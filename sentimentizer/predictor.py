"""Prediction engine for sentiment analysis and review routing.

Encapsulates model loading, inference, and result formatting so that
serving (HTTP API) and other callers share a single code path.

Usage::

    from sentimentizer.predictor import SentimentPredictor

    predictor = SentimentPredictor(model_name="encoder")
    result = predictor.predict("This restaurant was amazing!")
    # {"label": "positive", "score": 0.88,
    #  "scores": {"negative": 0.02, ...}, "model": "encoder", "positive": 0.88}

    batch = predictor.predict_batch(["Great food!", "Terrible service."])
    # [{"label": "positive", "score": 0.88, "scores": {...},
    #  "token_count": 3, "model": "encoder", "positive": 0.88}, ...]

    route = predictor.classify("Is the pasta gluten-free?")
    # {"prediction": {"label": "dietary", "score": 0.95, "token_count": 4}}

    tokens = predictor.tokenize("Hello world")
    # {"text": "Hello world", "tokens": ["hello", "world"], "token_ids": [...], "token_count": 2}
"""

from __future__ import annotations

import importlib.metadata
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sentimentizer import logger
from sentimentizer.config import (
    DecoderConfig,
    EncoderConfig,
    ModernBERTConfig,
    RNNConfig,
    auto_detect_device,
)
from sentimentizer.models.base import BaseSentimentModel, get_trained_model
from sentimentizer.router.config import DEFAULT_ROUTER_MODEL_PATH, FOOD_TASK_SPEC
from sentimentizer.router.model import RouterModel
from sentimentizer.tokenizer import Tokenizer, get_trained_tokenizer, regex_tokenize, text_sequencer

# ---------------------------------------------------------------------------
# Router imports (optional — sentence-transformers may not be installed)
# ---------------------------------------------------------------------------

_HF_MAX_SEQ_LENGTH = ModernBERTConfig.max_seq_length

try:
    # router.config/model import lazily, so probe sentence_transformers
    # (the actual optional dep) here to set _ROUTER_AVAILABLE correctly.
    import sentence_transformers  # noqa: F401

    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False


class SentimentPredictor:
    """High-level prediction engine wrapping model loading and inference.

    Handles:
      - Single-text and batch sentiment prediction
      - Single-text and batch review routing
      - Standalone tokenization
      - Model and router lifecycle
    """

    def __init__(
        self,
        model_name: str = "encoder",
        router_model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.device: str = device or auto_detect_device("auto")
        self.model_name: str = model_name.lower().strip()

        # Only load the GloVe dictionary tokenizer for models that need it.
        # HF models (e.g. ModernBERT) ship their own tokenizer and don't use
        # the shared yelp.dictionary, so skip the download on first run.
        from sentimentizer.hf import _HF_MODEL_TYPES

        if self.model_name not in _HF_MODEL_TYPES:
            self.tokenizer: Tokenizer | None = get_trained_tokenizer()
        else:
            self.tokenizer = None

        # --- Sentiment model ---
        self.model, self._model_error = self._load_sentiment_model(self.model_name)

        # --- Router model (lazy — loaded on first classify() call) ---
        if router_model_path is None:
            router_model_path = DEFAULT_ROUTER_MODEL_PATH
        self._router_model_path: str = router_model_path
        self._initialize_router_state()

        # --- Version ---
        self._version = self._detect_version()

    def _initialize_router_state(self) -> None:
        """Create per-predictor coordination before the instance becomes shared."""
        self.router: Any = None
        self._router_error: str | None = None
        self._router_loaded: bool = False
        self._router_loading = False
        self._router_condition = threading.Condition()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_version() -> str:
        try:
            return importlib.metadata.version("sentimentizer")
        except Exception:
            return "unknown"

    def _load_sentiment_model(self, model_name: str) -> tuple[torch.nn.Module | None, str | None]:
        """Load a sentiment model. Returns (model, error).

        Returns (None, error_message) on failure so callers can decide
        whether to crash or degrade gracefully.
        """
        try:
            model = get_trained_model(model_name, device=self.device)
            # Only assign default tokenizer if model doesn't already have a custom one
            if getattr(model, "tokenizer", None) is None:
                model.tokenizer = self.tokenizer
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded sentiment model: {model_name}")
            return model, None
        except Exception as exc:
            logger.exception(f"Failed to load sentiment model: {model_name}")
            return None, str(exc)

    def _load_router_model(self, router_model_path: str) -> tuple[Any, str | None]:
        """Load the RouterModel; only the default food path can auto-download.

        Returns (None, error_message) on failure so callers can return 503
        instead of crashing the whole deployment.
        """
        if not _ROUTER_AVAILABLE:
            msg = "sentence-transformers package not installed"
            logger.warning(f"Router not available: {msg}")
            return None, msg

        path = Path(router_model_path)
        try:
            is_default = path.resolve() == Path(DEFAULT_ROUTER_MODEL_PATH).resolve()
            if not path.exists() and is_default:
                logger.info(f"Router model not found at {path}, downloading from HF Hub...")
                try:
                    self._download_router_artifact(path)
                    logger.info(f"Router model downloaded to {path}")
                except Exception:
                    logger.exception("Failed to download router model from HF Hub")

            if path.exists():
                logger.info(f"Loading router model from {path}")
                model = RouterModel.from_pretrained(str(path))
                if is_default and model.task is None and not model.labels:
                    # Only the known default food artifact can omit integer label metadata.
                    classes = np.asarray(model.head.classes_)
                    if (
                        classes.ndim == 1
                        and classes.dtype.kind in "iu"
                        and sorted(classes.tolist()) == list(range(len(FOOD_TASK_SPEC.labels)))
                    ):
                        model = RouterModel(model.backbone, model.head, task=FOOD_TASK_SPEC)
                model.class_labels()
            else:
                msg = f"Router model path not found: {path}"
                logger.warning(msg)
                return None, msg
            return model, None
        except Exception as exc:
            logger.exception("Failed to load router model")
            return None, str(exc)

    @staticmethod
    def _download_router_artifact(path: Path) -> None:
        """Download the pinned router artifact, publishing it only once complete.

        The download path doubles as the "already downloaded" marker, so a partial
        directory left by an interrupted download would never be repaired.
        """
        from huggingface_hub import snapshot_download

        from sentimentizer.config import HF_ROUTER_REPO, HF_ROUTER_REVISION

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}-") as staging:
            staged = Path(staging) / path.name
            snapshot_download(
                repo_id=HF_ROUTER_REPO,
                revision=HF_ROUTER_REVISION,
                local_dir=str(staged),
            )
            staged.replace(path)

    def _ensure_router_loaded(self) -> None:
        """Lazily load the router model on first classify() call.

        No-op if already loaded (or if a previous load attempt failed —
        in which case the cached error is reused so we don't retry on
        every request).
        """
        with self._router_condition:
            self._router_condition.wait_for(lambda: not self._router_loading)
            if self._router_loaded or self._router_error is not None:
                return
            self._router_loading = True
        router, error = None, "Router loading was interrupted"
        started = time.monotonic()
        try:
            # Construction/network I/O occurs without holding the condition.
            router, error = self._load_router_model(self._router_model_path)
        except Exception as exc:
            error = f"Router load failed ({type(exc).__name__})"
            logger.exception("Router construction failed")
        finally:
            with self._router_condition:
                self.router = router
                self._router_error = (
                    (error or "Router model did not load") if router is None else None
                )
                self._router_loaded = router is not None
                self._router_loading = False
                self._router_condition.notify_all()
            logger.info(
                "Router load attempt completed",
                loaded=router is not None,
                duration_s=time.monotonic() - started,
            )

    def load_router(self) -> bool:
        """Warm up the router, preserving a previously memoized failure."""
        self._ensure_router_loaded()
        return self.router_loaded

    def retry_router_load(self) -> bool:
        """Explicitly retry terminal failure, or join an already active attempt."""
        with self._router_condition:
            if self._router_loading:
                self._router_condition.wait_for(lambda: not self._router_loading)
                return self.router_loaded
            if self.router_loaded:
                return True
            self._router_error = None
        return self.load_router()

    @property
    def router_loading(self) -> bool:
        """Report construction progress without starting or waiting for a load."""
        with self._router_condition:
            return self._router_loading

    @property
    def router_decisions_ready(self) -> bool:
        """Report whether the published router has an acceptance policy."""
        with self._router_condition:
            return self._router_loaded and self.router.acceptance_policy is not None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        """Whether the sentiment model loaded successfully."""
        return self.model is not None

    @property
    def router_loaded(self) -> bool:
        """Whether the router model loaded successfully."""
        with self._router_condition:
            return self._router_loaded

    @property
    def router_error(self) -> str | None:
        """Error message if router failed to load, else None."""
        with self._router_condition:
            return self._router_error

    @property
    def can_tokenize(self) -> bool:
        """Whether a tokenizer is available; HF models only have the one they load with."""
        return self.tokenizer is not None or (
            self.model_loaded and getattr(type(self.model), "NEEDS_TOKENIZE_STAGE", True) is False
        )

    @property
    def model_error(self) -> str | None:
        """Error message if sentiment model failed to load, else None."""
        return self._model_error

    @property
    def version(self) -> str:
        """Package version string."""
        return self._version

    # ------------------------------------------------------------------
    # Sentiment inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict[str, Any]:
        """Run sentiment analysis on a single text.

        Returns:
            Dict with label, score, all class scores, token usage,
            and model name.
            Example: ``{"label": "positive", "score": 0.88,
            "scores": {"negative": 0.02, "neutral": 0.10,
            "positive": 0.88}, "token_count": 5, "model": "encoder",
            "positive": 0.88}``
        """
        if not self.model_loaded:
            raise RuntimeError(f"Sentiment model not loaded: {self._model_error}")
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Run sentiment analysis on multiple texts — single forward pass.

        Tokenizes all texts, stacks into a single sequence batch,
        and runs one forward pass.

        Returns:
            List of dicts, each with label, score, all class scores,
            token usage, and the model name. Uses additive format with
            both the dynamic winning-class key (deprecated) and explicit
            fields.
            Example: ``[{"positive": 0.88, "label": "positive",
            "score": 0.88, "scores": {...}, "token_count": 12,
            "model": "encoder"}, ...]``
        """
        if not self.model_loaded:
            raise RuntimeError(f"Sentiment model not loaded: {self._model_error}")

        if getattr(type(self.model), "NEEDS_TOKENIZE_STAGE", True) is False:
            encoded = self.model.tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=_HF_MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            probs = self.model.predict(encoded)
        else:
            token_arrays = [self.model.tokenizer.tokenize_text(t) for t in texts]
            batch = np.concatenate(token_arrays, axis=0)
            probs = self.model.predict(batch)

        token_counts = [len(regex_tokenize(t)) for t in texts]

        label_names = BaseSentimentModel.LABEL_NAMES
        label_idx = probs.argmax(dim=1)
        results: list[dict[str, Any]] = []
        for i in range(len(texts)):
            label = label_names[label_idx[i]]
            score = probs[i, label_idx[i]].item()
            all_scores = {name: probs[i, j].item() for j, name in enumerate(label_names)}
            results.append(
                {
                    label: score,
                    "label": label,
                    "score": score,
                    "scores": all_scores,
                    "token_count": token_counts[i],
                    "model": self.model_name,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Router inference
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict[str, Any]:
        """Classify a single text into a route category.

        Returns:
            Dict with key ``prediction`` containing label, score, and token_count.
        """
        results = self.classify_batch([text])
        return results[0]

    def classify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify multiple texts into route categories.

        Returns:
            List of dicts, each with key ``prediction`` containing label, score, and token_count.
        """
        self._ensure_router_loaded()
        if not self.router_loaded:
            raise RuntimeError(f"Router model not loaded: {self._router_error}")

        predictions = self.router.classify_batch(texts)
        results: list[dict[str, Any]] = []
        for text, prediction in zip(texts, predictions, strict=True):
            token_count = len(regex_tokenize(text))
            results.append(
                {
                    "prediction": {
                        **prediction.model_dump(mode="json"),
                        "token_count": token_count,
                    },
                }
            )
        return results

    def decide(self, text: str) -> dict[str, Any]:
        """Return a predicted topic and a separate accept/abstain decision."""
        return self.decide_batch([text])[0]

    def decide_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Apply the artifact's calibrated policy while preserving input order."""
        self._ensure_router_loaded()
        if not self.router_loaded:
            raise RuntimeError(f"Router model not loaded: {self._router_error}")
        predictions = self.router.decide_batch(texts)
        return [
            {
                "prediction": {
                    **prediction.model_dump(mode="json"),
                    "token_count": len(regex_tokenize(text)),
                }
            }
            for text, prediction in zip(texts, predictions, strict=True)
        ]

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> dict[str, Any]:
        """Tokenize text without running inference.

        Returns:
            Dict with keys ``text``, ``tokens``, ``token_ids``, ``token_count``.

        Raises:
            RuntimeError: If the model that owns the tokenizer failed to load.
        """
        if not self.can_tokenize:
            raise RuntimeError(f"Tokenizer not available: {self._model_error}")
        if self.model_loaded and getattr(type(self.model), "NEEDS_TOKENIZE_STAGE", True) is False:
            encoded = self.model.tokenizer(text, truncation=True, max_length=_HF_MAX_SEQ_LENGTH)
            tokens = self.model.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            return {
                "text": text,
                "tokens": tokens,
                "token_ids": encoded["input_ids"],
                "token_count": len(encoded["input_ids"]),
            }

        tokens = regex_tokenize(text)
        token_ids = text_sequencer(self.tokenizer.dictionary, tokens, self.tokenizer.cfg.max_len)
        return {
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids.tolist(),
            "token_count": len(tokens),
        }

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    def get_sentiment_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded sentiment model."""
        import dataclasses

        from sentimentizer.config import EmbeddingsConfig

        config_entry = _MODEL_CONFIGS.get(self.model_name)
        if config_entry is None:
            return {"model": self.model_name, "status": "unknown_config"}

        cfg = config_entry["config_class"]()
        cfg_dict = dataclasses.asdict(cfg)
        param_count = sum(p.numel() for p in self.model.parameters()) if self.model_loaded else 0

        max_seq_len = self.tokenizer.cfg.max_len if self.tokenizer else 512
        if hasattr(cfg, "max_seq_length"):
            max_seq_len = cfg.max_seq_length

        embedding_dim = EmbeddingsConfig.emb_length
        if self.model_loaded:
            inner = self.model.module if hasattr(self.model, "module") else self.model
            if (
                hasattr(inner, "backbone")
                and hasattr(inner.backbone, "config")
                and hasattr(inner.backbone.config, "hidden_size")
            ):
                embedding_dim = inner.backbone.config.hidden_size

        return {
            self.model_name: {
                "architecture": config_entry["architecture"],
                "device": self.device,
                "max_sequence_length": max_seq_len,
                "embedding_dim": embedding_dim,
                "parameters": param_count,
                "status": "loaded" if self.model_loaded else "error",
                **cfg_dict,
            }
        }

    def get_router_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded router model."""
        with self._router_condition:
            router = self.router if self._router_loaded else None
            error = self._router_error
        info: dict[str, Any] = {
            "model_path": self._router_model_path,
            "categories": [],
            "task_id": None,
            "taxonomy_version": None,
            "status": "loaded" if router is not None else "not_loaded",
        }
        task = None
        if router is not None:
            task = router.task
            info["categories"] = router.labels or router.class_labels()
        else:
            try:
                metadata = RouterModel.read_metadata(self._router_model_path)
                if metadata is not None:
                    task = metadata.task
                    info["categories"] = metadata.labels
                elif (
                    Path(self._router_model_path).resolve()
                    == Path(DEFAULT_ROUTER_MODEL_PATH).resolve()
                ):
                    task = FOOD_TASK_SPEC
            except (ValueError, OSError, KeyError):
                info["error"] = "Router artifact metadata is invalid or unreadable"
        if task is not None:
            info.update(
                categories=list(task.labels),
                task_id=task.task_id,
                taxonomy_version=task.taxonomy_version,
            )
        if error:
            info["error"] = error
        return info


# ---------------------------------------------------------------------------
# Model config registry — metadata derived from config dataclasses
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "rnn": {
        "architecture": "Bidirectional LSTM",
        "config_class": RNNConfig,
    },
    "encoder": {
        "architecture": "Transformer Encoder (CLS token)",
        "config_class": EncoderConfig,
    },
    "decoder": {
        "architecture": "Encoder-Decoder Transformer (cross-attention)",
        "config_class": DecoderConfig,
    },
    "modernbert": {
        "architecture": "ModernBERT Transformer",
        "config_class": ModernBERTConfig,
    },
}
