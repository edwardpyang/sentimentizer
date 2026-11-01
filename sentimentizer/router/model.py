"""RouterModel: sentence-transformer backbone + LogisticRegression head.

Drop-in replacement for SetFitModel with equivalent API:
- from_pretrained() / save_pretrained() / push_to_hub()
- model_encode() / predict() / predict_proba()

The model consists of:
- backbone: SentenceTransformer for encoding text to embeddings
- head: LogisticRegression for classification on embeddings
- labels: List of category names (e.g., ["dietary", "service", "general"])
"""

from __future__ import annotations

import json
import logging
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from sentimentizer.router.head import (
    HEAD_FILENAME,
    PICKLED_HEAD_FILENAMES,
    PICKLED_HEAD_PATTERNS,
    load_head,
    save_head,
)
from sentimentizer.router.schemas import (
    RouterAcceptancePolicy,
    RouterArtifactMetadata,
    RouterClassification,
    RouterDecision,
    RouterProvenance,
    RouterTaskSpec,
)

logger = logging.getLogger(__name__)

# Lazy imports — these are optional dependencies
_sentence_transformers: Any = None


class RouterPolicyUnavailableError(RuntimeError):
    """Signal that classification is available but calibrated decisions are not."""


def _get_sentence_transformers() -> Any:
    """Lazy import sentence_transformers."""
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer

        _sentence_transformers = SentenceTransformer
    return _sentence_transformers


class RouterModel:
    """Sentence-transformer backbone + LogisticRegression head.

    Drop-in replacement for SetFitModel with equivalent API:
    - from_pretrained() / save_pretrained() / push_to_hub()
    - model_encode() / predict() / predict_proba()
    """

    def __init__(
        self,
        backbone: Any,
        head: Any | None = None,
        labels: list[str] | None = None,
        *,
        task: RouterTaskSpec | None = None,
        provenance: RouterProvenance | None = None,
        acceptance_policy: RouterAcceptancePolicy | None = None,
    ) -> None:
        """Initialize RouterModel.

        Args:
            backbone: SentenceTransformer model for encoding.
            head: Sklearn classifier (e.g., LogisticRegression). If None,
                  predict/predict_proba will raise RuntimeError.
            labels: List of label names in index order.
        """
        self.backbone = backbone
        self.head = head
        if task is not None and labels is not None and tuple(labels) != task.labels:
            raise ValueError("Model labels conflict with task labels")
        metadata = RouterArtifactMetadata(labels=labels or [], task=task)
        self.task = metadata.task
        self.labels = list(metadata.labels)
        self.provenance = provenance
        self.acceptance_policy = acceptance_policy

    @staticmethod
    def read_metadata(path: str | Path) -> RouterArtifactMetadata | None:
        """Read a sidecar without importing the optional model dependencies."""
        path = Path(path)
        for name in ("router_config.json", "config_setfit.json"):
            try:
                with (path / name).open() as source:
                    data = json.load(source)
            except FileNotFoundError:
                continue
            if not isinstance(data, dict):
                raise ValueError("Router metadata must be an object")
            labels = data.get("labels", [])
            if isinstance(labels, dict):
                data["labels"] = [labels[str(i)] for i in range(len(labels))]
            return RouterArtifactMetadata.model_validate(data)
        return None

    # ------------------------------------------------------------------
    # Class methods: loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str | Path) -> RouterModel:
        """Load a RouterModel from a local directory.

        Expects the directory to contain:
        - SentenceTransformer backbone files (model.safetensors, config.json, etc.)
        - router_head.safetensors (classifier parameters; see ``router.head``)
        - router_config.json (metadata with labels, model_type, etc.)

        Pickled heads are never loaded, because that would run code from the artifact.

        Args:
            path: Path to the saved model directory.

        Returns:
            Loaded RouterModel.

        Raises:
            FileNotFoundError: If the path doesn't exist.
            RuntimeError: If the artifact has no pickle-free head.
            ValueError: If the head or metadata is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        metadata = cls.read_metadata(path) or RouterArtifactMetadata()

        head_path = path / HEAD_FILENAME
        if not head_path.exists():
            if any((path / name).exists() for name in PICKLED_HEAD_FILENAMES):
                raise RuntimeError(
                    f"Router artifact at {path} has a pickled head, which is no longer "
                    f"loaded. Run: sentimentizer router convert-head --model-path {path}"
                )
            raise RuntimeError(f"Router artifact at {path} has no {HEAD_FILENAME}")
        logger.info(f"Loading classification head from {head_path}")
        head = load_head(head_path)

        SentenceTransformer = _get_sentence_transformers()

        # Load backbone
        logger.info(f"Loading backbone from {path}")
        backbone = SentenceTransformer(str(path))

        return cls(
            backbone=backbone,
            head=head,
            labels=metadata.labels,
            task=metadata.task,
            provenance=metadata.provenance,
            acceptance_policy=metadata.acceptance_policy,
        )

    # ------------------------------------------------------------------
    # Save / push
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """Save the RouterModel to a local directory.

        Saves:
        - SentenceTransformer backbone files
        - router_head.safetensors (classifier parameters; see ``router.head``)
        - router_config.json (metadata)

        Args:
            path: Directory to save the model to.

        Raises:
            UnsupportedRouterHeadError: If the head cannot be stored without pickle.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save backbone
        logger.info(f"Saving backbone to {path}")
        self.backbone.save(str(path))

        # Save head
        if self.head is not None:
            head_path = path / HEAD_FILENAME
            save_head(self.head, head_path)
            logger.info(f"Saved classification head to {head_path}")

        # Save config
        metadata = RouterArtifactMetadata(
            labels=self.labels,
            head_type=type(self.head).__name__ if self.head is not None else None,
            task=self.task,
            provenance=self.provenance,
            acceptance_policy=self.acceptance_policy,
        )
        config_path = path / "router_config.json"
        # A metadata reader sees the old complete file or the new complete file.
        with tempfile.NamedTemporaryFile(mode="w", dir=path, delete=False) as target:
            temporary_path = Path(target.name)
            try:
                target.write(metadata.model_dump_json(indent=2))
                target.close()
                temporary_path.replace(config_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        logger.info(f"Saved router config to {config_path}")

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        """Push the RouterModel to Hugging Face Hub.

        Saves the model to a temporary directory first, then uploads.

        Args:
            repo_id: Hugging Face repository ID (e.g., "user/model").
            **kwargs: Additional arguments passed to huggingface_hub.
                Older pickled heads are deleted from the repo unless
                ``delete_patterns`` is given.
        """
        import tempfile

        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api = HfApi()
            kwargs.setdefault("delete_patterns", list(PICKLED_HEAD_PATTERNS))
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                **kwargs,
            )
        logger.info(f"Pushed RouterModel to {repo_id}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def class_labels(self) -> list[str]:
        """Resolve probability-column labels from the classifier's class order."""
        if self.head is None:
            raise RuntimeError("Router classification head is not loaded")
        classes = np.asarray(self.head.classes_)
        if classes.ndim != 1 or len(classes) < 2:
            raise ValueError("Router head must have at least two classes")
        values = classes.tolist()
        labels = list(self.task.labels) if self.task is not None else list(self.labels)
        if not labels and all(isinstance(value, str) for value in values):
            labels = values
        if len(labels) != len(values) or len(set(labels)) != len(labels):
            raise ValueError("Router head classes must cover the saved label vocabulary")
        resolved = []
        for value in values:
            if isinstance(value, str) and value in labels:
                resolved.append(value)
            elif (
                isinstance(value, Integral)
                and not isinstance(value, bool)
                and 0 <= int(value) < len(labels)
            ):
                resolved.append(labels[int(value)])
            else:
                raise ValueError("Router head class has no valid saved label mapping")
        if len(set(resolved)) != len(labels):
            raise ValueError("Router head classes must map uniquely to saved labels")
        return resolved

    def classify_batch(self, texts: list[str]) -> list[RouterClassification]:
        """Classify texts with one encoding pass and validated native values."""
        labels = self.class_labels()
        if not texts:
            return []
        probabilities = self.probabilities_batch(texts)
        winners = probabilities.argmax(axis=1)
        return [
            RouterClassification(label=labels[int(column)], score=float(row[column]))
            for row, column in zip(probabilities, winners, strict=True)
        ]

    def probabilities_batch(self, texts: list[str]) -> np.ndarray:
        """Validate probability dimensions, values and normalization in one pass."""
        labels = self.class_labels()
        if not texts:
            return np.empty((0, len(labels)))
        probabilities = np.asarray(self.head.predict_proba(self.model_encode(texts)))
        if probabilities.dtype.kind not in "fiu":
            raise ValueError("Router probabilities must be numeric values, not booleans or strings")
        if probabilities.shape != (len(texts), len(labels)):
            raise ValueError("Router probability dimensions do not match inputs and classes")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError("Router probabilities must be finite and between zero and one")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
            raise ValueError("Router probability rows must sum to one")
        return probabilities

    def decide_batch(self, texts: list[str]) -> list[RouterDecision]:
        """Apply the saved acceptance policy without changing the predicted label."""
        if self.acceptance_policy is None:
            raise RouterPolicyUnavailableError(
                "Router has no acceptance policy; calibrate the artifact first"
            )
        threshold = self.acceptance_policy.threshold
        return [
            RouterDecision(
                **prediction.model_dump(),
                decision="accept"
                if threshold is not None and prediction.score >= threshold
                else "abstain",
            )
            for prediction in self.classify_batch(texts)
        ]

    def model_encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode texts to embeddings using the backbone.

        Args:
            texts: Single text or list of texts to encode.

        Returns:
            Numpy array of shape (N, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.backbone.encode(texts, convert_to_numpy=True)

    def predict(self, texts: list[str] | str) -> list | np.ndarray:
        """Predict class labels for texts.

        Args:
            texts: Single text or list of texts.

        Returns:
            List of predicted labels (ints or strings depending on head).
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model_encode(texts)
        return self.head.predict(embeddings)

    def predict_proba(self, texts: list[str] | str) -> np.ndarray:
        """Predict class probabilities for texts.

        Args:
            texts: Single text or list of texts.

        Returns:
            Numpy array of shape (N, num_classes) with class probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model_encode(texts)
        return self.head.predict_proba(embeddings)
