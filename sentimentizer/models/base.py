"""Base class and registry for sentiment analysis models.

Provides:
- BaseSentimentModel: Shared predict() and predict_text() methods for inference
- MODEL_REGISTRY: Dict mapping model_type → (model_class, config_class, new_model_func)
- get_trained_model: Unified model loading from registry (replaces per-file duplicates)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from sentimentizer.config import VALID_DEVICES, OptimizationParams, SchedulerParams

if TYPE_CHECKING:
    from sentimentizer.tokenizer import Tokenizer


def unpack_saved_weights(checkpoint: dict, default_config: Any) -> tuple[dict, Any]:
    """Split a saved weights file into its state dict and architecture config.

    Weights files are a bare state dict (older files and Hub downloads) or a metadata
    dict with ``model_state_dict``, plus ``model_config`` when the architecture was
    tuned. ``n_heads`` cannot be recovered from weight shapes, so the saved config must
    win over defaults.
    """
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    saved_config = checkpoint.get("model_config")
    if not saved_config:
        return state_dict, default_config
    known = {f.name for f in dataclasses.fields(default_config)}
    overrides = {key: value for key, value in saved_config.items() if key in known}
    return state_dict, dataclasses.replace(default_config, **overrides)


class BaseSentimentModel(nn.Module):
    """Base class for all sentiment analysis models.

    Provides the shared predict() and predict_text() methods that were
    previously duplicated across RNN, Encoder, and Decoder. Subclasses
    must implement forward().

    Output contract (3-class):
    - forward(): returns (B, num_classes) raw logits — never squeeze
    - predict(): returns (B, num_classes) softmax probabilities
    - predict_text(): returns dict[str, float] mapping label names to
      probabilities, e.g. {"negative": 0.05, "neutral": 0.12, "positive": 0.83}
    """

    LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]
    NUM_CLASSES: int = 3

    # ── Capability declarations (class-level, no dispatch) ──────────────────
    MODEL_TYPE: ClassVar[str] = ""
    OPT_PARAMS_CLS: ClassVar[type] = OptimizationParams
    SCHED_PARAMS_CLS: ClassVar[type] = SchedulerParams
    NEEDS_TOKENIZE_STAGE: ClassVar[bool] = True
    DDP_FIND_UNUSED_PARAMS: ClassVar[bool] = False
    SUPPORTS_ONNX: ClassVar[bool] = True
    STEP_SCHEDULER_PER_BATCH: ClassVar[bool] = True

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer  # Tokenizer will be set externally after model creation

    def prepare_batch(self, batch: dict, device: str) -> tuple[dict, torch.Tensor]:
        """Convert a raw batch dict into (model_inputs, target).

        Maps legacy 'data' key to 'input_ids' so that model(**inputs)
        correctly passes the positional argument to forward().
        """
        target = batch["target"].to(device)
        inputs = {}
        for k, v in batch.items():
            if k == "target":
                continue
            # GloVe models expect 'input_ids'; the dataset uses 'data'
            key = "input_ids" if k == "data" else k
            inputs[key] = v.to(device)
        return inputs, target

    def save_to_checkpoint_dir(self, ckpt_dir: Path, tokenizer: Any | None = None) -> dict:
        """Save model state into ckpt_dir. Return metadata dict."""
        return {"model_state_dict": self.state_dict()}

    @classmethod
    def load_from_checkpoint_dir(
        cls, ckpt_dir: Path, metadata: dict, device: str
    ) -> BaseSentimentModel:
        """Construct an instance and restore weights."""
        _ensure_registry()
        model_type = cls.MODEL_TYPE
        state_dict, config = unpack_saved_weights(
            metadata, _MODEL_REGISTRY[model_type]["config_class"]()
        )

        emb_shape = state_dict["embed_layer.weight"].shape
        empty_embeddings = torch.zeros(emb_shape)

        num_classes = state_dict.get("_metadata", {}).get("num_classes", 3)
        if "classifier.3.weight" in state_dict:
            num_classes = state_dict["classifier.3.weight"].shape[0]

        if model_type == "rnn":
            hidden_size = state_dict["classifier.0.weight"].shape[1] // 2
            model = cls(
                emb_weights=empty_embeddings,
                hidden_size=hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                num_classes=num_classes,
            )
        elif model_type == "encoder":
            d_model = state_dict["proj.weight"].shape[0]
            model = cls(
                d_model=d_model,
                n_heads=config.n_heads,
                emb_weights=empty_embeddings,
                n_layers=config.n_layers,
                dropout=config.dropout,
                ff_multiplier=config.ff_multiplier,
                num_classes=num_classes,
            )
        elif model_type == "decoder":
            d_model = state_dict["proj.weight"].shape[0]
            model = cls(
                d_model=d_model,
                n_heads=config.n_heads,
                emb_weights=empty_embeddings,
                n_encoder_layers=config.n_encoder_layers,
                n_decoder_layers=config.n_decoder_layers,
                dropout=config.dropout,
                ff_multiplier=config.ff_multiplier,
                num_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unsupported GloVe model type for load_from_checkpoint_dir: {model_type}"
            )

        model.load_state_dict(state_dict)
        return model.to(device)

    def unfreeze_backbone(self) -> None:
        """No-op for GloVe models."""
        pass

    def predict(self, inputs: dict[str, torch.Tensor] | np.ndarray) -> torch.Tensor:
        """Run inference with softmax activation.

        Args:
            inputs: Dict of input tensors, or numpy array of token IDs (legacy format).

        Returns:
            Probability matrix of shape (B, num_classes) with softmax probabilities.
        """
        with torch.no_grad():
            self.eval()
            device = next(self.parameters()).device
            if isinstance(inputs, np.ndarray):
                tensor_input = torch.from_numpy(inputs).to(device)
                return torch.softmax(self.forward(tensor_input), dim=-1)
            else:
                device_inputs = {k: v.to(device) for k, v in inputs.items()}
                return torch.softmax(self.forward(**device_inputs), dim=-1)

    def predict_text(self, text: str) -> dict[str, float]:
        """Tokenize raw text and run sentiment prediction in one call.

        Args:
            text: Raw text string to classify.

        Returns:
            Dict mapping label names to probabilities.
        """
        from sentimentizer.tokenizer import Tokenizer

        if isinstance(self.tokenizer, Tokenizer):
            token_ids = self.tokenizer.tokenize_text(text)
            inputs = {"input_ids": torch.from_numpy(token_ids)}
        elif self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v for k, v in encoded.items()}
        else:
            raise ValueError("Tokenizer not set on model. Cannot predict_text.")

        probs = self.predict(inputs)
        return {label: probs[0, i].item() for i, label in enumerate(self.LABEL_NAMES)}


# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────

# Registry maps model_type → (model_class, config_class, new_model_func)
# Populated lazily to avoid circular imports at module load time.
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def _ensure_registry() -> None:
    """Populate the model registry if not yet populated."""
    if _MODEL_REGISTRY:
        return

    from sentimentizer.config import DecoderConfig, EncoderConfig, ModernBERTConfig, RNNConfig
    from sentimentizer.models.decoder import Decoder
    from sentimentizer.models.decoder import new_model as _decoder_new
    from sentimentizer.models.encoder import Encoder
    from sentimentizer.models.encoder import new_model as _encoder_new
    from sentimentizer.models.modernbert import ModernBERT
    from sentimentizer.models.modernbert import new_modernbert_model as _modernbert_new
    from sentimentizer.models.rnn import RNN
    from sentimentizer.models.rnn import new_model as _rnn_new

    _MODEL_REGISTRY["rnn"] = {
        "model_class": RNN,
        "config_class": RNNConfig,
        "new_model": _rnn_new,
        "weights_key": "rnn",
    }
    _MODEL_REGISTRY["encoder"] = {
        "model_class": Encoder,
        "config_class": EncoderConfig,
        "new_model": _encoder_new,
        "weights_key": "encoder",
    }
    _MODEL_REGISTRY["decoder"] = {
        "model_class": Decoder,
        "config_class": DecoderConfig,
        "new_model": _decoder_new,
        "weights_key": "decoder",
    }
    _MODEL_REGISTRY["modernbert"] = {
        "model_class": ModernBERT,
        "config_class": ModernBERTConfig,
        "new_model": _modernbert_new,
        "weights_key": "modernbert",
    }


def get_model_registry() -> dict[str, dict[str, Any]]:
    """Return the model registry, populating it lazily if needed."""
    _ensure_registry()
    return dict(_MODEL_REGISTRY)


def get_trained_model(model_type: str, device: str) -> nn.Module:
    """Load a pre-trained model from saved weights using the model registry."""
    _ensure_registry()

    if device not in VALID_DEVICES:
        raise ValueError("device must be cpu, cuda, or mps")

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type!r}")

    entry = _MODEL_REGISTRY[model_type]
    ModelClass = entry["model_class"]

    from sentimentizer.config import DriverConfig, weights_path_for

    weights_path = weights_path_for(model_type)

    # Try local file first; if missing, download from Hugging Face Hub
    if not Path(weights_path).exists():
        from sentimentizer.hf import _HF_MODEL_TYPES, download_weights

        is_hf_model = model_type in _HF_MODEL_TYPES
        downloaded = download_weights(
            model_type,
            weights_path,
            dict_path=None if is_hf_model else DriverConfig.files.dictionary_file_path,
        )
        if downloaded is None:
            raise FileNotFoundError(
                f"Weights file not found at {weights_path} and could not be "
                "downloaded from Hugging Face Hub. Please train the model "
                f"first: python workflows/driver.py --device cuda --model {model_type} "
                "--type new --save True"
            )

    checkpoint = torch.load(
        weights_path, map_location=torch.device(device=device), weights_only=True
    )

    return ModelClass.load_from_checkpoint_dir(
        Path(weights_path).parent, metadata=checkpoint, device=device
    )
