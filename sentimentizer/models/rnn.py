from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim import corpora

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    VALID_DEVICES,
    EmbeddingsConfig,
    RNNConfig,
    weights_path_for,
)
from sentimentizer.extractor import new_embedding_weights
from sentimentizer.models.base import BaseSentimentModel, unpack_saved_weights

logger = new_logger(DEFAULT_LOG_LEVEL)

# Module-level singleton for B008 compliance — used as default in function signatures
_DEFAULT_RNN_CONFIG = RNNConfig()


class RNN(BaseSentimentModel):
    """Bidirectional LSTM for sentiment classification.

    Uses pre-trained GloVe embeddings, a multi-layer bidirectional LSTM that
    processes tokens in sequence order (batch_first=True), and concatenates
    the final forward and backward hidden states for classification.

    Args:
        emb_weights: Pre-trained embedding weights of shape (vocab_size, emb_dim)
        hidden_size: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        verbose: Whether to log debug shapes
        dropout: Dropout probability
    """

    MODEL_TYPE: str = "rnn"

    def __init__(
        self,
        emb_weights: torch.Tensor,
        hidden_size: int = RNNConfig.hidden_size,
        num_layers: int = RNNConfig.num_layers,
        verbose: bool = False,
        dropout: float = RNNConfig.dropout,
        num_classes: int = RNNConfig.num_classes,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        emb_dim = emb_weights.shape[1]

        # Embedding layer with pre-trained GloVe weights.  padding_idx=0 keeps
        # the pad token's vector fixed at zero so it doesn't drift during
        # training (pack_padded_sequence excludes pad positions from the LSTM
        # but the embedding still receives gradients without padding_idx).
        self.embed_layer = nn.Embedding(emb_weights.shape[0], emb_dim, padding_idx=0)
        self.fc0 = nn.Linear(emb_dim, emb_dim)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Bidirectional LSTM with batch_first=True
        # Processes tokens in sequence order — word 1, word 2, ..., word 200
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Classification head: concatenated final hidden states → class logits
        # hidden_size * 2 because bidirectional (forward + backward)
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        self.verbose = verbose

        # Load pre-trained GloVe weights, freeze all rows except OOV (last row).
        # See encoder.py for the full rationale and the per-row hook pattern.
        self.embed_layer.load_state_dict({"weight": emb_weights})  # type: ignore

        if freeze_embeddings:

            def _freeze_glove_grads(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[:-1] = 0.0  # zero GloVe rows; row 0 (pad) already zeroed by padding_idx
                return grad

            self.embed_layer.weight.register_hook(_freeze_glove_grads)

    def forward(
        self,
        input_ids: torch.Tensor,
        onnx_export: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass producing raw logits.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
                    Zero-padding is used to compute sequence lengths.
            onnx_export: If True, skip pack_padded_sequence for ONNX compatibility.
                         Uses masked LSTM output instead. Slightly different numerics.
                         Do NOT pass this manually — it is set by _RNNOnnxWrapper
                         during torch.onnx.export() tracing.
            **kwargs: Ignored.

        Returns:
            Logits of shape ``(batch, num_classes)``
        """
        inputs = input_ids
        embeds = self.embed_layer(inputs)  # (B, seq_len, emb_dim)
        embeds = self.dropout_layer(embeds)
        if self.verbose:
            logger.info(f"embedding shape {embeds.shape}")

        embeds = F.relu(self.fc0(embeds), inplace=True)  # (B, seq_len, emb_dim)

        # Compute actual (non-padding) lengths so the bidirectional LSTM
        # can skip padding.  Without packing, the backward pass starts
        # from position max_len-1 (all zeros), so its final hidden state
        # is dominated by padding noise instead of real content.
        lengths = (inputs != 0).sum(dim=1).clamp(min=1)  # (B,)

        if onnx_export:
            # ONNX-compatible path: no pack_padded_sequence.
            # The LSTM processes all positions including padding. Since padding
            # tokens map to the zero-vector in the embedding layer (index 0),
            # their contribution is limited to bias-driven drift in the hidden
            # state. For sequences near max_len=200, this drift is typically <1e-2.
            lstm_out, _ = self.lstm(embeds)  # (B, seq_len, hidden_size * 2)
            B = inputs.size(0)
            hidden_size = self.lstm.hidden_size

            # Forward final state: extract at the index of the last real token
            idx = (lengths - 1).clamp(max=inputs.size(1) - 1)
            forward_hidden = lstm_out[torch.arange(B, device=inputs.device), idx, :hidden_size]

            # Backward final state: at index 0 of the output.
            # NOTE: In the unpacked path, the backward LSTM at position 0 has
            # processed ALL tokens including padding. Since padding maps to
            # near-zero embeddings, the bias-driven drift is typically <1e-2
            # for sequences near max_len=200. validate_onnx_export() verifies
            # this tolerance.
            backward_hidden = lstm_out[:, 0, hidden_size:]

            hidden_cat = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # Standard path with packed sequences (more accurate)
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)

            # hidden shape: (num_layers * 2, B, hidden_size)
            # Take final layer's forward and backward hidden states.
            # With packed input, these are correctly computed from the
            # last real token (forward) and first real token (backward).
            hidden_fwd = hidden[-2]  # last forward layer: (B, hidden_size)
            hidden_bwd = hidden[-1]  # last backward layer: (B, hidden_size)
            hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (B, hidden*2)

        if self.verbose:
            logger.info(f"hidden cat shape {hidden_cat.shape}")

        logits = self.classifier(hidden_cat)  # (B, num_classes)
        return logits


def new_model(
    dict_path: str,
    embeddings_config: EmbeddingsConfig,
    model_config: RNNConfig = _DEFAULT_RNN_CONFIG,
    freeze_embeddings: bool = True,
) -> RNN:
    """Create a new RNN model with pre-trained GloVe embeddings.

    Args:
        dict_path: Path to the gensim dictionary file
        embeddings_config: Configuration for GloVe embeddings
        model_config: RNN architecture configuration (defaults from RNNConfig)
    """
    dict_yelp = corpora.Dictionary.load(dict_path)

    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = RNN(
        emb_weights=emb_t,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        num_classes=model_config.num_classes,
        freeze_embeddings=freeze_embeddings,
    )
    return model


def get_trained_model(
    device: str,
    model_config: RNNConfig = _DEFAULT_RNN_CONFIG,
) -> RNN:
    """Load a pre-trained RNN model from saved weights.

    If local weights are not found, attempts to download them from the
    Hugging Face Hub (``ryeyoo/sentimentizer-rnn``).

    Args:
        device: Device to load weights onto ("cpu", "cuda", or "mps")
        model_config: RNN architecture configuration (must match saved weights)

    Returns:
        RNN model with loaded weights
    """
    if device not in VALID_DEVICES:
        raise ValueError("device must be cpu, cuda, or mps")

    weights_path = weights_path_for("rnn")

    # Try local file first; if missing, download from Hugging Face Hub
    if not Path(weights_path).exists():
        from sentimentizer.config import DriverConfig
        from sentimentizer.hf import download_weights

        downloaded = download_weights(
            "rnn", weights_path, dict_path=DriverConfig.files.dictionary_file_path
        )
        if downloaded is None:
            raise FileNotFoundError(
                f"Weights file not found at {weights_path} and could not be "
                "downloaded from Hugging Face Hub. Please train the model "
                "first: python workflows/driver.py --device cuda --model rnn "
                "--type new --save True"
            )

    weights = torch.load(weights_path, map_location=torch.device(device=device), weights_only=True)
    weights, model_config = unpack_saved_weights(weights, model_config)

    # Check if weights are from the new architecture (has 'classifier' keys)
    if "classifier.0.weight" not in weights:
        raise RuntimeError(
            "Saved weights are from a previous model architecture and are incompatible. "
            "Please retrain the model: "
            "python workflows/driver.py --device cuda --model rnn --type new --save True"
        )

    emb_shape = weights["embed_layer.weight"].shape
    hidden_size = weights["classifier.0.weight"].shape[1] // 2

    # Infer num_classes from the final linear layer output dimension
    num_classes = weights["classifier.3.weight"].shape[0]
    if num_classes == 1:
        raise RuntimeError(
            "Saved weights are from a binary classification model (num_classes=1). "
            "3-class migration requires retraining: "
            "python workflows/driver.py --device cuda --model rnn --type new --save True"
        )

    # Also check _metadata if available for robust detection
    if "_metadata" in weights and "num_classes" in weights["_metadata"]:
        num_classes = weights["_metadata"]["num_classes"]

    empty_embeddings = torch.zeros(emb_shape)
    model = RNN(
        emb_weights=empty_embeddings,
        hidden_size=hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        num_classes=num_classes,
    )

    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        raise RuntimeError(
            "Saved weights are incompatible with the current model architecture. "
            "Please retrain the model: "
            "python workflows/driver.py --device cuda --model rnn --type new --save True"
        ) from e

    return model
