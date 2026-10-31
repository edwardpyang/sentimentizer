import math
from pathlib import Path
from typing import Any

import torch
from gensim import corpora
from torch import nn

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    VALID_DEVICES,
    DecoderConfig,
    EmbeddingsConfig,
    weights_path_for,
)
from sentimentizer.extractor import new_embedding_weights
from sentimentizer.models.base import BaseSentimentModel, unpack_saved_weights

logger = new_logger(DEFAULT_LOG_LEVEL)

# Module-level singleton for B008 compliance — used as default in function signatures
_DEFAULT_DECODER_CONFIG = DecoderConfig()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe: torch.Tensor = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe.narrow(1, 0, x.size(1))
        return self.dropout(x)


class Decoder(BaseSentimentModel):
    """Encoder-Decoder Transformer for sentiment classification.

    The input text is encoded by a small TransformerEncoder into memory
    representations. A learnable query token cross-attends to the memory
    through TransformerDecoder layers. The query output is then classified
    into a sentiment score.

    Args:
        emb_weights: Pre-trained embedding weights of shape (vocab_size, emb_dim)
        d_model: Internal Transformer dimension (projected from embeddings)
        n_heads: Number of attention heads
        n_encoder_layers: Number of Transformer encoder layers
        n_decoder_layers: Number of Transformer decoder layers
        verbose: Whether to log debug shapes
        dropout: Dropout probability
        ff_multiplier: Feed-forward dim = d_model * ff_multiplier
    """

    MODEL_TYPE: str = "decoder"

    def __init__(
        self,
        emb_weights: torch.Tensor,
        d_model: int = DecoderConfig.d_model,
        n_heads: int = DecoderConfig.n_heads,
        n_encoder_layers: int = DecoderConfig.n_encoder_layers,
        n_decoder_layers: int = DecoderConfig.n_decoder_layers,
        verbose: bool = False,
        dropout: float = DecoderConfig.dropout,
        ff_multiplier: int = DecoderConfig.ff_multiplier,
        num_classes: int = DecoderConfig.num_classes,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.verbose = verbose
        self.num_classes = num_classes

        # Embedding layer.  padding_idx=0 keeps the pad token's vector fixed
        # at zero so it doesn't drift during training (the encoder/decoder
        # attention masks ignore these positions, but the embedding still
        # receives gradients without padding_idx).
        self.embed_layer = nn.Embedding(emb_weights.shape[0], emb_weights.shape[1], padding_idx=0)

        # Project GloVe embeddings to d_model
        self.proj = nn.Linear(emb_weights.shape[1], d_model)

        # Learnable query token for cross-attention decoding
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding for the input sequence (default max_len=500)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder — encodes input text into memory.
        # Pre-LN + GELU for stable gradients from init (see encoder.py rationale).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            enable_nested_tensor=False,
        )

        # Transformer decoder — query token cross-attends to encoded memory.
        # Pre-LN + GELU for the same stability reasons as the encoder above.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
        )

        # Classification head: query output → class logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Load pre-trained GloVe weights, freeze the matrix, keep OOV row trainable.
        # See encoder.py for the full rationale.
        self.embed_layer.load_state_dict({"weight": emb_weights})  # type: ignore

        if freeze_embeddings:

            def freeze_glove_grads(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[:-1] = 0.0
                return grad

            self.embed_layer.weight.register_hook(freeze_glove_grads)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass producing raw logits.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
                    Zero-padding is used to build the padding mask.
            **kwargs: Ignored.

        Returns:
            Logits of shape ``(batch, num_classes)``
        """
        inputs = input_ids
        # Embed and project
        embeds = self.embed_layer(inputs)  # (B, seq_len, emb_dim)
        if self.verbose:
            logger.info(f"embedding shape {embeds.shape}")

        projected = self.proj(embeds) * math.sqrt(self.d_model)  # (B, seq_len, d_model)

        # Add positional encoding to input
        memory_input = self.pos_encoder(projected)  # (B, seq_len, d_model)

        # Padding mask for encoder: True where padding (attention ignores those positions)
        src_key_padding_mask = inputs == 0  # (B, seq_len)

        # Encode input text into memory (with padding mask)
        memory = self.encoder(
            memory_input, src_key_padding_mask=src_key_padding_mask
        )  # (B, seq_len, d_model)
        if self.verbose:
            logger.info(f"memory shape {memory.shape}")

        # Expand query token for the batch
        query = self.query_token.expand(inputs.size(0), -1, -1)  # (B, 1, d_model)

        # Query cross-attends to memory via decoder (memory_key_padding_mask
        # ensures cross-attention also ignores padding)
        decoded = self.decoder(
            tgt=query, memory=memory, memory_key_padding_mask=src_key_padding_mask
        )  # (B, 1, d_model)
        if self.verbose:
            logger.info(f"decoded shape {decoded.shape}")

        # Extract query output and classify
        query_out = decoded.squeeze(1)  # (B, d_model)
        logits = self.classifier(query_out)  # (B, num_classes)
        return logits


def new_model(
    dict_path: str,
    embeddings_config: EmbeddingsConfig,
    model_config: DecoderConfig = _DEFAULT_DECODER_CONFIG,
    freeze_embeddings: bool = True,
) -> Decoder:
    """Create a new Decoder model with pre-trained GloVe embeddings.

    Args:
        dict_path: Path to the gensim dictionary file
        embeddings_config: Configuration for GloVe embeddings
        model_config: Decoder architecture configuration (defaults from DecoderConfig)
    """

    dict_yelp = corpora.Dictionary.load(dict_path)
    embedding_matrix = new_embedding_weights(dict_yelp, embeddings_config)
    emb_t = torch.from_numpy(embedding_matrix)
    model = Decoder(
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        emb_weights=emb_t,
        n_encoder_layers=model_config.n_encoder_layers,
        n_decoder_layers=model_config.n_decoder_layers,
        dropout=model_config.dropout,
        ff_multiplier=model_config.ff_multiplier,
        num_classes=model_config.num_classes,
        freeze_embeddings=freeze_embeddings,
    )
    return model


def get_trained_model(
    device: str,
    model_config: DecoderConfig = _DEFAULT_DECODER_CONFIG,
) -> Decoder:
    """Load a pre-trained Decoder model from saved weights.

    If local weights are not found, attempts to download them from the
    Hugging Face Hub (``ryeyoo/sentimentizer-decoder``).

    Args:
        device: Device to load weights onto ("cpu", "cuda", or "mps")
        model_config: Decoder architecture configuration (must match saved weights)

    Returns:
        Decoder model with loaded weights
    """
    if device not in VALID_DEVICES:
        raise ValueError("device must be cpu, cuda, or mps")

    weights_path = weights_path_for("decoder")

    # Try local file first; if missing, download from Hugging Face Hub
    if not Path(weights_path).exists():
        from sentimentizer.config import DriverConfig
        from sentimentizer.hf import download_weights

        downloaded = download_weights(
            "decoder", weights_path, dict_path=DriverConfig.files.dictionary_file_path
        )
        if downloaded is None:
            raise FileNotFoundError(
                f"Weights file not found at {weights_path} and could not be "
                "downloaded from Hugging Face Hub. Please train the model "
                "first: python workflows/driver.py --device cuda --model decoder "
                "--type new --save True"
            )

    weights = torch.load(
        weights_path,
        map_location=torch.device(device=device),
        weights_only=True,
    )
    weights, model_config = unpack_saved_weights(weights, model_config)

    # Infer dimensions from saved weights
    emb_shape = weights["embed_layer.weight"].shape
    d_model = weights["proj.weight"].shape[0]

    # Infer num_classes from the final linear layer output dimension
    num_classes = weights["classifier.3.weight"].shape[0]
    if num_classes == 1:
        raise RuntimeError(
            "Saved weights are from a binary classification model (num_classes=1). "
            "3-class migration requires retraining: "
            "python workflows/driver.py --device cuda --model decoder --type new --save True"
        )

    # Also check _metadata if available for robust detection
    if "_metadata" in weights and "num_classes" in weights["_metadata"]:
        num_classes = weights["_metadata"]["num_classes"]

    empty_embeddings = torch.zeros(emb_shape)
    model = Decoder(
        d_model=d_model,
        n_heads=model_config.n_heads,
        emb_weights=empty_embeddings,
        n_encoder_layers=model_config.n_encoder_layers,
        n_decoder_layers=model_config.n_decoder_layers,
        dropout=model_config.dropout,
        ff_multiplier=model_config.ff_multiplier,
        num_classes=num_classes,
    )

    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        raise RuntimeError(
            "Saved weights are incompatible with the current model architecture. "
            "Please retrain the model: "
            "python workflows/driver.py --device cuda --model decoder --type new --save True"
        ) from e

    return model
