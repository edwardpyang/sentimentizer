import os
from dataclasses import dataclass
from logging import INFO

import numpy as np

from sentimentizer import root
from sentimentizer.device import resolve_device as auto_detect_device  # noqa: F401 — re-export

data_path = os.path.join(root, "sentimentizer")
external_data = root.parent / "data"  # data/ one level above project root

DEFAULT_LOG_LEVEL = INFO

# Embedding constants
EMBEDDING_DTYPE = np.float32
EMBEDDING_RANDOM_MEAN: float = 0.0
EMBEDDING_RANDOM_STD: float = 0.32

# Classification constants
NUM_CLASSES: int = 3
LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]

# Logging constants
EXTRACT_LOG_INTERVAL: int = 100000


def default_epochs(model_type: str) -> int:
    """Return default epochs for the model type.

    RNNs need enough epochs to learn compositional patterns like negation.
    Transformers need more epochs for attention patterns to develop.
    """
    if model_type == "modernbert":
        return 3
    if model_type in ("encoder", "decoder"):
        return 12
    return 4


def default_dataloader_workers(device: str) -> int:
    """Return optimal DataLoader workers for the device.

    MPS has issues with multiprocessing, so use 0 workers.
    CUDA and CPU benefit from multiple workers for data loading.
    """
    if device == "mps":
        return 0
    return min(os.cpu_count() or 4, 10)


# Valid device strings for model loading (excludes "auto" — must be resolved first)
VALID_DEVICES: frozenset[str] = frozenset(("cpu", "cuda", "mps"))


@dataclass
class OptimizationParams:
    """Default optimization params (used for RNN).

    Encoder and Decoder models override these via their config.
    """

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4


@dataclass
class EncoderOptimizationParams(OptimizationParams):
    """Optimization params for Transformer models (Encoder, Decoder).

    Uses lower LR and AdamW-style weight decay for stable training.
    Inherits betas from OptimizationParams; overrides lr and weight_decay.
    """

    lr: float = 0.0003
    weight_decay: float = 0.01


@dataclass
class DecoderOptimizationParams(OptimizationParams):
    """Optimization params for Decoder model.

    The decoder has more parameters than the encoder due to cross-attention
    layers, so it uses a lower LR and higher weight decay to prevent
    overfitting on the sentiment classification task.
    Inherits betas from OptimizationParams; overrides lr and weight_decay.
    """

    lr: float = 0.0003
    weight_decay: float = 0.02


@dataclass
class ModernBERTOptimizationParams(OptimizationParams):
    """Optimization params for ModernBERT.

    Uses lower LR (2e-5) and AdamW-style weight decay (0.01).
    """

    lr: float = 2e-5
    weight_decay: float = 0.01


@dataclass
class SchedulerParams:
    """Base scheduler params.

    T_max, warmup_epochs, and last_epoch are legacy fields from when
    per-epoch stepping was used. With per-batch stepping, the scheduler
    is rebuilt in Trainer.fit() / _train_func() once the DataLoader
    length is known, using warmup_ratio and dynamically computed
    total_steps. These legacy fields are kept for checkpoint compat.
    """

    T_max: int = 4
    eta_min: float = 1e-6
    last_epoch: int = -1
    warmup_epochs: int = 0
    warmup_ratio: float = 0.06


@dataclass
class RNNSchedulerParams(SchedulerParams):
    """Scheduler params for RNN model.

    Uses per-batch warmup+cosine decay. warmup_ratio controls what fraction
    of total optimizer steps are spent warming up. T_max and warmup_epochs
    are inherited but unused — the per-batch scheduler is rebuilt in
    Trainer.fit() / _train_func() once the DataLoader length is known.
    """

    warmup_ratio: float = 0.06
    eta_min: float = 1e-6


@dataclass
class EncoderSchedulerParams(SchedulerParams):
    """Scheduler params for Encoder model.

    Uses per-batch warmup+cosine decay via warmup_ratio. T_max and
    warmup_epochs are legacy fields from per-epoch stepping; the
    per-batch scheduler is rebuilt in Trainer.fit() / _train_func()
    once the DataLoader length is known.
    """

    T_max: int = 24
    warmup_epochs: int = 1
    warmup_ratio: float = 0.06
    eta_min: float = 1e-6


@dataclass
class DecoderSchedulerParams(SchedulerParams):
    """Scheduler params for Decoder model.

    Uses per-batch warmup+cosine decay via warmup_ratio. Higher eta_min
    than encoder because the decoder has more parameters and is more
    prone to overfitting during early training. T_max and warmup_epochs
    are legacy fields from per-epoch stepping.
    """

    T_max: int = 24
    eta_min: float = 1e-5
    warmup_epochs: int = 1
    warmup_ratio: float = 0.06


@dataclass
class ModernBERTSchedulerParams(SchedulerParams):
    """Scheduler params for ModernBERT model.

    Uses per-batch (per-optimizer-step) scheduler stepping via
    STEP_SCHEDULER_PER_BATCH=True. warmup_ratio controls what fraction
    of total optimizer steps are spent warming up; total_steps is computed
    dynamically from dataset size at training time.

    T_max and warmup_epochs are inherited but unused for this model —
    the per-batch scheduler is rebuilt in Trainer.fit() / _train_func()
    once the DataLoader length is known.
    """

    warmup_ratio: float = 0.06
    eta_min: float = 1e-6


@dataclass(frozen=True)
class TokenizerConfig:
    text_col: str = "tokens"
    label_col: str = "stars"
    inputs: str = "data"
    labels: str = "target"
    stop: int = 10000
    max_len: int = 200
    dict_min: int = 3
    dict_keep: int = 20000
    no_above: float = 0.99999
    save_dictionary: bool = True
    include_neutral: bool = True  # include 3-star reviews (3-class) or drop them (binary)


@dataclass(frozen=True)
class FileConfig:
    archive_file_path: str = str(external_data / "yelp_dataset.tar")
    raw_file_path: str = "yelp_academic_dataset_review.json"
    dictionary_file_path: str = f"{data_path}/data/yelp.dictionary"
    raw_reviews_file_path: str = f"{data_path}/data/review_data_raw.parquet"
    processed_reviews_file_path: str = f"{data_path}/data/review_data.parquet"
    hf_processed_reviews_file_path: str = f"{data_path}/data/review_data_hf.parquet"
    rnn_weights_file_path: str = f"{data_path}/data/rnn_weights.pth"
    encoder_weights_file_path: str = f"{data_path}/data/encoder_weights.pth"
    decoder_weights_file_path: str = f"{data_path}/data/decoder_weights.pth"
    hf_weights_dir: str = f"{data_path}/data/hf_weights"


def weights_path_for(model_type: str) -> str:
    """Return the weights file path for the given model type.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder', 'modernbert'.

    Returns:
        Absolute path to the weights file for that model type.
    """
    if model_type == "rnn":
        return FileConfig.rnn_weights_file_path
    elif model_type == "encoder":
        return FileConfig.encoder_weights_file_path
    elif model_type == "decoder":
        return FileConfig.decoder_weights_file_path
    elif model_type == "modernbert":
        return f"{FileConfig.hf_weights_dir}/head.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


@dataclass
class TrainerConfig:
    batch_size: int = 8
    epochs: int = -1  # -1 = model-specific default (3/4/12 for modernbert/rnn/transformer)
    early_stopping_patience: int = 3  # stop if val_loss doesn't improve for this many epochs
    dataloader_workers: int = -1  # -1 means auto-detect based on device
    ray_workers: int = 1  # Ray Train workers (only used with --distributed)
    device: str = "auto"  # "auto" detects best available: cuda > mps > cpu
    memory: bool = True
    checkpoint_dir: str = ""  # directory to save checkpoints (empty = no checkpointing)
    checkpoint_every: int = 1  # save checkpoint every N epochs (0 = disabled)
    checkpoint_best: bool = True  # save the best model (lowest val loss) separately
    class_weights: list[float] | None = None  # per-class weights for CrossEntropyLoss
    balance_strategy: str = "class_weights_only"  # undersample/oversample/class_weights_only
    weight_smoothing: float = 0.5  # inverse-frequency exponent (1.0=full, 0.5=sqrt, 0.0=uniform)
    loss_type: str = "focal"  # "cross_entropy" or "focal" (focal is default for 3-class)
    focal_gamma: float = 2.0  # focal loss focusing parameter (loss_type="focal" only)
    label_smoothing: float = 0.1  # softens hard targets, reduces overconfident predictions
    neutral_oversample_ratio: float = 0.0  # 0.0=disabled, 0.20=neutral to 20% of data
    use_amp: bool = True  # mixed-precision training (bfloat16/CUDA, float32/CPU)
    gradient_accumulation_steps: int = (
        8  # effectively increases batch size for VRAM-constrained training
    )
    use_8bit_optimizer: bool = True  # bitsandbytes AdamW8bit (falls back to AdamW)
    ray_update_every: int = (
        -1
    )  # -1 means auto-detect based on model type (e.g. 100 for modernbert, 500 for others)
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            import datetime
            import random

            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            rand = random.randint(1000, 9999)
            self.run_id = f"run_{now}_{rand}"


@dataclass
class EmbeddingsConfig:
    model_name: str = "glove-wiki-gigaword-300"  # auto-downloaded via gensim.downloader
    emb_length: int = 300


# Per-model Hugging Face Hub repository IDs for pre-trained weights.
# Each model type maps to its own repo; weights are stored as
# ``{model_type}_weights.pth`` inside the repo.
HF_WEIGHTS_REPOS: dict[str, str] = {
    "rnn": "ryeyoo/sentimentizer-rnn",
    "encoder": "ryeyoo/sentimentizer-encoder",
    "decoder": "ryeyoo/sentimentizer-decoder",
    "modernbert": "ryeyoo/sentimentizer-modernbert",
}

# Hugging Face Hub repository for the pre-trained router model.
# The router uses snapshot_download (whole-repo) rather than individual
# file downloads, so it lives outside HF_WEIGHTS_REPOS.
HF_ROUTER_REPO: str = "ryeyoo/sentimentizer-router"
# Auto-download only this exact commit, so a later upload to the repo cannot change what
# deployments load. Update it after each `make upload-router`.
HF_ROUTER_REVISION: str = "eabc9b71594063948f1f6da3ecc28d0ce02d41f8"


@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face Hub interactions.

    Attributes:
        repo_id: Default repo ID (used by push_model_to_hub).
            Per-model repos are defined in HF_WEIGHTS_REPOS above.
    """

    repo_id: str = "ryeyoo/sentimentizer"


@dataclass(frozen=True)
class RNNConfig:
    """Configuration for the RNN model architecture."""

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 3


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the Transformer Encoder model architecture."""

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.2
    ff_multiplier: int = 4  # dim_feedforward = d_model * ff_multiplier
    num_classes: int = 3


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for the Transformer Decoder model architecture.

    The decoder uses fewer layers and higher dropout than the encoder
    because the cross-attention mechanism (2 encoder + 2 decoder = 4
    layers total) adds significant capacity.  With the original 2+4
    configuration the model had ~5.9M parameters and overfit quickly.
    """

    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    dropout: float = 0.3
    ff_multiplier: int = 4
    num_classes: int = 3


@dataclass(frozen=True)
class ModernBERTConfig:
    """Configuration for the ModernBERT model architecture."""

    model_name: str = "answerdotai/ModernBERT-base"
    dropout: float = 0.1
    num_classes: int = 3
    max_seq_length: int = 512
    freeze_backbone_epochs: int = 0
    gradient_checkpointing: bool = True  # saves ~60% activation VRAM at ~30% compute cost


@dataclass
class DriverConfig:
    files: type[FileConfig] = FileConfig
    embeddings: type[EmbeddingsConfig] = EmbeddingsConfig
    tokenizer: type[TokenizerConfig] = TokenizerConfig
    trainer: type[TrainerConfig] = TrainerConfig
    rnn: type[RNNConfig] = RNNConfig
    encoder: type[EncoderConfig] = EncoderConfig
    decoder: type[DecoderConfig] = DecoderConfig
    modernbert: type[ModernBERTConfig] = ModernBERTConfig
    hf: type[HuggingFaceConfig] = HuggingFaceConfig


def validate_config_consistency(config: DriverConfig) -> None:
    """Validate that include_neutral and num_classes settings are consistent.

    Raises ValueError if include_neutral=True but num_classes!=3, or
    include_neutral=False but num_classes!=2.
    """
    tokenizer_cfg = config.tokenizer
    model_configs = [
        ("rnn", config.rnn),
        ("encoder", config.encoder),
        ("decoder", config.decoder),
        ("modernbert", config.modernbert),
    ]
    for model_type, model_cfg in model_configs:
        nc = model_cfg.num_classes
        if tokenizer_cfg.include_neutral and nc != 3:
            raise ValueError(
                f"include_neutral=True requires num_classes=3, got {nc} for {model_type}"
            )
        if not tokenizer_cfg.include_neutral and nc != 2:
            raise ValueError(
                f"include_neutral=False requires num_classes=2, got {nc} for {model_type}"
            )
