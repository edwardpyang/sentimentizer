from __future__ import annotations

import json
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, HF_WEIGHTS_REPOS

logger = new_logger(DEFAULT_LOG_LEVEL)


# ---------------------------------------------------------------------------
# Model descriptions per model type
# ---------------------------------------------------------------------------

_MODEL_DESCRIPTIONS: dict[str, str] = {
    "rnn": (
        "A bidirectional LSTM for sentiment classification built on pre-trained GloVe "
        "embeddings. The model processes token sequences through a multi-layer "
        "bidirectional LSTM, concatenates the final forward and backward hidden "
        "states, and classifies via a two-layer MLP head."
    ),
    "encoder": (
        "A Transformer Encoder for sentiment classification built on pre-trained GloVe "
        "embeddings. The model uses multi-head self-attention with positional "
        "encodings and a classification token (CLS) to produce a sentiment score."
    ),
    "decoder": (
        "A Transformer Encoder-Decoder for sentiment classification built on "
        "pre-trained GloVe embeddings. The encoder processes the input sequence, "
        "and the decoder attends to the encoder outputs to produce a sentiment "
        "prediction."
    ),
    "modernbert": (
        "A ModernBERT-base model fine-tuned for 3-class sentiment classification "
        "(negative, neutral, positive). The backbone uses mean pooling over "
        "non-padding tokens followed by a two-layer classifier head with GELU "
        "activation. Supports 8-bit AdamW optimization and layer-wise unfreezing."
    ),
}

_HF_MODEL_TYPES = {"modernbert"}

_BACKBONE_FILENAMES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer.model",
]
_REQUIRED_BACKBONE_FILENAMES = {"config.json", "model.safetensors"}


def _copy_atomic(source: str | Path, destination: Path) -> None:
    """Copy a file so an interrupted copy never looks like a complete download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.partial")
    try:
        shutil.copy2(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_backbone_dir(repo_id: str, local_backbone_dir: Path, model_type: str) -> None:
    """Download the backbone directory from the Hugging Face Hub.

    Downloads all known HuggingFace backbone files (``backbone/*``) to
    *local_backbone_dir*.  Optional files absent from the repo are skipped since
    not every model ships every file.

    Args:
        repo_id: Hugging Face repository ID.
        local_backbone_dir: Local directory to write backbone files into.
        model_type: Model type string (for log messages).

    Raises:
        FileNotFoundError: If a required backbone file is absent from the repo.
        Exception: Any non-"not found" download failure propagates, because a
            partial backbone fails later with a far less useful error.
    """
    local_backbone_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading backbone directory from {repo_id}/backbone/ ...")

    for filename in _BACKBONE_FILENAMES:
        remote_path = f"backbone/{filename}"
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
            )
            _copy_atomic(downloaded, local_backbone_dir / filename)
            logger.info(f"  Downloaded {filename} -> {local_backbone_dir / filename}")
        except EntryNotFoundError as exc:
            if filename in _REQUIRED_BACKBONE_FILENAMES:
                raise FileNotFoundError(
                    f"Required backbone file {filename} not found in {repo_id}"
                ) from exc
            logger.debug(f"  Skipped {filename} (not found in repo)")

    logger.info(f"Backbone directory downloaded to {local_backbone_dir}")


def create_model_card(
    model_type: str,
    tuning_result: dict[str, Any] | None = None,
) -> str:
    """Generate a HuggingFace model card (README.md) for a Sentimentizer model.

    The model card includes YAML frontmatter, a description of the model
    architecture, usage instructions, and optionally the tuning metrics
    from a ``TuningRunResult``.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        tuning_result: Optional dict with tuning metrics (as produced by
            ``TuningRunResult`` or loaded from the saved JSON). If provided,
            the model card includes a metrics table.

    Returns:
        The model card content as a string.

    Raises:
        ValueError: If *model_type* is not recognized.
    """
    if model_type not in _MODEL_DESCRIPTIONS:
        valid_types = ", ".join(_MODEL_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown model type: {model_type!r}. Must be one of: {valid_types}")

    is_hf_model = model_type in _HF_MODEL_TYPES
    description = _MODEL_DESCRIPTIONS[model_type]
    repo_id = HF_WEIGHTS_REPOS.get(model_type, f"ryeyoo/sentimentizer-{model_type}")

    # ── YAML frontmatter ──
    yaml_block = (
        "---\n"
        "language: en\n"
        "license: mit\n"
        "tags:\n"
        "  - sentiment-analysis\n"
        "  - text-classification\n"
        f"  - {model_type}\n"
        "library_name: sentimentizer\n"
        "task: text-classification\n"
        "---\n"
    )

    # ── Title and description ──
    title = f"# Sentimentizer {model_type.upper()} Sentiment Model"
    desc_section = f"\n## Description\n\n{description}\n"

    # ── Training data ──
    if is_hf_model:
        training_section = (
            "\n## Training Data\n\n"
            "Trained on the [Yelp Open Dataset](https://www.yelp.com/dataset) reviews "
            "with 3-class labels (negative/neutral/positive). The ModernBERT tokenizer "
            "handles subword tokenization natively—no custom dictionary is needed.\n"
        )
    else:
        training_section = (
            "\n## Training Data\n\n"
            "Trained on the [Yelp Open Dataset](https://www.yelp.com/dataset) reviews, "
            "with GloVe Wiki-Gigaword-100 pre-trained embeddings. Reviews are tokenized "
            "with a custom dictionary (20k vocab, min frequency 3) and padded/truncated "
            "to 200 tokens.\n"
        )

    # ── Metrics section (from tuning result) ──
    metrics_section = ""
    if tuning_result is not None:
        metrics_section = _format_metrics_section(tuning_result)

    # ── Usage section ──
    usage_section = _format_usage_section(model_type, repo_id)

    # ── Files section ──
    if is_hf_model:
        files_section = (
            "\n## Files\n\n"
            f"- `{model_type}_weights.pth` — Classifier head weights and config metadata\n"
            "- `backbone/` — Hugging Face transformer backbone (safetensors + config)\n"
            "- `backbone/tokenizer.json` — Full tokenizer data\n"
            "- `backbone/tokenizer_config.json` — Tokenizer configuration\n"
        )
    else:
        files_section = (
            "\n## Files\n\n"
            f"- `{model_type}_weights.pth` — Model state dictionary\n"
            "- `yelp.dictionary` — Gensim dictionary for tokenization\n"
        )

    # ── Combine ──
    return (
        yaml_block
        + title
        + desc_section
        + training_section
        + metrics_section
        + usage_section
        + files_section
    )


def _format_metrics_section(tuning_result: dict[str, Any]) -> str:
    """Format tuning metrics into a Markdown section for the model card."""
    lines = ["\n## Metrics\n"]

    # Table of key metrics
    table_lines = [
        "| Metric | Value |",
        "|--------|------|",
    ]

    metric_keys = [
        ("best_accuracy", "Accuracy"),
        ("best_loss", "Loss"),
        ("best_balanced_accuracy", "Balanced Accuracy"),
        ("best_cohen_kappa", "Cohen's Kappa"),
        ("best_mcc", "MCC"),
        ("best_negative_f1", "Negative F1"),
        ("best_neutral_f1", "Neutral F1"),
        ("best_positive_f1", "Positive F1"),
        ("best_macro_f1", "Macro F1"),
        ("best_weighted_f1", "Weighted F1"),
    ]

    for key, label in metric_keys:
        val = tuning_result.get(key)
        if val is not None:
            if isinstance(val, float):
                table_lines.append(f"| {label} | {val:.4f} |")
            else:
                table_lines.append(f"| {label} | {val} |")

    lines.append("\n" + "\n".join(table_lines) + "\n")

    # Validation status
    validation_passed = tuning_result.get("validation_passed")
    if validation_passed is not None:
        status = "✅ Passed" if validation_passed else "❌ Failed"
        lines.append(f"\n**Validation:** {status}")

    # Tuning mode and iterations
    mode = tuning_result.get("mode", "")
    iterations = tuning_result.get("iterations_completed")
    converged = tuning_result.get("converged")
    elapsed = tuning_result.get("elapsed_seconds")

    meta_parts: list[str] = []
    if mode:
        meta_parts.append(f"Mode: `{mode}`")
    if iterations is not None:
        meta_parts.append(f"Iterations: {iterations}")
    if converged is not None:
        meta_parts.append(f"Converged: {'Yes' if converged else 'No'}")
    if elapsed is not None:
        meta_parts.append(f"Elapsed: {elapsed:.1f}s")

    if meta_parts:
        lines.append("\n" + " | ".join(meta_parts) + "\n")

    # Best config
    best_config = tuning_result.get("best_config")
    if best_config and isinstance(best_config, dict):
        config_lines = ["\n<details><summary>Best Configuration</summary>\n", "```json"]
        config_lines.append(json.dumps(best_config, indent=2))
        config_lines.append("```\n</details>\n")
        lines.extend(config_lines)

    return "".join(lines)


def _upload_model_card(
    api: HfApi,
    repo_id: str,
    model_type: str,
    tuning_result: dict[str, Any] | None,
) -> None:
    """Generate and upload a model card (README.md) to the Hugging Face Hub.

    Writes the model card content to a temporary file and uploads it
    as ``README.md`` to the repository.

    Args:
        api: HfApi instance for uploading.
        repo_id: Hugging Face repository ID.
        model_type: Model type for the model card.
        tuning_result: Optional tuning metrics to include.
    """
    try:
        model_card_content = create_model_card(model_type, tuning_result=tuning_result)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(model_card_content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Update {model_type} model card",
        )
        logger.info(f"Successfully pushed model card to {repo_id}/README.md")

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to push model card: {e}")
        # Clean up temp file on error too
        try:
            if "tmp_path" in dir():
                Path(tmp_path).unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass


def _format_usage_section(model_type: str, repo_id: str) -> str:
    """Format usage instructions for the model card."""
    is_hf_model = model_type in _HF_MODEL_TYPES

    if is_hf_model:
        return (
            "\n## Usage\n\n"
            "```python\n"
            "from sentimentizer.hf import download_weights\n"
            "from sentimentizer.config import weights_path_for\n\n"
            f"# Download classifier head + backbone from Hugging Face Hub\n"
            f'weights_path = weights_path_for("{model_type}")\n'
            f"download_weights(\n"
            f'    "{model_type}",\n'
            f"    weights_path,\n"
            f'    repo_id="{repo_id}",\n'
            f")\n\n"
            f"# Load and run inference\n"
            f"from sentimentizer.models.{model_type} import new_modernbert_model\n"
            f"from sentimentizer.predictor import SentimentPredictor\n\n"
            f"model = new_modernbert_model()\n"
            f"predictor = SentimentPredictor(model)\n\n"
            f"result = predictor.predict('amazing food great service')\n"
            f"print(result)\n"
            f"# e.g. {{'label': 'positive', 'score': 0.83, ...}}\n"
            "```\n"
        )

    return (
        "\n## Usage\n\n"
        "```python\n"
        "from sentimentizer.hf import download_weights\n"
        "from sentimentizer.config import DriverConfig, weights_path_for\n\n"
        f"# Download weights + dictionary from Hugging Face Hub\n"
        f'weights_path = weights_path_for("{model_type}")\n'
        f"download_weights(\n"
        f'    "{model_type}",\n'
        f"    weights_path,\n"
        f'    repo_id="{repo_id}",\n'
        f"    dict_path=DriverConfig.files.dictionary_file_path,\n"
        f")\n\n"
        f"# Load and run inference\n"
        f"from sentimentizer.models.{model_type} import get_trained_model\n"
        f"from sentimentizer.tokenizer import get_trained_tokenizer\n\n"
        f'model = get_trained_model(device="cpu")\n'
        f"tokenizer = get_trained_tokenizer()\n\n"
        f"probs = model.predict_text('amazing food great service')\n"
        f"for label, prob in sorted(probs.items(), key=lambda x: -x[1]):\n"
        f"    print(f'{{label}}: {{prob:.4f}}')\n"
        f"# e.g. positive: 0.8300, neutral: 0.1200, negative: 0.0500\n"
        "```\n"
    )


def _upload_backbone_dir(api: HfApi, backbone_dir: Path, repo_id: str, model_type: str) -> None:
    """Upload the HuggingFace backbone directory to the Hub.

    Uploads all files in *backbone_dir* under the ``backbone/`` prefix in the
    repository.  This includes ``model.safetensors``, ``config.json``, and
    tokenizer files.

    Args:
        api: HfApi instance for uploading.
        backbone_dir: Local path to the backbone directory (must exist).
        repo_id: Hugging Face repository ID.
        model_type: Model type string (for commit messages).

    Raises:
        FileNotFoundError: If *backbone_dir* does not exist, since weights without
            their backbone cannot be loaded.
    """
    if not backbone_dir.is_dir():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_dir}")

    logger.info(f"Pushing backbone directory to Hugging Face Hub: {repo_id}/backbone/...")
    for file_path in sorted(backbone_dir.iterdir()):
        if file_path.is_file():
            path_in_repo = f"backbone/{file_path.name}"
            logger.info(f"  Uploading {file_path.name} -> {repo_id}/{path_in_repo}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                commit_message=f"Update {model_type} backbone: {file_path.name}",
            )
    logger.info(f"Successfully pushed backbone directory to {repo_id}/backbone/")


def _upload_model_artifacts(
    api: HfApi,
    *,
    weights_path: Path,
    model_type: str,
    repo_id: str,
    dict_path: str | Path | None,
    tuning_result: dict[str, Any] | None,
    backbone_path: str | Path | None,
) -> None:
    """Upload weights, the HF backbone or dictionary, and the model card."""
    filename = f"{model_type}_weights.pth"
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo=filename,
        repo_id=repo_id,
        commit_message=f"Update {model_type} weights",
    )
    logger.info(f"Successfully pushed weights to {repo_id}/{filename}")

    if model_type in _HF_MODEL_TYPES:
        backbone = Path(backbone_path) if backbone_path is not None else None
        _upload_backbone_dir(api, backbone or weights_path.parent / "backbone", repo_id, model_type)

    if dict_path is not None:
        dict_file = Path(dict_path)
        if dict_file.exists():
            logger.info(f"Pushing dictionary to Hugging Face Hub: {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(dict_file),
                path_in_repo=dict_file.name,
                repo_id=repo_id,
                commit_message=f"Update dictionary for {model_type}",
            )
            logger.info(f"Successfully pushed dictionary to {repo_id}/{dict_file.name}")
        else:
            logger.error(f"Dictionary file not found: {dict_file}")

    _upload_model_card(api, repo_id, model_type, tuning_result)


def push_model_to_hub(
    local_path: str | Path,
    model_type: str,
    repo_id: str | None = None,
    dict_path: str | Path | None = None,
    tuning_result: dict[str, Any] | None = None,
    backbone_path: str | Path | None = None,
) -> bool:
    """Upload model weights, dictionary, and model card to the Hugging Face Hub.

    If *repo_id* is provided, it uploads to that repository.
    Otherwise, it looks up the per-model repository from ``HF_WEIGHTS_REPOS``.

    For Hugging Face transformer models (e.g. ModernBERT), also uploads the
    ``backbone/`` directory containing the pre-trained weights and tokenizer.

    When *tuning_result* is provided, a model card (README.md) is generated
    with the tuning metrics and uploaded alongside the weights.

    Args:
        local_path: Path to the local .pth weight file.
        model_type: Model type ('rnn', 'encoder', 'decoder', or 'modernbert').
        repo_id: Optional Hugging Face repository ID.
        dict_path: Optional path to the dictionary file to also upload.
        tuning_result: Optional dict with tuning metrics to include in the model card.
        backbone_path: Optional path to the backbone directory (for HF models).
            If not provided and the model type is an HF model, defaults to
            ``<local_path_parent>/backbone/``.

    Returns:
        True if the weights and required artifacts were uploaded, False otherwise.
    """
    if repo_id is None:
        repo_id = HF_WEIGHTS_REPOS.get(model_type)

    if repo_id is None:
        logger.error(f"No Hugging Face repo configured for model type {model_type!r}")
        return False

    path = Path(local_path)
    if not path.exists():
        logger.error(f"Local weight file not found: {path}")
        return False

    api = HfApi()
    upload = partial(
        _upload_model_artifacts,
        api,
        weights_path=path,
        model_type=model_type,
        repo_id=repo_id,
        dict_path=dict_path,
        tuning_result=tuning_result,
        backbone_path=backbone_path,
    )

    logger.info(f"Pushing {model_type}_weights.pth to Hugging Face Hub: {repo_id}...")
    try:
        upload()
        return True
    except Exception as e:
        if "Repository Not Found" not in str(e) and "404 Client Error" not in str(e):
            logger.error(f"Failed to push weights to Hugging Face Hub: {e}")
            return False

    logger.info(f"Repository {repo_id} not found. Attempting to create it...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        upload()
        return True
    except Exception as create_error:
        logger.error(f"Failed to create repository or retry upload: {create_error}")
        return False


def pull_model_from_hub(
    repo_id: str,
    model_type: str,
    local_path: str | Path,
    dict_path: str | Path | None = None,
) -> bool:
    """Download model weights and optionally the dictionary from the Hugging Face Hub.

    For Hugging Face transformer models (e.g. ModernBERT), also downloads the
    ``backbone/`` directory containing pre-trained weights and tokenizer files.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/repo').
        model_type: Model type ('rnn', 'encoder', 'decoder', or 'modernbert').
        local_path: Local path where the weights should be saved.
        dict_path: Optional local path where the dictionary should be saved.

    Returns:
        True if the weight download was successful, False otherwise.
    """
    return download_weights(model_type, local_path, repo_id, dict_path) is not None


def download_weights(
    model_type: str,
    local_path: str | Path,
    repo_id: str | None = None,
    dict_path: str | Path | None = None,
) -> Path | None:
    """Download model weights and optionally the dictionary from the Hugging Face Hub.

    If *repo_id* is provided, it downloads from that repository.
    Otherwise, it looks up the per-model repository from ``HF_WEIGHTS_REPOS``.

    For Hugging Face transformer models (e.g. ModernBERT), also downloads the
    ``backbone/`` directory containing pre-trained weights and tokenizer files.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder', or 'modernbert'.
        local_path: Destination path on the local filesystem.
        repo_id: Optional Hugging Face repository ID.
        dict_path: Optional destination path for the dictionary on the local filesystem.

    Returns:
        The path to the downloaded weights file, or ``None`` if the download
        failed (e.g. the repo or file doesn't exist, or there is no network).
    """
    if repo_id is None:
        repo_id = HF_WEIGHTS_REPOS.get(model_type)

    if repo_id is None:
        logger.error(f"No Hugging Face repo configured for model type {model_type!r}")
        return None

    local_path = Path(local_path)
    filename = f"{model_type}_weights.pth"
    is_hf_model = model_type in _HF_MODEL_TYPES

    logger.info(f"Downloading {filename} from {repo_id} ...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        # Weights go last: callers treat the weights file as proof the download is
        # complete, and skip downloading if it exists.
        if is_hf_model:
            backbone_dir = local_path.parent / "backbone"
            _download_backbone_dir(repo_id, backbone_dir, model_type)

        _copy_atomic(downloaded_path, local_path)
        logger.info(f"Weights downloaded to {local_path}")

        if dict_path is not None:
            dict_file = Path(dict_path)
            try:
                logger.info(f"Downloading dictionary from {repo_id} ...")
                downloaded_dict = hf_hub_download(
                    repo_id=repo_id,
                    filename=dict_file.name,
                )
                _copy_atomic(downloaded_dict, dict_file)
                logger.info(f"Dictionary downloaded to {dict_file}")
            except EntryNotFoundError:
                logger.warning(f"Dictionary file {dict_file.name} not found in {repo_id}")
            except Exception as e:
                logger.error(f"Failed to download dictionary from Hugging Face Hub: {e}")

        return local_path

    except EntryNotFoundError:
        logger.warning(f"Weights file {filename} not found in {repo_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to download weights from Hugging Face Hub: {e}")
        return None
