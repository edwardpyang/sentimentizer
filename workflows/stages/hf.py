"""HuggingFace Hub push/pull stage.

Heavy imports are DEFERRED to function bodies to avoid importing the ML stack
at module level. Do NOT add module-level imports of torch, ray, gensim, or
sentimentizer.config here.
"""

from __future__ import annotations

from workflows.lifecycle import State, logger


def run_hf_push(state: State, *, repo_id: str | None) -> None:
    """Push model weights to Hugging Face Hub."""
    from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig, weights_path_for
    from sentimentizer.hf import _HF_MODEL_TYPES, push_model_to_hub

    weights_path = weights_path_for(state.model)
    resolved_repo = repo_id if repo_id is not None else HF_WEIGHTS_REPOS.get(state.model)

    is_hf_model = state.model in _HF_MODEL_TYPES

    pushed = push_model_to_hub(
        local_path=weights_path,
        model_type=state.model,
        repo_id=resolved_repo,
        dict_path=None if is_hf_model else DriverConfig.files.dictionary_file_path,
    )
    if not pushed:
        raise RuntimeError(f"Failed to push {state.model} weights to HF Hub")


def run_hf_pull(state: State, *, repo_id: str | None) -> None:
    """Pull model weights from Hugging Face Hub."""
    from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig, weights_path_for
    from sentimentizer.hf import _HF_MODEL_TYPES, download_weights

    weights_path = weights_path_for(state.model)
    resolved_repo = repo_id if repo_id is not None else HF_WEIGHTS_REPOS.get(state.model)

    is_hf_model = state.model in _HF_MODEL_TYPES

    result_path = download_weights(
        model_type=state.model,
        local_path=weights_path,
        repo_id=resolved_repo,
        dict_path=None if is_hf_model else DriverConfig.files.dictionary_file_path,
    )
    if not result_path:
        raise RuntimeError(f"Failed to pull {state.model} weights from HF Hub")
    logger.info(f"Pulled {state.model} weights from HF Hub to {result_path}")
