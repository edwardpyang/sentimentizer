"""Tune stage: hyperparameter tuning (agent or standalone Ray Tune).

Heavy imports (torch, ray, config) are DEFERRED to function bodies to avoid
importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

from workflows.lifecycle import State, _cuda_cleanup, _ensure_ray_initialized, logger


def run_tune(
    state: State,
    *,
    mode: str,
    agent_config: str | None,
    tune_samples: int | None,
    tune_max_iterations: int | None,
    tune_output_dir: str,
    no_validate: bool,
    validation_threshold: float,
    max_retries: int,
    save: bool,
    push_to_hub: bool,
    balance_classes: bool,
    balance_seed: int,
    weight_smoothing: float = 0.5,
    loss_type: str = "cross_entropy",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    neutral_oversample_ratio: float = 0.0,
    balance_strategy: str = "class_weights_only",
) -> None:
    """Hyperparameter tuning (LLM-guided agent or standalone Ray Tune run)."""
    from workflows.lifecycle import is_ray_available

    if not is_ray_available():
        raise ImportError("Ray is required for tuning. Install with: pip install '.[ray]'")
    _ensure_ray_initialized()

    from sentimentizer.agent.diagnose_model import TuningRun, TuningRunConfig
    from sentimentizer.device import resolve_device

    device = resolve_device(state.device)

    config = TuningRunConfig(
        model_type=state.model,
        mode=mode,
        config_path=agent_config,
        num_samples=tune_samples,
        max_iterations=tune_max_iterations,
        device=device,
        save_best_model=save,
        output_dir=tune_output_dir,
        validate_predictions=not no_validate,
        validation_threshold=validation_threshold,
        max_retries=max_retries,
        balance_classes=balance_classes,
        balance_seed=balance_seed,
        weight_smoothing=weight_smoothing,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        neutral_oversample_ratio=neutral_oversample_ratio,
        balance_strategy=balance_strategy,
        push_to_hub=push_to_hub,
    )

    run = TuningRun(config)

    logger.info(  # type: ignore[call-arg]
        "starting_tuning_run",
        model_type=config.model_type,
        mode=config.mode,
        device=config.device,
        validate=config.validate_predictions,
        max_retries=config.max_retries,
    )

    try:
        result = run.execute()
    finally:
        _cuda_cleanup()

    logger.info(  # type: ignore[call-arg]
        "tuning_run_complete",
        best_accuracy=result.best_accuracy,
        best_loss=result.best_loss,
        best_f1=result.best_positive_f1,
        best_cohen_kappa=result.best_cohen_kappa,
        best_negative_f1=result.best_negative_f1,
        best_neutral_f1=result.best_neutral_f1,
        best_positive_f1=result.best_positive_f1,
        best_macro_f1=result.best_macro_f1,
        best_weighted_f1=result.best_weighted_f1,
        best_mcc=result.best_mcc,
        best_balanced_accuracy=result.best_balanced_accuracy,
        iterations=result.iterations_completed,
        converged=result.converged,
        validation_passed=result.validation_passed,
        retry_count=result.retry_count,
        model_path=result.model_path,
        results_path=result.results_path,
        elapsed_seconds=result.elapsed_seconds,
    )

    if result.validation_passed:
        logger.info("tuning_run_passed: model predictions validated successfully")
    else:
        logger.warning(
            "tuning_run_failed: model predictions did not meet validation threshold "
            f"({config.validation_threshold}) after {result.retry_count} retries"
        )
