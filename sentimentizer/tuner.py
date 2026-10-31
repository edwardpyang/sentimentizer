"""Ray Tune + Optuna hyperparameter search for sentimentizer models.

Translates the YAML-driven search space configuration into Ray Tune
search spaces, runs trials with ASHA scheduling, and returns the best
configuration found.

This module has NO dependency on the agent package — it is a standalone
tuning library that the agent calls into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from ray import tune
    from ray.tune import CLIReporter, RunConfig
    from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler, MedianStoppingRule
    from ray.tune.search.optuna import OptunaSearch

    TuneCallback = tune.callback.Callback
except ImportError:
    TuneCallback = object

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


# ---------------------------------------------------------------------------
# Prometheus callback for Ray Tune
# ---------------------------------------------------------------------------


class TunePrometheusCallback(TuneCallback):
    """Ray Tune callback that pushes trial metrics to Prometheus gauges.

    This callback updates the ``sentimentizer_tune_*`` Prometheus gauges
    defined in ``sentimentizer/exporter.py`` so that Grafana can display
    real-time tuning progress alongside system and Ray metrics.

    Inherits from ``ray.tune.Callback`` to satisfy the callback interface
    contract (including the ``setup`` method that Ray calls during tuning
    initialization).

    Usage::

        from sentimentizer.tuner import TunePrometheusCallback

        callback = TunePrometheusCallback(model_type="rnn")
        result = tune.Tuner(
            ...,
            run_config=RunConfig(callbacks=[callback]),
        ).fit()
    """

    def __init__(self, model_type: str) -> None:
        super().__init__()
        self.model_type = model_type
        self._completed_trials: int = 0
        self._total_trials: int = 0
        self._best_accuracy: float = 0.0
        self._best_loss: float = float("inf")
        self._best_f1: float = 0.0

    def _update_trial_gauges(self, trial_id: str, metrics: dict) -> None:
        """Push per-trial metrics to Prometheus gauges."""
        try:
            from sentimentizer.exporter import (
                TUNE_TRIAL_COMPLETED_COUNT,
                TUNE_TRIAL_COUNT,
                TUNE_TRIAL_EPOCH,
                TUNE_TRIAL_TRAIN_LOSS,
                TUNE_TRIAL_VAL_ACCURACY,
                TUNE_TRIAL_VAL_BALANCED_ACCURACY,
                TUNE_TRIAL_VAL_COHEN_KAPPA,
                TUNE_TRIAL_VAL_LOSS,
                TUNE_TRIAL_VAL_MACRO_F1,
                TUNE_TRIAL_VAL_MCC,
                TUNE_TRIAL_VAL_NEGATIVE_F1,
                TUNE_TRIAL_VAL_NEUTRAL_F1,
                TUNE_TRIAL_VAL_POSITIVE_F1,
                TUNE_TRIAL_VAL_WEIGHTED_F1,
            )

            TUNE_TRIAL_VAL_ACCURACY.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_accuracy", 0.0)
            )
            TUNE_TRIAL_VAL_LOSS.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_loss", float("inf"))
            )
            TUNE_TRIAL_TRAIN_LOSS.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("train_loss", 0.0)
            )
            TUNE_TRIAL_VAL_COHEN_KAPPA.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_cohen_kappa", 0.0)
            )
            positive_f1 = metrics.get("val_positive_f1", 0.0)
            TUNE_TRIAL_VAL_BALANCED_ACCURACY.labels(
                trial_id=trial_id, model_type=self.model_type
            ).set(metrics.get("val_balanced_accuracy", 0.0))
            TUNE_TRIAL_VAL_NEGATIVE_F1.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_negative_f1", 0.0)
            )
            TUNE_TRIAL_VAL_NEUTRAL_F1.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_neutral_f1", 0.0)
            )
            TUNE_TRIAL_VAL_POSITIVE_F1.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_positive_f1", 0.0)
            )
            TUNE_TRIAL_VAL_MACRO_F1.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_macro_f1", 0.0)
            )
            TUNE_TRIAL_VAL_WEIGHTED_F1.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_weighted_f1", 0.0)
            )
            TUNE_TRIAL_VAL_MCC.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("val_mcc", 0.0)
            )
            TUNE_TRIAL_EPOCH.labels(trial_id=trial_id, model_type=self.model_type).set(
                metrics.get("epoch", 0)
            )

            # Track best metrics across trials
            val_accuracy = metrics.get("val_accuracy", 0.0)
            val_loss = metrics.get("val_loss", float("inf"))
            positive_f1 = metrics.get("val_positive_f1", 0.0)

            if val_accuracy > self._best_accuracy:
                self._best_accuracy = val_accuracy
                from sentimentizer.exporter import TUNE_BEST_VAL_ACCURACY

                TUNE_BEST_VAL_ACCURACY.labels(model_type=self.model_type).set(self._best_accuracy)

            if val_loss < self._best_loss:
                self._best_loss = val_loss
                from sentimentizer.exporter import TUNE_BEST_VAL_LOSS

                TUNE_BEST_VAL_LOSS.labels(model_type=self.model_type).set(self._best_loss)

            if positive_f1 > self._best_f1:
                self._best_f1 = positive_f1
                from sentimentizer.exporter import TUNE_BEST_VAL_F1

                TUNE_BEST_VAL_F1.labels(model_type=self.model_type).set(self._best_f1)

            TUNE_TRIAL_COUNT.labels(model_type=self.model_type).set(self._total_trials)
            TUNE_TRIAL_COMPLETED_COUNT.labels(model_type=self.model_type).set(
                self._completed_trials
            )

        except ImportError:
            logger.warning(
                "prometheus_client_not_available", message="Cannot push tune metrics to Prometheus"
            )

    # Ray Tune callback interface methods
    # Ray dispatches these hooks by keyword (CallbackList.on_trial_*(**info)), so the
    # parameter names are part of its interface and must not be renamed.
    def on_trial_result(
        self,
        iteration: int,
        trials: list,
        trial: Any,
        result: dict,
        **info: Any,
    ) -> None:
        """Called after each trial reports intermediate results.

        In Ray 2.55+, the ``trial`` parameter is a ``Trial`` object with a
        ``trial_id`` attribute — it is no longer a plain string.
        """
        # Ray 2.55+ passes a Trial object with .trial_id; fall back for older versions
        trial_id: str = getattr(trial, "trial_id", None) or (
            trial if isinstance(trial, str) else str(trial)
        )
        self._update_trial_gauges(trial_id, result)

    def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        """Called when a trial completes."""
        self._completed_trials += 1
        try:
            from sentimentizer.exporter import TUNE_TRIAL_COMPLETED_COUNT

            TUNE_TRIAL_COMPLETED_COUNT.labels(model_type=self.model_type).set(
                self._completed_trials
            )
        except ImportError:
            pass

    def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        """Called when a trial starts."""
        self._total_trials = len(trials) if trials else 0
        try:
            from sentimentizer.exporter import TUNE_TRIAL_COUNT

            TUNE_TRIAL_COUNT.labels(model_type=self.model_type).set(self._total_trials)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TunerConfig:
    """Ray Tune + Optuna hyperparameter search configuration."""

    scheduler: str = "asha"
    metric: str = "val_accuracy"
    mode: str = "max"
    num_samples: int = 20
    grace_period: int = 2
    reduction_factor: int = 3
    search_spaces: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict,
    )


def _get_default_config_path() -> Path:
    """Return path to the default config.yaml bundled with the agent package."""
    return Path(__file__).parent / "agent" / "config.yaml"


def load_tuner_config(path: str | Path | None = None) -> TunerConfig:
    """Load tuner configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, uses the default
              config.yaml bundled with the package.

    Returns:
        TunerConfig with all settings populated.
    """
    path = _get_default_config_path() if path is None else Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return TunerConfig(**raw.get("tuner", {}))


def load_search_space(
    model_type: str,
    tuner_config: TunerConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the raw search space for a specific model type.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        tuner_config: Optional pre-loaded TunerConfig. If None, loads
                      from config_path.
        config_path: Path to YAML config file (only used if
                     tuner_config is None).

    Returns:
        Dict mapping parameter names to their search space specs,
        e.g. {'lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2}}

    Raises:
        ValueError: If model_type is not found in the search spaces.
    """
    if tuner_config is None:
        tuner_config = load_tuner_config(config_path)

    if model_type not in tuner_config.search_spaces:
        available = list(tuner_config.search_spaces.keys())
        raise ValueError(
            f"No search space defined for model type '{model_type}'. Available: {available}"
        )

    return tuner_config.search_spaces[model_type]


# ---------------------------------------------------------------------------
# Search space builder
# ---------------------------------------------------------------------------


def build_search_space(
    model_type: str,
    overrides: dict[str, dict[str, Any]] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Ray Tune search space dict from the YAML config.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        overrides: Optional dict of parameter overrides from the agent.
                   Keys are parameter names, values are search space dicts
                   (same format as YAML). These override the YAML defaults.
        config_path: Path to config YAML (uses default if None).

    Returns:
        Dict mapping parameter names to Ray Tune search space objects.
    """
    from ray import tune

    raw_space = load_search_space(model_type, config_path=config_path)

    # Apply agent overrides (narrowing/widening the search space)
    if overrides:
        for param_name, override_spec in overrides.items():
            if param_name in raw_space:
                raw_space[param_name] = override_spec

    search_space: dict[str, Any] = {}
    for param_name, spec in raw_space.items():
        param_type = spec["type"]

        if param_type == "loguniform":
            search_space[param_name] = tune.loguniform(spec["low"], spec["high"])
        elif param_type == "uniform":
            search_space[param_name] = tune.uniform(spec["low"], spec["high"])
        elif param_type == "choice":
            search_space[param_name] = tune.choice(spec["values"])
        elif param_type == "randint":
            search_space[param_name] = tune.randint(spec["low"], spec["high"])
        else:
            raise ValueError(f"Unknown search space type: {param_type}")

    return search_space


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def _get_scheduler(
    tuner_config: TunerConfig,
    max_epochs: int,
) -> ASHAScheduler | HyperBandScheduler | MedianStoppingRule:
    """Create the appropriate Ray Tune scheduler from config.

    Args:
        tuner_config: Tuner configuration.
        max_epochs: Epochs per trial. Each trial reports once per epoch, so this is the
            scheduler's ``max_t`` in its ``training_iteration`` unit.
    """
    if tuner_config.scheduler == "asha":
        return ASHAScheduler(
            time_attr="training_iteration",
            metric=tuner_config.metric,
            mode=tuner_config.mode,
            max_t=max_epochs,
            grace_period=tuner_config.grace_period,
            reduction_factor=tuner_config.reduction_factor,
        )
    elif tuner_config.scheduler == "hyperband":
        return HyperBandScheduler(
            time_attr="training_iteration",
            metric=tuner_config.metric,
            mode=tuner_config.mode,
            max_t=max_epochs,
            reduction_factor=tuner_config.reduction_factor,
        )
    elif tuner_config.scheduler == "median":
        return MedianStoppingRule(
            time_attr="training_iteration",
            metric=tuner_config.metric,
            mode=tuner_config.mode,
            grace_period=tuner_config.grace_period,
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {tuner_config.scheduler}. Use 'asha', 'hyperband', or 'median'."
        )


# ---------------------------------------------------------------------------
# Ray Tune trainable
# ---------------------------------------------------------------------------


def _trainable_wrapper(config: dict, train_dataset: Any = None, val_dataset: Any = None) -> None:
    """Ray Tune trainable that wraps sentimentizer training.

    This function is executed by each Ray Tune trial. It:
    1. Constructs a model with the trial's hyperparameters
    2. Trains for one epoch at a time (reporting metrics per epoch)
    3. Reports validation accuracy and loss back to Ray Tune

    The config dict must contain:
    - All model hyperparameters from the search space
    - 'model_type': str
    - 'device': str
    - 'dict_path': str
    - 'embeddings_model_name': str
    - 'embeddings_emb_length': int
    """
    from ray import tune

    from sentimentizer.config import DriverConfig, EmbeddingsConfig, TrainerConfig, default_epochs
    from sentimentizer.trainer import _new_loaders, new_trainer

    model_type = config["model_type"]
    dict_path = config["dict_path"]
    embeddings_config = EmbeddingsConfig(
        model_name=config["embeddings_model_name"],
        emb_length=config["embeddings_emb_length"],
    )
    device = config.get("device", "cpu")

    # Validate d_model divisible by n_heads for transformer models
    if model_type in ("encoder", "decoder"):
        d_model = config.get("d_model", 256)
        n_heads = config.get("n_heads", 4)
        if d_model % n_heads != 0:
            for nh in sorted([2, 4, 8], reverse=True):
                if d_model % nh == 0:
                    config["n_heads"] = nh
                    break

    # Create model with trial hyperparameters
    model_config = _build_model_config(model_type, config)

    if model_type == "rnn":
        from sentimentizer.models.rnn import new_model
    elif model_type == "encoder":
        from sentimentizer.models.encoder import new_model
    elif model_type == "decoder":
        from sentimentizer.models.decoder import new_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = new_model(
        dict_path=dict_path,
        embeddings_config=embeddings_config,
        model_config=model_config,
    )

    epochs = config.get("epochs", default_epochs(model_type))
    trainer_config = TrainerConfig(
        epochs=epochs,
        batch_size=config.get("batch_size", TrainerConfig.batch_size),
        device=device,
        dataloader_workers=0,
        class_weights=config.get("class_weights"),
        weight_smoothing=config.get("weight_smoothing", 0.5),
        loss_type=config.get("loss_type", "cross_entropy"),
        focal_gamma=config.get("focal_gamma", 2.0),
        label_smoothing=config.get("label_smoothing", 0.1),
    )

    trainer = new_trainer(
        model=model,
        cfg=trainer_config,
        model_type=model_type,
        lr=config.get("lr"),
        weight_decay=config.get("weight_decay"),
    )

    if train_dataset is None or val_dataset is None:
        from sentimentizer.loader import load_train_val_corpus_datasets

        balance_classes = config.get("balance_classes", True)
        random_state = config.get("balance_seed", 42)
        train_dataset, val_dataset = load_train_val_corpus_datasets(
            DriverConfig.files.processed_reviews_file_path,
            balance_classes=balance_classes,
            random_state=random_state,
        )

    train_loader, val_loader = _new_loaders(
        train_dataset,
        val_dataset,
        trainer_config,
    )
    model.to(device)
    # Trainer.fit() normally does this; without it the placeholder decays to eta_min
    # within the first few optimizer steps.
    trainer.rebuild_scheduler(len(train_loader), epochs)

    for epoch in range(1, epochs + 1):
        trainer._train_epoch(model, train_loader, epoch)  # noqa: SLF001
        trainer.evaluate(model, val_loader, epoch)

        metrics = trainer.latest_metrics
        val_loss = trainer.val_loss

        tune.report(
            {
                "val_accuracy": metrics.accuracy,
                "val_balanced_accuracy": metrics.balanced_accuracy,
                "val_loss": val_loss,
                "train_loss": trainer.latest_train_loss,
                "val_precision": metrics.positive_precision,
                "val_recall": metrics.positive_recall,
                "val_f1": metrics.positive_f1,
                "val_cohen_kappa": metrics.cohen_kappa,
                "val_negative_f1": metrics.negative_f1,
                "val_neutral_f1": metrics.neutral_f1,
                "val_positive_f1": metrics.positive_f1,
                "val_macro_f1": metrics.macro_f1,
                "val_weighted_f1": metrics.weighted_f1,
                "val_mcc": metrics.mcc,
                "epoch": epoch,
            }
        )


def _build_model_config(model_type: str, config: dict) -> Any:
    """Build a model config dataclass from the trial hyperparameters."""
    if model_type == "rnn":
        from sentimentizer.config import RNNConfig

        return RNNConfig(
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
        )
    elif model_type == "encoder":
        from sentimentizer.config import EncoderConfig

        return EncoderConfig(
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            dropout=config.get("dropout", 0.2),
            ff_multiplier=config.get("ff_multiplier", 4),
        )
    elif model_type == "decoder":
        from sentimentizer.config import DecoderConfig

        return DecoderConfig(
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_encoder_layers=config.get("n_encoder_layers", 2),
            n_decoder_layers=config.get("n_decoder_layers", 2),
            dropout=config.get("dropout", 0.3),
            ff_multiplier=config.get("ff_multiplier", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tune_model(
    model_type: str,
    tuner_config: TunerConfig | None = None,
    search_space_overrides: dict[str, dict[str, Any]] | None = None,
    config_path: str | Path | None = None,
    balance_classes: bool = False,
    balance_seed: int = 42,
    class_weights: list[float] | None = None,
    weight_smoothing: float = 0.5,
    loss_type: str = "cross_entropy",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> dict[str, Any]:
    """Run hyperparameter tuning using Ray Tune + Optuna.

    See Ray Tune docs:
    https://docs.ray.io/en/2.55.1/tune/api/doc/ray.tune.Tuner.html
    https://docs.ray.io/en/2.55.1/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        tuner_config: Tuner configuration (loaded from YAML if None).
        search_space_overrides: Optional overrides from the agent's
            TuningDecision.
        config_path: Path to config YAML.
        balance_classes: Whether to balance classes in training data
            by undersampling the majority class (default: True).
        balance_seed: Random seed for class balancing (default: 42).

    Returns:
        Dict with keys:
        - 'best_config': dict of best hyperparameters
        - 'best_accuracy': float — overall accuracy
        - 'best_balanced_accuracy': float — mean of per-class recalls
        - 'best_loss': float — best validation loss
        - 'best_precision': float — positive-class precision
        - 'best_recall': float — positive-class recall
        - 'best_f1': float — positive-class F1 score
        - 'best_cohen_kappa': float — Cohen's kappa coefficient
        - 'best_negative_f1': float — negative-class F1 score
        - 'best_neutral_f1': float — neutral-class F1 score
        - 'best_positive_f1': float — positive-class F1 score
        - 'best_macro_f1': float — macro-averaged F1 score
        - 'best_weighted_f1': float — weighted-averaged F1 score
        - 'best_mcc': float — Matthews correlation coefficient
        - 'trial_count': int
        - 'results': list of all trial results
    """
    if tuner_config is None:
        tuner_config = load_tuner_config(config_path)

    search_space = build_search_space(
        model_type,
        overrides=search_space_overrides,
        config_path=config_path,
    )

    from sentimentizer.config import DriverConfig, auto_detect_device

    search_space["model_type"] = model_type
    search_space["device"] = auto_detect_device("auto")
    search_space["dict_path"] = DriverConfig.files.dictionary_file_path
    search_space["embeddings_model_name"] = DriverConfig.embeddings.model_name
    search_space["embeddings_emb_length"] = DriverConfig.embeddings.emb_length
    search_space["balance_classes"] = balance_classes
    search_space["balance_seed"] = balance_seed
    search_space["class_weights"] = class_weights
    search_space["weight_smoothing"] = weight_smoothing
    search_space["loss_type"] = loss_type
    search_space["focal_gamma"] = focal_gamma
    search_space["label_smoothing"] = label_smoothing

    from sentimentizer.loader import load_train_val_corpus_datasets

    logger.info("loading_datasets_for_tuning")
    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.processed_reviews_file_path,
        balance_classes=balance_classes,
        random_state=balance_seed,
    )

    from sentimentizer.config import default_epochs

    scheduler = _get_scheduler(tuner_config, max_epochs=default_epochs(model_type))
    search_alg = OptunaSearch(
        metric=tuner_config.metric,
        mode=tuner_config.mode,
    )

    reporter = CLIReporter(
        metric_columns=[
            "val_accuracy",
            "val_balanced_accuracy",
            "val_loss",
            "train_loss",
            "val_f1",
            "val_cohen_kappa",
            "val_negative_f1",
            "val_neutral_f1",
            "val_positive_f1",
            "val_macro_f1",
            "val_weighted_f1",
            "val_mcc",
            "epoch",
        ],
    )

    logger.info(
        "starting_ray_tune",
        model_type=model_type,
        num_samples=tuner_config.num_samples,
        scheduler=tuner_config.scheduler,
        search_algorithm="optuna",
    )

    _check_memory_for_workers(num_workers=4)

    # Start a Prometheus metrics server on port 8082 so that
    # the tune gauges in exporter.py are scrapable by Prometheus.
    # Port 8081 is used by the standalone exporter; we use 8082
    # here to avoid conflicts when both are running.
    prom_callback = TunePrometheusCallback(model_type=model_type)
    _ensure_prometheus_server(port=8082)

    result = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                _trainable_wrapper,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            ),
            resources={"cpu": 1, "gpu": 0.25 if _gpu_available() else 0},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=tuner_config.num_samples,
            max_concurrent_trials=4,
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            name=f"sentimentizer_{model_type}_tune",
            callbacks=[prom_callback],
        ),
    ).fit()

    best_result = result.get_best_result(
        metric=tuner_config.metric,
        mode=tuner_config.mode,
    )
    best_config = best_result.config
    # Result.metrics is Optional[Dict] in Ray 2.55+ — guard against None
    best_metrics = best_result.metrics or {}

    internal_keys = {
        "model_type",
        "device",
        "dict_path",
        "embeddings_model_name",
        "embeddings_emb_length",
        "balance_classes",
        "balance_seed",
    }
    clean_config = {k: v for k, v in best_config.items() if k not in internal_keys}

    # ResultGrid is NOT directly iterable in Ray 2.55+.
    # Use index-based iteration with len() instead of ``for x in result:``.
    all_results = []
    for i in range(len(result)):
        trial_result = result[i]
        trial_config = {k: v for k, v in trial_result.config.items() if k not in internal_keys}
        all_results.append(
            {"config": trial_config, "metrics": trial_result.metrics or {}},
        )

    output = {
        "best_config": clean_config,
        "best_accuracy": best_metrics.get("val_accuracy", 0.0),
        "best_balanced_accuracy": best_metrics.get("val_balanced_accuracy", 0.0),
        "best_loss": best_metrics.get("val_loss", float("inf")),
        "best_precision": best_metrics.get("val_precision", 0.0),
        "best_recall": best_metrics.get("val_recall", 0.0),
        "best_f1": best_metrics.get("val_f1", 0.0),
        "best_cohen_kappa": best_metrics.get("val_cohen_kappa", 0.0),
        "best_negative_f1": best_metrics.get("val_negative_f1", 0.0),
        "best_neutral_f1": best_metrics.get("val_neutral_f1", 0.0),
        "best_positive_f1": best_metrics.get("val_positive_f1", 0.0),
        "best_macro_f1": best_metrics.get("val_macro_f1", 0.0),
        "best_weighted_f1": best_metrics.get("val_weighted_f1", 0.0),
        "best_mcc": best_metrics.get("val_mcc", 0.0),
        "trial_count": len(result),
        "results": all_results,
    }

    logger.info(
        "tuning_complete",
        best_accuracy=output["best_accuracy"],
        best_balanced_accuracy=output["best_balanced_accuracy"],
        best_loss=output["best_loss"],
        best_f1=output["best_f1"],
        best_cohen_kappa=output["best_cohen_kappa"],
        best_negative_f1=output["best_negative_f1"],
        best_neutral_f1=output["best_neutral_f1"],
        best_positive_f1=output["best_positive_f1"],
        best_macro_f1=output["best_macro_f1"],
        best_mcc=output["best_mcc"],
        trial_count=output["trial_count"],
        best_config=clean_config,
    )

    return output


# Track whether we've already started a Prometheus server to avoid
# "Address already in use" errors when tune_model is called more
# than once in the same process.
_prometheus_server_started: bool = False


def _ensure_prometheus_server(port: int = 8082) -> None:
    """Start a prometheus_client HTTP server for tune metrics if not running.

    This is called by ``tune_model`` so that the ``sentimentizer_tune_*``
    gauges are scrapable by Prometheus while tuning is in progress.
    It uses a module-level flag to ensure the server is started at most once.
    """
    global _prometheus_server_started
    if _prometheus_server_started:
        logger.info("prometheus_server_already_running", port=port)
        return
    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        _prometheus_server_started = True
        logger.info(
            "prometheus_server_started",
            port=port,
            message="Tune metrics available for Prometheus scraping",
        )
    except OSError as exc:
        # Port already in use — most likely the standalone exporter
        # is running on 8081, or a previous tune run started 8082.
        logger.warning("prometheus_server_port_in_use", port=port, error=str(exc))
    except ImportError:
        logger.warning(
            "prometheus_client_not_available",
            message="Cannot start Prometheus server for tune metrics",
        )


def _gpu_available() -> bool:
    """Check if a GPU is available for Ray Tune trials."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_memory_for_workers(num_workers: int) -> None:
    """Check and log if there is enough GPU memory for the requested workers."""
    if not _gpu_available():
        return

    import torch

    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_mem_gb = free_mem / (1024**3)

        # Estimate ~1.5 GB per worker for these text classification models
        # (embeddings + hidden states + adam optimizer)
        required_gb = num_workers * 1.5

        if free_mem_gb < required_gb:
            logger.warning(
                "insufficient_gpu_memory_warning",
                free_mem_gb=f"{free_mem_gb:.2f}GB",
                estimated_required_gb=f"{required_gb:.2f}GB",
                num_workers=num_workers,
                message="You may experience CUDA OutOfMemory errors during tuning.",
            )
        else:
            logger.info(
                "gpu_memory_check_passed",
                free_mem_gb=f"{free_mem_gb:.2f}GB",
                estimated_required_gb=f"{required_gb:.2f}GB",
                num_workers=num_workers,
            )
    except Exception as e:
        logger.warning(f"Could not calculate GPU memory: {e}")
