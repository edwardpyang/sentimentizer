"""TuningRun workflow: create a tuning run and work with the agent until convergence.

This module provides a high-level API for orchestrating hyperparameter
tuning runs. It supports two modes:

1. **Agent-guided** — uses the Pydantic AI + LangGraph agent loop
   (analyze → decide → tune → evaluate) to iteratively refine the
   search space until convergence or max iterations.

2. **Standalone** — runs a single Ray Tune + Optuna search without
   the LLM agent, useful for quick sweeps or when Ollama is unavailable.

Both modes produce a ``TuningRunResult`` with the best configuration,
metrics, and an optional trained model saved to disk.

The workflow also includes model validation: after tuning, it trains a final
model with the best config and validates predictions against known
sentiment examples (negative, neutral, positive). If the model doesn't
meet quality criteria, it can re-tune with adjusted parameters.

Usage::

    from sentimentizer.agent.diagnose_model import TuningRun, TuningRunConfig

    config = TuningRunConfig(model_type="rnn")
    run = TuningRun(config)
    result = run.execute()
    print(f"Best accuracy: {result.best_accuracy:.4f}")
    print(f"Best macro F1: {result.best_macro_f1:.4f}")
    print(f"Validation passed: {result.validation_passed}")
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sentimentizer import new_logger
from sentimentizer.agent.loader import AgentConfig, load_agent_config
from sentimentizer.agent.models import AgentRunResult, TuningResult
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DriverConfig,
    EmbeddingsConfig,
    TrainerConfig,
    default_epochs,
    weights_path_for,
)
from sentimentizer.device import resolve_device
from sentimentizer.tuner import TunerConfig, tune_model

logger = new_logger(DEFAULT_LOG_LEVEL)


# ---------------------------------------------------------------------------
# Known sentiment examples for validation
# ---------------------------------------------------------------------------

KNOWN_SENTIMENT_EXAMPLES: list[dict[str, Any]] = [
    {"text": "amazing food great service", "expected": "positive"},
    {"text": "terrible experience worst ever", "expected": "negative"},
    {"text": "no good", "expected": "negative"},
    {"text": "loved it so much", "expected": "positive"},
    {"text": "awful and disgusting", "expected": "negative"},
    {"text": "best restaurant in town", "expected": "positive"},
    {"text": "would not recommend", "expected": "negative"},
    {"text": "fantastic place", "expected": "positive"},
    {"text": "it was okay", "expected": "neutral"},
    {"text": "average experience", "expected": "neutral"},
    {"text": "nothing special", "expected": "neutral"},
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TuningRunConfig:
    """Configuration for a ``TuningRun``.

    Attributes:
        model_type: Model architecture to tune ('rnn', 'encoder', 'decoder').
        config_path: Path to agent config YAML. Uses default if None.
        mode: 'agent' for LLM-guided loop, 'standalone' for single Ray Tune run.
        max_iterations: Maximum agent loop iterations (agent mode only).
            None means "use the value from the agent config YAML".
        convergence_threshold: Min accuracy improvement to continue (agent mode).
            None means "use the value from the agent config YAML".
        num_samples: Number of Ray Tune trials per iteration.
            None means "use the value from the tuner config YAML".
        scheduler: Ray Tune scheduler ('asha', 'hyperband', 'median').
            None means "use the value from the tuner config YAML".
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        save_best_model: Whether to train and save a final model with best config.
        output_dir: Directory to save results and model weights.
        search_space_overrides: Manual overrides for the YAML search space.
        stop: Number of data rows to load for training.
        validate_predictions: Whether to validate model predictions after training.
        validation_threshold: Minimum fraction of correct predictions to pass.
        max_retries: Maximum number of re-tuning attempts if validation fails.
        weight_smoothing: Inverse-frequency exponent for class weights
            (1.0=full, 0.5=sqrt, 0.0=uniform).
        loss_type: Loss function type ('cross_entropy' or 'focal').
        focal_gamma: Focal loss focusing parameter (loss_type='focal' only).
        label_smoothing: Softens hard targets, reduces overconfident predictions.
        neutral_oversample_ratio: 0.0=disabled, 0.20=neutral to 20% of data.
        balance_strategy: Strategy for handling class imbalance
            ('class_weights_only', 'undersample', 'oversample').
        push_to_hub: Whether to push model weights, dictionary, and model card
            to Hugging Face Hub after successful validation.
    """

    model_type: str = "rnn"
    config_path: str | None = None
    mode: str = "agent"
    max_iterations: int | None = None
    convergence_threshold: float | None = None
    num_samples: int | None = None
    scheduler: str | None = None
    device: str = "auto"
    save_best_model: bool = True
    output_dir: str = "tuning_results"
    search_space_overrides: dict[str, dict[str, Any]] | None = None
    stop: int = 100000
    validate_predictions: bool = True
    validation_threshold: float = 0.75
    max_retries: int = 2
    balance_classes: bool = False
    balance_seed: int = 42
    weight_smoothing: float = 0.5
    loss_type: str = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    neutral_oversample_ratio: float = 0.0
    balance_strategy: str = "class_weights_only"
    push_to_hub: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TuningRunResult:
    """Result of a ``TuningRun.execute()`` call.

    Attributes:
        best_config: Best hyperparameter configuration found.
        best_accuracy: Best validation accuracy achieved.
        best_loss: Best validation loss achieved.
        best_balanced_accuracy: Best balanced accuracy (mean of per-class recalls).
        best_negative_precision: Best precision for the negative class.
        best_negative_recall: Best recall for the negative class.
        best_negative_f1: Best F1 score for the negative class.
        best_neutral_precision: Best precision for the neutral class.
        best_neutral_recall: Best recall for the neutral class.
        best_neutral_f1: Best F1 score for the neutral class.
        best_positive_precision: Best precision for the positive class.
        best_positive_recall: Best recall for the positive class.
        best_positive_f1: Best F1 score for the positive class.
        best_macro_f1: Best macro-averaged F1 (mean of per-class F1).
        best_weighted_f1: Best weighted-averaged F1 (weighted by class frequency).
        best_cohen_kappa: Best Cohen's kappa coefficient achieved.
        best_mcc: Best Matthews correlation coefficient achieved.
        iterations_completed: Number of tuning iterations completed.
        converged: Whether the agent converged before max iterations.
        history: List of TuningResult dicts from each iteration.
        model_path: Path to the saved model weights (empty string if not saved).
        results_path: Path to the saved JSON results file.
        elapsed_seconds: Wall-clock time for the entire run.
        validation_passed: Whether model predictions passed validation.
        validation_results: Detailed validation results per example.
        validation_metrics: Classification metrics from model validation.
        retry_count: Number of re-tuning attempts due to failed validation.
    """

    best_config: dict[str, Any] = field(default_factory=dict)
    best_accuracy: float = 0.0
    best_loss: float = float("inf")
    best_balanced_accuracy: float = 0.0
    best_negative_precision: float = 0.0
    best_negative_recall: float = 0.0
    best_negative_f1: float = 0.0
    best_neutral_precision: float = 0.0
    best_neutral_recall: float = 0.0
    best_neutral_f1: float = 0.0
    best_positive_precision: float = 0.0
    best_positive_recall: float = 0.0
    best_positive_f1: float = 0.0
    best_macro_f1: float = 0.0
    best_weighted_f1: float = 0.0
    best_cohen_kappa: float = 0.0
    best_mcc: float = 0.0
    iterations_completed: int = 0
    converged: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    model_path: str = ""
    results_path: str = ""
    elapsed_seconds: float = 0.0
    validation_passed: bool = False
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    validation_metrics: dict[str, Any] | None = None
    retry_count: int = 0


# ---------------------------------------------------------------------------
# TuningRun
# ---------------------------------------------------------------------------


class TuningRun:
    """Orchestrate a hyperparameter tuning run from start to finish.

    This is the main entry point for creating a tuning run and working
    with the agent until a good model is found. It handles:

    - Loading configuration from YAML
    - Running agent-guided or standalone tuning
    - Training a final model with the best configuration
    - Validating model predictions against known sentiment examples
    - Re-tuning if validation fails (up to max_retries)
    - Saving results and model weights to disk

    Args:
        config: A ``TuningRunConfig`` with run settings.

    Example::

        run = TuningRun(TuningRunConfig(model_type="encoder", mode="agent"))
        result = run.execute()
        if result.validation_passed:
            print(f"Model ready! Accuracy: {result.best_accuracy:.4f}")
    """

    def __init__(self, config: TuningRunConfig) -> None:
        self.config = config
        self._resolve_device()
        self._output_dir = Path(config.output_dir)
        self._agent_config: AgentConfig | None = None
        self._tuner_config: TunerConfig | None = None

    def _resolve_device(self) -> None:
        """Resolve 'auto' device to the best available."""
        if self.config.device == "auto":
            self.config.device = resolve_device("auto")
        logger.info("device_resolved", device=self.config.device)

    def _load_configs(self) -> tuple[AgentConfig, TunerConfig]:
        """Load agent and tuner configs, applying overrides from TuningRunConfig."""
        agent_config, tuner_config = load_agent_config(self.config.config_path)

        # Apply TuningRunConfig overrides. None means "not specified —
        # keep the YAML value"; any explicit value is applied, even one
        # that happens to equal a default.
        if self.config.max_iterations is not None:
            agent_config.max_iterations = self.config.max_iterations
        if self.config.convergence_threshold is not None:
            agent_config.convergence_threshold = self.config.convergence_threshold

        if self.config.num_samples is not None:
            tuner_config.num_samples = self.config.num_samples
        if self.config.scheduler is not None:
            tuner_config.scheduler = self.config.scheduler

        self._agent_config = agent_config
        self._tuner_config = tuner_config
        return agent_config, tuner_config

    def execute(self) -> TuningRunResult:
        """Execute the full tuning run with validation and retry logic.

        Runs either agent-guided or standalone tuning based on
        ``self.config.mode``, then optionally trains a final model
        with the best configuration found, validates predictions,
        and re-tunes if validation fails.

        Returns:
            ``TuningRunResult`` with best config, metrics, and file paths.
        """
        start_time = time.time()
        logger.info(
            "tuning_run_start",
            model_type=self.config.model_type,
            mode=self.config.mode,
            device=self.config.device,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        retry_count = 0
        result = self._run_tuning()

        # Train final model and validate predictions
        if self.config.save_best_model and result.best_accuracy > 0.0:
            while retry_count <= self.config.max_retries:
                model_path = self._train_final_model(result.best_config)
                result.model_path = str(model_path)

                if self.config.validate_predictions:
                    validation_passed, validation_results, validation_metrics = (
                        self._validate_model(model_path)
                    )
                    result.validation_passed = validation_passed
                    result.validation_results = validation_results
                    result.validation_metrics = validation_metrics

                    if validation_passed:
                        logger.info(
                            "validation_passed",
                            model_type=self.config.model_type,
                            retry_count=retry_count,
                        )
                        break

                    logger.warning(
                        "validation_failed",
                        model_type=self.config.model_type,
                        retry_count=retry_count,
                        max_retries=self.config.max_retries,
                    )
                    retry_count += 1

                    if retry_count <= self.config.max_retries:
                        # Re-tune with narrowed search space
                        result = self._run_tuning()
                        if result.best_accuracy <= 0.0:
                            logger.warning("re_tuning_produced_no_results")
                            break
                else:
                    # No validation requested — mark as passed
                    result.validation_passed = True
                    break

        result.retry_count = retry_count

        # Copy best weights to model-specific path if validation passed
        if result.validation_passed and result.model_path:
            model_weights_path = Path(weights_path_for(self.config.model_type))
            try:
                model_weights_path.parent.mkdir(parents=True, exist_ok=True)
                _publish_tuned_weights(Path(result.model_path), model_weights_path)
                logger.info(
                    "weights_copied_to_default",
                    source=result.model_path,
                    destination=str(model_weights_path),
                )
            except OSError as e:
                logger.warning(f"failed_to_copy_weights: {e}")

            # Push to Hugging Face Hub if requested
            if self.config.push_to_hub:
                self._push_to_hub(result, model_weights_path)

        # Save results to JSON
        results_path = self._save_results(result)
        result.results_path = str(results_path)

        result.elapsed_seconds = time.time() - start_time
        logger.info(
            "tuning_run_complete",
            best_accuracy=result.best_accuracy,
            best_loss=result.best_loss,
            best_macro_f1=result.best_macro_f1,
            best_cohen_kappa=result.best_cohen_kappa,
            best_mcc=result.best_mcc,
            best_balanced_accuracy=result.best_balanced_accuracy,
            best_positive_f1=result.best_positive_f1,
            best_negative_f1=result.best_negative_f1,
            iterations=result.iterations_completed,
            converged=result.converged,
            validation_passed=result.validation_passed,
            retry_count=result.retry_count,
            elapsed_seconds=result.elapsed_seconds,
        )

        return result

    # ------------------------------------------------------------------
    # Hugging Face Hub push
    # ------------------------------------------------------------------

    def _push_to_hub(self, result: TuningRunResult, weights_path: Path) -> None:
        """Push model weights, dictionary, and model card to Hugging Face Hub.

        Args:
            result: The ``TuningRunResult`` with metrics for the model card.
            weights_path: Path to the model weights file to upload.
        """
        from sentimentizer.hf import push_model_to_hub

        tuning_result_dict = {
            "best_accuracy": result.best_accuracy,
            "best_loss": result.best_loss,
            "best_balanced_accuracy": result.best_balanced_accuracy,
            "best_negative_precision": result.best_negative_precision,
            "best_negative_recall": result.best_negative_recall,
            "best_negative_f1": result.best_negative_f1,
            "best_neutral_precision": result.best_neutral_precision,
            "best_neutral_recall": result.best_neutral_recall,
            "best_neutral_f1": result.best_neutral_f1,
            "best_positive_precision": result.best_positive_precision,
            "best_positive_recall": result.best_positive_recall,
            "best_positive_f1": result.best_positive_f1,
            "best_macro_f1": result.best_macro_f1,
            "best_weighted_f1": result.best_weighted_f1,
            "best_cohen_kappa": result.best_cohen_kappa,
            "best_mcc": result.best_mcc,
            "best_config": result.best_config,
            "validation_passed": result.validation_passed,
            "mode": self.config.mode,
            "iterations_completed": result.iterations_completed,
            "converged": result.converged,
            "elapsed_seconds": result.elapsed_seconds,
        }

        logger.info(
            "pushing_to_hub",
            model_type=self.config.model_type,
            weights_path=str(weights_path),
        )

        push_model_to_hub(
            local_path=str(weights_path),
            model_type=self.config.model_type,
            dict_path=DriverConfig.files.dictionary_file_path,
            tuning_result=tuning_result_dict,
        )

    # ------------------------------------------------------------------
    # Tuning dispatch
    # ------------------------------------------------------------------

    def _run_tuning(self) -> TuningRunResult:
        """Run the appropriate tuning mode (agent or standalone)."""
        if self.config.mode == "agent":
            return self._run_agent_guided()
        elif self.config.mode == "standalone":
            return self._run_standalone()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode!r}. Use 'agent' or 'standalone'.")

    # ------------------------------------------------------------------
    # Agent-guided mode
    # ------------------------------------------------------------------

    def _run_agent_guided(self) -> TuningRunResult:
        """Run the full agent-guided tuning loop.

        Uses the LangGraph state graph (analyze → decide → tune → evaluate)
        with Pydantic AI agents powered by GLM 5.1 via Ollama.
        """
        import asyncio

        from sentimentizer.agent.graph import run_agent_tuning

        agent_config, tuner_config = self._load_configs()

        logger.info(
            "starting_agent_guided_tuning",
            model_type=self.config.model_type,
            max_iterations=agent_config.max_iterations,
            convergence_threshold=agent_config.convergence_threshold,
            num_samples=tuner_config.num_samples,
        )

        agent_result: AgentRunResult = asyncio.run(
            run_agent_tuning(
                model_type=self.config.model_type,
                config_path=self.config.config_path,
            )
        )

        return TuningRunResult(
            best_config=agent_result.best_config,
            best_accuracy=agent_result.best_accuracy,
            best_loss=agent_result.best_loss,
            iterations_completed=agent_result.iterations_completed,
            converged=agent_result.converged,
            history=[
                h.model_dump() if hasattr(h, "model_dump") else h for h in agent_result.history
            ],
        )

    # ------------------------------------------------------------------
    # Standalone mode
    # ------------------------------------------------------------------

    def _run_standalone(self) -> TuningRunResult:
        """Run a single Ray Tune + Optuna search without the LLM agent.

        This is useful for quick sweeps or when Ollama is not available.
        """
        _, tuner_config = self._load_configs()

        logger.info(
            "starting_standalone_tuning",
            model_type=self.config.model_type,
            num_samples=tuner_config.num_samples,
            scheduler=tuner_config.scheduler,
        )

        result = tune_model(
            model_type=self.config.model_type,
            tuner_config=tuner_config,
            search_space_overrides=self.config.search_space_overrides,
            config_path=self.config.config_path,
            balance_classes=self.config.balance_classes,
            balance_seed=self.config.balance_seed,
            weight_smoothing=self.config.weight_smoothing,
            loss_type=self.config.loss_type,
            focal_gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing,
        )

        tuning_result = TuningResult(
            best_accuracy=result["best_accuracy"],
            best_loss=result["best_loss"],
            best_config=result["best_config"],
            trial_count=result["trial_count"],
            # No previous iteration to improve over in a single sweep.
            improvement_over_last=0.0,
        )

        return TuningRunResult(
            best_config=result["best_config"],
            best_accuracy=result["best_accuracy"],
            best_loss=result["best_loss"],
            best_balanced_accuracy=result.get("best_balanced_accuracy", 0.0),
            best_negative_precision=result.get("best_negative_precision", 0.0),
            best_negative_recall=result.get("best_negative_recall", 0.0),
            best_negative_f1=result.get("best_negative_f1", 0.0),
            best_neutral_precision=result.get("best_neutral_precision", 0.0),
            best_neutral_recall=result.get("best_neutral_recall", 0.0),
            best_neutral_f1=result.get("best_neutral_f1", 0.0),
            best_positive_precision=result.get("best_positive_precision", 0.0),
            best_positive_recall=result.get("best_positive_recall", 0.0),
            best_positive_f1=result.get("best_positive_f1", 0.0),
            best_macro_f1=result.get("best_macro_f1", 0.0),
            best_weighted_f1=result.get("best_weighted_f1", 0.0),
            best_cohen_kappa=result.get("best_cohen_kappa", 0.0),
            best_mcc=result.get("best_mcc", 0.0),
            iterations_completed=1,
            # Standalone is a single sweep, not an iterative loop — there
            # is no convergence notion (unlike agent mode, where converged
            # means the accuracy delta fell below convergence_threshold).
            converged=False,
            history=[tuning_result.model_dump()],
        )

    # ------------------------------------------------------------------
    # Final model training
    # ------------------------------------------------------------------

    def _train_final_model(self, best_config: dict[str, Any]) -> Path:
        """Train a final model with the best configuration found.

        Creates a model using the best hyperparameters from the tuning
        run, trains it on the full dataset, and saves the weights.

        Args:
            best_config: Best hyperparameter configuration from tuning.

        Returns:
            Path to the saved model weights.
        """
        import torch

        from sentimentizer.loader import load_train_val_corpus_datasets
        from sentimentizer.trainer import new_trainer, restore_best_checkpoint

        model_type = self.config.model_type
        device = self.config.device

        logger.info(
            "training_final_model",
            model_type=model_type,
            device=device,
            best_config=best_config,
        )

        # Build model config from best hyperparameters
        model_config = _build_model_config(model_type, best_config)

        # Create model
        model = _create_model(model_type, model_config)
        model.to(device)

        # Train for the model's default epoch count. The per-batch scheduler
        # computes total_steps dynamically from epochs * steps_per_epoch, so
        # the cosine decay scale always matches the actual training length.
        # Early stopping (patience=3, set below) handles convergence.
        final_epochs = default_epochs(model_type)

        trainer_config = TrainerConfig(
            epochs=final_epochs,
            batch_size=best_config.get("batch_size", TrainerConfig.batch_size),
            device=device,
            early_stopping_patience=3,
            checkpoint_dir=str(self._output_dir / "checkpoints"),
            checkpoint_every=1,
            checkpoint_best=True,
            weight_smoothing=self.config.weight_smoothing,
            loss_type=self.config.loss_type,
            focal_gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing,
            neutral_oversample_ratio=self.config.neutral_oversample_ratio,
            balance_strategy=self.config.balance_strategy,
        )

        trainer = new_trainer(
            model=model,
            cfg=trainer_config,
            model_type=model_type,
            lr=best_config.get("lr"),
            weight_decay=best_config.get("weight_decay"),
        )

        # Load data
        train_dataset, val_dataset = load_train_val_corpus_datasets(
            DriverConfig.files.processed_reviews_file_path,
            balance_classes=self.config.balance_classes,
            random_state=self.config.balance_seed,
        )

        # Train
        trainer.fit(model, train_data=train_dataset, val_data=val_dataset)
        restore_best_checkpoint(model, trainer_config.checkpoint_dir, device=device)

        # Save model weights
        model_path = self._output_dir / f"best_model_{model_type}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"final_model_saved: {model_path}")

        # Save the architecture config alongside the weights so the model
        # can be reconstructed exactly. n_heads cannot be recovered from
        # weight shapes (nn.MultiheadAttention packs QKV regardless of head
        # count), so weight-shape inference alone is insufficient.
        config_path = model_path.with_name(f"{model_path.stem}_config.json")
        with open(config_path, "w") as f:
            json.dump({"model_type": model_type, **asdict(model_config)}, f, indent=2)
        logger.info(f"final_model_config_saved: {config_path}")

        return model_path

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    def _validate_model(self, model_path: str) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        """Validate model predictions against known sentiment examples.

        Loads the trained model and tests it against a set of known
        negative, neutral, and positive phrases. A prediction is correct
        if the highest-probability class label matches the expected label.

        Also computes comprehensive classification metrics including
        per-class precision/recall/F1, macro/weighted F1, Cohen's kappa,
        MCC, balanced accuracy, and per-class AUC-ROC.

        Before validation, runs diagnostic checks to detect common
        training issues like dictionary misalignment. If diagnostics
        find critical issues, validation fails early with details.

        Args:
            model_path: Path to the saved model weights.

        Returns:
            Tuple of (passed, results, metrics) where passed is True if
            the fraction of correct predictions >= validation_threshold,
            results is a list of per-example validation details, and
            metrics is a dict of ClassificationMetrics fields.
        """

        from sentimentizer.metrics import compute_metrics_from_examples

        model_type = self.config.model_type
        device = self.config.device

        logger.info(
            "validating_model",
            model_type=model_type,
            model_path=model_path,
            threshold=self.config.validation_threshold,
        )

        # Run diagnostic checks before validation
        diagnostics = diagnose_training_issues(model_type=model_type)
        if diagnostics["has_critical_issues"]:
            logger.warning(
                "validation_skipped_due_to_diagnostics",
                critical_issues=diagnostics["critical_issues"],
            )
            error_results = [
                {"error": issue, "severity": "critical"} for issue in diagnostics["critical_issues"]
            ]
            return False, error_results, {}

        # Load tokenizer
        from sentimentizer.tokenizer import get_trained_tokenizer

        try:
            tokenizer = get_trained_tokenizer()
        except Exception as e:
            logger.warning(f"tokenizer_load_failed: {e}")
            return False, [{"error": str(e)}], {}

        # Load model
        try:
            model = _load_trained_model(model_type, model_path, device)
        except Exception as e:
            logger.warning(f"model_load_failed: {e}")
            return False, [{"error": str(e)}], {}

        model.to(device)
        model.eval()

        results: list[dict[str, Any]] = []
        correct = 0

        for example in KNOWN_SENTIMENT_EXAMPLES:
            text = example["text"]
            expected = example["expected"]

            # predict_text returns dict like {"negative": 0.05, "neutral": 0.12, "positive": 0.83}
            scores: dict[str, float] = model.predict_text(text, tokenizer)

            # Determine if prediction is correct: highest-probability class matches expected label
            predicted_label = max(scores, key=scores.get)  # type: ignore[arg-type]
            is_correct = predicted_label == expected

            if is_correct:
                correct += 1

            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "predicted": predicted_label,
                    "correct": is_correct,
                }
            )

            logger.info(
                "validation_example",
                text=text,
                expected=expected,
                predicted=predicted_label,
                scores={k: round(v, 4) for k, v in scores.items()},
                correct=is_correct,
            )

        accuracy = correct / len(KNOWN_SENTIMENT_EXAMPLES)
        passed = accuracy >= self.config.validation_threshold

        # Compute comprehensive classification metrics
        metrics_obj = compute_metrics_from_examples(results)
        metrics_dict = metrics_obj.to_dict()

        logger.info(
            "validation_summary",
            correct=correct,
            total=len(KNOWN_SENTIMENT_EXAMPLES),
            accuracy=round(accuracy, 4),
            passed=passed,
            balanced_accuracy=metrics_obj.balanced_accuracy,
            macro_f1=metrics_obj.macro_f1,
            cohen_kappa=metrics_obj.cohen_kappa,
            mcc=metrics_obj.mcc,
            positive_f1=metrics_obj.positive_f1,
            negative_f1=metrics_obj.negative_f1,
            neutral_f1=metrics_obj.neutral_f1,
        )

        return passed, results, metrics_dict

    # ------------------------------------------------------------------
    # Results persistence
    # ------------------------------------------------------------------

    def _save_results(self, result: TuningRunResult) -> Path:
        """Save tuning results to a JSON file.

        Args:
            result: The ``TuningRunResult`` to save.

        Returns:
            Path to the saved JSON file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_path = self._output_dir / f"tuning_results_{self.config.model_type}.json"

        output = {
            "model_type": self.config.model_type,
            "mode": self.config.mode,
            "device": self.config.device,
            "best_config": result.best_config,
            "best_accuracy": result.best_accuracy,
            "best_loss": result.best_loss,
            "best_balanced_accuracy": result.best_balanced_accuracy,
            "best_negative_precision": result.best_negative_precision,
            "best_negative_recall": result.best_negative_recall,
            "best_negative_f1": result.best_negative_f1,
            "best_neutral_precision": result.best_neutral_precision,
            "best_neutral_recall": result.best_neutral_recall,
            "best_neutral_f1": result.best_neutral_f1,
            "best_positive_precision": result.best_positive_precision,
            "best_positive_recall": result.best_positive_recall,
            "best_positive_f1": result.best_positive_f1,
            "best_macro_f1": result.best_macro_f1,
            "best_weighted_f1": result.best_weighted_f1,
            "best_cohen_kappa": result.best_cohen_kappa,
            "best_mcc": result.best_mcc,
            "iterations_completed": result.iterations_completed,
            "converged": result.converged,
            "history": result.history,
            "model_path": result.model_path,
            "validation_passed": result.validation_passed,
            "validation_results": result.validation_results,
            "validation_metrics": result.validation_metrics,
            "retry_count": result.retry_count,
            "elapsed_seconds": result.elapsed_seconds,
        }

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"results_saved: {results_path}")
        return results_path

    @staticmethod
    def load_results(path: str | Path) -> dict[str, Any]:
        """Load tuning results from a JSON file.

        Args:
            path: Path to the JSON results file.

        Returns:
            Dict with the saved tuning results.
        """
        path = Path(path)
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diagnose_training_issues(model_type: str = "") -> dict[str, Any]:
    """Run diagnostic checks to detect common training issues.

    Checks for problems that cause incorrect or "whacky" predictions,
    such as dictionary misalignment between tokenization and embedding
    matrix, class imbalance, and data integrity issues.

    This function is designed to be called before or after training to
    verify the pipeline is producing correct results. It can also be
    used as a troubleshooting tool when predictions look wrong.

    Diagnostics performed:

    1. **Dictionary alignment** — Verifies that the saved dictionary
       (used by ``new_model()`` to build the embedding matrix) has
       the same token-to-ID mapping as a freshly built dictionary
       from the raw data. A mismatch means training data token IDs
       don't align with embedding rows, causing the model to learn
       wrong word associations.

    2. **Embedding matrix alignment** — Verifies that each row in the
       embedding matrix corresponds to the correct word in the
       dictionary (row k = embedding for token ID k-1).

    3. **Class balance** — Reports the positive/negative class ratio
       in the processed training data and warns about severe imbalance.

    4. **Data integrity** — Checks that token IDs in the processed
       data fall within the valid range for the dictionary.

    Args:
        model_type: Model type to check ('rnn', 'encoder', 'decoder').

    Returns:
        Dict with keys:
        - ``has_critical_issues`` (bool): True if any issue would cause
          incorrect predictions.
        - ``critical_issues`` (list[str]): Descriptions of critical issues.
        - ``warnings`` (list[str]): Descriptions of non-critical issues.
        - ``checks`` (dict): Detailed results of each diagnostic check.
    """
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from gensim import corpora

    from sentimentizer.config import DriverConfig, EmbeddingsConfig, TokenizerConfig
    from sentimentizer.extractor import new_embedding_weights

    critical_issues: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    dict_path = DriverConfig.files.dictionary_file_path
    processed_path = DriverConfig.files.processed_reviews_file_path
    raw_path = DriverConfig.files.raw_reviews_file_path

    # ------------------------------------------------------------------
    # Check 1: Dictionary alignment
    # ------------------------------------------------------------------
    # The saved dictionary is used by new_model() to build the embedding
    # matrix. If the tokenization step used a different dictionary (e.g.,
    # from from_dataset() which previously did NOT save), the token IDs
    # in the training data won't match the embedding rows.
    # ------------------------------------------------------------------
    dict_check: dict[str, Any] = {"name": "dictionary_alignment"}

    try:
        saved_dict = corpora.Dictionary.load(dict_path)
        dict_check["saved_dict_terms"] = len(saved_dict)

        # Build a fresh dictionary from raw data (same pipeline as
        # _build_dictionary_distributed, but using pandas)
        if Path(raw_path).exists():
            cfg = TokenizerConfig()
            raw_df = pd.read_parquet(raw_path)

            total_word_freq: Counter = Counter()
            total_doc_freq: Counter = Counter()
            num_docs = 0

            batch_size = 10000
            for start in range(0, len(raw_df), batch_size):
                batch = raw_df.iloc[start : start + batch_size]
                for doc_tokens in batch[cfg.text_col]:
                    num_docs += 1
                    total_word_freq.update(doc_tokens)
                    total_doc_freq.update(set(doc_tokens))

            new_dict = corpora.Dictionary()
            new_dict.num_docs = num_docs
            for idx, word in enumerate(
                sorted(total_word_freq.keys(), key=lambda w: (-total_word_freq[w], w))
            ):
                new_dict.token2id[word] = idx
                new_dict.dfs[idx] = total_doc_freq[word]
            new_dict.num_pos = sum(total_word_freq.values())
            new_dict.num_nnz = sum(total_doc_freq.values())
            new_dict.filter_extremes(
                no_below=cfg.dict_min,
                no_above=cfg.no_above,
                keep_n=cfg.dict_keep,
            )
            new_dict.compactify()

            dict_check["new_dict_terms"] = len(new_dict)

            # Compare token2id mappings
            mismatches = 0
            common_words = 0
            sample_mismatches: list[dict[str, Any]] = []
            for word in saved_dict.token2id:
                if word in new_dict.token2id:
                    common_words += 1
                    if saved_dict.token2id[word] != new_dict.token2id[word]:
                        mismatches += 1
                        if len(sample_mismatches) < 5:
                            sample_mismatches.append(
                                {
                                    "word": word,
                                    "saved_id": saved_dict.token2id[word],
                                    "new_id": new_dict.token2id[word],
                                }
                            )

            mismatch_rate = mismatches / common_words if common_words > 0 else 0.0
            dict_check["common_words"] = common_words
            dict_check["mismatches"] = mismatches
            dict_check["mismatch_rate"] = round(mismatch_rate, 4)
            dict_check["sample_mismatches"] = sample_mismatches
            dict_check["passed"] = mismatch_rate < 0.01  # tolerate <1% drift

            if mismatch_rate >= 0.5:
                issue = (
                    f"Dictionary misalignment: {mismatches}/{common_words} "
                    f"({mismatch_rate:.1%}) token IDs differ between saved "
                    f"dictionary and freshly built dictionary. This means "
                    f"training data token IDs don't match embedding matrix "
                    f"rows, causing incorrect predictions. Re-tokenize with: "
                    f"python workflows/driver.py tokenize --type new"
                )
                critical_issues.append(issue)
            elif mismatch_rate >= 0.01:
                warnings.append(
                    f"Minor dictionary drift: {mismatches}/{common_words} "
                    f"({mismatch_rate:.1%}) token IDs differ. Consider "
                    f"re-tokenizing."
                )
        else:
            dict_check["skipped"] = True
            dict_check["skip_reason"] = f"Raw data not found at {raw_path}"
            warnings.append("Could not verify dictionary alignment: raw data file not found")

    except Exception as e:
        dict_check["error"] = str(e)
        dict_check["passed"] = False
        critical_issues.append(f"Dictionary check failed: {e}")

    checks["dictionary_alignment"] = dict_check

    # ------------------------------------------------------------------
    # Check 2: Embedding matrix alignment
    # ------------------------------------------------------------------
    # Verify that row k in the embedding matrix corresponds to the word
    # with token ID k-1 in the dictionary (offset by 1 for padding).
    # ------------------------------------------------------------------
    emb_check: dict[str, Any] = {"name": "embedding_alignment"}

    try:
        saved_dict = corpora.Dictionary.load(dict_path)
        emb_config = EmbeddingsConfig()
        embedding_matrix = new_embedding_weights(saved_dict, emb_config)

        expected_shape = (len(saved_dict) + 2, emb_config.emb_length)
        emb_check["actual_shape"] = embedding_matrix.shape
        emb_check["expected_shape"] = expected_shape
        emb_check["shape_matches"] = embedding_matrix.shape == expected_shape

        # Check that row 0 is all zeros (padding)
        emb_check["padding_row_is_zero"] = bool(np.allclose(embedding_matrix[0], 0.0))

        # Check that the OOV row (last row) is NOT all zeros
        emb_check["oov_row_is_nonzero"] = bool(not np.allclose(embedding_matrix[-1], 0.0))

        # Spot-check a few words: verify that the embedding for a word
        # matches the GloVe vector at the correct row
        test_words = ["the", "good", "bad", "food", "service"]
        spot_checks: list[dict[str, Any]] = []
        for word in test_words:
            if word in saved_dict.token2id:
                token_id = saved_dict.token2id[word]
                row_idx = token_id + 1  # offset for padding
                spot_checks.append(
                    {
                        "word": word,
                        "token_id": token_id,
                        "row_idx": row_idx,
                        "row_is_nonzero": bool(not np.allclose(embedding_matrix[row_idx], 0.0)),
                    }
                )
        emb_check["spot_checks"] = spot_checks
        emb_check["passed"] = emb_check["shape_matches"] and emb_check["padding_row_is_zero"]

        if not emb_check["shape_matches"]:
            critical_issues.append(
                f"Embedding matrix shape mismatch: got {embedding_matrix.shape}, "
                f"expected {expected_shape}. Dictionary and embedding matrix are "
                f"incompatible."
            )

    except Exception as e:
        emb_check["error"] = str(e)
        emb_check["passed"] = False
        critical_issues.append(f"Embedding check failed: {e}")

    checks["embedding_alignment"] = emb_check

    # ------------------------------------------------------------------
    # Check 3: Class balance
    # ------------------------------------------------------------------
    balance_check: dict[str, Any] = {"name": "class_balance"}

    try:
        if Path(processed_path).exists():
            processed_df = pd.read_parquet(processed_path)

            if "target" in processed_df.columns:
                target_counts = processed_df["target"].value_counts().to_dict()
                balance_check["target_counts"] = {str(k): int(v) for k, v in target_counts.items()}

                neg_count = target_counts.get(0.0, target_counts.get(0, 0))
                neu_count = target_counts.get(1.0, target_counts.get(1, 0))
                pos_count = target_counts.get(2.0, target_counts.get(2, 0))
                total = neg_count + neu_count + pos_count

                if total > 0:
                    balance_check["negative_count"] = neg_count
                    balance_check["neutral_count"] = neu_count
                    balance_check["positive_count"] = pos_count
                    balance_check["negative_ratio"] = round(neg_count / total, 4)
                    balance_check["neutral_ratio"] = round(neu_count / total, 4)
                    balance_check["positive_ratio"] = round(pos_count / total, 4)
                    counts = [c for c in [neg_count, neu_count, pos_count] if c > 0]
                    if len(counts) >= 2:
                        balance_check["imbalance_ratio"] = round(max(counts) / min(counts), 2)
                    else:
                        balance_check["imbalance_ratio"] = float("inf")

                    if balance_check["imbalance_ratio"] > 3:
                        warnings.append(
                            f"Severe class imbalance: negative={neg_count}, "
                            f"neutral={neu_count}, positive={pos_count}, "
                            f"ratio={balance_check['imbalance_ratio']}:1. "
                            f"Consider enabling class balancing or adjusting "
                            f"class_weights."
                        )

                balance_check["passed"] = True
            else:
                balance_check["skipped"] = True
                balance_check["skip_reason"] = "'target' column not found"
        else:
            balance_check["skipped"] = True
            balance_check["skip_reason"] = f"Processed data not found at {processed_path}"

    except Exception as e:
        balance_check["error"] = str(e)
        balance_check["passed"] = False
        warnings.append(f"Class balance check failed: {e}")

    checks["class_balance"] = balance_check

    # ------------------------------------------------------------------
    # Check 4: Data integrity — token ID range
    # ------------------------------------------------------------------
    integrity_check: dict[str, Any] = {"name": "data_integrity"}

    try:
        if Path(processed_path).exists() and Path(dict_path).exists():
            processed_df = pd.read_parquet(processed_path)
            saved_dict = corpora.Dictionary.load(dict_path)

            if "data" in processed_df.columns:
                max_valid_id = len(saved_dict)  # token IDs are 0..N-1, +1 offset = 1..N
                oov_id = len(saved_dict) + 1

                # Sample a subset for speed
                sample = processed_df["data"].iloc[:1000]
                max_token_id = 0
                oov_count = 0
                invalid_count = 0

                for row in sample:
                    arr = np.asarray(row)
                    row_max = int(arr.max())
                    if row_max > max_token_id:
                        max_token_id = row_max
                    oov_count += int(np.sum(arr == oov_id))
                    # Invalid: token ID > oov_id or between max_valid_id+1 and oov_id-1
                    invalid_count += int(
                        np.sum((arr > max_valid_id) & (arr != oov_id) & (arr != 0))
                    )

                integrity_check["max_valid_token_id"] = max_valid_id
                integrity_check["oov_id"] = oov_id
                integrity_check["max_token_id_in_data"] = max_token_id
                integrity_check["oov_count_in_sample"] = oov_count
                integrity_check["invalid_token_count_in_sample"] = invalid_count
                integrity_check["passed"] = invalid_count == 0

                if invalid_count > 0:
                    critical_issues.append(
                        f"Invalid token IDs in training data: {invalid_count} "
                        f"tokens exceed valid range (max valid={max_valid_id}, "
                        f"OOV={oov_id}, found max={max_token_id})"
                    )
            else:
                integrity_check["skipped"] = True
                integrity_check["skip_reason"] = "'data' column not found"
        else:
            integrity_check["skipped"] = True
            integrity_check["skip_reason"] = "Processed data or dictionary not found"

    except Exception as e:
        integrity_check["error"] = str(e)
        integrity_check["passed"] = False
        warnings.append(f"Data integrity check failed: {e}")

    checks["data_integrity"] = integrity_check

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    result = {
        "has_critical_issues": len(critical_issues) > 0,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "checks": checks,
    }

    if critical_issues:
        logger.warning(
            "diagnostics_found_critical_issues",
            critical_issues=critical_issues,
        )
    if warnings:
        logger.info(
            "diagnostics_found_warnings",
            warnings=warnings,
        )
    if not critical_issues and not warnings:
        logger.info("diagnostics_all_checks_passed")

    return result


# ---------------------------------------------------------------------------
# Module-level helpers (shared with tuner.py)
# ---------------------------------------------------------------------------


def _build_model_config(model_type: str, config: dict[str, Any]) -> Any:
    """Build a model config dataclass from the best hyperparameters."""
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


def _create_model(model_type: str, model_config: Any) -> Any:
    """Create a model instance from the model type and config."""
    embeddings_config = EmbeddingsConfig()
    dict_path = DriverConfig.files.dictionary_file_path

    if model_type == "rnn":
        from sentimentizer.models.rnn import new_model

        return new_model(
            dict_path=dict_path,
            embeddings_config=embeddings_config,
            model_config=model_config,
        )
    elif model_type == "encoder":
        from sentimentizer.models.encoder import new_model

        return new_model(
            dict_path=dict_path,
            embeddings_config=embeddings_config,
            model_config=model_config,
        )
    elif model_type == "decoder":
        from sentimentizer.models.decoder import new_model

        return new_model(
            dict_path=dict_path,
            embeddings_config=embeddings_config,
            model_config=model_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _publish_tuned_weights(source: Path, destination: Path) -> None:
    """Write tuned weights to the serving path with their architecture embedded.

    Serving and Hub consumers only receive the weights file, so a sidecar next to the
    tuning output would not reach them, and non-default architectures would fail to
    load or, for ``n_heads``, load silently wrong.
    """
    import torch

    config_path = source.with_name(f"{source.stem}_config.json")
    if not config_path.exists():
        shutil.copy2(source, destination)
        return
    with open(config_path) as f:
        model_config = json.load(f)
    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    torch.save({"model_state_dict": state_dict, "model_config": model_config}, destination)


def _load_trained_model(model_type: str, model_path: str, device: str) -> Any:
    """Load a trained model from saved weights.

    Reconstructs the architecture from the sidecar config JSON saved
    next to the weights (``<name>_config.json``). If no sidecar exists
    (e.g. Hugging Face downloads or older checkpoints, which always use
    the default architecture), falls back to inferring the dimensions
    that are recoverable from weight shapes.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        model_path: Path to the saved model weights (.pth).
        device: Device to load the model onto.

    Returns:
        Model with loaded weights in eval mode.
    """
    import torch

    weights = torch.load(model_path, map_location=device, weights_only=True)
    emb_shape = weights["embed_layer.weight"].shape
    empty_emb = torch.zeros(emb_shape)

    # Prefer the sidecar config — it captures architecture params (notably
    # n_heads) that cannot be inferred from weight shapes.
    config_path = Path(model_path).with_name(f"{Path(model_path).stem}_config.json")
    model_config = None
    if config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)
        saved.pop("model_type", None)
        model_config = _build_model_config(model_type, saved)
    else:
        logger.warning(
            "model_config_sidecar_missing",
            config_path=str(config_path),
            note="reconstructing with default architecture; non-default "
            "n_heads/n_layers will fail to load",
        )

    if model_type == "rnn":
        from sentimentizer.models.rnn import RNN

        if model_config is not None:
            model: Any = RNN(
                emb_weights=empty_emb,
                hidden_size=model_config.hidden_size,
                num_layers=model_config.num_layers,
                dropout=model_config.dropout,
            )
        else:
            hidden_size = weights["classifier.0.weight"].shape[1] // 2
            model = RNN(emb_weights=empty_emb, hidden_size=hidden_size)
    elif model_type == "encoder":
        from sentimentizer.models.encoder import Encoder

        if model_config is not None:
            model = Encoder(
                emb_weights=empty_emb,
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                n_layers=model_config.n_layers,
                dropout=model_config.dropout,
                ff_multiplier=model_config.ff_multiplier,
            )
        else:
            d_model = weights["proj.weight"].shape[0]
            model = Encoder(emb_weights=empty_emb, d_model=d_model)
    elif model_type == "decoder":
        from sentimentizer.models.decoder import Decoder

        if model_config is not None:
            model = Decoder(
                emb_weights=empty_emb,
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                n_encoder_layers=model_config.n_encoder_layers,
                n_decoder_layers=model_config.n_decoder_layers,
                dropout=model_config.dropout,
                ff_multiplier=model_config.ff_multiplier,
            )
        else:
            d_model = weights["proj.weight"].shape[0]
            model = Decoder(emb_weights=empty_emb, d_model=d_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(weights)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def create_tuning_run(
    model_type: str = "rnn",
    mode: str = "agent",
    config_path: str | None = None,
    num_samples: int | None = None,
    max_iterations: int | None = None,
    device: str = "auto",
    save_best_model: bool = True,
    output_dir: str = "tuning_results",
    search_space_overrides: dict[str, dict[str, Any]] | None = None,
    validate_predictions: bool = True,
    validation_threshold: float = 0.75,
    max_retries: int = 2,
    balance_classes: bool = False,
    balance_seed: int = 42,
    weight_smoothing: float = 0.5,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    neutral_oversample_ratio: float = 0.0,
    balance_strategy: str = "class_weights_only",
    push_to_hub: bool = False,
) -> TuningRunResult:
    """Create and execute a tuning run with sensible defaults.

    This is the simplest way to start a tuning run. It creates a
    ``TuningRunConfig``, builds a ``TuningRun``, and executes it.

    The workflow works with the agent until a good model is found:
    it tunes hyperparameters, trains a final model, validates
    predictions, and re-tunes if validation fails.

    Args:
        model_type: Model architecture ('rnn', 'encoder', 'decoder').
        mode: 'agent' for LLM-guided loop, 'standalone' for single search.
        config_path: Path to agent config YAML (uses default if None).
        num_samples: Number of Ray Tune trials per iteration.
            None means "use the value from the tuner config YAML".
        max_iterations: Max agent loop iterations (agent mode).
            None means "use the value from the agent config YAML".
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        save_best_model: Whether to train and save a final model.
        output_dir: Directory to save results and model weights.
        search_space_overrides: Manual overrides for the search space.
        validate_predictions: Whether to validate model predictions.
        validation_threshold: Minimum fraction of correct predictions.
        max_retries: Max re-tuning attempts if validation fails.
        balance_classes: Whether to balance classes by undersampling (default: False).
        balance_seed: Random seed for class balancing (default: 42).
        weight_smoothing: Inverse-frequency exponent for class weights (default: 0.5).
        loss_type: Loss function type ('cross_entropy' or 'focal', default: 'focal').
        balance_strategy: Strategy for class imbalance (default: 'class_weights_only').
        push_to_hub: Whether to push model to Hugging Face Hub after validation (default: False).

    Returns:
        ``TuningRunResult`` with best config, metrics, and file paths.

    Example::

        # Agent-guided tuning with validation
        result = create_tuning_run(model_type="rnn", mode="agent")
        if result.validation_passed:
            print(f"Model ready! Accuracy: {result.best_accuracy:.4f}")

        # Quick standalone sweep with Hub push
        result = create_tuning_run(model_type="encoder", mode="standalone", push_to_hub=True)
    """
    config = TuningRunConfig(
        model_type=model_type,
        mode=mode,
        config_path=config_path,
        num_samples=num_samples,
        max_iterations=max_iterations,
        device=device,
        save_best_model=save_best_model,
        output_dir=output_dir,
        search_space_overrides=search_space_overrides,
        validate_predictions=validate_predictions,
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
    return run.execute()
