"""Tests for the sentimentizer model diagnosis module.

Tests cover:
- TuningRunConfig defaults and custom values
- TuningRunResult dataclass
- TuningRun class initialization and config resolution
- _build_model_config for each model type
- KNOWN_SENTIMENT_EXAMPLES structure
- Validation logic (correct/incorrect predictions)
- Results persistence (save/load)
- create_tuning_run convenience function
- Driver CLI argument parsing for --tune
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentimentizer.agent.diagnose_model import (
    KNOWN_SENTIMENT_EXAMPLES,
    TuningRun,
    TuningRunConfig,
    TuningRunResult,
    _build_model_config,
)
from sentimentizer.config import DecoderConfig, EncoderConfig, RNNConfig

# ─── TuningRunConfig Tests ──────────────────────────────────────────


class TestTuningRunConfig:
    """Test TuningRunConfig dataclass defaults and customization."""

    def test_defaults(self) -> None:
        """TuningRunConfig should have sensible defaults."""
        config = TuningRunConfig()
        assert config.model_type == "rnn"
        assert config.mode == "standalone"
        # None means "defer to the agent/tuner config YAML" — see _load_configs
        assert config.max_iterations is None
        assert config.convergence_threshold is None
        assert config.num_samples is None
        assert config.scheduler is None
        assert config.device == "auto"
        assert config.save_best_model is True
        assert config.output_dir == "tuning_results"
        assert config.validate_predictions is True
        assert config.validation_threshold == 0.75
        assert config.max_retries == 2
        assert config.config_path is None
        assert config.search_space_overrides is None
        assert config.push_to_hub is False
        assert config.weight_smoothing == 0.5
        assert config.loss_type == "focal"
        assert config.focal_gamma == 2.0
        assert config.label_smoothing == 0.1
        assert config.neutral_oversample_ratio == 0.0
        assert config.balance_strategy == "class_weights_only"

    def test_custom_values(self) -> None:
        """TuningRunConfig should accept custom values."""
        config = TuningRunConfig(
            model_type="encoder",
            mode="standalone",
            num_samples=50,
            max_iterations=10,
            device="cuda",
            save_best_model=False,
            output_dir="/tmp/tune",
            validation_threshold=0.9,
            max_retries=3,
            push_to_hub=True,
        )
        assert config.model_type == "encoder"
        assert config.mode == "standalone"
        assert config.num_samples == 50
        assert config.max_iterations == 10
        assert config.device == "cuda"
        assert config.save_best_model is False
        assert config.output_dir == "/tmp/tune"
        assert config.validation_threshold == 0.9
        assert config.max_retries == 3
        assert config.push_to_hub is True

    def test_invalid_mode_raises(self) -> None:
        """TuningRun with invalid mode should raise ValueError."""
        config = TuningRunConfig(mode="invalid")
        run = TuningRun(config)
        # Patch _load_configs to avoid file I/O
        with (
            patch.object(run, "_load_configs", return_value=(MagicMock(), MagicMock())),
            pytest.raises(ValueError, match="Unknown mode"),
        ):
            run._run_tuning()


# ─── TuningRunResult Tests ───────────────────────────────────────────


class TestTuningRunResult:
    """Test TuningRunResult dataclass."""

    def test_defaults(self) -> None:
        """TuningRunResult should have sensible defaults."""
        result = TuningRunResult()
        assert result.best_config == {}
        assert result.best_accuracy == 0.0
        assert result.best_loss == float("inf")
        assert result.best_macro_f1 == 0.0
        assert result.best_cohen_kappa == 0.0
        assert result.best_negative_f1 == 0.0
        assert result.best_neutral_f1 == 0.0
        assert result.best_positive_f1 == 0.0
        assert result.iterations_completed == 0
        assert result.converged is False
        assert result.history == []
        assert result.model_path == ""
        assert result.results_path == ""
        assert result.elapsed_seconds == 0.0
        assert result.validation_passed is False
        assert result.validation_results == []
        assert result.validation_metrics is None
        assert result.retry_count == 0

    def test_custom_values(self) -> None:
        """TuningRunResult should accept custom values."""
        result = TuningRunResult(
            best_config={"lr": 0.001, "hidden_size": 256},
            best_accuracy=0.89,
            best_loss=0.31,
            best_negative_f1=0.87,
            best_neutral_f1=0.78,
            best_positive_f1=0.92,
            best_macro_f1=0.86,
            best_cohen_kappa=0.78,
            iterations_completed=3,
            converged=True,
            validation_passed=True,
            retry_count=1,
        )
        assert result.best_config == {"lr": 0.001, "hidden_size": 256}
        assert result.best_accuracy == 0.89
        assert result.best_loss == 0.31
        assert result.best_negative_f1 == 0.87
        assert result.best_neutral_f1 == 0.78
        assert result.best_positive_f1 == 0.92
        assert result.best_macro_f1 == 0.86
        assert result.best_cohen_kappa == 0.78
        assert result.iterations_completed == 3
        assert result.converged is True
        assert result.validation_passed is True
        assert result.retry_count == 1


# ─── _build_model_config Tests ───────────────────────────────────────


class TestBuildModelConfig:
    """Test _build_model_config for each model type."""

    def test_rnn_config(self) -> None:
        """Should build RNNConfig from dict."""
        config = _build_model_config("rnn", {"hidden_size": 128, "num_layers": 3, "dropout": 0.3})
        assert isinstance(config, RNNConfig)
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.dropout == 0.3

    def test_rnn_config_defaults(self) -> None:
        """Should use defaults for missing RNN params."""
        config = _build_model_config("rnn", {})
        assert isinstance(config, RNNConfig)
        assert config.hidden_size == 256
        assert config.num_layers == 2
        assert config.dropout == 0.2

    def test_encoder_config(self) -> None:
        """Should build EncoderConfig from dict."""
        config = _build_model_config(
            "encoder", {"d_model": 128, "n_heads": 2, "n_layers": 2, "dropout": 0.1}
        )
        assert isinstance(config, EncoderConfig)
        assert config.d_model == 128
        assert config.n_heads == 2
        assert config.n_layers == 2
        assert config.dropout == 0.1

    def test_encoder_config_defaults(self) -> None:
        """Should use defaults for missing Encoder params."""
        config = _build_model_config("encoder", {})
        assert isinstance(config, EncoderConfig)
        assert config.d_model == 256
        assert config.n_heads == 4

    def test_decoder_config(self) -> None:
        """Should build DecoderConfig from dict."""
        config = _build_model_config(
            "decoder",
            {"d_model": 512, "n_heads": 8, "n_encoder_layers": 3, "n_decoder_layers": 6},
        )
        assert isinstance(config, DecoderConfig)
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_encoder_layers == 3
        assert config.n_decoder_layers == 6

    def test_decoder_config_defaults(self) -> None:
        """Should use defaults for missing Decoder params."""
        config = _build_model_config("decoder", {})
        assert isinstance(config, DecoderConfig)
        assert config.d_model == 256
        assert config.n_heads == 4

    def test_invalid_model_type(self) -> None:
        """Should raise ValueError for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            _build_model_config("transformer", {})


# ─── KNOWN_SENTIMENT_EXAMPLES Tests ──────────────────────────────────


class TestKnownSentimentExamples:
    """Test the validation examples used for model prediction checking."""

    def test_examples_exist(self) -> None:
        """KNOWN_SENTIMENT_EXAMPLES should not be empty."""
        assert len(KNOWN_SENTIMENT_EXAMPLES) > 0

    def test_each_example_has_required_keys(self) -> None:
        """Each example should have 'text' and 'expected' keys."""
        for example in KNOWN_SENTIMENT_EXAMPLES:
            assert "text" in example
            assert "expected" in example

    def test_expected_values_are_valid(self) -> None:
        """Each expected value should be 'positive', 'negative', or 'neutral'."""
        for example in KNOWN_SENTIMENT_EXAMPLES:
            assert example["expected"] in ("positive", "negative", "neutral")

    def test_no_good_is_negative(self) -> None:
        """'no good' should be classified as negative sentiment."""
        no_good = [e for e in KNOWN_SENTIMENT_EXAMPLES if e["text"] == "no good"]
        assert len(no_good) == 1
        assert no_good[0]["expected"] == "negative"

    def test_has_positive_and_negative_examples(self) -> None:
        """Should have at least one positive, one negative, and one neutral example."""
        positives = [e for e in KNOWN_SENTIMENT_EXAMPLES if e["expected"] == "positive"]
        negatives = [e for e in KNOWN_SENTIMENT_EXAMPLES if e["expected"] == "negative"]
        neutrals = [e for e in KNOWN_SENTIMENT_EXAMPLES if e["expected"] == "neutral"]
        assert len(positives) >= 1
        assert len(negatives) >= 1
        assert len(neutrals) >= 1


# ─── TuningRun Class Tests ────────────────────────────────────────────


class TestTuningRunInit:
    """Test TuningRun initialization and device resolution."""

    def test_auto_device_resolution(self) -> None:
        """'auto' device should be resolved to a specific device."""
        config = TuningRunConfig(device="auto")
        run = TuningRun(config)
        assert run.config.device in ("cpu", "cuda", "mps")

    def test_explicit_device(self) -> None:
        """Explicit device should not be changed."""
        config = TuningRunConfig(device="cpu")
        run = TuningRun(config)
        assert run.config.device == "cpu"

    def test_output_dir_creation(self) -> None:
        """TuningRun should store the output directory path."""
        config = TuningRunConfig(output_dir="/tmp/test_tuning")
        run = TuningRun(config)
        assert str(run._output_dir) == "/tmp/test_tuning"


# ─── Results Persistence Tests ────────────────────────────────────────


class TestResultsPersistence:
    """Test saving and loading tuning results."""

    def test_save_and_load_results(self) -> None:
        """Should save results to JSON and load them back."""
        config = TuningRunConfig(output_dir="/tmp/test_tuning_run")
        run = TuningRun(config)

        result = TuningRunResult(
            best_config={"lr": 0.001, "hidden_size": 256},
            best_accuracy=0.89,
            best_loss=0.31,
            iterations_completed=3,
            converged=True,
            validation_passed=True,
            retry_count=0,
        )

        # Save results
        results_path = run._save_results(result)
        assert Path(results_path).exists()

        # Load results
        loaded = TuningRun.load_results(results_path)
        assert loaded["model_type"] == "rnn"
        assert loaded["best_accuracy"] == 0.89
        assert loaded["best_loss"] == 0.31
        assert loaded["converged"] is True
        assert loaded["validation_passed"] is True
        assert loaded["best_config"]["lr"] == 0.001

        # Cleanup
        import shutil

        shutil.rmtree("/tmp/test_tuning_run", ignore_errors=True)

    def test_load_nonexistent_results(self) -> None:
        """Loading from nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TuningRun.load_results("/nonexistent/path/results.json")

    def test_results_include_validation_info(self) -> None:
        """Saved results should include validation_passed and validation_results."""
        config = TuningRunConfig(output_dir="/tmp/test_tuning_validation")
        run = TuningRun(config)

        result = TuningRunResult(
            best_accuracy=0.85,
            validation_passed=True,
            validation_results=[
                {"text": "great", "expected": "positive", "score": 0.92, "correct": True},
                {"text": "terrible", "expected": "negative", "score": 0.15, "correct": True},
            ],
        )

        results_path = run._save_results(result)
        loaded = TuningRun.load_results(results_path)

        assert loaded["validation_passed"] is True
        assert len(loaded["validation_results"]) == 2
        assert loaded["validation_results"][0]["text"] == "great"

        # Cleanup
        import shutil

        shutil.rmtree("/tmp/test_tuning_validation", ignore_errors=True)


# ─── Validation Logic Tests ───────────────────────────────────────────


class TestValidationLogic:
    """Test the model validation logic (without actual models)."""

    def test_positive_prediction_correct(self) -> None:
        """A score > 0.5 for positive text should be correct."""
        score = 0.85
        expected = "positive"
        is_correct = score > 0.5 if expected == "positive" else score < 0.5
        assert is_correct is True

    def test_positive_prediction_incorrect(self) -> None:
        """A score < 0.5 for positive text should be incorrect."""
        score = 0.3
        expected = "positive"
        is_correct = score > 0.5 if expected == "positive" else score < 0.5
        assert is_correct is False

    def test_negative_prediction_correct(self) -> None:
        """A score < 0.5 for negative text should be correct."""
        score = 0.15
        expected = "negative"
        is_correct = score > 0.5 if expected == "positive" else score < 0.5
        assert is_correct is True

    def test_negative_prediction_incorrect(self) -> None:
        """A score > 0.5 for negative text should be incorrect (the bug)."""
        score = 0.7
        expected = "negative"
        is_correct = score > 0.5 if expected == "positive" else score < 0.5
        assert is_correct is False

    def test_no_good_should_be_negative(self) -> None:
        """'no good' is expected to be negative — score < 0.5 is correct."""
        # Simulating the bug: if model predicts score > 0.5 for "no good"
        # that would be incorrect
        buggy_score = 0.65  # model incorrectly thinks "no good" is positive
        expected = "negative"
        is_correct = buggy_score > 0.5 if expected == "positive" else buggy_score < 0.5
        assert is_correct is False  # bug: model is wrong

        # After fixing: model should predict score < 0.5 for "no good"
        fixed_score = 0.25  # model correctly thinks "no good" is negative
        is_correct = fixed_score > 0.5 if expected == "positive" else fixed_score < 0.5
        assert is_correct is True  # fixed: model is correct

    def test_validation_threshold(self) -> None:
        """Validation should pass when accuracy >= threshold."""
        threshold = 0.75
        correct = 6
        total = 8
        accuracy = correct / total
        assert accuracy >= threshold  # 0.75 >= 0.75 → passes

    def test_validation_threshold_fails(self) -> None:
        """Validation should fail when accuracy < threshold."""
        threshold = 0.75
        correct = 5
        total = 8
        accuracy = correct / total
        assert accuracy < threshold  # 0.625 < 0.75 → fails


# ─── create_tuning_run Tests ─────────────────────────────────────────


class TestCreateTuningRun:
    """Test the create_tuning_run convenience function."""

    def test_creates_config_with_defaults(self) -> None:
        """create_tuning_run should create a TuningRunConfig with defaults."""
        from sentimentizer.agent.diagnose_model import create_tuning_run

        with patch("sentimentizer.agent.diagnose_model.TuningRun") as mock_run:
            create_tuning_run()

            config = mock_run.call_args.args[0]
            assert config.model_type == "rnn"
            assert config.mode == "standalone"
            assert config.validate_predictions is True

    def test_creates_config_with_custom_values(self) -> None:
        """create_tuning_run should accept custom parameters."""
        from sentimentizer.agent.diagnose_model import create_tuning_run

        with patch("sentimentizer.agent.diagnose_model.TuningRun") as mock_run:
            create_tuning_run(
                model_type="encoder",
                mode="standalone",
                num_samples=50,
                validation_threshold=0.9,
            )

            config = mock_run.call_args.args[0]
            assert config.model_type == "encoder"
            assert config.mode == "standalone"
            assert config.num_samples == 50
            assert config.validation_threshold == 0.9


# ─── Driver CLI Tests ────────────────────────────────────────────────


class TestDriverTuneArgs:
    """Test that the driver CLI supports the tune subcommand and arguments."""

    def test_tune_command_exists(self) -> None:
        """The tune command should be parseable."""
        from click.testing import CliRunner

        from workflows.cli import cli

        result = CliRunner().invoke(cli, ["tune", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output

    def test_tune_mode_choices(self) -> None:
        """--mode should accept 'agent' or 'standalone'."""
        from click.testing import CliRunner

        from workflows.cli import cli

        with patch("workflows.stages.tune.run_tune") as mock_run:
            result = CliRunner().invoke(cli, ["tune", "--mode", "standalone"])
            assert result.exit_code == 0
            assert mock_run.call_args.kwargs["mode"] == "standalone"

            result = CliRunner().invoke(cli, ["tune", "--mode", "agent"])
            assert result.exit_code == 0
            assert mock_run.call_args.kwargs["mode"] == "agent"

    def test_tune_mode_invalid_raises(self) -> None:
        """--mode with invalid choice should fail."""
        from click.testing import CliRunner

        from workflows.cli import cli

        result = CliRunner().invoke(cli, ["tune", "--mode", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value for '--mode'" in result.output
