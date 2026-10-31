"""Tests for the sentimentizer tuning agent.

Tests cover:
- Config loader (YAML → dataclass)
- Pydantic models (structured validation)
- Search space building
- Convergence checking
- Graph construction
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

# ─── Config Loader Tests ───────────────────────────────────────────


class TestConfigLoader:
    """Test YAML config loading for AgentConfig and TunerConfig."""

    def test_load_default_config(self):
        """Loading the bundled config.yaml should succeed with defaults."""
        from sentimentizer.agent.loader import load_agent_config

        agent_cfg, tuner_cfg = load_agent_config()
        assert agent_cfg.model_name == "glm-5.1:cloud"
        assert agent_cfg.ollama_base_url == "http://localhost:11434/v1"
        assert agent_cfg.max_iterations == 5
        assert agent_cfg.convergence_threshold == 0.005
        assert agent_cfg.temperature == 0.3
        assert agent_cfg.max_tokens == 2048
        assert agent_cfg.checkpointing.enabled is True
        assert agent_cfg.human_in_the_loop is False

    def test_load_tuner_config(self):
        """TunerConfig should have correct scheduler and search spaces."""
        from sentimentizer.agent.loader import load_agent_config

        _, tuner_cfg = load_agent_config()
        assert tuner_cfg.scheduler == "asha"
        assert tuner_cfg.metric == "val_accuracy"
        assert tuner_cfg.mode == "max"
        assert tuner_cfg.num_samples == 20
        assert "rnn" in tuner_cfg.search_spaces
        assert "encoder" in tuner_cfg.search_spaces
        assert "decoder" in tuner_cfg.search_spaces

    def test_rnn_search_space_has_expected_params(self):
        """RNN search space should contain lr, hidden_size, etc."""
        from sentimentizer.agent.loader import load_agent_config
        from sentimentizer.tuner import load_search_space

        _, tuner_cfg = load_agent_config()
        space = load_search_space("rnn", tuner_config=tuner_cfg)
        assert "lr" in space
        assert "hidden_size" in space
        assert "num_layers" in space
        assert "dropout" in space
        assert "weight_decay" in space
        assert "batch_size" in space

    def test_encoder_search_space(self):
        """Encoder search space should have transformer-specific params."""
        from sentimentizer.agent.loader import load_agent_config
        from sentimentizer.tuner import load_search_space

        _, tuner_cfg = load_agent_config()
        space = load_search_space("encoder", tuner_config=tuner_cfg)
        assert "lr" in space
        assert "d_model" in space
        assert "n_heads" in space
        assert "n_layers" in space

    def test_decoder_search_space(self):
        """Decoder search space should have encoder/decoder layer counts."""
        from sentimentizer.agent.loader import load_agent_config
        from sentimentizer.tuner import load_search_space

        _, tuner_cfg = load_agent_config()
        space = load_search_space("decoder", tuner_config=tuner_cfg)
        assert "n_encoder_layers" in space
        assert "n_decoder_layers" in space

    def test_invalid_model_type_raises(self):
        """Loading search space for unknown model type should raise ValueError."""
        from sentimentizer.tuner import load_search_space

        with pytest.raises(ValueError, match="No search space defined"):
            load_search_space("transformer")

    def test_custom_config_file(self):
        """Loading from a custom YAML file should override defaults."""
        from sentimentizer.agent.loader import load_agent_config

        custom_config = {
            "agent": {
                "model_name": "llama3",
                "ollama_base_url": "http://custom-host:11434/v1",
                "max_iterations": 10,
                "temperature": 0.7,
                "checkpointing": {"enabled": False, "db_path": "custom.db"},
            },
            "tuner": {
                "scheduler": "hyperband",
                "num_samples": 50,
                "search_spaces": {
                    "rnn": {
                        "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
                        "hidden_size": {"type": "choice", "values": [64, 128, 256]},
                    }
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom_config, f)
            f.flush()
            try:
                agent_cfg, tuner_cfg = load_agent_config(f.name)
                assert agent_cfg.model_name == "llama3"
                assert agent_cfg.max_iterations == 10
                assert agent_cfg.temperature == 0.7
                assert agent_cfg.checkpointing.enabled is False
                assert agent_cfg.checkpointing.db_path == "custom.db"
                assert tuner_cfg.scheduler == "hyperband"
                assert tuner_cfg.num_samples == 50
                assert "rnn" in tuner_cfg.search_spaces
            finally:
                os.unlink(f.name)

    def test_missing_config_file_raises(self):
        """Loading from a nonexistent path should raise FileNotFoundError."""
        from sentimentizer.agent.loader import load_agent_config

        with pytest.raises(FileNotFoundError):
            load_agent_config("/nonexistent/path/config.yaml")

    def test_env_var_config_path(self):
        """get_config_path should read from environment variable."""
        from sentimentizer.agent.loader import get_config_path

        original = os.environ.get("SENTIMENTIZER_AGENT_CONFIG")
        try:
            os.environ["SENTIMENTIZER_AGENT_CONFIG"] = "/custom/path/config.yaml"
            result = get_config_path()
            assert result == Path("/custom/path/config.yaml")
        finally:
            if original is None:
                os.environ.pop("SENTIMENTIZER_AGENT_CONFIG", None)
            else:
                os.environ["SENTIMENTIZER_AGENT_CONFIG"] = original

    def test_env_var_not_set(self):
        """get_config_path should return None when env var is not set."""
        from sentimentizer.agent.loader import get_config_path

        original = os.environ.pop("SENTIMENTIZER_AGENT_CONFIG", None)
        try:
            result = get_config_path()
            assert result is None
        finally:
            if original is not None:
                os.environ["SENTIMENTIZER_AGENT_CONFIG"] = original


# ─── Pydantic Models Tests ─────────────────────────────────────────


class TestPydanticModels:
    """Test structured output validation for agent models."""

    def test_analysis_result_creation(self):
        """AnalysisResult should validate and store fields."""
        from sentimentizer.agent.models import AnalysisResult

        result = AnalysisResult(
            summary="Model is overfitting slightly",
            overfitting=True,
            underfitting=False,
            lr_status="appropriate",
            suggested_focus=["dropout", "weight_decay"],
        )
        assert result.overfitting is True
        assert result.lr_status == "appropriate"
        assert "dropout" in result.suggested_focus

    def test_tuning_decision_creation(self):
        """TuningDecision should validate strategy and search space."""
        from sentimentizer.agent.models import SearchSpaceParam, TuningDecision

        decision = TuningDecision(
            reasoning="Narrowing search around best lr",
            strategy="narrow",
            search_space={
                "lr": SearchSpaceParam(type="loguniform", low=1e-4, high=1e-3),
                "dropout": SearchSpaceParam(type="uniform", low=0.2, high=0.4),
            },
            num_samples=15,
        )
        assert decision.strategy == "narrow"
        assert "lr" in decision.search_space
        assert decision.num_samples == 15

    def test_tuning_decision_invalid_strategy(self):
        """TuningDecision with invalid strategy should raise validation error."""
        from pydantic import ValidationError

        from sentimentizer.agent.models import TuningDecision

        with pytest.raises(ValidationError):
            TuningDecision(
                reasoning="test",
                strategy="invalid_strategy",  # type: ignore[arg-type]
                search_space={},
                num_samples=10,
            )

    def test_tuning_result_creation(self):
        """TuningResult should store metrics from a tuning run."""
        from sentimentizer.agent.models import TuningResult

        result = TuningResult(
            best_accuracy=0.89,
            best_loss=0.31,
            best_config={"lr": 0.001, "hidden_size": 256},
            trial_count=20,
            improvement_over_last=0.02,
        )
        assert result.best_accuracy == 0.89
        assert result.trial_count == 20
        assert result.improvement_over_last == 0.02

    def test_agent_run_result_creation(self):
        """AgentRunResult should store final tuning outcome."""
        from sentimentizer.agent.models import AgentRunResult

        result = AgentRunResult(
            best_config={"lr": 0.001, "hidden_size": 256},
            best_accuracy=0.89,
            best_loss=0.31,
            iterations_completed=3,
            converged=True,
        )
        assert result.converged is True
        assert result.iterations_completed == 3

    def test_search_space_param_loguniform(self):
        """SearchSpaceParam should store loguniform params."""
        from sentimentizer.agent.models import SearchSpaceParam

        param = SearchSpaceParam(type="loguniform", low=1e-5, high=1e-2)
        assert param.type == "loguniform"
        assert param.low == 1e-5
        assert param.high == 1e-2

    def test_search_space_param_choice(self):
        """SearchSpaceParam should store choice params."""
        from sentimentizer.agent.models import SearchSpaceParam

        param = SearchSpaceParam(type="choice", values=[128, 256, 512])
        assert param.type == "choice"
        assert param.values == [128, 256, 512]

    def test_search_space_param_invalid_type(self):
        """SearchSpaceParam with invalid type should raise validation error."""
        from pydantic import ValidationError

        from sentimentizer.agent.models import SearchSpaceParam

        with pytest.raises(ValidationError):
            SearchSpaceParam(type="invalid", low=0, high=1)  # type: ignore[arg-type]


# ─── Search Space Building Tests ────────────────────────────────────


class TestSearchSpaceBuilding:
    """Test Ray Tune search space construction from YAML config."""

    def test_build_search_space_rnn(self):
        """Building RNN search space should produce valid Ray Tune domains."""
        pytest.importorskip("ray")

        from sentimentizer.tuner import build_search_space

        space = build_search_space("rnn")
        # Should have keys for each hyperparameter
        assert "lr" in space
        assert "hidden_size" in space
        assert "num_layers" in space
        assert "dropout" in space

    def test_build_search_space_encoder(self):
        """Building encoder search space should include transformer params."""
        pytest.importorskip("ray")

        from sentimentizer.tuner import build_search_space

        space = build_search_space("encoder")
        assert "lr" in space
        assert "d_model" in space
        assert "n_heads" in space

    def test_build_search_space_with_overrides(self):
        """Overrides should replace default search space entries."""
        pytest.importorskip("ray")

        from sentimentizer.tuner import build_search_space

        overrides = {
            "lr": {"type": "loguniform", "low": 1e-4, "high": 1e-3},
        }
        space = build_search_space("rnn", overrides=overrides)
        assert "lr" in space

    def test_build_search_space_invalid_model(self):
        """Building search space for invalid model should raise ValueError."""
        pytest.importorskip("ray")

        from sentimentizer.tuner import build_search_space

        with pytest.raises(ValueError):
            build_search_space("invalid_model")


# ─── Convergence Tests ──────────────────────────────────────────────


class TestConvergence:
    """Test convergence checking logic."""

    def test_convergence_on_agent_stop(self):
        """Should converge when agent decides to stop."""
        from sentimentizer.agent.models import TuningDecision
        from sentimentizer.agent.nodes import _check_convergence

        decision = TuningDecision(
            reasoning="Results converged",
            strategy="stop",
            search_space={},
        )
        assert _check_convergence(
            history=[{"improvement_over_last": 0.001}],
            threshold=0.005,
            max_iterations=5,
            iteration=1,
            decision=decision,
        )

    def test_convergence_on_max_iterations(self):
        """Should converge when max iterations reached."""
        from sentimentizer.agent.nodes import _check_convergence

        assert _check_convergence(
            history=[],
            threshold=0.005,
            max_iterations=5,
            iteration=5,
        )

    def test_convergence_on_low_improvement(self):
        """Should converge when improvement is below threshold for 3 iterations."""
        from sentimentizer.agent.nodes import _check_convergence

        history = [
            {"improvement_over_last": 0.001},
            {"improvement_over_last": 0.002},
            {"improvement_over_last": 0.001},
        ]
        assert _check_convergence(
            history=history,
            threshold=0.005,
            max_iterations=10,
            iteration=5,
        )

    def test_no_convergence_when_improving(self):
        """Should not converge when improvement is above threshold."""
        from sentimentizer.agent.nodes import _check_convergence

        history = [
            {"improvement_over_last": 0.05},
        ]
        assert not _check_convergence(
            history=history,
            threshold=0.005,
            max_iterations=10,
            iteration=1,
        )

    def test_no_convergence_at_start(self):
        """Should not converge on first iteration with no history."""
        from sentimentizer.agent.nodes import _check_convergence

        assert not _check_convergence(
            history=[],
            threshold=0.005,
            max_iterations=10,
            iteration=0,
        )


# ─── Graph Construction Tests ───────────────────────────────────────


class TestGraphConstruction:
    """Test LangGraph state graph construction."""

    def test_build_graph(self):
        """Building the graph should produce a compiled graph."""
        from sentimentizer.agent.graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_create_initial_state(self):
        """Initial state should have all required keys."""
        from sentimentizer.agent.graph import create_initial_state

        state = create_initial_state(model_type="encoder")
        assert state["iteration"] == 0
        assert state["model_type"] == "encoder"
        assert state["history"] == []
        assert state["best_accuracy"] == 0.0
        assert state["converged"] is False


class TestSearchBudget:
    """The web search budget is documented as per agent run."""

    def test_each_agent_run_gets_a_fresh_budget(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from sentimentizer.agent import graph as graph_module
        from sentimentizer.agent import websearch

        for _ in range(websearch.MAX_CALLS_PER_RUN):
            websearch._check_rate_limit()
        with pytest.raises(RuntimeError, match="rate limit"):
            websearch._check_rate_limit()

        compiled = AsyncMock()
        compiled.ainvoke.return_value = {"best_config": {}, "best_accuracy": 0.5}
        with (
            patch.object(graph_module, "build_graph", return_value=compiled),
            patch.object(graph_module, "create_initial_state", return_value={}),
        ):
            asyncio.run(graph_module.run_agent_tuning("rnn"))

        websearch._check_rate_limit()
        websearch.reset_rate_limit()
