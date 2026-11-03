"""Sentimentizer hyperparameter tuning agent.

Uses Pydantic AI Slim (GLM 5.1 via Ollama) for LLM reasoning,
LangGraph for workflow orchestration, and Ray Tune + Optuna for
hyperparameter search.

LangGraph and Pydantic AI come from the optional ``agent`` extra, so
``run_agent_tuning`` is imported lazily; everything else here works without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentimentizer.agent.diagnose_model import (
    TuningRun,
    TuningRunConfig,
    TuningRunResult,
    create_tuning_run,
    diagnose_training_issues,
    require_agent_extra,
)
from sentimentizer.agent.loader import AgentConfig, TunerConfig, load_agent_config
from sentimentizer.agent.websearch import (
    WebSearchResult,
    reset_rate_limit,
    sanitize_content,
    validate_query,
    web_search,
)

if TYPE_CHECKING:
    from sentimentizer.agent.graph import run_agent_tuning

__all__ = [
    "AgentConfig",
    "TunerConfig",
    "TuningRun",
    "TuningRunConfig",
    "TuningRunResult",
    "WebSearchResult",
    "create_tuning_run",
    "diagnose_training_issues",
    "load_agent_config",
    "require_agent_extra",
    "reset_rate_limit",
    "run_agent_tuning",
    "sanitize_content",
    "validate_query",
    "web_search",
]


def __getattr__(name: str) -> Any:
    """Import ``run_agent_tuning`` on first access so the agent extra stays optional."""
    if name == "run_agent_tuning":
        require_agent_extra()
        from sentimentizer.agent.graph import run_agent_tuning

        return run_agent_tuning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
