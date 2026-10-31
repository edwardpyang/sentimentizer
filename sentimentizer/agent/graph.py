"""LangGraph state graph for the tuning agent.

Defines the workflow: analyze → decide → tune → evaluate → (loop or end)

Uses LangGraph for orchestration and checkpointing, with Pydantic AI
agents (GLM 5.1 via Ollama) as the LLM reasoning layer inside nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from sentimentizer import new_logger
from sentimentizer.agent.loader import load_agent_config
from sentimentizer.agent.models import AgentRunResult
from sentimentizer.agent.nodes import analyze, decide, evaluate, tune
from sentimentizer.agent.state import AgentState
from sentimentizer.agent.websearch import reset_rate_limit
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


def should_continue(state: AgentState) -> str:
    """Conditional edge: decide whether to continue or end the loop.

    Returns 'analyze' to continue iterating, or 'end' to stop.
    """
    if state.get("converged", False):
        return "end"

    # Safety check: if iteration exceeds a hard limit, stop
    iteration = state.get("iteration", 0)
    config_path = state.get("agent_config_path")
    try:
        agent_config, _ = load_agent_config(config_path)
        hard_limit = agent_config.max_iterations * 2  # allow some extra
    except Exception:
        hard_limit = 20

    if iteration >= hard_limit:
        logger.info("hard_limit_reached", iteration=iteration, limit=hard_limit)
        return "end"

    return "analyze"


def build_graph(
    model_type: str = "",
    config_path: str | Path | None = None,
) -> StateGraph:
    """Build the LangGraph tuning agent graph.

    The graph has four nodes in a loop:
    1. analyze — Pydantic AI analysis agent examines metrics
    2. decide — Pydantic AI strategy agent chooses next action
    3. tune — Ray Tune + Optuna executes the search
    4. evaluate — Check convergence, update best results

    After evaluate, a conditional edge either loops back to
    analyze or goes to END.

    Args:
        model_type: Unused, kept for API compatibility.
        config_path: Unused, kept for API compatibility.

    Returns:
        Compiled StateGraph ready to run.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze)
    graph.add_node("decide", decide)
    graph.add_node("tune", tune)
    graph.add_node("evaluate", evaluate)

    # Define edges
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "decide")
    graph.add_edge("decide", "tune")
    graph.add_edge("tune", "evaluate")

    # Conditional edge after evaluate: loop back or end
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {"analyze": "analyze", "end": END},
    )

    return graph.compile()


def create_initial_state(
    model_type: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the initial AgentState for a new tuning run.

    Args:
        model_type: Model to tune ('rnn', 'encoder', 'decoder').
        config_path: Path to agent config YAML.

    Returns:
        Initial state dict for the graph.
    """
    return {
        "iteration": 0,
        "model_type": model_type,
        "history": [],
        "best_config": {},
        "best_accuracy": 0.0,
        "best_loss": float("inf"),
        "search_space_overrides": {},
        "agent_config_path": str(config_path) if config_path else None,
        "converged": False,
    }


async def run_agent_tuning(
    model_type: str,
    config_path: str | Path | None = None,
) -> AgentRunResult:
    """Run the complete agent tuning loop.

    This is the main entry point for the tuning agent. It:
    1. Builds the LangGraph state graph
    2. Creates initial state
    3. Runs the graph to completion
    4. Returns the best configuration found

    Args:
        model_type: Model to tune ('rnn', 'encoder', 'decoder').
        config_path: Path to agent config YAML (uses default if None).

    Returns:
        AgentRunResult with the best configuration and full history.
    """
    graph = build_graph(model_type, config_path)
    initial_state = create_initial_state(model_type, config_path)

    logger.info(
        "starting_agent_tuning",
        model_type=model_type,
        config_path=str(config_path) if config_path else "default",
    )

    # The search budget is per run; without this it lasts for the process lifetime.
    reset_rate_limit()

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    # Extract final result
    final_result = final_state.get("final_result")
    if final_result is None:
        # Build result from state if graph didn't set it explicitly
        final_result = AgentRunResult(
            best_config=final_state.get("best_config", {}),
            best_accuracy=final_state.get("best_accuracy", 0.0),
            best_loss=final_state.get("best_loss", float("inf")),
            iterations_completed=final_state.get("iteration", 0),
            converged=final_state.get("converged", False),
        )

    logger.info(
        "agent_tuning_complete",
        best_accuracy=final_result.best_accuracy,
        best_loss=final_result.best_loss,
        iterations=final_result.iterations_completed,
        converged=final_result.converged,
    )

    return final_result
