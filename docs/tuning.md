# Hyperparameter Tuning

> **Note:** Hyperparameter tuning requires the `ray` extra: `uv add "sentimentizer[ray]"`
>
> The LLM-guided mode (`--mode agent`) also needs the
> `agent` extra (LangGraph + Pydantic AI): `uv sync --extra ray --extra agent`.
> `--mode standalone`, the default, does not.

Sentimentizer offers three ways to tune hyperparameters, each at a different level of automation:

| | **Standalone** | **Iterative Agent** | **Tuning Workflow (Fixed Loop)** |
|---|---|---|---|
| **What it does** | Single Ray Tune + Optuna search | LangGraph-guided iterative search loop | High-level pipeline: tune → train → validate → retry |
| **LLM involved** | ❌ No | ✅ GLM 5.1 via Ollama | ✅ (in agent mode) or ❌ (in standalone mode) |
| **Iterative** | ❌ One-shot sweep | ✅ Refines search space each iteration | ✅ Refines + validates + retries |
| **Model validation** | ❌ | ❌ | ✅ Tests predictions on known examples |
| **Auto-retry on failure** | ❌ | ❌ | ✅ Re-tunes up to `max_retries` times |
| **Saves final model** | ❌ | ❌ | ✅ Trains & saves best model weights |
| **Requires Ollama** | ❌ No | ✅ Yes | Only in agent mode |
| **CLI** | `tune --mode standalone` | `tune --mode agent` | `tune --save` (standalone by default) |
| **When to use** | Quick sweep, no Ollama available | You want LLM-guided search but will handle model training yourself | You want a complete end-to-end pipeline |

## Standalone Tuning

Runs a single Ray Tune + Optuna hyperparameter search with no LLM involvement. Best for quick sweeps or when Ollama is unavailable.

```bash
# Via Makefile
make tune-standalone

# Via CLI
python workflows/driver.py --model rnn --tune --tune-mode standalone --save
```

This executes one [`tune_model()`](../sentimentizer/tuner.py) call — it searches the space defined in [`sentimentizer/agent/config.yaml`](../sentimentizer/agent/config.yaml) and returns the best configuration found. No iterative refinement, no model validation.

### Output

Returns a dict with the best configuration and metrics from the single search:

| Key | Description |
|-----|-------------|
| `best_config` | Best hyperparameter configuration found (e.g., `{"lr": 0.003, "hidden_size": 256}`) |
| `best_accuracy` | Best validation accuracy across all trials |
| `best_loss` | Best validation loss across all trials |
| `best_balanced_accuracy` | Best mean of per-class recalls |
| `best_cohen_kappa` | Best Cohen's kappa coefficient |
| `best_negative_f1` | Best negative-class F1 score |
| `best_neutral_f1` | Best neutral-class F1 score |
| `best_positive_f1` | Best positive-class F1 score |
| `best_macro_f1` | Best macro-averaged F1 score |
| `best_weighted_f1` | Best weighted-averaged F1 score |
| `best_mcc` | Best Matthews correlation coefficient |
| `trial_count` | Number of Ray Tune trials completed |

When run via the Tuning Workflow (`--tune --tune-mode standalone`), this is wrapped with model training, validation, and retry logic (see below).

## Iterative Agent Tuning

An LLM-guided hyperparameter tuning loop that uses **Pydantic AI Slim** (GLM 5.1 via Ollama) for reasoning, **LangGraph** for workflow orchestration, and **Ray Tune + Optuna** for the search backend. The agent iteratively refines the search space based on results from previous iterations.

### Architecture

```
analyze (GLM 5.1) → decide (GLM 5.1) → tune (Ray Tune + Optuna) → evaluate
     ↑                                                              │
     └──────────────────────────────────────────────────────────────┘
                          (loop until converged)
```

1. **analyze** — GLM 5.1 examines training metrics, detects overfitting/underfitting, assesses learning rate
2. **decide** — GLM 5.1 chooses a strategy (widen, narrow, change_focus, increase_epochs, stop) and produces a validated `TuningDecision` with an updated search space
3. **tune** — Ray Tune + Optuna executes the hyperparameter search with ASHA scheduling
4. **evaluate** — Checks convergence (improvement below threshold for 3 iterations, max iterations reached, or agent decides to stop)

### Prerequisites

Install [Ollama](https://ollama.ai) and pull the GLM 5.1 model:

```bash
ollama pull glm5.1
```

### Output

The agent returns an [`AgentRunResult`](../sentimentizer/agent/models.py) with:

| Field | Description |
|-------|-------------|
| `best_config` | Best hyperparameter configuration found (e.g., `{"lr": 0.003, "hidden_size": 256}`) |
| `best_accuracy` | Best validation accuracy achieved across all iterations |
| `best_loss` | Best validation loss achieved |
| `iterations_completed` | Number of agent loop iterations that ran |
| `converged` | Whether the agent converged before reaching `max_iterations`. Always `false` in standalone mode — a single sweep has no convergence notion. |
| `history` | List of [`TuningResult`](../sentimentizer/agent/models.py) from each iteration |

The result is always written to `best_config.json`:

```json
{
  "best_config": {"lr": 0.003, "hidden_size": 256, "num_layers": 2, "dropout": 0.2},
  "best_accuracy": 0.89,
  "best_loss": 0.31,
  "iterations": 3,
  "converged": true
}
```

> **Note:** Agent tuning (`tune --mode agent`) only runs the LLM-guided search loop — it finds the best hyperparameters but does **not** train a final model or validate predictions. Add `--save`, or use the Tuning Workflow below, to get a trained and validated model.

### Usage

```bash
# Via Makefile (standalone is the default; tune-agent selects the LLM loop)
make tune MODEL=encoder
make tune-agent MODEL=encoder

# Via CLI
sentimentizer --model encoder tune --mode agent --save

# With a custom agent config
sentimentizer --model encoder tune --mode agent --agent-config path/to/custom.yaml --save

# Standalone (single Ray Tune sweep, no LLM agent)
sentimentizer --model encoder tune --mode standalone --save
```

The tuning mode is selected with `--mode agent|standalone`; there is no
`--agent-tune` flag. The default is `standalone`; `make tune-agent` is the
Makefile shortcut for the agent form.

## Tuning Workflow Pipeline

The **Tuning Workflow** (`TuningRun` in [`sentimentizer/agent/diagnose_model.py`](../sentimentizer/agent/diagnose_model.py)) is the highest-level tuning interface. It wraps either agent-guided or standalone tuning with additional post-tuning steps:

1. **Tune** — Runs agent-guided (`mode="agent"`) or standalone (`mode="standalone"`) hyperparameter search
2. **Train** — Trains a final model using the best configuration found (2× default epochs for better convergence)
3. **Validate** — Tests the trained model against known sentiment examples (e.g., "amazing food great service" → positive, "terrible experience" → negative)
4. **Retry** — If validation fails (accuracy below threshold), re-tunes with adjusted parameters up to `max_retries` times

```
┌──────────────────────────────────────────────────┐
│              Tuning Workflow                     │
│                                                   │
│  ┌─────────┐    ┌─────────┐    ┌──────────────┐  │
│  │  Tune    │───▶│  Train  │───▶│  Validate    │  │
│  │(agent or │    │  final  │    │  predictions │  │
│  │standalone)│   │  model  │    │  on known    │  │
│  └─────────┘    └─────────┘    │  examples    │  │
│       ▲                        └──────┬───────┘  │
│       │                               │           │
│       └─────────── retry ────────────┘           │
│              (if validation fails)                 │
└──────────────────────────────────────────────────┘
```

### Output

Returns a [`TuningRunResult`](../sentimentizer/agent/diagnose_model.py) with:

| Field | Description |
|-------|-------------|
| `best_config` | Best hyperparameter configuration found |
| `best_accuracy` | Best validation accuracy achieved |
| `best_loss` | Best validation loss achieved |
| `best_balanced_accuracy` | Best mean of per-class recalls |
| `best_cohen_kappa` | Best Cohen's kappa coefficient |
| `best_negative_f1` | Best negative-class F1 score |
| `best_neutral_f1` | Best neutral-class F1 score |
| `best_positive_f1` | Best positive-class F1 score |
| `best_macro_f1` | Best macro-averaged F1 score |
| `best_weighted_f1` | Best weighted-averaged F1 score |
| `best_mcc` | Best Matthews correlation coefficient |
| `iterations_completed` | Number of tuning iterations (1 for standalone, variable for agent) |
| `converged` | Whether the agent converged before max iterations. Always `false` for standalone mode (single sweep, no convergence notion). |
| `model_path` | Path to the saved model weights (`.pth` file) |
| `results_path` | Path to the saved JSON results file |
| `validation_passed` | Whether model predictions met the validation threshold |
| `validation_results` | Per-example validation details (text, expected, score, correct) |
| `validation_metrics` | Full [`ClassificationMetrics`](../sentimentizer/metrics.py) dict from model validation |
| `retry_count` | Number of re-tuning attempts due to failed validation |
| `elapsed_seconds` | Wall-clock time for the entire run |

Results are saved to `tuning_results/tuning_results_{model_type}.json`. If validation passes, the best model weights are also copied to the default weights path for serving.

### Usage

```bash
# Agent-guided tuning with model validation (recommended, defaults to RNN)
make tune

# Tune specific models
make tune-rnn
make tune-encoder
make tune-decoder

# Standalone mode (no LLM, single Ray Tune sweep, still validates model)
make tune-standalone

# Customize the number of trials and agent iterations
make tune-custom SAMPLES=50 ITERATIONS=10

# Skip model validation
make tune-no-validate
```

Via CLI:

```bash
# Agent-guided tuning (default)
python workflows/driver.py --model rnn --tune --save

# Standalone tuning (no LLM)
python workflows/driver.py --model rnn --tune --tune-mode standalone --save

# Customize trials, iterations, and validation
python workflows/driver.py --model encoder --tune --save \
  --tune-samples 50 \
  --tune-max-iterations 10 \
  --validation-threshold 0.8 \
  --max-retries 3

# Skip validation
python workflows/driver.py --model rnn --tune --no-validate --save
```

Programmatic API:

```python
from sentimentizer.agent.diagnose_model import TuningRun, TuningRunConfig

# Agent-guided tuning with validation (recommended)
config = TuningRunConfig(model_type="rnn", mode="agent")
result = TuningRun(config).execute()
print(f"Best accuracy: {result.best_accuracy:.4f}")
print(f"Validation passed: {result.validation_passed}")

# Standalone tuning with validation
config = TuningRunConfig(model_type="encoder", mode="standalone")
result = TuningRun(config).execute()

# Quick convenience function
from sentimentizer.agent.diagnose_model import create_tuning_run
result = create_tuning_run(model_type="rnn", mode="agent")
```

## Configuration

Agent and tuner settings are defined in [`sentimentizer/agent/config.yaml`](../sentimentizer/agent/config.yaml):

```yaml
agent:
  model_name: glm5.1                    # Ollama model name
  ollama_base_url: http://localhost:11434/v1
  max_iterations: 5                      # Max agent loop iterations
  convergence_threshold: 0.005           # Stop if avg improvement < threshold over 3 iterations
  temperature: 0.3                       # LLM sampling temperature
  max_tokens: 2048                       # Max LLM output tokens
  checkpointing:
    enabled: true
    db_path: agent_checkpoints.db
  human_in_the_loop: false               # Require human approval (future)

tuner:
  scheduler: asha                        # asha, hyperband, or median
  metric: val_accuracy
  mode: max
  num_samples: 20                        # Trials per tuning iteration
  grace_period: 2
  reduction_factor: 3
  search_spaces:
    rnn:
      lr: { type: loguniform, low: 1e-5, high: 1e-2 }
      hidden_size: { type: choice, values: [128, 256, 512] }
      ...
```

Override the config path via the `SENTIMENTIZER_AGENT_CONFIG` environment variable.
