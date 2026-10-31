"""Click CLI: commands and shared options.

IMPORTANT: This module imports workflows.lifecycle at the top level
(which triggers env setup and cleanup registration). It does NOT import
any stage modules at module level — stages are imported inside command
handlers to keep ``--help`` fast and avoid pulling in torch/ray.
"""

from __future__ import annotations

from typing import Any

import click

from workflows.lifecycle import State

# ──────────────────────────────────────────────
# Shared decorator for train/run flags
# ──────────────────────────────────────────────


def shared_train_options(func: click.Command) -> click.Command:
    """Shared flags for ``train`` and ``run`` — keeps their surfaces in lockstep.

    ``--resume`` is intentionally NOT in this list because ``run`` exposes
    it as split flags (--resume-tokenize / --resume-train) while ``train``
    keeps the unified ``--resume``.
    """
    options = [
        click.option("--distributed", is_flag=True, help="Use Ray Train for distributed training"),
        click.option("--num-workers", default=1, type=int, help="Number of Ray Train workers"),
        click.option("--save/--no-save", default=False, help="Save model weights after training"),
        click.option("--checkpoint-dir", default="", type=str, help="Directory for checkpoints"),
        click.option(
            "--checkpoint-every", default=1, type=int, help="Save checkpoint every N epochs"
        ),
        click.option(
            "--balance-classes",
            is_flag=True,
            help="Enable class balancing (undersampling majority)",
        ),
        click.option(
            "--balance-seed", default=42, type=int, help="Random seed for class balancing"
        ),
        click.option(
            "--num-classes",
            default=3,
            type=int,
            help="Number of classification classes (must be 3 for negative/neutral/positive)",
        ),
        click.option(
            "--include-neutral/--no-include-neutral",
            default=True,
            help="Include 3-star (neutral) reviews in training data",
        ),
        click.option(
            "--loss-type",
            default="cross_entropy",
            type=click.Choice(["cross_entropy", "focal"]),
            help="Loss function type",
        ),
        click.option(
            "--focal-gamma",
            default=2.0,
            type=float,
            help="Focal loss focusing parameter (only used with --loss-type=focal)",
        ),
        click.option(
            "--label-smoothing",
            default=0.1,
            type=float,
            help="Label smoothing for CrossEntropyLoss (0.0 = no smoothing)",
        ),
        click.option(
            "--weight-smoothing",
            default=0.5,
            type=float,
            help="Exponent on inverse-frequency class weights (1.0=full, 0.5=sqrt, 0.0=uniform)",
        ),
        click.option(
            "--neutral-oversample-ratio",
            default=0.0,
            type=float,
            help="Oversample neutral class to this ratio (0.0=disabled, 0.20=20%)",
        ),
        click.option(
            "--balance-strategy",
            default="class_weights_only",
            type=click.Choice(["class_weights_only", "undersample", "oversample"]),
            help="Class balancing strategy",
        ),
        click.option(
            "--freeze-embeddings/--no-freeze-embeddings",
            default=True,
            help="Freeze pre-trained GloVe embedding weights (unfreeze to fine-tune)",
        ),
        click.option("--push-to-hub", is_flag=True, help="Push model weights to Hugging Face Hub"),
        click.option(
            "--pull-from-hub", is_flag=True, help="Pull model weights from Hugging Face Hub"
        ),
        click.option("--hf-repo", default=None, type=str, help="Hugging Face repository ID"),
        click.option(
            "--run-id",
            default="",
            type=str,
            help="Unique training run ID for metrics tracking",
        ),
        click.option(
            "--ray-update-every",
            default=-1,
            type=int,
            help="Batch update/logging frequency for distributed training (-1 = auto)",
        ),
    ]
    for option in reversed(options):
        func = option(func)  # type: ignore[misc]
    return func


# ──────────────────────────────────────────────
# CLI group definition
# ──────────────────────────────────────────────


@click.group()
@click.option(
    "--model",
    default="rnn",
    type=click.Choice(["rnn", "encoder", "decoder", "modernbert"]),
    envvar="SENTIMENTIZER_MODEL",
    help="Model architecture: rnn, encoder, decoder, or modernbert",
)
@click.option(
    "--device",
    default="auto",
    envvar="SENTIMENTIZER_DEVICE",
    help="Compute device: auto, cuda, mps, or cpu (resolved lazily)",
)
@click.option(
    "--run-type",
    default="new",
    type=click.Choice(["new", "update"]),
    help="new = build/train fresh; update = reuse existing artifacts",
)
@click.pass_context
def cli(ctx: click.Context, model: str, device: str, run_type: str) -> None:
    """Sentimentizer ML pipeline."""
    # Store raw values only. Do NOT call auto_detect_device here or import
    # sentimentizer.config — it transitively imports torch, which would
    # trigger torch loading for every <subcommand> --help.
    ctx.obj = State(model=model, device=device, run_type=run_type)


# ── extract ──────────────────────────────────


@cli.command()
@click.option("--stop", default=10000, type=int, help="Number of review lines to load")
@click.pass_context
def extract(ctx: click.Context, stop: int) -> None:
    """Extract raw reviews into parquet."""
    from workflows.stages.extract import run_extract

    run_extract(ctx.obj, stop=stop)


# ── tokenize ─────────────────────────────────


@cli.command()
@click.option(
    "--resume",
    is_flag=True,
    help="Update the existing dictionary from new data",
)
@click.pass_context
def tokenize(ctx: click.Context, resume: bool) -> None:
    """Build/update dictionary and write processed parquet."""
    from workflows.stages.tokenize import run_tokenize

    run_tokenize(ctx.obj, resume=resume)


# ── train ────────────────────────────────────


@cli.command()
@shared_train_options
@click.option(
    "--resume",
    is_flag=True,
    help="Resume training from latest checkpoint in --checkpoint-dir",
)
@click.pass_context
def train(ctx: click.Context, resume: bool, **kwargs: Any) -> None:
    """Fit the model (single-node or distributed)."""
    from workflows.stages.train import run_train

    run_train(ctx.obj, resume=resume, **kwargs)


# ── tune ─────────────────────────────────────


@cli.command()
@click.option(
    "--mode",
    default="agent",
    type=click.Choice(["agent", "standalone"]),
    help="Tuning mode: agent (LLM-guided loop) or standalone (Ray Tune)",
)
@click.option("--agent-config", default=None, type=str, help="Path to agent config YAML")
@click.option(
    "--samples",
    "tune_samples",
    default=None,
    type=int,
    help="Ray Tune trials per iteration (default: from agent config YAML)",
)
@click.option(
    "--max-iterations",
    "tune_max_iterations",
    default=None,
    type=int,
    help="Maximum agent tuning iterations (default: from agent config YAML)",
)
@click.option(
    "--output-dir",
    "tune_output_dir",
    default="tuning_results",
    type=str,
    help="Directory for tuning results",
)
@click.option("--no-validate", is_flag=True, help="Skip model prediction validation")
@click.option(
    "--validation-threshold",
    default=0.75,
    type=float,
    help="Minimum fraction of correct predictions to pass validation",
)
@click.option("--max-retries", default=2, type=int, help="Max re-tuning attempts on failure")
@click.option("--save/--no-save", default=False, help="Save best model after tuning")
@click.option("--push-to-hub", is_flag=True, help="Push model weights to Hugging Face Hub")
@click.option("--balance-classes", is_flag=True, help="Enable class balancing in training data")
@click.option("--balance-seed", default=42, type=int, help="Random seed for class balancing")
@click.option(
    "--weight-smoothing",
    default=0.5,
    type=float,
    help="Inverse-frequency exponent (1.0=full, 0.5=sqrt, 0.0=uniform)",
)
@click.option(
    "--loss-type",
    default="cross_entropy",
    type=click.Choice(["cross_entropy", "focal"]),
    help="Loss function type",
)
@click.option(
    "--focal-gamma",
    default=2.0,
    type=float,
    help="Focal loss focusing parameter (loss_type=focal only)",
)
@click.option("--label-smoothing", default=0.1, type=float, help="Label smoothing coefficient")
@click.option(
    "--neutral-oversample-ratio",
    default=0.0,
    type=float,
    help="Neutral class oversample ratio (0.0=disabled)",
)
@click.option(
    "--balance-strategy",
    default="class_weights_only",
    type=click.Choice(["class_weights_only", "undersample", "oversample"]),
    help="Class imbalance strategy",
)
@click.pass_context
def tune(ctx: click.Context, mode: str, **kwargs: Any) -> None:
    """Hyperparameter tuning (LLM-guided agent or standalone Ray Tune run)."""
    from workflows.stages.tune import run_tune

    run_tune(ctx.obj, mode=mode, **kwargs)


# ── hf ───────────────────────────────────────


@cli.group()
def hf() -> None:
    """Hugging Face Hub operations."""


@hf.command("push")
@click.option("--repo-id", default=None, type=str, help="Hugging Face repository ID")
@click.pass_context
def hf_push(ctx: click.Context, repo_id: str | None) -> None:
    """Push model weights to Hugging Face Hub."""
    from workflows.stages.hf import run_hf_push

    run_hf_push(ctx.obj, repo_id=repo_id)


@hf.command("pull")
@click.option("--repo-id", default=None, type=str, help="Hugging Face repository ID")
@click.pass_context
def hf_pull(ctx: click.Context, repo_id: str | None) -> None:
    """Pull model weights from Hugging Face Hub."""
    from workflows.stages.hf import run_hf_pull

    run_hf_pull(ctx.obj, repo_id=repo_id)


# ── export ────────────────────────────────────


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["rnn", "encoder", "decoder", "modernbert"]),
    required=True,
    help="Model to export to ONNX",
)
@click.option("--quantize/--no-quantize", default=True, help="Apply INT8 quantization")
@click.option("--output-dir", default="onnx_artifacts", help="Output directory")
@click.pass_context
def export(ctx: click.Context, model: str, quantize: bool, output_dir: str) -> None:
    """Export a trained model to ONNX format."""
    from workflows.stages.export import run_export

    run_export(ctx.obj, model_type=model, quantize=quantize, output_dir=output_dir)


# ── router ─────────────────────────────────────


@cli.group()
def router() -> None:
    """Router operations (train, evaluate, push)."""


@router.command("train")
@click.option("--data", required=True, type=click.Path(exists=True), help="Training JSONL data")
@click.option(
    "--validation-data", type=click.Path(exists=True), help="Independent validation JSONL"
)
@click.option(
    "--base-model",
    default=None,
    help="Sentence-transformer base model (overrides config default)",
)
@click.option("--output-dir", default=None, help="Output directory for trained model")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to router config YAML (defaults to router/config.yaml)",
)
@click.pass_context
def router_train(
    ctx: click.Context,
    data: str,
    validation_data: str | None,
    base_model: str | None,
    output_dir: str | None,
    config_path: str | None,
) -> None:
    """Train the router model."""
    from dataclasses import replace
    from pathlib import Path

    from sentimentizer.router.config import load_router_config
    from sentimentizer.router.dataset import (
        assert_disjoint,
        load_router_dataset,
        load_router_records,
    )
    from sentimentizer.router.train_router import train_router

    train_cfg, _ = load_router_config(config_path)
    config = replace(
        train_cfg,
        base_model=base_model or train_cfg.base_model,
        output_dir=Path(output_dir) if output_dir else train_cfg.output_dir,
    )
    if validation_data:
        from datasets import Dataset

        train_rows, eval_rows = load_router_records(data), load_router_records(validation_data)
        assert_disjoint(train_rows, eval_rows)
        train_ds, eval_ds = Dataset.from_list(train_rows), Dataset.from_list(eval_rows)
    else:
        train_ds, eval_ds = load_router_dataset(data, seed=config.seed)
    train_router(config, train_ds, eval_ds)


@router.command("split")
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option(
    "--test-size", default=0.2, type=click.FloatRange(min=0, max=1, min_open=True, max_open=True)
)
@click.option(
    "--validation-size",
    default=0.2,
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
)
@click.option("--seed", default=42, type=int)
@click.option("--calibration-size", default=0.0, type=click.FloatRange(min=0, max=1, max_open=True))
@click.option("--policy-size", default=0.0, type=click.FloatRange(min=0, max=1, max_open=True))
def router_split(
    data: str,
    output_dir: str,
    test_size: float,
    validation_size: float,
    seed: int,
    calibration_size: float,
    policy_size: float,
) -> None:
    """Split original sources BEFORE augmenting the training partition."""
    from pathlib import Path

    from sentimentizer.router.dataset import (
        assert_disjoint,
        load_router_records,
        save_dataset_to_jsonl,
        split_router_records,
    )

    sizes = {
        "test": test_size,
        "validation": validation_size,
        "calibration": calibration_size,
        "policy": policy_size,
    }
    if sum(sizes.values()) >= 1:
        raise click.UsageError("Held-out split fractions must sum to less than one")
    train = load_router_records(data)
    remaining = 1.0
    partitions = {}
    for name, size in sizes.items():
        if size:
            train, partitions[name] = split_router_records(train, size / remaining, seed)
            remaining -= size
    partitions["train"] = train
    assert_disjoint(*partitions.values())
    for name, rows in partitions.items():
        save_dataset_to_jsonl(rows, str(Path(output_dir) / f"{name}.jsonl"))
        click.echo(f"{name}: {len(rows)} records")


@router.command("augment")
@click.option(
    "--seeds",
    "seed_path_override",
    type=click.Path(exists=True),
    help="Training partition JSONL; overrides seed_path",
)
@click.option(
    "--output",
    default=None,
    help="Output JSONL file path (overrides config default)",
)
@click.option(
    "--model",
    default=None,
    help="Ollama model name (overrides config default)",
)
@click.option(
    "--variations",
    default=None,
    type=int,
    help="Number of variations per seed (overrides config default)",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama API endpoint (overrides config default)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume missing variations using persisted seed lineage and counts",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to router config YAML (defaults to router/config.yaml)",
)
@click.pass_context
def router_augment(
    ctx: click.Context,
    seed_path_override: str | None,
    output: str | None,
    model: str | None,
    variations: int | None,
    ollama_url: str | None,
    resume: bool,
    config_path: str | None,
) -> None:
    """Augment seed utterances using GLM 5.1 via Ollama.

    Streams each entry to the output JSONL file as it's generated,
    so you can see progress in real-time. Defaults come from
    router/config.yaml; CLI flags override when provided.

    Use --resume to retry missing variations for incomplete seeds.
    """
    from pathlib import Path

    from sentimentizer.router.augment import augment_seeds, augmentation_counts, prepare_seeds
    from sentimentizer.router.config import FOOD_TASK_SPEC, load_router_config
    from sentimentizer.router.dataset import load_router_records
    from sentimentizer.router.seeds import SEED_UTTERANCES

    train_cfg, augment_cfg = load_router_config(config_path)
    if seed_path_override or augment_cfg.seed_path:
        seed_path = Path(seed_path_override or augment_cfg.seed_path)
        if not seed_path_override and config_path and not seed_path.is_absolute():
            seed_path = Path(config_path).parent / seed_path
        seeds = load_router_records(seed_path)
    elif train_cfg.task == FOOD_TASK_SPEC:
        seeds = SEED_UTTERANCES
    else:
        raise click.UsageError("Custom tasks require augmentation.seed_path")
    effective_model = model or augment_cfg.model
    effective_url = ollama_url or augment_cfg.ollama_url
    effective_variations = variations if variations is not None else augment_cfg.variations_per_seed
    effective_output = output or augment_cfg.output_path

    if resume:
        click.echo(f"Resuming augmentation to {effective_output}...")
    else:
        click.echo(f"Augmenting {len(seeds)} seeds with model={effective_model}...")
    click.echo(f"Streaming output to {effective_output} (entries written as generated)")
    augmented = augment_seeds(
        seeds,
        model=effective_model,
        ollama_url=effective_url,
        variations_per_seed=effective_variations,
        output_path=effective_output,
        resume=resume,
        task=train_cfg.task,
    )
    counts = augmentation_counts(augmented)
    incomplete = sum(
        counts.get(seed["seed_id"], 0) < effective_variations
        for seed in prepare_seeds(seeds, train_cfg.task)
    )
    if incomplete:
        raise click.ClickException(
            f"{incomplete} seeds incomplete; partial output saved. Retry with --resume"
        )
    click.echo(f"Done: {len(augmented)} total utterances written to {effective_output}")


@router.command("evaluate")
@click.option("--model-path", required=True, help="Path to trained router model")
@click.option("--data", type=click.Path(exists=True), help="Path to evaluation JSONL data")
@click.pass_context
def router_evaluate(ctx: click.Context, model_path: str, data: str | None) -> None:
    """Evaluate the router model."""
    import json

    from sentimentizer.router.evaluate import evaluate_router
    from sentimentizer.router.model import RouterModel

    model = RouterModel.from_pretrained(model_path)
    if data:
        from sentimentizer.router.dataset import load_router_records

        records = load_router_records(data)
        eval_ds = {
            "text": [row["text"] for row in records],
            "label": [row["label"] for row in records],
        }
        if any("group_id" in row for row in records):
            eval_ds["group_id"] = [row.get("group_id") for row in records]
        report = evaluate_router(model, eval_ds)
        click.echo(
            json.dumps(
                {**report, "similarity_matrix": report["similarity_matrix"].tolist()}, indent=2
            )
        )
    else:
        click.echo("No evaluation data provided. Use --data to specify a JSONL file.")


@router.command("calibrate")
@click.option("--model-path", required=True, type=click.Path(exists=True))
@click.option("--calibration-data", required=True, type=click.Path(exists=True))
@click.option("--policy-data", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--min-accuracy", default=None, type=click.FloatRange(min=0, max=1, min_open=True))
@click.option("--min-accepted", default=None, type=click.IntRange(min=1))
def router_calibrate(
    model_path: str,
    calibration_data: str,
    policy_data: str,
    output_dir: str,
    min_accuracy: float | None,
    min_accepted: int | None,
) -> None:
    """Fit probability calibration and select acceptance on separate held-out files."""
    from pathlib import Path

    from sentimentizer.router.calibration import (
        DEFAULT_MIN_ACCEPTED,
        DEFAULT_MIN_ACCURACY,
        calibrate_router,
    )
    from sentimentizer.router.dataset import load_router_records
    from sentimentizer.router.model import RouterModel

    if Path(model_path).resolve() == Path(output_dir).resolve():
        raise click.UsageError("Use a different output-dir to preserve the uncalibrated artifact")
    model = calibrate_router(
        RouterModel.from_pretrained(model_path),
        load_router_records(calibration_data),
        load_router_records(policy_data),
        min_accuracy=min_accuracy if min_accuracy is not None else DEFAULT_MIN_ACCURACY,
        min_accepted=min_accepted if min_accepted is not None else DEFAULT_MIN_ACCEPTED,
    )
    model.save_pretrained(output_dir)
    if model.acceptance_policy is not None:
        click.echo(model.acceptance_policy.model_dump_json(indent=2))


@router.command("benchmark")
@click.option("--train-data", required=True, type=click.Path(exists=True))
@click.option("--validation-data", required=True, type=click.Path(exists=True))
@click.option("--test-data", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--config", "config_path", type=click.Path(exists=True))
@click.option("--repeats", default=None, type=click.IntRange(min=1))
def router_benchmark(
    train_data: str,
    validation_data: str,
    test_data: str,
    output_dir: str,
    config_path: str | None,
    repeats: int | None,
) -> None:
    """Compare TF-IDF, frozen encoder and contrastive encoder on the same split."""
    from dataclasses import replace
    from pathlib import Path

    from sentimentizer.router.benchmark import BENCHMARK_REPEATS, benchmark_router
    from sentimentizer.router.config import load_router_config
    from sentimentizer.router.dataset import load_router_records

    config, _ = load_router_config(config_path)
    benchmark_router(
        replace(config, output_dir=Path(output_dir)),
        load_router_records(train_data),
        load_router_records(validation_data),
        load_router_records(test_data),
        repeats=repeats if repeats is not None else BENCHMARK_REPEATS,
    )
    click.echo(f"Benchmark report: {Path(output_dir) / 'benchmark_report.json'}")


@router.command("push")
@click.option("--model-path", default="models/router", help="Path to trained router model")
@click.option("--repo-id", default="ryeyoo/sentimentizer-router", help="Hugging Face repository ID")
@click.pass_context
def router_push(ctx: click.Context, model_path: str, repo_id: str) -> None:
    """Push the trained router model to Hugging Face Hub."""
    from sentimentizer.router.model import RouterModel

    try:
        model = RouterModel.from_pretrained(model_path)
        click.echo(f"Pushing router model to {repo_id}...")
        model.push_to_hub(repo_id)
        click.echo(f"Successfully pushed router model to {repo_id}")
    except Exception as e:
        click.echo(f"Failed to push router model: {e}", err=True)


# ── serve ───────────────────────────────────────


@cli.command("serve")
@click.option("--model-path", default="models/router", help="Path to trained router model")
@click.option("--host", default=None, help="Override serve_host from service.yaml")
@click.option(
    "--port",
    default=None,
    type=click.IntRange(min=1, max=65535),
    help="Override serve_port from service.yaml",
)
@click.pass_context
def serve_cmd(
    ctx: click.Context,
    model_path: str,
    host: str | None,
    port: int | None,
) -> None:
    """Serve the unified Sentimentizer API via Ray Serve.

    Starts a REST API serving both sentiment analysis and review routing:

    Sentiment endpoints (v1):
        POST /v1/predict, POST /v1/batch, POST /v1/tokenize, GET /v1/models

    Router endpoints (v1):
        POST /v1/router/predict, POST /v1/router/batch, GET /v1/router/models

    Infrastructure endpoints (unversioned):
        GET /health, GET /health/live, GET /health/ready

    Requires the ray extra: pip install -e ".[ray]"
    """
    import os

    os.environ.setdefault("ROUTER_MODEL_PATH", model_path)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

    from sentimentizer.serve import main as serve_main

    serve_main(host=host, port=port)


# ── diagnose ─────────────────────────────────


@cli.group()
def diagnose() -> None:
    """Pipeline diagnostics."""


@diagnose.command("env")
@click.pass_context
def diagnose_env(ctx: click.Context) -> None:
    """Fast environment check (Python / CUDA paths / Ray env vars). No torch/ray."""
    from workflows.stages.diagnose import run_diagnose_env

    run_diagnose_env(ctx.obj)


@diagnose.command("pipeline")
@click.pass_context
def diagnose_pipeline(ctx: click.Context) -> None:
    """Heavy pipeline check. Imports the ML stack."""
    from workflows.stages.diagnose import run_diagnose_pipeline

    run_diagnose_pipeline(ctx.obj)


# ── run (full pipeline) ──────────────────────


@cli.command()
@click.option("--stop", default=10000, type=int, help="Number of review lines to load")
@click.option(
    "--resume-tokenize",
    is_flag=True,
    help="Update existing dictionary during tokenize stage",
)
@click.option(
    "--resume-train",
    is_flag=True,
    help="Resume training from checkpoint",
)
@shared_train_options
@click.pass_context
def run(
    ctx: click.Context,
    stop: int,
    resume_tokenize: bool,
    resume_train: bool,
    **train_kwargs: Any,
) -> None:
    """Run the full pipeline: extract → tokenize → train.

    Mirrors today's default ``python driver.py --type new`` invocation.
    The single legacy ``--resume`` flag is split into stage-specific flags
    so each stage's resume can be controlled independently.
    """
    from workflows.stages.extract import run_extract
    from workflows.stages.tokenize import run_tokenize
    from workflows.stages.train import run_train

    run_extract(ctx.obj, stop=stop)
    run_tokenize(ctx.obj, resume=resume_tokenize)
    run_train(ctx.obj, resume=resume_train, **train_kwargs)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cli()
