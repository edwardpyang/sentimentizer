"""Compare lexical, frozen-embedding and contrastive routers on identical inputs."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from sentimentizer.router.config import RouterConfig
from sentimentizer.router.dataset import assert_disjoint, dataset_fingerprint
from sentimentizer.router.evaluate import evaluate_router, prediction_metrics
from sentimentizer.router.provenance import library_versions

BENCHMARK_REPEATS = 5
MAX_LATENCY_TEXTS = 20
TFIDF_NGRAM_RANGE = (1, 2)


def measure_inference(
    predict: Callable[[list[str]], Any], texts: list[str], repeats: int = BENCHMARK_REPEATS
) -> dict:
    """Measure warmed single-request p95 and batch throughput in this process."""
    if not texts or repeats < 1:
        raise ValueError("Benchmark requires texts and positive repeats")
    predict(texts[:1])
    singles = []
    batches = []
    for _ in range(repeats):
        for text in texts[:MAX_LATENCY_TEXTS]:
            start = time.perf_counter()
            predict([text])
            singles.append(time.perf_counter() - start)
        start = time.perf_counter()
        predict(texts)
        batches.append(time.perf_counter() - start)
    return {
        "single_request_p95_s": float(np.percentile(singles, 95)),
        "single_request_samples": len(singles),
        "batch_size": len(texts),
        "batch_median_s": float(np.median(batches)),
        "batch_texts_per_second": len(texts) / max(float(np.median(batches)), np.finfo(float).eps),
        "repeats": repeats,
    }


def benchmark_router(
    config: RouterConfig,
    train_records: list[dict],
    validation_records: list[dict],
    test_records: list[dict],
    *,
    repeats: int = BENCHMARK_REPEATS,
) -> dict:
    """Train three baselines, persist artifacts, and report held-out quality/cost.

    RSS values describe the whole process, including shared libraries and caches;
    they are not isolated model memory or GPU-memory measurements.
    """
    import joblib
    import psutil
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from sentimentizer.router.train_router import train_router

    if not train_records or not validation_records or not test_records or repeats < 1:
        raise ValueError("Benchmark needs nonempty train/validation/test data and positive repeats")
    assert_disjoint(train_records, validation_records, test_records)
    for rows in (train_records, validation_records, test_records):
        for row in rows:
            config.task.label_id(row["label"])
    if {row["label"] for row in train_records} != set(range(len(config.task.labels))):
        raise ValueError("Benchmark training data must contain every configured class")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    texts = [row["text"] for row in test_records]
    targets = [config.task.labels[row["label"]] for row in test_records]
    report: dict[str, Any] = {
        "datasets": {
            name: dataset_fingerprint(rows)
            for name, rows in (
                ("train", train_records),
                ("validation", validation_records),
                ("test", test_records),
            )
        },
        "library_versions": library_versions(),
        "memory_scope": (
            "Whole-process RSS before/after training; sequential runs share libraries/caches. "
            "Not isolated model or GPU memory."
        ),
        "runs": {},
    }
    before = process.memory_info().rss
    started = time.monotonic()
    lexical = make_pipeline(
        TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE),
        LogisticRegression(max_iter=config.head_max_iter, random_state=config.seed),
    )
    lexical.fit([row["text"] for row in train_records], [row["label"] for row in train_records])
    duration = time.monotonic() - started
    probabilities = lexical.predict_proba(texts)
    winners = probabilities.argmax(axis=1)
    labels = [config.task.labels[int(lexical.classes_[i])] for i in winners]
    artifact = config.output_dir / "tfidf.joblib"
    joblib.dump(lexical, artifact)
    (config.output_dir / "tfidf_task.json").write_text(config.task.model_dump_json(indent=2))
    report["runs"]["tfidf"] = {
        "training_config": {
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "head_max_iter": config.head_max_iter,
            "seed": config.seed,
        },
        **prediction_metrics(
            targets, labels, probabilities.max(axis=1).tolist(), list(config.task.labels)
        ),
        "latency": measure_inference(lexical.predict_proba, texts, repeats),
        "training_duration_s": duration,
        "process_rss_before_bytes": before,
        "process_rss_after_bytes": process.memory_info().rss,
        "artifact": str(artifact),
    }
    del lexical
    gc.collect()
    for mode in ("frozen", "contrastive"):
        before = process.memory_info().rss
        started = time.monotonic()
        run_config = replace(config, training_mode=mode, output_dir=config.output_dir / mode)
        model = train_router(run_config, _columns(train_records), _columns(validation_records))
        duration = time.monotonic() - started
        metrics = evaluate_router(model, _columns(test_records))
        metrics["similarity_matrix"] = metrics["similarity_matrix"].tolist()
        report["runs"][mode] = {
            **metrics,
            "latency": measure_inference(model.probabilities_batch, texts, repeats),
            "training_duration_s": duration,
            "process_rss_before_bytes": before,
            "process_rss_after_bytes": process.memory_info().rss,
            "artifact": str(run_config.output_dir),
            "device": str(model.backbone.device),
            "provenance": model.provenance.model_dump(mode="json") if model.provenance else None,
        }
        del model
        gc.collect()
    (config.output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    return report


def _columns(records: list[dict]) -> dict[str, list]:
    """Keep optional source metadata aligned when converting records to columns."""
    return {
        key: [row.get(key) for row in records]
        for key in set().union(*(row.keys() for row in records))
    }
