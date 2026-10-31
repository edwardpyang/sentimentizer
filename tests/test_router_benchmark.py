"""Exercise comparison tooling with real sklearn heads and a tiny local encoder."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from sentimentizer.router.benchmark import benchmark_router, measure_inference
from sentimentizer.router.config import RouterConfig
from sentimentizer.router.model import RouterModel
from sentimentizer.router.provenance import add_dataset, training_provenance
from sentimentizer.router.schemas import RouterTaskSpec


def test_measurement_reports_samples_and_nonnegative_latency():
    result = measure_inference(lambda texts: [0] * len(texts), ["first", "second"], repeats=2)
    assert result["single_request_samples"] == 4
    assert result["single_request_p95_s"] >= 0
    assert result["batch_texts_per_second"] > 0
    with pytest.raises(ValueError):
        measure_inference(lambda _: [], [], repeats=0)


def test_benchmark_uses_identical_splits_and_saves_all_comparisons(tmp_path, monkeypatch):
    class Backbone:
        device = "cpu"

        def encode(self, texts, **kwargs):
            return np.array([[1.0 if text.startswith("positive") else -1.0, 1.0] for text in texts])

    def rows(name):
        return [
            {
                "text": f"{'positive' if i % 2 else 'negative'} {name} sample{i}",
                "label": i % 2,
                "group_id": f"{name}-{i}",
            }
            for i in range(8)
        ]

    calls = []

    def train(config, train, validation):
        calls.append((config.training_mode, train, validation))
        backbone = Backbone()
        head = LogisticRegression().fit(backbone.encode(train["text"]), train["label"])
        provenance = training_provenance(config, rows("train"), backbone)
        add_dataset(provenance, "validation", rows("validation"))
        return RouterModel(backbone, head, task=config.task, provenance=provenance)

    monkeypatch.setattr("sentimentizer.router.train_router.train_router", train)
    config = RouterConfig(
        task=RouterTaskSpec(task_id="test", labels=("negative", "positive")), output_dir=tmp_path
    )
    report = benchmark_router(config, rows("train"), rows("validation"), rows("test"), repeats=1)
    assert set(report["runs"]) == {"tfidf", "frozen", "contrastive"}
    assert [call[0] for call in calls] == ["frozen", "contrastive"]
    assert calls[0][1:] == calls[1][1:]
    for result in report["runs"].values():
        assert result["accuracy"] == 1
        assert result["process_rss_after_bytes"] > 0
        assert result["latency"]["repeats"] == 1
    assert (tmp_path / "tfidf.joblib").exists()
    assert (
        json.loads((tmp_path / "benchmark_report.json").read_text())["datasets"]
        == report["datasets"]
    )
    with pytest.raises(ValueError, match="overlap"):
        benchmark_router(config, rows("train"), rows("validation"), rows("train"))
