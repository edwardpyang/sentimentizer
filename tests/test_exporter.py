"""Tests for the standalone exporter's training-metrics file ingestion."""

import json
from unittest.mock import MagicMock, patch

import pytest

from sentimentizer import exporter
from sentimentizer.metrics_publisher import write_json_atomic


def _value(gauge, model_type: str, run_id: str) -> float:
    return gauge.labels(model_type=model_type, run_id=run_id)._value.get()


def _series(gauge) -> set[tuple[str, ...]]:
    return set(gauge._metrics)


@pytest.fixture
def metrics_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "_METRICS_DIR", tmp_path)
    return tmp_path


def test_epoch_file_populates_gauges(metrics_dir):
    (metrics_dir / "rnn_metrics.json").write_text(
        json.dumps({"run_id": "r1", "val_loss": 0.4, "accuracy": 0.8, "epoch": 3, "lr": 0.01})
    )
    exporter._update_training_metrics()

    assert _value(exporter.TRAINING_VAL_LOSS, "rnn", "r1") == 0.4
    assert _value(exporter.TRAINING_VAL_ACCURACY, "rnn", "r1") == 0.8
    assert _value(exporter.TRAINING_EPOCH, "rnn", "r1") == 3


def test_missing_files_produce_zero_baseline_series(metrics_dir):
    exporter._update_training_metrics()
    assert _value(exporter.TRAINING_VAL_LOSS, "decoder", "") == 0


def test_corrupt_file_does_not_blank_other_models(metrics_dir):
    """A bad file used to abort the pass after every gauge had been cleared."""
    (metrics_dir / "rnn_metrics.json").write_text('{"run_id": "partial')
    (metrics_dir / "encoder_metrics.json").write_text(json.dumps({"run_id": "r2", "accuracy": 0.7}))

    exporter._update_training_metrics()

    assert _value(exporter.TRAINING_VAL_ACCURACY, "encoder", "r2") == 0.7
    assert not any(labels[0] == "rnn" for labels in _series(exporter.TRAINING_VAL_ACCURACY))


def test_reset_file_skips_batch_gauges(metrics_dir):
    (metrics_dir / "rnn_metrics.json").write_text(json.dumps({"run_id": "r3", "_reset": True}))
    (metrics_dir / "rnn_batch.json").write_text(json.dumps({"run_id": "r3", "batch": 9}))

    exporter._update_training_metrics()

    assert ("rnn", "r3") not in _series(exporter.TRAINING_BATCH)


def test_batch_snapshot_populates_intra_epoch_gauges(metrics_dir):
    (metrics_dir / "rnn_metrics.json").write_text(json.dumps({"run_id": "r4", "epoch": 1}))
    (metrics_dir / "rnn_batch.json").write_text(
        json.dumps({"run_id": "r4", "batch": 9, "loss_ema": 0.5, "epoch": 2, "lr": 0.1})
    )

    exporter._update_training_metrics()

    assert _value(exporter.TRAINING_BATCH, "rnn", "r4") == 9
    assert _value(exporter.TRAINING_TRAIN_LOSS_EMA, "rnn", "r4") == 0.5
    assert _value(exporter.TRAINING_EPOCH, "rnn", "r4") == 2


def test_corrupt_batch_snapshot_keeps_epoch_gauges(metrics_dir):
    (metrics_dir / "rnn_metrics.json").write_text(json.dumps({"run_id": "r5", "accuracy": 0.6}))
    (metrics_dir / "rnn_batch.json").write_text("{bad")

    exporter._update_training_metrics()

    assert _value(exporter.TRAINING_VAL_ACCURACY, "rnn", "r5") == 0.6
    assert ("rnn", "r5") not in _series(exporter.TRAINING_BATCH)


class TestWriteJsonAtomic:
    def test_replaces_content_and_leaves_no_temporary_files(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text("old")

        write_json_atomic(path, {"a": 1})

        assert json.loads(path.read_text()) == {"a": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["metrics.json"]
        assert path.stat().st_mode & 0o777 == 0o644

    def test_failed_serialization_keeps_previous_file(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text('{"old": true}')

        with pytest.raises(TypeError):
            write_json_atomic(path, {"bad": object()})

        assert json.loads(path.read_text()) == {"old": True}
        assert [p.name for p in tmp_path.iterdir()] == ["metrics.json"]


class TestRayScrape:
    def test_unreachable_ray_endpoint_does_not_raise(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            exporter._update_ray_metrics("http://localhost:1")

    def test_collector_loop_survives_failures(self):
        stop = RuntimeError("stop loop")
        with (
            patch.object(exporter, "_update_system_metrics", side_effect=ValueError("sys")),
            patch.object(exporter, "_update_gpu_metrics", side_effect=ValueError("gpu")),
            patch.object(exporter, "_update_ray_metrics", side_effect=ValueError("ray")),
            patch.object(exporter, "_update_training_metrics", MagicMock()) as training,
            patch.object(exporter.time, "sleep", side_effect=stop),
            pytest.raises(RuntimeError, match="stop loop"),
        ):
            exporter._metrics_loop(1, "http://localhost:1")
        training.assert_called_once()
