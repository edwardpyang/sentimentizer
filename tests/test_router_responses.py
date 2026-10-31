"""Contract tests for reusable router artifacts and validated classifications."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from sentimentizer.router.config import FOOD_TASK_SPEC, RouterConfig, load_router_config
from sentimentizer.router.model import RouterModel
from sentimentizer.router.schemas import RouterClassification, RouterTaskSpec


def make_router(labels=("first", "second"), classes=(0, 1), probabilities=None):
    backbone = MagicMock()
    backbone.encode.return_value = np.zeros((1, 4))
    head = MagicMock()
    head.classes_ = np.array(classes)
    head.predict_proba.return_value = np.array(
        probabilities if probabilities is not None else [[0.2, 0.8]]
    )
    task = RouterTaskSpec(task_id="test_task", labels=labels)
    return RouterModel(backbone, head, task=task)


@pytest.mark.parametrize(
    "overrides",
    [
        {"labels": ["same", "same"]},
        {"labels": ["one"]},
        {"labels": [" ", "two"]},
        {"fallback_label": "missing"},
        {"label_descriptions": {"missing": "Unknown"}},
        {"task_id": ""},
        {"unknown": "value"},
    ],
)
def test_task_rejects_invalid_vocabularies(overrides):
    with pytest.raises(ValidationError):
        RouterTaskSpec.model_validate({"task_id": "test", "labels": ["a", "b"], **overrides})


@pytest.mark.parametrize("value", [True, False, 0.5, -1, 2, "0", np.bool_(True)])
def test_task_rejects_invalid_class_ids(value):
    with pytest.raises(ValueError):
        RouterTaskSpec(task_id="test", labels=("a", "b")).label_id(value)


def test_food_config_defaults_are_independent_and_legacy_yaml_still_loads(tmp_path):
    first, second = RouterConfig(), RouterConfig()
    first.task.label_descriptions["dietary"] = "Changed"
    assert second.task == FOOD_TASK_SPEC
    path = tmp_path / "config.yaml"
    path.write_text("training:\n  num_epochs: 3\n")
    config, _ = load_router_config(path)
    assert config.task == FOOD_TASK_SPEC
    assert config.num_epochs == 3


def test_external_task_config_and_seed_env_override(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        "training:\n  task:\n    task_id: test\n    labels: [a, b]\n"
        "augmentation:\n  seed_path: seeds.jsonl\n"
    )
    monkeypatch.setenv("SENTIMENTIZER_AUGMENT_SEED_PATH", "override.jsonl")
    config, augmentation = load_router_config(path)
    assert config.task.labels == ("a", "b")
    assert augmentation.seed_path == "override.jsonl"


def test_schemas_import_without_model_or_serving_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sentimentizer.router.schemas import RouterTaskSpec; "
            "import sys; assert not {'sentence_transformers', 'ray', "
            "'sentimentizer.serve.config'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("classes, expected", [((1, 0), "first"), (("second", "first"), "first")])
def test_probability_columns_follow_head_classes_and_encode_once(classes, expected):
    model = make_router(classes=classes)
    predictions = model.classify_batch(["example"])
    assert predictions == [RouterClassification(label=expected, score=0.8)]
    assert predictions[0].model_dump(mode="json") == {"label": expected, "score": 0.8}
    model.backbone.encode.assert_called_once()
    model.head.predict.assert_not_called()


def test_five_class_model_and_empty_batch():
    model = make_router(
        labels=("a", "b", "c", "d", "e"),
        classes=(4, 3, 2, 1, 0),
        probabilities=[[0.1, 0.1, 0.1, 0.6, 0.1]],
    )
    assert model.classify_batch([]) == []
    model.backbone.encode.assert_not_called()
    assert model.classify_batch(["example"])[0].label == "b"


@pytest.mark.parametrize("classes", [(0, 0), (0, 2), (True, False), (0.0, 1.0), ("a", "b"), (0,)])
def test_invalid_class_mappings_fail_before_encoding(classes):
    model = make_router(classes=classes)
    with pytest.raises(ValueError):
        model.classify_batch(["example"])
    model.backbone.encode.assert_not_called()


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [[0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
        [[float("nan"), 0.5]],
        [[float("inf"), 0]],
        [[-0.1, 0.9]],
        [[1.1, 0]],
        [0.5, 0.5],
    ],
)
def test_invalid_probability_batches_fail(probabilities):
    model = make_router(probabilities=probabilities)
    with pytest.raises(ValueError):
        model.classify_batch(["example"])


@pytest.mark.parametrize("probabilities", [[[True, False]], [["0.2", "0.8"]]])
def test_non_numeric_probability_types_are_rejected(probabilities):
    with pytest.raises(ValueError, match="numeric"):
        make_router(probabilities=probabilities).classify_batch(["example"])


def test_legacy_string_head_needs_no_guessed_food_labels():
    model = make_router(classes=("first", "second"))
    legacy = RouterModel(model.backbone, model.head)
    assert legacy.classify_batch(["example"])[0].label == "second"
    model.head.classes_ = np.array([0, 1])
    with pytest.raises(ValueError):
        legacy.classify_batch(["example"])


def test_explicit_conflicting_model_labels_fail():
    with pytest.raises(ValueError, match="conflict"):
        RouterModel(MagicMock(), labels=["x", "y"], task=FOOD_TASK_SPEC)


def test_task_artifact_round_trip_and_atomic_publication(tmp_path, monkeypatch):
    model = make_router()
    joblib = MagicMock()
    joblib.dump.side_effect = lambda head, path: Path(path).touch()
    joblib.load.return_value = model.head
    monkeypatch.setattr("sentimentizer.router.model._get_joblib", lambda: joblib)
    monkeypatch.setattr(
        "sentimentizer.router.model._get_sentence_transformers", lambda: lambda path: model.backbone
    )
    sidecar = tmp_path / "router_config.json"
    sidecar.write_text(json.dumps({"labels": ["old", "labels"]}))
    original_replace = Path.replace

    def observe_publication(source, target):
        assert RouterModel.read_metadata(tmp_path).labels == ["old", "labels"]
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_publication)
    model.save_pretrained(tmp_path)
    loaded = RouterModel.from_pretrained(tmp_path)
    assert loaded.task == model.task
    assert loaded.classify_batch(["example"]) == model.classify_batch(["example"])
    assert {p.name for p in tmp_path.iterdir()} == {"router_head.joblib", "router_config.json"}


def test_legacy_metadata_and_conflicting_sidecar(tmp_path):
    (tmp_path / "config_setfit.json").write_text(json.dumps({"labels": {"0": "a", "1": "b"}}))
    assert RouterModel.read_metadata(tmp_path).labels == ["a", "b"]
    (tmp_path / "router_config.json").write_text(
        json.dumps(
            {
                "labels": ["wrong", "labels"],
                "task": RouterTaskSpec(task_id="test", labels=("a", "b")).model_dump(mode="json"),
            }
        )
    )
    with pytest.raises(ValidationError):
        RouterModel.read_metadata(tmp_path)


def make_predictor(path, model=None):
    from sentimentizer.predictor import SentimentPredictor

    predictor = SentimentPredictor.__new__(SentimentPredictor)
    predictor._initialize_router_state()
    predictor._router_model_path = str(path)
    predictor.router = model
    predictor._router_loaded = model is not None
    predictor._router_error = None
    return predictor


def test_predictor_reuses_model_labels_and_memoizes_loading(tmp_path):
    model = make_router()
    predictor = make_predictor(tmp_path)
    predictor._load_router_model = MagicMock(return_value=(model, None))
    assert predictor.classify("test text") == {
        "prediction": {"label": "second", "score": 0.8, "token_count": 2},
    }
    predictor.classify("another text")
    predictor._load_router_model.assert_called_once()
    assert predictor.get_router_model_info()["categories"] == ["first", "second"]


def test_predictor_metadata_reads_task_without_loading(tmp_path):
    model = make_router()
    (tmp_path / "router_config.json").write_text(
        json.dumps(
            {
                "labels": model.labels,
                "task": model.task.model_dump(mode="json"),
            }
        )
    )
    predictor = make_predictor(tmp_path)
    predictor._load_router_model = MagicMock()
    info = predictor.get_router_model_info()
    assert (info["categories"], info["task_id"], info["status"]) == (
        ["first", "second"],
        "test_task",
        "not_loaded",
    )
    predictor._load_router_model.assert_not_called()


def test_custom_missing_model_does_not_download_food_and_failure_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("sentimentizer.predictor._ROUTER_AVAILABLE", True)
    download = MagicMock()
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    predictor = make_predictor(tmp_path / "missing")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="not loaded"):
            predictor.classify("example")
    download.assert_not_called()
    assert predictor.get_router_model_info()["status"] == "not_loaded"
    assert predictor.router_error is not None


@pytest.mark.parametrize("metadata", ["[]", "invalid", '{"labels": ["a", "a"]}'])
def test_invalid_metadata_is_reported_without_loading(tmp_path, metadata):
    (tmp_path / "router_config.json").write_text(metadata)
    predictor = make_predictor(tmp_path)
    assert "metadata" in predictor.get_router_model_info()["error"]


def test_failed_metadata_publication_keeps_previous_file_and_removes_temp(tmp_path, monkeypatch):
    model = make_router()
    monkeypatch.setattr("sentimentizer.router.model._get_joblib", lambda: MagicMock())
    sidecar = tmp_path / "router_config.json"
    original = '{"labels": ["old", "labels"]}'
    sidecar.write_text(original)

    def fail_publication(source, target):
        raise OSError("publication failed")

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError):
        model.save_pretrained(tmp_path)
    assert sidecar.read_text() == original
    assert list(tmp_path.iterdir()) == [sidecar]


def test_default_food_legacy_mapping_supports_reordered_classes(tmp_path, monkeypatch):
    model = make_router(
        labels=FOOD_TASK_SPEC.labels, classes=(1, 2, 0), probabilities=[[0.2, 0.3, 0.5]]
    )
    legacy = RouterModel(model.backbone, model.head)
    monkeypatch.setattr("sentimentizer.predictor._ROUTER_AVAILABLE", True)
    monkeypatch.setattr("sentimentizer.predictor.DEFAULT_ROUTER_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(RouterModel, "from_pretrained", lambda path: legacy)
    predictor = make_predictor(tmp_path)
    assert predictor.classify("example")["prediction"]["label"] == "dietary"
    assert predictor.router.task == FOOD_TASK_SPEC


def test_unlabeled_numeric_custom_artifact_is_not_treated_as_food(tmp_path, monkeypatch):
    model = make_router()
    legacy = RouterModel(model.backbone, model.head)
    monkeypatch.setattr("sentimentizer.predictor._ROUTER_AVAILABLE", True)
    monkeypatch.setattr(RouterModel, "from_pretrained", lambda path: legacy)
    predictor = make_predictor(tmp_path)
    with pytest.raises(RuntimeError, match="vocabulary"):
        predictor.classify("example")
    assert not predictor.router_loaded


def test_invalid_sidecar_fails_before_optional_model_import(tmp_path, monkeypatch):
    (tmp_path / "router_config.json").write_text("[]")
    loader = MagicMock()
    monkeypatch.setattr("sentimentizer.router.model._get_sentence_transformers", loader)
    with pytest.raises(ValueError, match="metadata"):
        RouterModel.from_pretrained(tmp_path)
    loader.assert_not_called()


def test_native_sklearn_head_round_trip_keeps_probabilities(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    embeddings = np.array([[1, 0], [0, 1], [2, 0], [0, 2]])
    head = LogisticRegression().fit(embeddings, [0, 1, 0, 1])
    backbone = MagicMock()
    backbone.encode.return_value = embeddings[:2]
    task = RouterTaskSpec(task_id="test", labels=("a", "b"))
    original = RouterModel(backbone, head, task=task)
    expected = original.classify_batch(["first", "second"])
    original.save_pretrained(tmp_path)
    monkeypatch.setattr(
        "sentimentizer.router.model._get_sentence_transformers", lambda: lambda path: backbone
    )
    restored = RouterModel.from_pretrained(tmp_path)
    assert restored.head is not head
    assert restored.task == task
    assert restored.classify_batch(["first", "second"]) == expected
