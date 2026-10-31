"""Exercise task-aware training, augmentation, evaluation, and CLI wiring."""

import importlib
import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from sentimentizer.router.augment import _build_prompt, augment_seeds
from sentimentizer.router.config import RouterConfig
from sentimentizer.router.dataset import load_router_records, save_dataset_to_jsonl
from sentimentizer.router.model import RouterModel
from sentimentizer.router.schemas import RouterTaskSpec


@pytest.mark.parametrize("size", [2, 5])
def test_augmentation_uses_every_configured_category(size):
    task = RouterTaskSpec(
        task_id="test",
        description="Route test messages",
        labels=tuple(f"class_{i}" for i in range(size)),
    )
    prompt = _build_prompt("An example", 0, 5, task=task)
    assert task.description in prompt
    assert all(label in prompt for label in task.labels)
    assert "Yelp" not in prompt


def test_augmentation_rejects_wrong_labels_and_preserves_parent_metadata(monkeypatch):
    task = RouterTaskSpec(task_id="test", labels=("a", "b"))
    response = MagicMock()
    response.json.return_value = {
        "response": "\n".join(
            [
                json.dumps({"text": "wrong", "label": 1}),
                json.dumps({"text": "fractional", "label": 0.2}),
                json.dumps({"text": "valid", "label": 0, "group_id": "wrong-group"}),
            ]
        )
    }
    monkeypatch.setattr("sentimentizer.router.augment.requests.post", lambda *a, **k: response)
    monkeypatch.setattr("sentimentizer.router.augment.time.sleep", lambda _: None)
    seed = {"text": "seed", "label": 0, "group_id": "parent"}
    records = augment_seeds([seed], task=task)
    assert [row["text"] for row in records] == ["seed", "valid"]
    assert all(row["label"] == 0 and row["group_id"] == "parent" for row in records)
    assert records[0]["seed_id"] == records[1]["seed_id"]
    assert records[1]["is_seed"] is False


def test_jsonl_preserves_grouping_and_extra_metadata(tmp_path):
    rows = [{"text": "example", "label": 0, "group_id": "parent", "source": {"id": "s1"}}]
    path = save_dataset_to_jsonl(rows, str(tmp_path / "data.jsonl"))
    assert load_router_records(path) == rows


@pytest.mark.parametrize("row", ['{"text": "" , "label": 0}', '{"text": "a"}', "[]", "invalid"])
def test_invalid_jsonl_reports_line_without_exposing_contents(tmp_path, row):
    path = tmp_path / "data.jsonl"
    path.write_text(row)
    with pytest.raises(ValueError, match="line 1"):
        load_router_records(path)


def test_evaluation_handles_missing_test_classes_and_no_general_label():
    from sentimentizer.router.evaluate import calibrate_threshold, evaluate_router

    task = RouterTaskSpec(task_id="test", labels=("a", "b", "c", "d", "e"))
    backbone = MagicMock()
    backbone.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 2))
    head = MagicMock()
    head.classes_ = np.array([4, 3, 2, 1, 0])
    head.predict_proba.return_value = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
    model = RouterModel(backbone, head, task=task)
    dataset = {"text": ["first", "second"], "label": [0, 1]}
    report = evaluate_router(model, dataset)
    assert report["classification_report"]["a"]["recall"] == 1
    assert report["classification_report"]["e"]["support"] == 0
    assert report["tau_threshold"] is None
    assert report["similarity_labels"] == ["a", "b"]
    assert report["confidence_statistics"]["mean"] == 1
    assert calibrate_threshold(model, dataset) is None
    assert head.predict_proba.call_count == 1


def test_training_uses_task_and_all_loaded_settings(monkeypatch, tmp_path):
    pytest.importorskip("sentence_transformers")
    module = importlib.import_module("sentimentizer.router.train_router")
    backbone = MagicMock()
    backbone.encode.return_value = np.array([[1, 0], [0, 1], [2, 0], [0, 2]])
    monkeypatch.setattr(module, "SentenceTransformer", lambda _: backbone)
    monkeypatch.setattr(module, "CosineSimilarityLoss", lambda model: None)
    save = MagicMock()
    monkeypatch.setattr(RouterModel, "save_pretrained", save)
    task = RouterTaskSpec(task_id="test", labels=("a", "b"))
    config = RouterConfig(
        task=task,
        max_seq_length=64,
        num_epochs=3,
        num_iterations=2,
        batch_size=4,
        output_dir=tmp_path,
    )
    model = module.train_router(config, {"text": ["a", "b", "aa", "bb"], "label": [0, 1, 0, 1]})
    assert model.task == task
    assert model.labels == ["a", "b"]
    assert backbone.max_seq_length == 64
    assert backbone.fit.call_args.kwargs["epochs"] == 3
    assert backbone.fit.call_args.kwargs["train_objectives"][0][0].batch_size == 4
    save.assert_called_once()


@pytest.mark.parametrize("labels", [[0, 0], [0, True], [0, 1.0], [0, -1], [0, 2]])
def test_invalid_training_labels_fail_before_backbone_load(monkeypatch, labels):
    pytest.importorskip("sentence_transformers")
    module = importlib.import_module("sentimentizer.router.train_router")
    loader = MagicMock()
    monkeypatch.setattr(module, "SentenceTransformer", loader)
    with pytest.raises(ValueError):
        module.train_router(
            RouterConfig(task=RouterTaskSpec(task_id="test", labels=("a", "b"))),
            {"text": ["first", "second"], "label": labels},
        )
    loader.assert_not_called()


def write_config(path, *, seeds=False):
    path.write_text(
        "training:\n  num_epochs: 3\n  seed: 17\n  max_seq_length: 64\n"
        "  task:\n    task_id: test\n    labels: [a, b]\n"
        + ("augmentation:\n  seed_path: seeds.jsonl\n" if seeds else "")
    )


def test_train_cli_preserves_loaded_config(monkeypatch, tmp_path):
    pytest.importorskip("sentence_transformers")
    from workflows.cli import cli

    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    data = tmp_path / "data.jsonl"
    data.write_text("{}")
    loader = MagicMock(return_value=("train", "eval"))
    trainer = MagicMock()
    monkeypatch.setattr("sentimentizer.router.dataset.load_router_dataset", loader)
    monkeypatch.setattr("sentimentizer.router.train_router.train_router", trainer)
    result = CliRunner().invoke(
        cli,
        [
            "router",
            "train",
            "--config",
            str(config_path),
            "--data",
            str(data),
            "--base-model",
            "override",
        ],
    )
    assert result.exit_code == 0, result.output
    config = trainer.call_args.args[0]
    assert (config.task.task_id, config.num_epochs, config.seed, config.max_seq_length) == (
        "test",
        3,
        17,
        64,
    )
    assert config.base_model == "override"
    loader.assert_called_once_with(str(data), seed=17)


def test_custom_task_augmentation_requires_seeds_and_resolves_relative_paths(monkeypatch, tmp_path):
    from workflows.cli import cli

    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    from sentimentizer.router.augment import prepare_seeds

    seed = {"text": "example", "label": 0}
    prepared = prepare_seeds([seed], RouterTaskSpec(task_id="test", labels=("a", "b")))[0]
    augment = MagicMock(return_value=[prepared] + [{**prepared, "is_seed": False}] * 50)
    monkeypatch.setattr("sentimentizer.router.augment.augment_seeds", augment)
    args = ["router", "augment", "--config", str(config_path)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert "seed_path" in result.output
    augment.assert_not_called()
    write_config(config_path, seeds=True)
    seeds = [{"text": "example", "label": 0}]
    (tmp_path / "seeds.jsonl").write_text(json.dumps(seeds[0]) + "\n")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert augment.call_args.args[0] == seeds
    assert augment.call_args.kwargs["task"].task_id == "test"


def test_evaluation_cli_uses_all_explicit_evaluation_rows(monkeypatch, tmp_path):
    from workflows.cli import cli

    data = tmp_path / "eval.jsonl"
    rows = [{"text": "a", "label": 0}, {"text": "b", "label": 1}]
    save_dataset_to_jsonl(rows, str(data))
    evaluate = MagicMock(return_value={"similarity_matrix": np.eye(2), "tau_threshold": None})
    monkeypatch.setattr(RouterModel, "from_pretrained", lambda _: "model")
    monkeypatch.setattr("sentimentizer.router.evaluate.evaluate_router", evaluate)
    result = CliRunner().invoke(
        cli, ["router", "evaluate", "--model-path", "artifact", "--data", str(data)]
    )
    assert result.exit_code == 0, result.output
    evaluate.assert_called_once_with("model", {"text": ["a", "b"], "label": [0, 1]})
    assert json.loads(result.output)["similarity_matrix"] == [[1.0, 0.0], [0.0, 1.0]]
