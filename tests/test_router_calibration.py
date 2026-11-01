"""Validate held-out calibration, abstention, provenance and sklearn persistence."""

import json
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner
from sklearn.linear_model import LogisticRegression

from sentimentizer.router.calibration import calibrate_router, select_acceptance_policy
from sentimentizer.router.config import RouterConfig
from sentimentizer.router.evaluate import evaluate_router, prediction_metrics
from sentimentizer.router.model import RouterModel
from sentimentizer.router.provenance import add_dataset, training_provenance
from sentimentizer.router.schemas import RouterAcceptancePolicy, RouterProvenance, RouterTaskSpec


class TinyBackbone:
    device = "cpu"

    def encode(self, texts, **kwargs):
        return np.array([[float(text.split()[0]), 1.0] for text in texts])

    def save(self, path):
        pass


def records(partition, count=8):
    return [
        {
            "text": f"{(-1 if i % 2 == 0 else 1) * (i + 1)} {partition} {i}",
            "label": i % 2,
            "group_id": f"{partition}-{i}",
        }
        for i in range(count)
    ]


def tiny_router():
    backbone = TinyBackbone()
    rows = records("train")
    head = LogisticRegression().fit(
        backbone.encode([row["text"] for row in rows]), [row["label"] for row in rows]
    )
    provenance = RouterProvenance()
    add_dataset(provenance, "train", rows)
    return RouterModel(
        backbone,
        head,
        task=RouterTaskSpec(task_id="test", labels=("negative", "positive")),
        provenance=provenance,
    )


def test_threshold_maximizes_coverage_and_respects_ties():
    policy = select_acceptance_policy(
        [0.6, 0.8, 0.8, 0.9], [False, True, False, True], min_accuracy=1.0, min_accepted=1
    )
    assert policy.threshold == 0.9
    assert policy.accepted == 1
    assert policy.accuracy == 1


def test_insufficient_evidence_produces_abstain_all():
    policy = select_acceptance_policy([0.99], [True], min_accepted=2)
    assert policy.threshold is None
    model = tiny_router()
    model.acceptance_policy = policy
    assert model.decide_batch(["1 new"])[0].decision == "abstain"


def test_decisions_preserve_topic_and_classification_contract():
    model = tiny_router()
    with pytest.raises(RuntimeError, match="calibrate"):
        model.decide_batch(["1 new"])
    model.acceptance_policy = RouterAcceptancePolicy(
        threshold=0.9, min_accepted=1, examples=1, accepted=1, accuracy=1
    )
    model.head = MagicMock(classes_=np.array([1, 0]))
    model.head.predict_proba.return_value = np.array([[0.6, 0.4], [0.95, 0.05]])
    decisions = model.decide_batch(["1 new", "2 newer"])
    assert [row.decision for row in decisions] == ["abstain", "accept"]
    assert [row.label for row in decisions] == ["positive", "positive"]
    assert set(model.classify_batch(["1 new", "2 newer"])[0].model_dump()) == {"label", "score"}


def test_real_calibration_keeps_base_head_frozen_and_round_trips(tmp_path, monkeypatch):
    original = tiny_router()
    coefficients = original.head.coef_.copy()
    calibrated = calibrate_router(
        original, records("calibration"), records("policy"), min_accuracy=1.0, min_accepted=2
    )
    np.testing.assert_array_equal(original.head.coef_, coefficients)
    assert original.acceptance_policy is None
    assert calibrated.provenance.calibration_method == "sigmoid"
    assert set(calibrated.provenance.dataset_fingerprints) == {"train", "calibration", "policy"}
    assert calibrated.acceptance_policy.accepted == 8
    calibrated.save_pretrained(tmp_path)
    monkeypatch.setattr(
        "sentimentizer.router.model._get_sentence_transformers", lambda: lambda _: TinyBackbone()
    )
    restored = RouterModel.from_pretrained(tmp_path)
    assert restored.acceptance_policy == calibrated.acceptance_policy
    assert restored.provenance == calibrated.provenance
    np.testing.assert_allclose(
        restored.predict_proba(["1 independent", "-2 independent"]),
        calibrated.predict_proba(["1 independent", "-2 independent"]),
    )


def test_calibration_rejects_training_overlap_and_missing_classes():
    model = tiny_router()
    with pytest.raises(ValueError, match="overlap"):
        calibrate_router(model, records("train"), records("policy"))
    with pytest.raises(ValueError, match="overlap"):
        calibrate_router(model, records("calibration"), records("calibration"))
    with pytest.raises(ValueError, match="every configured class"):
        calibrate_router(model, records("calibration")[::2], records("policy"))
    model.provenance = None
    with pytest.raises(ValueError, match="provenance"):
        calibrate_router(model, records("calibration"), records("policy"))


def test_evaluation_rejects_data_used_for_training_calibration_or_policy():
    model = calibrate_router(
        tiny_router(), records("calibration"), records("policy"), min_accepted=2
    )
    for partition in ("train", "calibration", "policy"):
        with pytest.raises(ValueError, match="overlap"):
            evaluate_router(model, records(partition))
    report = evaluate_router(model, records("test"))
    assert report["policy_result"]["accuracy"] == 1
    assert report["policy_result"]["coverage"] == 1


def test_metrics_include_zero_support_classes_reliability_and_coverage():
    report = prediction_metrics(["a", "a"], ["a", "b"], [0.9, 0.6], ["a", "b", "c"])
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
    assert report["classification_report"]["c"]["support"] == 0
    assert report["expected_calibration_error"] == pytest.approx(0.35)
    assert (
        next(row for row in report["accuracy_coverage"] if row["threshold"] == 0.8)["accuracy"] == 1
    )


def test_frozen_training_uses_validation_and_persists_provenance(tmp_path, monkeypatch):
    import sentimentizer.router.train_router as module

    backbone = TinyBackbone()
    backbone.fit = MagicMock()
    monkeypatch.setattr(module, "SentenceTransformer", lambda _: backbone)
    config = RouterConfig(task=tiny_router().task, training_mode="frozen", output_dir=tmp_path)
    train = {key: [row[key] for row in records("train")] for key in ("text", "label", "group_id")}
    validation = {key: [row[key] for row in records("validation")] for key in train}
    model = module.train_router(config, train, validation)
    backbone.fit.assert_not_called()
    assert model.provenance.training_config["seed"] == config.seed
    assert set(model.provenance.dataset_fingerprints) == {"train", "validation"}
    assert json.loads((tmp_path / "validation_report.json").read_text())["accuracy"] == 1
    with pytest.raises(ValueError, match="overlap"):
        module.train_router(config, train, train)


def test_training_provenance_records_resolved_config():
    model = tiny_router()
    config = replace(RouterConfig(), task=model.task)
    provenance = training_provenance(config, records("train"), model.backbone)
    assert provenance.training_config["task"]["labels"] == ["negative", "positive"]
    assert "scikit-learn" in provenance.library_versions


def test_probability_rows_must_be_normalized():
    model = tiny_router()
    model.head = MagicMock(classes_=np.array([0, 1]))
    model.head.predict_proba.return_value = [[0.2, 0.2]]
    with pytest.raises(ValueError, match="sum to one"):
        model.classify_batch(["1 new"])


def test_offline_contrastive_training_and_reload_are_reproducible(tmp_path, monkeypatch):
    """Exercise the actual encoder optimizer without fetching pretrained weights."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
    from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer

    import sentimentizer.router.train_router as module

    vocabulary = ["[PAD]", "good", "bad", "a", "b", "c"]

    def build(_):
        words = WordEmbeddings(
            WhitespaceTokenizer(vocabulary, stop_words=[]),
            torch.randn(len(vocabulary), 4),
            update_embeddings=True,
        )
        return SentenceTransformer(modules=[words, Pooling(4)], device="cpu")

    monkeypatch.setattr(module, "SentenceTransformer", build)
    config = RouterConfig(
        task=RouterTaskSpec(task_id="offline", labels=("negative", "positive")),
        num_iterations=1,
        batch_size=2,
        seed=17,
        learning_rate=0.01,
        output_dir=tmp_path / "first",
    )
    data = {"text": ["good a", "bad a", "good b", "bad b"], "label": [1, 0, 1, 0]}
    first = module.train_router(config, data)
    second = module.train_router(replace(config, output_dir=tmp_path / "second"), data)
    np.testing.assert_allclose(
        first.predict_proba(["good c", "bad c"]),
        second.predict_proba(["good c", "bad c"]),
        atol=1e-7,
    )
    restored = RouterModel.from_pretrained(tmp_path / "first")
    np.testing.assert_allclose(
        first.predict_proba(["good c", "bad c"]),
        restored.predict_proba(["good c", "bad c"]),
        atol=1e-7,
    )


def test_calibration_cli_saves_separate_artifact_and_rejects_overwrite(tmp_path, monkeypatch):
    from sentimentizer.router.dataset import save_dataset_to_jsonl
    from workflows.cli import cli

    original = tmp_path / "original"
    original.mkdir()
    calibration = save_dataset_to_jsonl(records("calibration"), str(tmp_path / "calibration.jsonl"))
    policy = save_dataset_to_jsonl(records("policy"), str(tmp_path / "policy.jsonl"))
    monkeypatch.setattr(RouterModel, "from_pretrained", lambda _: tiny_router())
    args = [
        "router",
        "calibrate",
        "--model-path",
        str(original),
        "--calibration-data",
        str(calibration),
        "--policy-data",
        str(policy),
        "--min-accepted",
        "2",
        "--output-dir",
    ]
    result = CliRunner().invoke(cli, args + [str(tmp_path / "calibrated")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["accepted"] == 8
    metadata = RouterModel.read_metadata(tmp_path / "calibrated")
    assert metadata.acceptance_policy is not None
    result = CliRunner().invoke(cli, args + [str(original)])
    assert result.exit_code != 0
    assert "different output-dir" in result.output


def test_multiclass_calibration_retains_probability_label_mapping():
    class OneHotBackbone:
        def encode(self, texts, **kwargs):
            return np.eye(3)[[int(text.split()[0]) for text in texts]]

    def rows(partition):
        return [
            {"text": f"{label} {partition} {i}", "label": label}
            for label in range(3)
            for i in range(3)
        ]

    backbone = OneHotBackbone()
    train = rows("train")
    head = LogisticRegression().fit(
        backbone.encode([row["text"] for row in train]), [row["label"] for row in train]
    )
    provenance = RouterProvenance()
    add_dataset(provenance, "train", train)
    model = RouterModel(
        backbone,
        head,
        task=RouterTaskSpec(task_id="multiclass", labels=("a", "b", "c")),
        provenance=provenance,
    )
    calibrated = calibrate_router(model, rows("calibration"), rows("policy"), min_accepted=1)
    assert [item.label for item in calibrated.classify_batch(["0 test", "1 test", "2 test"])] == [
        "a",
        "b",
        "c",
    ]
    np.testing.assert_allclose(
        calibrated.probabilities_batch(["0 test", "1 test", "2 test"]).sum(axis=1), 1
    )


def test_calibration_supports_native_string_classes():
    model = tiny_router()
    train = records("train")
    model.head = LogisticRegression().fit(
        model.model_encode([row["text"] for row in train]),
        [model.task.labels[row["label"]] for row in train],
    )
    calibrated = calibrate_router(model, records("calibration"), records("policy"), min_accepted=1)
    assert [item.label for item in calibrated.classify_batch(["-1 test", "2 test"])] == [
        "negative",
        "positive",
    ]


def test_training_cli_persists_optional_group_ids_and_rejects_later_overlap(tmp_path, monkeypatch):
    import sentimentizer.router.train_router as module
    from sentimentizer.router.dataset import record_keys, save_dataset_to_jsonl
    from workflows.cli import cli

    train_rows = [
        {"text": "-1 standalone", "label": 0},
        {"text": "2 source A snippet", "label": 1, "group_id": "review-A"},
        {"text": "-3 source B snippet", "label": 0, "group_id": "review-B"},
        {"text": "4 source C snippet", "label": 1, "group_id": "review-C"},
    ]
    train = save_dataset_to_jsonl(train_rows, str(tmp_path / "train.jsonl"))
    validation = save_dataset_to_jsonl(records("validation"), str(tmp_path / "validation.jsonl"))
    config = tmp_path / "config.yaml"
    config.write_text(
        "training:\n  training_mode: frozen\n  task:\n"
        "    task_id: test\n    labels: [negative, positive]\n"
    )
    output = tmp_path / "model"
    monkeypatch.setattr(module, "SentenceTransformer", lambda _: TinyBackbone())
    result = CliRunner().invoke(
        cli,
        [
            "router",
            "train",
            "--data",
            str(train),
            "--validation-data",
            str(validation),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    metadata = RouterModel.read_metadata(output)
    assert set(record_keys(train_rows)["groups"]) <= set(metadata.provenance.used_keys["groups"])
    monkeypatch.setattr(
        "sentimentizer.router.model._get_sentence_transformers", lambda: lambda _: TinyBackbone()
    )
    restored = RouterModel.from_pretrained(output)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_router(
            restored, [{"text": "6 another source A snippet", "label": 1, "group_id": "review-A"}]
        )


def test_calibrating_a_calibrated_artifact_is_rejected():
    """The stored format holds one sigmoid layer, so stacking must fail early, not at save."""
    calibrated = calibrate_router(
        tiny_router(), records("calibration"), records("policy"), min_accuracy=1.0, min_accepted=2
    )
    with pytest.raises(ValueError, match="already calibrated"):
        calibrate_router(calibrated, records("calibration"), records("policy"))
