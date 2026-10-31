"""Prevent source leakage and recover interrupted synthetic generation."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from sentimentizer.router.augment import augment_seeds, augmentation_counts, prepare_seeds
from sentimentizer.router.config import FOOD_TASK_SPEC
from sentimentizer.router.dataset import assert_disjoint, split_router_records


def source_rows():
    return [
        {"text": f"source {group} variation {v}", "label": group % 3, "group_id": str(group)}
        for group in range(30)
        for v in range(4)
    ]


def test_group_split_preserves_classes_is_disjoint_and_reproducible():
    rows = source_rows()
    train, test = split_router_records(rows)
    assert_disjoint(train, test)
    assert {row["label"] for row in train} == {0, 1, 2}
    assert {row["label"] for row in test} == {0, 1, 2}
    assert len(train) + len(test) == len(rows)
    assert (train, test) == split_router_records(rows)


def test_duplicate_text_links_different_source_groups():
    rows = source_rows()
    rows.append({"text": " SOURCE   0 VARIATION 0 ", "label": 0, "group_id": "3"})
    train, test = split_router_records(rows)
    assert_disjoint(train, test)
    assert ({"0", "3"} <= {row["group_id"] for row in train}) or (
        {"0", "3"} <= {row["group_id"] for row in test}
    )


def test_insufficient_independent_classes_fail_instead_of_leaking():
    with pytest.raises(ValueError, match="independent|preserve"):
        split_router_records([{"text": "one", "label": 0}, {"text": "two", "label": 1}])


def test_conflicting_duplicate_labels_fail():
    with pytest.raises(ValueError, match="conflicting"):
        split_router_records([{"text": "same", "label": 0}, {"text": " SAME ", "label": 1}])


def test_overlap_rejects_both_sources_and_normalized_duplicates():
    first = [{"text": "hello world", "label": 0, "group_id": "source"}]
    for second in (
        [{"text": " HELLO  World ", "label": 0}],
        [{"text": "different", "label": 1, "group_id": "source"}],
    ):
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint(first, second)


def install_response(monkeypatch, rows):
    response = MagicMock()
    response.json.return_value = {"response": "\n".join(json.dumps(row) for row in rows)}
    post = MagicMock(return_value=response)
    monkeypatch.setattr("sentimentizer.router.augment.requests.post", post)
    monkeypatch.setattr("sentimentizer.router.augment.time.sleep", lambda _: None)
    return post


def test_resume_seed_only_file_generates_and_completed_resume_does_no_work(tmp_path, monkeypatch):
    seed = {"text": "Original", "label": 0}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(seed) + "\n")
    post = install_response(monkeypatch, [{"text": "New variation", "label": 0}])
    rows = augment_seeds([seed], variations_per_seed=1, output_path=str(path), resume=True)
    assert post.call_count == 1
    assert list(augmentation_counts(rows).values()) == [1]
    post.reset_mock()
    assert augment_seeds([seed], variations_per_seed=1, output_path=str(path), resume=True) == rows
    post.assert_not_called()


def test_shortfalls_are_bounded_deduplicated_and_resumable(tmp_path, monkeypatch):
    seeds = [{"text": "Original", "label": 0, "group_id": "source"}]
    path = tmp_path / "rows.jsonl"
    post = install_response(
        monkeypatch,
        [
            {"text": "ORIGINAL", "label": 0},
            {"text": "variation", "label": 0, "group_id": "untrusted"},
            {"text": " VARIATION ", "label": 0},
            {"text": "wrong class", "label": 1},
        ],
    )
    rows = augment_seeds(seeds, variations_per_seed=2, output_path=str(path), max_attempts=2)
    assert post.call_count == 2
    assert len(rows) == 2
    assert rows[1]["group_id"] == "source"
    post = install_response(monkeypatch, [{"text": "second variation", "label": 0}])
    rows = augment_seeds(seeds, variations_per_seed=2, output_path=str(path), resume=True)
    assert len(rows) == 3
    assert "Generate 1 variations" in post.call_args.kwargs["json"]["prompt"]


def test_resume_repairs_truncated_tail_but_rejects_ambiguous_legacy_rows(tmp_path, monkeypatch):
    seed = {"text": "Original", "label": 0}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(seed) + '\n{"text":')
    install_response(monkeypatch, [{"text": "variation", "label": 0}])
    assert (
        len(augment_seeds([seed], variations_per_seed=1, output_path=str(path), resume=True)) == 2
    )
    path.write_text(json.dumps({"text": "untraceable variation", "label": 0}) + "\n")
    with pytest.raises(ValueError, match="lineage"):
        augment_seeds([seed], output_path=str(path), resume=True)


@pytest.mark.parametrize("with_calibration", [False, True])
def test_split_cli_writes_disjoint_partitions(tmp_path, with_calibration):
    from workflows.cli import cli

    data = tmp_path / "sources.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in source_rows()))
    output = tmp_path / "split"
    args = ["router", "split", "--data", str(data), "--output-dir", str(output)]
    if with_calibration:
        args += ["--calibration-size", "0.1", "--policy-size", "0.1"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    from sentimentizer.router.dataset import load_router_records

    names = ["train", "validation", "test"] + (
        ["calibration", "policy"] if with_calibration else []
    )
    partitions = [load_router_records(output / f"{name}.jsonl") for name in names]
    assert_disjoint(*partitions)
    assert sum(map(len, partitions)) == len(source_rows())


def test_seed_identity_is_stable_and_task_scoped():
    first = prepare_seeds([{"text": "Seed", "label": 0}], FOOD_TASK_SPEC)[0]
    second = prepare_seeds([{"text": " SEED ", "label": 0}], FOOD_TASK_SPEC)[0]
    assert first["seed_id"] == second["seed_id"]


def test_resume_does_not_silently_relabel_old_seed_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"text": "seed", "label": 1}) + "\n")
    with pytest.raises(ValueError, match="conflicts"):
        augment_seeds([{"text": "seed", "label": 0}], output_path=str(path), resume=True)


def test_null_source_still_gets_seed_lineage_and_numpy_ids_fingerprint():
    from sentimentizer.router.dataset import dataset_fingerprint

    seed = prepare_seeds([{"text": "seed", "label": 0, "group_id": None}], FOOD_TASK_SPEC)[0]
    assert seed["group_id"] == seed["seed_id"]
    assert dataset_fingerprint([{"text": "seed", "label": np.int64(0)}]) == dataset_fingerprint(
        [{"text": "seed", "label": 0}]
    )


def test_source_identifiers_retain_case_while_text_duplicates_are_normalized():
    assert_disjoint(
        [{"text": "first", "label": 0, "group_id": "A"}],
        [{"text": "second", "label": 1, "group_id": "a"}],
    )
