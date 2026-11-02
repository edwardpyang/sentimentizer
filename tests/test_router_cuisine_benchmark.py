"""Protect the label and split rules that make the cuisine benchmark meaningful."""

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "router_cuisine_benchmark.py"
spec = importlib.util.spec_from_file_location("router_cuisine_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def _reviews(businesses_per_label=20, reviews_per_business=40):
    return {
        label: {
            f"{label}-{business}": [
                f"{label} {business} review {review}" for review in range(reviews_per_business)
            ]
            for business in range(businesses_per_label)
        }
        for label in benchmark.LABELS
    }


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        ("Restaurants, Italian", "italian"),
        ("Sushi Bars, Restaurants", "japanese"),
        ("Tex-Mex, Mexican, Restaurants", "mexican"),
        ("Restaurants, Italian, French", None),
        ("Restaurants, Thai, Asian Fusion", None),
        ("Italian, Grocery", None),
        ("Restaurants, Pizza", None),
        (None, None),
    ],
)
def test_only_unambiguous_single_cuisine_restaurants_get_a_label(categories, expected):
    assert benchmark.cuisine_label(categories) == expected


def test_split_keeps_businesses_out_of_both_sides_and_nests_training_samples():
    train, test = benchmark.make_split(_reviews(), seed=0)
    test_groups = {row["group_id"] for row in test}
    for shots, rows in train.items():
        assert not test_groups & {row["group_id"] for row in rows}
        assert len(rows) == shots * len(benchmark.LABELS)
    for smaller, larger in zip(benchmark.TRAIN_SHOTS, benchmark.TRAIN_SHOTS[1:], strict=False):
        assert {row["text"] for row in train[smaller]} <= {row["text"] for row in train[larger]}


def test_small_training_samples_spread_across_businesses_before_repeating_one():
    train, _ = benchmark.make_split(_reviews(), seed=0)
    italian = [
        row["group_id"] for row in train[8] if row["label"] == benchmark.LABELS.index("italian")
    ]
    assert len(set(italian)) == len(italian)


def test_split_is_reproducible_and_changes_with_the_seed():
    first = benchmark.make_split(_reviews(), seed=1)
    assert first == benchmark.make_split(_reviews(), seed=1)
    assert first != benchmark.make_split(_reviews(), seed=2)


def test_split_rejects_a_cuisine_without_enough_reviews():
    with pytest.raises(ValueError, match="Not enough reviews"):
        benchmark.make_split(_reviews(businesses_per_label=4, reviews_per_business=2), seed=0)


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("Best Italian in town", True),
        ("the tex-mex plates", True),
        ("Thai basil everywhere", True),
        ("great pasta and pho", False),
        ("the chinatown branch", False),
    ],
)
def test_keyword_subset_only_excludes_reviews_naming_a_cuisine(text, named):
    assert bool(benchmark.CUISINE_KEYWORD.search(text)) is named


def _archive(path, businesses, reviews):
    with tarfile.open(path, "w:gz") as tar:
        for name, rows in (
            ("yelp_academic_dataset_business.json", businesses),
            ("yelp_academic_dataset_review.json", reviews),
        ):
            data = "".join(json.dumps(row) + "\n" for row in rows).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_keeps_long_unique_reviews_of_labelled_restaurants(tmp_path):
    words = "word " * benchmark.MIN_REVIEW_WORDS
    archive = tmp_path / "yelp.tar"
    _archive(
        archive,
        [
            {"business_id": "kept", "categories": "Restaurants, Thai"},
            {"business_id": "mixed", "categories": "Restaurants, Thai, Korean"},
        ],
        [
            {"business_id": "kept", "text": f"first {words}"},
            {"business_id": "kept", "text": f"FIRST   {words}"},
            {"business_id": "kept", "text": "too short"},
            {"business_id": "mixed", "text": f"mixed {words}"},
        ],
    )
    output = tmp_path / "reviews.jsonl"
    benchmark.extract(str(archive), str(output))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row["label"], row["group_id"]) for row in rows] == [("thai", "kept")]
    assert not output.with_name("reviews.jsonl.tmp").exists()


def test_extract_fails_without_replacing_output_when_nothing_matches(tmp_path):
    archive = tmp_path / "yelp.tar"
    _archive(archive, [{"business_id": "b", "categories": "Doctors"}], [])
    output = tmp_path / "reviews.jsonl"
    output.write_text("previous\n")
    with pytest.raises(ValueError):
        benchmark.extract(str(archive), str(output))
    assert output.read_text() == "previous\n"


def test_result_log_resumes_finished_runs(tmp_path):
    path = str(tmp_path / "log.jsonl")
    key = {"kind": "frozen", "encoder": "e", "shots": 8, "seed": 0, "head": "default"}
    log = benchmark.ResultLog(path)
    assert not log.has(**key)
    log.add({**key, "accuracy": 0.5, "seconds": 1.0})
    assert benchmark.ResultLog(path).has(**key)
    assert not benchmark.ResultLog(path).has(**{**key, "seed": 1})


def test_selective_metrics_reward_confidence_that_ranks_errors_last():
    targets = np.array([0, 0, 1, 1])
    well_ranked = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6]])
    badly_ranked = np.array([[0.6, 0.4], [0.7, 0.3], [0.9, 0.1], [0.45, 0.55]])
    good = benchmark.selective_metrics(well_ranked, targets)
    bad = benchmark.selective_metrics(badly_ranked, targets)
    assert good["accuracy_at_50pct_coverage"] == 1.0
    assert bad["accuracy_at_50pct_coverage"] == 0.5
    assert good["aurc"] < bad["aurc"]
    assert good["correct"] == [True, True, False, True]


def test_selective_metrics_penalize_confident_mistakes_in_log_loss_and_brier():
    targets = np.array([0])
    mild = benchmark.selective_metrics(np.array([[0.4, 0.6]]), targets)
    severe = benchmark.selective_metrics(np.array([[0.01, 0.99]]), targets)
    assert severe["nll"] > mild["nll"]
    assert severe["brier"] > mild["brier"]


def test_score_handles_a_test_set_with_and_without_named_cuisines():
    test = [
        {"text": "great noodles", "label": 0},
        {"text": "best thai food", "label": 1},
    ]
    probabilities = np.zeros((2, len(benchmark.LABELS)))
    probabilities[0, 0] = probabilities[1, 2] = 1.0
    row = benchmark.score(test, probabilities, 1.0, encoder="e")
    assert row["accuracy"] == 0.5
    assert row["no_keyword_accuracy"] == 1.0
    assert row["no_keyword_n"] == 1
    assert row["per_class_recall"]["chinese"] == 1.0
    assert row["per_class_precision"]["chinese"] == 1.0
    assert row["per_class_recall"]["french"] == 0.0
    assert row["per_class_precision"]["greek"] == 0.0
    assert row["encoder"] == "e"


def test_report_prints_per_label_results_and_tolerates_older_rows(tmp_path, capsys):
    labels = {label: 0.5 for label in benchmark.LABELS}
    base = {"kind": "frozen", "encoder": "e", "shots": 8, "head": "default"}
    metrics = {"accuracy": 0.5, "macro_f1": 0.5, "no_keyword_accuracy": 0.4, "ece": 0.1}
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({**base, **metrics, "seed": 0})
        + "\n"
        + json.dumps(
            {
                **base,
                **metrics,
                "seed": 0,
                "encoder": "new",
                "nll": 1.0,
                "per_class_precision": labels,
                "per_class_recall": labels,
            }
        )
        + "\n"
    )
    benchmark.report(str(log))
    output = capsys.readouterr().out
    assert "korean" in output and "precision=0.500" in output
    assert "nll=1.000" in output
