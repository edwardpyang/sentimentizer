#!/usr/bin/env python3
"""Compare router encoders on Yelp cuisine labels, a proxy for real review routing.

Cuisine comes from the business, not from a reader annotating the review, so these labels
are noisy and test a different task than the production topics. See
reports/router-encoder-benchmark.md for the design, results and limits.

    python scripts/router_cuisine_benchmark.py extract --archive yelp_dataset.tar --output c.jsonl
    python scripts/router_cuisine_benchmark.py frozen --data c.jsonl --log r.jsonl --work tmp \
        --batch-size 8 --max-seq-length 128
    python scripts/router_cuisine_benchmark.py contrastive --data c.jsonl --log r.jsonl --work tmp \
        --batch-size 8 --max-seq-length 128
    python scripts/router_cuisine_benchmark.py report --log r.jsonl

Results append to the JSONL log after every run, and finished runs are skipped on restart.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import random
import re
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

LABELS = (
    "chinese",
    "french",
    "greek",
    "indian",
    "italian",
    "japanese",
    "korean",
    "mexican",
    "thai",
    "vietnamese",
)
CATEGORY_LABELS = {
    "Italian": "italian",
    "Mexican": "mexican",
    "Tex-Mex": "mexican",
    "Chinese": "chinese",
    "Japanese": "japanese",
    "Sushi Bars": "japanese",
    "Thai": "thai",
    "Vietnamese": "vietnamese",
    "Korean": "korean",
    "Indian": "indian",
    "Greek": "greek",
    "French": "french",
}
# Mixed-cuisine categories make the single business-level label unreliable.
EXCLUDED_CATEGORIES = {
    "Asian Fusion",
    "Pan Asian",
    "Fusion",
    "Mediterranean",
    "Middle Eastern",
    "Buffets",
}
ENCODERS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
}
REVIEWS_PER_BUSINESS = 40
MIN_REVIEW_WORDS = 30
TEST_PER_CLASS = 150
TEST_PER_BUSINESS = 3
TEST_BUSINESS_FRACTION = 0.3
TRAIN_SHOTS = (8, 32, 128, 512)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_CONTRASTIVE_SHOTS = (8, 32)
DEFAULT_C_GRID = (1.0, 10.0, 100.0)
ENCODE_BATCH_SIZE = 32
COVERAGE_LEVELS = (0.25, 0.5, 0.75)
REPORT_METRICS = (
    "accuracy",
    "macro_f1",
    "no_keyword_accuracy",
    "nll",
    "ece",
    "aurc",
    "accuracy_at_50pct_coverage",
)
# Scores the subset of reviews that cannot be classified by the cuisine's own name.
CUISINE_KEYWORD = re.compile(
    r"\b(ital(y|ian)\w*|mexic\w*|tex-?mex|chin(a|ese)|japan\w*|thai\w*|vietnam\w*|korea\w*"
    r"|india\w*|greek\w*|greece|french|france)\b",
    re.IGNORECASE,
)

Record = dict[str, Any]
ReviewsByLabel = dict[str, dict[str, list[str]]]


def cuisine_label(categories: str | None) -> str | None:
    """Return the cuisine of a restaurant with exactly one, or None if it is ambiguous."""
    names = {name.strip() for name in (categories or "").split(",")}
    found = {CATEGORY_LABELS[name] for name in names & CATEGORY_LABELS.keys()}
    if "Restaurants" not in names or len(found) != 1 or names & EXCLUDED_CATEGORIES:
        return None
    return found.pop()


def extract(archive: str, output: str) -> None:
    """Stream the Yelp tar without unpacking it and sample reviews per business."""
    rng = random.Random(0)
    businesses: dict[str, str] = {}
    reservoirs: dict[str, list[str]] = collections.defaultdict(list)
    seen: collections.Counter[str] = collections.Counter()
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            source = tar.extractfile(member) if member.isfile() else None
            if source is None:
                continue
            if member.name.endswith("business.json"):
                for line in source:
                    business = json.loads(line)
                    label = cuisine_label(business.get("categories"))
                    if label is not None:
                        businesses[business["business_id"]] = label
            elif member.name.endswith("review.json"):
                if not businesses:
                    raise ValueError("The archive listed reviews before businesses")
                for line in source:
                    review = json.loads(line)
                    business_id = review["business_id"]
                    if (
                        business_id not in businesses
                        or len(review["text"].split()) < MIN_REVIEW_WORDS
                    ):
                        continue
                    seen[business_id] += 1
                    if len(reservoirs[business_id]) < REVIEWS_PER_BUSINESS:
                        reservoirs[business_id].append(review["text"])
                    else:
                        slot = rng.randrange(seen[business_id])
                        if slot < REVIEWS_PER_BUSINESS:
                            reservoirs[business_id][slot] = review["text"]
                break
    if not reservoirs:
        raise ValueError(f"No cuisine reviews found in {archive}")
    temporary = Path(f"{output}.tmp")
    duplicates: set[str] = set()
    with temporary.open("w") as sink:
        for business_id, texts in reservoirs.items():
            for text in texts:
                key = " ".join(text.casefold().split())
                if key not in duplicates:
                    duplicates.add(key)
                    row = {"text": text, "label": businesses[business_id], "group_id": business_id}
                    sink.write(json.dumps(row) + "\n")
    temporary.replace(output)
    print(f"Wrote {len(duplicates)} reviews from {len(reservoirs)} businesses to {output}")


def load_reviews(path: str) -> ReviewsByLabel:
    """Group sampled reviews by cuisine and business."""
    reviews: ReviewsByLabel = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path) as source:
        for line in source:
            row = json.loads(line)
            reviews[row["label"]][row["group_id"]].append(row["text"])
    return reviews


def make_split(reviews: ReviewsByLabel, seed: int) -> tuple[dict[int, list[Record]], list[Record]]:
    """Build a business-disjoint test set and nested per-class training samples.

    Training reviews round-robin over businesses so each one contributes one review before
    any contributes a second. Reviews of one restaurant share its name and dishes, so
    letting them straddle the split would measure recall of that restaurant, not cuisine.
    """
    from sentimentizer.router.dataset import assert_disjoint

    rng = random.Random(seed)
    test: list[Record] = []
    train: dict[int, list[Record]] = {shots: [] for shots in TRAIN_SHOTS}
    for label_id, label in enumerate(LABELS):
        groups = sorted(reviews[label])
        rng.shuffle(groups)
        cut = int(len(groups) * TEST_BUSINESS_FRACTION)
        candidates: list[tuple[str, str]] = []
        for group in groups[:cut]:
            texts = sorted(reviews[label][group])
            rng.shuffle(texts)
            candidates += [(text, group) for text in texts[:TEST_PER_BUSINESS]]
        rng.shuffle(candidates)
        test += [
            {"text": text, "label": label_id, "group_id": group}
            for text, group in candidates[:TEST_PER_CLASS]
        ]
        pools = {
            group: rng.sample(sorted(reviews[label][group]), len(reviews[label][group]))
            for group in groups[cut:]
        }
        order: list[tuple[str, str]] = []
        depth = 0
        while len(order) < max(TRAIN_SHOTS):
            before = len(order)
            order += [
                (pools[group][depth], group) for group in groups[cut:] if depth < len(pools[group])
            ]
            if len(order) == before:
                raise ValueError(f"Not enough reviews for {label}")
            depth += 1
        for shots in TRAIN_SHOTS:
            train[shots] += [
                {"text": text, "label": label_id, "group_id": group}
                for text, group in order[:shots]
            ]
    for shots in TRAIN_SHOTS:
        assert_disjoint(train[shots], test)
    return train, test


def selective_metrics(probabilities: np.ndarray, targets: np.ndarray) -> Record:
    """Score confidence ranking and probabilities, independent of any fixed threshold.

    Fixed thresholds confound heads with different confidence scales, so accuracy is taken at
    fixed coverage. ECE alone cannot rank models: a near-chance model can be well calibrated.
    """
    rows = np.arange(len(targets))
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence, kind="stable")
    running_risk = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    onehot = np.eye(probabilities.shape[1])[targets]
    return {
        "nll": float(-np.log(np.clip(probabilities[rows, targets], 1e-12, None)).mean()),
        "brier": float(((probabilities - onehot) ** 2).sum(axis=1).mean()),
        "aurc": float(running_risk.mean()),
        **{
            f"accuracy_at_{int(coverage * 100)}pct_coverage": float(
                correct[order[: max(1, round(coverage * len(order)))]].mean()
            )
            for coverage in COVERAGE_LEVELS
        },
        "confidence": [round(float(value), 5) for value in confidence],
        "correct": [bool(value) for value in correct],
    }


def score(test: list[Record], probabilities: np.ndarray, seconds: float, **key: Any) -> Record:
    """Score one run, including accuracy on reviews that never name a cuisine."""
    from sentimentizer.router.evaluate import prediction_metrics

    names = list(LABELS)
    targets = np.array([row["label"] for row in test])
    predicted_ids = probabilities.argmax(axis=1)
    metrics = prediction_metrics(
        [names[index] for index in targets],
        [names[index] for index in predicted_ids],
        probabilities.max(axis=1).tolist(),
        names,
    )
    subset = [i for i, row in enumerate(test) if not CUISINE_KEYWORD.search(row["text"])]
    correct = predicted_ids == targets
    report = metrics["classification_report"]
    return {
        **key,
        "accuracy": metrics["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "ece": metrics["expected_calibration_error"],
        "no_keyword_accuracy": float(correct[subset].mean()),
        "no_keyword_n": len(subset),
        "test_n": len(test),
        "per_class_precision": {name: report[name]["precision"] for name in names},
        "per_class_recall": {name: report[name]["recall"] for name in names},
        "per_class_f1": {name: report[name]["f1-score"] for name in names},
        "confusion": metrics["confusion_matrix"],
        "seconds": seconds,
        **selective_metrics(probabilities, targets),
    }


class ResultLog:
    """Append finished runs to JSONL so an interrupted benchmark can resume."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.done: set[tuple] = set()
        if os.path.exists(path):
            with open(path) as source:
                self.done = {self._key(json.loads(line)) for line in source if line.strip()}

    @staticmethod
    def _key(row: Record) -> tuple:
        return (row["kind"], row["encoder"], row["shots"], row["seed"], row.get("head", ""))

    def has(self, **key: Any) -> bool:
        return self._key(key) in self.done

    def add(self, row: Record) -> None:
        with open(self.path, "a") as sink:
            sink.write(json.dumps(row) + "\n")
        self.done.add(self._key(row))
        print(
            f"{row['kind']} {row['encoder']} shots={row['shots']} seed={row['seed']} "
            f"head={row['head']} accuracy={row['accuracy']:.3f} seconds={row['seconds']:.0f}",
            flush=True,
        )


def run_tfidf(reviews: ReviewsByLabel, seeds: list[int], log: ResultLog) -> None:
    """Fit the same lexical baseline as the repository benchmark."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from sentimentizer.router.benchmark import TFIDF_NGRAM_RANGE
    from sentimentizer.router.config import RouterConfig

    for seed in seeds:
        train, test = make_split(reviews, seed)
        for shots in TRAIN_SHOTS:
            key = {
                "kind": "tfidf",
                "encoder": "tfidf",
                "shots": shots,
                "seed": seed,
                "head": "default",
            }
            if log.has(**key):
                continue
            started = time.monotonic()
            pipeline = make_pipeline(
                TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE),
                LogisticRegression(max_iter=RouterConfig().head_max_iter, random_state=seed),
            )
            pipeline.fit([row["text"] for row in train[shots]], [row["label"] for row in train[shots]])
            probabilities = pipeline.predict_proba([row["text"] for row in test])
            seconds = time.monotonic() - started
            log.add(score(test, probabilities, seconds, **key))


def run_frozen(
    reviews: ReviewsByLabel,
    args: argparse.Namespace,
    log: ResultLog,
) -> None:
    """Fit heads on cached embeddings, which matches train_router's frozen mode.

    The production head is unchanged in "default". The "l2_C*" heads unit-normalize first so
    encoders without a Normalize module (mxbai) are compared under the same regularization.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    from sentimentizer.router.config import RouterConfig

    splits = {seed: make_split(reviews, seed) for seed in args.seeds}
    texts = sorted(
        {
            row["text"]
            for train, test in splits.values()
            for rows in (*train.values(), test)
            for row in rows
        }
    )
    index = {text: i for i, text in enumerate(texts)}
    heads = ["default"] + [f"l2_C{value:g}" for value in args.c_grid]
    for name in args.encoders:
        if all(
            log.has(kind="frozen", encoder=name, shots=shots, seed=seed, head=head)
            for shots in args.eval_shots
            for seed in args.seeds
            for head in heads
        ):
            continue
        model = SentenceTransformer(ENCODERS[name])
        model.max_seq_length = args.max_seq_length
        embeddings = model.encode(
            texts, convert_to_numpy=True, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=False
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        unit = embeddings / np.clip(norms, 1e-12, None)
        variants = [("default", embeddings, 1.0)] + [
            (f"l2_C{value:g}", unit, value) for value in args.c_grid
        ]
        for seed, (train, test) in splits.items():
            test_ids = [index[row["text"]] for row in test]
            for shots in args.eval_shots:
                train_ids = [index[row["text"]] for row in train[shots]]
                labels = [row["label"] for row in train[shots]]
                for head, matrix, regularization in variants:
                    key = {
                        "kind": "frozen",
                        "encoder": name,
                        "shots": shots,
                        "seed": seed,
                        "head": head,
                    }
                    if log.has(**key):
                        continue
                    started = time.monotonic()
                    classifier = LogisticRegression(
                        max_iter=RouterConfig().head_max_iter,
                        solver="lbfgs",
                        random_state=seed,
                        C=regularization,
                    ).fit(matrix[train_ids], labels)
                    probabilities = classifier.predict_proba(matrix[test_ids])
                    seconds = time.monotonic() - started
                    log.add(score(test, probabilities, seconds, **key))


def run_contrastive(reviews: ReviewsByLabel, args: argparse.Namespace, log: ResultLog) -> None:
    """Fine-tune through the production train_router entry point."""
    import torch

    from sentimentizer.router.config import RouterConfig
    from sentimentizer.router.dataset import records_to_columns
    from sentimentizer.router.schemas import RouterTaskSpec
    from sentimentizer.router.train_router import train_router

    task = RouterTaskSpec(task_id="yelp_cuisine_proxy", labels=LABELS)
    for shots in args.contrastive_shots:
        for seed in args.seeds:
            for name in args.encoders:
                key = {
                    "kind": "contrastive",
                    "encoder": name,
                    "shots": shots,
                    "seed": seed,
                    "head": "train_router",
                }
                if log.has(**key):
                    continue
                train, test = make_split(reviews, seed)
                output = Path(args.work) / f"{name}-{shots}-{seed}"
                started = time.monotonic()
                model = train_router(
                    RouterConfig(
                        base_model=ENCODERS[name],
                        task=task,
                        seed=seed,
                        training_mode="contrastive",
                        output_dir=output,
                        batch_size=args.batch_size,
                        max_seq_length=args.max_seq_length,
                    ),
                    records_to_columns(train[shots]),
                )
                seconds = time.monotonic() - started
                if tuple(model.class_labels()) != LABELS:
                    raise ValueError("Probability columns do not follow the task label order")
                probabilities = model.probabilities_batch([row["text"] for row in test])
                log.add(score(test, probabilities, seconds, **key))
                shutil.rmtree(output, ignore_errors=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()


def paired_interval(differences: np.ndarray) -> float:
    """Return the 95% t half-width of paired per-seed differences."""
    from scipy import stats

    if len(differences) < 2:
        return float("nan")
    spread = differences.std(ddof=1) / math.sqrt(len(differences))
    return float(stats.t.ppf(0.975, len(differences) - 1) * spread)


def report(path: str) -> None:
    """Print seed-averaged results with paired differences from the best encoder."""
    with open(path) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    grouped: dict[tuple, dict[str, dict[int, Record]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    for row in rows:
        grouped[(row["kind"], row["head"], row["shots"])][row["encoder"]][row["seed"]] = row
    for (kind, head, shots), encoders in sorted(grouped.items()):
        print(f"\n{kind} head={head} shots={shots}")
        ranked = sorted(
            encoders.items(),
            key=lambda item: -np.mean([row["accuracy"] for row in item[1].values()]),
        )
        best = ranked[0][1]
        for name, by_seed in ranked:
            common = sorted(set(by_seed) & set(best))
            difference = np.array([by_seed[s]["accuracy"] - best[s]["accuracy"] for s in common])
            summary = " ".join(
                f"{metric}={np.mean([row[metric] for row in by_seed.values()]):.3f}"
                for metric in REPORT_METRICS
                if all(metric in row for row in by_seed.values())
            )
            print(
                f"  {name:22s} n={len(by_seed)} {summary} "
                f"gap={difference.mean():+.3f}±{paired_interval(difference):.3f}"
            )
            if not all("per_class_precision" in row for row in by_seed.values()):
                continue
            for label in LABELS:
                precision, recall = (
                    np.mean([row[field][label] for row in by_seed.values()])
                    for field in ("per_class_precision", "per_class_recall")
                )
                print(f"    {label:12s} precision={precision:.3f} recall={recall:.3f}")


def parse_args() -> argparse.Namespace:
    """Define one subcommand per pipeline stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--archive", required=True)
    extract_parser.add_argument("--output", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--log", required=True)
    for command in ("frozen", "contrastive"):
        stage = commands.add_parser(command)
        stage.add_argument("--data", required=True)
        stage.add_argument("--log", required=True)
        stage.add_argument("--work", required=True)
        stage.add_argument("--batch-size", type=int, required=True)
        stage.add_argument("--max-seq-length", type=int, required=True)
        stage.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
        stage.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
        stage.add_argument("--eval-shots", type=int, nargs="+", default=list(TRAIN_SHOTS))
        stage.add_argument(
            "--contrastive-shots", type=int, nargs="+", default=list(DEFAULT_CONTRASTIVE_SHOTS)
        )
        stage.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
        stage.add_argument(
            "--mxbai-path",
            help="Float32 copy of mxbai: its checkpoint is float16, which train_router cannot tune",
        )
        stage.add_argument("--skip-tfidf", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested stage."""
    args = parse_args()
    if args.command == "extract":
        extract(args.archive, args.output)
    elif args.command == "report":
        report(args.log)
    else:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if args.mxbai_path:
            ENCODERS["mxbai-embed-large-v1"] = args.mxbai_path
        reviews = load_reviews(args.data)
        log = ResultLog(args.log)
        if args.command == "contrastive":
            run_contrastive(reviews, args, log)
        else:
            if not args.skip_tfidf:
                run_tfidf(reviews, args.seeds, log)
            run_frozen(reviews, args, log)


if __name__ == "__main__":
    main()
