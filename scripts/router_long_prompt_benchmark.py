#!/usr/bin/env python3
"""Test whether router encoders find a short signal buried in a very long prompt.

No encoder reads 200k tokens in one pass, so each prompt is split into windows. A synthetic
prompt holds a cuisine review at a known position, surrounded by non-food Yelp reviews. This
measures which window size and aggregation survive where the signal sits, not how real
prompts route. Signals never straddle a window boundary, which flatters every method.

    python scripts/router_long_prompt_benchmark.py extract-filler \
        --archive yelp_dataset.tar --output filler.jsonl
    python scripts/router_long_prompt_benchmark.py run --data c.jsonl --filler filler.jsonl \
        --log r.jsonl --work tmp
    python scripts/router_cuisine_benchmark.py report --log r.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from router_cuisine_benchmark import (
    CUISINE_KEYWORD,
    LABELS,
    MIN_REVIEW_WORDS,
    Record,
    ResultLog,
    ReviewsByLabel,
    load_reviews,
    make_split,
    score,
)

# (encoder name, Hub id, window tokens). The full-precision Qwen3-Embedding-8B is left out:
# its weights alone take about 15 GB of a 16 GB card.
WINDOW_CONFIGS = (
    ("nemotron-1b", "nvidia/Nemotron-3-Embed-1B-BF16", 8192),
    ("nemotron-1b", "nvidia/Nemotron-3-Embed-1B-BF16", 32768),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", 512),
    ("mxbai-embed-large-v1", "mixedbread-ai/mxbai-embed-large-v1", 512),
    ("bge-m3", "BAAI/bge-m3", 512),
    ("bge-m3", "BAAI/bge-m3", 8192),
    ("modernbert-large", "answerdotai/ModernBERT-large", 8192),
    ("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", 512),
    ("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", 8192),
    ("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", 32768),
    ("qwen3-4b", "Qwen/Qwen3-Embedding-4B", 8192),
)
# NVFP4 checkpoints run only through vLLM, which needs its own environment and server.
VLLM_CONFIGS = (
    ("qwen3-8b-nvfp4", "alexliap/Qwen3-Embedding-8B-NVFP4", 8192),
    ("qwen3-8b-nvfp4", "alexliap/Qwen3-Embedding-8B-NVFP4", 32768),
    ("nemotron-1b-nvfp4", "nvidia/Nemotron-3-Embed-1B-NVFP4", 8192),
    ("nemotron-1b-nvfp4", "nvidia/Nemotron-3-Embed-1B-NVFP4", 32768),
)
VLLM_TIMEOUT_S = 600
ENCODE_BATCH_BY_WINDOW = {512: 32, 8192: 2, 32768: 1}
DOC_TOKENS = (8192, 32768, 200000)
POSITIONS = ("start", "middle", "end", "scattered")
WINDOW_METHODS = ("first", "last", "first_last", "max_window")
SCATTERED_SIGNALS = 3
TRAIN_SHOTS = 32
TEST_DOCS_PER_CLASS = 20
POOL_TOKENS = 1_600_000
HEAD_C = 10.0
HEAD_MAX_ITER = 1000
DEFAULT_SEEDS = (0, 1, 2)
FILLER_REVIEWS = 30000
FOOD_CATEGORY_TERMS = (
    "Restaurants",
    "Food",
    "Bars",
    "Nightlife",
    "Coffee",
    "Cafes",
    "Bakeries",
    "Grocery",
    "Desserts",
    "Event Planning",
)
FILLER_LABEL = len(LABELS)


def is_filler_business(categories: str | None) -> bool:
    """Return whether a business has categories and none of them concern food."""
    return bool(categories) and not any(term in categories for term in FOOD_CATEGORY_TERMS)


def extract_filler(archive: str, output: str) -> None:
    """Sample non-food reviews that never name a cuisine, so filler holds no signal."""
    rng = random.Random(0)
    businesses: set[str] = set()
    sample: list[str] = []
    seen = 0
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            source = tar.extractfile(member) if member.isfile() else None
            if source is None:
                continue
            if member.name.endswith("business.json"):
                for line in source:
                    business = json.loads(line)
                    if is_filler_business(business.get("categories")):
                        businesses.add(business["business_id"])
            elif member.name.endswith("review.json"):
                if not businesses:
                    raise ValueError("The archive listed reviews before businesses")
                for line in source:
                    review = json.loads(line)
                    text = review["text"]
                    if (
                        review["business_id"] not in businesses
                        or len(text.split()) < MIN_REVIEW_WORDS
                        or CUISINE_KEYWORD.search(text)
                    ):
                        continue
                    seen += 1
                    if len(sample) < FILLER_REVIEWS:
                        sample.append(text)
                    else:
                        slot = rng.randrange(seen)
                        if slot < FILLER_REVIEWS:
                            sample[slot] = text
                break
    if not sample:
        raise ValueError(f"No filler reviews found in {archive}")
    temporary = Path(f"{output}.tmp")
    with temporary.open("w") as sink:
        for text in sample:
            sink.write(json.dumps({"text": text}) + "\n")
    temporary.replace(output)
    print(f"Wrote {len(sample)} filler reviews to {output}")


def place_signal(
    signal: list[int], context: list[int], window: int, rng: random.Random
) -> list[int]:
    """Return one window of tokens holding the signal at a random offset within filler."""
    kept = signal[:window]
    offset = rng.randrange(window - len(kept) + 1)
    return context[:offset] + kept + context[offset : window - len(kept)]


def signal_indices(windows: int, position: str) -> list[int]:
    """Return which windows of a prompt hold signal for one position."""
    if position == "start":
        return [0]
    if position == "end":
        return [windows - 1]
    if position == "middle":
        return [windows // 2]
    if position != "scattered":
        raise ValueError(f"Unknown position: {position}")
    count = min(SCATTERED_SIGNALS, windows)
    return sorted({int(i) for i in np.linspace(0, windows - 1, count).round()})


def method_rows(method: str, rows: np.ndarray) -> np.ndarray:
    """Select the window probabilities one aggregation method may look at."""
    if method == "first":
        return rows[:1]
    if method == "last":
        return rows[-1:]
    if method == "first_last":
        return rows[[0, -1]]
    if method == "max_window":
        return rows
    raise ValueError(f"Unknown method: {method}")


def most_cuisine_like(rows: np.ndarray) -> np.ndarray:
    """Return cuisine probabilities of the window most confident in any cuisine.

    The last column is the filler class. It is dropped after choosing, so a prompt that looks
    entirely like filler still gets a forced cuisine and scores low on confidence.
    """
    cuisines = rows[:, :FILLER_LABEL]
    best = cuisines[cuisines.max(axis=1).argmax()]
    return best / np.clip(best.sum(), 1e-12, None)


def unit(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows, so encoders with and without a Normalize module compare fairly."""
    return vectors / np.clip(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12, None)


def fit_head(features: np.ndarray, labels: np.ndarray, seed: int) -> Any:
    """Fit the classifier used by every window size and aggregation."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=HEAD_C, max_iter=HEAD_MAX_ITER, random_state=seed).fit(
        features, labels
    )


class VllmEmbeddingModel:
    """Embed through a local vLLM server with the interface WindowEncoder expects."""

    def __init__(self, url: str, model_id: str, window: int, tokenizer: Any) -> None:
        self.url = url
        self.model_id = model_id
        self.window = window
        self.tokenizer = tokenizer

    def encode(self, texts: list[str], batch_size: int, **_: Any) -> np.ndarray:
        """Request embeddings in batches, restoring input order from the response."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            payload = {
                "model": self.model_id,
                "input": texts[start : start + batch_size],
                "truncate_prompt_tokens": self.window,
            }
            request = urllib.request.Request(
                f"{self.url}/v1/embeddings",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=VLLM_TIMEOUT_S) as response:
                rows = json.load(response)["data"]
            vectors += [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]
        return np.array(vectors, dtype=np.float32)


class WindowEncoder:
    """Build token windows for one encoder and cache their embeddings on disk."""

    def __init__(self, model: Any, name: str, window: int, batch_size: int, work: Path) -> None:
        self.model = model
        self.name = name
        self.window = window
        self.batch_size = batch_size
        self.work = work

    def tokens(self, text: str) -> list[int]:
        """Tokenize without special tokens, since the model adds its own."""
        return self.model.tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(self, windows: list[list[int]]) -> np.ndarray:
        """Embed token windows, truncating any drift from decoding at the model limit."""
        texts = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in windows]
        started = time.monotonic()
        prompts = getattr(self.model, "prompts", None) or {}
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name="document" if "document" in prompts else None,
        )
        print(
            f"{self.name}@{self.window} encoded {len(windows)} windows "
            f"in {time.monotonic() - started:.0f}s",
            flush=True,
        )
        return unit(np.asarray(vectors, dtype=np.float32))

    def cached(self, stem: str, build: Any) -> np.ndarray:
        """Return cached embeddings, publishing a finished array atomically."""
        path = self.work / f"{self.name}@{self.window}-{stem}.npy"
        if path.exists():
            return np.load(path)
        vectors = self.encode(build())
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as sink:
            np.save(sink, vectors)
        temporary.replace(path)
        return vectors


def test_rows(test: list[Record], per_class: int) -> list[Record]:
    """Take the first prompts per class from an already shuffled test set."""
    taken: dict[int, int] = dict.fromkeys(range(len(LABELS)), 0)
    rows = []
    for row in test:
        if taken[row["label"]] < per_class:
            taken[row["label"]] += 1
            rows.append(row)
    return rows


def sibling_texts(reviews: ReviewsByLabel, row: Record) -> list[str]:
    """Return the review plus others from its restaurant, which never reaches training."""
    others = sorted(
        text for text in reviews[LABELS[row["label"]]][row["group_id"]] if text != row["text"]
    )
    random.Random(row["text"]).shuffle(others)
    return [row["text"], *others][:SCATTERED_SIGNALS]


def evaluate_documents(
    encoder_key: str,
    pool: np.ndarray,
    train: np.ndarray,
    train_labels: np.ndarray,
    signals: np.ndarray,
    rows: list[Record],
    seed: int,
    doc_tokens: int,
    window: int,
    log: ResultLog,
) -> None:
    """Score every aggregation and position for one prompt length."""
    rng = np.random.default_rng(seed)
    windows = math.ceil(doc_tokens / window)
    filler_labels = np.full(len(train), FILLER_LABEL)
    filler_ids = rng.choice(len(pool), size=min(len(train), len(pool)), replace=False)
    window_head = fit_head(
        np.vstack([train, pool[filler_ids]]),
        np.concatenate([train_labels, filler_labels[: len(filler_ids)]]),
        seed,
    )
    pool_probabilities = window_head.predict_proba(pool)
    signal_probabilities = window_head.predict_proba(signals.reshape(-1, signals.shape[-1]))
    signal_probabilities = signal_probabilities.reshape(*signals.shape[:2], -1)

    def sample_filler(count: int) -> np.ndarray:
        return rng.choice(len(pool), size=count, replace=count > len(pool))

    train_means = np.stack(
        [
            unit((vector + pool[sample_filler(windows - 1)].sum(axis=0)) / windows)
            for vector in train
        ]
    )
    mean_head = fit_head(train_means, train_labels, seed)

    for position in POSITIONS:
        where = signal_indices(windows, position)
        method_probabilities: dict[str, list[np.ndarray]] = {
            method: [] for method in (*WINDOW_METHODS, "mean")
        }
        for doc, _ in enumerate(rows):
            order = sample_filler(windows)
            document_rows = pool_probabilities[order].copy()
            vectors = pool[order].copy()
            for slot, window_index in enumerate(where):
                document_rows[window_index] = signal_probabilities[doc, slot]
                vectors[window_index] = signals[doc, slot]
            for method in WINDOW_METHODS:
                method_probabilities[method].append(
                    most_cuisine_like(method_rows(method, document_rows))
                )
            method_probabilities["mean"].append(
                mean_head.predict_proba(unit(vectors.mean(axis=0))[None])[0]
            )
        for method, probabilities in method_probabilities.items():
            key = {
                "kind": "long_prompt",
                "encoder": encoder_key,
                "shots": doc_tokens,
                "seed": seed,
                "head": f"{method}/{position}",
            }
            if not log.has(**key):
                log.add(score(rows, np.stack(probabilities), 0.0, **key))


def run_config(
    encoder: WindowEncoder,
    reviews: ReviewsByLabel,
    filler_texts: list[str],
    args: argparse.Namespace,
    log: ResultLog,
) -> None:
    """Embed windows once for an encoder and window size, then score every prompt length."""
    window = encoder.window
    stream: list[int] = []
    needed = args.pool_tokens + window * 4
    for text in filler_texts:
        stream += encoder.tokens(text)
        if len(stream) >= needed:
            break
    if len(stream) < 2 * window:
        raise ValueError(f"Filler holds too few tokens for a {window}-token window")
    pool_size = max(2, min(args.pool_tokens, len(stream)) // window)
    pool = encoder.cached(
        "filler",
        lambda: [stream[i * window : (i + 1) * window] for i in range(pool_size)],
    )
    for seed in args.seeds:
        run_seed(encoder, stream, pool, reviews, seed, args, log)


def run_seed(
    encoder: WindowEncoder,
    stream: list[int],
    pool: np.ndarray,
    reviews: ReviewsByLabel,
    seed: int,
    args: argparse.Namespace,
    log: ResultLog,
) -> None:
    """Embed one seed's signal windows and score every prompt length."""
    window = encoder.window
    rng = random.Random(seed)

    def in_context(text: str) -> list[int]:
        start = rng.randrange(max(1, len(stream) - window))
        return place_signal(encoder.tokens(text), stream[start : start + window], window, rng)

    train_split, test = make_split(reviews, seed)
    train_records = train_split[args.train_shots]
    rows = test_rows(test, args.test_docs_per_class)
    train = encoder.cached(
        f"train-{seed}", lambda: [in_context(row["text"]) for row in train_records]
    )
    flat = encoder.cached(
        f"signals-{seed}",
        lambda: [
            in_context(text)
            for row in rows
            for text in (sibling_texts(reviews, row) * SCATTERED_SIGNALS)[:SCATTERED_SIGNALS]
        ],
    )
    signals = flat.reshape(len(rows), SCATTERED_SIGNALS, -1)
    train_labels = np.array([row["label"] for row in train_records])
    for doc_tokens in args.doc_tokens:
        if doc_tokens >= window:
            evaluate_documents(
                f"{encoder.name}@{window}",
                pool,
                train,
                train_labels,
                signals,
                rows,
                seed,
                doc_tokens,
                window,
                log,
            )


def run_all(args: argparse.Namespace) -> None:
    """Run each configuration, recording memory failures since they are results here."""
    import torch
    from sentence_transformers import SentenceTransformer

    reviews = load_reviews(args.data)
    with open(args.filler) as source:
        filler_texts = [json.loads(line)["text"] for line in source]
    random.Random(0).shuffle(filler_texts)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    log = ResultLog(args.log)
    if args.vllm_url:
        run_vllm(args, reviews, filler_texts, work, log)
        return
    failures: list[str] = []
    for name, model_id, window in WINDOW_CONFIGS:
        if args.only and f"{name}@{window}" not in args.only:
            continue
        model = None
        try:
            model = SentenceTransformer(model_id)
            model.max_seq_length = window
            encoder = WindowEncoder(model, name, window, ENCODE_BATCH_BY_WINDOW[window], work)
            run_config(encoder, reviews, filler_texts, args, log)
        except torch.cuda.OutOfMemoryError as error:
            failures.append(f"{name}@{window}")
            print(f"{name}@{window} ran out of memory: {error}", file=sys.stderr, flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"Out of memory: {', '.join(failures)}")


def run_vllm(
    args: argparse.Namespace,
    reviews: ReviewsByLabel,
    filler_texts: list[str],
    work: Path,
    log: ResultLog,
) -> None:
    """Run the configurations --only selects against the model the vLLM server has loaded."""
    from transformers import AutoTokenizer

    for name, model_id, window in VLLM_CONFIGS:
        if f"{name}@{window}" not in (args.only or ()):
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VllmEmbeddingModel(args.vllm_url, model_id, window, tokenizer)
        encoder = WindowEncoder(model, name, window, ENCODE_BATCH_BY_WINDOW[window], work)
        run_config(encoder, reviews, filler_texts, args, log)


def parse_args() -> argparse.Namespace:
    """Define the filler extraction and benchmark stages."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    filler = commands.add_parser("extract-filler")
    filler.add_argument("--archive", required=True)
    filler.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--data", required=True)
    run.add_argument("--filler", required=True)
    run.add_argument("--log", required=True)
    run.add_argument("--work", required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    run.add_argument("--doc-tokens", type=int, nargs="+", default=list(DOC_TOKENS))
    run.add_argument("--train-shots", type=int, default=TRAIN_SHOTS)
    run.add_argument("--test-docs-per-class", type=int, default=TEST_DOCS_PER_CLASS)
    run.add_argument("--pool-tokens", type=int, default=POOL_TOKENS)
    run.add_argument("--only", nargs="+", help="Limit to configurations such as bge-m3@8192")
    run.add_argument("--vllm-url", help="Local vLLM server for the NVFP4 configurations in --only")
    return parser.parse_args()


def main() -> None:
    """Run the requested stage."""
    args = parse_args()
    if args.command == "extract-filler":
        extract_filler(args.archive, args.output)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
