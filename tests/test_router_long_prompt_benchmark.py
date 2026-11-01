"""Protect the prompt construction and aggregation rules of the long-prompt benchmark."""

import argparse
import importlib.util
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "router_long_prompt_benchmark", SCRIPTS / "router_long_prompt_benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class FakeTokenizer:
    """Map each word to one id, like a tokenizer with no subword splitting."""

    def __init__(self):
        self.ids = {}
        self.words = []

    def __call__(self, text, add_special_tokens=False):
        tokens = []
        for word in text.split():
            if word not in self.ids:
                self.ids[word] = len(self.words)
                self.words.append(word)
            tokens.append(self.ids[word])
        return {"input_ids": tokens}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(self.words[token] for token in tokens)


class FakeModel:
    """Embed text by which cuisine words it holds, so only signal windows look like a cuisine."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = 0

    def encode(self, texts, **_):
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        vectors = rng.normal(0, 0.02, (len(texts), len(benchmark.LABELS) + 1))
        for row, text in enumerate(texts):
            words = text.split()
            for index, label in enumerate(benchmark.LABELS):
                vectors[row, index] += words.count(label)
            vectors[row, -1] += words.count("filler") / 8
        return vectors


def _reviews():
    return {
        label: {
            f"{label}-{business}": [f"{label} {business} review {review}" for review in range(40)]
            for business in range(20)
        }
        for label in benchmark.LABELS
    }


def _tar(path, business_rows, review_rows):
    with tarfile.open(path, "w") as tar:
        for name, rows in (("business.json", business_rows), ("review.json", review_rows)):
            payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def test_signal_stays_whole_inside_a_full_window():
    rng = random.Random(0)
    signal = [1, 2, 3]
    window = benchmark.place_signal(signal, [9] * 10, 10, rng)
    assert len(window) == 10
    start = next(i for i, token in enumerate(window) if token == 1)
    assert window[start : start + 3] == signal
    assert window.count(9) == 7


def test_signal_longer_than_a_window_fills_it_without_overflowing():
    assert benchmark.place_signal(list(range(20)), [99] * 8, 8, random.Random(0)) == list(range(8))


@pytest.mark.parametrize(
    ("windows", "position", "expected"),
    [
        (7, "start", [0]),
        (7, "middle", [3]),
        (7, "end", [6]),
        (7, "scattered", [0, 3, 6]),
        (1, "scattered", [0]),
        (1, "end", [0]),
    ],
)
def test_signal_position_maps_to_the_expected_windows(windows, position, expected):
    assert benchmark.signal_indices(windows, position) == expected


def test_unknown_position_and_method_are_rejected():
    with pytest.raises(ValueError, match="position"):
        benchmark.signal_indices(3, "nowhere")
    with pytest.raises(ValueError, match="method"):
        benchmark.method_rows("nope", np.zeros((1, 3)))


def test_most_cuisine_like_ignores_confident_filler_windows():
    labels = len(benchmark.LABELS)
    filler = np.zeros(labels + 1)
    filler[-1] = 0.99
    filler[0] = 0.01
    signal = np.zeros(labels + 1)
    signal[3] = 0.6
    signal[-1] = 0.4
    chosen = benchmark.most_cuisine_like(np.stack([filler, signal]))
    assert chosen.argmax() == 3
    assert chosen.sum() == pytest.approx(1.0)


def test_extract_filler_drops_food_businesses_and_reviews_naming_a_cuisine(tmp_path):
    archive = tmp_path / "yelp.tar"
    long_text = " ".join(["word"] * benchmark.MIN_REVIEW_WORDS)
    _tar(
        archive,
        [
            {"business_id": "garage", "categories": "Auto Repair, Automotive"},
            {"business_id": "diner", "categories": "Restaurants, Italian"},
            {"business_id": "unknown", "categories": None},
        ],
        [
            {"business_id": "garage", "text": f"kept {long_text}"},
            {"business_id": "garage", "text": "too short"},
            {"business_id": "garage", "text": f"italian mechanic {long_text}"},
            {"business_id": "diner", "text": f"food {long_text}"},
            {"business_id": "unknown", "text": f"unlabelled {long_text}"},
        ],
    )
    output = tmp_path / "filler.jsonl"
    benchmark.extract_filler(str(archive), str(output))
    rows = [json.loads(line)["text"] for line in output.read_text().splitlines()]
    assert rows == [f"kept {long_text}"]


def test_extract_filler_keeps_previous_output_when_nothing_matches(tmp_path):
    archive = tmp_path / "yelp.tar"
    _tar(archive, [{"business_id": "garage", "categories": "Automotive"}], [])
    output = tmp_path / "filler.jsonl"
    output.write_text("previous")
    with pytest.raises(ValueError, match="No filler"):
        benchmark.extract_filler(str(archive), str(output))
    assert output.read_text() == "previous"


def _args(tmp_path):
    return argparse.Namespace(
        seeds=[0],
        doc_tokens=[16, 160],
        train_shots=8,
        test_docs_per_class=4,
        pool_tokens=800,
        work=str(tmp_path),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_methods_only_score_well_when_they_can_see_the_signal(tmp_path):
    log = benchmark.ResultLog(str(tmp_path / "log.jsonl"))
    encoder = benchmark.WindowEncoder(FakeModel(), "fake", 16, 4, tmp_path)
    filler = ["filler " * 40] * 200
    benchmark.run_config(encoder, _reviews(), filler, _args(tmp_path), log)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    accuracy = {(row["head"], row["shots"]): row["accuracy"] for row in rows}
    assert accuracy[("first/start", 160)] > 0.9
    assert accuracy[("first/end", 160)] < 0.4
    assert accuracy[("last/end", 160)] > 0.9
    assert accuracy[("first_last/middle", 160)] < 0.4
    assert accuracy[("max_window/middle", 160)] > 0.9


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_rerun_reuses_cached_embeddings_and_finished_rows(tmp_path):
    path = tmp_path / "log.jsonl"
    filler = ["filler " * 40] * 200
    args = _args(tmp_path)
    first = benchmark.WindowEncoder(FakeModel(), "fake", 16, 4, tmp_path)
    benchmark.run_config(first, _reviews(), filler, args, benchmark.ResultLog(str(path)))
    before = path.read_text()
    second_model = FakeModel()
    second = benchmark.WindowEncoder(second_model, "fake", 16, 4, tmp_path)
    benchmark.run_config(second, _reviews(), filler, args, benchmark.ResultLog(str(path)))
    assert second_model.calls == 0
    assert path.read_text() == before
    assert not list(tmp_path.glob("*.tmp"))


def _serve_embeddings(monkeypatch, rows):
    requests = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        requests.append((request.full_url, json.loads(request.data), timeout))
        return Response(json.dumps({"data": rows}).encode())

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", urlopen)
    return requests


def test_vllm_model_restores_input_order_and_truncates_at_the_window(monkeypatch):
    rows = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
    requests = _serve_embeddings(monkeypatch, rows)
    model = benchmark.VllmEmbeddingModel("http://localhost:8000", "m", 512, None)
    vectors = model.encode(["a", "b"], batch_size=2)
    assert vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    url, payload, timeout = requests[0]
    assert url == "http://localhost:8000/v1/embeddings"
    assert payload == {"model": "m", "input": ["a", "b"], "truncate_prompt_tokens": 512}
    assert timeout == benchmark.VLLM_TIMEOUT_S


def test_vllm_server_failure_reaches_the_caller_without_a_cached_result(monkeypatch, tmp_path):
    def urlopen(request, timeout):
        raise benchmark.urllib.error.URLError("connection refused")

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", urlopen)
    model = benchmark.VllmEmbeddingModel("http://localhost:8000", "m", 8, FakeTokenizer())
    encoder = benchmark.WindowEncoder(model, "fake", 8, 2, tmp_path)
    with pytest.raises(benchmark.urllib.error.URLError):
        encoder.cached("filler", lambda: [[]])
    assert not list(tmp_path.iterdir())
