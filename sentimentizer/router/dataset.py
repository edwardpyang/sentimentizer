"""Dataset loading and train/test split for the router.

Loads JSONL files with text and integer task-label IDs
and splits into train/test datasets for router training.
"""

import hashlib
import json
import logging
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

SPLIT_CANDIDATES = 128


def text_key(text: str) -> str:
    """Hash normalized text for duplicate and overlap checks without storing it."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def record_keys(records: list[dict]) -> dict[str, list[str]]:
    """Describe text and caller-supplied source groups for leakage checks."""
    return {
        "texts": sorted({text_key(row["text"]) for row in records}),
        "groups": sorted(
            {text_key(str(row["group_id"])) for row in records if row.get("group_id") is not None}
        ),
    }


def assert_disjoint(*partitions: list[dict]) -> None:
    """Reject shared sources or normalized texts across dataset partitions."""
    assert_disjoint_keys(*(record_keys(rows) for rows in partitions))


def assert_disjoint_keys(*partitions: dict[str, list[str]]) -> None:
    """Check persisted fingerprints against new data without retaining raw text."""
    for i, first in enumerate(partitions):
        for second in partitions[i + 1 :]:
            if any(
                set(first.get(key, [])) & set(second.get(key, [])) for key in ("texts", "groups")
            ):
                raise ValueError("Router datasets overlap by source group or normalized text")


def dataset_fingerprint(records: list[dict]) -> str:
    """Fingerprint labeled inputs independently of their row order."""
    content = sorted(
        (text_key(row["text"]), row["label"], str(row.get("group_id", ""))) for row in records
    )
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def split_router_records(
    records: list[dict], test_size: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split connected source/duplicate groups while retaining all classes.

    The requested fraction is approximate when groups have unequal sizes.
    Legacy rows without group_id are grouped only by normalized text; that
    cannot identify paraphrases and does not prove semantic independence.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if not 0 < test_size < 1 or not records:
        raise ValueError("A nonempty dataset and test_size between zero and one are required")
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[tuple[str, str], int] = {}
    text_labels: dict[str, int] = {}
    for index, row in enumerate(records):
        key = text_key(row["text"])
        if key in text_labels and text_labels[key] != row["label"]:
            raise ValueError(
                "Duplicate router texts have conflicting labels; audit the annotations"
            )
        text_labels[key] = row["label"]
        keys = [("text", key)]
        if row.get("group_id") is not None:
            keys.append(("group", str(row["group_id"])))
        for group_key in keys:
            if group_key in owners:
                parents[root(index)] = root(owners[group_key])
            else:
                owners[group_key] = index
    groups = [root(index) for index in range(len(records))]
    if len(set(groups)) < 2:
        raise ValueError("Router splitting requires at least two independent source groups")
    labels = {row["label"] for row in records}
    splitter = GroupShuffleSplit(n_splits=SPLIT_CANDIDATES, test_size=test_size, random_state=seed)
    best: tuple[float, list[int], list[int]] | None = None
    for train_ids, test_ids in splitter.split(records, groups=groups):
        if any({records[int(i)]["label"] for i in ids} != labels for ids in (train_ids, test_ids)):
            continue
        size_error = abs(len(test_ids) / len(records) - test_size)
        balance_error = sum(
            abs(
                sum(records[int(i)]["label"] == label for i in test_ids) / len(test_ids)
                - sum(row["label"] == label for row in records) / len(records)
            )
            for label in labels
        )
        score = size_error + balance_error
        if best is None or score < best[0]:
            best = (score, train_ids.tolist(), test_ids.tolist())
    if best is None:
        raise ValueError(
            "Cannot preserve every class in both group-disjoint splits; "
            "add independent sources per class or adjust split sizes"
        )
    train, test = ([records[i] for i in indices] for indices in best[1:])
    assert_disjoint(train, test)
    if any(row.get("group_id") is None for row in records):
        logger.warning("Missing source group_id: only exact duplicate leakage can be checked")
    return train, test


def load_router_records(data_path: str | Path) -> list[dict]:
    """Read labeled JSONL records, preserving caller-supplied metadata."""
    records = []
    with Path(data_path).open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("text"), str)
                or not record["text"].strip()
                or "label" not in record
            ):
                raise ValueError(f"Expected nonempty text and label at line {line_number}")
            records.append(record)
    return records


def load_router_dataset(
    data_path: str = "augmented_yelp.jsonl",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple["Dataset", "Dataset"]:
    """Load JSONL dataset and split into train/test.

    Labels are integer IDs in the configured task's ordered vocabulary.

    Args:
        data_path: Path to the JSONL file.
        test_size: Fraction of data for test split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    from datasets import Dataset

    train, test = split_router_records(load_router_records(data_path), test_size, seed)
    logger.info(f"Loaded router dataset from {data_path}: {len(train)} train, {len(test)} test")
    return Dataset.from_list(train), Dataset.from_list(test)


def save_dataset_to_jsonl(
    dataset: list[dict],
    output_path: str,
) -> Path:
    """Save a list of utterance dicts to JSONL format.

    Args:
        dataset: List of {"text": str, "label": int} dicts.
        output_path: Path to write the JSONL file.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(dataset)} utterances to {path}")
    return path
