"""Dataset loading and train/test split for the router.

Loads JSONL files with text and integer task-label IDs
and splits into train/test datasets for router training.
"""

import hashlib
import json
import logging
import unicodedata
from numbers import Integral
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
            {
                hashlib.sha256(str(row["group_id"]).encode()).hexdigest()
                for row in records
                if row.get("group_id") is not None
            }
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
        (
            text_key(row["text"]),
            int(row["label"]) if isinstance(row["label"], Integral) else row["label"],
            str(row["group_id"]) if row.get("group_id") is not None else "",
        )
        for row in records
    )
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def records_to_columns(records: list[dict]) -> dict[str, list]:
    """Preserve every optional field, filling missing entries with nulls."""
    keys = sorted({key for row in records for key in row})
    return {key: [row.get(key) for row in records] for key in keys}


def records_to_dataset(records: list[dict]) -> "Dataset":
    """Build a dataset without allowing the first row to define its columns."""
    from datasets import Dataset

    return Dataset.from_dict(records_to_columns(records))


def _label_aware_split(
    groups: list[int], labels: list[int], test_size: float, seed: int
) -> tuple[list[int], list[int]]:
    """Find a class-complete group assignment when random candidates miss one.

    Binary variables assign whole connected groups to test. Coverage constraints
    require at least one group for each class on each side; slack variables
    minimize deviation from the requested row and per-class counts. Solver
    failure is distinguished from a proof that the constraints are infeasible.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, vstack

    group_ids = np.random.default_rng(seed).permutation(sorted(set(groups))).tolist()
    group_columns = {group: column for column, group in enumerate(group_ids)}
    label_rows = {label: row for row, label in enumerate(sorted(set(labels)))}
    num_groups, num_labels = len(group_ids), len(label_rows)
    counts = coo_matrix(
        (
            np.ones(len(labels)),
            ([label_rows[label] for label in labels], [group_columns[group] for group in groups]),
        ),
        shape=(num_labels, num_groups),
    ).tocsr()
    support = counts.copy()
    support.data[:] = 1
    class_sizes = np.asarray(counts.sum(axis=1)).ravel()
    group_sizes = np.asarray(counts.sum(axis=0)).ravel()
    totals = np.r_[len(labels), class_sizes]
    count_rows = vstack([csr_matrix(group_sizes[None, :]), counts])
    slack = eye(num_labels + 1)
    matrix = vstack(
        [
            hstack([support, csr_matrix((num_labels, num_labels + 1))]),
            hstack([count_rows, -slack]),
            hstack([-count_rows, -slack]),
        ]
    )
    targets = test_size * totals
    result = milp(
        # Prioritize the requested number of rows; class balance breaks ties.
        c=np.r_[np.zeros(num_groups), 1.0, 1.0 / (len(labels) * class_sizes)],
        integrality=np.r_[np.ones(num_groups), np.zeros(num_labels + 1)],
        bounds=Bounds(
            np.zeros(num_groups + num_labels + 1),
            np.r_[np.ones(num_groups), np.full(num_labels + 1, np.inf)],
        ),
        constraints=LinearConstraint(
            matrix,
            np.r_[np.ones(num_labels), np.full(2 * (num_labels + 1), -np.inf)],
            np.r_[np.asarray(support.sum(axis=1)).ravel() - 1, targets, -targets],
        ),
    )
    if result.status == 2:
        raise ValueError(
            "Cannot preserve every class in both group-disjoint splits; "
            "source-group constraints are infeasible"
        )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Router grouped split solver failed (status={result.status}); feasibility is unknown"
        )
    selected = {
        group for group, value in zip(group_ids, result.x[:num_groups], strict=True) if value >= 0.5
    }
    train = [index for index, group in enumerate(groups) if group not in selected]
    test = [index for index, group in enumerate(groups) if group in selected]
    if any({labels[index] for index in indices} != set(labels) for indices in (train, test)):
        raise RuntimeError("Router grouped split solver returned an invalid class assignment")
    return train, test


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
        logger.info("Random candidates missed class coverage; solving source-group constraints")
        indices = _label_aware_split(groups, [row["label"] for row in records], test_size, seed)
    else:
        indices = best[1:]
    train, test = ([records[i] for i in partition] for partition in indices)
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
    train, test = split_router_records(load_router_records(data_path), test_size, seed)
    logger.info(f"Loaded router dataset from {data_path}: {len(train)} train, {len(test)} test")
    return records_to_dataset(train), records_to_dataset(test)


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
