"""Dataset loading and train/test split for the router.

Loads JSONL files with text and integer task-label IDs
and splits into train/test datasets for router training.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


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
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path)
    split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    logger.info(
        f"Loaded router dataset from {data_path}: "
        f"{len(split['train'])} train, {len(split['test'])} test"
    )
    return split["train"], split["test"]


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
