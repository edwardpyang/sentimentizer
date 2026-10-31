"""Generate auditable seed variations with recoverable, counted progress."""

import hashlib
import json
import logging
import time
from pathlib import Path

import requests

from sentimentizer.router.config import FOOD_TASK_SPEC
from sentimentizer.router.dataset import text_key
from sentimentizer.router.schemas import RouterTaskSpec

logger = logging.getLogger(__name__)

AUGMENT_REQUEST_TIMEOUT_S = 120
AUGMENT_PAUSE_S = 0.5
AUGMENT_MAX_ATTEMPTS = 3


def _build_prompt(
    seed_text: str, seed_label: int, num_variations: int, *, task: RouterTaskSpec | None = None
) -> str:
    """Build a task-specific prompt with a single required source label."""
    from sentimentizer.prompts import load_prompt

    task = task or FOOD_TASK_SPEC
    category = task.labels[task.label_id(seed_label)]
    return load_prompt("augment_router").format(
        num_variations=num_variations,
        seed_text=seed_text,
        seed_category=category,
        task_description=task.description,
        label_descriptions=json.dumps(task.label_descriptions),
        other_categories=json.dumps([label for label in task.labels if label != category]),
        seed_label=seed_label,
    )


def prepare_seeds(seeds: list[dict], task: RouterTaskSpec) -> list[dict]:
    """Attach stable seed and source identities, preserving caller metadata."""
    prepared = []
    seen: set[str] = set()
    for seed in seeds:
        label = task.label_id(seed.get("label"))
        if not isinstance(seed.get("text"), str) or not seed["text"].strip():
            raise ValueError("Augmentation seeds require nonempty text")
        identity = json.dumps([task.task_id, task.taxonomy_version, label, text_key(seed["text"])])
        seed_id = hashlib.sha256(identity.encode()).hexdigest()
        if seed_id in seen:
            continue
        seen.add(seed_id)
        prepared.append(
            {**seed, "seed_id": seed_id, "group_id": seed.get("group_id", seed_id), "is_seed": True}
        )
    return prepared


def augmentation_counts(records: list[dict]) -> dict[str, int]:
    """Count accepted variations rather than treating original seeds as progress."""
    counts: dict[str, int] = {}
    for row in records:
        if row.get("is_seed") is False and isinstance(row.get("seed_id"), str):
            counts[row["seed_id"]] = counts.get(row["seed_id"], 0) + 1
    return counts


def _load_existing_entries(output_path: str) -> list[dict]:
    """Recover a truncated final write; reject corruption in complete records."""
    path = Path(output_path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                logger.warning("Discarding truncated final augmentation record")
                break
            raise ValueError(f"Malformed augmentation record at line {index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Invalid augmentation record at line {index + 1}")
        rows.append(row)
    return rows


def _resume_records(existing: list[dict], seeds: list[dict], task: RouterTaskSpec) -> list[dict]:
    """Validate lineage before resuming, including old files containing only seeds."""
    by_id = {seed["seed_id"]: seed for seed in seeds}
    by_text = {text_key(seed["text"]): seed for seed in seeds}
    rows = []
    seen: dict[str, int] = {}
    for row in existing:
        task.label_id(row.get("label"))
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Invalid text in augmentation resume file")
        key = text_key(row["text"])
        if key in seen:
            if seen[key] != row["label"]:
                raise ValueError("Conflicting labels in augmentation resume file")
            continue
        seen[key] = row["label"]
        if row.get("seed_id") is None and key in by_text:
            row = by_text[key]
        parent = by_id.get(row.get("seed_id"))
        if parent is None or row.get("is_seed") not in (True, False):
            raise ValueError("Cannot resume legacy variations without seed lineage; regenerate")
        if row["label"] != parent["label"] or row.get("group_id") != parent["group_id"]:
            raise ValueError("Resume record conflicts with its seed label or source group")
        if row["is_seed"] and key != text_key(parent["text"]):
            raise ValueError("Resume seed identity conflicts with its text")
        rows.append(row)
    seen_seeds = {row["seed_id"] for row in rows if row["is_seed"]}
    return rows + [seed for seed in seeds if seed["seed_id"] not in seen_seeds]


def augment_seeds(
    seeds: list[dict],
    model: str = "glm-5.1:cloud",
    ollama_url: str = "http://localhost:11434/api/generate",
    variations_per_seed: int = 50,
    batch_size: int = 5,
    output_path: str | None = None,
    resume: bool = False,
    *,
    task: RouterTaskSpec | None = None,
    max_attempts: int = AUGMENT_MAX_ATTEMPTS,
) -> list[dict]:
    """Generate unique, label-validated variations and retain partial progress.

    Each seed is attempted at most max_attempts times per invocation. Resume
    requests only the missing count. Callers must inspect counts before training;
    the CLI fails incomplete jobs while leaving their output recoverable.
    """
    task = task or FOOD_TASK_SPEC
    if variations_per_seed < 1 or batch_size < 1 or max_attempts < 1:
        raise ValueError("Variation count, batch size and max_attempts must be positive")
    seeds = prepare_seeds(seeds, task)
    rows = (
        _resume_records(_load_existing_entries(output_path), seeds, task)
        if resume and output_path
        else list(seeds)
    )
    seen = {text_key(row["text"]) for row in rows}
    counts = augmentation_counts(rows)
    path = Path(output_path) if output_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace the recovered snapshot atomically before appending new records.
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as snapshot:
            temporary = Path(snapshot.name)
            try:
                for row in rows:
                    snapshot.write(json.dumps(row) + "\n")
                snapshot.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    for seed in seeds:
        seed_id = seed["seed_id"]
        for attempt in range(max_attempts):
            remaining = variations_per_seed - counts.get(seed_id, 0)
            if remaining <= 0:
                break
            start = time.monotonic()
            logger.info(
                "Augmenting seed",
                extra={"seed_id": seed_id, "remaining": remaining, "attempt": attempt + 1},
            )
            try:
                response = requests.post(
                    ollama_url,
                    json={
                        "model": model,
                        "prompt": _build_prompt(seed["text"], seed["label"], remaining, task=task),
                        "stream": False,
                    },
                    timeout=AUGMENT_REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
                payload = response.json().get("response", "")
                if not isinstance(payload, str):
                    raise ValueError("Augmentation response must contain text")
                for line in payload.splitlines():
                    if counts.get(seed_id, 0) >= variations_per_seed:
                        break
                    try:
                        entry = json.loads(line)
                        if (
                            not isinstance(entry, dict)
                            or task.label_id(entry.get("label")) != seed["label"]
                        ):
                            raise ValueError("Variation label does not match its source")
                        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
                            raise ValueError("Variation text is empty")
                    except (json.JSONDecodeError, ValueError):
                        continue
                    key = text_key(entry["text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    entry = {
                        **seed,
                        "text": entry["text"],
                        "is_seed": False,
                        "augmentation_model": model,
                    }
                    rows.append(entry)
                    counts[seed_id] = counts.get(seed_id, 0) + 1
                    if path:
                        with path.open("a") as target:
                            target.write(json.dumps(entry) + "\n")
            except (requests.RequestException, ValueError) as exc:
                logger.warning(
                    "Augmentation attempt failed",
                    extra={"seed_id": seed_id, "error_type": type(exc).__name__},
                )
            logger.info(
                "Augmentation attempt complete",
                extra={
                    "seed_id": seed_id,
                    "accepted": counts.get(seed_id, 0),
                    "duration_s": time.monotonic() - start,
                },
            )
            if counts.get(seed_id, 0) < variations_per_seed and attempt + 1 < max_attempts:
                time.sleep(AUGMENT_PAUSE_S)
    incomplete = sum(counts.get(seed["seed_id"], 0) < variations_per_seed for seed in seeds)
    if incomplete:
        logger.warning(
            "Augmentation incomplete; resume to retry shortfalls",
            extra={"incomplete_seeds": incomplete},
        )
    return rows
