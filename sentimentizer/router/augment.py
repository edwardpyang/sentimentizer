"""Augment task-specific seed utterances using GLM 5.1 via Ollama API.

Generates hard negatives: text that sounds like one category but
belongs to another (e.g., "The bread was so dry" → General, not Dietary).
"""

import json
import logging
import time

import requests

from sentimentizer.router.config import FOOD_TASK_SPEC
from sentimentizer.router.schemas import RouterTaskSpec

logger = logging.getLogger(__name__)

AUGMENT_REQUEST_TIMEOUT_S = 120
AUGMENT_PAUSE_S = 0.5


def _build_prompt(
    seed_text: str, seed_label: int, num_variations: int, *, task: RouterTaskSpec | None = None
) -> str:
    """Build the augmentation prompt for GLM 5.1.

    Loads the prompt template from sentimentizer/prompts/augment_router.txt
    and fills in the template variables.

    Args:
        seed_text: Original utterance text.
        seed_label: Original class ID in the task's ordered vocabulary.
        num_variations: Number of variations to generate.
        task: Label definitions; defaults to the shipped food task.

    Returns:
        Formatted prompt string for the Ollama API.
    """
    from sentimentizer.prompts import load_prompt

    task = task or FOOD_TASK_SPEC
    seed_category = task.labels[task.label_id(seed_label)]
    other_categories = [label for label in task.labels if label != seed_category]
    template = load_prompt("augment_router")
    return template.format(
        num_variations=num_variations,
        seed_text=seed_text,
        seed_category=seed_category,
        task_description=task.description,
        label_descriptions=json.dumps(task.label_descriptions),
        other_categories=json.dumps(other_categories),
        seed_label=seed_label,
    )


def _load_existing_entries(output_path: str) -> list[dict]:
    """Load existing entries from a JSONL file for resume.

    Args:
        output_path: Path to the existing JSONL file.

    Returns:
        List of entry dicts from the file, or empty list if file doesn't exist.
    """
    from pathlib import Path

    path = Path(output_path)
    if not path.exists():
        return []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {output_path}: {line[:80]}")
    return entries


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
) -> list[dict]:
    """Expand seed utterances using GLM 5.1 via Ollama API.

    Generates hard negatives: text that sounds like one category but
    belongs to another. For example, "The bread was so dry" sounds
    like a General complaint but could be labeled Dietary if it's
    about gluten-free bread.

    When output_path is provided, each entry is written to the JSONL
    file immediately as it's generated (streaming). This allows you
    to see progress in real-time and avoids losing data if the process
    is interrupted. Seed utterances are written first, then augmented
    entries are appended as each Ollama API call completes.

    When resume=True and output_path points to an existing file, seeds
    that already appear in the file are skipped. Only unprocessed seeds
    are sent to the Ollama API, and new entries are appended to the file.

    Args:
        seeds: List of {"text": str, "label": int} seed utterances.
        model: Ollama model name.
        ollama_url: Ollama API endpoint.
        variations_per_seed: Number of variations to generate per seed.
        batch_size: Number of seeds to process in one API call.
        output_path: If provided, stream each entry to this JSONL file
            as it's generated. Seed utterances are written first.
        resume: If True and output_path exists, skip seeds already in
            the file and append new entries to the existing file.

    Returns:
        Original seeds + augmented utterances with the same format.
    """
    from pathlib import Path

    task = task or FOOD_TASK_SPEC
    for seed in seeds:
        task.label_id(seed["label"])
        if not isinstance(seed.get("text"), str) or not seed["text"].strip():
            raise ValueError("Augmentation seeds require nonempty text")

    # Resume mode: load existing entries and determine which seeds to process
    if resume and output_path and Path(output_path).exists():
        existing = _load_existing_entries(output_path)
        existing_texts = {e["text"] for e in existing if "text" in e}
        seeds_to_process = [s for s in seeds if s["text"] not in existing_texts]
        skipped = len(seeds) - len(seeds_to_process)
        logger.info(
            f"Resume mode: {len(existing)} existing entries, "
            f"skipping {skipped} seeds, processing {len(seeds_to_process)} remaining"
        )
        if not seeds_to_process:
            logger.info("All seeds already processed — nothing to do")
            return existing
        augmented = existing
        seeds_for_augment = seeds_to_process
    else:
        augmented = list(seeds)  # start with originals
        seeds_for_augment = seeds

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if resume and path.exists():
            # Append to existing file
            with open(path, "a") as out_file:
                augmented = _augment_loop(
                    seeds_for_augment,
                    augmented,
                    model,
                    ollama_url,
                    variations_per_seed,
                    batch_size,
                    out_file,
                    task=task,
                )
        else:
            # Write new file (seed utterances first)
            with open(path, "w") as out_file:
                for entry in seeds:
                    out_file.write(json.dumps(entry) + "\n")
                out_file.flush()
                logger.info(f"Streaming augmentation to {path} ({len(seeds)} seeds written)")
                augmented = _augment_loop(
                    seeds_for_augment,
                    augmented,
                    model,
                    ollama_url,
                    variations_per_seed,
                    batch_size,
                    out_file,
                    task=task,
                )
    else:
        augmented = _augment_loop(
            seeds_for_augment,
            augmented,
            model,
            ollama_url,
            variations_per_seed,
            batch_size,
            task=task,
        )

    logger.info(f"Augmented {len(seeds)} seeds → {len(augmented)} total utterances")
    if output_path:
        logger.info(f"Saved to {output_path}")
    return augmented


def _augment_loop(
    seeds: list[dict],
    augmented: list[dict],
    model: str,
    ollama_url: str,
    variations_per_seed: int,
    batch_size: int,
    out_file: object | None = None,
    *,
    task: RouterTaskSpec | None = None,
) -> list[dict]:
    """Inner loop for augmentation, optionally streaming to an open file.

    Args:
        seeds: List of seed utterances to process.
        augmented: Accumulator list (starts with existing entries).
        model: Ollama model name.
        ollama_url: Ollama API endpoint.
        variations_per_seed: Number of variations per seed.
        batch_size: Number of seeds per batch.
        out_file: Open file handle for streaming writes (or None).

    Returns:
        Updated accumulator list with augmented entries.
    """
    task = task or FOOD_TASK_SPEC
    for i in range(0, len(seeds), batch_size):
        batch = seeds[i : i + batch_size]
        for seed in batch:
            prompt = _build_prompt(seed["text"], seed["label"], variations_per_seed, task=task)
            try:
                response = requests.post(
                    ollama_url,
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=AUGMENT_REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
                result = response.json()
                text = result.get("response", "")
                for line in text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if (
                            isinstance(entry, dict)
                            and isinstance(entry.get("text"), str)
                            and entry["text"].strip()
                            and task.label_id(entry.get("label")) == seed["label"]
                        ):
                            entry.update(
                                {k: v for k, v in seed.items() if k not in {"text", "label"}}
                            )
                            augmented.append(entry)
                            if out_file:
                                out_file.write(json.dumps(entry) + "\n")
                    except (json.JSONDecodeError, ValueError):
                        logger.warning("Skipped invalid augmentation entry")
                if out_file:
                    out_file.flush()
            except requests.RequestException as e:
                logger.error(f"Ollama API request failed: {e}")
            time.sleep(AUGMENT_PAUSE_S)
    return augmented
