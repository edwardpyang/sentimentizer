"""Capture experiment inputs without retaining raw training reviews in metadata."""

from contextlib import suppress
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from sentimentizer.router.config import RouterConfig
from sentimentizer.router.dataset import dataset_fingerprint, record_keys
from sentimentizer.router.schemas import RouterProvenance

RECORDED_LIBRARIES = ("sentimentizer", "torch", "sentence-transformers", "scikit-learn", "numpy")


def library_versions() -> dict[str, str]:
    """Record installed versions without importing optional heavy libraries."""
    versions = {}
    for name in RECORDED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def dataset_records(dataset: Any) -> list[dict]:
    """Retain source lineage for dict-of-columns and Hugging Face datasets."""
    if isinstance(dataset, dict):
        return [
            dict(zip(dataset, values, strict=True))
            for values in zip(*dataset.values(), strict=True)
        ]
    return [dict(row) for row in dataset]


def add_dataset(provenance: RouterProvenance, name: str, records: list[dict]) -> None:
    """Record all data influencing an artifact, including calibration/policy selection."""
    provenance.dataset_fingerprints[name] = dataset_fingerprint(records)
    for key, values in record_keys(records).items():
        provenance.used_keys[key] = sorted(set(provenance.used_keys.get(key, [])) | set(values))


def training_provenance(
    config: RouterConfig, records: list[dict], backbone: Any
) -> RouterProvenance:
    """Describe a training run with its actual resolved model revision when available."""
    configuration = asdict(config)
    configuration["task"] = config.task.model_dump(mode="json")
    configuration["output_dir"] = str(config.output_dir)
    revision = None
    with suppress(AttributeError, IndexError, KeyError, TypeError):
        revision = backbone[0].auto_model.config._commit_hash
    provenance = RouterProvenance(
        training_config=configuration,
        library_versions=library_versions(),
        base_model_revision=revision if isinstance(revision, str) else None,
    )
    add_dataset(provenance, "train", records)
    return provenance
