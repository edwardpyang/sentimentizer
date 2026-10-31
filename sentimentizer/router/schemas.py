"""Shared task, artifact, and classification contracts for the router."""

from numbers import Integral
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonemptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class RouterTaskSpec(BaseModel):
    """Define an ordered label vocabulary and its input contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: NonemptyString
    taxonomy_version: NonemptyString = "1"
    description: str = ""
    labels: tuple[NonemptyString, ...] = Field(min_length=2)
    label_descriptions: dict[str, str] = Field(default_factory=dict)
    fallback_label: str | None = None
    input_format: NonemptyString = "text-v1"

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        """Reject ambiguous labels and references outside the vocabulary."""
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Task labels must be unique")
        if set(self.label_descriptions) - set(self.labels):
            raise ValueError("Label descriptions must reference task labels")
        if self.fallback_label is not None and self.fallback_label not in self.labels:
            raise ValueError("Fallback label must belong to task labels")
        return self

    def label_id(self, value: object) -> int:
        """Validate a dataset class ID before any integer coercion."""
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Class IDs must be integers, not booleans or fractional values")
        label_id = int(value)
        if not 0 <= label_id < len(self.labels):
            raise ValueError("Class ID is outside the task vocabulary")
        return label_id


class RouterClassification(BaseModel):
    """Represent a prediction independently of its HTTP envelope."""

    model_config = ConfigDict(extra="forbid")

    label: NonemptyString
    score: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)


class RouterArtifactMetadata(BaseModel):
    """Read current and legacy router sidecars without loading model weights."""

    model_type: str = "router"
    labels: list[NonemptyString] = Field(default_factory=list)
    head_type: str | None = None
    task: RouterTaskSpec | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        """Keep the legacy label list consistent with task metadata."""
        if self.task is not None:
            if not self.labels:
                self.labels = list(self.task.labels)
            elif tuple(self.labels) != self.task.labels:
                raise ValueError("Artifact labels conflict with task labels")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Artifact labels must be unique")
        return self
