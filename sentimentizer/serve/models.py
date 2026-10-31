from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentimentizer.router.schemas import RouterClassification
from sentimentizer.serve.config import cfg


class PredictRequest(BaseModel):
    """Single text sentiment prediction request."""

    text: Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]
    model: str | None = Field(
        default=None,
        description="Model name to use for prediction. "
        "If omitted, uses the default model. "
        "Returns 400 if the requested model is not loaded.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "The food was terrific!"},
                {"text": "Terrible service, would not recommend."},
            ]
        }
    }


class BatchRequest(BaseModel):
    """Multiple text sentiment prediction request."""

    texts: list[Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]] = Field(
        ..., min_length=1, max_length=cfg.max_batch_size
    )
    model: str | None = Field(
        default=None,
        description="Model name to use for prediction. "
        "If omitted, uses the default model. "
        "Returns 400 if the requested model is not loaded.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": [
                        "Great food!",
                        "Terrible service.",
                    ],
                },
            ]
        }
    }


class TokenizeRequest(BaseModel):
    """Tokenize text without running inference."""

    text: Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "The food was terrific!"},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response models for Swagger docs
# ---------------------------------------------------------------------------


class SentimentPrediction(BaseModel):
    """Single sentiment prediction in the response."""

    label: str = Field(..., description="Predicted sentiment label")
    score: float = Field(..., description="Confidence score for the predicted label")
    token_count: int = Field(..., description="Number of tokens in the input text")
    model: str = Field(..., description="Model name used for prediction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "positive",
                    "score": 0.92,
                    "token_count": 4,
                    "model": "encoder",
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Response from /v1/predict."""

    prediction: SentimentPrediction
    latency_s: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": {
                        "label": "positive",
                        "score": 0.92,
                        "token_count": 4,
                        "model": "encoder",
                    },
                    "latency_s": 0.0043,
                }
            ]
        }
    }


class BatchResultItem(BaseModel):
    """Single item in a batch response."""

    prediction: SentimentPrediction


class BatchResponse(BaseModel):
    """Response from /v1/batch."""

    results: list[BatchResultItem]
    count: int
    latency_s: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "prediction": {
                                "label": "positive",
                                "score": 0.89,
                                "token_count": 2,
                                "model": "encoder",
                            },
                        }
                    ],
                    "count": 1,
                    "latency_s": 0.0031,
                }
            ]
        }
    }


class TokenizeResponse(BaseModel):
    """Response from /v1/tokenize."""

    text: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "The food was terrific!",
                    "tokens": ["the", "food", "was", "terrific"],
                    "token_ids": [42, 156, 23, 789],
                    "token_count": 4,
                }
            ]
        }
    }


class ModelDetailResponse(BaseModel):
    """Response from /v1/models/{model_name}."""

    model: str
    info: dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "encoder",
                    "info": {
                        "architecture": "Transformer Encoder (CLS token)",
                        "device": "cpu",
                        "status": "loaded",
                    },
                }
            ]
        }
    }


class ModelsResponse(BaseModel):
    """Response from /v1/models."""

    models: dict[str, Any]
    default: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "models": {
                        "encoder": {
                            "architecture": "Transformer Encoder (CLS token)",
                            "device": "cpu",
                            "status": "loaded",
                        }
                    },
                    "default": "encoder",
                }
            ]
        }
    }


class _RouterResponseModel(BaseModel):
    """Reject unexpected fields in router response envelopes."""

    model_config = ConfigDict(extra="forbid")


class RouterPrediction(RouterClassification):
    """Single router prediction.

    Note: omits model field since there is only one router model.
    See /v1/router/models for router model metadata.
    """

    token_count: int = Field(strict=True, ge=0, description="Number of regex tokens in the input")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "dietary",
                    "score": 0.98,
                    "token_count": 5,
                }
            ]
        }
    }


class RouterPredictResponse(_RouterResponseModel):
    """Response from /v1/router/predict."""

    prediction: RouterPrediction
    latency_s: float = Field(strict=True, ge=0, allow_inf_nan=False)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": {
                        "label": "dietary",
                        "score": 0.98,
                        "token_count": 5,
                    },
                    "latency_s": 0.0051,
                }
            ]
        }
    }


class RouterBatchResultItem(_RouterResponseModel):
    """Single item in a router batch response."""

    prediction: RouterPrediction


class RouterBatchResponse(_RouterResponseModel):
    """Response from /v1/router/batch."""

    results: list[RouterBatchResultItem]
    count: int = Field(strict=True, ge=0)
    latency_s: float = Field(strict=True, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_count(self) -> Self:
        """Require count to reflect every serialized result."""
        if self.count != len(self.results):
            raise ValueError("Router batch count must match results")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "results": [
                        {
                            "prediction": {
                                "label": "dietary",
                                "score": 0.98,
                                "token_count": 5,
                            }
                        }
                    ],
                    "count": 1,
                    "latency_s": 0.0062,
                }
            ]
        }
    }


class RouterModelsResponse(_RouterResponseModel):
    """Response from /v1/router/models."""

    model_path: str
    categories: list[str]
    status: Literal["loaded", "not_loaded"]
    error: str | None = None
    task_id: str | None = None
    taxonomy_version: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_path": "models/router",
                    "categories": ["dietary", "service", "general"],
                    "status": "loaded",
                }
            ]
        }
    }


class HealthLiveResponse(BaseModel):
    """Response from /health/live."""

    status: Literal["live"] = "live"
    uptime_s: float

    model_config = {"json_schema_extra": {"examples": [{"status": "live", "uptime_s": 123.4}]}}


class ImageModelsStatus(BaseModel):
    """Status of image models within health ready response."""

    krea_2: Literal["enabled", "disabled"]
    ideogram_4: Literal["enabled", "disabled"]


class HealthReadyResponse(BaseModel):
    """Response from /health/ready."""

    status: Literal["ready", "not_ready"]
    device: str
    version: str
    uptime_s: float
    model_loaded: str
    router_loaded: bool
    router_error: str | None = None
    image_models: ImageModelsStatus | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ready",
                    "device": "cpu",
                    "version": "0.0.0",
                    "uptime_s": 123.4,
                    "model_loaded": "encoder",
                    "router_loaded": True,
                    "router_error": None,
                    "image_models": {
                        "krea_2": "enabled",
                        "ideogram_4": "disabled",
                    },
                }
            ]
        }
    }
