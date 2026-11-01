# Enforce types at schema/config boundaries while legacy modules migrate.
# pyright: reportArgumentType=error, reportReturnType=error, reportCallIssue=error
# pyright: reportAttributeAccessIssue=error, reportOptionalMemberAccess=error
# pyright: reportGeneralTypeIssues=error, reportMissingImports=error

"""Configuration for the Ray Serve deployment.

Loads defaults from service.yaml, with environment variables
taking priority. Use ``load_serve_config()`` to load with the
standard precedence: YAML defaults < env vars.

Environment variables:
    SENTIMENTIZER_SERVE_CONFIG  -- Path to custom YAML config file
    SENTIMENTIZER_SERVE_HOST -- Override serve_host
    SENTIMENTIZER_SERVE_PORT -- Override serve_port
    SENTIMENTIZER_RAY_OBJECT_STORE_MEMORY_MB -- Override ray_object_store_memory_mb
    SENTIMENTIZER_DEFAULT_MODEL -- Override default_model
    SENTIMENTIZER_MAX_BATCH_SIZE -- Override max_batch_size
    SENTIMENTIZER_MAX_TEXT_LENGTH -- Override max_text_length
    SENTIMENTIZER_PREDICT_BATCH_SIZE -- Override predict_batch_size
    SENTIMENTIZER_PREDICT_BATCH_WAIT_S -- Override predict_batch_wait_s
    SENTIMENTIZER_CLASSIFY_BATCH_SIZE -- Override classify_batch_size
    SENTIMENTIZER_CLASSIFY_BATCH_WAIT_S -- Override classify_batch_wait_s
    ROUTER_MODEL_PATH -- Override router_model_path
    SENTIMENTIZER_CORS_ORIGINS -- Override cors_origins (comma-separated)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import ConfigDict, TypeAdapter

from sentimentizer import logger


@dataclass(frozen=True)
class ServeConfig:
    """Configuration for the Sentimentizer Ray Serve deployment.

    Attributes:
        serve_host: Address used by the Ray Serve HTTP proxy.
        serve_port: Port used by the Ray Serve HTTP proxy.
        ray_object_store_memory_mb: Ray object-store allocation in MiB.
        default_model: Default sentiment model to load (rnn, encoder, decoder).
        router_model_path: Path to trained router model directory.
        max_batch_size: Maximum texts per /v1/batch request.
        max_text_length: Maximum characters per text input.
        predict_batch_size: Maximum requests collected per /v1/predict batch.
        predict_batch_wait_s: Seconds to wait before processing a partial batch.
        classify_batch_size: Maximum requests collected per /v1/router/predict batch.
        classify_batch_wait_s: Seconds to wait before processing a partial router batch.
        cors_origins: Allowed CORS origins (comma-separated for env var).
        request_timeout_s: HTTP request timeout in seconds.
        bge_m3_num_replicas: Number of BGE-M3 Ray Serve replicas.
        bge_m3_max_ongoing_requests: Maximum concurrent requests per BGE-M3 replica.
        bge_m3_num_cpus: Ray CPU reservation per BGE-M3 replica.
        bge_m3_num_gpus: Ray GPU reservation per BGE-M3 replica.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    serve_host: str = "0.0.0.0"
    serve_port: int = 8000
    ray_object_store_memory_mb: int = 384
    default_model: str = "encoder"
    router_model_path: str = "models/router"
    max_batch_size: int = 64
    max_text_length: int = 10000
    predict_batch_size: int = 32
    predict_batch_wait_s: float = 0.05
    classify_batch_size: int = 32
    classify_batch_wait_s: float = 0.05
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    request_timeout_s: int = 600
    embeddings_enabled: bool = False
    embeddings_device: str = "auto"
    dense_embedding_model_id: str = "nomic-ai/nomic-embed-text-v1.5"
    dense_embedding_revision: str = "161a1515eaa28482dfe6ceae96cdefdd20dabcbd"
    bge_m3_enabled: bool = False
    bge_m3_model_id: str = "BAAI/bge-m3"
    bge_m3_use_fp16: bool = False
    bge_m3_batch_size: int = 8
    bge_m3_batch_wait_s: float = 0.01
    bge_m3_num_replicas: int = 1
    bge_m3_max_ongoing_requests: int = 64
    bge_m3_num_cpus: float = 2.0
    bge_m3_num_gpus: float = 0.0

    def __post_init__(self) -> None:
        """Validate that numeric fields are positive."""
        _positive_ints = {
            "serve_port": self.serve_port,
            "ray_object_store_memory_mb": self.ray_object_store_memory_mb,
            "max_batch_size": self.max_batch_size,
            "max_text_length": self.max_text_length,
            "predict_batch_size": self.predict_batch_size,
            "classify_batch_size": self.classify_batch_size,
            "bge_m3_batch_size": self.bge_m3_batch_size,
            "bge_m3_num_replicas": self.bge_m3_num_replicas,
            "bge_m3_max_ongoing_requests": self.bge_m3_max_ongoing_requests,
        }
        _positive_floats = {
            "predict_batch_wait_s": self.predict_batch_wait_s,
            "classify_batch_wait_s": self.classify_batch_wait_s,
            "bge_m3_batch_wait_s": self.bge_m3_batch_wait_s,
        }
        for name, val in _positive_ints.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        for name, val in _positive_floats.items():
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if not self.serve_host.strip():
            raise ValueError("serve_host must not be empty")
        if self.serve_port > 65535:
            raise ValueError(f"serve_port must be <= 65535, got {self.serve_port}")
        for name, val in {
            "bge_m3_num_cpus": self.bge_m3_num_cpus,
            "bge_m3_num_gpus": self.bge_m3_num_gpus,
        }.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean, accepting common truthy/falsy values.

    Raises ValueError for unrecognized strings (e.g. "maybe").
    This avoids the Python footgun where ``bool("false")`` is ``True``.
    """
    lower = value.lower().strip()
    if lower in _TRUTHY:
        return True
    if lower in _FALSY:
        return False
    raise ValueError(f"Cannot interpret {value!r} as boolean")


def _parse_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


# Mapping from env var name to ServeConfig field name
_ENV_OVERRIDES: dict[str, str] = {
    "SENTIMENTIZER_SERVE_HOST": "serve_host",
    "SENTIMENTIZER_SERVE_PORT": "serve_port",
    "SENTIMENTIZER_RAY_OBJECT_STORE_MEMORY_MB": "ray_object_store_memory_mb",
    "SENTIMENTIZER_DEFAULT_MODEL": "default_model",
    "ROUTER_MODEL_PATH": "router_model_path",
    "SENTIMENTIZER_MAX_BATCH_SIZE": "max_batch_size",
    "SENTIMENTIZER_MAX_TEXT_LENGTH": "max_text_length",
    "SENTIMENTIZER_PREDICT_BATCH_SIZE": "predict_batch_size",
    "SENTIMENTIZER_PREDICT_BATCH_WAIT_S": "predict_batch_wait_s",
    "SENTIMENTIZER_CLASSIFY_BATCH_SIZE": "classify_batch_size",
    "SENTIMENTIZER_CLASSIFY_BATCH_WAIT_S": "classify_batch_wait_s",
    "SENTIMENTIZER_CORS_ORIGINS": "cors_origins",
    "SENTIMENTIZER_REQUEST_TIMEOUT_S": "request_timeout_s",
    "SENTIMENTIZER_EMBEDDINGS_ENABLED": "embeddings_enabled",
    "SENTIMENTIZER_EMBEDDINGS_DEVICE": "embeddings_device",
    "SENTIMENTIZER_DENSE_EMBEDDING_MODEL_ID": "dense_embedding_model_id",
    "SENTIMENTIZER_DENSE_EMBEDDING_REVISION": "dense_embedding_revision",
    "SENTIMENTIZER_BGE_M3_ENABLED": "bge_m3_enabled",
    "SENTIMENTIZER_BGE_M3_MODEL_ID": "bge_m3_model_id",
    "SENTIMENTIZER_BGE_M3_USE_FP16": "bge_m3_use_fp16",
    "SENTIMENTIZER_BGE_M3_BATCH_SIZE": "bge_m3_batch_size",
    "SENTIMENTIZER_BGE_M3_BATCH_WAIT_S": "bge_m3_batch_wait_s",
    "SENTIMENTIZER_BGE_M3_NUM_REPLICAS": "bge_m3_num_replicas",
    "SENTIMENTIZER_BGE_M3_MAX_ONGOING_REQUESTS": "bge_m3_max_ongoing_requests",
    "SENTIMENTIZER_BGE_M3_NUM_CPUS": "bge_m3_num_cpus",
    "SENTIMENTIZER_BGE_M3_NUM_GPUS": "bge_m3_num_gpus",
}

# Type coercion for non-string fields
# list[str] fields use _parse_list for comma-separated env var parsing
_FIELD_TYPES: dict[str, Callable[[str], object]] = {
    "serve_port": int,
    "ray_object_store_memory_mb": int,
    "max_batch_size": int,
    "max_text_length": int,
    "predict_batch_size": int,
    "predict_batch_wait_s": float,
    "classify_batch_size": int,
    "classify_batch_wait_s": float,
    "cors_origins": list,
    "request_timeout_s": int,
    "embeddings_enabled": _parse_bool,
    "bge_m3_enabled": _parse_bool,
    "bge_m3_use_fp16": _parse_bool,
    "bge_m3_batch_size": int,
    "bge_m3_batch_wait_s": float,
    "bge_m3_num_replicas": int,
    "bge_m3_max_ongoing_requests": int,
    "bge_m3_num_cpus": float,
    "bge_m3_num_gpus": float,
}


# Image generation moved to the imagenate service. Keep warning for one release so a
# stale deployment does not lose images silently.
_REMOVED_IMAGE_FIELDS = frozenset(
    {
        "krea_2_enabled",
        "krea_2_license_accepted",
        "ideogram_4_enabled",
        "ideogram_4_license_accepted",
        "comfyui_base_url",
        "comfyui_timeout_s",
        "comfyui_poll_interval_s",
        "comfyui_temp_directory",
        "image_moderation_url",
        "image_moderation_api_key",
        "image_moderation_timeout_s",
        "default_image_model",
        "api_keys",
        "rate_limit_per_min",
        "rate_limit_burst",
        "idempotency_ttl_s",
        "job_ttl_s",
        "prompt_blocklist_path",
    }
)
_REMOVED_IMAGE_ENV_PREFIXES = (
    "SENTIMENTIZER_DIFFUSION_KREA_2_",
    "SENTIMENTIZER_DIFFUSION_IDEOGRAM_4_",
)


def _warn_removed_image_settings(yaml_keys: set[str]) -> None:
    """Log one warning per image setting that this service no longer reads."""
    removed_env = {f"SENTIMENTIZER_{name.upper()}" for name in _REMOVED_IMAGE_FIELDS}
    for name in sorted(os.environ):
        if name in removed_env or name.startswith(_REMOVED_IMAGE_ENV_PREFIXES):
            logger.warning("image_setting_ignored_moved_to_imagenate", env_var=name)
    for key in sorted(yaml_keys):
        logger.warning("image_setting_ignored_moved_to_imagenate", yaml_key=key)


def _default_config_path() -> Path:
    """Return path to the default service.yaml bundled with the package."""
    return Path(__file__).parent / "service.yaml"


_CONFIG_ADAPTER = TypeAdapter(ServeConfig)


def load_serve_config(path: str | Path | None = None) -> ServeConfig:
    """Load serve configuration from YAML with environment variable overrides.

    Precedence (highest wins):
        1. Environment variables (SENTIMENTIZER_DEFAULT_MODEL, etc.)
        2. YAML config file values
        3. Dataclass defaults

    Args:
        path: Path to YAML config file. If None, checks the
              SENTIMENTIZER_SERVE_CONFIG env var, then falls back to
              the bundled service.yaml.

    Returns:
        A ServeConfig instance with resolved values.
    """
    if path is None:
        env_path = os.environ.get("SENTIMENTIZER_SERVE_CONFIG")
        path = Path(env_path) if env_path else _default_config_path()
    else:
        path = Path(path)

    # Start with dataclass defaults
    values: dict[str, object] = {}

    # Layer 1: YAML file overrides (if it exists)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("Serve configuration must be a YAML mapping")
        values.update(raw or {})
    stale_yaml_keys = values.keys() & _REMOVED_IMAGE_FIELDS
    for key in stale_yaml_keys:
        del values[key]
    _warn_removed_image_settings(set(stale_yaml_keys))

    # Layer 2: Environment variable overrides
    for env_var, field_name in _ENV_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            coerce = _FIELD_TYPES.get(field_name, str)
            if coerce is list:
                values[field_name] = _parse_list(env_value)
            else:
                try:
                    values[field_name] = coerce(env_value)
                except ValueError:
                    raise ValueError(
                        f"Invalid {env_var}={env_value!r}: expected {coerce.__name__}"
                    ) from None

    return _CONFIG_ADAPTER.validate_python(values)


# Global singleton instance loaded once at import time
cfg = load_serve_config()
