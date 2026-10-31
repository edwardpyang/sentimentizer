"""Tests for the minimal BGE-M3 Ray Serve application."""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sentimentizer.serve.__main__ import build_parser, run
from sentimentizer.serve.bge_only_app import (
    BGEM3OnlyDeployment,
    bge_app,
    initialize_ray,
)
from sentimentizer.serve.embeddings_models import EmbeddingsRequest

_IngressWrapper = getattr(BGEM3OnlyDeployment, "func_or_class", BGEM3OnlyDeployment)
_Deployment = next(base for base in _IngressWrapper.__mro__ if "embeddings" in base.__dict__)


def _vector(value: float) -> dict[str, object]:
    return {
        "dense": [value] * 1024,
        "sparse_indices": [1],
        "sparse_values": [value],
    }


def test_constructor_creates_only_one_bge_predictor() -> None:
    predictor_type = MagicMock()
    with patch.dict(_Deployment.__init__.__globals__, {"BGEM3Predictor": predictor_type}):
        deployment = _Deployment()

    predictor_type.assert_called_once()
    assert deployment._predictor is predictor_type.return_value
    assert "DenseEmbeddingPredictor" not in _Deployment.__init__.__code__.co_names
    assert "SentimentPredictor" not in _Deployment.__init__.__code__.co_names


def test_bge_asgi_app_is_ray_serializable() -> None:
    """Guard the Ray/FastAPI class-based ingress serialization boundary."""
    import ray.cloudpickle as pickle

    pickle.dumps(bge_app)


def test_embeddings_preserve_batch_order() -> None:
    predictor_type = MagicMock()
    with patch.dict(_Deployment.__init__.__globals__, {"BGEM3Predictor": predictor_type}):
        predictor_type.return_value.encode.return_value = [_vector(1.0), _vector(2.0)]
        deployment = _Deployment()
        result = asyncio.run(deployment.embeddings(EmbeddingsRequest(texts=["first", "second"])))

    predictor_type.return_value.encode.assert_called_once_with(["first", "second"])
    assert result["vectors"][0]["dense"][0] == 1.0
    assert result["vectors"][1]["dense"][0] == 2.0


def test_health_routes_report_live_and_ready() -> None:
    with patch.dict(_Deployment.__init__.__globals__, {"BGEM3Predictor": MagicMock()}):
        deployment = _Deployment()
        live = asyncio.run(deployment.health_live())
        ready = asyncio.run(deployment.health_ready())
        compatibility = asyncio.run(deployment.health())

    assert live["status"] == "alive"
    assert live["uptime_s"] >= 0
    assert ready["status"] == "ready"
    assert compatibility == ready


def test_metrics_report_embedding_requests() -> None:
    predictor_type = MagicMock()
    with patch.dict(_Deployment.__init__.__globals__, {"BGEM3Predictor": predictor_type}):
        predictor_type.return_value.encode.return_value = [_vector(1.0)]
        deployment = _Deployment()
        asyncio.run(deployment.embeddings(EmbeddingsRequest(texts=["hello"])))
        response = asyncio.run(deployment.metrics())

    body = response.body.decode()
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    assert "sentimentizer_bge_m3_service_ready 1" in body
    assert "sentimentizer_bge_m3_request_total 1" in body
    assert "sentimentizer_bge_m3_error_total 0" in body


def test_metrics_report_embedding_errors() -> None:
    predictor_type = MagicMock()
    with patch.dict(_Deployment.__init__.__globals__, {"BGEM3Predictor": predictor_type}):
        predictor_type.return_value.encode.side_effect = RuntimeError("encode failed")
        deployment = _Deployment()
        with pytest.raises(RuntimeError, match="encode failed"):
            asyncio.run(deployment.embeddings(EmbeddingsRequest(texts=["hello"])))
        response = asyncio.run(deployment.metrics())

    assert "sentimentizer_bge_m3_error_total 1" in response.body.decode()


def test_non_embedding_routes_are_not_registered() -> None:
    client = TestClient(bge_app)
    route_paths = [route.path for route in bge_app.routes]

    assert "/metrics" in route_paths
    assert client.post("/v1/sentiment/predict", json={"text": "hello"}).status_code == 404
    assert client.post("/v1/router/predict", json={"text": "hello"}).status_code == 404
    assert client.post("/v1/embeddings/dense", json={"texts": ["hello"]}).status_code == 404
    assert client.post("/v1/images/generate", json={"prompt": "hello"}).status_code == 404


def test_embedding_request_rejects_batches_over_maximum() -> None:
    with pytest.raises(ValidationError):
        EmbeddingsRequest(texts=["text"] * 65)


def test_initialize_ray_sets_object_store_memory() -> None:
    ray_module = MagicMock()

    initialize_ray(ray_module, 384, python_executable="/venv/bin/python")

    ray_module.init.assert_called_once_with(
        ignore_reinit_error=True,
        namespace="sentimentizer-bge",
        object_store_memory=384 * 1024 * 1024,
        runtime_env={"py_executable": "/venv/bin/python"},
    )


def test_initialize_ray_rejects_nonpositive_allocation() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        initialize_ray(MagicMock(), 0)


def test_main_disables_uv_run_runtime_env_hook() -> None:
    """bge_only_app sets RAY_ENABLE_UV_RUN_RUNTIME_ENV=false at module import
    time to skip the working-dir packaging + per-session venv creation on
    cold start (~37s savings). Must be set before any Ray import — Ray reads
    this constant at ray_constants import time, not at ray.init() call time."""
    # Importing the module sets the env var. Clear it first to prove the
    # import is what sets it, not a pre-existing value.
    with patch.dict(os.environ, {}, clear=True):
        # Force a re-import so the module-level setdefault runs fresh.
        import importlib

        import sentimentizer.serve.bge_only_app as bga

        importlib.reload(bga)
        assert os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV") == "false"


def test_main_respects_caller_override_of_uv_run_env() -> None:
    """A caller that pre-sets RAY_ENABLE_UV_RUN_RUNTIME_ENV=true keeps the
    uv hook enabled — setdefault does not overwrite an explicit choice."""
    with patch.dict(os.environ, {"RAY_ENABLE_UV_RUN_RUNTIME_ENV": "true"}, clear=True):
        import importlib

        import sentimentizer.serve.bge_only_app as bga

        importlib.reload(bga)
        assert os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV") == "true"


def test_bge_only_parser_defaults() -> None:
    args = build_parser().parse_args(["--bge-m3-only"])

    assert args.bge_m3_only is True
    assert args.host is None
    assert args.port is None
    assert args.ray_object_store_memory_mb is None


def test_bundled_server_settings() -> None:
    from sentimentizer.serve.config import load_serve_config

    serve_cfg = load_serve_config()

    assert serve_cfg.serve_host == "0.0.0.0"
    assert serve_cfg.serve_port == 8000
    assert serve_cfg.ray_object_store_memory_mb == 384
    assert serve_cfg.bge_m3_batch_size == 8
    assert serve_cfg.bge_m3_num_replicas == 1
    assert serve_cfg.bge_m3_max_ongoing_requests == 64
    assert serve_cfg.bge_m3_num_cpus == 2.0
    assert serve_cfg.bge_m3_num_gpus == 0.0


def test_bge_only_run_uses_yaml_server_settings(tmp_path) -> None:
    config_path = tmp_path / "custom-serve.yaml"
    config_path.write_text(
        "serve_host: 127.0.0.1\nserve_port: 9010\nray_object_store_memory_mb: 512\n"
    )

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("sentimentizer.serve.bge_only_app.main") as bge_main,
    ):
        args = build_parser().parse_args(["--bge-m3-only", "--config", str(config_path)])
        run(args)

    bge_main.assert_called_once_with(
        host="127.0.0.1",
        port=9010,
        object_store_memory_mb=512,
    )


def test_bge_only_cli_server_settings_override_yaml(tmp_path) -> None:
    config_path = tmp_path / "custom-serve.yaml"
    config_path.write_text(
        "serve_host: 127.0.0.1\nserve_port: 9010\nray_object_store_memory_mb: 512\n"
    )

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("sentimentizer.serve.bge_only_app.main") as bge_main,
    ):
        args = build_parser().parse_args(
            [
                "--bge-m3-only",
                "--config",
                str(config_path),
                "--host",
                "0.0.0.0",
                "--port",
                "9020",
                "--ray-object-store-memory-mb",
                "640",
            ]
        )
        run(args)

    bge_main.assert_called_once_with(
        host="0.0.0.0",
        port=9020,
        object_store_memory_mb=640,
    )


def test_server_settings_support_environment_overrides(tmp_path) -> None:
    from sentimentizer.serve.config import load_serve_config

    missing_config = tmp_path / "missing.yaml"
    with patch.dict(
        os.environ,
        {
            "SENTIMENTIZER_SERVE_HOST": "127.0.0.2",
            "SENTIMENTIZER_SERVE_PORT": "9030",
            "SENTIMENTIZER_RAY_OBJECT_STORE_MEMORY_MB": "768",
            "SENTIMENTIZER_BGE_M3_NUM_REPLICAS": "2",
            "SENTIMENTIZER_BGE_M3_MAX_ONGOING_REQUESTS": "32",
            "SENTIMENTIZER_BGE_M3_NUM_CPUS": "1.5",
            "SENTIMENTIZER_BGE_M3_NUM_GPUS": "0.5",
        },
        clear=True,
    ):
        serve_cfg = load_serve_config(missing_config)

    assert serve_cfg.serve_host == "127.0.0.2"
    assert serve_cfg.serve_port == 9030
    assert serve_cfg.ray_object_store_memory_mb == 768
    assert serve_cfg.bge_m3_num_replicas == 2
    assert serve_cfg.bge_m3_max_ongoing_requests == 32
    assert serve_cfg.bge_m3_num_cpus == 1.5
    assert serve_cfg.bge_m3_num_gpus == 0.5


@pytest.mark.parametrize(
    ("argv", "env"),
    [(["--diffusion"], {}), ([], {"SENTIMENTIZER_DIFFUSION_ENABLED": "1"})],
)
def test_diffusion_flag_exits_with_pointer_to_imagenate(capsys, argv, env) -> None:
    with patch.dict(os.environ, env):
        args = build_parser().parse_args(argv)
    with pytest.raises(SystemExit) as exc_info:
        run(args)
    assert exc_info.value.code == 2
    assert "imagenate" in capsys.readouterr().err
