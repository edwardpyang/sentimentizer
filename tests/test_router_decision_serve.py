"""Check real HTTP serialization for decisions, warmup, retry and readiness."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from sentimentizer.predictor import SentimentPredictor
from sentimentizer.router.config import FOOD_TASK_SPEC
from sentimentizer.router.model import RouterModel
from sentimentizer.router.schemas import RouterAcceptancePolicy
from sentimentizer.serve.app import SentimentizerDeployment, create_fastapi_app
from sentimentizer.serve.base import ServiceMetrics
from sentimentizer.serve.models import (
    BatchRequest,
    PredictRequest,
    RouterDecisionBatchResponse,
    RouterDecisionResponse,
    RouterLoadRequest,
    RouterModelsResponse,
    RouterPredictResponse,
)

Deployment = SentimentizerDeployment.func_or_class


def router():
    backbone = MagicMock()
    backbone.encode.side_effect = lambda texts, **kw: np.array(
        [[0.7 if "uncertain" in text else 0.95] for text in texts]
    )
    head = MagicMock(classes_=np.array([0, 1, 2]))
    head.predict_proba.side_effect = lambda embeddings: np.array(
        [[row[0], (1 - row[0]) / 2, (1 - row[0]) / 2] for row in embeddings]
    )
    return RouterModel(
        backbone,
        head,
        task=FOOD_TASK_SPEC,
        acceptance_policy=RouterAcceptancePolicy(
            threshold=0.9, min_accepted=1, examples=1, accepted=1, accuracy=1
        ),
    )


def deployment(model=None):
    predictor = SentimentPredictor.__new__(SentimentPredictor)
    predictor._initialize_router_state()
    predictor._router_model_path = "unused"
    predictor._load_router_model = MagicMock(return_value=(model or router(), None))
    dep = SimpleNamespace()
    for name in (
        "_require_ready",
        "_router_decisions",
        "router_decide",
        "router_decide_batch",
        "router_load",
        "health_router",
        "router_predict",
    ):
        setattr(dep, name, MethodType(getattr(Deployment, name), dep))
    dep.predictor = predictor
    dep._ready = True
    dep._load_error = None
    dep._router_metrics = ServiceMetrics(prefix="router")

    async def decide(inputs):
        return predictor.decide_batch([inputs["text"]])[0]

    async def classify(inputs):
        return predictor.classify_batch([inputs["text"]])[0]

    dep.decide_route = decide
    dep.classify_route = classify
    return dep


def client_for(dep):
    api = create_fastapi_app("Router decisions", "Router decisions")

    @api.post("/v1/router/decide", response_model=RouterDecisionResponse)
    async def single(body: PredictRequest, request: Request):
        return await dep.router_decide(body, request)

    @api.post("/v1/router/decide/batch", response_model=RouterDecisionBatchResponse)
    async def batch(body: BatchRequest, request: Request):
        return await dep.router_decide_batch(body, request)

    @api.post("/v1/router/load", response_model=RouterModelsResponse)
    async def load(body: RouterLoadRequest):
        return await dep.router_load(body)

    @api.get("/health/router", response_model=None)
    async def ready():
        return await dep.health_router()

    @api.post("/v1/router/predict", response_model=RouterPredictResponse)
    async def classify(body: PredictRequest, request: Request):
        return await dep.router_predict(body, request)

    return TestClient(api, raise_server_exceptions=False)


def test_readiness_does_not_load_and_warmup_enables_decisions():
    dep = deployment()
    with client_for(dep) as client:
        initial = client.get("/health/router")
        assert initial.status_code == 503
        assert initial.json()["router_loading"] is False
        dep.predictor._load_router_model.assert_not_called()
        assert client.post("/v1/router/load", json={}).status_code == 200
        ready = client.get("/health/router")
        assert ready.status_code == 200
        assert ready.json()["decision_ready"] is True
        assert client.post("/v1/router/load", json={}).status_code == 200
    dep.predictor._load_router_model.assert_called_once()


def test_single_and_batch_decisions_keep_label_and_input_order():
    dep = deployment()
    with client_for(dep) as client:
        single = client.post("/v1/router/decide", json={"text": "uncertain allergy mention"})
        assert single.status_code == 200
        assert single.json()["prediction"] == {
            "label": "dietary",
            "score": 0.7,
            "token_count": 3,
            "decision": "abstain",
        }
        batch = client.post(
            "/v1/router/decide/batch",
            json={"texts": ["allergy mention", "uncertain allergy mention"]},
        )
        assert batch.status_code == 200
        assert batch.json()["count"] == 2
        assert [row["prediction"]["decision"] for row in batch.json()["results"]] == [
            "accept",
            "abstain",
        ]
        original = client.post("/v1/router/predict", json={"text": "uncertain allergy mention"})
        assert set(original.json()["prediction"]) == {"label", "score", "token_count"}
    assert dep._router_metrics.error_count == 0


def test_missing_policy_is_explicitly_unavailable_while_classification_works():
    model = router()
    model.acceptance_policy = None
    dep = deployment(model)
    with client_for(dep) as client:
        response = client.post("/v1/router/decide", json={"text": "allergy mention"})
        assert response.status_code == 503
        assert "calibrate" in response.json()["error"]["message"]
        assert client.get("/health/router").json()["decision_ready"] is False
        assert (
            client.post("/v1/router/predict", json={"text": "allergy mention"}).status_code == 200
        )
    assert dep._router_metrics.error_count == 1


def test_load_failure_requires_explicit_retry_and_strict_request():
    dep = deployment()
    dep.predictor._load_router_model.side_effect = [(None, "transient failure"), (router(), None)]
    with client_for(dep) as client:
        assert client.post("/v1/router/load", json={}).status_code == 503
        assert client.post("/v1/router/load", json={}).status_code == 503
        assert dep.predictor._load_router_model.call_count == 1
        assert client.post("/v1/router/load", json={"retry_failed": "true"}).status_code == 422
        assert client.post("/v1/router/load", json={"retry_failed": True}).status_code == 200
    assert dep.predictor._load_router_model.call_count == 2


@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"prediction": {"label": "dietary", "score": 0.8, "token_count": 2, "decision": "maybe"}}],
        [
            {
                "prediction": {
                    "label": "dietary",
                    "score": float("nan"),
                    "token_count": 2,
                    "decision": "accept",
                }
            }
        ],
    ],
)
def test_invalid_or_partial_decisions_fail_before_success_accounting(results):
    dep = deployment()
    dep.predictor.load_router()
    dep.predictor.decide_batch = MagicMock(return_value=results)
    with client_for(dep) as client:
        response = client.post("/v1/router/decide/batch", json={"texts": ["one example"]})
    assert response.status_code == 500
    assert dep._router_metrics.error_count == 1
    assert dep._router_metrics.request_count == 1
