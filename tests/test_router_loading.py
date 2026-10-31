"""Force competing lazy-load and retry interleavings without timing guesses."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from sentimentizer.predictor import SentimentPredictor


class ObservedCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


def predictor():
    instance = SentimentPredictor.__new__(SentimentPredictor)
    instance._initialize_router_state()
    instance._router_model_path = "unused"
    instance._router_condition = ObservedCondition()
    return instance


@pytest.mark.parametrize("outcome", ["success", "failure", "exception", "interrupt"])
def test_concurrent_load_has_one_owner_and_releases_waiter_on_every_outcome(outcome):
    instance = predictor()
    entered, release = threading.Event(), threading.Event()
    model = MagicMock()

    def load(_):
        entered.set()
        assert release.wait(5), "Test failed to release router construction"
        if outcome == "exception":
            raise ValueError("controlled constructor failure")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        return (model, None) if outcome == "success" else (None, "unavailable")

    instance._load_router_model = MagicMock(side_effect=load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(instance.load_router)
        try:
            assert entered.wait(5)
            waiter = pool.submit(instance.load_router)
            assert instance._router_condition.waiting.wait(5)
            assert instance.router_loading
            assert not instance.router_loaded
            assert instance._load_router_model.call_count == 1
        finally:
            release.set()
        if outcome == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                owner.result(timeout=5)
        else:
            assert owner.result(timeout=5) is (outcome == "success")
        assert waiter.result(timeout=5) is (outcome == "success")
    assert not instance.router_loading
    instance.load_router()
    assert instance._load_router_model.call_count == 1
    if outcome != "success":
        instance._load_router_model.return_value = (model, None)
        instance._load_router_model.side_effect = None
        assert instance.retry_router_load()
        assert instance._load_router_model.call_count == 2
        assert instance.router_error is None


def test_retry_joins_active_attempt_and_does_not_clear_its_failure():
    instance = predictor()
    entered, release = threading.Event(), threading.Event()

    def load(_):
        entered.set()
        assert release.wait(5)
        return None, "controlled failure"

    instance._load_router_model = MagicMock(side_effect=load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(instance.load_router)
        try:
            assert entered.wait(5)
            retry = pool.submit(instance.retry_router_load)
            assert instance._router_condition.waiting.wait(5)
        finally:
            release.set()
        assert not owner.result(timeout=5)
        assert not retry.result(timeout=5)
    assert instance._load_router_model.call_count == 1
    assert instance.router_error == "controlled failure"


def test_retry_of_loaded_model_is_idempotent():
    instance = predictor()
    instance._load_router_model = MagicMock(return_value=(MagicMock(), None))
    assert instance.load_router()
    assert instance.retry_router_load()
    instance._load_router_model.assert_called_once()
