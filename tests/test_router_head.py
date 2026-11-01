"""Tests for the router's pickle-free head format and its migration path."""

import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from click.testing import CliRunner
from joblib.numpy_pickle import NumpyArrayWrapper
from safetensors.numpy import save_file
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sentimentizer.router.head import (
    HEAD_FILENAME,
    CalibratedLinearHead,
    UnsupportedRouterHeadError,
    _load_pickled_head,
    convert_pickled_head,
    load_head,
    save_head,
)
from sentimentizer.router.model import RouterModel
from workflows.cli import cli

_RNG = np.random.default_rng(7)
_X = _RNG.normal(size=(60, 4))
_NEW = _RNG.normal(scale=4.0, size=(40, 4))
_BINARY = (_X[:, 0] > 0).astype(int)
_MULTI = np.digitize(_X[:, 0], [-0.5, 0.5])


def _calibrated(labels) -> CalibratedClassifierCV:
    indices = np.arange(len(_X))
    head = CalibratedClassifierCV(
        FrozenEstimator(LogisticRegression().fit(_X, labels)),
        method="sigmoid",
        cv=[(indices, indices)],
        ensemble=False,
    )
    return head.fit(_X, labels)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "labels",
        [_BINARY, _MULTI, np.array(["a", "b"])[_BINARY], np.array(["x", "y", "z"])[_MULTI]],
        ids=["binary-int", "multi-int", "binary-str", "multi-str"],
    )
    @pytest.mark.parametrize("calibrate", [False, True], ids=["plain", "calibrated"])
    def test_probabilities_and_classes_are_identical(self, tmp_path, labels, calibrate):
        head = _calibrated(labels) if calibrate else LogisticRegression().fit(_X, labels)
        path = tmp_path / HEAD_FILENAME

        save_head(head, path)
        restored = load_head(path)

        np.testing.assert_allclose(
            restored.predict_proba(_NEW), head.predict_proba(_NEW), atol=1e-12
        )
        np.testing.assert_array_equal(restored.classes_, head.classes_)
        np.testing.assert_array_equal(restored.predict(_NEW), head.predict(_NEW))
        assert isinstance(restored, CalibratedLinearHead if calibrate else LogisticRegression)

    def test_fortran_ordered_coefficients_survive_storage(self, tmp_path):
        """scikit-learn can fit Fortran-ordered coefficients; safetensors stores raw memory."""
        head = LogisticRegression().fit(_X, _MULTI)
        head.coef_ = np.asfortranarray(head.coef_)
        path = tmp_path / HEAD_FILENAME

        save_head(head, path)

        np.testing.assert_array_equal(load_head(path).coef_, head.coef_)

    def test_restored_calibrated_head_can_be_saved_again(self, tmp_path):
        first, second = tmp_path / "first", tmp_path / "second"
        save_head(_calibrated(_MULTI), first)
        restored = load_head(first)

        save_head(restored, second)

        np.testing.assert_array_equal(
            load_head(second).predict_proba(_NEW), restored.predict_proba(_NEW)
        )

    def test_restored_plain_head_can_still_be_calibrated(self, tmp_path):
        """Calibration needs a real scikit-learn estimator, not a lookalike."""
        path = tmp_path / HEAD_FILENAME
        save_head(LogisticRegression().fit(_X, _MULTI), path)
        indices = np.arange(len(_X))

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(load_head(path)),
            method="sigmoid",
            cv=[(indices, indices)],
            ensemble=False,
        ).fit(_X, _MULTI)

        assert calibrated.predict_proba(_NEW).shape == (40, 3)


class TestSaveRejections:
    def test_unsupported_head_is_rejected_without_writing(self, tmp_path):
        path = tmp_path / HEAD_FILENAME
        with pytest.raises(UnsupportedRouterHeadError, match="SVC"):
            save_head(SVC(probability=True).fit(_X, _BINARY), path)
        assert not path.exists()

    def test_isotonic_calibration_is_rejected(self, tmp_path):
        indices = np.arange(len(_X))
        head = CalibratedClassifierCV(
            FrozenEstimator(LogisticRegression().fit(_X, _BINARY)),
            method="isotonic",
            cv=[(indices, indices)],
            ensemble=False,
        ).fit(_X, _BINARY)
        with pytest.raises(UnsupportedRouterHeadError, match="isotonic"):
            save_head(head, tmp_path / HEAD_FILENAME)

    def test_head_that_would_change_probabilities_is_rejected(self, tmp_path):
        class Drifted(LogisticRegression):
            def predict_proba(self, X):
                return 1 - super().predict_proba(X)

        path = tmp_path / HEAD_FILENAME
        with pytest.raises(UnsupportedRouterHeadError, match="change probabilities"):
            save_head(Drifted().fit(_X, _BINARY), path)
        assert not path.exists()

    def test_failed_write_keeps_previous_file(self, tmp_path):
        path = tmp_path / HEAD_FILENAME
        save_head(LogisticRegression().fit(_X, _BINARY), path)
        before = path.read_bytes()

        with (
            patch("safetensors.numpy.save_file", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            save_head(LogisticRegression().fit(_X, _MULTI), path)

        assert path.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == [HEAD_FILENAME]


class TestLoadRejections:
    GOOD = {"coef": np.zeros((1, 4)), "intercept": np.zeros(1)}
    META = {"format": "sentimentizer-linear-head", "version": "1", "classes": "[0, 1]"}

    @pytest.mark.parametrize(
        ("tensors", "metadata"),
        [
            (GOOD, {**META, "format": "other"}),
            (GOOD, {**META, "version": "99"}),
            (GOOD, {k: v for k, v in META.items() if k != "classes"}),
            (GOOD, {**META, "classes": "not json"}),
            (GOOD, {**META, "classes": "[true, false]"}),
            (GOOD, {**META, "classes": "[1.5, 2.5]"}),
            (GOOD, {**META, "classes": "[0]"}),
            (GOOD, {**META, "classes": "[0, 0]"}),
            (GOOD, {**META, "classes": '{"a": 1}'}),
            ({"coef": np.zeros((1, 4))}, META),
            ({"coef": np.zeros((3, 4)), "intercept": np.zeros(1)}, META),
            ({"coef": np.zeros(4), "intercept": np.zeros(1)}, META),
            ({"coef": np.zeros((1, 4)), "intercept": np.zeros(3)}, META),
            ({"coef": np.full((1, 4), np.nan), "intercept": np.zeros(1)}, META),
            ({"coef": np.zeros((1, 4), dtype=np.int64), "intercept": np.zeros(1)}, META),
            ({**GOOD, "sigmoid_a": np.zeros(1)}, META),
            ({**GOOD, "sigmoid_a": np.zeros(3), "sigmoid_b": np.zeros(3)}, META),
            ({**GOOD, "sigmoid_a": np.full(1, np.inf), "sigmoid_b": np.zeros(1)}, META),
        ],
    )
    def test_invalid_contents_raise_value_error(self, tmp_path, tensors, metadata):
        path = tmp_path / HEAD_FILENAME
        save_file(tensors, str(path), metadata=metadata)
        with pytest.raises(ValueError):
            load_head(path)

    def test_corrupt_and_pickled_bytes_raise_value_error(self, tmp_path):
        path = tmp_path / HEAD_FILENAME
        path.write_bytes(pickle.dumps(LogisticRegression()))
        with pytest.raises(ValueError, match="unreadable"):
            load_head(path)


class _Exploit:
    """Pickle payload that leaves a marker file behind if it ever runs."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


class TestPickleMigration:
    @pytest.mark.parametrize("name", ["router_head.joblib", "model_head.pkl"])
    def test_converts_real_joblib_files_identically(self, tmp_path, name):
        head = LogisticRegression().fit(_X, _MULTI)
        joblib.dump(head, tmp_path / name)

        target = convert_pickled_head(tmp_path)

        np.testing.assert_array_equal(
            load_head(target).predict_proba(_NEW), head.predict_proba(_NEW)
        )
        assert (tmp_path / name).exists()

    def test_converts_calibrated_joblib_file(self, tmp_path):
        head = _calibrated(_MULTI)
        joblib.dump(head, tmp_path / "router_head.joblib")

        target = convert_pickled_head(tmp_path)

        np.testing.assert_allclose(
            load_head(target).predict_proba(_NEW), head.predict_proba(_NEW), atol=1e-12
        )

    def test_malicious_pickle_never_runs(self, tmp_path):
        marker = tmp_path / "executed"
        path = tmp_path / "router_head.joblib"
        path.write_bytes(pickle.dumps(_Exploit(marker)))

        with pytest.raises(pickle.UnpicklingError, match="posix.system|os.system"):
            convert_pickled_head(tmp_path)

        assert not marker.exists()
        assert not (tmp_path / HEAD_FILENAME).exists()

    def test_malicious_payload_in_object_array_never_runs(self, tmp_path):
        """joblib reads object arrays with unrestricted pickle.load; that path is blocked."""
        marker = tmp_path / "executed"
        wrapper = NumpyArrayWrapper(
            subclass=np.ndarray, shape=(1,), order="C", dtype=np.dtype(object)
        )
        path = tmp_path / "router_head.joblib"
        path.write_bytes(pickle.dumps(wrapper) + pickle.dumps(_Exploit(marker)))

        with pytest.raises(pickle.UnpicklingError, match="non-numeric"):
            _load_pickled_head(path)

        assert not marker.exists()

    def test_dotted_global_names_cannot_escape_the_allowlist(self, tmp_path):
        path = tmp_path / "router_head.joblib"
        path.write_bytes(b"\x80\x04\x8c\x05numpy\x8c\x0cdtype.__init\x93.")
        with pytest.raises(pickle.UnpicklingError, match="references"):
            _load_pickled_head(path)

    def test_existing_head_is_not_overwritten(self, tmp_path):
        joblib.dump(LogisticRegression().fit(_X, _BINARY), tmp_path / "router_head.joblib")
        convert_pickled_head(tmp_path)
        with pytest.raises(FileExistsError):
            convert_pickled_head(tmp_path)

    def test_missing_pickle_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_pickled_head(tmp_path)


class TestArtifactLoading:
    def test_pickled_artifact_is_refused_without_being_opened(self, tmp_path):
        marker = tmp_path / "executed"
        (tmp_path / "router_head.joblib").write_bytes(pickle.dumps(_Exploit(marker)))
        backbone = MagicMock()

        with (
            patch("sentimentizer.router.model._get_sentence_transformers", lambda: backbone),
            pytest.raises(RuntimeError, match="convert-head"),
        ):
            RouterModel.from_pretrained(tmp_path)

        assert not marker.exists()
        backbone.assert_not_called()

    def test_legacy_setfit_pickle_is_refused_without_being_opened(self, tmp_path):
        marker = tmp_path / "executed"
        (tmp_path / "model_head.pkl").write_bytes(pickle.dumps(_Exploit(marker)))

        with pytest.raises(RuntimeError, match="convert-head"):
            RouterModel.from_pretrained(tmp_path)

        assert not marker.exists()

    def test_artifact_without_head_is_reported(self, tmp_path):
        with pytest.raises(RuntimeError, match=HEAD_FILENAME):
            RouterModel.from_pretrained(tmp_path)

    def test_push_removes_old_pickles_from_the_repo(self):
        backbone = MagicMock()
        model = RouterModel(backbone, LogisticRegression().fit(_X, _BINARY), labels=["a", "b"])

        with patch("huggingface_hub.HfApi") as api:
            model.push_to_hub("me/repo")

        kwargs = api.return_value.upload_folder.call_args.kwargs
        assert set(kwargs["delete_patterns"]) == {"*.joblib", "*.pkl", "*.pickle"}


class TestCli:
    def test_convert_head_then_artifact_loads(self, tmp_path):
        head = LogisticRegression().fit(_X, _MULTI)
        joblib.dump(head, tmp_path / "router_head.joblib")
        (tmp_path / "router_config.json").write_text(json.dumps({"labels": ["a", "b", "c"]}))

        result = CliRunner().invoke(cli, ["router", "convert-head", "--model-path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        with patch(
            "sentimentizer.router.model._get_sentence_transformers",
            lambda: lambda _path: MagicMock(),
        ):
            loaded = RouterModel.from_pretrained(tmp_path)
        assert loaded.labels == ["a", "b", "c"]
        np.testing.assert_array_equal(loaded.head.predict_proba(_NEW), head.predict_proba(_NEW))

    def test_convert_head_rejects_malicious_file_with_nonzero_exit(self, tmp_path):
        marker = tmp_path / "executed"
        (tmp_path / "router_head.joblib").write_bytes(pickle.dumps(_Exploit(marker)))

        result = CliRunner().invoke(cli, ["router", "convert-head", "--model-path", str(tmp_path)])

        assert result.exit_code == 1
        assert "Failed to convert" in result.output
        assert not marker.exists()

    def test_failed_push_exits_nonzero(self, tmp_path):
        with patch.object(RouterModel, "from_pretrained", side_effect=RuntimeError("bad artifact")):
            result = CliRunner().invoke(
                cli, ["router", "push", "--model-path", str(tmp_path), "--repo-id", "me/repo"]
            )
        assert result.exit_code == 1
        assert "bad artifact" in result.output
