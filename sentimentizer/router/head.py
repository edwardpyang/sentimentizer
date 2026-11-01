"""Store the router's classification head without pickle.

Pickle runs code from the file being loaded, so serving only reads
``router_head.safetensors``, which holds numbers and a small JSON header. Two heads
can be stored: a LogisticRegression, and one wrapped with sigmoid calibration.

Older artifacts hold a pickled head. ``convert_pickled_head`` migrates them once,
through an unpickler that can only build the classes those artifacts contain.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

HEAD_FILENAME = "router_head.safetensors"
PICKLED_HEAD_FILENAMES = ("router_head.joblib", "model_head.pkl", "model_head.pickle")
PICKLED_HEAD_PATTERNS = ("*.joblib", "*.pkl", "*.pickle")

_FORMAT_NAME = "sentimentizer-linear-head"
_FORMAT_VERSION = "1"
_EQUIVALENCE_TOLERANCE = 1e-9
_PROBE_ROWS = 16
_PROBE_SEED = 0
_PROBE_SCALE = 3.0

_ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        ("joblib.numpy_pickle", "NumpyArrayWrapper"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("sklearn.calibration", "CalibratedClassifierCV"),
        ("sklearn.calibration", "_CalibratedClassifier"),
        ("sklearn.calibration", "_SigmoidCalibration"),
        ("sklearn.frozen._frozen", "FrozenEstimator"),
        ("sklearn.linear_model._logistic", "LogisticRegression"),
        ("sklearn.preprocessing._label", "LabelEncoder"),
    }
)


class UnsupportedRouterHeadError(TypeError):
    """Signal a head that the pickle-free format cannot represent."""


class CalibratedLinearHead:
    """Apply saved per-class sigmoid calibration to a logistic regression.

    Matches ``CalibratedClassifierCV`` with ``method="sigmoid"``; ``save_head``
    verifies that against the original head before publishing a file.
    """

    def __init__(self, base: Any, sigmoid_a: np.ndarray, sigmoid_b: np.ndarray) -> None:
        self.base = base
        self.sigmoid_a = sigmoid_a
        self.sigmoid_b = sigmoid_b

    @property
    def classes_(self) -> np.ndarray:
        """Return class values in probability-column order."""
        return self.base.classes_

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities with one row per embedding."""
        from scipy.special import expit

        scores = np.asarray(self.base.decision_function(embeddings)).reshape(
            len(embeddings), len(self.sigmoid_a)
        )
        calibrated = expit(-(self.sigmoid_a * scores + self.sigmoid_b))
        if len(self.classes_) == 2:
            return np.column_stack([1.0 - calibrated[:, 0], calibrated[:, 0]])
        denominator = calibrated.sum(axis=1, keepdims=True)
        uniform = np.full_like(calibrated, 1 / len(self.classes_))
        return np.divide(calibrated, denominator, out=uniform, where=denominator != 0)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the most probable class for each embedding."""
        return self.classes_[self.predict_proba(embeddings).argmax(axis=1)]


def _build_logistic_regression(coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray) -> Any:
    """Rebuild a fitted LogisticRegression from its parameters."""
    from sklearn.linear_model import LogisticRegression

    head = LogisticRegression()
    head.classes_ = classes
    head.coef_ = coef
    head.intercept_ = intercept
    head.n_features_in_ = coef.shape[1]
    head.n_iter_ = np.zeros(1, dtype=np.int32)
    return head


def _unwrap_calibrated(head: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return the base estimator and sigmoid parameters of a calibrated head."""
    from sklearn.frozen import FrozenEstimator

    if len(head.calibrated_classifiers_) != 1:
        raise UnsupportedRouterHeadError("Only a single calibrated classifier can be stored")
    calibrated = head.calibrated_classifiers_[0]
    if calibrated.method != "sigmoid":
        raise UnsupportedRouterHeadError(f"Cannot store {calibrated.method!r} calibration")
    base = calibrated.estimator
    if isinstance(base, FrozenEstimator):
        base = base.estimator
    sigmoid_a = np.array([item.a_ for item in calibrated.calibrators], dtype=np.float64)
    sigmoid_b = np.array([item.b_ for item in calibrated.calibrators], dtype=np.float64)
    return base, sigmoid_a, sigmoid_b


def _head_parameters(head: Any) -> tuple[Any, np.ndarray | None, np.ndarray | None]:
    """Return a head's logistic regression and optional sigmoid parameters."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    base, sigmoid_a, sigmoid_b = head, None, None
    if isinstance(head, CalibratedLinearHead):
        base, sigmoid_a, sigmoid_b = head.base, head.sigmoid_a, head.sigmoid_b
    elif isinstance(head, CalibratedClassifierCV):
        base, sigmoid_a, sigmoid_b = _unwrap_calibrated(head)
    if not isinstance(base, LogisticRegression):
        raise UnsupportedRouterHeadError(
            f"Cannot store a {type(base).__name__} router head without pickle; "
            "use LogisticRegression, optionally with sigmoid calibration"
        )
    return base, sigmoid_a, sigmoid_b


def _rebuild_head(
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
    sigmoid_a: np.ndarray | None,
    sigmoid_b: np.ndarray | None,
) -> Any:
    """Build the servable head from stored parameters."""
    base = _build_logistic_regression(coef, intercept, classes)
    if sigmoid_a is None or sigmoid_b is None:
        return base
    return CalibratedLinearHead(base, sigmoid_a, sigmoid_b)


def _validate_parameters(
    coef: np.ndarray,
    intercept: np.ndarray,
    classes: np.ndarray,
    sigmoid_a: np.ndarray | None,
    sigmoid_b: np.ndarray | None,
) -> None:
    """Reject parameters that could not have come from a fitted head."""
    if classes.ndim != 1 or len(classes) < 2 or classes.dtype.kind not in "iuU":
        raise ValueError("Router head classes must be two or more integers or strings")
    if len(set(classes.tolist())) != len(classes):
        raise ValueError("Router head classes must be unique")
    outputs = 1 if len(classes) == 2 else len(classes)
    if coef.ndim != 2 or coef.shape[0] != outputs or coef.shape[1] < 1:
        raise ValueError("Router head coefficients do not match its classes")
    if intercept.shape != (outputs,):
        raise ValueError("Router head intercept does not match its classes")
    if (sigmoid_a is None) != (sigmoid_b is None):
        raise ValueError("Router head sigmoid parameters must be stored together")
    arrays = [coef, intercept]
    if sigmoid_a is not None and sigmoid_b is not None:
        if sigmoid_a.shape != (outputs,) or sigmoid_b.shape != (outputs,):
            raise ValueError("Router head sigmoid parameters do not match its classes")
        arrays += [sigmoid_a, sigmoid_b]
    if not all(array.dtype.kind == "f" and np.isfinite(array).all() for array in arrays):
        raise ValueError("Router head parameters must be finite numbers")


def save_head(head: Any, path: Path) -> None:
    """Write a head atomically, after proving the written file predicts identically.

    Raises:
        UnsupportedRouterHeadError: If the format cannot represent *head*, or the
            rebuilt head would change its probabilities.
    """
    from safetensors.numpy import save_file

    base, sigmoid_a, sigmoid_b = _head_parameters(head)
    # safetensors writes raw memory as C order, but scikit-learn can fit Fortran-ordered arrays.
    coef = np.array(base.coef_, dtype=np.float64, order="C")
    intercept = np.array(base.intercept_, dtype=np.float64, order="C")
    classes = np.asarray(base.classes_)
    _validate_parameters(coef, intercept, classes, sigmoid_a, sigmoid_b)

    tensors = {"coef": coef, "intercept": intercept}
    if sigmoid_a is not None and sigmoid_b is not None:
        tensors |= {"sigmoid_a": sigmoid_a, "sigmoid_b": sigmoid_b}
    metadata = {
        "format": _FORMAT_NAME,
        "version": _FORMAT_VERSION,
        "classes": json.dumps(classes.tolist()),
    }
    temporary_path = path.with_name(f".{path.name}.partial")
    try:
        save_file(tensors, str(temporary_path), metadata=metadata)
        # Check the written bytes, since a mistake in storage is what this guards against.
        generator = np.random.default_rng(_PROBE_SEED)
        probe = generator.normal(scale=_PROBE_SCALE, size=(_PROBE_ROWS, coef.shape[1]))
        if not np.allclose(
            head.predict_proba(probe),
            load_head(temporary_path).predict_proba(probe),
            atol=_EQUIVALENCE_TOLERANCE,
        ):
            raise UnsupportedRouterHeadError("The stored router head would change probabilities")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_head(path: Path) -> Any:
    """Read and validate a stored head; this never runs code from the file.

    Raises:
        ValueError: If the file is not a valid stored head.
    """
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(path), framework="numpy") as source:
            metadata = source.metadata() or {}
            tensors = {name: source.get_tensor(name) for name in source.keys()}  # noqa: SIM118
    except (SafetensorError, OSError) as exc:
        raise ValueError(f"Router head file is unreadable: {exc}") from exc
    if metadata.get("format") != _FORMAT_NAME or metadata.get("version") != _FORMAT_VERSION:
        raise ValueError("Router head file has an unsupported format")
    if not {"coef", "intercept"} <= tensors.keys():
        raise ValueError("Router head file is missing its parameters")
    try:
        class_values = json.loads(metadata["classes"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Router head file has invalid classes") from exc
    if not isinstance(class_values, list) or not (
        all(type(value) is int for value in class_values)
        or all(type(value) is str for value in class_values)
    ):
        raise ValueError("Router head classes must be two or more integers or strings")

    parameters = (
        tensors["coef"],
        tensors["intercept"],
        np.array(class_values),
        tensors.get("sigmoid_a"),
        tensors.get("sigmoid_b"),
    )
    _validate_parameters(*parameters)
    return _rebuild_head(*parameters)


def _load_pickled_head(path: Path) -> Any:
    """Unpickle a legacy head while refusing anything beyond the expected classes."""
    from joblib.numpy_pickle import NumpyArrayWrapper, NumpyUnpickler

    class _PlainArrayWrapper(NumpyArrayWrapper):
        """Refuse object arrays, which joblib reads with an unrestricted pickle.load."""

        def read(self, unpickler: Any, ensure_native_byte_order: bool) -> Any:
            if self.dtype.hasobject or self.subclass is not np.ndarray:
                raise pickle.UnpicklingError("Pickled head contains a non-numeric array")
            return super().read(unpickler, ensure_native_byte_order)

    class _RestrictedUnpickler(NumpyUnpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) not in _ALLOWED_PICKLE_GLOBALS:
                raise pickle.UnpicklingError(f"Pickled head references {module}.{name}")
            if (module, name) == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
                return _PlainArrayWrapper
            return super().find_class(module, name)

    with path.open("rb") as source:
        return _RestrictedUnpickler(str(path), source, ensure_native_byte_order=True).load()


def convert_pickled_head(model_dir: str | Path) -> Path:
    """Migrate an artifact's pickled head to the pickle-free format.

    Returns:
        Path to the new head file. The pickled file is left in place, unused.

    Raises:
        FileExistsError: If the artifact already has a pickle-free head.
        FileNotFoundError: If the artifact has no pickled head.
        pickle.UnpicklingError: If the pickle references anything unexpected.
        UnsupportedRouterHeadError: If the head cannot be stored without pickle.
    """
    model_dir = Path(model_dir)
    target = model_dir / HEAD_FILENAME
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    for name in PICKLED_HEAD_FILENAMES:
        source = model_dir / name
        if source.exists():
            save_head(_load_pickled_head(source), target)
            return target
    raise FileNotFoundError(f"No pickled router head found in {model_dir}")
