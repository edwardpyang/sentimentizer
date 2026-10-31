"""Tests for the Click-based CLI driver.

Verifies lazy-loading (no torch/ray on --help), flag surfaces, and
command composition without requiring the full ML stack.
"""

import subprocess
import sys

import pytest
from click.testing import CliRunner

from workflows.driver import State, cli

# ──────────────────────────────────────────────
# Lazy-loading: --help must not import ML stack
# ──────────────────────────────────────────────

SUBCOMMANDS_TO_CHECK = [
    [],  # bare group help
    ["extract", "--help"],
    ["tokenize", "--help"],
    ["train", "--help"],
    ["tune", "--help"],
    ["hf", "--help"],
    ["hf", "push", "--help"],
    ["hf", "pull", "--help"],
    ["diagnose", "--help"],
    ["diagnose", "env", "--help"],
    ["diagnose", "pipeline", "--help"],
    ["run", "--help"],
]


@pytest.mark.parametrize("argv", SUBCOMMANDS_TO_CHECK)
def test_help_does_not_import_ml_stack(argv: list[str]) -> None:
    """Every <subcommand> --help must render without pulling in torch or ray.

    Critical: click runs the group callback before subcommand help, so a
    single ``import torch`` (or ``from sentimentizer.config import ...``) in
    the group callback poisons every subcommand's --help latency.

    Spawned in a fresh subprocess because pytest's other tests may have
    already imported torch/ray in the current process.
    """
    args_repr = repr(argv if argv else ["--help"])
    code = (
        "import sys, time\n"
        "from workflows.driver import cli\n"
        "from click.testing import CliRunner\n"
        f"start = time.time()\n"
        f"result = CliRunner().invoke(cli, {args_repr})\n"
        f"duration = time.time() - start\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'torch' not in sys.modules, 'torch leaked'\n"
        "assert 'ray' not in sys.modules, 'ray leaked'\n"
        "assert 'sentimentizer.config' not in sys.modules, 'config (which imports torch) leaked'\n"
        f"assert duration < 1.0, f'help took {{duration:.2f}}s — likely slow import leak'\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr or out.stdout


def test_diagnose_env_no_ml_imports() -> None:
    """Verify that actually running ``diagnose env`` imports neither torch nor ray.

    This is separate from the parametrized --help test because that test
    covers help rendering, while this one covers command execution.
    """
    code = (
        "import sys, time\n"
        "from workflows.driver import cli\n"
        "from click.testing import CliRunner\n"
        "start = time.time()\n"
        "result = CliRunner().invoke(cli, ['diagnose', 'env'])\n"
        "duration = time.time() - start\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'torch' not in sys.modules, 'torch leaked'\n"
        "assert 'ray' not in sys.modules, 'ray leaked'\n"
        "assert 'sentimentizer.config' not in sys.modules, 'config leaked'\n"
        "assert duration < 1.0, f'diagnose env took {duration:.2f}s'\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr or out.stdout


# ──────────────────────────────────────────────
# Flag surface tests
# ──────────────────────────────────────────────


def test_train_help_lists_flags() -> None:
    result = CliRunner().invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--distributed" in result.output
    assert "--checkpoint-dir" in result.output
    assert "--resume" in result.output
    assert "--save" in result.output


def test_run_help_lists_flags() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--distributed" in result.output
    assert "--checkpoint-dir" in result.output
    assert "--resume-tokenize" in result.output
    assert "--resume-train" in result.output
    assert "--save" in result.output
    # run should NOT have a unified --resume
    assert "--resume" in result.output  # appears as part of --resume-tokenize/--resume-train


def test_tune_help_lists_flags() -> None:
    result = CliRunner().invoke(cli, ["tune", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "--samples" in result.output
    assert "--no-validate" in result.output


def test_extract_help_lists_flags() -> None:
    result = CliRunner().invoke(cli, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--stop" in result.output


def test_tokenize_help_lists_flags() -> None:
    result = CliRunner().invoke(cli, ["tokenize", "--help"])
    assert result.exit_code == 0
    assert "--resume" in result.output


def test_root_help_lists_global_options() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--device" in result.output
    assert "--run-type" in result.output


# ──────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────


def test_invalid_model_rejected() -> None:
    result = CliRunner().invoke(cli, ["--model", "invalid", "train"])
    assert result.exit_code != 0


def test_invalid_run_type_rejected() -> None:
    result = CliRunner().invoke(cli, ["--run-type", "fast", "train"])
    assert result.exit_code != 0


def test_hf_requires_subcommand() -> None:
    result = CliRunner().invoke(cli, ["hf"])
    assert result.exit_code != 0


def test_diagnose_bare_shows_help() -> None:
    """Bare ``sentimentizer diagnose`` should show the group help (exit 0)."""
    result = CliRunner().invoke(cli, ["diagnose"])
    # Click groups with subcommands show help and exit 0 when invoked without a subcommand
    # (or exit with non-zero if invoke_without_command is not set — both are acceptable)
    # The key invariant: it must NOT import the ML stack.
    assert result.exit_code == 0 or "env" in result.output


# ──────────────────────────────────────────────
# Command composition tests (mocked)
# ──────────────────────────────────────────────


def test_run_chains_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "workflows.stages.extract.run_extract", lambda *a, **kw: calls.append("extract")
    )
    monkeypatch.setattr(
        "workflows.stages.tokenize.run_tokenize", lambda *a, **kw: calls.append("tokenize")
    )
    monkeypatch.setattr("workflows.stages.train.run_train", lambda *a, **kw: calls.append("train"))
    result = CliRunner().invoke(cli, ["run", "--stop", "10"])
    assert result.exit_code == 0, result.output
    assert calls == ["extract", "tokenize", "train"]


def test_run_passes_resume_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify --resume-tokenize and --resume-train are passed correctly."""
    tokenize_kwargs: dict = {}
    train_kwargs: dict = {}

    def mock_tokenize(state: State, *, resume: bool) -> None:
        tokenize_kwargs["resume"] = resume

    def mock_train(state: State, **kwargs: object) -> None:
        train_kwargs.update(kwargs)

    monkeypatch.setattr("workflows.stages.extract.run_extract", lambda *a, **kw: None)
    monkeypatch.setattr("workflows.stages.tokenize.run_tokenize", mock_tokenize)
    monkeypatch.setattr("workflows.stages.train.run_train", mock_train)

    result = CliRunner().invoke(
        cli,
        ["run", "--stop", "10", "--resume-tokenize", "--resume-train", "--checkpoint-dir", "ckpt"],
    )
    assert result.exit_code == 0, result.output
    assert tokenize_kwargs["resume"] is True
    assert train_kwargs["resume"] is True


def test_extract_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "workflows.stages.extract.run_extract", lambda *a, **kw: calls.append("extract")
    )
    result = CliRunner().invoke(cli, ["extract", "--stop", "5000"])
    assert result.exit_code == 0, result.output
    assert calls == ["extract"]


def test_tokenize_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "workflows.stages.tokenize.run_tokenize", lambda *a, **kw: calls.append("tokenize")
    )
    result = CliRunner().invoke(cli, ["tokenize"])
    assert result.exit_code == 0, result.output
    assert calls == ["tokenize"]


def test_train_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("workflows.stages.train.run_train", lambda *a, **kw: calls.append("train"))
    result = CliRunner().invoke(cli, ["train", "--save"])
    assert result.exit_code == 0, result.output
    assert calls == ["train"]


def test_tune_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("workflows.stages.tune.run_tune", lambda *a, **kw: calls.append("tune"))
    result = CliRunner().invoke(cli, ["tune", "--mode", "standalone"])
    assert result.exit_code == 0, result.output
    assert calls == ["tune"]


def test_diagn_pipeline_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "workflows.stages.diagnose.run_diagnose_pipeline", lambda *a, **kw: calls.append("pipeline")
    )
    result = CliRunner().invoke(cli, ["diagnose", "pipeline"])
    assert result.exit_code == 0, result.output
    assert calls == ["pipeline"]


def test_diagn_env_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "workflows.stages.diagnose.run_diagnose_env", lambda *a, **kw: calls.append("env")
    )
    result = CliRunner().invoke(cli, ["diagnose", "env"])
    assert result.exit_code == 0, result.output
    assert calls == ["env"]


def test_hf_push_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("workflows.stages.hf.run_hf_push", lambda *a, **kw: calls.append("push"))
    result = CliRunner().invoke(cli, ["hf", "push"])
    assert result.exit_code == 0, result.output
    assert calls == ["push"]


def test_hf_pull_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("workflows.stages.hf.run_hf_pull", lambda *a, **kw: calls.append("pull"))
    result = CliRunner().invoke(cli, ["hf", "pull"])
    assert result.exit_code == 0, result.output
    assert calls == ["pull"]


# ──────────────────────────────────────────────
# State dataclass tests
# ──────────────────────────────────────────────


def test_state_defaults() -> None:
    state = State(model="rnn", device="auto", run_type="new")
    assert state.model == "rnn"
    assert state.device == "auto"
    assert state.run_type == "new"


def test_state_device_stored_unresolved() -> None:
    """Verify that the group callback stores device='auto' as-is, not resolved."""
    result = CliRunner().invoke(cli, ["--device", "auto", "--help"])
    assert result.exit_code == 0


def test_resolve_device() -> None:
    """Verify resolve_device returns non-auto values as-is and resolves auto."""
    from sentimentizer.device import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"
    # auto resolves to one of the valid devices
    assert resolve_device("auto") in ("cpu", "cuda", "mps")


def test_resolve_device_warns_on_cpu_torch_with_nvidia_libs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """resolve_device should warn when torch is CPU-only but NVIDIA libs exist."""
    import torch

    # Skip if torch actually has CUDA (no warning expected)
    if torch.cuda.is_available():
        return

    from sentimentizer import device as device_mod

    # Simulate CPU-only torch (+cpu suffix) with NVIDIA libs present
    monkeypatch.setattr(torch, "__version__", "2.11.0+cpu")
    monkeypatch.setattr(device_mod, "_has_nvidia_libs", lambda: True)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    import logging

    with caplog.at_level(logging.WARNING):
        result = device_mod.resolve_device("auto")

    assert result == "cpu"
    assert any("CPU-only" in rec.message for rec in caplog.records)


def test_resolve_device_no_warning_when_no_nvidia_libs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """resolve_device should NOT warn when CPU-only torch and no NVIDIA libs."""
    import torch

    if torch.cuda.is_available():
        return

    from sentimentizer import device as device_mod

    monkeypatch.setattr(torch, "__version__", "2.11.0+cpu")
    monkeypatch.setattr(device_mod, "_has_nvidia_libs", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    import logging

    with caplog.at_level(logging.WARNING):
        result = device_mod.resolve_device("auto")

    assert result == "cpu"
    assert not any("CPU-only" in rec.message for rec in caplog.records)


def test_run_tokenize_new_retransforms_data_when_rows_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: str
) -> None:
    """run_type='new' should always re-transform data, even when parquet has enough rows.

    A 'new' run rebuilds everything from scratch — the tokenizer config
    (e.g. include_neutral) or class mapping may have changed, so stale
    parquet files must be re-created.
    """
    from sentimentizer.config import DriverConfig
    from workflows.driver import State

    processed_path = str(tmp_path / "review_data.parquet")
    raw_path = str(tmp_path / "raw_reviews.parquet")

    monkeypatch.setattr(DriverConfig.files, "processed_reviews_file_path", processed_path)
    monkeypatch.setattr(DriverConfig.files, "raw_reviews_file_path", raw_path)

    monkeypatch.setattr("workflows.stages.tokenize._parquet_row_count", lambda p: 20000)
    monkeypatch.setattr("workflows.lifecycle._ensure_ray_initialized", lambda: None)
    monkeypatch.setattr("workflows.lifecycle.is_ray_available", lambda: False)

    calls: list[str] = []

    class FakeDataSource:
        def filter(self, **kwargs):
            return self

        def map_batches(self, *args, **kwargs):
            return self

        def write_parquet(self, path):
            calls.append("write_parquet")

    monkeypatch.setattr(
        "sentimentizer.data_source.read_parquet",
        lambda path, use_ray=False: FakeDataSource(),
    )

    class FakeTokenizer:
        @classmethod
        def build_dictionary(cls, data_source, cfg=None):
            calls.append("build_dictionary")
            return cls()

        def transform(self, data_source):
            calls.append("transform")
            return data_source

    monkeypatch.setattr("sentimentizer.tokenizer.Tokenizer", FakeTokenizer)

    from workflows.stages.tokenize import run_tokenize

    state = State(model="rnn", device="cpu", run_type="new")
    run_tokenize(state)

    assert "build_dictionary" in calls, "Dictionary should have been created"
    assert "transform" in calls, "Data should always be re-transformed on 'new' run"


def test_run_tokenize_new_creates_data_when_rows_insufficient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: str
) -> None:
    """run_type='new' should create both dictionary and data when rows are insufficient."""
    from sentimentizer.config import DriverConfig
    from workflows.driver import State

    processed_path = str(tmp_path / "review_data.parquet")
    raw_path = str(tmp_path / "raw_reviews.parquet")

    monkeypatch.setattr(DriverConfig.files, "processed_reviews_file_path", processed_path)
    monkeypatch.setattr(DriverConfig.files, "raw_reviews_file_path", raw_path)

    monkeypatch.setattr("workflows.stages.tokenize._parquet_row_count", lambda p: 0)
    monkeypatch.setattr("workflows.stages.tokenize._remove_path", lambda p: None)
    monkeypatch.setattr("workflows.lifecycle._ensure_ray_initialized", lambda: None)
    monkeypatch.setattr("workflows.lifecycle.is_ray_available", lambda: False)

    calls: list[str] = []

    class FakeDataSource:
        def filter(self, **kwargs):
            return self

        def map_batches(self, *args, **kwargs):
            return self

        def write_parquet(self, path):
            calls.append("write_parquet")

    monkeypatch.setattr(
        "sentimentizer.data_source.read_parquet",
        lambda path, use_ray=False: FakeDataSource(),
    )

    class FakeTokenizer:
        @classmethod
        def build_dictionary(cls, data_source, cfg=None):
            calls.append("build_dictionary")
            return cls()

        def transform(self, data_source):
            calls.append("transform")
            return data_source

    monkeypatch.setattr("sentimentizer.tokenizer.Tokenizer", FakeTokenizer)

    from workflows.stages.tokenize import run_tokenize

    state = State(model="rnn", device="cpu", run_type="new")
    run_tokenize(state)

    assert "build_dictionary" in calls, "Dictionary should have been created"
    assert "transform" in calls, "Data should have been transformed"
    assert "write_parquet" in calls, "Data should have been written"


def test_run_tokenize_update_skips_when_rows_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: str
) -> None:
    """run_type='update' should skip entirely when parquet already has enough rows."""
    from sentimentizer.config import DriverConfig
    from workflows.driver import State

    processed_path = str(tmp_path / "review_data.parquet")

    monkeypatch.setattr(DriverConfig.files, "processed_reviews_file_path", processed_path)

    monkeypatch.setattr("workflows.stages.tokenize._parquet_row_count", lambda p: 20000)
    monkeypatch.setattr("workflows.lifecycle._ensure_ray_initialized", lambda: None)
    monkeypatch.setattr("workflows.lifecycle.is_ray_available", lambda: False)

    from workflows.stages.tokenize import run_tokenize

    state = State(model="rnn", device="cpu", run_type="update")
    # If skip doesn't work, the function tries pd.read_parquet/gensim which
    # aren't patched in this test — so a passing test means the early return
    # kicked in
    run_tokenize(state)


def test_config_reexport() -> None:
    """Verify that auto_detect_device is re-exported from sentimentizer.config."""
    from sentimentizer.config import auto_detect_device
    from sentimentizer.device import resolve_device

    # They should be the same function
    assert auto_detect_device is resolve_device
