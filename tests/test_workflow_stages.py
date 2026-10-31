"""Tests for thin workflow stages and helpers that had no direct coverage."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from workflows.helpers import _get_model_config, _load_model, _parquet_row_count, _remove_path
from workflows.stages.export import run_export
from workflows.stages.extract import run_extract
from workflows.stages.hf import run_hf_pull, run_hf_push


class TestPathHelpers:
    def test_remove_path_handles_file_directory_and_missing(self, tmp_path):
        file_path = tmp_path / "file"
        file_path.write_text("x")
        directory = tmp_path / "dir"
        (directory / "nested").mkdir(parents=True)

        for path in (file_path, directory, tmp_path / "missing"):
            _remove_path(str(path))
            assert not path.exists()

    def test_parquet_row_count_for_file_directory_and_missing(self, tmp_path):
        frame = pd.DataFrame({"a": [1, 2, 3]})
        single = tmp_path / "single.parquet"
        frame.to_parquet(single)
        parts = tmp_path / "parts"
        parts.mkdir()
        frame.to_parquet(parts / "0.parquet")
        frame.to_parquet(parts / "1.parquet")

        assert _parquet_row_count(str(single)) == 3
        assert _parquet_row_count(str(parts)) == 6
        assert _parquet_row_count(str(tmp_path / "missing")) == 0


class TestModelLoading:
    @pytest.mark.parametrize("model", ["rnn", "encoder", "decoder", "modernbert"])
    def test_config_exists_for_each_model(self, model):
        assert _get_model_config(model) is not None

    def test_unknown_model_config_raises(self):
        with pytest.raises(ValueError, match="no matching model config"):
            _get_model_config("mystery")

    def test_missing_dictionary_is_reported_before_building(self, tmp_path):
        state = SimpleNamespace(model="encoder", run_type="new")
        with (
            patch("sentimentizer.config.DriverConfig.files.dictionary_file_path", tmp_path / "x"),
            pytest.raises(FileNotFoundError, match="Dictionary file not found"),
        ):
            _load_model(state, "cpu")

    def test_invalid_run_type_raises(self):
        state = SimpleNamespace(model="modernbert", run_type="sideways")
        with pytest.raises(ValueError, match="invalid run_type"):
            _load_model(state, "cpu")

    def test_new_glove_model_receives_freeze_flag(self, tmp_path):
        dictionary = tmp_path / "dict"
        dictionary.write_text("x")
        state = SimpleNamespace(model="rnn", run_type="new")
        with (
            patch("sentimentizer.config.DriverConfig.files.dictionary_file_path", dictionary),
            patch("sentimentizer.models.rnn.new_model") as new_model,
        ):
            model = _load_model(state, "cpu", freeze_embeddings=False)

        assert model is new_model.return_value
        assert new_model.call_args.kwargs["freeze_embeddings"] is False
        assert new_model.call_args.kwargs["dict_path"] == dictionary

    def test_update_loads_trained_glove_model(self, tmp_path):
        dictionary = tmp_path / "dict"
        dictionary.write_text("x")
        state = SimpleNamespace(model="encoder", run_type="update")
        with (
            patch("sentimentizer.config.DriverConfig.files.dictionary_file_path", dictionary),
            patch("sentimentizer.models.encoder.get_trained_model") as trained,
        ):
            assert _load_model(state, "cpu") is trained.return_value
        assert trained.call_args.kwargs["device"] == "cpu"


class TestExportStage:
    @pytest.mark.parametrize("passed", [True, False])
    def test_returns_pipeline_results(self, passed, tmp_path):
        results = {
            "onnx_path": "x.onnx",
            "validation": {"passed": passed, "max_diff": 0.1, "tolerance": 0.01},
        }
        with patch("workflows.stages.export.export_pipeline", return_value=results) as pipeline:
            assert run_export(None, "rnn", quantize=False, output_dir=str(tmp_path)) is results
        assert pipeline.call_args.kwargs["model_type"] == "rnn"
        assert pipeline.call_args.kwargs["quantize"] is False


class TestExtractStage:
    @pytest.fixture
    def files(self, tmp_path):
        files = SimpleNamespace(
            raw_reviews_file_path=str(tmp_path / "out" / "reviews.parquet"),
            archive_file_path="archive",
            raw_file_path="raw",
        )
        with (
            patch("sentimentizer.config.DriverConfig.files", files),
            patch("workflows.stages.extract._ensure_ray_initialized"),
        ):
            yield files

    def test_skips_when_enough_rows_exist(self, files):
        with (
            patch("workflows.stages.extract._parquet_row_count", return_value=10),
            patch("sentimentizer.extractor.extract_data_local") as extract,
            patch("workflows.lifecycle.is_ray_available", return_value=False),
        ):
            run_extract(MagicMock(), stop=5)
        extract.assert_not_called()

    def test_local_path_writes_parquet(self, files):
        frame = pd.DataFrame({"text": ["a"], "stars": [5]})
        with (
            patch("sentimentizer.extractor.extract_data_local", return_value=frame),
            patch("workflows.lifecycle.is_ray_available", return_value=False),
        ):
            run_extract(MagicMock(), stop=1)
        assert len(pd.read_parquet(files.raw_reviews_file_path)) == 1

    def test_ray_path_writes_dataset(self, files):
        dataset = MagicMock()
        with (
            patch("sentimentizer.extractor.extract_data_ray", return_value=dataset),
            patch("workflows.lifecycle.is_ray_available", return_value=True),
        ):
            run_extract(MagicMock(), stop=1)
        dataset.write_parquet.assert_called_once_with(files.raw_reviews_file_path)


class TestHubStages:
    STATE = SimpleNamespace(model="modernbert")

    def test_failed_push_raises_so_the_command_exits_nonzero(self):
        with (
            patch("sentimentizer.hf.push_model_to_hub", return_value=False),
            pytest.raises(RuntimeError, match="Failed to push modernbert"),
        ):
            run_hf_push(self.STATE, repo_id=None)

    def test_successful_push_skips_dictionary_for_hf_model(self):
        with patch("sentimentizer.hf.push_model_to_hub", return_value=True) as push:
            run_hf_push(self.STATE, repo_id="me/repo")
        assert push.call_args.kwargs["repo_id"] == "me/repo"
        assert push.call_args.kwargs["dict_path"] is None

    def test_failed_pull_raises_so_the_command_exits_nonzero(self):
        with (
            patch("sentimentizer.hf.download_weights", return_value=None),
            pytest.raises(RuntimeError, match="Failed to pull modernbert"),
        ):
            run_hf_pull(self.STATE, repo_id=None)

    def test_successful_pull_includes_dictionary_for_glove_model(self):
        state = SimpleNamespace(model="rnn")
        with patch("sentimentizer.hf.download_weights", return_value="weights") as download:
            run_hf_pull(state, repo_id=None)
        assert download.call_args.kwargs["dict_path"] is not None
