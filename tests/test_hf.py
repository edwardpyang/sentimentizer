"""Tests for Hugging Face Hub weight downloading and pushing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentimentizer.config import HF_WEIGHTS_REPOS
from sentimentizer.hf import (
    _HF_MODEL_TYPES,
    download_weights,
    pull_model_from_hub,
    push_model_to_hub,
)


class TestHFWeightsRepos:
    """Test the HF_WEIGHTS_REPOS configuration mapping."""

    def test_rnn_repo_exists(self) -> None:
        assert "rnn" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["rnn"] == "ryeyoo/sentimentizer-rnn"

    def test_encoder_repo_exists(self) -> None:
        assert "encoder" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["encoder"] == "ryeyoo/sentimentizer-encoder"

    def test_decoder_repo_exists(self) -> None:
        assert "decoder" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["decoder"] == "ryeyoo/sentimentizer-decoder"

    def test_unknown_model_type_not_in_repos(self) -> None:
        assert "unknown" not in HF_WEIGHTS_REPOS

    def test_modernbert_repo_exists(self) -> None:
        assert "modernbert" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["modernbert"] == "ryeyoo/sentimentizer-modernbert"


class TestDownloadWeights:
    """Test the download_weights convenience function."""

    def test_download_weights_unknown_model_type(self) -> None:
        """download_weights returns None for unknown model types."""
        result = download_weights("unknown", "/tmp/unknown_weights.pth")
        assert result is None

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_success(self, mock_download: MagicMock) -> None:
        """download_weights copies downloaded file to local path."""
        import os
        import tempfile

        # Create a fake downloaded file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                dest_path = os.path.join(dest_dir, "rnn_weights.pth")
                result = download_weights("rnn", dest_path)

                assert result is not None
                assert Path(dest_path).exists()
                mock_download.assert_called_once_with(
                    repo_id="ryeyoo/sentimentizer-rnn",
                    filename="rnn_weights.pth",
                )
        finally:
            os.unlink(tmp_path)

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_entry_not_found(self, mock_download: MagicMock) -> None:
        """download_weights returns None when file not found on Hub."""
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("rnn_weights.pth not found")

        result = download_weights("rnn", "/tmp/rnn_weights.pth")
        assert result is None

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_network_error(self, mock_download: MagicMock) -> None:
        """download_weights returns None on network errors."""
        mock_download.side_effect = ConnectionError("No network")

        result = download_weights("rnn", "/tmp/rnn_weights.pth")
        assert result is None

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_creates_parent_dir(self, mock_download: MagicMock) -> None:
        """download_weights creates parent directories if they don't exist."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                nested_path = os.path.join(dest_dir, "nested", "dir", "rnn_weights.pth")
                result = download_weights("rnn", nested_path)

                assert result is not None
                assert Path(nested_path).exists()
        finally:
            os.unlink(tmp_path)


class TestPullModelFromHub:
    """Test the pull_model_from_hub function."""

    @patch("sentimentizer.hf.hf_hub_download")
    def test_pull_success(self, mock_download: MagicMock) -> None:
        """pull_model_from_hub returns True on successful download."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                dest_path = os.path.join(dest_dir, "rnn_weights.pth")
                result = pull_model_from_hub("ryeyoo/sentimentizer-rnn", "rnn", dest_path)
                assert result is True
        finally:
            os.unlink(tmp_path)

    @patch("sentimentizer.hf.hf_hub_download")
    def test_pull_entry_not_found(self, mock_download: MagicMock) -> None:
        """pull_model_from_hub returns False when file not found."""
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("not found")

        result = pull_model_from_hub("ryeyoo/sentimentizer-rnn", "rnn", "/tmp/weights.pth")
        assert result is False

    @patch("sentimentizer.hf.hf_hub_download")
    def test_pull_network_error(self, mock_download: MagicMock) -> None:
        """pull_model_from_hub returns False on network errors."""
        mock_download.side_effect = ConnectionError("No network")

        result = pull_model_from_hub("ryeyoo/sentimentizer-rnn", "rnn", "/tmp/weights.pth")
        assert result is False


class TestPushModelToHub:
    """Test the push_model_to_hub function."""

    @patch("sentimentizer.hf.HfApi")
    def test_push_file_not_found(self, mock_api_cls: MagicMock) -> None:
        """push_model_to_hub logs error and returns when file doesn't exist."""
        push_model_to_hub("/nonexistent/path.pth", "ryeyoo/sentimentizer-rnn", "rnn")
        # HfApi should not be instantiated if file doesn't exist
        mock_api_cls.assert_not_called()

    @patch("sentimentizer.hf.HfApi")
    def test_push_success(self, mock_api_cls: MagicMock) -> None:
        """push_model_to_hub uploads file to Hub."""
        import tempfile

        mock_api_instance = MagicMock()
        mock_api_cls.return_value = mock_api_instance

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            push_model_to_hub(tmp_path, "ryeyoo/sentimentizer-rnn", "rnn")
            mock_api_instance.upload_file.assert_called_once()
        finally:
            import os

            os.unlink(tmp_path)


class TestCreateModelCard:
    """Test the create_model_card function."""

    def test_rnn_model_card_without_metrics(self) -> None:
        """Should generate an RNN model card without tuning metrics."""
        from sentimentizer.hf import create_model_card

        card = create_model_card("rnn")
        assert "---\n" in card
        assert "sentiment-analysis" in card
        assert "text-classification" in card
        assert "RNN" in card
        assert "Bidirectional LSTM" in card or "bidirectional LSTM" in card
        assert "yelp.dictionary" in card
        assert "rnn_weights.pth" in card
        assert "## Usage" in card
        assert "## Files" in card
        assert "## Description" in card
        assert "## Training Data" in card

    def test_encoder_model_card_without_metrics(self) -> None:
        """Should generate an Encoder model card without tuning metrics."""
        from sentimentizer.hf import create_model_card

        card = create_model_card("encoder")
        assert "ENCODER" in card
        assert "Transformer Encoder" in card or "transformer encoder" in card.lower()
        assert "encoder_weights.pth" in card

    def test_decoder_model_card_without_metrics(self) -> None:
        """Should generate a Decoder model card without tuning metrics."""
        from sentimentizer.hf import create_model_card

        card = create_model_card("decoder")
        assert "DECODER" in card
        assert "Transformer Encoder-Decoder" in card or "encoder-decoder" in card.lower()
        assert "decoder_weights.pth" in card

    def test_model_card_with_tuning_result(self) -> None:
        """Should include metrics section when tuning_result is provided."""
        from sentimentizer.hf import create_model_card

        tuning_result = {
            "best_accuracy": 0.8923,
            "best_loss": 0.3145,
            "best_balanced_accuracy": 0.8765,
            "best_cohen_kappa": 0.7843,
            "best_mcc": 0.7812,
            "best_negative_f1": 0.8690,
            "best_neutral_f1": 0.7521,
            "best_positive_f1": 0.9156,
            "best_macro_f1": 0.8456,
            "best_config": {"lr": 0.001, "hidden_size": 256},
            "validation_passed": True,
            "mode": "standalone",
            "iterations_completed": 1,
            "converged": True,
            "elapsed_seconds": 42.5,
        }

        card = create_model_card("rnn", tuning_result=tuning_result)
        assert "## Metrics" in card
        assert "| Accuracy | 0.8923 |" in card
        assert "| Loss | 0.3145 |" in card
        assert "| Negative F1 | 0.8690 |" in card
        assert "✅ Passed" in card
        assert "Mode: `standalone`" in card
        assert "Iterations: 1" in card
        assert "Converged: Yes" in card
        assert "Elapsed: 42.5s" in card
        assert "Best Configuration" in card
        assert '"lr": 0.001' in card

    def test_model_card_failed_validation(self) -> None:
        """Should show failed validation status."""
        from sentimentizer.hf import create_model_card

        tuning_result = {
            "validation_passed": False,
        }
        card = create_model_card("rnn", tuning_result=tuning_result)
        assert "❌ Failed" in card

    def test_invalid_model_type_raises(self) -> None:
        """Should raise ValueError for unknown model type."""
        from sentimentizer.hf import create_model_card

        with pytest.raises(ValueError, match="Unknown model type"):
            create_model_card("transformer")

    def test_model_card_yaml_frontmatter(self) -> None:
        """Should include valid YAML frontmatter."""
        from sentimentizer.hf import create_model_card

        card = create_model_card("rnn")
        assert card.startswith("---\n")
        assert "language: en" in card
        assert "license: mit" in card
        assert "library_name: sentimentizer" in card
        assert "task: text-classification" in card
        assert "- rnn" in card

    def test_model_card_usage_section_has_download(self) -> None:
        """Should include download_weights in usage section."""
        from sentimentizer.hf import create_model_card

        card = create_model_card("encoder")
        assert "download_weights" in card
        assert "get_trained_model" in card
        assert "encoder" in card

    def test_model_card_partial_metrics(self) -> None:
        """Should handle tuning_result with only some metrics."""
        from sentimentizer.hf import create_model_card

        tuning_result = {
            "best_accuracy": 0.85,
            "validation_passed": True,
        }
        card = create_model_card("rnn", tuning_result=tuning_result)
        assert "| Accuracy | 0.8500 |" in card
        assert "✅ Passed" in card
        # Should not include metrics that are None/missing
        assert "Loss" not in card


class TestPushModelToHubWithModelCard:
    """Test that push_model_to_hub accepts and uploads tuning_result."""

    @patch("sentimentizer.hf.HfApi")
    @patch("sentimentizer.hf._upload_model_card")
    def test_push_with_tuning_result_uploads_model_card(
        self, mock_upload_card: MagicMock, mock_api_cls: MagicMock
    ) -> None:
        """push_model_to_hub should call _upload_model_card when tuning_result is provided."""
        import tempfile

        from sentimentizer.hf import push_model_to_hub

        mock_api_instance = MagicMock()
        mock_api_cls.return_value = mock_api_instance

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            tuning_result = {"best_accuracy": 0.89, "validation_passed": True}
            push_model_to_hub(
                local_path=tmp_path,
                model_type="rnn",
                tuning_result=tuning_result,
            )
            # _upload_model_card should have been called
            mock_upload_card.assert_called_once()
            call_args = mock_upload_card.call_args
            assert call_args[0][2] == "rnn"  # model_type
            assert call_args[0][3] == tuning_result  # tuning_result
        finally:
            import os

            os.unlink(tmp_path)


class TestGetTrainedModelHFDownload:
    """Test that get_trained_model falls back to HF Hub when local weights missing."""

    def test_rnn_downloads_from_hub_when_local_missing(self) -> None:
        """RNN get_trained_model downloads from HF Hub when local file missing."""
        from unittest.mock import ANY, patch

        import torch

        from sentimentizer.models.rnn import get_trained_model

        # Create fake weights that match the RNN architecture
        emb_weights = torch.zeros(100, 100)
        from sentimentizer.models.rnn import RNN

        model = RNN(emb_weights=emb_weights)
        fake_state_dict = model.state_dict()

        with (
            patch("sentimentizer.models.rnn.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights") as mock_download,
            patch("sentimentizer.models.rnn.torch.load", return_value=fake_state_dict),
        ):
            # Simulate successful download
            mock_download.return_value = Path("/fake/rnn_weights.pth")

            model = get_trained_model("cpu")
            assert isinstance(model, RNN)
            mock_download.assert_called_once_with(
                "rnn", mock_download.call_args[0][1], dict_path=ANY
            )

    def test_rnn_raises_when_hub_fails(self) -> None:
        """RNN get_trained_model raises FileNotFoundError when Hub download fails."""
        from unittest.mock import patch

        from sentimentizer.models.rnn import get_trained_model

        with (
            patch("sentimentizer.models.rnn.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights", return_value=None),
            pytest.raises(FileNotFoundError, match="Weights file not found"),
        ):
            get_trained_model("cpu")

    def test_encoder_downloads_from_hub_when_local_missing(self) -> None:
        """Encoder get_trained_model downloads from HF Hub when local file missing."""
        from unittest.mock import ANY, patch

        import torch

        from sentimentizer.models.encoder import Encoder, get_trained_model

        # Create fake weights that match the Encoder architecture
        emb_weights = torch.zeros(100, 100)
        model = Encoder(emb_weights=emb_weights)
        fake_state_dict = model.state_dict()

        with (
            patch("sentimentizer.models.encoder.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights") as mock_download,
            patch("sentimentizer.models.encoder.torch.load", return_value=fake_state_dict),
        ):
            mock_download.return_value = Path("/fake/encoder_weights.pth")

            model = get_trained_model("cpu")
            assert isinstance(model, Encoder)
            mock_download.assert_called_once_with(
                "encoder", mock_download.call_args[0][1], dict_path=ANY
            )

    def test_decoder_downloads_from_hub_when_local_missing(self) -> None:
        """Decoder get_trained_model downloads from HF Hub when local file missing."""
        from unittest.mock import ANY, patch

        import torch

        from sentimentizer.models.decoder import Decoder, get_trained_model

        # Create fake weights that match the Decoder architecture
        emb_weights = torch.zeros(100, 100)
        model = Decoder(emb_weights=emb_weights)
        fake_state_dict = model.state_dict()

        with (
            patch("sentimentizer.models.decoder.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights") as mock_download,
            patch("sentimentizer.models.decoder.torch.load", return_value=fake_state_dict),
        ):
            mock_download.return_value = Path("/fake/decoder_weights.pth")

            model = get_trained_model("cpu")
            assert isinstance(model, Decoder)
            mock_download.assert_called_once_with(
                "decoder", mock_download.call_args[0][1], dict_path=ANY
            )


class TestHFModelTypes:
    """Test the _HF_MODEL_TYPES set."""

    def test_modernbert_is_hf_model(self) -> None:
        assert "modernbert" in _HF_MODEL_TYPES

    def test_gloves_are_not_hf_models(self) -> None:
        assert "rnn" not in _HF_MODEL_TYPES
        assert "encoder" not in _HF_MODEL_TYPES
        assert "decoder" not in _HF_MODEL_TYPES


class TestModernBERTModelCard:
    """Test model card generation for ModernBERT."""

    def test_modernbert_model_card(self) -> None:
        from sentimentizer.hf import create_model_card

        card = create_model_card("modernbert")
        assert "---\n" in card
        assert "MODERNBERT" in card
        assert "ModernBERT" in card
        assert "modernbert_weights.pth" in card
        assert "backbone/" in card
        assert "sentiment-analysis" in card

    def test_modernbert_model_card_no_dictionary(self) -> None:
        from sentimentizer.hf import create_model_card

        card = create_model_card("modernbert")
        assert "yelp.dictionary" not in card
        assert "GloVe" not in card

    def test_modernbert_model_card_has_backbone_files(self) -> None:
        from sentimentizer.hf import create_model_card

        card = create_model_card("modernbert")
        assert "safetensors" in card
        assert "tokenizer.json" in card

    def test_modernbert_model_card_usage_section(self) -> None:
        from sentimentizer.hf import create_model_card

        card = create_model_card("modernbert")
        assert "new_modernbert_model" in card
        assert "download_weights" in card
        assert "SentimentPredictor" in card


class TestPushModelToHubBackbone:
    """Test that push_model_to_hub uploads backbone for HF models."""

    @patch("sentimentizer.hf._upload_backbone_dir")
    @patch("sentimentizer.hf.HfApi")
    def test_push_modernbert_uploads_backbone(
        self, mock_api_cls: MagicMock, mock_upload_backbone: MagicMock
    ) -> None:
        import tempfile

        mock_api_instance = MagicMock()
        mock_api_cls.return_value = mock_api_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "modernbert_weights.pth"
            weights_path.write_bytes(b"fake weights")
            backbone_dir = Path(tmpdir) / "backbone"
            backbone_dir.mkdir()
            (backbone_dir / "model.safetensors").write_bytes(b"fake backbone")
            (backbone_dir / "config.json").write_text("{}")

            push_model_to_hub(
                local_path=str(weights_path),
                model_type="modernbert",
                backbone_path=str(backbone_dir),
            )

            mock_upload_backbone.assert_called_once()
            call_args = mock_upload_backbone.call_args
            assert call_args[0][0] == mock_api_instance
            assert call_args[0][1] == backbone_dir
            assert call_args[0][2] == "ryeyoo/sentimentizer-modernbert"

    @patch("sentimentizer.hf._upload_backbone_dir")
    @patch("sentimentizer.hf.HfApi")
    def test_push_rnn_does_not_upload_backbone(
        self, mock_api_cls: MagicMock, mock_upload_backbone: MagicMock
    ) -> None:
        import tempfile

        mock_api_instance = MagicMock()
        mock_api_cls.return_value = mock_api_instance

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            push_model_to_hub(tmp_path, "ryeyoo/sentimentizer-rnn", "rnn")
            mock_upload_backbone.assert_not_called()
        finally:
            import os

            os.unlink(tmp_path)


class TestDownloadBackboneDir:
    """Test _download_backbone_dir functionality."""

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_modernbert_downloads_backbone(self, mock_download: MagicMock) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                dest_path = os.path.join(dest_dir, "hf_weights", "head.pth")
                result = download_weights("modernbert", dest_path)

                assert result is not None
                assert mock_download.call_count >= 2
                first_call_filename = mock_download.call_args_list[0][1].get(
                    "filename",
                    (
                        mock_download.call_args_list[0][0][0]
                        if mock_download.call_args_list[0][0]
                        else ""
                    ),
                )
                assert first_call_filename == "modernbert_weights.pth"
        finally:
            os.unlink(tmp_path)

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_rnn_no_backbone(self, mock_download: MagicMock) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                dest_path = os.path.join(dest_dir, "rnn_weights.pth")
                result = download_weights("rnn", dest_path)

                assert result is not None
                mock_download.assert_called_once()
        finally:
            os.unlink(tmp_path)


class TestGetTrainedTokenizerDownload:
    """Test that get_trained_tokenizer downloads the dictionary if missing."""

    @patch("sentimentizer.tokenizer.Path.exists")
    @patch("sentimentizer.tokenizer.corpora.Dictionary.load")
    @patch("sentimentizer.hf.download_weights")
    def test_tokenizer_exists_no_download(
        self, mock_download: MagicMock, mock_load: MagicMock, mock_exists: MagicMock
    ) -> None:
        """When dictionary exists, load it directly without downloading."""
        from sentimentizer.tokenizer import get_trained_tokenizer

        mock_exists.return_value = True
        mock_load.return_value = MagicMock()

        tokenizer = get_trained_tokenizer()

        assert tokenizer is not None
        mock_exists.assert_called()
        mock_download.assert_not_called()
        mock_load.assert_called_once()

    @patch("sentimentizer.tokenizer.Path.exists")
    @patch("sentimentizer.tokenizer.corpora.Dictionary.load")
    @patch("sentimentizer.hf.download_weights")
    def test_tokenizer_missing_download_success(
        self, mock_download: MagicMock, mock_load: MagicMock, mock_exists: MagicMock
    ) -> None:
        """When dictionary is missing, it should attempt to download and then load."""
        from sentimentizer.tokenizer import get_trained_tokenizer

        # First exists() call returns False (missing), second returns True (downloaded)
        mock_exists.side_effect = [False, True]
        mock_load.return_value = MagicMock()

        tokenizer = get_trained_tokenizer()

        assert tokenizer is not None
        assert mock_exists.call_count == 2
        mock_download.assert_called_once()
        # Verify it was called with model_type="encoder", dict_path, and local_path
        assert mock_download.call_args[1]["model_type"] == "encoder"
        assert "yelp.dictionary" in str(mock_download.call_args[1]["dict_path"])
        assert "encoder_weights.pth" in str(mock_download.call_args[1]["local_path"])
        mock_load.assert_called_once()

    @patch("sentimentizer.tokenizer.Path.exists")
    @patch("sentimentizer.tokenizer.corpora.Dictionary.load")
    @patch("sentimentizer.hf.download_weights")
    def test_tokenizer_missing_download_exception(
        self, mock_download: MagicMock, mock_load: MagicMock, mock_exists: MagicMock
    ) -> None:
        """When download raises an exception, raise FileNotFoundError."""
        from sentimentizer.tokenizer import get_trained_tokenizer

        mock_exists.return_value = False  # Always missing
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(FileNotFoundError, match="Dictionary file not found"):
            get_trained_tokenizer()

        mock_download.assert_called_once()
        mock_load.assert_not_called()

    @patch("sentimentizer.tokenizer.Path.exists")
    @patch("sentimentizer.tokenizer.corpora.Dictionary.load")
    @patch("sentimentizer.hf.download_weights")
    def test_tokenizer_missing_download_still_missing(
        self, mock_download: MagicMock, mock_load: MagicMock, mock_exists: MagicMock
    ) -> None:
        """When download returns but file is still missing, raise FileNotFoundError."""
        from sentimentizer.tokenizer import get_trained_tokenizer

        mock_exists.return_value = False  # Always missing (even after download)
        mock_download.return_value = None

        with pytest.raises(FileNotFoundError, match="Dictionary file not found"):
            get_trained_tokenizer()

        mock_download.assert_called_once()
        mock_load.assert_not_called()


class TestPredictorConditionalTokenizer:
    """Test that SentimentPredictor skips the GloVe tokenizer for HF models."""

    @patch("sentimentizer.predictor.get_trained_model")
    @patch("sentimentizer.predictor.get_trained_tokenizer")
    @patch("sentimentizer.predictor.auto_detect_device", return_value="cpu")
    @patch("sentimentizer.predictor._ROUTER_AVAILABLE", False)
    def test_encoder_loads_glove_tokenizer(
        self,
        _mock_device: MagicMock,
        mock_get_tokenizer: MagicMock,
        mock_get_model: MagicMock,
    ) -> None:
        """GloVe models (encoder/rnn/decoder) should call get_trained_tokenizer."""
        from sentimentizer.predictor import SentimentPredictor

        mock_get_model.side_effect = FileNotFoundError("no weights")

        SentimentPredictor(model_name="encoder")

        mock_get_tokenizer.assert_called_once()

    @patch("sentimentizer.predictor.get_trained_model")
    @patch("sentimentizer.predictor.get_trained_tokenizer")
    @patch("sentimentizer.predictor.auto_detect_device", return_value="cpu")
    @patch("sentimentizer.predictor._ROUTER_AVAILABLE", False)
    def test_modernbert_skips_glove_tokenizer(
        self,
        _mock_device: MagicMock,
        mock_get_tokenizer: MagicMock,
        mock_get_model: MagicMock,
    ) -> None:
        """ModernBERT should NOT call get_trained_tokenizer (uses its own HF tokenizer)."""
        from sentimentizer.predictor import SentimentPredictor

        mock_get_model.side_effect = FileNotFoundError("no weights")

        predictor = SentimentPredictor(model_name="modernbert")

        mock_get_tokenizer.assert_not_called()
        assert predictor.tokenizer is None


class TestRouterAutoDownload:
    """Test that SentimentPredictor auto-downloads the router model from HF Hub.

    The router loads lazily — download happens on the first classify() call,
    not at construction time.
    """

    @patch("sentimentizer.predictor.get_trained_model")
    @patch("sentimentizer.predictor.get_trained_tokenizer")
    @patch("sentimentizer.predictor.auto_detect_device", return_value="cpu")
    @patch("sentimentizer.predictor._ROUTER_AVAILABLE", True)
    def test_router_downloads_when_missing(
        self,
        _mock_device: MagicMock,
        _mock_tokenizer: MagicMock,
        mock_get_model: MagicMock,
    ) -> None:
        """Only the default food router downloads lazily when its path is missing."""
        from sentimentizer.config import HF_ROUTER_REPO
        from sentimentizer.predictor import SentimentPredictor
        from sentimentizer.router.config import DEFAULT_ROUTER_MODEL_PATH

        mock_get_model.side_effect = FileNotFoundError("no weights")
        with (
            patch("sentimentizer.predictor.Path.exists", return_value=False),
            patch("huggingface_hub.snapshot_download") as mock_snap,
        ):
            predictor = SentimentPredictor(model_name="encoder")
            mock_snap.assert_not_called()
            with pytest.raises(RuntimeError):
                predictor.classify("hello")
            mock_snap.assert_called_once_with(
                repo_id=HF_ROUTER_REPO,
                local_dir=DEFAULT_ROUTER_MODEL_PATH,
            )

    @patch("sentimentizer.predictor.get_trained_model")
    @patch("sentimentizer.predictor.get_trained_tokenizer")
    @patch("sentimentizer.predictor.auto_detect_device", return_value="cpu")
    @patch("sentimentizer.predictor._ROUTER_AVAILABLE", True)
    def test_router_download_failure_degrades_gracefully(
        self,
        _mock_device: MagicMock,
        _mock_tokenizer: MagicMock,
        mock_get_model: MagicMock,
    ) -> None:
        """When router download fails, predictor should still init without crashing."""
        import tempfile

        from sentimentizer.predictor import SentimentPredictor

        mock_get_model.side_effect = FileNotFoundError("no weights")

        with tempfile.TemporaryDirectory() as tmpdir:
            router_path = Path(tmpdir) / "nonexistent_router"

            with (
                patch(
                    "huggingface_hub.snapshot_download",
                    side_effect=ConnectionError("No network"),
                ) as download,
                patch("sentimentizer.predictor.DEFAULT_ROUTER_MODEL_PATH", str(router_path)),
            ):
                predictor = SentimentPredictor(
                    model_name="encoder",
                    router_model_path=str(router_path),
                )
                # Construction succeeds without loading the router.
                assert predictor.router is None
                assert predictor.router_error is None
                # Lazy load on first classify() captures the error.
                with pytest.raises(RuntimeError):
                    predictor.classify("hello")
                assert predictor.router is None
                assert predictor.router_error is not None
                with pytest.raises(RuntimeError):
                    predictor.classify("hello again")
                download.assert_called_once()

    @patch("sentimentizer.predictor.get_trained_model")
    @patch("sentimentizer.predictor.get_trained_tokenizer")
    @patch("sentimentizer.predictor.auto_detect_device", return_value="cpu")
    @patch("sentimentizer.predictor._ROUTER_AVAILABLE", False)
    def test_router_no_download_without_sentence_transformers(
        self,
        _mock_device: MagicMock,
        _mock_tokenizer: MagicMock,
        mock_get_model: MagicMock,
    ) -> None:
        """When sentence-transformers is not installed, router download is skipped."""
        from sentimentizer.predictor import SentimentPredictor

        mock_get_model.side_effect = FileNotFoundError("no weights")

        with patch("huggingface_hub.snapshot_download") as mock_snap:
            predictor = SentimentPredictor(model_name="encoder")
            mock_snap.assert_not_called()
            assert predictor.router is None
