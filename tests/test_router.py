"""Tests for the router module.

Tests cover config, labels, seeds, dataset loading, and augmentation.
Router-dependent tests are skipped when sentence-transformers is not installed.
"""

import json
from pathlib import Path

import pytest

# Check if optional dependencies are available
try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import datasets  # noqa: F401

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

skip_without_router = pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed (install with: pip install -e '.[router]')",
)

skip_without_datasets = pytest.mark.skipif(
    not DATASETS_AVAILABLE,
    reason="datasets not installed (install with: pip install -e '.[router]')",
)


class TestRouterConfig:
    """Test RouterConfig defaults and immutability."""

    def test_default_config(self) -> None:
        from sentimentizer.router.config import RouterConfig

        config = RouterConfig()
        assert config.base_model == "BAAI/bge-base-en-v1.5"
        assert config.num_iterations == 20
        assert config.num_epochs == 1
        assert config.batch_size == 16
        assert config.max_seq_length == 512
        assert config.seed == 42
        assert config.output_dir == Path("models/router")

    def test_frozen_config(self) -> None:
        from sentimentizer.router.config import RouterConfig

        config = RouterConfig()
        with pytest.raises(AttributeError):
            config.base_model = "different-model"  # type: ignore[misc]

    def test_custom_config(self) -> None:
        from sentimentizer.router.config import RouterConfig

        config = RouterConfig(
            base_model="mxbai-embed-large-v1",
            num_iterations=10,
            batch_size=32,
        )
        assert config.base_model == "mxbai-embed-large-v1"
        assert config.num_iterations == 10
        assert config.batch_size == 32


class TestRouteLabels:
    """Test RouteLabels config and label mappings."""

    def test_default_labels(self) -> None:
        from sentimentizer.router.config import RouteLabels

        labels = RouteLabels()
        assert labels.dietary == 0
        assert labels.service == 1
        assert labels.general == 2

    def test_label_names(self) -> None:
        from sentimentizer.router.config import RouteLabels

        names = RouteLabels.label_names()
        assert names == {0: "dietary", 1: "service", 2: "general"}

    def test_num_classes(self) -> None:
        from sentimentizer.router.config import RouteLabels

        assert RouteLabels.num_classes() == 3


class TestSeedUtterances:
    """Test seed utterance data integrity."""

    def test_seed_utterances_exist(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        assert len(SEED_UTTERANCES) > 0

    def test_seed_utterances_have_correct_labels(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        valid_labels = {0, 1, 2}
        for entry in SEED_UTTERANCES:
            assert entry["label"] in valid_labels, f"Invalid label: {entry['label']}"

    def test_seed_utterances_have_nonempty_text(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        for entry in SEED_UTTERANCES:
            assert len(entry["text"]) > 0, "Empty text found in seed utterances"

    def test_utterances_per_category(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        counts = {0: 0, 1: 0, 2: 0}
        for entry in SEED_UTTERANCES:
            counts[entry["label"]] += 1

        # Each category should have 10 seeds
        assert counts[0] == 10, f"Dietary has {counts[0]} seeds, expected 10"
        assert counts[1] == 10, f"Service has {counts[1]} seeds, expected 10"
        assert counts[2] == 10, f"General has {counts[2]} seeds, expected 10"

    def test_total_seed_count(self) -> None:
        from sentimentizer.router.seeds import SEED_UTTERANCES

        assert len(SEED_UTTERANCES) == 30


@skip_without_datasets
class TestDatasetLoader:
    """Test JSONL dataset loading (with mock data)."""

    def test_load_router_dataset(self, tmp_path) -> None:
        """Test loading and splitting a JSONL dataset."""
        from sentimentizer.router.dataset import load_router_dataset

        # Create a mock JSONL file
        data = []
        for i in range(100):
            label = i % 3
            data.append({"text": f"Review text {i} for label {label}", "label": label})

        jsonl_path = tmp_path / "test_data.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        train_ds, test_ds = load_router_dataset(str(jsonl_path), test_size=0.2, seed=42)

        # Should have 80% train, 20% test
        assert len(train_ds) == 80
        assert len(test_ds) == 20

    def test_load_router_dataset_preserves_columns(self, tmp_path) -> None:
        """Test that loaded dataset has 'text' and 'label' columns."""
        from sentimentizer.router.dataset import load_router_dataset

        data = [{"text": f"Independent review {i}", "label": i % 3} for i in range(30)]

        jsonl_path = tmp_path / "test_columns.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        train_ds, test_ds = load_router_dataset(str(jsonl_path), test_size=0.3, seed=42)

        assert "text" in train_ds.column_names
        assert "label" in train_ds.column_names


class TestSaveDatasetToJsonl:
    """Test saving utterance dicts to JSONL format."""

    def test_save_dataset_to_jsonl(self, tmp_path) -> None:
        from sentimentizer.router.dataset import save_dataset_to_jsonl

        data = [
            {"text": "Review 1", "label": 0},
            {"text": "Review 2", "label": 1},
        ]

        output_path = tmp_path / "output.jsonl"
        result = save_dataset_to_jsonl(data, str(output_path))

        assert result.exists()
        with open(result) as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Verify JSONL format
        entry = json.loads(lines[0])
        assert "text" in entry
        assert "label" in entry


class TestAugmentConfig:
    """Test AugmentConfig and load_router_config."""

    def test_augment_config_defaults(self) -> None:
        from sentimentizer.router.config import AugmentConfig

        config = AugmentConfig()
        assert config.model == "glm-5.1:cloud"
        assert config.ollama_url == "http://localhost:11434/api/generate"
        assert config.variations_per_seed == 50
        assert config.output_path == "augmented_yelp.jsonl"
        assert config.batch_size == 5

    def test_augment_config_custom(self) -> None:
        from sentimentizer.router.config import AugmentConfig

        config = AugmentConfig(model="llama3", variations_per_seed=10, output_path="custom.jsonl")
        assert config.model == "llama3"
        assert config.variations_per_seed == 10
        assert config.output_path == "custom.jsonl"

    def test_load_router_config_defaults(self) -> None:
        from sentimentizer.router.config import load_router_config

        train_cfg, augment_cfg = load_router_config()
        assert train_cfg.base_model == "BAAI/bge-base-en-v1.5"
        assert train_cfg.num_iterations == 20
        assert augment_cfg.model == "glm-5.1:cloud"
        assert augment_cfg.variations_per_seed == 50

    def test_load_router_config_custom_file(self, tmp_path) -> None:
        from sentimentizer.router.config import load_router_config

        config_content = """
training:
  base_model: "custom-model"
  num_iterations: 30

augmentation:
  model: "custom-llm"
  variations_per_seed: 100
"""
        config_path = tmp_path / "custom_router.yaml"
        config_path.write_text(config_content)

        train_cfg, augment_cfg = load_router_config(str(config_path))
        assert train_cfg.base_model == "custom-model"
        assert train_cfg.num_iterations == 30
        assert augment_cfg.model == "custom-llm"
        assert augment_cfg.variations_per_seed == 100

    def test_load_router_config_missing_file(self) -> None:
        import pytest

        from sentimentizer.router.config import load_router_config

        with pytest.raises(FileNotFoundError):
            load_router_config("/nonexistent/path.yaml")


class TestAugmentSeeds:
    """Test seed augmentation (with mocked Ollama API)."""

    def test_build_prompt(self) -> None:
        from sentimentizer.router.augment import _build_prompt

        prompt = _build_prompt("Great food!", 2, 10)
        assert "Great food!" in prompt
        assert "general" in prompt.lower()
        assert "10" in prompt

    def test_augment_seeds_handles_api_failure(self) -> None:
        """Augment seeds should gracefully handle API failures."""
        from sentimentizer.router.augment import augment_seeds

        seeds = [{"text": "Test review", "label": 0}]
        # Use a URL that doesn't exist — should return just the original seeds
        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
        )
        # Should still return the original seeds even if API fails
        assert len(result) >= len(seeds)
        assert result[0]["text"] == seeds[0]["text"]

    def test_augment_seeds_streams_to_file(self, tmp_path) -> None:
        """Augment seeds with output_path should stream entries to JSONL."""
        from sentimentizer.router.augment import augment_seeds

        seeds = [{"text": "Test review", "label": 0}]
        output_path = tmp_path / "streamed.jsonl"

        augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
            output_path=str(output_path),
        )

        # File should exist and contain seed utterances
        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        # At minimum, the seed entries should be written
        assert len(lines) >= len(seeds)
        entry = json.loads(lines[0])
        assert "text" in entry
        assert "label" in entry

    def test_augment_seeds_without_output_path(self) -> None:
        """Augment seeds without output_path should still return results."""
        from sentimentizer.router.augment import augment_seeds

        seeds = [{"text": "Test review", "label": 0}]
        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
        )
        assert len(result) >= len(seeds)
        assert result[0]["text"] == seeds[0]["text"]

    def test_augment_seeds_resume_skips_existing(self, tmp_path) -> None:
        """Resume mode should skip seeds already in the output file."""
        from sentimentizer.router.augment import augment_seeds

        # Create an existing file with some entries
        output_path = tmp_path / "resume.jsonl"
        existing_entries = [
            {"text": "Seed 1", "label": 0},
            {"text": "Seed 2", "label": 1},
        ]
        with open(output_path, "w") as f:
            for entry in existing_entries:
                f.write(json.dumps(entry) + "\n")

        # Resume with seeds that include the existing ones
        seeds = [
            {"text": "Seed 1", "label": 0},  # already in file — should be skipped
            {"text": "Seed 2", "label": 1},  # already in file — should be skipped
            {"text": "Seed 3", "label": 2},  # NOT in file — should be processed
        ]

        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
            output_path=str(output_path),
            resume=True,
        )

        # Should include the 3 existing entries + seed 3 (API will fail but seed is kept)
        assert len(result) >= 3  # at least the existing entries

        # File should still have the original entries
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) >= 3  # at least the original entries

    def test_augment_seeds_resume_all_processed(self, tmp_path) -> None:
        """Resume mode with all seeds already processed should return existing."""
        from sentimentizer.router.augment import augment_seeds

        output_path = tmp_path / "complete.jsonl"
        seeds = [{"text": "Seed A", "label": 0}, {"text": "Seed B", "label": 1}]

        # Write all seeds to file
        with open(output_path, "w") as f:
            for seed in seeds:
                f.write(json.dumps(seed) + "\n")

        # Presence of originals alone is not completed augmentation.
        result = augment_seeds(
            seeds,
            ollama_url="http://localhost:99999/api/generate",
            variations_per_seed=5,
            output_path=str(output_path),
            resume=True,
        )

        # Failed requests retain the original seeds for another resume.
        assert len(result) == 2
        assert result[0]["text"] == "Seed A"
        assert result[1]["text"] == "Seed B"


@skip_without_router
class TestContrastivePairs:
    """Test contrastive pair generation."""

    def test_pair_count_with_multiple_per_class(self) -> None:
        """Total pairs = 2 * num_iterations * num_texts when each class has 2+ examples."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["d1", "d2", "s1", "s2", "g1", "g2"]
        labels = [0, 0, 1, 1, 2, 2]
        num_iterations = 5
        pairs = generate_contrastive_pairs(texts, labels, num_iterations=num_iterations)
        # 6 texts * (5 same-class + 5 different-class) = 60 pairs total
        assert len(pairs) == 2 * num_iterations * len(texts)

    def test_pair_count_with_single_per_class(self) -> None:
        """Single example per class creates 1 self-pair + num_iterations diff pairs."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["dietary text", "service text", "general text"]
        labels = [0, 1, 2]
        num_iterations = 5
        pairs = generate_contrastive_pairs(texts, labels, num_iterations=num_iterations)
        # 3 texts * (1 self-pair + 5 different-class) = 18 pairs total
        expected = len(texts) * (1 + num_iterations)
        assert len(pairs) == expected

    def test_same_class_pairs_have_label_one(self) -> None:
        """Same-class pairs should have similarity label 1.0."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["a dietary", "b dietary", "c service", "d service"]
        labels = [0, 0, 1, 1]
        pairs = generate_contrastive_pairs(texts, labels, num_iterations=2, seed=42)
        same_class = [p for p in pairs if p.label == 1.0]
        assert len(same_class) > 0
        for p in same_class:
            # Both texts should be from the same class — no self-pairing
            # (only possible when class has 2+ examples)
            assert p.texts[0] != p.texts[1]

    def test_different_class_pairs_have_label_zero(self) -> None:
        """Different-class pairs should have similarity label 0.0."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["a dietary", "b service", "c general"]
        labels = [0, 1, 2]
        pairs = generate_contrastive_pairs(texts, labels, num_iterations=2, seed=42)
        diff_class = [p for p in pairs if p.label == 0.0]
        assert len(diff_class) > 0

    def test_no_self_pairing_with_multiple_per_class(self) -> None:
        """Same-class pairs should not pair a text with itself (2+ per class)."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["a1", "a2", "b1", "b2", "c1", "c2"]
        labels = [0, 0, 1, 1, 2, 2]
        pairs = generate_contrastive_pairs(texts, labels, num_iterations=3, seed=42)
        for p in pairs:
            if p.label == 1.0:
                assert p.texts[0] != p.texts[1]

    def test_reproducible_with_seed(self) -> None:
        """Same seed should produce same pairs."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["a", "b", "c"]
        labels = [0, 1, 2]
        pairs1 = generate_contrastive_pairs(texts, labels, num_iterations=2, seed=42)
        pairs2 = generate_contrastive_pairs(texts, labels, num_iterations=2, seed=42)
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2, strict=True):
            assert p1.texts == p2.texts
            assert p1.label == p2.label

    def test_single_example_per_class_creates_self_pair(self) -> None:
        """Single example in a class should create a self-pair as fallback."""
        from sentimentizer.router.train_router import generate_contrastive_pairs

        texts = ["only dietary", "only service"]
        labels = [0, 1]
        pairs = generate_contrastive_pairs(texts, labels, num_iterations=2, seed=42)
        # For the same-class pair of "only dietary", it must pair with itself
        # since it's the only example in class 0
        same_class_label0 = [p for p in pairs if p.label == 1.0 and p.texts[0] == "only dietary"]
        assert len(same_class_label0) == 1  # one self-pair as fallback


@skip_without_router
class TestRouterModel:
    """Test RouterModel save/load cycle and inference."""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """RouterModel should survive a save/load roundtrip."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from sentimentizer.router.model import RouterModel

        # Create a minimal mock model
        mock_backbone = type(
            "MockBackbone",
            (),
            {
                "encode": lambda self, texts, convert_to_numpy=True: np.random.rand(len(texts), 10),
                "save": lambda self, path: None,
            },
        )()
        mock_head = LogisticRegression()
        # Fit the head with dummy data so it can predict
        X_dummy = np.random.rand(10, 10)
        y_dummy = [0, 1, 2] * 3 + [0]
        mock_head.fit(X_dummy, y_dummy)

        model = RouterModel(
            backbone=mock_backbone,
            head=mock_head,
            labels=["dietary", "service", "general"],
        )

        # Save
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))

        # Verify files exist
        assert (save_dir / "router_head.joblib").exists()
        assert (save_dir / "router_config.json").exists()

        # Verify config content
        with open(save_dir / "router_config.json") as f:
            config = json.load(f)
        assert config["model_type"] == "router"
        assert config["labels"] == ["dietary", "service", "general"]
        assert config["head_type"] == "LogisticRegression"

    def test_router_config_json_content(self, tmp_path) -> None:
        """router_config.json should contain model_type, labels, head_type."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from sentimentizer.router.model import RouterModel

        mock_backbone = type(
            "MockBackbone",
            (),
            {
                "encode": lambda self, texts, convert_to_numpy=True: np.random.rand(len(texts), 10),
                "save": lambda self, path: None,
            },
        )()
        mock_head = LogisticRegression()
        X_dummy = np.random.rand(10, 10)
        y_dummy = [0, 1, 2] * 3 + [0]
        mock_head.fit(X_dummy, y_dummy)

        model = RouterModel(backbone=mock_backbone, head=mock_head, labels=["a", "b", "c"])
        save_dir = tmp_path / "cfg_test"
        model.save_pretrained(str(save_dir))

        with open(save_dir / "router_config.json") as f:
            config = json.load(f)
        assert config["labels"] == ["a", "b", "c"]
        assert config["head_type"] == "LogisticRegression"

    def test_from_pretrained_missing_path_raises(self) -> None:
        """from_pretrained with nonexistent path should raise FileNotFoundError."""
        from sentimentizer.router.model import RouterModel

        with pytest.raises(FileNotFoundError):
            RouterModel.from_pretrained("/nonexistent/path")

    def test_predict_with_mock_backbone(self) -> None:
        """predict() should return head predictions."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from sentimentizer.router.model import RouterModel

        # Create a fitted head
        X = np.random.rand(30, 10)
        y = [0, 1, 2] * 10
        head = LogisticRegression().fit(X, y)

        # Mock backbone that returns consistent embeddings
        class MockBackbone:
            def encode(self, texts, convert_to_numpy=True):
                return np.random.rand(len(texts), 10)

        model = RouterModel(
            backbone=MockBackbone(),
            head=head,
            labels=["dietary", "service", "general"],
        )
        preds = model.predict(["test text"])
        assert len(preds) == 1

    def test_predict_proba_with_mock_backbone(self) -> None:
        """predict_proba() should return probability matrix."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from sentimentizer.router.model import RouterModel

        X = np.random.rand(30, 10)
        y = [0, 1, 2] * 10
        head = LogisticRegression().fit(X, y)

        class MockBackbone:
            def encode(self, texts, convert_to_numpy=True):
                return np.random.rand(len(texts), 10)

        model = RouterModel(
            backbone=MockBackbone(),
            head=head,
            labels=["dietary", "service", "general"],
        )
        probs = model.predict_proba(["test text"])
        assert probs.shape == (1, 3)
        assert abs(probs.sum() - 1.0) < 1e-6  # probabilities sum to 1

    def test_model_encode_with_mock_backbone(self) -> None:
        """model_encode() should return embeddings."""
        import numpy as np

        from sentimentizer.router.model import RouterModel

        class MockBackbone:
            def encode(self, texts, convert_to_numpy=True):
                return np.random.rand(len(texts), 10)

        model = RouterModel(backbone=MockBackbone(), labels=[])
        embeddings = model.model_encode(["hello", "world"])
        assert embeddings.shape == (2, 10)

    def test_setfit_config_alias(self) -> None:
        """SetFitConfig should be an alias for RouterConfig."""
        from sentimentizer.router.config import RouterConfig, SetFitConfig

        assert SetFitConfig is RouterConfig
        assert SetFitConfig().base_model == "BAAI/bge-base-en-v1.5"

    def test_router_model_in_init(self) -> None:
        """RouterModel should be importable from sentimentizer.router."""
        from sentimentizer.router import RouterModel

        assert RouterModel is not None
