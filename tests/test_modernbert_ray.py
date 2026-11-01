import pytest

# ruff: noqa: E402
ray = pytest.importorskip("ray")

import pandas as pd

from sentimentizer.config import TrainerConfig
from sentimentizer.loader import load_train_val_ray_datasets
from sentimentizer.trainer import new_ray_trainer


@pytest.fixture
def modernbert_ray_datasets(tmp_path):
    """Own the Ray session, including cleanup when dataset setup fails."""
    path = tmp_path / "reviews.parquet"
    pd.DataFrame(
        {
            "input_ids": [[1, 2, 3]] * 12,
            "attention_mask": [[1, 1, 1]] * 12,
            "target": [0, 1, 2] * 4,
        }
    ).to_parquet(path)
    try:
        ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
        yield load_train_val_ray_datasets(str(path), test_size=0.5)
    finally:
        ray.shutdown()


class TestModernBERTRay:
    """Ray-specific training integration checks for ModernBERT."""

    def test_new_ray_trainer_creation(self, modernbert_ray_datasets) -> None:
        """Verify new_ray_trainer can successfully construct a TorchTrainer with modernbert."""
        from ray.train.torch import TorchTrainer

        train_ds, val_ds = modernbert_ray_datasets
        cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
        trainer = new_ray_trainer(
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            model_type="modernbert",
        )

        assert isinstance(trainer, TorchTrainer)
        assert trainer.train_loop_config["model_type"] == "modernbert"
        assert trainer.train_loop_config["lr"] == 2e-5  # ModernBERT default LR
