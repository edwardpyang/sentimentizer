"""Train a task-configured router with food-review defaults.

Uses CosineSimilarityLoss for contrastive learning on the embedding
space, then fits a LogisticRegression classification head on top of
the fine-tuned embeddings.

This module replaces the SetFit-based training pipeline with direct
use of sentence-transformers + sklearn, eliminating the setfit
dependency.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from sentimentizer.router.config import RouterConfig
from sentimentizer.router.model import RouterModel

logger = logging.getLogger(__name__)


def generate_contrastive_pairs(
    texts: list[str],
    labels: list[int],
    num_iterations: int = 20,
    seed: int = 42,
) -> list[InputExample]:
    """Generate contrastive sentence pairs for training.

    Ported from SetFit's ContrastiveDataset logic:
    - For each text, generate `num_iterations` same-class pairs (label=1.0)
    - For each text, generate `num_iterations` different-class pairs (label=0.0)
    - No self-pairing (text can't pair with itself)

    Args:
        texts: List of text strings.
        labels: List of integer labels (same length as texts).
        num_iterations: Number of same-class and different-class pairs
                       per text. Total pairs = 2 * num_iterations * len(texts).
        seed: Random seed for reproducibility.

    Returns:
        List of InputExample objects with texts and cosine similarity labels
        (1.0 for same class, 0.0 for different class).
    """
    rng = random.Random(seed)

    # Group texts by label for efficient sampling
    texts_by_label: dict[int, list[str]] = {}
    for text, label in zip(texts, labels, strict=True):
        texts_by_label.setdefault(label, []).append(text)

    unique_labels = list(texts_by_label.keys())
    if len(unique_labels) < 2:
        raise ValueError("Contrastive training requires at least two classes")
    pairs: list[InputExample] = []

    for text, label in zip(texts, labels, strict=True):
        # Same-class pairs (positive examples, cosine similarity = 1.0)
        same_class = [t for t in texts_by_label[label] if t != text]
        if same_class:
            for _ in range(num_iterations):
                pair_text = rng.choice(same_class)
                pairs.append(InputExample(texts=[text, pair_text], label=1.0))
        else:
            # Single example in class — create a self-pair as fallback
            logger.debug(f"Only one example for label {label}, creating self-pair")
            pairs.append(InputExample(texts=[text, text], label=1.0))

        # Different-class pairs (negative examples, cosine similarity = 0.0)
        other_labels = [lbl for lbl in unique_labels if lbl != label]
        for _ in range(num_iterations):
            other_label = rng.choice(other_labels)
            pair_text = rng.choice(texts_by_label[other_label])
            pairs.append(InputExample(texts=[text, pair_text], label=0.0))

    logger.info(
        f"Generated {len(pairs)} contrastive pairs "
        f"({num_iterations} same-class + {num_iterations} different-class "
        f"per {len(texts)} texts)"
    )
    return pairs


def train_router(
    config: RouterConfig,
    train_dataset: Any,
    eval_dataset: Any = None,
) -> RouterModel:
    """Train the router model.

    Uses CosineSimilarityLoss for contrastive learning on the embedding
    space, then fits a LogisticRegression classification head on top of
    the fine-tuned embeddings.

    Args:
        config: RouterConfig with model parameters.
        train_dataset: Training dataset with 'text' and 'label' columns.
                       Can be a HuggingFace Dataset or a dict with
                       'text' and 'label' keys.
        eval_dataset: Optional evaluation dataset (unused in training,
                      reserved for future use).

    Returns:
        Trained RouterModel.
    """
    # Extract texts and labels from dataset
    if hasattr(train_dataset, "__getitem__"):
        # HuggingFace Dataset or dict-like
        texts = train_dataset["text"]
        labels = train_dataset["label"]
        # Convert to plain lists if needed
        if hasattr(texts, "tolist"):
            texts = texts.tolist() if not isinstance(texts, list) else list(texts)
        if hasattr(labels, "tolist"):
            labels = labels.tolist() if not isinstance(labels, list) else list(labels)
        texts = list(texts)
        labels = [config.task.label_id(lbl) for lbl in labels]
    else:
        raise ValueError(f"Unsupported train_dataset type: {type(train_dataset)}")

    if len(texts) != len(labels) or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("Training requires one nonempty text per class ID")
    if set(labels) != set(range(len(config.task.labels))):
        raise ValueError("Training data must include every configured task class")
    if config.max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    logger.info(f"Training router model on {len(texts)} examples, {len(set(labels))} classes")

    # Step 1: Load backbone
    logger.info(f"Loading backbone: {config.base_model}")
    backbone = SentenceTransformer(config.base_model)
    backbone.max_seq_length = config.max_seq_length

    # Step 2: Generate contrastive pairs
    pairs = generate_contrastive_pairs(
        texts=texts,
        labels=labels,
        num_iterations=config.num_iterations,
        seed=config.seed,
    )

    # Step 3: Fine-tune backbone with CosineSimilarityLoss
    train_dataloader = DataLoader(pairs, batch_size=config.batch_size, shuffle=True)
    train_loss = CosineSimilarityLoss(model=backbone)

    logger.info(
        f"Starting contrastive fine-tuning: "
        f"{config.num_epochs} epochs, batch_size={config.batch_size}"
    )
    backbone.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.num_epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        show_progress_bar=True,
    )

    # Step 4: Encode training texts and fit classification head
    logger.info("Encoding training texts with fine-tuned backbone...")
    embeddings = backbone.encode(texts, convert_to_numpy=True)

    logger.info("Fitting LogisticRegression classification head...")
    head = LogisticRegression(max_iter=1000, solver="lbfgs")
    head.fit(embeddings, labels)

    # Step 5: Build RouterModel
    model = RouterModel(backbone=backbone, head=head, task=config.task)

    # Step 6: Save model
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    logger.info(f"Router model saved to {output_path}")

    return model
