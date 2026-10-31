"""Hugging Face Tokenizer wrapper for sentimentizer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL, ModernBERTConfig
from sentimentizer.tokenizer import vectorized_convert_ratings

logger = new_logger(DEFAULT_LOG_LEVEL)


class HFTokenizer:
    """Wrapper class for Hugging Face tokenizers.

    Enables text tokenization using Hugging Face AutoTokenizer and structures
    tokenized batches for downstream training or inference.
    """

    def __init__(self, tokenizer: Any, model_name: str) -> None:
        self.tokenizer = tokenizer
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name: str) -> HFTokenizer:
        """Load tokenizer from pre-trained model name or path."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers package is required to use HFTokenizer. "
                "Please install it using: pip install sentimentizer[transformers]"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(tokenizer=tokenizer, model_name=model_name)

    def tokenize_text(
        self, text: str, max_length: int = ModernBERTConfig.max_seq_length
    ) -> dict[str, torch.Tensor]:
        """Tokenize a single string."""
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def transform(self, data_source: Any) -> Any:
        """Transform a DataSource (pandas or Ray) by tokenizing text columns."""

        def transform_batch(batch: dict) -> dict:
            input_ids = []
            attention_masks = []

            # Use 'text' or 'data' column if 'text' isn't present
            text_col = "text" if "text" in batch else "data"

            for text in batch[text_col]:
                encoded = self.tokenizer(
                    str(text),
                    padding=False,
                    truncation=True,
                    max_length=ModernBERTConfig.max_seq_length,
                )
                input_ids.append(encoded["input_ids"])
                attention_masks.append(encoded["attention_mask"])

            result: dict = {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
            }

            if "stars" in batch:
                result["target"] = vectorized_convert_ratings(np.asarray(batch["stars"]))
            elif "target" in batch:
                result["target"] = np.asarray(batch["target"])

            return result

        return data_source.map_batches(transform_batch, batch_format="numpy")
