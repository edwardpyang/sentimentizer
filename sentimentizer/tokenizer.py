from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd
from gensim import corpora

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

if TYPE_CHECKING:
    import ray.data

from sentimentizer import new_logger, time_decorator
from sentimentizer.config import DEFAULT_LOG_LEVEL, FileConfig, TokenizerConfig

logger = new_logger(DEFAULT_LOG_LEVEL)

TokenizerType = TypeVar("TokenizerType", bound="Tokenizer")

pattern = re.compile(r"[a-z0-9'-]+")


def convert_rating(rating: int) -> int:
    """Scale a single star rating to a class index.

    3-class mapping: 1-2★ → 0 (negative), 3★ → 1 (neutral), 4-5★ → 2 (positive).
    Used for single-item inference (tokenize_text). For batch processing,
    prefer vectorized_convert_ratings() instead.
    """
    if rating >= 4:
        return 2  # positive
    elif rating == 3:
        return 1  # neutral
    else:
        return 0  # negative


def vectorized_convert_ratings(stars: np.ndarray) -> np.ndarray:
    """Vectorized star-to-label conversion using NumPy.

    3-class mapping: 1-2★ → 0, 3★ → 1, 4-5★ → 2.
    ~100x faster than applying convert_rating() per element.
    """
    return np.select([stars <= 2, stars == 3, stars >= 4], [0, 1, 2]).astype(np.int64)


def text_sequencer(
    dictionary: corpora.Dictionary, text: list[str], max_len: int = 200
) -> np.ndarray:
    """
    converts tokens to numeric representation by dictionary;
    zero is considered padding
    """

    processed = np.zeros(max_len, dtype=int)
    # OOV token index: after compactify(), IDs are 0..N-1 so the
    # embedding matrix has rows [pad, word_0, word_1, ..., word_{N-1}, OOV].
    # Row N+1 = len(dictionary) + 1 is the dedicated OOV row.
    dict_final = len(dictionary) + 1

    for i, word in enumerate(text):
        if i >= max_len:
            return processed
        if word in dictionary.token2id:
            # the ids have an offset of 1 for this because
            # 0 represents a padded value in pytorch
            processed[i] = dictionary.token2id[word] + 1
        else:
            processed[i] = dict_final

    return processed


def regex_tokenize(x: str) -> list[str]:
    """regex tokenize, less accurate than spacy"""
    return pattern.findall(x.lower())


def _dictionary_from_counts(
    word_freq: Counter, doc_freq: Counter, num_docs: int, cfg: TokenizerConfig
) -> corpora.Dictionary:
    """Build a filtered, contiguous dictionary from aggregated vocabulary counts."""
    dictionary = corpora.Dictionary()
    dictionary.num_docs = num_docs

    # Sort by (descending frequency, ascending word) so the mapping is deterministic
    # regardless of batch processing order.
    for idx, word in enumerate(sorted(word_freq.keys(), key=lambda w: (-word_freq[w], w))):
        dictionary.token2id[word] = idx
        dictionary.dfs[idx] = doc_freq[word]

    dictionary.num_pos = sum(word_freq.values())
    dictionary.num_nnz = sum(doc_freq.values())

    dictionary.filter_extremes(
        no_below=cfg.dict_min,
        no_above=cfg.no_above,
        keep_n=cfg.dict_keep,
    )
    # Remap IDs to contiguous 0..N-1 after filter_extremes leaves gaps.
    # Required for embedding matrix row alignment (row k = token ID k-1).
    dictionary.compactify()
    return dictionary


def new_dictionary(data: pd.DataFrame, cfg: TokenizerConfig) -> corpora.Dictionary:
    """builds a dictionary from a dataframe"""
    # Create dictionary from text column
    # To guarantee alignment with _build_dictionary_distributed, we must assign
    # IDs by descending frequency and ascending word.

    word_freq = Counter()
    doc_freq = Counter()
    num_docs = 0

    for doc_tokens in data[cfg.text_col]:
        if not isinstance(doc_tokens, list):
            try:
                doc_tokens = list(doc_tokens)
            except TypeError:
                doc_tokens = regex_tokenize(str(doc_tokens))
        num_docs += 1
        word_freq.update(doc_tokens)
        doc_freq.update(set(doc_tokens))

    dictionary = _dictionary_from_counts(word_freq, doc_freq, num_docs, cfg)
    logger.info("dictionary created...")

    if cfg.save_dictionary:
        dictionary.save(f"{FileConfig.dictionary_file_path}")
        logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

    return dictionary


@dataclass
class Tokenizer:
    """wrapper class for handling tokenization of datasets"""

    dictionary: corpora.Dictionary
    cfg: TokenizerConfig = field(default_factory=TokenizerConfig)

    @classmethod
    def build_dictionary(
        cls,
        data_source: object,
        cfg: TokenizerConfig | None = None,
    ) -> Tokenizer:
        """Build dictionary from any DataSource (pandas or Ray).

        Replaces ``new_dictionary()`` and ``_build_dictionary_distributed()``
        with a single unified entry point.
        """
        if cfg is None:
            cfg = TokenizerConfig(save_dictionary=True)

        total_word_freq: Counter = Counter()
        total_doc_freq: Counter = Counter()
        total_num_docs = 0

        for batch in data_source.iter_batches(batch_size=10000, batch_format="numpy"):
            wf, df, nd = _count_vocab_batch(batch, cfg.text_col)
            total_word_freq += wf
            total_doc_freq += df
            total_num_docs += nd

        dictionary = _dictionary_from_counts(total_word_freq, total_doc_freq, total_num_docs, cfg)

        if cfg.save_dictionary:
            dictionary.save(str(FileConfig.dictionary_file_path))

        logger.info(f"dictionary created: {len(dictionary)} terms, {total_num_docs} documents")
        return cls(dictionary=dictionary, cfg=cfg)

    @classmethod
    def from_data(cls: type[TokenizerType], data: pd.DataFrame) -> TokenizerType:
        """creates tokenizer from dataframe"""
        return cls(dictionary=new_dictionary(data, TokenizerConfig(save_dictionary=False)))

    @time_decorator
    def transform(self, data_source: object) -> object:
        """Transform any DataSource: drop neutral, tokenize, convert ratings.

        Replaces ``transform_dataframe()`` and ``transform_dataset()``.
        Uses expression-based filter (Risk 9 mitigation) for the neutral-drop
        step on the pandas path.
        """
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")

        dictionary = self.dictionary
        cfg = self.cfg

        # Step 1: Conditionally drop neutral (3-star) reviews
        if not cfg.include_neutral:
            filtered = data_source.filter(col=cfg.label_col, op="ne", value=3)
        else:
            filtered = data_source  # keep all reviews including 3-star

        # Step 2: Map batches to convert text to sequences
        def transform_batch(batch: dict) -> dict:
            inputs = []
            for text in batch[cfg.text_col]:
                inputs.append(text_sequencer(dictionary, text, cfg.max_len))
            batch[cfg.inputs] = np.array(inputs)

            if cfg.label_col in batch:
                batch[cfg.labels] = vectorized_convert_ratings(np.asarray(batch[cfg.label_col]))

            # Drop variable-length columns
            cols_to_keep = {cfg.inputs, cfg.labels}
            for col in list(batch.keys()):
                if col not in cols_to_keep:
                    del batch[col]
            return batch

        return filtered.map_batches(transform_batch, batch_size=10000, batch_format="numpy")

    def update_dictionary(self, data_source: object) -> None:
        """Update existing dictionary with new tokens from any DataSource.

        Replaces ``update_from_dataset()``. Skips ``filter_extremes`` and
        ``compactify`` to preserve existing token IDs.
        """
        cfg = self.cfg
        total_word_freq: Counter = Counter()
        total_doc_freq: Counter = Counter()
        total_num_docs = 0

        for batch in data_source.iter_batches(batch_size=10000, batch_format="numpy"):
            wf, df, nd = _count_vocab_batch(batch, cfg.text_col)
            total_word_freq += wf
            total_doc_freq += df
            total_num_docs += nd

        old_len = len(self.dictionary)
        for word in total_word_freq:
            if word not in self.dictionary.token2id:
                idx = len(self.dictionary)
                self.dictionary.token2id[word] = idx
                self.dictionary.dfs[idx] = total_doc_freq[word]
            else:
                idx = self.dictionary.token2id[word]
                self.dictionary.dfs[idx] += total_doc_freq[word]

        self.dictionary.num_docs += total_num_docs
        self.dictionary.num_pos += sum(total_word_freq.values())
        self.dictionary.num_nnz += sum(total_doc_freq.values())

        new_len = len(self.dictionary)
        logger.info(
            f"dictionary updated: added {new_len - old_len} new terms, "
            f"total {new_len} terms, {total_num_docs} new documents"
        )

        if cfg.save_dictionary:
            self.dictionary.save(str(FileConfig.dictionary_file_path))

    def transform_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """transforms dataframe with text and target (legacy shim).

        Preserves original columns and adds ``inputs``/``labels``.
        """
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")

        cfg = self.cfg
        data = data.copy()
        if not cfg.include_neutral:
            data = data[data[cfg.label_col] != 3].copy()  # type: ignore
        data[cfg.inputs] = data[cfg.text_col].map(
            lambda text: text_sequencer(self.dictionary, text, cfg.max_len)
        )
        data[cfg.labels] = vectorized_convert_ratings(data[cfg.label_col].values)
        logger.info("converted tokens to numbers...")
        return data

    def tokenize_text(self, text: str) -> np.ndarray:
        """converts string phrase to numpy array"""
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")
        tokens = regex_tokenize(text)
        return text_sequencer(self.dictionary, tokens, self.cfg.max_len).reshape(
            1, self.cfg.max_len
        )

    @classmethod
    def from_dataset(
        cls: type[TokenizerType],
        ds: ray.data.Dataset,
        cfg: TokenizerConfig | None = None,
    ) -> TokenizerType:
        """Creates tokenizer from ray dataset using distributed Map-Reduce.

        Instead of pulling every row to the driver via iter_rows(), this
        distributes word counting across Ray workers and aggregates the
        results on the driver to build the Gensim dictionary.

        By default the dictionary is saved to disk (save_dictionary=True)
        so that ``new_model()`` loads the same token-to-ID mapping that was
        used during tokenization.  Pass ``TokenizerConfig(save_dictionary=False)``
        to skip persistence (e.g. in tests).

        Args:
            ds: Ray dataset with a text column to build the dictionary from.
            cfg: Optional tokenizer config.  Defaults to
                ``TokenizerConfig(save_dictionary=True)``.
        """
        if cfg is None:
            cfg = TokenizerConfig(save_dictionary=True)

        dictionary = _build_dictionary_distributed(ds, cfg)

        if cfg.save_dictionary:
            dictionary.save(f"{FileConfig.dictionary_file_path}")
            logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

        return cls(dictionary=dictionary, cfg=cfg)

    def update_from_dataset(self, ds: ray.data.Dataset) -> None:
        """Updates the existing dictionary with new tokens from a Ray dataset.

        Uses the same distributed Map-Reduce approach as ``from_dataset()``
        to count tokens, but calls ``add_documents()`` on the existing
        dictionary instead of creating a new one.

        Args:
            ds: Ray dataset with a text column to update the dictionary from.
        """
        _update_dictionary_distributed(self.dictionary, ds, self.cfg)

        if self.cfg.save_dictionary:
            self.dictionary.save(f"{FileConfig.dictionary_file_path}")
            logger.info(f"updated dictionary saved to {FileConfig.dictionary_file_path}...")

    @time_decorator
    def transform_dataset(self, ds: ray.data.Dataset) -> ray.data.Dataset:
        """transforms ray dataset with text and target"""
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")

        # Capture for closure
        dictionary = self.dictionary
        cfg = self.cfg

        # Conditionally drop neutral (3-star) reviews
        if not cfg.include_neutral:
            ds = ds.filter(lambda row: row[cfg.label_col] != 3)

        def transform_batch(batch: dict) -> dict:
            inputs = []
            for text in batch[cfg.text_col]:
                inputs.append(text_sequencer(dictionary, text, cfg.max_len))

            batch[cfg.inputs] = np.array(inputs)

            if cfg.label_col in batch:
                batch[cfg.labels] = vectorized_convert_ratings(np.asarray(batch[cfg.label_col]))

            # Drop variable-length columns that cause Arrow conversion issues
            # Keep only the numeric columns needed for training
            cols_to_keep = {cfg.inputs, cfg.labels}
            for col in list(batch.keys()):
                if col not in cols_to_keep:
                    del batch[col]

            return batch

        ds = ds.map_batches(transform_batch, batch_format="numpy")
        logger.info("converted tokens to numbers...")
        return ds


def get_trained_tokenizer() -> Tokenizer:
    """Load the pre-trained dictionary, downloading from HF Hub if missing.

    On a fresh install the ``yelp.dictionary`` file won't exist locally.
    When that happens, we download it from the canonical encoder repo
    (``ryeyoo/sentimentizer-encoder``) — all three GloVe model types
    share the same dictionary.
    """
    dict_path = str(files("sentimentizer.data").joinpath("yelp.dictionary"))

    if not Path(dict_path).exists():
        logger.info(
            "Dictionary not found locally, downloading from HF Hub...",
            path=dict_path,
        )
        try:
            from sentimentizer.hf import download_weights

            download_weights(
                model_type="encoder",
                local_path=Path(dict_path).parent / "encoder_weights.pth",
                dict_path=dict_path,
            )
        except Exception:
            logger.exception("Failed to auto-download dictionary from HF Hub")

    if not Path(dict_path).exists():
        raise FileNotFoundError(
            f"Dictionary file not found at {dict_path} and could not be "
            "downloaded from Hugging Face Hub. Train a model first or run:\n"
            '  python -c "from sentimentizer.hf import download_weights; '
            "download_weights('encoder', 'sentimentizer/data/encoder_weights.pth', "
            "dict_path='sentimentizer/data/yelp.dictionary')\""
        )

    corp_dict = corpora.Dictionary.load(dict_path)
    return Tokenizer(dictionary=corp_dict)


# ---------------------------------------------------------------------------
# Distributed dictionary building (Map-Reduce)
# ---------------------------------------------------------------------------


def _count_vocab_batch(batch: dict, text_col: str) -> tuple[Counter, Counter, int]:
    """Count word frequencies and document frequencies for a batch.

    Returns:
        word_freq: Counter of total word occurrences across all docs in batch
        doc_freq: Counter of how many docs each word appears in
        num_docs: Number of documents in this batch
    """
    word_freq: Counter = Counter()
    doc_freq: Counter = Counter()
    num_docs = 0

    for doc_tokens in batch[text_col]:
        if not isinstance(doc_tokens, list):
            try:
                doc_tokens = list(doc_tokens)
            except TypeError:
                doc_tokens = regex_tokenize(str(doc_tokens))
        num_docs += 1
        word_freq.update(doc_tokens)
        doc_freq.update(set(doc_tokens))

    return word_freq, doc_freq, num_docs


def _build_dictionary_distributed(ds: ray.data.Dataset, cfg: TokenizerConfig) -> corpora.Dictionary:
    """Build a Gensim dictionary using distributed Map-Reduce over Ray.

    Instead of streaming every row to the driver via iter_rows(),
    this distributes word counting across Ray workers and aggregates
    the results on the driver to construct the dictionary.
    """
    text_col = cfg.text_col

    # Map phase: count vocab across distributed batches
    total_word_freq: Counter = Counter()
    total_doc_freq: Counter = Counter()
    total_num_docs = 0

    for batch in ds.iter_batches(batch_size=10000, batch_format="numpy"):
        wf, df, nd = _count_vocab_batch(batch, text_col)
        total_word_freq += wf
        total_doc_freq += df
        total_num_docs += nd

    # Reduce phase: build Gensim dictionary from aggregated counts
    dictionary = _dictionary_from_counts(total_word_freq, total_doc_freq, total_num_docs, cfg)
    logger.info(
        f"dictionary created via Map-Reduce: {len(dictionary)} terms, {total_num_docs} documents"
    )

    return dictionary


def _update_dictionary_distributed(
    dictionary: corpora.Dictionary, ds: ray.data.Dataset, cfg: TokenizerConfig
) -> None:
    """Update a Gensim dictionary using distributed Map-Reduce over Ray."""
    text_col = cfg.text_col

    # Map phase: count vocab across distributed batches
    total_word_freq: Counter = Counter()
    total_doc_freq: Counter = Counter()
    total_num_docs = 0

    for batch in ds.iter_batches(batch_size=10000, batch_format="numpy"):
        wf, df, nd = _count_vocab_batch(batch, text_col)
        total_word_freq += wf
        total_doc_freq += df
        total_num_docs += nd

    # Reduce phase: update existing dictionary
    # We simulate add_documents by updating the internal structures.
    # Existing IDs are preserved.

    old_len = len(dictionary)
    for word in total_word_freq:
        if word not in dictionary.token2id:
            idx = len(dictionary)
            dictionary.token2id[word] = idx
            dictionary.dfs[idx] = total_doc_freq[word]
        else:
            idx = dictionary.token2id[word]
            dictionary.dfs[idx] += total_doc_freq[word]

    dictionary.num_docs += total_num_docs
    dictionary.num_pos += sum(total_word_freq.values())
    dictionary.num_nnz += sum(total_doc_freq.values())

    # We SKIP filter_extremes and compactify because they reassign IDs,
    # which would break existing model weights.

    new_len = len(dictionary)
    logger.info(
        f"dictionary updated via Map-Reduce: added {new_len - old_len} new terms, "
        f"total {new_len} terms, {total_num_docs} new documents"
    )
