# Router encoder benchmark on Yelp cuisine labels

**Status: encoder ranking complete (2026-11-01).** All 60 contrastive runs finished with 5 seeds on
an RTX 5080. The ranking covers review-length text on a proxy task. It does not cover long prompts.

## Summary

| Rank | Encoder | Accuracy at 8 / 32 per class | Gap to top at 32 | Seconds at 32 |
|---|---|---|---|---|
| 1 (tied) | `bge-base-en-v1.5` | 0.706 / 0.720 | -0.011 ± 0.011 | 101 |
| 1 (tied) | `mxbai-embed-large-v1` | 0.711 / 0.730 | — | 303 |
| 3 | `bge-small-en-v1.5` | 0.624 / 0.685 | -0.046 ± 0.010 | 45 |
| 4 | `all-MiniLM-L6-v2` | 0.621 / 0.676 | -0.055 ± 0.010 | 25 |
| 5 | `multi-qa-MiniLM-L6` | 0.595 / 0.664 | -0.067 ± 0.015 | 25 |
| 6 | `modernbert-base` | 0.375 / 0.672 | -0.058 ± 0.008 | 138 |

- **`bge-base-en-v1.5` and `mxbai-embed-large-v1` are not separable on accuracy.** `mxbai` is nominally
  ahead, but neither interval excludes zero.
- **This benchmark gives no reason to replace the default, `bge-base-en-v1.5`.** It ranks errors
  better (lower AURC), has lower log loss and calibration error at 32 per class, and tunes three times
  faster. This is a proxy task, so a labeled sample of the real task is still needed before changing
  the default either way.
- **`modernbert-base` ranks last because it needs more data.** At 32 per class it ties `all-MiniLM-L6-v2`
  (-0.003 ± 0.011), but at 8 per class it is unusable. It is also slower and worse calibrated. Its
  8,192-token limit is its only advantage here, and these runs used 128 tokens.
- **Seeing more of each review helps.** `bge-base-en-v1.5` gained about 2 points at batch 16 and 256
  tokens. `mxbai` was not run at those settings, so this does not change the ranking.
- **Long prompts are untested.** Almost all these reviews fit in 512 tokens, so these results do not
  transfer to prompts of up to 200k tokens. See
  [the long-prompt plan](../docs/router-long-prompt-benchmark-plan.md).

## Method

This tests encoder quality on real review text, using a task with a known answer. Cuisine is
a **proxy**: the production task is dietary/service/general, and this does not measure it.

| Choice | Detail |
|---|---|
| Data | Yelp Open Dataset: up to 40 reviews of at least 30 words per restaurant, 406,180 reviews from 14,828 restaurants |
| Labels | 10 cuisines, from restaurants with exactly one of them. Fusion, Mediterranean, Middle Eastern and buffet restaurants are excluded. Sushi Bars count as Japanese and Tex-Mex as Mexican |
| Split | By `business_id`, so reviews of one restaurant never reach both train and test |
| Test set | 150 reviews per class, at most 3 per restaurant, redrawn for each seed |
| Comparison | 5 seeds, with paired differences and 95% t-intervals across seeds |
| Training | The production `train_router` at commit `e95a8be`, with default hyperparameters except batch size and sequence length |
| Software | torch 2.11.0, sentence-transformers 5.5.0, scikit-learn 1.8.0 |
| Harder subset | "No-keyword" accuracy scores only reviews that never name a cuisine, about 60% of the test set |

Chance accuracy is 10%. Labels come from the restaurant, so a review about parking still
counts as its restaurant's cuisine, and no model can reach 100%. Compare models with each
other, not with a ceiling.

Runs use batch 8 and 128 tokens for all encoders, the only tested setting that fit `mxbai` on the 11
GB GPU used for the first runs. A float32 copy of `mxbai` is used because of the bug below. Many reviews
are truncated, as measured on 20,000 reviews with the `bge-base-en-v1.5` tokenizer:

| Token limit | Reviews truncated |
|---|---|
| 128 | 38.8% |
| 256 | 11.5% |
| 384 | 4.1% |
| 512 | 1.8% |

## Contrastive results

Gap is paired accuracy against the top encoder at that training size.

### 8 reviews per class

| Encoder | Accuracy | Gap | Macro-F1 | No-keyword accuracy | Accuracy on most confident 50% | Log loss | ECE | AURC | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| `mxbai-embed-large-v1` | 0.711 | — | 0.716 | 0.578 | 0.951 | 1.344 | 0.191 | 0.091 | 79 |
| `bge-base-en-v1.5` | 0.706 | -0.005 ± 0.026 | 0.710 | 0.567 | 0.956 | 1.152 | 0.220 | 0.088 | 27 |
| `bge-small-en-v1.5` | 0.624 | -0.087 ± 0.051 | 0.626 | 0.465 | 0.912 | 1.402 | 0.248 | 0.141 | 15 |
| `all-MiniLM-L6-v2` | 0.621 | -0.090 ± 0.051 | 0.631 | 0.453 | 0.903 | 1.471 | 0.268 | 0.146 | 11 |
| `multi-qa-MiniLM-L6` | 0.595 | -0.115 ± 0.076 | 0.605 | 0.419 | 0.886 | 1.503 | 0.242 | 0.161 | 9 |
| `modernbert-base` | 0.375 | -0.336 ± 0.208 | 0.370 | 0.269 | 0.506 | 2.981 | 0.417 | 0.468 | 35 |

### 32 reviews per class

| Encoder | Accuracy | Gap | Macro-F1 | No-keyword accuracy | Accuracy on most confident 50% | Log loss | ECE | AURC | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| `mxbai-embed-large-v1` | 0.730 | — | 0.734 | 0.608 | 0.957 | 1.628 | 0.220 | 0.082 | 303 |
| `bge-base-en-v1.5` | 0.720 | -0.011 ± 0.011 | 0.722 | 0.590 | 0.970 | 1.022 | 0.094 | 0.078 | 101 |
| `bge-small-en-v1.5` | 0.685 | -0.046 ± 0.010 | 0.686 | 0.538 | 0.946 | 1.107 | 0.120 | 0.102 | 45 |
| `all-MiniLM-L6-v2` | 0.676 | -0.055 ± 0.010 | 0.680 | 0.525 | 0.957 | 1.092 | 0.095 | 0.100 | 25 |
| `modernbert-base` | 0.672 | -0.058 ± 0.008 | 0.680 | 0.530 | 0.944 | 2.129 | 0.274 | 0.113 | 138 |
| `multi-qa-MiniLM-L6` | 0.664 | -0.067 ± 0.015 | 0.669 | 0.509 | 0.954 | 1.137 | 0.096 | 0.107 | 25 |

Seed 3 at 8 per class scored about 6 to 13 points below the other seeds for the three smaller encoders,
which widens their intervals at that size.

### Why `modernbert-base` is unstable at 8 per class

Its five seeds scored 0.579, 0.325, 0.295, 0.503 and 0.171, then 0.663 to 0.679 at 32 per class. It was
added after the first five encoders, with the same settings and seeds. Diagnostics:

| Check at 8 per class | Result | What it shows |
|---|---|---|
| Rerun of seeds 1, 2 and 4 | Same accuracies | Deterministic for a seed on this setup |
| Embeddings | No NaNs, mean norm about 35 | Training did not diverge |
| Training accuracy | 1.000 on all seeds | The head memorizes the training reviews |
| Frozen encoder, default head | 0.257 at 8 per class, 0.422 at 32 | Its untuned embeddings carry little cuisine signal |
| Learning rate 2e-6 instead of 2e-5 | 0.155 to 0.213 | A lower rate does not stabilize it |

It starts from a general text encoder rather than one trained for sentence similarity, so nearly all its
cuisine signal must come from fine-tuning. The likely explanation is that 200 fine-tuning steps at 8 per
class are too few and 800 at 32 are enough. The step count itself was not varied.

### `bge-base-en-v1.5` at batch 16 and 256 tokens

| Reviews per class | Accuracy | Gain over its batch 8, 128-token run | No-keyword accuracy | Log loss | ECE | AURC | Seconds |
|---|---|---|---|---|---|---|---|
| 8 | 0.729 | +0.023 ± 0.005 | 0.588 | 1.158 | 0.280 | 0.075 | 39 |
| 32 | 0.742 | +0.022 ± 0.011 | 0.611 | 0.938 | 0.094 | 0.069 | 150 |

Batch size and sequence length changed together, so the gain cannot be credited to length alone. These
runs also beat `mxbai` at 128 tokens at 32 per class by 0.011 ± 0.006, but that is not a fair comparison
because `mxbai` did not get the longer input.

### Per-label results at 32 reviews per class

Precision here is measured on the balanced test set. "Natural precision" reweights it with the
confusion matrix to the sampled review mix, where Mexican and Italian are about 30% each and
Korean about 1.5%. The sample caps reviews per restaurant, so this approximates real traffic.

| Label | mxbai precision | mxbai recall | mxbai natural precision | bge-base precision | bge-base recall | bge-base natural precision |
|---|---|---|---|---|---|---|
| chinese | 0.58 | 0.77 | 0.65 | 0.60 | 0.71 | 0.69 |
| french | 0.64 | 0.62 | 0.28 | 0.67 | 0.61 | 0.33 |
| greek | 0.71 | 0.67 | 0.28 | 0.66 | 0.67 | 0.22 |
| indian | 0.85 | 0.81 | 0.73 | 0.87 | 0.79 | 0.78 |
| italian | 0.72 | 0.70 | 0.93 | 0.68 | 0.71 | 0.90 |
| japanese | 0.79 | 0.75 | 0.82 | 0.80 | 0.73 | 0.80 |
| korean | 0.77 | 0.68 | 0.31 | 0.75 | 0.66 | 0.32 |
| mexican | 0.79 | 0.80 | 0.93 | 0.69 | 0.83 | 0.90 |
| thai | 0.80 | 0.75 | 0.58 | 0.82 | 0.74 | 0.67 |
| vietnamese | 0.84 | 0.75 | 0.64 | 0.80 | 0.75 | 0.59 |

Rare labels lose the most precision once reweighted, which matters for any rare route. The
largest confusions are among French, Greek and Italian, and among Chinese, Korean and Vietnamese.

### Agreement with the first runs

The first five encoders' 8-per-class results reproduce on different hardware to within 0.7 points for every encoder. The
first runs had only 2 seeds at 32 per class and put `bge-base-en-v1.5` ahead of `mxbai` (0.725 and
0.718). With all 5 seeds the order reverses, so those two seeds were too few to order the pair. The RTX
5080 runs took a little under half the earlier time. Raw logs from the first runs were lost with that machine,
so all contrastive figures here come from the rerun.

## Frozen results

This sweep comes from the first machine, and its raw logs were lost with it. It used 512 tokens and 5
seeds. Accuracy is shown at 8 / 32 / 128 / 512 reviews per class, with the best of the L2-normalized
heads at C = 1, 10 and 100.

| Encoder | Accuracy | No-keyword accuracy at 32 |
|---|---|---|
| `mxbai-embed-large-v1` | 0.651 / 0.743 / 0.775 / 0.789 | 0.621 |
| `bge-base-en-v1.5` | 0.623 / 0.733 / 0.769 / 0.780 | 0.608 |
| `bge-small-en-v1.5` | 0.553 / 0.677 / 0.724 / 0.749 | 0.532 |
| `all-MiniLM-L6-v2` | 0.592 / 0.679 / 0.716 / 0.739 | 0.531 |
| `multi-qa-MiniLM-L6` | 0.556 / 0.650 / 0.696 / 0.720 | 0.495 |
| TF-IDF baseline (its only head) | 0.365 / 0.562 / 0.686 / 0.745 | 0.437 |

**Do not read the gain from fine-tuning off these tables.** Frozen runs used 512 tokens and contrastive
runs used 128. The paired frozen run at 128 tokens was skipped because only the encoder ranking was
needed, so the value of fine-tuning remains untested.

The published `ryeyoo/sentimentizer-router` backbone reached only 11% at 8 per class, or 21%
at best with stronger heads. It was fine-tuned for other labels, so this says nothing about
its performance on its own task.

## Problems found

1. **`train_router` cannot fine-tune `mxbai-embed-large-v1`.** The checkpoint declares
   `float16`, the loaded model honors it, and `backward()` fails with
   `RuntimeError: Found dtype Float but expected Half`. `router/config.py` names this
   encoder as the upgrade path. Not fixed here.
2. **Production settings run out of memory on full-length reviews.** With every sequence full,
   on an 11 GB GPU, `bge-base-en-v1.5` fits at batch 16 and 256 tokens or batch 8 and 512,
   but not at batch 16 and 512. `mxbai` fits only at batch 8 and 128 of the settings tested. These limits
   were not retested on the 16 GB RTX 5080.
3. **The default head is underconfident on unit-norm embeddings.** For `bge-base-en-v1.5` at 32
   per class, moving C from 1 to 10 raises accuracy from 0.717 to 0.733 and cuts ECE from
   0.523 to 0.249. Acceptance thresholds depend on these scores. `mxbai` has no `Normalize`
   module (mean norm 18.42), so the default head regularizes it less, which flatters it in
   default-head comparisons.
4. **`prediction_metrics` cannot rank encoders for abstaining.** Fixed thresholds are confounded
   by each head's confidence scale, and ECE can be near zero for a useless model. This
   benchmark adds accuracy at fixed coverage, AURC, log loss and Brier score. The shared
   metrics are unchanged.
5. **Long input is silently truncated.** The router keeps only the first `max_seq_length` tokens and
   gives no warning. Over HTTP, `max_text_length` rejects text over 10,000 characters.

## Reproduce

```bash
python scripts/router_cuisine_benchmark.py extract --archive yelp_dataset.tar --output c.jsonl
python scripts/router_cuisine_benchmark.py contrastive --data c.jsonl --log r.jsonl --work tmp \
  --batch-size 8 --max-seq-length 128 --mxbai-path <float32-copy-of-mxbai>
python scripts/router_cuisine_benchmark.py contrastive --data c.jsonl --log r-b16.jsonl --work tmp \
  --batch-size 16 --max-seq-length 256 --encoders bge-base-en-v1.5
python scripts/router_cuisine_benchmark.py report --log r.jsonl
```

Create the float32 copy with `SentenceTransformer(name).float().save(path)`. Wrap long runs in
`systemd-inhibit --what=sleep`, because suspending pauses them. Use a separate log per batch size and
sequence length, because the resume key contains neither. Raw logs are not committed, and review text
from the Yelp dataset must not be committed.
