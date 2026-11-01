# Router encoder benchmark on Yelp cuisine labels

**Status: preliminary.** Contrastive results cover 35 of 50 runs at 2026-11-01 15:38 EST.
The 8-per-class runs have all 5 seeds. The 32-per-class runs have 2 seeds, so treat those as
indicative only. The paired frozen baseline at 128 tokens and the `bge-base-en-v1.5` run at
batch 16, 256 tokens have not run yet. This page does not recommend changing the default
encoder.

## Summary so far

- **`mxbai-embed-large-v1` and `bge-base-en-v1.5` are the strongest encoders.** They are
  not separable from each other: at 8 reviews per class the gap is 1.0 ± 1.7 points.
- **`bge-base-en-v1.5` costs far less to tune:** about 55 s against 170 s per 8-per-class run,
  and mxbai needs more memory than most settings allow (see below).
- **Both beat the smaller encoders** by about 9 to 12 points at 8 per class.
- **Fine-tuning appears to help most with little data.** Compared with the frozen sweep, the
  8-per-class gain is 3 to 14 points. At 32 per class it is about zero. These comparisons
  are confounded: frozen used 512 tokens and contrastive used 128. The paired frozen run
  will settle this.

## Method

This tests encoder quality on real review text, using a task with a known answer. Cuisine is
a **proxy**: the production task is dietary/service/general, and this does not measure it.
A labeled sample of the real task is still needed before changing the default.

| Choice | Detail |
|---|---|
| Data | Yelp Open Dataset: up to 40 reviews of at least 30 words per restaurant, about 406k reviews from 14.8k restaurants |
| Labels | 10 cuisines, from restaurants with exactly one of them. Fusion, Mediterranean, Middle Eastern and buffet restaurants are excluded. Sushi Bars count as Japanese and Tex-Mex as Mexican |
| Split | By `business_id`, so reviews of one restaurant never reach both train and test |
| Test set | 150 reviews per class, at most 3 per restaurant, redrawn for each seed |
| Comparison | 5 seeds, with paired differences and 95% t-intervals across seeds |
| Training | The production `train_router` at commit `448dff9`, with default hyperparameters except batch size and sequence length |
| Harder subset | "No-keyword" accuracy scores only reviews that never name a cuisine, about 60% of the test set |

Chance accuracy is 10%. Labels come from the restaurant, so a review about parking still
counts as its restaurant's cuisine, and no model can reach 100%. Compare models with each
other, not with a ceiling.

Contrastive runs use batch 8 and 128 tokens for all encoders, the only tested setting that
fits mxbai on an 11 GB GPU. Many reviews are truncated: the median review is 82 words and
the 90th percentile is 215. A float32 copy of mxbai is used because of the bug below.

## Contrastive results

Gap is paired accuracy against the top encoder at that training size.

### 8 reviews per class (5 seeds)

| Encoder | Accuracy | Gap | Macro-F1 | No-keyword accuracy | Accuracy on most confident 50% | Log loss | ECE | AURC | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| `mxbai-embed-large-v1` | 0.718 | — | 0.725 | 0.588 | 0.951 | 1.340 | 0.189 | 0.090 | 170 |
| `bge-base-en-v1.5` | 0.708 | -0.010 ± 0.017 | 0.711 | 0.571 | 0.958 | 1.152 | 0.223 | 0.087 | 55 |
| `bge-small-en-v1.5` | 0.623 | -0.095 ± 0.048 | 0.625 | 0.464 | 0.913 | 1.403 | 0.252 | 0.140 | 20 |
| `all-MiniLM-L6-v2` | 0.620 | -0.097 ± 0.037 | 0.630 | 0.450 | 0.910 | 1.469 | 0.267 | 0.145 | 11 |
| `multi-qa-MiniLM-L6` | 0.595 | -0.123 ± 0.063 | 0.605 | 0.419 | 0.886 | 1.505 | 0.242 | 0.161 | 11 |

### 32 reviews per class (2 seeds, incomplete)

| Encoder | Accuracy | No-keyword accuracy | Accuracy on most confident 50% | Log loss | ECE | AURC | Seconds |
|---|---|---|---|---|---|---|---|
| `bge-base-en-v1.5` | 0.725 | 0.592 | 0.969 | 1.015 | 0.076 | 0.078 | 223 |
| `mxbai-embed-large-v1` | 0.718 | 0.584 | 0.960 | 1.684 | 0.229 | 0.081 | 682 |
| `bge-small-en-v1.5` | 0.680 | 0.528 | 0.956 | 1.105 | 0.124 | 0.101 | 88 |
| `all-MiniLM-L6-v2` | 0.673 | 0.521 | 0.966 | 1.106 | 0.096 | 0.098 | 45 |
| `multi-qa-MiniLM-L6` | 0.657 | 0.496 | 0.961 | 1.139 | 0.102 | 0.105 | 45 |

The `bge-small-en-v1.5` seed 0 run at 32 per class was in progress when the machine
suspended. It looks normal but will be rerun.

### Per-label results at 8 reviews per class (5 seeds)

Precision here is measured on the balanced test set. "Natural precision" reweights it with the
confusion matrix to the sampled review mix, where Mexican and Italian are about 30% each and
Korean about 1.5%. The sample caps reviews per restaurant, so this approximates real traffic.

| Label | mxbai precision | mxbai recall | mxbai natural precision | bge-base precision | bge-base recall | bge-base natural precision |
|---|---|---|---|---|---|---|
| chinese | 0.68 | 0.72 | 0.75 | 0.69 | 0.72 | 0.79 |
| french | 0.66 | 0.60 | 0.31 | 0.64 | 0.59 | 0.32 |
| greek | 0.75 | 0.63 | 0.22 | 0.71 | 0.59 | 0.22 |
| indian | 0.80 | 0.81 | 0.64 | 0.81 | 0.79 | 0.68 |
| italian | 0.72 | 0.71 | 0.92 | 0.69 | 0.72 | 0.91 |
| japanese | 0.85 | 0.74 | 0.89 | 0.79 | 0.74 | 0.85 |
| korean | 0.86 | 0.64 | 0.51 | 0.89 | 0.61 | 0.59 |
| mexican | 0.72 | 0.82 | 0.90 | 0.61 | 0.85 | 0.85 |
| thai | 0.70 | 0.76 | 0.40 | 0.73 | 0.75 | 0.48 |
| vietnamese | 0.81 | 0.74 | 0.61 | 0.85 | 0.73 | 0.71 |

Rare labels lose the most precision once reweighted, which matters for any rare route. The
largest confusions are Greek/Italian/French and Korean/Chinese.

## Frozen results

This complete sweep used 512 tokens and 5 seeds. Accuracy is shown at 8 / 32 / 128 / 512
reviews per class, with the best of the L2-normalized heads at C = 1, 10 and 100.

| Encoder | Accuracy | No-keyword accuracy at 32 |
|---|---|---|
| `mxbai-embed-large-v1` | 0.651 / 0.743 / 0.775 / 0.789 | 0.621 |
| `bge-base-en-v1.5` | 0.623 / 0.733 / 0.769 / 0.780 | 0.608 |
| `bge-small-en-v1.5` | 0.553 / 0.677 / 0.724 / 0.749 | 0.532 |
| `all-MiniLM-L6-v2` | 0.592 / 0.679 / 0.716 / 0.739 | 0.531 |
| `multi-qa-MiniLM-L6` | 0.556 / 0.650 / 0.696 / 0.720 | 0.495 |
| TF-IDF baseline (its only head) | 0.365 / 0.562 / 0.686 / 0.745 | 0.437 |

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
   but not at batch 16 and 512. mxbai fits only at batch 8 and 128 of the settings tested.
3. **The default head is underconfident on unit-norm embeddings.** For `bge-base-en-v1.5` at 32
   per class, moving C from 1 to 10 raises accuracy from 0.717 to 0.733 and cuts ECE from
   0.523 to 0.249. Acceptance thresholds depend on these scores. mxbai has no `Normalize`
   module (mean norm 18.42), so the default head regularizes it less, which flatters it in
   default-head comparisons.
4. **`prediction_metrics` cannot rank encoders for abstaining.** Fixed thresholds are confounded
   by each head's confidence scale, and ECE can be near zero for a useless model. This
   benchmark adds accuracy at fixed coverage, AURC, log loss and Brier score. The shared
   metrics are unchanged.

## Reproduce

```bash
python scripts/router_cuisine_benchmark.py extract --archive yelp_dataset.tar --output c.jsonl
python scripts/router_cuisine_benchmark.py contrastive --data c.jsonl --log r.jsonl --work tmp \
  --batch-size 8 --max-seq-length 128 --mxbai-path <float32-copy-of-mxbai>
python scripts/router_cuisine_benchmark.py frozen --data c.jsonl --log r.jsonl --work tmp \
  --batch-size 8 --max-seq-length 128 --eval-shots 8 32 --c-grid 10 --skip-tfidf \
  --mxbai-path <float32-copy-of-mxbai>
python scripts/router_cuisine_benchmark.py report --log r.jsonl
```

Create the float32 copy with `SentenceTransformer(name).float().save(path)`. Wrap long runs in
`systemd-inhibit --what=sleep`, because suspending pauses them. Raw logs are not committed,
and review text from the Yelp dataset must not be committed.
