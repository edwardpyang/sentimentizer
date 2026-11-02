# Router long-prompt benchmark plan

**Status:** ⏳ Partially implemented (2026-11-01). A draft script and tests exist, uncommitted. They
have only run against a fake model, and the review below found defects to fix first. No long-prompt
run has started. Do not start one until this plan is approved.

## Goal

Find which encoder, window size and aggregation method can route a prompt of up to 200k tokens,
given a short deciding signal at an unknown position. No encoder checked reads 200k tokens in one
pass, so every candidate splits the prompt into windows.

## Scope

**In scope**

- A synthetic benchmark built from the Yelp archive and the cuisine labels.
- The encoders, window sizes and aggregation methods listed below.
- Text-generation models and an end-to-end `ModernBERT` classifier as comparisons.
- A results section in `reports/router-encoder-benchmark.md`.

**Out of scope**

- Changing `RouterConfig`, `RouterModel`, the default encoder, or any production code.
- Serving long prompts. `max_text_length` is 10,000 characters and the body limit is 1 MiB, so a
  200k-token prompt gets a 422 today. That needs its own plan once a method is chosen.
- Contrastive fine-tuning at long windows. All heads here sit on frozen encoders.
- Abstaining on prompts with no signal. This is the most useful follow-up.

**What the results cannot show.** This measures whether a method survives the signal's position and
dilution. It does not predict real routing accuracy. Real prompts differ in filler, signal and
labels, so a labeled sample of real prompts is still required before any production change.

## Test design

A prompt is a cuisine review (the signal) inside filler taken from non-food Yelp reviews that never
name a cuisine. The label is the signal's cuisine, so chance accuracy is 10%.

| Factor | Values |
|---|---|
| Prompt tokens | 8,192 / 32,768 / 200,000, skipping any shorter than the window |
| Signal position | start / middle / end / scattered (3 reviews from one restaurant) |
| Aggregation | first window / last window / first and last / most cuisine-confident window / mean of all windows |
| Training | 32 reviews per class, each placed in a window of the tested size, plus an equal count of filler windows as an 11th class |
| Test | 20 prompts per class, from businesses held out of training |
| Repeats | 3 seeds |
| Head | L2-normalized logistic regression, `C=10` |

### Configurations

| Encoder | Weights | Window tokens | Runtime |
|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | float32 | 512 | sentence-transformers |
| `mixedbread-ai/mxbai-embed-large-v1` | float16 | 512 | sentence-transformers |
| `BAAI/bge-m3` | float32 | 512, 8,192 | sentence-transformers |
| `answerdotai/ModernBERT-large` | float32 | 8,192 | sentence-transformers |
| `Qwen/Qwen3-Embedding-0.6B` | bfloat16 | 512, 8,192, 32,768 | sentence-transformers |
| `Qwen/Qwen3-Embedding-4B` | bfloat16, about 8 GB | 8,192 | sentence-transformers |
| `nvidia/Nemotron-3-Embed-1B-BF16` | bfloat16 | 8,192, 32,768 | sentence-transformers |
| `alexliap/Qwen3-Embedding-8B-NVFP4` | NVFP4, 5.15 GB | 8,192, 32,768 | vLLM 0.25.0 |
| `nvidia/Nemotron-3-Embed-1B-NVFP4` | NVFP4, 1.03 GB | 8,192, 32,768 | vLLM 0.25.0 |

Full-precision `Qwen/Qwen3-Embedding-8B` is excluded because its 15.13 GB of weights leave almost
nothing on a 16 GB card.

### Text-model comparisons

These test whether a model that reads the text directly beats an embedding and a head. They are the
only candidates that might read a whole 200k-token prompt in one pass, so they are part of the
benchmark rather than an optional extra.

| Comparison | Task | Status |
|---|---|---|
| Text-generation model, no examples | Cuisine ranking set and long prompts | Needs a model choice |
| Text-generation model, 8 and 32 examples per class | Cuisine ranking set only, since examples add to prompt length | Needs a model choice |
| End-to-end `ModernBERT` classifier at up to 8,192 tokens | Cuisine ranking set, then long prompts | Needs new training code, because the repo's version targets the 3-class sentiment task. Test more than 32 examples per class |

**Text-generation design**

- Use the same seeds, splits and test reviews as the encoder ranking, so results pair seed by seed.
- Keep the prompt template, labels and generation settings as module constants. Use temperature 0.
- Restrict output to the 10 labels where the runtime supports it. Otherwise parse strictly, score any
  unparseable reply as wrong, and report how many there were.
- Report accuracy, macro-F1 and no-keyword accuracy. Report AURC and log loss only if the runtime
  returns usable label probabilities, because a bare label has no confidence to rank.
- Record input tokens, seconds per prompt, peak GPU memory and any cost for every run.
- Treat prompt text as untrusted data, as `.rules/untrusted-input.md` requires.

**Model candidates, checked 2026-11-01**

| Option | Examples | Fit |
|---|---|---|
| Local NVFP4 | `nvidia/Qwen3.6-27B-NVFP4` (21.9 GB), `RedHatAI/gemma-4-26B-A4B-it-NVFP4` (16.4 GB), `nvidia/Qwen3.6-35B-A3B-NVFP4` (23.4 GB) | The weights alone do not fit on a 16 GB card |
| Local, smaller | Not chosen yet. Weights must stay near 10 GB to leave room for context | Runs short reviews. Long windows and 200k tokens are unlikely to fit |
| Local with CPU offload | Any of the above, using the machine's 60 GB of RAM | Possible but slow. Measure before relying on it |
| Hosted | `glm-5.2:cloud` and `kimi-k2.6:cloud`, already configured in Ollama here | Most likely route to 200k tokens. Sends text to a third party, and context limits are unverified |

Listed limits of 262,144 positions come from model configs and are not proof that a model works at
that length. Nemotron's embedding model lists the same value but supports 32,768.

**Prompt budget.** Long prompts at 8k, 32k and 200k tokens, 4 positions, 20 prompts per class and 3
seeds come to 2,400 prompts per length and about 576 M input tokens. Pilot with 1 seed and 5 prompts
per class, about 48 M input tokens, and record the measured cost before going further.

Synthetic prompts can measure dilution and false alarms. They cannot measure links between distant
parts of a prompt, because each synthetic signal stands alone. That needs real prompts.

### Compute budget

Each seed embeds 920 signal windows (320 train, 600 test). Each configuration also embeds its filler
pool once.

| Window | Configurations | Windows each, all seeds | Tokens each | Tokens total |
|---|---|---|---|---|
| 512 | 4 | 5,885 | 3 M | 12 M |
| 8,192 | 7 | 2,955 | 24 M | 169 M |
| 32,768 | 4 | 2,808 | 92 M | 368 M |

The total is about 550 M tokens. This is the full matrix. At the estimated speeds it would take weeks, so
the phases below run a much smaller set first.

## Architecture

```
scripts/
  router_long_prompt_benchmark.py        NEW  extract-filler and run stages
  router_cuisine_benchmark.py                 unchanged; supplies splits, scoring, ResultLog, report
tests/
  test_router_long_prompt_benchmark.py   NEW
reports/
  router-encoder-benchmark.md            MOD  long-prompt section
docs/
  router.md                              MOD  link to the results
```

Data lives in `~/projects/data/router_cuisine/`, outside the repo. Review text must never be
committed. vLLM has its own venv there (`vllm-venv/`), so its dependencies cannot affect the project
environment. The benchmark calls it over localhost.

## Public interfaces

```bash
python scripts/router_long_prompt_benchmark.py extract-filler --archive <tar> --output <jsonl>
python scripts/router_long_prompt_benchmark.py preflight --filler <jsonl> --only <configs>
python scripts/router_long_prompt_benchmark.py run --data <jsonl> --filler <jsonl> --log <jsonl> \
    --work <dir> [--seeds ...] [--doc-tokens ...] [--only <name@window> ...] [--vllm-url <url>]
```

Results append to the JSONL log with `kind="long_prompt"`, `encoder="<name>@<window>"`,
`shots=<prompt tokens>` and `head="<method>/<position>"`. The existing `report --log` reads them.

New or changed functions for the Phase 1 fixes:

```python
def split_filler(stream: list[int]) -> tuple[list[int], list[int]]: ...
def run_preflight(args: argparse.Namespace) -> None: ...
def cache_stem(kind: str, seed: int | None, args: argparse.Namespace) -> str: ...
```

`run_seed` and `evaluate_documents` change to take separate train and test filler. They have one
call site each.

## Implementation order

The work is split so the cheapest answer comes first, and each phase decides whether the next is worth
running. Every code step ends with `make ci`. Time estimates extrapolate from one measurement,
`bge-base-en-v1.5` at 512 tokens (115,645 tokens/s, 1.1 GB peak), and could be off by about 3x either way.
Preflight replaces them with measurements.

### Phase 1: fast comparison (about 3 hours)

Does a longer window beat small windows with good aggregation? This phase uses 1 seed and 10 prompts
per class, at all three prompt lengths.

| Configuration | Tokens | Estimated time |
|---|---|---|
| `bge-base-en-v1.5@512` | 2 M | Under 1 min |
| `mxbai-embed-large-v1@512`, `bge-m3@512` | 2 M each | About 1 min each |
| `qwen3-0.6b@512` | 2 M | About 2 min |
| `bge-m3@8192` | 7 M | About 50 min |
| `modernbert-large@8192` | 7 M | Up to about 1 hour |
| `qwen3-0.6b@8192` | 7 M | About 80 min |

1. **Fix the draft script.** Resolve every open item in the review findings and add a regression test
   for each. Add the `preflight` stage.
2. **Preflight the Phase 1 configurations.** Encode a few full-length windows, and record peak GPU memory
   and tokens per second. Record out-of-memory as a result.
3. **Pilot.** Run `bge-base@512` first and check the sanity conditions in the analysis section.
4. **Run the rest**, cheapest first, and report the results.

**Gate:** the user decides on Phase 2 from these results:

- If a 512-token window already holds up at every position, longer windows have little left to prove.
  Skip the rest of Phase 2.
- If accuracy improves from 512 to 8,192, Phase 2 tests whether that continues at 32,768.
- If nothing beats chance at 200k tokens, go to Phase 4, since text-generation models become the main
  hope.

### Phase 2: long windows and NVFP4 (about 2 days)

Uses the same 1 seed and 10 prompts per class.

1. **Validate NVFP4 embeddings.** Start one vLLM server per model. Embed 200 short reviews and compare
   them with the full-precision reference by cosine similarity, running the reference on CPU if it
   will not fit on the GPU. Require a median above 0.95, a threshold chosen to catch broken pooling
   rather than small quantization drift. Exclude any checkpoint that fails and record why.
2. **Preflight, then run:** `qwen3-0.6b@32768` (about 17 hours), `qwen3-8b-nvfp4@8192` (about 15 hours),
   and `nemotron-1b-nvfp4@8192` (about 2 hours). Add `nemotron-1b-nvfp4@32768` (about 30 hours) only if
   `qwen3-0.6b@32768` beat its 8,192-token result.

### Phase 3: confirm the leaders (time set by earlier measurements)

Run the top two or three configurations with 3 seeds and 20 prompts per class. The other configurations,
including `qwen3-4b@8192`, `nemotron-1b` BF16 and `qwen3-8b-nvfp4@32768` (estimated at hundreds of hours), run
only if a result points to them.

### Phase 4: text-generation models

1. Add the adapter and prompt constants, with tests using a fake server.
2. Run the chosen models on the cuisine ranking set, at 0, 8 and 32 examples per class.
3. Run the long-prompt pilot from the prompt budget. Continue only if the measured time and cost are
   approved. Hosted runs need explicit approval first.

### Phase 5: end-to-end `ModernBERT`

Add training code for arbitrary labels, then run it on both tasks.

### Report

Update `reports/router-encoder-benchmark.md` and `docs/router.md` after each phase. Mark this plan
complete when the phases that were run are reported and the rest are recorded as skipped, with reasons.

## Analysis

**Primary measure:** accuracy on 200k-token prompts at each position, with paired differences across
seeds.

**Choosing a method:** rank by accuracy at the worst position, since the real signal position is
unknown. Treat gaps smaller than the paired interval as ties and break them by lower compute. With 3
seeds, the 95% t-interval is wide, so report small gaps as unresolved.

**Sanity conditions, checked on the pilot:**

- The first-window method with the signal at the end must score near chance (below about 15%). A
  higher score means filler or context is leaking label information.
- The first-window method with the signal at the start must match the last-window method with the
  signal at the end, within noise.
- At a 512-token window, single-window accuracy should be broadly in line with the frozen 512-token
  results, after allowing for filler sharing the window.

**Report for every configuration:** accuracy by position and prompt length, tokens per second, peak
GPU memory and windows per 200k-token prompt.

## Testing strategy

`tests/test_router_long_prompt_benchmark.py` runs on CPU with a fake model and covers:

- Signal placement, truncation and window positions.
- Filler extraction, including no partial output on failure.
- Aggregation that ignores confident filler windows.
- Methods scoring well only when they can see the signal, and at chance otherwise.
- Resume with no re-encoding, and failures leaving no cached file.
- vLLM response ordering, truncation and server failure.

New for the Phase 1 fixes:

- Train and test filler are disjoint.
- Changing a run parameter does not reuse cached embeddings.
- The vLLM adapter applies the document prompt.
- Timing reaches the log.

## Review findings

First review of the draft script. All are open.

| Finding | Effect | Resolution |
|---|---|---|
| The window head trains on filler windows that also appear in test prompts | Flatters the most-confident-window method by suppressing false positives on filler it has memorized | Split the filler stream in two. Train and test draw pool windows and signal context from separate halves |
| Cache names and log keys omit training size, test size and pool size | A rerun with different settings can silently reuse stale embeddings or skip runs | Put those parameters in the cache name. Use a separate log per parameter set |
| Run `seconds` is logged as 0 | Throughput, a required output, is lost | Log per-configuration encode time and token count |
| The vLLM adapter does not apply `passage: ` but the sentence-transformers path does | Nemotron BF16 and NVFP4 would not be comparable | Apply the document prompt in both paths |
| A vLLM failure stops the run, unlike the handling of out-of-memory | Only matters if several models share one run | Keep it: `run_vllm` serves one model at a time, so a failure should be loud |

Second review, after those resolutions:

| Risk | Resolution |
|---|---|
| 8B NVFP4 is a third-party build with no sentence-transformers pooling metadata, and vLLM may pool it wrongly | Phase 2 compares it with the original. Exclude it if that fails |
| vLLM 0.25.0 is NVIDIA's tested version for Nemotron, not necessarily for the Qwen build | Phase 2 validation covers this. Record the outcome rather than assuming |
| `truncate_prompt_tokens` and `--max-model-len` behavior is unverified for this path | Phase 2 validation sends over-length input and checks the result |
| `Nemotron-3-Embed-1B-BF16` is named in the NVFP4 model card but has not been loaded | Preflight confirms it if that configuration is ever run. Drop the configuration if it fails |
| Long windows run out of memory | Recorded as a result. The batch size is already 1 at 32,768 |
| The full run may be too slow | Phases run the cheapest configurations first, and preflight measures before each |
| Signals never straddle a window boundary | Flatters all windowed methods, most of all small windows. Stated in the report. A straddling test is a possible follow-up if small windows win narrowly |
| Filler differs from real prompts | Stated in the report. Only a real prompt sample resolves it |
| A hosted model receives review text from a third-party dataset, and later real prompts | No hosted run starts without explicit approval. Check the Yelp dataset terms first |
| Text-generation models cannot fit 200k tokens on this card | Expected for local runs. Report the longest length that ran, and record out-of-memory as a result |
| Text inside a prompt can steer a text-generation router, including a malicious instruction | Add a test prompt with an injected routing instruction. Report the hijack rate beside accuracy |
| Few-shot examples lengthen every prompt, at about 40k tokens for 32 per class | Limit few-shot runs to the ranking set, and record the input tokens |
| Frozen `ModernBERT` embeddings carry little signal: `modernbert-base` scored 0.257 at 8 per class and 0.422 at 32 at 128 tokens | Keep `ModernBERT-large@8192` as a long-window reference, but expect it to rank low. Long-prompt runs do not fine-tune it |
| Hub downloads stall, as the Xet transfer did for mxbai | Set `HF_HUB_DISABLE_XET=1`. Partial downloads resume |

## Assumptions

Checked 2026-11-01 from Hub metadata unless marked otherwise.

- Token limits: `bge-m3` and `ModernBERT-large` 8,192; `Qwen3-Embedding` 0.6B 32,768, 4B and 8B
  40,960; Nemotron 32,768 per its model card (its config's 262,144 positions is not the supported
  limit).
- Filler supply: 1.36 M eligible non-food reviews before filtering. A sample of 30,000 is extracted,
  which should cover the 3.2 M tokens split filler needs (estimated, not yet counted).
- RTX 5080 with 16 GB. Whether long windows fit is **not yet tested**.
- This hardware runs NVFP4 through vLLM: **not yet tested**.
- `sentence-transformers` 5.5.0 returns bfloat16 embeddings as usable arrays: **not yet tested**.

## Convention updates

- `docs/router.md`: link to the long-prompt results and state that router input beyond
  `max_seq_length` is silently truncated.
- `AGENTS.md`: none, since no production behavior changes.
- `pyproject.toml`: no new dependencies. This branch's version bump already covers a pull request.
