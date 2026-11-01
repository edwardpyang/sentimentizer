# Model Training and Checkpointing

This document describes how to train Sentimentizer models (RNN, Encoder, Decoder, or ModernBERT), manage training checkpoints, and leverage distributed training with Ray Train.

## Prerequisites

Before starting training, you must download the pre-trained GloVe word embeddings and obtain the Yelp reviews dataset.

```bash
# Downloads GloVe embeddings and prints Yelp dataset instructions
make download-data
```

`scripts/download_data.sh` fetches GloVe automatically. The Yelp dataset cannot
be downloaded programmatically — the script prints the manual steps: register at
<https://www.yelp.com/dataset>, download the archive, and place it at
`static/yelp_dataset.tar`.

---

## Single-Node Training

To run the pipeline on a single node, you can use the `sentimentizer` CLI.

### Pipeline Stage Isolation

The pipeline is split into three main stages (extract -> tokenize -> train). You can run the entire pipeline at once, or run stages independently.

```bash
# Run the entire pipeline: extract -> tokenize -> train
sentimentizer run --save

# Extract stage only (outputs static/reviews.parquet)
sentimentizer extract --stop 10000

# Tokenization stage only (outputs static/dictionary.txt and static/dataset.parquet)
sentimentizer tokenize

# Train stage only (requires existing parquet/dictionary artifacts)
sentimentizer train --save
```

### Global Options

The CLI supports global options that must be provided **before** the subcommand:

- `--model [rnn|encoder|decoder|modernbert]`: Model type to train (default: `rnn`).
- `--device [auto|cuda|mps|cpu]`: Device to use (default: `auto`).
- `--run-type [new|update]`: controls fresh-build vs reuse. `new` (default) creates new pipeline artifacts. `update` reuses existing ones.

*Example:*
```bash
sentimentizer --model encoder --device cuda train --save
```

### Sleep Prevention (Linux only)

To prevent your system from suspending or sleeping during long-running training epochs, the Makefile targets automatically wrap execution using `systemd-inhibit` (if available on the host OS):

```bash
systemd-inhibit --what=sleep --who='training' --why='Model training in progress' --mode=block
```

If `systemd-inhibit` is not found, it falls back to direct execution. You can disable this behavior manually by running:

```bash
make train INHIBIT_SLEEP=
```

---

## Distributed Training

Sentimentizer supports multi-GPU/multi-node distributed training using Ray Train.

```bash
# Run distributed training with 2 Ray Train workers
sentimentizer train --distributed --num-workers 2 --save
```

Under the hood, Ray Train orchestrates PyTorch `DistributedDataParallel` (DDP) across the workers. Prometheus metrics are collected from worker rank 0 and published to the metrics endpoints.

Single-node, callback, and Ray worker loops share the same accumulation and
per-update scheduler implementation. Each epoch starts with a fresh accumulation
counter. An incomplete final window averages the microbatches it actually saw,
then performs one optimizer update and one scheduler step. Scheduler budgets
include this final update using `ceil(num_batches / gradient_accumulation_steps)`.
DDP continues to synchronize every backward pass.

---

## Checkpointing

Model checkpointing is enabled by default during training to protect against system crashes, preemptions, or accidental interruptions.

### Checkpoint Structure

Checkpoints are saved inside the `checkpoints/<model_type>/` directory:
- `checkpoint_epoch_<epoch>.pth`: Periodic checkpoint saved at the end of each epoch (frequency controlled by `--checkpoint-every`).
- `best_model.pth`: Saved whenever the validation loss improves. `--save` writes these best weights rather than the final epoch's; without a checkpoint directory it falls back to the final epoch.
- `checkpoint_epoch_<epoch>/` and `best_model/`: Per-checkpoint `backbone/` for ModernBERT, about one full model per retained checkpoint. GloVe models keep everything in the `.pth`.

Epochs are ordered numerically when finding the latest checkpoint. Distributed runs store Ray checkpoints in `--checkpoint-dir` (or Ray's default location) every epoch, keeping the best by validation loss and the latest; `--checkpoint-every` does not apply to them.

A checkpoint file contains a dictionary of:
- `model_state_dict`: Model weights.
- `optimizer_state_dict`: Optimizer states.
- `scheduler_state_dict`: Learning rate scheduler states.
- `epoch`: The epoch number at which the checkpoint was saved.
- `val_loss`: Validation loss for tracking the best model.
- `config`: Hyperparameters and configuration.

### CLI Checkpoint Controls

- **Save Checkpoint Every N Epochs**: `--checkpoint-every N` (default: 1)
- **Disable Checkpoints / Saving**: `--no-save` (disables all checkpoint saving and weight persistence)
- **Resume Training**: `--resume` (or `--resume-train` inside `sentimentizer run`) resumes training from the latest checkpoint found in the checkpoint directory. It requires `--checkpoint-dir` and is not supported with `--distributed`, which also rejects `--run-type update`.

*Example (Resuming training):*
```bash
sentimentizer train --resume --checkpoint-dir checkpoints/rnn --save
```

---

## CLI Options Reference

Below is a complete reference of the CLI flags available under the `train` (and `run`) subcommands.

| Option | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--distributed` | flag | `False` | Use Ray Train for distributed training. |
| `--num-workers` | integer | `1` | Number of Ray Train workers to spin up. |
| `--save` / `--no-save` | flag | `--no-save` | Persist final weights & checkpoints after training. |
| `--checkpoint-dir` | string | `""` | Custom directory for saving checkpoints. |
| `--checkpoint-every` | integer | `1` | Save checkpoint every N epochs. |
| `--balance-classes` | flag | `False` | Enable class balancing (undersamples majority classes). |
| `--balance-seed` | integer | `42` | Random seed to use for class balancing. |
| `--num-classes` | integer | `3` | Number of classification classes (must be 3 for sentiment). |
| `--include-neutral` / `--no-include-neutral` | flag | `True` | Include 3-star (neutral) reviews in training data. |
| `--loss-type` | `cross_entropy`, `focal` | `cross_entropy` | Loss function type. |
| `--focal-gamma` | float | `2.0` | Focal loss focusing parameter (only used with focal loss). |
| `--label-smoothing` | float | `0.1` | Label smoothing for CrossEntropyLoss (0.0 = no smoothing). |
| `--weight-smoothing` | float | `0.5` | Exponent on inverse-frequency class weights (1.0=full, 0.0=uniform). |
| `--neutral-oversample-ratio` | float | `0.0` | Oversample neutral class to this ratio (e.g. 0.20 = 20%, 0.0 = disabled). |
| `--balance-strategy` | `class_weights_only`, `undersample`, `oversample` | `class_weights_only` | Class balancing strategy. |
| `--freeze-embeddings` / `--no-freeze-embeddings` | flag | `True` | Freeze pre-trained GloVe embedding weights. |
| `--push-to-hub` | flag | `False` | Push model weights to Hugging Face Hub after successful training. |
| `--pull-from-hub` | flag | `False` | Pull model weights from Hugging Face Hub before starting training. |
| `--hf-repo` | string | `None` | Hugging Face repository ID. |
| `--run-id` | string | `None` | Unique training run ID for metrics tracking. |
| `--ray-update-every` | integer | `-1` | Batch update/logging frequency for distributed training (-1 = auto). |
| `--resume` | flag | `False` | Resume training from latest checkpoint in checkpoint directory. |
