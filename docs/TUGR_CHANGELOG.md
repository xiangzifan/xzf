# TUGR v1 code changes

Date: 2026-10-21

Parent: `IN2R_A5Delete_RESB_v1_20261020`

This file records implementation facts. It does not claim a retrieval gain;
that claim requires the planned AutoDL reproduction.

## `co_train.py`

### GraphRefiner

- Added four controlled modes: `mean`, `dua_pool`, `qdmp_mean`, and
  `tugr_full`.
- Added DUA support utility: positive query/support cosine minus a
  temperature-scaled LogMeanExp over the other queries in the mini-batch.
- Added a batch-size-one finite fallback.
- Detached utility routing by default; an explicit opt-in flag enables the
  end-to-end variant.
- Added QDMP by adding standardized sender utility to the last (sender) axis
  of every attention head's logits.
- Added DUA pooling and soft effective-K diagnostics.
- Kept the legacy unused `edge_nn` module frozen so that the original random
  initialization sequence and state layout are not silently changed.
- Added strict shape, device, dtype, finite-value, and hyperparameter checks.
- Added cumulative utility, pooling, effective-K, and message-entropy logs.

### Runtime construction and optimization

- Centralized Memory and GraphRefiner creation in
  `ensure_rectification_modules`.
- Added only trainable GraphRefiner parameters to Adam; frozen legacy edge
  parameters are excluded.
- Changed gradient clipping from `net.params` only to every unique parameter
  registered in the optimizer, so both GraphRefiners are clipped.
- Explicitly switches the writer GraphRefiners to train mode and the peer
  GraphRefiners to evaluation mode for each co-training phase.
- Replaced broad iterator exception handling with `StopIteration` handling so
  data/shape/runtime failures are no longer silently treated as epoch wraps.

### Loss accounting

- Split i2t and t2i unlabeled losses into independent variables and meters.
- Removed the previous last-assignment behaviour where the t2i branch could
  overwrite the reported i2t unlabeled loss.
- Non-finite branch losses are skipped and replaced by zero for aggregate
  logging; non-finite gradients still skip the optimizer update.

### Checkpointing and strict resume

- Numbered/model-best checkpoints retain the encoder and GraphRefiner weights
  needed to identify the trained method.
- A single `resume_latest.pth.tar` stores the complete epoch-boundary runtime
  state: both encoders, both directional GraphRefiners for both networks, both
  Memory banks, both Adam states, GMM loss history, best score, and Python,
  NumPy, PyTorch, and CUDA RNG states.
- Added strict resume-configuration checks for dimensions, K, Memory mode and
  all TUGR routing settings.
- The resume file is overwritten each epoch instead of duplicating large
  Memory and optimizer states in every numbered checkpoint.

## `memory_bank.py`

- Passes the current query tensor to GraphRefiner after support retrieval.
- Keeps support selection under `no_grad`, while the subsequent GraphRefiner
  path remains differentiable.
- Upgraded Memory serialization to schema 2: only the valid prefix is saved,
  then restored into the configured full-capacity tensors.
- Retains backward loading support for the earlier schema-1 full-capacity
  state.

## `run.py`

- Added TUGR mode, attention-head, dropout, utility-temperature,
  pool-temperature, message-strength, detach/gradient, and epsilon arguments.
- `quality_diverse + tugr_full + K=5` is the default full method.
- Added argument-domain validation.
- Added `--resume` for strict epoch-boundary continuation.
- Added `--co_train_epochs`, allowing the established fixed-20 execution to
  retain `num_epochs=40` as the cosine-scheduler horizon without an external
  process killer.

## `utils.py`

- Checkpoints are first written to a temporary sibling file and then replaced
  atomically. The same rule is used for `model_best.pth.tar`.

## Tests

- Updated the Memory/Refiner interface test to verify that the exact query
  tensor reaches the refiner.
- Added `test_tugr.py` covering legacy mean equivalence, analytical DUA,
  batch-size-one handling, all factorial modes, unit-norm outputs, finite
  backward gradients, frozen legacy edges, input validation, and runtime-state
  restoration.
- Added an atomic-checkpoint round-trip test that verifies both the numbered
  checkpoint and `model_best.pth.tar` and rejects leftover temporary files.

## Deliberately not included

- QAD hard query gating;
- G5/R5 relation mismatch as a routing prior;
- the old discrete/prefix Adaptive-K;
- clean/noisy ground-truth labels in Memory or GraphRefiner;
- online leave-one-out counterfactual utility;
- test-set-driven parameter selection.
