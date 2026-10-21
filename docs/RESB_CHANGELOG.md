# RESB v1 controlled change record

## Parent

This package was copied from:

`code_new/母版（之前的A5-delete）/IN2R_A5_delete_boundary_tokens/IN2R-main`

The parent directory was not modified.

## Algorithmic scope

Only co-training Memory construction and retrieval are changed.  The following
remain the A5-Delete implementation:

- encoders and similarity model;
- boundary-token deletion;
- clean/noisy split and peer clean probability;
- fixed `K=5`;
- GraphRefiner layers, attention, mean pooling, and normalization;
- labeled and unlabeled losses and their weights;
- optimizer and learning-rate schedule.

## Files changed

### `memory_bank.py` (new)

Implements one controlled class with three modes:

1. `fifo`: original circular FIFO and raw top-K retrieval;
2. `id_refresh`: stable-record consolidation, reliability-aware EMA refresh,
   and exact self-record exclusion;
3. `quality_diverse`: ID refresh plus candidate reranking by peer clean
   probability and feature freshness, followed by MMR selection in the paired
   value modality.

The class contains no trainable parameters and never accepts synthetic
clean/noisy ground-truth labels.

### `data.py`

Preserves the original training-record index before rebuilding the labeled and
unlabeled subsets.  This ID is returned in the existing loader ID field.  No
label or ground-truth mapping is added.

### `co_train.py`

- imports and constructs the selected Memory mode;
- passes stable record IDs and existing peer clean probabilities on enqueue;
- passes query record IDs for exact self-exclusion;
- passes the current epoch for feature-freshness calculation;
- emits cumulative Memory diagnostics after each network-training phase.

The `GraphRefiner` block is byte-for-byte identical to the parent block.

### `run.py`

Adds validated command-line options for Memory mode and RESB hyperparameters.
The default mode in this experimental package is `quality_diverse`.

### `test_memory_bank.py` (new)

CPU tests cover FIFO wrap/retrieval, stable IDs after subset reconstruction,
single-slot ID refresh, self-record exclusion, reliability reranking, MMR
diversity, and complete state round-trip.

## First controlled experiment

Keep every non-Memory option identical and compare:

```text
--memory_mode fifo
--memory_mode id_refresh
--memory_mode quality_diverse
```

Do not combine this first experiment with Adaptive-K, RelationPool, QAD, or a
new loss.  If RESB is independently stable, combination tests can follow.

## Verification performed before packaging

- all Python files compiled successfully;
- eight deterministic CPU unit tests passed;
- the GraphRefiner block matched the A5-Delete parent exactly;
- all unmodified source files matched their parent SHA-256 hashes;
- the package manifest is `code_checksums.sha256`.
