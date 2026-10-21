# Local validation report

Date: 2026-10-22

Target: A5-Delete + RESB M2 + controlled Mean/TUGR/QASR GraphRefiners

## Passed locally

1. The modified Python sources compile successfully with `py_compile`.
2. Twenty-two deterministic CPU tests pass:
   - nine RESB/data/metadata-interface tests;
   - legacy mean-path numerical equivalence;
   - analytical TUGR two-query utility and batch-size-one fallback;
   - output shape, finiteness, normalization, and pooling normalization for all
     five modes;
   - Sparsemax probability-simplex projection and exact sparsity;
   - bounded QASR query/consensus/metadata evidence;
   - strict rejection of missing or misaligned QASR metadata;
   - QASR's exact fallback to the legacy mean path under uniform evidence;
   - finite TUGR and QASR backward gradients;
   - exact alignment between RESB-selected slots and quality/freshness passed
     to QASR;
   - compact Memory and directional GraphRefiner state round-trip;
   - Memory inheritance of the encoder device when CUDA availability is mocked;
   - atomic checkpoint completion and temporary-file cleanup.
3. The command completed with no failures:

   ```text
   python -m unittest -v tests/test_memory_bank.py tests/test_tugr.py
   Ran 22 tests ... OK
   ```

4. `git diff --check` reports no whitespace errors.
5. A deterministic synthetic diagnostic smoke check over three seeds produced
   finite QASR outputs, pooling sums within `1.2e-7` of one, active support
   counts around four of five, inverse-concentration effective K around three,
   and finite residual gates. This is a numerical check, not retrieval evidence.

## Regression protections

- The `mean`, `dua_pool`, `qdmp_mean`, and `tugr_full` branches retain their
  previous equations.
- Metadata is passed only when a refiner explicitly declares
  `requires_support_metadata`; the legacy refiner interface remains valid.
- QASR is rejected at the command line unless Memory mode is
  `quality_diverse`.
- All QASR hyperparameters are finite/range checked and included in strict
  resume compatibility when either side of a resume is QASR.
- QASR evidence is detached by default while the message/value path remains
  differentiable.
- Sparsemax validates finite inputs and explicitly renormalizes floating-point
  projection drift.
- Existing full-optimizer gradient checking/clipping, split i2t/t2i loss
  accounting, atomic checkpointing, compact Memory state, optimizer state and
  RNG restoration remain in place.

## Environment-limited checks

Importing the unmodified `vse.py` on this Windows host fails inside `torchtext`
with `OSError: [WinError 127]`; this is the known local PyTorch/torchtext binary
mismatch. The unit tests isolate that unrelated import. Full CUDA VSE training
must run in the established AutoDL environment.

The local WSL launcher returns `E_ACCESSDENIED`. Consequently the two Bash
launchers require a final `bash -n` check on AutoDL. Both tracked launchers run
Python compilation and all unit tests before starting training.

No QASR retrieval improvement is claimed from local testing.

## Required AutoDL gate

Before training:

1. verify the supplied commit SHA and a clean Git worktree;
2. run `bash -n` on both new launchers;
3. run the 22 unit tests and source compilation;
4. verify the fixed noise-map SHA-256;
5. start the same-code RESB-M2 `mean` control and QASR in separate output
   directories;
6. require finite QASR `pool_*`, `active_k`, `effective_k`, `message_entropy`,
   `qasr_*`, and correction-gate diagnostics;
7. choose checkpoints only by Validation and evaluate Test once after the
   configuration is frozen.
