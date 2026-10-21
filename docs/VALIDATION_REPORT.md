# Local validation report

Date: 2026-10-21

Target: A5-Delete + RESB M2 + TUGR-Full

## Passed locally

1. Every Python source compiled successfully with `python -m compileall -q .`.
2. Sixteen deterministic CPU tests passed:
   - eight inherited RESB/data tests;
   - exact propagation of query embeddings from Memory to GraphRefiner;
   - legacy mean-path numerical equivalence;
   - analytical two-query DUA value;
   - finite batch-size-one utility fallback;
   - shape, finiteness, and unit-norm checks for all four factorial modes;
   - finite TUGR backward gradients and frozen unused legacy edges;
   - query-alignment input validation;
   - compact Memory plus directional GraphRefiner state round-trip;
   - explicit Memory inheritance of the encoder device while CUDA availability
     is simulated independently of the CPU-resident test fixture;
   - atomic checkpoint completion and temporary-file cleanup.
3. The test command completed with no failures:

   ```text
   python -m unittest -v test_memory_bank.py test_tugr.py
   Ran 16 tests ... OK
   ```

4. The new `mean` mode was evaluated against an explicit implementation of
   the parent A5-Delete forward equations after copying the same weights; the
   outputs matched within `atol=1e-7, rtol=1e-6`.
5. DUA was checked against a closed-form two-query example with expected
   utilities `[[1, -1], [1, -1]]`.
6. Compact Memory serialization restores the ID-to-slot mapping and selected
   output; the complete rectification state restores both directional
   GraphRefiner tensors exactly.
7. The CUDA-host regression no longer depends on the test machine's actual GPU:
   it mocks CUDA availability while keeping `DummyNet` on CPU, and verifies that
   Memory follows the encoder device rather than the global CUDA fallback.

## Engineering protections added

- full-optimizer gradient finiteness checks and clipping;
- separate i2t and t2i unlabeled-loss accounting;
- atomic checkpoint replacement;
- one non-duplicated strict-resume checkpoint;
- complete Memory/optimizer/RNG epoch-boundary state;
- critical-configuration compatibility checks on resume;
- numerical and shape validation at the GraphRefiner boundary.

## Environment-limited check

Importing the unmodified `vse.py` on this Windows host fails in `torchtext`
with `OSError: [WinError 127]`. The installed torchtext binary is incompatible
with the installed PyTorch runtime. GraphRefiner tests isolate that unrelated
import; the full CUDA VSE training path cannot be executed on this machine.
The local WSL launcher also returns `E_ACCESSDENIED`, so the supplied Bash
launcher must receive its final `bash -n` check in AutoDL.

No claim is made that a complete GPU epoch or a retrieval improvement has been
observed locally.

## Required AutoDL checks before the fixed-20 reproduction

In the known working IN2R environment:

1. run `python -m unittest -v test_memory_bank.py test_tugr.py`;
2. compile with `python -m compileall -q .`;
3. run one short TUGR-Full smoke job that reaches both co-training phases and
   validation;
4. verify finite `utility_*`, `pool_*`, `effective_k`, and `message_entropy`;
5. interrupt after an epoch and resume once from `resume_latest.pth.tar`;
6. then launch the fixed seed-42 run with `num_epochs=40` as the cosine
   horizon and `co_train_epochs=20` as the actual execution budget, without
   changing the default detached-utility setting.
