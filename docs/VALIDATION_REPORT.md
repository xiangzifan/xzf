# Local validation report

Date: 2026-10-20

## Passed

1. Every Python file compiled with `python -m compileall`.
2. Eight deterministic CPU unit tests passed:
   - stable record IDs after labeled-subset reconstruction;
   - FIFO wrap and original top-K retrieval behaviour;
   - single-slot ID refresh and metadata update;
   - exact self-record exclusion;
   - reliability reranking;
   - MMR value diversity;
   - Memory state round-trip;
   - gradient flow from selected supports into a differentiable refiner.
3. The `GraphRefiner` source block is byte-for-byte identical to the parent
   A5-Delete block.
4. All files declared unchanged in `RESB_CHANGELOG.md` match their parent
   SHA-256 hashes.
5. Every entry in `code_checksums.sha256` matched the delivered file.

## Environment-limited check

Importing the complete training entry point on this Windows host fails inside
the unchanged `vse.py` import of `torchtext`, with `OSError: [WinError 127]`.
The installed `torchtext` binary is incompatible with the installed PyTorch
runtime.  `vse.py` itself is byte-identical to the A5-Delete parent, so this is
an environment dependency failure rather than a RESB regression.

No claim is made that a complete GPU training epoch was run locally.

## Required AutoDL smoke test before a full experiment

In the known working IN2R environment:

1. run `python -m unittest -v test_memory_bank.py`;
2. launch `run.py` with the established A5-Delete paths/noise map, one short
   controlled job per Memory mode;
3. confirm that both model phases reach validation and that Memory diagnostics
   are finite;
4. only then launch the full seed-42 ablation.
