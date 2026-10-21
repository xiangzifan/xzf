# QASR changelog

## 2026-10-22 — QASR v1

Base: `c787d56387b5268871dc4ef8e9ef7555dfc06040`.

- Added the independent `qasr_full` GraphRefiner mode.
- Preserved all legacy Mean/TUGR forward branches.
- Added bounded query relevance, support consensus, and RESB
  quality-freshness evidence without batch-negative coupling or per-query
  variance normalization.
- Added a bounded query-anchor outer-product prior to support graph logits.
- Added vectorized Sparsemax pooling with exact zero support weights.
- Added an evidence-gated residual from the legacy mean prototype to the sparse
  selected prototype.
- Added explicit selected-slot metadata transfer from RESB Memory to metadata-
  aware refiners while retaining the old refiner API for all other modes.
- Added QASR CLI validation, strict-resume configuration checks, and diagnostics.
- Added same-code RESB-Mean and QASR fixed seed-42 launchers with clean-tree,
  source-compile, unit-test, environment, Git, and noise-file evidence gates.
- Expanded deterministic CPU coverage from 16 to 22 tests.

No retrieval improvement is claimed until the controlled AutoDL runs finish.
