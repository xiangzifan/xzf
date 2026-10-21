# IN2R Research Repository

Private research repository for reproducible experiments derived from the official
[IN2R project](https://github.com/liuyyy111/IN2R).

The current mother code is the verified **A5-Delete + Reliability-aware Evolving
Support Bank (RESB v1)** snapshot used in the controlled AutoDL experiments on
2026-10-20. Future changes should start from this repository rather than from
dated ZIP copies.

> This repository is private because the upstream repository currently does not
> publish an explicit software license. Do not make this repository public or add
> an open-source license until redistribution permission is confirmed.

## Mother-code scope

The repository keeps one coherent training stack:

- `run.py`: co-training entry point.
- `co_train.py`: A5-Delete co-training and GraphRefiner pipeline.
- `memory_bank.py`: RESB memory implementation.
- `data.py`: dataset loading and stable record identifiers.
- `model.py`, `vse.py`, `vse_.py`: retrieval models and losses.
- `evaluation.py`: validation and test metrics.
- `train.py` and `single_run.py`: single-model/warm-up path.
- `utils.py` and `vocab.py`: shared utilities.
- `tests/test_memory_bank.py`: CPU-compatible Memory regression tests.

Deliberately excluded: datasets, GloVe files, NLTK assets, noise-map binaries,
checkpoints, logs, output directories, SSH credentials, ZIP/TAR archives,
historical duplicate source trees, and the old hard-coded `test.py`.

## RESB modes

`run.py` exposes a common `--memory_mode` switch:

- `fifo`: original circular FIFO and raw top-K retrieval.
- `id_refresh`: stable-record refresh and exact self-record exclusion.
- `quality_diverse`: reliability/freshness reranking plus value-space MMR.

For a controlled ablation, change only `--memory_mode`; keep the data, noise
map, seed and remaining hyperparameters fixed.

## Verified environment

The controlled AutoDL runs used:

- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1
- torchvision 0.16.0
- torchtext 0.16.0
- torchdata 0.7.0
- NumPy 1.26.4
- SciPy 1.11.4
- scikit-learn 1.3.2
- NLTK 3.8.1
- matplotlib 3.7.5

Install the pinned direct dependencies with:

```bash
python -m pip install -r requirements.txt
```

## CPU regression test

Run from the repository root:

```bash
python -m unittest -v tests/test_memory_bank.py
```

At initialization, all eight Memory tests passed.

## AutoDL workflow

Clone once into persistent storage:

```bash
cd /root/autodl-fs
git clone https://github.com/zifanx0329-cpu/IN2R.git IN2R_repo
cd IN2R_repo
```

Before every experiment:

```bash
git fetch origin --tags
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

Datasets, assets and outputs must remain outside this checkout. The archived
controllers under `scripts/autodl/` preserve the exact paths and policies used
for the 2026-10-20 experiment; review their path variables before reuse.

## Reproducibility rule

Every experiment output must record at least:

- Git commit SHA and dirty/clean status.
- Complete launch command and parsed configuration.
- Dataset/noise-map checksum.
- Python package snapshot and GPU information.
- Validation curve, selected checkpoint and final test metrics.

## Known limitation

The current training checkpoint is not yet a strict-resume checkpoint. It does
not persist the complete GraphRefiner, optimizer/scheduler, loss-history, RNG and
full Memory runtime state as one atomic training state. Restarting from an
encoder checkpoint is therefore not equivalent to uninterrupted training.
Implementing and testing strict resume should be the first major repository
change.

## Attribution

This work builds on the official IN2R implementation and also follows components
credited upstream to NCR, GPO and ConVSE. Preserve upstream attribution in future
changes. No license is granted by this repository.
