# IN2R A5-Delete + RESB + TUGR

Private research repository for reproducible experiments derived from the
official [IN2R project](https://github.com/liuyyy111/IN2R). The current `main`
branch is built from the verified `IN2R_A5Delete_RESB_v1_20261020` mother code.

> The upstream repository currently does not publish an explicit software
> license. Keep this repository private and do not add an open-source license
> until redistribution permission is confirmed.

The default experiment combines:

1. **A5-Delete**: the confirmed IN2R baseline variant;
2. **RESB M2 (`quality_diverse`)**: ID refresh plus reliability, freshness,
   and MMR-aware support selection;
3. **TUGR-Full**: Query-conditioned Discriminative Message Passing (QDMP)
   followed by Discriminative Utility-Aligned pooling (DUA-Pool).

No synthetic clean/noisy ground-truth label is read by RESB or TUGR.

## TUGR mechanism

For query `i`, support `j`, and every query `r` in the mini-batch, TUGR first
computes cosine similarities `c[r,i,j]`. Its support utility is

```text
u[i,j] = c[i,i,j]
         - T * logmeanexp_{r != i}(c[r,i,j] / T)
```

The detached, within-query standardized utility biases each support sender in
the support self-attention logits (QDMP). A softmax over the same utility then
replaces equal mean pooling (DUA-Pool). Its inverse concentration
`1 / sum_j pi[i,j]^2` is logged as the soft effective support count.

The released default deliberately keeps the utility detached. Encoder and
GraphRefiner values remain differentiable through the retrieval objective,
while the encoder cannot manipulate the routing score as an easy shortcut.

## Controlled GraphRefiner modes

The implementation keeps a strict 2x2 factorial switch even though the first
recommended run is the full model:

| `--refiner_mode` | Message passing | Pooling |
|---|---|---|
| `mean` | original support self-attention | equal mean |
| `dua_pool` | original support self-attention | DUA |
| `qdmp_mean` | QDMP | equal mean |
| `tugr_full` | QDMP | DUA |

`tugr_full` is the default.

## Controlled Memory modes

| `--memory_mode` | Meaning |
|---|---|
| `fifo` | original circular FIFO and raw Top-K |
| `id_refresh` | one slot per stable record and self-ID exclusion |
| `quality_diverse` | ID refresh plus quality/freshness reranking and MMR |

`quality_diverse` is the default.

## Recommended Flickr30K seed-42 run

Use the same data, vocabulary, fixed 60% noise map, and training policy as the
accepted RESB seed-42 reproduction. That run completed 20 co-training epochs
while retaining a 40-epoch cosine-scheduler horizon. A typical command is:

```bash
python run.py \
  --data_name f30k_precomp \
  --data_path "$IN2R_DATA_PATH" \
  --vocab_path "$IN2R_VOCAB_PATH" \
  --noise_ratio 0.6 \
  --noise_file "$IN2R_NOISE_02" \
  --warmup_epoch 5 \
  --seed 42 \
  --num_epochs 40 \
  --co_train_epochs 20 \
  --memory_mode quality_diverse \
  --refiner_mode tugr_full \
  --k 5 \
  --refiner_heads 4 \
  --refiner_utility_temperature 0.1 \
  --refiner_pool_temperature 0.1 \
  --refiner_message_prior_strength 0.5 \
  --output_dir "$IN2R_OUTPUT_DIR"
```

Do not add `--refiner_end_to_end_utility` for the first reproduction.

## Strict resume

Every co-training epoch writes:

- numbered evaluation checkpoints;
- `model_best.pth.tar` for the normal test path;
- one atomic `resume_latest.pth.tar` containing encoders, both GraphRefiners,
  both Memory banks, optimizer moments, loss history, and RNG state.

Resume only from the last file:

```bash
python run.py <the same arguments> \
  --resume "$IN2R_OUTPUT_DIR/resume_latest.pth.tar"
```

The loader rejects changes to critical Memory/GraphRefiner settings. The
Memory checkpoint stores only its valid prefix to avoid serializing unused
queue capacity, while restoring the exact configured capacity.

## Diagnostics

At the end of each model phase, logs contain separate i2t/t2i unlabeled losses
and GraphRefiner statistics:

- utility mean and standard deviation;
- pooling top-1 weight and entropy;
- soft effective K;
- maximum pooling normalization error;
- message entropy;
- RESB quality, freshness, overlap, and value-diversity diagnostics.

## Validation

CPU-compatible deterministic tests:

```bash
python -m unittest -v tests/test_memory_bank.py tests/test_tugr.py
```

The tests cover the original mean-path numerical equivalence, the closed-form
DUA value in a two-query example, all four factorial modes, finite backward
gradients, query propagation through Memory, compact state round-trips, and
strict runtime rectification restoration.

The local Windows machine has a PyTorch/torchtext binary mismatch, so full VSE
GPU training must be run in the established AutoDL environment. See
`docs/VALIDATION_REPORT.md` for the exact local checks and remaining server
smoke test. The ready-to-run fixed-20 launcher is
`scripts/autodl/run_tugr_seed42_fixed20.sh`.

## Repository layout

- `run.py`: co-training entry point and experiment configuration;
- `co_train.py`: A5-Delete co-training plus TUGR GraphRefiner;
- `memory_bank.py`: RESB Memory implementation;
- `data.py`: dataset loading and stable record identifiers;
- `model.py`, `vse.py`, `vse_.py`: retrieval models and losses;
- `evaluation.py`: validation and test metrics;
- `tests/`: CPU regression tests;
- `docs/`: RESB/TUGR change and validation records;
- `scripts/autodl/`: controlled AutoDL launchers.

Datasets, GloVe/NLTK assets, noise-map binaries, checkpoints, logs, output
directories, credentials, archives, and historical duplicate source trees are
deliberately excluded.

## Verified environment

The controlled AutoDL environment uses Python 3.10, PyTorch 2.1.0 + CUDA 12.1,
torchvision 0.16.0, torchtext 0.16.0, torchdata 0.7.0, NumPy 1.26.4,
SciPy 1.11.4, scikit-learn 1.3.2, NLTK 3.8.1, and matplotlib 3.7.5.

```bash
python -m pip install -r requirements.txt
```

## AutoDL Git workflow

```bash
cd /root/autodl-fs
git clone https://github.com/xiangzifan/xzf.git IN2R_repo
cd IN2R_repo
git fetch origin --tags
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

Every experiment output should record the Git SHA and dirty/clean status, the
complete launch command and parsed configuration, the dataset/noise-map
checksum, the package/GPU snapshot, validation curve, selected checkpoint, and
final test metrics.

## Current validation boundary

The Memory/TUGR unit tests and source compilation pass locally. Full VSE CUDA
training has not been executed on the local Windows host because its installed
torchtext binary is incompatible with PyTorch. No rSum improvement is claimed
until the fixed AutoDL reproduction completes.

## Attribution

This work builds on the official IN2R implementation and components credited
upstream to NCR, GPO, and ConVSE. Preserve upstream attribution. No license is
granted by this repository.
