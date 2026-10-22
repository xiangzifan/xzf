#!/usr/bin/env bash
set -Eeuo pipefail

# CUSR v1: RESB quality-diverse memory + legacy mean GraphRefiner +
# counterfactual utility-guided selective rectification.
CODE_DIR="${IN2R_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${IN2R_PYTHON:-/root/miniconda3/envs/in2r/bin/python}"
ENV_ROOT="${IN2R_ENV_ROOT:-/root/miniconda3/envs/in2r}"
DATA_DIR="${IN2R_DATA_PATH:-/root/autodl-tmp/Flickr_xzf}"
NOISE_FILE="${IN2R_NOISE_02:-/root/autodl-fs/f30k_precomp_0.6_seed42.npy}"
OUTPUT_DIR="${IN2R_OUTPUT_DIR:-/root/autodl-fs/IN2R_work/results/CUSR_v1_seed42}"
RESUME_FILE="${IN2R_RESUME:-}"

unset PYTHONPATH OMP_NUM_THREADS
export PYTHONNOUSERSITE=1
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib/python3.10/site-packages/torch/lib:$ENV_ROOT/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NLTK_DATA="${NLTK_DATA:-/root/autodl-tmp/nltk_data}"
export IN2R_GLOVE_CACHE="${IN2R_GLOVE_CACHE:-/root/autodl-tmp/IN2R_assets/vector_cache}"

if [[ -z "$RESUME_FILE" && -e "$OUTPUT_DIR" ]]; then
  echo "ABORT: output already exists: $OUTPUT_DIR" >&2
  exit 51
fi
if [[ -n "$RESUME_FILE" && ! -s "$RESUME_FILE" ]]; then
  echo "ABORT: resume checkpoint not found: $RESUME_FILE" >&2
  exit 52
fi
if [[ ! -s "$NOISE_FILE" ]]; then
  echo "ABORT: fixed noise file not found: $NOISE_FILE" >&2
  exit 53
fi

cd "$CODE_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ABORT: Git worktree is not clean" >&2
  git status --short >&2
  exit 54
fi

mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
git rev-parse HEAD^{tree} > "$OUTPUT_DIR/git_tree.txt"
git status --short --branch > "$OUTPUT_DIR/git_status.txt"
sha256sum "$NOISE_FILE" > "$OUTPUT_DIR/noise_sha256.txt"
{
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c 'import numpy, sklearn, torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("numpy=" + numpy.__version__); print("sklearn=" + sklearn.__version__); print("cuda_available=" + str(torch.cuda.is_available()))'
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  else
    echo "nvidia-smi unavailable"
  fi
} > "$OUTPUT_DIR/environment.txt"
"$PYTHON_BIN" -m py_compile co_train.py memory_bank.py run.py \
  selective_rectification.py tests/test_memory_bank.py tests/test_tugr.py \
  tests/test_cusr.py 2>&1 | tee "$OUTPUT_DIR/compile.log"
"$PYTHON_BIN" -m unittest -v tests/test_memory_bank.py tests/test_tugr.py \
  tests/test_cusr.py 2>&1 | tee "$OUTPUT_DIR/unit_tests.log"

cmd=(
  "$PYTHON_BIN" -u run.py
  --data_path="$DATA_DIR" --data_name=f30k_precomp --vocab_path="$DATA_DIR"
  --noise_ratio=0.6 --noise_file="$NOISE_FILE" --batch_size=128
  --num_epochs=40 --co_train_epochs=20 --warmup_epoch=5
  --learning_rate=0.0005 --lr_update=15 --lr_scheduler=cosine
  --lambda_intra=0.5 --margin=0.2 --dropout_rate=0.3 --grad_clip=2.0
  --p_threshold=0.5 --soft_margin=exponential --queue_size=65536 --k=5
  --memory_candidate_pool_size=20 --memory_quality_weight=0.25
  --memory_freshness_weight=0.10 --memory_freshness_tau=2.0
  --memory_mmr_balance=0.80 --memory_alpha_min=0.20 --memory_alpha_max=0.80
  --memory_quality_momentum=0.50 --memory_mode=quality_diverse
  --refiner_mode=mean --refiner_heads=4 --refiner_dropout=0.1
  --rectification_policy=cusr --correction_weight=0.5
  --cusr_utility_temperature=0.05 --cusr_gate_temperature=0.05
  --cusr_min_gain=0.0 --cusr_min_utility=0.0
  --refiner_qasr_pool_temperature=0.25
  --refiner_qasr_query_weight=1.0 --refiner_qasr_consensus_weight=0.5
  --refiner_qasr_metadata_weight=0.25 --refiner_qasr_graph_strength=0.5
  --refiner_qasr_residual_strength=1.0
  --refiner_utility_temperature=0.1 --refiner_pool_temperature=0.1
  --refiner_message_prior_strength=0.5
  --i2t --t2i --workers=0 --gpu=0 --seed=42 --log_step=100
  --output_dir="$OUTPUT_DIR"
)

if [[ -n "$RESUME_FILE" ]]; then
  cmd+=(--resume="$RESUME_FILE")
fi

printf '%q ' "${cmd[@]}" > "$OUTPUT_DIR/launch_command.txt"
printf '\n' >> "$OUTPUT_DIR/launch_command.txt"
date --iso-8601=seconds > "$OUTPUT_DIR/start_time.txt"
"${cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/launcher.log"
status="${PIPESTATUS[0]}"
date --iso-8601=seconds > "$OUTPUT_DIR/end_time.txt"
exit "$status"
