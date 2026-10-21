#!/usr/bin/env bash
set -Eeuo pipefail

CODE=/root/autodl-fs/IN2R_work/IN2R_A5Delete_RESB_v1_20261020
BASE=/root/autodl-fs/IN2R_work/results/RESB_v1_seed42
PY=/root/miniconda3/envs/in2r/bin/python
ENV_ROOT=/root/miniconda3/envs/in2r
DATA=/root/autodl-tmp/Flickr_xzf
NOISE=/root/autodl-fs/f30k_precomp_0.6_seed42.npy
EXPECTED_NOISE_SHA=2cbb83d4f7b88c3877b028730876bb200f89a2b72c89e93d3bc68c62f525aeb7

unset PYTHONPATH OMP_NUM_THREADS
export PYTHONNOUSERSITE=1
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib/python3.10/site-packages/torch/lib:$ENV_ROOT/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONHASHSEED=42
export NLTK_DATA=/root/autodl-tmp/nltk_data
export IN2R_GLOVE_CACHE=/root/autodl-tmp/IN2R_assets/vector_cache
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$BASE/preflight"
exec > >(tee -a "$BASE/controller.log") 2>&1

echo "[$(date --iso-8601=seconds)] Stage-1 controller started"
echo "Environment adjustment: unset invalid inherited OMP_NUM_THREADS=0"

if pgrep -af '[p]ython.*run.py' >/dev/null; then
  echo "ABORT: another run.py process is active"
  pgrep -af '[p]ython.*run.py'
  exit 20
fi

cd "$CODE"
sha256sum -c code_checksums.sha256 | tee "$BASE/preflight/code_checksums_before_training.txt"
printf '%s  %s\n' "$EXPECTED_NOISE_SHA" "$NOISE" | sha256sum -c - | tee "$BASE/preflight/noise_checksum_verification.txt"

for required in \
  "$DATA/f30k_precomp/train_ims.npy" \
  "$DATA/f30k_precomp/dev_ims.npy" \
  "$DATA/f30k_precomp/test_ims.npy" \
  "$DATA/f30k_precomp_vocab.json" \
  "$IN2R_GLOVE_CACHE/glove.840B.300d.txt.pt" \
  "$NLTK_DATA/tokenizers/punkt/PY3/english.pickle"; do
  test -f "$required" || { echo "ABORT: missing $required"; exit 21; }
done

run_one() {
  local mode="$1"
  local label="$2"
  local out="$BASE/$label"

  if [ -e "$out" ]; then
    echo "ABORT: refusing to reuse output directory $out"
    exit 22
  fi
  mkdir -p "$out"

  local -a cmd=(
    "$PY" -u run.py
    --data_path="$DATA"
    --data_name=f30k_precomp
    --vocab_path="$DATA"
    --noise_ratio=0.6
    --noise_file="$NOISE"
    --batch_size=128
    --num_epochs=40
    --warmup_epoch=5
    --warmup_model_path=
    --learning_rate=0.0005
    --lr_update=15
    --lr_scheduler=cosine
    --lambda_intra=0.5
    --margin=0.2
    --dropout_rate=0.3
    --grad_clip=2.0
    --p_threshold=0.5
    --soft_margin=exponential
    --queue_size=65536
    --k=5
    --memory_candidate_pool_size=20
    --memory_quality_weight=0.25
    --memory_freshness_weight=0.10
    --memory_freshness_tau=2.0
    --memory_mmr_balance=0.80
    --memory_alpha_min=0.20
    --memory_alpha_max=0.80
    --memory_quality_momentum=0.50
    --i2t
    --t2i
    --workers=0
    --gpu=0
    --seed=42
    --log_step=100
    --memory_mode="$mode"
    --output_dir="$out"
  )

  printf '%q ' "${cmd[@]}" > "$out/launch_command.txt"
  printf '\n' >> "$out/launch_command.txt"
  date --iso-8601=seconds > "$out/start_time.txt"
  echo "[$(date --iso-8601=seconds)] START $label ($mode)"

  set +e
  "${cmd[@]}" 2>&1 | tee "$out/launcher.log"
  local rc=${PIPESTATUS[0]}
  set -e
  date --iso-8601=seconds > "$out/end_time.txt"

  if [ "$rc" -ne 0 ]; then
    echo "ABORT: $label exited with code $rc"
    exit "$rc"
  fi
  if grep -Ein 'Traceback|RuntimeError|CUDA out of memory|non-finite|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' "$out/launcher.log" > "$out/error_scan.txt"; then
    echo "ABORT: fatal-pattern scan matched for $label"
    cat "$out/error_scan.txt"
    exit 23
  fi
  echo "No fatal patterns found." > "$out/error_scan.txt"
  test -s "$out/model_best.pth.tar" || { echo "ABORT: model_best missing for $label"; exit 24; }
  grep -q 'final model' "$out/log.txt" || { echo "ABORT: final test missing for $label"; exit 25; }
  echo "[$(date --iso-8601=seconds)] COMPLETE $label"
}

run_one fifo M0_fifo

M0_LOG="$BASE/M0_fifo/log.txt"
M0_TEST_RSUM=$(awk '/final model/{seen=1; next} seen && /rsum:/{v=$NF} END{print v}' "$M0_LOG")
echo "$M0_TEST_RSUM" > "$BASE/M0_fifo/test_fusion_rsum.txt"
if [ -z "$M0_TEST_RSUM" ]; then
  echo "ABORT: unable to extract M0 final Test Fusion rSum"
  exit 26
fi
if ! awk -v x="$M0_TEST_RSUM" 'BEGIN { d=x-501.3; if (d<0) d=-d; exit(d<=1.5 ? 0 : 1) }'; then
  echo "PAUSE: M0 Test Fusion rSum=$M0_TEST_RSUM differs from 501.3 by more than 1.5"
  exit 27
fi
echo "M0 gate passed: Test Fusion rSum=$M0_TEST_RSUM"

run_one id_refresh M1_id_refresh
run_one quality_diverse M2_quality_diverse

echo "[$(date --iso-8601=seconds)] All three controlled runs completed"
