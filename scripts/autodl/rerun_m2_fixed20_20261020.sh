#!/usr/bin/env bash
set -Eeuo pipefail

CODE=/root/autodl-fs/IN2R_work/IN2R_A5Delete_RESB_v1_20261020
BASE=/root/autodl-fs/IN2R_work/results/RESB_v1_seed42
OUT="$BASE/M2_quality_diverse"
PY=/root/miniconda3/envs/in2r/bin/python
ENV_ROOT=/root/miniconda3/envs/in2r
DATA=/root/autodl-tmp/Flickr_xzf
NOISE=/root/autodl-fs/f30k_precomp_0.6_seed42.npy

unset PYTHONPATH OMP_NUM_THREADS
export PYTHONNOUSERSITE=1
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib/python3.10/site-packages/torch/lib:$ENV_ROOT/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONHASHSEED=42
export NLTK_DATA=/root/autodl-tmp/nltk_data
export IN2R_GLOVE_CACHE=/root/autodl-tmp/IN2R_assets/vector_cache
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$BASE"
exec > >(tee -a "$BASE/m2_rerun_fixed20_controller.log") 2>&1

latest_completed_epoch() {
  awk '
    /\* Co-training/ { co=1 }
    co && /Epoch \[[0-9]+\/40\]/ {
      s=$0; sub(/^.*Epoch \[/,"",s); sub(/\/40\].*$/,"",s); current=s+0
    }
    co && /R_sum:/ { latest=current }
    END { if (latest != "") print latest }
  ' "$1"
}

train_pid_for_out() {
  pgrep -f "^/root/miniconda3/envs/in2r/bin/python -u run.py .*--output_dir=$OUT$" | head -1 || true
}

test ! -e "$OUT" || { echo "ABORT: output already exists: $OUT"; exit 51; }
pgrep -af '[p]ython.*run.py' && { echo "ABORT: another run.py is active"; exit 52; } || true
cd "$CODE"
sha256sum -c code_checksums.sha256
mkdir -p "$OUT"

cmd=(
  "$PY" -u run.py
  --data_path="$DATA" --data_name=f30k_precomp --vocab_path="$DATA"
  --noise_ratio=0.6 --noise_file="$NOISE" --batch_size=128
  --num_epochs=40 --warmup_epoch=5 --warmup_model_path=
  --learning_rate=0.0005 --lr_update=15 --lr_scheduler=cosine
  --lambda_intra=0.5 --margin=0.2 --dropout_rate=0.3 --grad_clip=2.0
  --p_threshold=0.5 --soft_margin=exponential --queue_size=65536 --k=5
  --memory_candidate_pool_size=20 --memory_quality_weight=0.25
  --memory_freshness_weight=0.10 --memory_freshness_tau=2.0
  --memory_mmr_balance=0.80 --memory_alpha_min=0.20 --memory_alpha_max=0.80
  --memory_quality_momentum=0.50 --i2t --t2i --workers=0 --gpu=0
  --seed=42 --log_step=100 --memory_mode=quality_diverse --output_dir="$OUT"
)

printf '%q ' "${cmd[@]}" > "$OUT/launch_command.txt"
printf '\n' >> "$OUT/launch_command.txt"
printf '%s\n' \
  "max_epochs=40" \
  "fixed_completed_epochs=20" \
  "stop_after_zero_based_epoch=19" \
  "metric=Validation Fusion rSum" \
  "scheduler_horizon=40" \
  "test_checkpoint=model_best.pth.tar" > "$OUT/fixed20_policy.txt"
date --iso-8601=seconds > "$OUT/start_time.txt"
echo "[$(date --iso-8601=seconds)] START M2 clean rerun; fixed20; cosine horizon 40"

set +e
"${cmd[@]}" 2>&1 | tee "$OUT/launcher.log" &
pipeline_pid=$!
set -e
stopped=0

while kill -0 "$pipeline_pid" 2>/dev/null; do
  sleep 15
  latest=""
  [ ! -f "$OUT/log.txt" ] || latest=$(latest_completed_epoch "$OUT/log.txt")
  if [ -n "$latest" ]; then
    printf '%s latest_completed_epoch=%s target=19\n' "$(date --iso-8601=seconds)" "$latest" > "$OUT/fixed20_status.txt"
    if [ "$latest" -ge 19 ]; then
      train_pid=$(train_pid_for_out)
      test -n "$train_pid" || { echo "ABORT: training process missing at stop point"; exit 53; }
      echo "[$(date --iso-8601=seconds)] FIXED20 STOP M2 after completed epoch $latest"
      kill -TERM "$train_pid"
      stopped=1
      break
    fi
  fi
done

set +e
wait "$pipeline_pid"
rc=$?
set -e
date --iso-8601=seconds > "$OUT/end_time.txt"

if [ "$stopped" -eq 0 ]; then
  echo "ABORT: M2 exited before fixed20 target, rc=$rc"
  exit 54
fi
test -s "$OUT/model_best.pth.tar" || { echo "ABORT: missing model_best"; exit 55; }
if grep -Ein 'Traceback|RuntimeError|CUDA out of memory|non-finite|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' "$OUT/launcher.log" > "$OUT/error_scan.txt"; then
  echo "ABORT: fatal pattern found"
  cat "$OUT/error_scan.txt"
  exit 56
fi
echo "No fatal patterns found." > "$OUT/error_scan.txt"

echo "[$(date --iso-8601=seconds)] TEST M2 best checkpoint from epochs 0-19"
"$PY" -c "import logging; logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s'); from evaluation import evalrank; evalrank('$OUT/model_best.pth.tar', split='test')" \
  2>&1 | tee "$OUT/test_fixed20.log"
test "${PIPESTATUS[0]}" -eq 0
date --iso-8601=seconds > "$OUT/complete_time.txt"
echo "[$(date --iso-8601=seconds)] COMPLETE M2 clean rerun"
