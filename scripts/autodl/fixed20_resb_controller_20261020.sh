#!/usr/bin/env bash
set -Eeuo pipefail

CODE=/root/autodl-fs/IN2R_work/IN2R_A5Delete_RESB_v1_20261020
BASE=/root/autodl-fs/IN2R_work/results/RESB_v1_seed42
PY=/root/miniconda3/envs/in2r/bin/python
ENV_ROOT=/root/miniconda3/envs/in2r
DATA=/root/autodl-tmp/Flickr_xzf
NOISE=/root/autodl-fs/f30k_precomp_0.6_seed42.npy
OLD_CONTROLLER_PID="${1:?old controller pid required}"

unset PYTHONPATH OMP_NUM_THREADS
export PYTHONNOUSERSITE=1
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib/python3.10/site-packages/torch/lib:$ENV_ROOT/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONHASHSEED=42
export NLTK_DATA=/root/autodl-tmp/nltk_data
export IN2R_GLOVE_CACHE=/root/autodl-tmp/IN2R_assets/vector_cache
export CUDA_VISIBLE_DEVICES=0

exec > >(tee -a "$BASE/fixed20_controller.log") 2>&1

latest_completed_epoch() {
  local log="$1"
  awk '
    /\* Co-training/ { co=1 }
    co && /Epoch \[[0-9]+\/40\]/ {
      s=$0; sub(/^.*Epoch \[/,"",s); sub(/\/40\].*$/,"",s); current=s+0
    }
    co && /R_sum:/ { latest=current }
    END { if (latest != "") print latest }
  ' "$log"
}

train_pid_for() {
  local out="$1"
  pgrep -f "^/root/miniconda3/envs/in2r/bin/python -u run.py .*--output_dir=$out$" | head -1 || true
}

record_fixed_policy() {
  local out="$1"
  {
    echo "policy_override=fixed_20_completed_epochs"
    echo "stop_after_completed_epoch_zero_based=19"
    echo "trained_epochs=20"
    echo "scheduler_horizon=40"
    echo "test_checkpoint=model_best.pth.tar"
  } >> "$out/early_stop_policy.txt"
}

wait_and_stop_at_20() {
  local label="$1"
  local out="$BASE/$label"
  local train_pid latest metric_log
  while true; do
    train_pid=$(train_pid_for "$out")
    if [ -z "$train_pid" ]; then
      echo "ABORT: no active run.py for $label before fixed-20 stop"
      exit 41
    fi
    metric_log="$out/log.txt"
    [ ! -f "$out/train_fixed20.log" ] || metric_log="$out/train_fixed20.log"
    latest=""
    [ ! -f "$metric_log" ] || latest=$(latest_completed_epoch "$metric_log")
    if [ -n "$latest" ]; then
      printf '%s label=%s latest_completed_epoch=%s target=19\n' "$(date --iso-8601=seconds)" "$label" "$latest" > "$out/fixed20_status.txt"
      if [ "$latest" -ge 19 ]; then
        echo "[$(date --iso-8601=seconds)] FIXED20 STOP $label after completed epoch $latest (20 epochs total)"
        record_fixed_policy "$out"
        kill -TERM "$train_pid"
        for _ in $(seq 1 60); do
          kill -0 "$train_pid" 2>/dev/null || break
          sleep 1
        done
        kill -0 "$train_pid" 2>/dev/null && { echo "ABORT: $label did not stop after SIGTERM"; exit 42; }
        if [ -f "$out/train_fixed20.log" ] && [ ! -e "$out/log.txt" ]; then
          mv "$out/train_fixed20.log" "$out/log.txt"
        fi
        return
      fi
    fi
    sleep 15
  done
}

scan_and_test() {
  local label="$1"
  local out="$BASE/$label"
  test -s "$out/model_best.pth.tar" || { echo "ABORT: missing model_best for $label"; exit 43; }
  if grep -Ein 'Traceback|RuntimeError|CUDA out of memory|non-finite|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' "$out/launcher.log" > "$out/error_scan.txt"; then
    echo "ABORT: fatal pattern found for $label"
    cat "$out/error_scan.txt"
    exit 44
  fi
  echo "No fatal patterns found." > "$out/error_scan.txt"
  echo "[$(date --iso-8601=seconds)] TEST $label using best checkpoint from epochs 0-19"
  cd "$CODE"
  "$PY" -c "import logging; logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s'); from evaluation import evalrank; evalrank('$out/model_best.pth.tar', split='test')" \
    2>&1 | tee "$out/test_fixed20.log"
  test "${PIPESTATUS[0]}" -eq 0
  date --iso-8601=seconds > "$out/fixed20_complete_time.txt"
  echo "[$(date --iso-8601=seconds)] COMPLETE $label"
}

launch_m2() {
  local out="$BASE/M2_quality_diverse"
  test ! -e "$out" || { echo "ABORT: M2 output already exists: $out"; exit 45; }
  mkdir -p "$out"
  local -a cmd=(
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
    --seed=42 --log_step=100 --memory_mode=quality_diverse --output_dir="$out"
  )
  printf '%q ' "${cmd[@]}" > "$out/launch_command.txt"
  printf '\n' >> "$out/launch_command.txt"
  printf '%s\n' "max_epochs=40" "fixed_completed_epochs=20" "stop_after_zero_based_epoch=19" "metric=Validation Fusion rSum" "scheduler_horizon=40" > "$out/early_stop_policy.txt"
  date --iso-8601=seconds > "$out/start_time.txt"
  echo "[$(date --iso-8601=seconds)] START M2_quality_diverse mode=quality_diverse fixed20, scheduler_horizon=40"
  cd "$CODE"
  set +e
  "${cmd[@]}" 2>&1 | tee "$out/launcher.log" &
  local pipeline_pid=$!
  set -e
  wait_and_stop_at_20 M2_quality_diverse
  set +e
  wait "$pipeline_pid"
  set -e
  date --iso-8601=seconds > "$out/end_time.txt"
}

echo "[$(date --iso-8601=seconds)] TAKEOVER: fixed 20 completed epochs; cosine horizon remains 40"
kill -0 "$OLD_CONTROLLER_PID"
wait_and_stop_at_20 M1_id_refresh
scan_and_test M1_id_refresh
launch_m2
scan_and_test M2_quality_diverse
kill -KILL "$OLD_CONTROLLER_PID" 2>/dev/null || true
echo "[$(date --iso-8601=seconds)] M1 and M2 completed under fixed-20 policy"
