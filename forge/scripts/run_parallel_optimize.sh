#!/bin/bash
# Run N parallel optimization instances, one per GPU.
# Each instance gets its own output directory and GPU via HIP_VISIBLE_DEVICES.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
ROUNDS=${OPT_ROUNDS:-3}
TARGET_DIR=${TARGET_DIR:-targets/aiter_mha}
OUTPUT_BASE=${OUTPUT_BASE:-optimization_logs}

echo "=== 8-GPU Parallel Optimization ==="
echo "GPUs: ${NUM_GPUS}, Rounds: ${ROUNDS}, Target: ${TARGET_DIR}"

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  output_dir="${OUTPUT_BASE}/gpu_${gpu_id}"
  echo "[GPU ${gpu_id}] Starting optimization → ${output_dir}"
  HIP_VISIBLE_DEVICES="$gpu_id" python -m aiter_forge.mini_loop \
    --target "$TARGET_DIR" \
    --output "$output_dir" \
    --rounds "$ROUNDS" \
    --auto \
    --gpu-id "$gpu_id" \
    --seed "$gpu_id" &
  pids+=($!)
done

echo "Waiting for ${#pids[@]} processes..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Process $pid failed"
    failed=$((failed + 1))
  fi
done

if [ "$failed" -gt 0 ]; then
  echo "WARNING: $failed / $NUM_GPUS runs failed"
fi

# Merge results from all successful runs
echo ""
echo "=== Merging Results ==="
python -m aiter_forge.merge_results \
  --dirs "${OUTPUT_BASE}"/gpu_* \
  --metric tflops \
  --output "${OUTPUT_BASE}/merged_report.json"

echo ""
echo "=== Done ==="
cat "${OUTPUT_BASE}/merged_report.json"

# Exit non-zero if any GPU optimization run failed
if [ "$failed" -gt 0 ]; then
  exit 1
fi
