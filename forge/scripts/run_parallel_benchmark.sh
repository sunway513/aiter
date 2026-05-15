#!/bin/bash
# Run MHA benchmark across N GPUs in parallel.
# Each GPU runs the same benchmark shapes independently for consistency.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
AITER_ROOT=${AITER_ROOT:?AITER_ROOT must be set}
OUTPUT_BASE=${OUTPUT_BASE:-benchmark_logs}

echo "=== Multi-GPU Benchmark ==="
echo "GPUs: ${NUM_GPUS}, AITER_ROOT: ${AITER_ROOT}"

mkdir -p "${OUTPUT_BASE}"

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  log_file="${OUTPUT_BASE}/gpu_${gpu_id}.log"
  echo "[GPU ${gpu_id}] Starting benchmark → ${log_file}"
  (
    HIP_VISIBLE_DEVICES="$gpu_id" python3 -c "
import torch
dev = torch.device('cuda:0')
print(f'GPU {$gpu_id}: {torch.cuda.get_device_name(dev)}')
print(f'  Memory: {torch.cuda.get_device_properties(dev).total_memory / 1e9:.1f} GB')
"
    # Run benchmark shapes from target config
    cd "${AITER_ROOT}"
    HIP_VISIBLE_DEVICES="$gpu_id" python op_tests/op_benchmarks/triton/bench_mha.py \
      -b 4 -hq 16 -hk 16 -d 128 -sq 4096 -sk 4096 -metric throughput 2>&1
  ) > "${log_file}" 2>&1 &
  pids+=($!)
done

echo "Waiting for ${#pids[@]} benchmark processes..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Process $pid failed"
    failed=$((failed + 1))
  fi
done

# Print results summary
echo ""
echo "=== Benchmark Results ==="
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  log_file="${OUTPUT_BASE}/gpu_${gpu_id}.log"
  echo "--- GPU ${gpu_id} ---"
  cat "${log_file}" 2>/dev/null || echo "(no output)"
  echo ""
done

if [ "$failed" -gt 0 ]; then
  echo "WARNING: $failed / $NUM_GPUS benchmarks failed"
  exit 1
fi
