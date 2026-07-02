#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch bench_moe under Nsight Systems, capturing ONLY the measured MoE forward.
#
# bench_moe's `--nsys` flag brackets the measured region with cudaProfilerStart/Stop
# (after warmup) and wraps the pure forward with the `bench_moe.measured` NVTX range.
# Combined with `nsys -c cudaProfilerApi --capture-range-end stop`, the resulting
# .nsys-rep contains only the measured forward -- no autotune, no warmup, no graph
# capture. Analyze the `bench_moe.measured` NVTX range (nsys-ui or `nsys stats`) for
# the pure-forward view (the L2-flush memsets fall outside that NVTX range).
#
# NOTES:
#   * `--nsys` disables the CUPTI kernel breakdown (nsys and CUPTI cannot coexist),
#     so pass `--analysis none`.
#   * Prefer a SINGLE token / single candidate per capture for a clean timeline.
#   * For multi-GPU, place `nsys profile` AFTER `mpirun` so each rank writes its own
#     report; keep NCCL_NVLS_ENABLE=0 to avoid the B300 NVLS hang.
#
# Usage:
#   bash run_nsys.sh                 # single-GPU example (DeepSeek-V3 256e, 4096 tokens)
#   WORLD_SIZE=4 bash run_nsys.sh    # 4-GPU TTP example
set -eu

OUT_DIR=${OUT_DIR:-/tmp/bench_moe_nsys}
WORLD_SIZE=${WORLD_SIZE:-1}
TOKENS=${TOKENS:-4096}
mkdir -p "$OUT_DIR"

NSYS_ARGS=(
  -t cuda,nvtx
  -c cudaProfilerApi --capture-range-end stop
  --force-overwrite true
)

# DeepSeek-V3 (256 experts) shape; edit for other shapes.
SHAPE=(
  --model deepseek_v3 --num_experts 256 --top_k 8 --hidden_size 7168 --intermediate_size 2048
  --n_group 8 --topk_group 4 --routing_method DEEPSEEK_V3
  --quant FP8_BLOCK_SCALES --n_shared_experts 1 --shared_expert_mode fused
  --backend TRTLLM
)

if [ "$WORLD_SIZE" = "1" ]; then
  nsys profile "${NSYS_ARGS[@]}" -o "$OUT_DIR/bench_moe.nsys-rep" \
    python -m tests.microbenchmarks.bench_moe --world_size 1 \
      "${SHAPE[@]}" --balanced_total_num_tokens "$TOKENS" --nsys --analysis none
else
  NCCL_NVLS_ENABLE=0 mpirun -np "$WORLD_SIZE" \
    nsys profile "${NSYS_ARGS[@]}" \
      -o "$OUT_DIR/bench_moe_rank%q{OMPI_COMM_WORLD_RANK}.nsys-rep" \
    python -m tests.microbenchmarks.bench_moe --world_size "$WORLD_SIZE" \
      "${SHAPE[@]}" --parallel_mode TTP --comm_pattern random --expert_pattern random \
      --balanced_total_num_tokens "$TOKENS" --nsys --analysis none
fi

echo "nsys report(s) written under: $OUT_DIR"
echo "Inspect the measured forward with:"
echo "  nsys stats --report nvtx_pushpop_trace $OUT_DIR/bench_moe*.nsys-rep"
