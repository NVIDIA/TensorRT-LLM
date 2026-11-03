#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run one benchmark arm of CONFIDENCE_REPRODUCTION.md on 8 GPUs.
# Usage: run_benchmark.sh <checkpoint> <config.yaml> <dataset.jsonl> <batch_size> <output_dir>
#
# Constant concurrency equal to the batch size: a new request starts as soon as one finishes.
# Greedy decoding; each request stops at EOS or after 1,024 output tokens.
set -euo pipefail
if [ $# -ne 5 ]; then
  sed -n 4,8p "$0"
  exit 2
fi
MODEL=$1
CONFIG=$2
DATASET=$3
BS=$4
OUT=$5

# Requests and warmup requests per batch size, as used for the published results.
case $BS in
  64) N=1280; WARMUP=256 ;;
  128) N=2560; WARMUP=512 ;;
  256) N=5120; WARMUP=512 ;;
  384) N=7680; WARMUP=768 ;;
  512) N=10240; WARMUP=1024 ;;
  768) N=10240; WARMUP=768 ;;
  1024) N=10240; WARMUP=1024 ;;
  2048) N=20480; WARMUP=2048 ;;
  *) echo "Choose request and warmup counts for batch size $BS (about 10-20x and 1-4x BS)"; exit 2 ;;
esac

# The published results were measured with this allocator setting.
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99999

mkdir -p "$OUT"
printf '{"dataset": "%s", "batch_size": %d, "config": "%s"}\n' \
  "$(basename "$DATASET" .jsonl)" "$BS" "$(realpath "$CONFIG")" > "$OUT/run.json"
trtllm-bench --model "$MODEL" --model_path "$MODEL" --workspace "$OUT/workspace" \
  throughput --backend pytorch --config "$CONFIG" \
  --sampler_options "$(dirname "$(realpath "$0")")/sampler.yaml" \
  --dataset "$DATASET" --num_requests "$N" --concurrency "$BS" --warmup "$WARMUP" \
  --tp 8 --pp 1 --ep 8 --max_batch_size $((BS / 8)) --max_num_tokens 4096 --max_seq_len 2560 \
  --kv_cache_free_gpu_mem_fraction 0.5 --disable_chunked_context --custom_tokenizer deepseek_v4 \
  --eos_id 1 --beam_width 1 \
  --report_json "$OUT/report.json" --request_json "$OUT/requests.json" > "$OUT/log.txt" 2>&1
echo "$OUT: $(python3 -c "import json, sys; p = json.load(open(sys.argv[1]))['performance']; \
print(f\"{p['system_output_throughput_tok_s'] / 8:.0f} tok/s/GPU, \
{p['output_throughput_per_user_tok_s']:.1f} tok/s/user\")" "$OUT/report.json")"
