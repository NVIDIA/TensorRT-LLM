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

# Attention DP or not, and the batch and token limits, from the config.
read -r ADP MAX_BS MAX_TOKENS < <(python3 -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); \
print(int(bool(c.get('enable_attention_dp'))), c['max_batch_size'], c['max_num_tokens'])" "$CONFIG")

# Requests and warmup requests per batch size, as used for the published results.
if [ "$ADP" = 0 ]; then
  # Tensor-parallel attention (section 9): 20x and 2x the batch size, at least 320 and 64.
  N=$((BS * 20 > 320 ? BS * 20 : 320))
  WARMUP=$((BS * 2 > 64 ? BS * 2 : 64))
else
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
fi

# The published results were measured with this allocator setting.
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99999

# The output directory keeps the config it ran; a rerun first drops the previous report.
mkdir -p "$OUT"
rm -f "$OUT/report.json" "$OUT/requests.json"
cp "$CONFIG" "$OUT/config.yaml"
printf '{"dataset": "%s", "batch_size": %d}\n' "$(basename "$DATASET" .jsonl)" "$BS" > "$OUT/run.json"
trtllm-bench --model "$MODEL" --model_path "$MODEL" --workspace "$OUT/workspace" \
  throughput --backend pytorch --config "$OUT/config.yaml" \
  --sampler_options "$(dirname "$(realpath "$0")")/sampler.yaml" \
  --dataset "$DATASET" --num_requests "$N" --concurrency "$BS" --warmup "$WARMUP" \
  --tp 8 --pp 1 --ep 8 --max_batch_size "$MAX_BS" --max_num_tokens "$MAX_TOKENS" --max_seq_len 2560 \
  --kv_cache_free_gpu_mem_fraction 0.5 --disable_chunked_context --custom_tokenizer deepseek_v4 \
  --eos_id 1 --beam_width 1 \
  --report_json "$OUT/report.json" --request_json "$OUT/requests.json" > "$OUT/log.txt" 2>&1
echo "$OUT: $(python3 -c "import json, sys; p = json.load(open(sys.argv[1]))['performance']; \
print(f\"{p['system_output_throughput_tok_s'] / 8:.0f} tok/s/GPU, \
{p['output_throughput_per_user_tok_s']:.1f} tok/s/user\")" "$OUT/report.json")"
