#!/bin/bash

set -euo pipefail

# Common settings and preparation

model_path="$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2/"
NP=4
BATCH_SIZE=32

export TLLM_AUTOTUNER_CACHE_PATH=autotuner_cache/sample_performance_calibrating_cache.json

mkdir -p profiles
mkdir -p autotuner_cache

python3 ../../benchmarks/cpp/prepare_dataset.py \
    --tokenizer "$model_path" \
    --stdout \
    --random-seed 42 \
    token-norm-dist \
    --num-requests $((BATCH_SIZE*NP)) \
    --input-mean 8192 \
    --input-stdev 0 \
    --output-mean 1024 \
    --output-stdev 0 \
    >/tmp/dataset.jsonl

# Step 1

rm -f -- "$TLLM_AUTOTUNER_CACHE_PATH"

cat <<EOF >/tmp/config_collect.yaml
enable_attention_dp: true
layer_wise_benchmarks_config:
    calibration_mode: COLLECT
    calibration_file_path: profiles/calibration_data.json
moe_config:
    backend: CUTLASS
print_iter_log: true
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

TLLM_PROFILE_START_STOP=$((BATCH_SIZE + 10))-$((BATCH_SIZE + 35)) \
NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
nsys profile \
    -t cuda,nvtx \
    --cpuctxsw none --cuda-event-trace false \
    --cuda-graph-trace node \
    -c cudaProfilerApi --capture-range-end stop \
    -o profiles/report_e2e_collect_rank%q{RANK}.nsys-rep \
    --force-overwrite true \
trtllm-llmapi-launch \
trtllm-bench \
    --model deepseek-ai/DeepSeek-V3 \
    --model_path "${model_path}" \
    throughput \
    --tp $NP \
    --ep $NP \
    --warmup 0 \
    --dataset /tmp/dataset.jsonl \
    --max_batch_size $BATCH_SIZE \
    --max_num_tokens 10240 \
    --disable_chunked_context \
    --num_requests $((BATCH_SIZE*NP)) \
    --concurrency $((BATCH_SIZE*NP)) \
    --config /tmp/config_collect.yaml

# Step 2

cat <<EOF >/tmp/config_mark.yaml
cuda_graph_config: null
enable_attention_dp: true
layer_wise_benchmarks_config:
    calibration_mode: MARK
moe_config:
    backend: CUTLASS
print_iter_log: true
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

TLLM_PROFILE_START_STOP=$((BATCH_SIZE + 10))-$((BATCH_SIZE + 35)) \
NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
nsys profile \
    -t cuda,nvtx \
    --cpuctxsw none --cuda-event-trace false \
    --cuda-graph-trace node \
    -c cudaProfilerApi --capture-range-end stop \
    -o profiles/report_e2e_mark_rank%q{RANK}.nsys-rep \
    --force-overwrite true \
trtllm-llmapi-launch \
trtllm-bench \
    --model deepseek-ai/DeepSeek-V3 \
    --model_path "${model_path}" \
    throughput \
    --tp $NP \
    --ep $NP \
    --warmup 0 \
    --dataset /tmp/dataset.jsonl \
    --max_batch_size $BATCH_SIZE \
    --max_num_tokens 10240 \
    --disable_chunked_context \
    --num_requests $((BATCH_SIZE*NP)) \
    --concurrency $((BATCH_SIZE*NP)) \
    --config /tmp/config_mark.yaml

# Step 3

NP=$NP ./mpi_launch.sh ./run.sh config_gen.yaml \
    --model "$model_path" \
    --load-format AUTO \
    --layer-indices 5,6,7 \
    --batch-size 32 \
    --seq-len-q 4 \
    --seq-len-kv-cache $((8193 + (BATCH_SIZE / 2 + 25) * 3)) \
    --balance-method NotModified \
    --replay-file-path profiles/calibration_data.json \
    --replay-start $((BATCH_SIZE + 10 + 5)) \
    --replay-stop $((BATCH_SIZE + 35))

# Step 4

seq 0 $((NP - 1)) | xargs -I% python3 parse_e2e.py \
    --eager-trace profiles/report_e2e_mark_rank%.nsys-rep \
    --graph-trace profiles/report_e2e_collect_rank%.nsys-rep \
    --layer-indices 5,6,7 \
    --warmup-times 5 \
    -o profiles/report_e2e_mark_rank%.json
seq 0 $((NP - 1)) | xargs -I% python3 parse.py \
    --world-size $NP \
    --rank %

# Step 5

python3 correlation.py \
    --reference profiles/report_e2e_mark_rank0.json \
    $(seq 1 $((NP - 1)) | xargs -I% echo "--target profiles/report_e2e_mark_rank%.json") \
    $(seq 0 $((NP - 1)) | xargs -I% echo "--target profiles/report_np4_rank%.json") \
    -o profiles/correlation.html
