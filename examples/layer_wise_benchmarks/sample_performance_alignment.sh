#!/bin/bash

set -euo pipefail

# Common settings and preparation

MODEL="${MODEL:-$LLM_MODELS_ROOT/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2}"
NP=${NP:-4}
BATCH_SIZE=32

export PROFILE_DIR="${PROFILE_DIR:-profiles}"
export TLLM_AUTOTUNER_CACHE_PATH="$PROFILE_DIR/sample_performance_alignment_cache.json"

mkdir -p -- "$PROFILE_DIR"
mkdir -p -- "$(dirname -- "$TLLM_AUTOTUNER_CACHE_PATH")"

python3 ../../benchmarks/cpp/prepare_dataset.py \
    --tokenizer "$MODEL" \
    --stdout \
    --random-seed 42 \
    token-norm-dist \
    --num-requests $((BATCH_SIZE * NP)) \
    --input-mean 2048 \
    --input-stdev 0 \
    --output-mean 256 \
    --output-stdev 0 \
    >/tmp/dataset.jsonl

# Step 1

rm -f -- "$TLLM_AUTOTUNER_CACHE_PATH"

cat <<EOF >/tmp/config_collect.yaml
enable_attention_dp: true
layer_wise_benchmarks_config:
    calibration_mode: COLLECT
    calibration_file_path: "$PROFILE_DIR/calibration_data.json"
moe_config:
    backend: CUTLASS
print_iter_log: true
EOF

TLLM_PROFILE_START_STOP=$((BATCH_SIZE + 10))-$((BATCH_SIZE + 35)) \
NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
nsys profile \
    -t cuda,nvtx \
    --cpuctxsw none --cuda-event-trace false \
    --cuda-graph-trace node \
    -c cudaProfilerApi --capture-range-end stop \
    -o "$PROFILE_DIR/report_e2e_collect_rank%q{RANK}.nsys-rep" \
    --force-overwrite true \
trtllm-llmapi-launch \
trtllm-bench \
    --model deepseek-ai/DeepSeek-V3 \
    --model_path "$MODEL" \
    throughput \
    --tp $NP \
    --ep $NP \
    --warmup 0 \
    --dataset /tmp/dataset.jsonl \
    --max_batch_size $BATCH_SIZE \
    --max_num_tokens 3072 \
    --disable_chunked_context \
    --num_requests $((BATCH_SIZE * NP)) \
    --concurrency $((BATCH_SIZE * NP)) \
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
EOF

TLLM_PROFILE_START_STOP=$((BATCH_SIZE + 10))-$((BATCH_SIZE + 35)) \
NP=$NP ./mpi_launch.sh middleware/mpi_env_from_ompi \
nsys profile \
    -t cuda,nvtx \
    --cpuctxsw none --cuda-event-trace false \
    --cuda-graph-trace node \
    -c cudaProfilerApi --capture-range-end stop \
    -o "$PROFILE_DIR/report_e2e_mark_rank%q{RANK}.nsys-rep" \
    --force-overwrite true \
trtllm-llmapi-launch \
trtllm-bench \
    --model deepseek-ai/DeepSeek-V3 \
    --model_path "$MODEL" \
    throughput \
    --tp $NP \
    --ep $NP \
    --warmup 0 \
    --dataset /tmp/dataset.jsonl \
    --max_batch_size $BATCH_SIZE \
    --max_num_tokens 3072 \
    --disable_chunked_context \
    --num_requests $((BATCH_SIZE * NP)) \
    --concurrency $((BATCH_SIZE * NP)) \
    --config /tmp/config_mark.yaml

# Step 3

NP=$NP ./mpi_launch.sh ./run.sh config_gen.yaml \
    --model "$MODEL" \
    --load-format AUTO \
    --layer-indices 5,6,7 \
    --batch-size $BATCH_SIZE \
    --seq-len-q 1 \
    --seq-len-kv-cache $((2049 + (BATCH_SIZE / 2 + 25) * 1)) \
    --balance-method NotModified \
    --replay-file-path "$PROFILE_DIR/calibration_data.json" \
    --replay-start $((BATCH_SIZE + 10 + 5)) \
    --replay-stop $((BATCH_SIZE + 35))

# Step 4

seq 0 $((NP - 1)) | xargs -I% python3 parse_e2e.py \
    --eager-trace "$PROFILE_DIR/report_e2e_mark_rank%.nsys-rep" \
    --graph-trace "$PROFILE_DIR/report_e2e_collect_rank%.nsys-rep" \
    --layer-indices 5,6,7 \
    --warmup-times 5 \
    -o "$PROFILE_DIR/report_e2e_collect_rank%.json"
seq 0 $((NP - 1)) | xargs -I% python3 parse.py \
    --profile-dir "$PROFILE_DIR" \
    --world-size $NP \
    --rank %

# Step 5

targets=()
for i in $(seq 1 $((NP - 1))); do
    targets+=(--target "$PROFILE_DIR/report_e2e_collect_rank$i.json")
done
for i in $(seq 0 $((NP - 1))); do
    targets+=(--target "$PROFILE_DIR/report_np${NP}_rank$i.json")
done
python3 correlation.py \
    --reference "$PROFILE_DIR/report_e2e_collect_rank0.json" \
    "${targets[@]}" \
    -o "$PROFILE_DIR/correlation.html"
