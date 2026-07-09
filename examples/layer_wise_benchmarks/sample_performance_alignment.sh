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

# Notes on profiling Steps 1 and 2 under recent nsys versions
# (https://nvbugs/6127669):
#   - The whole benchmark process is traced instead of gating the collection
#     with "-c cudaProfilerApi": kernels launched by CUDA graphs that were
#     instantiated before the capture range opened are exported without
#     runtime correlation (correlationId 0), which breaks parse_e2e.py, and
#     the engine captures its CUDA graphs during warmup, before any capture
#     range can open. TLLM_PROFILE_START_STOP still bounds the calibration
#     data collection, and parse_e2e.py selects the matching iterations via
#     --start-iter/--stop-iter.
#   - max_num_tokens admits all prefills in a single iteration, so all
#     requests finish at the same iteration. Draining the batch through
#     progressively smaller CUDA graph batch sizes with the profiler attached
#     has been observed to wedge the device stream and hang the executor;
#     a uniform batch also matches the steady-state assumption of the
#     correlation methodology.

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
    -t cuda,nvtx -s none \
    --cpuctxsw none --cuda-event-trace false \
    --cuda-graph-trace node \
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
    --max_num_tokens $((BATCH_SIZE * 2048)) \
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
    -t cuda,nvtx -s none \
    --cpuctxsw none --cuda-event-trace false \
    --cuda-graph-trace node \
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
    --max_num_tokens $((BATCH_SIZE * 2048)) \
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
    --seq-len-kv-cache $((2048 + BATCH_SIZE + 22)) \
    --balance-method NotModified \
    --replay-file-path "$PROFILE_DIR/calibration_data.json" \
    --replay-start-iter $((BATCH_SIZE + 10 + 5)) \
    --replay-stop-iter $((BATCH_SIZE + 34))
# The calibration file contains the 25 iterations [BATCH_SIZE+10, BATCH_SIZE+34]:
# collection starts at the TLLM_PROFILE_START_STOP start iteration and the stop
# iteration itself is not collected. Replaying [BATCH_SIZE+15, BATCH_SIZE+34]
# matches the 20 iterations that parse_e2e.py keeps after --warmup-times 5.
# --seq-len-kv-cache is the average past length over the replayed iterations:
# 2048 prompt tokens (all requests prefill at iteration 1 and decode in
# lockstep) plus the mid-window decode offset.

# Step 4

seq 0 $((NP - 1)) | xargs -I% python3 parse_e2e.py \
    --eager-trace "$PROFILE_DIR/report_e2e_mark_rank%.nsys-rep" \
    --graph-trace "$PROFILE_DIR/report_e2e_collect_rank%.nsys-rep" \
    --layer-indices 5,6,7 \
    --warmup-times 5 \
    --start-iter $((BATCH_SIZE + 10)) \
    --stop-iter $((BATCH_SIZE + 34)) \
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
