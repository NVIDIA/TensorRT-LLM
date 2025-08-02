#!/bin/bash

model_card="deepseek-ai/DeepSeek-R1"
model_path="/llm-models/DeepSeek-R1/DeepSeek-R1-FP4/"
dataset_file="/tmp/aa_prompt_50000.txt"
nsys_on="true"
#nsys_on="false"

# isl=8192
# osl=1024

multi_round=1
disable_overlap_scheduler="false"

# Default batch split configuration
USE_BATCH_SPLIT_MODEL="true"
BATCH_SPLIT_RATIO=0.5
USE_SEPARATE_STREAMS="true"
SYNC_AFTER_ATTENTION="true"
ENABLE_LAYER_COMPLETION_EVENTS="false"
ENABLE_INTRA_LAYER_BATCH_SPLITTING="true"
ENABLE_ATTENTION_METADATA_SPLITTING="true"
ENABLE_SEPARATE_KV_CACHE_MANAGERS="true"

# Function to print usage
print_usage() {
    echo "Usage: $0 <concurrency> <max_batch_size> <max_num_tokens> <tp> <ep> <gpu_fraction> <eplb_num_slots> <updates_per_iter> <log_dir> [mode]"
    echo ""
    echo "Required arguments:"
    echo "  concurrency      - Number of concurrent requests"
    echo "  max_batch_size   - Maximum batch size"
    echo "  max_num_tokens   - Maximum number of tokens"
    echo "  tp               - Tensor parallelism"
    echo "  ep               - Expert parallelism"
    echo "  gpu_fraction     - GPU memory fraction for KV cache"
    echo "  eplb_num_slots   - EPLB number of slots (0 to disable)"
    echo "  updates_per_iter - Updates per iteration"
    echo "  log_dir          - Log directory"
    echo ""
    echo "Optional mode argument (default: 4):"
    echo "  1 - No batch splitting and no inter-layer overlap"
    echo "  2 - Only inter-layer overlap (pipeline parallelism)"
    echo "  3 - Only batch splitting (no inter-layer overlap)"
    echo "  4 - Both inter-layer and batch splitting (full overlap)"
    echo ""
    echo "Examples:"
    echo "  $0 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_batch_split_20250716"
    echo "  $0 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_batch_split_20250716 2"
    echo "  $0 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_batch_split_20250716 1"
}

# Function to configure batch splitting based on mode
configure_batch_splitting() {
    local mode=$1
    
    case $mode in
        1)
            echo "Mode 1: No batch splitting and no inter-layer overlap"
            USE_BATCH_SPLIT_MODEL="false"
            USE_SEPARATE_STREAMS="false"
            ENABLE_LAYER_COMPLETION_EVENTS="false"
            ENABLE_INTRA_LAYER_BATCH_SPLITTING="false"
            ENABLE_ATTENTION_METADATA_SPLITTING="false"
            ENABLE_SEPARATE_KV_CACHE_MANAGERS="false"
            SYNC_AFTER_ATTENTION="false"
            ;;
        2)
            echo "Mode 2: Only inter-layer overlap (pipeline parallelism)"
            USE_BATCH_SPLIT_MODEL="true"
            USE_SEPARATE_STREAMS="true"
            ENABLE_LAYER_COMPLETION_EVENTS="true"
            ENABLE_INTRA_LAYER_BATCH_SPLITTING="false"
            ENABLE_ATTENTION_METADATA_SPLITTING="false"
            ENABLE_SEPARATE_KV_CACHE_MANAGERS="false"
            SYNC_AFTER_ATTENTION="false"
            ;;
        3)
            echo "Mode 3: Only batch splitting (no inter-layer overlap)"
            USE_BATCH_SPLIT_MODEL="true"
            USE_SPLIT_KV_CACHE="true"
            USE_SEPARATE_STREAMS="true"
            ENABLE_LAYER_COMPLETION_EVENTS="false"
            ENABLE_INTRA_LAYER_BATCH_SPLITTING="true"
            ENABLE_ATTENTION_METADATA_SPLITTING="true"
            ENABLE_SEPARATE_KV_CACHE_MANAGERS="true"
            SYNC_AFTER_ATTENTION="true"
            ;;
        4)
            echo "Mode 4: Both inter-layer and batch splitting (full overlap)"
            USE_BATCH_SPLIT_MODEL="true"
            USE_SPLIT_KV_CACHE="true"
            USE_SEPARATE_STREAMS="true"
            ENABLE_LAYER_COMPLETION_EVENTS="true"
            ENABLE_INTRA_LAYER_BATCH_SPLITTING="true"
            ENABLE_ATTENTION_METADATA_SPLITTING="true"
            ENABLE_SEPARATE_KV_CACHE_MANAGERS="true"
            SYNC_AFTER_ATTENTION="true"
            ;;
        *)
            echo "Error: Invalid mode '$mode'. Must be 1, 2, 3, or 4."
            print_usage
            exit 1
            ;;
    esac
}

# Parameter validation and default values
if [ $# -lt 9 ]; then
    echo "Error: Insufficient arguments provided."
    print_usage
    exit 1
fi

concurrency=${1:-256}
max_batch_size=${2:-64}
max_num_tokens=${3:-2400}
tp=${4:-4}
ep=${5:-4}
gpu_fraction=${6:-0.5}
eplb_num_slots=${7:-0}
updates_per_iter=${8:-1}
log_dir=${9:-"/scratch/TensorRT-LLM/results_batch_split_$(date +%Y%m%d)"}
mode=${10:-2}  # Default to mode 2 (only inter-layer overlap)

# Validate that eplb_num_slots is a number
if ! [[ "${eplb_num_slots}" =~ ^[0-9]+$ ]]; then
    echo "Error: eplb_num_slots must be a non-negative integer, got: '${eplb_num_slots}'"
    exit 1
fi

# Validate that updates_per_iter is a number
if ! [[ "${updates_per_iter}" =~ ^[0-9]+$ ]]; then
    echo "Error: updates_per_iter must be a non-negative integer, got: '${updates_per_iter}'"
    exit 1
fi

# Configure batch splitting based on mode
configure_batch_splitting $mode

echo "Parameters:"
echo "  concurrency: ${concurrency}"
echo "  max_batch_size: ${max_batch_size}"
echo "  max_num_tokens: ${max_num_tokens}"
echo "  tp: ${tp}"
echo "  ep: ${ep}"
echo "  gpu_fraction: ${gpu_fraction}"
echo "  eplb_num_slots: ${eplb_num_slots}"
echo "  updates_per_iter: ${updates_per_iter}"
echo "  log_dir: ${log_dir}"
echo "  mode: ${mode}"
echo

gen_aa_dataset() {
    local isl=$1
    local osl=$2
    local num_prompts=$3
    local dataset_file=$4

    # run the loadgen
    loadgen_config_file=/tmp/loadgen_config.json
    # Generate the loadgen_config.json file
    cat << EOF > ${loadgen_config_file}
{
    "dataset": {
        "type": "aa",
        "model_name": "${model_path}",
        "seed": 42,
        "max_count": ${num_prompts},
        "approximate_input_tokens": ${isl},
        "num_sentences_to_translate": 100,
        "output_tokens": ${osl}
    },
    "inference_server": {
        "type": "dump_dataset",
        "output_file_name": "${dataset_file}"
    },
    "timing_strategy": {
        "type": "fixed",
        "desired_rps": -1
    },
    "post_processors": [],
    "timeout": null
}
EOF
    infserver_loadgen ${loadgen_config_file} --output_dir /tmp/ 
    echo "Dataset generated to ${dataset_file}"
}

check_dataset_exists() {
    local dataset_file=$1
    local expected_lines=$2
    
    if [ -f "${dataset_file}" ]; then
        local actual_lines=$(wc -l < "${dataset_file}")
        if [ "${actual_lines}" -eq "${expected_lines}" ]; then
            echo "Dataset ${dataset_file} already exists with ${actual_lines} lines. Skipping generation."
            return 0
        else
            echo "Dataset ${dataset_file} exists but has ${actual_lines} lines (expected ${expected_lines}). Regenerating..."
            return 1
        fi
    else
        echo "Dataset ${dataset_file} does not exist. Generating..."
        return 1
    fi
}

gen_attn_dp_config() {
    local cuda_graph_batch=$1
    local eplb_num_slots=$2
    local moe_max_num_tokens=$3
    local updates_per_iter=$4
    local sub_dir=$5
    
    extra_llm_api_file=${sub_dir}/extra-llm-api-config.yml
    echo "extra_llm_api_file: ${extra_llm_api_file}"

    cat << EOF > ${extra_llm_api_file}
enable_attention_dp: true
enable_layerwise_nvtx_marker: true
print_iter_log: true
moe_max_num_tokens: ${moe_max_num_tokens}
enable_chunked_prefill: false
moe_backend: WideEP
kv_cache_dtype: fp8
#cuda_graph_config:
#  padding_enabled: true
#  max_batch_size: ${cuda_graph_batch}
load_format: dummy
EOF

    if [ "${eplb_num_slots}" -gt 0 ]; then
        eplb_config_file=${sub_dir}/moe_load_balancer.yaml
        cat << EOF > ${eplb_config_file}
num_slots: ${eplb_num_slots}
layer_updates_per_iter: ${updates_per_iter}   
EOF

        cat << EOF >> ${extra_llm_api_file}
moe_load_balancer: ${sub_dir}/moe_load_balancer.yaml
EOF
    fi
}   

run_dep_benchmark() {
    local concurrency=$1
    local max_batch_size=$2
    local max_num_tokens=$3
    local tp=$4
    local ep=$5
    local eplb_num_slots=$6
    local updates_per_iter=$7
    local sub_dir=$8

    num_requests=$((concurrency * multi_round))
    cuda_graph_batch=$((concurrency / tp))

    extra_llm_api_file=${sub_dir}/extra-llm-api-config.yml

    nsys_prefix=""
    # check NSYS_MODE is not empty
    if [ "${nsys_on}" == "true" ]; then
        nsys_file=${sub_dir}/nsys_worker_proc_${SLURM_PROCID}
        nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
        export TLLM_PROFILE_START_STOP=700-750
        export TLLM_PROFILE_RECORD_GC=1
        export TLLM_NVTX_DEBUG=1
    fi

    trtllm-bench -m ${model_card} --model_path ${model_path} throughput \
        --tp ${tp} \
        --ep ${ep} \
        --warmup 0 \
        --dataset ${dataset_file} \
        --backend pytorch \
        --max_batch_size ${max_batch_size} \
        --max_num_tokens ${max_num_tokens} \
        --max_seq_len 2304 \
	    --kv_cache_free_gpu_mem_fraction ${gpu_fraction} \
        --extra_llm_api_options ${extra_llm_api_file} \
        --num_requests ${num_requests} \
        --concurrency ${concurrency} \
        --report_json ${sub_dir}/trtllm_bench_report.json \
        --iteration_log ${sub_dir}/iteration.log \
        |& tee ${sub_dir}/trtllm_bench_run.log
}

sub_dir=${log_dir}/dep${ep}_concurrency${concurrency}_eplb${eplb_num_slots}_updates${updates_per_iter}_mode${mode}
mkdir -p ${sub_dir}
echo "sub_dir: ${sub_dir}"

# Define moe_max_num_tokens in global scope
moe_max_num_tokens=16384
if [ "${concurrency}" -gt "${moe_max_num_tokens}" ]; then
    moe_max_num_tokens=${concurrency}
fi

# Set batch split environment variables
export USE_BATCH_SPLIT_MODEL=${USE_BATCH_SPLIT_MODEL}
export USE_SPLIT_KV_CACHE=${USE_SPLIT_KV_CACHE}
export BATCH_SPLIT_RATIO=${BATCH_SPLIT_RATIO}
export USE_SEPARATE_STREAMS=${USE_SEPARATE_STREAMS}
export SYNC_AFTER_ATTENTION=${SYNC_AFTER_ATTENTION}
export ENABLE_LAYER_COMPLETION_EVENTS=${ENABLE_LAYER_COMPLETION_EVENTS}
export ENABLE_INTRA_LAYER_BATCH_SPLITTING=${ENABLE_INTRA_LAYER_BATCH_SPLITTING}
export ENABLE_ATTENTION_METADATA_SPLITTING=${ENABLE_ATTENTION_METADATA_SPLITTING}
export ENABLE_SEPARATE_KV_CACHE_MANAGERS=${ENABLE_SEPARATE_KV_CACHE_MANAGERS}

# Debug: Check environment variables and module import
echo "=== DEBUG: Environment Variables ==="
echo "USE_BATCH_SPLIT_MODEL: ${USE_BATCH_SPLIT_MODEL}"
echo "USE_SPLIT_KV_CACHE: ${USE_SPLIT_KV_CACHE}"
echo "ENABLE_SEPARATE_KV_CACHE_MANAGERS: ${ENABLE_SEPARATE_KV_CACHE_MANAGERS}"

echo "=== DEBUG: Module Import Test ==="
python -c "
import os
print('Environment variables in Python:')
print('  USE_BATCH_SPLIT_MODEL:', os.environ.get('USE_BATCH_SPLIT_MODEL', 'not set'))
print('  USE_SPLIT_KV_CACHE:', os.environ.get('USE_SPLIT_KV_CACHE', 'not set'))

try:
    import tensorrt_llm._torch.models.deepseek_v3_batch_split_moe
    print('✓ Batch split module imported successfully')
    
    # Check if the registration happened
    from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
    print('✓ AutoModelForCausalLM imported successfully')
    
except ImportError as e:
    print('✗ Failed to import batch split module:', e)
except Exception as e:
    print('✗ Error during import:', e)
"

export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1
export TRTLLM_MNNVL_AR_ENABLED=1
export FLASHINFER_WORKSPACE_BASE=/scratch/TensorRT-LLM/flashinfer/

export TLLM_OVERRIDE_LAYER_NUM=5

export EXPERT_STATISTIC_PATH=${sub_dir}/expert_statistic
export EXPERT_STATISTIC_ITER_RANGE=600-650

echo "Batch Split Configuration:"
echo "  USE_BATCH_SPLIT_MODEL: ${USE_BATCH_SPLIT_MODEL}"
echo "  BATCH_SPLIT_RATIO: ${BATCH_SPLIT_RATIO}"
echo "  USE_SEPARATE_STREAMS: ${USE_SEPARATE_STREAMS}"
echo "  SYNC_AFTER_ATTENTION: ${SYNC_AFTER_ATTENTION}"
echo "  ENABLE_LAYER_COMPLETION_EVENTS: ${ENABLE_LAYER_COMPLETION_EVENTS}"
echo "  ENABLE_INTRA_LAYER_BATCH_SPLITTING: ${ENABLE_INTRA_LAYER_BATCH_SPLITTING}"
echo "  ENABLE_ATTENTION_METADATA_SPLITTING: ${ENABLE_ATTENTION_METADATA_SPLITTING}"
echo "  ENABLE_SEPARATE_KV_CACHE_MANAGERS: ${ENABLE_SEPARATE_KV_CACHE_MANAGERS}"
echo

echo "run_dep_benchmark ${concurrency} ${max_batch_size} ${max_num_tokens} ${tp} ${ep} ${eplb_num_slots} ${updates_per_iter} ${sub_dir}"

cuda_graph_batch=$((concurrency / tp))

# Check if dataset exists before generating
if ! check_dataset_exists "${dataset_file}" 50000; then
    gen_aa_dataset 1024 1024 50000 /tmp/aa_prompt_50000.txt
fi

gen_attn_dp_config ${cuda_graph_batch} ${eplb_num_slots} ${moe_max_num_tokens} ${updates_per_iter} ${sub_dir}

# Force import the batch split module to ensure registration happens
echo "=== FORCING BATCH SPLIT MODULE IMPORT ==="
python -c "
import os
print('Forcing import of batch split module...')
try:
    import tensorrt_llm._torch.models.deepseek_v3_batch_split_moe
    print('✓ Batch split module imported and registered')
except Exception as e:
    print('✗ Error importing batch split module:', e)
"

run_dep_benchmark ${concurrency} ${max_batch_size} ${max_num_tokens} ${tp} ${ep} ${eplb_num_slots} ${updates_per_iter} ${sub_dir}
#python /scratch/TensorRT-LLM/examples/wide_ep/ep_load_balancer/report_load_statistics.py --expert_statistic_path $EXPERT_STATISTIC_PATH

# Usage Examples:
# Mode 1: No batch splitting and no inter-layer overlap
# ./run_deepseek_batch_split.sh 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_20250716 1
#
# Mode 2: Only inter-layer overlap (pipeline parallelism)
# ./run_deepseek_batch_split.sh 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_20250716 2
#
# Mode 3: Only batch splitting (no inter-layer overlap)
# ./run_deepseek_batch_split.sh 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_20250716 3
#
# Mode 4: Both inter-layer and batch splitting (full overlap) - DEFAULT
# ./run_deepseek_batch_split.sh 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_20250716 4
# ./run_deepseek_batch_split.sh 256 64 2400 4 4 0.5 0 1 /scratch/TensorRT-LLM/results_20250716  # Same as mode 4

# Errors
# ./run_deepseek_batch_split.sh 256 64 2400 4 4 0.5 256 4 /scratch/TensorRT-LLM/results_20250716
# - load balancing fails due to gdr library missing on platform 