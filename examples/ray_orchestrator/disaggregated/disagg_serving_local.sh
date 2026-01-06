#!/bin/bash

# Parse command line arguments
BACKEND="ray"
ATTACH_MODE=false
MODEL_DIR="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TP_SIZE=1
USAGE="Usage: $0 [--executor ray|mpi] [--attach] [--model model_dir] [--tp_size N] [--help]"

while [[ $# -gt 0 ]]; do
    case $1 in
        --executor)
            BACKEND="$2"
            shift 2
            ;;
        --attach)
            ATTACH_MODE=true
            shift
            ;;
        --model)
            MODEL_DIR="$2"
            shift 2
            ;;
        --tp_size)
            TP_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "$USAGE"
            echo "Options:"
            echo "  --executor ray|mpi   Choose distributed executor (default: ray)"
            echo "  --attach             Attach to existing ray cluster (skip ray start/stop)"
            echo "  --model model_dir    Model directory (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
            echo "  --tp_size N          Tensor parallel size (default: 1)"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

if [[ "$BACKEND" != "ray" && "$BACKEND" != "mpi" ]]; then
    echo "Error: Executor must be either 'ray' or 'mpi'"
    echo "$USAGE"
    exit 1
fi

echo "Executor: $BACKEND"
echo "Tensor parallel size: $TP_SIZE"
if [[ "$ATTACH_MODE" == "true" ]]; then
    echo "Attach mode enabled - will not manage ray cluster"
fi

# Generate extra_llm_config.yaml based on executor type
echo "Generating extra_llm_config.yaml for executor: $BACKEND"
if [[ "$BACKEND" == "ray" ]]; then
    # Ray backend generates per-server configs with placement_groups (GPU indices)
    # Context server uses GPUs 0 to (TP_SIZE-1)
    # Generation server uses GPUs TP_SIZE to (2*TP_SIZE-1)

    # Generate placement groups array for context server: [[0, 1, ..., TP_SIZE-1]]
    # Format: List of placement groups, each group is a list of GPU indices
    CTX_PLACEMENT_GROUPS="[["
    for ((i=0; i<TP_SIZE; i++)); do
        if [[ $i -gt 0 ]]; then CTX_PLACEMENT_GROUPS+=", "; fi
        CTX_PLACEMENT_GROUPS+="$i"
    done
    CTX_PLACEMENT_GROUPS+="]]"

    # Generate bundle indices for context server: [[0, 1, ..., TP_SIZE-1]]
    CTX_BUNDLE_INDICES="[["
    for ((i=0; i<TP_SIZE; i++)); do
        if [[ $i -gt 0 ]]; then CTX_BUNDLE_INDICES+=", "; fi
        CTX_BUNDLE_INDICES+="$i"
    done
    CTX_BUNDLE_INDICES+="]]"

    # Generate placement groups array for generation server: [[TP_SIZE, TP_SIZE+1, ..., 2*TP_SIZE-1]]
    GEN_PLACEMENT_GROUPS="[["
    for ((i=TP_SIZE; i<2*TP_SIZE; i++)); do
        if [[ $i -gt TP_SIZE ]]; then GEN_PLACEMENT_GROUPS+=", "; fi
        GEN_PLACEMENT_GROUPS+="$i"
    done
    GEN_PLACEMENT_GROUPS+="]]"

    # Generate bundle indices for generation server: [[0, 1, ..., TP_SIZE-1]]
    GEN_BUNDLE_INDICES="[["
    for ((i=0; i<TP_SIZE; i++)); do
        if [[ $i -gt 0 ]]; then GEN_BUNDLE_INDICES+=", "; fi
        GEN_BUNDLE_INDICES+="$i"
    done
    GEN_BUNDLE_INDICES+="]]"

    echo "Context server placement_groups: $CTX_PLACEMENT_GROUPS"
    echo "Context server placement_bundle_indices: $CTX_BUNDLE_INDICES"
    echo "Generation server placement_groups: $GEN_PLACEMENT_GROUPS"
    echo "Generation server placement_bundle_indices: $GEN_BUNDLE_INDICES"

    # Context server config
    cat > extra_llm_config_ctx.yaml << EOF
# extra_llm_config.yaml for context server with Ray placement
cache_transceiver_config:
    backend: "UCX"
    max_tokens_in_buffer: 1024
disable_overlap_scheduler: true
orchestrator_type: "ray"
max_batch_size: 1
max_num_tokens: 512
max_seq_len: 128
kv_cache_config:
    free_gpu_memory_fraction: 0.8
# Ray placement config - specifies which GPUs to use
# placement_groups: List of GPU index lists (one list per placement group)
# placement_bundle_indices: Which bundle indices to use from each placement group
ray_placement_config:
    placement_groups: $CTX_PLACEMENT_GROUPS
    placement_bundle_indices: $CTX_BUNDLE_INDICES
EOF

    # Generation server config
    cat > extra_llm_config_gen.yaml << EOF
# extra_llm_config.yaml for generation server with Ray placement
cache_transceiver_config:
    backend: "UCX"
    max_tokens_in_buffer: 1024
disable_overlap_scheduler: true
orchestrator_type: "ray"
max_batch_size: 1
max_num_tokens: 512
max_seq_len: 128
kv_cache_config:
    free_gpu_memory_fraction: 0.8
# Ray placement config - specifies which GPUs to use
ray_placement_config:
    placement_groups: $GEN_PLACEMENT_GROUPS
    placement_bundle_indices: $GEN_BUNDLE_INDICES
EOF

    # Also create a generic one for backward compatibility
    cat > extra_llm_config.yaml << EOF
# extra_llm_config.yaml when launching disaggregated server instances.
cache_transceiver_config:
    backend: "UCX"
    max_tokens_in_buffer: 1024
disable_overlap_scheduler: true
orchestrator_type: "ray"
max_batch_size: 1
max_num_tokens: 512
max_seq_len: 128
kv_cache_config:
    free_gpu_memory_fraction: 0.8
EOF
else
    cat > extra_llm_config.yaml << EOF
# extra_llm_config.yaml when launching disaggregated server instances.
cache_transceiver_config:
    backend: "UCX"
    max_tokens_in_buffer: 1024
disable_overlap_scheduler: true
# Using default executor MPI (no orchestrator_type specified)
# Memory-saving parameters
max_batch_size: 1
max_num_tokens: 512
max_seq_len: 128
kv_cache_config:
    free_gpu_memory_fraction: 0.8
EOF
fi

# Generate disaggregated server config
echo "Generating disagg_config_local.yaml"
cat > disagg_config_local.yaml << EOF
hostname: localhost
port: 8000
model: $MODEL_DIR
free_gpu_memory_fraction: 0.8
backend: "pytorch"
disable_overlap_scheduler: True
context_servers:
  num_instances: 1
  tensor_parallel_size: $TP_SIZE
  pipeline_parallel_size: 1
  max_batch_size: 1
  max_num_tokens: 512
  max_seq_len: 128
  kv_cache_config:
    free_gpu_memory_fraction: 0.8
  cache_transceiver_config:
    backend: "UCX"
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  tensor_parallel_size: $TP_SIZE
  pipeline_parallel_size: 1
  max_batch_size: 1
  max_num_tokens: 512
  max_seq_len: 128
  kv_cache_config:
    free_gpu_memory_fraction: 0.8
  cache_transceiver_config:
    backend: "UCX"
  urls:
      - "localhost:8002"
EOF

# Conditionally start ray head if using ray backend and not in attach mode
RAY_STARTED=false
if [[ "$BACKEND" == "ray" && "$ATTACH_MODE" != "true" ]]; then
    echo "Checking if ray cluster is already running..."
    if ray status > /dev/null 2>&1; then
        echo "Ray cluster is already running. Stopping existing cluster first..."
        ray stop
        sleep 2
    fi
    echo "Launching ray head..."
    ray start --head --disable-usage-stats
    RAY_STARTED=true
elif [[ "$BACKEND" == "ray" && "$ATTACH_MODE" == "true" ]]; then
    echo "Attach mode: Skipping ray cluster management"
fi

# Launching context servers
echo "Launching context servers..."
if [[ "$BACKEND" == "mpi" ]]; then
    CTX_GPUS=$(seq -s, 0 $((TP_SIZE - 1)))
    echo "Context server using GPUs: $CTX_GPUS (via CUDA_VISIBLE_DEVICES)"
    (
        export CUDA_VISIBLE_DEVICES=$CTX_GPUS
        trtllm-serve $MODEL_DIR --host localhost --tp_size $TP_SIZE --port 8001 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config.yaml
    ) &> output_ctx0 &
elif [[ "$BACKEND" == "ray" ]]; then
    echo "Context server using GPUs: 0 to $((TP_SIZE - 1)) (via ray_placement_config.gpu_indices)"
    (
        # Ray backend uses ray_placement_config.gpu_indices from YAML instead of CUDA_VISIBLE_DEVICES
        trtllm-serve $MODEL_DIR --host localhost --tp_size $TP_SIZE --port 8001 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config_ctx.yaml
    ) &> output_ctx0 &
else
    trtllm-serve $MODEL_DIR --host localhost --tp_size $TP_SIZE --port 8001 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config.yaml &> output_ctx0 &
fi
CTX_PID=$!
echo "Context server started with PID: $CTX_PID"

# Launching generation servers
echo "Launching generation servers..."
if [[ "$BACKEND" == "mpi" ]]; then
    GEN_GPUS=$(seq -s, $TP_SIZE $((2 * TP_SIZE - 1)))
    echo "Generation server using GPUs: $GEN_GPUS (via CUDA_VISIBLE_DEVICES)"
    (
        export CUDA_VISIBLE_DEVICES=$GEN_GPUS
        trtllm-serve $MODEL_DIR --host localhost --tp_size $TP_SIZE --port 8002 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config.yaml
    ) &> output_gen0 &
elif [[ "$BACKEND" == "ray" ]]; then
    echo "Generation server using GPUs: $TP_SIZE to $((2 * TP_SIZE - 1)) (via ray_placement_config.gpu_indices)"
    (
        # Ray backend uses ray_placement_config.gpu_indices from YAML instead of CUDA_VISIBLE_DEVICES
        trtllm-serve $MODEL_DIR --host localhost --tp_size $TP_SIZE --port 8002 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config_gen.yaml
    ) &> output_gen0 &
else
    trtllm-serve $MODEL_DIR --host localhost --tp_size $TP_SIZE --port 8002 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config.yaml &> output_gen0 &
fi
GEN_PID=$!
echo "Generation server started with PID: $GEN_PID"

# Launching disaggregated server
echo "Launching disaggregated server..."
trtllm-serve disaggregated -c disagg_config_local.yaml

# Cleanup
if [[ "$RAY_STARTED" == "true" && "$ATTACH_MODE" != "true" ]]; then
    echo "Stopping ray..."
    ray stop
fi

echo "Cleaning up generated config files..."
rm -f extra_llm_config.yaml extra_llm_config_ctx.yaml extra_llm_config_gen.yaml
