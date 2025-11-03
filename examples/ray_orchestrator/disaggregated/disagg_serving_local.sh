#!/bin/bash

# Parse command line arguments
BACKEND="ray"
ATTACH_MODE=false
TP_SIZE=1
USAGE="Usage: $0 [--executor ray|mpi] [--attach] [--tp_size N] [--help]"

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
        --tp_size)
            TP_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "$USAGE"
            echo "Options:"
            echo "  --executor ray|mpi   Choose distributed executor (default: ray)"
            echo "  --attach             Attach to existing ray cluster (skip ray start/stop)"
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
    cat > extra_llm_config.yaml << EOF
# extra_llm_config.yaml when launching disaggregated server instances.
cache_transceiver_config:
    backend: "UCX"
    max_tokens_in_buffer: 2048
disable_overlap_scheduler: true
# Ray executor configuration
orchestrator_type: "ray"
EOF
else
    cat > extra_llm_config.yaml << EOF
# extra_llm_config.yaml when launching disaggregated server instances.
cache_transceiver_config:
    backend: "UCX"
    max_tokens_in_buffer: 2048
disable_overlap_scheduler: true
# Using default executor MPI (no orchestrator_type specified)
EOF
fi

# Generate disaggregated server config
echo "Generating disagg_config_local.yaml"
cat > disagg_config_local.yaml << EOF
hostname: localhost
port: 8000
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
free_gpu_memory_fraction: 0.25
backend: "pytorch"
disable_overlap_scheduler: True
context_servers:
  num_instances: 1
  tensor_parallel_size: $TP_SIZE
  pipeline_parallel_size: 1
  kv_cache_config:
    free_gpu_memory_fraction: 0.2
  cache_transceiver_config:
    backend: "UCX"
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  tensor_parallel_size: $TP_SIZE
  pipeline_parallel_size: 1
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
    export CUDA_VISIBLE_DEVICES=0
fi

trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --tp_size $TP_SIZE --port 8001 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config.yaml &> output_ctx0 &

if [[ "$BACKEND" == "mpi" ]]; then
    export CUDA_VISIBLE_DEVICES=1
fi
# Launching generation servers
echo "Launching generation servers..."
trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --tp_size $TP_SIZE --port 8002 --kv_cache_free_gpu_memory_fraction 0.15 --backend pytorch --extra_llm_api_options extra_llm_config.yaml &> output_gen0 &

# Launching disaggregated server
echo "Launching disaggregated server..."
trtllm-serve disaggregated -c disagg_config_local.yaml

# Cleanup
if [[ "$RAY_STARTED" == "true" && "$ATTACH_MODE" != "true" ]]; then
    echo "Stopping ray..."
    ray stop
fi

echo "Cleaning up generated extra_llm_config.yaml..."
rm -f extra_llm_config.yaml
