#!/bin/bash
set -Eeuo pipefail

# Parse arguments
jobWorkspace=$1
llmTarfile=$2
tarName=$3
llmSrcNode=$4
stageName=$5
perfMode=$6
resourcePathNode=$7
pytestCommand=$8
pytestWithoutLLMAPILaunchCommand=$9
coverageConfigFile=${10}
container=${11}
mounts=${12}
scriptRunNode=${13}
scriptInstallNode=${14}
configYamlFile=${15}
testListPathNode=${16}
num_ctx_servers=${17}
num_gen_servers=${18}
gpus_per_node=${19}
gpus_per_ctx_server=${20}
gpus_per_gen_server=${21}
nodes_per_ctx_server=${22}
nodes_per_gen_server=${23}
total_nodes=${24}
total_gpus=${25}

echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

# Export environment variables
export jobWorkspace=$jobWorkspace
export tarName=$tarName
export llmTarfile=$llmTarfile
export llmSrcNode=$llmSrcNode
export stageName=$stageName
export perfMode=$perfMode
export resourcePathNode=$resourcePathNode
export pytestCommand="$pytestCommand"
export pytestWithoutLLMAPILaunchCommand="$pytestWithoutLLMAPILaunchCommand"
export coverageConfigFile="$coverageConfigFile"
export NVIDIA_IMEX_CHANNELS=0
export NVIDIA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi --query-gpu=count -i 0 --format=noheader)-1)))

# Setup srun arguments
srunArgs=(
    "--container-image=$container"
    "--container-name=disagg-test-$SLURM_JOB_ID"
    "--container-workdir=/home/svc_tensorrt/bloom/scripts"
    "--container-mounts=$mounts"
    "--container-env=NVIDIA_IMEX_CHANNELS"
    "--container-env=DISAGG_SERVER_IDX"
    "--mpi=pmix"
)

# Start container
echo "Starting container..."
if ! srun "${srunArgs[@]}" echo "Container up." &> $jobWorkspace/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check $jobWorkspace/container_launch.log"
fi

# Install tensorrt-llm on each node
echo "Installing TensorRT-LLM..."
chmod +x $scriptInstallNode
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -N $SLURM_NNODES --ntasks-per-node=1 --ntasks=$SLURM_NNODES $scriptInstallNode &> $jobWorkspace/install.log; then
    cleanup_on_failure "TensorRT-LLM installation failed. Check $jobWorkspace/install.log for details"
fi
echo "TensorRT-LLM installation completed successfully"

# Print received hardware configuration
echo "Hardware configuration (from slurm_exec.py):"
echo "  num_ctx_servers: $num_ctx_servers"
echo "  num_gen_servers: $num_gen_servers"
echo "  gpus_per_node: $gpus_per_node"
echo "  gpus_per_ctx_server: $gpus_per_ctx_server"
echo "  gpus_per_gen_server: $gpus_per_gen_server"
echo "  nodes_per_ctx_server: $nodes_per_ctx_server"
echo "  nodes_per_gen_server: $nodes_per_gen_server"
echo "  total_nodes: $total_nodes"
echo "  total_gpus: $total_gpus"

# Calculate node counts for splitting
total_ctx_nodes=$((num_ctx_servers * nodes_per_ctx_server))
total_gen_nodes=$((num_gen_servers * nodes_per_gen_server))

echo "  total_ctx_nodes: $total_ctx_nodes"
echo "  total_gen_nodes: $total_gen_nodes"

# Get enable_pdl from config yaml
# enable_pdl=false
# echo "enable_pdl: $enable_pdl"

# Get node lists from SLURM_NODELIST
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
total_nodes_num=${#all_nodes[@]}
echo "all_nodes: ${all_nodes[@]}, total_nodes_num: $total_nodes_num"

# Split nodes: gen_nodes, ctx_nodes
gen_node_list=(${all_nodes[@]:0:$total_gen_nodes})
ctx_node_list=(${all_nodes[@]:$total_gen_nodes:$total_ctx_nodes})

echo "gen_nodes: ${gen_node_list[@]}"
echo "ctx_nodes: ${ctx_node_list[@]}"

chmod +x $scriptRunNode

# Start gen servers
echo "Starting gen servers..."
for i in $(seq 0 $((num_gen_servers - 1))); do
    gen_world_size=$((nodes_per_gen_server * gpus_per_node))
    export DISAGG_SERVER_IDX="GEN_$i"
    srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
        -N $nodes_per_gen_server \
        --ntasks=$gen_world_size \
        --ntasks-per-node=$gpus_per_node \
        --container-env=DISAGG_SERVER_IDX \
        $scriptRunNode &> $jobWorkspace/gen_server_$i.log &
    echo "Started gen server $i"
done

# Start ctx servers
echo "Starting ctx servers..."
for i in $(seq 0 $((num_ctx_servers - 1))); do
    ctx_world_size=$((nodes_per_ctx_server * gpus_per_node))
    export DISAGG_SERVER_IDX="CTX_$i"
    srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
        -N $nodes_per_ctx_server \
        --ntasks=$ctx_world_size \
        --ntasks-per-node=$gpus_per_node \
        --container-env=DISAGG_SERVER_IDX \
        $scriptRunNode &> $jobWorkspace/ctx_server_$i.log &
    echo "Started ctx server $i"
done

# Start disagg server
echo "Starting disagg server..."
export DISAGG_SERVER_IDX="DISAGG_SERVER"
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    -w ${gen_node_list[0]} \
    --container-env=DISAGG_SERVER_IDX \
    $scriptRunNode &> $jobWorkspace/disagg_server.log &

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVER_IDX="BENCHMARK"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    -w ${gen_node_list[0]} \
    --container-env=DISAGG_SERVER_IDX \
    $scriptRunNode &> $jobWorkspace/benchmark.log; then
    cleanup_on_failure "Benchmark failed. Check logs in ${jobWorkspace} for details"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"

scancel ${SLURM_JOB_ID}
