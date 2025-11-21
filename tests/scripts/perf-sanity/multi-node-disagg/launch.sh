#!/bin/bash

set -Eeuo pipefail

num_ctx_servers=$1
num_gen_servers=$2
gpus_per_node=$3
gpus_per_ctx_server=$4
gpus_per_gen_server=$5
nodes_per_ctx_server=$6
nodes_per_gen_server=$7
total_nodes=$8
total_gpus=$9
trtllmsrc=${10}
jobworkspace=${11}
script_dir=${12}
stagename=${13}
test_name=${14}
mounts=${15}
llm_models_root=${16}
build_wheel=${17}

# Set WORKDIR to trtllmsrc
WORKDIR=$trtllmsrc
cd $WORKDIR

# Get container image from jenkins properties
IMAGE=$(grep LLM_SBSA_DOCKER_IMAGE ${trtllmsrc}/jenkins/current_image_tags.properties | head -1 | awk -F "=" '{print $2}')
CONT=$(echo $IMAGE | sed 's|urm.nvidia.com/|urm.nvidia.com#|g')
MOUNTS=$mounts

# Get commit and timestamp
commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
CONTAINER_NAME="disagg-test-$commit"

# Export environment variables
export LLM_MODELS_ROOT="${llm_models_root}"
export OPEN_SEARCH_DB_BASE_URL="http://gpuwa.nvidia.com"
export llmsrc=$WORKDIR
export jobWorkspace="${jobworkspace}"
export stageName="${stagename}"
export testName="${test_name}"
export NVIDIA_IMEX_CHANNELS=0
export NVIDIA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi --query-gpu=count -i 0 --format=noheader 2>/dev/null || echo 8)-1)))

# Build pytest command
export pytestPrefix="LLM_ROOT=${llmsrc} LLM_BACKEND_ROOT=${llmsrc}/triton_backend __LUNOWUD=\"-thread_pool_size=12\""
export LLMAPILaunch="${llmsrc}/tensorrt_llm/llmapi/trtllm-llmapi-launch"
export pytestCommand="pytest -v --perf --perf-log-formats=csv --perf-log-formats yaml --timeout-method=thread --timeout=14400 --rootdir ${llmsrc}/tests/integration/defs --test-prefix=${stageName} --output-dir=${jobWorkspace} --csv=${jobWorkspace}/report.csv --junit-xml ${jobWorkspace}/results.xml -o junit_logging=out-err --test-list=${jobWorkspace}/test_list.txt --splitting-algorithm least_duration --splits 1 --group 1"

echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

# Create job workspace
mkdir -p ${jobWorkspace}

# Create test list file
echo "perf/test_perf.py::test_perf[perf_sanity_upload-${testName}]" > ${jobWorkspace}/test_list.txt

# Get node lists from SLURM_NODELIST
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
total_nodes_num=${#all_nodes[@]}
echo "all_nodes: ${all_nodes[@]}, total_nodes_num: $total_nodes_num"

# Calculate total ctx and gen nodes
total_ctx_nodes=$((num_ctx_servers * nodes_per_ctx_server))
total_gen_nodes=$((num_gen_servers * nodes_per_gen_server))

# Split nodes: gen_nodes, ctx_nodes
gen_node_list=(${all_nodes[@]:0:$total_gen_nodes})
ctx_node_list=(${all_nodes[@]:$total_gen_nodes:$total_ctx_nodes})

echo "gen_nodes: ${gen_node_list[@]}"
echo "ctx_nodes: ${ctx_node_list[@]}"

# Setup srun arguments
srunArgs=(
    "--container-image=$CONT"
    "--container-name=$CONTAINER_NAME"
    "--container-workdir=$WORKDIR"
    "--container-mounts=$MOUNTS"
    "--container-env=NVIDIA_IMEX_CHANNELS"
    "--container-env=DISAGG_SERVER_IDX"
    "--mpi=pmix"
)

# Start container
echo "Starting container..."
if ! srun "${srunArgs[@]}" -l echo "Container up." &> ${jobWorkspace}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${jobWorkspace}/container_launch.log"
fi

# Build TensorRT-LLM wheel if needed
if [ "${build_wheel}" = "true" ]; then
    echo "Building TensorRT-LLM wheel on one node..."
    build_command="python3 ./scripts/build_wheel.py --benchmarks --use_ccache --clean --cuda_architectures '100-real'"
    if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -N 1 --ntasks-per-node=1 --ntasks=1 \
        bash -c "cd ${WORKDIR} && ${build_command}" \
        &> ${jobWorkspace}/build.log; then
        cleanup_on_failure "TensorRT-LLM build failed. Check ${jobWorkspace}/build.log for details"
    fi
    echo "TensorRT-LLM build completed successfully"
fi

# Install tensorrt-llm on each node
echo "Installing TensorRT-LLM..."
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -N $SLURM_NNODES --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash ${script_dir}/install.sh &> ${jobWorkspace}/install.log; then
    cleanup_on_failure "TensorRT-LLM installation failed. Check ${jobWorkspace}/install.log for details"
fi
echo "TensorRT-LLM installation completed successfully"

# Start gen servers
echo "Starting gen servers..."
for i in $(seq 0 $((num_gen_servers - 1))); do
    gen_world_size=$((nodes_per_gen_server * gpus_per_node))
    export DISAGG_SERVER_IDX="GEN_$i"
    srun "${srunArgs[@]}" -l --kill-on-bad-exit=1 \
        -N $nodes_per_gen_server \
        --ntasks=$gen_world_size \
        --ntasks-per-node=$gpus_per_node \
        --container-env=DISAGG_SERVER_IDX \
        bash ${script_dir}/run.sh &> ${jobWorkspace}/gen_server_$i.log &
    echo "Started gen server $i"
done

# Start ctx servers
echo "Starting ctx servers..."
for i in $(seq 0 $((num_ctx_servers - 1))); do
    ctx_world_size=$((nodes_per_ctx_server * gpus_per_node))
    export DISAGG_SERVER_IDX="CTX_$i"
    srun "${srunArgs[@]}" -l --kill-on-bad-exit=1 \
        -N $nodes_per_ctx_server \
        --ntasks=$ctx_world_size \
        --ntasks-per-node=$gpus_per_node \
        --container-env=DISAGG_SERVER_IDX \
        bash ${script_dir}/run.sh &> ${jobWorkspace}/ctx_server_$i.log &
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
    bash ${script_dir}/run.sh &> ${jobWorkspace}/disagg_server.log &
echo "Started disagg server"

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVER_IDX="BENCHMARK"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    -w ${gen_node_list[0]} \
    --container-env=DISAGG_SERVER_IDX \
    bash ${script_dir}/run.sh &> ${jobWorkspace}/benchmark.log; then
    cleanup_on_failure "Benchmark failed. Check logs in ${jobWorkspace} for details"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"

sleep 60
scancel ${SLURM_JOB_ID}
