#!/bin/bash

set -Eeuo pipefail

gpus=$1
nodes=$2
trtllmsrc=$3
jobworkspace=$4
script_dir=$5
stagename=$6
test_name=$7
mounts=$8
llm_models_root=$9
build_wheel=$10

# Set WORKDIR to trtllmsrc
WORKDIR=$trtllmsrc
cd $WORKDIR

IMAGE=$(grep LLM_SBSA_DOCKER_IMAGE ${trtllmsrc}/jenkins/current_image_tags.properties | head -1 | awk -F "=" '{print $2}')
CONT=$(echo $IMAGE | sed 's|urm.nvidia.com/|urm.nvidia.com#|g')
MOUNTS=$mounts

commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
CONTAINER_NAME="aggr-test-${commit}"

# Export environment variables
export LLM_MODELS_ROOT="${llm_models_root}"
export OPEN_SEARCH_DB_BASE_URL="http://gpuwa.nvidia.com"
export llmsrc=$WORKDIR
export jobWorkspace="${jobworkspace}"
export stageName="${stagename}"
export testName="${test_name}"
export NVIDIA_IMEX_CHANNELS=0
export NVIDIA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi --query-gpu=count -i 0 --format=noheader 2>/dev/null || echo 8)-1)))
export pytestCommand="LLM_ROOT=${llmsrc} LLM_BACKEND_ROOT=${llmsrc}/triton_backend __LUNOWUD=\"-thread_pool_size=12\" OPEN_SEARCH_DB_BASE_URL=\"http://gpuwa.nvidia.com\" ${llmsrc}/tensorrt_llm/llmapi/trtllm-llmapi-launch pytest -v --perf --perf-log-formats=csv --perf-log-formats yaml --timeout-method=thread --timeout=14400 --rootdir ${llmsrc}/tests/integration/defs --test-prefix=${stageName} --output-dir=${jobWorkspace} --csv=${jobWorkspace}/report.csv --junit-xml ${jobWorkspace}/results.xml -o junit_logging=out-err --test-list=${jobWorkspace}/test_list.txt --splitting-algorithm least_duration --splits 1 --group 1"

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

# Setup srun arguments
srunArgs=(
    "--container-image=$CONT"
    "--container-name=$CONTAINER_NAME"
    "--container-workdir=$WORKDIR"
    "--container-mounts=$MOUNTS"
    "--mpi=pmi2"
)

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

# Run install.sh on each node
echo "Running install.sh..."
if ! srun "${srunArgs[@]}" \
        -n ${nodes} \
        --nodes ${nodes} \
        bash ${script_dir}/install.sh &> ${jobWorkspace}/install.log; then
    cleanup_on_failure "TensorRT-LLM installation failed. Check ${jobWorkspace}/install.log for details"
fi
echo "TensorRT-LLM installation completed successfully"

# Run run.sh
echo "Running run.sh..."
srun "${srunArgs[@]}" \
        --container-remap-root \
        -s \
        --overlap \
        -n ${gpus} \
        --nodes ${nodes} \
        bash ${script_dir}/run.sh &> ${jobWorkspace}/run.log

echo "Run completed. Log: ${jobWorkspace}/run.log"
echo "Job completed successfully!"
