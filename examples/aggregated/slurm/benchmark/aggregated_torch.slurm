#!/bin/bash
set -euo pipefail

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --work-dir) work_dir="$2"; shift 2 ;;
        --full-logdir) full_logdir="$2"; shift 2 ;;
        --container-name) container_name="$2"; shift 2 ;;
        --container-mount) container_mount="$2"; shift 2 ;;
        --container-image) container_image="$2"; shift 2 ;;
        --trtllm-repo) trtllm_repo="$2"; shift 2 ;;
        --build-wheel) build_wheel="$2"; shift 2 ;;
        --cuda-architectures) cuda_architectures="$2"; shift 2 ;;
        --trtllm-wheel-path) trtllm_wheel_path="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Parsed arguments:"
echo "  work_dir: ${work_dir}"
echo "  full_logdir: ${full_logdir}"
echo "  container_mount: ${container_mount}"
echo "  container_image: ${container_image}"
echo "  trtllm_repo: ${trtllm_repo}"
echo "  build_wheel: ${build_wheel}"
echo "  cuda_architectures: ${cuda_architectures}"
echo "  trtllm_wheel_path: ${trtllm_wheel_path}"

cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

env > ${full_logdir}/environment.txt

# Start container
echo "Starting container..."
if ! srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        echo "Container up." &> ${full_logdir}/1_container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${full_logdir}/1_container_launch.log"
fi

# Install TensorRT-LLM
if [ -n "${trtllm_wheel_path}" ]; then
    echo "Installing TensorRT-LLM from wheel: ${trtllm_wheel_path}..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "pip install ${trtllm_wheel_path}[devel]" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM wheel installation failed. Check ${full_logdir}/2_install.log"
    fi
    echo "TensorRT-LLM wheel installation completed"
elif [ -d "${trtllm_repo}" ]; then
    echo "Installing TensorRT-LLM from ${trtllm_repo}..."
    TRT_LLM_GIT_COMMIT=$(git -C ${trtllm_repo} rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "TRT_LLM_GIT_COMMIT: ${TRT_LLM_GIT_COMMIT}"

    if [ "${build_wheel}" = "true" ]; then
        echo "Building TensorRT-LLM wheel..."
        build_command="python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --clean"
        if [ -n "${cuda_architectures:-}" ]; then
            build_command="${build_command} --cuda_architectures \"${cuda_architectures}\""
        fi
        if ! srun --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
            bash -c "cd ${trtllm_repo} && rm -rf .venv-* && ${build_command}" \
            &> ${full_logdir}/2_build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${full_logdir}/2_build.log"
        fi
        echo "TensorRT-LLM build completed"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && pip install -e .[devel]" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${full_logdir}/2_install.log"
    fi
    echo "TensorRT-LLM installation completed"
else
    echo "No trtllm_wheel_path or trtllm_repo provided, using container's TensorRT-LLM"
fi

# Get hostname for server
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
server_hostname=$(scontrol show hostname $SLURM_NODELIST | head -1)
echo "Server hostname: ${server_hostname}"

# Replace <HOSTNAME> placeholder in command files
sed -i "s/<HOSTNAME>/${server_hostname}/g" ${full_logdir}/wait_server_cmds.sh
sed -i "s/<HOSTNAME>/${server_hostname}/g" ${full_logdir}/client_cmds.sh

# Start server
echo "Starting server..."
start_server_cmd=$(cat ${full_logdir}/start_server_cmds.sh)
echo "Executing: ${start_server_cmd}"
eval "${start_server_cmd}"

# Wait for server to be healthy
echo "Waiting for server to become healthy..."
wait_cmd=$(cat ${full_logdir}/wait_server_cmds.sh)
echo "Executing: ${wait_cmd}"
eval "${wait_cmd}"
echo "Server is ready!"

# Run benchmark
echo "Starting benchmark..."
while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    echo "Executing: ${cmd}"
    eval "${cmd}"
    if [ $? -ne 0 ]; then
        cleanup_on_failure "Command failed: ${cmd}"
    fi
done < "${full_logdir}/client_cmds.sh"

echo "Job completed successfully, total runtime: $SECONDS seconds"

scancel ${SLURM_JOB_ID}
