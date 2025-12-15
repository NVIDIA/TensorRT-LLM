#!/bin/bash
set -euo pipefail

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Worker configuration
        --num-ctx-servers) num_ctx_servers="$2"; shift 2 ;;
        --num-gen-servers) num_gen_servers="$2"; shift 2 ;;
        --concurrency-list) concurrency_list="$2"; shift 2 ;;

        # Sequence and benchmark parameters
        --isl) isl="$2"; shift 2 ;;
        --osl) osl="$2"; shift 2 ;;
        --multi-round) multi_round="$2"; shift 2 ;;
        --benchmark-ratio) benchmark_ratio="$2"; shift 2 ;;
        --streaming) streaming="$2"; shift 2 ;;
        --use-nv-sa-benchmark) use_nv_sa_benchmark="$2"; shift 2 ;;

        # Environment and paths
        --dataset-file) dataset_file="$2"; shift 2 ;;
        --model-path) model_path="$2"; shift 2 ;;
        --trtllm-repo) trtllm_repo="$2"; shift 2 ;;
        --work-dir) work_dir="$2"; shift 2 ;;
        --full-logdir) full_logdir="$2"; shift 2 ;;
        --container-mount) container_mount="$2"; shift 2 ;;
        --container-image) container_image="$2"; shift 2 ;;
        --build-wheel) build_wheel="$2"; shift 2 ;;
        --cuda-architectures) cuda_architectures="$2"; shift 2 ;;
        --trtllm-wheel-path) trtllm_wheel_path="$2"; shift 2 ;;

        # Accuracy evaluation
        --enable-accuracy-test) enable_accuracy_test="$2"; shift 2 ;;
        --accuracy-model) accuracy_model="$2"; shift 2 ;;
        --accuracy-tasks) accuracy_tasks="$2"; shift 2 ;;
        --model-args-extra) model_args_extra="$2"; shift 2 ;;

        # Server environment variables
        --server-env-var) server_env_var="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Print all parsed arguments
echo "Parsed arguments:"
echo
echo "Worker Configuration:"
echo "  num_ctx_servers: ${num_ctx_servers}"
echo "  num_gen_servers: ${num_gen_servers}"
echo "  concurrency_list: ${concurrency_list}"
echo
echo "Benchmark Configuration:"
echo "  isl: ${isl}"
echo "  osl: ${osl}"
echo "  multi_round: ${multi_round}"
echo "  benchmark_ratio: ${benchmark_ratio}"
echo "  streaming: ${streaming}"
echo "  use_nv_sa_benchmark: ${use_nv_sa_benchmark}"
echo
echo "Environment Configuration:"
echo "  dataset_file: ${dataset_file}"
echo "  model_path: ${model_path}"
echo "  trtllm_repo: ${trtllm_repo}"
echo "  work_dir: ${work_dir}"
echo "  full_logdir: ${full_logdir}"
echo "  container_mount: ${container_mount}"
echo "  container_image: ${container_image}"
echo "  build_wheel: ${build_wheel}"
echo "  trtllm_wheel_path: ${trtllm_wheel_path}"
echo
echo "Accuracy Configuration:"
echo "  enable_accuracy_test: ${enable_accuracy_test}"
echo "  accuracy_model: ${accuracy_model}"
echo "  accuracy_tasks: ${accuracy_tasks}"
echo "  model_args_extra: ${model_args_extra}"
echo
echo "Server Environment Variables:"
echo "  server_env_var: ${server_env_var}"

# Set TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 for gen_only_no_context mode
if [ "${benchmark_mode}" = "gen_only_no_context" ]; then
    export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1
    worker_env_var="${worker_env_var} TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1"
    server_env_var="${server_env_var} TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1"
    echo "Setting TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 for gen_only_no_context mode"
fi

container_name="disaggr-test"

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}" > ${full_logdir}/job_info.txt
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
    # Install from pre-built wheel if path is provided
    echo "Installing TensorRT-LLM from wheel: ${trtllm_wheel_path}..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "pip install ${trtllm_wheel_path}" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM wheel installation failed. Check ${full_logdir}/2_install.log for details"
    fi
    echo "TensorRT-LLM wheel installation completed successfully"
elif [ -d "${trtllm_repo}" ]; then
    # Build and install from repository if no wheel path provided
    echo "Installing TensorRT-LLM from ${trtllm_repo}..."
    TRT_LLM_GIT_COMMIT=$(git -C ${trtllm_repo} rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "TRT_LLM_GIT_COMMIT: ${TRT_LLM_GIT_COMMIT}"

    if [ "${build_wheel}" = "true" ]; then
        echo "Building TensorRT-LLM wheel on one node..."
        build_command="python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --clean"
        if [ -n "${cuda_architectures:-}" ]; then
            build_command="${build_command} --cuda_architectures \"${cuda_architectures}\""
        fi
        if ! srun --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
            bash -c "cd ${trtllm_repo} && ${build_command}" \
            &> ${full_logdir}/2_build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${full_logdir}/2_build.log for details"
        fi
        echo "TensorRT-LLM build completed successfully"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && pip install -e ." \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${full_logdir}/2_install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
fi

# Get node lists and replace the placeholder with the actual node names
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
all_nodes_str=$(IFS=','; echo "${all_nodes[*]}")
echo "all_nodes_str: ${all_nodes_str}"

start_worker_cmds_file=${full_logdir}/start_worker_cmds.txt
IFS=',' read -r -a node_array <<< "$all_nodes_str"
for i in "${!node_array[@]}"; do
    current_val="${node_array[$i]}"
    placeholder="<node${i}_placeholder>"

    # Use sed to replace the placeholder with the value in-place
    sed -i "s|$placeholder|$current_val|g" "${start_worker_cmds_file}"
    echo "Replaced $placeholder with $current_val"
done

# start the workers (skip ctx workers if TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set).
echo "Starting worker commands from ${start_worker_cmds_file}..."
cat ${start_worker_cmds_file} | while read cmd; do
    # Skip ctx worker commands if in gen-only mode
    # CTX appears as argument to start_worker.sh and in log filename
    if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ] && [[ "$cmd" == *"start_worker.sh CTX"* ]]; then
        echo "Skipping ctx worker command (TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set): ${cmd}"
        continue
    fi
    echo "Starting worker command: ${cmd}"
    eval "${cmd}"
done

# start the server (in background)
echo "Starting server..."
srun -l --container-name=${container_name} \
    --container-image=${container_image} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${work_dir}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${full_logdir} ${work_dir} "${server_env_var}" \
    &> ${full_logdir}/4_output_server.log &

# Wait for server to be ready (runs synchronously)
echo "Waiting for server to be ready..."
if ! srun -l --container-name=${container_name} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${work_dir}/wait_server.sh ${full_logdir} \
    &> ${full_logdir}/5_wait_server.log; then
    cleanup_on_failure "Server failed to become ready. Check ${full_logdir}/5_wait_server.log for details"
fi
echo "Server is ready!"

# Start benchmarking
echo "Starting benchmark..."
if [ "${use_nv_sa_benchmark}" = "true" ]; then
    echo "Using NVIDIA SA benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark_nv_sa.sh \
            "${model_path}" "${isl}" "${osl}" "${benchmark_ratio}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${full_logdir}/" \
            &> ${full_logdir}/6_bench.log; then
        cleanup_on_failure "NVIDIA SA benchmark failed. Check ${full_logdir}/6_bench.log for details"
    fi
else
    echo "Using default benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark.sh \
            "${model_path}" "${dataset_file}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${full_logdir}/" \
            &> ${full_logdir}/6_bench.log; then
        cleanup_on_failure "Benchmark failed. Check ${full_logdir}/6_bench.log for details"
    fi
fi
echo "Benchmark completed successfully"

# Run accuracy evaluation if enabled
if [ "${enable_accuracy_test}" = "true" ]; then
    echo "Starting accuracy evaluation..."
    if ! srun -l --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash ${work_dir}/accuracy_eval.sh \
        "${full_logdir}" "${accuracy_model}" "${accuracy_tasks}" "${model_path}" \
        "${model_args_extra}" "${full_logdir}/accuracy_eval" \
        &> ${full_logdir}/7_accuracy_eval.log; then
        cleanup_on_failure "Accuracy evaluation failed. Check ${full_logdir}/7_accuracy_eval.log for details"
    fi
    echo "Accuracy evaluation completed successfully"
fi

echo "Total runtime: $SECONDS seconds"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
