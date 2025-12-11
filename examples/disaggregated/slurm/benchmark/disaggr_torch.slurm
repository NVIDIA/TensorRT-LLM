#!/bin/bash
set -euo pipefail

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Hardware configuration
        --gpus-per-node) gpus_per_node="$2"; shift 2 ;;
        --numa-bind) numa_bind="$2"; shift 2 ;;
        --ctx-nodes) ctx_nodes="$2"; shift 2 ;;
        --gen-nodes) gen_nodes="$2"; shift 2 ;;
        --ctx-world-size) ctx_world_size="$2"; shift 2 ;;
        --gen-world-size) gen_world_size="$2"; shift 2 ;;
        # Worker configuration
        --num-ctx-servers) num_ctx_servers="$2"; shift 2 ;;
        --ctx-config-path) ctx_config_path="$2"; shift 2 ;;
        --num-gen-servers) num_gen_servers="$2"; shift 2 ;;
        --gen-config-path) gen_config_path="$2"; shift 2 ;;
        --concurrency-list) concurrency_list="$2"; shift 2 ;;
        # Sequence and benchmark parameters
        --isl) isl="$2"; shift 2 ;;
        --osl) osl="$2"; shift 2 ;;
        --multi-round) multi_round="$2"; shift 2 ;;
        --benchmark-ratio) benchmark_ratio="$2"; shift 2 ;;
        --streaming) streaming="$2"; shift 2 ;;
        --use-nv-sa-benchmark) use_nv_sa_benchmark="$2"; shift 2 ;;
        --benchmark-mode) benchmark_mode="$2"; shift 2 ;;
        --cache-max-tokens) cache_max_tokens="$2"; shift 2 ;;
        # Environment and paths
        --dataset-file) dataset_file="$2"; shift 2 ;;
        --model-path) model_path="$2"; shift 2 ;;
        --trtllm-repo) trtllm_repo="$2"; shift 2 ;;
        --work-dir) work_dir="$2"; shift 2 ;;
        --full-logdir) full_logdir="$2"; shift 2 ;;
        --container-mount) container_mount="$2"; shift 2 ;;
        --container-image) container_image="$2"; shift 2 ;;
        --build-wheel) build_wheel="$2"; shift 2 ;;
        --trtllm-wheel-path) trtllm_wheel_path="$2"; shift 2 ;;
        # Profiling
        --nsys-on) nsys_on="$2"; shift 2 ;;
        --ctx-profile-range) ctx_profile_range="$2"; shift 2 ;;
        --gen-profile-range) gen_profile_range="$2"; shift 2 ;;
        # Accuracy evaluation
        --enable-accuracy-test) enable_accuracy_test="$2"; shift 2 ;;
        --accuracy-model) accuracy_model="$2"; shift 2 ;;
        --accuracy-tasks) accuracy_tasks="$2"; shift 2 ;;
        --model-args-extra) model_args_extra="$2"; shift 2 ;;
        # Worker environment variables
        --worker-env-var) worker_env_var="$2"; shift 2 ;;
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
echo "Hardware Configuration:"
echo "  gpus_per_node: ${gpus_per_node}"
echo "  numa_bind: ${numa_bind}"
echo "  ctx_nodes: ${ctx_nodes}"
echo "  gen_nodes: ${gen_nodes}"
echo "  ctx_world_size: ${ctx_world_size}"
echo "  gen_world_size: ${gen_world_size}"
echo
echo "Worker Configuration:"
echo "  num_ctx_servers: ${num_ctx_servers}"
echo "  ctx_config_path: ${ctx_config_path}"
echo "  num_gen_servers: ${num_gen_servers}"
echo "  gen_config_path: ${gen_config_path}"
echo "  concurrency_list: ${concurrency_list}"
echo
echo "Benchmark Configuration:"
echo "  use_nv_sa_benchmark: ${use_nv_sa_benchmark}"
echo "  isl: ${isl}"
echo "  osl: ${osl}"
echo "  multi_round: ${multi_round}"
echo "  benchmark_ratio: ${benchmark_ratio}"
echo "  streaming: ${streaming}"
echo "  cache_max_tokens: ${cache_max_tokens}"
echo "  benchmark_mode: ${benchmark_mode}"
echo
echo "Environment Configuration:"
echo "  dataset_file: ${dataset_file}"
echo "  container_mount: ${container_mount}"
echo "  container_image: ${container_image}"
echo "  model_path: ${model_path}"
echo "  trtllm_repo: ${trtllm_repo}"
echo "  build_wheel: ${build_wheel}"
echo "  trtllm_wheel_path: ${trtllm_wheel_path}"
echo "  work_dir: ${work_dir}"
echo "  nsys_on: ${nsys_on}"
echo "  ctx_profile_range: ${ctx_profile_range}"
echo "  gen_profile_range: ${gen_profile_range}"
echo
echo "Accuracy Configuration:"
echo "  enable_accuracy_test: ${enable_accuracy_test}"
echo "  accuracy_model: ${accuracy_model}"
echo "  accuracy_tasks: ${accuracy_tasks}"
echo "  model_args_extra: ${model_args_extra}"
echo
echo "Worker Environment Variables:"
echo "  worker_env_var: ${worker_env_var}"
echo
echo "Server Environment Variables:"
echo "  server_env_var: ${server_env_var}"

container_name="disaggr-test"

echo "Log directory: ${full_logdir}"

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

# Get node lists
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
all_nodes_str=$(IFS=','; echo "${all_nodes[*]}")

echo "Assigning nodes..."
srun -l --container-name=${container_name} \
    --container-image=${container_image} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    python3 ${work_dir}/assign_nodes.py \
        --all-nodes ${all_nodes_str} \
        --gpus-per-node ${gpus_per_node} \
        --num-gen-servers ${num_gen_servers} \
        --num-ctx-servers ${num_ctx_servers} \
        --gen-world-size ${gen_world_size} \
        --ctx-world-size ${ctx_world_size} \
        --log-dir ${full_logdir} \
        &> ${full_logdir}/3_assign_nodes.log

echo "Generating start worker commands..."
srun -l --container-name=${container_name} \
    --container-image=${container_image} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    python3 ${work_dir}/gen_start_worker_cmds.py \
        --allocations ${full_logdir}/allocations.json \
        --gen-world-size ${gen_world_size} \
        --ctx-world-size ${ctx_world_size} \
        --gpus-per-node ${gpus_per_node} \
        --container-image ${container_image} \
        --container-name ${container_name} \
        --container-mount ${container_mount} \
        --work-dir ${work_dir} \
        --model-path ${model_path} \
        --benchmark-mode ${benchmark_mode} \
        --concurrency-list ${concurrency_list} \
        --numa-bind ${numa_bind} \
        --log-dir ${full_logdir} \
        --nsys-on ${nsys_on} \
        --ctx-profile-range ${ctx_profile_range} \
        --gen-profile-range ${gen_profile_range} \
        --ctx-config-path ${ctx_config_path} \
        --gen-config-path ${gen_config_path} \
        --worker-env-var "${worker_env_var}" \
        &> ${full_logdir}/4_gen_start_worker_cmds.log

start_worker_cmds_file=${full_logdir}/start_worker_cmds.txt
echo "Starting worker commands from ${start_worker_cmds_file}..."
cat ${start_worker_cmds_file} | while read cmd; do
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
    &> ${full_logdir}/5_output_server.log &

# Wait for server to be ready (runs synchronously)
echo "Waiting for server to be ready..."
if ! srun -l --container-name=${container_name} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${work_dir}/wait_server.sh ${full_logdir} \
    &> ${full_logdir}/6_wait_server.log; then
    cleanup_on_failure "Server failed to become ready. Check ${full_logdir}/6_wait_server.log for details"
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
            &> ${full_logdir}/7_bench.log; then
        cleanup_on_failure "NVIDIA SA benchmark failed. Check ${full_logdir}/7_bench.log for details"
    fi
else
    echo "Using default benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark.sh \
            "${model_path}" "${dataset_file}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${full_logdir}/" \
            &> ${full_logdir}/8_bench.log; then
        cleanup_on_failure "Benchmark failed. Check ${full_logdir}/8_bench.log for details"
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
        &> ${full_logdir}/9_accuracy_eval.log; then
        cleanup_on_failure "Accuracy evaluation failed. Check ${full_logdir}/9_accuracy_eval.log for details"
    fi
    echo "Accuracy evaluation completed successfully"
fi

echo "Total runtime: $SECONDS seconds"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
