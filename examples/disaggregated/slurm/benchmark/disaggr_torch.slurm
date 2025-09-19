#!/bin/bash
set -euo pipefail

# Validate required environment variables
check_env_var() {
    local var_name=$1
    local var_value=${!var_name}
    if [[ -z "${var_value}" ]]; then
        echo "Error: ${var_name} environment variable is not set. Please set it in your submit script." >&2
        exit 1
    fi
}

# Hardware Configuration
check_env_var "GPUS_PER_NODE"

# Benchmark Configuration
check_env_var "USE_NV_SA_BENCHMARK"
check_env_var "ISL"
check_env_var "OSL"
check_env_var "MULTI_ROUND"
check_env_var "BENCHMARK_RATIO"
check_env_var "STREAMING"
check_env_var "CACHE_MAX_TOKENS"
check_env_var "DATASET_FILE"

# Environment Configuration
check_env_var "MOUNT_DIR"
check_env_var "CONTAINER_IMAGE"

# Model Configuration
check_env_var "MODEL_PATH"
check_env_var "TRTLLM_REPO"
check_env_var "BUILD_WHEEL"

# Workspace Configuration
check_env_var "WORK_DIR"

# Profiling Configuration
check_env_var "NSYS_ON"

container_name="disaggr-test"

# Parse arguments
num_ctx_servers=$1
ctx_tp_size=$2
ctx_batch_size=$3
ctx_max_num_tokens=$4
ctx_enable_attention_dp=$5
ctx_gpu_frac=$6
num_gen_servers=$7
gen_tp_size=$8
gen_batch_size=$9
gen_max_num_tokens=${10}
gen_enable_attention_dp=${11}
gen_gpu_memory_fraction=${12}
eplb_num_slots=${13}
mtp_size=${14}
concurrency_list=${15}

# Setup logging directory
log_base="${WORK_DIR}/${ISL}-${OSL}/" # path to the log directory
full_logdir=${log_base}/ctx${num_ctx_servers}_gen${num_gen_servers}_dep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
    echo "enable_pdl: ${enable_pdl}"
    full_logdir=${log_base}/ctx${num_ctx_servers}_gen${num_gen_servers}_tep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
fi
mkdir -p "${full_logdir}"

echo "concurrency_list: ${concurrency_list}"
echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

# Start container
echo "Starting container..."
if ! srun -l --container-image=${CONTAINER_IMAGE} \
        --container-name=${container_name} \
        --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
        --mpi=pmix \
        echo "Container up." &> ${full_logdir}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${full_logdir}/container_launch.log"
fi

# Build TensorRT-LLM if needed
if [ -d "${TRTLLM_REPO}" ]; then
    echo "Installing TensorRT-LLM from ${TRTLLM_REPO}..."

    if [ "${BUILD_WHEEL}" = "true" ]; then
        echo "Building TensorRT-LLM wheel on one node..."
        build_command="python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures '100-real' --binding_type pybind --clean"
        if ! srun --container-name=${container_name} \
            --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
            --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
            bash -c "cd ${TRTLLM_REPO} && ${build_command}" \
            &> ${full_logdir}/build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${full_logdir}/build.log for details"
        fi
        echo "TensorRT-LLM build completed successfully"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${TRTLLM_REPO} && pip install -e ." \
        &> ${full_logdir}/install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${full_logdir}/install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
fi

# Generate worker config
echo "Generating worker config..."
# Sequence Length Configuration
seq_offset=203            # Offset added to sequence lengths
ctx_max_seq_len=$((ISL + seq_offset))
gen_max_seq_len=$((ISL + OSL + seq_offset))
if ! srun -l -N 1 -n 1 \
        --container-name=${container_name} \
        --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
        --mpi=pmix --overlap \
        python3 ${WORK_DIR}/gen_worker_config.py \
                --work_dir ${full_logdir} \
                --ctx_tp_size ${ctx_tp_size} \
                --ctx_pp_size 1 \
                --ctx_batch_size ${ctx_batch_size} \
                --ctx_max_num_tokens ${ctx_max_num_tokens} \
                --ctx_max_seq_len ${ctx_max_seq_len} \
                --ctx_free_gpu_memory_fraction ${ctx_gpu_frac} \
                --gen_tp_size ${gen_tp_size} \
                --gen_pp_size 1 \
                --gen_batch_size ${gen_batch_size} \
                --gen_max_num_tokens ${gen_max_num_tokens} \
                --gen_max_seq_len ${gen_max_seq_len} \
                --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
                --eplb_num_slots ${eplb_num_slots} \
                --mtp_size ${mtp_size} \
                --cache_transceiver_max_num_tokens ${CACHE_MAX_TOKENS} \
                $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
                $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
                &> ${full_logdir}/gen_worker_config.log; then
    cleanup_on_failure "Failed to generate worker config. Check ${full_logdir}/gen_worker_config.log for details"
fi
echo "Worker config generated successfully"

# Calculate nodes needed
ctx_nodes_num=$(((ctx_tp_size + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
gen_nodes_num=$(((gen_tp_size + GPUS_PER_NODE - 1) / GPUS_PER_NODE))

all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
total_nodes_num=${#all_nodes[@]}
echo "all_nodes: ${all_nodes[@]}, total_nodes_num: ${total_nodes_num}"

# get the node list for the gen workers
total_gen_nodes_num=$((gen_nodes_num * num_gen_servers))
gen_nodes=(${all_nodes[@]:0:${total_gen_nodes_num}})
echo "gen_nodes: ${gen_nodes[@]}, total_gen_nodes_num: ${total_gen_nodes_num}"

# get the node list for the ctx workers
total_ctx_nodes_num=$((ctx_nodes_num * num_ctx_servers))
ctx_nodes=(${all_nodes[@]:${total_gen_nodes_num}:${total_nodes_num}})
echo "ctx_nodes: ${ctx_nodes[@]}, total_ctx_nodes_num: ${total_ctx_nodes_num}"

rm -rf ${full_logdir}/hostnames
rm -rf ${full_logdir}/server_config.yaml

# start the gen workers
echo "Starting gen workers..."
for i in $(seq 0 $((num_gen_servers - 1))); do
    srun -l -N ${gen_nodes_num} \
        --ntasks=${gen_tp_size} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --container-image=${CONTAINER_IMAGE} \
        --container-name=${container_name} \
        --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
        --mpi=pmix \
        bash ${WORK_DIR}/start_worker.sh \
        "GEN" ${i} ${MODEL_PATH} "8336" "${enable_pdl}" "${full_logdir}" ${NSYS_ON} \
        &> ${full_logdir}/output_gen_${i}.log &
done

# start the ctx workers
echo "Starting ctx workers..."
for i in $(seq 0 $((num_ctx_servers - 1))); do
    srun -l -N ${ctx_nodes_num} \
        --ntasks=${ctx_tp_size} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --container-image=${CONTAINER_IMAGE} \
        --container-name=${container_name} \
        --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
        --mpi=pmix \
        bash ${WORK_DIR}/start_worker.sh \
        "CTX" ${i} ${MODEL_PATH} "8336" "${enable_pdl}" "${full_logdir}" ${NSYS_ON} \
        &> ${full_logdir}/output_ctx_${i}.log &
done

# start the server
echo "Starting server..."
srun -l --container-name=${container_name} \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${WORK_DIR}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${full_logdir} ${WORK_DIR} \
        &> ${full_logdir}/output_server.log &

# Start benchmarking
echo "Starting benchmark..."
if [ "${USE_NV_SA_BENCHMARK}" = "true" ]; then
    echo "Using NVIDIA SA benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${WORK_DIR}/run_benchmark_nv_sa.sh \
            "${MODEL_PATH}" "${ISL}" "${OSL}" "${BENCHMARK_RATIO}" "${MULTI_ROUND}" "${num_gen_servers}" "${concurrency_list}" "${STREAMING}" "${full_logdir}/" \
            &> ${full_logdir}/bench.log; then
        cleanup_on_failure "NVIDIA SA benchmark failed. Check ${full_logdir}/bench.log for details"
    fi
else
    echo "Using default benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${MOUNT_DIR}:${MOUNT_DIR} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${WORK_DIR}/run_benchmark.sh \
            "${MODEL_PATH}" "${DATASET_FILE}" "${MULTI_ROUND}" "${num_gen_servers}" "${concurrency_list}" "${STREAMING}" "${full_logdir}/" \
            &> ${full_logdir}/bench.log; then
        cleanup_on_failure "Benchmark failed. Check ${full_logdir}/bench.log for details"
    fi
fi
echo "Benchmark completed successfully"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
