#!/bin/bash
set -euo pipefail

# Parse arguments
num_ctx_servers=${1}
ctx_tp_size=${2}
ctx_pp_size=${3}
ctx_batch_size=${4}
ctx_max_num_tokens=${5}
ctx_enable_attention_dp=${6}
ctx_gpu_frac=${7}
num_gen_servers=${8}
gen_tp_size=${9}
gen_pp_size=${10}
gen_batch_size=${11}
gen_max_num_tokens=${12}
gen_enable_attention_dp=${13}
gen_gpu_frac=${14}
eplb_num_slots=${15}
mtp_size=${16}
concurrency_list=${17}
gpus_per_node=${18}
use_nv_sa_benchmark=${19}
isl=${20}
osl=${21}
multi_round=${22}
benchmark_ratio=${23}
streaming=${24}
cache_max_tokens=${25}
dataset_file=${26}
container_mount=${27}
container_image=${28}
model_path=${29}
trtllm_repo=${30}
build_wheel=${31}
work_dir=${32}
nsys_on=${33}
seq_offset=${34}  # Offset added to sequence lengths
numa_bind=${35}    # Whether to enable NUMA binding
benchmark_mode=${36}  # Benchmark mode: e2e or gen_only

# Print all parsed arguments
echo "Parsed arguments:"
echo "Context Server Configuration:"
echo "  num_ctx_servers: ${num_ctx_servers}"
echo "  ctx_tp_size: ${ctx_tp_size}"
echo "  ctx_pp_size: ${ctx_pp_size}"
echo "  ctx_batch_size: ${ctx_batch_size}"
echo "  ctx_max_num_tokens: ${ctx_max_num_tokens}"
echo "  ctx_enable_attention_dp: ${ctx_enable_attention_dp}"
echo "  ctx_gpu_frac: ${ctx_gpu_frac}"
echo
echo "Generation Server Configuration:"
echo "  num_gen_servers: ${num_gen_servers}"
echo "  gen_tp_size: ${gen_tp_size}"
echo "  gen_pp_size: ${gen_pp_size}"
echo "  gen_batch_size: ${gen_batch_size}"
echo "  gen_max_num_tokens: ${gen_max_num_tokens}"
echo "  gen_enable_attention_dp: ${gen_enable_attention_dp}"
echo "  gen_gpu_frac: ${gen_gpu_frac}"
echo "  eplb_num_slots: ${eplb_num_slots}"
echo "  mtp_size: ${mtp_size}"
echo "  concurrency_list: ${concurrency_list}"
echo
echo "Hardware Configuration:"
echo "  gpus_per_node: ${gpus_per_node}"
echo
echo "Benchmark Configuration:"
echo "  use_nv_sa_benchmark: ${use_nv_sa_benchmark}"
echo "  isl: ${isl}"
echo "  osl: ${osl}"
echo "  multi_round: ${multi_round}"
echo "  benchmark_ratio: ${benchmark_ratio}"
echo "  streaming: ${streaming}"
echo "  cache_max_tokens: ${cache_max_tokens}"
echo "  seq_offset: ${seq_offset}"
echo "  numa_bind: ${numa_bind}"
echo "  benchmark_mode: ${benchmark_mode}"
echo
echo "Environment Configuration:"
echo "  dataset_file: ${dataset_file}"
echo "  container_mount: ${container_mount}"
echo "  container_image: ${container_image}"
echo "  model_path: ${model_path}"
echo "  trtllm_repo: ${trtllm_repo}"
echo "  build_wheel: ${build_wheel}"
echo "  work_dir: ${work_dir}"
echo "  nsys_on: ${nsys_on}"

container_name="disaggr-test"

# Setup logging directory
log_base="${work_dir}/${isl}-${osl}/" # path to the log directory
full_logdir=${log_base}/ctx${num_ctx_servers}_gen${num_gen_servers}_dep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
    echo "enable_pdl: ${enable_pdl}"
    full_logdir=${log_base}/ctx${num_ctx_servers}_gen${num_gen_servers}_tep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
fi
mkdir -p "${full_logdir}"

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

# Start container
echo "Starting container..."
if ! srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        echo "Container up." &> ${full_logdir}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${full_logdir}/container_launch.log"
fi

# Build TensorRT-LLM if needed
if [ -d "${trtllm_repo}" ]; then
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
            &> ${full_logdir}/build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${full_logdir}/build.log for details"
        fi
        echo "TensorRT-LLM build completed successfully"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && pip install -e ." \
        &> ${full_logdir}/install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${full_logdir}/install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
fi

# Generate worker config
echo "Generating worker config..."
# Sequence Length Configuration
ctx_max_seq_len=$((isl + seq_offset))  # Add sequence offset to input sequence length
gen_max_seq_len=$((isl + osl + seq_offset))  # Add sequence offset to total sequence length
if ! srun -l -N 1 -n 1 \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix --overlap \
        python3 ${work_dir}/gen_worker_config.py \
                --work_dir ${full_logdir} \
                --ctx_tp_size ${ctx_tp_size} \
                --ctx_pp_size ${ctx_pp_size} \
                --ctx_batch_size ${ctx_batch_size} \
                --ctx_max_num_tokens ${ctx_max_num_tokens} \
                --ctx_max_seq_len ${ctx_max_seq_len} \
                --ctx_free_gpu_memory_fraction ${ctx_gpu_frac} \
                --gen_tp_size ${gen_tp_size} \
                --gen_pp_size ${gen_pp_size} \
                --gen_batch_size ${gen_batch_size} \
                --gen_max_num_tokens ${gen_max_num_tokens} \
                --gen_max_seq_len ${gen_max_seq_len} \
                --gen_gpu_memory_fraction ${gen_gpu_frac} \
                --eplb_num_slots ${eplb_num_slots} \
                --mtp_size ${mtp_size} \
                --cache_transceiver_max_num_tokens ${cache_max_tokens} \
                $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
                $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
                &> ${full_logdir}/gen_worker_config.log; then
    cleanup_on_failure "Failed to generate worker config. Check ${full_logdir}/gen_worker_config.log for details"
fi
echo "Worker config generated successfully"

# Calculate nodes needed
ctx_nodes_num=$(((ctx_tp_size + gpus_per_node - 1) / gpus_per_node))
gen_nodes_num=$(((gen_tp_size + gpus_per_node - 1) / gpus_per_node))

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
        --ntasks-per-node=${gpus_per_node} \
        --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        bash ${work_dir}/start_worker.sh \
        "GEN" ${i} ${model_path} "8336" "${benchmark_mode}" "${concurrency_list}" "${enable_pdl}" "${numa_bind}" "${full_logdir}" "${nsys_on}" \
        &> ${full_logdir}/output_gen_${i}.log &
done

# start the ctx workers
echo "Starting ctx workers..."
for i in $(seq 0 $((num_ctx_servers - 1))); do
    srun -l -N ${ctx_nodes_num} \
        --ntasks=${ctx_tp_size} \
        --ntasks-per-node=${gpus_per_node} \
        --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        bash ${work_dir}/start_worker.sh \
        "CTX" ${i} ${model_path} "8336" "${benchmark_mode}" "${concurrency_list}" "${enable_pdl}" "${numa_bind}" "${full_logdir}" "${nsys_on}" \
        &> ${full_logdir}/output_ctx_${i}.log &
done

# start the server
echo "Starting server..."
srun -l --container-name=${container_name} \
    --container-image=${container_image} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${work_dir}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${full_logdir} ${work_dir} \
        &> ${full_logdir}/output_server.log &

# Start benchmarking
echo "Starting benchmark..."
if [ "${use_nv_sa_benchmark}" = "true" ]; then
    echo "Using NVIDIA SA benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark_nv_sa.sh \
            "${model_path}" "${isl}" "${osl}" "${benchmark_ratio}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${full_logdir}/" \
            &> ${full_logdir}/bench.log; then
        cleanup_on_failure "NVIDIA SA benchmark failed. Check ${full_logdir}/bench.log for details"
    fi
else
    echo "Using default benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark.sh \
            "${model_path}" "${dataset_file}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${full_logdir}/" \
            &> ${full_logdir}/bench.log; then
        cleanup_on_failure "Benchmark failed. Check ${full_logdir}/bench.log for details"
    fi
fi
echo "Benchmark completed successfully"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
