#!/bin/bash
set -euo pipefail

# Parse arguments
# Hardware configuration
gpus_per_node=${1}
numa_bind=${2}
ctx_nodes=${3}  # Number of nodes needed for ctx workers
gen_nodes=${4}  # Number of nodes needed for gen workers

# Worker configuration
num_ctx_servers=${5}
ctx_config_path=${6}
num_gen_servers=${7}
gen_config_path=${8}
eplb_num_slots=${9}
mtp_size=${10}
concurrency_list=${11}

# Sequence and benchmark parameters
isl=${12}
osl=${13}
multi_round=${14}
benchmark_ratio=${15}
streaming=${16}
use_nv_sa_benchmark=${17}
benchmark_mode=${18}
cache_max_tokens=${19}

# Environment and paths
dataset_file=${20}
model_path=${21}
trtllm_repo=${22}
work_dir=${23}
log_dir=${24}
container_mount=${25}
container_image=${26}
build_wheel=${27}

# Profiling
nsys_on=${28}

# Print all parsed arguments
echo "Parsed arguments:"
echo "Hardware Configuration:"
echo "  gpus_per_node: ${gpus_per_node}"
echo "  numa_bind: ${numa_bind}"
echo
echo "Worker Configuration:"
echo "  num_ctx_servers: ${num_ctx_servers}"
echo "  ctx_config_path: ${ctx_config_path}"
echo "  num_gen_servers: ${num_gen_servers}"
echo "  gen_config_path: ${gen_config_path}"
echo "  eplb_num_slots: ${eplb_num_slots}"
echo "  mtp_size: ${mtp_size}"
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
echo "  work_dir: ${work_dir}"
echo "  nsys_on: ${nsys_on}"

container_name="disaggr-test"

# Log directory is now passed directly
echo "Log directory: ${log_dir}"

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
        echo "Container up." &> ${log_dir}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${log_dir}/container_launch.log"
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
            &> ${log_dir}/build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${log_dir}/build.log for details"
        fi
        echo "TensorRT-LLM build completed successfully"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && pip install -e ." \
        &> ${log_dir}/install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${log_dir}/install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
fi

# Get enable_pdl from gen config
enable_pdl=$(python3 -c "import yaml; f=open('${gen_config_path}'); c=yaml.safe_load(f); print(str(not c['enable_attention_dp']).lower())")

# Get node lists
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
total_nodes_num=${#all_nodes[@]}
echo "all_nodes: ${all_nodes[@]}, total_nodes_num: ${total_nodes_num}"

# Split nodes between gen and ctx workers
gen_node_list=(${all_nodes[@]:0:${gen_nodes}})
ctx_node_list=(${all_nodes[@]:${gen_nodes}:${total_nodes_num}})

echo "gen_nodes: ${gen_node_list[@]}, num_nodes: ${gen_nodes}"
echo "ctx_nodes: ${ctx_node_list[@]}, num_nodes: ${ctx_nodes}"

rm -rf ${log_dir}/hostnames
rm -rf ${log_dir}/server_config.yaml

# start the gen workers
echo "Starting gen workers..."
for i in $(seq 0 $((num_gen_servers - 1))); do
    srun -l -N ${gen_nodes} \
        --ntasks=$((gen_nodes * gpus_per_node)) \
        --ntasks-per-node=${gpus_per_node} \
        --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        bash ${work_dir}/start_worker.sh \
        "GEN" ${i} ${model_path} "8336" "${benchmark_mode}" "${concurrency_list}" "${enable_pdl}" "${numa_bind}" "${log_dir}" "${nsys_on}" "${gen_config_path}" \
        &> ${log_dir}/output_gen_${i}.log &
done

# start the ctx workers
echo "Starting ctx workers..."
for i in $(seq 0 $((num_ctx_servers - 1))); do
    srun -l -N ${ctx_nodes} \
        --ntasks=$((ctx_nodes * gpus_per_node)) \
        --ntasks-per-node=${gpus_per_node} \
        --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        bash ${work_dir}/start_worker.sh \
        "CTX" ${i} ${model_path} "8336" "${benchmark_mode}" "${concurrency_list}" "${enable_pdl}" "${numa_bind}" "${log_dir}" "${nsys_on}" "${ctx_config_path}" \
        &> ${log_dir}/output_ctx_${i}.log &
done

# start the server
echo "Starting server..."
srun -l --container-name=${container_name} \
    --container-image=${container_image} \
    --container-mounts=${container_mount} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${work_dir}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${log_dir} ${work_dir} \
    &> ${log_dir}/output_server.log &

# Start benchmarking
echo "Starting benchmark..."
if [ "${use_nv_sa_benchmark}" = "true" ]; then
    echo "Using NVIDIA SA benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark_nv_sa.sh \
            "${model_path}" "${isl}" "${osl}" "${benchmark_ratio}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${log_dir}/" \
            &> ${log_dir}/bench.log; then
        cleanup_on_failure "NVIDIA SA benchmark failed. Check ${log_dir}/bench.log for details"
    fi
else
    echo "Using default benchmark script..."
    if ! srun -l --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 -n 1 \
            bash ${work_dir}/run_benchmark.sh \
            "${model_path}" "${dataset_file}" "${multi_round}" "${num_gen_servers}" "${concurrency_list}" "${streaming}" "${log_dir}/" \
            &> ${log_dir}/bench.log; then
        cleanup_on_failure "Benchmark failed. Check ${log_dir}/bench.log for details"
    fi
fi
echo "Benchmark completed successfully"

# try to kill the server and workers
scancel ${SLURM_JOB_ID}
