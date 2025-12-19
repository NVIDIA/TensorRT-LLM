#! /bin/bash
set -u
set -e
set -x

role=${1}
instance_id=${2}
model_path=${3}
port=${4}
benchmark_mode=${5}
concurrency=${6}
numa_bind=${7}
log_dir=${8}
enable_nsys=${9}
profile_range=${10}
config_file=${11}
worker_env_var=${12}

unset UCX_TLS
echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"

# Export worker environment variables from config
for env_var in ${worker_env_var}; do
    export "${env_var}"
    echo "Exported: ${env_var}"
done

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200/GB300 NVL72, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

if [ "${benchmark_mode}" = "gen_only" ]; then
    export TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1
    export TLLM_BENCHMARK_REQ_QUEUES_SIZE=${concurrency}
fi

echo "config_file: ${config_file}"

# if SLURM_NODEID is 0, save the hostname to a file
if [ "${SLURM_NODEID}" = "0" ]; then
    mkdir -p ${log_dir}/hostnames/
    echo $(hostname):${port} > ${log_dir}/hostnames/${role}_${instance_id}.txt
    echo "hostname:port saved to ${log_dir}/hostnames/${role}_${instance_id}.txt"
fi

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${role}_${instance_id}_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    export NSYS_MPI_STORE_TEAMS_PER_RANK=1
    export TLLM_PROFILE_START_STOP=${profile_range}
    echo "nsys is enabled on ${role} GPUs, TLLM_PROFILE_START_STOP=${profile_range}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
    trtllm-serve ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file}
