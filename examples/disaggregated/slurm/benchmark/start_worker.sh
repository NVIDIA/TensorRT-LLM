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
enable_pdl=${7}
numa_bind=${8}
log_dir=${9}
enable_nsys=${10}
config_file=${11}

unset UCX_TLS
echo "enable_pdl: ${enable_pdl}, log_dir: ${log_dir}"
echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"

export TLLM_LOG_LEVEL=INFO
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

if [ "${benchmark_mode}" = "gen_only" ]; then
    export TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1
    export TLLM_BENCHMARK_REQ_QUEUES_SIZE=${concurrency}
fi

echo "config_file: ${config_file}"

# save the hostname to a file

# if SLURM_NODEID is 0
if [ "${SLURM_NODEID}" = "0" ]; then
    mkdir -p ${log_dir}/hostnames/
    echo $(hostname) > ${log_dir}/hostnames/${role}_${instance_id}.txt
    echo "hostname saved to ${log_dir}/hostnames/${role}_${instance_id}.txt"
fi

#check if nsys is enabled
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
    trtllm-llmapi-launch ${numa_bind_cmd} trtllm-serve ${model_path} --host $(hostname) --port ${port} --extra_llm_api_options ${config_file}
else
    nsys_prefix=""
    nsys_file=${log_dir}/nsys_worker_proc_${role}_${instance_id}_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
    if [ "${role}" = "GEN" ]; then
        export TLLM_PROFILE_START_STOP=200-250
        echo "nsys is enabled on gen_gpus"
    elif [ "${role}" = "CTX" ]; then
        export TLLM_PROFILE_START_STOP=10-30
        echo "nsys is enabled on ctx_gpus"
    fi
    ${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
        trtllm-serve ${model_path} \
            --host $(hostname) --port ${port} \
            --extra_llm_api_options ${config_file}
fi
