#! /bin/bash
set -u
set -e
set -x

role=$1
instance_id=$2
model_path=$3
port=$4
enable_pdl=$5
work_dir=$6
enable_nsys=${7:-false}

unset UCX_TLS
echo "enable_pdl: ${enable_pdl}, work_dir: ${work_dir}"
echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"

export TLLM_LOG_LEVEL=INFO

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

numa_bind_cmd="numactl -m 0,1"
echo "numactl -m 0,1 - Only allocate memory from nodes on GB200"

if [ "${role}" = "CTX" ]; then
    config_file=${work_dir}/ctx_config.yaml
elif [ "${role}" = "GEN" ]; then
    config_file=${work_dir}/gen_config.yaml
else
    echo "Invalid role: ${role}"
    exit 1
fi
echo "config_file: ${config_file}"

# save the hostname to a file

# if SLURM_NODEID is 0
if [ "${SLURM_NODEID}" = "0" ]; then
    mkdir -p ${work_dir}/hostnames/
    echo $(hostname) > ${work_dir}/hostnames/${role}_${instance_id}.txt
    echo "hostname saved to ${work_dir}/hostnames/${role}_${instance_id}.txt"
fi

#check if nsys is enabled
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
    # Create log directory if it doesn't exist
    mkdir -p ${work_dir}
    # Start the worker and redirect output to the log file
    trtllm-llmapi-launch ${numa_bind_cmd} trtllm-serve ${model_path} --host $(hostname) --port ${port} --extra_llm_api_options ${config_file} 2>&1 | tee -a ${work_dir}/output_workers.log
else
    nsys_prefix=""
    nsys_file=${work_dir}/nsys_worker_proc_${instance_id}_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    if [ "${role}" = "GEN" ] && [ "$SLURM_PROCID" = "0" ]; then
        export TLLM_PROFILE_START_STOP=200-250
        nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
        echo "nsys_prefix: ${nsys_prefix}"
    elif [ "${role}" = "CTX" ]; then
        echo "nsys is not enabled on ctx_gpus"
    fi
    # Create log directory if it doesn't exist
    mkdir -p ${work_dir}
    # Start the worker with nsys profiling and redirect output to the log file
    ${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
        trtllm-serve ${model_path} \
            --host $(hostname) --port ${port} \
            --extra_llm_api_options ${config_file} 2>&1 | tee -a ${work_dir}/output_workers.log
fi
