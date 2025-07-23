#! /bin/bash

config_file=$1
concurrency=$2
enable_pdl=$3
ctx_gpus=$4
work_dir=$5
unset UCX_TLS
echo "config_file: ${config_file}, concurrency: ${concurrency}, enable_pdl: ${enable_pdl}, ctx_gpus: ${ctx_gpus}, work_dir: ${work_dir}"

export TLLM_LOG_LEVEL=INFO
export TRTLLM_USE_UCX_KVCACHE=1
export TLLM_BENCHMARK_REQ_QUEUES_SIZE=${concurrency}
export TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1
export TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER=1

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

#check if work_dir is provided
if [ -z "${work_dir}" ]; then
    echo "nsys is not enabled, start normal flow"
    trtllm-serve disaggregated_mpi_worker -c ${config_file}
else
    nsys_prefix=""
    nsys_file=${work_dir}/nsys_worker_proc_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    if [ "${SLURM_PROCID}" -ge "${ctx_gpus}" ]; then
        export TLLM_PROFILE_START_STOP=200-250
        nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
        echo "nsys_prefix: ${nsys_prefix}"
    else
        echo "nsys is not enabled on ctx_gpus"
    fi
    ${nsys_prefix} trtllm-serve disaggregated_mpi_worker -c ${config_file}
fi
