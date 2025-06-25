#! /bin/bash

config_file=$1
enable_pdl=$2
ctx_gpus=$3
work_dir=$4

export TLLM_LOG_LEVEL=INFO
export TRTLLM_USE_MPI_KVCACHE=1
export TRTLLM_MNNVL_AR_ENABLED=1

if [ "${enable_pdl}" = "true" ]; then
    export TRTLLM_ENABLE_PDL=1
fi

#check if work_dir is provided
if [ -z "${work_dir}" ]; then
    trtllm-serve disaggregated_mpi_worker -c ${config_file}
else
    nsys_prefix=""
    nsys_file=${work_dir}/nsys_worker_proc_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    if [ ${SLURM_PROCID} -ge ${ctx_gpus} ]; then
        export TLLM_PROFILE_START_STOP=300-400
    else
        export TLLM_PROFILE_START_STOP=25-100
    fi
    nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=all"

    ${nsys_prefix} trtllm-serve disaggregated_mpi_worker -c ${config_file}
fi
