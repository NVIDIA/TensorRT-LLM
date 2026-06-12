#! /bin/bash
set -u
set -e
set -x

config_file=${1}
numa_bind=${2}
log_dir=${3}
enable_nsys=${4}
ctx_profile_range=${5}
gen_profile_range=${6}
num_ctx_gpus=${7}
ctx_worker_env_var=${8}
gen_worker_env_var=${9}

if [ -z "${UCX_NET_DEVICES:-}" ]; then
    unset UCX_NET_DEVICES
else
    echo "Using UCX_NET_DEVICES: ${UCX_NET_DEVICES}"
fi

if [ -z "${UCX_TLS:-}" ]; then
    unset UCX_TLS
else
    echo "Using UCX_TLS: ${UCX_TLS}"
fi

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname)"

# NOTE: do NOT export CUDA_VISIBLE_DEVICES from this script.
#
# Restricting CUDA visibility to a single GPU (via CUDA driver isolation)
# breaks DWDP's intra-node peer GPU discovery:
#   - VA composite cuMemMap imports of peer GPUs' MNNVL fabric handles
#   - UCX cuda_ipc / cuda_copy intra-node transports for CTX->GEN KV
#   - PyTorch torch.cuda.device_count() peer enumeration
# All of these need the process to *see* peer GPUs on the same node,
# even if it only computes on one of them.
#
# Empirically (R1/T2/T3/T4 vs T5 on dwdp3 dg=4): exporting
# ``CUDA_VISIBLE_DEVICES=<single_gpu>`` blows TTFT std up 3x and drops
# per-CTX-GPU throughput by 15%, with TPOT unchanged.  Letting
# trtllm-serve auto-pick the device from SLURM_LOCALID restores Phase
# D's full perf.
#
# For audit, log which GPU SLURM would have given this rank.  With our
# compact-packing allocate_gpus, the natural mapping is:
#   gpu_id = (gpu_map_mpi_worker.txt[rank][2])   if gpu_map exists
#         == SLURM_LOCALID                       (always, by construction)
# so we log it but don't export.
gpu_map_file="${log_dir}/gpu_map_mpi_worker.txt"
if [ -f "${gpu_map_file}" ]; then
    expected_gpu=$(awk -v p="${SLURM_PROCID}" '$1==p {print $3; exit}' "${gpu_map_file}")
    echo "rank-to-gpu (arbitrary-dist path): SLURM_PROCID=${SLURM_PROCID} LOCALID=${SLURM_LOCALID} expected_gpu=${expected_gpu}"
else
    echo "rank-to-gpu (block-dist path): SLURM_PROCID=${SLURM_PROCID} LOCALID=${SLURM_LOCALID}"
fi

if [ "${SLURM_PROCID}" -lt "${num_ctx_gpus}" ]; then
    worker_role="CTX"
    worker_env_var=${ctx_worker_env_var}
    profile_range=${ctx_profile_range}
else
    worker_role="GEN"
    worker_env_var=${gen_worker_env_var}
    profile_range=${gen_profile_range}
fi

echo "worker_role: ${worker_role}, profile_range: ${profile_range}"

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

echo "config_file: ${config_file}"

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${worker_role}_${SLURM_PROCID}
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_NVTX_DEBUG=1
    export NSYS_MPI_STORE_TEAMS_PER_RANK=1
    export TLLM_PROFILE_START_STOP=${profile_range}
    echo "nsys is enabled on ${worker_role} ranks, TLLM_PROFILE_START_STOP=${profile_range}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

${nsys_prefix} ${numa_bind_cmd} trtllm-serve disaggregated_mpi_worker -c ${config_file}
