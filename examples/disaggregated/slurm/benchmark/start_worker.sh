#! /bin/bash
set -u
set -e
set -x

role=${1}
instance_id=${2}
model_path=${3}
port=${4}
numa_bind=${5}
log_dir=${6}
enable_nsys=${7}
config_file=${8}
cuda_devices=${9}
# CUDA_VISIBLE_DEVICES selection:
#   - Default packing (no gpu_map file): each node is dedicated to one
#     worker, so every rank on the node is given the node's full GPU list
#     (passed as ${9} by submit.py) and binds to its own device via
#     mapping.local_rank (= rank % gpus_per_node). Exposing the whole node
#     is required for intra-node TP custom all-reduce (attention_dp=false /
#     TEP): its cudaDeviceCanAccessPeer() topology check must be able to see
#     the peer GPUs. Pinning a single GPU per rank only works for DEP
#     (attention_dp=true), which has no intra-node TP all-reduce.
#   - Compact packing (gpu_map file emitted by submit.py): two workers may
#     share a node and would both see LOCALID=0, so look up the per-worker
#     gpu_map "<rank> <host> <local_gpu_id>" by SLURM_PROCID. srun
#     --distribution=arbitrary assigns PROCID in hostfile order, so it
#     indexes directly into the map.
gpu_map_file="${log_dir}/gpu_map_${role}_${instance_id}.txt"
if [ -f "${gpu_map_file}" ]; then
    gpu_id=$(awk -v p="${SLURM_PROCID}" '$1==p {print $3; exit}' "${gpu_map_file}")
    if [ -z "${gpu_id}" ]; then
        echo "ERROR: no GPU mapping for SLURM_PROCID=${SLURM_PROCID} in ${gpu_map_file}" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=${gpu_id}
else
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

# Clear UCX_TLS for specific clusters
unset UCX_TLS

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200/GB300 NVL72, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

echo "config_file: ${config_file}"

# The mooncake-store pool is described in the worker config and provisioned by
# trtllm-serve during bringup. Anchoring its run directory here keeps the
# master's log, the rendered client config and the published address in the
# job's log directory rather than in a temporary directory that shutdown
# removes, and it is how the ranks srun started, which never inherited the
# leader's environment, find that client config. An inherited
# MOONCAKE_CONFIG_PATH still wins, so an externally managed pool stays reachable.
export TRTLLM_MOONCAKE_RUN_DIR="${log_dir}"

# The generation servers wait for a master the context server starts. Both are
# launched together and the master comes up before its model loads, but the wait
# spans container start on another node, so it is given far more than the 60s
# default. Too short fails the job; too long costs nothing when the master is
# already there.
export TRTLLM_MOONCAKE_MASTER_TIMEOUT="${TRTLLM_MOONCAKE_MASTER_TIMEOUT:-900}"

# MiniMax-M3's MSA sparse attention JIT-compiles its FMHA kernels on first use,
# from inside the attention forward pass. One TP rank runs ninja while the
# others block on a file lock, so an uncached variant stalls the whole executor
# loop for ~8s, or ~70s when an iteration needs several. The cache defaults to
# ~/.cache, which is thrown away because the container is started with
# --no-container-mount-home, so every job would pay the compiles again during
# serving. Anchoring it next to this script puts it on the mounted filesystem
# at a path identical across jobs, so only the first run compiles.
if [ -z "${MINFER_FMHA_CACHE_DIR:-}" ]; then
    export MINFER_FMHA_CACHE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.cache/minfer/fmha_sm100"
    mkdir -p "${MINFER_FMHA_CACHE_DIR}"
    echo "MINFER_FMHA_CACHE_DIR: ${MINFER_FMHA_CACHE_DIR}"
fi

# Per-transfer KV timings (size, queue/transfer latency, throughput) as CSV next
# to the worker logs. These separate slow prefill from a slow prefill-to-decode
# handoff, which the aggregate benchmark numbers cannot. An explicit setting
# wins, and KV_TRANSFER_PERF_LOG=false turns it off.
if [ "${KV_TRANSFER_PERF_LOG:-true}" = "true" ] \
    && [ -z "${TLLM_KV_TRANSFER_PERF_LOG_FILE:-}" ]; then
    export TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO=1
    export TLLM_KV_TRANSFER_PERF_LOG_FILE="${log_dir}/kv_transfer_perf"
    echo "TLLM_KV_TRANSFER_PERF_LOG_FILE: ${TLLM_KV_TRANSFER_PERF_LOG_FILE}"
fi

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${role}_${instance_id}_${SLURM_PROCID}
    echo "nsys is enabled on ${role} GPUs, TLLM_PROFILE_START_STOP=${TLLM_PROFILE_START_STOP}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
    trtllm-serve ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file}
