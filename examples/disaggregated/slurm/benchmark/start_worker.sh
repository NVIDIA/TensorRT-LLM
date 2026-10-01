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

# Container runtimes (pyxis/enroot) reset image-defined variables like PATH
# at container start, so values passed via srun --export are lost for them.
# Allow the launcher config to prepend entries from inside the container.
# srun --export keeps any quotes in the exported values literal; strip them.
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND:-}"
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND#\'}"; TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND%\'}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND:-}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND#\'}"; TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND%\'}"
if [ -n "${TRTLLM_PATH_PREPEND:-}" ]; then
    export PATH="${TRTLLM_PATH_PREPEND}:${PATH}"
fi
if [ -n "${TRTLLM_PYTHONPATH_PREPEND:-}" ]; then
    export PYTHONPATH="${TRTLLM_PYTHONPATH_PREPEND}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Clear UCX_TLS for specific clusters. Some clusters instead need an
# explicit transport list (e.g. NVL72 nodes whose verbs transports cannot
# initialize): set TRTLLM_WORKER_UCX_TLS in worker_env_var to re-pin
# UCX_TLS here, after the container-provided value is cleared.
if [ -n "${TRTLLM_WORKER_UCX_TLS:-}" ]; then
    export UCX_TLS="${TRTLLM_WORKER_UCX_TLS}"
else
    unset UCX_TLS
fi

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

# The mooncake-store pool is named by the worker config, including
# mooncake_store.run_dir, which has to point inside the job's log directory
# because the ranks srun started never inherit the leader's environment and
# read the rendered client config back from there.
#
# run_dir also has to differ per server: two sharing one render a single client
# config between them, leaving a server transferring over another node's HCAs
# or lending under another's role. The worker config is shared by every server
# of a role and can only carry __LOG_DIR__, which is the whole job's, so the
# per-server part is substituted here, where the role and the instance are
# known. Each rank renders its own copy of the config, from the same value, so
# that they do not race over one file. See the Mooncake section of README.md.
if grep -q '__MOONCAKE_RUN_DIR__' "${config_file}"; then
    role_lower=$(echo "${role}" | tr '[:upper:]' '[:lower:]')
    mooncake_run_dir="${log_dir}/mooncake_${role_lower}_${instance_id}"
    rendered_config="${log_dir}/config_${role}_${instance_id}_rank${SLURM_PROCID}.yaml"
    sed "s|__MOONCAKE_RUN_DIR__|${mooncake_run_dir}|g" \
        "${config_file}" > "${rendered_config}"
    config_file="${rendered_config}"
    echo "mooncake_store.run_dir: ${mooncake_run_dir} (config ${config_file})"
fi

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

# In-place (.pth-style) TRT-LLM installs may lack the trtllm-serve console
# script; fall back to the module entry point in that case.
trtllm_serve_cmd="trtllm-serve"
if ! command -v trtllm-serve >/dev/null 2>&1; then
    trtllm_serve_cmd="python3 -m tensorrt_llm.commands.serve"
fi

${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
    ${trtllm_serve_cmd} ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file}
