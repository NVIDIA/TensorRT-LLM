#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 2-node KV cache transceiver bandwidth / UCX-tuning harness.
# Submitted by submit.py, which passes --partition/--account/--time/--job-name,
# --nodes=2, --ntasks-per-node=<gpus_per_node>, and exports CTT_CONFIG.
#
# For each UCX env-var set (a "sweep") this relaunches two srun steps -- one ctx
# instance on node A, one gen instance on node B, each its own MPI world of size
# N -- because UCX env vars only take effect at process start.

# Fallback batch log location; submit.py overrides this with --output pointing
# at <work_dir>/logs/ctt-%j.log (stderr merged in, no separate --error).
#SBATCH --output=ctt-%j.log

set -u

# submit.py exports CTT_SCRIPT_DIR (absolute path of this harness directory).
HERE="${CTT_SCRIPT_DIR:?CTT_SCRIPT_DIR must be exported by submit.py}"
CONFIG="${CTT_CONFIG:?CTT_CONFIG must be exported by submit.py}"

# Pull derived values out of the config.
eval "$(python3 "${HERE}/report.py" "${CONFIG}" --emit-launch-vars)"
echo "N=${N} IMAGE=${IMAGE} WORK=${WORK} NUM_SWEEPS=${NUM_SWEEPS} PER_SWEEP_TIMEOUT=${PER_SWEEP_TIMEOUT}"

mkdir -p "${WORK}/logs" "${WORK}/csv" "${WORK}/status"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if [ "${#NODES[@]}" -lt 2 ]; then
    echo "ERROR: need 2 nodes, got ${#NODES[@]}" >&2
    exit 1
fi
CTX_NODE="${NODES[0]}"
GEN_NODE="${NODES[1]}"
ZMQ_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
CONTAINER_NAME="ctt_${SLURM_JOB_ID}"
DRIVER="python3 ${HERE}/run_cache_transceiver_test.py"

echo "ctx_node=${CTX_NODE} gen_node=${GEN_NODE} zmq_port=${ZMQ_PORT}"

# ---------------------------------------------------------------------------
# One-time TensorRT-LLM install (mirrors benchmark/disaggr_torch.slurm).
# Uses --container-name so the install persists across the per-sweep srun steps.
# ---------------------------------------------------------------------------
if [ -n "${WHEEL_PATH}" ]; then
    INSTALL_CMD="pip install ${WHEEL_PATH}[devel]"
elif [ -n "${REPO}" ]; then
    if [ "${BUILD_WHEEL}" = "true" ]; then
        BUILD="cd ${REPO} && python3 scripts/build_wheel.py --trt_root /usr/local/tensorrt --use_ccache --clean"
        [ -n "${CUDA_ARCH}" ] && BUILD="${BUILD} --cuda_architectures \"${CUDA_ARCH}\""
        INSTALL_CMD="${BUILD} && cd ${REPO} && pip install -e .[devel]"
    else
        INSTALL_CMD="cd ${REPO} && pip install -e .[devel]"
    fi
else
    INSTALL_CMD="echo 'Using TensorRT-LLM pre-installed in the container'"
fi

echo "Install: ${INSTALL_CMD}"
if ! srun --container-image="${IMAGE}" --container-name="${CONTAINER_NAME}" \
        --container-mounts="${MOUNTS}" -N2 --ntasks-per-node=1 \
        bash -c "${INSTALL_CMD}" &> "${WORK}/logs/install.log"; then
    echo "ERROR: install failed; see ${WORK}/logs/install.log" >&2
    exit 1
fi
echo "Install OK."

# ---------------------------------------------------------------------------
# Sweep loop.
# ---------------------------------------------------------------------------
# --kill-on-bad-exit=1: if any rank dies (e.g. the driver's watchdog SIGKILLs a
# rank wedged on a hung transfer), tear down the whole step instead of leaving
# the surviving ranks blocked in an MPI barrier waiting for the dead one.
COMMON="--container-image=${IMAGE} --container-name=${CONTAINER_NAME} \
        --container-mounts=${MOUNTS} --mpi=pmix --kill-on-bad-exit=1"

for SWEEP_IDX in $(seq 0 $((NUM_SWEEPS - 1))); do
    # This sweep's UCX_* exports (and CTT_SWEEP_NAME) as shell `export` lines.
    UCX_ENV="$(python3 "${HERE}/report.py" "${CONFIG}" --emit-ucx-env --sweep "${SWEEP_IDX}")"
    eval "${UCX_ENV}"  # local copy, for the log line below

    echo "=== sweep ${SWEEP_IDX} (${CTT_SWEEP_NAME}) UCX_TLS=${UCX_TLS:-} UCX_NET_DEVICES=${UCX_NET_DEVICES:-} ==="

    # IMPORTANT: some container images set their own default UCX_TLS via the
    # entrypoint/profile, which would override anything passed through
    # `srun --export`. So we (re-)export the sweep's UCX vars INSIDE the
    # container `bash -c`, after the profile has run, where they win.
    PRELUDE="${UCX_ENV}
export UCX_PROTO_INFO=used"

    EXPORTS="ALL,CTX_NODE=${CTX_NODE},ZMQ_PORT=${ZMQ_PORT},CTT_CONFIG=${CONFIG},CTT_SWEEP=${SWEEP_IDX}"

    # `timeout -k 60 --signal=TERM`: send SIGTERM at PER_SWEEP_TIMEOUT, then
    # SIGKILL 60s later. Without --kill-after, a srun/native process that
    # ignores SIGTERM would leave `timeout` (and this loop) blocked forever, so
    # a single hung sweep would prevent every later UCX env set from running.
    # One combined log per rank: omit --error so srun merges stderr into
    # --output (stdout + stderr, incl. UCX_PROTO_INFO, in a single .log file).
    timeout -k 60 --signal=TERM "${PER_SWEEP_TIMEOUT}" srun ${COMMON} --export="${EXPORTS}" \
        --nodelist="${CTX_NODE}" -N1 --ntasks-per-node="${N}" \
        --output="${WORK}/logs/sweep${SWEEP_IDX}_ctx_rank%t.log" \
        bash -c "${PRELUDE}
${DRIVER} --role ctx" &
    CTX_PID=$!

    timeout -k 60 --signal=TERM "${PER_SWEEP_TIMEOUT}" srun ${COMMON} --export="${EXPORTS}" \
        --nodelist="${GEN_NODE}" -N1 --ntasks-per-node="${N}" \
        --output="${WORK}/logs/sweep${SWEEP_IDX}_gen_rank%t.log" \
        bash -c "${PRELUDE}
${DRIVER} --role gen" &
    GEN_PID=$!

    wait "${CTX_PID}"; CTX_RC=$?
    wait "${GEN_PID}"; GEN_RC=$?
    echo "sweep ${SWEEP_IDX} done (ctx_rc=${CTX_RC} gen_rc=${GEN_RC})"
done

# ---------------------------------------------------------------------------
# Aggregate results.
# ---------------------------------------------------------------------------
python3 "${HERE}/report.py" "${CONFIG}" --aggregate --out "${WORK}/results.json"
echo "Results written to ${WORK}/results.json"
