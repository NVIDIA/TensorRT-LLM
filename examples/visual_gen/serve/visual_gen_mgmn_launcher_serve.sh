#!/bin/bash
#SBATCH -A <account>    # parameter
#SBATCH -p <partition>  # parameter
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o visual-gen-serve-mgmn.out
#SBATCH -e visual-gen-serve-mgmn.err
#SBATCH -J visual-gen-serve-mgmn

##############################################################################
# OVERVIEW:
# This script launches trtllm-serve for VisualGen across multiple nodes with
# the MGMN launcher (trtllm-llmapi-launch) — the same recipe used for LLM
# workloads (see examples/llm-api/llm_mgmn_trtllm_serve.sh).
#
# One MPI rank runs per GPU (--mpi=pmix, --ntasks-per-node = GPUs per node).
# Only rank 0 executes trtllm-serve; the launcher hosts a diffusion worker on
# every rank. No MASTER_ADDR/MASTER_PORT plumbing is needed: the rank-0
# worker resolves the torch rendezvous on its own host and distributes it to
# all ranks over MPI.
#
# The HTTP server listens on the FIRST node of the allocation. The srun runs
# in the foreground; the server serves requests until the job hits its time
# limit or is cancelled with scancel.
#
# CONTRACT: total MPI ranks (nodes × ntasks-per-node) must equal
# parallel_config.n_workers (= cfg_size × cp_size × ulysses_size × tp_size)
# of the --visual_gen_args YAML. This script's defaults launch 2 × 4 = 8
# ranks, matching examples/visual_gen/configs/wan2.2-t2v-fp8-8gpu.yaml
# (cfg_size=2 × ulysses_size=4). A mismatch fails fast at startup.
#
# WHAT TO MODIFY:
# 1. SLURM Parameters (lines above):
#    - Replace <account> with your SLURM account name
#    - Replace <partition> with your SLURM partition name
#    - Adjust -N, --ntasks-per-node, and --gpus-per-node; keep
#      ntasks-per-node equal to gpus-per-node (one rank per GPU) and the
#      total rank count equal to the config's n_workers
#
# 2. Environment Variables (set before running sbatch, or edit defaults below):
#    - CONTAINER_IMAGE:   Docker image or enroot .sqsh image with TensorRT-LLM installed
#    - MOUNT_DIR:         host directory to mount into the container (default: $HOME)
#    - MOUNT_DEST:        mount destination path inside the container (default: $HOME)
#    - PROJECT_ROOT:      path to TensorRT-LLM source on the shared filesystem
#    - MODEL:             local model directory (downloading from HF is not
#                         recommended in multi-node Slurm jobs; pre-download it)
#    - VISUAL_GEN_CONFIG: --visual_gen_args YAML whose parallel_config yields
#                         n_workers == total MPI ranks
#    - SERVER_PORT:       HTTP port for trtllm-serve (default: 8000)
#
# EXAMPLE USAGE:
#   export CONTAINER_IMAGE="/path/to/tensorrt-llm.sqsh"
#   export PROJECT_ROOT="/path/to/TensorRT-LLM"
#   export MODEL="/path/to/Wan2.2-T2V-A14B-Diffusers"
#   sbatch visual_gen_mgmn_launcher_serve.sh
#
# ALTERNATIVE: benchmark_visual_gen_mgmn_distributed.sh launches the server
# in the external-launch (SPMD) mode instead — plain srun without the
# launcher, every rank running trtllm-serve, with explicit MASTER_ADDR/
# MASTER_PORT — and adds a benchmark client step.
##############################################################################

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/TensorRT-LLM}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/path/to/tensorrt-llm.sqsh}"
export MOUNT_DIR="${MOUNT_DIR:-$HOME}"
export MOUNT_DEST="${MOUNT_DEST:-$HOME}"

export MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
export VISUAL_GEN_CONFIG="${VISUAL_GEN_CONFIG:-${PROJECT_ROOT}/examples/visual_gen/configs/wan2.2-t2v-fp8-8gpu.yaml}"
export SERVER_PORT="${SERVER_PORT:-8000}"

# ---------------------------------------------------------------------------
# Derived values — do not edit
# ---------------------------------------------------------------------------

# First node of the allocation: rank 0 — and therefore the HTTP server —
# runs here. Used only for the endpoint printout below, not for rendezvous.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "============================================"
echo "VisualGen MGMN Serving (trtllm-llmapi-launch)"
echo "============================================"
echo "Model:               ${MODEL}"
echo "VisualGen config:    ${VISUAL_GEN_CONFIG}"
echo "Server:              http://${HEAD_NODE}:${SERVER_PORT}"
echo "Nodes:               ${SLURM_NNODES}"
echo "Tasks per node:      ${SLURM_NTASKS_PER_NODE}"
echo "Total MPI ranks:     ${SLURM_NTASKS} (must equal the config's n_workers)"
echo "============================================"
echo ""
echo "Once healthy, query the server from a login or compute node:"
echo "  curl http://${HEAD_NODE}:${SERVER_PORT}/health"
echo ""

srun -l \
    --mpi=pmix \
    --kill-on-bad-exit=1 \
    --export=ALL \
    --container-image "${CONTAINER_IMAGE}" \
    --container-workdir "${PROJECT_ROOT}" \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    trtllm-llmapi-launch \
      trtllm-serve "${MODEL}" \
        --host 0.0.0.0 \
        --port "${SERVER_PORT}" \
        --enable_visual_gen \
        --visual_gen_args "${VISUAL_GEN_CONFIG}"
