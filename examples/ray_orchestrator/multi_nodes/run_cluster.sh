#!/usr/bin/env bash

# NOTE: Multi-node with Ray orchestrator in TensorRT-LLM is an experimental feature and may not work on all systems.
# This script launches a Ray cluster and connects all allocated nodes for multi-node inference.

# The following variables are expected to be set in the environment:
#   CONTAINER: the path of the container image that has the desired TensorRT-LLM version installed.
#   MOUNTS: directory mount specification in format src:dest

# Run inside an already‑active allocated node.
# To start a Ray cluster across nodes:
#       >> bash -e launch_ray.sh
#
# See multi_nodes/README.md for more details.
#

set -euo pipefail

: "${CONTAINER:?Set CONTAINER to the container image path (e.g. /path/to/trtllm.sqfs)}"
: "${MOUNTS:?Set MOUNTS to mount spec src:dst[,src2:dst2...] (no spaces)}"
RAY_PORT=${RAY_PORT:-6379}

RUN_ID=$(date +%m%d-%H%M-%S)
HEAD_NAME="ray-head-${RUN_ID}"

SLURM_SUBMIT_DIR=$PWD
BASE_LOG_DIR=${BASE_LOG_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
LOG_DIR="$BASE_LOG_DIR/${SLURM_JOB_ID}-logs"
mkdir -p "$LOG_DIR"

COMMAND=""
if [[ "$#" -gt 0 ]]; then
  for arg; do [[ "$arg" == "--" ]] && shift && break; done
  COMMAND="$*"
fi

MIN_WORKER_PORT=${MIN_WORKER_PORT:-54001}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-54257}

COMMON_SRUN_ARGS+=" --mpi=pmix"
COMMON_SRUN_ARGS+=" --container-remap-root --container-writable"
COMMON_SRUN_ARGS+=" --container-mounts=$MOUNTS"
COMMON_SRUN_ARGS+=" --container-image=$CONTAINER"
COMMON_SRUN_ARGS+=" --container-workdir=$SLURM_SUBMIT_DIR"
COMMON_SRUN_ARGS+=" -p $SLURM_JOB_PARTITION"
COMMON_SRUN_ARGS+=" -A $SLURM_JOB_ACCOUNT"
COMMON_SRUN_ARGS+=" --gres=gpu:$SLURM_GPUS_ON_NODE"

# Getting the node names and IP addresses in the SLURM allocation
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
ip_addresses_array=()

for node in $nodes; do
    ip_address=$(host $node | awk '/has address/ { print $4 }')
    ip_addresses_array+=("$ip_address")
done

head_node=${nodes_array[0]}
head_node_ip=${ip_addresses_array[0]}
WORKERS=("${nodes_array[@]:1}")

ip_head=$head_node_ip:$RAY_PORT

BLUE='\e[96m'
GREEN='\e[32m'
RESET='\e[0m'

echo -e "${BLUE}[INFO] Head node : $head_node${RESET}"
echo -e "${BLUE}[INFO] Worker(s) : ${WORKERS[*]}${RESET}"
echo -e "${BLUE}[INFO] GPUs per node : ${SLURM_GPUS_ON_NODE}${RESET}"
echo -e "${BLUE}[INFO] CONTAINER: '$CONTAINER'${RESET}"
echo -e "${BLUE}[INFO] COMMAND = '$COMMAND'${RESET}"
echo -e "${BLUE}[INFO] Logs      : $LOG_DIR${RESET}"

########################################################
# Start Ray cluster on head node
########################################################

# enabled dashboard only for debug
# Add apt-get install -y --no-install-recommends libzmq3-dev for multi-node disagg

head_cmd=$(cat <<EOF
# WAR: clean all slurm / MPI / PMIx env to avoid pmix mismatch error
for v in \$(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \$1}'); do
    unset "\$v"
done
# ---- mark that the container shell is alive ----
touch "$LOG_DIR/STARTED_RAY_HEAD"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export TRTLLM_UCX_INTERFACE=eth0

ray start --head \
  --port=$RAY_PORT \
  --node-ip-address="$head_node_ip" \
  --disable-usage-stats \
  --include-dashboard=true \
  --min-worker-port=${MIN_WORKER_PORT} \
  --max-worker-port=${MAX_WORKER_PORT} \
  --num-cpus=16 \
  --block
EOF
)

echo "head_cmd: $head_cmd"
srun --overlap $COMMON_SRUN_ARGS  --job-name=ray-head --container-name="${HEAD_NAME}" --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "$head_cmd" 2>&1 | tee -a "$LOG_DIR/ray-head.log" &


########################################################
# Wait til Ray cluster ready on head
########################################################

sleep 30

echo "[INFO] waiting for head container..."
while ! srun --overlap -N1 -n1 -w "$head_node" bash -c "test -f '$LOG_DIR/STARTED_RAY_HEAD'" >/dev/null 2>&1; do
    echo "[INFO][$(date)] Waiting for head node container to start..."
    sleep 2
done

ATTEMPTS=15
until srun --overlap --nodes=1 --ntasks=1 --cpu-bind=none -w "$head_node" \
           --container-name="${HEAD_NAME}" bash -lc 'ray status >/dev/null 2>&1'
do
    ((ATTEMPTS--)) || { echo "[ERROR] Ray head did not come up."; exit 1; }
    sleep 4
done
echo -e "${GREEN}[INFO] Ray head is UP! ✔${RESET}"


########################################################
# Start Ray worker nodes
########################################################

NUM_WORKERS=${#WORKERS[@]}
for idx in "${!WORKERS[@]}"; do
    W=${WORKERS[$idx]}

    worker_cmd=$(
    cat <<EOF
# WAR: clean all slurm / MPI / PMIx env to avoid pmix mismatch error
for v in \$(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \$1}'); do
    unset "\$v"
done

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export TRTLLM_UCX_INTERFACE=eth0

ray start --address="$ip_head" \
          --disable-usage-stats \
          --min-worker-port=${MIN_WORKER_PORT} \
          --max-worker-port=${MAX_WORKER_PORT} \
          --num-cpus=16 \
          --block
EOF
    )

    echo "worker_cmd (node=$W, idx=$idx): $worker_cmd"
    srun --overlap $COMMON_SRUN_ARGS --exact --container-name="ray-worker-${idx}" --nodes=1 --ntasks=1 --cpu-bind=none \
         -w "$W" bash -lc "$worker_cmd" 2>&1 | tee -a "$LOG_DIR/ray-worker-${idx}.log" &
done


##############################################################################
# Wait until every node is connected in the Ray cluster
##############################################################################

expected_nodes=$(( ${#WORKERS[@]} + 1 ))       # head + workers
echo "[INFO] Waiting for ${#WORKERS[@]} worker nodes to join..."

ATTEMPTS=30
while (( ATTEMPTS-- )); do
    # run ray status inside the head container
    status=$(srun --overlap --nodes=1 --ntasks=1 --cpu-bind=none \
                  -w "$head_node" --container-name="${HEAD_NAME}" ray status 2>/dev/null)

    # to count active nodes
    active_count=$(
        sed -n '/^Active:/,/^Pending:/p' <<<"$status" |
        grep -c 'node_'
    )

    echo "[INFO] Detected $active_count worker node(s) out of $expected_nodes total expected node(s)."

    [[ $active_count -eq $expected_nodes ]] && break
    sleep 4
done

[[ $active_count -eq $expected_nodes ]] || { echo '[ERROR] Ray nodes timed-out'; exit 1; }

echo -e "${GREEN}[INFO] All nodes have joined Ray cluster. ✔  \`ray status\`:${RESET}"
printf '%s\n' "$status"


##############################################################################
# Run TRT-LLM driver (if given)
##############################################################################

if [[ -n "$COMMAND" ]]; then
  driver_srun_cmd=(
    srun --overlap
    --no-container-mount-home
    --container-name="${HEAD_NAME}"
    --nodes=1 --ntasks=1 -w "$head_node"
    bash -lc "$COMMAND"
  )
  # --c ontainer-workdir="$CONTAINER_CWD"

  echo -e "${BLUE}[INFO] Driver srun command:${RESET}"
  printf '  %q ' "${driver_srun_cmd[@]}"
  echo

  set +e
  "${driver_srun_cmd[@]}" 2>&1 | tee -a "$LOG_DIR/trtllm-command.log"
  DRIVER_RC=$?
  set -e

  if [[ $DRIVER_RC -ne 0 ]]; then
    echo "[WARN] driver exited with status $DRIVER_RC – Ray cluster left running."
  fi
else
  echo "[INFO] No COMMAND supplied. Idling. Press Ctrl+C to exit."
fi

wait
