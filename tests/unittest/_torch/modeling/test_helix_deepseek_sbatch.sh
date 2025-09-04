#!/bin/bash

if [ -z "$WORKDIR" || \
     -z "$CONTAINER_NAME" || \
     -z "$CONTAINER_IMAGE" || \
     -z "$CONTAINER_MOUNT" || \
     -z "$REPO_DIR" || \
     -z "$LLM_MODELS_ROOT" || \
     -z "$SLURM_ACCOUNT" || \
     -z "$SLURM_JOB_NAME" || \
     -z "$SLURM_PARTITION" ]; then
  echo "Environment not set, please source test_helix_deepseek_sbatch_env.sh first"
  exit 1
fi

TP=${1:-8}
KVP=${2:-1}
EP=${3:-2}
DENSE=${4:-0}
dense_arg=""
if (( DENSE == 1 )); then
  dense_arg="--dense"
fi
world_size=$((TP * KVP))
if (( world_size % EP != 0 )); then
  echo "World size $world_size must be a multiple of EP $EP"
  exit 1
fi
gpus_per_node=4
NODES=$(((world_size + gpus_per_node - 1) / gpus_per_node))
gpus=$((NODES * gpus_per_node))
sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=01:00:00
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --comment=fact_off
#SBATCH --gres=gpu:${gpus_per_node}
#SBATCH --segment=${NODES}
#SBATCH --ntasks=${gpus}
#SBATCH --ntasks-per-node=${gpus_per_node}

cleanup_on_failure() {
    echo "Error: \$1"
    scancel \${SLURM_JOB_ID}
    exit 1
}

set -x

logdir=\${WORKDIR}/slurm-\${SLURM_JOB_ID}
mkdir -p \${logdir}
full_logdir=\${logdir}

echo "Starting container..."
if ! srun -l --container-image=\${CONTAINER_IMAGE} \
        --container-name=\${CONTAINER_NAME} \
        --container-mounts=\${CONTAINER_MOUNT} \
        --mpi=pmix \
        echo "Container up." &> \${full_logdir}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check \${full_logdir}/container_launch.log"
fi

export TLLM_LOG_LEVEL="INFO" # DEBUG is verbose
export TRTLLM_ENABLE_PDL=1
export LD_LIBRARY_PATH=/workspace/TensorRT-LLM/cpp/build/tensorrt_llm/executor/cache_transmission/ucx_utils:\${LD_LIBRARY_PATH}

echo "====== Baseline ========"
srun --mpi pmix -N ${NODES} --ntasks-per-node ${gpus_per_node} \
  --container-env=MASTER_ADDR,MASTER_PORT \
  --container-name=\${CONTAINER_NAME} \
  --container-mounts=\${CONTAINER_MOUNT} \
  python3 \${REPO_DIR}/tests/unittest/_torch/modeling/test_helix_deepseek.py --type v3 --tp ${TP} --kvp ${KVP} --ep ${EP} ${dense_arg} \
    &> \${full_logdir}/benchmark.log 2>&1
EOF
