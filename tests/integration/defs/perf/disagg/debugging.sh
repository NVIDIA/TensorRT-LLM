
# How to collect the entire test list
# Please apply the corresponding env file under ./envs before collect the test list

cd /lustre/fsw/portfolios/coreai/users/fredricz/tensorrt_llm/tests/integration/defs/perf/disagg

source ./perf/disagg/envs/.env_oci

# Local test cases collect
export WORK_DIR="/mnt/c/code/TensorRT-LLM/tests/integration/defs/perf/disagg"
export OUTPUT_PATH=/mnt/c/code/TensorRT-LLM/tests/integration/defs/perf/disagg/output
poetry run pytest --disagg --collect-only -q &> testlist_h100.txt


# run compare_backends.py to generate backend comparison report
poetry run python compare_backends.py \
    --csv-path "$CSV_PATH" \
    --threshold 5.0 \
    --default-backend NIXL \
    --output backend_comparison.csv \
    --html backend_comparison.html

# run with test list file
poetry run pytest --disagg test_disagg.py -s -vv --disagg-test-list=./testlist/testlist_gb200_debug.txt

# Remove the .cache directory when it's too big
# This one will lead the poetry install command to fail
rm -rf ~/.cache

# Test simple collect with TensorRT-LLM installed from source
srun -N1 -n1 \
    --partition=batch \
    --account=coreai_comparch_trtllm \
    --gres=gpu:4 \
    --time=01:00:00 \
    --container-image=${CONTAINER_IMAGE} \
    --container-name=debug-collect \
    --container-mounts=${WORK_DIR}:${WORK_DIR},${OUTPUT_PATH}:${OUTPUT_PATH},${REPO_DIR}:${REPO_DIR} \
    bash -c "cd ${REPO_DIR}
     pip3 install -r ${REPO_DIR}/requirements-dev.txt || echo '⚠️  requirements-dev.txt install failed, continuing...'
     echo '📦 Step 2: Installing TensorRT-LLM wheel...'
     pip3 install ${REPO_DIR}/build/*.whl --extra-index-url https://gitlab-master.nvidia.com/api/v4/projects/100660/packages/pypi/simple
     cd ${WORK_DIR}
     python3 simple_collect.py ${OUTPUT_PATH}"

# Test simple collect with TensorRT-LLM built in
srun -N1 -n1 \
    --partition=batch \
    --account=coreai_comparch_trtllm \
    --gres=gpu:4 \
    --time=01:00:00 \
    --container-image=${CONTAINER_IMAGE} \
    --container-name=debug-collect \
    --container-mounts=${WORK_DIR}:${WORK_DIR},${OUTPUT_PATH}:${OUTPUT_PATH} \
    bash -c "cd ${WORK_DIR} && python3 simple_collect.py ${OUTPUT_PATH}"

# Get max GPU frequency and memory frequency
srun -N1 -n1 \
    --partition=batch \
    --account=coreai_comparch_trtllm \
    --gres=gpu:4 \
    --time=00:10:00 \
    --container-image=${CONTAINER_IMAGE} \
    --container-name=debug-collect \
    --container-mounts=${WORK_DIR}:${WORK_DIR},${OUTPUT_PATH}:${OUTPUT_PATH} \
    bash -c "nvidia-smi --query-gpu=index,name,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory --format=csv"
