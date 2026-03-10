#!/bin/bash
# Run all aggregated perf sanity tests on OCI.
# No wheel build, no nsys capture.

set -euo pipefail

REPO_ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/chenfeiz/repo/trtllm-2"
LOCAL_DIR="${REPO_ROOT}/jenkins/scripts/perf/local"
IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.12-py3-aarch64-ubuntu24.04-trt10.14.1.48-skip-tritondevel-202602011118-10901"
MOUNTS="/home/chenfeiz/:/home/chenfeiz/,/home/chenfeiz/.cache:/root/.cache,/lustre/fsw/portfolios/coreai/:/lustre/fsw/portfolios/coreai/,/lustre/fs1/portfolios/coreai/:/lustre/fs1/portfolios/coreai/"
LLM_MODELS_ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models"
PARTITION="batch"
ACCOUNT="coreai_comparch_trtllm"
JOB_NAME="perf_test"

TESTS=(
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_v32_fp4_grace_blackwell-v32_fp4_tep4_mtp3_1k1k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_v32_fp4_grace_blackwell-v32_fp4_tep4_mtp3_8k1k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-k2_thinking_fp4_2_nodes_grace_blackwell-k2_thinking_fp4_dep8_32k8k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-k2_thinking_fp4_2_nodes_grace_blackwell-k2_thinking_fp4_tep8_32k8k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-k2_thinking_fp4_grace_blackwell-k2_thinking_fp4_tep4_8k1k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_dep4_mtp1_1k8k]"
    # "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_v32_fp4_blackwell-v32_fp4_dep8_mtp1_8k1k]"
    # "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_blackwell-r1_fp4_v2_dep8_mtp1_8k1k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_tp4_mtp3_1k8k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_tep4_mtp3_8k1k]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_r1_fp4_v2_2_nodes_grace_blackwell-r1_fp4_v2_tep8_mtp3]"
    "perf/test_perf_sanity.py::test_e2e[aggr_upload-ctx_only-gb200_kimi-k2-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-UCX]"
    # "perf/test_perf_sanity.py::test_e2e[aggr_upload-deepseek_v32_fp4_blackwell-v32_fp4_tep8_mtp3_8k1k]"
)

for TEST in "${TESTS[@]}"; do
    # Extract the config name between "aggr_upload-" and the closing "]"
    CONFIG_NAME=$(echo "$TEST" | sed 's/.*aggr_upload-\(.*\)]/\1/')

    cd "${LOCAL_DIR}"
    python3 submit.py --test-list "$TEST" \
        --partition "$PARTITION" \
        --account "$ACCOUNT" \
        --job-name "$JOB_NAME" \
        --image "$IMAGE" \
        --mounts "$MOUNTS" \
        --llm-models-root "$LLM_MODELS_ROOT" \
        --work-dir "${LOCAL_DIR}/agg-${CONFIG_NAME}"

    cd "${LOCAL_DIR}/agg-${CONFIG_NAME}" && sbatch slurm_launch.sh
done
