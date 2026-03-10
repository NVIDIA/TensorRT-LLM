#!/bin/bash
# Run all GB200 disagg gen_only perf sanity tests on OCI.
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
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_gpt-oss-120b-fp4_1k1k_con2048_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_gpt-oss-120b-fp4_1k1k_con512_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_gpt-oss-120b-fp4_8k1k_con512_ctx1_tp1_gen1_dep2_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_gpt-oss-120b-fp4_1k1k_con64_ctx1_tp1_gen1_tp4_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_gpt-oss-120b-fp4_8k1k_con128_ctx1_tp1_gen1_tp4_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_gpt-oss-120b-fp4_8k1k_con4_ctx1_tp1_gen1_tp4_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_deepseek-r1-fp4_1k1k_con3072_ctx1_dep4_gen1_dep4_eplb0_mtp1_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_deepseek-v32-fp4_1k1k_con2048_ctx1_dep4_gen1_dep4_eplb0_mtp1_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_kimi-k2-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_qwen3-235b-fp4_8k1k_con1024_ctx1_tp1_gen1_dep8_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_deepseek-r1-fp4_1k1k_con1024_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_deepseek-r1-fp4_8k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-UCX]"
    "perf/test_perf_sanity.py::test_e2e[disagg_upload-gen_only-gb200_deepseek-r1-fp4_128k8k_con1_ctx1_pp8_gen1_tep8_eplb0_mtp3_ccb-UCX]"
)

for TEST in "${TESTS[@]}"; do
    # Extract the config name between "gen_only-" and the closing "]"
    CONFIG_NAME=$(echo "$TEST" | sed 's/.*gen_only-\(.*\)]/\1/')

    cd "${LOCAL_DIR}"
    python3 submit.py --test-list "$TEST" \
        --partition "$PARTITION" \
        --account "$ACCOUNT" \
        --job-name "$JOB_NAME" \
        --image "$IMAGE" \
        --mounts "$MOUNTS" \
        --llm-models-root "$LLM_MODELS_ROOT" \
        --work-dir "${LOCAL_DIR}/gen_only-1-${CONFIG_NAME}"

    cd "${LOCAL_DIR}/gen_only-1-${CONFIG_NAME}" && sbatch slurm_launch.sh
done
