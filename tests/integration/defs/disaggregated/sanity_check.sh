#!/bin/bash
set -x
pkill -9 -f trtllm-serve || true

rm -rf output.json || true
rm -rf output_streaming.json || true

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=$1

TEST_DESC=${2:-"2_ranks"}
NUM_ITERS=${3:-5}
SKIP_KILL=${4:-"no"}

CLIENT_DIR=${EXAMPLE_DIR}/clients

NUM_RANKS=""
CONFIG_FILE=""
if [[ "${TEST_DESC}" == "2_ranks" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${EXAMPLE_DIR}/disagg_config.yaml
elif [[ "${TEST_DESC}" == "gen_only" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_gen_only.yaml
elif [[ "${TEST_DESC}" == "cuda_graph" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_cuda_graph_padding.yaml
elif [[ "${TEST_DESC}" == "mixed" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_mixed.yaml
elif [[ "${TEST_DESC}" == "overlap" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_overlap.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp_8_overlap_dp" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_overlap_dp.yaml
elif [[ "${TEST_DESC}" == "4_ranks" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp2_gentp1.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp8" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp8_attention_dp" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp8_attention_dp_one" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp8_attention_dp_one_mtp" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp2_gentp2_deepseek_v3_lite_attention_dp_one_mtp.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp8_tp1_mtp" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite_one_mtp.yaml
elif [[ "${TEST_DESC}" == "deepseek_v3_lite_fp8_tp1" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/test_configs/disagg_config_ctxtp1_gentp1_deepseek_v3_lite.yaml
else
  echo "Invalid test description: ${TEST_DESC}"
  exit 1
fi

if [[ "${TEST_DESC}" == "gen_only" ]]; then
    export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1
fi

mpirun --allow-run-as-root -n ${NUM_RANKS} trtllm-serve disaggregated_mpi_worker -c ${CONFIG_FILE} &> output_workers &
trtllm-serve disaggregated --server_start_timeout 900 -c ${CONFIG_FILE} &> output_disagg &

for i in $(seq 1 ${NUM_ITERS}); do
    python3 ${CLIENT_DIR}/disagg_client.py -c ${EXAMPLE_DIR}/disagg_config.yaml -p ${CLIENT_DIR}/prompts.json --server-start-timeout 950
    python3 ${CLIENT_DIR}/disagg_client.py -c ${EXAMPLE_DIR}/disagg_config.yaml -p ${CLIENT_DIR}/prompts.json --server-start-timeout 950 --streaming -o output_streaming.json
done

echo "------------------"
echo "Workers output:"
echo "------------------"
cat output_workers

echo ""
echo ""
echo "------------------"
echo "Disagg server output"
echo "------------------"
cat output_disagg

if [[ "${SKIP_KILL}" != "yes" ]]; then
  pkill -9 -f trtllm-serve || true
fi

# Check that size of output is size of prompts.json
num_outputs=$(jq '. | length' output.json)
num_outputs_streaming=$(jq '. | length' output_streaming.json)
expected_num_outputs=$(jq '. | length' ${CLIENT_DIR}/prompts.json)

if [[ "${num_outputs}" -ne "${expected_num_outputs}" ]]; then
  echo "Expected ${expected_num_outputs} outputs, got ${num_outputs} outputs"
  exit 1
fi

if [[ "${num_outputs_streaming}" -ne "${expected_num_outputs}" ]]; then
  echo "Expected ${expected_num_outputs} outputs, got ${num_outputs_streaming} streaming outputs"
  exit 1
fi

if [[ "${TEST_DESC}" != "gen_only" ]]; then
  expected_strings=("The capital of Germany is Berlin" "Asyncio is a Python library")
  if [[ "${TEST_DESC}" =~ "deepseek_v3_lite" ]]; then
    expected_strings=("Berlin" "Asyncio is a powerful tool")
  fi

  for expected_string in "${expected_strings[@]}"; do
      grep "${expected_string}" output.json
      grep "${expected_string}" output_streaming.json
  done
fi
