#!/bin/bash
pkill -9 -f launch_disaggregated || true
rm -rf output.json || true

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=${SCRIPT_DIR}/..
CLIENT_DIR=${EXAMPLE_DIR}/clients
TEST_DESC=${1:-"2_ranks"}
NUM_ITERS=${2:-5}

NUM_RANKS=""
CONFIG_FILE=""
if [[ "${TEST_DESC}" == "2_ranks" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${EXAMPLE_DIR}/disagg_config.yaml
elif [[ "${TEST_DESC}" == "cuda_graph" ]]; then
  NUM_RANKS=2
  CONFIG_FILE=${SCRIPT_DIR}/disagg_config_cuda_graph_padding.yaml
elif [[ "${TEST_DESC}" == "4_ranks" ]]; then
  NUM_RANKS=4
  CONFIG_FILE=${SCRIPT_DIR}/disagg_config_ctxtp2_gentp1.yaml
else
  echo "Invalid test description: ${TEST_DESC}"
  exit 1
fi

mpirun --allow-run-as-root -n ${NUM_RANKS} python3 ${EXAMPLE_DIR}/launch_disaggregated_workers.py -c ${CONFIG_FILE} &> output_workers &
python3 ${EXAMPLE_DIR}/launch_disaggregated_server.py -c ${CONFIG_FILE}  &> output_disagg &

for i in $(seq 1 ${NUM_ITERS}); do
    python3 ${CLIENT_DIR}/disagg_client.py -c ${EXAMPLE_DIR}/disagg_config.yaml -p ${CLIENT_DIR}/prompts.json --server-start-timeout 180
    python3 ${CLIENT_DIR}/disagg_client.py -c ${EXAMPLE_DIR}/disagg_config.yaml -p ${CLIENT_DIR}/prompts.json --server-start-timeout 180 --streaming -o output_streaming.json
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

pkill -9 -f launch_disaggregated || true

expected_strings=("The capital of France is Paris" "Asyncio is a Python library")

for expected_string in "${expected_strings[@]}"; do
    grep "${expected_string}" output.json
    grep "${expected_string}" output_streaming.json
done
