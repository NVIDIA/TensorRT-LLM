#!/bin/bash
pkill -9 -f launch_disaggregated || true
rm -rf output.json || true

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXAMPLE_DIR=${SCRIPT_DIR}/..
CLIENT_DIR=${EXAMPLE_DIR}/clients

if [[ "$1" == "2_ranks" ]]; then
  mpirun --allow-run-as-root -n 2 python3 ${EXAMPLE_DIR}/launch_disaggregated_workers.py -c ${EXAMPLE_DIR}/disagg_config.yaml &> output_workers &
  python3 ${EXAMPLE_DIR}/launch_disaggregated_server.py -c ${EXAMPLE_DIR}/disagg_config.yaml  &> output_disagg &
elif [[ "$1" == "4_ranks" ]]; then
  mpirun --allow-run-as-root -n 4 python3 ${EXAMPLE_DIR}/launch_disaggregated_workers.py -c ${SCRIPT_DIR}/disagg_config_ctxtp2_gentp1.yaml &> output_workers &
  python3 ${EXAMPLE_DIR}/launch_disaggregated_server.py -c ${SCRIPT_DIR}/disagg_config_ctxtp2_gentp1.yaml  &> output_disagg &
fi

python3 ${CLIENT_DIR}/disagg_client.py -c ${EXAMPLE_DIR}/disagg_config.yaml -p ${CLIENT_DIR}/prompts.json --server-start-timeout 180
python3 ${CLIENT_DIR}/disagg_client.py -c ${EXAMPLE_DIR}/disagg_config.yaml -p ${CLIENT_DIR}/prompts.json --server-start-timeout 180 --streaming -o output_streaming.json

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
