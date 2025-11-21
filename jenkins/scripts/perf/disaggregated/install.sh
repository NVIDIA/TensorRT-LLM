#!/bin/bash
set -Eeuo pipefail

cd $resourcePathNode
llmSrcNode=$resourcePathNode/TensorRT-LLM/src

echo $SLURM_JOB_ID > $jobWorkspace/slurm_job_id.txt

wget -nv $llmTarfile
tar -zxf $tarName
which python3
python3 --version
apt-get install -y libffi-dev
nvidia-smi && nvidia-smi -q && nvidia-smi topo -m
if [[ $pytestCommand == *--run-ray* ]]; then
    pip3 install ray[default]
fi
cd $llmSrcNode && pip3 install --retries 1 -r requirements-dev.txt
cd $resourcePathNode &&  pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl
git config --global --add safe.directory "*"
gpuUuids=$(nvidia-smi -q | grep "GPU UUID" | awk '{print $4}' | tr '\n' ',' || true)
hostNodeName="${HOST_NODE_NAME:-$(hostname -f || hostname)}"
echo "HOST_NODE_NAME = $hostNodeName ; GPU_UUIDS = $gpuUuids ; STAGE_NAME = $stageName"
