#!/bin/bash
set -Eeuo pipefail

# Determine working directory based on mode
if [ -n "${resourcePathNode:-}" ] && [ "$resourcePathNode" != "" ]; then
    # CI mode
    cd $resourcePathNode/TensorRT-LLM/src
else
    # Local mode
    cd $llmSrc
fi

pip install -e .
pip install -r requirements-dev.txt

hostname
nvidia-smi

echo "Installation completed on $(hostname)"
