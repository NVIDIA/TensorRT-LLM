#!/bin/bash
ENV_DIR=/tmp/$USER/uv-envs/trtllm
uv venv -p 3.12.11 $ENV_DIR
source $ENV_DIR/bin/activate
uv pip install --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -e .