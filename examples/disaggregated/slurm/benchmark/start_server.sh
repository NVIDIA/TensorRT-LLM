#!/bin/bash
set -u
set -e
set -x

config_file=$1

# Container runtimes (pyxis/enroot) reset image-defined variables like PATH
# at container start, so values passed via srun --export are lost for them.
# Allow the launcher config to prepend entries from inside the container.
# srun --export keeps any quotes in the exported values literal; strip them.
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND:-}"
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND#\'}"; TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND%\'}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND:-}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND#\'}"; TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND%\'}"
if [ -n "${TRTLLM_PATH_PREPEND:-}" ]; then
    export PATH="${TRTLLM_PATH_PREPEND}:${PATH}"
fi
if [ -n "${TRTLLM_PYTHONPATH_PREPEND:-}" ]; then
    export PYTHONPATH="${TRTLLM_PYTHONPATH_PREPEND}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# In-place (.pth-style) TRT-LLM installs may lack the trtllm-serve console
# script; fall back to the module entry point in that case.
trtllm_serve_cmd="trtllm-serve"
if ! command -v trtllm-serve >/dev/null 2>&1; then
    trtllm_serve_cmd="python3 -m tensorrt_llm.commands.serve"
fi

${trtllm_serve_cmd} disaggregated -c ${config_file} -t 7200 -r 7200
