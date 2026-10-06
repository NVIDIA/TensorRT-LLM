#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 || ( $1 != build && $1 != install ) ]]; then
    echo "Usage: $0 {build|install} WHEEL_DIR" >&2
    exit 2
fi
wheel_dir=$2

torch_abi() {
    python3 - <<'PY'
import json
import platform
import sysconfig

import torch

print(json.dumps({
    "torch": torch.__version__,
    "torch_git": torch.version.git_version,
    "cuda": torch.version.cuda,
    "cxx11_abi": torch.compiled_with_cxx11_abi(),
    "python_abi": sysconfig.get_config_var("SOABI"),
    "machine": platform.machine(),
}, indent=2))
PY
}

if [[ $1 == build ]]; then
    build_dir=$(mktemp -d)
    trap 'rm -rf "$build_dir"' EXIT
    mkdir -p "$build_dir/causal-conv1d" "$build_dir/mamba" "$wheel_dir"

    # causal-conv1d 1.7.0 and mamba-ssm 2.3.0 (Mamba2, without Mamba3 dependencies).
    curl --fail --location --retry 3 \
        https://github.com/Dao-AILab/causal-conv1d/archive/cd81f0413cad2fc1e6f17e785ac39f59aae690cd.tar.gz \
        --output "$build_dir/causal-conv1d.tar.gz"
    curl --fail --location --retry 3 \
        https://github.com/state-spaces/mamba/archive/f1493ff6e9335160eb134eb67e59f8e4d9adefd6.tar.gz \
        --output "$build_dir/mamba.tar.gz"
    tar -xzf "$build_dir/causal-conv1d.tar.gz" -C "$build_dir/causal-conv1d" --strip-components=1
    tar -xzf "$build_dir/mamba.tar.gz" -C "$build_dir/mamba" --strip-components=1

    # PyTorch 2.14 headers require C++20 in both C++ and CUDA translation units.
    sed -i 's/-std=c++17/-std=c++20/g' "$build_dir/mamba/setup.py"

    export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
    export CAUSAL_CONV1D_SKIP_CUDA_BUILD=FALSE MAMBA_SKIP_CUDA_BUILD=FALSE
    export CAUSAL_CONV1D_FORCE_CXX11_ABI=FALSE MAMBA_FORCE_CXX11_ABI=FALSE
    export MAX_JOBS="${MAX_JOBS:-8}"
    python3 -m pip wheel --no-build-isolation --no-deps --no-cache-dir \
        --wheel-dir "$wheel_dir" "$build_dir/causal-conv1d" "$build_dir/mamba"
    torch_abi > "$wheel_dir/torch_abi.json"
else
    # NGC releases can have incompatible torch ABIs even with the same major/minor version.
    current_abi=$(torch_abi)
    if ! diff -u "$wheel_dir/torch_abi.json" - <<< "$current_abi"; then
        echo "Mamba wheels do not match this environment; rebuild with: $0 build WHEEL_DIR" >&2
        exit 1
    fi
    python3 -m pip install --no-index --no-deps --force-reinstall \
        "$wheel_dir"/causal_conv1d-*.whl "$wheel_dir"/mamba_ssm-*.whl

    python3 - <<'PY'
import torch
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import (
    mamba_chunk_scan_combined,
    mamba_split_conv1d_scan_combined,
)
from transformers import Qwen3_5MoeForConditionalGeneration

print(f"torch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"causal-conv1d={causal_conv1d.__version__}, mamba-ssm={mamba_ssm.__version__}")
PY
fi
