#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

curl --fail --location --retry 3 \
    https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.7.0.tar.gz \
    --output "$build_dir/causal-conv1d.tar.gz"
curl --fail --location --retry 3 \
    https://github.com/state-spaces/mamba/archive/refs/tags/v2.3.2.post1.tar.gz \
    --output "$build_dir/mamba.tar.gz"
tar -xzf "$build_dir/causal-conv1d.tar.gz" -C "$build_dir"
tar -xzf "$build_dir/mamba.tar.gz" -C "$build_dir"

# PyTorch 2.14 headers require C++20 in both C++ and CUDA translation units.
sed -i 's/-std=c++17/-std=c++20/g' "$build_dir/mamba-2.3.2.post1/setup.py"

# Compile against the installed torch, including its headers and C++ ABI.
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-8}"
python3 -m pip install --no-build-isolation --no-deps --no-cache-dir \
    "$build_dir/causal-conv1d-1.7.0" "$build_dir/mamba-2.3.2.post1"

python3 - <<'PY'
import torch
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda

print(f"torch={torch.__version__}, CUDA={torch.version.cuda}")
print(f"causal-conv1d={causal_conv1d.__version__}, mamba-ssm={mamba_ssm.__version__}")
PY
