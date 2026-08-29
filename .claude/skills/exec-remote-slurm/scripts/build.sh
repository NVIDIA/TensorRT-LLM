#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# build.sh — Runs inside the container on the compute node.
# Builds TensorRT-LLM from source using build_wheel.py.
#
# Usage: build.sh <repo_dir> [build_wheel_args...]
#
# Default build_wheel.py flags (used when no extra args are provided):
#   -a "100-real" --nvtx
# Any extra arguments after repo_dir are forwarded verbatim to build_wheel.py.

set -euo pipefail

repo_dir=${1:?Usage: build.sh <repo_dir> [build_wheel_args...]}
shift

cd "${repo_dir}"

if [[ $# -gt 0 ]]; then
    echo "[build.sh] Running: python3 ./scripts/build_wheel.py $*"
    python3 ./scripts/build_wheel.py "$@"
else
    echo "[build.sh] Running default build"
    python3 ./scripts/build_wheel.py \
        -a "100-real" \
        --nvtx
fi
