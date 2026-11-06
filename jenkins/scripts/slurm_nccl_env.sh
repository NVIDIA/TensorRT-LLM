#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Chain the CI image's BASH_ENV (docker/Dockerfile.multi) before applying
# overrides. Restore it while sourcing so nested initialization cannot recurse.
export BASH_ENV=/etc/bash.bashrc
if [ -f "$BASH_ENV" ]; then
    source "$BASH_ENV"
fi
export BASH_ENV="${BASH_SOURCE[0]}"

# Applied on every Bash startup, including the multi-node Python launcher.
if [ "${NCCL_NET_PLUGIN:-}" = "gcp" ] || [ "${NCCL_ENV_PLUGIN:-}" = "gcp" ]; then
    export NCCL_IB_DISABLE=0
    export NCCL_NET=gIB
    export NCCL_PROFILER_PLUGIN=none
fi
