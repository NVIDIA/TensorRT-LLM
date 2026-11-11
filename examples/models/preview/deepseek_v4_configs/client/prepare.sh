#!/bin/bash
# Optional, once: fills the AgentX dataset cache $CLIENT_HF_HOME on shared storage, so jobs do not
# download the corpus each time (needed when CLIENT_OFFLINE=1). It also checks that the client
# installs. Run it inside the TensorRT-LLM container on a node with internet access, for example:
#   srun -N 1 --container-image="$IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
#       --no-container-mount-home bash client/prepare.sh "$PWD/cluster.env"
#   usage: prepare.sh <cluster env>
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

set -euo pipefail
source "$1"
venv=${CLIENT_VENV:-$(mktemp -d /tmp/agentx-client.XXXXXX)/venv}
[ -x "$venv/bin/aiperf" ] || bash "$(dirname "$0")/install.sh" "$venv"

mkdir -p "$CLIENT_HF_HOME"
HF_HOME=$CLIENT_HF_HOME "$venv/bin/python" - << 'PY'
from datasets import load_dataset
from huggingface_hub import dataset_info
repo = "semianalysisai/cc-traces-weka-062126"
print(len(load_dataset(repo, split="train")), "traces in", repo, "at revision", dataset_info(repo).sha)
PY
cp "$venv/pip-freeze.txt" "$CLIENT_HF_HOME/client-pip-freeze.txt" 2> /dev/null || true
echo "client installs; dataset cache in $CLIENT_HF_HOME"
