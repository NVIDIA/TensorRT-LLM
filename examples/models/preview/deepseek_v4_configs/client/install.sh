#!/bin/bash
# Installs the AgentX client as the measured runs did: aiperf from SemiAnalysisAI/agentx-harness
# into a venv laid over the TensorRT-LLM container's Python. Put the venv on node-local storage;
# a venv on NFS installs tens of thousands of small files slowly.
#   usage: install.sh <venv dir>
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
venv=$1
aiperf_ref=754356e9a39acc6cc6afb242d123bb57c3fb6f75   # SemiAnalysisAI/agentx-harness

python3 -m venv --system-site-packages --without-pip "$venv"
pip=("$venv/bin/python" -m pip install -q)
# The measured runs' install steps, in their order: aiperf without its dependencies, its CLI and
# serialization dependencies, a `datasets` new enough for the trace corpus, and three pins.
"${pip[@]}" --force-reinstall --no-deps \
    "aiperf @ git+https://github.com/SemiAnalysisAI/agentx-harness.git@$aiperf_ref"
"${pip[@]}" cyclopts "ruamel.yaml" msgspec orjson kaleido dash dash-bootstrap-components \
    textual starlette-compress setproctitle jmespath aiofiles ffmpeg-python \
    pydantic-settings prometheus_client fastapi soundfile plotly seaborn tiktoken \
    tqdm "uvicorn[standard]" uvloop crick zstandard
"${pip[@]}" -U datasets
"${pip[@]}" "transformers==5.8.1"
"${pip[@]}" --force-reinstall --no-deps "pydantic==2.12.5" "pydantic-core==2.41.5"
"$venv/bin/python" -m pip freeze > "$venv/pip-freeze.txt"
