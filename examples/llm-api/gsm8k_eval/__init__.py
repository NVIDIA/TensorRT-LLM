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
"""Public pinned GSM8K evaluation over an OpenAI-compatible chat endpoint.

``protocol`` defines the pinned dataset items, prompt, request and scorer; ``records`` the
durable per-question results, summary and verdict; ``server`` the lifecycle of a managed
``trtllm-serve`` process; ``runner`` drives one run against an existing endpoint or a managed
server. ``manifest.json`` is the frozen public first-100 item manifest. The command line is
``examples/llm-api/eval_gsm8k.py``. Nothing here imports TensorRT-LLM, CUDA or pytest.
"""
