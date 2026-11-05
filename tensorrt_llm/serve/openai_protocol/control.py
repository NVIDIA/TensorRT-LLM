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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

from typing import Dict, List, Literal, Optional

from pydantic import Field, NonNegativeInt, PositiveInt

from tensorrt_llm.serve.openai_protocol.common import OpenAIBaseModel


class MemoryUpdateRequest(OpenAIBaseModel):
    tags: list[str] = Field(
        min_length=1,
        description="Memory tags to release/resume, e.g. ['model', 'kv_cache']")


class RuntimeMemoryUpdateRequest(OpenAIBaseModel):
    tags: Optional[List[str]] = Field(
        default=None,
        description="Memory tags to release/resume; omit to select defaults")


class UpdateWeightsRequest(OpenAIBaseModel):
    weights: Optional[Dict[str, str]] = Field(
        default=None,
        description="Weight handles dict, or None to finalize update")


ProfileActivity = Literal["CPU", "GPU", "CUDA_PROFILER"]


class StartProfileRequest(OpenAIBaseModel):
    """Request body for the POST /start_profile endpoint."""
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory where chrome traces are written. Defaults "
        "to the TLLM_TORCH_PROFILER_DIR environment variable, "
        "then /tmp.")
    num_steps: Optional[PositiveInt] = Field(
        default=None,
        description="Number of iterations to profile. Must be >= 1 if "
        "provided; if omitted, profiling runs until /stop_profile is "
        "called. ``num_steps == 0`` is rejected because the profile "
        "window would never close — the stop iteration would equal the "
        "start iteration, profile_step() would discard the stop marker "
        "as stale, and the window would run forever.")
    start_step: NonNegativeInt = Field(
        default=0,
        description="Skip this many iterations before profiling begins. "
        "Must be >= 0.")
    activities: List[ProfileActivity] = Field(
        default_factory=lambda: ["CPU", "GPU"],
        description="Which activities to trace. Supported values: "
        "'CPU', 'GPU', 'CUDA_PROFILER'. Unknown values are rejected at "
        "schema validation time.")
