# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TRT-LLM Usage Telemetry.

Collects anonymous usage statistics to help improve TensorRT-LLM.
Data is sent to NVIDIA's telemetry service (NvTelemetry/GXT).

Opt-out:
  - Set environment variable TRTLLM_NO_USAGE_STATS=1
  - Set environment variable TELEMETRY_DISABLED=true or TELEMETRY_DISABLED=1
  - Set environment variable DO_NOT_TRACK=1
  - Create file ~/.config/trtllm/do_not_track
  - Pass TelemetryConfig(disabled=True) to LLM() or --no-telemetry via CLI
  - Automatically disabled in CI/test environments (override with TRTLLM_USAGE_FORCE_ENABLED=1)
"""

from tensorrt_llm.usage import config as _config
from tensorrt_llm.usage import usage_lib as _usage_lib

TelemetryConfig = _config.TelemetryConfig
TelemetryField = _config.TelemetryField
UsageContext = _config.UsageContext
report_usage = _usage_lib.report_usage
report_exit = _usage_lib.report_exit
start_usage_session = _usage_lib.start_usage_session
record_llm_initialization_attempt = _usage_lib.record_llm_initialization_attempt
record_llm_initialization_failure = _usage_lib.record_llm_initialization_failure
record_llm_initialized = _usage_lib.record_llm_initialized
record_llm_shutdown = _usage_lib.record_llm_shutdown
record_observed_signal = _usage_lib.record_observed_signal
record_termination_observation = _usage_lib.record_termination_observation
set_lifecycle_phase = _usage_lib.set_lifecycle_phase
set_usage_context = _usage_lib.set_usage_context
get_observed_signal = _usage_lib.get_observed_signal
get_termination_observation = _usage_lib.get_termination_observation
is_usage_stats_enabled = _usage_lib.is_usage_stats_enabled

__all__ = [
    "TelemetryConfig",
    "TelemetryField",
    "UsageContext",
    "get_observed_signal",
    "get_termination_observation",
    "record_llm_initialization_attempt",
    "record_llm_initialization_failure",
    "record_llm_initialized",
    "record_llm_shutdown",
    "record_observed_signal",
    "record_termination_observation",
    "report_exit",
    "report_usage",
    "set_lifecycle_phase",
    "set_usage_context",
    "start_usage_session",
    "is_usage_stats_enabled",
]
