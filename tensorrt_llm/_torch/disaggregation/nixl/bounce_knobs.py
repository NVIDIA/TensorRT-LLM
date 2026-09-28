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
"""Valid agent_bounce_params keys. Must stay dependency-free: imported from the
llm_args validator, where the transfer-agent bindings may not be loadable."""

# TRTLLM_NIXL_BOUNCE_* env-var names without the prefix and trailing _BYTES, lowercased.
# KEEP IN SYNC with kEnvKnobs in cpp/.../nixl_utils/bounce/BounceConfig.h (sync-tested).
AGENT_BOUNCE_PARAM_KEYS: frozenset = frozenset(
    {
        "max_chunk_size",
        "arena_allocation_granularity",
        "max_inflight_chunks_per_request",
        "copy_stream_count",
        "scatter_worker_count",
        "min_descriptor_count",
        "max_average_descriptor_size",
        "request_timeout_ms",
        "disable_fabric_memory",
        "enable_eager_gather",
        "use_zero_copy_arguments",
    }
)
