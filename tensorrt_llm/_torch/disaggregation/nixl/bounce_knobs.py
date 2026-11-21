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
"""Valid agent_bounce_params keys and a model's bounce defaults. Must stay dependency-free:
imported from the llm_args validator, where the transfer-agent bindings may not be loadable."""

import os

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
        "scatter_on_order_stream",
        "scatter_copy_engine",
        "copy_stream_priority",
    }
)


# Bounce knobs a model opts into (RuntimeHooks.bounce_scatter_on_order_stream): received KV is
# scattered on the execution stream, between forwards, and the private copy streams take the
# default priority. Each key maps to (value, the TRTLLM_NIXL_BOUNCE_* variable that overrides it).
ORDER_STREAM_DEFAULTS: dict = {
    "scatter_on_order_stream": ("1", "TRTLLM_NIXL_BOUNCE_SCATTER_ON_ORDER_STREAM"),
    "copy_stream_priority": ("default", "TRTLLM_NIXL_BOUNCE_COPY_STREAM_PRIORITY"),
}
_model_defaults: dict = {}


def set_model_scatter_on_order_stream(enabled: bool) -> None:
    """Apply (or drop) the order-stream defaults to transfer agents created from now on."""
    _model_defaults.clear()
    if enabled:
        _model_defaults.update(ORDER_STREAM_DEFAULTS)


def effective_agent_bounce_params(params) -> dict:
    """``params`` (agent_bounce_params) over the model's defaults: a default applies only where
    neither ``params`` nor its TRTLLM_NIXL_BOUNCE_* variable sets the knob, so the precedence stays
    dict > variable > model default > built-in default."""
    merged = {
        key: value
        for key, (value, env_name) in _model_defaults.items()
        if not os.environ.get(env_name)
    }
    merged.update(params or {})
    return merged
