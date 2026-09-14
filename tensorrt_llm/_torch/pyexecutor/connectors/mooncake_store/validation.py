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
"""Startup gates for the Mooncake store connector.

Every rejection here is a configuration whose failure mode is a wrong answer
rather than a slow one: KV that gets replayed without all of the state it was
computed with. Beam search, attention data parallelism, host and disk cache
tiers, and Mamba caches are rejected for all connectors in `py_executor`, so
they are not repeated.

Checks run at construction, before any request is admitted, so a bad deployment
fails at startup instead of after the first cache hit.
"""

from typing import TYPE_CHECKING

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

if TYPE_CHECKING:
    from ..kv_cache_layout import KvCacheLayout

__all__ = ["validate_layout", "validate_llm_args"]


def validate_llm_args(llm_args: TorchLlmArgs) -> None:
    """Reject parallel and model configurations this connector cannot serve."""
    if getattr(llm_args, "context_parallel_size", 1) > 1:
        raise NotImplementedError(
            "The mooncake-store connector does not support context parallelism. "
            "A stored page is keyed by the tokens it holds, but under context "
            "parallelism a rank holds a slice of the sequence rather than whole "
            "blocks of it, so the same key would name different bytes on "
            "different ranks."
        )

    if getattr(llm_args, "pipeline_parallel_size", 1) > 1:
        raise NotImplementedError(
            "The mooncake-store connector does not support pipeline parallelism. "
            "Keys are namespaced per rank, so each stage would store only its own "
            "layers and a prefix hit would require every stage to agree; that path "
            "is untested. Run with tensor parallelism only."
        )

    sparse_config = getattr(llm_args, "sparse_attention_config", None)
    if sparse_config is not None and not getattr(sparse_config, "sparse_disable_index_value", True):
        raise NotImplementedError(
            "The mooncake-store connector requires "
            "sparse_attention_config.sparse_disable_index_value=True. The index-V "
            "cache is a plain tensor outside the KV cache manager's paged pools, "
            "so it is neither described to the connector nor transferred; a "
            "replayed prefix would carry index-K from the store alongside stale "
            "index-V. This is the same restriction disaggregated serving applies."
        )


def validate_layout(layout: "KvCacheLayout") -> None:
    """Reject KV cache geometries this connector cannot key correctly."""
    windowed = [group.layer_group_id for group in layout.groups if group.window_size is not None]
    if windowed:
        raise NotImplementedError(
            "The mooncake-store connector does not support sliding-window "
            f"attention (layer groups {windowed} declare a window size). A page's "
            "validity then depends on where the window sits, which is a property "
            "of the request that read it rather than of the tokens it holds, so "
            "content-addressed reuse across instances is not sound."
        )

    if not layout.groups:
        raise ValueError(
            "The KV cache layout describes no layer groups, so there is nothing "
            "for the mooncake-store connector to transfer."
        )
