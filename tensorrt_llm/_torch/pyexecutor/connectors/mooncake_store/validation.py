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
rather than a slow one: KV replayed without all of the state it was computed
with. Beam search, attention data parallelism and Mamba caches are rejected for
all connectors in `py_executor`, so they are not repeated here.

Checks run at construction, before any request is admitted, so a bad deployment
fails at startup instead of after the first cache hit.
"""

import os
from typing import TYPE_CHECKING, Optional

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from ..kv_cache_layout import KvCacheLayout
    from .config import MooncakeStoreConnectorConfig

__all__ = ["validate_layout", "validate_llm_args", "validate_node_budget"]

_GIB = 1 << 30

#: Host memory to leave unclaimed after the segments, covering page cache,
#: allocator arenas and the runtime's own host buffers, none of which are
#: accounted for below.
NODE_BUDGET_RESERVE_BYTES = 32 * _GIB


def validate_llm_args(llm_args: TorchLlmArgs) -> None:
    """Reject parallel and model configurations this connector cannot serve."""
    if llm_args.context_parallel_size > 1:
        raise NotImplementedError(
            "The mooncake-store connector does not support context parallelism. "
            "A stored page is keyed by the tokens it holds, but under context "
            "parallelism a rank holds a slice of the sequence rather than whole "
            "blocks of it, so the same key would name different bytes on "
            "different ranks."
        )

    if llm_args.pipeline_parallel_size > 1:
        raise NotImplementedError(
            "The mooncake-store connector does not support pipeline parallelism. "
            "Keys are namespaced per rank, so each stage would store only its own "
            "layers and a prefix hit would require every stage to agree; that path "
            "is untested. Run with tensor parallelism only."
        )

    if getattr(llm_args.kv_cache_config, "enable_swa_scratch_reuse", False):
        raise NotImplementedError(
            "The mooncake-store connector does not support "
            "kv_cache_config.enable_swa_scratch_reuse. The connector saves a "
            "sliding window's pages at the end of a prompt from the page indices "
            "the cache reported when it allocated them, and scratch reuse lets a "
            "prefill write a window block into a page it does not keep."
        )

    spec_config = getattr(llm_args, "speculative_config", None)
    recompute_tail = (
        getattr(spec_config, "context_recompute_tail", 0) if spec_config is not None else 0
    )
    if recompute_tail:
        raise NotImplementedError(
            "The mooncake-store connector does not support "
            "speculative_config.context_recompute_tail. The runtime trims a reserved prefix to leave that tail for "
            "the drafter after the lookup verified the sliding window at the full end, so the window the trimmed "
            "load needs was never checked against the store. Leave context_recompute_tail at 0."
        )

    # Only configs with an index-V cache declare the setting.
    sparse_config = llm_args.sparse_attention_config
    if sparse_config is not None and not getattr(sparse_config, "sparse_disable_index_value", True):
        raise NotImplementedError(
            "The mooncake-store connector requires "
            "sparse_attention_config.sparse_disable_index_value=True. The index-V "
            "cache is a plain tensor outside the KV cache manager's paged pools, "
            "so it is neither described to the connector nor transferred; a "
            "replayed prefix would carry index-K from the store alongside stale "
            "index-V. This is the same restriction disaggregated serving applies."
        )


def _available_host_memory() -> Optional[int]:
    """Host memory the kernel says is available, or `None` if unknowable.

    `MemAvailable` rather than free pages: weights read during startup fill the
    page cache, which a segment allocation reclaims but `SC_AVPHYS_PAGES` does
    not count, understating what is usable by hundreds of gigabytes.
    """
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    # "MemAvailable:   123456 kB"
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    # Kernels before 3.14 and non-Linux hosts do not publish MemAvailable, so
    # fall back to free pages and keep the check conservative.
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError):
        return None


def validate_node_budget(
    config: "MooncakeStoreConnectorConfig",
    *,
    ranks_on_node: Optional[int] = None,
    available_bytes: Optional[int] = None,
    load_threads: int = 1,
) -> None:
    """Reject a segment size this node cannot afford once every rank asks.

    Contribution is per rank but host DRAM is a per-node limit, so eight ranks
    at 160 GiB want 1280 GiB of a node that has around 956 GiB. Unchecked, that
    surfaces as the OOM killer minutes later rather than as a failed
    allocation. A server that lends nothing is still checked: its staging
    buffers are pinned either way.

    Args:
        config: The resolved connector configuration.
        ranks_on_node: Ranks of this server sharing this physical node.
            Defaults to the local MPI communicator's size.
        available_bytes: Host memory to measure against. Defaults to the
            kernel's own figure.
        load_threads: Asynchronous loader threads per loading rank, each
            pinning a staging pool of its own.
    """
    from tensorrt_llm._utils import local_mpi_size

    from .staging import MAX_STAGING_BUFFER_BYTES

    if ranks_on_node is None:
        try:
            ranks_on_node = max(1, local_mpi_size())
        except Exception:  # noqa: BLE001 - a missing communicator must not block startup
            ranks_on_node = 1

    segment_claim = ranks_on_node * max(0, config.global_segment_size)
    # Staging is pinned for the process's lifetime and comes out of the same
    # DRAM, so it belongs in the sum even when no segment is mounted. Loads get
    # a pool for the executor and one per asynchronous loader thread, saves
    # one, and a capacity-only rank opens none.
    directions = (1 + load_threads) * int(config.role.loads) + int(config.role.saves)
    staging_claim = ranks_on_node * directions * MAX_STAGING_BUFFER_BYTES

    claimed = segment_claim + staging_claim
    if claimed <= 0:
        return

    if available_bytes is None:
        available_bytes = _available_host_memory()
    if available_bytes is None:
        logger.warning(
            "mooncake-store: cannot read this node's available memory, so the "
            f"{claimed / _GIB:.1f} GiB the {ranks_on_node} rank(s) here claim "
            "for the pool is not checked against it."
        )
        return

    budget = available_bytes - NODE_BUDGET_RESERVE_BYTES
    if claimed <= budget:
        logger.warning(
            f"mooncake-store: {ranks_on_node} rank(s) on this node will claim "
            f"{claimed / _GIB:.1f} GiB of host memory for the pool, within the "
            f"{budget / _GIB:.1f} GiB available after a "
            f"{NODE_BUDGET_RESERVE_BYTES / _GIB:.0f} GiB reserve."
        )
        return

    affordable = (budget - staging_claim) // ranks_on_node
    if affordable > 0:
        remedy = (
            f"Lower segment_size to at most {affordable / _GIB:.1f} GiB, or run "
            f"fewer ranks per node."
        )
    else:
        remedy = (
            f"Lending nothing would not be enough: the {staging_claim / _GIB:.1f} "
            f"GiB of pinned staging buffers alone overruns the budget. Run fewer "
            f"ranks per node."
        )
    raise ValueError(
        f"mooncake-store: this node cannot afford what the pool is configured "
        f"to claim from it. {ranks_on_node} rank(s) here would lend "
        f"{config.global_segment_size / _GIB:.1f} GiB each and pin "
        f"{staging_claim / _GIB:.1f} GiB of staging between them, "
        f"{claimed / _GIB:.1f} GiB in total, but only "
        f"{available_bytes / _GIB:.1f} GiB is available and "
        f"{NODE_BUDGET_RESERVE_BYTES / _GIB:.0f} GiB of that is reserved for "
        f"weights and the runtime's own host buffers. {remedy} "
        f"Contribution is per rank, so a server's demand on its node grows "
        f"with its parallelism."
    )


def validate_layout(layout: "KvCacheLayout") -> None:
    """Reject KV cache geometries this connector cannot key correctly.

    Sliding-window groups are accepted. A page's bytes are fixed by the tokens
    up to and including its block, which its key already chains, so a stored
    page is never wrong for a request that shares the prefix. What a window
    changes is which pages exist: the scheduler saves a window's pages only at
    the end of each prompt, and a lookup resumes only where those are present.
    """
    if not layout.groups:
        raise ValueError(
            "The KV cache layout describes no layer groups, so there is nothing "
            "for the mooncake-store connector to transfer."
        )
