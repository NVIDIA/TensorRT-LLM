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
"""The DFlash2 drafter's windowed context in a Kimi K3.1 KV cache manager.

The DFlash2 drafter attends to a sliding window of its context (``dflash2_draft_window``, 1024
positions), so the standalone draft layers the engine appends to a manager keep only that window
(the manager's ``draft_window_size``) instead of the whole history. Shared by the family's KDA + DSA
managers (``KimiK3DSAHybridCacheManager`` and the Rubin ``RubinKdaDsaHybridCacheManager``):
:func:`set_draft_window` and :func:`window_draft_cost` for the construction and the static
estimate, :class:`DraftWindowCacheMixin` for the history the window follows.
"""

from __future__ import annotations

import sys
from dataclasses import replace

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_draft_pages import (
    dflash2_draft_window,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BatchDesc, KVCacheDesc


def set_draft_window(kwargs: dict) -> None:
    """Keep only the DFlash2 drafter's sliding window of its context in the standalone draft layers.

    The engine appends the drafter's layers to the manager (``draft_*`` keyword arguments) when the
    drafter opts into the unified draft cache; the window is the drafter checkpoint's
    (``dflash2_draft_window``), passed as the manager's ``draft_window_size``.
    """
    if kwargs.get("draft_num_layers") and kwargs.get("draft_window_size") is None:
        window = dflash2_draft_window(kwargs.get("spec_config"))
        if window is not None:
            kwargs["draft_window_size"] = window


def window_draft_cost(
    slope: int,
    fixed_cost: int,
    *,
    tokens_per_block: int,
    max_batch_size: int,
    mapping: Mapping,
    **kwargs,
) -> tuple[int, int]:
    """The hybrid estimate with the standalone draft layers costed at the drafter's window.

    The hybrid estimate charges standalone draft layers as full history; with the window set by
    :func:`set_draft_window` they hold the window plus the scratch per request instead, which
    the manager's runtime accounting already uses. Swaps one draft term for the other.
    """
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        _estimate_draft_cache_size_components,
        _get_draft_layer_sizes,
    )

    draft_layer_sizes = _get_draft_layer_sizes(**kwargs)
    if not draft_layer_sizes:
        return slope, fixed_cost
    window = kwargs.get("draft_window_size") or dflash2_draft_window(kwargs.get("spec_config"))
    if window is None:
        return slope, fixed_cost
    helix_cp_size = mapping.cp_size if mapping.has_cp_helix() else 1
    common = dict(
        generation_capacity_headroom=1,
        scratch_tokens=int(kwargs.get("draft_scratch_tokens") or 0),
        helix_cp_size=helix_cp_size // int(kwargs.get("draft_cp_size") or 1),
    )
    _, full_slope, full_fixed = _estimate_draft_cache_size_components(
        draft_layer_sizes, tokens_per_block, **common
    )
    _, window_slope, window_fixed = _estimate_draft_cache_size_components(
        draft_layer_sizes, tokens_per_block, window_size=window, **common
    )
    return (
        slope - full_slope + window_slope,
        fixed_cost + max_batch_size * (window_fixed - full_fixed),
    )


class DraftWindowCacheMixin:
    """The history of a KDA hybrid manager whose standalone draft layers keep a sliding window.

    Ahead of the hybrid manager in the MRO. The draft layers keep only the window before a
    request's history length; a context chunk allocates draft pages for all of its positions, and
    pages that fall out of the window are released when the history length passes them. The
    recurrent-state cache holds the history length at the last committed snapshot point while a
    later point is pending, which with a snapshot at the prompt end is the whole prompt, so the
    draft pages of every chunk since that point would stay locked. These overrides advance the
    history length to each chunk's end and size the draft pages for the windows of every resident
    request plus one step of chunks. A manager without a draft window keeps the plain behavior.
    """

    def _windowed_draft_pages(self) -> bool:
        """Whether this manager holds a sliding window of a standalone drafter's context."""
        return (
            bool(getattr(self, "_draft_layer_sizes", None)) and self.draft_window_size is not None
        )

    def _draft_page_floor_tokens(self) -> int:
        """Draft-page tokens that every resident request's window plus one step of chunks need.

        Each request holds the blocks of its retention window plus up to two partial blocks; a
        step's chunks add ``max_num_tokens``. Leases are counted twice over the batch size, for the
        requests that wait on their KV transfer (disaggregated) or on the next step (overlap
        scheduler) while others are scheduled. Blocks are the manager's ledger blocks: under Helix
        or DKV one spans several pages.
        """
        block = self._ledger_tokens_per_block
        retention_blocks = -(-self.draft_retention_window_size // block)
        per_lease = (retention_blocks + 2) * block
        leases = 2 * self._max_resident_sequences()
        return (self.max_num_tokens or 0) + leases * per_lease

    @override
    def _build_cache_config(self, config):
        """Add a draft-page floor: the windows of every lease plus one step of chunks.

        One request of that many tokens with no history needs exactly as many draft pages, a few
        tens of thousands of tokens of the other layers' pages and one recurrent state.
        """
        config = super()._build_cache_config(config)
        if not self._windowed_draft_pages():
            return config
        floor = BatchDesc([KVCacheDesc(capacity=self._draft_page_floor_tokens(), history_length=0)])
        return replace(config, constraints=[*config.constraints, floor])

    def _advance_history(self, request, kv_cache, history_length: int) -> None:
        if not kv_cache.resize(None, history_length):
            raise ValueError(
                f"Failed to advance the KV cache history of request {request.py_request_id} to "
                f"{history_length} tokens"
            )

    @override
    def try_commit_blocks(self, request, kv_cache=None) -> None:
        """Commit at a snapshot point from a history length that already passed the last commit.

        A commit with recurrent snapshots must start or end at the history length; after
        :meth:`update_context_resources` advanced it past the last commit, a commit at the point
        the context reached ends there.
        """
        if (
            self._windowed_draft_pages()
            and self.enable_block_reuse
            and not request.is_dummy_request
        ):
            cache = (
                kv_cache if kv_cache is not None else self.kv_cache_map.get(request.py_request_id)
            )
            position = request.context_current_position
            if (
                cache is not None
                and cache.is_active
                and position in request.expect_snapshot_points
                and cache.num_committed_tokens < cache.history_length < position
            ):
                self._advance_history(request, cache, position)
        super().try_commit_blocks(request, kv_cache)

    @override
    def update_context_resources(self, scheduled_batch) -> None:
        """After the step's commits, advance each context request's history length to its chunk's
        end, which releases the draft pages that fell out of the window."""
        super().update_context_resources(scheduled_batch)
        if not self._windowed_draft_pages():
            return
        for request in scheduled_batch.context_requests:
            cache = self.kv_cache_map.get(request.py_request_id)
            position = request.context_current_position
            if cache is not None and cache.is_active and cache.history_length < position:
                self._advance_history(request, cache, position)
