# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-forward recurrent-state snapshots of the GDN layers.

With hybrid block reuse and snapshot offsets from the prompt end, a context
chunk that ends at the last snapshot point P runs through the prompt end L
instead, saving the extra iteration of the tail [P, L). The GDN layers leave
the state at P in the request's live slot and stage the state at L in a row of
their own; after the forward the cache manager commits the request up to P and
copies the staged state back (``InforwardSnapshots.finish``).

The staging rows (the requests per batch that can take the path, at most
``max_batch_size`` and ``_INFORWARD_SNAPSHOT_MAX_ROWS``, within
``_INFORWARD_SNAPSHOT_STAGING_BYTES``) are allocated after the pools, outside
the GPU cache quota.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

_INFORWARD_SNAPSHOT_MAX_ROWS = 8
_INFORWARD_SNAPSHOT_STAGING_BYTES = 512 << 20


class InforwardSnapshotEntry(NamedTuple):
    """One request's in-forward snapshot in this step's context chunk."""

    row: int  # staging row
    point: int  # snapshot point P
    num_before: int  # chunk tokens before P
    num_after: int  # chunk tokens from P to the prompt end


class InforwardSnapshots:
    """The staging rows and this step's plan of one GDN hybrid cache manager.

    The manager (``Qwen35HybridCacheManager``) owns one instance and hands
    itself to the methods that read its recurrent-state pools and caches.
    """

    def __init__(self) -> None:
        # Staging rows per batch; 0 when the path does not apply.
        self.rows = 0
        # The longest tail, and the fewest chunk tokens before P (the
        # convolution window at P must lie inside the chunk).
        self.max_tail = 0
        self.min_main = 0
        # This step's plan {request id: entry}; replaced, never mutated in place.
        self.plan: dict[int, InforwardSnapshotEntry] = {}
        # [layer, row, ...]: one dense tensor per layer, written by the GDN
        # layer with index_copy_.
        self._stage_ssm: torch.Tensor | None = None
        self._stage_conv: torch.Tensor | None = None

    @staticmethod
    def max_tail_of(manager: MambaHybridCacheManagerV2) -> int:
        """The longest tail the in-forward snapshots cover (the largest
        ``mamba_state_config.additional_snapshot_offsets_from_end``)."""
        return max(
            manager.kv_cache_config.mamba_state_config.additional_snapshot_offsets_from_end,
            default=0,
        )

    @classmethod
    def layout(cls, manager: MambaHybridCacheManagerV2) -> tuple[int, int]:
        """(staging rows, bytes per row) of the in-forward snapshots; no rows
        where the path does not apply.

        The GDN layers implement the split with the q_k_v convolution layout.
        Every rank of a server must schedule the same chunks, so PP stages
        without local GDN layers rule the path out.
        """
        row_bytes = manager.local_num_mamba_layers * (manager.ssm_bytes + manager.conv_bytes)
        if not (
            manager._state_layout.conv_state_layout == "q_k_v"
            and cls.max_tail_of(manager) > 0
            and manager.kv_cache_config.enable_block_reuse
            and not manager.is_draft
            and manager.mapping.pp_size == 1
            and row_bytes > 0
        ):
            return 0, row_bytes
        rows = max(
            1,
            min(
                manager.max_batch_size,
                _INFORWARD_SNAPSHOT_MAX_ROWS,
                _INFORWARD_SNAPSHOT_STAGING_BYTES // row_bytes,
            ),
        )
        return rows, row_bytes

    def setup(self, manager: MambaHybridCacheManagerV2) -> None:
        """Allocate the staging rows when the path applies."""
        rows, row_bytes = self.layout(manager)
        if rows == 0 or not manager.enable_block_reuse:
            return
        layers = manager.local_num_mamba_layers
        max_tail = self.max_tail_of(manager)
        device = manager.all_ssm_states[0].device
        self._stage_ssm = torch.empty(
            [layers, rows] + manager.ssm_state_shape,
            dtype=manager.ssm_state_dtype,
            device=device,
        )
        self._stage_conv = torch.empty(
            [layers, rows] + manager.conv_state_shape,
            dtype=manager.conv_state_dtype,
            device=device,
        )
        self.rows = rows
        self.max_tail = max_tail
        self.min_main = max(1, manager.conv_state_shape[-1])
        logger.info(
            f"In-forward Mamba snapshots: {rows} staging rows "
            f"({rows * row_bytes / 2**20:.0f} MiB), tail <= {max_tail} tokens"
        )

    def stage_views(self, layer_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A local GDN layer's staging rows.

        Args:
            layer_offset: The layer's offset among the local recurrent layers.

        Returns:
            SSM ``[rows, H, V, K]`` and conv ``[rows, conv_dim, d_conv - 1]``.
        """
        return self._stage_ssm[layer_offset], self._stage_conv[layer_offset]

    def plan_batch(self, scheduled_batch: ScheduledRequests) -> None:
        """Turn the scheduler's in-forward snapshot marks into this step's plan."""
        plan: dict[int, InforwardSnapshotEntry] = {}
        context_only = not scheduled_batch.generation_requests
        for req in scheduled_batch.context_requests:
            point = req.py_mamba_inforward_snapshot_point
            if point is None:
                continue
            req.py_mamba_inforward_snapshot_point = None
            pos = req.context_current_position
            end = pos + req.context_chunk_size
            if (
                context_only
                and self.rows > len(plan)
                and not req.is_dummy_request
                and pos + self.min_main <= point < end
                and end == req.prompt_len
                and end - point <= self.max_tail
                and req.expect_snapshot_points
                and point == req.expect_snapshot_points[-1]
            ):
                plan[req.py_request_id] = InforwardSnapshotEntry(
                    len(plan), point, point - pos, end - point
                )
            else:
                # The scheduler checks the same conditions, so this is not
                # expected. The chunk still runs through the prompt end, so no
                # snapshot at P is recorded for this request.
                logger.warning(
                    f"Request {req.py_request_id} "
                    f"dropped from the in-forward snapshot plan (P={point}, "
                    f"chunk [{pos}, {end}), prompt {req.prompt_len}, "
                    f"context-only batch {context_only})"
                )
        self.plan = plan

    def finish(
        self, manager: MambaHybridCacheManagerV2, scheduled_batch: ScheduledRequests
    ) -> None:
        """Commit each planned request up to P from its live state, then copy its staged prompt-end state back.

        Called on the execution stream right after the forward, so the copies
        precede the batch's sampler event.
        """
        plan = self.plan
        if not plan:
            return
        self.plan = {}
        dst, src = [], []
        for request in scheduled_batch.context_requests:
            entry = plan.get(request.py_request_id)
            kv_cache = manager.kv_cache_map.get(request.py_request_id)
            if entry is None or kv_cache is None or not kv_cache.is_active:
                continue
            point = entry.point
            if (
                not request.is_dummy_request
                and kv_cache.num_committed_tokens < point
                and kv_cache.history_length <= point
            ):
                if kv_cache.history_length < point and not kv_cache.resize(None, point):
                    raise ValueError(
                        "Failed to resize history length of V2 Mamba cache "
                        f"for request {request.py_request_id} to the in-forward "
                        f"snapshot point {point}"
                    )
                kv_cache.commit(
                    manager._augment_tokens_for_block_reuse(
                        manager._reuse_token_source(request),
                        request,
                        start=kv_cache.num_committed_tokens,
                        end=point,
                    )
                )
                if manager._branch_snapshot_points.get(request.py_request_id) == point:
                    manager._branch_snapshots_taken_total += 1
            slot = kv_cache.get_ssm_block_base_index(manager.ssm_layer_group_id)
            if slot < 0:
                raise RuntimeError(
                    f"In-forward Mamba snapshot: request {request.py_request_id} lost its live SSM slot"
                )
            dst += [state[slot] for state in manager.all_ssm_states]
            dst += [state[slot] for state in manager.all_conv_states]
            src += list(self._stage_ssm[:, entry.row].unbind(0))
            src += list(self._stage_conv[:, entry.row].unbind(0))
        if dst:
            with torch.cuda.stream(manager._stream):
                torch._foreach_copy_(dst, src, non_blocking=True)

    def shutdown(self) -> None:
        self.rows = 0
        self.plan = {}
        self._stage_ssm = None
        self._stage_conv = None


__all__ = [
    "InforwardSnapshotEntry",
    "InforwardSnapshots",
]
