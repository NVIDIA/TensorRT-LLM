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
the GPU cache quota. ``TRTLLM_INFORWARD_SNAPSHOT_MAX_ROWS`` (rows) and
``TRTLLM_INFORWARD_SNAPSHOT_STAGING_MIB`` (MiB) override the two caps. Requests
beyond the rows run their tails a step later. Every rank schedules on its own
and the scheduler budgets split rows from these caps, so both must be set
identically on every rank of a server.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import _host_slots_to_device
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

_INFORWARD_SNAPSHOT_MAX_ROWS = 8
_INFORWARD_SNAPSHOT_STAGING_BYTES = 512 << 20
_INFORWARD_SNAPSHOT_MAX_ROWS_ENV = "TRTLLM_INFORWARD_SNAPSHOT_MAX_ROWS"
_INFORWARD_SNAPSHOT_STAGING_MIB_ENV = "TRTLLM_INFORWARD_SNAPSHOT_STAGING_MIB"


def _env_positive_int(name: str, default: int) -> int:
    """A positive int from the environment, or `default` when unset."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from err
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


class InforwardSnapshotEntry(NamedTuple):
    """One request's in-forward snapshot in this step's context chunk."""

    row: int  # staging row
    point: int  # snapshot point P
    num_before: int  # chunk tokens before P
    num_after: int  # chunk tokens from P to the prompt end


class InforwardSnapshots:
    """The staging rows and this step's plan of one GDN hybrid cache manager.

    The manager (an ``InforwardSnapshotHost``) owns one instance and hands
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
        max_rows = _env_positive_int(_INFORWARD_SNAPSHOT_MAX_ROWS_ENV, _INFORWARD_SNAPSHOT_MAX_ROWS)
        staging_mib = _env_positive_int(
            _INFORWARD_SNAPSHOT_STAGING_MIB_ENV, _INFORWARD_SNAPSHOT_STAGING_BYTES >> 20
        )
        rows = max(1, min(manager.max_batch_size, max_rows, (staging_mib << 20) // row_bytes))
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
        self,
        manager: MambaHybridCacheManagerV2,
        scheduled_batch: ScheduledRequests,
        extra_buffers: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
    ) -> None:
        """Commit each planned request up to P from its live state, then copy its staged prompt-end state back.

        Called on the execution stream right after the forward, so the copies
        precede the batch's sampler event. One index_copy_ per staged buffer
        covers every planned request.

        Args:
            manager: The owning cache manager.
            scheduled_batch: The batch the plan was made for.
            extra_buffers: (live pool ``[slots, ...]``, staging ``[rows, ...]``)
                of other model state that shares the GDN slot, restored with the
                SSM and conv states.
        """
        plan = self.plan
        if not plan:
            return
        self.plan = {}
        slots, rows = [], []
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
            slots.append(slot)
            rows.append(entry.row)
        if not slots:
            return
        # (live pool [slots, ...], staging [rows, ...]) of every buffer the snapshot staged.
        buffers = list(zip(manager.all_ssm_states, self._stage_ssm.unbind(0)))
        buffers += list(zip(manager.all_conv_states, self._stage_conv.unbind(0)))
        buffers += extra_buffers
        device = self._stage_ssm.device
        with torch.cuda.stream(manager._stream):
            slot_index = _host_slots_to_device(slots, device)
            # Rows follow batch order, so they are usually 0..n-1 and a slice replaces the gather.
            leading = rows == list(range(len(rows)))
            row_index = None if leading else _host_slots_to_device(rows, device)
            for pool, stage in buffers:
                src = stage[: len(rows)] if leading else stage.index_select(0, row_index)
                pool.index_copy_(0, slot_index, src)

    def shutdown(self) -> None:
        self.rows = 0
        self.plan = {}
        self._stage_ssm = None
        self._stage_conv = None


class InforwardSnapshotHost:
    """The in-forward snapshot API of a hybrid cache manager whose GDN layers implement the split.

    Mixed into a ``MambaHybridCacheManagerV2`` subclass, which creates
    ``_inforward_snapshots`` before the base constructor and calls the hooks:
    ``_setup_inforward_snapshots`` once its recurrent-state pools are bound,
    ``_plan_inforward_snapshots`` in ``prepare_resources``, and
    ``_inforward_snapshots.shutdown`` in ``shutdown``. The executor calls
    ``finish_inforward_snapshots`` after the forward.
    """

    _inforward_snapshots: InforwardSnapshots
    # Whether the model's recurrent layers implement the split; a manager may
    # decide per instance.
    _supports_inforward_snapshot = True

    @property
    def inforward_snapshot_rows(self) -> int:
        """Staging rows per batch of the in-forward snapshots; 0 when the path does not apply."""
        return self._inforward_snapshots.rows

    @property
    def inforward_snapshot_max_tail(self) -> int:
        """The longest prompt tail an in-forward snapshot covers."""
        return self._inforward_snapshots.max_tail

    @property
    def inforward_snapshot_min_main(self) -> int:
        """The fewest chunk tokens before an in-forward snapshot point."""
        return self._inforward_snapshots.min_main

    @property
    def inforward_snapshot_plan(self) -> dict[int, InforwardSnapshotEntry]:
        """This step's in-forward snapshots by request id."""
        return self._inforward_snapshots.plan

    def inforward_snapshot_stage_views(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A layer's staging rows: SSM ``[rows, H, V, K]`` and conv ``[rows, conv_dim, d_conv - 1]``."""
        return self._inforward_snapshots.stage_views(self.mamba_layer_offsets[layer_idx])

    def state_index(self, request_id: int) -> int:
        """The live state slot of a request scheduled in this step."""
        return self._request_id_to_state_index[request_id]

    def finish_inforward_snapshots(self, scheduled_batch: ScheduledRequests) -> None:
        """Commit each planned request up to P from its live state, then copy its staged prompt-end state back.

        Called on the execution stream right after the forward, so the copies
        precede the batch's sampler event.
        """
        if self._inforward_snapshots.plan:
            self._inforward_snapshots.finish(
                self, scheduled_batch, self._inforward_extra_restore_buffers()
            )

    def _setup_inforward_snapshots(self) -> None:
        """Allocate the staging rows when the model implements the split and the path applies."""
        if self._supports_inforward_snapshot and self.local_num_mamba_layers > 0:
            self._inforward_snapshots.setup(self)

    def _plan_inforward_snapshots(self, scheduled_batch: ScheduledRequests) -> None:
        """Plan this step's in-forward snapshots from the scheduler's marks."""
        if self.inforward_snapshot_rows > 0:
            self._inforward_snapshots.plan_batch(scheduled_batch)

    def _inforward_extra_restore_buffers(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(live pool ``[slots, ...]``, staging ``[rows, ...]``) of model state that shares the GDN slot.

        ``finish_inforward_snapshots`` restores these with the SSM and conv states; none by default.
        """
        return []


__all__ = [
    "InforwardSnapshotEntry",
    "InforwardSnapshotHost",
    "InforwardSnapshots",
]
