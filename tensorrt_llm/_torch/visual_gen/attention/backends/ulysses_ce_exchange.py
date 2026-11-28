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
"""Overlap both Ulysses exchanges with full-sequence attention on consecutive head groups."""

import socket
import traceback
from functools import lru_cache
from itertools import accumulate

import torch
import torch.distributed as dist
from cuda.bindings import runtime as cudart

from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div
from tensorrt_llm.visual_gen.args import _normalize_ulysses_head_plan

from ...cuda_graph_runner import in_cuda_graph_runner_scope
from ...parallel.ce_transport import CeTransport
from ...parallel.symm_mem_pool import SymmMemAllocationError
from .interface import AttentionBackend, AttentionTensorLayout

_QUERY_TILE_ROWS = 256


def resolve_ulysses_head_plan(spec: str, local_heads: int) -> tuple[int, ...]:
    """Resolve an expert plan once the model's head count per Ulysses rank is known."""
    h = local_heads
    if h < 1:
        raise ValueError(f"Local head count must be positive, got {h}")
    spec = _normalize_ulysses_head_plan(spec)
    if spec == "whole":
        return (h,)
    if spec == "per_head":
        return (1,) * h
    parts = spec.split(",")
    rest = h - sum(int(part) for part in parts if part != "rest")
    groups = tuple(rest if part == "rest" else int(part) for part in parts)
    if sum(groups) != h or any(width < 1 for width in groups):
        raise ValueError(f"Head group widths must be positive and sum to {h}, got {groups}")
    return groups


@lru_cache(maxsize=256)
def plan_head_groups(
    local_heads: int, sequence: int, batch: int, sm_count: int, spec: str = "auto"
) -> tuple[int, ...]:
    """Choose ordered widths. The wave estimate uses the global sequence and 256 query rows/CTA.

    The planner follows chang-l/TensorRT-LLM#13. A tie on whole waves favors the shortest
    first input and last output transfers, then the fewest attention launches.
    """
    if min(local_heads, sequence, batch, sm_count) < 1:
        raise ValueError(
            "Head count, global sequence length, batch size and SM count must be positive"
        )
    h = local_heads
    spec = spec.strip()
    if spec != "auto":
        return resolve_ulysses_head_plan(spec, h)

    candidates = [(h,), (1,) * h]
    candidates.extend((w,) * (h // w) for w in range(2, h) if h % w == 0)
    candidates.extend((w, h - w) for w in range(1, h // 2 + 1))
    candidates.extend((w, h - 2 * w, w) for w in range(1, (h - 1) // 2 + 1))
    tiles_per_head = batch * ceil_div(sequence, _QUERY_TILE_ROWS)

    def cost(groups: tuple[int, ...]) -> tuple[int, int, int, int]:
        waves = sum(ceil_div(width * tiles_per_head, sm_count) for width in groups)
        return waves, groups[0], groups[-1], len(groups)

    return min(candidates, key=cost)


def _copy_head_stripe(dst: torch.Tensor, src: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Copy [B, S, width, D] head stripes directly between source and final row pitches."""
    batch, rows, width, dim = src.shape
    elem = src.element_size()
    row_bytes = width * dim * elem
    src_base, dst_base = src.data_ptr(), dst.data_ptr()
    # A size-one row axis can legally have any stride, even when CUDA requires pitch >= width.
    src_pitch = max(src.stride(1) * elem, row_bytes)
    dst_pitch = max(dst.stride(1) * elem, row_bytes)
    for b in range(batch):
        CUASSERT(
            cudart.cudaMemcpy2DAsync(
                dst_base + b * dst.stride(0) * elem,
                dst_pitch,
                src_base + b * src.stride(0) * elem,
                src_pitch,
                row_bytes,
                rows,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream.cuda_stream,
            )
        )
    # The producer may drop its last reference while this side stream is still reading.
    src.record_stream(stream)


def _head_contiguous(x: torch.Tensor) -> torch.Tensor:
    # BHSD outputs transposed to BSHD and arbitrary input views may separate adjacent heads.
    # The pitched descriptor below needs the trailing [width, D] span and independent rows.
    if (
        x.stride(3) != 1
        or (x.size(2) > 1 and x.stride(2) != x.size(3))
        or x.stride(1) < x.size(2) * x.size(3)
    ):
        return x.contiguous()
    return x


class HeadChunkedUlysses:
    """One reusable exchange per group. Ranks issue calls in the same order.

    An all-rank host failure drains the interrupted eager exchange before control
    returns to the caller. A lost or unmatched peer requires a group restart.
    """

    @torch.inference_mode(False)
    def __init__(self, group: dist.ProcessGroup, device: torch.device, plan: str = "auto"):
        self.group = group
        self.device = device
        self.plan = plan.strip()
        self.size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.transport: CeTransport | None = None
        self._tried_transport = False
        self._captured = False
        self._eager_stream: torch.cuda.Stream | None = None
        # Check once before a peer-memory rendezvous; ranks with different SM counts could
        # otherwise select different head groups and wait on incompatible channels.
        props = torch.cuda.get_device_properties(device)
        local = (
            socket.gethostname(),
            props.major,
            props.minor,
            props.multi_processor_count,
            self.plan,
        )
        members = [local] * self.size
        dist.all_gather_object(members, local, group=group)
        if any(member[-1] != self.plan for member in members):
            raise ValueError("All ranks in a Ulysses group must use the same ulysses_head_plan")
        self._sms = (
            props.multi_processor_count
            if len(set(members)) == 1 and local[1:3] in ((10, 0), (10, 3))
            else 0
        )
        if self._sms and self.plan != "auto":
            # The expert split applies only on CE-capable fabrics. One input, one output
            # and the completion barrier are the minimum channels for any such exchange.
            self._create_transport(3)

    def _create_transport(self, channels: int) -> None:
        self.transport = CeTransport.create(self.group, self.device, min_channels=channels)
        self._tried_transport = True

    def _reserve(self, slot_bytes: int, channels: int, capturing: bool) -> bool:
        if not self._tried_transport:
            if capturing:
                return False
            self._create_transport(channels)
        if self.transport is None:
            return False
        pool = self.transport.pool
        if pool.reserved and pool.channels < channels:
            return False
        if pool.slot_nbytes < slot_bytes:
            # A captured graph holds the current pool addresses until pipeline teardown.
            if capturing or self._captured:
                return False
            try:
                with torch.inference_mode(False):
                    pool.reserve(slot_bytes, slots=1, min_channels=channels)
            except SymmMemAllocationError as exc:
                logger.warning(
                    f"Ulysses using NCCL after symmetric-memory allocation failed: {exc}"
                )
                self.close()
                return False
        return True

    def close(self) -> None:
        self._tried_transport = True
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        inner_backend: AttentionBackend,
        **kwargs,
    ) -> torch.Tensor | None:
        """Exchange equal BF16 Q/K/V tensors in [B, S_local, H, D] with a BF16-output backend.

        Forward writes each sender's token rows into [3, B, S, group_heads, D]. Reverse writes
        each sender's head columns into [B, S_local, H, D]. Ranks write disjoint slices in both
        directions. The returned tensor owns its storage when the next call reuses the slot.
        """
        if not self._sms or (self._tried_transport and self.transport is None):
            return None
        batch, rows, heads, dim = q.shape
        local_heads = heads // self.size
        sequence = self.size * rows
        groups = plan_head_groups(local_heads, sequence, batch, self._sms, self.plan)
        if self.plan == "auto" and len(groups) == 1:
            return None  # A single FMHA has no neighboring group to overlap.
        offsets = tuple(accumulate(groups, initial=0))
        plane_size = batch * sequence * dim
        output_offset = 3 * local_heads * plane_size
        capturing = torch.cuda.is_current_stream_capturing()
        # A standalone graph can replay on an independent stream, bypassing Python
        # scratch ownership. The VisualGen worker orders managed replays with its forwards.
        if capturing and not in_cuda_graph_runner_scope():
            return None
        compute = torch.cuda.current_stream(self.device)
        if not capturing:
            if self._eager_stream is not None and compute != self._eager_stream:
                compute.wait_stream(self._eager_stream)
            self._eager_stream = compute
        if self.transport is not None and self.transport.interrupted:
            raise RuntimeError(
                "Previous Ulysses exchange could not recover. Restart the process group."
            )
        if not self._reserve(
            4 * local_heads * plane_size * q.element_size(), 2 * len(groups) + 1, capturing
        ):
            return None
        transport = self.transport
        assert transport is not None
        is_hnd = inner_backend.preferred_layout == AttentionTensorLayout.HND
        kwargs.update(batch_size=batch, seq_len=sequence, seq_len_kv=sequence)
        # Every group's Q/K/V is already in its consumer's full-sequence layout.
        slots = [transport.pool.peer_slot(p, 0).view(q.dtype) for p in range(self.size)]
        outputs = [slot.narrow(0, output_offset, q.numel()).view_as(q) for slot in slots]
        inputs = tuple(_head_contiguous(x) for x in (q, k, v))

        def group_input(peer: int, g: int) -> torch.Tensor:
            return (
                slots[peer]
                .narrow(0, 3 * offsets[g] * plane_size, 3 * groups[g] * plane_size)
                .view(3, batch, sequence, groups[g], dim)
            )

        def send_input(g: int) -> tuple[torch.cuda.Event, ...]:
            def copy_to(peer: int) -> None:
                head = peer * local_heads + offsets[g]
                dst = group_input(peer, g).narrow(2, self.rank * rows, rows)
                stream = torch.cuda.current_stream(self.device)
                for src, target in zip(inputs, dst.unbind(0)):
                    block = src.narrow(2, head, groups[g])
                    if peer == self.rank:
                        target.copy_(block)
                    else:
                        _copy_head_stripe(target, block, stream)

            channel = 2 * g
            published = transport.push(
                channel,
                copy_to,
                record_publication=len(transport.peers) > 1 and transport.fused_waits,
            )
            copy_to(self.rank)
            return published

        def send_output(g: int, out: torch.Tensor) -> None:
            out = _head_contiguous(out)
            head = self.rank * local_heads + offsets[g]

            def copy_to(peer: int) -> None:
                src = out.narrow(1, peer * rows, rows)
                dst = outputs[peer].narrow(2, head, groups[g])
                if peer == self.rank:
                    dst.copy_(src)
                else:
                    _copy_head_stripe(dst, src, torch.cuda.current_stream(self.device))

            transport.push(2 * g + 1, copy_to)
            copy_to(self.rank)

        expected = ((src, ch) for ch in range(2 * len(groups)) for src in transport.peers)
        try:
            with transport.boundary(expected=expected):
                published = send_input(0)
                for g in range(len(groups)):
                    following = send_input(g + 1) if g + 1 < len(groups) else ()
                    transport.wait_many((src, 2 * g) for src in transport.peers)
                    # FMHA may occupy every SM. Publish this group's Q/K/V to every peer
                    # before starting it, so a peer's input signal cannot sit behind our FMHA.
                    for event in published:
                        compute.wait_event(event)
                    published = following
                    qkv = group_input(self.rank, g)
                    if is_hnd:
                        qkv = qkv.transpose(2, 3)
                    qg, kg, vg = qkv.unbind(0)
                    out = inner_backend.forward_head_group(q=qg, k=kg, v=vg, **kwargs)
                    send_output(g, out.transpose(1, 2) if is_hnd else out)
                # All returned head groups share one consumer: the final copy out of scratch.
                transport.wait_many(
                    (src, 2 * g + 1) for g in range(len(groups)) for src in transport.peers
                )
                # Releasing the slot lets the next invocation overwrite it.
                output = outputs[self.rank].clone()
        except Exception as exc:
            if not transport.interrupted:
                raise
            if capturing:
                raise RuntimeError(
                    "Ulysses graph capture failed. Restart the process group."
                ) from exc
            # A peer may have reached the completion fence already. Report the original
            # failure before recovery, which then waits until the process-group deadline.
            logger.error(
                f"Rank {self.rank} interrupted its Ulysses exchange:\n{traceback.format_exc()}"
            )
            if not transport.ready():
                raise RuntimeError(
                    "Ulysses exchange could not recover. Restart the process group."
                ) from exc
            # A cached graph can replay directly on another stream after the exception.
            compute.synchronize()
            raise
        self._captured |= capturing
        return output
