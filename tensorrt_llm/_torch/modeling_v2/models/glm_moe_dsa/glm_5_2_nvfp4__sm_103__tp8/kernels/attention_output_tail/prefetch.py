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
"""Off-path L2 prefetch of the attention_output_tail weights (``attention_output_tail_prefetch``).

At M = 6 the tail (W_UV + local o_proj) is a 26 MiB weight stream (``w_o``
24 MiB, ``w_uv`` 2 MiB per rank) that starts only after the attention kernel
has produced the latent. The attention region before it (absorption BMM,
RoPE / FP8 quantization / cache append, top-k bridge, sparse attention) is
latency-bound and leaves HBM mostly idle, and it lasts longer than the
stream. ``WeightPrefetch`` uses that window: when the attention of an armed
layer starts, it forks a private auxiliary stream from the caller's stream,
launches a small kernel there that issues ``cp.async.bulk.prefetch.L2`` over
the tail's weights, and records an event that the caller's stream waits on
later in the same layer (``join``). The tail kernels then read their weights
from L2.

A prefetch has no architectural effect: it neither reads nor writes any
tensor the model observes, so it needs no producer/consumer protocol and no
memory-order contract with the attention kernels, and the results are
identical with and without it. The fork and the join only add graph edges;
``join`` must run before the layer's forward returns so that a CUDA-graph
capture always sees the auxiliary stream rejoin the capturing stream.

The kernel is a fire-and-forget issuer: ``ctas`` CTAs of ``threads`` threads
each issue one ``chunk``-byte prefetch per iteration over the concatenated
weight ranges, optionally pausing ``gap_ns`` between iterations to pace the
traffic.
"""

import functools
from dataclasses import dataclass
from typing import List, Sequence

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class PrefetchConfig:
    """Launch parameters of the L2 prefetch kernel."""

    chunk: int = 4096
    """Bytes per ``cp.async.bulk.prefetch.L2`` instruction (multiple of 16)."""
    ctas: int = 52
    """CTAs of the prefetch kernel."""
    threads: int = 32
    """Threads per CTA (each issues one prefetch per iteration)."""
    gap_ns: int = 0
    """Pause between iterations of one thread (0: issue back to back)."""


@triton.jit
def _bulk_prefetch_l2(addr, nbytes):
    tl.inline_asm_elementwise(
        "cp.async.bulk.prefetch.L2.global [$1], $2; // dummy $0",
        "=r,l,r",
        [addr, nbytes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _nanosleep(ns):
    tl.inline_asm_elementwise(
        "nanosleep.u32 $1; // dummy $0",
        "=r,r",
        [ns],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _l2_prefetch_kernel(
    addr0,
    chunks0,
    addr1,
    chunks1,
    addr2,
    chunks2,
    CHUNK: tl.constexpr,
    NPROG: tl.constexpr,
    NT: tl.constexpr,
    ITERS: tl.constexpr,
    GAP_NS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, NT).to(tl.int64)
    total = chunks0 + chunks1 + chunks2
    for it in tl.static_range(ITERS):
        c = (it * NPROG + pid) * NT + lanes
        # Lanes past the end re-issue chunk 0, which is harmless.
        c = tl.where(c < total, c, 0)
        c12 = c - chunks0
        addr = tl.where(
            c < chunks0,
            addr0 + c * CHUNK,
            tl.where(c12 < chunks1, addr1 + c12 * CHUNK, addr2 + (c12 - chunks1) * CHUNK),
        )
        _bulk_prefetch_l2(addr, tl.full([NT], CHUNK, tl.int32))
        if GAP_NS > 0:
            _nanosleep(tl.full([NT], GAP_NS, tl.int32))


def check_prefetch_tensors(tensors: Sequence[torch.Tensor], chunk: int) -> str:
    """Return why ``tensors`` cannot be prefetched in ``chunk`` pieces, or ''."""
    if not 1 <= len(tensors) <= 3:
        return "expects one to three weight tensors"
    if chunk % 16 or chunk <= 0:
        return "chunk is not a positive multiple of 16 bytes"
    for t in tensors:
        if not t.is_cuda or not t.is_contiguous():
            return "weight is not a contiguous CUDA tensor"
        nbytes = t.numel() * t.element_size()
        if t.data_ptr() % 16 or nbytes % chunk:
            return "weight size or alignment is not a chunk multiple"
    return ""


def launch_prefetch(
    tensors: Sequence[torch.Tensor], cfg: PrefetchConfig = PrefetchConfig()
) -> None:
    """Issue L2 prefetches of one to three contiguous tensors on the current
    stream."""
    regions = [(t.data_ptr(), t.numel() * t.element_size() // cfg.chunk) for t in tensors]
    regions += [(regions[0][0], 0)] * (3 - len(regions))
    per_iter = cfg.ctas * cfg.threads
    iters = -(-sum(n for _, n in regions) // per_iter)
    _l2_prefetch_kernel[(cfg.ctas,)](
        regions[0][0],
        regions[0][1],
        regions[1][0],
        regions[1][1],
        regions[2][0],
        regions[2][1],
        CHUNK=cfg.chunk,
        NPROG=cfg.ctas,
        NT=cfg.threads,
        ITERS=iters,
        GAP_NS=cfg.gap_ns,
        num_warps=max(1, cfg.threads // 32),
        num_stages=1,
    )


@functools.lru_cache(maxsize=None)
def _aux_stream(device_index: int) -> torch.cuda.Stream:
    # One private stream per device for the package; every fork is joined
    # within the same layer, so layers never overlap on it.
    return torch.cuda.Stream(device=device_index)


class WeightPrefetch:
    """Fork/join of the off-path weight prefetch of one tail (layer, rank).

    Owns its own fork and completion events, so no state is shared between
    layers beyond the device's auxiliary stream.
    """

    def __init__(self, cfg: PrefetchConfig = PrefetchConfig()):
        self.cfg = cfg
        self._fork_event = torch.cuda.Event()
        self._done_event = torch.cuda.Event()
        self._forked = False
        self.forks = 0

    @property
    def forked(self) -> bool:
        return self._forked

    def fork(self, tensors: List[torch.Tensor]) -> None:
        """Launch the prefetch of ``tensors`` on the auxiliary stream."""
        if self._forked:
            raise RuntimeError(
                "[glm_moe_dsa] attention_output_tail_prefetch: prefetch forked twice without a join"
            )
        caller = torch.cuda.current_stream()
        aux = _aux_stream(caller.device.index)
        self._fork_event.record(caller)
        aux.wait_event(self._fork_event)
        with torch.cuda.stream(aux):
            launch_prefetch(tensors, self.cfg)
            self._done_event.record(aux)
        self._forked = True
        self.forks += 1

    def join(self) -> None:
        """Make the caller's stream wait for the prefetch kernel (no-op if
        nothing was forked)."""
        if not self._forked:
            return
        torch.cuda.current_stream().wait_event(self._done_event)
        self._forked = False


__all__: List[str] = [
    "PrefetchConfig",
    "WeightPrefetch",
    "check_prefetch_tensors",
    "launch_prefetch",
]
