# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 head GEMV: ``x @ weight^T`` for at most 8 bf16 rows and a large weight (an lm_head vocabulary shard) on a
persistent CuTe DSL kernel, over a caller-owned :class:`K3HeadGemvWorkspace`.

``K3HeadGemvWorkspace`` is re-exported here so that a target creates it through the catalog: once per weight shape
and schedule, eagerly, before any CUDA-graph capture.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

# Importing the op module also registers torch.ops.trtllm.k3_head_gemv.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_head_gemv.op import (
    K3HeadGemvWorkspace,
)

__all__ = ["K3HeadGemvWorkspace", "k3_head_gemv"]


class K3HeadGemv(OpWrapper):
    """Return ``x @ weight^T`` as bf16 ``[M, N]`` (fp32 accumulation, one bf16 rounding) in one k3_head_gemv call over
    ``workspace``, whose schedule and ``chunk_tiles`` the call takes. The calls that share a workspace must be ordered
    on one stream."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 LM-head shard, stream-K workspace across all decode row counts",
            spec=dict(n=10240, k=7168, weight_index=0, ms=range(1, 9), keep_tiles=(40, 80)),
        ),
        Cell(
            why="second head weight of the same shape sharing the workspace layout",
            spec=dict(n=10240, k=7168, weight_index=1, ms=range(1, 9), keep_tiles=(40, 80)),
        ),
    )

    note = """
    Each cell covers M=1..8, ring=6, prefetch=16 and keep_tiles=0, plus
    keep_tiles=40,80 at M=1,8, with x~N(0,1) and weight~0.02*N(0,1)
    rounded to bf16. The shape is K3's 163840-row LM head at TP16.
    Dynamic scheduling is accepted but not certified.

    Stream-K cuts T=(N/128)*(K/128) tile-major items into G=min(SMs,T)
    contiguous CTA ranges [c*T/G,(c+1)*T/G). An unsplit tile accumulates in
    k order. For split tiles, pieces accumulate their k-tiles in order in
    fp32; pieces after 0 write partials and raise flags, and piece 0 waits,
    adds them in piece order and rounds once to bf16. Dynamic scheduling
    chunks K by chunk_tiles; persistent CTAs claim units from a counter and
    the last unit adds partials in chunk order. Both orders are fixed;
    changing SM count can change stream-K bits. Repeated calls and rows
    across M are bit-identical, not necessarily identical to cuBLAS.
    Bias, logit processing, fp32 conversion and vocabulary gather are
    outside. The result is the first M rows of a new [8,N] buffer, whose
    eight rows are all written with zero-filled activation rows past M.

    K3HeadGemvWorkspace, re-exported here from kernels/k3_head_gemv/op.py,
    is a caller-owned frozen dataclass: n_out, k_in, schedule, chunk_tiles
    and three tensors. For tiles=N/128 and
    P=streamk_max_pieces(N,K,G), stream-K partials hold tiles*P*128*8 fp32
    values; slot (t,p>0) holds [128 rows][8 tokens], and p=0 is unused.
    Flags hold tiles*P int32 words raised by pieces and cleared by the
    finalizer. Claim is one int32 word, unused by stream-K. At 148/152 SMs
    the certified shape has 80 tiles, 56 k-tiles, P=3, 240 flag words
    (960 B) and 245760 partial floats (960 KiB), independent of M.
    Dynamic workspaces hold every (tile,chunk) partial, per-tile unit counts
    and the unit counter.

    The target creates one per weight shape/schedule in post_load_weights
    with create(n_out,k_in,device,schedule="streamk",chunk_tiles=0), eagerly
    outside capture. Creation allocates uninitialized partials and zeroes
    flags/claim on the current stream; the first call must follow that
    initialization. It is single-GPU, not collective, reads no environment
    variable, rejects capture with RuntimeError and unknown schedules with
    ValueError. Sizes depend on device SM count.

    Only calls of that shape, schedule and device may share the object;
    any M, weight of that shape, keep_tiles, ring or prefetch may use it.
    Calls must not overlap: order them on one stream or explicitly order
    different streams with synchronization/events. PDL accesses workspace
    only after griddepcontrol.wait. Graph replays may alternate with eager
    calls on the same stream. Concurrent users can read another launch's
    partial or hang on a flag another launch cleared. Every flag must be
    zero at launch; dynamic unit counts and claim must also be zero.
    Partials are written before use and need no clearing. Finalizers reset
    flags; the dynamic last claim resets its counter, re-arming the workspace
    regardless of M. A launch that never completes can leave raised words:
    recreate the workspace. The layout depends on (N,K,schedule,SMs), never
    earlier M. Buffer dtype/device/size mismatches are loudly rejected.
    Tests retain dipping/growing M, alternating weights and workspaces,
    graph/eager interleaving, zeroed-word checks and creation/call refusals.

    The current stream, device SM count and TRTLLM_ENABLE_PDL (default 1)
    are consumed. Result-neutral process compile keys are
    (N,K,ring,G,P,prefetch,PDL) for stream-K and
    (N,K,chunk_tiles,ring,G,PDL) for dynamic; M and keep_tiles are runtime.
    Warm keys eagerly before capture or RuntimeError is raised. cutlass
    and cuda.bindings must be importable. prefetch must be nonnegative.
    Under PDL, initial ring loads and prefetch k-tiles enter before the
    dependency wait, so weight must not be written by preceding in-flight
    work. x, workspace and y accesses follow the wait. Stream-K releases
    dependents after issuing all loads, dynamic after its last claim;
    dependents must wait before reading y. Tiles below keep_tiles use
    normal L2 priority, the rest EVICT_FIRST; this changes no bits.
    The grid has G persistent 256-thread CTAs (dynamic: one per SM);
    stream-K uses at most one CTA per item so no empty range can omit a
    required flag, including smaller, uncertified weights.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        workspace: K3HeadGemvWorkspace,
        keep_tiles: int = 0,
        ring: int = 6,
        prefetch: int = 16,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_head_gemv(
            x,
            weight,
            workspace.partials,
            workspace.flags,
            workspace.claim,
            keep_tiles,
            workspace.chunk_tiles,
            ring,
            workspace.schedule,
            prefetch,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        workspace: K3HeadGemvWorkspace,
        keep_tiles: int = 0,
        ring: int = 6,
        prefetch: int = 16,
    ) -> torch.Tensor:
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).bfloat16()

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        workspace: K3HeadGemvWorkspace,
        keep_tiles: int = 0,
        ring: int = 6,
        prefetch: int = 16,
    ) -> None:
        assert x.data_ptr() % 16 == 0, "x must start at a 16-byte-aligned address"
        assert weight.data_ptr() % 16 == 0, "weight must start at a 16-byte-aligned address"
        assert weight.device == x.device, "weight must be on x's device"


k3_head_gemv = K3HeadGemv()
