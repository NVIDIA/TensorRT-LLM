# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's MLA decode attention over the paged latent cache, with v_b and the output gate in the same launch, over
a caller-owned :class:`K3MlaAttnWorkspace`; also the plain forms (the attention output itself)."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_mla import (
    op as _k3_mla_op,  # noqa: F401 — registers the ops
)

from .k3_mla_attn_workspace import K3MlaAttnWorkspace


def _attention_reference(q, pool, row_stride, page_table, page_offset, seq_len, softmax_scale):
    tokens = q.shape[0] // seq_len.numel()
    heads = q.shape[1] // 576
    table = page_table.reshape(1, -1) if page_table.ndim == 1 else page_table
    outs = []
    with true_fp32_matmul():
        for i, length in enumerate(seq_len.tolist()):
            positions = torch.arange(length, device=q.device)
            indices = (table[i, positions // 64].long() + page_offset) * 64 + positions % 64
            kv = pool.view(-1, row_stride)[indices, :576].float()
            qi = q[i * tokens : (i + 1) * tokens].view(tokens, heads, 576).float()
            scores = torch.einsum("thd,ld->thl", qi, kv) * softmax_scale
            limit = length - tokens + torch.arange(tokens, device=q.device)
            hidden = positions[None, :] > limit[:, None]
            scores = scores.masked_fill(hidden[:, None, :], float("-inf"))
            outs.append(torch.einsum("thl,ld->thd", torch.softmax(scores, dim=-1), kv[:, :512]))
    return torch.cat(outs)


class K3MlaAttnVbOut(OpWrapper):
    """Write ``out`` [M, heads * 128] = per head ``bf16(bf16(o) @ w_vb[h]^T)`` (times the gate's sigmoid with
    ``gate``) for the decode attention o of R <= 8 requests of T <= 8 tokens over their pages of ``pool``. Uses
    ``workspace``'s partial slots and, past 7 clusters, advances its counters."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 eight-token verification with three clusters",
            spec=dict(num_requests=3, tokens=8, heads=6),
        ),
        Cell(
            why="TP16 eight-request decode crossing into no-cluster mode",
            spec=dict(num_requests=8, tokens=1, heads=6),
        ),
        Cell(
            why="TP4 verification with four head groups and eight clusters",
            spec=dict(num_requests=2, tokens=8, heads=24),
        ),
    )

    note = """
    q is request-major fused_q from k3_mla_qkv, which must append the step's
    cache rows first. The attention, v_b and output gate are fused; the query
    path/cache append and final output projection are outside. gate already
    contains sigmoid values: this op only multiplies. The gated output is
    bit-identical to bf16(plain * gate). k3_mla_attn_out writes the bf16
    [M, heads * 512] attention itself, and k3_mla_attn returns identical bits
    in a new tensor at page offset zero, suitable for a one-layer pool or
    layer zero of an interleaved pool.

    q, w_vb and out are contiguous bf16 CUDA tensors, respectively
    [R*T, heads*576], [heads,128,512], [R*T,heads*128]. The flat bf16 pool
    has row_stride >=576 divisible by 8. The int32 page table is [R,W] with
    unit column stride and arbitrary row stride >=W, or [W] for R=1;
    seq_len is contiguous int32 [R]. gate is None or bf16 [R*T,C] with unit
    column stride, any row stride, and 0 <= gate_col0 <= C-heads*128.
    The kernel supports SM100/SM103; only SM100 has certification.
    Cells cover cached lengths 64..4100, page crossings and multiple tiles
    per CTA with a real KVCacheManager and a 1e-2 per-request relative bound.

    One cluster of 16 CTAs handles each request/group of six heads, splitting
    KV over 128-row tiles. Softmax is fp32, each partial O/l is fp16, and
    the 16 partials merge in a fixed fp32 order. Above CLUSTER_WAVE=7
    clusters, no-cluster mode uses workspace exchange if all CTAs fit
    simultaneously, one per SM (at most nine clusters on 148 SMs); otherwise
    clusters launch in multiple waves. Both modes merge the same values in
    the same order, and request rows match their single-request calls.

    K3MlaAttnWorkspace is caller-owned, one per device and head-group count
    (heads/6: one at TP16, four at TP4), shared by all MLA layers of that shape.
    This call only reads the KV manager's pool. The workspace is one fp16
    buffer of attn_workspace_elems(groups): partial slots for eight requests
    x groups x 16 CTAs, 64 KiB per CTA (8 MiB per group), then 48 KiB/group
    of fp32 (m,l) exchange and two int32 arrival counters per request/group,
    each on its own 128-byte line (2 KiB/group). Total: 8,439,808 bytes/group,
    33,759,232 at TP4, independent of R and T.

    The target creates it in post_load_weights with create(device, groups).
    Creation allocates eagerly and refuses CUDA-graph capture; it zeroes the
    counters, leaves partials as allocated, then synchronizes the device so
    the result is ready on any stream. All three attention forms may share
    it. Separate workspaces are independent: interleaved three-step calls
    match fresh workspaces bit for bit and count only their own launches.

    Calls sharing one workspace must run one at a time. All workspace
    accesses follow the grid-dependency wait, so same-stream calls may follow
    directly if every intervening kernel waits on its predecessor or launches
    without PDL, as decode kernels do. Overlapping streams or graph/eager
    calls overwrite partials and counters and are outside the contract.
    A later launch reads only counters, only in no-cluster mode, after its
    wait and before arrival; it waits for 16 arrivals past count & ~15.
    Partials and exchange are written/read within the call; poisoning them
    with NaNs preserves output bits. Before its wait, a launch reads only
    page tables, lengths and pages before the page containing L_i-T; q and
    that page onward are read after the wait.

    Never re-arm: each no-cluster launch adds 16 to both counters of its
    requests/head groups, leaving other counters untouched; cluster launches
    leave all counters untouched. Signed-difference comparisons handle int32
    wrap: starts at 2^31-16 and -16 match zeroed counters. Launches require
    multiples of 16, which create() and complete launches leave. Certification
    runs three layers x two eager steps on one shared workspace, captures a
    step's three calls and replays two steps with rewritten q/prepared metadata,
    comparing fresh-workspace bits, the reference, and final counters.
    Wrong head-group layout, dtype, device or size raises before launch;
    negative controls preserve out and workspace for four-group, fp32 and
    eight-elements-short buffers supplied to a one-group call.

    k3_mla_decode_view supplies page tables/lengths from prepared
    kv_cache_block_offsets and kv_lens_cuda_runtime, the layer page_offset,
    pool token row_stride, and scale 1/(sqrt(qk_nope+qk_rope)*q_scaling).
    Shape/layout errors (R,T <=8, heads divisible by six, tensors above)
    already raise ValueError before launch. First calls compile outside graph
    capture. is_valid checks lengths, not whether an earlier call actually
    wrote the rows or whether another stream is using the workspace.
    tests/unittest/_torch/modeling_v2/models/kimi_k3_vl/test_k3_mla_attn.py additionally
    covers every split of up to eight tokens, R x 8 verification, stock
    cute_dsl_mla_decode_fp16_blackwell parity, 640-element interleaved rows,
    poison/counter-wrap checks, refusals and optional K3_BASE_TRTLLM identity.
    """

    def __call__(
        self,
        q: torch.Tensor,
        pool: torch.Tensor,
        row_stride: int,
        page_table: torch.Tensor,
        page_offset: int,
        seq_len: torch.Tensor,
        softmax_scale: float,
        w_vb: torch.Tensor,
        out: torch.Tensor,
        workspace: K3MlaAttnWorkspace,
        gate: Optional[torch.Tensor] = None,
        gate_col0: int = 0,
    ) -> None:
        torch.ops.trtllm.k3_mla_attn_vb_out(
            q,
            pool,
            row_stride,
            page_table,
            page_offset,
            seq_len,
            softmax_scale,
            w_vb,
            out,
            workspace.buffer,
            gate,
            gate_col0,
        )

    def reference(
        self,
        q: torch.Tensor,
        pool: torch.Tensor,
        row_stride: int,
        page_table: torch.Tensor,
        page_offset: int,
        seq_len: torch.Tensor,
        softmax_scale: float,
        w_vb: torch.Tensor,
        out: torch.Tensor,
        workspace: K3MlaAttnWorkspace,
        gate: Optional[torch.Tensor] = None,
        gate_col0: int = 0,
    ) -> None:
        o = (
            _attention_reference(
                q, pool, row_stride, page_table, page_offset, seq_len, softmax_scale
            )
            .bfloat16()
            .float()
        )
        with true_fp32_matmul():
            y = torch.einsum("thc,hvc->thv", o, w_vb.float()).reshape(out.shape).bfloat16()
        if gate is not None:
            y = (y.float() * gate[:, gate_col0 : gate_col0 + out.shape[1]].float()).bfloat16()
        out.copy_(y)

    def is_valid(
        self,
        q: torch.Tensor,
        pool: torch.Tensor,
        row_stride: int,
        page_table: torch.Tensor,
        page_offset: int,
        seq_len: torch.Tensor,
        softmax_scale: float,
        w_vb: torch.Tensor,
        out: torch.Tensor,
        workspace: K3MlaAttnWorkspace,
        gate: Optional[torch.Tensor] = None,
        gate_col0: int = 0,
    ) -> None:
        tokens = q.shape[0] // seq_len.numel()
        assert bool((seq_len >= tokens).all()), "seq_len must include every token of the step"


k3_mla_attn_vb_out = K3MlaAttnVbOut()


def k3_mla_attn_out(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    page_offset: int,
    seq_len: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor,
    workspace: K3MlaAttnWorkspace,
) -> None:
    """Write the attention output itself, ``out`` [M, heads * 512] bf16 (no v_b), over ``workspace``."""
    torch.ops.trtllm.k3_mla_attn_out(
        q, pool, row_stride, page_table, page_offset, seq_len, softmax_scale, out, workspace.buffer
    )


def k3_mla_attn(
    q: torch.Tensor,
    pool: torch.Tensor,
    row_stride: int,
    page_table: torch.Tensor,
    seq_len: torch.Tensor,
    softmax_scale: float,
    workspace: K3MlaAttnWorkspace,
) -> torch.Tensor:
    """Return the attention output [M, heads * 512] bf16 in a new tensor (page offset 0), over ``workspace``."""
    return torch.ops.trtllm.k3_mla_attn(
        q, pool, row_stride, page_table, seq_len, softmax_scale, workspace.buffer
    )
