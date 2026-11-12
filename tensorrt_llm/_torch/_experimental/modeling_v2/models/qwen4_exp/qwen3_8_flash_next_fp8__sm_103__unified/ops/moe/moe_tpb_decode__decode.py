# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode step's routed experts over 128-row intermediate blocks, in one launch of 4- or 8-CTA
clusters (`native_ops.fp8_block_moe_tpb`, CUDA C++ in this target's `csrc/fp8BlockMoeTpb.cu`).

An expert's intermediate (640) is five 128-row blocks and its activation needs nothing outside a
block, so the routed experts are dealt by blocks over the four tep4 ranks: expert e's block b lives
on rank (e + b) % 4, and each rank returns its partial for the TP sum after the layer. A cluster
runs a block's FC1 -> activation -> FC2, sharing the FC1 outputs through distributed shared
memory; partials are summed in cluster order (deterministic).

Operands (`BlockExperts`), block-major, a rank's 640 blocks:
`w1 [640, 256, 2560]` e4m3 (rows: the block's 128 up rows, then its 128 gate rows), `s1 [640, 2, 20]`
fp32, `w2 [640, 2560, 128]` e4m3 (the block's 128 down_proj columns), `s2 [640, 20]` fp32.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ... import native_ops
from .fp8_block_scale_moe_runner import dequantize_block_weight, requantize_1x128

EXPERTS = 512
TOP_K = 10
HIDDEN = 2560
INTER = 640
#: 128-row intermediate blocks an expert has.
EXPERT_BLOCKS = INTER // 128
#: Blocks a rank holds (tep4 round robin: 512 * 5 / 4).
BLOCKS = 640
RANKS = 4
MAX_ROWS = 8


class BlockExperts(NamedTuple):
    """One MoE layer's routed experts at this rank, block-major (see the module docstring)."""

    w1: torch.Tensor
    s1: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    rank: int


#: The shared expert's intermediate, and the work items its TP slice is folded in as (`SharedBlocks`).
SHARED_INTER = 640
SHARED_ITEMS = 5
SHARED_ITEM_COLS = 32


class SharedBlocks(NamedTuple):
    """This rank's TP4 slice of the bf16 shared expert, folded into the launch (3..8 rows): `x` the
    MoE site's `[T, 2560]` bf16 rows, `gu` `[5, 64, 2560]` the slice's gate / up rows (item sb = columns
    `[160 rank + 32 sb, +32)` as 8 groups of 4 columns, each `[gate 4 ; up 4]`), `gl` the `[2560]` gate-logit row,
    `dn` `[5, 2560, 32]` the slice's down_proj columns (`shared_blocks`)."""

    x: torch.Tensor
    gu: torch.Tensor
    gl: torch.Tensor
    dn: torch.Tensor


def shared_blocks(gate, up, dn, rank: int):
    """The checkpoint's shared-expert gate / up rows `[640, 2560]` and down `[2560, 640]` (bf16) -> this rank's
    `(gu [5, 64, 2560], dn [5, 2560, 32])`."""
    w = SHARED_INTER // RANKS
    lo = rank * w
    g = gate[lo : lo + w].reshape(SHARED_ITEMS, 8, 4, -1)
    u = up[lo : lo + w].reshape(SHARED_ITEMS, 8, 4, -1)
    gu = torch.stack([g, u], dim=2).reshape(SHARED_ITEMS, 64, -1).contiguous()
    d = (
        dn[:, lo : lo + w]
        .reshape(dn.shape[0], SHARED_ITEMS, SHARED_ITEM_COLS)
        .permute(1, 0, 2)
        .contiguous()
    )
    return gu, d


def block_ids(rank: int) -> tuple[list[int], list[int]]:
    """(expert, block) of each of a rank's 640 blocks: j < 512 is expert j's block (rank - j) % 4,
    j >= 512 is expert 4 (j - 512) + rank's block 4."""
    experts = [j if j < EXPERTS else 4 * (j - EXPERTS) + rank for j in range(BLOCKS)]
    blocks = [(rank - j) % RANKS if j < EXPERTS else EXPERT_BLOCKS - 1 for j in range(BLOCKS)]
    return experts, blocks


def block_major(fc1_w, fc1_s, fc2_w, fc2_s, rank: int) -> BlockExperts:
    """Whole-expert operands of all 512 experts indexed by routing id (`[512, 1280, 2560]` rows
    `[up ; gate]`, `[512, 10, 20]`, `[512, 2560, 640]`, `[512, 20, 5]`) -> the rank's block-major
    operands."""
    experts, blocks = block_ids(rank)
    e = torch.tensor(experts, device=fc1_w.device)
    b = torch.tensor(blocks, device=fc1_w.device)
    n = fc1_w.shape[0]
    w1 = fc1_w.view(n, 2, EXPERT_BLOCKS, 128, HIDDEN)[e, :, b].reshape(BLOCKS, 256, HIDDEN)
    s1 = fc1_s.view(n, 2, EXPERT_BLOCKS, HIDDEN // 128)[e, :, b].reshape(BLOCKS, 2, HIDDEN // 128)
    w2 = fc2_w.view(n, HIDDEN, EXPERT_BLOCKS, 128)[e, :, b]
    s2 = fc2_s[e, :, b]
    return BlockExperts(w1.contiguous(), s1.contiguous(), w2.contiguous(), s2.contiguous(), rank)


_WORKSPACE: dict = {}


def _workspace(device: torch.device) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    ws = _WORKSPACE.get(index)
    if ws is None:
        nbytes = native_ops.fp8_block_moe_tpb_workspace_bytes()
        ws = torch.zeros(nbytes, dtype=torch.uint8, device=torch.device("cuda", index))
        _WORKSPACE[index] = ws
    return ws


@functools.cache
def _pdl(device_index: int) -> bool:
    return get_env_enable_pdl() and torch.cuda.get_device_capability(device_index)[0] >= 9


def _cells():
    eager = tuple(
        Cell(
            why=f"decode routed experts at {r} rows, round-robin blocks of tep4 rank {rank}",
            spec=dict(rows=r, rank=rank, capture=False),
        )
        for r, rank in ((1, 0), (8, 3))
    )
    captured = tuple(
        Cell(
            why=f"captured decode routed experts at {r} rows, round-robin blocks, replayed twice",
            spec=dict(rows=r, rank=2, capture=True),
        )
        for r in (3,)
    )
    return eager + captured


class MoeTpbDecode(OpWrapper):
    """`partial = moe_tpb_decode(logits, xq, xsf, blocks, shared=None)`.

    `logits` `[T, 512]` bf16 (the front end's), `xq` `[T, 2560]` e4m3 and `xsf` `[20, >= T]` fp32 its
    1x128 scales transposed (`moe_front_router_quant`'s), `blocks` this rank's `BlockExperts`, `shared`
    this rank's slice of the shared expert to fold in (`SharedBlocks`, 3..8 rows). Returns the
    `[T, 2560]` bf16 partial of this rank's blocks: the ranks' partials sum to the layer's routed
    experts (plus `sigmoid(gate logit) * shared` with `shared`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Source: `torch_ext::fp8BlockMoeTpb` (`csrc/fp8BlockMoeTpbOp.cpp`, kernel in
    `csrc/fp8BlockMoeTpb.cu`). The clusters' partials meet through counters in the op's
    per-device workspace, so calls on one device must not overlap in time. PDL per
    `TRTLLM_ENABLE_PDL` on sm_90+.
    """

    def __call__(self, logits, xq, xsf, blocks: BlockExperts, shared: SharedBlocks | None = None):
        extra = {}
        if shared is not None:
            extra.update(sh_x=shared.x, sh_gu=shared.gu, sh_gl=shared.gl, sh_dn=shared.dn)
        return native_ops.fp8_block_moe_tpb(
            logits,
            xq,
            xsf,
            blocks.w1,
            blocks.s1,
            blocks.w2,
            blocks.s2,
            int(blocks.rank),
            _workspace(logits.device),
            _pdl(logits.device.index),
            **extra,
        )

    def reference(self, logits, xq, xsf, blocks: BlockExperts, shared: SharedBlocks | None = None):
        """The runner reference's arithmetic over this rank's blocks of `renorm_moe_routing_op`'s
        routing, fp32, rounded once; with `shared`, plus `sigmoid(gate logit) * slice`."""
        routed = self._routed(logits, xq, xsf, blocks)
        if shared is None:
            return routed.to(torch.bfloat16)
        scale, part = shared_reference(shared)
        return (routed + scale * part).to(torch.bfloat16)

    @staticmethod
    def _routed(logits, xq, xsf, blocks: BlockExperts):
        ids, w = renorm_moe_routing_op(logits, topk=TOP_K, output_dtype=torch.bfloat16)
        T = xq.shape[0]
        nb = HIDDEN // 128
        act_in = (
            xq.float().reshape(T, nb, 128) * xsf[:, :T].t().reshape(T, nb, 1).float()
        ).reshape(T, HIDDEN)
        out = torch.zeros(T, HIDDEN, dtype=torch.float32, device=xq.device)
        experts, _ = block_ids(blocks.rank)
        by_expert: dict[int, list[int]] = {}
        for j, e in enumerate(experts):
            by_expert.setdefault(e, []).append(j)
        for e in torch.unique(ids).tolist():
            tok, slot = (ids.long() == e).nonzero(as_tuple=True)
            for j in by_expert.get(e, []):
                w1 = dequantize_block_weight(blocks.w1[j], blocks.s1[j])
                y1 = requantize_1x128(act_in[tok] @ w1.t())
                act = requantize_1x128(torch.nn.functional.silu(y1[:, 128:]) * y1[:, :128])
                w2 = dequantize_block_weight(blocks.w2[j], blocks.s2[j].view(-1, 1))
                out.index_add_(0, tok, (act @ w2.t()) * w[tok, slot].float().unsqueeze(1))
        return out

    def is_valid(self, logits, xq, xsf, blocks: BlockExperts, shared: SharedBlocks | None = None):
        rows = logits.shape[0]
        assert logits.dtype == torch.bfloat16 and logits.is_contiguous(), logits.dtype
        assert tuple(logits.shape[1:]) == (EXPERTS,) and 1 <= rows <= MAX_ROWS, tuple(logits.shape)
        assert xq.dtype == torch.float8_e4m3fn and tuple(xq.shape) == (rows, HIDDEN)
        assert xq.is_contiguous()
        assert xsf.dtype == torch.float32 and xsf.shape[0] == HIDDEN // 128 and xsf.shape[1] >= rows
        assert xsf.stride(1) == 1, "xsf is read as [20, stride] with unit column stride"
        assert tuple(blocks.w1.shape) == (BLOCKS, 256, HIDDEN) and blocks.w1.is_contiguous()
        assert tuple(blocks.s1.shape) == (BLOCKS, 2, HIDDEN // 128) and blocks.s1.is_contiguous()
        assert tuple(blocks.w2.shape) == (BLOCKS, HIDDEN, 128) and blocks.w2.is_contiguous()
        assert tuple(blocks.s2.shape) == (BLOCKS, HIDDEN // 128) and blocks.s2.is_contiguous()
        assert 0 <= blocks.rank < RANKS, blocks.rank
        if shared is not None:
            assert 3 <= rows <= MAX_ROWS, rows
            assert shared.x.dtype == torch.bfloat16 and tuple(shared.x.shape) == (rows, HIDDEN)
            assert shared.x.is_contiguous()
            assert shared.gu.dtype == torch.bfloat16 and tuple(shared.gu.shape) == (
                SHARED_ITEMS,
                64,
                HIDDEN,
            )
            assert shared.gl.dtype == torch.bfloat16 and shared.gl.numel() == HIDDEN
            assert shared.dn.dtype == torch.bfloat16 and tuple(shared.dn.shape) == (
                SHARED_ITEMS,
                HIDDEN,
                SHARED_ITEM_COLS,
            )
            assert (
                shared.gu.is_contiguous()
                and shared.gl.is_contiguous()
                and shared.dn.is_contiguous()
            )


def shared_reference(shared: SharedBlocks) -> tuple[torch.Tensor, torch.Tensor]:
    """`(sigmoid(gate logit) [T, 1], slice [T, 2560])` in fp32 with the unfused path's bf16 roundings
    (the gate|up GEMM's outputs, `silu_and_mul_rows`' activation); the slice summed over the ranks
    is the shared expert's output before its bf16 rounding."""
    x = shared.x.float()
    gu = shared.gu.float().reshape(SHARED_ITEMS, 8, 2, 4, HIDDEN)
    gate = gu[:, :, 0].reshape(-1, HIDDEN)
    up = gu[:, :, 1].reshape(-1, HIDDEN)
    g = (x @ gate.t()).to(torch.bfloat16).float()
    u = (x @ up.t()).to(torch.bfloat16).float()
    logit = (x @ shared.gl.float().reshape(HIDDEN, 1)).to(torch.bfloat16).float()
    act = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).float()
    dn = shared.dn.float().permute(1, 0, 2).reshape(HIDDEN, -1)
    return torch.sigmoid(logit), act @ dn.t()


moe_tpb_decode = MoeTpbDecode()
