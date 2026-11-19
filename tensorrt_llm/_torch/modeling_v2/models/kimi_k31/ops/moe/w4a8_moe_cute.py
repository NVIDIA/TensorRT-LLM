# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts on the CuTe DSL W4A8 grouped GEMMs (MXFP4 weights x MXFP8 activations).

The same computation as the TRTLLM-Gen MXFP4 x MXFP8 runner (:class:`K31W4A8MoeRunner`): FC1 + SiTU +
MXFP8 requant + FC2, FP32 accumulation, BF16 expert outputs, the routing weight applied at the
finalize. It reads the runner's own prepared weights in place and, on the routing the MoE front or
the route tail already sorts, it needs neither the runner's routing-from-ids kernel nor (when the
fused MoE tail folds the top-k sum) its finalize kernel. Two GEMM kernels: the 1-CTA one with 8 / 16-row
token tiles for TP-split experts (TEP), the CTA-pair one with 128-row tiles and the MMA width sized to
each tile's rows for full-width experts (DEP16 decode and prefill); :func:`pick_tiling` chooses.
Measured bitwise-equal to the runner's output on the TEP and DEP16 shapes (see the test).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

from ...kernels.moe.moe_finalize import run_moe_finalize
from ...kernels.moe.runner_sort import CLUSTER_MAX_ROWS, runner_sort
from ...kernels.moe.w4a8_moe_gemm import (
    HIDDEN,
    MMA_M,
    SUPPORTED_N_TILES,
    run_w4a8_moe,
    workspace_bytes,
)
from ...kernels.moe.w4a8_moe_gemm2 import PAIR_M, SUPPORTED_PAIR_N_TILES, run_w4a8_moe_2cta

#: K3.1 routed experts: 896 experts, top-16.
NUM_EXPERTS = 896
TOP_K = 16
_SF_VEC = 32
#: Rows up to which FC1 releases FC2 right after its grid dependency (decode / verify rows of the attention-TP
#: generation targets; above them FC1 fills the GPU and FC2's CTAs would only queue behind it). A caller may pass
#: its own bound (``early_release_max_rows``).
EARLY_RELEASE_MAX_ROWS = 32
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
#: Stored row of a 32-row block <- row of the block before the runner's shuffle (inverse of
#: srcToDstBlk32RowMap: stored 8 b + a <- 4 a + b).
_BLK32_SRC = tuple(4 * (r % 8) + r // 8 for r in range(32))


def moe_workspace_bytes(max_tiles: int, inter: int, n_tile: int) -> int:
    """Bytes of the zeroed workspace the op needs for up to ``max_tiles`` tiles (kept zero by every
    call, so one workspace serves every call of a stream)."""
    return workspace_bytes(max_tiles, inter, HIDDEN, n_tile)


def pick_tiling(rows: int, inter: int) -> tuple[int, bool]:
    """``(n_tile, pair)`` for ``rows`` routed rows of one rank and experts of ``inter`` intermediate
    columns: full-width experts (``inter`` >= 2048) take the CTA-pair kernel with 128-row tiles at every row
    count (an expert's weights stream once per tile with the shared-memory footprint of a 64-row tile);
    TP-split experts take the 1-CTA kernel with :func:`pick_n_tile`'s tiles up to 511 rows, the CTA-pair
    kernel from 512 rows."""
    if inter >= 2048 or rows >= 512:
        return 128, True
    return pick_n_tile(rows), False


def pick_n_tile(rows: int) -> int:
    """Token tile for ``rows`` routed rows of one rank (rows of the step under attention TP, dispatched rows under
    attention DP): 8 up to 47 rows (1-3 rows per touched expert), 16 up to 511, 64 up to 1536, else 128."""
    if rows < 48:
        return 8
    if rows < 512:
        return 16
    if rows <= 1536:
        return 64
    return 128


def max_tiles_for(rows: int, local_num_experts: int, n_tile: int) -> int:
    """The tile count bound of the runner sort (``filled + (expanded - filled) // n_tile``)."""
    expanded = rows * TOP_K
    filled = min(NUM_EXPERTS, local_num_experts, expanded)
    return filled + (expanded - filled) // n_tile


def _sort(topk_ids, local_expert_offset, local_num_experts, n_tile, early_release=False):
    """``(e2p, p2t, t2e, t2l, num tiles)``: the CuTe runner sort (tokens in p2t, rows of an expert in token
    order). ``early_release``: the sort releases the next grid right after its own wait on the previous one
    (that grid must wait for the sort before reading it)."""
    e2p, p2t, _total, t2e, t2l, nne = runner_sort(
        topk_ids.int().contiguous(),
        local_expert_offset,
        local_num_experts,
        n_tile,
        early_release=early_release,
    )
    return e2p, p2t, t2e, t2l, nne


def prerouted_sort(
    topk_ids: torch.Tensor, local_expert_offset: int, local_num_experts: int, n_tile: int
) -> list:
    """The ``prerouted`` layout :class:`K31W4A8MoeCute` sorts itself for these ids and ``n_tile`` (the CuTe runner
    sort of :func:`_sort`), for a caller that issues the sort ahead of the experts."""
    return list(
        runner_sort(topk_ids.int().contiguous(), local_expert_offset, local_num_experts, n_tile)
    )


class K31W4A8MoeCute(OpWrapper):
    """One rank's routed experts (FC1 + SiTU + MXFP8 requant + FC2 [+ finalize]) on the CuTe DSL
    W4A8 GEMMs; see the module docstring.

    ``prerouted`` (optional): the runner-sort layout of this rank's local experts with tiles of
    ``n_tile`` rows, as the MoE front / route tail write it: ``[expanded_idx_to_permuted_idx,
    permuted_idx_to_token_idx, total_num_padded_tokens, tile_idx_to_expert_idx,
    tile_idx_to_mn_limit, num_non_exiting_tiles]``; without it the op sorts (the CuTe runner sort, power-of-two
    tiles, up to :data:`.CLUSTER_MAX_ROWS` rows).
    ``pair``: FC1 / FC2 on CTA pairs (:mod:`.w4a8_moe_gemm2`, ``n_tile`` 64..256; no workspace
    needed) instead of the 1-CTA kernel; see :func:`pick_tiling`.
    ``early_release_max_rows``: rows up to which the 1-CTA FC1 releases FC2 right after its grid
    dependency (default :data:`EARLY_RELEASE_MAX_ROWS`); only the launch point of FC2 moves, not the bits.
    Returns ``[rows, 3584]`` BF16 (``finalize``) or the unfinalized ``(y [permuted rows, 3584] BF16,
    topk_weights, expanded_idx_to_permuted_idx)`` the fused MoE tail folds.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP32 (MoE TP4 x EP8) decode step: 8 rows, 112 local experts x 768, tile 8",
            spec=dict(rows=8, local=112, inter=768, n_tile=8),
        ),
        Cell(
            why="TEP16 (MoE TP4 x EP4) decode: 72 rows over 224 experts, tile 16",
            spec=dict(rows=72, local=224, inter=768, n_tile=16),
        ),
        Cell(
            why="unfinalized output for the fused MoE tail (fold_finalize)",
            spec=dict(rows=24, local=224, inter=768, n_tile=8, finalize=False),
        ),
        Cell(
            why="TEP16 above the MoE front's rows (33..192): self-sorted, tile 16, unfinalized for the tail's fold",
            spec=dict(rows=192, local=224, inter=768, n_tile=16, finalize=False),
        ),
        *(
            Cell(
                why=f"TEP8 MoE TP2 x EP4: {r} rows, 224 x 1536 experts, tile {t}, unfinalized",
                spec=dict(rows=r, local=224, inter=1536, n_tile=t, finalize=False),
            )
            for r, t in ((20, 8), (48, 16))
        ),
        *(
            Cell(
                why=f"TEP8 MoE TP4 x EP2: {r} rows, 448 x 768 experts, tile {t}, unfinalized",
                spec=dict(rows=r, local=448, inter=768, n_tile=t, finalize=False),
            )
            for r, t in ((32, 8), (48, 16))
        ),
        Cell(
            why="128-row token tiles (4 gather warps, 32-column epilogue chunks) over 56 x 3072 experts",
            spec=dict(rows=1024, local=56, inter=3072, n_tile=128),
        ),
        Cell(
            why="DEP16 decode on CTA pairs: 2048 dispatched rows, 56 x 3072 experts, 128-row tiles",
            spec=dict(rows=2048, local=56, inter=3072, n_tile=128, pair=True),
        ),
    )

    note = """
    Bitwise: bit-identical to K31W4A8MoeRunner (the TRTLLM-Gen runner on the same ids, weights and MXFP8 input) on
    the cells (1-CTA tiles 8 / 16 / 64 / 128, CTA-pair tiles 64 / 128). Weights: the runner layout
    (``ops/moe/mxfp4_expert_pack``: ``w3_w1_weight`` [L, 2I, 1792] e2m1 rows permuted (gated interleave + 32-row
    shuffle), ``w2_weight`` [L, 3584, I/2] (32-row shuffle), the 128x4-interleaved UE8M0 scales). ``workspace``:
    :func:`moe_workspace_bytes` zeroed bytes, re-armed by every call, so two calls that can run at the same time
    need two workspaces (unused by the CTA-pair kernel). Every output buffer comes from the caching allocator;
    CUDA-graph safe.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        local_expert_offset: int,
        local_num_experts: int,
        n_tile: int,
        workspace: torch.Tensor,
        prerouted: Optional[Sequence[torch.Tensor]] = None,
        finalize: bool = True,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        output: Optional[torch.Tensor] = None,
        pair: bool = False,
        early_release_max_rows: Optional[int] = None,
    ):
        rows = hidden_states.shape[0]
        inter = gemm1_weights.shape[1] // 2
        if rows == 0:  # a rank without rows this step: nothing to launch
            dev = hidden_states.device
            if not finalize:
                empty_y = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=dev)
                return empty_y, topk_weights, torch.empty(0, dtype=torch.int32, device=dev)
            return (
                output
                if output is not None
                else torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=dev)
            )
        if prerouted is None:
            # The 1-CTA kernel waits for the sort's grid before reading it, so the sort may release it early
            # (its launch and setup then overlap the sort).
            e2p, p2t, t2e, t2l, nne = _sort(
                topk_ids, local_expert_offset, local_num_experts, n_tile, early_release=not pair
            )
        else:
            e2p, p2t, _total, t2e, t2l, nne = prerouted
        max_tiles = t2e.shape[0]
        dev = hidden_states.device
        h = torch.empty(max_tiles * n_tile, inter, dtype=torch.float8_e4m3fn, device=dev)
        h_sf = torch.empty(max_tiles * n_tile, inter // _SF_VEC, dtype=torch.uint8, device=dev)
        y = torch.empty(max_tiles * n_tile, HIDDEN, dtype=torch.bfloat16, device=dev)
        x_sf = hidden_states_scale.view(rows, HIDDEN // _SF_VEC)
        if pair:
            kw = dict(n_tile=n_tile, gate_cap=gate_cap, linear_cap=linear_cap, max_acc_stages=3)
            run_w4a8_moe_2cta(gemm1_weights, gemm1_weights_scale, None, None, hidden_states, x_sf, h, h_sf, None,
                                  t2e, t2l, p2t, nne, **kw)  # fmt: skip
            run_w4a8_moe_2cta(None, None, gemm2_weights, gemm2_weights_scale, None, None, h, h_sf, y, t2e, t2l,
                                  p2t, nne, early_weights=True, **kw)  # fmt: skip
        else:
            common = dict(
                n_tile=n_tile, max_tiles=max_tiles, gate_cap=gate_cap, linear_cap=linear_cap
            )
            # Decode rows: FC1 releases FC2 right after its own grid dependency, so FC2 streams its first weights
            # (early_weights) while FC1 runs.
            run_w4a8_moe(gemm1_weights, gemm1_weights_scale, None, None, hidden_states, x_sf, h, h_sf, None,
                             t2e, t2l, p2t, nne, workspace, early_release=rows <= (
                                 EARLY_RELEASE_MAX_ROWS if early_release_max_rows is None
                                 else early_release_max_rows), **common)  # fmt: skip
            run_w4a8_moe(None, None, gemm2_weights, gemm2_weights_scale, None, None, h, h_sf, y, t2e, t2l,
                             p2t, nne, workspace, early_weights=True, **common)  # fmt: skip
        if not finalize:
            return y, topk_weights, e2p
        out = (
            output
            if output is not None
            else torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        )
        run_moe_finalize(y, e2p, topk_weights, out)
        return out

    def reference(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        local_expert_offset: int,
        local_num_experts: int,
        n_tile: int,
        workspace: torch.Tensor,
        prerouted: Optional[Sequence[torch.Tensor]] = None,
        finalize: bool = True,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        output: Optional[torch.Tensor] = None,
        pair: bool = False,
        early_release_max_rows: Optional[int] = None,
    ):
        """The runner class in plain torch from the stored weights: dequantized MXFP8 input and MXFP4
        experts (the stored row maps undone), FP32 FC1, SiTU, the ``mxfp8_quantize`` requant, FP32 FC2,
        BF16 expert rows, FP32 slot-order finalize. Ignores ``n_tile``, ``workspace``, ``output``, ``pair``,
        ``early_release_max_rows``; with ``finalize=False`` returns the expert rows
        in the order of the sort (``prerouted`` or the runner sort with ``n_tile``) like the op."""
        rows = hidden_states.shape[0]
        inter = gemm1_weights.shape[1] // 2
        dev = hidden_states.device
        x = _deq_e4m3(hidden_states, hidden_states_scale.view(rows, HIDDEN // _SF_VEC))
        ids = topk_ids.long() - local_expert_offset
        local = (ids >= 0) & (ids < local_num_experts)
        lin_rows, gate_rows = _fc1_rows(inter, dev)
        hid_of_stored = _fc2_hidden(HIDDEN, dev)
        out = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=dev)
        y_slot = torch.zeros(rows, TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
        with true_fp32_matmul():
            for e in torch.unique(ids[local]).tolist():
                tok, slot = (ids == e).nonzero(as_tuple=True)
                w1 = _deq_e2m1(
                    gemm1_weights[e],
                    _deinterleave(gemm1_weights_scale[e], 2 * inter, HIDDEN // _SF_VEC),
                )
                acc = x[tok] @ w1.t()
                hgl = _situ(acc[:, gate_rows], acc[:, lin_rows], gate_cap, linear_cap)
                hq, hs = _mxfp8_quant(hgl)
                w2 = _deq_e2m1(
                    gemm2_weights[e],
                    _deinterleave(gemm2_weights_scale[e], HIDDEN, inter // _SF_VEC),
                )
                o2 = _deq_e4m3(hq, hs) @ w2.t()
                ye = torch.empty_like(o2)
                ye[:, hid_of_stored] = o2
                y_slot[tok, slot] = ye.to(torch.bfloat16)
        if not finalize:
            if prerouted is None:
                e2p = _sort(topk_ids, local_expert_offset, local_num_experts, n_tile)[0]
                max_tiles = max_tiles_for(rows, local_num_experts, n_tile)
            else:
                e2p, max_tiles = prerouted[0].reshape(-1), prerouted[3].shape[0]
            y = torch.zeros(max_tiles * n_tile, HIDDEN, dtype=torch.bfloat16, device=dev)
            sel = e2p >= 0
            y[e2p[sel].long()] = y_slot.reshape(rows * TOP_K, HIDDEN)[sel]
            return y, topk_weights, e2p
        for k in range(TOP_K):
            m = local[:, k]
            out[m] += topk_weights[m, k].float()[:, None] * y_slot[m, k].float()
        return out.to(torch.bfloat16)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        local_expert_offset: int,
        local_num_experts: int,
        n_tile: int,
        workspace: torch.Tensor,
        prerouted: Optional[Sequence[torch.Tensor]] = None,
        finalize: bool = True,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        output: Optional[torch.Tensor] = None,
        pair: bool = False,
        early_release_max_rows: Optional[int] = None,
    ) -> None:
        tiles = SUPPORTED_PAIR_N_TILES if pair else SUPPORTED_N_TILES
        if n_tile not in tiles:
            raise ValueError(f"n_tile {n_tile} not in {tiles}")
        inter = gemm1_weights.shape[1] // 2
        if (2 * inter) % MMA_M or inter % MMA_M or gemm1_weights.shape[2] * 2 != HIDDEN:
            raise ValueError(
                f"weights {tuple(gemm1_weights.shape)}: need 2I % 128 == 0, I % 128 == 0, K = {HIDDEN}"
            )
        if gemm1_weights.shape[0] != local_num_experts:
            raise ValueError("gemm1_weights must hold exactly the rank's local experts")
        if prerouted is None and (
            n_tile & (n_tile - 1) or hidden_states.shape[0] > CLUSTER_MAX_ROWS
        ):
            raise ValueError(
                f"n_tile {n_tile}, {hidden_states.shape[0]} rows: the op's own sort takes power-of-two tiles "
                f"and up to {CLUSTER_MAX_ROWS} rows; pass prerouted"
            )
        if pair:
            if (2 * inter) % PAIR_M or HIDDEN % PAIR_M:
                raise ValueError(f"pair: 2I = {2 * inter} must be a multiple of {PAIR_M}")
            return
        rows = hidden_states.shape[0]
        max_tiles = (
            prerouted[3].shape[0]
            if prerouted is not None
            else max_tiles_for(rows, local_num_experts, n_tile)
        )
        need = moe_workspace_bytes(max_tiles, inter, n_tile)
        if workspace.numel() * workspace.element_size() < need:
            raise ValueError(
                f"workspace {workspace.numel() * workspace.element_size()} B < {need} B"
            )


# ---------------------------------------------------------------------- reference helpers
def _ue8m0(sf: torch.Tensor) -> torch.Tensor:
    bits = sf.to(torch.int32) << 23
    bits = torch.where(sf == 0, torch.full_like(bits, 0x00400000), bits)
    return bits.view(torch.float32)


def _deq_e4m3(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return (q.float().unflatten(-1, (-1, _SF_VEC)) * _ue8m0(sf)[..., None]).flatten(-2)


def _deq_e2m1(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 [M, K/2] (element 2i = low nibble of byte i) + UE8M0 [M, K/32] -> FP32 [M, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    vals = lut[torch.stack([(packed & 0xF).long(), (packed >> 4).long()], dim=-1).flatten(-2)]
    return (vals.unflatten(-1, (-1, _SF_VEC)) * _ue8m0(sf)[..., None]).flatten(-2)


def _deinterleave(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """128x4-interleaved scales (one expert) -> linear [rows, cols]."""
    m = torch.arange(rows, device=flat.device)[:, None]
    k = torch.arange(cols, device=flat.device)[None, :]
    off = (
        (m // 128) * ((cols + 3) // 4 * 512)
        + (k // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + k % 4
    )
    return flat.reshape(-1)[off.reshape(-1)].reshape(rows, cols)


def _fc1_rows(inter: int, dev):
    """Stored rows of (linear, gate) for every intermediate j: interleaved row 2 j (linear) / 2 j + 1
    (gate), each 32-row block shuffled (stored 8 b + a <- 4 a + b)."""
    j = torch.arange(inter, device=dev)
    inv = torch.empty(32, dtype=torch.long)
    for stored, src in enumerate(_BLK32_SRC):
        inv[src] = stored
    inv = inv.to(dev)
    lin = 2 * j
    gate = 2 * j + 1
    return (lin // 32) * 32 + inv[lin % 32], (gate // 32) * 32 + inv[gate % 32]


def _fc2_hidden(hidden: int, dev):
    """Hidden index of every stored FC2 row."""
    r = torch.arange(hidden, device=dev)
    src = torch.tensor(_BLK32_SRC, dtype=torch.long, device=dev)
    return (r // 32) * 32 + src[r % 32]


def _situ(gate: torch.Tensor, lin: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    return beta * torch.tanh(lin / beta) * alpha * torch.tanh(gate / alpha) * torch.sigmoid(gate)


def _mxfp8_quant(v: torch.Tensor):
    """``mxfp8_quantize``'s arithmetic on FP32 [N, K]: UE8M0 = amax * fp32(1 / 448) rounded up to a power
    of two; E4M3 round to nearest even, saturating."""
    n, k = v.shape
    blk = v.reshape(n, k // _SF_VEC, _SF_VEC)
    amax = blk.abs().amax(-1)
    sf = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=v.device)
    bits = sf.view(torch.int32)
    ex = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    sub = torch.where(mant > 0x400000, torch.ones_like(ex), torch.zeros_like(ex))
    eb = torch.where(ex == 0, sub, ex + (mant != 0).to(torch.int32)).clamp(max=254)
    sb = (254 - eb) << 23
    sb = torch.where(eb == 254, torch.full_like(sb, 0x00400000), sb)
    sb = torch.where(eb == 0, torch.full_like(sb, 0x3F800000), sb)
    sb = torch.where(amax == 0, torch.zeros_like(sb), sb)
    q = (
        (blk * sb.view(torch.float32)[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(n, k)
    )
    return q, eb.to(torch.uint8)


k31_w4a8_moe_cute = K31W4A8MoeCute()

__all__ = [
    "K31W4A8MoeCute",
    "k31_w4a8_moe_cute",
    "max_tiles_for",
    "moe_workspace_bytes",
    "pick_n_tile",
    "prerouted_sort",
]
