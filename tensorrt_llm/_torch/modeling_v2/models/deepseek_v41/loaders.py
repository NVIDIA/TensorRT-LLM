# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading shared by the deepseek_v41 targets: the attention projections, the LM head, the FP8 block-32
dequantization, the NVFP4 expert layout helpers of the TRTLLM fp4 runner, and the per-layer release of the loading
dict. Each target's `weights.py` says which checkpoint tensor fills which parameter; these are the pieces they share.
"""

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
    mxfp8_row_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (
    requantize_block128,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.nvfp4_fc1_scale_fold import (
    fc1_global_scale,
    fold_block_scales,
)
from tensorrt_llm.logger import logger

BLOCK = 32  # FP8 block edge of the checkpoint's dense weights
SF_BLOCK = 16  # NVFP4: one e4m3 scale per 16 elements along K


def block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of 32 rows, source
    row `4u + v` (0<=u<8, 0<=v<4) lands at destination row `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index interleaving the `[up | gate]` halves of a `2*I` row stack into up0, gate0, up1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def fc1_perm(rows: int, device) -> torch.Tensor:
    return interleave_perm(rows, device)[block32_perm(rows, device)]


def swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of a `[M, C]` uint8 scale matrix (`M % 128 == 0`, `C % 4 == 0`)."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous())


def pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    return t if t.shape[0] == rows else torch.cat([t, t.new_zeros(rows - t.shape[0], t.shape[1])])


def pad_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    return (
        t if t.shape[1] == cols else torch.cat([t, t.new_zeros(t.shape[0], cols - t.shape[1])], 1)
    )


def dequant_block32(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 [R, C] weight with a [ceil(R/32), ceil(C/32)] UE8M0 scale -> bf16 [R, C]."""
    r, c = w.shape
    sf = s.float().repeat_interleave(BLOCK, 0)[:r].repeat_interleave(BLOCK, 1)[:, :c]
    return (w.float() * sf).to(torch.bfloat16)


def release_consumed(weights, prefix: str) -> None:
    """Drop the checkpoint tensors under `prefix` from the engine's loading dict (`ConsumableWeightsDict.mark_consumed`)
    once their layer is loaded, as stock's module loading does. The loader hands each rank host copies of the tensors
    it reads, and the dict keeps them until the whole load ends; released layer by layer, a rank's load-time host
    peak is one layer's tensors instead of every layer's. A plain dict (tests) is left alone."""
    consume = getattr(weights, "mark_consumed", None)
    if consume is not None:
        consume(prefix)


def load_mxfp8(a, name: str, weights: dict, key: str, device, rows=None, cols=None) -> None:
    """Copy the E4M3 weight `key` and its per-row 1x32 UE8M0 scales into `a.<name>` / `a.<name>_s`.

    `rows` / `cols` are this rank's `(first, end)` slice of the weight (tensor parallelism). The weight bytes are the
    checkpoint's; only the scale metadata is expanded from 32x32 blocks to rows (exactly).
    """
    w = weights[key].to(device)
    assert w.dtype == torch.float8_e4m3fn, (
        f"{key}: the attention projections are E4M3 in the checkpoint"
    )
    scale = mxfp8_row_scales(weights[key[: -len("weight")] + "scale"].to(device), tuple(w.shape))
    if rows is not None:
        w, scale = w[rows[0] : rows[1]], scale[rows[0] : rows[1]]
    if cols is not None:
        w, scale = w[:, cols[0] : cols[1]], scale[:, cols[0] // 32 : cols[1] // 32]
    getattr(a, name).data.copy_(w)
    getattr(a, name + "_s").data.copy_(scale)


def load_o_a(a, weights: dict, key: str, device, rows) -> None:
    """The grouped `wo_a` rows `rows = (first, end)` of this rank as E4M3 with 128x128 power-of-two block scales.

    The checkpoint stores it with 32x32 UE8M0 scales; the engine's O-A kernels read 128x128 blocks, so (as the engine's
    loader does) the bf16 values are requantized to power-of-two block multipliers.
    """
    w = weights[key].to(device)
    assert w.dtype == torch.float8_e4m3fn
    s = (
        weights[key[: -len("weight")] + "scale"]
        .to(device)
        .float()
        .repeat_interleave(32, 0)[: w.shape[0]]
    )
    dense = (w.float() * s.repeat_interleave(32, 1)[:, : w.shape[1]])[rows[0] : rows[1]].to(
        torch.bfloat16
    )
    q, scale = requantize_block128(dense)
    a.wo_a.data.copy_(q)
    a.wo_a_scale.data.copy_(scale)


def load_attention(a, weights: dict, p: str, device) -> None:
    """Fill one layer's `_Attention` from `layers.N.attn.*`; under TP this rank takes its heads and groups."""
    t = a.tp_rank
    h0, h1 = t * a.hl, (t + 1) * a.hl
    g0, g1 = t * a.gl, (t + 1) * a.gl
    cp = lambda dst, src: dst.data.copy_(src)  # noqa: E731
    cp(a.attn_sink, weights[p + "attn_sink"].to(device).float()[h0:h1])
    cp(a.q_norm, weights[p + "q_norm.weight"].to(device))
    cp(a.kv_norm, weights[p + "kv_norm.weight"].to(device))
    load_mxfp8(a, "wq_a", weights, p + "wq_a.weight", device)
    load_mxfp8(a, "wq_b", weights, p + "wq_b.weight", device, rows=(h0 * a.hd, h1 * a.hd))
    load_mxfp8(a, "wkv", weights, p + "wkv.weight", device)
    load_o_a(a, weights, p + "wo_a.weight", device, (g0 * a.o_rank, g1 * a.o_rank))
    load_mxfp8(a, "wo_b", weights, p + "wo_b.weight", device, cols=(g0 * a.o_rank, g1 * a.o_rank))
    if a.is_owner:
        cp(a.c_norm, weights[p + "compressor.norm.weight"].to(device))
        cp(a.c_wkv, weights[p + "compressor.wkv.weight"].to(device).to(a.c_wkv.dtype))
        if a.ratio == 2:
            cp(a.c_wgate, weights[p + "compressor.wgate.weight"].to(device).float())
        cp(a.i_wk, weights[p + "indexer.wk.weight"].to(device))
        cp(a.i_knorm, weights[p + "indexer.k_norm.weight"].to(device))
    if a.is_index_source:
        load_mxfp8(a, "i_wq_b", weights, p + "indexer.wq_b.weight", device)
        cp(a.i_wproj, weights[p + "indexer.weights_proj.weight"].to(device))


def load_head(shell, weights: dict) -> None:
    """The LM head: column-parallel under TP, so this rank takes its slice of the vocabulary rows."""
    head = weights["head.weight"]
    rows = shell.lm_head.weight.shape[0]
    if rows != head.shape[0]:
        start = shell.model.mapping.tp_rank * rows
        head = head[start : start + rows]
    shell.lm_head.weight.data.copy_(head)


# ---- routed experts on the TRTLLM fp4 runner (the dep and tep8 targets) ----


def expert_scalars(
    isc1: torch.Tensor, isc2: torch.Tensor, ws1: torch.Tensor, ws2: torch.Tensor
) -> dict:
    """The runner's per-expert scalars, as stock's TRTLLM-Gen NVFP4 loader computes them (`_finalize_input_scales`,
    `load_expert_fc31_alpha_nvfp4`, `load_expert_fc2_alpha_nvfp4`, `_compute_fc31_scale_c`). `isc1` / `isc2` `[E]`:
    every expert's checkpoint input_scale of FC1 / FC2 (the layer's, all experts); `ws1` / `ws2` `[S]`: the FC1 / FC2
    weight_scale_2 of this rank's experts (FC1's is `fc1_global_scale`'s, the larger of w1's and w3's). One
    activation scale for FC1's input and one for FC2's over the whole layer (the reciprocal of the largest
    input_scale: the runner quantizes each batch, and each gated intermediate, on one grid), the alphas
    in stock's operation order (its float32 rounding differs from the plain product). -> `g1` `[1]` (fc31_input_scale),
    `o1_gate` (fc31_alpha), `o1` (fc31_scale_c = fc2_input_scale * fc31_alpha), `o2` (fc2_alpha), all float32."""
    g1 = isc1.float().max().reciprocal()
    g2 = isc2.float().max().reciprocal()
    fc31_alpha = (1.0 / (g1 * (1.0 / ws1.float()))).contiguous()
    return {
        "g1": g1.reshape(1).contiguous(),
        "o1_gate": fc31_alpha,
        "o1": (g2 * fc31_alpha).contiguous(),
        "o2": (1.0 / (g2 * (1.0 / ws2.float()))).contiguous(),
    }


def load_experts(moe, weights: dict, prefix: str, device) -> None:
    """Fill one layer's TRTLLM-runner `_Moe` (dep, tep8) from the checkpoint's
    `{e}.{w1,w2,w3}.{weight,weight_scale,...}`.

    w1 = gate, w3 = up, w2 = down. This rank keeps the experts it holds, `[held_offset, held_offset + held)` (its EP
    window `[offset, offset + local)`, or every expert under tep8's dynamic expert parallelism), and the `tp_rank`-th
    `1/tp` slice of the intermediate width, shuffled and swizzled for the runner."""
    cfg, inter_t = moe.cfg, moe.inter_t
    e_total, tp, tr = cfg.n_routed_experts, moe.tp, moe.tp_rank
    per = cfg.moe_intermediate_size // tp
    lo, hi = tr * per, (tr + 1) * per

    def scalar(e: int, w: str, name: str) -> torch.Tensor:
        return weights[f"{prefix}{e}.{w}.{name}"].reshape(()).float().to(device)

    isc1 = torch.stack(
        [
            torch.maximum(scalar(e, "w1", "input_scale"), scalar(e, "w3", "input_scale"))
            for e in range(e_total)
        ]
    )
    isc2 = torch.stack([scalar(e, "w2", "input_scale") for e in range(e_total)])
    ws1 = torch.stack([scalar(e, "w1", "weight_scale_2") for e in range(e_total)])
    ws3 = torch.stack([scalar(e, "w3", "weight_scale_2") for e in range(e_total)])
    ws13, w1_fold, w3_fold = fc1_global_scale(ws1, ws3)  # the fused FC1's one weight scalar
    ws2 = torch.stack([scalar(e, "w2", "weight_scale_2") for e in range(e_total)])
    first = getattr(moe, "held_offset", moe.offset)
    held = getattr(moe, "held", moe.local)
    win = slice(first, first + held)
    sc = expert_scalars(isc1, isc2, ws13[win], ws2[win])
    gate_alpha = sc["o1_gate"]
    moe.o1_gate.copy_(gate_alpha)
    moe.o1.copy_(sc["o1"])
    moe.o2.copy_(sc["o2"])
    moe.g1.copy_(sc["g1"])
    # The runner clamps in accumulator units (before the FC1 scalar is applied), so the real-valued
    # swiglu limit is divided by each expert's FC1 gate scalar. Measured against a torch reference.
    moe.clamp.copy_(cfg.swiglu_limit / gate_alpha)

    mism = 0
    for local in range(held):
        e = first + local
        q = f"{prefix}{e}."
        up_w = weights[q + "w3.weight"].to(device).view(torch.uint8)[lo:hi]
        gate_w = weights[q + "w1.weight"].to(device).view(torch.uint8)[lo:hi]
        up_s = weights[q + "w3.weight_scale"].to(device)[lo:hi]
        gate_s = weights[q + "w1.weight_scale"].to(device)[lo:hi]
        if not torch.equal(ws1[e], ws3[e]):
            # One FC1 weight scalar, the larger of the two, serves both halves (`fc1_global_scale`): the other
            # half's block scales are scaled DOWN by its own / the larger, so they cannot overflow e4m3.
            if ws3[e] > ws1[e]:
                gate_s = fold_block_scales(gate_s, w1_fold[e])
            else:
                up_s = fold_block_scales(up_s, w3_fold[e])
            mism += 1
        down_w = weights[q + "w2.weight"].to(device).view(torch.uint8)[:, lo // 2 : hi // 2]
        down_s = weights[q + "w2.weight_scale"].to(device)[:, lo // SF_BLOCK : hi // SF_BLOCK]
        # pad the intermediate width to the tile
        up_w, gate_w = pad_rows(up_w, inter_t), pad_rows(gate_w, inter_t)
        up_s = pad_rows(up_s.view(torch.uint8), inter_t)
        gate_s = pad_rows(gate_s.view(torch.uint8), inter_t)
        down_w = pad_cols(down_w, inter_t // 2)
        down_s = pad_cols(down_s.view(torch.uint8), inter_t // SF_BLOCK)

        w = torch.cat([up_w, gate_w], 0)
        moe.fc1_w[local].copy_(torch.index_select(w, 0, fc1_perm(w.shape[0], device)))
        s = torch.index_select(torch.cat([up_s, gate_s], 0), 0, fc1_perm(2 * inter_t, device))
        moe.fc1_s[local].copy_(swizzle(s).reshape(s.shape).view(torch.float8_e4m3fn))
        moe.fc2_w[local].copy_(torch.index_select(down_w, 0, block32_perm(down_w.shape[0], device)))
        d = torch.index_select(down_s, 0, block32_perm(down_s.shape[0], device))
        moe.fc2_s[local].copy_(swizzle(d).reshape(d.shape).view(torch.float8_e4m3fn))
    if mism:
        logger.info(
            f"modeling_v2 deepseek_v41: {prefix}: w1 / w3 weight_scale_2 differ in {mism} experts, the larger is shared"
        )
