# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sparse layers' NVFP4 QKV + indexer RMSNorm / RoPE / KV-insert producer at the TP2 heads, with its constant loads
staged before the PDL grid-dependency wait and an early dependents trigger
(``kernels/minimaxM3Sm107Tp2/producerPw2.cu``)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

__all__ = ["m3_producer"]

HEAD_DIM = 128
ROTARY_DIM = 64
PAGE = 128
#: Heads per rank at TP2 (MiniMax-M3: 64 query, 4 KV and 4 index heads in all; one index-K head per token).
HEADS_Q = 32
HEADS_KV = 2
HEADS_INDEX = 2
#: Packed heads per token: q, k, v, index-q and the index-k head.
HEADS_PACKED = HEADS_Q + 2 * HEADS_KV + HEADS_INDEX + 1
#: NVFP4 block: 16 values share one E4M3 scale.
FP4_BLOCK = 16
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _norm_rope(
    x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float
):
    """Gemma RMSNorm (``x * rsqrt(mean(x^2) + eps) * (1 + w)``) then NeoX RoPE on the first 64 dims, in fp32.
    x [T, 128] fp32; cos / sin [T, 32]."""
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.float())
    lo, hi, rest = y[:, :32], y[:, 32:64], y[:, 64:]
    return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin, rest], dim=-1)


def decode_nvfp4_token(
    data: torch.Tensor, scales: torch.Tensor, quant_scale: float
) -> torch.Tensor:
    """fp32 [128] of one token's packed E2M1 row (64 bytes, low nibble first) and its 8 E4M3 block scales."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=data.device)
    nib = torch.stack([data & 0xF, data >> 4], dim=-1).reshape(-1).long()
    vals = table[nib & 7] * torch.where(nib >= 8, -1.0, 1.0)
    s = scales.view(torch.float8_e4m3fn).float() / quant_scale
    return vals * s.repeat_interleave(FP4_BLOCK)


def scale_index(within_page: int, plane: int) -> torch.Tensor:
    """Positions of a token's 8 block scales in its page's [128 * 8] scale plane: row-major for K, the 4-token
    swizzle for V."""
    cols = torch.arange(8)
    if plane == 0:
        return within_page * 8 + cols
    return (within_page // 4) * 32 + cols * 4 + within_page % 4


class M3Producer(OpWrapper):
    """The NVFP4 QKV + indexer producer of a sparse layer: ``minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert`` at the
    TP2 heads (32 query, two KV and two index heads), head dim 128 and rotary dim 64, with the same arguments and
    outputs bit for bit.

    ``packed`` [T, 39 * 128] BF16 holds each token's 32 q heads, 2 k heads, 2 v heads, 2 index-q heads and the
    index-k head. Every head but v gets the Gemma RMSNorm with its weight and NeoX RoPE on its first 64 dims
    (``rotary_cos_sin[position]``). Returns (q FP8 [T, 32, 128], index-q FP8 [T, 2, 128]); writes the token's slot
    ``out_cache_loc[t]`` (skipped when negative or past the pool): each KV head's k and v as NVFP4 into its block of
    the HND pools (16-value blocks, E4M3 scales against ``kv_quant_scale[1]`` / ``[2]``, the V scales in the 4-token
    swizzle) and index-k as FP8. ``early`` triggers the dependent kernel on entry (its outputs are read only after the
    dependent's own grid wait); ``prewait_step_inputs`` reads the position ids, the slots and the RoPE rows before the
    kernel's own grid wait (none of them may be written by a PDL predecessor).
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token (c1 without speculation)", spec=dict(tokens=1, scale=1.0)),
        Cell(why="a c1 verify: 4 tokens", spec=dict(tokens=4, scale=3.0)),
        Cell(why="a c4 verify: 16 tokens, small activations", spec=dict(tokens=16, scale=1e-3)),
        Cell(
            why="a c8 verify: 32 tokens, large activations, a skipped and an out-of-pool slot",
            spec=dict(tokens=32, scale=40.0, bad_slots=True),
        ),
        Cell(why="the widest decode step: 48 tokens", spec=dict(tokens=48, scale=1.0)),
        Cell(
            why="a prefill chunk of 1000 tokens, early trigger off",
            spec=dict(tokens=1000, scale=1.0, early=False),
        ),
    )

    note = """
    The kernel is the native producer's arithmetic, compiled with the library's --use_fast_math, so it is held to that
    kernel bit for bit on every output and cache byte. The torch reference states the math in fp32: the FP8 outputs are
    held to it within one E4M3 step, and the NVFP4 K/V, decoded, within one E2M1 step of their block scale.
    """

    def __call__(
        self,
        packed: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        index_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        index_k_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        heads_q: int,
        pdl: bool = True,
        early: bool = True,
        prewait_step_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, index_q = torch.ops.trtllm.minimax_m3_sm107_tp2_producer_pw(
            packed,
            kv_data,
            kv_scale,
            index_k,
            out_cache_loc,
            kv_quant_scale,
            eps,
            q_weight,
            k_weight,
            index_q_weight,
            index_k_weight,
            rotary_cos_sin,
            position_ids,
            pdl,
            early,
            prewait_step_inputs,
        )
        return q, index_q

    def reference(
        self,
        packed: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        index_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        index_k_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        heads_q: int,
        pdl: bool = True,
        early: bool = True,
        prewait_step_inputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """fp32 values before their quantization: "q" [T, 32, 128], "index_q" [T, 2, 128], per token and KV head "k"
        and "v" [T, 2, 128], and "index_k" [T, 128] (the values written to the token's slot). The caches, ``pdl``,
        ``early`` and ``prewait_step_inputs`` do not change them."""
        t = packed.shape[0]
        x = packed.float().view(t, HEADS_PACKED, HEAD_DIM)
        rope = rotary_cos_sin[position_ids[:t].long()]
        cos, sin = rope[:, 0], rope[:, 1]

        def heads(first, count, weight):
            return torch.stack(
                [_norm_rope(x[:, h], weight, cos, sin, eps) for h in range(first, first + count)],
                dim=1,
            )

        k0 = HEADS_Q
        v0 = k0 + HEADS_KV
        iq0 = v0 + HEADS_KV
        return {
            "q": heads(0, HEADS_Q, q_weight),
            "k": heads(k0, HEADS_KV, k_weight),
            "v": x[:, v0 : v0 + HEADS_KV],
            "index_q": heads(iq0, HEADS_INDEX, index_q_weight),
            "index_k": _norm_rope(x[:, iq0 + HEADS_INDEX], index_k_weight, cos, sin, eps),
        }

    def is_valid(
        self,
        packed: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        index_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        index_q_weight: torch.Tensor,
        index_k_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        heads_q: int,
        pdl: bool = True,
        early: bool = True,
        prewait_step_inputs: bool = False,
    ) -> None:
        assert heads_q == HEADS_Q, f"the producer is built for {HEADS_Q} query heads, not {heads_q}"
        t = packed.shape[0]
        # The RoPE row is read at the position with no bound.
        pos = position_ids[:t]
        assert bool(((pos >= 0) & (pos < rotary_cos_sin.shape[0])).all()), (
            "position_ids must index rotary_cos_sin"
        )
        # Two tokens writing one slot race.
        slots = out_cache_loc[:t]
        live = slots[(slots >= 0) & (slots < kv_data.shape[0] * PAGE)]
        assert live.unique().numel() == live.numel(), "out_cache_loc must not repeat a slot"


m3_producer = M3Producer()
