# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-head QK RMS norm followed by RoPE, in place on a packed qkv buffer."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class FusedQkNormRope(OpWrapper):
    """In place on `qkv`: RMS-norm the q and k heads, then rotate them.
    Returns `qkv` -- the same buffer, mutated and handed back so a call site
    can show what it touched without allocating.

    The v heads are not touched. `qkv` is one `[num_tokens, (Hq + Hk + Hv) *
    head_dim]` buffer, which is the layout the fused projection already
    produces, so nothing is split or copied. The norm is `rmsnorm(x) * w`, or
    the Gemma form `rmsnorm(x) * (1 + w)` with `use_gemma`; it covers the whole
    head, the rotation only its first `rotary_dim` dims. With `use_mrope`,
    `position_ids` is `[3, T]` and rotary pair `j` takes its position from row 1
    when `j % 3 == 1 and j < 3 * mrope_section1`, from row 2 when
    `j % 3 == 2 and j < 3 * mrope_section2`, from row 0 otherwise -- the
    interleaved mRoPE layout. The caller owns the QKV projection before the
    call and the KV-cache append and attention after it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # gpt-oss-120b's default target asks for the RoPE half only:
    # `is_qk_norm=False`, neox pairing, and YaRN-blended frequencies. 64 query
    # heads, 8 key and 8 value, head_dim 64, all of which the cells carry
    # because the op reads them as scalars rather than deriving them. Under
    # attention TP each rank sees its own share of the heads.
    _GPT_OSS = dict(
        num_heads_q=64,
        num_heads_k=8,
        num_heads_v=8,
        head_dim=64,
        rotary_dim=64,
        base=150000.0,
        is_neox=True,
        is_qk_norm=False,
        # The checkpoint's YaRN correction range, from beta_fast=32 /
        # beta_slow=1 over its original 4096-token context.
        factor=32.0,
        low=8.0,
        high=18.0,
        attention_factor=1.0,
    )

    # qwen4_exp (Qwen3.8-Flash-Next) runs Gemma QK norm and an unscaled rope
    # on the first 64 dims (32 pairs), base 1e7, at two geometries. Its
    # mrope_section is [11, 11, 10], passed as sections 11 / 11. Positions are
    # drawn uniformly below `max_position` (a test axis, not an op argument):
    # the kernel's fast sincos loses phase as `position * freq` grows, so the
    # position range is part of what a cell certifies.
    _QWEN_ROPE = dict(
        rotary_dim=64,
        base=1e7,
        is_neox=True,
        is_qk_norm=True,
        use_gemma=True,
        max_position=32768,
    )
    _QWEN_MROPE = dict(use_mrope=True, mrope_section1=11, mrope_section2=11, **_QWEN_ROPE)
    # Main attention (trunk and MTP): 24 query heads, 2 kv, head_dim 256.
    _QWEN_MAIN = dict(num_heads_q=24, num_heads_k=2, num_heads_v=2, head_dim=256, **_QWEN_MROPE)
    # The QSA index plane's query rope: 4 index heads plus the one k and one v
    # slot of its scratch layout, over index_head_dim 128 (== 2 * rotary_dim,
    # so the partial factor is not reapplied).
    _QWEN_INDEX_HEADS = dict(num_heads_q=4, num_heads_k=1, num_heads_v=1, head_dim=128)
    _QWEN_INDEX_MROPE = dict(**_QWEN_INDEX_HEADS, **_QWEN_MROPE)
    _QWEN_INDEX_PLAIN = dict(**_QWEN_INDEX_HEADS, **_QWEN_ROPE)

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: one token", spec=dict(tokens=1, **_GPT_OSS)),
        Cell(why="prefill-shaped batch", spec=dict(tokens=2048, **_GPT_OSS)),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(tokens=37, **_GPT_OSS),
        ),
        Cell(
            why="default target at attention TP8: 8q/1k/1v heads per rank",
            spec=dict(
                tokens=64, **{**_GPT_OSS, "num_heads_q": 8, "num_heads_k": 1, "num_heads_v": 1}
            ),
        ),
        Cell(
            why="qwen4_exp main rope, decode: one token, Gemma norm and interleaved mRoPE "
            "with three distinct position rows",
            spec=dict(tokens=1, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp main rope, 19 tokens: the size its wiring and saturation tests run at",
            spec=dict(tokens=19, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp main rope, 8192-token prefill: the Gemma one-ulp tail, under "
            "GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 1 token: one k and one v head, the "
            "narrowest layout; 5 warps of an 8-warp block",
            spec=dict(tokens=1, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 2 tokens: the second token's heads "
            "straddle a block boundary",
            spec=dict(tokens=2, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 5 tokens: the last block runs one warp",
            spec=dict(tokens=5, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 17 tokens: one past a power of two",
            spec=dict(tokens=17, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 19 tokens: the size its wiring and "
            "saturation tests run at",
            spec=dict(tokens=19, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 33 tokens: one past a power of two",
            spec=dict(tokens=33, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 65 tokens: one past a power of two",
            spec=dict(tokens=65, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 8192-token prefill: the index site sees the "
            "whole packed batch; under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (plain), decode: the form the index plane runs "
            "today, row 0 alone as a flat [T] position_ids",
            spec=dict(tokens=1, **_QWEN_INDEX_PLAIN),
        ),
        Cell(
            why="qwen4_exp index rope (plain), 19 tokens: an odd count across blocks",
            spec=dict(tokens=19, **_QWEN_INDEX_PLAIN),
        ),
        Cell(
            why="qwen4_exp index rope (plain), 8192-token prefill: under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_INDEX_PLAIN),
        ),
    )

    note = """
    Certified by caller: gpt-oss runs `is_qk_norm=False` with YaRN; qwen4_exp
    runs `is_qk_norm=True, use_gemma=True`, unscaled, with and without
    interleaved mRoPE. Uncertified because no shipped target reaches them:
    `is_qk_norm=True` with `use_gemma=False` (the plain-weight norm the op's
    name advertises), `is_neox=False`, YaRN together with the norm or with
    mRoPE. A target that wants one adds the cell rather than assuming the path
    works because the argument exists.

    mRoPE preconditions. `position_ids` is a contiguous int32 `[3, T]`; the op
    rejects a strided view loudly, so an engine's `[3, 1, max_num_tokens]`
    position buffer has to be copied, not only reshaped. A HuggingFace
    `mrope_section` `[s0, s1, s2]` summing to `rotary_dim / 2` is passed as
    `mrope_section1 = s1`, `mrope_section2 = s2`; `s0` is implied. Only the
    interleaved layout exists here -- the contiguous-section variant is a
    different operator, measured ~500x out when wired in its place. Sections
    saturate: at 32 pairs `mrope_section2` 10, 11 and 12 are bit-identical (the
    largest `j % 3 == 2` below 32 is 29), while `mrope_section1` 10 moves pair
    31 back to row 0. Three equal rows reduce bitwise to the plain
    `use_mrope=False` call on that row, so a text batch cannot see a wrong
    split; an image span is where one shows.

    Metadata consumed: none.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        is_neox: bool,
        position_ids: torch.Tensor,
        factor: float = 1.0,
        low: float = 0.0,
        high: float = 0.0,
        attention_factor: float = 1.0,
        is_qk_norm: bool = True,
        use_gemma: bool = False,
        use_mrope: bool = False,
        mrope_section1: int = 0,
        mrope_section2: int = 0,
    ) -> torch.Tensor:
        """Writes `qkv` in place and returns it -- the op destroys the q/k
        heads' prior contents, so a caller that wants them has to have kept
        its own copy beforehand."""
        torch.ops.trtllm.fused_qk_norm_rope(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            factor,
            low,
            high,
            attention_factor,
            is_qk_norm,
            use_gemma,
            use_mrope,
            mrope_section1,
            mrope_section2,
        )
        return qkv

    def reference(
        self,
        qkv: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        is_neox: bool,
        position_ids: torch.Tensor,
        factor: float = 1.0,
        low: float = 0.0,
        high: float = 0.0,
        attention_factor: float = 1.0,
        is_qk_norm: bool = True,
        use_gemma: bool = False,
        use_mrope: bool = False,
        mrope_section1: int = 0,
        mrope_section2: int = 0,
    ) -> torch.Tensor:
        """The buffer the op writes back, fp32 throughout.

        Returns rather than mutates so a cell can drive both sides from the
        same input; the in-place write is what `__call__` does.
        """
        num_heads = num_heads_q + num_heads_k + num_heads_v
        x = qkv.float().view(-1, num_heads, head_dim).clone()
        half = rotary_dim // 2
        inv_freq = self._inv_freq(rotary_dim, base, factor, low, high, qkv.device)

        if use_mrope:
            angle = position_ids.float()[:, :, None] * inv_freq  # [3, T, half]
            cos, sin = angle.cos(), angle.sin()

            def pick(c: torch.Tensor) -> torch.Tensor:
                out = c[0].clone()
                out[:, 1 : mrope_section1 * 3 : 3] = c[1][:, 1 : mrope_section1 * 3 : 3]
                out[:, 2 : mrope_section2 * 3 : 3] = c[2][:, 2 : mrope_section2 * 3 : 3]
                return out

            cos, sin = pick(cos), pick(sin)
        else:
            angle = position_ids.float()[:, None] * inv_freq  # [T, half]
            cos, sin = angle.cos(), angle.sin()
        cos = cos[:, None, :] * attention_factor
        sin = sin[:, None, :] * attention_factor

        for start, count, weight in (
            (0, num_heads_q, q_weight.float()),
            (num_heads_q, num_heads_k, k_weight.float()),
        ):
            h = x[:, start : start + count, :]
            if is_qk_norm:
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
                h = h * ((1.0 + weight) if use_gemma else weight)
            r = h[..., :rotary_dim]
            if is_neox:
                x1, x2 = r[..., :half], r[..., half:]
                rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            else:
                x1, x2 = r[..., ::2], r[..., 1::2]
                rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(
                    -2
                )
            x[:, start : start + count, :] = torch.cat([rotated, h[..., rotary_dim:]], -1)
        return x.view(qkv.shape[0], -1).to(qkv.dtype)

    @staticmethod
    def _inv_freq(
        rotary_dim: int, base: float, factor: float, low: float, high: float, device
    ) -> torch.Tensor:
        """YaRN-blended inverse frequencies; plain RoPE when factor == 1."""
        j = torch.arange(rotary_dim // 2, dtype=torch.float32, device=device)
        pos_freqs = base ** (2.0 * j / rotary_dim)
        inv_extrapolation = 1.0 / pos_freqs
        inv_interpolation = 1.0 / (factor * pos_freqs)
        if high == low:
            high = high + 0.001  # the kernel guards the ramp singularity the same way
        ramp = ((j - low) / (high - low)).clamp(0.0, 1.0)
        extrapolation = 1.0 - ramp
        return inv_interpolation * (1.0 - extrapolation) + inv_extrapolation * extrapolation

    def is_valid(
        self,
        qkv: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        rotary_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        base: float,
        is_neox: bool,
        position_ids: torch.Tensor,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        # `eps`, `q_weight`, `k_weight`, `base` and `is_neox` are named though
        # nothing here reads them: they mirror `__call__`, and gpt-oss passes
        # them by keyword.
        #
        # The op reads the head counts as scalars and addresses qkv from
        # data_ptr(); a width that disagrees with them is read as whatever the
        # counts say and the heads silently come out shifted.
        expected = (num_heads_q + num_heads_k + num_heads_v) * head_dim
        assert qkv.shape[-1] == expected, (
            f"qkv's last dim must be (Hq + Hk + Hv) * head_dim = {expected}; got "
            f"{qkv.shape[-1]}. The op trusts the counts and reads the buffer flat"
        )
        assert qkv.is_contiguous(), "qkv must be contiguous; the op writes it through data_ptr()"
        assert rotary_dim <= head_dim, (
            f"rotary_dim ({rotary_dim}) exceeds head_dim ({head_dim}); the rotation "
            "would read past the end of each head"
        )
        assert rotary_dim % 2 == 0, f"rotary_dim must be even to pair; got {rotary_dim}"
        assert position_ids.shape[-1] == qkv.shape[0], (
            f"one position per token: {position_ids.shape[-1]} ids for {qkv.shape[0]} tokens"
        )


fused_qk_norm_rope = FusedQkNormRope()
