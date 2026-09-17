# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-head QK RMS norm followed by RoPE, in place on a packed qkv buffer."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, assert_within_ulp


class _FusedQkNormRope(OpWrapper):
    """In place on `qkv`: RMS-norm the q and k heads, then rotate them. Returns None.

    The v heads are not touched. `qkv` is one `[num_tokens, (Hq + Hk + Hv) *
    head_dim]` buffer, which is the layout the fused projection already
    produces, so nothing is split or copied.
    """

    ARCHS = frozenset({Arch.SM_103})

    # gpt-oss-120b is the only caller, and it asks for the RoPE half only:
    # `is_qk_norm=False`, neox pairing, and YaRN-blended frequencies. 64 query
    # heads, 8 key and 8 value, head_dim 64, all of which the cells carry
    # because the op reads them as scalars rather than deriving them.
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

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: one token", spec=dict(tokens=1, **_GPT_OSS)),
        Cell(why="prefill-shaped batch", spec=dict(tokens=2048, **_GPT_OSS)),
        Cell(
            why="a token count that is not a multiple of any tile",
            spec=dict(tokens=37, **_GPT_OSS),
        ),
    )

    note = """
    Three of this op's paths are uncertified because no shipped target reaches
    them, and the first is the one its own name advertises:

      * `is_qk_norm=True` -- gpt-oss passes False, so the norm half runs in
        `reference` but never against the kernel;
      * `is_neox=False`, the interleaved pairing;
      * `use_mrope`, `use_gemma`.

    Adding a target that wants any of them means adding the cell, not assuming
    the path works because the argument exists.
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
    ) -> None:
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

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """A rotation is a two-term combination, and two terms can cancel.

        `x1 * cos - x2 * sin` goes to zero wherever the two products nearly
        match, and there torch's per-element rtol is meaningless: at 2048 tokens
        roughly 2e-4 of the outputs land close enough that a 7.3e-4 absolute
        difference reads as a relative difference of 23. The decode and 37-token
        cells pass under torch's default band only because they are too small to
        sample one.

        So: 2 ulp element-wise against the row scale, 1 of relative RMS -- tight,
        because one rotation is the shortest chain in this catalog. Measured
        worst case across these cells is 7.3e-4 absolute at a row scale of ~4,
        about 0.05 ulp.
        """
        assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=1.0)

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


fused_qk_norm_rope = _FusedQkNormRope()
