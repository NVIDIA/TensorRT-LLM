# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's q side after its projection, in one CuTe DSL launch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_indexer_q_side_fused as _kernel

#: The indexer geometry the kernel is specialised for: 32 heads of 128, the
#: first 64 dims of each head rotated.
_HEADS = 32
_HEAD_DIM = 128
_ROPE_DIM = 64
_HALF = _ROPE_DIM // 2
_E4M3_MAX = 448.0
_MIN_AMAX = 1.0e-12


@dataclass(frozen=True)
class CompiledIndexerQSide:
    """The compiled launcher and the rope mode it was compiled for."""

    fn: Callable
    is_neox: bool


def build(device_id: int, *, is_neox: bool) -> CompiledIndexerQSide:
    """Compile the variant the target runs: ue8m0 row scales, PDL launch.

    `is_neox` is a compile-time constant of the kernel. One compile, a few
    seconds, cached per process by the kernel module; allocates nothing.
    """
    with torch.cuda.device(device_id):
        fn = _kernel.compile_indexer_q_side_fused(device_id, bool(is_neox), True, use_pdl=True)
    return CompiledIndexerQSide(fn=fn, is_neox=bool(is_neox))


class CuteDslIndexerQSideFused(OpWrapper):
    """Rope the indexer query, pack it to e4m3, and fold its scale into the head weights.

    `q` is the unrotated `[T, 32, 128]` bf16 indexer query, each head `[pe(64) |
    nope(64)]`. For every (token, head) row the op

    1. rotates `pe` at `position_ids[t]` from `cos_sin[pos] = [cos(32) | sin(32)]`
       (NeoX halves, or interleaved pairs when `is_neox` is False) and rounds the
       result to bf16;
    2. takes `amax` over the rotated `pe` and the untouched `nope`, sets the row
       scale to `max(amax, 1e-12) / 448` rounded up to a power of two, and writes
       `q_fp8_out[t, h] = e4m3([pe | nope] / scale)`;
    3. writes `weights_out[t, h] = (wq[t, h] * scale) * weight_scale`.

    `q` is not modified. Both outputs are caller-owned and returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target runs the checkpoint's interleaved rope (`indexer_is_neox` False),
    # one compiled launcher for every row count. Rows are generation tokens on a
    # decode step and the whole step on a prefill step; positions go up to the
    # engine's max_seq_len.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, one token: the smallest launch (one partial CTA)",
            spec=dict(tokens=1, positions="spread", max_pos=4096, is_neox=False),
        ),
        Cell(
            why="the full decode batch, 32 sequences x MTP 2",
            spec=dict(tokens=64, positions="spread", max_pos=8192, is_neox=False),
        ),
        Cell(
            why="a prefill chunk at max_num_tokens, positions 0..8191 in order",
            spec=dict(tokens=8192, positions="arange", max_pos=8192, is_neox=False),
        ),
        Cell(
            why="an odd row count with long-context positions near the table end",
            spec=dict(tokens=300, positions="tail", max_pos=131072, is_neox=False),
        ),
    )

    note = """
    `build(device_id, is_neox=...)` compiles the launcher once; the target calls
    it at load and passes the result as `compiled`. Row count is a runtime
    value, so one launcher serves decode and prefill and there is no stock-op
    fallback at the call site.

    The NeoX rope mode compiles and is verified by the kernel's own driver,
    but the target never builds it, so it is not certified here.

    `position_ids` is read on the device with no range check: a position at or
    past `cos_sin.shape[0]` reads past the table. `is_valid` states it.

    Bit-exact against the reference: every rounding the kernel does (bf16 after
    the rope FMA, the power-of-two scale, the two fp32 products) is reproduced.
    A bf16 subnormal in the rope half is flushed to signed zero first.
    """

    def __call__(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSide,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _kernel.indexer_q_side_fused(
            q,
            position_ids,
            cos_sin,
            wq,
            weight_scale,
            q_fp8_out,
            weights_out,
            is_neox=is_neox,
            use_ue8m0=True,
            compiled_fn=compiled.fn,
            use_pdl=True,
        )

    def reference(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSide,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fresh `(q_fp8, weights)`; `q_fp8_out`, `weights_out` and `compiled` are not read.

        The rope FMA is evaluated in fp64 and rounded to fp32 once, which is what
        a fused multiply-add does; every later step is an exact fp32 or bf16
        rounding, so the result is bit-exact rather than "close".
        """
        tokens = q.shape[0]
        pe_bf16 = q[..., :_ROPE_DIM]
        x = pe_bf16.float()
        bits = pe_bf16.contiguous().view(torch.int16)
        subnormal = (bits & 0x7F80) == 0
        x = torch.where(subnormal, torch.copysign(torch.zeros_like(x), x), x)
        nope = q[..., _ROPE_DIM:].float()

        table = cos_sin[position_ids.long()]  # [T, 64]
        cos, sin = table[:, :_HALF], table[:, _HALF:]
        if is_neox:
            partner = torch.cat([x[..., _HALF:], x[..., :_HALF]], dim=-1)
            cos_e = torch.cat([cos, cos], dim=-1)
            sin_e = torch.cat([sin, sin], dim=-1)
            sign = torch.cat([-torch.ones(_HALF), torch.ones(_HALF)])
        else:
            pairs = x.reshape(tokens, _HEADS, _HALF, 2)
            partner = pairs.flip(-1).reshape(tokens, _HEADS, _ROPE_DIM)
            cos_e = cos.repeat_interleave(2, dim=-1)
            sin_e = sin.repeat_interleave(2, dim=-1)
            sign = torch.tensor([-1.0, 1.0]).repeat(_HALF)
        sign = sign.to(device=q.device, dtype=torch.float32)
        cos_e, sin_e = cos_e[:, None, :], sin_e[:, None, :]
        partner_term = (partner * sin_e) * sign  # one fp32 rounding, then an exact sign flip
        rotated = (x.double() * cos_e.double() + partner_term.double()).float()
        rotated = rotated.to(torch.bfloat16).float()

        row = torch.cat([rotated, nope], dim=-1)  # [T, 32, 128]
        amax = row.abs().amax(dim=-1)
        floor = torch.tensor(_MIN_AMAX, dtype=torch.float32, device=q.device)
        inv_max = torch.tensor(1.0 / _E4M3_MAX, dtype=torch.float32, device=q.device)
        raw_scale = torch.maximum(amax, floor) * inv_max
        scale, inv_scale = _ue8m0_round_up(raw_scale)
        q_fp8 = (row * inv_scale[..., None]).to(torch.float8_e4m3fn)
        c = torch.tensor(weight_scale, dtype=torch.float32, device=q.device)
        weights = (wq * scale) * c
        return q_fp8, weights

    def is_valid(
        self,
        q: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        wq: torch.Tensor,
        weight_scale: float,
        q_fp8_out: torch.Tensor,
        weights_out: torch.Tensor,
        *,
        is_neox: bool,
        compiled: CompiledIndexerQSide,
    ) -> None:
        # The launcher's rope mode is fixed at compile time and the wrapper does
        # not check it against `is_neox`: the other mode returns plausible wrong
        # codes (max abs ~7.8 against 2.4e-4).
        assert compiled.is_neox == bool(is_neox), (
            f"compiled for is_neox={compiled.is_neox}, called with is_neox={is_neox}"
        )
        # Positions are read on the device with no bound.
        if position_ids.numel():
            lo, hi = int(position_ids.min()), int(position_ids.max())
            assert 0 <= lo and hi < cos_sin.shape[0], (
                f"position_ids span [{lo}, {hi}] but the rope table has "
                f"{cos_sin.shape[0]} rows; the kernel reads past it"
            )


def _ue8m0_round_up(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(2**ceil(log2 scale), 2**-ceil(log2 scale))` by exponent arithmetic, as the kernel does."""
    bits = scale.contiguous().view(torch.int32)
    exp_bits = bits & 0x7F800000
    exp_bits = torch.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
    inv_bits = 0x7F000000 - exp_bits
    return exp_bits.view(torch.float32), inv_bits.view(torch.float32)


cute_dsl_indexer_q_side_fused = CuteDslIndexerQSideFused()
