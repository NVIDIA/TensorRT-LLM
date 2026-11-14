# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Gated-DeltaNet decode step per row over the quantized state pool: gates, q/k L2
norm, the delta rule, and the state dequantized and requantized in place.

Triton (``_gdn_decode_t1``): the generic single-warp recurrent kernel specialized for
one token per sequence, with a flat ``N * HV * (V / BV)`` grid whose ``[BV, K]`` state
tiles are laid out like the pool so the int16 rows load as full vectors.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._gdn_decode_t1 import check_operands, launch_t1, reference_step
from ._gdn_state_q8 import dequantize_slots, store_slots

#: This checkpoint's Gated DeltaNet at tp1: 16 q/k heads, 48 v heads, head dims 128.
H, HV, K, V = 16, 48, 128, 128
#: A pool past the largest decode batch (128 rows) the target runs.
POOL_SLOTS = 160


def _cell(why: str, rows: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, **spec))


class GdnDecodeT1(OpWrapper):
    """One gated-delta-rule step for each of N sequences (one token each), state in the
    quantized pool. Row `n` uses slot `s = slots[n]`, value head `i` key head `i // (HV // H)`:

        g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
        q = q / sqrt(|q|^2 + 1e-6) * K^-0.5;  k = k / sqrt(|k|^2 + 1e-6)
        h = deq(pool[s]) * exp(g);  u = (v - h k) * beta;  h += u k^T;  o = h q
        pool[s] = quant(h)

    `deq` / `quant` are the pool's int16 rows with one fp32 scale (`amax / 32767`) per row
    (see `_gdn_state_q8`). A slot of -1 reads a zero state and writes nothing.

    Fusion boundary: both gates, the q/k L2 norm, the recurrence, the state's
    dequantize / requantize. The conv producing q/k/v, slot allocation and the gated
    norm on `o` are the caller's.

    Returns `o` `[N, HV, V]` bf16 -- `out` itself when given, else a new tensor.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: rows; pool ("compact": POOL_SLOTS contiguous slots | "coalesced": the cache
    # manager's view, slot stride 48 layers' states); pad (row 1 given slot -1); out
    # ("new" | "row_range": rows [P, P + N) of a mixed step's [1, T, HV, V] output).
    CELLS: tuple[Cell, ...] = (
        _cell(
            "decode 1 row (BV 8), coalesced pool: slot 100 past 2^31 bytes",
            1,
            pool="coalesced",
        ),
        _cell("decode 2 rows: N * HV = 96, the last BV 8 launch; a pad slot (-1)", 2, pad=True),
        _cell("decode 3 rows: the first BV 16 launch", 3),
        _cell("decode 128 rows (concurrency 128), BV 16", 128),
        _cell(
            "mixed step: 124 decode riders into rows [P, T) of the step output",
            124,
            out="row_range",
        ),
    )

    note = """
    Metadata consumed: `slots` -- the step's int32 state indices of these rows (pure
    decode: `state_indices[:N]`; a mixed step: its decode riders, `state_indices_d`).

    Call site (modeling): q, k `[1, N, 16, 128]` and v `[1, N, 48, 128]` views of the
    conv output rows (any token stride, heads packed: head stride 128, unit inner
    stride); a, b `[N, 48]` column views of the in_proj_ba output (unit column
    stride); A_log, dt_bias `[48]`; the layer's whole int8-typed pool `[slots, 48, 260, 128]`
    unsliced; `out` None at a pure decode step, `output[0, P:]` (the riders' rows of the
    contiguous `[1, T, 48, 128]` step output) at a mixed step.

    Pool contract: int8 `[S, HV, V + 4, K]`, inner strides `((V + 4) * K, K, 1)`, the
    slot stride and the base 16-byte aligned (the fp32 scales are read through the
    int8 base). Any slot stride (the cache manager's coalesced view is certified past
    2^31 bytes). Zero bytes are an exact zero state.

    Launch: BV 8 when N * HV <= 96 (one or two rows), else 16; one warp per program
    (every reduction in the generic single-warp kernel's order), one stage, grid
    `N * HV * (V / BV)`. No PDL.

    Numerics: fp32 inside, the state rounded once per step to int8 rows (absolute
    rounding <= 1/254 of the row maximum), `o` from the unrounded state and rounded
    once to bf16. The q/k L2 norm is `x / sqrt(sum x^2 + 1e-6)`, the generic decode
    kernel's and the prefill front end's form.

    Not checked by the op (is_valid covers it): pool format, operand strides, slots
    distinct and in range (two rows naming one slot race; an out-of-range slot is an
    illegal access).

    Hard limits
      slot offset  int64 (slot * slot stride)  certified at slot 100 x 38.9 MB = 3.9e9 B
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if out is None:
            out = q.new_empty(v.shape[1:])
        return launch_t1(q, k, v, a, b, A_log, dt_bias, pool, slots, out)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(o, pool_after)`: the float64 step from the dequantized states, `o` rounded
        once to `q.dtype`, the new states quantized into a copy of the pool (int16 rows,
        `amax / 32767` scales). Mutates nothing; `out` is ignored."""
        slot_list = slots.tolist()
        h = dequantize_slots(pool, slot_list)
        o, h_after = reference_step(q, k, v, a, b, A_log, dt_bias, h)
        pool_after = pool.clone()
        store_slots(pool_after, slot_list, h_after.float())
        return o.to(q.dtype), pool_after

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        pool: torch.Tensor,
        slots: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> None:
        check_operands(q, k, v, a, b, A_log, dt_bias, pool, slots)
        n, hv, vd = v.shape[1:]
        if out is not None:
            assert out.dtype == q.dtype and out.shape == (n, hv, vd) and out.is_contiguous(), (
                "out must be a contiguous [N, HV, V] tensor of q's dtype: the kernel writes "
                "row n, head i at (n * HV + i) * V"
            )


gdn_decode_t1 = GdnDecodeT1()
