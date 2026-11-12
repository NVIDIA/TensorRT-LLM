# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One Gated-DeltaNet decode step per row over the quantized state pool with the GDN gated
RMS norm (silu gate, norm before gate) applied to its output in the same kernel.

Triton (``_gdn_decode_t1``): one program of 4 or 8 warps owns a (row, value head)'s
whole ``[V, K]`` state, runs the single-token step, rounds ``o`` to bf16 and normalizes
it in registers -- bitwise the single-token kernel followed by the multi-row gated RMS
norm kernel.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major,
)

from ._gdn_decode_t1 import check_operands, launch_t1_gated_norm, reference_step
from ._gdn_state_q8 import dequantize_slots, store_slots

#: This checkpoint's Gated DeltaNet at tp1 and its gated norm's epsilon.
H, HV, K, V = 16, 48, 128, 128
EPS = 1e-6


def _cell(why: str, rows: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, **spec))


class GdnDecodeT1GatedNorm(OpWrapper):
    """`gdn_decode_t1`'s step for each of N sequences, then per (row, value head)
    `y = rms_norm(bf16(o)) * norm_weight * silu(z)`:

        o  = gdn_decode_t1(q, k, v, a, b, A_log, dt_bias, pool, slots, None)  # pool updated
        y  = rms_norm_gated_token_major(o.reshape(N * HV, V), z, norm_weight, eps)

    with `rms(x) = x / sqrt(mean(x^2) + eps)` over V. Returns `y` `[N * HV, V]` bf16
    (row `n * HV + i`), a new tensor; `pool` is updated in place as `gdn_decode_t1`
    updates it, bit for bit.

    Fusion boundary: `gdn_decode_t1`'s, plus the gated norm. The projections producing
    q/k/v and z and the output projection consuming `y` are the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: rows; pool ("compact" | "coalesced"); pad (row 1 given slot -1). z is the
    # [N, 6144] column view of the [N, 16384] in_proj_qkvz output.
    CELLS: tuple[Cell, ...] = (
        _cell("decode 5 rows: the first fused row count, 8 warps; a pad slot (-1)", 5, pad=True),
        _cell("decode 8 rows: 4 warps from here", 8),
        _cell("decode 32 rows, the largest fused row count; coalesced pool", 32, pool="coalesced"),
    )

    note = """
    Metadata consumed: `slots` -- the pure decode step's int32 state indices of its N rows.

    Call site (modeling): a pure decode step at 5..32 rows. `gdn_decode_t1`'s arguments
    (without `out`) as it takes them, then z the `[N, 48, 128]` view of the
    in_proj_qkvz output's z columns (token stride 16384, head stride 128, unit inner
    stride); `norm_weight` bf16 `[128]`; `eps` 1e-6. `y` goes straight to out_proj.

    Launch: grid `N * HV`, 8 warps below 8 rows, 4 from 8 rows, one stage. No PDL.

    Numerics: the state and `o` exactly as `gdn_decode_t1` (same q/k L2 norm form, same
    K-contraction order: warps split V only); `o` rounded to bf16, then the norm's
    sum of squares in the multi-row norm kernel's single-warp order
    (`_warp32x4_sum_sq`) and its elementwise ops in the same order, `y` rounded once.
    Bitwise equal to `gdn_decode_t1` followed by `rms_norm_gated_token_major` (y and
    pool) -- a property of the Triton build whose reduction order it replays.

    Not checked by the op (is_valid covers it): `gdn_decode_t1`'s conditions, z's
    layout.

    Hard limits
      K == V == 128  (static_assert in the kernel)
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
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return launch_t1_gated_norm(q, k, v, a, b, A_log, dt_bias, pool, slots, z, norm_weight, eps)

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
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(y, pool_after)`: `gdn_decode_t1`'s float64 step (`o` rounded to bf16, the
        new states quantized into a copy of the pool), then the gated norm entry's fp32
        reference. Mutates nothing."""
        slot_list = slots.tolist()
        h = dequantize_slots(pool, slot_list)
        o, h_after = reference_step(q, k, v, a, b, A_log, dt_bias, h)
        pool_after = pool.clone()
        store_slots(pool_after, slot_list, h_after.float())
        o = o.to(q.dtype).reshape(-1, o.shape[-1])
        y = rms_norm_gated_token_major.reference(o, z, norm_weight, eps, None, "silu")
        return y, pool_after

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
        z: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> None:
        check_operands(q, k, v, a, b, A_log, dt_bias, pool, slots)
        n, hv, vd = v.shape[1:]
        assert z.dtype == torch.bfloat16 and z.shape == (n, hv, vd), "z bf16 [N, HV, V]"
        assert z.stride(2) == 1 and z.stride(1) == vd, (
            "z's (HV, V) block must be contiguous per token: the kernel reads head i at "
            "token base + i * V"
        )
        assert norm_weight.shape == (vd,), "norm_weight [V]: the kernel reads V elements"


gdn_decode_t1_gated_norm = GdnDecodeT1GatedNorm()
