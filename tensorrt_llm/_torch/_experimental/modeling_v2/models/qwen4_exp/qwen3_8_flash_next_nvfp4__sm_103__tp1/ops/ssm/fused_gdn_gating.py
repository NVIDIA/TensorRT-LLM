# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated-DeltaNet decay gate: g = -exp(A_log) * softplus(a + dt_bias), per token
and per value head, in one Triton kernel.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.mamba.gdn_mixer.fused_gdn_gating`` (tier-2
sourcing: no op boundary exists). Its body allocates the fp32 output and
launches ``fused_gdn_gating_kernel`` once.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba.gdn_mixer import fused_gdn_gating as _fused_gdn_gating

#: This checkpoint's GDN value-head count at tp1, and the `[b | a]` width the
#: stock mixer slices `a` out of.
NUM_V_HEADS = 48
BA_WIDTH = 2 * NUM_V_HEADS

#: Read off this checkpoint's 48 layers of `linear_attn.A_log` /
#: `linear_attn.dt_bias` (bf16, 48 each): exp(A_log) spans [3.839e-03, 1.394e+02],
#: and dt_bias reaching 19.25 against the default threshold of 20.0 makes the
#: softplus's linear branch a live path, not a corner case.
A_LOG_RANGE = (-5.5625, 4.9375)
DT_BIAS_RANGE = (-8.9375, 19.25)

#: Relative error of the SASS approximate transcendentals (`ex2.approx.f32`,
#: `lg2.approx.f32`) Triton lowers `tl.exp` / `tl.log` to.
APPROX_ULP = 2.0**-22


class FusedGdnGating(OpWrapper):
    """`g[t, h] = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])`, computed in
    fp32 whatever the operand dtypes, into a fresh contiguous fp32
    `[batch, num_heads]` tensor. `batch, num_heads = a.shape`.

    `softplus(x) = log(1 + exp(beta * x)) / beta` while `beta * x <= threshold`,
    else `x` -- the unstabilised form: below `x = -24 ln 2 = -16.6355` the fp32
    sum drops `exp(x)` and `g` is exactly 0 (absolute error <= 6e-8 times
    exp(A_log), relative error 1.0). Harmless for a log-domain decay.

    Fusion boundary: bias add, softplus with its threshold branch, `exp(A_log)`
    and the product are inside. The projection producing `a`, the
    recurrence's other gate `beta = sigmoid(b)`, and the recurrence are not.
    No input is modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows; layout "contiguous" | "target_rows" (the decode rows of a
    # contiguous [T, 48] `a` projection, past T - rows prefill rows); fill
    # (constant a). `a` bf16, softplus beta 1.0, threshold 20 throughout.
    CELLS: tuple[Cell, ...] = (
        Cell(why="target verify call: 32 seqs x 2 tokens, contiguous a", spec=dict(rows=64)),
        Cell(
            why="target mixed verify: decode rows of a [T, 48] projection",
            spec=dict(rows=16, layout="target_rows"),
        ),
        Cell(
            why="a = 200: linear branch; exp(x) is evaluated, inf never leaks",
            spec=dict(rows=4, fill=200.0),
        ),
    )

    note = """
    Metadata consumed: none. Stateless and per (token, head).

    Call site (`_gdn_verify`): `(A_log, a_d, dt_bias, 1.0, 20.0)` with `a_d`
    the `[n * p, 48]` bf16 decode rows of the `a` projection (all of it on a
    pure verify step) and fp32 `A_log` / `dt_bias`; `g` is then reshaped to
    `[n, p, 48]` for `fused_recurrent_gated_delta_rule_update`. Certified at
    that call: 32 sequences of 2 tokens, the 16 decode rows of a mixed step's
    `[T, 48]` projection, and `a = 200` (the softplus's linear branch, where
    `exp` overflows).

    Raise loudly, so not guarded: a non-2-D `a` (`ValueError` unpacking
    `a.shape`), CPU tensors (`ValueError: Pointer argument cannot be accessed
    from Triton`).

    `a`'s row stride is honoured (a row is read at `row * a.stride(0)`; the
    target passes contiguous rows); everything else is addressed without
    strides. `is_valid` guards the three violations measured to return a finite wrong `g`: a stride-2 A_log
    view moved g by 6.031e+01 and a stride-2 dt_bias by 3.832e-01; 40-element
    A_log / dt_bias against 48 heads moved it by 1.009e+02 / 1.010e+02 (the
    per-head load is masked against `a.shape[1]` only); an `a` with head
    stride 2 by 1.153e+00.

    Deterministic: two launches agree bitwise.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> torch.Tensor:
        return _fused_gdn_gating(A_log, a, dt_bias, beta, threshold)

    def reference(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> torch.Tensor:
        """fp64 with the stabilised `log1p(exp(x))`, rounded once to fp32.

        The stabilised form makes the reference the accurate side; the
        kernel's unstabilised tail differs inside the test's absolute floor.
        """
        num_heads = a.shape[-1]
        x = a.double() + dt_bias[:num_heads].double()
        softplus = torch.where(beta * x <= threshold, torch.log1p(torch.exp(beta * x)) / beta, x)
        return (-torch.exp(A_log[:num_heads].double()) * softplus).float()

    def is_valid(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> None:
        # `shape[-1]` / `stride(-1)` so a non-2-D `a` still reaches the op's own
        # ValueError rather than being preempted here.
        num_heads = a.shape[-1]
        assert a.stride(-1) == 1, (
            f"`a`'s head axis must be unit-stride; got strides {a.stride()} -- the kernel "
            "reads a row as `row * a.stride(0) + head`, so another column stride returns "
            "a finite wrong g"
        )
        assert A_log.stride(0) == 1 and dt_bias.stride(0) == 1, (
            f"A_log and dt_bias must be contiguous 1-D; got strides {A_log.stride()} and "
            f"{dt_bias.stride()} -- both are read as `ptr + head`"
        )
        assert A_log.numel() >= num_heads and dt_bias.numel() >= num_heads, (
            f"A_log and dt_bias must hold at least a.shape[-1]={num_heads} elements; got "
            f"{A_log.numel()} and {dt_bias.numel()} -- the per-head load is masked against "
            "a.shape[-1] only, so a short tensor is read out of bounds"
        )


fused_gdn_gating = FusedGdnGating()
