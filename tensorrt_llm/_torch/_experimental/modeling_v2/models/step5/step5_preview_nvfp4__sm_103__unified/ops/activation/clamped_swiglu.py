# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU with the clamp after SiLU: `min(silu(g), L) * clamp(u, -L, L)`, the
activation of Step-5's dense MLP and of every MoE layer's shared expert.

Tier 4, composed from torch elementwise ops: no engine kernel computes this
function. Rejected at the cheaper rungs:

- `torch.ops.trtllm.silu_and_mul(x, swiglu_limit=L)` (`modules/swiglu.py`
  `silu_and_mul_kernel`) clamps the gate *before* SiLU, `silu(min(g, L))`.
  For `g > L` that is `silu(L)` (6.99362 at L = 7) where Step-5 takes `L`; it
  also passes `silu(g)` through for `g` in `(L, silu^-1(L)]` (up to 7.0064)
  where both agree. The test measures how far the two land apart.
- `catalog/activation/flashinfer_silu_and_mul` has no clamp at all, and
  FlashInfer's activation module has no clamped variant.
- trtllm-gen `gemm1_clamp_limit` is the same pre-SiLU clamp, inside an MoE
  runner rather than a standalone activation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: `swiglu_limits[i]` and `swiglu_limits_shared[i]` in the checkpoint's
#: `text_config`: 7.0 at every one of the 95 layers.
LIMIT = 7.0

#: Per-rank intermediate widths I (the activand is `[T, 2I]`): the dense MLP of
#: layers 0-2 and 91 (`intermediate_size` 13824) and the shared expert of
#: layers 3-90 (`share_expert_dim` 1536), unsplit under attention DP and split
#: four ways under TP4.
DENSE_DP, DENSE_TP4 = 13824, 3456
SHARED_DP, SHARED_TP4 = 1536, 384


def _cell(why: str, inter: int, rows: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, inter=inter, limit=LIMIT, dtype=torch.bfloat16))


class ClampedSwiglu(OpWrapper):
    """`out = min(silu(g), limit) * clamp(u, -limit, limit)` over
    `gate_up = [..., 2I]`, gate half first (`g = gate_up[..., :I]`,
    `u = gate_up[..., I:]`); a new `[..., I]` tensor in `gate_up`'s dtype.

    Both halves are widened to fp32 before SiLU, the clamps and the product, and
    only the product is rounded back, once. The gate has no lower clamp: SiLU is
    bounded below by about -0.278, so a lower clamp at `-limit` is inert.

    Fusion boundary: the activation only. The gate|up projection before it and
    the down projection after it are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inter (I), limit, dtype. The test fills the activand with a
    # bulk N(0, 3^2) and plants every eighth element of each half in the clamp
    # boundary band (gate in [6.9, 9], up in +-[6.9, 9]).
    CELLS: tuple[Cell, ...] = (
        _cell("dense dp I 13824: prefill 8192 rows, the envelope's largest", DENSE_DP, 8192),
        _cell("dense dp I 13824: decode 128 rows, largest captured batch", DENSE_DP, 128),
        _cell("dense dp I 13824: decode 1 row", DENSE_DP, 1),
        _cell("dense tp4 I 3456: prefill 8192 rows", DENSE_TP4, 8192),
        _cell("dense tp4 I 3456: decode 24 rows", DENSE_TP4, 24),
        _cell("dense tp4 I 3456: decode 1 row", DENSE_TP4, 1),
        _cell("shared dp I 1536: prefill 8192 rows", SHARED_DP, 8192),
        _cell("shared dp I 1536: chunked prefill 512 rows", SHARED_DP, 512),
        _cell("shared dp I 1536: decode 128 rows", SHARED_DP, 128),
        _cell("shared dp I 1536: decode 1 row", SHARED_DP, 1),
        _cell("shared tp4 I 384: prefill 8192 rows", SHARED_TP4, 8192),
        _cell("shared tp4 I 384: decode 24 rows", SHARED_TP4, 24),
        _cell("shared tp4 I 384: decode 1 row", SHARED_TP4, 1),
    )

    note = """
    Metadata consumed: none. Out-of-place; `gate_up` is not mutated (asserted
    per cell). Seven elementwise launches (widen gate, SiLU, clamp, widen up,
    clamp, multiply, narrow) with up to three fp32 `[T, I]` temporaries live.
    Measured on sm_103: 1327 us and 1.27 GiB peak above the input (output
    included) at 8192 x 13824; 344 us at 8192 x 3456; 136 us at 8192 x 1536;
    about 46 us at decode sizes, where the seven launches dominate. The tuning
    phase replaces it with one fused kernel.

    Caller preconditions: `gate_up`'s last dimension is `2I`, gate half first,
    as the fused gate|up projection writes it. The halves are taken as strided
    views of it; nothing is copied to compact them.

    Numerics: bf16 output equals the fp64 reference rounded once to bf16 except
    where the fp32 evaluation lands within its own few-ulp error of a bf16
    rounding midpoint; those elements are one bf16 ulp off. Measured: at most
    1 ulp, off-by-ulp fraction at most 2.6e-6 over the cells; the test gates
    each element at 1 ulp and the fraction at 1e-3. The rejected pre-SiLU clamp
    (`trtllm::silu_and_mul(swiglu_limit=7.0)`) is also never more than 1 ulp
    off, but at a fraction of 2.09e-2 on the boundary-rich activand, which the
    gate refuses; in fp32 it lands 4.46e-2 away where the entry's own deviation
    is 5.9e-6.

    Guards (`is_valid`): an odd last dimension (width 3 splits 2 | 1 and
    broadcasts to a plausible `[T, 2]` with no error); a negative or NaN
    `limit` (at -7 every output is 49; at NaN every output is NaN; no error).

    Measured, not certified (op and reference agree): gate +inf gives
    `limit * clamp(u)`; gate -inf gives NaN (SiLU's `-inf / inf`); gate -1e30
    gives -0.0; up +-inf saturates to +-limit; NaN in either half propagates.

    Certified: bf16 `[T, 2I]` contiguous activands at limit 7.0, I in
    {13824, 3456, 1536, 384}, T in {1, 24, 128, 512, 8192} as the cells name;
    CUDA-graph capture and replay at 128 x 2 * 13824 (the decode site is
    captured).

    Hard limits
      numel of `gate_up`  torch's elementwise kernels split or widen past
      2^31 elements  largest certified 8192 x 27648 = 2.26e8, below it; the
      envelope does not reach 2^31 (8192 rows would need I > 131072).

    Not certified: limits other than 7.0, fp16/fp32 activands (the fp32
    identity test runs the op in fp32 but certifies no cell), non-contiguous
    or 3-D activands, 0 rows, other widths, torch.compile.
    """

    def __call__(self, gate_up: torch.Tensor, limit: float) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate.float()).clamp_(max=limit)
        return activated.mul_(up.float().clamp(-limit, limit)).to(gate_up.dtype)

    def reference(self, gate_up: torch.Tensor, limit: float) -> torch.Tensor:
        """The closed form in fp64 from primitive ops, rounded once to `gate_up.dtype`.

        SiLU is written out as `g / (1 + exp(-g))` rather than taken from
        `F.silu`, the call `__call__` makes.
        """
        width = gate_up.shape[-1] // 2
        g = gate_up[..., :width].double()
        u = gate_up[..., width:].double()
        activated = (g / (1.0 + torch.exp(-g))).clamp(max=limit)
        return (activated * u.clamp(-limit, limit)).to(gate_up.dtype)

    def is_valid(self, gate_up: torch.Tensor, limit: float) -> None:
        assert gate_up.shape[-1] % 2 == 0, (
            f"clamped_swiglu: the last dimension {gate_up.shape[-1]} is not gate|up "
            "halves of equal width"
        )
        assert limit >= 0.0, f"clamped_swiglu: limit {limit} must be non-negative and not NaN"


clamped_swiglu = ClampedSwiglu()
