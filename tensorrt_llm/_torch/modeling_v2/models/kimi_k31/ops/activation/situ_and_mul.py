# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply (Kimi K3.1's dense and shared-expert MLP activation) via the fused Triton kernel."""

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.activation.situ_and_mul import (
    situ_and_mul as _situ_and_mul,
)

#: Kimi K3.1's activation_situ_beta and activation_situ_linear_beta.
_BETA, _LINEAR_BETA = 4.0, 25.0
#: The dense layer-0 MLP (intermediate 33792) splits over min(gcd(33792, tp), 4) ranks (one GB300
#: tray), the two shared experts (2 x 3072) over the full attention TP group; attention DP replicates
#: both.
_DENSE_WIDTHS = {1: 33792, 2: 16896, 4: 8448}
_SHARED_WIDTHS = {tp: 6144 // tp for tp in (1, 2, 4, 8, 16, 32)}


class K31SituAndMul(OpWrapper):
    """Return `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)` for `[g | u] = x`.

    `g, u = x[:, :d], x[:, d:]` with `d = x.shape[-1] // 2`. A new [M, d] tensor in `x.dtype`.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"K3.1 dense MLP over {tp} rank(s) (d={d}), M={rows}",
                spec=dict(shape=(rows, 2 * d), beta=_BETA, linear_beta=_LINEAR_BETA),
            )
            for tp, d in _DENSE_WIDTHS.items()
            for rows in (1, 8, 2048)
        ),
        *(
            Cell(
                why=f"K3.1 shared experts at attention TP{tp} (d={d}), M={rows}",
                spec=dict(shape=(rows, 2 * d), beta=_BETA, linear_beta=_LINEAR_BETA),
            )
            for tp, d in _SHARED_WIDTHS.items()
            for rows in (1, 5, 40, 2048)
        ),
        Cell(
            why="saturated positive, negative and zero gates at magnitude 1e4 (bf16)",
            spec=dict(shape=(4, 768), beta=_BETA, linear_beta=_LINEAR_BETA, saturated=True),
        ),
        *(
            Cell(
                why=f"K3.1 Rubin {what} (d={d}), M={rows}",
                spec=dict(shape=(rows, 2 * d), beta=_BETA, linear_beta=_LINEAR_BETA),
            )
            for what, d in (
                ("shared experts over TP8", 768),
                ("shared experts over TP2", 3072),
                ("shared experts unsplit", 6144),
                ("dense MLP over TP8", 33792 // 8),
                ("dense MLP unsplit", 33792),
            )
            for rows in (8, 96, 16384)
            # The unsplit dense MLP at 8 rows is already a cell above.
            if (d, rows) != (33792, 8)
        ),
    )
    note = """
    Activation and gating only, without quantization; `x` is read-only and must
    be 2-D with a unit last stride (rows may be strided). The caller owns the
    projections: in the family K31SituMLP the fused [gate | up] GEMM output feeds
    this op and its output feeds down_proj.
    Kimi K3.1 calls it in bf16 with beta 4.0 and linear_beta
    25.0 for the dense layer 0 and the two shared experts of every MoE layer;
    the routed experts run SiTU inside the MXFP4 MoE runner instead.

    No metadata or state. Triton specializes on dtype and on divisibility of
    widths, strides and pointers; warm each
    specialization outside CUDA-graph capture, after which replay is
    bit-identical to eager. PDL (TRTLLM_ENABLE_PDL, sm_90+) changes launch
    scheduling, not results. libdevice tanh differs from torch.tanh in the last
    fp32 bits, which moves a bf16 result by at most one ulp.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200).
    """

    def __call__(self, x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
        return _situ_and_mul(x, beta, linear_beta)

    def reference(self, x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
        """fp32: SiTU on the gate half times the soft-capped up half, rounded once to x.dtype."""
        d = x.shape[-1] // 2
        gate = x[:, :d].float()
        up = x[:, d:].float()
        situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        up = linear_beta * torch.tanh(up / linear_beta)
        return (situ * up).to(x.dtype)


situ_and_mul = K31SituAndMul()
