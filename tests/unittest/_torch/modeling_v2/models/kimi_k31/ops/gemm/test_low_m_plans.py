# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 low-M plans (``ops/gemm/low_m_plans``) on CPU: every projection a layout's plan lists is a per-rank
projection shape of that layout, and the helix8 drafter's bit-identical plan is its D-class plan's bitwise part."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
    K31_LOW_M_PLANS,
)

pytestmark = pytest.mark.cpu_only

H = 7168
#: KDA [q | k | v | g] rows; MLA g_proj rows and the KDA / MLA o_proj K (96 heads x 128); the packed DSA hidden
#: projection [q_a 1536 | kv_a 576 | wk 128 | weights_proj 32]; MLA q_b rows (96 x 192); shared-expert, drafter and
#: dense layer-0 MLP widths.
_KDA_QKVG, _HEADS_X_V, _DSA_HIDDEN, _Q_B = 4 * 12288, 12288, 2272, 18432
_SHARED_MLP, _DRAFTER_MLP, _DENSE_MLP = 6144, 14336, 33792
#: KDA heads and head dim: the gate pair [f_a | b] packs f_a (head dim rows) and b (one row per head), padded to 8 rows.
_KDA_HEADS, _KDA_HEAD_DIM = 96, 128

#: Plan -> (attention TP size, attention DP). ``tep8_draft`` is the DFlash2 drafter's own plan at TEP8;
#: ``tep16_shared`` is the TEP16 shared expert's plan.
LAYOUTS = {
    "tep32": (32, False),
    "tep16": (16, False),
    "tep16_shared": (16, False),
    "tep8": (8, False),
    "tep8_draft": (8, False),
    "dep16": (16, True),
    "helix8_draft": (8, False),
    "helix8_draft_d": (8, False),
}
#: The helix8 generation target's DFlash2 drafter: its projections replicated (96 heads, the 5 layers' history kv_a)
#: except o_proj (12 heads after the helix exchange) and the MLP down, row-parallel over the 8 ranks.
_HELIX8_DRAFT = [
    ("drafter_conv_proj", 2 * 2 * (H // 16), H),
    ("drafter_q_a", 1536, H),
    ("drafter_q_b", _Q_B, 1536),
    ("drafter_kv_a", 576, H),
    ("drafter_q_a_kv_a", 2112, H),
    ("drafter_o_proj", H, _HEADS_X_V // 8),
    ("drafter_gate_up", 2 * _DRAFTER_MLP // 8, H),
    ("drafter_down", H, _DRAFTER_MLP // 8),
    ("drafter_ctx_kv", 5 * 576, H),
]


def projections(layout: str) -> list:
    """``(label, N, K)`` per rank of the layout's planned projections. At TEP8 the DFlash2 drafter has its own plan
    (``tep8_draft``: its projections, q_b included) and the target plan leaves it out."""
    if layout.startswith("helix8_draft"):
        return list(_HELIX8_DRAFT)
    tp, dp = LAYOUTS[layout]
    t = 1 if dp else tp  # attention DP replicates every projection
    rows = [
        ("kda_qkvg", _KDA_QKVG // t, H),
        ("kda_gates", -(-(_KDA_HEAD_DIM + _KDA_HEADS // t) // 8) * 8, H),
        ("dsa_hidden_gate", _DSA_HIDDEN + _HEADS_X_V // t, H),
        ("dsa_hidden", _DSA_HIDDEN, H),
        ("mla_g_shared_gate_up", _HEADS_X_V // t, H),
        ("shared_down", H, _SHARED_MLP // t),
        ("mla_q_b", _Q_B // t, 1536),
        ("indexer_wq_b", 4096, 1536),
        ("latent_down", 3584, H),
        ("latent_up", H, 3584),
        ("drafter_conv_proj", 2 * 2 * (H // 16), H),
        ("drafter_q_a_kv_a", 2112, H),
        ("drafter_gate_up", 2 * _DRAFTER_MLP // t, H),
        ("drafter_ctx_kv", 5 * 576, H),
        ("dense_gate_up", 2 * _DENSE_MLP // t, H),
    ]
    if dp:
        # The attention-TP layouts run these two with their all-reduce in one kernel (the AR-push GEMM).
        rows += [
            ("o_proj", H, _HEADS_X_V),
            ("drafter_down", H, _DRAFTER_MLP),
        ]
    if layout == "tep8":
        rows = [r for r in rows if not r[0].startswith("drafter_")]
    elif layout == "tep8_draft":
        rows = [r for r in rows if r[0].startswith("drafter_")]
        rows.append(("drafter_q_b", _Q_B // t, 1536))
    return rows


@pytest.mark.parametrize("layout", sorted(K31_LOW_M_PLANS))
def test_plan_rows_are_model_projections(layout) -> None:
    shapes = {(n, k) for _, n, k in projections(layout)}
    assert set(K31_LOW_M_PLANS[layout]) <= shapes, sorted(set(K31_LOW_M_PLANS[layout]) - shapes)


def test_helix8_d_plan_names_a_kernel_wherever_the_bit_identical_plan_does() -> None:
    """Every (shape, rows) the helix8 drafter's bit-identical plan names a kernel for, its D-class plan names one too
    (the same, or a faster one of another summation order): the D-class lever never sends a planned row to cuBLAS."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import (
        plan_choice,
    )

    b, d = K31_LOW_M_PLANS["helix8_draft"], K31_LOW_M_PLANS["helix8_draft_d"]
    for n, k in b:
        for m in range(1, 129):
            if plan_choice(b, m, n, k) is not None:
                assert plan_choice(d, m, n, k) is not None, (n, k, m)
