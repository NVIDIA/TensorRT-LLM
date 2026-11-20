# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.registry import (
    FMHA_LIBS,
    get_enabled_fmha_lib_classes,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.components import (
    trtllm_attention_prims as prims,
)
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "value",
    [
        None,
        ",".join(FMHA_LIBS),
        "fallback,prims_ts_block_sparse,fallback",
        "+prims_ts,-fallback,+beam_shared_prefix,-prims_ts",
        "+unknown",
        "prims_ts,+fallback",
        "+",
    ],
)
def test_prims_fmha_policy(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", value)
    try:
        enabled = get_enabled_fmha_lib_classes()
    except ValueError as error:
        with pytest.raises(ValueError) as candidate:
            prims._enabled_fmha_libraries()
        assert str(candidate.value) == str(error)
    else:
        names = {cls: name for name, cls in FMHA_LIBS.items()}
        assert prims._enabled_fmha_libraries() == tuple(names[cls] for cls in enabled)


def test_optional_sage_plan_fallback(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts_block_sparse,fallback")
    recipe = QuantAttentionConfig(
        qk_dtype="fp8",
        v_dtype="fp8",
        q_block_size=1,
        k_block_size=16,
        v_block_size=1,
        smooth_v=True,
    )
    instance = prims.H3PrimTS(recipe, {})
    x = torch.empty(1, 32, 2, 64, dtype=torch.bfloat16)
    monkeypatch.setattr(
        prims,
        "prims_routed_attention",
        lambda *a, **kw: pytest.fail("unsupported head dimension reached compiler"),
    )
    assert instance(x, x, x, None, skip_softmax=False, skip_threshold_active=False) is None

    def broken(*a, **kw):
        raise ValueError("compiler error after profile admission")

    monkeypatch.setattr(prims, "prims_routed_attention", broken)
    y = torch.empty(1, 32, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="compiler error after profile admission"):
        instance(y, y, y, None, skip_softmax=False, skip_threshold_active=False)


def test_vc_denoising_schedule(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    recipe = QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8", vc_attention=True)
    state = {}
    impl = prims.H3PrimTS(recipe, state)
    x = torch.randn(1, 16, 1, 128, dtype=torch.bfloat16)
    calls = []
    refreshes = []

    def refresh(grouping, values):
        refreshes.append(state["denoise_step"][0])
        if grouping.permutation is None:
            grouping.permutation = (
                torch.arange(values.size(1), dtype=torch.int32).flip(0).view(1, 1, -1)
            )

    def capture(q, k, v, *, plans, quantize_qk, vc_permutation, vc_demean):
        calls.append((vc_demean, vc_permutation))
        return q

    monkeypatch.setattr(prims.PrimTSValueGrouping, "refresh", refresh)
    monkeypatch.setattr(prims, "prims_dense_attention", capture)
    for step in [7, 0, 1, 4, 4, 5]:
        state["denoise_step"] = (step, 20)
        impl(x, x, x, None, skip_softmax=False, skip_threshold_active=False)
    assert refreshes == [0, 4]
    assert [smooth for smooth, _ in calls] == [False, True, True, True, True, False]
    assert torch.equal(calls[0][1], torch.arange(16, dtype=torch.int32).view(1, 1, -1))
    assert all(perm is calls[1][1] for _, perm in calls[1:])
