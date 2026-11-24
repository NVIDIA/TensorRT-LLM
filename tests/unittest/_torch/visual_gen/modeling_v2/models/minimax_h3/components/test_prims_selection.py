# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components import (
    trtllm_attention_prims as prims,
)
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def prims_available(monkeypatch):
    monkeypatch.setattr(prims, "is_context_prims_available", lambda name: True)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("+unknown", "Unknown FMHA library"),
        ("prims_ts,+fallback", "either an exact comma-separated list"),
        ("+", "Invalid empty FMHA library entry"),
    ],
)
def test_h3_fmha_registry_errors(monkeypatch, value, message):
    monkeypatch.setenv("TLLM_FMHA_LIBS", value)
    with pytest.raises(ValueError, match=message):
        prims.H3PrimTS(None, {})


def test_sparse_only_admission_for_fp8_and_vc(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts_block_sparse")
    probes = []

    def available(name):
        probes.append(name)
        return True

    monkeypatch.setattr(prims, "is_context_prims_available", available)
    prims.H3PrimTS(None, {})
    with pytest.raises(RuntimeError, match="VC attention requires an available dense PrimTS"):
        prims.H3PrimTS(QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8", vc_attention=True), {})
    assert not probes
    prims.H3PrimTS(QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8"), {})
    assert probes == ["prims_ts_block_sparse"]


@pytest.mark.parametrize(
    ("value", "explicit_dense", "sparse_first"),
    [
        (None, False, True),
        ("fallback,prims_ts_block_sparse,fallback", False, False),
        ("fallback,prims_ts,prims_ts_block_sparse", True, False),
        ("+beam_shared_prefix", False, True),
        ("+prims_ts,-prims_ts", False, True),
        ("-prims_ts,+prims_ts,-fallback", True, True),
    ],
)
def test_h3_dense_prims_policy(monkeypatch, prims_available, value, explicit_dense, sparse_first):
    if value is None:
        monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    else:
        monkeypatch.setenv("TLLM_FMHA_LIBS", value)
    recipe = QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")
    assert prims.H3PrimTS(None, {}).smooth_v_plan is sparse_first
    q = torch.empty(1, 16, 1, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(prims, "prims_dense_attention", lambda q, k, v, **kwargs: v)
    if explicit_dense:
        impl = prims.H3PrimTS(recipe, {})
        assert impl(q, q, q, None, skip_softmax=False, skip_threshold_active=False) is q
    else:
        with pytest.raises(RuntimeError, match="per-tensor FP8"):
            impl = prims.H3PrimTS(recipe, {})
            impl(q, q, q, None, skip_softmax=False, skip_threshold_active=False)


def test_routed_sage_passes_quantization_recipe(monkeypatch, prims_available):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts_block_sparse")
    recipe = QuantAttentionConfig(
        qk_dtype="fp8", v_dtype="fp8", q_block_size=1, k_block_size=16, v_block_size=1
    )
    impl = prims.H3PrimTS(recipe, {})
    q = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)
    routes = prims.PrimsRoutes(
        q_block_size=128,
        kv_block_size=128,
        max_blocks_per_row=1,
        block_indptr=torch.tensor([[[0, 1]]], dtype=torch.int32),
        block_indices=torch.tensor([0], dtype=torch.int32),
    )

    def run(q, k, v, *, quantization, **kwargs):
        assert quantization == prims.PrimsSageQuantization(torch.float8_e4m3fn, 1, 16, False, False)
        return v

    monkeypatch.setattr(prims, "prims_routed_attention", run)
    assert impl(q, q, q, routes, skip_softmax=False, skip_threshold_active=False) is q


def test_optional_sage_plan_fallback(monkeypatch, prims_available):
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


def test_vc_denoising_schedule(monkeypatch, prims_available):
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
