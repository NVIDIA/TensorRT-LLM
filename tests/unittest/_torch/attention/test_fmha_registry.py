# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from tensorrt_llm._torch.attention_backend.fmha import flashinfer_sparse_mla
from tensorrt_llm._torch.attention_backend.fmha.registry import (
    DEFAULT_FMHA_LIBS,
    get_enabled_fmha_lib_names,
)


def test_flashinfer_sparse_mla_is_first_default_fmha():
    assert DEFAULT_FMHA_LIBS[0] == flashinfer_sparse_mla.FMHA_NAME


def test_flashinfer_sparse_mla_enablement_uses_registry_arch_and_op(monkeypatch):
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    monkeypatch.setattr(flashinfer_sparse_mla, "get_sm_version", lambda: 120)
    monkeypatch.setattr(flashinfer_sparse_mla, "_sparse_mla_op", lambda: object())

    assert flashinfer_sparse_mla.is_flashinfer_sparse_mla_enabled("dsa")
    assert flashinfer_sparse_mla.is_flashinfer_sparse_mla_enabled("deepseek_v4")
    assert not flashinfer_sparse_mla.is_flashinfer_sparse_mla_enabled("rocket")

    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")
    assert get_enabled_fmha_lib_names() == ("fallback",)
    assert not flashinfer_sparse_mla.is_flashinfer_sparse_mla_enabled("dsa")

    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback,flashinfer_sparse_mla")
    assert not flashinfer_sparse_mla.is_flashinfer_sparse_mla_enabled("dsa")


def test_flashinfer_sparse_mla_is_unavailable_off_sm120(monkeypatch):
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    monkeypatch.setattr(flashinfer_sparse_mla, "get_sm_version", lambda: 100)
    monkeypatch.setattr(flashinfer_sparse_mla, "_sparse_mla_op", lambda: object())

    attn = SimpleNamespace(
        is_mla_enable=True,
        sparse_params=SimpleNamespace(algorithm="dsa"),
    )
    assert not flashinfer_sparse_mla.FlashInferSparseMlaFmha.is_available(attn)


def test_flashinfer_sparse_mla_never_runtime_falls_back():
    fmha = object.__new__(flashinfer_sparse_mla.FlashInferSparseMlaFmha)
    assert fmha.is_supported(None, None, None, None, None)


def test_sparse_mla_rope_capability_is_callable_before_instantiation(monkeypatch):
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
        DeepseekV4TrtllmAttention,
    )
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSATrtllmAttention

    monkeypatch.setattr(
        flashinfer_sparse_mla,
        "is_flashinfer_sparse_mla_enabled",
        lambda algorithm: algorithm in ("deepseek_v4", "dsa"),
    )
    assert not DeepseekV4TrtllmAttention.support_fused_rope()
    assert not DSATrtllmAttention.support_fused_rope()

    monkeypatch.setattr(
        flashinfer_sparse_mla,
        "is_flashinfer_sparse_mla_enabled",
        lambda algorithm: False,
    )
    assert DeepseekV4TrtllmAttention.support_fused_rope()
    assert DSATrtllmAttention.support_fused_rope()


def test_flashinfer_sparse_mla_forward_dispatch(monkeypatch):
    from tensorrt_llm._torch.attention_backend.sparse import dsa_flashinfer
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import flashinfer

    calls = []
    monkeypatch.setattr(
        flashinfer,
        "run_flashinfer_sparse_mla",
        lambda *args: calls.append(("deepseek_v4", args)),
    )
    monkeypatch.setattr(
        dsa_flashinfer,
        "run_flashinfer_sparse_mla",
        lambda *args: calls.append(("dsa", args)),
    )

    for algorithm in ("deepseek_v4", "dsa"):
        attn = SimpleNamespace(sparse_params=SimpleNamespace(algorithm=algorithm))
        fmha = object.__new__(flashinfer_sparse_mla.FlashInferSparseMlaFmha)
        fmha._attn_ref = lambda: attn
        q, metadata, forward_args = object(), object(), object()

        fmha.forward(q, None, None, metadata, forward_args)

        assert calls[-1] == (algorithm, (attn, q, metadata, forward_args))
