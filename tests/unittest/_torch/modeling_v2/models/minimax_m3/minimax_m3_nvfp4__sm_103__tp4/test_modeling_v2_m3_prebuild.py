# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode kernels ``build_targets`` builds before the executor's warmup (``prebuild_kernels``), and that a forward
after it never compiles one: the MoE engines of every ``MOE_ENGINE`` / ``MOE16_ENGINE`` token count, the decode
GEMVs with their push variants, and the index scorer with and without its early-issue schedule at every decode
query length.

The engine builds are stubbed (a real one needs a loaded checkpoint's expert buffers; the engine kernels have their
own tests); the GEMVs and the scorer compile for real on the GPU.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers the trtllm ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4 import (  # noqa: E501
    dispatch,
    layers,
    linears,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_index_score import (  # noqa: E501
    m3_index_score,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_mx_gemv import (  # noqa: E501
    O_PROJ,
    SHARED_DOWN,
    SHARED_GATE_UP,
)
from tensorrt_llm._torch.cute_dsl_kernels.minimax_m3.mx_gemv import op as mx_gemv

assert torch.cuda.is_available(), "the prebuild checks require a CUDA device"

INDEX_HEADS = 4
INDEX_DIM = 128
DRAFT_LEN = 3


@pytest.fixture(autouse=True)
def _restore_modes():
    require, frozen = dispatch.require_mode(), mx_gemv.frozen()
    yield
    dispatch.set_require(require)
    mx_gemv.freeze(frozen)
    dispatch.reset_logged()


class _FakeSparseAttn:
    def __init__(self, indexer_kv_dtype: str):
        self.indexer_kv_dtype = indexer_kv_dtype
        self.m3_config = SimpleNamespace(num_index_heads=INDEX_HEADS, sparse_index_dim=INDEX_DIM)


def _moe(layer_idx: int, shared: bool = True) -> layers.M3MoE:
    """An M3MoE built by its own constructor over a stub stock block: only what prebuild_engines and the lookups
    read (the builds are stubbed)."""

    def stock_init(self, model_config, aux_stream_dict, layer_idx=None):
        nn.Module.__init__(self)
        # One rank: the build agreement takes no collective.
        self.mapping = SimpleNamespace(tp_size=1, enable_attention_dp=False)
        self.shared_experts = None

    with mock.patch.object(layers.MiniMaxM3MoE, "__init__", stock_init):
        moe = layers.M3MoE(None, {}, layer_idx)
    moe.shared_experts = object() if shared else None
    return moe


def _model(monkeypatch, num_layers: int = 3, indexer_kv_dtype: str = "fp8"):
    """The parts of MiniMaxM3Tp4ModelingV2 prebuild_kernels reads: layers (MoE + sparse attention), the embedding
    device and the speculative config."""
    monkeypatch.setattr(layers, "MiniMaxM3MsaSparseAttention", _FakeSparseAttn)
    model_layers = []
    for idx in range(num_layers):
        model_layers.append(
            SimpleNamespace(
                block_sparse_moe=_moe(idx, shared=idx != 1) if idx else None,
                self_attn=SimpleNamespace(
                    is_sparse_attention_layer=idx != 0, attn=_FakeSparseAttn(indexer_kv_dtype)
                ),
            )
        )
    stub = SimpleNamespace(
        layers=model_layers,
        embed_tokens=SimpleNamespace(weight=torch.empty(1, device="cuda")),
        model_config=SimpleNamespace(
            spec_config=SimpleNamespace(max_draft_len=DRAFT_LEN, max_total_draft_tokens=DRAFT_LEN)
        ),
    )
    stub.decode_query_lens = lambda: modeling.MiniMaxM3Tp4ModelingV2.decode_query_lens(stub)
    return stub


def _stub_engine_builds(monkeypatch, built: list) -> None:
    def build(self, num_tokens, shared=True):
        built.append(("moe", self.layer_idx, num_tokens, shared))
        return ("engine", self.layer_idx, num_tokens)

    def build16(self, num_tokens):
        built.append(("moe16", self.layer_idx, num_tokens))
        return ("engine16", self.layer_idx, num_tokens)

    monkeypatch.setattr(layers.M3MoE, "_build_engine", build)
    monkeypatch.setattr(layers.M3MoE, "_build_moe16_engine", build16)


def _forbid_compiles(monkeypatch) -> None:
    """Every compile / build entry point a forward could reach raises."""
    import cutlass.cute as cute

    def boom(*args, **kwargs):
        raise AssertionError("a compile entry point ran after prebuild")

    monkeypatch.setattr(cute, "compile", boom)
    monkeypatch.setattr(mx_gemv, "ensure_compiled", boom)
    monkeypatch.setattr(mx_gemv, "precompile", boom)
    monkeypatch.setattr(layers.M3MoE, "_build_engine", boom)
    monkeypatch.setattr(layers.M3MoE, "_build_moe16_engine", boom)
    monkeypatch.setattr(layers._scorer_runner(), "_compile", boom)


def test_decode_query_lens_follow_the_speculative_config(monkeypatch) -> None:
    model = _model(monkeypatch)
    assert model.decode_query_lens() == [1, 1 + DRAFT_LEN]
    model.model_config.spec_config = None
    assert model.decode_query_lens() == [1]
    model.model_config.spec_config = SimpleNamespace(
        max_draft_len=4, max_total_draft_tokens=4, draft_len_schedule={1: 4, 8: 2}
    )
    assert model.decode_query_lens() == [1, 3, 5]


def test_prebuild_builds_every_engine_count_and_scorer_variant(monkeypatch) -> None:
    built = []
    _stub_engine_builds(monkeypatch, built)
    monkeypatch.setattr(modeling, "adopt_m3_linears", lambda root: built.append(("linears",)))
    model = _model(monkeypatch)
    times = modeling.MiniMaxM3Tp4ModelingV2.prebuild_kernels(model)

    assert built[0] == ("linears",)
    assert mx_gemv.frozen(), "prebuild must close the GEMV compiles"
    # Layer 0 is dense, layer 1 has no shared expert (no engine takes it); layer 2 gets every count, in order.
    assert built[1:] == [
        ("moe", 2, n, True) for n in range(1, dispatch.MOE_ENGINE_MAX_TOKENS + 1)
    ] + [("moe16", 2, n) for n in range(dispatch.MOE16_MIN_TOKENS, dispatch.MOE16_MAX_TOKENS + 1)]
    moe = model.layers[2].block_sparse_moe
    assert set(moe._engines) == {(n, True, True) for n in dispatch.MOE_ENGINE.keys}
    assert set(moe._moe16_engines) == set(dispatch.MOE16_ENGINE.keys)

    runner = layers._scorer_runner()
    want = {
        (torch.float8_e4m3fn, INDEX_HEADS, qlen, INDEX_DIM, layers.INDEX_SCORE_SPLIT_K, early)
        for qlen in (1, 1 + DRAFT_LEN)
        for early in (True, False)
    }
    assert set(times["scorer_keys"]) == want
    assert want <= set(runner.kernel_cache)
    for key in ("gemv_s", "moe_s", "scorer_s"):
        assert times[key] >= 0.0

    # A second build_targets builds nothing again.
    built.clear()
    modeling.MiniMaxM3Tp4ModelingV2.prebuild_kernels(model)
    assert built == [("linears",)]


def test_engine_lookups_after_prebuild_never_build(monkeypatch) -> None:
    built = []
    _stub_engine_builds(monkeypatch, built)
    moe = _moe(5)
    moe.prebuild_engines()
    _forbid_compiles(monkeypatch)
    for require in (False, True):
        dispatch.set_require(require)
        for n in sorted(dispatch.MOE_ENGINE.keys):
            assert moe.engine(n, shared=True) == ("engine", 5, n)
        for n in sorted(dispatch.MOE16_ENGINE.keys):
            assert moe.moe16_engine(n) == ("engine16", 5, n)


def test_missing_engine_is_a_miss_not_a_build(monkeypatch) -> None:
    _forbid_compiles(monkeypatch)
    moe = _moe(7)
    dispatch.set_require(False)
    assert moe.engine(3, shared=True) is None
    assert moe.moe16_engine(10) is None
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="not built before warmup"):
        moe.engine(3, shared=True)
    with pytest.raises(dispatch.FastPathUnavailable, match="not built before warmup"):
        moe.moe16_engine(10)


def test_failed_engine_build_is_logged_then_missed(monkeypatch) -> None:
    def fail(self, num_tokens, shared=True):
        raise ValueError("experts backend is not TRTLLM-Gen NVFP4 SwiGLU-OAI")

    monkeypatch.setattr(layers.M3MoE, "_build_engine", fail)
    monkeypatch.setattr(layers.M3MoE, "_build_moe16_engine", lambda self, n: fail(self, n))
    moe = _moe(9)
    moe.prebuild_engines()
    assert all(v is None for v in moe._engines.values()) and len(moe._engines) == 8
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="build failed"):
        moe.engine(1, shared=True)


def _projection(nk, push):
    n, k = nk
    w = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    return SimpleNamespace(weight=wq.view(n, k), weight_scale=wsf, m3_push=push, tp_size=4)


@pytest.mark.parametrize(
    "nk,push",
    ((O_PROJ, linears.PUSH_ATTN), (SHARED_DOWN, linears.PUSH_MOE), (SHARED_GATE_UP, None)),
    ids=("o_proj", "shared_down", "shared_gate_up"),
)
def test_gemvs_after_prebuild_never_compile(monkeypatch, nk, push) -> None:
    use_pdl = True
    module = _projection(nk, push)
    variants = linears.gemv_variants(module)
    pushes = [v for v in variants if v[2] is not None]
    if push is None:
        assert not pushes
    else:
        covered = dispatch.ATTN_PUSH.keys if push == linears.PUSH_ATTN else dispatch.MOE_PUSH.keys
        assert {rows for rows, _, _ in pushes} == set(covered)
        assert {p[1] for _, _, p in pushes} == {b for b in (8, 16) if b <= max(covered)}
    mx_gemv.freeze(False)
    assert linears.prebuild_m3_gemvs(module, use_pdl) > 0
    mx_gemv.freeze(True)
    _forbid_compiles(monkeypatch)

    n, k = nk
    for rows, x_bf16, ar in variants:
        if ar is not None:
            # A push needs a multicast exchange to launch: its kernel is compiled (the forward finds it).
            resolved = mx_gemv._resolve(rows, n, k, x_bf16, True, None, use_pdl, (0, 0, 0, *ar))
            if resolved is not None:
                assert mx_gemv.is_compiled(resolved[0]), (rows, x_bf16, ar)
            continue
        if x_bf16:
            x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
            out = mx_gemv.m3_mx_gemv_bf16(x, module.weight, module.weight_scale, use_pdl=use_pdl)
        else:
            xq, xsf = torch.ops.trtllm.mxfp8_quantize(
                torch.randn(rows, k, device="cuda").to(torch.bfloat16), True
            )
            out = mx_gemv.m3_mx_gemv(
                xq.view(rows, k), xsf, module.weight, module.weight_scale, use_pdl=use_pdl
            )
        served = mx_gemv.serves(n, k, rows, x_bf16)
        assert (out is not None) == served, (rows, x_bf16)
        assert mx_gemv.last_uncompiled is None
    torch.cuda.synchronize()


def test_frozen_gemv_declines_an_uncompiled_kernel(monkeypatch) -> None:
    module = _projection(SHARED_DOWN, None)
    mx_gemv.freeze(True)
    _forbid_compiles(monkeypatch)
    # A launch option nothing prebuilt (no early trigger, without a push): declined, never compiled.
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(
        torch.randn(3, 768, device="cuda").to(torch.bfloat16), True
    )
    key = mx_gemv._resolve(3, 6144, 768, False, False, None, True, None)[0]
    if mx_gemv.is_compiled(key):
        pytest.skip("this process already compiled the probe kernel")
    out = mx_gemv.m3_mx_gemv(
        xq.view(3, 768), xsf, module.weight, module.weight_scale, trigger_early=False, use_pdl=True
    )
    assert out is None and mx_gemv.last_uncompiled == key


@pytest.mark.parametrize("indexer_kv_dtype", ("fp8", "bf16"))
def test_scorer_after_prebuild_never_compiles(monkeypatch, indexer_kv_dtype) -> None:
    model = _model(monkeypatch, indexer_kv_dtype=indexer_kv_dtype)
    keys = layers.prebuild_index_scorer(
        model.layers, model.decode_query_lens(), torch.device("cuda")
    )
    assert len(keys) == 4
    _forbid_compiles(monkeypatch)
    dtype = torch.float8_e4m3fn if indexer_kv_dtype == "fp8" else torch.bfloat16
    for _, heads, qlen, dim, split_k, early in keys:
        rows = 1 if early else 2
        blocks = 3
        idx_q = torch.randn(rows * qlen, heads, dim, device="cuda").to(dtype)
        idx_k = torch.randn(rows * blocks, 1, 128, dim, device="cuda").to(dtype)
        block_table = torch.arange(rows * blocks, dtype=torch.int32, device="cuda").view(
            rows, blocks
        )
        seq_lens = torch.full((rows,), blocks * 128 - 5, dtype=torch.int32, device="cuda")
        score = torch.empty(heads, blocks, rows * qlen, dtype=torch.float32, device="cuda")
        m3_index_score(
            idx_q,
            idx_k.squeeze(1),
            block_table,
            seq_lens,
            score.transpose(1, 2),
            qlen,
            split_k,
            early,
        )
    torch.cuda.synchronize()


def test_moe_keeps_its_layer_index():
    # prebuild_engines names the layer in its build and agreement messages.
    assert _moe(5).layer_idx == 5
