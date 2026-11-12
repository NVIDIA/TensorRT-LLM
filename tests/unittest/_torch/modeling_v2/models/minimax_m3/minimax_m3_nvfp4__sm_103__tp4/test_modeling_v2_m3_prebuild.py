# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode kernels the core builds before the executor's warmup (``prebuild_kernels``), and that a forward after it
never compiles one: the MoE engines of every ``MOE_ENGINE`` / ``MOE16_ENGINE`` token count, the decode GEMVs with
their push variants, and the index scorer with and without its early-issue schedule at every decode query length.

The engine builds are stubbed (a real one needs a loaded checkpoint's expert buffers; the engine kernels have their
own tests); the GEMVs and the scorer compile for real on the GPU.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers the trtllm ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops import (  # noqa: E501
    m3_index_score as index_score_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.m3_index_score import (  # noqa: E501
    m3_index_score,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_mx_gemv import (  # noqa: E501
    O_PROJ,
    SHARED_DOWN,
    SHARED_GATE_UP,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.mx_gemv import (  # noqa: E501
    op as mx_gemv,
)

dispatch = modeling

assert torch.cuda.is_available(), "the prebuild checks require a CUDA device"

INDEX_HEADS = 4
INDEX_DIM = 128
DRAFT_LEN = 3
_CORE_METHODS = (
    "prebuild_kernels",
    "_prebuild_engines",
    "_try_build",
    "engine",
    "moe16_engine",
    "_lookup",
    "_scorer_variants",
    "_prebuild_index_scorer",
    "decode_query_lens",
)


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


def _core(num_layers: int = 3, indexer_kv_dtype: str = "fp8", built=None, fail=False):
    """The parts of ``MiniMaxM3Tp4ModelingV2`` the prebuild reads: one rank (the build agreement takes no
    collective), MoE + sparse layers 1.. (layer 0 dense), no projections (the GEMV prebuild has its own test), the
    speculative config; the engine builds stubbed."""
    core = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=1, enable_attention_dp=False),
        moe_layers=frozenset(range(1, num_layers)),
        sparse_layers=frozenset(range(1, num_layers)),
        attn=[_FakeSparseAttn(indexer_kv_dtype) for _ in range(num_layers)],
        w={"embed": torch.empty(1, device="cuda")},
        mxfp8=SimpleNamespace(quant_method=SimpleNamespace(use_cutlass=True)),
        model_config=SimpleNamespace(
            spec_config=SimpleNamespace(max_draft_len=DRAFT_LEN, max_total_draft_tokens=DRAFT_LEN)
        ),
        moe_engines={},
        moe16_engines={},
        projections=lambda: [],
    )
    for name in _CORE_METHODS:
        setattr(core, name, getattr(modeling.MiniMaxM3Tp4ModelingV2, name).__get__(core))
    built = [] if built is None else built

    def build(i, num_tokens, shared=True):
        if fail:
            raise ValueError("the routed experts are not TRTLLM-Gen NVFP4 SwiGLU-OAI")
        built.append(("moe", i, num_tokens, shared))
        return ("engine", i, num_tokens)

    def build16(i, num_tokens):
        if fail:
            raise ValueError("the routed experts are not TRTLLM-Gen NVFP4 SwiGLU-OAI")
        built.append(("moe16", i, num_tokens))
        return ("engine16", i, num_tokens)

    core._build_engine, core._build_moe16_engine = build, build16
    return core


def _forbid_compiles(monkeypatch, core=None) -> None:
    """Every compile / build entry point a forward could reach raises."""
    import cutlass.cute as cute

    def boom(*args, **kwargs):
        raise AssertionError("a compile entry point ran after prebuild")

    monkeypatch.setattr(cute, "compile", boom)
    monkeypatch.setattr(mx_gemv, "ensure_compiled", boom)
    monkeypatch.setattr(mx_gemv, "precompile", boom)
    monkeypatch.setattr(index_score_kernel.IndexScoreRunner, "_compile", boom)
    if core is not None:
        core._build_engine = core._build_moe16_engine = boom


def test_decode_query_lens_follow_the_speculative_config() -> None:
    core = _core()
    assert core.decode_query_lens() == [1, 1 + DRAFT_LEN]
    core.model_config.spec_config = None
    assert core.decode_query_lens() == [1]
    core.model_config.spec_config = SimpleNamespace(
        max_draft_len=4, max_total_draft_tokens=4, draft_len_schedule={1: 4, 8: 2}
    )
    assert core.decode_query_lens() == [1, 3, 5]


def test_prebuild_builds_every_engine_count_and_scorer_variant() -> None:
    built = []
    core = _core(built=built)
    times = core.prebuild_kernels()

    assert mx_gemv.frozen(), "prebuild must close the GEMV compiles"

    # Layer 0 is dense; every MoE layer gets every count, in layer order.
    def per_layer(i):
        engines = [("moe", i, n, True) for n in range(1, dispatch.MOE_ENGINE_MAX_TOKENS + 1)]
        return engines + [
            ("moe16", i, n) for n in range(dispatch.MOE16_MIN_TOKENS, dispatch.MOE16_MAX_TOKENS + 1)
        ]

    assert built == per_layer(1) + per_layer(2)
    assert set(core.moe_engines[2]) == {(n, True, True) for n in dispatch.MOE_ENGINE.keys}
    assert set(core.moe16_engines[2]) == set(dispatch.MOE16_ENGINE.keys)

    want = {
        (torch.float8_e4m3fn, INDEX_HEADS, qlen, INDEX_DIM, modeling.INDEX_SCORE_SPLIT_K, early)
        for qlen in (1, 1 + DRAFT_LEN)
        for early in (True, False)
    }
    assert set(times["scorer_keys"]) == want
    assert want <= set(index_score_kernel.IndexScoreRunner.kernel_cache)
    for key in ("gemv_s", "moe_s", "scorer_s"):
        assert times[key] >= 0.0

    # A second build builds no engine again.
    built.clear()
    core.prebuild_kernels()
    assert built == []


def test_engine_lookups_after_prebuild_never_build(monkeypatch) -> None:
    core = _core()
    core._prebuild_engines(5)
    _forbid_compiles(monkeypatch, core)
    for require in (False, True):
        dispatch.set_require(require)
        for n in sorted(dispatch.MOE_ENGINE.keys):
            assert core.engine(5, n, shared=True) == ("engine", 5, n)
        for n in sorted(dispatch.MOE16_ENGINE.keys):
            assert core.moe16_engine(5, n) == ("engine16", 5, n)


def test_missing_engine_is_a_miss_not_a_build(monkeypatch) -> None:
    core = _core()
    core.moe_engines[7], core.moe16_engines[7] = {}, {}
    _forbid_compiles(monkeypatch, core)
    dispatch.set_require(False)
    assert core.engine(7, 3, shared=True) is None
    assert core.moe16_engine(7, 10) is None
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="not built before warmup"):
        core.engine(7, 3, shared=True)
    with pytest.raises(dispatch.FastPathUnavailable, match="not built before warmup"):
        core.moe16_engine(7, 10)


def test_failed_engine_build_is_logged_then_missed() -> None:
    core = _core(fail=True)
    core._prebuild_engines(9)
    assert all(v is None for v in core.moe_engines[9].values()) and len(core.moe_engines[9]) == 8
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="build failed"):
        core.engine(9, 1, shared=True)


def _projection(nk, push):
    n, k = nk
    w = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    return SimpleNamespace(weight=wq.view(n, k), weight_scale=wsf, push=push)


@pytest.mark.parametrize(
    "nk,push",
    ((O_PROJ, "attn"), (SHARED_DOWN, "moe"), (SHARED_GATE_UP, None)),
    ids=("o_proj", "shared_down", "shared_gate_up"),
)
def test_gemvs_after_prebuild_never_compile(monkeypatch, nk, push) -> None:
    use_pdl = True
    module = _projection(nk, push)
    variants = modeling._gemv_variants(*nk, push, 4)
    pushes = [v for v in variants if v[2] is not None]
    if push is None:
        assert not pushes
    else:
        covered = dispatch.ATTN_PUSH.keys if push == "attn" else dispatch.MOE_PUSH.keys
        assert {rows for rows, _, _ in pushes} == set(covered)
        assert {p[1] for _, _, p in pushes} == {b for b in (8, 16) if b <= max(covered)}
    mx_gemv.freeze(False)
    # As prebuild_kernels compiles one projection.
    mx_gemv.ensure_compiled(module.weight, module.weight_scale, use_pdl)
    for rows, x_bf16, ar in variants:
        mx_gemv.precompile(module.weight, module.weight_scale, rows, x_bf16, use_pdl=use_pdl, ar=ar)
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
    core = _core(indexer_kv_dtype=indexer_kv_dtype)
    keys = core._prebuild_index_scorer(core.decode_query_lens(), torch.device("cuda"))
    assert len(keys) == 4
    _forbid_compiles(monkeypatch, core)
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
