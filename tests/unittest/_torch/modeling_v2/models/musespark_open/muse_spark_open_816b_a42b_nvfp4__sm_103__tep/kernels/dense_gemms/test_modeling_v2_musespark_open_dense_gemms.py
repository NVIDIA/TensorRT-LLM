# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How ``dense_gemms`` / ``dense_gemm_prefetch`` (the dense projections' pinned kernels) are wired into
the tep8 target.

Host-only. The kernels themselves are checked on a GPU against float64 and the
built-in kernels (``gemm/test_modeling_v2_dense_gemm.py``); here: the kernel
table, which modules adoption re-classes and what it
refuses (leaving the model untouched), that the engine's later assignment of
its own low-M dispatcher lands as the fallback instead of replacing this one,
what each projection prefetches next, that a call the entry does not serve
reaches the engine's dispatcher with the same arguments, when a prefetch fork is
joined and when it is dropped, and that both targets join an outstanding fork at
the end of their forward.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.modules import low_m_gemm
from tensorrt_llm._torch.modules.linear import Linear

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_DENSE = dense_gemms.MuseSparkOpenTepDenseLinear
_TP = 8
#: Small stand-in widths with the checkpoint's proportions: (K, N) per projection.
_QKV, _O, _POOL_FULL, _UNPOOL_FULL = (64, 36), (16, 64), (64, 32 * _TP // 4), (32, 64)


# --- the kernel table ---------------------------------------------------------------------------


def test_the_default_plan_by_row_count():
    plan = dense_gemms.default_kernels()
    assert plan == {
        "qkv": ("direct:256,1",) * 2 + ("splitk:64,8,4,12",) * 6,
        "o_proj": ("direct:64,4",) * 2 + ("cublas",) * 6,
        "pool": ("direct:256,1",) * 2 + ("direct:128,2",) * 4 + ("splitk:64,8,4,12",) * 2,
        "unpool": ("direct:256,1",) * 2 + ("direct:64,2",) * 3 + ("splitk:64,8,4,12",) * 3,
    }
    assert dense_gemms.describe(plan) == (
        "qkv=direct:256,1@1-2|splitk:64,8,4,12@3-8, o_proj=direct:64,4@1-2|cublas@3-8, "
        "pool=direct:256,1@1-2|direct:128,2@3-6|splitk:64,8,4,12@7-8, "
        "unpool=direct:256,1@1-2|direct:64,2@3-5|splitk:64,8,4,12@6-8"
    )


def test_the_sm107_plan_by_row_count():
    plan = dense_gemms.default_kernels((10, 7))
    assert plan == {
        "qkv": ("direct:256,2", "direct:128,4", "direct:64,2") + ("splitk:64,8,4,12",) * 5,
        "o_proj": ("direct:64,4",) * 2 + ("splitk:64,8,1,12",) * 6,
        "pool": dense_gemms.default_kernels()["pool"],
        "unpool": ("direct:256,2",) + ("direct:128,2",) * 6 + ("splitk:64,8,4,12",),
    }
    with pytest.raises(ValueError, match=r"no kernel table for SM \(10, 0\)"):
        dense_gemms.default_kernels((10, 0))


def test_install_takes_a_kernel_table(low_m_on, monkeypatch):
    monkeypatch.setattr(dense_gemms, "device_capability", lambda: (10, 3))
    default = dense_gemms.default_kernels()
    table = dict(default, qkv=("cublas",) * 8)
    plan = dense_gemms.install(_builtin_text_model(), prefetch=False, kernels=table)
    assert plan.kernels == table and plan.capability == (10, 3)
    assert dense_gemms.describe(plan.kernels).startswith("qkv=cublas@1-8, o_proj=direct:64,4@1-2|")
    assert dense_gemms.install(_builtin_text_model(), prefetch=False).kernels == default


@pytest.mark.parametrize("capability", [(10, 3), (10, 7)])
def test_install_takes_the_devices_table(low_m_on, monkeypatch, capability):
    monkeypatch.setattr(dense_gemms, "device_capability", lambda: capability)
    plan = dense_gemms.install(_builtin_text_model(), prefetch=False)
    assert plan.kernels == dense_gemms.default_kernels(capability)
    assert plan.capability == capability


def test_install_refuses_a_device_without_a_table(low_m_on, monkeypatch):
    monkeypatch.setattr(dense_gemms, "device_capability", lambda: (10, 0))
    model = _builtin_text_model()
    with pytest.raises(ValueError, match="no kernel table"):
        dense_gemms.install(model, prefetch=False)
    assert not any(isinstance(m, _DENSE) for m in model.modules())


# --- stand-ins ---------------------------------------------------------------------------------


def _linear(n: int, k: int, bias: bool = False, custom_cublas: bool = False) -> Linear:
    """A built-in ``Linear`` as far as class, weight and the dispatch flags go."""
    linear = Linear.__new__(Linear)
    nn.Module.__init__(linear)
    linear.weight = nn.Parameter(torch.randn(n, k).to(torch.bfloat16), requires_grad=False)
    if bias:
        linear.bias = nn.Parameter(torch.zeros(n, dtype=torch.bfloat16), requires_grad=False)
    else:
        linear.register_parameter("bias", None)
    linear.use_custom_cublas_mm = custom_cublas
    return linear


class _Mapping:
    tp_size = _TP

    def __init__(self, rank: int) -> None:
        self.tp_rank = rank


def _builtin_text_model(num_layers: int = 3, rank: int = 5) -> MuseSparkOpenModel:
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    layers = []
    for _ in range(num_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = _linear(_QKV[1], _QKV[0])
        layer.self_attn.o_proj = _linear(_O[1], _O[0])
        layer.feed_forward = nn.Module()
        layer.feed_forward.pooler = _linear(_POOL_FULL[1], _POOL_FULL[0])
        layer.feed_forward.unpooler = _linear(_UNPOOL_FULL[1], _UNPOOL_FULL[0])
        layer.feed_forward.mapping = _Mapping(rank)
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


def _shard(model):
    """``model`` as the built-in constructor leaves it with the sharded projections on."""
    for layer in model.layers:
        layer.feed_forward._pool_gather_workspace = object()
        layer.feed_forward._unpool_gather_workspace = object()
    return model


def _projections(model):
    for layer in model.layers:
        yield layer.self_attn.qkv_proj
        yield layer.self_attn.o_proj
        yield layer.feed_forward.pooler
        yield layer.feed_forward.unpooler


@pytest.fixture
def low_m_on(monkeypatch):
    """The served ``TRTLLM_LOW_M_GEMM_BACKEND=auto``; kernels accepted for any stand-in shape."""
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(dense_gemms, "kernel_error", lambda kernel, rows, k, n: None)


# --- adoption ----------------------------------------------------------------------------------


def test_no_feature_leaves_the_scaffold(low_m_on):
    model = _builtin_text_model()
    _CORE.adopt(model)
    assert model._dense_gemm_plan is None
    assert all(type(linear) is Linear for linear in _projections(model))


@pytest.mark.parametrize("path, prefetch", [("dense_gemms", False), ("dense_gemm_prefetch", True)])
def test_the_core_installs_every_projection(low_m_on, path, prefetch):
    model = _shard(_builtin_text_model())
    modules_before = [name for name, _ in model.named_modules()]
    params_before = [name for name, _ in model.named_parameters()]
    _CORE.adopt(model, dispatch.plan_of(path))
    plan = model._dense_gemm_plan
    assert isinstance(plan, dense_gemms.DenseGemmPlan) and plan.prefetch is prefetch
    assert all(type(linear) is _DENSE for linear in _projections(model))
    assert [name for name, _ in model.named_modules()] == modules_before
    assert [name for name, _ in model.named_parameters()] == params_before


def test_the_engines_dispatcher_becomes_the_fallback(low_m_on):
    model = _builtin_text_model()
    dense_gemms.install(model, prefetch=False)
    engine = low_m_gemm.LowMGemmDispatcher()
    engine.attach(model, force=True)  # what prepare_low_m_gemm does after loading
    for linear in _projections(model):
        dispatcher = linear._low_m_gemm_dispatcher
        assert isinstance(dispatcher, dense_gemms.DenseGemmDispatcher)
        assert dispatcher.builtin is engine
        assert "_low_m_gemm_dispatcher" not in linear.__dict__


def test_an_engine_dispatcher_assigned_before_adoption_is_kept_as_the_fallback(low_m_on):
    model = _builtin_text_model(1)
    engine = low_m_gemm.LowMGemmDispatcher()
    engine.attach(model, force=True)
    dense_gemms.install(model, prefetch=False)
    assert all(linear._low_m_gemm_dispatcher.builtin is engine for linear in _projections(model))


@pytest.mark.parametrize(
    "break_it, error, match",
    [
        (
            lambda m: setattr(m.layers[1].self_attn, "o_proj", _linear(_O[1], _O[0], bias=True)),
            ValueError,
            "bias",
        ),
        (
            lambda m: setattr(
                m.layers[2].feed_forward, "unpooler", _linear(64, 32, custom_cublas=True)
            ),
            ValueError,
            "custom cuBLAS",
        ),
        (
            lambda m: setattr(m.layers[2].self_attn, "qkv_proj", nn.Linear(64, 36)),
            TypeError,
            "built-in Linear",
        ),
    ],
)
def test_a_refusal_leaves_the_model_untouched(low_m_on, break_it, error, match):
    model = _builtin_text_model()
    break_it(model)
    with pytest.raises(error, match=match):
        dense_gemms.install(_shard(model), prefetch=True)
    assert not any(type(m) is _DENSE for m in model.modules())


def test_a_kernel_that_cannot_serve_its_shape_is_refused(monkeypatch, low_m_on):
    monkeypatch.setattr(
        dense_gemms,
        "kernel_error",
        lambda kernel, rows, k, n: "too small" if n == 8 and rows == 5 else None,
    )
    model = _builtin_text_model()
    with pytest.raises(ValueError, match="cannot serve 5 x"):
        dense_gemms.install(_shard(model), prefetch=False)  # pool slice: N = 64 / 8
    assert not any(type(m) is _DENSE for m in model.modules())


def test_the_low_m_dispatch_must_be_on(monkeypatch, low_m_on):
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", False)
    with pytest.raises(ValueError, match="TRTLLM_LOW_M_GEMM_BACKEND"):
        dense_gemms.install(_builtin_text_model(), prefetch=False)


@pytest.mark.parametrize("sharded", (True, False), ids=("sharded", "replicated"))
def test_install_runs_on_meta_weights_under_meta_init(low_m_on, sharded):
    """The engine builds the model under ``MetaInitMode``, whose meta weights refuse every
    operation but a few initializers (a slice among the refused): adoption reads shapes only."""
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    model = _builtin_text_model()
    for linear in _projections(model):
        linear.weight = nn.Parameter(
            torch.empty(linear.weight.shape, dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
    with MetaInitMode():
        plan = dense_gemms.install(_shard(model) if sharded else model, prefetch=sharded)
    assert plan.prefetch is sharded
    pool = model.layers[0].feed_forward.pooler._low_m_gemm_dispatcher.site
    assert pool.shape == (_POOL_FULL[0], _POOL_FULL[1] // _TP if sharded else _POOL_FULL[1])


def test_gemm_prefetch_needs_the_sharded_projections(low_m_on):
    model = _builtin_text_model()
    with pytest.raises(ValueError, match="MUSESPARK_OPEN_SHARDED_PROJECTIONS"):
        dense_gemms.install(model, prefetch=True)
    assert not any(type(m) is _DENSE for m in model.modules())
    dense_gemms.install(model, prefetch=False)  # without the prefetch, no sharding is needed


# --- sites ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("sharded", (True, False), ids=("sharded", "replicated"))
def test_each_projection_knows_its_shape_and_what_comes_next(low_m_on, sharded):
    rank = 5
    model = _builtin_text_model(3, rank=rank)
    dense_gemms.install(_shard(model) if sharded else model, prefetch=sharded)
    layers = model.layers
    for i, layer in enumerate(layers):
        attention, moe = layer.self_attn, layer.feed_forward
        qkv, o, pool, unpool = (
            attention.qkv_proj._low_m_gemm_dispatcher.site,
            attention.o_proj._low_m_gemm_dispatcher.site,
            moe.pooler._low_m_gemm_dispatcher.site,
            moe.unpooler._low_m_gemm_dispatcher.site,
        )
        assert (qkv.role, o.role, pool.role, unpool.role) == dense_gemms.SITES
        assert qkv.shape == _QKV and o.shape == _O
        pool_n = _POOL_FULL[1] // _TP if sharded else _POOL_FULL[1]
        unpool_n = _UNPOOL_FULL[1] // _TP if sharded else _UNPOOL_FULL[1]
        assert pool.shape == (_POOL_FULL[0], pool_n) and unpool.shape == (_UNPOOL_FULL[0], unpool_n)
        assert qkv.next_weight() is attention.o_proj.weight
        nxt = o.next_weight()
        expected = (
            moe.pooler.weight[rank * pool_n : (rank + 1) * pool_n] if sharded else moe.pooler.weight
        )
        assert nxt.data_ptr() == expected.data_ptr() and nxt.shape == expected.shape
        assert pool.next_weight is None
        if i + 1 < len(layers):
            assert unpool.next_weight() is layers[i + 1].self_attn.qkv_proj.weight
        else:
            assert unpool.next_weight is None


class _Engine:
    def __init__(self):
        self.calls = []

    def apply(self, module, input_tensor, weight, bias, force_active=False):
        self.calls.append((module, input_tensor, weight, bias, force_active))
        return "engine"


@pytest.mark.parametrize("force_active", (False, True))
def test_a_call_the_entry_does_not_serve_reaches_the_engine_unchanged(low_m_on, force_active):
    model = _builtin_text_model(1)
    dense_gemms.install(_shard(model), prefetch=True)
    linear = model.layers[0].self_attn.qkv_proj
    engine = _Engine()
    linear._low_m_gemm_dispatcher = engine
    x = torch.randn(8, _QKV[0]).to(torch.bfloat16)  # a host tensor: not served
    assert (
        low_m_gemm.apply_low_m_gemm(linear, x, linear.weight, None, force_active=force_active)
        == "engine"
    )
    assert engine.calls == [(linear, x, linear.weight, None, force_active)]
    assert not linear._low_m_gemm_dispatcher.site.plan.pending


def test_join_is_a_no_op_without_a_fork():
    plan = dense_gemms.DenseGemmPlan(dense_gemms.default_kernels(), prefetch=True, enable_pdl=True)
    plan.join()
    assert not plan.pending


# --- the prefetch fork and its join ------------------------------------------------------------


class _Stream:
    """Records the streams it is made to wait for."""

    def __init__(self) -> None:
        self.waited = []

    def wait_stream(self, other) -> None:
        self.waited.append(other)


@pytest.mark.parametrize(
    "fork, now, waits",
    [
        (None, None, True),  # eager fork, eager consumer
        (7, 7, True),  # both inside capture 7
        (None, 7, False),  # an eager step left it; capture 7 must not wait on eager work
        (7, None, False),  # capture 7 was abandoned or a graph segment already joined it
        (7, 8, False),  # the next segment / capture
    ],
    ids=("eager", "same-capture", "eager-into-capture", "capture-into-eager", "other-capture"),
)
def test_a_join_waits_only_from_the_capture_state_of_its_fork(monkeypatch, fork, now, waits):
    plan = dense_gemms.DenseGemmPlan(dense_gemms.default_kernels(), prefetch=True, enable_pdl=True)
    side, current = _Stream(), _Stream()
    plan._stream, plan._fork_capture, plan.pending = side, fork, True
    monkeypatch.setattr(dense_gemms, "current_capture_id", lambda: now)
    monkeypatch.setattr(dense_gemms.torch.cuda, "current_stream", lambda *args: current)
    plan.join()
    assert not plan.pending
    assert current.waited == ([side] if waits else [])
    assert plan.dropped == (0 if waits else 1)
    plan.join()  # nothing outstanding any more
    assert len(current.waited) == int(waits) and plan.dropped == int(not waits)


class _OrderLayer(nn.Module):
    def __init__(self, log: list) -> None:
        super().__init__()
        self.log = log
        self.pre_qkv_proj_norm = nn.Identity()

    def forward(self, position_ids, hidden_states, attn_metadata, next_layer_norm=None, **kwargs):
        self.log.append("layer")
        return hidden_states if next_layer_norm is None else (hidden_states, hidden_states)


class _OrderNorm(nn.Module):
    def __init__(self, log: list) -> None:
        super().__init__()
        self.log = log

    def forward(self, x):
        self.log.append("norm")
        return x


def test_both_targets_join_an_outstanding_fork_at_the_end_of_their_forward(monkeypatch):
    """A fork whose consumer did not run (a sibling feature's GEMM, a short forward) is joined
    by the forward that made it, after everything else the forward launches."""
    log: list = []
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(16, 4)
    model.embed_norm = nn.Identity()
    model.norm = _OrderNorm(log)
    model.layers = nn.ModuleList([_OrderLayer(log) for _ in range(2)])
    model.dtype = torch.float32
    model.m_emb = 1.0
    core = _CORE.adopt(model)
    core._dense_gemm_plan = types.SimpleNamespace(join=lambda: log.append("join"))

    core(attn_metadata=types.SimpleNamespace(num_contexts=0), input_ids=torch.tensor([1, 2]))
    assert log == ["layer", "layer", "norm", "join"]

    log.clear()

    def builtin_forward(self, attn_metadata, **kwargs):
        log.append("built-in forward")
        return "hidden states"

    monkeypatch.setattr(MuseSparkOpenModel, "forward", builtin_forward)
    out = core(attn_metadata=types.SimpleNamespace(num_contexts=1), input_ids=torch.tensor([1]))
    assert out == "hidden states" and log == ["built-in forward", "join"]
