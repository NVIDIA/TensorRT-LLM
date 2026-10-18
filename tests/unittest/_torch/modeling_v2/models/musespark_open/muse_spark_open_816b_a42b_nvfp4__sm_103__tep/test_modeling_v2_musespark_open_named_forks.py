# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``dense_gemms.NamedFork``: side-stream work joined by one named projection site only.

Host-only (streams are stand-ins; the captured graph's edges are read on a GPU in
``gemm/test_modeling_v2_dense_gemm.py``). Here: registration and its refusals,
which dispatchers join a fork (its role's, every layer's, and no other), that a
plan without named forks runs its dispatchers exactly as before, the capture
rules (a fork inside a capture before the stream exists launches nothing; a
fork or join from another capture state drops the outstanding fork), and that
the targets join every named fork at the end of their forward.
"""

from __future__ import annotations

import contextlib
import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch.modules import low_m_gemm
from tensorrt_llm._torch.modules.linear import Linear

pytestmark = pytest.mark.cpu_only

_QKV, _O, _POOL, _UNPOOL = (64, 36), (16, 64), (64, 32), (32, 64)


class _Stream:
    """Records the streams it is made to wait for."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.waited = []

    def wait_stream(self, other) -> None:
        self.waited.append(other)


@pytest.fixture
def streams(monkeypatch):
    """A current stream stand-in, eager by default, and a no-op ``torch.cuda.stream``."""
    current = _Stream("current")
    state = types.SimpleNamespace(current=current, capture=None, entered=[])
    monkeypatch.setattr(dense_gemms, "current_capture_id", lambda: state.capture)
    monkeypatch.setattr(dense_gemms.torch.cuda, "current_stream", lambda *args: state.current)

    @contextlib.contextmanager
    def on(stream):
        state.entered.append(stream)
        yield

    monkeypatch.setattr(dense_gemms.torch.cuda, "stream", on)
    return state


def _plan() -> dense_gemms.DenseGemmPlan:
    return dense_gemms.DenseGemmPlan(dense_gemms.default_kernels(), prefetch=True, enable_pdl=True)


def _armed(fork: dense_gemms.NamedFork, capture=None) -> _Stream:
    """``fork`` with a stand-in stream and one outstanding fork made in ``capture``."""
    side = _Stream(fork.name)
    fork._stream, fork._fork_capture, fork.pending = side, capture, True
    return side


# --- registration ------------------------------------------------------------------------------


def test_forks_register_under_unique_names_and_a_site_role():
    plan = _plan()
    assert plan.named_forks == {}
    qkv = plan.add_fork("a", "qkv")
    free = plan.add_fork("b", None)
    assert plan.named_forks == {"a": qkv, "b": free}
    assert (qkv.name, qkv.join_role, free.join_role) == ("a", "qkv", None)
    assert not qkv.pending and qkv.stream is None and qkv.sink is None
    with pytest.raises(ValueError, match="already registered"):
        plan.add_fork("a", "o_proj")
    with pytest.raises(ValueError, match="join_role"):
        plan.add_fork("c", "router")
    assert list(plan.named_forks) == ["a", "b"]


# --- which join waits --------------------------------------------------------------------------


def test_a_role_join_waits_only_for_the_forks_of_that_role(streams):
    plan = _plan()
    qkv, o_proj, free = (
        plan.add_fork("q", "qkv"),
        plan.add_fork("o", "o_proj"),
        plan.add_fork("f", None),
    )
    sides = {fork.name: _armed(fork) for fork in (qkv, o_proj, free)}
    plan.join_role("unpool")
    assert streams.current.waited == [] and qkv.pending and o_proj.pending and free.pending
    plan.join_role("qkv")
    assert streams.current.waited == [sides["q"]]
    assert not qkv.pending and o_proj.pending and free.pending and qkv.joins == 1
    plan.join_role("qkv")  # nothing outstanding any more
    assert streams.current.waited == [sides["q"]]
    plan.join_all()
    assert streams.current.waited == [sides["q"], sides["o"], sides["f"]]
    assert not any(fork.pending for fork in (qkv, o_proj, free))


def test_join_all_also_joins_the_gemm_prefetch_fork(streams):
    plan = _plan()
    named = plan.add_fork("q", "qkv")
    gemm_prefetch_on, side = _Stream("gemm_prefetch"), _armed(named)
    plan._stream, plan._fork_capture, plan.pending = gemm_prefetch_on, None, True
    plan.join_all()
    assert (
        streams.current.waited == [gemm_prefetch_on, side]
        and not plan.pending
        and not named.pending
    )


@pytest.mark.parametrize(
    "fork, now, waits",
    [(None, None, True), (7, 7, True), (None, 7, False), (7, None, False), (7, 8, False)],
    ids=("eager", "same-capture", "eager-into-capture", "capture-into-eager", "other-capture"),
)
def test_a_join_waits_only_from_the_capture_state_of_its_fork(streams, fork, now, waits):
    named = _plan().add_fork("q", "qkv")
    side = _armed(named, capture=fork)
    streams.capture = now
    named.join()
    assert not named.pending
    assert streams.current.waited == ([side] if waits else [])
    assert (named.joins, named.dropped) == ((1, 0) if waits else (0, 1))


# --- the fork ----------------------------------------------------------------------------------


def test_a_fork_runs_its_launch_on_its_own_stream_after_the_current_work(streams):
    named = _plan().add_fork("q", "qkv")
    side = _Stream("side")
    named._stream, named.sink = side, "sink"
    launched = []
    assert named.fork(lambda: launched.append(streams.entered[-1]), device=torch.device("cpu"))
    assert side.waited == [streams.current] and launched == [side]
    assert named.pending and named.forks == 1 and named._fork_capture is None


def test_a_fork_inside_a_capture_before_the_stream_exists_launches_nothing(streams):
    named = _plan().add_fork("q", "qkv")
    streams.capture = 3
    launched = []
    assert not named.fork(lambda: launched.append(1), device=torch.device("cpu"))
    assert launched == [] and not named.pending and named.skipped == 1 and named.stream is None


def test_the_first_eager_fork_makes_a_dedicated_stream_and_the_sink(streams, monkeypatch):
    made = []

    def dedicated(device):
        made.append(device)
        return _Stream("dedicated")

    monkeypatch.setattr(dense_gemms, "dedicated_stream", dedicated)
    named = _plan().add_fork("q", "qkv")
    device = torch.device("cpu")
    assert named.fork(lambda: None, device=device)
    assert made == [device] and named.stream.name == "dedicated"
    assert named.sink.dtype == torch.int32 and named.sink.shape == (1,)
    named.fork(lambda: None, device=device)
    assert made == [device], "one stream per fork, made once"


def test_consecutive_forks_need_no_join_but_a_fork_from_another_capture_state_is_dropped(streams):
    named = _plan().add_fork("q", "qkv")
    side = _Stream("side")
    named._stream, named.sink = side, "sink"
    named.fork(lambda: None)
    named.fork(lambda: None)  # same (eager) state: ordered on the side stream, one join covers both
    assert named.dropped == 0 and named.forks == 2 and streams.current.waited == []
    streams.capture = 5
    named.fork(lambda: None)  # the eager fork cannot be waited on from capture 5
    assert side.waited == [streams.current] * 3
    assert named.dropped == 1 and named.pending and named._fork_capture == 5
    named.join()
    assert streams.current.waited == [side] and named.joins == 1


# --- dispatchers -------------------------------------------------------------------------------


def _linear(n: int, k: int) -> Linear:
    linear = Linear.__new__(Linear)
    nn.Module.__init__(linear)
    linear.weight = nn.Parameter(torch.randn(n, k).to(torch.bfloat16), requires_grad=False)
    linear.register_parameter("bias", None)
    linear.use_custom_cublas_mm = False
    return linear


class _Mapping:
    tp_size = 1
    tp_rank = 0


def _model(num_layers: int = 3):
    model = nn.Module()
    layers = []
    for _ in range(num_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = _linear(_QKV[1], _QKV[0])
        layer.self_attn.o_proj = _linear(_O[1], _O[0])
        layer.feed_forward = nn.Module()
        layer.feed_forward.pooler = _linear(_POOL[1], _POOL[0])
        layer.feed_forward.unpooler = _linear(_UNPOOL[1], _UNPOOL[0])
        layer.feed_forward.mapping = _Mapping()
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


class _Engine:
    def apply(self, module, input_tensor, weight, bias, force_active=False):
        return "engine"


@pytest.fixture
def low_m_on(monkeypatch):
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    monkeypatch.setattr(dense_gemms, "kernel_error", lambda kernel, rows, k, n: None)


def _call(linear) -> str:
    linear._low_m_gemm_dispatcher = _Engine()
    x = torch.randn(8, int(linear.weight.shape[1])).to(torch.bfloat16)  # host tensor: not served
    return low_m_gemm.apply_low_m_gemm(linear, x, linear.weight, None)


def test_every_qkv_dispatcher_joins_the_qkv_fork_and_no_other_dispatcher_does(low_m_on, streams):
    model = _model()
    plan = dense_gemms.install(model, prefetch=False)
    named = plan.add_fork("q", "qkv")
    for index, layer in enumerate(model.layers):
        side = _armed(named)
        for linear in (
            layer.feed_forward.unpooler,
            layer.feed_forward.pooler,
            layer.self_attn.o_proj,
        ):
            assert _call(linear) == "engine"
            assert named.pending, f"layer {index}: only the QKV dispatcher joins"
        assert _call(layer.self_attn.qkv_proj) == "engine"
        assert not named.pending and streams.current.waited[-1] is side
    assert named.joins == len(model.layers)


def test_without_named_forks_the_dispatchers_never_ask_for_a_role_join(low_m_on, monkeypatch):
    model = _model(2)
    plan = dense_gemms.install(model, prefetch=False)

    def refuse(self, role):
        raise AssertionError("join_role called without named forks")

    monkeypatch.setattr(dense_gemms.DenseGemmPlan, "join_role", refuse)
    for layer in model.layers:
        for linear in (layer.self_attn.qkv_proj, layer.feed_forward.unpooler):
            assert _call(linear) == "engine"
    assert plan.named_forks == {}


def test_the_end_of_a_forward_joins_every_named_fork():
    log = []
    plan = types.SimpleNamespace(
        named_forks={"q": object()},
        join=lambda: log.append("join"),
        join_all=lambda: log.append("join_all"),
    )
    modeling._join_prefetch(types.SimpleNamespace(_dense_gemm_plan=plan))
    assert log == ["join_all"]
    plan.named_forks = {}
    modeling._join_prefetch(types.SimpleNamespace(_dense_gemm_plan=plan))
    assert log == ["join_all", "join"]
    modeling._join_prefetch(types.SimpleNamespace(_dense_gemm_plan=None))
    assert log == ["join_all", "join"]
