# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How ``next_qkv_prefetch`` (the next layer's QKV weight L2-prefetched after the experts) is wired.

Host-only. The prefetch kernels are checked on a GPU (``gemm/test_modeling_v2_dense_gemm.py``:
memory unchanged, the bulk prefetches in the PTX, the captured edges of a fork joined at a
PDL GEMM), and the compositions with ``next_qkv_prefetch`` bit for bit in the 8-rank composition tests.
Here: what it builds on, ``post_wait_columns``, which layers get the hook (those whose
next QKV edge ``boundary_early_release`` converted), at which row counts a fork is made (those whose QKV
kernel stages its weight before its wait and loads its last k-tiles after it, whatever
kernel table the plan carries), that the fork is the ``dense_gemms`` plan's named fork joined at the
QKV site, that the ``ExpertsForward`` wrapper calls the hook after whichever expert call the
layer runs (the built-in one, or ``ring_buddy``'s stacked call), and that the decode forward
forks layer 0's prefetch before anything else.
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
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.next_qkv_prefetch import (  # noqa: E501
    prefetch as qkv_prefetch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm import (  # noqa: E501
    dense_gemm as gemm_ops,
)
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel

pytestmark = pytest.mark.cpu_only


#: The QKV kernel per row count, rows 1..8, of the default table.
_DEFAULT_QKV = dense_gemms.default_kernels()["qkv"]


# --- what it builds on -----------------------------------------------------------------------


def test_qkv_prefetch_builds_on_the_gemms_and_the_converted_edges():
    """The fork joins at the ``dense_gemms`` QKV site, after the edge ``boundary_early_release``
    converted, from the hook the experts' forward wrapper runs (any expert call, so not
    ``replicated_experts``)."""
    direct = {"dense_gemms", "boundary_early_release"}
    assert dispatch.closure({"next_qkv_prefetch"}) >= direct
    assert "replicated_experts" not in dispatch.closure({"next_qkv_prefetch"})
    for missing in sorted(direct):
        with pytest.raises(ValueError, match="builds on"):
            dispatch.Plan(dispatch.closure({"next_qkv_prefetch"}) - {missing})


# --- byte ranges --------------------------------------------------------------------------------


def test_the_post_wait_columns_of_the_qkv_kernel():
    """split-K (64,8,4,12) over K = 8192: 4 slices of 2048 columns = 16 k-tiles of 128; the
    first 12 are staged before the wait, the last 4 (columns 1536-2047 of each slice) after."""
    assert gemm_ops.SPLITK_CTA_K == 128
    assert gemm_ops.post_wait_columns("splitk:64,8,4,12", 8192) == (1536, 512, 2048, 4)
    assert gemm_ops.post_wait_columns("splitk:64,8,4,10", 8192) == (1280, 768, 2048, 4)
    assert gemm_ops.post_wait_columns("splitk:64,8,1,12", 1024) is None  # 8 tiles <= 12 stages
    assert gemm_ops.post_wait_columns("splitk:64,8,4,12", 4096) is None  # 8 tiles per slice
    assert gemm_ops.post_wait_columns("direct:256,1", 8192) is None
    assert gemm_ops.post_wait_columns("cublas", 8192) is None


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


def test_each_range_mode_launches_its_bytes(monkeypatch):
    full = _Recorder()
    monkeypatch.setattr(qkv_prefetch, "l2_prefetch", full)
    fork = types.SimpleNamespace(sink="sink")
    prefetch = qkv_prefetch.QkvPrefetch(fork, frozenset(range(3, 9)))
    weight = torch.empty(2304, 8192, dtype=torch.bfloat16)
    prefetch.launches(weight, 8)()
    assert full.calls == [(weight, "sink")]
    full.calls.clear()
    prefetch.launches(weight, 1)()
    assert full.calls == [(weight, "sink")]


# --- install --------------------------------------------------------------------------------------


class _Fork:
    """Records the launches it is asked to fork."""

    def __init__(self):
        self.forked = []
        self.sink = "sink"

    def fork(self, launch, device=None):
        self.forked.append(launch)
        return True


def _model(num_layers=4, converted=(0, 1, 2)):
    model = nn.Module()
    layers = []
    for index in range(num_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = types.SimpleNamespace(weight=torch.full((2, 2), float(index)))
        layer.feed_forward = nn.Module()
        layer.add_ffn = types.SimpleNamespace(
            _boundary_trigger="entry" if index in converted else "exit"
        )
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


def _plan(enable_pdl=True):
    return dense_gemms.DenseGemmPlan(
        dense_gemms.default_kernels(), prefetch=True, enable_pdl=enable_pdl
    )


def test_install_hooks_the_layers_whose_next_qkv_edge_early_release_converted():
    model = _model(5, converted=(0, 2, 3))
    plan = _plan()
    prefetch = qkv_prefetch.install(model, plan)
    assert set(plan.named_forks) == {qkv_prefetch.FORK_NAME}
    fork = plan.named_forks[qkv_prefetch.FORK_NAME]
    assert fork.join_role == "qkv" and prefetch.fork is fork
    assert prefetch.rows == frozenset(range(3, 9))  # split-K QKV rows of the default plan
    hooked = {
        index
        for index, layer in enumerate(model.layers)
        if hasattr(layer.feed_forward, "_qkv_prefetch_after_experts")
    }
    assert hooked == {0, 2, 3}
    for index in hooked:
        after = model.layers[index].feed_forward._qkv_prefetch_after_experts
        assert after.next_weight() is model.layers[index + 1].self_attn.qkv_proj.weight
    assert prefetch.first_weight() is model.layers[0].self_attn.qkv_proj.weight


@pytest.mark.parametrize(
    "break_it, match",
    [
        (lambda m: None, None),
        (
            lambda m: [setattr(layer.add_ffn, "_boundary_trigger", "exit") for layer in m.layers],
            "boundary_early_release converted no",
        ),
    ],
)
def test_install_refuses_without_a_converted_edge(break_it, match):
    model = _model()
    break_it(model)
    plan = _plan()
    if match is None:
        qkv_prefetch.install(model, plan)
        return
    with pytest.raises(ValueError, match=match):
        qkv_prefetch.install(model, plan)
    assert plan.named_forks == {}


def test_install_refuses_without_pdl_or_a_plan():
    with pytest.raises(ValueError, match="programmatic dependent launch"):
        qkv_prefetch.install(_model(), _plan(enable_pdl=False))
    with pytest.raises(ValueError, match="programmatic dependent launch"):
        qkv_prefetch.install(_model(), None)


def _plan_with(qkv_kernels):
    """The default table with ``qkv_kernels`` (rows 1..8) at the QKV site."""
    return dense_gemms.DenseGemmPlan(
        dict(dense_gemms.default_kernels(), qkv=tuple(qkv_kernels)),
        prefetch=True,
        enable_pdl=True,
    )


@pytest.mark.parametrize(
    "qkv_kernels",
    [(dense_gemms.CUBLAS,) * 8, _DEFAULT_QKV[:2] + ("splitk:64,8,8,12",) * 6],
    ids=["cublas", "splitk-8-tiles"],
)
def test_install_refuses_when_no_qkv_kernel_loads_weight_after_its_wait(qkv_kernels):
    """cuBLAS stages nothing before its wait; split-K (64,8,8,12) gives each CTA 8 k-tiles,
    all inside the 12-stage ring, so it loads nothing after its wait (and could trigger its
    dependents before it)."""
    plan = _plan_with(qkv_kernels)
    with pytest.raises(ValueError, match="before its wait and loads its last k-tiles after it"):
        qkv_prefetch.install(_model(), plan)
    assert plan.named_forks == {}


def test_install_forks_only_where_the_qkv_kernel_loads_weight_after_its_wait():
    """Rows 3-5 on split-K (64,8,8,12) (no post-wait k-tile), rows 6-8 on the default
    (64,8,4,12): only rows 6-8 fork."""
    plan = _plan_with(_DEFAULT_QKV[:2] + ("splitk:64,8,8,12",) * 3 + _DEFAULT_QKV[5:])
    kernels = plan.kernels["qkv"]
    assert [gemm_ops.post_wait_columns(kernels[r - 1], 8192) for r in (3, 6)] == [
        None,
        (1536, 512, 2048, 4),
    ]
    assert qkv_prefetch.fork_rows(kernels) == frozenset({6, 7, 8})
    prefetch = qkv_prefetch.install(_model(), plan)
    assert prefetch.rows == frozenset({6, 7, 8})
    assert qkv_prefetch.fork_rows(_DEFAULT_QKV) == frozenset(range(3, 9))


@pytest.mark.parametrize("rows, forks", [(1, 0), (2, 0), (3, 1), (8, 1), (9, 0), (16, 0)])
def test_a_fork_is_made_only_at_the_rows_early_release_launches_qkv_early(rows, forks):
    fork = _Fork()
    prefetch = qkv_prefetch.QkvPrefetch(fork, frozenset(range(3, 9)))
    weight = torch.empty(2, 2)
    after = qkv_prefetch.AfterExperts(prefetch, lambda: weight)
    after(rows)
    prefetch.first_weight = lambda: weight
    prefetch.graph_start(rows)
    assert len(fork.forked) == 2 * forks


# --- call sites ------------------------------------------------------------------------------------


def _hooked_moe(log):
    moe = nn.Module()
    moe.experts = nn.Module()
    moe._qkv_prefetch_after_experts = lambda rows: log.append(("qkv_prefetch", rows))
    return moe


def test_the_wrapper_calls_the_hook_after_the_built_in_expert_call():
    """At tp 4 (no ring-buddy stack) the layer runs the built-in expert forward; the
    wrapper forks the prefetch after it, on ``do_finalize=False`` calls only."""
    log = []
    moe = _hooked_moe(log)
    moe.experts.forward = (
        lambda x, logits, *a, do_finalize=True, **k: log.append(("builtin", do_finalize)) or "out"
    )
    model = types.SimpleNamespace(
        layers=[
            types.SimpleNamespace(feed_forward=moe),
            types.SimpleNamespace(feed_forward=nn.Module()),
        ]
    )
    assert qkv_prefetch.wrap_expert_forwards(model) == 1
    x = torch.zeros(8, 4)
    assert moe.experts(x, torch.zeros(8, 2), do_finalize=False, output_dtype=None) == "out"
    assert log == [("builtin", False), ("qkv_prefetch", 8)]
    log.clear()
    moe.experts(x, torch.zeros(8, 2), do_finalize=True)
    assert log == [("builtin", True)]
    # Idempotent: a second wrap (a weight reload) keeps the one wrapper.
    wrapper = moe.experts.forward
    assert qkv_prefetch.wrap_expert_forwards(model) == 0
    assert moe.experts.forward is wrapper


def test_the_wrapper_calls_the_hook_after_the_stacked_expert_call(monkeypatch):
    """At tp 8 the ring-buddy stacked call is patched in first; the wrapper sits outside it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
        ring_buddy,
    )

    log = []
    moe = _hooked_moe(log)
    routing = types.SimpleNamespace(
        apply=lambda logits, ids=None: log.append("routing"),
        routing_method_type=0,
    )
    moe.experts.routing_method = routing
    moe.experts.forward = lambda *a, **k: log.append("builtin")
    placement = (None, None, "ids", "weights", None)
    state = types.SimpleNamespace(
        active=True,
        kind="bf16",
        stacks="stacks",
        by_window={2 * ring_buddy.BLOCK: "stacks"},
        default_window=2 * ring_buddy.BLOCK,
        current=lambda logits: placement,
    )
    monkeypatch.setattr(ring_buddy, "_backend", lambda m: None)
    monkeypatch.setattr(
        ring_buddy,
        "_bf16_call",
        lambda backend, routing: lambda stacks, x, ids, w: log.append("experts") or "out",
    )
    ring_buddy._patch_experts(moe, state)
    x = torch.zeros(8, 4)
    # Without the wrapper the stacked call forks nothing.
    assert moe.experts.forward(x, torch.zeros(8, 2), do_finalize=False) == "out"
    assert log == ["routing", "experts"]
    log.clear()
    model = types.SimpleNamespace(layers=[types.SimpleNamespace(feed_forward=moe)])
    assert qkv_prefetch.wrap_expert_forwards(model) == 1
    assert moe.experts.forward(x, torch.zeros(8, 2), do_finalize=False) == "out"
    assert log == ["routing", "experts", ("qkv_prefetch", 8)]


class _OrderLayer(nn.Module):
    def __init__(self, log: list) -> None:
        super().__init__()
        self.log = log
        self.pre_qkv_proj_norm = nn.Identity()

    def forward(self, position_ids, hidden_states, attn_metadata, next_layer_norm=None, **kwargs):
        self.log.append("layer")
        return hidden_states if next_layer_norm is None else (hidden_states, hidden_states)


class _Embedding(nn.Embedding):
    def __init__(self, log):
        super().__init__(16, 4)
        self.log = log

    def forward(self, ids):
        self.log.append("embedding")
        return super().forward(ids)


def test_the_decode_forward_forks_layer_0s_prefetch_before_the_embedding(monkeypatch):
    log: list = []
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    model.embed_tokens = _Embedding(log)
    model.embed_norm = nn.Identity()
    model.norm = nn.Identity()
    model.layers = nn.ModuleList([_OrderLayer(log) for _ in range(2)])
    model.dtype = torch.float32
    model.m_emb = 1.0
    core = modeling.MuseSparkOpenTepModelingV2.adopt(model)
    core._decode.qkv_prefetch = types.SimpleNamespace(
        graph_start=lambda rows: log.append(("graph_start", rows))
    )
    core(attn_metadata=types.SimpleNamespace(num_contexts=0), input_ids=torch.tensor([1, 2, 3]))
    assert log[:2] == [("graph_start", 3), "embedding"]
    assert log[2:] == ["layer", "layer"]
