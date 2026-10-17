# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""attention_output_tail_prefetch: off-path L2 prefetch of the attention_output_tail weights (protocol and kernel).

Uses the real DSA ``MLA`` module of TP8 rank 0 with real layer-3
``v_b_proj``/``o_proj`` weights and synthetic latents (the attention_output_tail adapter test's
builders); the real ``MLA.forward`` path is in ``test_tail_prefetch_mla_forward.py``.

* the prefetch kernel runs on every config the benchmarks use and changes no
  tensor;
* attention_output_tail_prefetch's default tail (native W_UV + native o_proj on prefetched weights)
  is bit-identical to the native tail, and with the attention_output_tail kernels on prefetched
  weights it is bit-identical to the attention_output_tail: the prefetch has no effect on
  results;
* fork/join protocol: ``before_attention`` forks only on an armed call and
  only for layers the config selects; ``project`` always joins (engaged or
  declined), at every configured join point; a leftover fork raises in the
  next ``arm``;
* CUDA-graph capture of arm → before_attention → defer → project replays
  correctly with the auxiliary stream rejoined;
* ``make_attn_tail`` picks the attention_output_tail_prefetch schedule of the running
  neighbours exactly when ``attention_output_tail_prefetch`` runs;
* the adopted attention (``GlmMLA``) carries the attention_output_tail_prefetch flag and its tail
  uses ``TAIL_PREFETCH_SCHEDULE``; the layer adoption hands the MoE gate weight to the tail.
"""

import dataclasses
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "attention_output_tail"))
sys.path.insert(0, os.path.dirname(_HERE))
import plan_testlib  # noqa: E402
import tail_reference  # noqa: E402
import test_attn_tail_mla_adapter as tail_adapter  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E402
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail import (  # noqa: E402, E501
    prefetch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E402, E501
    DEFAULT_CONFIG,
    TAIL_PREFETCH_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
    TAIL_PREFETCH_STAGED_NATIVE_SCHEDULE,
    attn_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E402, E501
    TailConfig as TailConfig_,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E402, E501
    AttnTail,
    make_attn_tail,
)

pytestmark = tail_adapter.pytestmark

_R = dataclasses.replace
STAGED = TAIL_PREFETCH_STAGED_NATIVE_SCHEDULE
# attention_output_tail_prefetch's prefetch protocol with the attention_output_tail kernels (the
# default uses the native ones).
TAIL_PREFETCH_KERNEL_CONFIGS = _R(STAGED, oproj_engine="cute_stream")


def _no_ar():
    from tensorrt_llm._torch.distributed import AllReduceParams

    return AllReduceParams(enable_allreduce=False)


@pytest.mark.parametrize(
    "cfg",
    [
        prefetch.PrefetchConfig(),
        prefetch.PrefetchConfig(ctas=152),
        prefetch.PrefetchConfig(chunk=16384, ctas=26),
        prefetch.PrefetchConfig(gap_ns=500),
    ],
)
def test_prefetch_kernel_changes_nothing(cfg):
    mla = tail_adapter._mla()
    weights = [mla.v_b_proj, mla.o_proj.weight]
    assert prefetch.check_prefetch_tensors(weights, cfg.chunk) == ""
    before = [w.clone() for w in weights]
    probe = torch.randn(4096, device="cuda")
    probe_before = probe.clone()
    prefetch.launch_prefetch(weights, cfg)
    torch.cuda.synchronize()
    for w, b in zip(weights, before):
        assert torch.equal(w, b)
    assert torch.equal(probe, probe_before)


def test_prefetch_rejects_bad_inputs():
    a = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
    assert prefetch.check_prefetch_tensors([a, a, a, a], 4096) != ""
    assert prefetch.check_prefetch_tensors([a, a], 4000) != ""
    odd = torch.zeros(1000, dtype=torch.bfloat16, device="cuda")
    assert prefetch.check_prefetch_tensors([a, odd], 4096) != ""


def _drive(tail, mla, latent, out_rows=None, decline=False):
    rows = latent.shape[0]
    out_rows = out_rows or rows
    attn_out = torch.empty((out_rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point=tail.config.prefetch_fork)
    if decline:
        big = torch.zeros((9, 8, 512), dtype=torch.bfloat16, device="cuda")
        assert not tail.defer_w_uv(mla, meta, big, attn_out[:rows])
    else:
        assert tail.defer_w_uv(mla, meta, latent, attn_out[:rows])
    return tail.project(mla, attn_out, _no_ar())


@pytest.mark.parametrize("join", ["before_wuv", "before_oproj", "after_oproj"])
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_tail_prefetch_tail_bitwise_equals_attn_tail(join, rows):
    mla = tail_adapter._mla()
    latent = tail_adapter._latent(rows, seed=rows)
    ref = attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, rows, config=DEFAULT_CONFIG)
    tail = AttnTail("layer3", _R(TAIL_PREFETCH_KERNEL_CONFIGS, prefetch_join=join))
    got = _drive(tail, mla, latent)
    torch.cuda.synchronize()
    assert torch.equal(got, ref)
    assert tail.prefetch_forks == 1
    assert not tail._prefetch.forked and not tail._armed
    assert tail._pending is None


def test_fork_only_on_armed_calls_and_join_on_decline():
    mla = tail_adapter._mla()
    tail = AttnTail("layer3", TAIL_PREFETCH_KERNEL_CONFIGS)
    # Not armed: no fork.
    tail.before_attention(mla, point="attn")
    assert tail.prefetch_forks == 0
    assert not tail.arm(mla, tail_adapter._meta(6, num_contexts=1))
    tail.before_attention(mla, point="attn")
    assert tail.prefetch_forks == 0
    # Armed, but called at the other fork point: no fork.
    assert tail.arm(mla, tail_adapter._meta(6))
    tail.before_attention(mla, point="start")
    assert tail.prefetch_forks == 0
    tail._armed = False
    # Armed, forked, then declined at the W_UV step: project joins.
    assert _drive(tail, mla, tail_adapter._latent(6, seed=3), decline=True) is None
    assert tail.prefetch_forks == 1 and not tail._prefetch.forked
    # An attention_output_tail-only tail never forks.
    served_tail = AttnTail("layer3", DEFAULT_CONFIG)
    out = _drive(served_tail, mla, tail_adapter._latent(6, seed=4))
    assert out is not None and served_tail.prefetch_forks == 0
    # A leftover fork is an error at the next arm.
    assert tail.arm(mla, tail_adapter._meta(6))
    tail.before_attention(mla, point="attn")
    tail._armed = False
    with pytest.raises(RuntimeError, match="not consumed"):
        tail.arm(mla, tail_adapter._meta(6))
    tail._prefetch.join()


def test_layer_selection():
    mla = tail_adapter._mla()
    has_indexer = getattr(mla.mqa, "indexer", None) is not None
    for layers in ("all", "shared", "full"):
        tail = AttnTail("layer3", _R(TAIL_PREFETCH_KERNEL_CONFIGS, prefetch_layers=layers))
        _drive(tail, mla, tail_adapter._latent(6, seed=5))
        want = layers == "all" or (layers == "full") == has_indexer
        assert tail.prefetch_forks == int(want), (layers, has_indexer)


def _native_step(tail, mla, latent, out_rows=None):
    """What the patched op and MLA.forward do with attention_output_tail_prefetch's native engine."""
    from tensorrt_llm._torch.attention.mla import MLA

    rows = latent.shape[0]
    out_rows = out_rows or rows
    attn_out = torch.zeros((out_rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point="attn")
    # The op's native W_UV (the tail declines the W_UV step).
    assert not tail.defer_w_uv(mla, meta, latent, attn_out[:rows])
    v_view = attn_out[:rows].view(rows, mla.num_heads_tp, mla.v_head_dim)
    MLA._bmm_bf16_out(
        mla,
        latent.transpose(0, 1),
        mla.v_b_proj,
        mla.v_b_proj.transpose(1, 2),
        v_view.transpose(0, 1),
    )
    return tail.project(mla, attn_out, _no_ar())


@pytest.mark.parametrize("join", ["before_oproj", "after_oproj"])
@pytest.mark.parametrize("rows,out_rows", [(1, 1), (6, 6), (6, 8), (8, 8)])
def test_native_engine_bitwise_equals_native(join, rows, out_rows):
    mla = tail_adapter._mla()
    latent = tail_adapter._latent(rows, seed=10 + rows)
    native = tail_adapter._native(mla, latent, out_rows)
    tail = AttnTail("layer3", _R(STAGED, prefetch_join=join))
    got = _native_step(tail, mla, latent, out_rows)
    torch.cuda.synchronize()
    assert torch.equal(got[:rows], native[:rows])
    assert tail.prefetch_forks == 1 and tail.engaged_calls == 1
    assert not tail._prefetch.forked and not tail._armed


def test_native_engine_without_fork_takes_native_path():
    mla = tail_adapter._mla()
    # Layer selection excludes this layer: no fork, so the caller's native
    # projection runs (project returns None).
    has_indexer = getattr(mla.mqa, "indexer", None) is not None
    other = "shared" if has_indexer else "full"
    tail = AttnTail("layer3", _R(STAGED, prefetch_layers=other))
    rows = 6
    attn_out = torch.zeros((rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point="attn")
    assert tail.prefetch_forks == 0
    assert not tail.defer_w_uv(mla, meta, tail_adapter._latent(rows, seed=6), attn_out)
    assert tail.project(mla, attn_out, _no_ar()) is None


def test_native_engine_graph_capture_and_replay():
    mla = tail_adapter._mla()
    tail = AttnTail("layer3", STAGED)
    rows = 6
    lat = tail_adapter._latent(rows, seed=17).contiguous()
    _native_step(tail, mla, lat)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _native_step(tail, mla, lat)
    assert not tail._prefetch.forked
    for seed in range(6):
        lat.copy_(tail_adapter._latent(rows, seed=300 + seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, tail_adapter._native(mla, lat, rows))


def test_graph_capture_and_replay():
    mla = tail_adapter._mla()
    tail = AttnTail("layer3", TAIL_PREFETCH_KERNEL_CONFIGS)
    rows = 6
    meta = tail_adapter._meta(rows)
    lat = tail_adapter._latent(rows, seed=7).contiguous()
    out = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")

    def step():
        assert tail.arm(mla, meta)
        tail.before_attention(mla, point="attn")
        assert tail.defer_w_uv(mla, meta, lat, out)
        return tail.project(mla, out, _no_ar())

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    assert not tail._prefetch.forked
    for seed in range(6):
        lat.copy_(tail_adapter._latent(rows, seed=200 + seed))
        graph.replay()
        torch.cuda.synchronize()
        ref = attn_tail(lat, mla.v_b_proj, mla.o_proj.weight, rows, config=DEFAULT_CONFIG)
        assert torch.equal(captured, ref)


def test_make_attn_tail_selects_config(monkeypatch):
    mla = tail_adapter._build_mla(layer_idx=tail_adapter.LAYER, num_hidden_layers=78)
    cases = [
        ("attention_output_tail", DEFAULT_CONFIG),
        ("attention_output_tail,attention_output_tail_prefetch", TAIL_PREFETCH_SCHEDULE),
        (
            "attention_output_tail,attention_output_tail_prefetch,sparse_mla_decode",
            TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
        ),
        (
            "attention_output_tail,attention_output_tail_prefetch,sparse_mla_decode,post_attention_hop",
            TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
        ),
        (
            "attention_output_tail,attention_output_tail_prefetch,post_attention_hop",
            DEFAULT_CONFIG,
        ),
    ]
    for features, want in cases:
        plan_testlib.select(dispatch, features, monkeypatch)
        tail = make_attn_tail(mla)
        assert tail is not None and tail.config == want, features


@pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PREFETCH_ADAPTER_PATCH_APPLIED") != "1",
    reason="opt-in",
)
def test_adoption_selects_tail_prefetch(monkeypatch):
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
        attention_forward,
    )

    assert "before_attention" in inspect.getsource(attention_forward.forward_absorption_generation)
    plan_testlib.select(
        dispatch, "attention_output_tail,attention_output_tail_prefetch", monkeypatch
    )
    mla = tail_adapter._build_adopted_mla(layer_idx=tail_adapter.LAYER, num_hidden_layers=78)
    assert mla._tail_prefetch
    assert mla._attn_tail.config == TAIL_PREFETCH_SCHEDULE
    plan_testlib.select(dispatch, "attention_output_tail", monkeypatch)
    assert not tail_adapter._build_adopted_mla(
        layer_idx=tail_adapter.LAYER, num_hidden_layers=78
    )._tail_prefetch
    draft = tail_adapter._build_adopted_mla(layer_idx=78, num_hidden_layers=78)
    assert draft._attn_tail is None


@pytest.mark.parametrize("grid,slots,early", [(96, 3, 3), (96, 3, 0), (128, 3, 3), (64, 3, 2)])
@pytest.mark.parametrize("rows,out_rows", [(1, 1), (6, 6), (6, 8), (8, 8), (8, 16)])
def test_ring_oproj_within_budget(grid, slots, early, rows, out_rows):
    """The small-footprint o_proj against the FP64 oracle of the same ``v``
    (frozen attention_output_tail budget relative to the native GEMM), padded rows zero."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.cute_oproj_ring import (  # noqa: E501
        ring_oproj,
    )

    mla = tail_adapter._mla()
    latent = tail_adapter._latent(rows, seed=40 + rows)
    native, v = tail_reference.native_attn_tail(
        latent, mla.v_b_proj, mla.o_proj.weight, rows, out_rows=out_rows, return_v=True
    )
    out = torch.full(
        (out_rows, mla.o_proj.weight.shape[0]), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    ring_oproj(v, mla.o_proj.weight, out, use_pdl=False, grid=grid, slots=slots, early=early)
    torch.cuda.synchronize()
    ref = v[:rows].double() @ mla.o_proj.weight.double().t()
    e_ring = tail_reference.error_stats(out[:rows], ref)
    e_nat = tail_reference.error_stats(native[:rows], ref)
    assert e_ring["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (e_ring, e_nat)
    assert e_ring["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_ring["ref_max"])
    assert bool((out[rows:] == 0).all())


def test_ring_engine_graph_replay_matches_native():
    mla = tail_adapter._mla()
    cfg = _R(STAGED, native_oproj="ring")
    tail = AttnTail("layer3", cfg)
    rows = 6
    lat = tail_adapter._latent(rows, seed=51).contiguous()
    _native_step(tail, mla, lat)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _native_step(tail, mla, lat)
    for seed in range(6):
        lat.copy_(tail_adapter._latent(rows, seed=500 + seed))
        graph.replay()
        torch.cuda.synchronize()
        native = tail_adapter._native(mla, lat, rows)
        torch.testing.assert_close(
            captured.clone(),
            native,
            rtol=2**-6,
            atol=2**-6 * native.float().pow(2).mean().sqrt().item(),
        )
    assert tail.prefetch_forks == 2 and not tail._prefetch.forked


def test_fork_ahead_protocol():
    mla = tail_adapter._mla()
    prev = _R(STAGED, prefetch_fork="prev")
    # Other fork points ignore fork_ahead.
    tail = AttnTail("layer3", STAGED)
    tail.fork_ahead(mla)
    assert tail.prefetch_forks == 0
    # Forked ahead, then an armed call: before_attention does not fork
    # again, project runs the native o_proj and joins; bit-identical.
    tail = AttnTail("layer3", prev)
    rows = 6
    latent = tail_adapter._latent(rows, seed=60)
    tail.fork_ahead(mla)
    assert tail.prefetch_forks == 1 and tail._prefetch.forked
    meta = tail_adapter._meta(rows)
    attn_out = torch.zeros((rows, 2048), dtype=torch.bfloat16, device="cuda")
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point="attn")
    tail.before_attention(mla, point="start")
    assert tail.prefetch_forks == 1
    from tensorrt_llm._torch.attention.mla import MLA

    assert not tail.defer_w_uv(mla, meta, latent, attn_out)
    v_view = attn_out.view(rows, mla.num_heads_tp, mla.v_head_dim)
    MLA._bmm_bf16_out(
        mla,
        latent.transpose(0, 1),
        mla.v_b_proj,
        mla.v_b_proj.transpose(1, 2),
        v_view.transpose(0, 1),
    )
    got = tail.project(mla, attn_out, _no_ar())
    torch.cuda.synchronize()
    assert torch.equal(got, tail_adapter._native(mla, latent, rows))
    assert not tail._prefetch.forked and not tail._ahead
    # Forked ahead, then a call that declines: arm joins at once.
    tail.fork_ahead(mla)
    assert not tail.arm(mla, tail_adapter._meta(rows, num_contexts=1))
    assert not tail._prefetch.forked and not tail._ahead
    # Twice ahead without a call in between is a no-op the second time.
    tail.fork_ahead(mla)
    tail.fork_ahead(mla)
    assert tail.prefetch_forks == 3
    assert not tail.arm(mla, tail_adapter._meta(9))
    assert not tail._prefetch.forked


def test_fork_ahead_graph_capture():
    mla = tail_adapter._mla()
    tail = AttnTail("layer3", _R(STAGED, prefetch_fork="prev"))
    rows = 6
    lat = tail_adapter._latent(rows, seed=61).contiguous()

    def step():
        tail.fork_ahead(mla)
        return _native_step(tail, mla, lat)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    assert not tail._prefetch.forked
    for seed in range(4):
        lat.copy_(tail_adapter._latent(rows, seed=700 + seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, tail_adapter._native(mla, lat, rows))


def test_successor_prefetch_protocol_and_bitwise():
    mla = tail_adapter._mla()
    router = torch.randn(256, 6144, device="cuda").to(torch.bfloat16)
    before = router.clone()
    for staged in (False, True):
        cfg = TailConfig_(successor_prefetch=True, prefetch_l2=staged, oproj_engine="cute_stream")
        tail = AttnTail("layer3", cfg)
        rows = 6
        latent = tail_adapter._latent(rows, seed=80)
        # Without successor weights and without tail staging: no fork.
        if not staged:
            assert _drive(tail, mla, latent) is not None
            assert tail.prefetch_forks == 0
        tail.set_successor_weights([router])
        got = _drive(tail, mla, latent)
        torch.cuda.synchronize()
        ref = attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, rows, config=DEFAULT_CONFIG)
        assert torch.equal(got, ref)
        assert tail.prefetch_forks == 1 and not tail._prefetch.forked
    assert torch.equal(router, before)


def test_prefetch_kernel_three_regions():
    mla = tail_adapter._mla()
    router = torch.randn(256, 6144, device="cuda").to(torch.bfloat16)
    tensors = [mla.v_b_proj, mla.o_proj.weight, router]
    assert prefetch.check_prefetch_tensors(tensors, 4096) == ""
    copies = [t.clone() for t in tensors]
    prefetch.launch_prefetch(tensors, prefetch.PrefetchConfig())
    prefetch.launch_prefetch([router], prefetch.PrefetchConfig(ctas=8))
    torch.cuda.synchronize()
    for t, c in zip(tensors, copies):
        assert torch.equal(t, c)


@pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PREFETCH_ADAPTER_PATCH_APPLIED") != "1",
    reason="needs attention_output_tail and attention_output_tail_prefetch selected",
)
def test_adoption_registers_router_weight():
    """The layer adoption hands the MoE gate weight to the tail."""
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmDecoderLayer,
    )

    src = inspect.getsource(GlmDecoderLayer._adopt_tail_prefetch)
    assert "set_successor_weights" in src and 'is_enabled("attention_output_tail_prefetch")' in src


def test_ring_oproj_refuses_publishing():
    """The ring o_proj has no live-rows-only publish path (the into-hop entry needs
    every hop row); a caller that asks for it gets an error, not a partial
    publication.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.cute_oproj_ring import (  # noqa: E501
        ring_oproj,
    )

    mla = tail_adapter._mla()
    v = torch.zeros((6, mla.o_proj.weight.shape[1]), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((8, mla.o_proj.weight.shape[0]), dtype=torch.bfloat16, device="cuda")
    words = torch.zeros(16, dtype=torch.int32, device="cuda")
    flags = torch.zeros(4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="withdrawn"):
        ring_oproj(
            v, mla.o_proj.weight, out, use_pdl=False, mc_words=words, flags=flags, publish_rank=0
        )
