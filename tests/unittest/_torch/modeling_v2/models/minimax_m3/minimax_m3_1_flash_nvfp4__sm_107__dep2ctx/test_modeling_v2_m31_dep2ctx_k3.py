# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep2ctx large prefill breakable graphs (CPU, fakes): the RuntimeHooks extend the capture list with
DEP2CTX_BCG_GRID, and the pointer guard raises before the first captured segment replays when a noted address moved;
the engine's graph release resets the guard so a recapture on a new KV pool does not raise."""

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import bcg_ptr_guard
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import prefill_pad_rows as ppr

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        modeling,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_TARGET_CLASS = modeling.ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx


@pytest.fixture
def capturing(monkeypatch):
    state = types.SimpleNamespace(on=False, bcg=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.on)
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: state.bcg)
    return state


def _model(capacity=16, enabled=True):
    cls = _TARGET_CLASS
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model._pad_rows = ppr.PrefillPadRows(
        capacity, enabled=enabled, in_graph=lambda: modeling.is_in_breakable_cuda_graph()
    )
    model._pad_rows.allocate("cpu")
    model._bcg_reserved_logged = set()
    model._bcg_ptrs = bcg_ptr_guard.StaticPointerGuard("t")
    return model


class _Msa:
    def __init__(self, indexer="nvfp4"):
        self.indexer_kv_dtype = indexer


def _attn(**overrides):
    attn = types.SimpleNamespace(
        register_to_config=True,
        attn=_Msa(),
        main_kv_is_nvfp4=True,
        layer_idx=3,
        layer_idx_str="3",
        enable_fused_qkv_index_projection=True,
        main_kv_is_fp8=False,
        _bcg_ptrs=bcg_ptr_guard.StaticPointerGuard("t"),
    )
    attn.__dict__.update(overrides)
    return attn


def _a1_metadata(layers=(3,)):
    caches = {i: (torch.zeros(4), torch.zeros(4), torch.zeros(4)) for i in layers}
    return types.SimpleNamespace(
        msa_layer_cache_tensors=caches, msa_out_cache_loc=torch.full((16,), -1, dtype=torch.int32)
    )


def _sparse_attn(fused_results, calls, guard=None):
    results = list(fused_results)

    def fused(packed, position_ids, attn_metadata, *, cache_tensors=None, hidden_states=None):
        calls.append(("fused", packed, cache_tensors))
        return results.pop(0)

    return _attn(
        _a1_fused_in_bcg_takes=lambda md: True,
        _fused_fp8_qkv_indexer_norm_rope_kv_insert=fused,
        _forward_attention_core=lambda *a: calls.append(("core",) + a[:4]) or "o",
        o_proj=lambda o, all_reduce_params=None: ("proj", o),
        _bcg_ptrs=guard if guard is not None else bcg_ptr_guard.StaticPointerGuard("t"),
    )


class _Segment:
    """A recorded captured segment: replay() logs its name instead of launching kernels."""

    def __init__(self, name, events):
        self.name, self.events = name, events

    def replay(self):
        self.events.append(self.name)


@pytest.fixture
def replay_rig(monkeypatch):
    """The outer forward through the real BreakableCUDAGraphRunner.execute; segments record their replays and segment 0
    stands for the A1 producer + KV insert writing through the noted addresses."""
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import BreakableCUDAGraph
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (
        BreakableCUDAGraphRunner,
        BreakableCUDAGraphRunnerState,
    )

    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: object())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: object())
    guard = bcg_ptr_guard.StaticPointerGuard("t")
    monkeypatch.setattr(
        modeling.MiniMaxM3ForCausalLM,
        "forward",
        lambda self, attn_metadata, input_ids=None, **kwargs: self.model(input_ids),
    )
    events = []
    graph = BreakableCUDAGraph()
    graph._segments.extend([_Segment("producer", events), _Segment("tail", events)])
    graph._break_functions.append(lambda: events.append("attention break"))

    class Body(torch.nn.Module):
        def forward(self, input_ids):
            events.append("eager body")
            return "eager"

    model = _model(capacity=8)
    model._bcg_ptrs = guard
    model.model = Body()
    runner = BreakableCUDAGraphRunner(model.model)
    runner._graphs[8] = graph
    runner._outputs[8] = "replayed"
    md = _a1_metadata(layers=(3, 4))
    md.num_tokens, md.padded_num_tokens = 5, 8
    # What the capture noted: layer 3's producer in segment 0, layer 4's in a later one.
    for layer in (3, 4):
        guard.note(layer, modeling._a1_bcg_ptrs(md, layer))

    def replay():
        return runner.execute(
            8, lambda: model.forward(md, input_ids=torch.zeros(8, dtype=torch.int32))
        )

    def eager():
        return model.forward(md, input_ids=torch.zeros(8, dtype=torch.int32))

    idle = BreakableCUDAGraphRunnerState.IDLE
    return types.SimpleNamespace(
        md=md,
        guard=guard,
        events=events,
        model=model,
        runner=runner,
        replay=replay,
        eager=eager,
        idle=idle,
    )


def _swap_loc(md):
    md.msa_out_cache_loc = torch.full((16,), -1, dtype=torch.int32)


def test_a_replay_checks_the_pointers_before_the_first_segment(replay_rig):
    assert replay_rig.replay() == "replayed"
    assert replay_rig.events == ["producer", "attention break", "tail"]
    assert replay_rig.guard.checks == 1
    for _ in range(3):
        assert replay_rig.replay() == "replayed"
    assert replay_rig.guard.checks == 4  # every replay, not only the first ones


def _swap_kv(md):
    kv, index_k, scale = md.msa_layer_cache_tensors[3]
    md.msa_layer_cache_tensors[3] = (torch.zeros(4), index_k, scale)


def _swap_later_index_k(md):
    """Only the second noted layer's view moves; the first noted layer still matches."""
    kv, index_k, scale = md.msa_layer_cache_tensors[4]
    md.msa_layer_cache_tensors[4] = (kv, torch.zeros(4), scale)


def _drop_layer(md):
    md.msa_layer_cache_tensors = {}


@pytest.mark.parametrize(
    "swap",
    [_swap_loc, _swap_kv, _swap_later_index_k, _drop_layer],
    ids=["loc", "kv", "later-layer-index-k", "layer"],
)
def test_a_moved_buffer_raises_before_the_producer_segment_replays(replay_rig, swap):
    """The check runs before segment 0 (the A1 producer's KV insert) replays through the captured addresses, not in
    the attention break after it."""
    assert replay_rig.replay() == "replayed"
    replay_rig.events.clear()
    kept = (
        replay_rig.md.msa_out_cache_loc,
        dict(replay_rig.md.msa_layer_cache_tensors),
    )  # keep the old buffers live
    swap(replay_rig.md)
    with pytest.raises(RuntimeError, match="buffers moved since capture"):
        replay_rig.replay()
    assert replay_rig.events == []  # no segment and no break ran
    assert replay_rig.model._pad_rows._rows == 0 and replay_rig.model._pad_rows.writes == 1
    assert replay_rig.runner.state is replay_rig.idle
    del kept


def test_a_moved_index_k_sink_fails_closed(replay_rig):
    md = replay_rig.md
    sink = types.SimpleNamespace(t=torch.zeros(8, dtype=torch.uint8))
    md.kv_cache_manager = types.SimpleNamespace(get_nvfp4_index_k_sink=lambda: sink.t)
    ptrs = modeling._a1_bcg_ptrs(md, 3)
    assert ptrs[-1] == sink.t.data_ptr() and -1 not in ptrs
    replay_rig.guard.reset()
    replay_rig.guard.note(3, ptrs)
    replay_rig.replay()
    old = sink.t
    sink.t = torch.zeros(8, dtype=torch.uint8)  # the manager's sink is read live
    replay_rig.events.clear()
    with pytest.raises(RuntimeError, match="moved since capture"):
        replay_rig.replay()
    assert replay_rig.events == []
    del old


def test_the_graph_release_hook_resets_the_guard(capturing):
    """Phase 1 (KV-cache estimation) captures on its pool, the engine releases the graphs and rebuilds the pool and
    the metadata, phase 2 captures again: its note must not compare against phase 1's released addresses."""
    model = _model()
    guard = model._bcg_ptrs
    hooks = _TARGET_CLASS.runtime_hooks
    assert hooks.prefill_graphs_released is modeling._bcg_graphs_released
    phase1, phase2 = _a1_metadata(), _a1_metadata()
    for md in (phase1, phase2):
        attn = _sparse_attn([("q", "iq")], [], guard)
        if md is phase2:
            hooks.prefill_graphs_released(model)
            assert not guard.armed
        modeling.MiniMaxM3Attention._sparse_forward(attn, "pos", torch.zeros(8, 4), md)
        assert guard._noted[3] == modeling._a1_bcg_ptrs(md, 3)


def test_without_the_release_a_recapture_on_a_new_pool_raises(capturing):
    guard = bcg_ptr_guard.StaticPointerGuard("t")
    # Both phases stay live until the assertion, so the allocator cannot hand phase 1's buffers to phase 2.
    phase1, phase2 = _a1_metadata(), _a1_metadata()
    assert modeling._a1_bcg_ptrs(phase1, 3) != modeling._a1_bcg_ptrs(phase2, 3)
    modeling.MiniMaxM3Attention._sparse_forward(
        _sparse_attn([("q", "iq")], [], guard), "pos", torch.zeros(8, 4), phase1
    )
    with pytest.raises(RuntimeError, match="moved between captures"):
        modeling.MiniMaxM3Attention._sparse_forward(
            _sparse_attn([("q", "iq")], [], guard), "pos", torch.zeros(8, 4), phase2
        )
    assert phase1 is not phase2


def test_the_runtime_hooks_extend_the_breakable_capture_list():
    """The engine's capture list takes the target's extra prefill buckets up to max_num_tokens; a list that does not
    reach 4096 stays as it is."""
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import apply_prefill_capture_hooks

    hooks = _TARGET_CLASS.runtime_hooks
    if not modeling._BCG_LARGE:
        pytest.skip("DEP2CTX_BCG_LARGE is switched off in this environment")
    stock = list(range(256, 4097, 256)) + [16384]
    full = apply_prefill_capture_hooks(stock, hooks, max_num_tokens=16384, breakable=True)[0]
    small = apply_prefill_capture_hooks([256, 512], hooks, max_num_tokens=16384, breakable=True)[0]
    assert list(hooks.prefill_capture_extra_num_tokens) == list(ppr.DEP2CTX_BCG_GRID)
    assert hooks.prefill_capture_max_num_tokens is None
    assert full[-1] == 16384 and len(full) == 16 + len(ppr.DEP2CTX_BCG_GRID)
    assert small == [256, 512]
