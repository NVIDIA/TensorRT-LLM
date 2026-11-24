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
"""``CuteDslFc12FusedMoE`` on an output and counters zeroed ahead of the call.

With ``TRTLLM_K3_FC12_PREZERO`` the Kimi-K3 MoE runtime allocates the routed output and the
fused FC12 kernel's atomic counters where its latent chain forks and zeroes them on a stream of
their own (``new_zeroed_output`` / ``zeroed_output``); FC12's op then issues neither its counter
fill nor its output memset. The same kernel computes every routed row from the same values, so
outputs are bit-identical wherever FC12 reproduces itself: rows with at most two local experts
(``0 + a + b`` rounds the same in either order).
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.moe.test_cute_dsl_fc12_presorted import (
    HIDDEN,
    NUM_EXPERTS,
    SCALE,
    TILE,
    TOP_K,
    _assert_same_rows,
    _fc12_moe,
    _k3_module,
    _local_counts,
    _logits,
    _noaux_route,
    _quantized_input,
    _RankFc12,
    _tail_route,
    rubin_fc12,
)
from torch import nn

from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import fc12_counter_layout
from tensorrt_llm._torch.moe.fused_moe.cutedsl import fused_fc12 as fc12
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE
from tensorrt_llm._torch.utils import ActivationType

# (local_expert_offset, local_num_experts): EP8 rank 3, EP16 rank 7.
EP_LAYOUTS = [(3 * 112, 112), (7 * 56, 56)]


class _Event:
    """Stand-in for the zeroing stream's event: counts the joins."""

    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


# ---------------------------------------------------------------------------
# No GPU: counter layout, buffer layout, dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tiles", [1, 31, 32, 33, 112, 4096])
def test_counter_layout(num_tiles):
    """``fc1_ready`` first, each scheduler counter on its own 128-byte line after it."""
    lines = (num_tiles + 31) // 32
    assert fc12_counter_layout(num_tiles) == (32 * lines, 32 * lines + 32, 32 * lines + 64)


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 300])
@pytest.mark.parametrize("hidden", [3584, 6])
def test_new_zeroed_output_layout(num_tokens, hidden):
    """The output starts the buffer; the counters start on a 128-byte line after it and hold the
    counters of ``moe_sort``'s 128-row tile count."""
    num_local = 112
    backend = SimpleNamespace(expert_size_per_partition=num_local, hidden_size=hidden)
    ready = _Event()
    zeroed = CuteDslFc12FusedMoE.new_zeroed_output(
        backend, num_tokens, TOP_K, torch.device("cpu"), ready
    )
    assert zeroed.ready is ready
    assert zeroed.buffer.dtype == zeroed.counters.dtype == torch.int32
    assert zeroed.output.dtype == torch.bfloat16
    assert zeroed.output.shape == (num_tokens, hidden) and zeroed.output.is_contiguous()
    assert zeroed.output.data_ptr() == zeroed.buffer.data_ptr()
    offset = zeroed.counters.data_ptr() - zeroed.buffer.data_ptr()
    assert offset % 128 == 0 and offset >= num_tokens * hidden * 2
    assert zeroed.counters.data_ptr() + zeroed.counters.numel() * 4 == (
        zeroed.buffer.data_ptr() + zeroed.buffer.numel() * 4
    )
    expanded = num_tokens * TOP_K
    filled = min(num_local, expanded)
    max_tiles = filled + (expanded - filled) // TILE
    assert zeroed.counters.numel() == fc12_counter_layout(max_tiles)[2]


def _dispatch_backend(monkeypatch):
    """FC12 stand-in whose fused impl records the output and the zeroed counters it is given."""
    calls = []
    monkeypatch.setattr(
        fc12.AutoTuner,
        "get",
        staticmethod(lambda: SimpleNamespace(choose_one=lambda *args: (args[1][0], TILE))),
    )
    backend = SimpleNamespace(
        has_nvfp4=True,
        num_slots=NUM_EXPERTS,
        hidden_size=8,
        use_fused_finalize=True,
        _epilogue_identity=lambda: (int(ActivationType.SiTu), 4.0, 25.0),
    )

    def impl(*args, enable_alltoall=False, tile_size=TILE):
        calls.append((args[4], getattr(backend, "_zeroed_counters", "unset")))
        return args[4]

    backend.run_moe_nvfp4_impl = impl
    return backend, calls


def _run_dispatch(backend, num_tokens, moe_output=None):
    return CuteDslFc12FusedMoE.run_moe_nvfp4(
        backend,
        x=torch.empty(num_tokens, 4, dtype=torch.uint8),
        token_selected_experts=torch.zeros(num_tokens, TOP_K, dtype=torch.int32),
        token_final_scales=torch.ones(num_tokens, TOP_K),
        x_sf=torch.empty(0),
        moe_output=moe_output,
        weight_view=SimpleNamespace(expert_size_per_partition=112, slot_start=336),
    )


def _arm(backend, num_tokens, hidden):
    owner = SimpleNamespace(expert_size_per_partition=112, hidden_size=hidden)
    zeroed = CuteDslFc12FusedMoE.new_zeroed_output(
        owner, num_tokens, TOP_K, torch.device("cpu"), _Event()
    )
    backend.zeroed_output = zeroed
    backend._take_zeroed_output = CuteDslFc12FusedMoE._take_zeroed_output.__get__(backend)
    return zeroed


def test_dispatch_without_zeroed_output_is_unchanged(monkeypatch):
    """Switch off: the caller's output (or none) reaches the fused impl as before and no zeroed
    state is touched."""
    backend, calls = _dispatch_backend(monkeypatch)
    out = torch.empty(4, 8, dtype=torch.bfloat16)
    assert _run_dispatch(backend, 4, out) is out
    assert calls == [(out, "unset")]
    assert not hasattr(backend, "zeroed_output") and not hasattr(backend, "_zeroed_counters")


def test_dispatch_takes_zeroed_output(monkeypatch):
    """Armed: the call joins the zeroing stream once and runs into the zeroed output, whose
    counters wait for the fused op."""
    backend, calls = _dispatch_backend(monkeypatch)
    zeroed = _arm(backend, 4, 8)
    assert _run_dispatch(backend, 4) is zeroed.output
    assert zeroed.ready.waits == 1 and backend.zeroed_output is None
    ((output, state),) = calls
    assert output is zeroed.output
    assert state[0] is zeroed.output and state[1] is zeroed.counters


@pytest.mark.parametrize("case", ["caller_output", "other_rows"])
def test_dispatch_joins_unused_zeroed_output(monkeypatch, case):
    """A caller's output, or a token count other than the zeroed one: the zeroed buffer is joined
    and left unused."""
    backend, calls = _dispatch_backend(monkeypatch)
    zeroed = _arm(backend, 4, 8)
    out = torch.empty(4, 8, dtype=torch.bfloat16) if case == "caller_output" else None
    result = _run_dispatch(backend, 4 if out is not None else 5, out)
    assert zeroed.ready.waits == 1 and backend.zeroed_output is None
    assert result is not zeroed.output
    assert calls == [(result, None)]


def test_drop_zeroed_output_joins():
    backend = SimpleNamespace(zeroed_output=SimpleNamespace(ready=_Event()))
    ready = backend.zeroed_output.ready
    CuteDslFc12FusedMoE.drop_zeroed_output(backend)
    CuteDslFc12FusedMoE.drop_zeroed_output(backend)
    assert ready.waits == 1 and backend.zeroed_output is None


def _impl_backend():
    """Stand-in with the state ``run_moe_nvfp4_impl`` reads before the fused op."""
    return SimpleNamespace(
        num_slots=NUM_EXPERTS,
        use_fused_finalize=True,
        activation_type=ActivationType.SiTu,
        _situ_betas=lambda: (4.0, 25.0),
        act_clamp=float("inf"),
        fc2_input_scale=torch.ones(1),
        mapping=SimpleNamespace(moe_ep_size=8),
    )


def _run_impl(backend, monkeypatch, moe_output):
    tables = tuple(torch.zeros(1, dtype=torch.int32) for _ in range(6))
    monkeypatch.setattr(fc12, "sort_routing", lambda *args: tables)
    monkeypatch.delenv("TRTLLM_K3_ULL_EXPERTS", raising=False)
    op_calls = []
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_fc12_fused_rubin",
        lambda **kwargs: op_calls.append(kwargs),
        raising=False,
    )
    weights = torch.zeros(2, dtype=torch.uint8)
    weight_view = SimpleNamespace(
        w3_w1_weight=weights,
        fc1_weight_scale=weights,
        fc1_global_scale=torch.ones(1),
        w2_weight=weights,
        fc2_weight_scale=weights,
        fc2_global_scale=torch.ones(1),
        expert_size_per_partition=112,
        slot_start=336,
    )
    num_tokens = moe_output.size(0)
    CuteDslFc12FusedMoE.run_moe_nvfp4_impl(
        backend,
        torch.zeros(num_tokens, 4, dtype=torch.uint8),
        torch.zeros(num_tokens, TOP_K, dtype=torch.int32),
        torch.ones(num_tokens, TOP_K),
        torch.zeros(num_tokens, 1, dtype=torch.uint8),
        moe_output,
        weight_view,
    )
    (kwargs,) = op_calls
    return kwargs


def test_fused_op_kwargs_unchanged_without_zeroed_counters(monkeypatch):
    """Switch off: the fused op gets exactly the arguments it got before (no zeroed counters)."""
    out = torch.empty(4, 8, dtype=torch.bfloat16)
    kwargs = _run_impl(_impl_backend(), monkeypatch, out)
    assert "zeroed_counters" not in kwargs and kwargs["output"] is out


def test_fused_op_gets_zeroed_counters_once(monkeypatch):
    """The zeroed counters reach the fused op of the call on their output, once; a call on any
    other output (an autotuning profile) drops them."""
    backend = _impl_backend()
    out = torch.empty(4, 8, dtype=torch.bfloat16)
    counters = torch.zeros(96, dtype=torch.int32)
    backend._zeroed_counters = (out, counters)
    assert _run_impl(backend, monkeypatch, out)["zeroed_counters"] is counters
    assert backend._zeroed_counters is None
    assert "zeroed_counters" not in _run_impl(backend, monkeypatch, out)

    backend._zeroed_counters = (out, counters)
    other = torch.empty(4, 8, dtype=torch.bfloat16)
    assert "zeroed_counters" not in _run_impl(backend, monkeypatch, other)
    assert backend._zeroed_counters is None


# ---------------------------------------------------------------------------
# Rubin: bit-identical outputs, CUDA graph, KimiK3MoERuntime
# ---------------------------------------------------------------------------


def _few_local_routing(m, offset, num_local, seed):
    """Top-k ids with 0, 1 or 2 local experts per token (in turn) and positive weights, with
    this rank's ``moe_sort`` tables at tile 128 attached (so no tuner runs)."""
    gen = torch.Generator().manual_seed(seed)
    local = torch.arange(offset, offset + num_local)
    remote = torch.cat([torch.arange(0, offset), torch.arange(offset + num_local, NUM_EXPERTS)])
    rows = []
    for token in range(m):
        n_local = token % 3
        picks = local[torch.randperm(num_local, generator=gen)[:n_local]]
        rest = remote[torch.randperm(remote.numel(), generator=gen)[: TOP_K - n_local]]
        row = torch.cat([picks, rest])
        rows.append(row[torch.randperm(TOP_K, generator=gen)])
    ids = torch.stack(rows).to(torch.int32).cuda()
    weights = (torch.rand((m, TOP_K), generator=gen) + 0.1).cuda()
    tables = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=weights,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=num_local,
        tile_tokens_dim=TILE,
    )
    ids._k3_presorted = ((offset, num_local, TILE), tuple(tables))
    return ids, weights


def _rank_backend(offset, num_local, seed):
    backend = _RankFc12.build(offset, num_local, seed=seed)
    backend.expert_size_per_partition = num_local
    return backend


def _zero_beside(backend, m, side_stream, poison=True):
    """``new_zeroed_output`` on the current stream, poisoned with NaN there, zeroed on
    ``side_stream`` after a fork, and armed: the K3 runtime's sequence."""
    zeroed = backend.new_zeroed_output(m, TOP_K, torch.device("cuda"), torch.cuda.Event())
    if poison:
        zeroed.buffer.fill_(-1)  # 0xFFFF: BF16 NaN, counters -1
    fork = torch.cuda.Event()
    fork.record()
    with torch.cuda.stream(side_stream):
        fork.wait()
        zeroed.buffer.zero_()
        zeroed.ready.record()
    backend.zeroed_output = zeroed
    return zeroed


@pytest.fixture
def count_memsets(monkeypatch):
    """Counts FC12's in-op output memsets."""
    real = torch.ops.trtllm.moe_output_memset_inplace
    calls = []

    def counting(**kwargs):
        calls.append(kwargs["input"].data_ptr())
        return real(**kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "moe_output_memset_inplace", counting)
    return calls


@rubin_fc12
@pytest.mark.parametrize("offset,num_local", EP_LAYOUTS)
@pytest.mark.parametrize("m", [8, 32, 128])
def test_fc12_zeroed_output_bit_identical(count_memsets, m, offset, num_local):
    """FC12 into the zeroed output equals FC12 with its own fill and memset, bit for bit."""
    backend = _rank_backend(offset, num_local, seed=offset + m)
    x, x_sf = _quantized_input(m, seed=m)
    ids, weights = _few_local_routing(m, offset, num_local, seed=m + 1)

    expected = backend.run(x, x_sf, ids, weights)
    torch.cuda.synchronize()
    assert len(count_memsets) == 1

    side = torch.cuda.Stream()
    zeroed = _zero_beside(backend, m, side)
    actual = backend.run(x, x_sf, ids, weights)
    torch.cuda.synchronize()
    assert len(count_memsets) == 1, "FC12 zeroed the output although it came zeroed"
    assert actual.data_ptr() == zeroed.output.data_ptr()
    assert backend.zeroed_output is None and backend._zeroed_counters is None
    assert expected.abs().amax() > 0, "degenerate test: all-zero MoE output"
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@rubin_fc12
def test_fc12_zeroed_output_cuda_graph_replay(count_memsets):
    """Zeroing stream + routing tail + FC12 captured in one CUDA graph follow new logits on
    replay, equal to the eager path that zeroes in the op."""
    k3_route_tail = _k3_module("kernels.k3_route_tail")

    assert k3_route_tail.supports_device(torch.device("cuda"))
    m, offset, num_local = 16, 3 * 112, 112
    backend = _rank_backend(offset, num_local, seed=9)
    logits, bias = _logits(m, seed=90)
    x, x_sf = _quantized_input(m, seed=91)
    static_logits = logits.clone()
    side = torch.cuda.Stream()

    def step():
        _zero_beside(backend, m, side)
        return backend.run(x, x_sf, *_tail_route(static_logits, bias, offset, num_local))

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    assert count_memsets == []
    for seed in range(3):
        fresh, _ = _logits(m, seed=95 + seed)
        static_logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ref_ids, ref_weights = _noaux_route(static_logits, bias)
        expected = backend.run(x, x_sf, ref_ids, ref_weights, tile_size=TILE)
        _assert_same_rows(out, expected, _local_counts(ref_ids, offset, num_local))


@rubin_fc12
@pytest.mark.parametrize("m", [8, 64])
def test_moe_runtime_fc12_prezero(monkeypatch, count_memsets, m):
    """``TRTLLM_K3_FC12_PREZERO``: ``KimiK3MoERuntime`` with the latent, shared-expert and zeroing
    streams hands FC12 an output zeroed beside the routing (no in-op memset); the output equals
    the switch-off output."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    modeling_kimi_linear = _k3_module("text_model")
    _KimiK3RoutingMethod = modeling_kimi_linear._KimiK3RoutingMethod

    logits, bias = _logits(m, seed=120 + m)
    routing = _KimiK3RoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=SCALE,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    moe = _fc12_moe(routing)

    class _Identity(nn.Module):
        def forward(self, value, **kwargs):
            return value

    class _AllReduce(nn.Module):
        def forward(self, value, all_reduce_params=None):
            if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
                return value
            return value + 1.0

    class _SharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Module()
            self.down_proj.reduce_output = True
            self.down_proj.use_fused_gemm_allreduce = False
            self.down_proj.all_reduce = _AllReduce()

        def forward(self, value, final_all_reduce_params=None):
            return self.down_proj.all_reduce(value * 0.5, all_reduce_params=final_all_reduce_params)

    runtime = modeling_kimi_linear.KimiK3MoERuntime.__new__(modeling_kimi_linear.KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(
        compute_logits=lambda value: logits,
        e_score_correction_bias=bias,
        routed_scaling_factor=SCALE,
        num_experts=NUM_EXPERTS,
    )
    runtime.top_k = TOP_K
    runtime.num_experts = NUM_EXPERTS
    runtime.routed_experts = moe
    runtime.routed_expert_down_proj = _Identity()
    runtime.routed_expert_norm = _Identity()
    runtime.routed_expert_up_proj = _Identity()
    runtime.shared_experts = _SharedExperts()
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime.shared_expert_stream = torch.cuda.Stream()
    runtime.latent_down_stream = torch.cuda.Stream()
    for name in (
        "moe_main_event",
        "moe_shared_event",
        "moe_latent_fork_event",
        "moe_latent_ready_event",
        "moe_shared_gemm_event",
    ):
        setattr(runtime, name, torch.cuda.Event())
    runtime.moe_routed_event = None
    monkeypatch.setattr(runtime, "_can_fuse_router_down", lambda value: False)
    monkeypatch.setattr(runtime, "_routed_projection", lambda value, projection: value * 1.0)
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL", "1")
    monkeypatch.setattr(modeling_kimi_linear, "_K3_ROUTE_TAIL_ORDER", "0")

    real_new = moe.backend.new_zeroed_output
    zeroed = []

    def recording_new(*args, **kwargs):
        zeroed.append(real_new(*args, **kwargs))
        return zeroed[-1]

    monkeypatch.setattr(moe.backend, "new_zeroed_output", recording_new)

    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    outputs = {}
    for key, prezero in (("off", False), ("off again", False), ("on", True)):
        runtime.fc12_prezero_stream = torch.cuda.Stream() if prezero else None
        runtime.moe_prezero_event = torch.cuda.Event() if prezero else None
        count_memsets.clear()
        zeroed.clear()
        with torch.inference_mode(), with_multi_stream(True):
            outputs[key] = runtime(hidden)
        torch.cuda.synchronize()
        assert routing.deferred is None and moe.backend.zeroed_output is None
        assert len(count_memsets) == (0 if prezero else 1), f"{key}: memsets {count_memsets}"
        assert len(zeroed) == (1 if prezero else 0)

    ids, _ = _noaux_route(logits, bias)
    if torch.equal(outputs["off again"], outputs["off"]):
        assert torch.equal(outputs["on"], outputs["off"])
    else:
        _assert_same_rows(outputs["on"], outputs["off"], _local_counts(ids, 0, NUM_EXPERTS))
