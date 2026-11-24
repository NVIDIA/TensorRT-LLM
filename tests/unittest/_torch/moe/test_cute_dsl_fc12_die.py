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
"""Die-aware placement and tile claiming of the fused FC12 MoE kernel (Rubin R200, SM107).

``TRTLLM_K3_FC12_DIE_LAYOUT`` copies the expert weights into die-alternating memory (2 MiB chunks
alternating between the two dies, same bytes); ``TRTLLM_K3_FC12_DIE_SCHED`` then gives each FC12
phase one work counter per die so CTAs take the tiles whose weight rows sit on their own die.
Neither changes what a tile computes, only where its weights live and which CTA computes it:
outputs are bit-identical to the default wherever FC12 reproduces itself (rows with at most two
local experts, ``0 + a + b`` rounds the same in either order).
"""

import pytest
import torch
from _torch.moe.test_cute_dsl_fc12_presorted import (
    HIDDEN,
    NUM_EXPERTS,
    TILE,
    TOP_K,
    _assert_same_rows,
    _k3_module,
    _local_counts,
    _logits,
    _noaux_route,
    _quantized_input,
    _RankFc12,
    _tail_route,
    rubin_fc12,
)
from _torch.moe.test_cute_dsl_fc12_prezero import _few_local_routing, _rank_backend
from torch import nn

from tensorrt_llm._torch import die_locality
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops
from tensorrt_llm._torch.moe.fused_moe.cutedsl import fused_fc12 as fc12
from tensorrt_llm._torch.moe.fused_moe.cutedsl.kernel_inputs import NvFp4WeightView
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE
from tensorrt_llm._torch.utils import ActivationType

SCHED_ENV = "TRTLLM_K3_FC12_DIE_SCHED"
LAYOUT_ENV = "TRTLLM_K3_FC12_DIE_LAYOUT"
RESERVE_ENV = "TRTLLM_K3_FC12_SM_RESERVE"
LOOKAHEAD_ENV = "TRTLLM_K3_FC12_SCHED_LOOKAHEAD"
# (local_expert_offset, local_num_experts): EP8 rank 3, EP16 rank 7, EP4 rank 1.
EP_LAYOUTS = [(3 * 112, 112), (7 * 56, 56), (224, 224)]


# ---------------------------------------------------------------------------
# No GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tiles", [1, 32, 33, 4096])
def test_counter_layout_die_sched(num_tiles):
    """Die-local claiming: two scheduler lines per phase (die 0's, then die 1's)."""
    lines = (num_tiles + 31) // 32
    assert ops.fc12_counter_layout(num_tiles, die_sched=True) == (
        32 * lines,
        32 * lines + 64,
        32 * lines + 128,
    )
    assert ops.fc12_counter_layout(num_tiles) == (32 * lines, 32 * lines + 32, 32 * lines + 64)


def test_pack_sm_die_words():
    table = tuple((s >> 1) & 1 for s in range(200)) + (1, 0) * 6
    words = die_locality.pack_sm_die_words(table)
    assert len(words) == 7 and all(0 <= w < 1 << 32 for w in words)
    for smid, die in enumerate(table):
        assert (words[smid // 32] >> (smid % 32)) & 1 == die
    assert die_locality.pack_sm_die_words((0,) * 64) == (0, 0)


def test_die_sm_words_need_switch_and_placed_weights(monkeypatch):
    """No words (die-oblivious claiming) without the switch, for 2-CTA clusters, or for weights
    that are not die-alternating copies."""
    weight = torch.zeros(4, dtype=torch.uint8)
    monkeypatch.delenv(SCHED_ENV, raising=False)
    assert not ops.fc12_die_sched_requested()
    assert ops.fc12_die_sm_words(weight, weight, 1) is None
    monkeypatch.setenv(SCHED_ENV, "1")
    assert ops.fc12_die_sched_requested()
    assert ops.fc12_die_sm_words(weight, weight, 2) is None
    assert ops.fc12_die_sm_words(weight, weight, 1) is None


def test_die_sm_words_min_tokens(monkeypatch):
    """Calls below TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS keep die-oblivious claiming, before any
    weight or table lookup."""
    weight = torch.zeros(4, dtype=torch.uint8)
    monkeypatch.setenv(SCHED_ENV, "1")
    monkeypatch.setenv("TRTLLM_K3_FC12_DIE_SCHED_MIN_TOKENS", "24")
    monkeypatch.setattr(
        "tensorrt_llm._torch.die_locality.is_die_alternating",
        lambda tensor: pytest.fail("looked up the weights below the threshold"),
    )
    assert ops.fc12_die_sm_words(weight, weight, 1, num_tokens=8) is None
    assert ops.fc12_die_sm_words(weight, weight, 1, num_tokens=16) is None


@pytest.mark.parametrize("die_sched", [False, True])
def test_new_zeroed_output_counter_words(monkeypatch, die_sched):
    """A zeroed output armed while the switch is on carries the die-local counter lines."""
    if die_sched:
        monkeypatch.setenv(SCHED_ENV, "1")
    else:
        monkeypatch.delenv(SCHED_ENV, raising=False)
    owner = type("Owner", (), {"expert_size_per_partition": 112, "hidden_size": 8})()
    zeroed = CuteDslFc12FusedMoE.new_zeroed_output(owner, 4, TOP_K, torch.device("cpu"), None)
    max_tiles = 64
    assert zeroed.counters.numel() == ops.fc12_counter_layout(max_tiles, die_sched=die_sched)[2]


def test_kernel_rejects_die_sched_for_2cta():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.moe.kernels.rubin.rubin_contiguous_grouped_blockscaled_gemm_fused_fc12 import (  # noqa: E501
        Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel as Kernel,
    )

    common = dict(topk=TOP_K, scheduler="l2_atomic", die_sm_words=(0xFFFFFFFF, 1))
    kernel = Kernel(16, (128, 128, 128), (128, 128, 256), (1, 1), True, **common)
    assert kernel.use_die_scheduler and kernel.die_sm_words == (-1, 1)
    with pytest.raises(ValueError, match="1-CTA"):
        Kernel(16, (256, 128, 128), (256, 128, 256), (2, 1), True, **common)
    with pytest.raises(ValueError, match="l2_atomic"):
        Kernel(
            16,
            (128, 128, 128),
            (128, 128, 256),
            (1, 1),
            True,
            topk=TOP_K,
            scheduler="static",
            die_sm_words=(1,),
        )


# ---------------------------------------------------------------------------
# Rubin: placement, SM -> die table
# ---------------------------------------------------------------------------


def _two_dies() -> bool:
    return (
        torch.cuda.is_available()
        and die_locality.locality_domain_count(torch.cuda.current_device()) >= 2
    )


rubin_dies = pytest.mark.skipif(not _two_dies(), reason="needs a two-die GPU (Rubin R200)")


@rubin_dies
@pytest.mark.parametrize("nbytes", [1, 5 << 20, (7 << 20) + 3])
def test_die_alternating_copy(nbytes):
    """Same bytes; 4 MiB-aligned start; chunk i homed on die i % 2 as the driver reports it; the
    range is known while the copy lives and forgotten after."""
    source = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    copy = die_locality.die_alternating_copy(source)
    assert copy is not None and copy.shape == source.shape and copy.dtype == source.dtype
    assert torch.equal(copy, source)
    base = copy.data_ptr()
    assert base % (4 << 20) == 0 and die_locality.is_die_alternating(copy)
    assert not die_locality.is_die_alternating(source)
    chunks = -(-nbytes // die_locality.DIE_CHUNK_BYTES)
    assert die_locality.die_alternating_allocations(copy) in (min(2, chunks), chunks)
    assert die_locality.die_alternating_allocations(source) == 0
    for chunk in range(chunks):
        die = die_locality._chunk_die(base + chunk * die_locality.DIE_CHUNK_BYTES)
        assert die in (None, chunk % 2)
    del copy
    import gc

    gc.collect()
    assert base not in die_locality._ALTERNATING_RANGES


@rubin_dies
def test_die_alternating_copy_keeps_shape_and_dtype():
    source = torch.randn((3, 512, 1000), device="cuda").to(torch.bfloat16)
    copy = die_locality.die_alternating_copy(source[:, :, 7:])
    assert copy.is_contiguous() and torch.equal(copy, source[:, :, 7:])


@rubin_dies
@pytest.mark.parametrize(
    "nbytes,split",
    [
        (1, 0),
        (5 << 20, 3 << 20),
        ((7 << 20) + 3, 1 << 20),
        (9 << 20, 9 << 20),
        (6 << 20, 0),
        # One K3 FC1 expert (10.5 MiB) split at its half: the boundary moves to 6 MiB.
        (11010048, 5505024),
    ],
)
def test_die_split_copy(nbytes, split):
    """Same bytes; 4 MiB-aligned start; the chunks below the boundary nearest to the requested
    split on die 0 and the rest on die 1 as the driver reports it; one allocation per die that
    holds bytes; the range is known while the copy lives and forgotten after."""
    chunk = die_locality.DIE_CHUNK_BYTES
    source = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    copy = die_locality.die_split_copy(source, split)
    assert copy is not None and copy.shape == source.shape and torch.equal(copy, source)
    base = copy.data_ptr()
    chunks = -(-nbytes // chunk)
    die0_chunks = min((split + chunk // 2) // chunk, chunks)
    assert base % (4 << 20) == 0
    assert die_locality.die_split_request(copy) == (split, die0_chunks * chunk)
    assert die_locality.die_split_request(source) is None
    assert die_locality.die_split_allocations(copy) == (die0_chunks > 0) + (chunks > die0_chunks)
    for index in range(chunks):
        die = die_locality._chunk_die(base + index * chunk)
        assert die in (None, 0 if index < die0_chunks else 1)
    del copy
    import gc

    gc.collect()
    assert base not in die_locality._SPLIT_RANGES


@rubin_dies
def test_die_split_copy_keeps_shape_and_dtype():
    source = torch.randn((3, 512, 1000), device="cuda").to(torch.bfloat16)
    copy = die_locality.die_split_copy(source[:, :, 7:], 1 << 20)
    assert copy.is_contiguous() and torch.equal(copy, source[:, :, 7:])


@rubin_dies
def test_sm_die_table():
    """Every SM on a die, at least 40% of them on each; measured once per device."""
    device = torch.cuda.current_device()
    table = die_locality.sm_die_table(device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert table is not None and len(table) >= sms and set(table) <= {0, 1}
    assert min(table.count(0), table.count(1)) >= 0.4 * sms
    assert die_locality.sm_die_table(device) is table


# ---------------------------------------------------------------------------
# Rubin: FC12 outputs with placed weights and die-local claiming
# ---------------------------------------------------------------------------


def _placed(view: NvFp4WeightView) -> NvFp4WeightView:
    w1 = die_locality.die_alternating_copy(view.w3_w1_weight)
    w2 = die_locality.die_alternating_copy(view.w2_weight)
    return NvFp4WeightView(
        w3_w1_weight=w1,
        fc1_weight_scale=view.fc1_weight_scale,
        fc1_global_scale=view.fc1_global_scale,
        w2_weight=w2,
        fc2_weight_scale=view.fc2_weight_scale,
        fc2_global_scale=view.fc2_global_scale,
        expert_size_per_partition=view.expert_size_per_partition,
        slot_start=view.slot_start,
    )


def _run_view(backend, view, x, x_sf, ids, weights):
    original, backend.rank_weights = backend.rank_weights, view
    try:
        out = backend.run(x, x_sf, ids, weights)
        torch.cuda.synchronize()
        return out
    finally:
        backend.rank_weights = original


def _die_keys():
    """Kernel cache keys of die-local FC12 kernels (they end with the SM -> die words)."""
    return {
        key
        for key in ops.Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner.kernel_cache
        if len(key) > 11 and isinstance(key[11], tuple) and isinstance(key[11][0], int)
    }


def _skewed_routing(m, offset, num_local, hot, seed):
    """Like ``_few_local_routing`` (0, 1 or 2 local experts per token), but the local experts are
    drawn from the first ``hot`` ones: several 128-row tiles per expert, each with its tail."""
    gen = torch.Generator().manual_seed(seed)
    local = torch.arange(offset, offset + hot)
    remote = torch.cat([torch.arange(0, offset), torch.arange(offset + num_local, NUM_EXPERTS)])
    rows = []
    for token in range(m):
        n_local = token % 3
        picks = local[torch.randperm(hot, generator=gen)[:n_local]]
        rest = remote[torch.randperm(remote.numel(), generator=gen)[: TOP_K - picks.numel()]]
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


@rubin_fc12
@rubin_dies
@pytest.mark.parametrize("variant", ["layout", "die_sched"])
@pytest.mark.parametrize("offset,num_local", EP_LAYOUTS)
@pytest.mark.parametrize("m", [2, 8, 33, 64, 128, 200])
def test_fc12_die_bit_identical(monkeypatch, m, offset, num_local, variant):
    """Placed weights, with the default or the die-local scheduler, give the default output."""
    backend = _rank_backend(offset, num_local, seed=offset + m)
    x, x_sf = _quantized_input(m, seed=m + 7)
    ids, weights = _few_local_routing(m, offset, num_local, seed=m + 8)
    monkeypatch.delenv(SCHED_ENV, raising=False)
    expected = _run_view(backend, backend.rank_weights, x, x_sf, ids, weights)
    placed = _placed(backend.rank_weights)
    if variant == "die_sched":
        monkeypatch.setenv(SCHED_ENV, "1")
    before = _die_keys()
    actual = _run_view(backend, placed, x, x_sf, ids, weights)
    if variant == "die_sched":
        assert _die_keys(), "no die-local FC12 kernel was compiled"
    else:
        assert _die_keys() == before
    assert expected.abs().amax() > 0, "degenerate test: all-zero MoE output"
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@rubin_fc12
@rubin_dies
@pytest.mark.parametrize("offset,num_local", [(3 * 112, 112), (7 * 56, 56)])
@pytest.mark.parametrize("m,hot", [(300, 3), (700, 5), (96, 1)])
def test_fc12_die_sched_uneven_experts(monkeypatch, m, hot, offset, num_local):
    """Few hot experts with several M tiles each (and partial tail tiles): bit-identical."""
    backend = _rank_backend(offset, num_local, seed=hot + m)
    x, x_sf = _quantized_input(m, seed=m + 11)
    ids, weights = _skewed_routing(m, offset, num_local, hot, seed=m + hot)
    monkeypatch.delenv(SCHED_ENV, raising=False)
    expected = _run_view(backend, backend.rank_weights, x, x_sf, ids, weights)
    monkeypatch.setenv(SCHED_ENV, "1")
    actual = _run_view(backend, _placed(backend.rank_weights), x, x_sf, ids, weights)
    assert expected.abs().amax() > 0
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def _op(view, x, x_sf, weights, tables, out, mma_n):
    t2e, t2mn, e2p, p2e, _total, nonexit = tables
    torch.ops.trtllm.cute_dsl_nvfp4_fc12_fused_rubin(
        input=x.view(torch.float4_e2m1fn_x2),
        fc1_weight=view.w3_w1_weight.view(torch.float4_e2m1fn_x2),
        input_scale=x_sf.view(torch.uint8),
        fc1_weight_scale=view.fc1_weight_scale.view(torch.uint8),
        fc1_alpha=view.fc1_global_scale,
        tile_idx_to_group_idx=t2e,
        tile_idx_to_mn_limit=t2mn,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=nonexit,
        global_sf=torch.tensor([4.0], device="cuda"),
        fc2_weight=view.w2_weight.view(torch.float4_e2m1fn_x2),
        fc2_weight_scale=view.fc2_weight_scale.view(torch.uint8),
        fc2_alpha=view.fc2_global_scale,
        output=out,
        token_final_scales=weights,
        expanded_idx_to_permuted_idx=e2p,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_local_experts=view.expert_size_per_partition,
        local_expert_offset=view.slot_start,
        tile_size=TILE,
        swiglu_limit=float("inf"),
        ep_size=NUM_EXPERTS // view.expert_size_per_partition,
        enable_alltoall=False,
        scaling_vector_size=16,
        precomputed_tactic=str(((TILE, mma_n, 256), (TILE, mma_n, 128), (1, 1))),
        activation_type=int(ActivationType.SiTu),
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    return out


@rubin_fc12
@rubin_dies
@pytest.mark.parametrize("mma_n", [128, 256])
@pytest.mark.parametrize("reserve", [None, "100"])
@pytest.mark.parametrize("m", [8, 128])
def test_fc12_die_sched_tactics_and_reserve(monkeypatch, m, reserve, mma_n):
    """Both MMA N tactics, the full grid and a grid reduced by ``TRTLLM_K3_FC12_SM_RESERVE``."""
    offset, num_local = 3 * 112, 112
    backend = _rank_backend(offset, num_local, seed=m + mma_n)
    x, x_sf = _quantized_input(m, seed=m + 21)
    ids, weights = _few_local_routing(m, offset, num_local, seed=m + 22)
    tables = ids._k3_presorted[1]
    view = backend.rank_weights
    monkeypatch.delenv(SCHED_ENV, raising=False)
    monkeypatch.delenv(RESERVE_ENV, raising=False)
    expected = _op(
        view,
        x,
        x_sf,
        weights,
        tables,
        torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda"),
        mma_n,
    )
    monkeypatch.setenv(SCHED_ENV, "1")
    if reserve is not None:
        monkeypatch.setenv(RESERVE_ENV, reserve)
    actual = _op(
        _placed(view),
        x,
        x_sf,
        weights,
        tables,
        torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda"),
        mma_n,
    )
    torch.cuda.synchronize()
    assert expected.abs().amax() > 0
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@rubin_fc12
@rubin_dies
@pytest.mark.parametrize("die_sched", [False, True])
@pytest.mark.parametrize("m", [8, 64, 300])
def test_fc12_sched_lookahead(monkeypatch, m, die_sched):
    """``TRTLLM_K3_FC12_SCHED_LOOKAHEAD=2`` (alone or with die-local claiming): bit-identical."""
    offset, num_local = 3 * 112, 112
    backend = _rank_backend(offset, num_local, seed=m + 51)
    x, x_sf = _quantized_input(m, seed=m + 52)
    ids, weights = _few_local_routing(m, offset, num_local, seed=m + 53)
    monkeypatch.delenv(SCHED_ENV, raising=False)
    monkeypatch.delenv(LOOKAHEAD_ENV, raising=False)
    expected = _run_view(backend, backend.rank_weights, x, x_sf, ids, weights)
    monkeypatch.setenv(LOOKAHEAD_ENV, "2")
    view = backend.rank_weights
    if die_sched:
        monkeypatch.setenv(SCHED_ENV, "1")
        view = _placed(view)
    actual = _run_view(backend, view, x, x_sf, ids, weights)
    cache = ops.Sm107BlockScaledContiguousGroupedGemmFusedFc12Runner.kernel_cache
    assert any(("sched_lookahead", 2) in key for key in cache)
    assert expected.abs().amax() > 0
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_sched_lookahead_env(monkeypatch):
    monkeypatch.delenv(LOOKAHEAD_ENV, raising=False)
    assert ops.fc12_sched_lookahead(1) == 1
    monkeypatch.setenv(LOOKAHEAD_ENV, "2")
    assert ops.fc12_sched_lookahead(1) == 2 and ops.fc12_sched_lookahead(2) == 1
    monkeypatch.setenv(LOOKAHEAD_ENV, "3")
    with pytest.raises(ValueError, match=LOOKAHEAD_ENV):
        ops.fc12_sched_lookahead(1)


@rubin_fc12
@rubin_dies
def test_fc12_die_sched_cuda_graph_replay(monkeypatch):
    """Routing tail + die-local FC12 on placed weights in one CUDA graph: replays follow new
    logits and equal the default eager path."""
    k3_route_tail = _k3_module("kernels.k3_route_tail")

    assert k3_route_tail.supports_device(torch.device("cuda"))
    m, offset, num_local = 32, 3 * 112, 112
    backend = _rank_backend(offset, num_local, seed=31)
    default_view = backend.rank_weights
    logits, bias = _logits(m, seed=310)
    x, x_sf = _quantized_input(m, seed=311)
    static_logits = logits.clone()
    monkeypatch.setenv(SCHED_ENV, "1")
    backend.rank_weights = _placed(default_view)
    backend.run(x, x_sf, *_tail_route(static_logits, bias, offset, num_local))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = backend.run(x, x_sf, *_tail_route(static_logits, bias, offset, num_local))
    assert _die_keys()
    monkeypatch.delenv(SCHED_ENV)
    for seed in range(3):
        fresh, _ = _logits(m, seed=315 + seed)
        static_logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        ref_ids, ref_weights = _noaux_route(static_logits, bias)
        expected = _run_view(backend, default_view, x, x_sf, ref_ids, ref_weights)
        _assert_same_rows(out, expected, _local_counts(ref_ids, offset, num_local))


@rubin_fc12
@rubin_dies
def test_place_expert_weights_per_die(monkeypatch):
    """``place_expert_weights_per_die`` swaps both expert weights for die-alternating copies with
    the same bytes, releases the originals, and FC12 on them gives the same output."""
    offset, num_local, m = 3 * 112, 112, 64
    backend = _RankFc12.build(offset, num_local, seed=41)
    view = backend.rank_weights
    backend.w3_w1_weight = nn.Parameter(view.w3_w1_weight.clone(), requires_grad=False)
    backend.w2_weight = nn.Parameter(view.w2_weight.clone(), requires_grad=False)
    backend._using_load_balancer = lambda: False
    x, x_sf = _quantized_input(m, seed=42)
    ids, weights = _few_local_routing(m, offset, num_local, seed=43)
    monkeypatch.delenv(SCHED_ENV, raising=False)
    expected = _run_view(backend, view, x, x_sf, ids, weights)
    w1_ref, w2_ref = view.w3_w1_weight, view.w2_weight
    backend.rank_weights = None
    del view
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    monkeypatch.setenv(SCHED_ENV, "1")
    assert CuteDslFc12FusedMoE.place_expert_weights_per_die(backend)
    released = allocated - torch.cuda.memory_allocated()
    assert released >= w1_ref.numel() + w2_ref.numel()
    assert die_locality.is_die_alternating(backend.w3_w1_weight)
    assert die_locality.is_die_alternating(backend.w2_weight)
    assert isinstance(backend.w3_w1_weight, nn.Parameter)
    assert torch.equal(backend.w3_w1_weight, w1_ref) and torch.equal(backend.w2_weight, w2_ref)
    built = _RankFc12.build(offset, num_local, seed=41).rank_weights
    placed_view = NvFp4WeightView(
        w3_w1_weight=backend.w3_w1_weight,
        fc1_weight_scale=built.fc1_weight_scale,
        fc1_global_scale=built.fc1_global_scale,
        w2_weight=backend.w2_weight,
        fc2_weight_scale=built.fc2_weight_scale,
        fc2_global_scale=built.fc2_global_scale,
        expert_size_per_partition=num_local,
        slot_start=offset,
    )
    backend.rank_weights = placed_view
    actual = backend.run(x, x_sf, ids, weights)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_post_load_weights_places_only_with_switch(monkeypatch):
    """The base weight post-processing always runs; the per-die placement only with the
    switch."""
    calls = []
    backend = CuteDslFc12FusedMoE.__new__(CuteDslFc12FusedMoE)
    nn.Module.__init__(backend)
    monkeypatch.setattr(fc12.MoEImplBase, "post_load_weights", lambda self: calls.append("parent"))
    monkeypatch.setattr(
        CuteDslFc12FusedMoE, "place_expert_weights_per_die", lambda self: calls.append("place")
    )
    monkeypatch.delenv(LAYOUT_ENV, raising=False)
    monkeypatch.delenv("TRTLLM_K3_ULL_EXPERTS_DIE_MAJOR", raising=False)
    backend.post_load_weights()
    assert calls == ["parent"]
    monkeypatch.setenv(LAYOUT_ENV, "1")
    backend.post_load_weights()
    assert calls == ["parent", "parent", "place"]
