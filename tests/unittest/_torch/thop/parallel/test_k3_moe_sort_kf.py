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
"""``trtllm::kimi_k3_moe_sort_kf`` must produce the layout of ``trtllm::moe_sort`` for Kimi K3's EP16 rank.

The generated sort (one cluster launch) takes ``token_selected_experts`` [T, 16] of a rank with 56 local
experts out of 896 and returns the six tensors of ``moe_sort``: tile tables, the expanded <-> permuted row maps and
the two scalars. Everything that a consumer reads must equal ``moe_sort``'s (shapes, dtypes, tile tables, scalars, the
set of rows of every expert, -1 on non-local entries, the inverse maps); the order of the rows inside an expert is
deterministic but free. Checked on dispatch-like and adversarial ids (hot expert, all local, none local, ids at the
edges of the local range and wild ones), on every multiple of 16 from 16 to 8192 rows (the dispatch buffer of an EP16
rank has 16 x tokens-per-source-rank rows), under CUDA graph replay, on misaligned and strided inputs, through
``moe_permute`` + ``moe_unpermute_inplace`` against the same pipeline fed by ``moe_sort``, and through the
``TRTLLM_K3_MOE_SORT_KF`` gate of ``CuteDslFusedMoE``. The memory the op's outputs are about to take is poisoned with -7
before every call, so that an output the kernel leaves unwritten fails instead of passing by luck, and a producer that
triggers its dependents at its first instruction and writes the ids late checks that the kernel really waits
(``griddepcontrol.wait``) for the ids it sorts. The kernel has only been run on SM107 (Rubin): the op and the gate
refuse every other SM, and these tests need that GPU.

Tests that expect a ``RuntimeError`` from the op need C++ exceptions to propagate: in an image where libcurand.so.11
exports its own ``__gxx_personality_v0`` a thrown ``TORCH_CHECK`` aborts the process instead (rc 134); run those with
``LD_PRELOAD=libstdc++.so.6``.
"""

import os
import subprocess
import sys
import time
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 (Rubin) GPU: the kernel has only been run there",
)

NUM_EXPERTS = 896
TOP_K = 16
LOCAL = 56
TILE = 128
MAX_TOKENS = 8192
# multiples of 16: the shapes of the DEP16 decode buckets (768 .. 6144) have their own compile-time instantiations, the
# others take the generic launch path (1, 2, 4 or 8 CTAs)
TS = [16, 32, 48, 64, 96, 112, 128, 144, 192, 256, 496, 512, 768, 1008, 1024, 1536, 2048, 2064]
TS += [2560, 3072, 3584, 4096, 5008, 5120, 6144, 7168, 8176, 8192]
MODES = ["uniform", "dispatch", "hot", "all_local", "no_local", "edges", "wild"]
NAMES = [
    "tile_idx_to_expert_idx",
    "tile_idx_to_mn_limit",
    "expanded_idx_to_permuted_idx",
    "permuted_idx_to_expanded_idx",
    "total_num_padded_tokens",
    "num_non_exiting_tiles",
]


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)


def _max_tiles(t):
    """getMaxNumCtasInBatchDim(t, 16, 56, 128): a tile per expert while entries last, then one per 128 spare ones."""
    entries = TOP_K * t
    filled = min(LOCAL, entries)
    return filled + (entries - filled) // TILE


def _poison_allocator(t):
    """Leave -7 in the blocks that the op's six outputs for ``t`` rows are about to take from the caching allocator
    (same sizes, so best fit hands them back): an output that the kernel does not write shows up as -7 and not as a
    lucky zero."""
    nm = _max_tiles(t)
    junk = [
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in ((nm,), (nm,), (t, TOP_K), (nm * TILE,), (1,), (1,))
    ]
    del junk


def _kf(ids, scales, offset):
    """The op on K3's EP16 shape, with its output memory poisoned first."""
    _poison_allocator(ids.shape[0])
    return torch.ops.trtllm.kimi_k3_moe_sort_kf(
        ids, scales, NUM_EXPERTS, TOP_K, offset, LOCAL, TILE
    )


def _python(code, **env):
    return subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, **env),
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.fixture(scope="session")
def cpp_exceptions_propagate():
    """Whether a C++ ``TORCH_CHECK`` thrown by a thop op arrives as a ``RuntimeError``.

    Where libcurand.so.11 exports its own ``__gxx_personality_v0`` the unwinder lands in it and the process aborts
    (SIGABRT, rc 134) instead, so the probe runs in a subprocess.
    """
    probe = (
        "import torch, tensorrt_llm\n"
        "ids = torch.zeros((8, 16), dtype=torch.int32, device='cuda')\n"  # 8 rows: not a multiple of 16
        "try:\n"
        "    torch.ops.trtllm.kimi_k3_moe_sort_kf(ids, ids.float(), 896, 16, 56, 56, 128)\n"
        "except RuntimeError:\n"
        "    print('RAISED')\n"
    )
    result = _python(probe)
    return result.returncode == 0 and "RAISED" in result.stdout


def _inputs(t, mode, seed, offset):
    """int32 ids [t, 16] on the GPU."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 7919 + t)
    lo = offset
    if mode == "uniform":
        ids = torch.randint(-1, NUM_EXPERTS, (t, TOP_K), device="cuda", generator=gen)
    elif mode == "dispatch":
        # What the EP16 dispatch hands a rank: 16 distinct experts per token, kept rows packed at the head of every
        # source rank's segment, the rest -1.
        assert t % 16 == 0
        src = 16
        m = t // src
        score = torch.rand((src, m, NUM_EXPERTS), device="cuda", generator=gen)
        ids = score.topk(TOP_K, dim=-1).indices.to(torch.int32)
        valid = ((ids >= lo) & (ids < lo + LOCAL)).any(-1)
        order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)
        ids = torch.gather(ids, 1, order.unsqueeze(-1).expand(-1, -1, TOP_K))
        keep = torch.arange(m, device="cuda").unsqueeze(0) < valid.sum(1, keepdim=True)
        ids = torch.where(keep.unsqueeze(-1), ids, torch.full_like(ids, -1)).reshape(t, TOP_K)
    elif mode == "hot":
        ids = torch.randint(-1, NUM_EXPERTS, (t, TOP_K), device="cuda", generator=gen)
        ids[:, 3] = lo + 17  # every row sends one entry to the same expert
    elif mode == "all_local":
        ids = torch.randint(lo, lo + LOCAL, (t, TOP_K), device="cuda", generator=gen)
    elif mode == "no_local":
        ids = torch.randint(0, NUM_EXPERTS - LOCAL, (t, TOP_K), device="cuda", generator=gen)
        ids = torch.where((ids >= lo) & (ids < lo + LOCAL), ids + LOCAL, ids)
        ids[:, 0] = -1
    elif mode == "edges":
        edge = torch.tensor(
            [lo - 1, lo, lo + 1, lo + LOCAL - 1, lo + LOCAL, -1, 0, NUM_EXPERTS - 1], device="cuda"
        )
        ids = edge[torch.randint(0, edge.numel(), (t, TOP_K), device="cuda", generator=gen)]
    else:  # wild: far outside any expert range
        wild = torch.tensor(
            [-2, -100, 2**31 - 1, -(2**31), NUM_EXPERTS, 5000, lo, lo + 1], device="cuda"
        )
        ids = wild[torch.randint(0, wild.numel(), (t, TOP_K), device="cuda", generator=gen)]
    return ids.to(torch.int32).contiguous()


def _expected(ids, offset):
    """CPU facts of ``moe_sort``'s layout: counts, tile bases, tile tables and the row range of every expert."""
    flat = ids.reshape(-1).cpu().long()
    local = flat - offset
    is_local = (local >= 0) & (local < LOCAL)
    counts = torch.zeros(LOCAL, dtype=torch.long).scatter_add_(
        0, local[is_local], torch.ones_like(local[is_local])
    )
    tiles = (counts + TILE - 1) // TILE
    base = torch.cumsum(tiles, 0) - tiles
    ntiles = int(tiles.sum())
    expert_of_tile = torch.repeat_interleave(torch.arange(LOCAL), tiles)
    start = base[expert_of_tile] * TILE
    tile_in_expert = torch.arange(ntiles) - base[expert_of_tile]
    mn_limit = torch.minimum(start + (tile_in_expert + 1) * TILE, start + counts[expert_of_tile])
    return {
        "is_local": is_local,
        "local": local,
        "counts": counts,
        "base": base,
        "ntiles": ntiles,
        "tile_expert": expert_of_tile,
        "mn_limit": mn_limit,
    }


def _check_layout(outs, ids, offset, what):
    """The consumer-visible contract of the six outputs against the CPU reference."""
    t = ids.shape[0]
    got = dict(zip(NAMES, [o.cpu() for o in outs]))
    exp = _expected(ids, offset)
    nt, nm = exp["ntiles"], _max_tiles(t)
    assert got["tile_idx_to_expert_idx"].shape == (nm,) and got["tile_idx_to_mn_limit"].shape == (
        nm,
    ), what
    assert got["expanded_idx_to_permuted_idx"].shape == (t, TOP_K), what
    assert got["permuted_idx_to_expanded_idx"].shape == (nm * TILE,), what
    assert all(g.dtype == torch.int32 for g in got.values()), what
    assert int(got["num_non_exiting_tiles"]) == nt, f"{what}: num_non_exiting_tiles"
    assert int(got["total_num_padded_tokens"]) == nt * TILE, f"{what}: total_num_padded_tokens"
    assert torch.equal(got["tile_idx_to_expert_idx"][:nt].long(), exp["tile_expert"]), (
        f"{what}: tile_idx_to_expert_idx"
    )
    assert torch.equal(got["tile_idx_to_mn_limit"][:nt].long(), exp["mn_limit"]), (
        f"{what}: tile_idx_to_mn_limit"
    )
    e2p = got["expanded_idx_to_permuted_idx"].reshape(-1).long()
    p2e = got["permuted_idx_to_expanded_idx"].long()
    is_local = exp["is_local"]
    assert bool((e2p[~is_local] == -1).all()), f"{what}: non-local entries must be -1"
    rows = e2p[is_local]
    assert rows.numel() == int(is_local.sum())
    if rows.numel():
        lo_row = exp["base"][exp["local"][is_local]] * TILE
        hi_row = lo_row + exp["counts"][exp["local"][is_local]]
        assert bool(((rows >= lo_row) & (rows < hi_row)).all()), (
            f"{what}: a row left its expert's range"
        )
        assert torch.unique(rows).numel() == rows.numel(), f"{what}: rows are not distinct"
        assert torch.equal(p2e[rows], torch.nonzero(is_local).reshape(-1)), (
            f"{what}: p2e is not the inverse of e2p"
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("t", TS)
def test_layout_matches_contract(t, mode):
    offset = 56 * ((t // 16) % 16)
    for seed in (0, 1):
        ids = _inputs(t, mode, seed, offset)
        scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")
        outs = _kf(ids, scales, offset)
        torch.cuda.synchronize()
        _check_layout(outs, ids, offset, f"t={t} mode={mode} seed={seed}")


def test_layout_matches_contract_on_every_multiple_of_16():
    """Every row count the op accepts (16, 32, ..., 8192): the launch configuration (cluster size, threads, vectors per
    thread) changes with T, so each one is checked on a uniform and a hot-expert input."""
    for t in range(16, MAX_TOKENS + 1, 16):
        offset = 56 * ((t // 16) % 16)
        for mode in ("uniform", "hot"):
            ids = _inputs(t, mode, 2, offset)
            scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")
            outs = _kf(ids, scales, offset)
            _check_layout(outs, ids, offset, f"t={t} mode={mode}")


@pytest.mark.parametrize("t", [16, 768, 2048, 3072, 4096, 6144, 8192])
@pytest.mark.parametrize("mode", ["dispatch", "uniform", "hot"])
def test_equals_moe_sort_where_it_matters(t, mode):
    """Shapes, dtypes, tile tables and scalars are moe_sort's; the sets of rows per expert coincide."""
    offset = 56 * ((t // 16) % 16)
    ids = _inputs(t, mode, 3, offset)
    scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")
    kw = dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=LOCAL,
        tile_tokens_dim=TILE,
    )
    ref = torch.ops.trtllm.moe_sort(token_selected_experts=ids, token_final_scales=scales, **kw)
    _poison_allocator(t)
    kf = torch.ops.trtllm.kimi_k3_moe_sort_kf(
        token_selected_experts=ids, token_final_scales=scales, **kw
    )
    nt = int(ref[5])
    for name, a, b in zip(NAMES, ref, kf):
        assert a.shape == b.shape and a.dtype == b.dtype, name
    assert int(kf[5]) == nt and int(kf[4]) == int(ref[4])
    assert torch.equal(kf[0][:nt], ref[0][:nt]) and torch.equal(kf[1][:nt], ref[1][:nt])
    # -1 pattern of the expanded -> permuted map and, per expert, the set of expanded indices in its rows.
    e2p_ref, e2p_kf = ref[2].reshape(-1), kf[2].reshape(-1)
    assert torch.equal(e2p_ref < 0, e2p_kf < 0)
    p2e_ref, p2e_kf = ref[3], kf[3]
    mn = ref[1][:nt].long().cpu()
    exp_of_tile = ref[0][:nt].long().cpu()
    for tile in range(nt):
        start = tile * TILE
        end = int(mn[tile])
        a = torch.sort(p2e_ref[start:end]).values
        b = torch.sort(p2e_kf[start:end]).values
        # a tile holds a slice of one expert's rows; across the tiles of an expert the row order may differ, so compare
        # whole experts below; per tile only the number of valid rows must agree
        assert a.numel() == b.numel(), f"tile {tile} (expert {int(exp_of_tile[tile])})"
    for e in range(LOCAL):
        tiles = (exp_of_tile == e).nonzero().reshape(-1)
        if tiles.numel() == 0:
            continue
        first, last = int(tiles[0]), int(tiles[-1])
        a = torch.sort(p2e_ref[first * TILE : int(mn[last])]).values
        b = torch.sort(p2e_kf[first * TILE : int(mn[last])]).values
        assert torch.equal(a, b), f"expert {e}"


def test_deterministic():
    offset = 56 * 5
    ids = _inputs(3072, "dispatch", 11, offset)
    scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")
    first = _kf(ids, scales, offset)
    nt = int(first[5])
    rows = int(first[1][nt - 1])
    for _ in range(20):
        again = _kf(ids, scales, offset)
        assert torch.equal(first[2], again[2])
        assert torch.equal(first[0][:nt], again[0][:nt]) and torch.equal(
            first[1][:nt], again[1][:nt]
        )
        valid = torch.zeros(first[3].numel(), dtype=torch.bool, device="cuda")
        valid[(first[2].reshape(-1)[first[2].reshape(-1) >= 0]).long()] = True
        assert torch.equal(first[3][valid], again[3][valid]) and rows == int(again[1][nt - 1])


@pytest.mark.parametrize("t", [512, 3072, 4096])
def test_cuda_graph_replay(t):
    offset = 56 * 9
    static_ids = _inputs(t, "dispatch", 0, offset)
    scales = torch.ones(static_ids.shape, dtype=torch.float32, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.ops.trtllm.kimi_k3_moe_sort_kf(
            static_ids, scales, NUM_EXPERTS, TOP_K, offset, LOCAL, TILE
        )
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = torch.ops.trtllm.kimi_k3_moe_sort_kf(
            static_ids, scales, NUM_EXPERTS, TOP_K, offset, LOCAL, TILE
        )
    for seed, mode in (
        (1, "dispatch"),
        (2, "uniform"),
        (3, "hot"),
        (4, "no_local"),
        (5, "dispatch"),
    ):
        new = _inputs(t, mode, seed, offset)
        static_ids.copy_(new)
        for o in outs:
            o.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        _check_layout(outs, new, offset, f"graph t={t} mode={mode}")


def test_misaligned_and_strided_inputs():
    offset = 56 * 3
    t = 2048
    ids = _inputs(t, "dispatch", 0, offset)
    scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")
    for how in ("aligned_view", "misaligned_view", "transposed"):
        if how == "aligned_view":
            base = torch.empty(t * TOP_K + 4, dtype=torch.int32, device="cuda")
            base[4:] = ids.reshape(-1)
            view = base[4:].view(t, TOP_K)
        elif how == "misaligned_view":
            base = torch.empty(t * TOP_K + 4, dtype=torch.int32, device="cuda")
            base[1 : 1 + t * TOP_K] = ids.reshape(-1)
            view = base[1 : 1 + t * TOP_K].view(t, TOP_K)
        else:
            view = ids.t().contiguous().t()
            assert not view.is_contiguous()
            view.copy_(ids)
        outs = _kf(view, scales, offset)
        torch.cuda.synchronize()
        _check_layout(outs, ids, offset, how)


def test_rejects_unsupported_arguments(cpp_exceptions_propagate):
    if not cpp_exceptions_propagate:
        pytest.skip(
            "a C++ exception aborts the process here (libcurand.so.11 interposes __gxx_personality_v0): "
            "run with LD_PRELOAD=libstdc++.so.6"
        )
    offset = 56
    ids = _inputs(256, "dispatch", 0, offset)
    scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")
    op = torch.ops.trtllm.kimi_k3_moe_sort_kf
    with pytest.raises(RuntimeError):
        op(ids.long(), scales, NUM_EXPERTS, TOP_K, offset, LOCAL, TILE)
    with pytest.raises(RuntimeError):
        op(ids, scales, NUM_EXPERTS, TOP_K, offset, 112, TILE)
    with pytest.raises(RuntimeError):
        op(ids, scales, NUM_EXPERTS, TOP_K, offset, LOCAL, 64)
    with pytest.raises(RuntimeError):
        op(ids, scales, 256, TOP_K, offset, LOCAL, TILE)
    with pytest.raises(RuntimeError):
        op(ids, scales, NUM_EXPERTS, TOP_K, NUM_EXPERTS - LOCAL + 1, LOCAL, TILE)
    with pytest.raises(RuntimeError):
        op(ids[:0], scales[:0], NUM_EXPERTS, TOP_K, offset, LOCAL, TILE)
    for bad in (1, 8, 17, 100, 255):  # not a multiple of 16
        with pytest.raises(RuntimeError):
            op(ids[:bad], scales[:bad], NUM_EXPERTS, TOP_K, offset, LOCAL, TILE)
    too_many = torch.zeros((MAX_TOKENS + 16, TOP_K), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        op(too_many, too_many.float(), NUM_EXPERTS, TOP_K, offset, LOCAL, TILE)


@pytest.mark.parametrize("t", [512, 2048, 3072, 4096])
def test_moe_permute_unpermute_round_trip_equals_moe_sort(t):
    """The consumers see the same tokens: permute + unpermute (scaled sum over the local entries) is bit-identical
    whether moe_sort or the generated sort produced the maps."""
    offset = 56 * 7
    hidden = 512
    ids = _inputs(t, "dispatch", 21, offset)
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn((t, hidden), device="cuda", dtype=torch.bfloat16, generator=gen)
    scales = torch.rand((t, TOP_K), device="cuda", dtype=torch.float32, generator=gen) + 0.1
    kw = dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=LOCAL,
        tile_tokens_dim=TILE,
    )
    results = []
    for op in (torch.ops.trtllm.moe_sort, torch.ops.trtllm.kimi_k3_moe_sort_kf):
        _poison_allocator(t)
        tile_expert, mn_limit, e2p, p2e, total, ntiles = op(
            token_selected_experts=ids, token_final_scales=scales, **kw
        )
        permuted, _ = torch.ops.trtllm.moe_permute(x, None, mn_limit, p2e, ntiles, TILE, TOP_K)
        out = torch.zeros((t, hidden), device="cuda", dtype=torch.bfloat16)
        torch.ops.trtllm.moe_unpermute_inplace(permuted, out, e2p, scales)
        results.append(out)
    torch.cuda.synchronize()
    assert torch.equal(results[0], results[1])
    local = ((ids >= offset) & (ids < offset + LOCAL)).float()
    expected = (x.float() * (scales * local).sum(1, keepdim=True)).to(torch.bfloat16)
    torch.testing.assert_close(results[1].float(), expected.float(), rtol=2e-2, atol=2e-2)


class _CountingOp:
    """Stands in for torch.ops.trtllm.<name>: forwards to the real op and records the token counts it saw."""

    def __init__(self, real):
        self.real, self.calls = real, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs["token_selected_experts"].shape[0])
        return self.real(**kwargs)


def _k3_args(offset=56):
    return dict(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=LOCAL,
        tile_tokens_dim=TILE,
    )


MIN_TOKENS_ENV = "TRTLLM_K3_MOE_SORT_KF_MIN_TOKENS"


@pytest.fixture
def mod():
    """fused_moe_cute_dsl with the gate's cached settings (minimum token count, setup check) forgotten, before and
    after the test: they are read on first use."""
    from tensorrt_llm._torch.modules.fused_moe import fused_moe_cute_dsl

    fused_moe_cute_dsl._k3_moe_sort_kf_min_tokens.cache_clear()
    fused_moe_cute_dsl._validate_k3_moe_sort_kf.cache_clear()
    yield fused_moe_cute_dsl
    fused_moe_cute_dsl._k3_moe_sort_kf_min_tokens.cache_clear()
    fused_moe_cute_dsl._validate_k3_moe_sort_kf.cache_clear()


def test_gate_routes_k3_ep16_shapes_to_the_generated_sort(monkeypatch, mod):
    kf, prod = (
        _CountingOp(torch.ops.trtllm.kimi_k3_moe_sort_kf),
        _CountingOp(torch.ops.trtllm.moe_sort),
    )
    monkeypatch.setattr(torch.ops.trtllm, "kimi_k3_moe_sort_kf", kf, raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", prod, raising=False)
    monkeypatch.setenv(MIN_TOKENS_ENV, "128")
    ids = _inputs(1024, "dispatch", 0, 56)
    scales = torch.ones(ids.shape, dtype=torch.float32, device="cuda")

    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", False)
    mod._moe_sort(ids, scales, **_k3_args())
    assert (len(kf.calls), len(prod.calls)) == (0, 1), "flag off: moe_sort"

    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    outs = mod._moe_sort(ids, scales, **_k3_args())
    assert (len(kf.calls), len(prod.calls)) == (1, 1), "flag on, K3 EP16 shape: the generated sort"
    torch.cuda.synchronize()
    _check_layout(outs, ids, 56, "through the gate")

    small = _inputs(64, "dispatch", 0, 56)
    mod._moe_sort(small, torch.ones(small.shape, dtype=torch.float32, device="cuda"), **_k3_args())
    assert (len(kf.calls), len(prod.calls)) == (1, 2), "below the minimum token count: moe_sort"

    odd = _inputs(1024, "uniform", 0, 56)[:1000].contiguous()
    mod._moe_sort(odd, torch.ones(odd.shape, dtype=torch.float32, device="cuda"), **_k3_args())
    assert (len(kf.calls), len(prod.calls)) == (1, 3), "not a multiple of 16 rows: moe_sort"


def test_gate_decision_boundaries(monkeypatch, mod):
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    monkeypatch.setenv(MIN_TOKENS_ENV, "128")

    def decide(ids, **overrides):
        args = dict(_k3_args(), **overrides)
        return mod._use_k3_moe_sort_kf(
            ids,
            args["num_experts"],
            args["top_k"],
            args["local_expert_offset"],
            args["local_num_experts"],
            args["tile_tokens_dim"],
        )

    def rows(t, dtype=torch.int32):
        return torch.zeros((t, TOP_K), dtype=dtype, device="cuda")

    assert decide(rows(128)) and decide(rows(4096)) and decide(rows(MAX_TOKENS))
    assert not decide(rows(112)), "below the minimum token count"
    assert not decide(rows(127)) and not decide(rows(129)) and not decide(rows(4097)), (
        "not a multiple of 16"
    )
    assert not decide(rows(MAX_TOKENS + 16)), "above the kernel's limit"
    assert not decide(rows(1024, torch.int64)), "ids must be int32"
    assert not decide(rows(1024).cpu()), "ids must be on the GPU"
    assert not decide(rows(1024), local_num_experts=112), "EP8 rank"
    assert not decide(rows(1024), tile_tokens_dim=64)
    assert not decide(rows(1024), top_k=8)
    assert not decide(rows(1024), num_experts=256)
    assert not decide(rows(1024), local_expert_offset=NUM_EXPERTS - LOCAL + 1), (
        "offset outside the expert range"
    )
    assert not decide(rows(1024), local_expert_offset=-1)
    assert decide(rows(1024), local_expert_offset=NUM_EXPERTS - LOCAL)
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", False)
    assert not decide(rows(1024)), "flag off"


def test_gate_is_sm107_only(monkeypatch, mod):
    """The kernel has only run on SM107: no other SM takes it, whatever the shape."""
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    monkeypatch.setenv(MIN_TOKENS_ENV, "128")
    ids = torch.zeros((1024, TOP_K), dtype=torch.int32, device="cuda")
    for sm, expected in ((80, False), (90, False), (100, False), (103, False), (107, True)):
        monkeypatch.setattr(mod, "get_sm_version", lambda sm=sm: sm)
        decided = mod._use_k3_moe_sort_kf(ids, NUM_EXPERTS, TOP_K, 56, LOCAL, TILE)
        assert decided is expected, f"SM{sm}"
    for sm in (110, 120):
        monkeypatch.setattr(mod, "get_sm_version", lambda sm=sm: sm)
        assert not mod._use_k3_moe_sort_kf(ids, NUM_EXPERTS, TOP_K, 56, LOCAL, TILE), f"SM{sm}"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 64),
        ("", 64),
        ("   ", 64),
        ("128", 128),
        (" 128 ", 128),
        ("100", 100),
        ("16", 16),
        ("8", 16),
        ("1", 16),
        ("0", 16),
        ("-5", 16),
    ],
)
def test_minimum_token_count_is_never_below_the_token_multiple(monkeypatch, mod, value, expected):
    """``TRTLLM_K3_MOE_SORT_KF_MIN_TOKENS`` is clamped to 16: a smaller bound must not let a call of no rows (or of
    rows that are not a multiple of 16) reach the kernel. Unset, empty and blank are the default (64); the empty
    string is really set here, as ``MIN_TOKENS=`` in a launch script does."""
    if value is None:
        monkeypatch.delenv(MIN_TOKENS_ENV, raising=False)
    else:
        monkeypatch.setenv(MIN_TOKENS_ENV, value)
    assert mod._k3_moe_sort_kf_min_tokens() == expected


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_minimum_is_the_default_and_passes_the_setup_check(monkeypatch, mod, value):
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    monkeypatch.setenv(MIN_TOKENS_ENV, value)
    mod._validate_k3_moe_sort_kf()
    ids = torch.zeros((64, TOP_K), dtype=torch.int32, device="cuda")
    assert mod._use_k3_moe_sort_kf(ids, NUM_EXPERTS, TOP_K, 56, LOCAL, TILE)
    ids = torch.zeros((48, TOP_K), dtype=torch.int32, device="cuda")
    assert not mod._use_k3_moe_sort_kf(ids, NUM_EXPERTS, TOP_K, 56, LOCAL, TILE), (
        "below the default"
    )


def test_a_zero_minimum_still_keeps_empty_calls_out_of_the_kernel(monkeypatch, mod):
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    monkeypatch.setenv(MIN_TOKENS_ENV, "0")

    def decide(t):
        ids = torch.zeros((t, TOP_K), dtype=torch.int32, device="cuda")
        return mod._use_k3_moe_sort_kf(ids, NUM_EXPERTS, TOP_K, 56, LOCAL, TILE)

    assert not decide(0)
    assert decide(16) and decide(32)
    assert not decide(24), "not a multiple of 16"


def test_the_minimum_is_read_on_first_use_not_when_the_module_is_imported(monkeypatch, mod):
    monkeypatch.setenv(MIN_TOKENS_ENV, "256")
    assert mod._k3_moe_sort_kf_min_tokens() == 256
    monkeypatch.setenv(MIN_TOKENS_ENV, "512")
    assert mod._k3_moe_sort_kf_min_tokens() == 256, "read once"
    mod._k3_moe_sort_kf_min_tokens.cache_clear()
    assert mod._k3_moe_sort_kf_min_tokens() == 512


def test_a_bad_minimum_does_not_break_the_import_and_fails_when_a_layer_is_built():
    module = "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl"
    bad = {"TRTLLM_K3_MOE_SORT_KF": "1", MIN_TOKENS_ENV: "many"}
    imported = _python(f"import {module}", **bad)
    assert imported.returncode == 0, imported.stderr[-2000:]
    built = _python(f"import {module} as m; m._validate_k3_moe_sort_kf()", **bad)
    tail = built.stderr[-2000:]
    assert built.returncode != 0 and f"ValueError: {MIN_TOKENS_ENV}" in built.stderr, tail
    fine = _python(
        f"import {module} as m; m._validate_k3_moe_sort_kf()",
        TRTLLM_K3_MOE_SORT_KF="1",
        **{MIN_TOKENS_ENV: "128"},
    )
    assert fine.returncode == 0, fine.stderr[-2000:]


def test_the_setup_check_is_silent_with_the_flag_off_and_needs_the_op_with_it_on(monkeypatch, mod):
    """A build without the op (a Python overlay on an older image) with the flag on fails once, with a clear error,
    when a layer is built: not with an AttributeError at the first sort of a step."""
    monkeypatch.setattr(mod, "get_sm_version", lambda: 107)
    monkeypatch.setenv(MIN_TOKENS_ENV, "many")  # not even read with the flag off
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", False)
    mod._validate_k3_moe_sort_kf()
    mod._validate_k3_moe_sort_kf.cache_clear()
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    with pytest.raises(ValueError, match=MIN_TOKENS_ENV):
        mod._validate_k3_moe_sort_kf()  # the minimum is checked first; a failed check is not cached
    monkeypatch.setenv(MIN_TOKENS_ENV, "128")
    mod._validate_k3_moe_sort_kf()  # this build has the op
    monkeypatch.setattr(torch.ops, "trtllm", types.SimpleNamespace(), raising=False)
    mod._validate_k3_moe_sort_kf.cache_clear()
    with pytest.raises(RuntimeError, match="kimi_k3_moe_sort_kf"):
        mod._validate_k3_moe_sort_kf()


def test_the_setup_check_needs_the_op_only_where_the_gate_can_take_it(monkeypatch, mod):
    """Only SM107 sorts with the op, so only SM107 needs it: with the flag on, a build without the op on another GPU
    still builds its layers (it keeps ``moe_sort``); on SM107 it fails once, when a layer is built."""
    monkeypatch.setattr(mod, "_K3_MOE_SORT_KF", True)
    monkeypatch.setenv(MIN_TOKENS_ENV, "128")
    monkeypatch.setattr(
        torch.ops, "trtllm", types.SimpleNamespace(), raising=False
    )  # no op in this build
    for sm in (80, 90, 100, 103, 110, 120):
        monkeypatch.setattr(mod, "get_sm_version", lambda sm=sm: sm)
        mod._validate_k3_moe_sort_kf.cache_clear()
        mod._validate_k3_moe_sort_kf()
    monkeypatch.setattr(mod, "get_sm_version", lambda: 107)
    mod._validate_k3_moe_sort_kf.cache_clear()
    with pytest.raises(RuntimeError, match="kimi_k3_moe_sort_kf"):
        mod._validate_k3_moe_sort_kf()
    monkeypatch.setenv(MIN_TOKENS_ENV, "many")  # the minimum is checked on every SM
    monkeypatch.setattr(mod, "get_sm_version", lambda: 90)
    mod._k3_moe_sort_kf_min_tokens.cache_clear()
    mod._validate_k3_moe_sort_kf.cache_clear()
    with pytest.raises(ValueError, match=MIN_TOKENS_ENV):
        mod._validate_k3_moe_sort_kf()


# ---- the kernel waits for the ids: a producer that triggers its dependents first and writes them late -------------
# ``griddepcontrol.wait`` is the first thing the kernel does with global memory; PDL lets it launch as soon as every CTA
# of its producer has triggered, so a missing or misplaced wait reads the ids of the previous round. The producer
# triggers at its first instruction, spins until the host sets a flag (50 ms later) and only then writes the ids; the
# buffer holds other, valid-looking ids until then.

_RIG = {}
_SPIN_TIMEOUT_NS = 10_000_000_000  # a missing release ends the spin (and fails the comparison)
_RELEASE_AFTER_S = 0.05


def _rig():
    """Triton kernels of the late-producer test (built once)."""
    if _RIG:
        return _RIG
    import triton
    import triton.language as tl

    @triton.jit
    def _now():
        return tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )

    @triton.jit
    def _wait_for_predecessors():
        tl.inline_asm_elementwise(
            "griddepcontrol.wait; // dummy $0",
            "=r,~{memory}",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _late_copy(flag_addr, timeout_ns, src, dst, n, BLOCK: tl.constexpr):
        flag = flag_addr.to(tl.int64).to(tl.pointer_type(tl.int32))
        tl.extra.cuda.gdc_launch_dependents()
        t0 = _now()
        while (tl.load(flag, volatile=True) == 0) & (_now() - t0 < timeout_ns):
            pass
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    @triton.jit
    def _probe(flag_addr, stamps):
        flag = flag_addr.to(tl.int64).to(tl.pointer_type(tl.int32))
        at_entry = tl.load(flag, volatile=True)
        _wait_for_predecessors()
        after_wait = tl.load(flag, volatile=True)
        tl.store(stamps, at_entry)
        tl.store(stamps + 1, after_wait)

    _RIG.update(triton=triton, late_copy=_late_copy, probe=_probe)
    _warm_up_rig()
    return _RIG


def _warm_up_rig():
    """Compile both kernels now. The first launch of a Triton kernel compiles it, which takes seconds on a cold cache:
    the producer of the test would already be spinning (and time out) while the host compiled the probe behind it."""
    flag = torch.ones(1, dtype=torch.int32).pin_memory()  # already released: nothing spins
    src = torch.arange(1024, dtype=torch.int32, device="cuda")
    dst = torch.zeros_like(src)
    stamps = torch.zeros(2, dtype=torch.int32, device="cuda")
    _enqueue_late_producer(flag, src, dst)
    _RIG["probe"][(1,)](flag.data_ptr(), stamps, launch_pdl=True)
    torch.cuda.synchronize()
    assert torch.equal(dst, src) and stamps.tolist() == [1, 1]


def _enqueue_late_producer(flag, src, dst):
    """dst <- src once the host flag is set; triggers its dependents before it starts to wait for the flag. Launched
    as an ordinary kernel (after everything queued before it has completed): only a dependent needs the PDL launch
    attribute, and the trigger is an instruction."""
    rig = _rig()
    n = dst.numel()
    block = 1024
    rig["late_copy"][(rig["triton"].cdiv(n, block),)](
        flag.data_ptr(), _SPIN_TIMEOUT_NS, src, dst, n, BLOCK=block, num_warps=4
    )


def _release_after_a_delay(flag):
    time.sleep(_RELEASE_AFTER_S)
    flag[0] = 1  # a plain host store to pinned memory: the producer polls it


@pytest.mark.parametrize("t", [512, 3072])
def test_the_sort_waits_for_a_producer_that_triggers_early_and_writes_late(t):
    rig = _rig()
    offset = 56 * 3
    true_ids = _inputs(t, "dispatch", 31, offset)
    stale_ids = _inputs(t, "uniform", 32, offset)  # what a kernel that did not wait would sort
    assert not torch.equal(true_ids, stale_ids)
    scales = torch.ones(true_ids.shape, dtype=torch.float32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32).pin_memory()

    # The rig itself: a dependent launched behind the producer starts before the release (the flag is 0 at its entry)
    # and returns from its wait after it. Without that overlap the check below would prove nothing.
    stamps = torch.zeros(2, dtype=torch.int32, device="cuda")
    scratch = stale_ids.clone()
    _enqueue_late_producer(flag, true_ids, scratch)
    rig["probe"][(1,)](flag.data_ptr(), stamps, launch_pdl=True)
    _release_after_a_delay(flag)
    torch.cuda.synchronize()
    assert stamps.tolist() == [0, 1], (
        f"the dependent did not overlap its producer (flag at entry / after its wait: {stamps.tolist()})"
    )
    assert torch.equal(scratch, true_ids)

    # The sort behind the same producer: PDL launch (TRTLLM_ENABLE_PDL on, the default), outputs poisoned.
    flag[0] = 0
    ids = stale_ids.clone()
    _poison_allocator(t)
    _enqueue_late_producer(flag, true_ids, ids)
    outs = torch.ops.trtllm.kimi_k3_moe_sort_kf(
        ids, scales, NUM_EXPERTS, TOP_K, offset, LOCAL, TILE
    )
    _release_after_a_delay(flag)
    torch.cuda.synchronize()
    assert torch.equal(ids, true_ids)
    _check_layout(outs, true_ids, offset, f"late producer t={t}")
