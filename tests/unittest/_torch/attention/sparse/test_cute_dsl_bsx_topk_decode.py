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
"""BSX (direct/reg/tp CuTe DSL tiers) top-K decode tests.

CI-sized exactness grid for the guarded fp32 fast path inside
``trtllm::cute_dsl_gvr_topk_decode`` (next_n >= 1, cr in {1, 4}): every reg
launch-table instance once, the direct and tp tiers at a few npad each,
ragged (varlen) rows with POISONED tails (stale-garbage simulation: +1e30
beyond N_eff, which would dominate the top-K if the ragged-N masking were
broken), quantized-tie inputs, degenerate rows, pre_idx hardening,
host-only route-table asserts, a dispatcher-fallback check (bf16 routes to
the in-tree kernel), and the MTP axis: next_n in {2, 3, 4} x cr in {1, 4}
x all three tiers, each case checked against BOTH the torch.topk host
N_eff/offset simulation (``_tie_aware_check``) and a differential in-tree
arm (same inputs through the in-tree kernel via ``order_row``, per-row
value-multiset equality).
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops  # noqa: F401
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_bsx_dispatch as bsx_dispatch,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.single_pass_multi_cta_radix_topk_cluster import (  # noqa: E501
    _query_max_cluster_size,
)
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL BSX Top-K only supports SM 100/103, got SM {get_sm_version()}",
)

CR = 4  # default axis of the legacy (pre-MTP) cases: compress_ratio == 4 (DSv4)
NEXT_N = 1  # default axis of the legacy (pre-MTP) cases: next_n == 1


@pytest.fixture(autouse=True)
def _bsx_bands_off(monkeypatch):
    """Kernel-contract tests must reach the bsx tiers: disable the measured
    fallback band table (it legitimately routes several tested (npad, bs)
    shapes to the in-tree kernel in production). The table itself is covered
    by ``test_bsx_fallback_band_table``."""
    monkeypatch.setenv("TRTLLM_BSX_FALLBACK_BANDS", "0")
    bsx_dispatch._reset_env_cache()
    yield
    monkeypatch.delenv("TRTLLM_BSX_FALLBACK_BANDS", raising=False)
    bsx_dispatch._reset_env_cache()


@skip_not_sm100
@pytest.mark.parametrize("bands", ["on", "off"])
def test_bsx_cuda_graph_capture_replay(monkeypatch, bands):
    """The op must be CUDA-graph capturable and replay-consistent on both
    sides of the fallback band table. Shape (npad=32768, bs=64) sits inside
    a fallback bucket: bands=on captures the in-tree branch, bands=off the
    bsx tp branch — the dispatch decision is host-side per shape, so it
    bakes into the graph at capture time and must hold across replays,
    including replays over in-place rewritten inputs."""
    if bands == "on":
        # the autouse fixture pinned the table off; restore the default
        monkeypatch.delenv("TRTLLM_BSX_FALLBACK_BANDS", raising=False)
    bsx_dispatch._reset_env_cache()
    bs, npad, top_k = 64, 32768, 2048
    logits, pre_idx, seq_lens = _make_bsx_inputs(bs, npad, top_k, seed=1234)
    out = torch.empty(bs, top_k, dtype=torch.int32, device="cuda")

    def call():
        torch.ops.trtllm.cute_dsl_gvr_topk_decode(
            logits, pre_idx, seq_lens, out, top_k=top_k, next_n=NEXT_N, compress_ratio=CR
        )

    # warmup outside capture: JIT compile + any lazy init, on a side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    g.replay()
    torch.cuda.synchronize()
    _tie_aware_check(out, logits, seq_lens, top_k)

    # replay over in-place rewritten inputs (fresh logits + fresh hints)
    logits2, pre_idx2, seq_lens2 = _make_bsx_inputs(bs, npad, top_k, seed=4321)
    logits.copy_(logits2)
    pre_idx.copy_(pre_idx2)
    seq_lens.copy_(seq_lens2)
    g.replay()
    torch.cuda.synchronize()
    _tie_aware_check(out, logits, seq_lens, top_k)


def test_bsx_fallback_band_table(monkeypatch):
    """The measured (npad, bs) fallback buckets route to the in-tree kernel
    by default; the kill-switch restores bsx service; neighbours are not
    over-routed. Bands: full-grid calibration (2026-07-28), 131072
    lower bound recalibrated to 8 (2026-07-29)."""
    monkeypatch.delenv("TRTLLM_BSX_FALLBACK_BANDS", raising=False)
    bsx_dispatch._reset_env_cache()
    inb = bsx_dispatch._in_fallback_band
    # routed buckets (bucket floor <0.909 vs the in-tree head)
    assert inb(256, 8192) and inb(1024, 8192)
    assert inb(16, 32768) and inb(1024, 65536)
    assert inb(128, 131072) and inb(64, 262144)
    assert inb(8, 131072) and inb(8, 131136)  # 128K x BS8 (reg-tier floor)
    # ... but shapes that only ROUND into the 131072 bucket keep the
    # calibrated bs>=16 routing: at bs=8 the reg tier is 1.36-1.60x ahead
    # there, so the low-bs extension must not capture them.
    assert not inb(8, 163776)
    assert inb(16, 163776) and inb(255, 163776) and not inb(256, 163776)
    assert inb(48, 40960)  # off-grid npad resolves to the nearest pow2
    # neighbours stay on bsx
    assert not inb(128, 8192)  # direct/tp win band
    assert not inb(8, 32768)  # latency reg band
    assert not inb(256, 131072) and not inb(128, 262144)  # large-N tp win
    assert not inb(1024, 4096)  # small-npad tp win
    # kill-switch
    monkeypatch.setenv("TRTLLM_BSX_FALLBACK_BANDS", "0")
    bsx_dispatch._reset_env_cache()
    assert not inb(64, 32768)
    monkeypatch.delenv("TRTLLM_BSX_FALLBACK_BANDS", raising=False)
    bsx_dispatch._reset_env_cache()


# ---------------------------------------------------------------------------
# Shared helpers. ``_tie_aware_check`` is a local copy of the one in
# test_cute_dsl_gvr_topk_decode.py (same directory): the sibling test module
# is not reliably importable across the repo's pytest invocation styles
# (rootdir-dependent package resolution), so the checker is duplicated here
# verbatim-in-spirit with the same semantics. Keep the two in sync.
# ---------------------------------------------------------------------------
def _tie_aware_check(
    out_indices: torch.Tensor,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = NEXT_N,
    compress_ratio: int = CR,
) -> None:
    """Vectorized multi-row tie-aware correctness check (strict sort+allclose).

    Per row r the scan range is ``logits[r, :N_eff(r)]`` where
    ``N_eff = (seq_lens[r // next_n] - next_n + r % next_n + 1) // cr``
    (the kernels' exact formula). Checks: in-range indices, no duplicates,
    no selected value below the K-th reference value, and sorted-value
    multiset equality against torch.topk.
    """
    num_rows, top_k_out = out_indices.shape
    assert top_k_out == top_k
    device = logits.device
    logits_f32 = logits.to(torch.float32)
    N = logits.shape[1]

    row_idx = torch.arange(num_rows, device=device)
    group_idx = row_idx // next_n
    ofs = row_idx % next_n
    seq_lens_per_row = seq_lens.to(device=device, dtype=torch.long)[group_idx]
    actual_kv_len = seq_lens_per_row - next_n + ofs + 1
    N_eff = actual_kv_len // compress_ratio  # [num_rows]

    col_idx = torch.arange(N, device=device)
    in_range_mask = col_idx[None, :] < N_eff[:, None]
    masked_logits = torch.where(in_range_mask, logits_f32, float("-inf"))
    ref_vals, _ = torch.topk(masked_logits, k=top_k, largest=True, sorted=True, dim=-1)

    out_of_range = (out_indices < 0) | (out_indices >= N_eff[:, None])
    if bool(out_of_range.any().item()):
        bad_row = int(out_of_range.any(dim=1).int().argmax().item())
        raise AssertionError(
            f"row={bad_row}: out-of-range index "
            f"(N_eff={int(N_eff[bad_row].item())}, "
            f"indices={out_indices[bad_row].cpu().tolist()})"
        )

    sorted_idx, _ = out_indices.sort(dim=-1)
    has_dup = (sorted_idx[:, 1:] == sorted_idx[:, :-1]).any(dim=-1)
    if bool(has_dup.any().item()):
        bad_row = int(has_dup.int().argmax().item())
        raise AssertionError(
            f"row={bad_row}: duplicate indices: {out_indices[bad_row].cpu().tolist()}"
        )

    sel_vals = torch.gather(logits_f32, dim=-1, index=out_indices.long())
    kth_vals = ref_vals[:, -1:]
    n_below_per_row = (sel_vals < kth_vals).sum(dim=-1)
    if bool((n_below_per_row > 0).any().item()):
        bad_row = int(n_below_per_row.argmax().item())
        raise AssertionError(
            f"row={bad_row}: {int(n_below_per_row[bad_row].item())} selected "
            f"values < Kth-rank value ({float(kth_vals[bad_row, 0].item()):.6f})"
        )

    sel_sorted, _ = sel_vals.sort(dim=-1, descending=True)
    if not bool(torch.allclose(sel_sorted, ref_vals, rtol=1e-5, atol=1e-5)):
        per_row_max = (sel_sorted - ref_vals).abs().max(dim=-1).values
        bad_row = int(per_row_max.argmax().item())
        raise AssertionError(
            f"row={bad_row}: sorted-value mismatch — max diff "
            f"{float(per_row_max[bad_row].item()):.4e}"
        )


def _make_bsx_inputs(
    bs: int,
    npad: int,
    top_k: int,
    seed: int,
    kind: str = "randn",
    varlen: bool = True,
    preidx: str = "mixed",
    hit_rate: float = 0.5,
):
    """Build (logits fp32, pre_idx int32, seq_lens int32) for the bsx path.

    ``kind='ties'`` quantizes the logits to 0.25 steps so massive value
    plateaus straddle the K-th boundary (exercises the tie-ticket emits and
    the max-below plateau-descent path). Ragged rows: the tail beyond each
    row's N_eff is POISONED with +1e30 — stale garbage that would win the
    top-K if the lane masking were broken (production tails are stale
    values, not -FLT_MAX pad).

    ``preidx``: 'mixed' = argmax slot 0 + ``hit_rate`` real topk hints
    (default ~50%); 'zeros' = all-zero cold start; 'oor' = out-of-range
    garbage (negative / >= npad) that the kernels must clamp harmlessly.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = torch.randn(bs, npad, dtype=torch.float32, device="cuda") * 2.0
    if kind == "ties":
        logits = (logits * 4.0).round() * 0.25

    if varlen:
        lo = (top_k + 1) * CR  # keep every row non-degenerate (N_eff > K)
        seq_lens = torch.randint(lo, npad * CR + 1, (bs,), dtype=torch.int32, device="cuda")
    else:
        seq_lens = torch.full((bs,), npad * CR, dtype=torch.int32, device="cuda")

    n_eff = (seq_lens.long() // CR).clamp(max=npad)
    col = torch.arange(npad, device="cuda")
    tail = col[None, :] >= n_eff[:, None]
    logits = torch.where(tail, torch.full_like(logits, 1e30), logits)

    valid_logits = torch.where(tail, torch.full_like(logits, float("-inf")), logits)
    argmax_idx = valid_logits.argmax(dim=-1).int()
    if preidx == "zeros":
        pre_idx = torch.zeros(bs, top_k, dtype=torch.int32, device="cuda")
    elif preidx == "oor":
        pre_idx = torch.randint(-npad, 4 * npad, (bs, top_k), dtype=torch.int32, device="cuda")
    else:
        ref_topk = valid_logits.topk(top_k, dim=-1).indices.int()
        keep = torch.rand(ref_topk.shape, device="cuda") < hit_rate
        junk = torch.arange(top_k, dtype=torch.int32, device="cuda").expand(bs, -1)
        # junk arange is in-range for every row: N_eff > top_k here.
        pre_idx = torch.where(keep, ref_topk, junk).contiguous()
        pre_idx[:, 0] = argmax_idx
    return logits, pre_idx, seq_lens


def _run_op(logits, pre_idx, seq_lens, top_k):
    out = torch.empty(logits.shape[0], top_k, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out,
        top_k=top_k,
        next_n=NEXT_N,
        compress_ratio=CR,
    )
    torch.cuda.synchronize()
    return out


def _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, expected_tier):
    """Host-only: the dispatcher must accept this call and route it to
    ``expected_tier``."""
    out = torch.empty(logits.shape[0], top_k, dtype=torch.int32, device="cuda")
    assert bsx_dispatch.is_bsx_supported(
        logits, pre_idx, seq_lens, out, top_k, NEXT_N, CR, None, None
    ), "expected the bsx fast path to accept this call"
    bs, npad = logits.shape
    tier = bsx_dispatch.route(bs, npad, top_k)
    assert tier == expected_tier, f"route({bs}, {npad}, {top_k}) = {tier} != {expected_tier}"


def _skip_if_cluster_capped(bs, npad, top_k):
    cs = bsx_dispatch.route_cluster_size(bs, npad, top_k)
    torch.zeros(1, device="cuda")  # the driver-API query needs a live context
    hw_max = _query_max_cluster_size()
    if cs > hw_max:
        pytest.skip(f"tier cluster size {cs} exceeds device max {hw_max}")


# ---------------------------------------------------------------------------
# Host-only route-table asserts (mirror of the original CUDA dispatch).
# ---------------------------------------------------------------------------
def test_bsx_route_table():
    r = bsx_dispatch.route
    # latency ladder
    assert r(1, 4096, 512) == "direct"
    assert r(1, 12288, 2048) == "direct"
    assert r(1, 14336, 512) == "reg(cs=1,tb=512,maxv=8,ar=8)"
    assert r(1, 24576, 512) == "reg(cs=4,tb=512,maxv=4,ar=8)"
    assert r(1, 49152, 512) == "reg(cs=8,tb=512,maxv=3,ar=8)"
    assert r(1, 65536, 1024) == "reg(cs=8,tb=512,maxv=4,ar=8)"
    assert r(8, 131072, 512) == "reg(cs=8,tb=512,maxv=8,ar=8)"
    assert r(1, 163840, 2048) == "reg(cs=16,tb=512,maxv=5,ar=6)"
    assert r(4, 163840, 512) == "reg(cs=16,tb=512,maxv=5,ar=8)"
    assert r(1, 262144, 2048) == "reg(cs=16,tb=512,maxv=8,ar=8)"
    assert r(4, 262144, 1024) == "reg(cs=16,tb=512,maxv=8,ar=6)"
    # dense table
    assert r(64, 20480, 512) == "reg(cs=1,tb=1024,maxv=5,ar=8)"
    assert r(64, 28672, 512) == "reg(cs=1,tb=1024,maxv=8,ar=8)"
    assert r(8, 262144, 2048) == "reg(cs=8,tb=1024,maxv=8,ar=8)"
    # tp takeover
    assert r(256, 20480, 512) == "tp"
    assert r(128, 28672, 512) == "tp"
    assert r(16, 65536, 1024) == "tp"
    assert r(128, 262144, 2048) == "tp"
    # gvr tier only beyond the deployment envelope (guarded out upstream)
    assert r(1, 262208, 512).startswith("gvr")


def test_bsx_route_env_knobs(monkeypatch):
    """TRTLLM_BSX_TP_BS / TRTLLM_BSX_DENSE_BS keep the GVR_BSX_* semantics:
    unset/-1 -> baked bands, 0 -> disable, else explicit bs threshold."""
    r = bsx_dispatch.route
    try:
        monkeypatch.setenv("TRTLLM_BSX_DENSE_BS", "8")
        bsx_dispatch._reset_env_cache()
        assert r(8, 65536, 512) == "reg(cs=2,tb=1024,maxv=8,ar=8)"
        assert r(8, 131072, 1024) == "reg(cs=4,tb=1024,maxv=8,ar=8)"

        monkeypatch.setenv("TRTLLM_BSX_TP_BS", "0")  # 0 -> disable tp
        bsx_dispatch._reset_env_cache()
        assert r(1024, 65536, 512) != "tp"

        monkeypatch.setenv("TRTLLM_BSX_TP_BS", "4")
        bsx_dispatch._reset_env_cache()
        assert r(4, 65536, 512) == "tp"
    finally:
        monkeypatch.delenv("TRTLLM_BSX_TP_BS", raising=False)
        monkeypatch.delenv("TRTLLM_BSX_DENSE_BS", raising=False)
        bsx_dispatch._reset_env_cache()


# ---------------------------------------------------------------------------
# reg tier: every launch-table instance's ROUTE is asserted (host-only,
# free); a LAUNCH (JIT compile ~5-7s each) runs only for the 6 instances
# that together cover every codegen axis value (cs {1,2,8,16} x tb
# {512,1024} x ar {6,8} x maxv {5,8} + the dense-knob-only path) — the
# dropped launches differ only in axis combinations, not in code paths.
# The cs=2 dense instance is reachable only through the TRTLLM_BSX_DENSE_BS
# knob (default bands route around it), doubling as the env-knob dispatch
# test on a live launch.
# ---------------------------------------------------------------------------
_REG_INSTANCES = [
    # (npad, bs, K, dense_env, expected tier, launch)
    (14336, 1, 512, None, "reg(cs=1,tb=512,maxv=8,ar=8)", True),
    (24576, 1, 512, None, "reg(cs=4,tb=512,maxv=4,ar=8)", False),
    (49152, 1, 512, None, "reg(cs=8,tb=512,maxv=3,ar=8)", False),
    (65536, 1, 1024, None, "reg(cs=8,tb=512,maxv=4,ar=8)", False),
    (131072, 8, 512, None, "reg(cs=8,tb=512,maxv=8,ar=8)", True),
    (163840, 1, 2048, None, "reg(cs=16,tb=512,maxv=5,ar=6)", True),
    (163840, 4, 512, None, "reg(cs=16,tb=512,maxv=5,ar=8)", False),
    (262144, 1, 2048, None, "reg(cs=16,tb=512,maxv=8,ar=8)", False),
    (262144, 4, 1024, None, "reg(cs=16,tb=512,maxv=8,ar=6)", False),
    (20480, 64, 512, None, "reg(cs=1,tb=1024,maxv=5,ar=8)", False),
    (28672, 64, 512, None, "reg(cs=1,tb=1024,maxv=8,ar=8)", True),
    (65536, 8, 512, "8", "reg(cs=2,tb=1024,maxv=8,ar=8)", True),
    (131072, 8, 1024, "8", "reg(cs=4,tb=1024,maxv=8,ar=8)", False),
    (262144, 8, 2048, None, "reg(cs=8,tb=1024,maxv=8,ar=8)", True),
]


@skip_not_sm100
@pytest.mark.parametrize(
    "npad,bs,top_k,dense_env,expected,launch",
    _REG_INSTANCES,
    ids=[t[4] + f"_n{t[0]}_bs{t[1]}_k{t[2]}" for t in _REG_INSTANCES],
)
@pytest.mark.parametrize("kind", ["randn", "ties"])
def test_bsx_reg_launch_table(npad, bs, top_k, dense_env, expected, launch, kind, monkeypatch):
    if not launch and kind == "ties":
        pytest.skip("route-assert-only instance (single kind suffices)")
    _skip_if_cluster_capped(bs, npad, top_k)
    try:
        if dense_env is not None:
            monkeypatch.setenv("TRTLLM_BSX_DENSE_BS", dense_env)
            bsx_dispatch._reset_env_cache()
        logits, pre_idx, seq_lens = _make_bsx_inputs(
            bs, npad, top_k, seed=npad + bs + top_k, kind=kind, varlen=True
        )
        _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, expected)
        if launch:
            out = _run_op(logits, pre_idx, seq_lens, top_k)
            _tie_aware_check(out, logits, seq_lens, top_k)
    finally:
        if dense_env is not None:
            monkeypatch.delenv("TRTLLM_BSX_DENSE_BS", raising=False)
            bsx_dispatch._reset_env_cache()


# ---------------------------------------------------------------------------
# direct tier (npad <= DKCMAX): BS 1/8/64, randn + quantized ties, ragged +
# uniform seq_lens.
# ---------------------------------------------------------------------------
@skip_not_sm100
@pytest.mark.parametrize(
    "npad,bs,top_k,kind,varlen",
    [
        (4096, 1, 512, "randn", True),
        (8256, 8, 512, "ties", True),
        (12288, 64, 2048, "randn", True),
        (4096, 8, 512, "randn", False),  # uniform N_eff == npad (no tail)
    ],
)
def test_bsx_direct(npad, bs, top_k, kind, varlen):
    logits, pre_idx, seq_lens = _make_bsx_inputs(
        bs, npad, top_k, seed=npad * 3 + bs, kind=kind, varlen=varlen
    )
    _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, "direct")
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# tp tier (bs >= tp threshold): trivial npad<=kC path, cs>1 cluster path and
# cs=1 big-npad path.
# ---------------------------------------------------------------------------
@skip_not_sm100
@pytest.mark.parametrize(
    "npad,bs,top_k,kind",
    [
        (4096, 256, 512, "randn"),  # cs=1, trivial npad <= kC path
        (65536, 16, 1024, "ties"),  # cs=8 cluster path + tie plateaus
        (262144, 128, 2048, "randn"),  # cs=1, uf=8 streaming path
    ],
)
def test_bsx_tp(npad, bs, top_k, kind):
    _skip_if_cluster_capped(bs, npad, top_k)
    logits, pre_idx, seq_lens = _make_bsx_inputs(
        bs, npad, top_k, seed=npad + 7 * bs, kind=kind, varlen=True
    )
    _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, "tp")
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# Degenerate rows (N_eff <= K): in-kernel identity emit [0..N_eff-1] + -1
# pad, mixed with normal rows in the same batch, on all three tiers.
# ---------------------------------------------------------------------------
@skip_not_sm100
@pytest.mark.parametrize(
    "npad,bs,top_k,expected_kind",
    [
        (4096, 8, 2048, "direct"),
        (65536, 8, 512, "reg"),
        (65536, 16, 1024, "tp"),
    ],
)
def test_bsx_degenerate_rows(npad, bs, top_k, expected_kind):
    _skip_if_cluster_capped(bs, npad, top_k)
    logits, pre_idx, seq_lens = _make_bsx_inputs(
        bs, npad, top_k, seed=11, kind="randn", varlen=True
    )
    # Rows 0..3: degenerate (N_eff = 0 / 1 / K-1 / K); the rest keep their
    # non-degenerate varlen lengths. Poison the tails accordingly.
    for i, sl in enumerate([NEXT_N, CR + NEXT_N, (top_k - 1) * CR, top_k * CR]):
        seq_lens[i] = sl
    n_eff = (seq_lens.long() // CR).clamp(max=npad)
    col = torch.arange(npad, device="cuda")
    tail = col[None, :] >= n_eff[:, None]
    logits = torch.where(tail, torch.full_like(logits, 1e30), logits)

    tier = bsx_dispatch.route(bs, npad, top_k)
    assert tier.startswith(expected_kind) or tier == expected_kind
    out = _run_op(logits, pre_idx, seq_lens, top_k)

    for i in range(bs):
        ne = int(n_eff[i].item())
        if ne <= top_k:
            expect = torch.full((top_k,), -1, dtype=torch.int32, device="cuda")
            expect[:ne] = torch.arange(ne, dtype=torch.int32, device="cuda")
            assert torch.equal(out[i], expect), (
                f"row {i} (N_eff={ne}): degenerate identity emit mismatch: "
                f"{out[i, : max(ne + 2, 8)].cpu().tolist()}..."
            )
        else:
            _tie_aware_check(out[i : i + 1], logits[i : i + 1], seq_lens[i : i + 1], top_k)


# ---------------------------------------------------------------------------
# pre_idx hardening: all-zero (production cold start) and out-of-range
# garbage hints must neither fault nor corrupt the output.
# ---------------------------------------------------------------------------
@skip_not_sm100
@pytest.mark.parametrize("preidx", ["zeros", "oor"])
@pytest.mark.parametrize(
    "npad,bs,top_k",
    [
        (65536, 4, 512),  # reg tier
        (65536, 16, 1024),  # tp tier
    ],
)
def test_bsx_preidx_hardening(npad, bs, top_k, preidx):
    _skip_if_cluster_capped(bs, npad, top_k)
    logits, pre_idx, seq_lens = _make_bsx_inputs(
        bs, npad, top_k, seed=13, kind="randn", varlen=True, preidx=preidx
    )
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# tp-tier hint-ladder ADMISSION fast path (R0 parity): the P2a stage-0 pick
# pushes candidates at the tightest ladder rung whose sampled CI lies in
# [K, 0.6*kC] (the legacy pivot-band hi; a leaner SAFE legacy band pick
# overrides it — stage 0b), so high-hit-rate rows finish in one fused
# streaming pass. These
# cases pin the admission decision surface: hit-rate extremes, tie plateaus
# AT the admission threshold, forced count > kC overflow fallback, and a
# batch mixing admitted and fallback rows (per-row escape independence).
# ---------------------------------------------------------------------------
@skip_not_sm100
@pytest.mark.parametrize("hit_rate", [0.85, 0.15])
@pytest.mark.parametrize(
    "npad,bs,top_k",
    [
        (65536, 16, 1024),  # cs=8 cluster tp (pro-like)
        # cs=8 cluster tp, K=2048 (v32-like); npad pinned to 65536 so the
        # JIT variant is shared with the overflow test (route/cs and the
        # compiled kernel key on npad-derived cs, not on npad itself)
        (65536, 16, 2048),
        (131136, 128, 512),  # cs=1 streaming tp (flash 512k production shape)
    ],
)
def test_bsx_tp_admission_hitrate(npad, bs, top_k, hit_rate):
    """High-hr rows must admit (1-pass) and low-hr rows must stay exact via
    the pivot/secant fallback; both paths must produce exact top-K."""
    _skip_if_cluster_capped(bs, npad, top_k)
    logits, pre_idx, seq_lens = _make_bsx_inputs(
        bs, npad, top_k, seed=npad + bs + int(hit_rate * 100), varlen=True, hit_rate=hit_rate
    )
    _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, "tp")
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


@skip_not_sm100
def test_bsx_tp_admission_tie_plateau():
    """Tie plateau AT the admission threshold: K/2 distinct high values +
    a 3K-wide exact-tie plateau straddling the K-th rank. The hint set is
    the true top-K, so the ladder rungs land ON the plateau value and the
    admitted candidate set is tie-degenerate; the P4 tie-ticket emit must
    still return an exact value multiset."""
    bs, npad, top_k = 16, 65536, 1024  # cs=8 tp
    _skip_if_cluster_capped(bs, npad, top_k)
    torch.manual_seed(31)
    torch.cuda.manual_seed(31)
    logits = -torch.rand(bs, npad, dtype=torch.float32, device="cuda") - 1.0
    for r in range(bs):
        perm = torch.randperm(npad, device="cuda")
        hi = perm[: top_k // 2]
        plateau = perm[top_k // 2 : top_k // 2 + 3 * top_k]
        logits[r, hi] = 5.0 + torch.arange(top_k // 2, device="cuda").float() * 0.01
        logits[r, plateau] = 1.0  # exact ties straddling rank K
    seq_lens = torch.full((bs,), npad * CR, dtype=torch.int32, device="cuda")
    pre_idx = logits.topk(top_k, dim=-1).indices.int().contiguous()
    _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, "tp")
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


@skip_not_sm100
@pytest.mark.parametrize("top_k,n_plateau", [(1024, 8000), (2048, 12000)])
def test_bsx_tp_admission_overflow_fallback(top_k, n_plateau):
    """count(>= any admissible threshold) > kC: K-1 distinct high values,
    then an n_plateau-wide exact-tie plateau (n_plateau > kC = 6144/8192).
    Every threshold at or below the plateau overflows the candidate buffer
    (the uncapped push counter detects it) and every threshold above it
    undershoots (K-1 < K), so admission can never accept and the kernel
    must fall through to the max-below plateau-descent path and its direct
    emit. Exact answer: the K-1 highs plus exactly one plateau member."""
    bs, npad = 16, 65536  # cs=8 tp
    _skip_if_cluster_capped(bs, npad, top_k)
    torch.manual_seed(37)
    torch.cuda.manual_seed(37)
    logits = -torch.rand(bs, npad, dtype=torch.float32, device="cuda") - 1.0
    for r in range(bs):
        perm = torch.randperm(npad, device="cuda")
        hi = perm[: top_k - 1]
        plateau = perm[top_k - 1 : top_k - 1 + n_plateau]
        logits[r, hi] = 5.0 + torch.arange(top_k - 1, device="cuda").float() * 0.01
        logits[r, plateau] = 1.0
    seq_lens = torch.full((bs,), npad * CR, dtype=torch.int32, device="cuda")
    pre_idx = logits.topk(top_k, dim=-1).indices.int().contiguous()
    _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, "tp")
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


@skip_not_sm100
def test_bsx_tp_admission_mixed_batch():
    """Ragged batch mixing rows that admit (true-top-K hints) with rows
    that must fall back (all-zero cold-start hints): per-row admission
    escape is independent, so every row must stay exact regardless of
    which path its cluster takes."""
    bs, npad, top_k = 16, 65536, 1024  # cs=8 tp
    _skip_if_cluster_capped(bs, npad, top_k)
    logits, pre_idx, seq_lens = _make_bsx_inputs(
        bs, npad, top_k, seed=41, varlen=True, hit_rate=0.9
    )
    pre_idx[1::2] = 0  # odd rows: cold-start (degenerate ladder -> fallback)
    _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, "tp")
    out = _run_op(logits, pre_idx, seq_lens, top_k)
    _tie_aware_check(out, logits, seq_lens, top_k)


# ---------------------------------------------------------------------------
# Dispatcher fallback: unsupported inputs must route to the in-tree kernel
# and still produce its results (op contract unchanged).
# ---------------------------------------------------------------------------
@skip_not_sm100
def test_bsx_dispatcher_fallback_bf16():
    bs, npad, top_k = 4, 65536, 512
    torch.manual_seed(21)
    torch.cuda.manual_seed(21)
    logits = (torch.randn(bs, npad, device="cuda") * 2.0).to(torch.bfloat16)
    seq_lens = torch.full((bs,), npad * CR, dtype=torch.int32, device="cuda")
    pre_idx = torch.zeros(bs, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = logits.float().argmax(dim=-1).int()
    out = torch.empty(bs, top_k, dtype=torch.int32, device="cuda")

    assert not bsx_dispatch.is_bsx_supported(
        logits, pre_idx, seq_lens, out, top_k, NEXT_N, CR, None, None
    ), "bf16 must NOT take the bsx fast path"
    # Guard-only on purpose (host, no JIT): the bf16 in-tree execution the
    # op falls back to is exhaustively covered by
    # test_cute_dsl_gvr_topk_decode.py; compiling that variant here again
    # costs ~10s of CI for no added coverage.


@skip_not_sm100
def test_bsx_dispatcher_fallback_bad_shapes():
    """Host-only: MTP contract violations must fall back to the in-tree
    kernel (which asserts on them) rather than mis-launch a bsx tier."""
    bs, npad, top_k = 4, 65536, 512
    logits = torch.randn(bs * 2, npad, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), npad * 4, dtype=torch.int32, device="cuda")
    pre_idx = torch.zeros(bs, top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(bs * 2, top_k, dtype=torch.int32, device="cuda")
    ok = bsx_dispatch.is_bsx_supported
    # next_n=2 with request-level pre_idx/seq_lens: accepted.
    assert ok(logits, pre_idx, seq_lens, out, top_k, 2, 4, None, None)
    # cr outside {1, 4}: rejected.
    assert not ok(logits, pre_idx, seq_lens, out, top_k, 2, 2, None, None)
    # num_rows not divisible by next_n: rejected.
    assert not ok(logits, pre_idx, seq_lens, out, top_k, 3, 4, None, None)
    # row-level (non-request-level) pre_idx under next_n=2: rejected.
    pre_row = torch.zeros(bs * 2, top_k, dtype=torch.int32, device="cuda")
    assert not ok(logits, pre_row, seq_lens, out, top_k, 2, 4, None, None)
    # row-level seq_lens under next_n=2: rejected.
    sl_row = torch.full((bs * 2,), npad * 4, dtype=torch.int32, device="cuda")
    assert not ok(logits, pre_idx, sl_row, out, top_k, 2, 4, None, None)


# ---------------------------------------------------------------------------
# MTP (next_n > 1) + cr in {1, 4}: exactness on all three tiers, checked
# against BOTH the torch.topk host N_eff/offset simulation
# (``_tie_aware_check``) and a differential in-tree arm — the SAME inputs
# through the in-tree kernel (forced via ``order_row``, which the bsx guard
# rejects), per-row value-multiset equality. Ragged: per-request varlen
# seq_lens make N_eff differ across requests AND (via row % next_n) across
# the MTP rows of one request; tails are poisoned with +1e30.
# ---------------------------------------------------------------------------
def _n_eff_rows(seq_lens, num_rows, next_n, cr):
    r = torch.arange(num_rows, device="cuda")
    sl = seq_lens.to(device="cuda", dtype=torch.long)[r // next_n]
    return (sl - next_n + (r % next_n) + 1) // cr


def _make_mtp_inputs(
    bs_req, next_n, cr, npad, top_k, seed, kind, preidx, hit_rate=0.5, argmax_slot0=True
):
    """(logits [bs_req*next_n, npad] fp32, pre_idx [bs_req, K] int32,
    seq_lens [bs_req] int32) with poisoned per-row tails. ``preidx``:
    'noised' = per-request true-top-K hints (host-simulated cr==1 temporal
    offset: hint = ref_idx - 1) mixed with junk at ``hit_rate``; 'random' =
    out-of-range garbage the kernels must clamp; 'zeros' = cold start.

    ``argmax_slot0=True`` (default) enforces the op contract
    ``pre_idx[..., 0] = per-group argmax`` (over the min-N_eff window,
    mirroring the in-tree test suite). The in-tree kernel REQUIRES this
    invariant for exactness — the differential arm is only valid with it.
    The bsx tiers do not require it (clamp hardening); pass False for
    bsx-only robustness cases."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_rows = bs_req * next_n
    logits = torch.randn(num_rows, npad, dtype=torch.float32, device="cuda") * 2.0
    if kind == "ties":
        logits = (logits * 4.0).round() * 0.25
    lo = (top_k + 1) * cr + next_n  # every row non-degenerate (N_eff > K)
    seq_lens = torch.randint(lo, npad * cr + 1, (bs_req,), dtype=torch.int32, device="cuda")
    n_eff = _n_eff_rows(seq_lens, num_rows, next_n, cr)
    col = torch.arange(npad, device="cuda")
    tail = col[None, :] >= n_eff[:, None]
    logits = torch.where(tail, torch.full_like(logits, 1e30), logits)
    if preidx == "zeros":
        pre_idx = torch.zeros(bs_req, top_k, dtype=torch.int32, device="cuda")
    elif preidx == "random":
        pre_idx = torch.randint(-npad, 4 * npad, (bs_req, top_k), dtype=torch.int32, device="cuda")
    else:  # noised
        valid = torch.where(tail, torch.full_like(logits, float("-inf")), logits)
        ref = valid[::next_n].topk(top_k, dim=-1).indices.int()
        # host-side temporal-offset simulation: the cr==1 kernels read
        # logits[hint + (row % next_n) + 1], so a prev-step hint for a
        # top value at index i is (i - 1) for the first MTP row.
        off = 1 if cr == 1 else 0
        good = (ref - off).clamp(min=0)
        keep = torch.rand(good.shape, device="cuda") < hit_rate
        junk = torch.arange(top_k, dtype=torch.int32, device="cuda").expand(bs_req, -1)
        pre_idx = torch.where(keep, good, junk).contiguous()
    if argmax_slot0:
        min_ne = int(n_eff.min().item())
        pre_idx[:, 0] = logits[::next_n, :min_ne].argmax(dim=-1).int()
    return logits, pre_idx, seq_lens


def _run_mtp_both_arms(logits, pre_idx, seq_lens, top_k, next_n, cr):
    """bsx arm + in-tree differential arm (order_row forces the in-tree
    sort path; the bsx guard rejects order_row). Returns (out_bsx, out_ref)."""
    num_rows = logits.shape[0]
    out = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    assert bsx_dispatch.is_bsx_supported(
        logits, pre_idx, seq_lens, out, top_k, next_n, cr, None, None
    ), "expected the bsx fast path to accept this MTP call"
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre_idx, seq_lens, out, top_k=top_k, next_n=next_n, compress_ratio=cr
    )
    order_row = torch.argsort(seq_lens.long(), descending=True).int().contiguous()
    out_ref = torch.empty_like(out)
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits,
        pre_idx,
        seq_lens,
        out_ref,
        top_k=top_k,
        next_n=next_n,
        compress_ratio=cr,
        order_row=order_row,
    )
    torch.cuda.synchronize()
    return out, out_ref


_MTP_TIER_CELLS = {
    # tier -> (npad, bs_req_base, top_k) — bs_req is scaled so num_rows =
    # bs_req * next_n stays in the tier's route band for every next_n.
    "direct": (4096, 4, 512),
    "reg": (24576, 1, 2048),  # reg(cs=4,tb=512,maxv=4,ar=8) at num_rows < 16
    "tp": (65536, 8, 1024),  # tp takeover at num_rows >= 16 (npad >= 32768)
}

# (tier, next_n, cr) — every (next_n, cr) constexpr pair is a separate JIT
# compile of BOTH arms (~15-34s each), so the full 3-tier x {2,3,4} x {1,4}
# cross is prohibitively slow in CI. Coverage kept: the tp tier (the most
# complex MTP arithmetic: cluster exchange + admission escape) runs the
# full {2,3} x {1,4} cross; direct/reg run complementary (next_n, cr)
# diagonals so each tier still sees odd/even next_n and both cr values.
# next_n=4 is dropped: 2 covers the even/row-sharing arithmetic, 3 covers
# odd division (and is the production MTP depth).
_MTP_COMBOS = [
    ("tp", 2, 1),
    ("tp", 2, 4),
    ("tp", 3, 1),
    ("tp", 3, 4),
    ("direct", 2, 4),
    ("direct", 3, 1),
    ("reg", 2, 1),
    ("reg", 3, 4),
]

_MTP_KINDS = [
    ("randn", "random"),  # random logits + OOR-garbage hints (clamp hardening)
    ("randn", "noised"),  # realistic noised hints (+ host offset simulation)
    ("randn", "zeros"),  # all-zero cold-start hints
    ("ties", "noised"),  # quantized tie plateaus
]


@skip_not_sm100
@pytest.mark.parametrize(
    "tier,next_n,cr", _MTP_COMBOS, ids=[f"{t}-{n}-{c}" for t, n, c in _MTP_COMBOS]
)
@pytest.mark.parametrize("kind,preidx", _MTP_KINDS, ids=[f"{k}-{p}" for k, p in _MTP_KINDS])
def test_bsx_mtp_exactness(tier, next_n, cr, kind, preidx):
    npad, bs_req, top_k = _MTP_TIER_CELLS[tier]
    num_rows = bs_req * next_n
    if tier == "tp" and num_rows < 16:
        bs_req = (16 + next_n - 1) // next_n
        num_rows = bs_req * next_n
    _skip_if_cluster_capped(num_rows, npad, top_k)
    assert bsx_dispatch.route(num_rows, npad, top_k).startswith(tier)
    logits, pre_idx, seq_lens = _make_mtp_inputs(
        bs_req, next_n, cr, npad, top_k, seed=npad + 13 * next_n + cr, kind=kind, preidx=preidx
    )
    out, out_ref = _run_mtp_both_arms(logits, pre_idx, seq_lens, top_k, next_n, cr)
    # Independent torch.topk + host N_eff/offset simulation reference.
    _tie_aware_check(out, logits, seq_lens, top_k, next_n=next_n, compress_ratio=cr)
    # Differential oracle: per-row value multiset equal to the in-tree arm.
    sel = torch.gather(logits, -1, out.long()).sort(-1, descending=True).values
    sel_ref = torch.gather(logits, -1, out_ref.long()).sort(-1, descending=True).values
    assert torch.equal(sel, sel_ref), (
        f"bsx vs in-tree value-multiset mismatch (next_n={next_n}, cr={cr}, tier={tier})"
    )


@skip_not_sm100
@pytest.mark.parametrize("next_n,cr", [(2, 1), (3, 4)])
@pytest.mark.parametrize("preidx", ["zeros", "random"])
def test_bsx_mtp_preidx_hardening(next_n, cr, preidx):
    """bsx-only MTP robustness: hint sets that VIOLATE the op's argmax-
    slot-0 contract (pure zeros / out-of-range garbage) must neither fault
    nor break exactness on the bsx tiers. No differential arm here: the
    in-tree kernel requires the argmax invariant for exactness, the bsx
    tiers deliberately do not (clamp hardening). Cell and (next_n, cr)
    pairs match the reg combos of ``test_bsx_mtp_exactness`` so the JIT
    variants are reused (zero extra compiles)."""
    bs_req, npad, top_k = 1, 24576, 2048  # reg tier at num_rows 2/3
    num_rows = bs_req * next_n
    _skip_if_cluster_capped(num_rows, npad, top_k)
    logits, pre_idx, seq_lens = _make_mtp_inputs(
        bs_req,
        next_n,
        cr,
        npad,
        top_k,
        seed=19 + next_n + cr,
        kind="randn",
        preidx=preidx,
        argmax_slot0=False,
    )
    out = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    assert bsx_dispatch.is_bsx_supported(
        logits, pre_idx, seq_lens, out, top_k, next_n, cr, None, None
    )
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre_idx, seq_lens, out, top_k=top_k, next_n=next_n, compress_ratio=cr
    )
    torch.cuda.synchronize()
    _tie_aware_check(out, logits, seq_lens, top_k, next_n=next_n, compress_ratio=cr)


@skip_not_sm100
@pytest.mark.parametrize("next_n", [2, 3])
@pytest.mark.parametrize("cr", [1, 4])
def test_bsx_mtp_degenerate_rows(next_n, cr):
    """Degenerate MTP requests (N_eff <= K for some or all of the next_n
    rows): identity emit [0..N_eff-1] + -1 pad must hold per ROW, with the
    per-row N_eff = (seq_lens[req] - next_n + row % next_n + 1) // cr."""
    bs_req, npad, top_k = 8, 65536, 1024
    num_rows = bs_req * next_n
    _skip_if_cluster_capped(num_rows, npad, top_k)
    logits, pre_idx, seq_lens = _make_mtp_inputs(
        bs_req, next_n, cr, npad, top_k, seed=17, kind="randn", preidx="noised"
    )
    # Requests 0..3: degenerate / boundary — N_eff spans 0, 1, K-1, K, K+1
    # across their MTP rows.
    for i, sl in enumerate([next_n, cr + next_n, (top_k - 1) * cr + next_n - 1, top_k * cr]):
        seq_lens[i] = sl
    n_eff = _n_eff_rows(seq_lens, num_rows, next_n, cr)
    col = torch.arange(npad, device="cuda")
    tail = col[None, :] >= n_eff.clamp(min=0)[:, None]
    logits = torch.where(tail, torch.full_like(logits, 1e30), logits)

    out, out_ref = _run_mtp_both_arms(logits, pre_idx, seq_lens, top_k, next_n, cr)
    for r in range(num_rows):
        ne = int(n_eff[r].item())
        if ne <= top_k:
            expect = torch.full((top_k,), -1, dtype=torch.int32, device="cuda")
            if ne > 0:
                expect[:ne] = torch.arange(ne, dtype=torch.int32, device="cuda")
            assert torch.equal(out[r], expect), (
                f"row {r} (N_eff={ne}): degenerate identity emit mismatch"
            )
        else:
            # Per-row torch reference: collapse the row's MTP arithmetic
            # into an equivalent next_n=1 seq_lens so the shared checker
            # scans exactly N_eff(r) columns.
            sl_row = seq_lens[r // next_n : r // next_n + 1] - next_n + (r % next_n) + 1
            _tie_aware_check(
                out[r : r + 1], logits[r : r + 1], sl_row, top_k, next_n=1, compress_ratio=cr
            )
    nd = (n_eff > top_k).nonzero().flatten()
    if nd.numel():
        sel = torch.gather(logits[nd], -1, out[nd].long()).sort(-1, descending=True).values
        sel_ref = torch.gather(logits[nd], -1, out_ref[nd].long()).sort(-1, descending=True).values
        assert torch.equal(sel, sel_ref)
