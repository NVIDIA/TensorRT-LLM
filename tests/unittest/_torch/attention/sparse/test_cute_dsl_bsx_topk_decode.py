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
"""BSX (op43 direct/reg/tp CuTe DSL tiers) top-K decode tests.

CI-sized exactness grid for the guarded fp32/next_n=1/cr=4 fast path inside
``trtllm::cute_dsl_gvr_topk_decode``: every reg launch-table instance once,
the direct and tp tiers at a few npad each, ragged (varlen) rows with
POISONED tails (stale-garbage simulation: +1e30 beyond N_eff, which would
dominate the top-K if the ragged-N masking were broken), quantized-tie
inputs, degenerate rows, pre_idx hardening, host-only route-table asserts,
and a dispatcher-fallback check (bf16 / next_n=2 route to the in-tree
kernel).
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

CR = 4  # v1 dispatcher guard: compress_ratio == 4 (DSv4)
NEXT_N = 1  # v1 dispatcher guard: next_n == 1


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
# Host-only route-table asserts (mirror of the op42 gvr_bsx.cu dispatch).
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
# reg tier: every launch-table instance once (cs, tb, maxv, ar), ragged rows
# with poisoned tails. The two dense cs=2/cs=4 instances are reachable only
# through the TRTLLM_BSX_DENSE_BS knob (default bands route around them),
# which doubles as the env-knob dispatch test on a live launch.
# ---------------------------------------------------------------------------
_REG_INSTANCES = [
    # (npad, bs, K, dense_env, expected tier)
    (14336, 1, 512, None, "reg(cs=1,tb=512,maxv=8,ar=8)"),
    (24576, 1, 512, None, "reg(cs=4,tb=512,maxv=4,ar=8)"),
    (49152, 1, 512, None, "reg(cs=8,tb=512,maxv=3,ar=8)"),
    (65536, 1, 1024, None, "reg(cs=8,tb=512,maxv=4,ar=8)"),
    (131072, 8, 512, None, "reg(cs=8,tb=512,maxv=8,ar=8)"),
    (163840, 1, 2048, None, "reg(cs=16,tb=512,maxv=5,ar=6)"),
    (163840, 4, 512, None, "reg(cs=16,tb=512,maxv=5,ar=8)"),
    (262144, 1, 2048, None, "reg(cs=16,tb=512,maxv=8,ar=8)"),
    (262144, 4, 1024, None, "reg(cs=16,tb=512,maxv=8,ar=6)"),
    (20480, 64, 512, None, "reg(cs=1,tb=1024,maxv=5,ar=8)"),
    (28672, 64, 512, None, "reg(cs=1,tb=1024,maxv=8,ar=8)"),
    (65536, 8, 512, "8", "reg(cs=2,tb=1024,maxv=8,ar=8)"),
    (131072, 8, 1024, "8", "reg(cs=4,tb=1024,maxv=8,ar=8)"),
    (262144, 8, 2048, None, "reg(cs=8,tb=1024,maxv=8,ar=8)"),
]


@skip_not_sm100
@pytest.mark.parametrize(
    "npad,bs,top_k,dense_env,expected",
    _REG_INSTANCES,
    ids=[t[4] + f"_n{t[0]}_bs{t[1]}_k{t[2]}" for t in _REG_INSTANCES],
)
@pytest.mark.parametrize("kind", ["randn", "ties"])
def test_bsx_reg_launch_table(npad, bs, top_k, dense_env, expected, kind, monkeypatch):
    _skip_if_cluster_capped(bs, npad, top_k)
    try:
        if dense_env is not None:
            monkeypatch.setenv("TRTLLM_BSX_DENSE_BS", dense_env)
            bsx_dispatch._reset_env_cache()
        logits, pre_idx, seq_lens = _make_bsx_inputs(
            bs, npad, top_k, seed=npad + bs + top_k, kind=kind, varlen=True
        )
        _assert_bsx_routes(logits, pre_idx, seq_lens, top_k, expected)
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
# [K, kC], so high-hit-rate rows finish in one fused streaming pass. These
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
        (32832, 16, 2048),  # cs=4 cluster tp (v32 32k production shape)
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
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre_idx, seq_lens, out, top_k=top_k, next_n=NEXT_N, compress_ratio=CR
    )
    torch.cuda.synchronize()
    _tie_aware_check(out, logits, seq_lens, top_k)


@skip_not_sm100
def test_bsx_dispatcher_fallback_next_n2():
    """next_n=2 (cr=1, V3.2-style MTP rows) must fall back to the in-tree
    kernel and produce its results."""
    bs, npad, top_k, next_n, cr = 4, 65536, 2048, 2, 1
    num_rows = bs * next_n
    torch.manual_seed(23)
    torch.cuda.manual_seed(23)
    logits = (torch.randn(num_rows, npad, device="cuda") * 2.0).contiguous()
    seq_lens = torch.full((bs,), npad, dtype=torch.int32, device="cuda")
    # cr=1 kernel convention: it reads logits[pre_idx + (row % next_n) + 1].
    eff = npad - next_n  # safe hint range for every row
    pre_idx = torch.zeros(bs, top_k, dtype=torch.int32, device="cuda")
    pre_idx[:, 0] = logits[::next_n, :eff].argmax(dim=-1).int() - 1
    pre_idx[:, 1:] = torch.arange(1, top_k, dtype=torch.int32, device="cuda")
    out = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    assert not bsx_dispatch.is_bsx_supported(
        logits, pre_idx, seq_lens, out, top_k, next_n, cr, None, None
    ), "next_n=2 must NOT take the bsx fast path"
    torch.ops.trtllm.cute_dsl_gvr_topk_decode(
        logits, pre_idx, seq_lens, out, top_k=top_k, next_n=next_n, compress_ratio=cr
    )
    torch.cuda.synchronize()
    _tie_aware_check(out, logits, seq_lens, top_k, next_n=next_n, compress_ratio=cr)
