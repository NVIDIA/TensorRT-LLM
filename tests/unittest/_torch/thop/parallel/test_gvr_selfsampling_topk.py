# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exactness tests for the standalone self-sampling GVR top-K decode kernels
(`gvr_topk_decode_self_sampling[_host].py`).

Contract under test (see the host module docstring): batch-uniform host-int
``n_valid`` in compressed index space, fp32 logits with a 64-element-multiple
row stride, output = exact (tie-interchangeable) top-K indices of
``logits[:, :n_valid]`` per row.

Checks per case:
  - tie-aware exactness: the multiset of gathered output values equals the
    ``torch.topk`` value multiset bitwise (signed zeros normalized);
  - output indices are unique and within ``[0, n_valid)``;
  - padding immunity: the padded tail ``[n_valid, npad)`` is filled with huge
    values — any kernel read past ``n_valid`` fails the value comparison.
"""

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for gvr_selfsampling_topk tests", allow_module_level=True)

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("cutlass DSL is required for gvr_selfsampling_topk tests", allow_module_level=True)

if getSMVersion() != 100:
    pytest.skip("self-sampling GVR kernels target Blackwell sm_100", allow_module_level=True)

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
    gvr_topk_decode_self_sampling_host as ss_host,
)

_DEV = "cuda"


def _make_case(batch_size, n_valid, top_k, seed, hit_ratio=0.6):
    """Decode-like fp32 logits + prev-step hint. The padded tail is poisoned
    with +3e38 so any read past n_valid corrupts the top-K values."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    npad = (n_valid + 63) // 64 * 64
    logits = torch.randn((batch_size, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    logits[:, n_valid:] = 3e38

    ref_vals, ref_idx = torch.topk(logits[:, :n_valid].float(), top_k, dim=1, largest=True)
    # hint: argmax first (anchor), then a hit_ratio slice of the true top-K,
    # the rest random valid indices — mirrors the decode-step temporal hint.
    n_hits = int(top_k * hit_ratio)
    rand_fill = torch.randint(
        0, n_valid, (batch_size, top_k), generator=gen, dtype=torch.int32, device=_DEV
    )
    pre_idx = rand_fill.clone()
    pre_idx[:, :n_hits] = ref_idx[:, :n_hits].to(torch.int32)
    indices = torch.full((batch_size, top_k), -1, dtype=torch.int32, device=_DEV)
    return logits, pre_idx, indices, ref_vals


def _check_exact(logits, indices, n_valid, ref_vals):
    top_k = indices.shape[1]
    idx64 = indices.to(torch.int64)
    assert int(idx64.min()) >= 0, "negative output index"
    assert int(idx64.max()) < n_valid, "output index past n_valid"
    for row in range(indices.shape[0]):
        assert int(torch.unique(idx64[row]).numel()) == top_k, f"row {row}: duplicate indices"
    got = torch.gather(logits, 1, idx64)
    # +0.0 maps -0.0 to +0.0 so signed zeros compare equal bitwise
    got_sorted = torch.sort(got + 0.0, dim=1, descending=True).values
    ref_sorted = torch.sort(ref_vals + 0.0, dim=1, descending=True).values
    assert torch.equal(got_sorted, ref_sorted), (
        "top-K value multiset mismatch (inexact or padding read)"
    )


# (top_k, n_valid) — gate-edge (131075/131076 straddle the K=2048 hint-band
# gate), the small-N floor, the mid band, and the deployment-envelope top.
_CASES = [
    (512, 4099),
    (512, 65536),
    (512, 262143),
    (1024, 16387),
    (1024, 131072),
    (2048, 4111),
    (2048, 131075),
    (2048, 131076),
    (2048, 262144),
]


@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize("top_k,n_valid", _CASES, ids=[f"k{k}_n{n}" for k, n in _CASES])
def test_selfsampling_topk_exactness(batch_size, top_k, n_valid):
    logits, pre_idx, indices, ref_vals = _make_case(
        batch_size, n_valid, top_k, seed=n_valid * 31 + top_k + batch_size
    )
    ss_host.run(logits, pre_idx, n_valid, indices)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)


# n_valid <= top_k: every valid position is in the top-K. Production short
# path (heuristicTopKDecode.cu:72-84): identity indices + -1 tail padding.
_SHORT_CASES = [
    (512, 256),
    (512, 511),
    (512, 512),
    (1024, 64),
    (1024, 1024),
    (2048, 1000),
    (2048, 2047),
    (2048, 2048),
]


@pytest.mark.parametrize("batch_size", [1, 4], ids=lambda b: f"bs{b}")
@pytest.mark.parametrize(
    "top_k,n_valid", _SHORT_CASES, ids=[f"k{k}_n{n}" for k, n in _SHORT_CASES]
)
def test_selfsampling_topk_short_path(batch_size, top_k, n_valid):
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    npad = (n_valid + 63) // 64 * 64
    logits = torch.randn((batch_size, npad), generator=gen, dtype=torch.float32, device=_DEV)
    logits[:, n_valid:] = 3e38  # poison pad: the short path must never read it
    pre_idx = torch.randint(
        0, n_valid, (batch_size, top_k), generator=gen, dtype=torch.int32, device=_DEV
    )
    indices = torch.full((batch_size, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run(logits, pre_idx, n_valid, indices)
    torch.cuda.synchronize()
    head = indices[:, :n_valid].to(torch.int64)
    expect = torch.arange(n_valid, dtype=torch.int64, device=_DEV).expand(batch_size, n_valid)
    assert torch.equal(torch.sort(head, dim=1).values, expect), "short-path head not {0..n-1}"
    if n_valid < top_k:
        assert torch.equal(
            indices[:, n_valid:],
            torch.full((batch_size, top_k - n_valid), -1, dtype=torch.int32, device=_DEV),
        ), "short-path tail must be -1 padded"


def test_selfsampling_topk_values_output():
    """Opt-in values output (default None = off, matching dsa.py, which
    allocates the values scratch only for the non-CuTeDSL path): must equal
    the gathered top-K values and the torch.topk value multiset."""
    top_k, n_valid = 512, 8192
    logits, pre_idx, indices, ref_vals = _make_case(2, n_valid, top_k, seed=11)
    values = torch.full((2, top_k), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run(logits, pre_idx, n_valid, indices, values)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)
    assert torch.equal(values, torch.gather(logits, 1, indices.to(torch.int64)))
    assert torch.equal(
        torch.sort(values + 0.0, dim=1, descending=True).values,
        torch.sort(ref_vals + 0.0, dim=1, descending=True).values,
    )


def test_selfsampling_topk_values_short_path():
    """Short path with values: head copies logits, tail pads with -FLT_MAX
    (production heuristicTopKDecode pad convention)."""
    top_k, n_valid, bs = 1024, 512, 4
    gen = torch.Generator(device=_DEV).manual_seed(42)
    logits = torch.randn((bs, n_valid), generator=gen, dtype=torch.float32, device=_DEV)
    pre_idx = torch.randint(
        0, n_valid, (bs, top_k), generator=gen, dtype=torch.int32, device=_DEV
    )
    indices = torch.full((bs, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((bs, top_k), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run(logits, pre_idx, n_valid, indices, values)
    torch.cuda.synchronize()
    assert torch.equal(values[:, :n_valid], logits)
    fmin = torch.finfo(torch.float32).min
    assert bool((values[:, n_valid:] == fmin).all())
    assert bool((indices[:, n_valid:] == -1).all())


@pytest.mark.parametrize("hint_kind", ["all_zero", "all_same", "all_max", "half_dup"])
@pytest.mark.parametrize(
    "top_k,n_valid", [(512, 8192), (2048, 131075)], ids=["k512_n8192", "k2048_n131075"]
)
def test_selfsampling_topk_degenerate_hints(hint_kind, top_k, n_valid):
    """Hints only steer the sampling ladder — exactness must survive the
    degenerate hint buffers production can produce: the all-zero cold start
    (dsa.py ``heuristic_prev_topk.zero_()`` init corners), fully duplicated
    hints, and max-index hints (hint-robustness class of PR #17550)."""
    logits, _, indices, ref_vals = _make_case(2, n_valid, top_k, seed=n_valid + top_k)
    if hint_kind == "all_zero":
        pre_idx = torch.zeros((2, top_k), dtype=torch.int32, device=_DEV)
    elif hint_kind == "all_same":
        pre_idx = torch.full((2, top_k), 1234, dtype=torch.int32, device=_DEV)
    elif hint_kind == "all_max":
        pre_idx = torch.full((2, top_k), n_valid - 1, dtype=torch.int32, device=_DEV)
    else:
        gen = torch.Generator(device=_DEV).manual_seed(1)
        pre_idx = torch.randint(
            0, n_valid, (2, top_k), generator=gen, dtype=torch.int32, device=_DEV
        )
        pre_idx[:, top_k // 2 :] = pre_idx[:, :1]
    ss_host.run(logits, pre_idx, n_valid, indices)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)


def _run_varlen_case(kv, next_n, cr, top_k, seed, with_values=False, engine="auto"):
    """Build a per-row-poisoned varlen batch, run run_varlen, verify every
    row against its own n_r (production formula) — short rows included."""
    batch, rows = len(kv), len(kv) * next_n
    n_r = [(kv[r // next_n] - next_n + (r % next_n) + 1) // cr for r in range(rows)]
    npad = (max(n_r) + 63) // 64 * 64
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    logits = torch.randn((rows, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    for r in range(rows):
        logits[r, n_r[r] :] = 3e38  # poison beyond each row's OWN n_r
    pre_idx = torch.empty((batch, top_k), dtype=torch.int32, device=_DEV)
    for q in range(batch):
        nmin = max(min(n_r[q * next_n : (q + 1) * next_n]), 1)
        pre_idx[q] = torch.randint(0, nmin, (top_k,), generator=gen, dtype=torch.int32, device=_DEV)
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = (
        torch.full((rows, top_k), 7.0, dtype=torch.float32, device=_DEV) if with_values else None
    )
    kv_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(
        logits, pre_idx, kv_lens, indices,
        next_n=next_n, compress_ratio=cr, values=values, engine=engine,
    )
    torch.cuda.synchronize()
    fmin = torch.finfo(torch.float32).min
    for r in range(rows):
        n = n_r[r]
        if n <= top_k:
            head = indices[r, :n].to(torch.int64)
            assert torch.equal(torch.sort(head).values, torch.arange(n, device=_DEV))
            assert bool((indices[r, n:] == -1).all())
            if values is not None:
                assert torch.equal(values[r, :n], logits[r, :n])
                assert bool((values[r, n:] == fmin).all())
        else:
            idx = indices[r].to(torch.int64)
            assert int(idx.min()) >= 0 and int(idx.max()) < n
            assert int(torch.unique(idx).numel()) == top_k
            ref = torch.topk(logits[r, :n], top_k).values
            got = torch.sort(torch.gather(logits[r], 0, idx) + 0.0, descending=True).values
            assert torch.equal(got, torch.sort(ref + 0.0, descending=True).values), f"row {r} inexact"
            if values is not None:
                assert torch.equal(values[r], torch.gather(logits[r], 0, idx))


@pytest.mark.parametrize("engine", ["auto", "reference"])
@pytest.mark.parametrize(
    "kv,next_n,cr,top_k",
    [
        ([33000, 8200, 300], 1, 1, 512),  # v3.2-style heterogeneous + short row
        ([131075, 32800, 2000], 1, 4, 512),  # v4-style compressed index space
        ([9000, 5001], 2, 1, 512),  # MTP: n varies per row within a request
        ([65540], 4, 4, 1024),  # MTP: compressed-boundary-crossing rows
    ],
    ids=["cr1_hetero_short", "cr4_hetero_short", "cr1_mtp2", "cr4_mtp4"],
)
def test_selfsampling_topk_varlen(kv, next_n, cr, top_k, engine):
    """run_varlen production contract: per-row n from device kv_lens with the
    MTP window formula, request-level hints, per-row short path — on BOTH the
    per-row in-kernel engine ("auto") and the b=1 reference loop."""
    _run_varlen_case(kv, next_n, cr, top_k, seed=sum(kv) + next_n + cr, engine=engine)


def test_selfsampling_topk_varlen_engine_matches_reference():
    """Differential: the in-kernel engine's per-row value multisets must
    equal the reference loop's on a mixed batch (deep SPLIT rows, tsh band,
    short rows, compressed space)."""
    kv = [524288, 131075, 32800, 2000, 65540, 8192, 262144, 900]
    top_k, next_n, cr = 1024, 1, 4
    rows = len(kv)
    n_r = [(v - 1 + 1) // cr for v in kv]
    npad = (max(n_r) + 63) // 64 * 64
    gen = torch.Generator(device=_DEV).manual_seed(77)
    logits = torch.randn((rows, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    for r in range(rows):
        logits[r, n_r[r] :] = 3e38
    pre_idx = torch.empty((rows, top_k), dtype=torch.int32, device=_DEV)
    for q in range(rows):
        pre_idx[q] = torch.randint(
            0, max(n_r[q], 1), (top_k,), generator=gen, dtype=torch.int32, device=_DEV
        )
    kv_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    out_a = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    out_r = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, pre_idx, kv_lens, out_a, compress_ratio=cr)
    ss_host.run_varlen(logits, pre_idx, kv_lens, out_r, compress_ratio=cr, engine="reference")
    torch.cuda.synchronize()
    for r in range(rows):
        if n_r[r] <= top_k:
            assert torch.equal(out_a[r], out_r[r]) or torch.equal(
                torch.sort(out_a[r]).values, torch.sort(out_r[r]).values
            )
        else:
            ga = torch.sort(
                torch.gather(logits[r], 0, out_a[r].to(torch.int64)) + 0.0, descending=True
            ).values
            gr = torch.sort(
                torch.gather(logits[r], 0, out_r[r].to(torch.int64)) + 0.0, descending=True
            ).values
            assert torch.equal(ga, gr), f"row {r}: engine != reference"


def test_selfsampling_topk_varlen_values():
    _run_varlen_case([40000, 1900], 2, 4, 512, seed=5, with_values=True)


def test_selfsampling_topk_varlen_guards():
    logits = torch.randn((2, 8192), dtype=torch.float32, device=_DEV)
    pre_idx = torch.zeros((2, 512), dtype=torch.int32, device=_DEV)
    indices = torch.zeros((2, 512), dtype=torch.int32, device=_DEV)
    kv = torch.tensor([8192, 8192], dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="kv_lens length"):
        ss_host.run_varlen(logits, pre_idx, kv[:1], indices)
    with pytest.raises(RuntimeError, match="not divisible"):
        ss_host.run_varlen(logits, pre_idx, kv, indices, next_n=3)
    with pytest.raises(RuntimeError, match="compress_ratio"):
        ss_host.run_varlen(logits, pre_idx, kv, indices, compress_ratio=2)
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        ss_host.run_varlen(logits, pre_idx, kv.cpu(), indices)


def test_selfsampling_topk_run_ws_explicit_workspace():
    """run_ws with a caller-owned workspace must agree with run()."""
    top_k, n_valid = 1024, 65536
    logits, pre_idx, indices, ref_vals = _make_case(2, n_valid, top_k, seed=7)
    ws = torch.zeros(ss_host.workspace_bytes(), dtype=torch.uint8, device=_DEV)
    ss_host.run_ws(logits, pre_idx, n_valid, indices, ws)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)


def test_selfsampling_topk_guards():
    logits, pre_idx, indices, _ = _make_case(1, 8192, 512, seed=3)
    with pytest.raises(RuntimeError, match="float32"):
        ss_host.run(logits.to(torch.bfloat16), pre_idx, 8192, indices)
    with pytest.raises(RuntimeError, match="non-negative"):
        ss_host.run(logits, pre_idx, -1, indices)
    with pytest.raises(RuntimeError, match="batch dims"):
        ss_host.run(logits, pre_idx[:0], 8192, indices)


def test_selfsampling_route_factorization():
    """Two-time-scale dispatch groundwork: route() must factor losslessly
    into route_static (constant on n-bands, freezable at capture time) and
    route_dynamic (the n-continuous scalars the per-row device engine will
    recompute) — recombining them reproduces route() exactly. CPU-only."""
    npad = 1 << 20
    checked = 0
    for b in (1, 8, 16, 64, 148, 296, 1024):
        for k in (512, 1024, 2048):
            ns = set()
            for c in (
                2 * k, 3 * k, 4 * k + 64, 2560, 4096, 8192, 16384,
                4 * 1024, 4 * 4096, 4 * 32768, 65536, 131072, 262144,
            ):
                ns.update(v for v in range(c - 4, c + 5) if k < v <= npad)
            ns.update(range(k + 1, npad + 1, 4999))
            s = 12345
            for _ in range(400):
                s = (s * 1103515245 + 12345) % (1 << 31)
                ns.add(k + 1 + s % (npad - k - 1))
            for n in sorted(ns):
                assert ss_host.route_split(b, n, npad, k) == ss_host.route(b, n, npad, k), (
                    b, n, k,
                )
                checked += 1
    assert checked > 10_000


def test_selfsampling_route_bands_contiguous():
    """route_bands must tile the envelope contiguously with n-free statics."""
    bands = ss_host.route_bands(8, 262144, 1024)
    assert bands[0][0] == 1025 and bands[-1][1] == 262144
    for (_, h1, _), (l2, _, _) in zip(bands, bands[1:]):
        assert l2 == h1 + 1
    for _, _, st in bands:
        for f in ss_host._DYN_RT[st["kernel"]]:
            assert f not in st["rt"]


def test_selfsampling_dispatch_is_pure_and_total():
    """route(b, n, npad, k) must return a plan for every in-envelope shape."""
    for k in (512, 1024, 2048):
        for n in (k + 1, 4111, 65536, 131075, 131076, 262144):
            npad = (n + 63) // 64 * 64
            r = ss_host.route(4, n, npad, k)
            assert r["kernel"] in ("main", "reg", "clus", "reg_clus")
            assert r["block"] >= 128 and r["grid"][0] >= 1
