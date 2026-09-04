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

if getSMVersion() not in (100, 103):
    pytest.skip(
        "self-sampling GVR kernels target datacenter Blackwell (sm_100/103) "
        "— same gate as the production dispatch; consumer Blackwell "
        "(sm_120/121) lacks thread-block clusters",
        allow_module_level=True,
    )

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
# path convention: identity indices + -1 tail padding.
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
@pytest.mark.parametrize("top_k,n_valid", _SHORT_CASES, ids=[f"k{k}_n{n}" for k, n in _SHORT_CASES])
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
    pre_idx = torch.randint(0, n_valid, (bs, top_k), generator=gen, dtype=torch.int32, device=_DEV)
    indices = torch.full((bs, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((bs, top_k), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run(logits, pre_idx, n_valid, indices, values)
    torch.cuda.synchronize()
    assert torch.equal(values[:, :n_valid], logits)
    fmin = torch.finfo(torch.float32).min
    assert bool((values[:, n_valid:] == fmin).all())
    assert bool((indices[:, n_valid:] == -1).all())


@pytest.mark.parametrize(
    "hint_kind", ["all_zero", "all_same", "all_max", "half_dup", "minus_one_tail", "all_minus_one"]
)
@pytest.mark.parametrize(
    "top_k,n_valid", [(512, 8192), (2048, 131075)], ids=["k512_n8192", "k2048_n131075"]
)
def test_selfsampling_topk_degenerate_hints(hint_kind, top_k, n_valid):
    """Hints only steer the sampling ladder — exactness must survive the
    degenerate hint buffers production can produce: the all-zero cold start
    (dsa.py ``heuristic_prev_topk.zero_()`` init corners), fully duplicated
    hints, and max-index hints."""
    logits, _, indices, ref_vals = _make_case(2, n_valid, top_k, seed=n_valid + top_k)
    if hint_kind == "all_zero":
        pre_idx = torch.zeros((2, top_k), dtype=torch.int32, device=_DEV)
    elif hint_kind == "all_same":
        pre_idx = torch.full((2, top_k), 1234, dtype=torch.int32, device=_DEV)
    elif hint_kind == "all_max":
        pre_idx = torch.full((2, top_k), n_valid - 1, dtype=torch.int32, device=_DEV)
    elif hint_kind == "minus_one_tail":
        # production short-row pad convention writes -1 tails into prev_topk;
        # the next step feeds them back as hints — must stay in-bounds
        gen = torch.Generator(device=_DEV).manual_seed(2)
        pre_idx = torch.randint(
            0, n_valid, (2, top_k), generator=gen, dtype=torch.int32, device=_DEV
        )
        pre_idx[:, top_k // 3 :] = -1
    elif hint_kind == "all_minus_one":
        pre_idx = torch.full((2, top_k), -1, dtype=torch.int32, device=_DEV)
    else:
        gen = torch.Generator(device=_DEV).manual_seed(1)
        pre_idx = torch.randint(
            0, n_valid, (2, top_k), generator=gen, dtype=torch.int32, device=_DEV
        )
        pre_idx[:, top_k // 2 :] = pre_idx[:, :1]
    ss_host.run(logits, pre_idx, n_valid, indices)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n_valid, ref_vals)


@pytest.mark.parametrize("n_valid", [3072, 4096], ids=["n3072", "n4096"])
def test_selfsampling_topk_high_anchor_hint_completeness(n_valid):
    """Anchor-only hints whose gathered values all sit ABOVE the true k-th
    value (an argmax anchor over the all-zero cold-start buffer, with a high
    row head) bracket the sampling band so it contains fewer than top_k
    entries. The classify histogram then never reaches k and the crossing
    scan pins its bin-0 fallback as a fake crossing; the register-family
    whole-bin emit must escape to the key-space ranking instead of
    stopping at the histogram total (regression: rows exited with
    out[tot:k) unwritten -- a prefix-only write). Deterministic worst case:
    row[0] = second-max, so the bracket holds exactly two entries. The
    cells pin the vulnerable reg variant (hint-driven bracket + no-clamp
    classify: BRL variants clamp out-of-bracket values into bin 0 and
    cannot under-count, so batch/shape are chosen to compile BRL off).

    The production hint-free bracket cannot under-count (its k source
    values sit inside the band by construction), so this exercises the
    hinted codegen through the batch-uniform TESTING/BENCH entry."""
    top_k = 512
    bs = 256
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((bs, n_valid), generator=gen, dtype=torch.float32, device=_DEV)
    v2 = torch.topk(logits, 2, dim=1).values[:, 1]
    logits[:, 0] = v2  # row head = second-max: bracket = [second-max, max]
    ref_vals, _ = torch.topk(logits, top_k, dim=1)
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device=_DEV)
    pre_idx[:, 0] = logits.argmax(dim=1).to(torch.int32)
    indices = torch.full((bs, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run(logits, pre_idx, n_valid, indices)
    torch.cuda.synchronize()
    assert int((indices == -7).sum()) == 0, "unwritten output slots (prefix-only emit)"
    _check_exact(logits, indices, n_valid, ref_vals)


def test_selfsampling_topk_neginf_tail_completeness():
    """The DEG bracket arm folds the row tail element (the last n % 4
    columns live outside the float4 register batch) into the bracket
    without the > -inf guard the vector loop has: a single in-window -inf
    there drags the bracket low edge to -inf, every classify product
    becomes NaN, the histogram total is zero, and the row exited having
    written nothing (regression: hint-independent zero-write rows). The
    escape such rows now take must also not double-count the tail element
    when the tie class is the -inf key itself (regression: duplicate
    indices from the -inf fill lanes of the last partial float4) -- odd
    rows keep fewer than top_k finite entries to exercise that lane
    bound."""
    top_k = 1024
    bs, npad, n_valid = 256, 4096, 4093  # n_valid % 4 = 1: one tail column
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((bs, npad), generator=gen, dtype=torch.float32, device=_DEV)
    logits[:, n_valid:] = 3e38  # poison past the window
    logits[:, n_valid - 1] = float("-inf")  # in-window -inf in the tail column
    logits[1::2, 500:n_valid] = float("-inf")  # odd rows: n_finite < top_k
    masked = logits.clone()
    masked[:, n_valid:] = float("-inf")
    ref_vals, _ = torch.topk(masked, top_k, dim=1)
    indices = torch.full((bs, top_k), -7, dtype=torch.int32, device=_DEV)
    kv = torch.full((bs,), n_valid, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, kv, indices, max_seq_len=npad)
    torch.cuda.synchronize()
    assert int((indices == -7).sum()) == 0, "unwritten output slots (zero-write rows)"
    _check_exact(logits, indices, n_valid, ref_vals)


@pytest.mark.parametrize("pos", [1000, 3000], ids=["in_window", "out_of_window"])
def test_selfsampling_topk_posinf_completeness(pos):
    """A +inf in the register-family fold window drives the bracket max to
    +inf, so the bracket width GMAX-Tv=+inf and SC=rcp(+inf)=0 fold every
    value into bin 0; the whole-bin emit then drops the +inf from the top-k
    (regression, DKG issue #58). The infinite-width bracket must be rejected
    by the collapse guard and take the key-space escape, where fkey(+inf) is
    the maximum key. Both an in-window and an out-of-window +inf are pinned;
    N=4096 k=1024 keeps the register 'reg' family (not the streaming tiers,
    which never collapse this way)."""
    top_k = 1024
    n_valid = 4096
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((1, n_valid), generator=gen, dtype=torch.float32, device=_DEV) * 2.0
    logits[0, pos] = float("inf")
    ref_vals, _ = torch.topk(logits, top_k, dim=1)
    indices = torch.full((1, top_k), -7, dtype=torch.int32, device=_DEV)
    kv = torch.full((1,), n_valid, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, kv, indices, max_seq_len=n_valid)
    torch.cuda.synchronize()
    assert int((indices == -7).sum()) == 0, "unwritten output slots"
    assert torch.isinf(logits[0][indices[0].long()]).any(), "+inf dropped from the top-k"
    _check_exact(logits, indices, n_valid, ref_vals)


def test_selfsampling_topk_posinf_regclus_completeness() -> None:
    """A collapsed reg_clus bracket must use its whole-row key-space
    fallback instead of emitting from the lossy float-space histogram."""
    batch_size, n_valid, top_k = 4, 32768, 1024
    assert ss_host.route(batch_size, n_valid, n_valid, top_k)["kernel"] == "reg_clus"
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((batch_size, n_valid), generator=gen, dtype=torch.float32, device=_DEV)
    logits[:, 1000] = float("inf")
    ref_vals, _ = torch.topk(logits, top_k, dim=1)
    indices = torch.full((batch_size, top_k), -7, dtype=torch.int32, device=_DEV)
    kv = torch.full((batch_size,), n_valid, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, kv, indices, max_seq_len=n_valid)
    torch.cuda.synchronize()
    assert int((indices == -7).sum()) == 0, "unwritten output slots"
    assert torch.isposinf(logits.gather(1, indices.long())).any(dim=1).all()
    _check_exact(logits, indices, n_valid, ref_vals)


def _run_varlen_case(kv, next_n, cr, top_k, seed, with_values=False):
    """Build a per-row-poisoned varlen batch, run run_varlen, verify every
    row against its own n_r (production formula) — short rows included."""
    rows = len(kv) * next_n
    n_r = [(kv[r // next_n] - next_n + (r % next_n) + 1) // cr for r in range(rows)]
    npad = (max(n_r) + 63) // 64 * 64
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    logits = torch.randn((rows, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    for r in range(rows):
        logits[r, n_r[r] :] = 3e38  # poison beyond each row's OWN n_r
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = (
        torch.full((rows, top_k), 7.0, dtype=torch.float32, device=_DEV) if with_values else None
    )
    kv_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        next_n=next_n,
        compress_ratio=cr,
        values=values,
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
            assert torch.equal(got, torch.sort(ref + 0.0, descending=True).values), (
                f"row {r} inexact"
            )
            if values is not None:
                assert torch.equal(values[r], torch.gather(logits[r], 0, idx))


def _reference_varlen_indices(logits, kv_lens, next_n, compress_ratio, top_k):
    """Build a simple torch reference for the hint-free varlen contract."""
    reference = torch.full((logits.shape[0], top_k), -1, dtype=torch.int32, device=logits.device)
    lengths = kv_lens.tolist()
    for row in range(logits.shape[0]):
        valid = (
            max(
                lengths[row // next_n] - next_n + row % next_n + 1,
                0,
            )
            // compress_ratio
        )
        valid = min(valid, logits.shape[1])
        if valid <= 0:
            continue
        if valid <= top_k:
            reference[row, :valid] = torch.arange(valid, dtype=torch.int32, device=logits.device)
        else:
            reference[row] = torch.topk(logits[row, :valid], top_k).indices.to(torch.int32)
    return reference


@pytest.mark.parametrize(
    "kv,next_n,cr,top_k",
    [
        ([33000, 8200, 300], 1, 1, 512),  # v3.2-style heterogeneous + short row
        ([131075, 32800, 2000], 1, 4, 512),  # v4-style compressed index space
        ([9000, 5001], 2, 1, 512),  # MTP: n varies per row within a request
        ([65540], 4, 4, 1024),  # MTP: compressed-boundary-crossing rows
        ([40000, 7003], 3, 4, 512),  # MTP2 (next_n=3): unused in production
        # today, but the window formula must generalize
    ],
    ids=["cr1_hetero_short", "cr4_hetero_short", "cr1_mtp2", "cr4_mtp4", "cr4_mtp3"],
)
def test_selfsampling_topk_varlen(kv, next_n, cr, top_k):
    """Validate the hint-free per-row KV-length and MTP window contract."""
    _run_varlen_case(kv, next_n, cr, top_k, seed=sum(kv) + next_n + cr)


def test_selfsampling_topk_varlen_values():
    _run_varlen_case([40000, 1900], 2, 4, 512, seed=5, with_values=True)


@pytest.mark.parametrize(
    "rows,base,step,top_k",
    [
        (16, 40000, 977, 1024),  # R=9 SPLIT + per-row TSH runtime gate (tsh_en=1)
        (200, 6000, 64, 512),  # BLK=512, SPLIT=False, non-big launch mode
    ],
    ids=["b16_tsh_split", "b200_blk512_nonsplit"],
)
def test_selfsampling_topk_varlen_launch_modes(rows, base, step, top_k):
    """Exercise the varlen kernel's other launch modes: the SPLIT + per-row
    TSH-floor runtime gate domain (16 <= b <= 74, k <= 1024) and the
    BLK=512 non-split wide-batch plan."""
    _run_varlen_case([base + step * i for i in range(rows)], 1, 1, top_k, seed=rows)


def test_selfsampling_topk_varlen_zero_kv_slot():
    """Padded / evicted CUDA-graph request slots can carry kv_len < next_n
    (even 0): the engine must emit the empty short row (all -1), not raise —
    mixed with live MTP rows of another request in the same launch."""
    gen = torch.Generator(device=_DEV).manual_seed(9)
    logits = torch.randn((8, 8192), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    kv_lens = torch.tensor([0, 8192], dtype=torch.int32, device=_DEV)
    indices = torch.full((8, 512), -7, dtype=torch.int32, device=_DEV)
    for r in range(4, 8):
        n = 8192 - 4 + (r - 4) + 1
        logits[r, n:] = 3e38
    ss_host.run_varlen(logits, kv_lens, indices, next_n=4, compress_ratio=1)
    torch.cuda.synchronize()
    assert bool((indices[:4] == -1).all()), "kv=0 rows must be all -1"
    for r in range(4, 8):
        n = 8192 - 4 + (r - 4) + 1
        idx = indices[r].to(torch.int64)
        assert int(idx.min()) >= 0 and int(idx.max()) < n
        ref = torch.topk(logits[r, :n], 512).values
        got = torch.sort(torch.gather(logits[r], 0, idx) + 0.0, descending=True).values
        assert torch.equal(got, torch.sort(ref + 0.0, descending=True).values)


def test_selfsampling_topk_varlen_guards():
    logits = torch.randn((2, 8192), dtype=torch.float32, device=_DEV)
    indices = torch.zeros((2, 512), dtype=torch.int32, device=_DEV)
    kv = torch.tensor([8192, 8192], dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="kv_lens length"):
        ss_host.run_varlen(logits, kv[:1], indices)
    with pytest.raises(RuntimeError, match="not divisible"):
        ss_host.run_varlen(logits, kv, indices, next_n=3)
    with pytest.raises(RuntimeError, match="compress_ratio"):
        ss_host.run_varlen(logits, kv, indices, compress_ratio=2)
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        ss_host.run_varlen(logits, kv.cpu(), indices)
    with pytest.raises(RuntimeError, match="num_rows"):
        # request-level-shaped indices under MTP: MUST be rejected (the
        # kernel grid comes from logits rows — silent OOB writes otherwise)
        ss_host.run_varlen(logits, kv[:1], indices[:1], next_n=2)
    with pytest.raises(RuntimeError, match="contiguous"):
        strided = torch.zeros((2, 2), dtype=torch.int32, device=_DEV)[:, 0]
        ss_host.run_varlen(logits, strided, indices)


def test_selfsampling_topk_varlen_cuda_graph():
    """CUDA-graph safety of the in-kernel varlen engine: warm up, capture one
    launch with max_seq_len (capture-stable constant — no host reads, no JIT
    inside capture), then replay while kv_lens grows in place — including a
    row that crosses the n <= k short-path boundary INSIDE the graph and a
    row walking over the 131072 band edge."""
    rows, top_k, msl = 4, 512, 262144
    npad = msl
    logits = torch.randn((rows, npad), dtype=torch.float32, device=_DEV) - 2.0
    kv_lens = torch.tensor([100, 4099, 131070, 200000], dtype=torch.int32, device=_DEV)
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)

    def refresh(step):
        kv = [100 + step * 137, 4099 + step * 977, 131070 + step, 200000 + step * 3]
        kv_lens.copy_(torch.tensor(kv, dtype=torch.int32, device=_DEV))
        gen = torch.Generator(device=_DEV).manual_seed(1000 + step)
        logits.copy_(
            torch.randn((rows, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
        )
        for r in range(rows):
            n = min(kv[r], npad)
            logits[r, n:] = 3e38
        return kv

    refresh(0)
    ss_host.run_varlen(logits, kv_lens, indices, compress_ratio=1, max_seq_len=msl)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ss_host.run_varlen(logits, kv_lens, indices, compress_ratio=1, max_seq_len=msl)
    for step in range(1, 6):
        kv = refresh(step)
        indices.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for r in range(rows):
            n = min(kv[r], npad)
            if n <= top_k:
                head = torch.sort(indices[r, :n].to(torch.int64)).values
                assert torch.equal(head, torch.arange(n, device=_DEV))
                assert bool((indices[r, n:] == -1).all())
            else:
                idx = indices[r].to(torch.int64)
                assert int(idx.min()) >= 0 and int(idx.max()) < n
                assert int(torch.unique(idx).numel()) == top_k
                ref = torch.topk(logits[r, :n], top_k).values
                got = torch.sort(torch.gather(logits[r], 0, idx) + 0.0, descending=True).values
                assert torch.equal(got, torch.sort(ref + 0.0, descending=True).values), (
                    f"replay step {step} row {r} inexact (n={n})"
                )


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


def test_selfsampling_route_large_batch_domain():
    """Production num_rows = max_batch_size * next_n can exceed the b<=1024
    range the plans were tuned on (e.g. 1024 * 4 = 4096 rows). route() is a
    pure function — assert the full domain stays well-formed up to 8192 rows."""
    for k in (512, 1024, 2048):
        for n in (k + 1, 4096, 65536, 262144):
            npad = (n + 63) // 64 * 64
            for b in (1536, 2048, 4096, 8192):
                r = ss_host.route(b, n, npad, k)
                assert r is not None and len(r) >= 2, (b, n, k, r)


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
                2 * k,
                3 * k,
                4 * k + 64,
                2560,
                4096,
                8192,
                16384,
                4 * 1024,
                4 * 4096,
                4 * 32768,
                65536,
                131072,
                262144,
            ):
                ns.update(v for v in range(c - 4, c + 5) if k < v <= npad)
            ns.update(range(k + 1, npad + 1, 4999))
            s = 12345
            for _ in range(400):
                s = (s * 1103515245 + 12345) % (1 << 31)
                ns.add(k + 1 + s % (npad - k - 1))
            for n in sorted(ns):
                assert ss_host.route_split(b, n, npad, k) == ss_host.route(b, n, npad, k), (
                    b,
                    n,
                    k,
                )
                checked += 1
    assert checked > 10_000


def test_selfsampling_route_bands_contiguous():
    """route_bands must tile the envelope contiguously with n-free static fields."""
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


def test_selfsampling_topk_varlen_rejects_non_fp32():
    """Engine-level dtype contract: bf16/fp16 logits must raise a clear
    error (the dispatch seam falls through before this; direct callers get
    the loud contract message instead of a CuTe typing failure)."""
    logits = torch.randn(1, 8192, device=_DEV, dtype=torch.bfloat16)
    kv = torch.tensor([8000], dtype=torch.int32, device=_DEV)
    out = torch.empty(1, 512, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="float32"):
        ss_host.run_varlen(logits, kv, out, next_n=1, compress_ratio=4, max_seq_len=32768)


def test_selfsampling_topk_varlen_zero_window_rows():
    """CUDA-graph padding dummy rows: kv_lens < next_n makes every MTP-window
    row's valid length n <= 0. The kernel must clamp those rows onto the
    zero-work short path and pad-emit -1 for the whole row, while the normal
    request in the same batch stays exact."""
    torch.manual_seed(0)
    k, nn, cr, msl = 512, 4, 4, 40000
    kv = torch.tensor([msl, 1], dtype=torch.int32, device=_DEV)
    rows = kv.numel() * nn
    npad = (msl // cr + 63) // 64 * 64
    logits = torch.randn(rows, npad, dtype=torch.float32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, kv, out, next_n=nn, compress_ratio=cr, max_seq_len=msl)
    torch.cuda.synchronize()
    assert (out[nn:] == -1).all().item(), "n<=0 rows must be fully -1-padded"
    for r in range(nn):
        n_r = (msl - nn + r + 1) // cr
        ref = torch.topk(logits[r, :n_r], k).values.sort().values
        got = logits[r].gather(0, out[r].long().clamp_min(0)).sort().values
        assert torch.equal(ref, got)


def test_selfsampling_warmup_row_stride_matches_arena():
    """warmup_varlen(row_stride=...) must compile the SAME launcher key the
    dispatch derives from a column-sliced arena view (row stride wider than
    the logical width, like the DSL paged-MQA arena's 256-element rounding).
    msl is chosen so 64-rounding != 256-rounding: with a mismatched warmup
    stride, capture below hits the loud not-compiled raise."""
    k, msl = 512, 8300
    stride = (msl + 255) // 256 * 256
    rows = 2
    ss_host.warmup_varlen(
        k, msl, compress_ratio=1, next_n=1, num_rows_list=(rows,), row_stride=stride
    )
    arena = torch.randn(rows, stride, dtype=torch.float32, device=_DEV)
    logits = arena[:, :msl]  # non-contiguous column slice, like serving
    kv = torch.full((rows,), msl, dtype=torch.int32, device=_DEV)
    out = torch.empty(rows, k, dtype=torch.int32, device=_DEV)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ss_host.run_varlen(logits, kv, out, max_seq_len=msl)
    g.replay()
    torch.cuda.synchronize()
    ref = torch.topk(arena[:, :msl], k, dim=1).values.sort(dim=1).values
    got = arena.gather(1, out.long().clamp_min(0)).sort(dim=1).values
    assert torch.equal(ref, got)


def test_selfsampling_varlen_regclus_parity_and_oracle():
    """The varlen launcher must admit the clustered register-resident family
    exactly where the free route picks it (route() parity tier 1), and the
    per-row varlen port must match the reference oracle on a heterogeneous
    batch: long rows, a short row (n <= k, in-kernel identity + -1 tail) and
    a zero-window row, under MTP row windows (next_n=4)."""
    k, msl_c, nn, cr = 1024, 131072, 4, 4
    npad = msl_c  # 256-aligned already
    assert ss_host.route(8, msl_c, npad, k)["kernel"] == "reg_clus"
    batch = 3
    rows = batch * nn
    torch.manual_seed(7)
    lg = torch.randn(rows, npad, dtype=torch.float32, device=_DEV)
    kv = torch.tensor([msl_c * cr, 900, nn - 1], dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=nn, compress_ratio=cr, max_seq_len=msl_c * cr)
    key = (rows, npad, k, msl_c, nn, cr)
    assert ss_host._VARLEN_CACHE[key][0] == "reg_clus", ss_host._VARLEN_CACHE[key][0]
    torch.cuda.synchronize()
    ref = _reference_varlen_indices(lg, kv, nn, cr, k)
    for r in range(rows):
        if (ref[r] >= 0).any():
            row = lg[r].float()
            got = row[out[r].long().clamp_min(0)].sort().values
            want = row[ref[r].long().clamp_min(0)].sort().values
            assert torch.equal(got, want), f"row {r} value multiset mismatch"
            assert torch.equal(out[r] < 0, ref[r] < 0), f"row {r} pad mask mismatch"
        else:
            assert torch.equal(out[r], ref[r]), f"row {r} expected all -1"


def test_selfsampling_varlen_regclus_cuda_graph():
    """Cluster-family varlen engine must be CUDA-graph capturable: warmed
    engine, capture one launch, replay twice, tie-aware exact each time."""
    k, msl_c, cr = 1024, 131072, 4
    rows = 8
    torch.manual_seed(11)
    lg = torch.randn(rows, msl_c, dtype=torch.float32, device=_DEV)
    kv = torch.full((rows,), msl_c * cr, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    out.fill_(-7)
    with torch.cuda.graph(g):
        ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
    ref_v = torch.topk(lg.float(), k, dim=1).values.sort(dim=1).values
    for _ in range(2):
        out.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        got = lg.float().gather(1, out.long().clamp_min(0)).sort(dim=1).values
        assert torch.equal(got, ref_v)


def test_selfsampling_varlen_reg_parity_and_oracle():
    """The varlen launcher must admit the register-resident family (and its
    img flavor) exactly where the free route picks it (route() parity tier 2),
    and the per-row varlen port must match the reference oracle on a
    heterogeneous batch: long rows, a short row (n <= k; k can exceed BLK on
    this family, exercising the strided identity + -1 tail loop) and a
    zero-window row, under MTP row windows (next_n=4)."""
    torch.manual_seed(13)
    # (k, msl_c, cr, expected free-route family)
    cases = [
        (512, 6144, 4, "reg"),  # v4-style small-N band
        (2048, 8192, 1, "reg"),  # k > BLK: strided short-row emit
        (512, 3072, 4, "regimg"),  # img window (n4 in (512, 1024])
    ]
    nn = 4
    batch = 3
    rows = batch * nn
    for k, msl_c, cr, want in cases:
        npad = (msl_c + 63) // 64 * 64
        fam = ss_host.route(rows, msl_c, npad, k)["kernel"]
        assert fam == want, (fam, want, k, msl_c)
        msl = msl_c * cr
        lg = torch.randn(rows, npad, dtype=torch.float32, device=_DEV)
        kv = torch.tensor([msl, max((k - 3) * cr, nn), nn - 1], dtype=torch.int32, device=_DEV)
        out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_varlen(lg, kv, out, next_n=nn, compress_ratio=cr, max_seq_len=msl)
        key = (rows, npad, k, msl_c, nn, cr)
        assert ss_host._VARLEN_CACHE[key][0] == "reg", ss_host._VARLEN_CACHE[key][0]
        torch.cuda.synchronize()
        ref = _reference_varlen_indices(lg, kv, nn, cr, k)
        for r in range(rows):
            if (ref[r] >= 0).any():
                row = lg[r].float()
                got = row[out[r].long().clamp_min(0)].sort().values
                want_v = row[ref[r].long().clamp_min(0)].sort().values
                assert torch.equal(got, want_v), f"k={k} row {r} value multiset mismatch"
                assert torch.equal(out[r] < 0, ref[r] < 0), f"k={k} row {r} pad mask mismatch"
            else:
                assert torch.equal(out[r], ref[r]), f"k={k} row {r} expected all -1"


def test_selfsampling_varlen_clus_parity_and_oracle():
    """The varlen launcher must admit the cluster-split family exactly where
    the free route picks it (route() parity tier 3), and the per-row varlen
    port must match the reference oracle on a heterogeneous batch: full rows,
    mid rows BELOW the family's standalone admission floor (n <= SCAP,
    exercising the always-on per-row QUAD schedule), a short row (n <= k,
    in-kernel identity + -1 tail from cluster rank 0) and a zero-window row,
    under MTP row windows (next_n=4). Covers CS=2 and CS=4 clusters."""
    torch.manual_seed(23)
    nn = 4
    # (rows, msl_c, k, expected cluster size)
    cases = [(64, 131072, 1024, 2), (32, 131072, 1024, 4)]
    cr = 4
    for rows, msl_c, k, want_cs in cases:
        npad = msl_c
        plan = ss_host.route(rows, msl_c, npad, k)
        assert plan["kernel"] == "clus" and plan["cluster"] == want_cs, plan
        batch = rows // nn
        lg = torch.randn(rows, npad, dtype=torch.float32, device=_DEV)
        lens = [msl_c * cr, 900, nn - 1, 20000, 40000, 300000, msl_c * cr // 2, 5000]
        kv = torch.tensor(
            [lens[i % len(lens)] for i in range(batch)], dtype=torch.int32, device=_DEV
        )
        out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_varlen(lg, kv, out, next_n=nn, compress_ratio=cr, max_seq_len=msl_c * cr)
        key = (rows, npad, k, msl_c, nn, cr)
        assert ss_host._VARLEN_CACHE[key][0] == "clus", ss_host._VARLEN_CACHE[key][0]
        torch.cuda.synchronize()
        ref = _reference_varlen_indices(lg, kv, nn, cr, k)
        for r in range(rows):
            if (ref[r] >= 0).any():
                row = lg[r].float()
                got = row[out[r].long().clamp_min(0)].sort().values
                want = row[ref[r].long().clamp_min(0)].sort().values
                assert torch.equal(got, want), f"cs={want_cs} row {r} value multiset mismatch"
                assert torch.equal(out[r] < 0, ref[r] < 0), f"cs={want_cs} row {r} pad mask"
            else:
                assert torch.equal(out[r], ref[r]), f"cs={want_cs} row {r} expected all -1"


def test_selfsampling_varlen_clus_cuda_graph():
    """Cluster-split-family varlen engine must be CUDA-graph capturable:
    warmed engine, capture one launch, replay twice, tie-aware exact each
    time."""
    k, msl_c, cr = 1024, 131072, 4
    rows = 32
    torch.manual_seed(29)
    lg = torch.randn(rows, msl_c, dtype=torch.float32, device=_DEV)
    kv = torch.full((rows,), msl_c * cr, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
    torch.cuda.synchronize()
    key = (rows, msl_c, k, msl_c, 1, cr)
    assert ss_host._VARLEN_CACHE[key][0] == "clus", ss_host._VARLEN_CACHE[key][0]
    g = torch.cuda.CUDAGraph()
    out.fill_(-7)
    with torch.cuda.graph(g):
        ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
    ref_v = torch.topk(lg.float(), k, dim=1).values.sort(dim=1).values
    for _ in range(2):
        out.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        got = lg.float().gather(1, out.long().clamp_min(0)).sort(dim=1).values
        assert torch.equal(got, ref_v)


def test_selfsampling_varlen_reg_cuda_graph():
    """Register-family varlen engine must be CUDA-graph capturable: warmed
    engine, capture one launch, replay twice, tie-aware exact each time."""
    k, msl_c, cr = 512, 4096, 4
    rows = 16
    torch.manual_seed(17)
    lg = torch.randn(rows, msl_c, dtype=torch.float32, device=_DEV)
    kv = torch.full((rows,), msl_c * cr, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
    torch.cuda.synchronize()
    key = (rows, msl_c, k, msl_c, 1, cr)
    assert ss_host._VARLEN_CACHE[key][0] == "reg", ss_host._VARLEN_CACHE[key][0]
    g = torch.cuda.CUDAGraph()
    out.fill_(-7)
    with torch.cuda.graph(g):
        ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
    ref_v = torch.topk(lg.float(), k, dim=1).values.sort(dim=1).values
    for _ in range(2):
        out.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        got = lg.float().gather(1, out.long().clamp_min(0)).sort(dim=1).values
        assert torch.equal(got, ref_v)


def test_selfsampling_varlen_full_row_range():
    """Full-range production contract: with self-sampling enabled, EVERY row
    count dispatches to the self-sampling engines (no rows-based fall-through)
    — spot-check the throughput end (rows 304 and 1024) for tie-aware
    exactness at a mid-size envelope."""
    k, msl_c, cr = 1024, 65536, 4
    torch.manual_seed(3)
    for rows in (304, 1024):
        lg = torch.randn(rows, msl_c, dtype=torch.float32, device=_DEV)
        kv = torch.full((rows,), msl_c * cr, dtype=torch.int32, device=_DEV)
        out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=cr, max_seq_len=msl_c * cr)
        torch.cuda.synchronize()
        key = (rows, msl_c, k, msl_c, 1, cr)
        assert key in ss_host._VARLEN_CACHE, "row count must dispatch in-engine"
        ref_v = torch.topk(lg.float(), k, dim=1).values.sort(dim=1).values
        got = lg.float().gather(1, out.long().clamp_min(0)).sort(dim=1).values
        assert torch.equal(got, ref_v), f"rows={rows} value multiset mismatch"


def test_selfsampling_warmup_band_enumeration_bounded():
    """warmup_varlen must cover arbitrarily large batch lists by warming one
    representative row per distinct engine compile key — bounded time and
    memory (the representative rows saturate a few hundred, never the
    requested thousands)."""
    k, msl = 512, 65536  # kv tokens, cr=4 -> n_env 16384: few bands, fast
    before = len(ss_host._VARLEN_WARMUP_DONE)
    ss_host.warmup_varlen(
        k,
        msl,
        compress_ratio=4,
        next_n=4,
        num_rows_list=(4096,),
        row_stride=(msl // 4 + 255) // 256 * 256,
    )
    assert len(ss_host._VARLEN_WARMUP_DONE) == before + 1


def test_selfsampling_varlen_heterogeneous_lengths_main():
    """Row-independence contract at throughput scale on the streaming main
    family: rows are naturally independent tasks — per-row data AND length
    (random kv_lens spanning long / mid / short / n<=k / zero-window rows in
    one 304-row batch, MTP row windows via next_n=4). Every row must match
    its own per-prefix torch.topk value multiset; short rows must be
    identity + (-1) tail; zero-window rows all -1."""
    k, msl_c, cr, nn = 1024, 65536, 4, 4
    batch = 76
    rows = batch * nn  # 304 -> route_streaming main family
    torch.manual_seed(5)
    lg = torch.randn(rows, msl_c, dtype=torch.float32, device=_DEV)
    kv = torch.randint(1, msl_c * cr, (batch,), dtype=torch.int32, device=_DEV)
    kv[0] = msl_c * cr  # full length
    kv[1] = 900  # short (n <= k)
    kv[2] = nn - 1  # zero-window (every row of the request empty)
    kv[3] = k * cr + nn  # just above the short path
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=nn, compress_ratio=cr, max_seq_len=msl_c * cr)
    torch.cuda.synchronize()
    kl = kv.tolist()
    for r in range(rows):
        n_r = max(kl[r // nn] - nn + (r % nn) + 1, 0) // cr
        n_r = min(n_r, msl_c)
        if n_r <= 0:
            assert (out[r] == -1).all(), f"row {r}: zero-window must be all -1"
            continue
        if n_r <= k:
            want = list(range(n_r)) + [-1] * (k - n_r)
            assert out[r].tolist() == want, f"row {r}: short path identity+pad"
            continue
        row = lg[r].float()[:n_r]
        ref_v = torch.topk(row, k).values.sort().values
        got = row[out[r].long()].sort().values
        assert torch.equal(got, ref_v), f"row {r}: value multiset mismatch (n={n_r})"


# ===========================================================================
# ==== prefill: per-row [ks, ke) windows (run_prefill) ======================
# ===========================================================================
# Contract: row r selects the Top-K of logits[r, ks:ke] (ks=row_starts[r],
# ke=row_ends[r], compressed column units), output in the LOCAL frame
# (column - ks) with a trailing -1 pad; nv=ke-ks <= k gives identity 0..nv-1.
# The base is rounded down to a 16B boundary and the <=3 lead lanes are masked,
# so ks % 4 in {1,2,3} (2nd+ request of a multi-request chunk) is exercised
# with poison (+inf/NaN/3e38/-inf) written at [ks-3, ks) and [ke, npad).


def _prefill_reference(logits, row_starts, row_ends, top_k):
    rows, ncols = logits.shape
    out = torch.full((rows, top_k), -1, dtype=torch.int32, device=logits.device)
    ks, ke = row_starts.tolist(), row_ends.tolist()
    for r in range(rows):
        nv = max(min(ke[r], ncols) - ks[r], 0)
        if nv == 0:
            continue
        if nv <= top_k:
            out[r, :nv] = torch.arange(nv, dtype=torch.int32, device=logits.device)
        else:
            out[r] = torch.topk(logits[r, ks[r] : ks[r] + nv], top_k).indices.to(torch.int32)
    return out


def _check_prefill_exact(logits, got, row_starts, row_ends, top_k):
    """Tie-aware radix-parity check: trailing -1 pad from lengths; head unique
    and in [0, nv); identity for nv <= k; exact index set when the k-th value
    is unique, else strictly-above set + tie-class count (signed zeros and
    genuine +/-inf compare like radix). NaN-in-window rows are structure-only."""
    assert got.shape == (logits.shape[0], top_k) and got.dtype == torch.int32
    ks, ke = row_starts.tolist(), row_ends.tolist()
    got64 = got.to(torch.int64)
    dev = logits.device
    for r in range(logits.shape[0]):
        nv = max(min(ke[r], logits.shape[1]) - ks[r], 0)
        m = min(nv, top_k)
        row = got64[r]
        assert bool((row[m:] == -1).all()), f"row {r}: pad must be trailing -1 x{top_k - m}"
        head = row[:m]
        if m == 0:
            continue
        assert bool((head != -1).all()), f"row {r}: -1 inside the valid head"
        assert int(head.min()) >= 0 and int(head.max()) < nv, f"row {r}: index outside [0,{nv})"
        assert int(torch.unique(head).numel()) == m, f"row {r}: duplicate indices"
        win = logits[r, ks[r] : ks[r] + nv]
        if bool(torch.isnan(win).any()):
            continue  # NaN out of contract for both kernels; structure only
        if nv <= top_k:
            assert torch.equal(torch.sort(head).values, torch.arange(nv, device=dev)), (
                f"row {r}: short row must be identity"
            )
            continue
        vals = torch.sort(win, descending=True).values
        v_k, v_next = vals[top_k - 1], vals[top_k]
        got_vals = win[head]
        if bool(v_k != v_next):
            ref = (win >= v_k).nonzero(as_tuple=True)[0]
            assert ref.numel() == top_k
            assert torch.equal(torch.sort(head).values, ref), f"row {r}: index set mismatch"
        else:
            above = (win > v_k).nonzero(as_tuple=True)[0]
            got_above = head[got_vals > v_k]
            assert torch.equal(torch.sort(got_above).values, above), (
                f"row {r}: strictly-above set mismatch"
            )
            assert int((got_vals == v_k).sum()) == top_k - above.numel(), (
                f"row {r}: wrong number of boundary-tied picks"
            )
            assert bool((got_vals >= v_k).all()), f"row {r}: value below k-th selected"


def _make_prefill_case(rows, ncols, ks_list, ke_list, *, top_k, seed, dist="randn"):
    """DeepGEMM-like storage: stride = align(ncols + 256, 256), column slice
    [:, :ncols]; outside-window columns poisoned so an over-read/frame bug is
    caught (+inf at [ks-3, ks), rotating NaN/inf/3e38/-inf elsewhere)."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    stride = ((ncols + 256 + 255) // 256) * 256
    if dist == "randn":
        full = torch.randn((rows, stride), generator=gen, dtype=torch.float32, device=_DEV)
    elif dist == "equal":
        full = torch.ones((rows, stride), dtype=torch.float32, device=_DEV)
    elif dist == "twoval":
        full = torch.randint(0, 2, (rows, stride), generator=gen, device=_DEV).float()
    else:
        raise ValueError(dist)
    logits = full[:, :ncols]
    row_starts = torch.tensor(ks_list, dtype=torch.int32, device=_DEV)
    row_ends = torch.tensor(ke_list, dtype=torch.int32, device=_DEV)
    cols = torch.arange(stride, device=_DEV).unsqueeze(0)
    outside = (cols < row_starts.unsqueeze(1)) | (cols >= row_ends.unsqueeze(1))
    pat = torch.tensor([float("nan"), float("inf"), 3e38, float("-inf")], device=_DEV)[
        cols % 4
    ].expand(rows, -1)
    full.masked_scatter_(outside, pat[outside])
    for r, ks in enumerate(ks_list):
        full[r, max(ks - 3, 0) : ks] = float("inf")
    return logits, row_starts, row_ends


@pytest.mark.parametrize("top_k", [512, 1024, 2048], ids=lambda k: f"k{k}")
def test_prefill_causal_ramp(top_k):
    """Single-request causal ramp (ks=0): a run of short rows then long rows
    in one launch straddles the k boundary. Covers nv < k, == k, > k."""
    rows = 148
    ks = [0] * rows
    ke = list(range(1, rows + 1))
    lg, rs, re = _make_prefill_case(rows, rows, ks, ke, top_k=top_k, seed=top_k)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, top_k)


@pytest.mark.parametrize("lead", [1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_packed_misaligned_ks(top_k, lead):
    """Multi-request chunk: request 2 starts at ks % 4 == lead with +inf poison
    at [ks-lead, ks). A leaked lead lane would become top-1 (wrong)."""
    a = 300
    ks1 = ((a + 3) // 4) * 4 + lead
    n1 = 4096 + 17
    rows = a + n1
    ncols = ks1 + n1
    ks = [0] * a + [ks1] * n1
    ke = list(range(1, a + 1)) + [ks1 + n1] * n1
    lg, rs, re = _make_prefill_case(rows, ncols, ks, ke, top_k=top_k, seed=top_k * 100 + lead)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    assert int(out.min()) >= -1, "negative index leaked (missed -lead correction / guard)"
    _check_prefill_exact(lg, out, rs, re, top_k)


@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_short_rows(top_k):
    """nv in {0, 1, k-1, k, k+1}: identity 0..nv-1 + trailing -1 (radix short
    contract); nv==0 (ks==ke) -> all -1."""
    for nv in (0, 1, top_k - 1, top_k, top_k + 1):
        rows = 4
        ncols = max(nv, 1) + 8
        ks = [0] * rows
        ke = [nv] * rows
        lg, rs, re = _make_prefill_case(rows, ncols, ks, ke, top_k=top_k, seed=nv + top_k)
        out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_prefill(lg, rs, re, out)
        torch.cuda.synchronize()
        _check_prefill_exact(lg, out, rs, re, top_k)


@pytest.mark.parametrize("dist", ["equal", "twoval"], ids=lambda d: d)
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_ties_degenerate(top_k, dist):
    """All-equal (whole tie class) and two-valued (massive ties) rows drive the
    degenerate A/B narrowing paths; tie-aware acceptance."""
    rows = 16
    n = 4096
    ks = [0] * rows
    ke = [n] * rows
    lg, rs, re = _make_prefill_case(rows, n, ks, ke, top_k=top_k, seed=top_k, dist=dist)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, top_k)


@pytest.mark.parametrize("lead", [1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_neginf_tie_class(top_k, lead):
    """nv > k with fewer than k finite values -> the k-th boundary is in the
    -inf tie class (degen B), crossed with misaligned lead. A -inf-valued mask
    would be emitted here as a negative index (== -lead); assert none leaks."""
    n_finite = top_k - 100
    nv = top_k + 400
    ks1 = ((37 + 3) // 4) * 4 + lead
    ncols = ks1 + nv
    rows = 5
    gen = torch.Generator(device=_DEV).manual_seed(top_k * 10 + lead)
    stride = ((ncols + 256 + 255) // 256) * 256
    full = torch.full((rows, stride), float("-inf"), dtype=torch.float32, device=_DEV)
    for r in range(rows):
        full[r, ks1 : ks1 + n_finite] = torch.randn(n_finite, generator=gen, device=_DEV)
        full[r, :ks1] = float("inf")
        full[r, ks1 + nv :] = 3e38
    logits = full[:, :ncols]
    rs = torch.tensor([ks1] * rows, dtype=torch.int32, device=_DEV)
    re = torch.tensor([ks1 + nv] * rows, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(logits, rs, re, out)
    torch.cuda.synchronize()
    assert int(out.min()) >= -1, "negative index leaked in a -inf tie class"
    _check_prefill_exact(logits, out, rs, re, top_k)


@pytest.mark.parametrize("top_k, n", [(512, 4099), (2048, 131075)], ids=lambda v: f"n{v}")
def test_prefill_deepgemm_single_row_odd_width(top_k, n):
    """A 1-row tile with an odd num_k_tokens on a DeepGEMM-strided view must
    use stride(0) (not shape[1]) and stay exact — the fully-cached follow-up
    turn that the varlen 1-row rule would wrongly reject."""
    rows = 1
    ks = [0]
    ke = [n]
    lg, rs, re = _make_prefill_case(rows, n, ks, ke, top_k=top_k, seed=n)
    assert lg.stride(0) % 256 == 0 and lg.shape[1] == n  # DeepGEMM-like view
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, top_k)


def test_prefill_slab_over_gridy_limit():
    """> 65535 rows in one call must be slabbed (gridDim.y <= 65535)."""
    k = 512
    rows = 70000
    n = 2048
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    re = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    idx = torch.tensor([0, 1, 32767, 32768, 65535, 65536, 69999], device=_DEV)
    _check_prefill_exact(lg[idx], out[idx].contiguous(), rs[idx], re[idx], k)


def test_prefill_engine_key_distinct_from_decode():
    """The prefill compile shares the DSv3.2 decode varlen tuple (next_n=1,
    cr_shift=0) but has a distinct prologue, so the compile keys must differ."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
        gvr_topk_decode_self_sampling as dev,
    )

    tpl = (256, 8, 4, 256, 2, False, False, 1, 0, 1)
    a = dev.get_compiled(tpl, hint_free=True)
    b = dev.get_compiled(tpl, hint_free=True, prefill=True)
    assert a is not b


def test_prefill_guards():
    k = 512
    n = 4096
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((3, stride), dtype=torch.float32, device=_DEV)[:, :n]
    rs = torch.zeros((3,), dtype=torch.int32, device=_DEV)
    re = torch.full((3,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((3, k), -7, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="float32"):
        ss_host.run_prefill(lg.to(torch.bfloat16), rs, re, out)
    with pytest.raises(RuntimeError, match="row_starts"):
        ss_host.run_prefill(lg, rs.to(torch.int64), re, out)
    with pytest.raises(RuntimeError, match="row_starts/row_ends length"):
        ss_host.run_prefill(lg, rs[:2], re, out)
    with pytest.raises(RuntimeError, match="multiple of 4"):
        ss_host.run_prefill(lg, rs, re, torch.full((3, k + 2), -7, dtype=torch.int32, device=_DEV))
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        ss_host.run_prefill(lg[:, 1:], rs, re, out)  # base offset by 1 float


def test_prefill_warmup_idempotent_and_no_rejit():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
        gvr_topk_decode_self_sampling as dev,
    )

    k = 512
    ss_host.warmup_prefill(k, 32768)
    before = len(ss_host._PREFILL_WARMUP_DONE)
    ss_host.warmup_prefill(k, 32768)
    assert len(ss_host._PREFILL_WARMUP_DONE) == before, "warmup not idempotent"
    orig = dev.get_compiled
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    dev.get_compiled = counting
    try:
        for rows in (1, 8, 37, 74, 100, 296, 297, 4096):
            for nkv in (4096, 16384, 32768):
                stride = ((nkv + 256 + 255) // 256) * 256
                lg = torch.zeros((rows, stride), dtype=torch.float32, device=_DEV)[:, :nkv]
                rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
                re = torch.full((rows,), nkv, dtype=torch.int32, device=_DEV)
                out = torch.empty((rows, k), dtype=torch.int32, device=_DEV)
                ss_host.run_prefill(lg, rs, re, out, max_row_len=nkv)
    finally:
        dev.get_compiled = orig
    assert calls["n"] == 0, f"warmup missed keys: {calls['n']} live compiles"


def test_prefill_capture_no_host_sync():
    """After warmup, a run_prefill call captures under a CUDA graph (proves no
    .item()/.max() host read)."""
    k = 512
    n = 8192
    ss_host.warmup_prefill(k, max(n, 32768))
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((64, stride), dtype=torch.float32, device=_DEV)[:, :n]
    rs = torch.zeros((64,), dtype=torch.int32, device=_DEV)
    re = torch.full((64,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((64, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out, max_row_len=n)  # compile outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ss_host.run_prefill(lg, rs, re, out, max_row_len=n)
    g.replay()
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, k)
