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
The production ``run_varlen`` entry additionally accepts native 16-bit logits (BF16 and
FP16 on every family, the prefill window mode and the near-K complement
specialization) and per-request device KV lengths, including
CUDA graph replay.

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

from tensorrt_llm._torch.kernels.blackwell.top_k import (
    gvr_topk_decode_self_sampling_host as ss_host,
)

_DEV = "cuda"


def _varlen_cache_key(*parts: int) -> tuple[int, ...]:
    device = torch.cuda.current_device()
    return (*parts, ss_host._device_profile_key(device))


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
                12288,
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


def test_selfsampling_route_4k_8k_register_rungs() -> None:
    """Check register launch geometry, boundary tails and histogram layout."""
    for k in (512, 1024, 2048):
        for n in (4100, 6144, 8192, 8195):
            npad = (n + 63) // 64 * 64
            for b in (1, 64, 148):
                r = ss_host.route(b, n, npad, k)
                want = (512, 3, 1) if k == 2048 and n <= 6144 else (1024, 2, 1)
                assert r["kernel"] == "reg" and tuple(r["tpl"][:3]) == want, (b, n, k, r)
                assert r["tpl"][7] == 2 * ss_host.NB
            for b in (149, 160, 256, 296, 297):
                r = ss_host.route(b, n, npad, k)
                want = (512, 3, 2) if n <= 6144 else (512, 4, 2)
                assert r["kernel"] == "reg" and tuple(r["tpl"][:3]) == want, (b, n, k, r)
                assert r["tpl"][7] == ss_host.NB
        lower_edge = {
            512: ("regimg", (1024, 1, 2)),
            1024: ("reg", (1024, 1, 1)),
            2048: ("reg", (512, 2, 1)),
        }
        lo = ss_host.route(64, 4099, 4160, k)
        assert (lo["kernel"], tuple(lo["tpl"][:3])) == lower_edge[k]
        assert tuple(ss_host.route(160, 4099, 4160, k)["tpl"][:3]) == (512, 2, 4)
        hi = ss_host.route(64, 8196, 8256, k)
        assert hi["kernel"] == "reg" and tuple(hi["tpl"][:3]) == (1024, 3, 1)
        mid_hi = ss_host.route(160, 8196, 8256, k)
        assert mid_hi["kernel"] == "reg" and tuple(mid_hi["tpl"][:3]) == (512, 5, 2)
    for k in (512, 1024, 2048):
        for b in (1, 148, 149, 296, 297, 1024):
            for n in range(k + 1, 16385, 97):
                npad = (n + 63) // 64 * 64
                r = ss_host.route(b, n, npad, k)
                if r["kernel"] in ("reg", "regimg"):
                    assert r["rt"]["QC"] == 160, (b, n, k)
                    assert r["rt"]["IMGOFF"] == r["tpl"][7], (
                        b,
                        n,
                        k,
                        r["rt"]["IMGOFF"],
                        r["tpl"][7],
                    )
                    scratch = 2 * r["rt"]["CMP"]
                    if r["kernel"] == "regimg":
                        scratch = max(scratch, (n + 3) & ~3)
                    assert r["smem"] == (r["tpl"][7] + scratch) * 4, (b, n, k, r)


def test_selfsampling_register_rungs_follow_available_sms() -> None:
    """Only register wave boundaries follow the available SM count."""
    for num_sms in (148, 160, 208):
        for k in (512, 1024, 2048):
            n, npad = 8195, 8256
            wide = ss_host.route(num_sms, n, npad, k, num_sms)
            assert wide["kernel"] == "reg"
            assert tuple(wide["tpl"][:3]) == (1024, 2, 1)
            for rows in (num_sms + 1, 2 * num_sms):
                mid = ss_host.route(rows, n, npad, k, num_sms)
                assert mid["kernel"] == "reg"
                assert tuple(mid["tpl"][:3]) == (512, 4, 2)
            tall = ss_host.route(2 * num_sms + 1, n, npad, k, num_sms)
            assert tall["kernel"] == "reg"
            assert tuple(tall["tpl"][:3]) == (512, 4, 2)

            n, npad = 12288, 12288
            wide = ss_host.route(num_sms, n, npad, k, num_sms)
            assert wide["kernel"] == "reg"
            assert tuple(wide["tpl"][:3]) == (1024, 3, 1)
            mid = ss_host.route(num_sms + 1, n, npad, k, num_sms)
            assert mid["kernel"] == "reg"
            assert tuple(mid["tpl"][:3]) == (512, 6, 2)

        # Streaming and clustered-register tuning is intentionally invariant.
        for rows, n, k, family in (
            (52, 32768, 1024, "reg_clus"),
            (32, 262144, 1024, "clus"),
            (192, 131072, 2048, "main"),
            (320, 131072, 2048, "main"),
        ):
            baseline = ss_host.route(rows, n, n, k, 148)
            assert baseline["kernel"] == family
            assert ss_host.route(rows, n, n, k, num_sms) == baseline

        # The default preserves the 148-SM B200 behavior exactly.
        for rows in (148, 149, 296, 297):
            assert ss_host.route(rows, 8195, 8256, 1024) == ss_host.route(
                rows, 8195, 8256, 1024, 148
            )


def test_selfsampling_route_12k_register_coverage() -> None:
    """The 12K register envelope covers both architectures and all batch sizes."""
    for sm_version in (100, 103):
        for num_sms in (148, 160, 208):
            for k in (512, 1024, 2048):
                for rows in (
                    1,
                    num_sms,
                    num_sms + 1,
                    2 * num_sms,
                    2 * num_sms + 1,
                    512,
                    768,
                    1024,
                    1280,
                    2048,
                    8192,
                ):
                    for n in (8196, 10240, 12287, 12288):
                        for det in (False, True):
                            npad = (n + 63) // 64 * 64
                            plan = ss_host.route(rows, n, npad, k, num_sms, sm_version, det)
                            assert plan["kernel"] == "reg", (rows, n, k, plan)
                            assert plan["grid"] == (rows, 1) and plan["cluster"] == 1
                            assert not plan["ws"]
                            assert plan["tpl"][0] * plan["tpl"][1] >= n // 4
                            if det:
                                assert plan["tpl"][1] <= 8

                # K=2048 also retains a float4 rung through its three scalar tails.
                for n in (12289, 12290, 12291, 12292):
                    plan = ss_host.route(8192, n, 12352, k, num_sms, sm_version)
                    family = "reg" if k == 2048 and n <= 12291 else "main"
                    assert plan["kernel"] == family, (n, k, plan)


def test_selfsampling_route_sms_factorization() -> None:
    """The factored route preserves topology-dependent register plans."""
    for num_sms in (148, 160, 208):
        for rows in (num_sms, num_sms + 1, 2 * num_sms, 2 * num_sms + 1):
            for n in (4100, 8195, 12288, 12289, 12292, 131072):
                for k in (512, 1024, 2048):
                    npad = (n + 63) // 64 * 64
                    assert ss_host.route_split(rows, n, npad, k, num_sms) == ss_host.route(
                        rows, n, npad, k, num_sms
                    )


def test_selfsampling_route_sm103_large_batch_k2048_register_rung() -> None:
    """SM100 and SM103 share the large-batch K=2048 register launch geometry."""
    sm100 = ss_host.route(512, 8192, 8192, 2048, 148)
    tuned = ss_host.route(512, 8192, 8192, 2048, 148, sm_version=103)
    assert sm100 == tuned
    assert tuned["kernel"] == "reg"
    assert tuned["tpl"] == (512, 4, 2, 1, True, True, False, 1024)
    assert ss_host.route(1024, 8192, 8192, 2048, 148, 103)["kernel"] == "reg"

    # Nearby K, batch and topology points agree across SM100 and SM103.
    for rows, top_k, num_sms, sm_version in (
        (512, 1024, 148, 103),
        (384, 2048, 148, 103),
        (448, 2048, 148, 103),
        (1025, 2048, 148, 103),
        (512, 2048, 160, 103),
        (512, 2048, 148, 100),
    ):
        assert ss_host.route(rows, 8192, 8192, top_k, num_sms, sm_version) == ss_host.route(
            rows, 8192, 8192, top_k, num_sms
        )

    assert ss_host.route_split(512, 8192, 8192, 2048, 148, 103) == tuned

    for sm_version in (100, 103):
        for n, expected in (
            (4099, (512, 2, 4)),
            (4100, (256, 5, 4)),
            (4111, (256, 5, 4)),
            (6144, (512, 3, 2)),
            (8192, (512, 4, 2)),
            (8195, (512, 4, 2)),
            (8196, (512, 5, 2)),
        ):
            npad = (n + 63) // 64 * 64
            plan = ss_host.route(512, n, npad, 2048, 148, sm_version)
            assert plan["kernel"] == "reg"
            assert tuple(plan["tpl"][:3]) == expected, (n, sm_version, plan)


@pytest.mark.skipif(getSMVersion() not in (100, 103), reason="requires an SM100 or SM103 GPU")
def test_selfsampling_sm103_b512_k2048_cuda_graph() -> None:
    """Exercise the shared 148-SM SM100/SM103 plan through warmup and graph replay."""
    rows, n, npad, top_k = 512, 4111, 4160, 2048
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if properties.multi_processor_count != 148:
        pytest.skip("requires an SM100 or SM103 GPU with 148 SMs")
    plan = ss_host.route(
        rows,
        n,
        npad,
        top_k,
        properties.multi_processor_count,
        properties.major * 10 + properties.minor,
    )
    assert plan["kernel"] == "reg"
    assert tuple(plan["tpl"][:3]) == (256, 5, 4)

    ss_host.warmup_varlen(
        top_k,
        n,
        num_rows_list=(rows,),
        row_stride=npad,
    )
    generator = torch.Generator(device=_DEV).manual_seed(103_512_2048)
    logits = torch.randn((rows, npad), generator=generator, dtype=torch.float32, device=_DEV)
    kv_lens = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    kv_lens[1] = top_k - 5
    kv_lens[2] = 0
    output = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ss_host.run_varlen(logits, kv_lens, output, max_seq_len=n)
    key = _varlen_cache_key(rows, npad, top_k, n, 1, 1)
    assert ss_host._VARLEN_CACHE[key][0] == "reg"

    reference = _reference_varlen_indices(logits, kv_lens, 1, 1, top_k)
    for _ in range(2):
        output.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for row in range(rows):
            if (reference[row] >= 0).any():
                got = logits[row, output[row].long().clamp_min(0)].sort().values
                want = logits[row, reference[row].long().clamp_min(0)].sort().values
                assert torch.equal(got, want), f"row {row} value multiset mismatch"
                assert torch.equal(output[row] < 0, reference[row] < 0)
            else:
                assert torch.equal(output[row], reference[row])


def test_selfsampling_device_profile_is_cached_per_device(monkeypatch) -> None:
    """The SM-count and architecture query is cold-path-only and cached."""
    profiles = {0: (160, 10, 0), 1: (148, 10, 3)}
    queries = []

    class Properties:
        def __init__(self, multi_processor_count: int, major: int, minor: int) -> None:
            self.multi_processor_count = multi_processor_count
            self.major = major
            self.minor = minor

    def get_device_properties(device: int) -> Properties:
        queries.append(device)
        return Properties(*profiles[device])

    monkeypatch.setattr(ss_host, "_DEVICE_PROFILE", {})
    monkeypatch.setattr(ss_host, "_get_device_properties", get_device_properties)
    assert ss_host._unpack_device_profile(ss_host._device_profile_key(0)) == (160, 100)
    assert ss_host._unpack_device_profile(ss_host._device_profile_key(1)) == (148, 103)
    profiles[0] = (148, 10, 3)
    assert ss_host._device_num_sms(0) == 160
    assert queries == [0, 1]


def test_selfsampling_launcher_cache_identity_includes_device_profile(monkeypatch) -> None:
    """Uniform and varlen launchers compiled for different profiles never alias."""

    class FakeCudaTensor:
        is_cuda = True

        def __init__(self, shape: tuple[int, int], dtype: torch.dtype) -> None:
            self.shape = shape
            self.dtype = dtype

        def is_contiguous(self) -> bool:
            return True

        def data_ptr(self) -> int:
            return 0

        def get_device(self) -> int:
            return 0

    current_profile = [148, 100]
    builds = []

    def build_launcher(
        batch_size: int,
        n_valid: int,
        npad: int,
        top_k: int,
        num_sms: int,
        sm_version: int,
    ) -> tuple:
        builds.append((batch_size, n_valid, npad, top_k, num_sms, sm_version))
        return (lambda *args: None, (), False)

    monkeypatch.setattr(ss_host, "_LAUNCH_CACHE", {})
    monkeypatch.setattr(
        ss_host,
        "_device_profile_key",
        lambda device: ss_host._pack_device_profile(*current_profile),
    )
    monkeypatch.setattr(ss_host, "_build_launcher", build_launcher)
    logits = FakeCudaTensor((160, 8192), torch.float32)
    pre_idx = FakeCudaTensor((160, 512), torch.int32)
    indices = FakeCudaTensor((160, 512), torch.int32)

    ss_host._run_impl(logits, pre_idx, 8192, indices, None)
    ss_host._run_impl(logits, pre_idx, 8192, indices, None)
    current_profile[1] = 103
    ss_host._run_impl(logits, pre_idx, 8192, indices, None)
    current_profile[:] = [160, 100]
    ss_host._run_impl(logits, pre_idx, 8192, indices, None)
    assert builds == [
        (160, 8192, 8192, 512, 148, 100),
        (160, 8192, 8192, 512, 148, 103),
        (160, 8192, 8192, 512, 160, 100),
    ]
    assert set(ss_host._LAUNCH_CACHE) == {
        (160, 8192, 8192, 512, ss_host._pack_device_profile(148, 100)),
        (160, 8192, 8192, 512, ss_host._pack_device_profile(148, 103)),
        (160, 8192, 8192, 512, ss_host._pack_device_profile(160, 100)),
    }

    shape = (160, 8192, 512, 8192, 1, 1)
    launcher_sm100 = ("sm100",)
    launcher_sm103 = ("sm103",)
    monkeypatch.setattr(
        ss_host,
        "_VARLEN_CACHE",
        {
            (*shape, ss_host._pack_device_profile(148, 100)): launcher_sm100,
            (*shape, ss_host._pack_device_profile(148, 103)): launcher_sm103,
        },
    )
    assert ss_host._varlen_launcher(*shape, num_sms=148) is launcher_sm100
    assert ss_host._varlen_launcher(*shape, num_sms=148, sm_version=103) is launcher_sm103


def test_selfsampling_topk_varlen_rejects_unsupported_dtype():
    """Unsupported float64 logits fail before compilation; bf16 and fp16 are native dtypes."""
    logits = torch.randn(1, 8192, device=_DEV, dtype=torch.float64)
    kv = torch.tensor([8000], dtype=torch.int32, device=_DEV)
    out = torch.empty(1, 512, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="float32, bfloat16 or float16"):
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
    key = _varlen_cache_key(rows, npad, k, msl_c, nn, cr)
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


def test_selfsampling_route_k2048_regclus() -> None:
    """K=2048 rows above the register window take reg_clus with two hint
    slots per thread, except where the one-wave K=2048 register rung applies."""
    k = 2048
    for b, n in ((1, 16399), (8, 32771), (8, 65551), (16, 32771)):
        plan = ss_host.route(b, n, (n + 63) // 64 * 64, k)
        assert plan["kernel"] == "reg_clus" and plan["tpl"][3] == k, (b, n, plan["kernel"])
    for b in (32, 64):
        assert ss_host.route(b, 16399, 16448, k)["kernel"] == "reg"
    assert ss_host.route(1, 131087, 131136, k)["kernel"] == "main"
    assert ss_host.route(1, 65536, 65536, 1024)["tpl"][3] == 1024


@pytest.mark.parametrize("hint_free", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_selfsampling_regclus_static_k_cache(hint_free: bool, deterministic: bool) -> None:
    """Changing K at fixed cluster geometry must select the matching compiled variant."""
    b, n, npad = 2, 65551, 65600
    gen = torch.Generator(device=_DEV).manual_seed(20998)
    logits = torch.randn((b, npad), generator=gen, device=_DEV)
    logits[:, n:] = 3e38
    lengths = torch.full((b,), n, dtype=torch.int32, device=_DEV)
    # Revisit the first K after compiling both larger variants to exercise cache reuse.
    for k in (512, 1024, 2048, 512):
        assert ss_host.route(b, n, npad, k, det=deterministic)["kernel"] == "reg_clus"
        values, hints = logits[:, :n].topk(k)
        if hint_free:
            out = torch.full((b, k), -7, dtype=torch.int32, device=_DEV)
            ss_host.run_varlen(logits, lengths, out, max_seq_len=n, deterministic=deterministic)
        else:
            # A wider output follows the host's existing packed-row convention.
            out = torch.full((b, k + 64), -7, dtype=torch.int32, device=_DEV)
            ss_host.run(logits, hints.to(torch.int32), n, out, deterministic=deterministic)
            assert bool((out.flatten()[b * k :] == -7).all())
            out = out.flatten()[: b * k].view(b, k)
        torch.cuda.synchronize()
        _check_exact(logits, out[:, :k], n, values)


@pytest.mark.parametrize("pattern", ["randn", "ties", "equal"])
@pytest.mark.parametrize(
    "b,n", [(1, 16399), (4, 65551), (16, 32771), (1, 131072)], ids=lambda v: str(v)
)
def test_selfsampling_varlen_regclus_k2048_exact(b: int, n: int, pattern: str) -> None:
    """K=2048 reg_clus is exact on random, heavily tied and all-equal rows."""
    k, npad = 2048, (n + 63) // 64 * 64
    key = _varlen_cache_key(b, npad, k, n, 1, 1)
    gen = torch.Generator(device=_DEV).manual_seed(n + b)
    lg = torch.randn((b, npad), generator=gen, dtype=torch.float32, device=_DEV)
    if pattern == "ties":
        lg = (lg * 4).round() / 4
    elif pattern == "equal":
        lg.fill_(0.5)
    lg[:, n:] = 3e38
    kv = torch.full((b,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((b, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, max_seq_len=n)
    torch.cuda.synchronize()
    assert ss_host._VARLEN_CACHE[key][0] == "reg_clus"
    _check_exact(lg, out, n, torch.topk(lg[:, :n], k, dim=1).values)


def test_selfsampling_varlen_regclus_k2048_short_rows():
    """K=2048 reg_clus engine: short rows (n <= k, identity spans both hint
    slots), a zero-window row and long rows in one MTP launch."""
    k, msl, nn = 2048, 32771, 2
    npad = (msl + 63) // 64 * 64
    torch.manual_seed(13)
    lg = torch.randn(4 * nn, npad, dtype=torch.float32, device=_DEV)
    kv = torch.tensor([msl, 1500, nn - 1, 20000], dtype=torch.int32, device=_DEV)
    out = torch.full((4 * nn, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=nn, compress_ratio=1, max_seq_len=msl)
    assert ss_host._VARLEN_CACHE[_varlen_cache_key(4 * nn, npad, k, msl, nn, 1)][0] == "reg_clus"
    torch.cuda.synchronize()
    ref = _reference_varlen_indices(lg, kv, nn, 1, k)
    for r in range(4 * nn):
        valid = max(0, min(msl, int(kv[r // nn]) - nn + r % nn + 1))
        if valid <= k:
            assert torch.equal(out[r], ref[r]), f"row {r} short-row identity mismatch"
        else:
            row = lg[r]
            got = row[out[r].long().clamp_min(0)].sort().values
            want = row[ref[r].long().clamp_min(0)].sort().values
            assert torch.equal(got, want), f"row {r} value multiset mismatch"
            assert torch.equal(out[r] < 0, ref[r] < 0), f"row {r} pad mask mismatch"


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("b,n", [(1, 16399), (4, 65551), (8, 131071), (16, 32771)])
@pytest.mark.parametrize("pattern", ["hidden_peak", "outlier", "equal", "ties", "infinities"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_selfsampling_regclus_sampling_rescue(
    k: int, b: int, n: int, pattern: str, deterministic: bool
) -> None:
    """Exact selection and stable tie membership across all cluster fragment geometries."""
    npad = (n + 63) // 64 * 64
    gen = torch.Generator().manual_seed(20261026 + n + k)
    row = torch.randn(n, generator=gen)
    if pattern == "hidden_peak":
        row.fill_(1)
        # High scores occupy positions outside both endpoint-inclusive block samplers.
        sample_id = torch.arange(k)
        for width in (32, 64):
            positions = (sample_id // width) * (n - width) // (k // width - 1) + sample_id % width
            row[positions] = 0
    elif pattern == "outlier":
        row[0] = row[-1] = 1e9
    elif pattern == "equal":
        row.fill_(1)
    elif pattern == "ties":
        row.fill_(1)
        row[torch.randperm(n, generator=gen)[: k // 2]] = 2
    else:
        row[0::7] = float("inf")
        row[1::7] = -float("inf")
    logits = torch.full((b, npad), 3e38, dtype=torch.float32, device=_DEV)
    logits[:, :n] = row.to(_DEV)
    lengths = torch.full((b,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((b, k), -7, dtype=torch.int32, device=_DEV)
    kwargs = {"max_seq_len": n, "deterministic": deterministic}
    ss_host.run_varlen(logits, lengths, out, **kwargs)
    torch.cuda.synchronize()
    assert ss_host.route(b, n, npad, k, det=deterministic)["kernel"] == "reg_clus"
    reference = torch.topk(logits[:, :n], k, dim=1).values
    _check_exact(logits, out, n, reference)
    first = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ss_host.run_varlen(logits, lengths, out, **kwargs)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _check_exact(logits, out, n, reference)
        if deterministic:
            assert torch.equal(first, out), "rescue output order changed across graph replay"
    if deterministic:
        expected = torch.argsort(row, descending=True, stable=True)[:k].sort().values.to(_DEV)
        assert torch.equal(out.long().sort(dim=1).values, expected.expand(b, -1))
        ss_host.run_varlen(logits, lengths, out, sorting="ascending", **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(out.long(), expected.expand(b, -1))


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("deterministic", [False, True])
def test_selfsampling_regclus_rescue_negative_inf_tail(k: int, deterministic: bool) -> None:
    """Partial float4 slots must not duplicate the separately loaded scalar tail."""
    envelope = 131072
    valid_lengths = [base + tail for base in (k, 65536) for tail in (1, 2, 3)]
    rows = len(valid_lengths)
    assert ss_host.route(rows, envelope, envelope, k, det=deterministic)["kernel"] == "reg_clus"
    logits = torch.full((rows, envelope), 3e38, dtype=torch.float32, device=_DEV)
    for row, valid in enumerate(valid_lengths):
        logits[row, :valid] = -float("inf")
        # The threshold is -inf, while every real tail element is a strict winner.
        logits[row, valid // 4 * 4 : valid] = 1
    lengths = torch.tensor(valid_lengths, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, lengths, out, max_seq_len=envelope, deterministic=deterministic)
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        values = logits[row : row + 1, :valid]
        _check_exact(logits[row : row + 1], out[row : row + 1], valid, values.topk(k).values)
        if deterministic:
            expected = torch.argsort(values, descending=True, stable=True)[:, :k]
            assert torch.equal(out[row : row + 1].long().sort().values, expected.sort().values)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("candidates", [16384, 16385])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_selfsampling_regclus_rescue_candidate_limit(
    k: int, candidates: int, tied: bool, deterministic: bool
) -> None:
    """Both sides of the rescue limit preserve strict winners and threshold ties."""
    n, npad = 65551, 65600
    assert ss_host.route(1, n, npad, k, det=deterministic)["kernel"] == "reg_clus"
    logits = torch.full((1, npad), 3e38, dtype=torch.float32, device=_DEV)
    logits[:, :n] = 0
    logits[:, : k // 2] = 2
    begin = (n - candidates) // 2
    logits[:, begin : begin + candidates] = (
        1 if tied else torch.linspace(1, 1.0001, candidates, device=_DEV)
    )
    lengths = torch.tensor([n], dtype=torch.int32, device=_DEV)
    out = torch.full((1, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, lengths, out, max_seq_len=n, deterministic=deterministic)
    torch.cuda.synchronize()
    _check_exact(logits, out, n, logits[:, :n].topk(k).values)
    if deterministic:
        expected = torch.argsort(logits[:, :n], descending=True, stable=True)[:, :k]
        assert torch.equal(out.long().sort(dim=1).values, expected.sort(dim=1).values)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("deterministic", [False, True])
def test_selfsampling_regclus_rescue_mtp_short_rows(
    k: int, compress_ratio: int, deterministic: bool
) -> None:
    """An MTP launch may mix rescued long rows, identity rows and empty windows."""
    next_n, envelope = 4, 32771
    npad = (envelope + 63) // 64 * 64
    assert ss_host.route(4 * next_n, envelope, npad, k, det=deterministic)["kernel"] == "reg_clus"
    raw_lengths = [
        0,
        (k - 1) * compress_ratio,
        k * compress_ratio + next_n - 1,
        envelope * compress_ratio + next_n - 1,
    ]
    logits = torch.full((4 * next_n, npad), 3e38, dtype=torch.float32, device=_DEV)
    valid_lengths = []
    for row in range(4 * next_n):
        valid = max(
            0,
            min(
                envelope, (raw_lengths[row // next_n] - next_n + row % next_n + 1) // compress_ratio
            ),
        )
        valid_lengths.append(valid)
        logits[row, :valid] = 1
    lengths = torch.tensor(raw_lengths, dtype=torch.int32, device=_DEV)
    out = torch.full((4 * next_n, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(
        logits,
        lengths,
        out,
        next_n=next_n,
        compress_ratio=compress_ratio,
        max_seq_len=envelope * compress_ratio,
        deterministic=deterministic,
    )
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        if valid <= k:
            assert torch.equal(out[row, :valid], torch.arange(valid, device=_DEV))
            assert bool((out[row, valid:] == -1).all())
        else:
            _check_exact(
                logits[row : row + 1], out[row : row + 1], valid, torch.ones((1, k), device=_DEV)
            )
            if deterministic:
                assert torch.equal(out[row].sort().values, torch.arange(k, device=_DEV))


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
        key = _varlen_cache_key(rows, npad, k, msl_c, nn, cr)
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
        key = _varlen_cache_key(rows, npad, k, msl_c, nn, cr)
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
    key = _varlen_cache_key(rows, msl_c, k, msl_c, 1, cr)
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
    key = _varlen_cache_key(rows, msl_c, k, msl_c, 1, cr)
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


@pytest.mark.parametrize("rows,n", [(256, 8196), (1024, 9216), (256, 12288)])
def test_selfsampling_register_histogram_layout_exactness(rows: int, n: int) -> None:
    """Both register entry points use the shared-memory layout selected by NBH."""
    top_k = 512
    npad = (n + 63) // 64 * 64
    plan = ss_host.route(rows, n, npad, top_k)
    assert plan["kernel"] == "reg" and plan["tpl"][7] == 1024
    assert plan["rt"]["IMGOFF"] == 1024
    assert plan["smem"] == (1024 + 2 * plan["rt"]["CMP"]) * 4

    logits, pre_idx, indices, ref_vals = _make_case(rows, n, top_k, seed=rows + n)
    ss_host.run(logits, pre_idx, n, indices)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n, ref_vals)

    kv_lens = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    indices.fill_(-7)
    ss_host.run_varlen(logits, kv_lens, indices, max_seq_len=n)
    torch.cuda.synchronize()
    _check_exact(logits, indices, n, ref_vals)


@pytest.mark.parametrize(
    "rows,n,top_k,expected_tpl",
    [
        (64, 8192, 512, (1024, 2, 1)),
        (148, 4100, 2048, (512, 3, 1)),
        (160, 8195, 512, (512, 4, 2)),
        (256, 6144, 1024, (512, 3, 2)),
        (296, 8192, 2048, (512, 4, 2)),
    ],
    ids=["wide_b64_k512", "wide_b148_k2048", "mid_b160_k512", "mid_b256_k1024", "mid_b296_k2048"],
)
def test_selfsampling_varlen_4k_8k_rungs_exact(
    rows: int, n: int, top_k: int, expected_tpl: tuple[int, int, int]
) -> None:
    """The 4K < n <= 8K register rungs under the hint-free varlen engine on a
    heterogeneous batch: full rows, mid-length rows, a short row (n <= k), a
    two-valued row (boundary tie class of thousands -> the crossing-bin count
    saturates, non-quad refine) and a heavy-tailed row."""
    npad = (n + 63) // 64 * 64
    plan = ss_host.route(rows, n, npad, top_k)
    assert plan["kernel"] == "reg" and tuple(plan["tpl"][:3]) == expected_tpl, plan["tpl"]
    gen = torch.Generator(device=_DEV).manual_seed(rows + n)
    logits = torch.randn((rows, npad), generator=gen, dtype=torch.float32, device=_DEV) - 2.0
    kv = [n - (i % 7) * 3 for i in range(rows)]
    kv[1] = n // 2
    kv[2] = max(top_k - 5, 1)  # short row: identity head + -1 tail
    logits[3, : kv[3]] = torch.randint(0, 2, (kv[3],), generator=gen, device=_DEV).float()
    cauchy = torch.randn((kv[4],), generator=gen, device=_DEV) / torch.randn(
        (kv[4],), generator=gen, device=_DEV
    )
    logits[4, : kv[4]] = cauchy.clamp(-1e30, 1e30)
    for r in range(rows):
        logits[r, kv[r] :] = 3e38  # poison beyond each row's own length
    kv_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, kv_lens, out, next_n=1, compress_ratio=1, max_seq_len=n)
    torch.cuda.synchronize()
    key = _varlen_cache_key(rows, npad, top_k, n, 1, 1)
    assert ss_host._VARLEN_CACHE[key][0] == "reg", ss_host._VARLEN_CACHE[key][0]
    ref = _reference_varlen_indices(logits, kv_lens, 1, 1, top_k)
    for r in range(rows):
        row = logits[r]
        got = row[out[r].long().clamp_min(0)].sort().values
        want_v = row[ref[r].long().clamp_min(0)].sort().values
        assert torch.equal(got, want_v), f"row {r} value multiset mismatch"
        assert torch.equal(out[r] < 0, ref[r] < 0), f"row {r} pad mask mismatch"
        # the two-valued row has thousands of tied values: a duplicated index
        # would still match the value multiset, so pin uniqueness of the head
        head = out[r][out[r] >= 0]
        assert int(torch.unique(head).numel()) == head.numel(), f"row {r} duplicate indices"


def test_selfsampling_varlen_mid_rung_mtp_cr4():
    """The 148 < b <= 296 register rung under the production DSv4 window
    contract: compress_ratio=4, next_n=4 (40 requests x 4 rows), per-row n
    derived from KV tokens, with a short request and a zero-window request."""
    k, msl_c, cr, nn = 512, 8192, 4, 4
    rows = 160
    npad = msl_c
    plan = ss_host.route(rows, msl_c, npad, k)
    assert plan["kernel"] == "reg" and tuple(plan["tpl"][:3]) == (512, 4, 2), plan["tpl"]
    torch.manual_seed(23)
    lg = torch.randn(rows, npad, dtype=torch.float32, device=_DEV)
    kv_list = [msl_c * cr - (i % 5) * 7 for i in range(rows // nn)]
    kv_list[1] = (k - 3) * cr  # short rows: identity head + -1 tail
    kv_list[2] = nn - 1  # zero-window request: every row all -1
    kv_list[3] = msl_c * cr // 2
    kv = torch.tensor(kv_list, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=nn, compress_ratio=cr, max_seq_len=msl_c * cr)
    torch.cuda.synchronize()
    key = _varlen_cache_key(rows, npad, k, msl_c, nn, cr)
    assert ss_host._VARLEN_CACHE[key][0] == "reg", ss_host._VARLEN_CACHE[key][0]
    ref = _reference_varlen_indices(lg, kv, nn, cr, k)
    for r in range(rows):
        if (ref[r] >= 0).any():
            row = lg[r]
            got = row[out[r].long().clamp_min(0)].sort().values
            want_v = row[ref[r].long().clamp_min(0)].sort().values
            assert torch.equal(got, want_v), f"row {r} value multiset mismatch"
            assert torch.equal(out[r] < 0, ref[r] < 0), f"row {r} pad mask mismatch"
        else:
            assert torch.equal(out[r], ref[r]), f"row {r} expected all -1"


def test_selfsampling_varlen_mid_rung_cuda_graph():
    """The 148 < b <= 296 register rung must be CUDA-graph capturable: warmed
    engine, capture one launch, replay twice, tie-aware exact each time."""
    k, n, rows = 512, 8192, 160
    torch.manual_seed(19)
    lg = torch.randn(rows, n, dtype=torch.float32, device=_DEV)
    kv = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=1, max_seq_len=n)
    torch.cuda.synchronize()
    key = _varlen_cache_key(rows, n, k, n, 1, 1)
    assert ss_host._VARLEN_CACHE[key][0] == "reg", ss_host._VARLEN_CACHE[key][0]
    assert tuple(ss_host.route(rows, n, n, k)["tpl"][:3]) == (512, 4, 2)
    g = torch.cuda.CUDAGraph()
    out.fill_(-7)
    with torch.cuda.graph(g):
        ss_host.run_varlen(lg, kv, out, next_n=1, compress_ratio=1, max_seq_len=n)
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
        key = _varlen_cache_key(rows, msl_c, k, msl_c, 1, cr)
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


def test_selfsampling_varlen_narrow_logits_clamps_envelope_to_row_width() -> None:
    """Logits narrower than k + 1 with kv lengths beyond k: the launch
    envelope must be the physical row width, so every row takes the in-kernel
    short path (identity + -1 tail) instead of reading k + 1 elements across
    the row stride (DKG issue #60, item 1)."""
    rows, n, k = 2, 256, 512
    lg = torch.randn(rows, n, dtype=torch.float32, device=_DEV)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    lc = ss_host._varlen_launcher(rows, n, k, n, 1, 1)
    if lc[0] != "main":  # main derives the envelope from npad in-kernel
        env = lc[2] if not isinstance(lc[2], tuple) else lc[2][0]
        assert env == n, (lc[0], env)
    want = torch.full((rows, k), -1, dtype=torch.int32, device=_DEV)
    want[:, :n] = torch.arange(n, dtype=torch.int32, device=_DEV)
    for kv_len in (100000, n):  # kv far beyond the row width, and exactly the row width
        kv = torch.full((rows,), kv_len, dtype=torch.int32, device=_DEV)
        out.fill_(-7)
        ss_host.run_varlen(lg, kv, out, max_seq_len=n)
        torch.cuda.synchronize()
        assert torch.equal(out, want), kv_len


def test_selfsampling_warmup_incremental_rows_populate_exact_launchers() -> None:
    """A later warmup_varlen call whose new row count falls inside an already
    warmed engine band must still create that row count's launcher entry, so a
    CUDA-graph capture at it finds its key (DKG issue #60, item 2)."""
    k, msl, cr, nn = 512, 8192, 1, 1
    npad = (msl + 63) // 64 * 64
    ss_host.warmup_varlen(k, msl, compress_ratio=cr, next_n=nn, num_rows_list=(64,))
    key63 = _varlen_cache_key(63, npad, k, min(msl, npad), nn, cr)
    ss_host._VARLEN_CACHE.pop(key63, None)  # independent of earlier tests in this process
    ss_host.warmup_varlen(k, msl, compress_ratio=cr, next_n=nn, num_rows_list=(63, 64))
    assert key63 in ss_host._VARLEN_CACHE
    lg = torch.randn(63, npad, dtype=torch.float32, device=_DEV)
    kv = torch.full((63,), msl, dtype=torch.int32, device=_DEV)
    out = torch.empty(63, k, dtype=torch.int32, device=_DEV)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ss_host.run_varlen(lg[:, :msl], kv, out, max_seq_len=msl)
    g.replay()
    torch.cuda.synchronize()
    ref = torch.topk(lg[:, :msl], k, dim=1).values.sort(dim=1).values
    got = lg.gather(1, out.long().clamp_min(0)).sort(dim=1).values
    assert torch.equal(ref, got)


def test_validate_run_ws_requires_16_byte_alignment() -> None:
    """The workspace fake declares 16-byte alignment; the host check matches
    it so an 8-but-not-16-byte workspace fails here instead of at DSL
    conversion (DKG issue #60, item 3)."""
    logits = torch.zeros(1, 64, dtype=torch.float32, device=_DEV)
    base = torch.zeros(ss_host.WS_BYTES // 4 + 4, dtype=torch.int32, device=_DEV)
    ss_host.validate_run_ws(base[4:], logits)  # +16 B
    with pytest.raises(RuntimeError, match="16-byte"):
        ss_host.validate_run_ws(base[2:], logits)  # +8 B


# ===========================================================================
# ==== prefill: per-row [ks, ke) windows (run_prefill) ======================
# ===========================================================================
# Contract: Top-K of logits[r, ks:ke] in the local frame (column - ks), -1 pad,
# identity for nv <= k. ks % 4 in {1,2,3} is exercised with poison written at
# [ks-3, ks) and [ke, npad).


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
    """Tie-aware radix-parity check: -1 pad from lengths, unique head in [0, nv),
    identity for nv <= k; exact set when the k-th value is unique, else the
    strictly-above set plus a tie-class count. NaN-in-window rows: structure only."""
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


_DTYPE_IDS = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}
_PREFILL_DTYPES = pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=list(_DTYPE_IDS.values()))


def _make_prefill_case(
    rows, ncols, ks_list, ke_list, *, top_k, seed, dist="randn", dtype=torch.float32
):
    """DeepGEMM-like storage: stride = align(ncols + 256, 256), column slice
    [:, :ncols]; outside-window columns poisoned so an over-read/frame bug is
    caught (+inf at [ks-7, ks), rotating NaN/inf/3e38/-inf elsewhere). 16-bit
    cases are the RNE conversion of the fp32 construction (3e38 -> +inf in fp16,
    the finite 2.99e38 in bf16)."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    stride = ((ncols + 256 + 255) // 256) * 256
    if dist == "randn":
        full = torch.randn((rows, stride), generator=gen, dtype=torch.float32, device=_DEV)
    elif dist == "equal":
        full = torch.ones((rows, stride), dtype=torch.float32, device=_DEV)
    elif dist == "twoval":
        full = torch.randint(0, 2, (rows, stride), generator=gen, device=_DEV).float()
    elif dist == "subnormal":
        # fp16 subnormal band 2^-24 .. 2^-14 with random signs (I1): the K-th lands inside it
        full = torch.exp2(torch.rand((rows, stride), generator=gen, device=_DEV) * 10.0 - 24.0)
        full = torch.where(
            torch.rand((rows, stride), generator=gen, device=_DEV) < 0.5, -full, full
        )
    elif dist == "zero_band":
        # (0, 2^-25] casts to +-0 and (2^-25, 2^-24] to 0x0001 in fp16 (I5): the K-th lands inside
        full = torch.rand((rows, stride), generator=gen, device=_DEV) * 2.0**-24
        full = torch.where(
            torch.rand((rows, stride), generator=gen, device=_DEV) < 0.5, -full, full
        )
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
        full[r, max(ks - 7, 0) : ks] = float("inf")
    if dtype is not torch.float32:
        full = full.to(dtype)
        logits = full[:, :ncols]
    return logits, row_starts, row_ends


@_PREFILL_DTYPES
@pytest.mark.parametrize("top_k", [512, 1024, 2048], ids=lambda k: f"k{k}")
def test_prefill_causal_ramp(top_k, dtype):
    """Causal ramp (ks=0), 148 rows with nv = 1..148 < k: one tier-0 launch, all
    short-row identity. k boundary: test_prefill_short_rows; mixed short/long
    rows in one launch: test_prefill_packed_misaligned_ks."""
    rows = 148
    ks = [0] * rows
    ke = list(range(1, rows + 1))
    lg, rs, re = _make_prefill_case(rows, rows, ks, ke, top_k=top_k, seed=top_k, dtype=dtype)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, top_k)


@pytest.mark.parametrize(
    "dtype, lead",
    [(torch.float32, 1), (torch.float32, 2), (torch.float32, 3)]
    + [(dtype, lead) for dtype in (torch.bfloat16, torch.float16) for lead in range(1, 8)],
    ids=lambda v: f"lead{v}" if isinstance(v, int) else _DTYPE_IDS[v],
)
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_packed_misaligned_ks(top_k, lead, dtype):
    """Multi-request chunk: request 2 starts at ks % vector == lead (vector =
    4 fp32 / 8 bf16 elements) with +inf poison at [ks-lead, ks). A leaked lead
    lane would become top-1 (wrong)."""
    a = 300
    ks1 = ((a + 7) // 8) * 8 + lead
    n1 = 4096 + 17
    rows = a + n1
    ncols = ks1 + n1
    ks = [0] * a + [ks1] * n1
    ke = list(range(1, a + 1)) + [ks1 + n1] * n1
    lg, rs, re = _make_prefill_case(
        rows, ncols, ks, ke, top_k=top_k, seed=top_k * 100 + lead, dtype=dtype
    )
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    assert int(out.min()) >= -1, "negative index leaked (missed -lead correction / guard)"
    _check_prefill_exact(lg, out, rs, re, top_k)


@_PREFILL_DTYPES
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_short_rows(top_k, dtype):
    """nv in {0, 1, k-1, k, k+1}: identity 0..nv-1 + trailing -1 (radix short
    contract); nv==0 (ks==ke) -> all -1."""
    for nv in (0, 1, top_k - 1, top_k, top_k + 1):
        rows = 4
        ncols = max(nv, 1) + 8
        ks = [0] * rows
        ke = [nv] * rows
        lg, rs, re = _make_prefill_case(
            rows, ncols, ks, ke, top_k=top_k, seed=nv + top_k, dtype=dtype
        )
        out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_prefill(lg, rs, re, out)
        torch.cuda.synchronize()
        _check_prefill_exact(lg, out, rs, re, top_k)


@_PREFILL_DTYPES
@pytest.mark.parametrize("dist", ["equal", "twoval", "subnormal", "zero_band"], ids=lambda d: d)
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_ties_degenerate(top_k, dist, dtype):
    """All-equal (whole tie class) and two-valued (massive ties) rows drive the
    degenerate A/B narrowing paths; the fp16 subnormal and zero bands put the
    K-th among tiny codes (plain small normals in fp32/bf16); tie-aware acceptance."""
    rows = 16
    n = 4096
    ks = [0] * rows
    ke = [n] * rows
    lg, rs, re = _make_prefill_case(
        rows, n, ks, ke, top_k=top_k, seed=top_k, dist=dist, dtype=dtype
    )
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out)
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, top_k)


@pytest.mark.parametrize(
    "dtype, lead",
    [(torch.float32, 1), (torch.float32, 2), (torch.float32, 3)]
    + [(dtype, lead) for dtype in (torch.bfloat16, torch.float16) for lead in (1, 3, 5, 7)],
    ids=lambda v: f"lead{v}" if isinstance(v, int) else _DTYPE_IDS[v],
)
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_neginf_tie_class(top_k, lead, dtype):
    """nv > k with fewer than k finite values -> the k-th boundary is in the
    -inf tie class (degen B), crossed with misaligned lead. A -inf-valued mask
    would be emitted here as a negative index (== -lead); assert none leaks."""
    n_finite = top_k - 100
    nv = top_k + 400
    ks1 = ((37 + 7) // 8) * 8 + lead
    ncols = ks1 + nv
    rows = 5
    gen = torch.Generator(device=_DEV).manual_seed(top_k * 10 + lead)
    stride = ((ncols + 256 + 255) // 256) * 256
    full = torch.full((rows, stride), float("-inf"), dtype=torch.float32, device=_DEV)
    for r in range(rows):
        full[r, ks1 : ks1 + n_finite] = torch.randn(n_finite, generator=gen, device=_DEV)
        full[r, :ks1] = float("inf")
        full[r, ks1 + nv :] = 3e38
    full = full.to(dtype)
    logits = full[:, :ncols]
    rs = torch.tensor([ks1] * rows, dtype=torch.int32, device=_DEV)
    re = torch.tensor([ks1 + nv] * rows, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(logits, rs, re, out)
    torch.cuda.synchronize()
    assert int(out.min()) >= -1, "negative index leaked in a -inf tie class"
    _check_prefill_exact(logits, out, rs, re, top_k)


@_PREFILL_DTYPES
@pytest.mark.parametrize("top_k, n", [(512, 4099), (2048, 131075)], ids=lambda v: f"n{v}")
def test_prefill_deepgemm_single_row_odd_width(top_k, n, dtype):
    """A 1-row tile with an odd num_k_tokens on a DeepGEMM-strided view must
    use stride(0) (not shape[1]) and stay exact — the fully-cached follow-up
    turn that the varlen 1-row rule would wrongly reject."""
    rows = 1
    ks = [0]
    ke = [n]
    lg, rs, re = _make_prefill_case(rows, n, ks, ke, top_k=top_k, seed=n, dtype=dtype)
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
    from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling as dev

    tpl = (256, 8, 4, 256, 2, False, False, 1, 0, 1)
    a = dev.get_compiled(tpl, hint_free=True)
    b = dev.get_compiled(tpl, hint_free=True, prefill=True)
    assert a is not b
    # the bf16 prefill engine is a third, dtype-keyed specialization
    tpl16 = (256, 4, 4, 256, 2, False, False, 1, 0, 1)
    c = dev.get_compiled(tpl16, hint_free=True, prefill=True, dtype="bf16")
    d16 = dev.get_compiled(tpl16, hint_free=True, dtype="bf16")
    assert c is not d16 and c is not dev.get_compiled(tpl16, hint_free=True, prefill=True)
    # the fp16 engines are keyed apart from both the fp32 and the bf16 ones
    h = dev.get_compiled(tpl16, hint_free=True, prefill=True, dtype="f16")
    assert h is not c and h is not dev.get_compiled(tpl16, hint_free=True, dtype="f16")


def test_prefill_guards():
    k = 512
    n = 4096
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((3, stride), dtype=torch.float32, device=_DEV)[:, :n]
    rs = torch.zeros((3,), dtype=torch.int32, device=_DEV)
    re = torch.full((3,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((3, k), -7, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="float32, bfloat16 or float16"):
        ss_host.run_prefill(lg.to(torch.float64), rs, re, out)
    for dtype in (torch.bfloat16, torch.float16):
        # 16-bit rows must keep a 16-byte (8-element) row stride; a float4-only stride is rejected
        lg16_bad = torch.randn((3, n + 4), dtype=dtype, device=_DEV)[:, :n]
        with pytest.raises(RuntimeError, match="multiple of 8"):
            ss_host.run_prefill(lg16_bad, rs, re, out)
        # a well-formed 16-bit view runs (and is exact)
        lg16 = lg.to(dtype)
        out16 = torch.full((3, k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_prefill(lg16, rs, re, out16)
        torch.cuda.synchronize()
        _check_prefill_exact(lg16, out16, rs, re, k)
    with pytest.raises(RuntimeError, match="row_starts"):
        ss_host.run_prefill(lg, rs.to(torch.int64), re, out)
    with pytest.raises(RuntimeError, match="row_starts/row_ends length"):
        ss_host.run_prefill(lg, rs[:2], re, out)
    with pytest.raises(RuntimeError, match="multiple of 4"):
        ss_host.run_prefill(lg, rs, re, torch.full((3, k + 2), -7, dtype=torch.int32, device=_DEV))
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        ss_host.run_prefill(lg[:, 1:], rs, re, out)  # base offset by 1 float


def test_prefill_fp16_exact() -> None:
    """Stage A of the FP16 arm: the fp16 prefill engine (main family, window mode) is exact
    across misaligned window starts, a short row, an identity row and poisoned out-of-window
    columns, and the readiness query sees the compiled engine."""
    rows, n, k = 8, 8192, 512
    ks = [0, 5, 9, 16, 0, 3, 7, 1]
    ke = [n, n - 3, n // 2, n, k + 1, k - 5, n - 100, n]
    lg, rs, re = _make_prefill_case(rows, n, ks, ke, top_k=k, seed=1616, dtype=torch.float16)
    out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_prefill(lg, rs, re, out, max_row_len=n)
    torch.cuda.synchronize()
    _check_prefill_exact(lg, out, rs, re, k)
    assert ss_host.prefill_ready(lg, out, max_row_len=n)


@_PREFILL_DTYPES
def test_prefill_warmup_idempotent_and_no_rejit(dtype):
    from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling as dev

    k = 512
    ss_host.warmup_prefill(k, 32768, dtype=dtype)
    before = len(ss_host._PREFILL_WARMUP_DONE)
    ss_host.warmup_prefill(k, 32768, dtype=dtype)
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
                lg = torch.zeros((rows, stride), dtype=dtype, device=_DEV)[:, :nkv]
                rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
                re = torch.full((rows,), nkv, dtype=torch.int32, device=_DEV)
                out = torch.empty((rows, k), dtype=torch.int32, device=_DEV)
                ss_host.run_prefill(lg, rs, re, out, max_row_len=nkv)
    finally:
        dev.get_compiled = orig
    assert calls["n"] == 0, f"warmup missed keys: {calls['n']} live compiles"
    tag = _DTYPE_IDS[dtype]
    # the reachable engine set for this k above the k-th column: one tier-0 key
    # per bucket where a <= 148-row launch keeps the BLK=1024 plan at either
    # bucket edge, plus the bucket-free tier-1 and tier-2 keys — and nothing else
    # (buckets <= k belong to identity-only launches other tests make directly;
    # production never requests them: scores.shape[1] <= top_k takes the radix path)
    lo = ss_host._prefill_bucket(k + 1)
    want = {ss_host._prefill_cache_key(1, k, 0, tag), ss_host._prefill_cache_key(2, k, 0, tag)}
    bk = lo
    while bk <= 32768:
        for n_env in (bk // 2 + 1, bk):
            if ss_host._prefill_tier(1, n_env, k) == 0:
                want.add(ss_host._prefill_cache_key(0, k, bk, tag))
        bk <<= 1
    # deterministic prefill keys carry a fifth element; an earlier deterministic
    # test in the same process must not enter the comparison
    have = {
        kk
        for kk in ss_host._PREFILL_CACHE
        if len(kk) == 4 and kk[1] == k and kk[3] == tag and (kk[0] != 0 or kk[2] >= lo)
    }
    assert have == want, (have, want)
    # the capture-time readiness query agrees with the cache: warmed shape
    # ready, an unwarmed k (4 is valid but never compiled) not ready
    assert ss_host.prefill_ready(lg, out)
    assert not ss_host.prefill_ready(lg[:1], torch.empty((1, 4), dtype=torch.int32, device=_DEV))


@_PREFILL_DTYPES
def test_prefill_capture_no_host_sync(dtype):
    """After warmup, a run_prefill call captures under a CUDA graph (proves no
    .item()/.max() host read)."""
    k = 512
    n = 8192
    ss_host.warmup_prefill(k, max(n, 32768), dtype=dtype)
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((64, stride), dtype=torch.float32, device=_DEV).to(dtype)[:, :n]
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


@_PREFILL_DTYPES
def test_prefill_small_envelope_tier0_uses_sampled_plan(dtype):
    """<= 148-row prefill launches whose envelope lies above the tier-1 BLK=512
    plan's pair-sample gate (4096 for k <= 1024, 8192 for k > 1024) by the
    margin and at most 16384 must take that engine: the BLK=1024 tier-0 plan's
    compile-time gate is n > 16384, so under such an envelope every row would
    run the unsampled path. Tier selection is CPU-only; the launches below check
    the (tier, k, bucket) key each envelope creates, prefill_ready's agreement
    with it, and exactness on causal-ramp tail rows (ks = 0) and on a
    prefix-cache-hit window with a misaligned start (ks % 4 = lead) for every
    production K."""
    for rows in (1, 75, 148):
        for k in (512, 1024):
            for n_env in (1024, 2048, 4096, 4352):
                assert ss_host._prefill_tier(rows, n_env, k) == 0
            for n_env in (4353, 5000, 8192, 16384):
                assert ss_host._prefill_tier(rows, n_env, k) == 1
            assert (
                ss_host._prefill_tier(rows, 16385, k) == 0
                and ss_host._prefill_tier(rows, 32768, k) == 0
            )
        for n_env in (4099, 8192, 8448):
            assert ss_host._prefill_tier(rows, n_env, 2048) == 0
        for n_env in (8449, 12288, 16384):
            assert ss_host._prefill_tier(rows, n_env, 2048) == 1
        assert ss_host._prefill_tier(rows, 32768, 2048) == 0
    for rows in (149, 296):
        assert (
            ss_host._prefill_tier(rows, 4099, 512) == 1
            and ss_host._prefill_tier(rows, 32768, 512) == 1
        )
    assert (
        ss_host._prefill_tier(297, 8192, 512) == 2 and ss_host._prefill_tier(297, 32768, 2048) == 2
    )
    rows = 64
    for k in (512, 1024, 2048):
        for n in (4099, 8192, 16384, 32768):
            tier = ss_host._prefill_tier(rows, n, k)
            assert tier == (
                1 if ss_host._prefill_scpb_tier1(k) + ss_host._PREFILL_T1_MARGIN < n <= 16384 else 0
            )
            ks = [0] * rows
            ke = list(range(n - rows + 1, n + 1))
            lg, rs, re = _make_prefill_case(rows, n, ks, ke, top_k=k, seed=n + k, dtype=dtype)
            out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
            ss_host.run_prefill(lg, rs, re, out, max_row_len=n)
            torch.cuda.synchronize()
            bucket = ss_host._prefill_bucket(n)
            assert (
                ss_host._prefill_cache_key(tier, k, bucket, _DTYPE_IDS[dtype])
                in ss_host._PREFILL_CACHE
            )
            assert ss_host.prefill_ready(lg, out, max_row_len=n)
            _check_prefill_exact(lg, out, rs, re, k)
        # prefix-cache-hit window on the small-envelope tier-1 engine: ks % 4 == 1
        n = 8192
        ks = [4 * 100 + 1] * rows
        ke = list(range(n - rows + 1, n + 1))
        lg, rs, re = _make_prefill_case(rows, n, ks, ke, top_k=k, seed=7 * k, dtype=dtype)
        out = torch.full((rows, k), -7, dtype=torch.int32, device=_DEV)
        ss_host.run_prefill(lg, rs, re, out, max_row_len=n)
        torch.cuda.synchronize()
        _check_prefill_exact(lg, out, rs, re, k)


def _bf16_varlen_lengths(kv: list[int], next_n: int, compress_ratio: int, ncols: int) -> list[int]:
    return [
        min(max(kv[row // next_n] - next_n + row % next_n + 1, 0) // compress_ratio, ncols)
        for row in range(len(kv) * next_n)
    ]


_LOW_DTYPES = (torch.bfloat16, torch.float16)
_LOW_DTYPE_PARAMS = pytest.mark.parametrize(
    "dtype", _LOW_DTYPES, ids=[_DTYPE_IDS[dtype] for dtype in _LOW_DTYPES]
)
# fp16 edge patterns of DESIGN 2.8 (each exercises a numbered invariant; valid, if unremarkable, in bf16)
_FP16_EDGE_PATTERNS = (
    "subnormal",
    "zero_band",
    "zero_kth_pos",
    "zero_kth_neg",
    "posinf_many",
    "fp16max",
)


def _make_low_varlen_case(
    rows: int,
    ncols: int,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    pattern: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """16-bit varlen case in an arena of ``dtype`` (RNE cast of the fp32 construction): aligned
    storage offset, odd logical width (rounded up to a multiple of 8 for a single row, whose
    width is its npad), per-request windows (full row, shortened row, short path, empty).
    Patterns: the BF16 originals (ties / infinities / crosszero) and the fp16 edge patterns --
    subnormal band 2^-24..2^-14 with the K-th inside it (I1), zero band (0, 2^-25] -> +-0 and
    (2^-25, 2^-24] -> 0x0001 with the K-th inside it (I5), the K-th exactly +0.0 or -0.0 (6.N6),
    more than K elements equal to +inf (6.N7), the K-th at the fp16 maximum 65504 with fp32
    values above it casting to +inf (I6)."""
    generator = torch.Generator(device=_DEV).manual_seed(rows + ncols + top_k)
    # An aligned storage offset and an odd logical width exercise arena views.
    stride = (ncols + 255) // 256 * 256 + (256 if rows > 1 else 0)
    offset = 8
    storage = torch.empty(rows * stride + offset, dtype=dtype, device=_DEV)
    arena = storage[offset:].view(rows, stride)
    arena.copy_(torch.randn((rows, stride), generator=generator, device=_DEV))
    if pattern == "ties":
        arena.copy_((arena.float() * 2).round())
        arena[:, 0::7] = -0.0
        arena[:, 1::7] = 0.0
    elif pattern == "infinities":
        arena[:, 17] = float("inf")
        arena[1::4] = float("-inf")
        arena[2::4, top_k // 2 :] = float("-inf")
        arena[3::4, : top_k + 11] = float("inf")
    elif pattern == "crosszero":
        arena.copy_(-arena.float().abs() * 0.015625)
        arena[:, : top_k // 2].abs_()
        arena[:, top_k // 2 : 3 * top_k // 4] = 0.0
    elif pattern == "subnormal":
        # magnitudes 2^-24 .. 2^-14; top_k // 2 positives, so the K-th is a negative band value
        magnitude = torch.exp2(
            torch.rand((rows, stride), generator=generator, device=_DEV) * 10.0 - 24.0
        )
        arena.copy_(-magnitude)
        arena[:, : top_k // 2] = magnitude[:, : top_k // 2]
    elif pattern == "zero_band":
        arena.copy_(-arena.float().abs() - 0.5)
        band = torch.randperm(ncols, generator=generator, device=_DEV)[: 2 * top_k]
        signs = torch.where(
            torch.rand((rows, band.numel()), generator=generator, device=_DEV) < 0.5, -1.0, 1.0
        )
        arena[:, band] = (
            torch.rand((rows, band.numel()), generator=generator, device=_DEV) * 2.0**-24 * signs
        ).to(dtype)
    elif pattern in ("zero_kth_pos", "zero_kth_neg"):
        arena.copy_(-arena.float().abs() - 0.5)
        zeros = torch.randperm(ncols, generator=generator, device=_DEV)[: top_k + top_k // 4]
        arena[:, zeros] = 0.0 if pattern == "zero_kth_pos" else -0.0
    elif pattern == "posinf_many":
        many = torch.randperm(ncols, generator=generator, device=_DEV)[: top_k + 7]
        arena[:, many] = float("inf")
    elif pattern == "fp16max":
        # fp16 rounds the finite construction to the 65k binade; 70000 casts to +inf (finite in bf16)
        arena.copy_(torch.randn((rows, stride), generator=generator, device=_DEV) * 8.0 + 65400.0)
        arena[:, :5] = 70000.0
        arena[:, 5 : 5 + top_k + top_k // 4] = 65504.0
    else:
        raise ValueError(f"unknown test pattern: {pattern}")
    # A single row has no row stride: the host takes its logical width as npad (multiple of 8).
    logits = arena[:, : ncols if rows > 1 else (ncols + 7) // 8 * 8]
    kv_pattern = [ncols * compress_ratio, (ncols - 7) * compress_ratio, top_k - 3, 0]
    kv = [kv_pattern[request % len(kv_pattern)] for request in range(rows // next_n)]
    lengths = _bf16_varlen_lengths(kv, next_n, compress_ratio, ncols)
    for row, valid in enumerate(lengths):
        arena[row, valid:] = float("inf")
    return logits, torch.tensor(kv, dtype=torch.int32, device=_DEV), lengths


def _check_low_varlen_exact(
    logits: torch.Tensor,
    indices: torch.Tensor,
    lengths: list[int],
    values: torch.Tensor | None = None,
) -> None:
    top_k = indices.shape[1]
    output_values = None
    if values is not None:
        # Optional values retain the existing flattened [rows * K] ABI even
        # when the supplied scratch buffer has more than K columns.
        count = logits.shape[0] * top_k
        output_values = values.flatten()[:count].view(-1, top_k)
        assert bool((values.flatten()[count:] == 7.0).all())
    for row, valid in enumerate(lengths):
        count = min(valid, top_k)
        idx = indices[row, :count].long()
        assert bool(((idx >= 0) & (idx < valid)).all()), f"row {row}: out-of-range index"
        assert idx.unique().numel() == count, f"row {row}: duplicate indices"
        assert bool((indices[row, count:] == -1).all()), f"row {row}: invalid tail padding"
        gathered = logits[row, idx]
        reference = torch.topk(logits[row, :valid], count).values
        assert torch.equal(
            (gathered.float() + 0.0).sort().values,
            (reference.float() + 0.0).sort().values,
        ), f"row {row}: {logits.dtype} top-K value multiset mismatch"
        if output_values is not None:
            assert torch.equal(output_values[row, :count], gathered.float())
            assert bool((output_values[row, count:] == torch.finfo(torch.float32).min).all())


# stage-A names of the two helpers (kept for tests written against them)
_make_bf16_varlen_case = _make_low_varlen_case
_check_bf16_varlen_exact = _check_low_varlen_exact


def _launcher_family(
    dtype: torch.dtype,
    logits: torch.Tensor,
    top_k: int,
    n_env: int,
    next_n: int,
    compress_ratio: int,
) -> str:
    """Family of the cached 16-bit launcher a ``run_varlen`` call used (the launcher key layout
    of ``_run_varlen_16``: rows, row stride, K, clamped envelope, next_n, cr, device, profile)."""
    rows = logits.shape[0]
    npad = logits.stride(0) if rows > 1 else logits.shape[1]
    key = _varlen_cache_key(
        rows,
        npad,
        top_k,
        min(max(n_env, 1), npad),
        next_n,
        compress_ratio,
        torch.cuda.current_device(),
    )
    return str(ss_host._varlen_cache_for(dtype)[key][0])


_LOW_VARLEN_CASES = [
    pytest.param(1, 65536, 2048, 1, 1, "ties", id="hybrid128"),
    pytest.param(4, 131072, 1024, 1, 4, "infinities", id="regclus_cr4"),
    pytest.param(4, 262144, 2048, 1, 1, "infinities", id="regclus_cs16"),
    pytest.param(4, 32768, 512, 4, 4, "ties", id="regclus_mtp4"),
    pytest.param(4, 8192, 2048, 4, 1, "infinities", id="reg_mtp4"),
    pytest.param(8, 4099, 512, 4, 4, "ties", id="reg_odd_width"),
    pytest.param(4, 2048, 1024, 1, 4, "infinities", id="reg_small"),
    pytest.param(64, 65536, 2048, 1, 1, "crosszero", id="dense_crosszero_b64"),
    pytest.param(128, 65536, 2048, 1, 1, "crosszero", id="dense_crosszero_b128"),
    pytest.param(16, 262144, 2048, 1, 1, "infinities", id="dense_split"),
    pytest.param(256, 32768, 1024, 1, 4, "ties", id="main_nonsplit"),
    pytest.param(32, 262144, 512, 4, 4, "infinities", id="cluster_mtp4"),
    pytest.param(4, 8195, 512, 1, 1, "infinities", id="reg_odd_tail"),
]


@_LOW_DTYPE_PARAMS
@pytest.mark.parametrize("rows,ncols,top_k,next_n,compress_ratio,pattern", _LOW_VARLEN_CASES)
def test_selfsampling_topk_varlen_16bit_exactness(
    rows: int,
    ncols: int,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    pattern: str,
    dtype: torch.dtype,
) -> None:
    """Native 16-bit paths preserve values, index validity and padded row windows (an fp16 case
    control)."""
    logits, kv_lens, lengths = _make_low_varlen_case(
        rows, ncols, top_k, next_n, compress_ratio, pattern, dtype
    )
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        next_n=next_n,
        compress_ratio=compress_ratio,
        values=values,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    _check_low_varlen_exact(logits, indices, lengths, values)


@_LOW_DTYPE_PARAMS
@pytest.mark.parametrize("pattern", _FP16_EDGE_PATTERNS)
@pytest.mark.parametrize(
    "rows,ncols,top_k,compress_ratio,family",
    [
        pytest.param(128, 65536, 2048, 1, "main", id="main_dense"),
        pytest.param(512, 8192, 512, 4, "main", id="main_v16"),
        pytest.param(64, 65536, 1024, 4, "clus", id="clus"),
        pytest.param(4, 2048, 1024, 4, None, id="reg_small"),
        pytest.param(4, 131072, 1024, 4, None, id="regclus_cr4"),
        pytest.param(8, 1027, 1024, 4, "complement", id="complement"),
    ],
)
def test_selfsampling_topk_varlen_16bit_edge_patterns(
    rows: int,
    ncols: int,
    top_k: int,
    compress_ratio: int,
    family: str | None,
    pattern: str,
    dtype: torch.dtype,
) -> None:
    """fp16 edge patterns (DESIGN 2.8, 6.N5-N7) on every family -- subnormal band, zero band,
    K-th exactly +-0, more than K infinities, 65504 at the K-th -- with bf16 running the same
    rows as the control."""
    logits, kv_lens, lengths = _make_low_varlen_case(
        rows, ncols, top_k, 1, compress_ratio, pattern, dtype
    )
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        compress_ratio=compress_ratio,
        values=values,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    if family is not None:
        assert _launcher_family(dtype, logits, top_k, ncols, 1, compress_ratio) == family
    _check_low_varlen_exact(logits, indices, lengths, values)


@pytest.mark.parametrize(
    "rows,ncols,top_k,compress_ratio,family",
    [
        pytest.param(128, 65536, 2048, 1, "main", id="main_dense"),
        pytest.param(512, 8192, 512, 4, "main", id="main_v16"),
        pytest.param(512, 32768, 2048, 1, "main", id="main_v16_dense"),
        pytest.param(256, 32768, 1024, 4, "main", id="main_blk512"),
        pytest.param(1, 262144, 1024, 4, "main", id="main_split_bs1"),
        pytest.param(64, 65536, 1024, 4, "clus", id="clus"),
    ],
)
def test_selfsampling_topk_varlen_fp16_main_clus_smoke(
    rows: int, ncols: int, top_k: int, compress_ratio: int, family: str
) -> None:
    """Native FP16 on the streaming families: tie-aware exact
    top-K values, valid unique indices, -1 tails and the FP32 values output on random fp16
    rows whose windows cover the full row, a shortened row, the short path and an empty row."""
    generator = torch.Generator(device=_DEV).manual_seed(rows + ncols + top_k)
    logits = torch.randn((rows, ncols), generator=generator, device=_DEV).to(torch.float16)
    kv_pattern = [ncols * compress_ratio, (ncols - 7) * compress_ratio, top_k - 3, 0]
    kv = [kv_pattern[row % len(kv_pattern)] for row in range(rows)]
    lengths = _bf16_varlen_lengths(kv, 1, compress_ratio, ncols)
    for row, valid in enumerate(lengths):
        logits[row, valid:] = float("inf")
    kv_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        compress_ratio=compress_ratio,
        values=values,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    key = _varlen_cache_key(
        rows, ncols, top_k, ncols, 1, compress_ratio, torch.cuda.current_device()
    )
    assert ss_host._VARLEN_CACHE_FP16[key][0] == family
    _check_low_varlen_exact(logits, indices, lengths, values)


_FP16_REG_PATTERNS = (
    "ties",
    "infinities",
    "crosszero",
    "narrow_band",
    "subnormal_band",
    "zero_band",
    "zero_kth",
    "max65504",
    "posinf_many",
)


def _make_fp16_reg_case(
    rows: int,
    ncols: int,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """fp16 twin of ``_make_bf16_varlen_case``: an fp32 arena built per pattern and cast once
    (RNE) to float16, so the reference is torch.topk of exactly the rows the kernel reads.

    Beyond the three bf16 patterns, the fp16 ones exercise the numbered invariants of the FP16
    design: ``narrow_band`` (one binade [-2, -1) on a 1/256 grid: the K-th sits inside the 990-code
    bits-domain window, whole-bin emit of a tie class), ``subnormal_band`` (the first K columns hold
    the top-K with magnitudes 2^-24 .. 1.5 * 2^-14, so the window anchored at the K-th contains
    subnormal codes: every code16 site must take the halfword / cvt.rn path, I1), ``zero_band``
    ((0, 2^-25] -> +-0 and (2^-25, 2^-24] -> 0x0001, I5), ``zero_kth`` (the K-th exactly +-0),
    ``max65504`` (ties at -65504 / -65472 so the window anchor is the code next to -inf; fp32
    magnitudes above 65504 saturate to +-inf, I6) and ``posinf_many`` (more than K elements equal
    to +inf: the key-space escape).
    """
    generator = torch.Generator(device=_DEV).manual_seed(rows + ncols + top_k)
    stride = (ncols + 255) // 256 * 256 + (256 if rows > 1 else 0)
    offset = 8
    storage = torch.empty(rows * stride + offset, dtype=torch.float16, device=_DEV)
    arena = storage[offset:].view(rows, stride)
    base = torch.randn((rows, stride), generator=generator, device=_DEV)
    if pattern == "ties":
        base = (base * 2).round()
        base[:, 0::7] = -0.0
        base[:, 1::7] = 0.0
    elif pattern == "infinities":
        base[:, 17] = float("inf")
        base[1::4] = float("-inf")
        base[2::4, top_k // 2 :] = float("-inf")
        base[3::4, : top_k + 11] = float("inf")
    elif pattern == "crosszero":
        base = -base.abs() * 0.015625
        base[:, : top_k // 2].abs_()
        base[:, top_k // 2 : 3 * top_k // 4] = 0.0
    elif pattern == "narrow_band":
        base = -(1.0 + torch.rand((rows, stride), generator=generator, device=_DEV))
        base = (base * 256).round() / 256
    elif pattern == "subnormal_band":
        magnitude = torch.exp2(
            torch.rand((rows, stride), generator=generator, device=_DEV) * 3.0 - 13.0
        )
        head = torch.exp2(torch.rand((rows, top_k), generator=generator, device=_DEV) * 10.0 - 24.0)
        head[:, top_k - 1] = 1.5 * 2.0**-14
        magnitude[:, :top_k] = head
        base = -magnitude
        base[:, : top_k // 8] = magnitude[:, : top_k // 8]
    elif pattern == "zero_band":
        base = -base.abs() - 0.5
        band = torch.randperm(stride, generator=generator, device=_DEV)[: 2 * top_k]
        signs = torch.where(
            torch.rand((rows, 2 * top_k), generator=generator, device=_DEV) < 0.5, -1.0, 1.0
        )
        base[:, band] = (
            torch.rand((rows, 2 * top_k), generator=generator, device=_DEV) * 2.0**-24 * signs
        )
    elif pattern == "zero_kth":
        base = -base.abs() - 0.5
        zero_cols = torch.randperm(stride, generator=generator, device=_DEV)[: top_k + top_k // 4]
        base[:, zero_cols] = 0.0
        base[:, zero_cols[::2]] = -0.0
    elif pattern == "max65504":
        base = -((base * 2).round() + 65500.0)
        base[1::2, 5::97] = -70000.0
        base[:, 11::131] = 70000.0
    elif pattern == "posinf_many":
        base[:, torch.randperm(stride, generator=generator, device=_DEV)[: top_k + 7]] = float(
            "inf"
        )
    else:
        raise ValueError(f"unknown fp16 test pattern: {pattern}")
    arena.copy_(base)
    logits = arena[:, :ncols]
    kv_pattern = [ncols * compress_ratio, (ncols - 7) * compress_ratio, top_k - 3, 0]
    kv = [kv_pattern[request % len(kv_pattern)] for request in range(rows // next_n)]
    lengths = _bf16_varlen_lengths(kv, next_n, compress_ratio, ncols)
    for row, valid in enumerate(lengths):
        arena[row, valid:] = float("inf")
    return logits, torch.tensor(kv, dtype=torch.int32, device=_DEV), lengths


@pytest.mark.parametrize("pattern", _FP16_REG_PATTERNS)
@pytest.mark.parametrize(
    "rows,ncols,top_k,next_n,compress_ratio",
    [
        pytest.param(4, 8192, 2048, 4, 1, id="reg_mtp4"),
        pytest.param(8, 4099, 512, 4, 4, id="reg_odd_width"),
        pytest.param(4, 2048, 1024, 1, 4, id="reg_small"),
        pytest.param(4, 8195, 512, 1, 1, id="reg_odd_tail"),
    ],
)
def test_selfsampling_topk_varlen_fp16_reg_exactness(
    rows: int, ncols: int, top_k: int, next_n: int, compress_ratio: int, pattern: str
) -> None:
    """Native FP16 on the register family: the four register-routed shapes of the
    bf16 exactness set (bits-domain DEG window with and without the live packed words, bin
    proof, regimg, eager prefetch) under the bf16 patterns and the fp16 invariant patterns;
    tie-aware exact values, valid unique indices, -1 tails and the FP32 values output, with the
    register family asserted on the fp16 launcher cache."""
    logits, kv_lens, lengths = _make_fp16_reg_case(
        rows, ncols, top_k, next_n, compress_ratio, pattern
    )
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        next_n=next_n,
        compress_ratio=compress_ratio,
        values=values,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    npad = logits.stride(0) if rows > 1 else ncols
    families = {
        key: launcher[0]
        for key, launcher in ss_host._VARLEN_CACHE_FP16.items()
        if key[:6] == (rows, npad, top_k, ncols, next_n, compress_ratio)
    }
    assert set(families.values()) == {"reg"}, families
    _check_bf16_varlen_exact(logits, indices, lengths, values)


_FP16_REGCLUS_SHAPES = [
    pytest.param(1, 65536, 2048, 1, 1, id="hybrid128"),
    pytest.param(4, 131072, 1024, 1, 4, id="regclus_cr4"),
    pytest.param(4, 262144, 2048, 1, 1, id="regclus_cs16"),
    pytest.param(4, 32768, 512, 4, 4, id="regclus_mtp4"),
    # fp16 fkey16 window (stage B5) on the K=2048 geometries the bf16 window never ran (VPT == 1):
    # cs 2 / cs 4 / cs 8 single-vector builds and an MTP row set; the window either hits (whole-bin
    # emit of the K-th code) or misses into the linear rebuild, both tie-aware exact
    pytest.param(2, 16400, 2048, 1, 1, id="window_cs2"),
    pytest.param(4, 32768, 2048, 1, 1, id="window_cs4"),
    pytest.param(2, 65551, 2048, 1, 1, id="window_cs8_odd"),
    pytest.param(4, 32768, 2048, 4, 1, id="window_mtp4"),
]
_FP16_PATTERNS = (
    "ties",
    "infinities",
    "subnormal",
    "zero_kth",
    "large",
    "zero_band",
    "posinf_many",
)


def test_selfsampling_topk_fp16_dense_window_geometry() -> None:
    """The fp16 dense window keeps one exact code per bin over the value reach of the bf16
    window on the 1024-thread builds: 8x the bins (2,048, an 8 KB histogram), an 11-bit bin
    field in the staged word, a 1,023-code re-anchor below the sampled K-th, and the
    histogram's extra words taken from the staging capacity so the smem total (the
    shared-memory carveout) equals the bf16 build's. The 512/256-thread fp16 builds, bf16
    and fp32 builds, and fp16 linear-bin builds keep the 256-bin geometry and the
    module-level index width."""
    import cutlass

    from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling as dev

    def kernel(dtype, dense: bool, tpl):
        return dev.GvrMainKernel(
            *tpl, varlen=True, next_n=1, cr_shift=0, hint_free=True, dtype=dtype, dense=dense
        )

    def geometry(dtype, dense: bool, tpl=(1024, 4, 1, 256, 2, False, False)):
        kern = kernel(dtype, dense, tpl)
        return kern.nbd, kern.hb, kern.idxb, kern.idxm, kern.anchor_lo

    base = (256, 256, dev.IDXB__main, dev.IDXM__main, 127)
    wide = (2048, 2048, 21, (1 << 21) - 1, 1023)
    assert geometry(cutlass.Float16, True) == wide
    assert geometry(cutlass.Float16, True, (1024, 1, 1, 256, 2, True, False)) == wide
    assert geometry(cutlass.Float16, True, (512, 4, 2, 256, 4, False, False)) == base
    assert geometry(cutlass.Float16, True, (256, 4, 4, 256, 8, False, False)) == base
    assert geometry(cutlass.Float16, False) == base
    assert geometry(cutlass.BFloat16, True) == base
    assert geometry(cutlass.BFloat16, False) == base
    assert geometry(cutlass.Float32, False) == base
    assert dev.IDXB__main == 22
    for tpl in ((1024, 4, 1, 256, 2, False, False), (1024, 1, 1, 256, 2, True, False)):
        f16 = kernel(cutlass.Float16, True, tpl)
        bf16 = kernel(cutlass.BFloat16, True, tpl)
        # smem-neutral: blob + histogram equal; the staging gives up exactly the extra words
        assert f16.dyn_bytes + 4 * f16.hb == bf16.dyn_bytes + 4 * bf16.hb
        assert f16.scpb == f16.sample_scpb < bf16.scpb
        assert f16.cmpb == bf16.cmpb and f16.sampling_cap == bf16.sampling_cap
    for tpl in ((512, 4, 2, 256, 4, False, False), (256, 4, 4, 256, 8, False, False)):
        f16 = kernel(cutlass.Float16, True, tpl)
        bf16 = kernel(cutlass.BFloat16, True, tpl)
        assert (f16.hb, f16.scpb, f16.sample_scpb, f16.dyn_bytes) == (
            bf16.hb,
            bf16.scpb,
            bf16.sample_scpb,
            bf16.dyn_bytes,
        )


@pytest.mark.parametrize(
    "rows,ncols,top_k",
    [
        pytest.param(16, 262144, 2048, id="split"),
        pytest.param(128, 65536, 2048, id="blk1024"),
        pytest.param(1024, 65551, 2048, id="blk256_packed_idx"),
        pytest.param(512, 16399, 2048, id="v16"),
    ],
)
def test_selfsampling_topk_varlen_fp16_dense_window_exactness(
    rows: int, ncols: int, top_k: int
) -> None:
    """fp16 K=2048 rows whose top values spread over several binades (the K-th sits
    hundreds to thousands of fp16 codes above the sampled rung) stay exact on every
    dense-window build: SPLIT, 1024-thread, 256-thread packed-index and v16 staging."""
    generator = torch.Generator(device=_DEV).manual_seed(rows + ncols + top_k)
    # odd logical widths ride a 64-aligned row stride (the 16-bit host needs a multiple of 8);
    # the stride tail and every row's tail beyond its length carry the +inf pad
    stride = (ncols + 63) // 64 * 64
    arena = torch.randn((rows, stride), generator=generator, device=_DEV) - 4.0
    spread = torch.rand((rows, 3 * top_k), generator=generator, device=_DEV) * 60.0 + 0.5
    arena[:, : 3 * top_k] = spread  # wide upper tail: the K-th is ~2,048th of 6,144 spread values
    arena[1::3, top_k // 2 : top_k] = 7.0  # a tie class well below the K-th
    arena = arena.to(torch.float16)
    arena[:, ncols:] = float("inf")
    logits = arena[:, :ncols]
    kv_pattern = [ncols, ncols - 7, top_k - 3, 0]
    kv = [kv_pattern[row % len(kv_pattern)] for row in range(rows)]
    lengths = _bf16_varlen_lengths(kv, 1, 1, ncols)
    for row, valid in enumerate(lengths):
        arena[row, valid:] = float("inf")
    kv_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(logits, kv_lens, indices, values=values, max_seq_len=ncols)
    torch.cuda.synchronize()
    key = _varlen_cache_key(rows, stride, top_k, ncols, 1, 1, torch.cuda.current_device())
    assert ss_host._VARLEN_CACHE_FP16[key][0] == "main"
    _check_bf16_varlen_exact(logits, indices, lengths, values)


def _make_fp16_varlen_case(
    rows: int,
    ncols: int,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """fp16 twin of ``_make_bf16_varlen_case`` (same arena layout and KV pattern) with the fp16 patterns
    of DESIGN 2.8: ``subnormal`` = magnitudes 2^-24 .. 2^-14 around the K-th (I1 on every code16 site),
    ``zero_kth`` = more than K signed zeros so the K-th is exactly +-0 (6.N6), ``large`` = values near
    65504 with ties at 65504 and fp32 values above it saturating to +inf (I6), ``zero_band`` = elements in
    (0, 2^-24] that cast to +-0 or 0x0001 (I5), ``posinf_many`` = more than K elements equal to +inf
    (6.N7).  The fp32 construction is cast once (RNE) and the reference is the cast tensor."""
    generator = torch.Generator(device=_DEV).manual_seed(rows + ncols + top_k)
    stride = (ncols + 255) // 256 * 256 + (256 if rows > 1 else 0)
    offset = 8
    storage = torch.empty(rows * stride + offset, dtype=torch.float16, device=_DEV)
    arena = storage[offset:].view(rows, stride)
    base = torch.randn((rows, stride), generator=generator, device=_DEV)
    if pattern == "ties":
        base = (base * 2).round()
        base[:, 0::7] = -0.0
        base[:, 1::7] = 0.0
    elif pattern == "infinities":
        base[:, 17] = float("inf")
        base[1::4] = float("-inf")
        base[2::4, top_k // 2 :] = float("-inf")
        base[3::4, : top_k + 11] = float("inf")
    elif pattern == "subnormal":
        magnitude = torch.pow(
            2.0, torch.rand((rows, stride), generator=generator, device=_DEV) * 10.0 - 24.0
        )
        base = -magnitude
        base[:, : top_k // 2] = magnitude[:, : top_k // 2]
    elif pattern == "zero_kth":
        base = -base.abs() - 0.5
        zero_cols = torch.randperm(stride, generator=generator, device=_DEV)[: top_k + top_k // 4]
        base[:, zero_cols] = 0.0
        base[:, zero_cols[::2]] = -0.0
    elif pattern == "large":
        base = base * 8.0 + 65400.0
        base[:, 2::11] = 65504.0
        base[:, 5 :: max(ncols // 3, 1)] = 70000.0
    elif pattern == "zero_band":
        base = -base.abs() - 0.5
        band = torch.randperm(stride, generator=generator, device=_DEV)[: 2 * top_k]
        signs = torch.where(
            torch.rand((rows, 2 * top_k), generator=generator, device=_DEV) < 0.5, -1.0, 1.0
        )
        base[:, band] = torch.rand((rows, 2 * top_k), generator=generator, device=_DEV) * 2.0**-24
        base[:, band] *= signs
    elif pattern == "posinf_many":
        base = base * 1.5 - 3.5
        base[:, torch.randperm(stride, generator=generator, device=_DEV)[: top_k + 7]] = float(
            "inf"
        )
    else:
        raise ValueError(f"unknown fp16 test pattern: {pattern}")
    arena.copy_(base)
    logits = arena[:, :ncols]
    kv_pattern = [ncols * compress_ratio, (ncols - 7) * compress_ratio, top_k - 3, 0]
    kv = [kv_pattern[request % len(kv_pattern)] for request in range(rows // next_n)]
    lengths = _bf16_varlen_lengths(kv, next_n, compress_ratio, ncols)
    for row, valid in enumerate(lengths):
        arena[row, valid:] = float("inf")
    return logits, torch.tensor(kv, dtype=torch.int32, device=_DEV), lengths


@pytest.mark.parametrize("pattern", _FP16_PATTERNS)
@pytest.mark.parametrize("rows,ncols,top_k,next_n,compress_ratio", _FP16_REGCLUS_SHAPES)
def test_selfsampling_topk_varlen_fp16_regclus_exactness(
    rows: int,
    ncols: int,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    pattern: str,
) -> None:
    """Native FP16 on the clustered register family: the four bf16 reg_clus routes
    (hybrid128 = dense-key hybrid walk, cr4, cs16 = bits-domain window, mtp4) stay on reg_clus and are
    tie-aware exact on the bf16 patterns and the fp16-specific ones, with the FP32 values output.  The
    K=2048 shapes run the fp16 fkey16 window (stage B5: sampled-anchor single-code bins, 1024 bins);
    fp16 takes no hybrid plan, so hybrid128 is a window build too."""
    logits, kv_lens, lengths = _make_fp16_varlen_case(
        rows, ncols, top_k, next_n, compress_ratio, pattern
    )
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        next_n=next_n,
        compress_ratio=compress_ratio,
        values=values,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    npad = logits.stride(0) if rows > 1 else ncols
    key = _varlen_cache_key(
        rows, npad, top_k, ncols, next_n, compress_ratio, torch.cuda.current_device()
    )
    assert ss_host._VARLEN_CACHE_FP16[key][0] == "reg_clus"
    _check_bf16_varlen_exact(logits, indices, lengths, values)


@_LOW_DTYPE_PARAMS
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("extra", [1, 2, 3])
def test_selfsampling_topk_varlen_16bit_complement(
    top_k: int, extra: int, dtype: torch.dtype
) -> None:
    """Near-K complement selection (envelope K+1..K+3) remains exact across 16-bit ties and
    infinities, and the complement launcher is the one the host picks for both 16-bit dtypes."""
    rows, ncols = 4, top_k + extra
    logits, _, _ = _make_low_varlen_case(rows, ncols, top_k, 1, 1, "ties", dtype)
    logits[0, :ncols] = float("-inf")
    logits[0, 0] = float("inf")
    logits[1, :ncols] = 0.0
    lengths = [ncols, ncols, top_k - 1, 0]
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device=_DEV)
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(logits, kv_lens, indices, max_seq_len=ncols)
    torch.cuda.synchronize()
    assert _launcher_family(dtype, logits, top_k, ncols, 1, 1) == "complement"
    _check_low_varlen_exact(logits, indices, lengths)


@_LOW_DTYPE_PARAMS
@pytest.mark.parametrize("pattern", ("infinities", *_FP16_EDGE_PATTERNS))
@pytest.mark.parametrize(
    "rows,top_k,compress_ratio",
    [(1, 1024, 4), (64, 1024, 4), (1024, 1024, 4), (8, 512, 4), (8, 2048, 1)],
    ids=["pro_bs1", "pro_bs64", "pro_bs1024", "flash_bs8", "v32_bs8"],
)
def test_selfsampling_topk_varlen_16bit_complement_patterns(
    rows: int, top_k: int, compress_ratio: int, pattern: str, dtype: torch.dtype
) -> None:
    """The complement route at its grid envelopes (Pro N=1027 / Flash N=515 / V3.2 N=2051 = K+3)
    for BS 1..1024 on the infinities and fp16 edge patterns, with per-request windows that
    include the shortened, short-path and empty rows."""
    ncols = top_k + 3
    logits, kv_lens, lengths = _make_low_varlen_case(
        rows, ncols, top_k, 1, compress_ratio, pattern, dtype
    )
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        compress_ratio=compress_ratio,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    assert _launcher_family(dtype, logits, top_k, ncols, 1, compress_ratio) == "complement"
    _check_low_varlen_exact(logits, indices, lengths)


# stage B6: register plans of both 16-bit dtypes under the code-radix kernel (`kern_cr`) and, with the
# host switch forced off, under the bracket kernel; eager / DEG, regimg, binproof geometry, pk16,
# BLK-256 remap and staged-output (rows > SMs) plans with short, shortened and empty request windows
_B6_RADIX_PATTERNS = ("ties", "infinities", "subnormal", "zero_kth_neg", "fp16max")
_B6_RADIX_CASES = [
    pytest.param(4, 2048, 1024, 1, 4, id="reg_small"),
    pytest.param(8, 4099, 512, 4, 4, id="regimg_odd_width"),
    pytest.param(4, 8192, 2048, 4, 1, id="reg_mtp4_k2048"),
    pytest.param(1, 8195, 1024, 1, 4, id="reg_8k_bs1"),
    pytest.param(32, 16387, 1024, 1, 4, id="reg_16k_pk16"),
    pytest.param(256, 8207, 2048, 1, 1, id="reg_8k_staged"),
    pytest.param(1024, 4099, 1024, 1, 4, id="reg_blk256_remap"),
]


def _launcher_entry(
    dtype: torch.dtype,
    logits: torch.Tensor,
    top_k: int,
    n_env: int,
    next_n: int,
    compress_ratio: int,
) -> tuple:
    rows = logits.shape[0]
    npad = logits.stride(0) if rows > 1 else logits.shape[1]
    key = _varlen_cache_key(
        rows,
        npad,
        top_k,
        min(max(n_env, 1), npad),
        next_n,
        compress_ratio,
        torch.cuda.current_device(),
    )
    return ss_host._varlen_cache_for(dtype)[key]


@_LOW_DTYPE_PARAMS
@pytest.mark.parametrize("code_radix", (True, False), ids=("radix", "bracket"))
@pytest.mark.parametrize("pattern", _B6_RADIX_PATTERNS)
@pytest.mark.parametrize("rows,ncols,top_k,next_n,compress_ratio", _B6_RADIX_CASES)
def test_selfsampling_topk_varlen_16bit_reg_code_radix(
    rows: int,
    ncols: int,
    top_k: int,
    next_n: int,
    compress_ratio: int,
    pattern: str,
    code_radix: bool,
    dtype: torch.dtype,
    monkeypatch,
) -> None:
    """The code-radix register kernel (forced onto every register plan) and the bracket kernel
    (switch off) give the exact top-K value multiset, valid unique indices, the (-1) tail of short
    rows and the FP32 values output on every register plan geometry, for bf16 and fp16."""
    monkeypatch.setattr(ss_host, "_REG_CODE_RADIX_16BIT", code_radix)
    if code_radix:
        monkeypatch.setattr(ss_host, "_reg_code_radix", lambda k, n, rows, plan: True)
    monkeypatch.setattr(ss_host, "_VARLEN_CACHE_BF16", {})
    monkeypatch.setattr(ss_host, "_VARLEN_CACHE_FP16", {})
    logits, kv_lens, lengths = _make_low_varlen_case(
        rows, ncols, top_k, next_n, compress_ratio, pattern, dtype
    )
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    values = torch.full((rows, top_k + 13), 7.0, dtype=torch.float32, device=_DEV)
    ss_host.run_varlen(
        logits,
        kv_lens,
        indices,
        next_n=next_n,
        compress_ratio=compress_ratio,
        values=values,
        max_seq_len=ncols * compress_ratio,
    )
    torch.cuda.synchronize()
    family, compiled = _launcher_entry(dtype, logits, top_k, ncols, next_n, compress_ratio)[:2]
    assert family in ("reg", "regimg"), family
    # the launcher's compiled register kernel carries the requested mode (last field of its cache key)
    modes = {key[-1] for key, fn in ss_host._device()._COMPILE_CACHE__reg.items() if fn is compiled}
    assert modes == {code_radix}, modes
    _check_low_varlen_exact(logits, indices, lengths, values)


@_LOW_DTYPE_PARAMS
@pytest.mark.parametrize(
    "rows,ncols,top_k,next_n,compress_ratio",
    [
        pytest.param(8, 8195, 512, 4, 4, id="reg_mtp4"),
        pytest.param(4, 32768, 1024, 1, 4, id="regclus"),
        pytest.param(1, 65536, 2048, 1, 1, id="hybrid128"),
        pytest.param(16, 262144, 2048, 1, 1, id="main_split"),
        pytest.param(4, 1027, 1024, 1, 4, id="complement"),
    ],
)
def test_selfsampling_topk_varlen_16bit_cuda_graph(
    rows: int, ncols: int, top_k: int, next_n: int, compress_ratio: int, dtype: torch.dtype
) -> None:
    """16-bit warmup permits graph capture and replay with changed device KV lengths."""
    logits, kv_lens, lengths = _make_low_varlen_case(
        rows, ncols, top_k, next_n, compress_ratio, "ties", dtype
    )
    original = logits.clone()
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    ss_host.warmup_varlen(
        top_k,
        ncols * compress_ratio,
        next_n=next_n,
        compress_ratio=compress_ratio,
        num_rows_list=(rows,),
        row_stride=logits.stride(0) if rows > 1 else logits.shape[1],
        dtype=dtype,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ss_host.run_varlen(
            logits,
            kv_lens,
            indices,
            next_n=next_n,
            compress_ratio=compress_ratio,
            max_seq_len=ncols * compress_ratio,
        )
    for step in range(3):
        if step:
            options = [top_k - 3, ncols - 11, 0, top_k + 3]
            kv = [
                options[(request + step) % len(options)] * compress_ratio
                for request in range(rows // next_n)
            ]
            kv_lens.copy_(torch.tensor(kv, dtype=torch.int32, device=_DEV))
            lengths = _bf16_varlen_lengths(kv, next_n, compress_ratio, ncols)
            logits.copy_(original)
            for row, valid in enumerate(lengths):
                logits[row, valid:] = float("inf")
        indices.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        _check_low_varlen_exact(logits, indices, lengths)


@_LOW_DTYPE_PARAMS
def test_selfsampling_topk_varlen_16bit_layout_guards(dtype: torch.dtype) -> None:
    """Reject unsafe 16-bit vector-load layouts before compiling a launcher."""
    kv_lens = torch.full((2,), 8192, dtype=torch.int32, device=_DEV)
    indices = torch.empty((2, 512), dtype=torch.int32, device=_DEV)
    arena = torch.empty((2, 8256), dtype=dtype, device=_DEV)
    unaligned_storage = torch.empty(2 * 8256 + 1, dtype=dtype, device=_DEV)
    unaligned = unaligned_storage[1:].view(2, 8256)[:, :8192]
    for logits, message in (
        (arena[:, ::2], "inner stride"),
        (torch.empty((2, 8193), dtype=dtype, device=_DEV), "multiple of 8"),
        (unaligned, "aligned"),
    ):
        with pytest.raises(RuntimeError, match=message):
            ss_host.run_varlen(logits, kv_lens, indices, max_seq_len=8192)
    with pytest.raises(RuntimeError, match="float32"):
        ss_host.run_varlen(arena, kv_lens, indices, values=torch.empty_like(indices, dtype=dtype))


def test_selfsampling_topk_varlen_cache_dtype_device_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared launcher keeps dtypes, devices and architecture profiles isolated."""
    shape = (160, 8192, 512, 8192, 1, 1)
    profile_sm100 = ss_host._pack_device_profile(148, 100)
    profile_sm103 = ss_host._pack_device_profile(148, 103)
    launchers = {
        (*shape, 0, profile_sm100): ("device0_sm100",),
        (*shape, 1, profile_sm100): ("device1_sm100",),
        (*shape, 0, profile_sm103): ("device0_sm103",),
    }
    fp32 = ("fp32_sm100",)
    fp16 = {
        (*shape, 0, profile_sm100): ("device0_sm100_fp16",),
        (*shape, 1, profile_sm100): ("device1_sm100_fp16",),
    }
    monkeypatch.setattr(ss_host, "_VARLEN_CACHE", {(*shape, profile_sm100): fp32})
    monkeypatch.setattr(ss_host, "_VARLEN_CACHE_BF16", launchers)
    monkeypatch.setattr(ss_host, "_VARLEN_CACHE_FP16", fp16)
    assert ss_host._varlen_launcher(*shape, 148, 100) is fp32
    for key, launcher in fp16.items():
        assert (
            ss_host._varlen_launcher(*shape, 148, 100, dtype=torch.float16, device_index=key[-2])
            is launcher
        )
    for key, launcher in launchers.items():
        device_index, profile = key[-2:]
        num_sms, sm_version = ss_host._unpack_device_profile(profile)
        assert (
            ss_host._varlen_launcher(
                *shape,
                num_sms,
                sm_version,
                dtype=torch.bfloat16,
                device_index=device_index,
            )
            is launcher
        )
    assert ss_host._varlen_launcher(*shape, 148, 100, dtype=torch.float32) is fp32


@pytest.mark.parametrize(
    "rows,ncols,top_k,compress_ratio,dtypes",
    [
        pytest.param(4, 32768, 1024, 4, (torch.float32, torch.bfloat16), id="regclus_f32_bf16"),
        pytest.param(
            4, 32768, 1024, 4, (torch.float32, torch.bfloat16, torch.float16), id="regclus_three"
        ),
        pytest.param(
            256, 32768, 1024, 4, (torch.float32, torch.bfloat16, torch.float16), id="main_three"
        ),
    ],
)
def test_selfsampling_topk_varlen_mixed_dtype_warmup_capture(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    ncols: int,
    top_k: int,
    compress_ratio: int,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    """Repeated warmup of every dtype keeps each captured launcher usable; the graphs replay
    interleaved."""
    for name in ("_VARLEN_CACHE", "_VARLEN_CACHE_BF16", "_VARLEN_CACHE_FP16"):
        monkeypatch.setattr(ss_host, name, {})
    for name in ("_VARLEN_WARMUP_DONE", "_VARLEN_WARMUP_DONE_BF16", "_VARLEN_WARMUP_DONE_FP16"):
        monkeypatch.setattr(ss_host, name, set())

    generator = torch.Generator(device=_DEV).manual_seed(8173)
    original = torch.randn((rows, ncols), generator=generator, device=_DEV).round()
    logits = {dtype: original.to(dtype=dtype, copy=True) for dtype in dtypes}
    indices = {
        dtype: torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV) for dtype in dtypes
    }
    kv_lens = torch.full((rows,), ncols * compress_ratio, dtype=torch.int32, device=_DEV)
    for dtype in (*dtypes, *dtypes):
        ss_host.warmup_varlen(
            top_k,
            ncols * compress_ratio,
            compress_ratio=compress_ratio,
            num_rows_list=(rows,),
            row_stride=ncols,
            dtype=dtype,
        )

    graphs = {}
    for dtype in dtypes:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ss_host.run_varlen(
                logits[dtype],
                kv_lens,
                indices[dtype],
                compress_ratio=compress_ratio,
                max_seq_len=ncols * compress_ratio,
            )
        graphs[dtype] = graph

    for lengths in ([ncols, ncols - 11, top_k - 1, 0], [top_k + 3, 0, ncols // 2, top_k]):
        kv = [lengths[request % len(lengths)] for request in range(rows)]
        kv_lens.copy_(torch.tensor(kv, dtype=torch.int32, device=_DEV) * compress_ratio)
        for dtype in dtypes:
            logits[dtype].copy_(original)
            for row, valid in enumerate(kv):
                logits[dtype][row, valid:] = float("inf")
        for dtype in (*dtypes, torch.float32, *reversed(dtypes)):
            indices[dtype].fill_(-7)
            graphs[dtype].replay()
            torch.cuda.synchronize()
            _check_low_varlen_exact(logits[dtype], indices[dtype], kv)


@pytest.mark.parametrize("with_fp16", [False, True], ids=["bf16", "bf16_fp16"])
def test_selfsampling_topk_regclus_mixed_dtype(
    monkeypatch: pytest.MonkeyPatch, with_fp16: bool
) -> None:
    """Interleaved specializations preserve FP32 defaults and immutable layouts: regclus engines
    of every dtype (4-slot tpl; 512- and 1024-bin bf16, fp16) plus one main, clus, reg and
    complement engine per 16-bit dtype compile under the same module-constant guard (R8), and
    the compiled regclus engines run interleaved on their own dtype."""
    from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling as device

    constants = {
        name: value
        for name, value in vars(device).items()
        if isinstance(value, int) and (name.isupper() or name.endswith("__regclus"))
    }
    compile_kernel = device.cute.compile

    def guarded_compile(*args: object, **kwargs: object) -> object:
        assert {name: getattr(device, name) for name in constants} == constants
        compiled = compile_kernel(*args, **kwargs)
        assert {name: getattr(device, name) for name in constants} == constants
        return compiled

    for name in (
        "_COMPILE_CACHE",
        "_COMPILE_CACHE__reg",
        "_COMPILE_CACHE__clus",
        "_COMPILE_CACHE__regclus",
        "_COMPLEMENT_COMPILE_CACHE",
    ):
        monkeypatch.setattr(device, name, {})
    monkeypatch.setattr(device.cute, "compile", guarded_compile)
    tpl = (1024, 2, 4, 1024)
    options = {"varlen": True, "next_n": 1, "cr_shift": 0, "hint_free": True}
    fp32 = device.get_compiled__regclus(tpl, **options)
    bf16_512 = device.get_compiled__regclus(tpl, dtype="bf16", nbh=512, **options)
    bf16_1024 = device.get_compiled__regclus(tpl, dtype="bf16", nbh=1024, **options)
    assert fp32 is not bf16_512 and fp32 is not bf16_1024
    assert bf16_512 is not bf16_1024
    assert device.get_compiled__regclus(tpl, **options) is fp32
    # one engine of every other family per 16-bit dtype under the same guard (census tuples)
    for tag in ("bf16", "f16") if with_fp16 else ("bf16",):
        device.get_compiled(
            (1024, 1, 1, 256, 2, True, False, 1, 0, 40), hint_free=True, dtype=tag, dense=True
        )
        device.get_compiled__clus(
            (1024, 4, 1, 256, 4), varlen=True, next_n=1, cr_shift=2, hint_free=True, dtype=tag
        )
        device.get_compiled_complement(1024, 1027, 1, 2, dtype=tag)
        device.get_compiled__reg(
            (1024, 4, 1, 1, True, False, False, 2048),
            varlen=True,
            next_n=1,
            cr_shift=2,
            hint_free=True,
            static_k=1024,
            static_npad=16448,
            warp_rank=True,
            dtype=tag,
        )
    f16_512 = None
    if with_fp16:
        f16_512 = device.get_compiled__regclus(tpl, dtype="f16", nbh=512, **options)
        assert f16_512 is not bf16_512 and f16_512 is not fp32
        assert device.get_compiled__regclus(tpl, dtype="bf16", nbh=512, **options) is bf16_512

    rows, n_valid, top_k = 2, 32768, 1024
    generator = torch.Generator(device=_DEV).manual_seed(32768)
    logits_fp32 = torch.randn((rows, n_valid), generator=generator, device=_DEV)
    logits_bf16 = logits_fp32.to(torch.bfloat16)
    lengths = torch.full((rows,), n_valid, dtype=torch.int32, device=_DEV)
    indices = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    sequence = [
        (fp32, logits_fp32),
        (bf16_512, logits_bf16),
        (bf16_1024, logits_bf16),
        (fp32, logits_fp32),
    ]
    if f16_512 is not None:
        logits_f16 = logits_fp32.to(torch.float16)
        sequence += [(f16_512, logits_f16), (bf16_512, logits_bf16), (f16_512, logits_f16)]
    for compiled, logits in sequence:
        indices.fill_(-7)
        compiled(logits, indices, lengths, indices, n_valid)
        torch.cuda.synchronize()
        reference = torch.topk(logits, top_k, dim=1).values
        _check_exact(logits, indices, n_valid, reference)
