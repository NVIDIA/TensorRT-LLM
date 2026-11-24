# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_fp4_mqa_logits entry (FP4 prefill logits, DeepGEMM).

Each cell is a prefill chunk: per request a cached prefix and new query rows, the chunk's keys
back to back, each row's window its request's causal keys.
"""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp4_mqa_logits import (  # noqa: E402,E501
    rubin_fp4_mqa_logits as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_fp4_quantize import (  # noqa: E402,E501
    mxfp4_values,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_fp4_quantize import (  # noqa: E402,E501
    rubin_idx_fp4_quantize as quantize,
)

__extra_import_path__ = ["../../../..", "."]
import _rubin_fp4_inputs as fp4  # noqa: E402

if torch.cuda.is_available():  # the GPU cells' guard interposer: a CPU run needs neither
    from _validating import certifying, validating  # noqa: E402,F401 -- path above; fixture by name

cpu = pytest.mark.cpu_only
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="drives the kernel on sm_107")


def _windows(requests, q_rows=None):
    """Row windows of a prefill chunk: ``requests`` are (cached keys, new rows); the chunk's key
    matrix holds each request's cached and new keys back to back. Returns (ks, ke, num_keys)
    (int32 [rows]) for the rows ``q_rows`` = (begin, end) of the chunk, or all rows."""
    ks, ke, start = [], [], 0
    for cached, new in requests:
        for i in range(new):
            ks.append(start)
            ke.append(start + cached + i + 1)
        start += cached + new
    ks = torch.tensor(ks, dtype=torch.int32)
    ke = torch.tensor(ke, dtype=torch.int32)
    if q_rows is not None:
        ks, ke = ks[q_rows[0] : q_rows[1]], ke[q_rows[0] : q_rows[1]]
    return ks, ke, start


def _schedule(args: dict) -> torch.Tensor:
    """DeepGEMM's schedule of the call ``args`` describe."""
    from tensorrt_llm.deep_gemm import get_mqa_logits_metadata

    return get_mqa_logits_metadata(
        args["cu_seqlen_ks"], args["cu_seqlen_ke"], args["k"].shape[0], args["q"].shape[1]
    )


def _build(spec, seed: int, device="cuda") -> dict:
    gen = torch.Generator().manual_seed(seed)
    ks, ke, num_keys = _windows(spec["requests"], spec["q_rows"])
    rows = ks.shape[0]
    k, sf_k = fp4.fp4_rows(num_keys, spec["data"], gen, fp4.KEY_EXPONENTS)
    q, sf_q = fp4.fp4_rows(rows * fp4.HEADS, spec["data"], gen, fp4.QUERY_EXPONENTS)
    weights = fp4.weights(rows, spec["data"], gen)
    args = dict(
        q=q.view(rows, fp4.HEADS, fp4.DATA).to(device),
        sf_q=sf_q.view(rows, fp4.HEADS).to(device),
        k=k.to(device),
        sf_k=sf_k.to(device),
        weights=weights.to(device),
        cu_seqlen_ks=ks.to(device),
        cu_seqlen_ke=ke.to(device),
        clean_logits=spec["clean_logits"],
    )
    if spec["scheduled"] and device == "cuda":
        args["schedule_meta"] = _schedule(args)
    return args


# ---- CPU ------------------------------------------------------------------------------------------


def _small(requests, q_rows=None, seed=0) -> dict:
    spec = dict(requests=requests, q_rows=q_rows, clean_logits=False, data="int", scheduled=False)
    return _build(spec, seed, device="cpu")


@cpu
def test_reference_is_the_windowed_weighted_relu_sum():
    # Packed requests: a cached prefix and a one-row prompt.
    args = _small([(0, 30), (16, 20), (0, 1)], seed=3)
    ref = op.reference(**args)
    rows, num_keys = args["q"].shape[0], args["k"].shape[0]
    assert ref.dtype == torch.float32 and ref.shape == (rows, num_keys)
    queries = mxfp4_values(args["q"], args["sf_q"]).double()
    keys = mxfp4_values(args["k"], args["sf_k"]).double()
    scores = torch.einsum("rhd,jd->rhj", queries, keys).relu()
    want = torch.einsum("rh,rhj->rj", args["weights"].double(), scores).float()
    cols = torch.arange(num_keys)[None, :]
    window = (cols >= args["cu_seqlen_ks"][:, None]) & (cols < args["cu_seqlen_ke"][:, None])
    want = want.masked_fill(~window, float("-inf"))
    assert torch.equal(ref, want)


@cpu
def test_guard_admits_a_well_formed_call():
    args = _small([(0, 30), (16, 20), (0, 1)])
    op.is_valid(**args)
    # The paged gather types the 64 key bytes as float8 and the words as fp32: the key bytes pass
    # as they are, the words once reinterpreted.
    gathered = args["k"].view(torch.float8_e4m3fn)
    op.is_valid(**dict(args, k=gathered, sf_k=args["sf_k"].view(-1, 1)))


@cpu
@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda a: dict(a, q=torch.zeros(a["q"].shape[0], fp4.HEADS, 128, dtype=torch.uint8)),
            "q must",
        ),
        (lambda a: dict(a, k=torch.zeros(a["k"].shape[0], 128, dtype=torch.uint8)), "k must"),
        (lambda a: dict(a, sf_q=a["sf_q"].view(torch.float32)), "sf_q"),
        (lambda a: dict(a, sf_q=a["sf_q"][:, :16].contiguous()), "sf_q"),
        (lambda a: dict(a, sf_k=a["sf_k"].view(torch.float32)), "sf_k"),
        (lambda a: dict(a, sf_k=a["sf_k"][:-1]), "sf_k"),
        (lambda a: dict(a, weights=_padded_rows(a["weights"])), "packed"),
        (lambda a: dict(a, weights=a["weights"].bfloat16()), "fp32"),
        (lambda a: dict(a, cu_seqlen_ke=a["cu_seqlen_ke"] + a["k"].shape[0]), "past the key"),
        (lambda a: dict(a, cu_seqlen_ks=a["cu_seqlen_ks"] - 1), "before the key"),
    ],
    ids=[
        "fp8-wide q",
        "fp8-wide k",
        "fp32 sf_q",
        "sf_q heads",
        "fp32 sf_k",
        "sf_k count",
        "weights rows",
        "weights dtype",
        "window past keys",
        "window before keys",
    ],
)
def test_guard_refuses(change, match):
    args = _small([(0, 30), (16, 20), (0, 1)])
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**change(args))


def _padded_rows(w: torch.Tensor) -> torch.Tensor:
    """The same weights in rows padded to twice their width."""
    wide = torch.zeros(w.shape[0], 2 * w.shape[1], dtype=w.dtype)
    wide[:, : w.shape[1]] = w
    return wide[:, : w.shape[1]]


# ---- sm_107 ---------------------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args)
    again = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    assert out.shape == ref.shape and out.dtype == torch.float32
    window = torch.isfinite(ref)
    assert torch.equal(out[window], again[window]), "two identical calls disagree"
    if cell.spec["clean_logits"]:
        assert torch.isneginf(out[~window]).all(), "clean_logits left a finite value outside"
    if cell.spec["data"] == "int":
        assert torch.equal(out[window], ref[window])
    else:
        err = (out.double() - ref.double()).masked_fill(~window, 0).abs()
        band = fp4.dense_band(args, window)
        assert (err <= band).all(), f"max excess {float((err - band).max()):.3e}"


# The scheduled cells run with their schedule in test_certified_cells.
_UNSCHEDULED = [c for c in op.CELLS if not c.spec["scheduled"]]


@gpu
@pytest.mark.parametrize("cell", _UNSCHEDULED, ids=[c.why[:48] for c in _UNSCHEDULED])
def test_a_schedule_places_work_and_changes_no_logit(cell):
    """The call with DeepGEMM's schedule against the same call without one: equal bit for bit in
    every row's window, and -inf outside when the call cleans."""
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    args.pop("schedule_meta", None)
    plain = op(**args)
    scheduled = op(**args, schedule_meta=_schedule(args))
    torch.cuda.synchronize()
    cols = torch.arange(plain.shape[1], device="cuda")[None, :]
    ks, ke = args["cu_seqlen_ks"], args["cu_seqlen_ke"]
    window = (cols >= ks.long()[:, None]) & (cols < ke.long()[:, None])
    assert torch.equal(plain[window], scheduled[window]), "the schedule changed a logit"
    if cell.spec["clean_logits"]:
        assert torch.isneginf(scheduled[~window]).all(), "clean_logits left a finite value outside"


@gpu
def test_gathered_keys_score_as_the_reference():
    """The default target's prefill path on one chunk: the keys quantized, written with the FP4
    scatter, gathered back by the family gather at the 64-byte payload, then scored."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.gather_k_cache_paged import (  # noqa: E501
        k31_gather_k_cache_paged as gather,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_k_cache_scatter_fp4 import (  # noqa: E501
        rubin_idx_k_cache_scatter_fp4 as scatter,
    )

    gen = torch.Generator().manual_seed(13)
    page, num_keys = 64, 3000
    ks, ke, _ = _windows([(1000, 2000)])
    num_pages = -(-num_keys // page) + 3
    cache = torch.zeros(num_pages, page, 1, fp4.TOKEN, dtype=torch.uint8, device="cuda")
    slots = torch.randperm(num_pages * page, generator=gen)[:num_keys]
    block = slots // page * page * fp4.TOKEN
    slot_k = (block + slots % page * fp4.DATA).cuda()
    slot_s = (block + page * fp4.DATA + slots % page * 4).cuda()
    keys = torch.randn(num_keys, fp4.DIM, generator=gen).bfloat16().cuda()
    k4, sk = quantize(keys[:, :64], keys[:, 64:])
    scatter(k4, sk, cache, slot_k, slot_s, num_keys)
    gathered, words = gather(cache, slot_k, slot_s, 0, num_keys, fp4.DATA)
    queries = torch.randn(ks.shape[0], fp4.HEADS, fp4.DIM, generator=gen).bfloat16().cuda()
    q4, sq = quantize(queries[..., :64], queries[..., 64:])
    args = dict(
        q=q4.view(-1, fp4.HEADS, fp4.DATA),
        sf_q=sq.view(-1, fp4.HEADS),
        k=gathered,
        sf_k=words.view(torch.int32),
        weights=torch.randn(ks.shape[0], fp4.HEADS, generator=gen).cuda(),
        cu_seqlen_ks=ks.cuda(),
        cu_seqlen_ke=ke.cuda(),
        clean_logits=False,
    )
    with validating(op):
        out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    assert torch.equal(gathered.view(torch.uint8), k4.view(torch.uint8))
    assert torch.equal(words.view(torch.int32).view(-1), sk.view(-1))
    window = torch.isfinite(ref)
    err = (out.double() - ref.double()).masked_fill(~window, 0).abs()
    band = fp4.dense_band(args, window)
    assert (err <= band).all(), f"max excess {float((err - band).max()):.3e}"


@gpu
def test_guard_refuses_a_schedule_built_for_another_call():
    """A schedule of the first half of the rows is not this call's."""
    args = _build(op.CELLS[3].spec, seed=11)
    half = args["cu_seqlen_ks"].shape[0] // 2
    other = _schedule(
        dict(
            args,
            cu_seqlen_ks=args["cu_seqlen_ks"][:half].contiguous(),
            cu_seqlen_ke=args["cu_seqlen_ke"][:half].contiguous(),
        )
    )
    with pytest.raises(AssertionError, match="another call"):
        with validating(op):
            op(**args, schedule_meta=other)
