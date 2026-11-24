# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_fp4_paged_mqa_logits entry (FP4 decode / verify logits, CuTe DSL).

The keys live on random physical pages of a [pages, P, 1, 68] FP4 cache view.
"""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp4_paged_mqa_logits import (  # noqa: E402,E501
    SPLIT_KV,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_fp4_paged_mqa_logits import (  # noqa: E402,E501
    rubin_fp4_paged_mqa_logits as op,
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

_SENTINEL = 12345.0


def _out(rows: int, max_context_len: int, pitch: int | None = None, device="cpu") -> torch.Tensor:
    """A sentinel-filled caller buffer with one spare row and 256 columns of slack past the
    aligned width (or the given pitch), and its [rows, max_context_len] view."""
    aligned = -(-max_context_len // SPLIT_KV) * SPLIT_KV
    pitch = aligned + SPLIT_KV if pitch is None else pitch
    base = torch.full((rows + 1, pitch), _SENTINEL, dtype=torch.float32, device=device)
    return base[:rows, :max_context_len]


def _schedule(lens: torch.Tensor) -> torch.Tensor:
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(lens.device).multi_processor_count
    return get_paged_mqa_logits_metadata(lens.reshape(-1, 1), fp4.SCHEDULE_BLOCK_KV, num_sms)


def _lens_2d(args) -> torch.Tensor:
    """Each query row's request length (the entry takes one length per request)."""
    return args["context_lens"][:, None].expand(-1, args["q"].shape[1])


def _build(spec, seed: int, device="cuda") -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n = spec["batch"], spec["next_n"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    if spec.get("zero_every"):
        lens[:: spec["zero_every"]] = 0
    kv_cache, table, _ = fp4.paged_cache(
        lens.tolist(), spec["tokens_per_block"], spec["data"], gen, spec["page_stride_pad"], device
    )
    rows = batch * next_n
    q, sf_q = fp4.fp4_rows(rows * fp4.HEADS, spec["data"], gen, fp4.QUERY_EXPONENTS)
    lens = lens.to(device)
    return dict(
        q=q.view(batch, next_n, fp4.HEADS, fp4.DATA).to(device),
        sf_q=sf_q.view(batch, next_n, fp4.HEADS).to(device),
        kv_cache=kv_cache,
        weights=fp4.weights(rows, spec["data"], gen).to(device),
        context_lens=lens,
        block_table=table,
        # A CPU schedule: the guard checks its SM count on the GPU only.
        schedule_meta=_schedule(lens)
        if device == "cuda"
        else torch.zeros((2, 2), dtype=torch.int32),
        max_context_len=spec["max_context_len"],
        num_next_n_atoms=spec["num_next_n_atoms"],
        atom=spec["atom"],
        kv_evict_first=spec["kv_evict_first"],
        out=_out(rows, spec["max_context_len"], device=device),
    )


def _untouched(out: torch.Tensor, max_context_len: int) -> None:
    """The columns past the aligned width and the spare row past the view keep the sentinel."""
    rows, pitch = out.shape[0], out.stride(0)
    aligned = -(-max_context_len // SPLIT_KV) * SPLIT_KV
    if pitch > aligned:
        slack = out.as_strided((rows, pitch - aligned), (pitch, 1), out.storage_offset() + aligned)
        assert bool((slack == _SENTINEL).all()), "the kernel wrote past the aligned row width"
    spare = out.as_strided((1, pitch), (pitch, 1), out.storage_offset() + rows * pitch)
    assert bool((spare == _SENTINEL).all()), "the kernel wrote past the last row"


# ---- CPU ------------------------------------------------------------------------------------------


def _small(lens, next_n, max_context_len, page=16, pad=0, seed=0) -> dict:
    """A CPU call on exact data over a small cache (P = 16)."""
    gen = torch.Generator().manual_seed(seed)
    kv_cache, table, _ = fp4.paged_cache(lens, page, "int", gen, pad)
    rows = len(lens) * next_n
    q, sf_q = fp4.fp4_rows(rows * fp4.HEADS, "int", gen, fp4.QUERY_EXPONENTS)
    return dict(
        q=q.view(len(lens), next_n, fp4.HEADS, fp4.DATA),
        sf_q=sf_q.view(len(lens), next_n, fp4.HEADS),
        kv_cache=kv_cache,
        weights=fp4.weights(rows, "int", gen),
        context_lens=torch.tensor(lens, dtype=torch.int32),
        block_table=table,
        schedule_meta=torch.zeros((2, 2), dtype=torch.int32),
        max_context_len=max_context_len,
        num_next_n_atoms=1,
        atom=None,
        kv_evict_first=True,
        out=_out(rows, max_context_len),
    )


@cpu
@pytest.mark.parametrize(
    "lens,next_n,pad",
    [
        ([1, 40], 8, 0),  # a context shorter than the window: rows with no key
        ([90, 17, 64], 6, 48),  # several pages on a padded page stride (the V2 per-layer view)
        ([0, 40], 2, 0),  # a request without keys on this rank (helix8): its rows stay -inf
    ],
)
def test_reference_is_the_causal_weighted_relu_sum(lens, next_n, pad):
    args = _small(lens, next_n, max_context_len=160, pad=pad, seed=len(lens) * 7 + next_n)
    before = args["out"].clone()
    ref = op.reference(**args)
    assert ref.dtype == torch.float32 and ref.shape == (len(lens) * next_n, 160)
    assert torch.equal(ref, fp4.paged_logits(args, _lens_2d(args)))
    assert torch.equal(args["out"], before), "the reference must leave out untouched"


@cpu
def test_guard_admits_a_well_formed_call():
    args = _small([70, 130], 2, max_context_len=160)
    op.is_valid(**args)
    # The decode arena's pitch: exactly the aligned width.
    op.is_valid(**dict(args, out=_out(4, 160, pitch=256)))
    op.is_valid(**dict(args, q=args["q"].view(torch.int8)))


def _unpacked(kv):
    """The same page view with each token's 68 bytes spread over 136."""
    wide = torch.zeros(kv.shape[0], kv.shape[1], 1, 2 * kv.shape[3], dtype=torch.uint8)
    return wide[..., : kv.shape[3]]


@cpu
@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: dict(a, sf_q=a["sf_q"].view(torch.float32)), "sf_q"),
        (lambda a: dict(a, sf_q=a["sf_q"][:, :, :16].contiguous()), "sf_q"),
        (lambda a: dict(a, kv_cache=torch.zeros(9, 16, 1, 132, dtype=torch.uint8)), "68"),
        (lambda a: dict(a, kv_cache=_unpacked(a["kv_cache"])), "packed"),
        (lambda a: dict(a, max_context_len=100, out=_out(4, 100)), "max_context_len"),
        (lambda a: dict(a, out=a["out"][:-1]), "out must be"),
        (lambda a: dict(a, out=a["out"][:, :-1]), "out must be"),
        (lambda a: dict(a, out=_out(4, 160).double()), "fp32"),
        (lambda a: dict(a, out=_out(4, 160, pitch=160)), "room"),
        (lambda a: dict(a, weights=a["weights"][:, :16].contiguous()), "one row per query row"),
        (lambda a: dict(a, q=a["q"].transpose(2, 3).contiguous().transpose(2, 3)), "packed"),
        (lambda a: dict(a, block_table=a["block_table"][:, :-1]), "block_table"),
    ],
    ids=[
        "fp32 sf_q",
        "sf_q heads",
        "fp8 cache",
        "kv rows",
        "context past max",
        "short out",
        "narrow out",
        "out dtype",
        "pitch below aligned",
        "weights",
        "q",
        "short table",
    ],
)
def test_guard_refuses(change, match):
    args = _small([70, 130], 2, max_context_len=160)
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**change(args))


@cpu
def test_guard_refuses_a_buffer_that_ends_inside_the_last_row_padding():
    args = _small([70, 130], 2, max_context_len=160)
    flat = torch.full((3 * 256 + 160,), _SENTINEL)  # row 3 ends at column 160 of its 256
    out = flat.as_strided((4, 160), (256, 1))
    with pytest.raises(AssertionError, match="room"):
        op.is_valid(**dict(args, out=out))


# ---- sm_107 ---------------------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args)
    assert out.data_ptr() == args["out"].data_ptr(), "the entry must write the caller's buffer"
    first = out.clone()
    again = op(**dict(args, out=_out(*out.shape, device="cuda")))
    flipped = op(
        **dict(args, kv_evict_first=not args["kv_evict_first"], out=_out(*out.shape, device="cuda"))
    )
    ref = op.reference(**args)
    torch.cuda.synchronize()
    batch, next_n = args["q"].shape[:2]
    assert out.shape == (batch * next_n, args["max_context_len"]) and out.dtype == torch.float32
    causal = torch.isfinite(ref)
    assert torch.equal(first[causal], again[causal]), "two identical calls disagree"
    assert torch.equal(first[causal], flipped[causal]), "the L2 policy changed the logits"
    _untouched(out, args["max_context_len"])
    if cell.spec["data"] == "int":
        assert torch.equal(first[causal], ref[causal])
    else:
        fp4.assert_paged_band(first, ref, args, _lens_2d(args))


@gpu
def test_the_aligned_pitch_of_the_decode_arena_gives_the_same_logits():
    spec = dict(op.CELLS[5].spec, max_context_len=10000)
    args = _build(spec, seed=11)
    rows = args["q"].shape[0] * args["q"].shape[1]
    with validating(op):
        tight = op(**dict(args, out=_out(rows, 10000, pitch=10240, device="cuda"))).clone()
        padded = op(**dict(args, out=_out(rows, 10000, device="cuda")))
    torch.cuda.synchronize()
    causal = torch.isfinite(op.reference(**args))
    assert torch.equal(tight[causal], padded[causal])


@gpu
def test_graph_replay_scores_a_new_step():
    spec = next(c.spec for c in op.CELLS if c.spec["batch"] == 12 and c.spec["data"] == "int")
    first = _build(spec, seed=41)
    gen = torch.Generator().manual_seed(42)
    rows = spec["batch"] * spec["next_n"]
    q, sf_q = fp4.fp4_rows(rows * fp4.HEADS, "int", gen, fp4.QUERY_EXPONENTS)
    shorter = first["context_lens"] - torch.randint(0, 300, (spec["batch"],), generator=gen).cuda()
    second = dict(
        first,
        q=q.view_as(first["q"]).cuda(),
        sf_q=sf_q.view_as(first["sf_q"]).cuda(),
        weights=fp4.weights(rows, "int", gen).cuda(),
        context_lens=shorter.clamp_min(1).to(torch.int32),
    )
    second["schedule_meta"] = _schedule(second["context_lens"])
    static = dict(first, **{k: first[k].clone() for k in ("q", "sf_q", "weights", "context_lens")})
    static["schedule_meta"] = first["schedule_meta"].clone()
    op(**static)  # compile and warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**static)
    for name in ("q", "sf_q", "weights", "context_lens", "schedule_meta"):
        static[name].copy_(second[name])
    graph.replay()
    eager = op(**dict(second, out=_out(rows, spec["max_context_len"], device="cuda")))
    torch.cuda.synchronize()
    causal = torch.isfinite(op.reference(**second))
    assert torch.equal(static["out"][causal], eager[causal])


@gpu
def test_quantized_and_scattered_keys_score_as_the_reference():
    """The default target's chain on one layer: quantize the keys and queries, scatter the keys
    into the paged cache, score the verify window."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_k_cache_scatter_fp4 import (  # noqa: E501
        rubin_idx_k_cache_scatter_fp4 as scatter,
    )

    gen = torch.Generator().manual_seed(7)
    batch, next_n, page, max_len = 4, 8, 64, 8192
    lens = torch.randint(3000, 8000, (batch,), generator=gen, dtype=torch.int32)
    pages_per_req = [-(-int(c) // page) for c in lens]
    num_pages = sum(pages_per_req) + 4
    cache = torch.zeros(num_pages, page, 1, fp4.TOKEN, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(num_pages, generator=gen)
    table = torch.zeros(batch, max(pages_per_req), dtype=torch.int32)
    slots_k, slots_s, keys = [], [], []
    used = 0
    for b, npages in enumerate(pages_per_req):
        ids = perm[used : used + npages]
        used += npages
        table[b, :npages] = ids.to(torch.int32)
        pos = torch.arange(int(lens[b]))
        block = ids[pos // page] * page * fp4.TOKEN
        slots_k.append(block + (pos % page) * fp4.DATA)
        slots_s.append(block + page * fp4.DATA + (pos % page) * 4)
        keys.append(torch.randn(int(lens[b]), fp4.DIM, generator=gen))
    keys = torch.cat(keys).bfloat16().cuda()
    k4, ks = quantize(keys[:, :64], keys[:, 64:])
    slot_k, slot_s = torch.cat(slots_k).cuda(), torch.cat(slots_s).cuda()
    with validating(scatter):
        scatter(k4, ks, cache, slot_k, slot_s, keys.shape[0])
    queries = torch.randn(batch * next_n, fp4.HEADS, fp4.DIM, generator=gen).bfloat16().cuda()
    q4, qs = quantize(queries[..., :64], queries[..., 64:])
    lens = lens.cuda()
    args = dict(
        q=q4.view(batch, next_n, fp4.HEADS, fp4.DATA),
        sf_q=qs.view(batch, next_n, fp4.HEADS),
        kv_cache=cache,
        weights=torch.randn(batch * next_n, fp4.HEADS, generator=gen).cuda(),
        context_lens=lens,
        block_table=table.cuda(),
        schedule_meta=_schedule(lens),
        max_context_len=max_len,
        num_next_n_atoms=1,
        atom=None,
        kv_evict_first=True,
        out=_out(batch * next_n, max_len, device="cuda"),
    )
    with validating(op):
        out = op(**args).clone()
    ref = op.reference(**args)
    torch.cuda.synchronize()
    # The cache holds exactly the quantized keys, request by request.
    first = 0
    for b in range(batch):
        n = int(lens[b])
        ids = table[b, : -(-n // page)].long().cuda()
        raw = cache[ids].reshape(len(ids), -1)
        assert torch.equal(
            raw[:, : page * fp4.DATA].reshape(-1, fp4.DATA)[:n],
            k4[first : first + n].view(torch.uint8),
        )
        words = raw[:, page * fp4.DATA :].contiguous().view(torch.int32).reshape(-1)[:n]
        assert torch.equal(words, ks[first : first + n].view(-1))
        first += n
    fp4.assert_paged_band(out, ref, args, _lens_2d(args))
