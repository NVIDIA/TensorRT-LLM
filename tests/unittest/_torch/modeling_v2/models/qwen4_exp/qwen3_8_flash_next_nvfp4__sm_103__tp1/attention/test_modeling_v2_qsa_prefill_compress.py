# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_prefill_compress catalog entry.

The paged state is real: an index-key pool of random raw keys, a position
plane holding `(t, h, w)` for every allocated position, and the engine's
`[slots, width]` table -- more slots than requests, each live row's pages drawn
from a seeded permutation of the pool, `-1` past its allocation, rows past the
batch holding a previous step's plausible page ids. The cos|sin table is built
the way the target builds it at load.

A padding row at logical position -1 is deliberately absent: it is an illegal
memory access that poisons the process for every test after it.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_prefill_compress import (  # noqa: E501
    qsa_prefill_compress as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_prefill_compress requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise over the whole cache.

    Measured on sm_103: the kernel matches this fp32 reference on every
    element of every cell, so there is no tolerance to derive. It has to
    be bitwise -- a pool accumulated in bf16 instead of fp32 lands only
    0.03125 away, inside any bf16 band. Comparing every slot, not just the
    anchors, is the untouched-slot lock.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)


DEV = "cuda"
ROTARY_DIM = 64
THETA = 1.0e7
MAX_POSITIONS = 16384


def _cos_sin() -> torch.Tensor:
    """The target's `[max_positions, 2 * pairs]` float32 cos|sin table."""
    exponent = torch.arange(0, ROTARY_DIM, 2, dtype=torch.float64, device=DEV)
    inv_freq = (1.0 / (THETA ** (exponent / ROTARY_DIM))).to(torch.float32)
    positions = torch.arange(MAX_POSITIONS, dtype=torch.float32, device=DEV)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()


def _build(spec, seed: int):
    """Real caches, table and rows for one cell; returns the op's kwargs and extras."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, head_dim = spec["tokens_per_block"], spec["head_dim"]
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    index_cache = torch.randn(pages, tpb, 1, head_dim, generator=gen).to(torch.bfloat16)
    position_cache = torch.zeros(pages, tpb, 3, dtype=torch.int32)
    used = 0
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
        p = torch.arange(cols * tpb).view(cols, tpb)
        if spec["distinct_axes"]:
            coords = (p, (3 * p + 5 * r) % 500, (5 * p + 11 * r) % 500)
        else:
            coords = (p, p, p)
        position_cache[table[r, :cols].long()] = torch.stack(coords, dim=-1).to(torch.int32)
    assert used <= pages and len(segments) < slots, "pool or table too small for the batch"
    k_norm_weight = (torch.randn(head_dim, generator=gen) * 0.1).to(torch.bfloat16)

    logical = torch.tensor(positions, dtype=spec["logical_dtype"])
    request = torch.tensor(requests, dtype=spec["request_dtype"])
    if spec["duplicate"]:
        logical, request = torch.cat([logical, logical]), torch.cat([request, request])
    table, index_cache, position_cache = table.to(DEV), index_cache.to(DEV), position_cache.to(DEV)

    extras = {}
    if spec["layout"] == "strided":
        wide_positions = torch.randint(0, 500, (pages, tpb, 6), dtype=torch.int32, device=DEV)
        wide_positions[..., ::2] = position_cache
        position_cache = wide_positions[..., ::2]
        wide_table = torch.randint(0, pages, (slots, 2 * width), dtype=torch.int32, device=DEV)
        wide_table[:, ::2] = table
        table = wide_table[:, ::2]
        pool = torch.randn(2 * pages, tpb, 1, head_dim, device=DEV).to(torch.bfloat16)
        pool[pages:] = index_cache
        index_cache = pool[pages:]
        extras["neighbour"] = (pool[:pages], pool[:pages].clone())
    elif spec["layout"] == "head_dim_strided":
        wide = torch.randn(pages, tpb, 1, 2 * head_dim, device=DEV).to(torch.bfloat16)
        wide[..., ::2] = index_cache
        index_cache = wide[..., ::2]
        extras["neighbour"] = (wide[..., 1::2], wide[..., 1::2].clone())

    args = dict(
        logical_positions=logical.to(DEV),
        request_indices=request.to(DEV),
        block_table=table,
        index_cache=index_cache,
        position_cache=position_cache,
        k_norm_weight=k_norm_weight.to(DEV),
        cos_sin=_cos_sin(),
        eps=spec["eps"],
        tokens_per_block=tpb,
        compress_ratio=spec["compress_ratio"],
        mrope_section=spec["mrope_section"],
    )
    return args, extras


def _anchors(args) -> int:
    """Distinct anchor slots the rows name."""
    logical = args["logical_positions"].long()
    keyed = args["request_indices"].long() * (1 << 32) + logical
    return int(torch.unique(keyed[(logical + 1) % args["compress_ratio"] == 0]).numel())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = args["index_cache"].clone()
    ref = op.reference(**args)
    out = op(**args)
    assert out is args["index_cache"]
    _compare(out, ref)
    changed = (out.view(torch.int16) != before.view(torch.int16)).any(-1).sum()
    assert int(changed) == _anchors(args), "exactly the anchor slots are written"
    if "neighbour" in extras:
        view, saved = extras["neighbour"]
        assert torch.equal(view, saved), "wrote outside the strided view"


def _scene(seed: int, **overrides):
    """Two requests of 64 tokens from 0 under mRoPE, at the checkpoint geometry."""
    spec = {**op.CELLS[0].spec, "segments": ((0, 64),) * 2, "pages": 4, **overrides}
    return _build(spec, seed=seed)[0]


def test_text_rows_reduce_mrope_to_plain_rope() -> None:
    """With t == h == w, the mRoPE selection is an identity, bit for bit."""
    args = _scene(seed=7, distinct_axes=False)
    plain = dict(args, index_cache=args["index_cache"].clone(), mrope_section=None)
    op(**args)
    op(**plain)
    assert torch.equal(args["index_cache"], plain["index_cache"])


def test_a_second_call_over_the_same_rows_is_not_idempotent() -> None:
    """Recorded, not tolerated: the anchor slot is one of its own inputs."""
    args = _scene(seed=14)
    once = op(**args).clone()
    twice = op(**args)
    changed = (twice.view(torch.int16) != once.view(torch.int16)).any(-1).sum()
    assert int(changed) == _anchors(args)


def test_cuda_graph_capture_and_replay() -> None:
    """The launch grid is fixed by the row count, so it captures and replays."""
    args = _scene(seed=16)
    cache = args["index_cache"]
    before = cache.clone()
    ref = op.reference(**args)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            cache.copy_(before)
            op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    cache.copy_(before)
    with torch.cuda.graph(graph):
        op(**args)
    for _ in range(2):
        cache.copy_(before)
        graph.replay()
        torch.cuda.synchronize()
        _compare(cache, ref)


def test_stream_ordering_needs_no_explicit_synchronization() -> None:
    """The launch is on the current stream; a read enqueued after it sees it."""
    args = _scene(seed=17)
    before = args["index_cache"].clone()
    ref = op.reference(**args)
    for _ in range(20):
        args["index_cache"].copy_(before)
        _compare(op(**args), ref)


def test_callable_rejects_its_documented_domain_errors() -> None:
    """The callable's own loud errors: why `is_valid` carries only stride guards."""
    args = _scene(seed=13)
    cache, logical, request = (
        args["index_cache"],
        args["logical_positions"],
        args["request_indices"],
    )
    before = cache.clone()
    cases = [
        ("request indices must match logical positions", dict(request_indices=request[:-1])),
        (
            "request metadata must be one-dimensional",
            dict(logical_positions=logical.view(2, -1), request_indices=request.view(2, -1)),
        ),
        ("request metadata must use integer storage", dict(logical_positions=logical.float())),
        (
            "requires a two-dimensional int32 block table",
            dict(block_table=args["block_table"].long()),
        ),
        ("must have one KV head", dict(index_cache=cache.expand(-1, -1, 2, -1).contiguous())),
        (
            "position cache must contain three axes",
            dict(position_cache=args["position_cache"][..., :2].contiguous()),
        ),
        (
            "position cache must use int32 storage",
            dict(position_cache=args["position_cache"].long()),
        ),
        ("index cache must use BF16", dict(index_cache=cache.half())),
        ("pages do not match tokens_per_block", dict(tokens_per_block=16)),
        ("packed 2D cos/sin tensor", dict(cos_sin=args["cos_sin"][:, :-1].contiguous())),
        ("half-width rotary embedding", dict(cos_sin=args["cos_sin"][:, :32].contiguous())),
        ("power-of-two head and compression widths", dict(compress_ratio=3)),
        (
            "norm weight does not match the index head dimension",
            dict(k_norm_weight=args["k_norm_weight"][:-1].contiguous()),
        ),
        ("requires CUDA tensors", dict(block_table=args["block_table"].cpu())),
        ("MRoPE sections do not cover the rotary width", dict(mrope_section=(1, 1, 1))),
        ("MRoPE sections do not cover the rotary width", dict(mrope_section=(16, 16))),
    ]
    for message, override in cases:
        with pytest.raises(ValueError, match=message):
            op(**{**args, **override})
    assert torch.equal(cache, before), "a rejected call still wrote"


def _strided(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """The same values as a stride-2 view along `dim` of a zero-filled buffer."""
    shape = list(tensor.shape)
    shape[dim] *= 2
    wide = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    index = [slice(None)] * tensor.dim()
    index[dim] = slice(None, None, 2)
    wide[tuple(index)] = tensor
    return wide[tuple(index)]


@pytest.mark.parametrize(
    "name, dim, match",
    [
        ("logical_positions", 0, "logical_positions must be contiguous"),
        ("request_indices", 0, "request_indices must be contiguous"),
        ("k_norm_weight", 0, "k_norm_weight must be contiguous"),
        ("cos_sin", 1, "unit stride"),
    ],
)
def test_guard_refuses_the_strides_the_callable_takes_silently(name, dim, match) -> None:
    """Shown unguarded first -- accepted and wrong -- then refused.

    Old receipt's max abs error against the right answer: 4.72, 6.20, 0.80
    and 7.16 respectively; the floor below states a bound, not one draw.
    """
    args = _scene(seed=12)
    before = args["index_cache"].clone()
    ref = op.reference(**args)
    bad = dict(args, **{name: _strided(args[name], dim)})
    assert not bad[name].is_contiguous()

    out = op(**bad)
    error = (out.float() - ref.float()).abs().max().item()
    assert error > 0.25, f"strided {name} was expected to be read wrongly; off by {error}"

    args["index_cache"].copy_(before)
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(**bad)
    assert torch.equal(args["index_cache"], before), "a refused call still wrote"


def _variant(args, defect):
    """The compression with one deliberate defect, in plain torch."""
    cache, cos_sin = args["index_cache"], args["cos_sin"]
    cr, tpb = args["compress_ratio"], args["tokens_per_block"]
    out = cache.clone()
    logical = args["logical_positions"].long()
    anchored = (logical + 1) % cr == 0
    anchor, request = logical[anchored], args["request_indices"].long()[anchored]
    members = anchor[:, None] - (cr - 1) + torch.arange(cr, device=DEV)
    pages = args["block_table"].long()[request[:, None], members // tpb]
    slots = members % tpb
    raw = cache[pages, slots, 0]
    if defect == "pool_in_bfloat16":
        acc = raw[:, 0]
        for i in range(1, cr):
            acc = (acc + raw[:, i]).to(torch.bfloat16)
        pooled = (acc / cr).to(torch.bfloat16).float()
    else:
        pooled = (raw.float().sum(dim=1) / cr).to(torch.bfloat16).float()
    scale = torch.rsqrt(pooled.mul(pooled).sum(-1, keepdim=True) / cache.shape[3] + args["eps"])
    weight = args["k_norm_weight"].float()
    gamma = weight if defect == "plain_weight" else weight + 1.0
    normed = (pooled * scale * gamma).to(cos_sin.dtype)
    source = -1 if defect == "rope_at_last_member" else 0
    t, h, w = args["position_cache"][pages[:, source], slots[:, source], :3].long().unbind(-1)
    n = cos_sin.shape[1] // 2
    q = torch.arange(n, device=DEV)
    section = None if defect == "ignore_mrope" else args["mrope_section"]
    if section is None:
        pos = t[:, None].expand(-1, n)
    else:
        pos = torch.where(
            (q % 3 == 1) & (q < 3 * section[1]),
            h[:, None],
            torch.where((q % 3 == 2) & (q < 3 * section[2]), w[:, None], t[:, None]),
        )
    cos, sin = cos_sin[pos, q], cos_sin[pos, n + q]
    rot = normed[:, : 2 * n]
    if defect == "gptj_pair_split":
        first, second = rot[:, ::2], rot[:, 1::2]
        head = torch.stack([first * cos - second * sin, second * cos + first * sin], -1).flatten(1)
    else:
        first, second = rot[:, :n], rot[:, n:]
        head = torch.cat([first * cos - second * sin, second * cos + first * sin], -1)
    out[pages[:, -1], slots[:, -1], 0] = torch.cat([head, normed[:, 2 * n :]], -1).to(
        torch.bfloat16
    )
    return out


@pytest.mark.parametrize(
    "defect, floor",
    [
        ("rope_at_last_member", 1.0),
        ("plain_weight", 1.0),
        ("ignore_mrope", 1.0),
        ("gptj_pair_split", 1.0),
        ("pool_in_bfloat16", 0.0),
    ],
)
def test_wrong_variants_land_outside_the_gate(defect: str, floor: float) -> None:
    """Every defect differs in at least one bit; four of them by more than 1.0.

    Pooling in bf16 moves the result by only ~0.03 (old receipt: 2124 elements,
    max 0.03125) -- inside any bf16 band, which is why the gate is bitwise.
    """
    args = _scene(seed=15)
    ref = op.reference(**args)
    assert torch.equal(_variant(args, None).view(torch.int16), ref.view(torch.int16))
    wrong = _variant(args, defect)
    out = op(**args)
    _compare(out, ref)
    assert int((out.view(torch.int16) != wrong.view(torch.int16)).sum()) > 0, defect
    assert (out.float() - wrong.float()).abs().max().item() > floor
