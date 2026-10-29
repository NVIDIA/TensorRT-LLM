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


def _compare(out: torch.Tensor, ref: torch.Tensor, flips: int = 0) -> None:
    """Bitwise over the whole cache, except for `flips` elements one bf16 step off.

    Measured on sm_103: the kernel matches this fp32 reference on every element of every
    cell but one, so there is no tolerance to derive. It has to be bitwise -- a pool
    accumulated in bf16 instead of fp32 lands only 0.03125 away, inside any bf16 band.
    Comparing every slot, not just the anchors, is the untouched-slot lock.

    The exception is the cell of 32 MTP7 verify blocks (64 anchors, `rounding_flips`): the
    kernel takes `tl.rsqrt`, the hardware approximation, where the reference takes torch's,
    and the two differ by an fp32 ulp, which moves the bf16 rounding of an element whose fp32
    value sits on a rounding boundary. One element of its 8192 anchor elements is one bf16
    step off.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if flips == 0:
        torch.testing.assert_close(out.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)
        return
    distance = (out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
    assert int(distance.max()) <= 1 and int((distance != 0).sum()) <= flips, (
        f"{int((distance != 0).sum())} elements differ, by up to {int(distance.max())} bf16 "
        f"steps; at most {flips} may differ by one"
    )


DEV = "cuda"
ROTARY_DIM = 64
THETA = 1.0e7
MAX_POSITIONS = 16384
SENTINEL = 0x3C3C


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
        coords = (p, (3 * p + 5 * r) % 500, (5 * p + 11 * r) % 500)
        position_cache[table[r, :cols].long()] = torch.stack(coords, dim=-1).to(torch.int32)
    assert used <= pages and len(segments) < slots, "pool or table too small for the batch"
    k_norm_weight = (torch.randn(head_dim, generator=gen) * 0.1).to(torch.bfloat16)

    logical = torch.tensor(positions, dtype=spec["logical_dtype"])
    request = torch.tensor(requests, dtype=spec["request_dtype"])
    table, index_cache, position_cache = table.to(DEV), index_cache.to(DEV), position_cache.to(DEV)

    extras = {}
    if spec["layout"] == "shared_page":
        # An NVFP4 pool's page: `roles` index pages, then the position page. The other
        # index roles hold a sentinel and must survive, as must the position page.
        roles = spec["roles"]
        index_bytes = tpb * head_dim * 2
        page = torch.empty(pages, roles * index_bytes + tpb * 3 * 4, dtype=torch.uint8, device=DEV)
        others = page[:, : (roles - 1) * index_bytes].view(torch.int16)
        others.fill_(SENTINEL)
        shared_index = page[:, (roles - 1) * index_bytes : roles * index_bytes]
        shared_index = shared_index.view(torch.bfloat16).view(pages, tpb, 1, head_dim)
        shared_position = page[:, roles * index_bytes :].view(torch.int32).view(pages, tpb, 3)
        shared_index.copy_(index_cache)
        shared_position.copy_(position_cache)
        assert shared_index.stride(0) == roles * tpb * head_dim + tpb * 3 * 2
        assert shared_position.stride() == (roles * tpb * head_dim // 2 + tpb * 3, 3, 1)
        index_cache, position_cache = shared_index, shared_position
        extras["neighbour"] = (others, others.clone())
        extras["position_page"] = (position_cache, position_cache.clone())
    else:
        assert spec["layout"] == "plain", spec["layout"]

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
    _compare(out, ref, cell.spec.get("rounding_flips", 0))
    changed = (out.view(torch.int16) != before.view(torch.int16)).any(-1).sum()
    assert int(changed) == _anchors(args), "exactly the anchor slots are written"
    if "neighbour" in extras:
        view, saved = extras["neighbour"]
        assert torch.equal(view, saved), "wrote into another index role of the page"
    if "position_page" in extras:
        view, saved = extras["position_page"]
        assert torch.equal(view, saved), "wrote into the position page"


def _scene(seed: int, **overrides):
    """Two requests of 64 tokens from 0, the target's call, at the checkpoint geometry."""
    spec = {**op.CELLS[0].spec, "segments": ((0, 64),) * 2, "pages": 4, **overrides}
    return _build(spec, seed=seed)[0]


#: The cell of 32 MTP7 verify blocks (256 rows, two anchors each): the segments, pool and table.
_MTP7_32 = next(c.spec for c in op.CELLS if c.why.startswith("decode MTP7: 32 requests"))
_MTP7_BATCH = {key: _MTP7_32[key] for key in ("segments", "pages", "slots", "rounding_flips")}


def test_cuda_graph_capture_and_replay() -> None:
    """The launch grid is fixed by the row count, so it captures and replays: 32 MTP7
    verify blocks, the decode target's captured batch."""
    batch = _MTP7_BATCH
    args = _scene(seed=16, **batch)
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
        _compare(cache, ref, batch["rounding_flips"])


def _strided(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """The same values as a stride-2 view along `dim` of a zero-filled buffer."""
    shape = list(tensor.shape)
    shape[dim] *= 2
    wide = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    index = [slice(None)] * tensor.dim()
    index[dim] = slice(None, None, 2)
    wide[tuple(index)] = tensor
    return wide[tuple(index)]


def test_guard_refuses_strided_logical_positions() -> None:
    """Shown unguarded first -- accepted and wrong -- then refused: a 1-D operand the
    kernel reads flat (`request_indices` and `k_norm_weight` carry the same
    `is_contiguous` guard). Old receipt's max abs error against the right answer: 4.72;
    the floor below states a bound, not one draw.
    """
    args = _scene(seed=12)
    before = args["index_cache"].clone()
    ref = op.reference(**args)
    bad = dict(args, logical_positions=_strided(args["logical_positions"], 0))
    assert not bad["logical_positions"].is_contiguous()

    out = op(**bad)
    error = (out.float() - ref.float()).abs().max().item()
    assert error > 0.25, (
        f"strided logical_positions was expected to be read wrongly; off by {error}"
    )

    args["index_cache"].copy_(before)
    with pytest.raises(AssertionError, match="logical_positions must be contiguous"):
        with validating(op):
            op(**bad)
    assert torch.equal(args["index_cache"], before), "a refused call still wrote"
