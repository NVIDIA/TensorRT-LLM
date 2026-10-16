# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_index_store catalog entry.

The paged state is real: an index-key plane and a position plane of `pages`
pages, and the engine's `[slots, width]` table -- more slots than requests,
each live row's pages drawn from a seeded permutation of the pool, `-1` past
its allocation, rows past the batch holding a previous step's plausible page
ids. Both planes are pre-filled with a sentinel so a stray write shows.

Out-of-range pages, requests and slots are deliberately absent: each is a CUDA
device-side assert that poisons the process for every test after it.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_index_store import (  # noqa: E501
    qsa_paged_index_store as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_index_store requires a CUDA device"


def _compare(out, ref) -> None:
    """Bitwise over both whole planes, as raw bits.

    A store is a move with one conversion that `reference` performs too, so
    there is nothing to spend a tolerance on; comparing every slot rather
    than the written ones is the stray-write check.
    """
    for name, got, want in zip(("index_cache", "position_cache"), out, ref):
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert got.shape == want.shape, (name, got.shape, want.shape)
        torch.testing.assert_close(
            _bits(got), _bits(want), rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


DEV = "cuda"
KEY_FILL = -7.0
POS_FILL = -7
ENGINE_MAX_NUM_TOKENS = 8192


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(as_int)


def _build(spec, seed: int):
    """Real planes, table and rows for one cell; returns the op's kwargs and extras."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tpb, hkv, head_dim, axes = (spec[k] for k in ("tokens_per_block", "hkv", "head_dim", "axes"))
    slots, width, pages = spec["slots"], spec["width"], spec["pages"]
    segments = spec["segments"]
    requests = [r for r, (_, n) in enumerate(segments) for _ in range(n)]
    positions = [p for start, n in segments for p in range(start, start + n)]
    rows = len(positions)

    table = torch.randint(0, pages, (slots, width), generator=gen, dtype=torch.int32)
    order = torch.randperm(pages, generator=gen).to(torch.int32)
    used = 0
    for r, (start, n) in enumerate(segments):
        cols = (start + n - 1) // tpb + 1
        table[r] = -1
        table[r, :cols] = order[used : used + cols]
        used += cols
    assert used <= pages and len(segments) < slots, "pool or table too small for the batch"
    table = table.to(DEV)
    if spec["table_layout"] == "column_slice":
        wide = torch.full((slots, width + 4), -1, dtype=torch.int32, device=DEV)
        wide[:, 2 : 2 + width] = table
        table = wide[:, 2 : 2 + width]
        assert not table.is_contiguous()

    projection = torch.randn(
        rows, spec["index_heads"] + hkv, head_dim, generator=gen, dtype=torch.float32
    ).to(device=DEV, dtype=spec["key_dtype"])
    token_k = projection[:, spec["index_heads"] :]
    if spec["key_layout"] == "contiguous":
        token_k = token_k.contiguous()

    request_indices = torch.tensor(requests, dtype=spec["request_dtype"], device=DEV)
    logical_positions = torch.tensor(positions, dtype=spec["position_dtype"], device=DEV)
    if spec["meta_layout"] == "strided":
        wide_requests = torch.zeros(rows, 2, dtype=spec["request_dtype"], device=DEV)
        wide_positions = torch.zeros(rows, 3, dtype=spec["position_dtype"], device=DEV)
        wide_requests[:, 0] = request_indices
        wide_positions[:, 1] = logical_positions
        request_indices, logical_positions = wide_requests[:, 0], wide_positions[:, 1]
        assert not logical_positions.is_contiguous()

    # Three genuinely different axes, as an image span has: t, t + 7, t + 14.
    coords = logical_positions.to(torch.int32)[:, None] + 7 * torch.arange(
        3, dtype=torch.int32, device=DEV
    )
    if spec["coords_layout"] == "transposed":
        # The engine's `[3, max_num_tokens]` position buffer, transposed.
        buffer = torch.full((3, ENGINE_MAX_NUM_TOKENS), -1, dtype=torch.int32, device=DEV)
        buffer[:, :rows] = coords.t()
        coords = buffer[:, :rows].t()
    elif spec["coords_layout"] == "stride0":
        coords = logical_positions[:, None].expand(-1, 3)
        assert coords.stride(1) == 0
    else:
        coords = coords.contiguous()

    extras = {}
    if spec["cache_layout"] == "role_view":
        pool = torch.full((pages, 2, tpb, hkv, head_dim), KEY_FILL, device=DEV).to(torch.bfloat16)
        index_cache = pool[:, 0]
        extras["neighbour"] = (pool[:, 1], pool[:, 1].clone())
        assert not index_cache.is_contiguous()
    else:
        index_cache = torch.full((pages, tpb, hkv, head_dim), KEY_FILL, device=DEV).to(
            torch.bfloat16
        )
    position_cache = torch.full((pages, tpb, axes), POS_FILL, dtype=torch.int32, device=DEV)
    args = dict(
        token_k=token_k,
        position_coordinates=coords,
        index_cache=index_cache,
        position_cache=position_cache,
        request_indices=request_indices,
        logical_positions=logical_positions,
        block_table=table,
        tokens_per_block=tpb,
    )
    return args, extras


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = args["position_cache"].clone()
    ref = op.reference(**args)
    out = op(**args)
    assert out[0] is args["index_cache"] and out[1] is args["position_cache"]
    _compare(out, ref)
    if "neighbour" in extras:
        view, saved = extras["neighbour"]
        assert torch.equal(_bits(view), _bits(saved)), "the neighbouring role moved"
    if cell.spec["axes"] > 3:
        assert torch.equal(out[1][..., 3:], before[..., 3:]), "axes past the coordinates moved"
    if cell.spec["coords_layout"] == "stride0":
        for axis in (1, 2):
            assert torch.equal(out[1][..., axis], out[1][..., 0])


def _small(seed: int, **overrides):
    """Four decode rows at the checkpoint's call-site layouts."""
    spec = {**op.CELLS[0].spec, "segments": ((0, 1), (7, 1), (32, 1), (40, 1)), "pages": 8}
    return _build({**spec, **overrides}, seed=seed)[0]


def test_only_the_named_slots_change() -> None:
    """The exact set of (page, slot) pairs written, independent of the reference."""
    args = _small(seed=13)
    before_index = args["index_cache"].clone()
    before_position = args["position_cache"].clone()
    op(**args)
    table = args["block_table"]
    written = {(int(table[r, p // 32]), p % 32) for r, p in enumerate((0, 7, 32, 40))}
    changed_index = {
        (int(p), int(s))
        for p, s in torch.nonzero(
            (_bits(args["index_cache"]) != _bits(before_index)).flatten(2).any(-1)
        )
    }
    changed_position = {
        (int(p), int(s))
        for p, s in torch.nonzero((args["position_cache"] != before_position).any(-1))
    }
    assert changed_index == written, f"index plane touched {changed_index}, expected {written}"
    assert changed_position == written, f"position plane touched {changed_position}"


def _one_row(seed: int):
    """Row 0 of `_small` alone, so nothing else can land where it does."""
    args = _small(seed=seed)
    for name in ("token_k", "position_coordinates", "request_indices", "logical_positions"):
        args[name] = args[name][:1]
    return args


def test_coordinates_wider_than_int32_wrap_rather_than_raise() -> None:
    """int64 coordinates are truncated into the int32 plane with nothing raised."""
    args = _small(seed=39)
    args["position_coordinates"] = torch.full((4, 3), 2**31 + 5, dtype=torch.int64, device=DEV)
    op(**args)
    page = int(args["block_table"][0, 0])
    assert int(args["position_cache"][page, 0, 0]) == -2147483643


def test_unallocated_page_sentinel_writes_the_last_page() -> None:
    """The `-1` sentinel is not skipped: it wraps to the pool's last page."""
    args = _one_row(seed=42)
    pages = args["index_cache"].shape[0]
    args["block_table"][0] = -1
    before = args["index_cache"].clone()
    op(**args)
    assert torch.equal(
        _bits(args["index_cache"][pages - 1, 0]), _bits(args["token_k"][0].to(torch.bfloat16))
    ), "the -1 row was expected on the last page"
    assert torch.equal(_bits(args["index_cache"][:-1]), _bits(before[:-1]))


def test_negative_logical_position_lands_on_the_last_table_column() -> None:
    """`-1 // 32 == -1` and `-1 % 32 == 31`: the last column's page, last slot."""
    args = _one_row(seed=44)
    args["logical_positions"][0] = -1
    args["block_table"][0, -1] = 5
    op(**args)
    assert torch.equal(
        _bits(args["index_cache"][5, 31]), _bits(args["token_k"][0].to(torch.bfloat16))
    )


def test_two_rows_on_one_slot_store_one_of_them() -> None:
    """Nothing orders duplicate destinations; the slot holds one of the two rows."""
    args = _small(seed=47)
    args["logical_positions"][1] = args["logical_positions"][0]
    args["request_indices"][1] = args["request_indices"][0]
    args["token_k"] = torch.stack(
        [torch.full((1, 128), float(v), dtype=torch.bfloat16, device=DEV) for v in (1, 2, 3, 4)]
    )
    op(**args)
    stored = float(args["index_cache"][int(args["block_table"][0, 0]), 0, 0, 0])
    assert stored in (1.0, 2.0), f"a duplicate slot held {stored}, neither row's value"


def test_repeats_are_stream_ordered_and_idempotent() -> None:
    """A read enqueued after the store sees it; repeating the call repeats it."""
    args = _small(seed=49)
    op(**args)
    first = (args["index_cache"].clone(), args["position_cache"].clone())
    for _ in range(7):
        op(**args)  # no synchronize: the clone below is on the same stream
        assert torch.equal(_bits(args["index_cache"]), _bits(first[0]))
        assert torch.equal(args["position_cache"], first[1])


def test_cuda_graph_capture_and_replay() -> None:
    """Captured once, replayed against new positions written into the same buffer."""
    args = _small(seed=52)
    positions = args["logical_positions"]
    replayed = positions.clone()
    positions.zero_()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(**args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)

    positions.copy_(replayed)
    args["index_cache"].fill_(KEY_FILL)
    args["position_cache"].fill_(POS_FILL)
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare((args["index_cache"], args["position_cache"]), ref)


def test_the_stored_key_is_raw_not_normalized() -> None:
    """Discrimination against a Gemma-normed key, the decode kernel's form.

    A key stored normed double-normalizes every anchor `qsa_prefill_compress`
    builds. Old receipt at its seeds: 3.0625 max abs apart; the floor states a
    bound rather than one draw.
    """
    args = _small(seed=66)
    out = op(**args)
    gen = torch.Generator(device=DEV).manual_seed(68)
    weight = torch.randn(128, generator=gen, device=DEV) * 0.5
    x = args["token_k"].float()
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + weight)).to(
        torch.bfloat16
    )
    page = int(args["block_table"][0, 0])
    stored = out[0][page, 0]
    assert torch.equal(_bits(stored), _bits(args["token_k"][0]))
    distance = (stored.float() - normed[0].float()).abs().max().item()
    assert distance > 0.25, f"a normed key differs from the raw one by only {distance}"


# ---------------------------------------------------------------- the guards
#
# Each case is shown first unguarded -- accepted, and silently wrong -- then
# refused under `validating`.


def _unguarded_then_refused(args, match: str) -> None:
    op(**args)
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(**args)


def test_guard_refuses_two_dimensional_request_indices() -> None:
    args = _small(seed=60)
    args["request_indices"] = args["request_indices"][:, None]
    _unguarded_then_refused(args, "one-dimensional")


def test_guard_refuses_metadata_of_unequal_length() -> None:
    """A length-1 position vector collapses every row onto one slot."""
    args = _small(seed=61)
    args["request_indices"] = torch.zeros(4, dtype=torch.int32, device=DEV)
    args["logical_positions"] = args["logical_positions"][:1]
    before = args["index_cache"].clone()
    op(**args)
    changed = (_bits(args["index_cache"]) != _bits(before)).flatten(1).any(-1).sum()
    assert int(changed) == 1, "four rows were expected to land on one page"
    with pytest.raises(AssertionError, match="logical_positions has"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_single_token_k_row() -> None:
    """One key row broadcasts into every destination slot."""
    args = _small(seed=62)
    args["token_k"] = args["token_k"][:1]
    op(**args)
    table = args["block_table"]
    last = args["index_cache"][int(table[3, 1]), 8]
    assert torch.equal(_bits(last), _bits(args["token_k"][0])), "row 0 should fill row 3's slot"
    with pytest.raises(AssertionError, match="token_k has"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_single_coordinate_row() -> None:
    args = _small(seed=63)
    args["position_coordinates"] = args["position_coordinates"][:1]
    _unguarded_then_refused(args, "position_coordinates has")


def test_guard_refuses_a_page_size_that_is_not_the_cache_s() -> None:
    """Position 40 at tokens_per_block 16 is page column 2, slot 8; its slot at
    the cache's 32 is column 1, slot 8 -- which the call never writes."""
    args = _small(seed=64)
    args["tokens_per_block"] = 16
    # Column 2 is unallocated for both rows reaching it; point it inside the pool
    # at a page that is not the truth page.
    args["block_table"][2:4, 2] = args["block_table"][0, 0]
    truth = int(args["block_table"][3, 1]), 8
    before = args["index_cache"][truth].clone()
    op(**args)
    assert torch.equal(_bits(args["index_cache"][truth]), _bits(before))
    with pytest.raises(AssertionError, match="does not match index_cache"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_position_plane_with_a_different_page_size() -> None:
    """Refused outright: past 16 positions the unguarded write is a device-side
    assert that would poison every later test, so it is not shown."""
    args = _small(seed=65)
    args["position_cache"] = torch.full((8, 16, 3), POS_FILL, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="does not match position_cache"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_cache_with_more_heads_than_the_key() -> None:
    """A one-head key broadcasts into both heads of a two-head cache."""
    args = _small(seed=67)
    args["index_cache"] = torch.full((8, 32, 2, 128), KEY_FILL, device=DEV).to(torch.bfloat16)
    op(**args)
    page = int(args["block_table"][0, 0])
    assert torch.equal(_bits(args["index_cache"][page, 0, 1]), _bits(args["token_k"][0, 0])), (
        "the second head was expected to receive the broadcast"
    )
    with pytest.raises(AssertionError, match="index-KV heads"):
        with validating(op):
            op(**args)
