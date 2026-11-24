# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rubin_idx_k_cache_scatter_fp4 entry (FP4 index keys into the paged cache)."""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_k_cache_scatter_fp4 import (  # noqa: E402,E501
    rubin_idx_k_cache_scatter_fp4 as op,
)

__extra_import_path__ = ["../../../.."]
if torch.cuda.is_available():  # the GPU cells' guard interposer: a CPU run needs neither
    from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

cpu = pytest.mark.cpu_only
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="drives the kernel on sm_107")

DATA, SCALE = 64, 4
TOKEN = DATA + SCALE


def _slots(token_slots: torch.Tensor, page: int, layout_token: int = TOKEN, data: int = DATA):
    """Flat byte offsets of the keys and exponent words of the given global token slots."""
    pages, pos = token_slots // page, token_slots % page
    base = pages * page * layout_token
    return (base + pos * data).to(torch.int64), (base + page * data + pos * SCALE).to(torch.int64)


def _build(spec, seed: int, device="cpu") -> tuple[dict, torch.Tensor]:
    """The call's arguments over a random-byte cache, and the cache's base tensor (padding
    included). Distinct random token slots; slot mappings longer than num_tokens when the cell
    says so (the extra rows point at real slots and must stay unwritten); every
    ``skipped_every``-th row's slots -1 when the cell says so (a helix8 rank's rows of another)."""
    gen = torch.Generator().manual_seed(seed)
    page, pages, pad = spec["tokens_per_block"], spec["num_pages"], spec["page_stride_pad"]
    rows = spec["slot_rows"] or spec["num_tokens"]
    base = torch.randint(0, 256, (pages, page * TOKEN + pad), generator=gen).to(torch.uint8)
    base = base.to(device)
    token_slots = torch.randperm(pages * page, generator=gen)[:rows]
    slot_k, slot_s = _slots(token_slots, page)
    if spec.get("skipped_every"):
        slot_k[:: spec["skipped_every"]] = -1
        slot_s[:: spec["skipped_every"]] = -1
    k = torch.randint(-128, 128, (rows, DATA), generator=gen).to(torch.int8)
    scale = torch.randint(-(2**31), 2**31, (rows, 1), generator=gen, dtype=torch.int64)
    args = dict(
        k_fp4=k.to(device),
        k_scale=scale.to(torch.int32).to(device),
        k_cache=base[:, : page * TOKEN].view(pages, page, 1, TOKEN),
        slot_mapping_fp4=slot_k.to(device),
        slot_mapping_scale=slot_s.to(device),
        num_tokens=spec["num_tokens"],
    )
    return args, base


@cpu
def test_reference_writes_each_key_and_word_where_the_page_layout_puts_them():
    # A padded page stride (the V2 per-layer view): the cache view is not contiguous.
    spec = dict(num_tokens=37, tokens_per_block=16, num_pages=9, page_stride_pad=52, slot_rows=50)
    args, base = _build(spec, seed=55)
    before = base.clone()
    op.reference(**args)
    page = spec["tokens_per_block"]
    expected = before.clone()
    for t in range(spec["num_tokens"]):
        p, off = divmod(int(args["slot_mapping_fp4"][t]), page * TOKEN)
        expected[p, off : off + DATA] = args["k_fp4"][t].view(torch.uint8)
        p, off = divmod(int(args["slot_mapping_scale"][t]), page * TOKEN)
        expected[p, off : off + SCALE] = args["k_scale"][t].view(torch.uint8)
    assert torch.equal(base, expected)
    # The rows past num_tokens were not written, and a pad of a wider page stride is untouched.
    assert torch.equal(base[:, page * TOKEN :], before[:, page * TOKEN :])


@cpu
def test_reference_skips_a_negative_slot_and_writes_nothing_for_no_tokens():
    """A token whose slots are -1 (a helix8 row another rank owns) is skipped, as the compiled op
    skips it; the guard admits such a call; no tokens write nothing."""
    spec = dict(
        num_tokens=12,
        tokens_per_block=16,
        num_pages=4,
        page_stride_pad=0,
        slot_rows=None,
        skipped_every=3,
    )
    args, base = _build(spec, seed=5)
    op.is_valid(**args)
    before = base.clone()
    op.reference(**args)
    page = spec["tokens_per_block"]
    expected = before.clone()
    for t in range(spec["num_tokens"]):
        if t % 3 == 0:
            continue
        p, off = divmod(int(args["slot_mapping_fp4"][t]), page * TOKEN)
        expected[p, off : off + DATA] = args["k_fp4"][t].view(torch.uint8)
        p, off = divmod(int(args["slot_mapping_scale"][t]), page * TOKEN)
        expected[p, off : off + SCALE] = args["k_scale"][t].view(torch.uint8)
    assert torch.equal(base, expected)
    written = base.clone()
    op.reference(**dict(args, num_tokens=0))
    assert torch.equal(base, written)
    skipped_only = dict(args, slot_mapping_fp4=args["slot_mapping_fp4"][::3])
    skipped_only.update(slot_mapping_scale=args["slot_mapping_scale"][::3], num_tokens=4)
    op.is_valid(**skipped_only)
    op.reference(**skipped_only)
    assert torch.equal(base, written)


@cpu
def test_guard_admits_a_well_formed_call():
    args, _ = _build(
        dict(num_tokens=20, tokens_per_block=16, num_pages=4, page_stride_pad=0, slot_rows=None),
        seed=1,
    )
    op.is_valid(**args)
    op.is_valid(**dict(args, k_fp4=args["k_fp4"].view(torch.uint8)))
    op.is_valid(**dict(args, num_tokens=0))


def _fp8_layout(args):
    """The same tokens' slot mappings computed for the FP8 cache's 132-byte tokens."""
    page = args["k_cache"].shape[1]
    tokens = (
        args["slot_mapping_fp4"] // (page * TOKEN) * page
        + (args["slot_mapping_fp4"] % (page * TOKEN)) // DATA
    )
    k, s = _slots(tokens, page, layout_token=132, data=128)
    logical = args["k_cache"].numel()
    keep = (k + 128 <= logical) & (s + 4 <= logical)
    return dict(
        args, slot_mapping_fp4=k[keep], slot_mapping_scale=s[keep], num_tokens=int(keep.sum())
    )


@cpu
@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: dict(a, k_fp4=torch.zeros(a["k_fp4"].shape[0], 128, dtype=torch.int8)), "64"),
        (lambda a: dict(a, k_scale=a["k_scale"].view(torch.float32)), "int32"),
        (lambda a: dict(a, k_cache=torch.zeros(4, 16, 1, 132, dtype=torch.uint8)), "68"),
        (lambda a: dict(a, slot_mapping_fp4=a["slot_mapping_fp4"] + a["k_cache"].numel()), "past"),
        (lambda a: dict(a, slot_mapping_scale=a["slot_mapping_scale"] - 2), "exponent"),
        (lambda a: dict(a, slot_mapping_fp4=a["slot_mapping_fp4"] + 8), "key rows"),
        (_fp8_layout, "key rows|exponent"),
        (lambda a: dict(a, num_tokens=a["slot_mapping_fp4"].shape[0] + 1), "num_tokens"),
    ],
    ids=[
        "fp8 keys",
        "fp32 scales",
        "fp8 cache",
        "past the cache",
        "scale off a word",
        "key off a row",
        "fp8 slot layout",
        "too many tokens",
    ],
)
def test_guard_refuses(change, match):
    args, _ = _build(
        dict(num_tokens=20, tokens_per_block=16, num_pages=4, page_stride_pad=0, slot_rows=None),
        seed=2,
    )
    with pytest.raises(AssertionError, match=match):
        op.is_valid(**change(args))


def _pages_for(spec) -> dict:
    """A cell's spec with enough pages for its token slots."""
    need = -(-(spec["slot_rows"] or spec["num_tokens"]) // spec["tokens_per_block"]) + 2
    return dict(spec, num_pages=max(spec["num_pages"], need))


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = _pages_for(cell.spec)
    args, base = _build(spec, zlib.crc32(cell.why.encode()) % 2**31, device="cuda")
    ref_base = base.clone()
    page = spec["tokens_per_block"]
    ref_args = dict(args, k_cache=ref_base[:, : page * TOKEN].view(-1, page, 1, TOKEN))
    assert op(**args) is None
    op.reference(**ref_args)
    torch.cuda.synchronize()
    assert torch.equal(base, ref_base), "the op wrote other bytes than the reference"


@gpu
def test_graph_replay_writes_the_new_keys_at_the_new_slots():
    spec = dict(num_tokens=96, tokens_per_block=64, num_pages=64, page_stride_pad=0, slot_rows=None)
    first, base = _build(spec, seed=31, device="cuda")
    second, _ = _build(spec, seed=32, device="cuda")
    static = {
        k: v.clone() if torch.is_tensor(v) and k != "k_cache" else v for k, v in first.items()
    }
    op(**static)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**static)
    for name in ("k_fp4", "k_scale", "slot_mapping_fp4", "slot_mapping_scale"):
        static[name].copy_(second[name])
    want = base.clone()
    graph.replay()
    op.reference(**dict(second, k_cache=want[:, : 64 * TOKEN].view(-1, 64, 1, TOKEN)))
    torch.cuda.synchronize()
    assert torch.equal(base, want)
