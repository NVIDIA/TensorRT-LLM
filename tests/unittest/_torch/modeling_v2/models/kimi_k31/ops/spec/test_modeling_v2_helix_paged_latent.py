# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the DFlash worker's Helix paged-latent entries (``ops/spec/helix_paged_latent.py``).

Under Helix context parallelism global page ``g`` is local page ``g // cp_size`` of rank ``g % cp_size``. Each CP rank
prepares a generation step over its own pages (``helix_paged_latent_prepare``): the context latents of the step go to
rank-local columns, the store flag is -1 on other ranks' positions, ``num_ctx`` is capped at
``capacity * cp_size - block_size``, and ``ctx_len`` / ``seq_lens_cuda`` advance in place.
``helix_store_context_latents`` then stores the flagged latents and leaves the -1 slots untouched.

The cells of both entries run against the entry's reference, byte for byte, in-place updates included.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.helix_paged_latent import (
    helix_paged_latent_prepare,
    helix_store_context_latents,
)

__extra_import_path__ = ["../..", "../../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)

_MAX_BLOCKS = 4


def _generator(why: str) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(why.encode()) % 2**31)


def _prepare_inputs(gen, gens, cp_size, page_size, block_size, width, first_row=1) -> dict:
    """One generation step of ``gens`` requests behind ``first_row`` context rows, as the worker hands it over: the
    first request's context crosses a page, the second request (if any) is a padding row on the dummy slot, the
    third (if any) runs past the context its rank's pages cover and past ``max_ctx``."""
    rows = first_row + gens
    slots_total = gens + 3
    dummy = slots_total - 1

    def ints(lo, hi, shape, dtype):
        return torch.randint(lo, hi, shape, generator=gen, device="cuda", dtype=dtype)

    slots = torch.randperm(slots_total - 1, generator=gen, device="cuda")[:gens].to(torch.int32)
    cap_tokens = _MAX_BLOCKS * page_size * cp_size
    ctx_len = ints(0, cap_tokens + 2 * page_size, (slots_total,), torch.int64)
    ctx_len[slots[0]] = page_size - 2
    if gens > 1:
        slots[1] = dummy
    if gens > 2:
        ctx_len[slots[2]] = cap_tokens + page_size
    return dict(
        num_accepted_tokens=ints(1, width + 1, (rows,), torch.int32),
        accepted_tokens=ints(0, 163840, (rows, width), torch.int32),
        slots=slots,
        ctx_len=ctx_len,
        base_pages=ints(0, 3, (slots_total,), torch.int64),
        block_counts=ints(1, _MAX_BLOCKS + 1, (rows,), torch.int64),
        block_tables=ints(0, 4096, (rows, _MAX_BLOCKS), torch.int32),
        seq_lens_cuda=ints(1, 100000, (rows,), torch.int32),
        first_row=first_row,
        dummy_slot=dummy,
        max_ctx=cap_tokens + page_size // 2,
        page_size=page_size,
        block_size=block_size,
        width=width,
        cp_size=cp_size,
    )


def _clone(inputs: dict) -> dict:
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _assert_prepare_matches(inputs: dict, cp_rank: int) -> None:
    """The entry and its reference on copies of ``inputs``: every output and the in-place updates, byte for byte."""
    ours, ref = _clone(inputs), _clone(inputs)
    out = helix_paged_latent_prepare(**ours, cp_rank=cp_rank)
    want = helix_paged_latent_prepare.reference(**ref, cp_rank=cp_rank)
    for field in out._fields:
        a, b = getattr(out, field), getattr(want, field)
        assert a.dtype == b.dtype and a.shape == b.shape, (
            field,
            a.dtype,
            b.dtype,
            a.shape,
            b.shape,
        )
        assert torch.equal(_bytes(a), _bytes(b)), field
    assert torch.equal(ours["ctx_len"], ref["ctx_len"]), "ctx_len"
    assert torch.equal(ours["seq_lens_cuda"], ref["seq_lens_cuda"]), "seq_lens_cuda"


def _store_inputs(gen, cp_size, tokens, layers, head_dim, dtype, page_size=64, pages=64):
    """Latents and pools for ``tokens`` distinct destinations, flags 1 / 0 on this rank and -1 on the others."""
    latent = torch.randn((tokens, layers, head_dim), generator=gen, device="cuda").to(dtype)
    pools = [
        torch.randn((pages, 1, 1, page_size, head_dim), generator=gen, device="cuda").to(dtype)
        for _ in range(layers)
    ]
    dest = torch.randperm(pages * page_size, generator=gen, device="cuda")[:tokens]
    flag = torch.randint(0, 2, (tokens,), generator=gen, device="cuda", dtype=torch.int8)
    elsewhere = torch.randint(0, cp_size, (tokens,), generator=gen, device="cuda") != 0
    mask = torch.where(elsewhere, torch.full_like(flag, -1), flag)
    return latent, pools, dest // page_size, dest % page_size, mask


def _store(fn, latent, pools, pages, offsets, mask) -> list:
    pools = [p.clone() for p in pools]
    ptrs = torch.tensor([p.data_ptr() for p in pools], dtype=torch.int64, device="cuda")
    fn(latent, pools, ptrs, pages, offsets, mask)
    return pools


PREPARE_CELLS = sample_cells(helix_paged_latent_prepare.CELLS)[:2]


@pytest.mark.parametrize(
    "cell",
    PREPARE_CELLS,
    ids=[c.why[:48] for c in PREPARE_CELLS],
)
def test_prepare_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(helix_paged_latent_prepare)
    spec = dict(cell.spec)
    cp_rank = spec.pop("cp_rank")
    inputs = _prepare_inputs(_generator(cell.why), **spec)
    _assert_prepare_matches(inputs, cp_rank)


STORE_CELLS = sample_cells(helix_store_context_latents.CELLS)[:2]


@pytest.mark.parametrize(
    "cell",
    STORE_CELLS,
    ids=[c.why[:48] for c in STORE_CELLS],
)
def test_store_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(helix_store_context_latents)
    args = _store_inputs(_generator(cell.why), **cell.spec)
    ours = _store(helix_store_context_latents, *args)
    want = _store(helix_store_context_latents.reference, *args)
    for layer, (a, b) in enumerate(zip(ours, want)):
        assert torch.equal(_bytes(a), _bytes(b)), f"layer {layer}"
