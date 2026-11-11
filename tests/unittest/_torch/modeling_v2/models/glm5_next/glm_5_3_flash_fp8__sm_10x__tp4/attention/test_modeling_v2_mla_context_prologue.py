# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry mla_context_prologue."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.mla_context_prologue import (  # noqa: E501
    mla_context_prologue,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
TPB, PAGES = 32, 4100
Q_LORA, KV_LORA, HD, INDEX_WIDTH = 1536, 512, 128, 288
RMS_EPS, LN_EPS = 1e-5, 1e-6


def _args(spec: dict, seed: int):
    """The op's arguments for the packed context rows of ``(cached, new)`` requests, and the
    storage behind both pool views.

    Every page a request reaches gets its own slot. The pools are coalesced views (the
    latent pool the first of two FP8 subpages per slot, the indexer pool the second of three
    BF16 subpages with padded rows) holding stale finite values. ``tables`` adds generation
    tables after the context ones, as in a mixed step.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    positions, request_ids, pages = [], [], []
    for i, (cached, new) in enumerate(spec["context"]):
        positions.append(torch.arange(cached, cached + new))
        request_ids.append(torch.full((new,), i, dtype=torch.int32))
        pages.append(-(-(cached + new) // TPB))
    positions = torch.cat(positions).to(DEVICE)
    request_ids = torch.cat(request_ids).to(DEVICE)
    num_tables = max(spec.get("tables", 0), len(pages))
    tables = torch.zeros(num_tables, spec.get("pages", PAGES), dtype=torch.int64, device=DEVICE)
    slots = 1 + sum(pages)
    order = 1 + torch.randperm(slots - 1, generator=g, device=DEVICE)
    start = 0
    for t, count in enumerate(pages):
        tables[t, :count] = order[start : start + count]
        start += count
    rows = positions.shape[0]
    qkv = torch.randn(rows, Q_LORA + KV_LORA, generator=g, device=DEVICE).to(torch.bfloat16)
    idx = torch.randn(rows, INDEX_WIDTH, generator=g, device=DEVICE).to(torch.bfloat16)
    weights = [
        (torch.randn(d, generator=g, device=DEVICE) * 0.2 + 1.0).to(torch.bfloat16)
        for d in (Q_LORA, KV_LORA, HD)
    ]
    bias = (torch.randn(HD, generator=g, device=DEVICE) * 0.1).to(torch.bfloat16)
    latent_storage = torch.randn(slots, 2, TPB, KV_LORA, generator=g, device=DEVICE) * 8
    latent_storage = latent_storage.to(torch.float8_e4m3fn)
    index_storage = torch.randn(slots, 3, TPB, 3 * HD + 64, generator=g, device=DEVICE)
    index_storage = index_storage.to(torch.bfloat16)
    scale = torch.tensor(0.7, device=DEVICE)
    args = [qkv, idx, *weights, bias, RMS_EPS, LN_EPS, positions, request_ids, tables]
    return args, latent_storage, index_storage, scale


def _pools(latent_storage, index_storage):
    return latent_storage[:, 0], index_storage[:, 1, :, : 3 * HD]


def _bytes(pool):
    """A pool's rows as bytes: boolean indexing does not take every FP8 dtype."""
    return pool.view(torch.uint8)


@pytest.mark.parametrize(
    "cell", mla_context_prologue.CELLS, ids=[c.why[:44] for c in mla_context_prologue.CELLS]
)
def test_mla_context_prologue_matches_reference(cell, certifying):  # noqa: F811
    """The q_a rows, the latent rows and the indexer rows equal the reference's up to one
    rounding step, nothing else moves.

    Both sides normalize in FP32 and round once, but the kernel's sums and reciprocal square
    roots differ from torch's in the last FP32 bits, which can move a value to the
    neighbouring BF16 one: at most one BF16 step, ``2**-7 |x|``, for the q_a rows and the
    normalized indexer keys. The latent is rounded to BF16 and then quantized: a BF16 step can
    move its E4M3 code by one step, at most ``2**-3 |x|`` (``2**-9`` among subnormals). The
    gate columns are copied, bit for bit.
    """
    certifying(mla_context_prologue)
    args, latent_storage, index_storage, scale = _args(
        cell.spec, zlib.crc32(cell.why.encode()) % 2**31
    )
    stale = (latent_storage.clone(), index_storage.clone())
    ref_storage = (latent_storage.clone(), index_storage.clone())
    q_resid = mla_context_prologue(*args, *_pools(latent_storage, index_storage), TPB, scale)
    q_ref = mla_context_prologue.reference(*args, *_pools(*ref_storage), TPB, scale)
    assert q_resid.shape == (args[0].shape[0], Q_LORA) and q_resid.dtype == torch.bfloat16
    assert bool(((q_resid.float() - q_ref.float()).abs() <= 2**-7 * q_ref.float().abs()).all())

    positions, request_ids, tables = args[8], args[9], args[10]
    slot = tables[request_ids.long(), positions // TPB]
    row = positions % TPB
    written = torch.zeros(latent_storage.shape[0], TPB, dtype=torch.bool, device=DEVICE)
    written[slot, row] = True
    # Untouched: the foreign subpages, the padding, the pool keys and every unwritten row.
    assert torch.equal(_bytes(latent_storage[:, 1]), _bytes(stale[0][:, 1]))
    for i in (0, 2):
        assert torch.equal(index_storage[:, i], stale[1][:, i])
    assert torch.equal(index_storage[:, 1, :, 2 * HD :], stale[1][:, 1, :, 2 * HD :])
    for got, want, old in zip(
        _pools(latent_storage, index_storage), _pools(*ref_storage), _pools(*stale)
    ):
        assert torch.equal(_bytes(got)[~written], _bytes(old)[~written])
        assert torch.equal(_bytes(want)[~written], _bytes(old)[~written])
    latent, latent_ref = _pools(latent_storage, index_storage)[0], _pools(*ref_storage)[0]
    got = _bytes(latent)[written].view(torch.float8_e4m3fn).float()
    want = _bytes(latent_ref)[written].view(torch.float8_e4m3fn).float()
    assert bool(((got - want).abs() <= torch.clamp(2**-3 * want.abs(), min=2**-9)).all())
    index, index_ref = _pools(latent_storage, index_storage)[1], _pools(*ref_storage)[1]
    got, want = index[written][:, :HD].float(), index_ref[written][:, :HD].float()
    assert bool(((got - want).abs() <= 2**-7 * want.abs()).all())
    assert torch.equal(index[written][:, HD : 2 * HD], index_ref[written][:, HD : 2 * HD])


def test_guard_refuses_a_position_written_twice():
    """Two rows on one cache position race on its latent and indexer rows."""
    args, latent_storage, index_storage, scale = _args(dict(context=((0, 8),), pages=4), seed=0)
    args[8] = torch.tensor([0, 1, 2, 3, 4, 5, 6, 6], device=DEVICE)
    with pytest.raises(AssertionError, match="written twice"):
        with validating(mla_context_prologue):
            mla_context_prologue(*args, *_pools(latent_storage, index_storage), TPB, scale)
