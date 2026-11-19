# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 gather_k_cache_paged entry: a sample of its cells' gathered bytes exactly,
against the reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.gather_k_cache_paged import (  # noqa: E501
    k31_gather_k_cache_paged as op,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_dsa_inputs import (  # noqa: E402 — needs the path declared above
    BYTES_PER_TOKEN,
    raw_index_cache,
    slot_mappings,
)
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "gather_k_cache_paged requires a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)
#: A typical cell of ``op.CELLS``.
_CELLS = sample_cells(CELLS, size_key="tokens")[1:2]


def _build(spec, seed):
    """The cells' 128-wide index-key cache (132 bytes per token, pages at a page stride 1024
    bytes wider than a page) and the slot mappings of tokens ``[0, start + tokens)``."""
    gen = torch.Generator().manual_seed(seed)
    tokens, start, tpb, head_dim = (
        spec["tokens"],
        spec["start"],
        spec["tokens_per_block"],
        spec["head_dim"],
    )
    pages = (start + tokens) // tpb + 2
    _, cache = raw_index_cache(pages, tpb, gen, page_stride_pad=1024)
    fp8, scale = slot_mappings(start + tokens, pages, tpb, gen)
    if spec.get("holes"):
        fp8[start::5] = -1
    if spec.get("misaligned"):
        # Any byte offset inside a page that keeps the key (or the scale) inside it.
        page_bytes = tpb * BYTES_PER_TOKEN
        first = scale // page_bytes * page_bytes
        fp8 = first + torch.randint(0, page_bytes - head_dim + 1, fp8.shape, generator=gen).cuda()
        scale = first + torch.randint(0, page_bytes - 4 + 1, fp8.shape, generator=gen).cuda()
    return cache, fp8, scale, start, tokens, head_dim


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:44] for c in _CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        k, s = op(*args)
    ref_k, ref_s = op.reference(*args)
    _, fp8, _, start, tokens, _ = args
    ok = fp8[start : start + tokens] >= 0
    assert torch.equal(k.view(torch.uint8)[ok], ref_k.view(torch.uint8)[ok])
    assert torch.equal(s.view(torch.int32)[ok], ref_s.view(torch.int32)[ok])
