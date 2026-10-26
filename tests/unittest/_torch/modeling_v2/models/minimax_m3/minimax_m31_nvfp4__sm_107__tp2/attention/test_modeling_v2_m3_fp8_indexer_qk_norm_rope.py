# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_fp8_indexer_qk_norm_rope entry of the MiniMax-M3.1 sm_107 tp2 target: index-Q out and index-K in
the cache against the in-place fused norm + RoPE followed by the E4M3 cast, byte for byte up to the 0.1% budget the
in-tree test of the kernel allows. The entry's cells (one index head, its TP4 lineage) and the TP2 calls (two index
heads per rank, the M3.1 RoPE base 1e7: the unfused index projection of a prefill or mixed step)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention.m3_fp8_indexer_qk_norm_rope import (  # noqa: E501
    m3_fp8_indexer_qk_norm_rope as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

DIM, ROTARY, EPS, THETA, PAGE = 128, 64, 1e-6, 5e6, 128
# The TP2 calls: two index heads per rank, the M3.1 checkpoint's RoPE base.
TP2_INDEX_HEADS, M31_THETA = 2, 1e7


def _bytes_close(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.float8_e4m3fn
    differ = int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
    assert differ <= (a.numel() - 1) // 1000, f"{differ} of {a.numel()} E4M3 bytes differ"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    _check(cell.spec["m"], cell.spec["heads"], THETA, zlib.crc32(cell.why.encode()) % 2**31)


@pytest.mark.parametrize("m", [1, 8, 32, 2048])
def test_tp2_index_heads(m) -> None:
    _check(m, TP2_INDEX_HEADS, M31_THETA, 1000 + m)


def _check(m: int, heads: int, theta: float, seed: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(seed)
    qk = torch.randn(m, (heads + 1) * DIM, generator=g, device="cuda").to(torch.bfloat16)
    qw = (torch.randn(DIM, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    kw = (torch.randn(DIM, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    pos = torch.randint(0, 200000, (m,), generator=g, device="cuda", dtype=torch.int32)
    pages = (m + PAGE - 1) // PAGE + 1
    cache = torch.zeros(pages, 1, PAGE, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    slots = torch.randperm(pages * PAGE, generator=g, device="cuda")[:m].to(torch.int32)
    before = qk.clone()
    args = (qk, cache, slots, heads, DIM, ROTARY, EPS, qw, kw, theta, pos)
    with validating(op):
        q = op(*args)
    assert torch.equal(qk, before), "the BF16 input is left untouched"
    q_ref, k_ref = op.reference(*args)
    _bytes_close(q.reshape(m, heads, DIM), q_ref)
    flat = cache.view(-1, DIM)
    _bytes_close(flat[slots.long()], k_ref)
