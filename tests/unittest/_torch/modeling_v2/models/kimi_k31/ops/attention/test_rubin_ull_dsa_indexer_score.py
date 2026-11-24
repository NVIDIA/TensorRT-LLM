# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_indexer_score entry: every cell against the FP64 reference, the
unwritten columns untouched, and the vector maxima bitwise against the kernel's own logits."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_indexer_score import (  # noqa: E501
    DIM,
    HEADS,
    PAGE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_indexer_score import (  # noqa: E501
    dsa_indexer_score as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the tcgen05 scorer needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

POISON = -12345.0


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lens = spec["lens"]
    requests, next_n = len(lens), len(lens[0])
    rows = requests * next_n
    pages = [-(-max(r) // PAGE) for r in lens]
    blocks = sum(pages) + 3
    q = (torch.randn(requests, next_n, HEADS, DIM, generator=g, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    codes = (torch.randn(blocks, PAGE, DIM, generator=g, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(blocks, PAGE, generator=g, device="cuda") * 0.02 + 0.001
    kv = torch.cat([codes.view(torch.uint8).view(blocks, -1), scales.view(torch.uint8)], dim=1)
    kv = kv.view(blocks, PAGE, 1, DIM + 4)
    weights = torch.randn(rows, HEADS, generator=g, device="cuda") * 0.1
    perm = torch.randperm(blocks, generator=g, device="cuda").to(torch.int32)
    width = max(pages)
    table = torch.zeros(requests, width + 1, dtype=torch.int32, device="cuda")
    start = 0
    for b, n in enumerate(pages):
        table[b, :n] = perm[start : start + n]
        start += n
    row_lens = torch.tensor([n for r in lens for n in r], dtype=torch.int32, device="cuda")
    pitch = -(-max(max(r) for r in lens) // 8) * 8 + 8
    out = torch.full((rows, pitch), POISON, device="cuda")
    tops = None
    if spec["tops_from"]:
        tops = torch.full((rows, pitch // 8), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
    return (q, kv, weights, row_lens, table, out, tops, max(spec["tops_from"], 1))


def _ordered(x: torch.Tensor) -> torch.Tensor:
    """``ptx.ordered_bits`` of FP32 values, as int64 in [0, 2**32)."""
    bits = x.view(torch.int32).long() & 0xFFFFFFFF
    neg = bits >= 0x80000000
    return torch.where(neg, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    q, kv, weights, row_lens, table, out, tops, tops_from = args
    with validating(op):
        op(*args)
    torch.cuda.synchronize()
    ref = op.reference(*args)
    lens = row_lens.tolist()
    flat = kv.view(kv.shape[0], -1)
    scales = flat[:, PAGE * DIM :].contiguous().view(torch.float32).view(-1, PAGE)
    for t, n in enumerate(lens):
        b = t // q.shape[1]
        pages = table[b, : -(-n // PAGE)].long()
        keys = flat[pages, : PAGE * DIM].contiguous().view(torch.float8_e4m3fn).view(-1, DIM)[:n]
        # The kernel's error: FP32 sums of 128 exact FP8 products, then 32 FP32 FMAs, so the band
        # is a few FP32 ulps of the magnitudes it sums.
        mag = (
            (q[b, t % q.shape[1]].float().double().abs() @ keys.float().double().abs().T)
            * weights[t].double().abs()[:, None]
        ).sum(0) * scales[pages].reshape(-1)[:n].double()
        err = (out[t, :n].double() - ref[t]).abs()
        assert bool((err <= 64 * 2.0**-24 * mag + 1e-30).all()), (
            f"row {t}: max err {err.max().item():.3e}"
        )
        assert bool((out[t, n:] == POISON).all()), f"row {t}: wrote past its {n} keys"
    if tops is not None:
        for t, n in enumerate(lens):
            if n < tops_from:
                assert bool((tops[t] == 0x5A5A5A5A).all()), f"row {t}: maxima below tops_from"
                continue
            vectors = -(-n // 8)
            img = torch.zeros(vectors * 8, dtype=torch.long, device="cuda")
            img[:n] = _ordered(out[t, :n].contiguous())
            want = img.view(vectors, 8).amax(1)
            got = tops[t, :vectors].long() & 0xFFFFFFFF
            assert torch.equal(got, want), f"row {t}: vector maxima differ"


def test_is_valid_refuses_what_the_kernel_would_read_wrongly() -> None:
    args = list(_build(CELLS[0].spec, seed=1))
    with pytest.raises(ValueError, match="unsupported"):
        op.is_valid(args[0][..., :64].contiguous(), *args[1:])  # head dim 64
    short = args[5][:, :64]
    with pytest.raises(ValueError, match="pitch"):
        op.is_valid(*args[:5], short, None, 1)
