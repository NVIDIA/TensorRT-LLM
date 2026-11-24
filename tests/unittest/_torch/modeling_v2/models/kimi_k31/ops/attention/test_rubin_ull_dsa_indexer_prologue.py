# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_indexer_prologue entry: FP8 queries and head weights bitwise the
exact quantization, the written keys within the FP8 band of the FP64 LayerNorm, nothing else in the
cache moved."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention._rubin_ull_indexer_math import (  # noqa: E501
    DIM,
    HEADS,
    PAGE,
    PAGE_BYTES,
    cache_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_indexer_prologue import (  # noqa: E501
    dsa_indexer_prologue as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the CuTe DSL prologue runs where the K3.1 ULL target does (SM10x)",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

WEIGHT_SCALE = 128**-0.5 * 32**-0.5
BLOCKS = 40


def _runner():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.rubin_ull_dsa_index import (  # noqa: E501
        indexer_prologue,
    )

    return indexer_prologue(torch.device("cuda", torch.cuda.current_device()))


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["rows"]
    # As the target calls it: q the cuBLAS query rows [M, 4096] viewed per head, k and w column
    # views of the DSA hidden projection rows.
    q = torch.randn(rows, HEADS * DIM, generator=g, device="cuda").bfloat16().view(rows, HEADS, DIM)
    hidden = torch.randn(rows, 3040, generator=g, device="cuda").bfloat16()
    k, w = hidden[:, 2112:2240], hidden[:, 2240:2272]
    ln_w = 1 + 0.1 * torch.randn(DIM, generator=g, device="cuda")
    ln_b = 0.05 * torch.randn(DIM, generator=g, device="cuda")
    cache = torch.zeros(BLOCKS, PAGE, 1, DIM + 4, dtype=torch.uint8, device="cuda")
    where = torch.randperm(BLOCKS * PAGE, generator=g, device="cuda")[:rows]
    block, pos = where // PAGE, where % PAGE
    slot_fp8 = block * PAGE_BYTES + pos * DIM
    slot_scale = block * PAGE_BYTES + PAGE * DIM + pos * 4
    q_out = torch.empty(rows, HEADS, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    w_out = torch.full((rows, HEADS), float("nan"), device="cuda")
    return (_runner(), k, q, w, ln_w, ln_b, 1e-6, WEIGHT_SCALE, q_out, w_out, cache, slot_fp8,
            slot_scale, spec["n_write"])  # fmt: skip


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    _, k, q, w, ln_w, ln_b, eps, ws, q_out, w_out, cache, slot_fp8, slot_scale, n_write = args
    before = cache.clone()
    with validating(op):
        op(*args)
    torch.cuda.synchronize()
    want_q, want_w, keys = op.reference(*args)
    assert torch.equal(q_out.view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(w_out, want_w)
    # Keys: one E4M3 step (2^-3 relative) of the BF16 LayerNorm, itself within 2^-8 of the FP64
    # LayerNorm (rsqrt.approx), plus the subnormal floor (scale * 2^-9).
    codes, scales = cache_rows(cache, slot_fp8, slot_scale, n_write)
    if n_write:
        deq = codes.double() * scales.double().unsqueeze(1)
        band = (2.0**-3 + 2.0**-7) * keys.abs() + scales.double().unsqueeze(1) * 2.0**-9
        assert bool(((deq - keys).abs() <= band).all())
    written = torch.zeros_like(cache.view(-1), dtype=torch.bool)
    for s, width in ((slot_fp8, DIM), (slot_scale, 4)):
        for t in range(n_write):
            written[int(s[t]) : int(s[t]) + width] = True
    assert torch.equal(cache.view(-1)[~written], before.view(-1)[~written])


def test_is_valid_refuses_what_the_kernel_would_read_wrongly() -> None:
    args = list(_build(CELLS[0].spec, seed=1))
    short = list(args)
    short[9] = args[9][:4]  # w_out of 4 rows
    with pytest.raises(ValueError, match="every row"):
        op.is_valid(*short)
    far = list(args)
    far[11] = args[11].clone()
    far[11][0] = args[10].numel()
    with pytest.raises(ValueError, match="past the cache"):
        op.is_valid(*far)
