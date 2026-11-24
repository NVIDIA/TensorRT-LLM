# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_index_front entry: the query rows against the FP32 reference, the
FP8 queries and head weights bitwise the quantization of the kernel's own rows, the written keys
within the FP8 band of the FP64 LayerNorm, and nothing written past ``n_write``. Rows 17-32 run
Rubin's row (``RUBIN_TACTICS_FRONT``, two 16-token tiles) on every device: GB300's table has none."""

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
    quantize_queries,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_index_front import (  # noqa: E501
    dsa_index_front as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the tcgen05 front needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

WEIGHT_SCALE = 128**-0.5 * 32**-0.5
BLOCKS = 40
_FRONTS: dict = {}


def _front():
    """One layer's front for every cell's row count (the target builds it per layer at load)."""
    if "front" not in _FRONTS:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import (  # noqa: E501  # noqa: E501
            RUBIN_TACTICS_FRONT,
            TACTICS,
            IndexFront,
            NormWeight,
            _rubin_rows,
            tactic_for,
        )

        g = torch.Generator(device="cuda").manual_seed(4096)
        weight = (torch.randn(HEADS * DIM, 1536, generator=g, device="cuda") * 0.03).bfloat16()
        gamma = (1 + 0.1 * torch.randn(1536, generator=g, device="cuda")).bfloat16()
        ln_w = 1 + 0.1 * torch.randn(DIM, generator=g, device="cuda")
        ln_b = 0.05 * torch.randn(DIM, generator=g, device="cuda")
        rows = sorted({c.spec["rows"] for c in CELLS})
        table = RUBIN_TACTICS_FRONT if _rubin_rows(weight.device) else TACTICS
        # The device's row where it has one, else Rubin's (rows 17-32 exist only in RUBIN_TACTICS_FRONT).
        tactics = {r: tactic_for(r, table) or tactic_for(r, RUBIN_TACTICS_FRONT) for r in rows}
        _FRONTS["front"] = IndexFront(
            weight, NormWeight(gamma, 1e-6), ln_w, ln_b, 1e-6, rows, tactics=tactics
        )
    return _FRONTS["front"]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["rows"]
    # The hidden projection row: q_a 1536 | kv 512 | pe 64 | index k 128 | index w 32 | gate 768.
    hidden = torch.randn(rows, 3040, generator=g, device="cuda").bfloat16()
    qa, ik, iw = hidden[:, :1536], hidden[:, 2112:2240], hidden[:, 2240:2272]
    cache = torch.zeros(BLOCKS, PAGE, 1, DIM + 4, dtype=torch.uint8, device="cuda")
    where = torch.randperm(BLOCKS * PAGE, generator=g, device="cuda")[:rows]
    block, pos = where // PAGE, where % PAGE
    slot_fp8 = block * PAGE_BYTES + pos * DIM
    slot_scale = block * PAGE_BYTES + PAGE * DIM + pos * 4
    n_write = spec["n_write"]
    return (qa, ik, iw, WEIGHT_SCALE, cache, slot_fp8, slot_scale, n_write, None)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    front = _front()
    qa, ik, iw, ws, cache, slot_fp8, slot_scale, n_write, _ = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    q_rows = torch.full((qa.shape[0], HEADS * DIM), float("nan"), dtype=torch.bfloat16,
                        device="cuda")  # fmt: skip
    args = (front, qa, ik, iw, ws, cache, slot_fp8, slot_scale, n_write, q_rows)
    before = cache.clone()
    with validating(op):
        q_fp8, index_w = op(*args)
    torch.cuda.synchronize()
    _, _, ref_rows, keys = op.reference(*args)
    # The query rows: BF16 products of the BF16-rounded normed rows, summed in FP32 over a K split:
    # within a BF16 ulp of the reference except where the RMSNorm's own rounding moved an operand
    # by one BF16 ulp (rsqrt.approx); 4 ulp of the row maximum covers both.
    assert_within_ulp(q_rows, ref_rows, element_ulp=4.0, rms_ulp=1.0)
    # The epilogue quantizes the rows it stored: bitwise the exact quantization of those rows.
    want_q, want_w = quantize_queries(q_rows, iw, ws)
    assert torch.equal(q_fp8.view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(index_w, want_w)
    # Keys: the BF16 LayerNorm then E4M3 at a UE8M0 scale; the dequantized code is within one E4M3
    # step (2^-3 relative) of the BF16 value, itself within 2^-8 of the FP64 LayerNorm, plus the
    # subnormal floor (scale * 2^-9).
    codes, scales = cache_rows(cache, slot_fp8, slot_scale, n_write)
    if n_write:
        deq = codes.double() * scales.double().unsqueeze(1)
        band = (2.0**-3 + 2.0**-7) * keys.abs() + scales.double().unsqueeze(1) * 2.0**-9
        assert bool(((deq - keys).abs() <= band).all())
    # Nothing else in the cache moved.
    written = torch.zeros_like(cache.view(-1), dtype=torch.bool)
    for s, width in ((slot_fp8, DIM), (slot_scale, 4)):
        for t in range(n_write):
            written[int(s[t]) : int(s[t]) + width] = True
    assert torch.equal(cache.view(-1)[~written], before.view(-1)[~written])


def test_is_valid_refuses_what_the_kernel_would_read_wrongly() -> None:
    front = _front()
    qa, ik, iw, ws, cache, slot_fp8, slot_scale, _, _ = _build(CELLS[0].spec, seed=1)
    with pytest.raises(ValueError, match="past the step"):
        op.is_valid(front, qa, ik, iw, ws, cache, slot_fp8, slot_scale, qa.shape[0] + 1)
    far = slot_fp8.clone()
    far[0] = cache.numel()
    with pytest.raises(ValueError, match="past the cache"):
        op.is_valid(front, qa, ik, iw, ws, cache, far, slot_scale, 1)
