# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one GPU) of the shared low-M / mid-M MXFP8 GEMM's row store, the epilogue
``rubin_mxfp8_linear_prequantized_push`` runs.

``fp8_block_scale_low_m_gemm.run_prequantized(..., row_store=...)`` hands every 16-byte vector of
the output rows to the row store instead of storing the output. Here the row store copies each
vector to its place in a BF16 ``[M, N]`` tensor, so the tensor must equal the plain call's output
bit for bit, at every row count of helix8's verify (6..126 rows: the low-M kernel at split-K 4 and
2, the mid-M kernel) and on shapes with the low-M kernel without split-K, the mid-M kernel with
split-K and a partial last weight tile. The rows past ``M`` are not written; a CUDA graph of the
row-store call replays to the same bytes.
"""

import dataclasses

import pytest
import torch
from _rubin_gemm_util import activation, generator, mxfp8_weight, requires_sm107

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    _low_m,
    low_m_gemm_max_m,
    mxfp8_act_layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear_push import (  # noqa: E501
    pushes_from_epilogue,
)

pytestmark = requires_sm107


@dataclasses.dataclass(frozen=True)
class _RowCopy:
    """A row store that writes each vector into the BF16 ``[M, n]`` tensor at ``desc``."""

    n: int

    def begin(self, desc):
        return ()

    def store(self, desc, state, token, feature, w0, w1, w2, w3):
        from cutlass import Int64

        from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as lowm

        lowm._stg_v4_b32(desc + Int64(token * self.n + feature) * 2, w0, w1, w2, w3)


def _quantize(x: torch.Tensor, sf_rows: int, layout: str):
    """The low-M GEMM's MXFP8 of BF16 rows, scales in ``layout`` over ``sf_rows`` rows (as
    test_rubin_mxfp8_linear_prequantized.py builds them)."""
    m, k = x.shape
    bits = x.view(torch.int16).to(torch.int32) & 0x7FFF
    amax = bits.view(m, k // 32, 32).amax(-1).clamp(min=0x38D2)
    exp = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    inv = ((254 - exp) << 7).to(torch.int16).view(torch.bfloat16)
    q = (x.view(m, k // 32, 32) * inv.unsqueeze(-1)).to(torch.float8_e4m3fn).view(m, k)
    rows = torch.full((sf_rows, k // 32), 127, dtype=torch.uint8, device=x.device)
    rows[:m] = exp.to(torch.uint8)
    tiles = rows.view(sf_rows // 128, 4, 32, k // 128, 4)
    order = (3, 0, 2, 1, 4) if layout == "mid" else (0, 3, 2, 1, 4)
    return q.contiguous(), tiles.permute(*order).reshape(-1).contiguous()


#: (M, N, K): helix8's o_proj at every verify row count, then other tactic kinds.
_SHAPES = (
    *((rows, 7168, 1536) for rows in range(6, 127, 6)),
    (1, 7168, 1536),
    (64, 7168, 1536),
    (65, 7168, 1536),
    (128, 7168, 1536),
    (24, 18432, 1536),  # low-M without split-K
    (96, 1536, 7168),  # mid-M with split-K
    (54, 6288, 7168),  # a partial last weight tile (the KDA stack)
    (126, 6288, 7168),
)


@pytest.mark.parametrize("m,n,k", _SHAPES, ids=[f"{m}x{n}x{k}" for m, n, k in _SHAPES])
def test_row_store_writes_the_output_bytes(m, n, k) -> None:
    low_m = _low_m()
    assert low_m is not None and m <= low_m_gemm_max_m(n, k)
    gen = generator(f"row store {m} {n} {k}")
    x = activation(m, k, gen)
    weight, weight_scale = mxfp8_weight(n, k, gen)
    act, act_scale = _quantize(x, *mxfp8_act_layout(m, n, k))
    assert pushes_from_epilogue(m, n, k, weight_scale)
    plain = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    low_m.run_prequantized(act, act_scale, weight, weight_scale, plain, act_r128c4=True)
    rows = torch.full((m + 2, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    store = _RowCopy(n)
    low_m.run_prequantized(
        act,
        act_scale,
        weight,
        weight_scale,
        None,
        act_r128c4=True,
        row_store=store,
        store_desc=rows,
    )
    torch.cuda.synchronize()
    assert torch.equal(rows[:m].view(torch.int16), plain.view(torch.int16))
    assert torch.isnan(rows[m:].float()).all(), "rows past M are not written"

    # One CUDA graph of the row-store call replays to the same bytes on new operands.
    x2 = activation(m, k, gen)
    act2, act_scale2 = _quantize(x2, *mxfp8_act_layout(m, n, k))
    live_act, live_scale = act.clone(), act_scale.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        low_m.run_prequantized(
            live_act,
            live_scale,
            weight,
            weight_scale,
            None,
            act_r128c4=True,
            row_store=store,
            store_desc=rows,
        )
    live_act.copy_(act2)
    live_scale.copy_(act_scale2)
    rows.fill_(float("nan"))
    graph.replay()
    low_m.run_prequantized(act2, act_scale2, weight, weight_scale, plain, act_r128c4=True)
    torch.cuda.synchronize()
    assert torch.equal(rows[:m].view(torch.int16), plain.view(torch.int16))


def test_the_store_needs_its_descriptor() -> None:
    gen = generator("row store args")
    weight, weight_scale = mxfp8_weight(7168, 1536, gen)
    act, act_scale = _quantize(activation(6, 1536, gen), *mxfp8_act_layout(6, 7168, 1536))
    out = torch.empty(6, 7168, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="row_store"):
        _low_m().run_prequantized(
            act, act_scale, weight, weight_scale, out, row_store=_RowCopy(7168)
        )
    with pytest.raises(ValueError, match="row_store"):
        _low_m().run_prequantized(act, act_scale, weight, weight_scale, None)
