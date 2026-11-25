# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kernels/fused_out_a2a.py``: the Ulysses output exchange as NVFP4 codes and block scales.

One process stands in for the U ranks: each rank's head slice goes through ``quant_to_send``, the
chunks are permuted as ``all_to_all_single`` delivers them, and ``assemble_from_recv`` must give,
for every rank, the bytes ``trtllm::fp4_quantize`` produces on that rank's gathered BF16 rows.
The tiled ``fused_out_quant`` kernel on the gathered rows is checked as a second assertion.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  -- loads libth_common.so (registers the trtllm ops)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
    fused_out_a2a,
    fused_out_quant,
)

pytestmark = skip_pre_blackwell

WIDTH = 5120  # H * D of the gathered row; a head is 128 wide, so a 16-element block never straddles ranks


def _activation(rows: int, width: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, width), generator=gen, device="cuda", dtype=torch.float32)
    x[0, :16] *= 64.0
    x[0, 16:32] = 0.0
    x[rows - 1, width - 16 :] *= 32.0
    x[rows // 2] = 0.0
    return x.to(torch.bfloat16)


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.float().abs().max().clamp(min=1e-3)
    return (448.0 * 6.0 / amax).to(torch.float32)


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.dtype == torch.uint8 and expected.dtype == torch.uint8, what
    assert actual.numel() == expected.numel(), (
        f"{what}: {actual.numel()} vs {expected.numel()} bytes"
    )
    assert torch.equal(actual.reshape(-1), expected.reshape(-1)), what


@pytest.mark.parametrize("local_rows", (1017, 9455))
@pytest.mark.parametrize("ulysses", (2, 4))
def test_quantize_exchange_assemble_equals_fp4_quantize_of_gathered_rows(ulysses, local_rows):
    seq = local_rows * ulysses
    k_local = WIDTH // ulysses
    full = _activation(seq, WIDTH, seed=seq)  # [S, H * D]: what the BF16 exchange would gather
    gs = _global_scale(full)

    # Rank r holds the full sequence for its heads: columns [r * k_local, (r + 1) * k_local).
    sends = [
        fused_out_a2a.quant_to_send(full[:, r * k_local : (r + 1) * k_local].contiguous(), gs)
        for r in range(ulysses)
    ]
    for send in sends:
        assert send.shape == (seq, k_local // 2 + k_local // 16) and send.dtype == torch.uint8

    for u in range(ulysses):
        # all_to_all_single: rank u receives chunk u of every source rank, in source-rank order.
        recv = torch.cat([send[u * local_rows : (u + 1) * local_rows] for send in sends], dim=0)
        codes, scales = fused_out_a2a.assemble_from_recv(recv, ulysses, k_local)
        gathered = full[u * local_rows : (u + 1) * local_rows].contiguous()
        ref_codes, ref_scales = torch.ops.trtllm.fp4_quantize(gathered, gs, 16, False)
        assert codes.shape == (local_rows, WIDTH // 2)
        _assert_same_bytes(codes, ref_codes, f"rank {u}: packed e2m1 codes")
        _assert_same_bytes(scales, ref_scales, f"rank {u}: swizzled e4m3 block scales")
        tiled_codes, tiled_scales = fused_out_quant._launch(
            gathered, gs, rb=16, ch=512, num_warps=4
        )
        _assert_same_bytes(codes, tiled_codes, f"rank {u}: codes vs fused_out_quant")
        _assert_same_bytes(scales, tiled_scales, f"rank {u}: scales vs fused_out_quant")
