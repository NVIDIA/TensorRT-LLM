# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The low-M FP8 GEMM's Lamport epilogue publish (``lamport=(slot_address, row_bytes)``).

The published words must be, bit for bit, the BF16 output the same GEMM stores without
``lamport`` (same tactic, so the same split-K order) with every ``-0`` turned into ``+0``, at
``slot + token * row_bytes + column * 2``; every other word of the region stays the sentinel
(armed), including the other ranks' column blocks and rows past M. Single GPU: any device address
works for the stores (the multicast address of a real mailbox is covered by the multi-GPU test).
"""

import pytest
import torch

from tensorrt_llm._utils import is_sm_100f

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() == 0 or not is_sm_100f(),
    reason="requires an SM100/SM103 GPU (tcgen05 block-scaled MMA)",
)

SENTINEL = -0x80000000
_SHAPES = [(7168, 1536), (7168, 3584)]
_ROWS = [1, 3, 8, 13, 16, 32]
WORLD = 8
RANK = 5


def _quantized_weight(n, k, generator):
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8
    from tensorrt_llm.quantization.utils.fp8_utils import (
        resmooth_to_fp8_e8m0,
        transform_sf_into_required_layout,
    )

    weight = (torch.randn(n, k, generator=generator, device="cuda") * 0.02).to(torch.bfloat16)
    weight_fp8, block_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
    weight_fp8, block_scale = resmooth_to_fp8_e8m0(
        weight_fp8.contiguous(), block_scale.contiguous().float()
    )
    packed = transform_sf_into_required_layout(
        block_scale.float(), mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )
    return weight_fp8.contiguous(), packed


@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("rows", _ROWS)
def test_lamport_publish_matches_store(n, k, rows):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import low_m_fp8_blockscaled_splitk as lowm

    generator = torch.Generator(device="cuda").manual_seed(99 + n + k + rows)
    weight, weight_scale = _quantized_weight(n, k, generator)
    x = (torch.randn(rows, k, generator=generator, device="cuda") * 4).to(torch.bfloat16)
    act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, False)
    row_words = WORLD * n // 2
    for tactic in lowm.candidate_tactics(rows, n, k):
        for pdl in (False, True):
            out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
            lowm.run_fp8_blockscaled_splitk(act, act_scale, weight, weight_scale, out, pdl, tactic)
            region = torch.full((rows + 2, row_words), SENTINEL, dtype=torch.int32, device="cuda")
            base = region[1 : rows + 1]
            slot = region.data_ptr() + RANK * n * 2 + region.stride(0) * 4
            dummy = torch.zeros(rows, n, dtype=torch.bfloat16, device="cuda")
            lowm.run_fp8_blockscaled_splitk(
                act,
                act_scale,
                weight,
                weight_scale,
                dummy,
                pdl,
                tactic,
                lamport=(slot, row_words * 4),
            )
            torch.cuda.synchronize()
            assert not dummy.any(), "lamport mode must not store to out"
            bits = out.view(torch.int16).clone()
            bits[bits == -0x8000] = 0
            expected = bits.view(torch.int32)  # [rows, n / 2] words (little endian pairs)
            got = base[:, RANK * n // 2 : (RANK + 1) * n // 2]
            assert torch.equal(got, expected), f"{tactic} pdl={pdl}: published words differ"
            mask = torch.ones_like(region, dtype=torch.bool)
            mask[1 : rows + 1, RANK * n // 2 : (RANK + 1) * n // 2] = False
            assert bool((region[mask] == SENTINEL).all()), f"{tactic}: wrote outside its slot"
