# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU checks of the weight side of ``rubin_mxfp8_linear`` on sm_107.

A stacked KDA projection gives every projection its columns and zero padding columns; the direct
K32 relay gives the bytes of the engine's ``block_scale_interleave`` and of the K128 transform
the engine's MXFP8 weights took.
"""

import pytest
import torch
from _rubin_gemm_util import (
    activation,
    generator,
    mxfp8_weight_scales,
    needs_kernel,
    requires_sm107,
    stack_mxfp8_weights,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
    quantize_mxfp8_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    rubin_mxfp8_linear as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = requires_sm107
needs_a1 = needs_kernel("rubin_dense_blockscaled_gemm")


def _checkpoint_weight(rows: int, k: int, gen: torch.Generator):
    """E4M3 values and row-major K32 scales of a ``[rows, k]`` weight of row norm ~1."""
    w = (torch.randn(rows, k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)
    return quantize_mxfp8_reference(w)


@needs_a1
@pytest.mark.parametrize("tp", [8, 1], ids=["tp8", "dep16"])
def test_a_stacked_kda_projection(tp) -> None:
    """[q | k | v | g] of this rank's heads, f_a (128 rows) and b (one row per head)."""
    heads = 96 // tp
    gen = generator(f"kda stack tp{tp}")
    parts = [_checkpoint_weight(n, 7168, gen) for n in (4 * heads * 128, 128, heads)]
    weight, weight_scale = stack_mxfp8_weights([(v, mxfp8_weight_scales(s)) for v, s in parts])
    x = activation(96, 7168, gen)
    with validating(op):
        out = op(x, weight, weight_scale)
    # The cells' band: exact FP32 terms summed in another order, one BF16 rounding.
    assert_within_ulp(out, op.reference(x, weight, weight_scale), element_ulp=2.0, rms_ulp=0.5)
    real = sum(v.shape[0] for v, _ in parts)
    assert torch.count_nonzero(out[:, real:]) == 0, "zero rows give zero columns"


def test_the_direct_k32_relay_gives_the_engines_bytes() -> None:
    """The engine's interleave of the same K32 bytes, and its K128 transform of K128 scales."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
    from tensorrt_llm.quantization.utils.fp8_utils import (
        transform_k128_scales_to_cutedsl_mxfp8_layout,
    )

    gen = generator("k32 relay")
    for rows, k in ((6288, 7168), (12, 7168), (7168, 1536), (1536, 7168), (7168, 768)):
        scales = torch.randint(0, 255, (rows, k // 32), generator=gen, device="cuda")
        scales = scales.to(torch.uint8)
        engine = torch.ops.trtllm.block_scale_interleave(scales)
        assert torch.equal(mxfp8_weight_scales(scales), engine.reshape(-1)), (rows, k)
        # One FP32 power of two per 128 x 128 block; the K128 transform repeats it to K32.
        exponents = torch.randint(-20, 5, (-(-rows // 128), k // 128), generator=gen, device="cuda")
        k128 = torch.exp2(exponents.float())
        old = transform_k128_scales_to_cutedsl_mxfp8_layout(k128, rows, k)
        k32 = (k128.view(torch.int32) >> 23).to(torch.uint8)
        k32 = k32.repeat_interleave(128, dim=0)[:rows].repeat_interleave(4, dim=1)
        assert torch.equal(mxfp8_weight_scales(k32), old.reshape(-1)), (rows, k)
