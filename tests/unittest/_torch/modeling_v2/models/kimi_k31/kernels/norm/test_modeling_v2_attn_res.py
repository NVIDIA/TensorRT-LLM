# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's PDL attn_res kernel against the engine's C++ attn_res op.

The engine op is what the general target runs today. The kernel's claims against its fused add +
RMSNorm: the updated prefix sum is bit-identical, and the normalized output differs by at most one
BF16 rounding step in a small fraction of the elements (the statistics are summed in another order).
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res kernel is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm import attn_res  # noqa: E402

__extra_import_path__ = ["../../ops/norm"]
from _k31_attn_res_inputs import (  # noqa: E402 — needs the path declared above
    OUT_EPS,
    RMS_EPS,
    assert_mix_close,
    mix_inputs,
)

assert torch.cuda.is_available(), "the attn_res kernel requires a CUDA device"
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
    pytest.skip("the attn_res kernel runs on SM100 / SM103 / SM107", allow_module_level=True)

H = attn_res.HIDDEN_SIZE


def _inputs(m: int, n: int, seed: int):
    """(layer / prefix, addend, block, res_w, rms_w, out_w) of ``m`` rows and ``n`` candidates."""
    layer, addend, block, weights = mix_inputs(dict(rows=m, candidates=n), seed)
    return layer, addend, block, *weights


def _engine_add_rmsnorm(prefix, addend, block, res_w, rms_w, out_w):
    m, k = prefix.shape[0], block.shape[0]
    updated, out = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(
        prefix.reshape(m, 1, H),
        addend.reshape(m, 1, H),
        block.reshape(k, m, 1, H),
        res_w,
        rms_w,
        out_w,
        RMS_EPS,
        OUT_EPS,
    )
    return updated.reshape(m, H), out.reshape(m, H)


# (rows, candidates): an 8-CTA cluster (<= 16 rows, >= 4 candidates) and the kernel's row limit
# (1024).
SHAPES = ((1, 12), (1024, 5))


@pytest.mark.parametrize("m,n", SHAPES)
def test_add_rmsnorm_vs_engine(m: int, n: int) -> None:
    prefix, addend, block, res_w, rms_w, out_w = _inputs(m, n, seed=m * 16 + n + 7)
    updated, out = attn_res.k31_attn_res_add_rmsnorm(
        prefix, addend, block, res_w, rms_w, out_w, RMS_EPS, OUT_EPS
    )
    ref_updated, ref_out = _engine_add_rmsnorm(prefix, addend, block, res_w, rms_w, out_w)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16)), (
        "the updated prefix sum must be bit-identical to the engine op's"
    )
    fraction = assert_mix_close(out, ref_out, out_w)
    print(
        f"add_rmsnorm m={m} n={n}: {fraction:.4%} of the outputs one BF16 step from the engine op"
    )
