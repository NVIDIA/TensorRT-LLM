# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_pre_proj_split entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "cute_dsl_indexer_pre_proj_split runs an sm_103 CuTe DSL kernel", allow_module_level=True
    )

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_indexer_pre_proj_split import (  # noqa: E402, E501
    PREFILL_MAX_ROWS,
    PrefillSplitHandle,
    build_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_indexer_pre_proj_split import (  # noqa: E402, E501
    compile as compile_handle,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_indexer_pre_proj_split import (  # noqa: E402, E501
    cute_dsl_indexer_pre_proj_split as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, KEY, HEADS = 6144, 128, 32
#: One unit of fp32's mantissa.
FP32_ULP = 2.0**-23


def _compare(key, head_weights, ref_key, ref_head_weights) -> None:
    """Products of bf16 operands are exact in fp32; kernel and reference
    differ only in how K = 6144 of them are summed (fp32 over an 8-way
    split-K cluster, or the prefill GEMM's single fp32 accumulator, against
    fp64). The prefill path stores fp32 and rounds the keys to bf16 in a copy:
    the same one rounding.

    key, rounded once to bf16 on both sides: 4 ulp element-wise and 2 ulp
    RMS, as for every bf16 projection in this target.

    head_weights, fp32: a random-walk estimate of the reordering is a few
    fp32 ulps of the row scale. The band is 512 ulp (2**-14) element-wise and
    128 ulp (2**-16) RMS, wide enough for an accumulator that truncates
    instead of rounding; a wrong column or row is O(1).
    """
    assert_within_ulp(key, ref_key, element_ulp=4.0, rms_ulp=2.0)
    assert_within_ulp(head_weights, ref_head_weights, 512.0, 128.0, ulp=FP32_ULP)


def _build(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    weight = torch.randn(KEY + HEADS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5
    # NaN, so an element the kernel never writes fails the comparison.
    key = torch.full((rows, KEY), float("nan"), dtype=torch.bfloat16, device="cuda")
    head_weights = torch.full((rows, HEADS), float("nan"), dtype=torch.float32, device="cuda")
    return x, weight.to(torch.bfloat16), key, head_weights


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    rows = cell.spec["rows"]
    x, weight, key, head_weights = _build(rows, zlib.crc32(cell.why.encode()) % 2**31)
    if cell.spec.get("prefill"):
        # Built on another weight: the target builds once and runs every layer through it.
        compiled = build_prefill(torch.zeros_like(weight))
    else:
        compiled = compile_handle(torch.cuda.current_device(), rows)
    with validating(op):
        got = op(x, weight, key, head_weights, compiled)
    assert got[0] is key and got[1] is head_weights, "the op writes both buffers and returns them"
    _compare(key, head_weights, *op.reference(x, weight, key, head_weights, compiled))


def test_a_handle_refuses_another_row_count() -> None:
    """Each handle is compiled for one row count; the target picks it by the
    step's rows, so a mismatch must raise rather than compute a partial tile."""
    compiled = compile_handle(torch.cuda.current_device(), 2)
    with pytest.raises(ValueError, match="compiled for M=2"):
        op(*_build(4, seed=4), compiled)


def test_guard_refuses_bf16_head_weights_on_the_prefill_handle() -> None:
    """The prefill path copies the fp32 product into the buffers, and a copy casts:
    bf16 head weights would round silently where the scorers expect fp32."""
    x, weight, key, head_weights = _build(80, seed=5)
    handle = PrefillSplitHandle(PREFILL_MAX_ROWS, None)
    with pytest.raises(AssertionError, match="head_weights must be contiguous fp32"):
        with validating(op):
            op(x, weight, key, head_weights.to(torch.bfloat16), handle)
