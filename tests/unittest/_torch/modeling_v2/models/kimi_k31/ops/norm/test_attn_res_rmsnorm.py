# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_rmsnorm and attn_res_add_rmsnorm entries."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm._attn_res_common import (  # noqa: E501
    HIDDEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm import (  # noqa: E501
    k31_attn_res_add_rmsnorm as add_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm import (  # noqa: E501
    k31_attn_res_rmsnorm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm import (  # noqa: E501
    set_early_trigger_max_rows,
    set_reg_cap_min_rows,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_attn_res_inputs import OUT_EPS, RMS_EPS, assert_mix_close, mix_inputs  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res entries run a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "the attn_res entries require a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
# attn_res_rmsnorm's cells (and, below, attn_res_add_rmsnorm's).
CELLS = certified_cells(op, TARGET_CELLS)
ADD_CELLS = certified_cells(add_op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    layer, _, block, weights = mix_inputs(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    args = (layer, block, *weights, RMS_EPS, OUT_EPS)
    with validating(op):
        out = op(*args)
    assert_mix_close(out, op.reference(*args), weights[2])


@pytest.mark.parametrize("cell", ADD_CELLS, ids=[c.why[:44] for c in ADD_CELLS])
def test_add_certified_cells(cell) -> None:
    prefix, addend, block, weights = mix_inputs(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**30
    )
    args = (prefix, addend, block, *weights, RMS_EPS, OUT_EPS)
    with validating(add_op):
        updated, out = add_op(*args)
    ref_updated, ref_out = add_op.reference(*args)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    assert_mix_close(out, ref_out, weights[2])


@pytest.mark.parametrize("rows, candidates", [(8, 9), (65, 5), (120, 7), (191, 9)])
def test_row_limits_keep_the_bits(rows, candidates) -> None:
    """The attention-TP target's process-wide limits (early trigger up to 191 rows, register-capped builds from 65
    rows) change the launch, not the result: the mixes give the bits of the default builds."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm import attn_res

    prefix, addend, block, weights = mix_inputs(
        dict(rows=rows, candidates=candidates), seed=rows * 16 + candidates
    )
    default = (attn_res._EARLY_TRIGGER_MAX_M, attn_res.REG_CAP_MIN_TOKENS)
    out = op(prefix, block, *weights, RMS_EPS, OUT_EPS)
    updated, add_out = add_op(prefix, addend, block, *weights, RMS_EPS, OUT_EPS)
    try:
        set_early_trigger_max_rows(191)
        set_reg_cap_min_rows(65)
        assert (attn_res._EARLY_TRIGGER_MAX_M, attn_res.REG_CAP_MIN_TOKENS) == (191, 65)
        limited = op(prefix, block, *weights, RMS_EPS, OUT_EPS)
        limited_updated, limited_add_out = add_op(prefix, addend, block, *weights, RMS_EPS, OUT_EPS)
    finally:
        set_early_trigger_max_rows(default[0])
        set_reg_cap_min_rows(default[1])
    assert torch.equal(limited.view(torch.int16), out.view(torch.int16))
    assert torch.equal(limited_updated.view(torch.int16), updated.view(torch.int16))
    assert torch.equal(limited_add_out.view(torch.int16), add_out.view(torch.int16))


def test_guard_refuses_another_hidden_size() -> None:
    """The kernel is built for 7168 columns and would read any other width as 7168."""
    layer = torch.randn(4, 4096, device="cuda").to(torch.bfloat16)
    block = torch.randn(2, 4, 4096, device="cuda").to(torch.bfloat16)
    w = torch.randn(4096, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError, match="layer must be"):
        with validating(op):
            op(layer, block, w, w, w, RMS_EPS, OUT_EPS)


def test_guard_refuses_too_many_candidates() -> None:
    layer, _, block, weights = mix_inputs(dict(rows=2, candidates=12), seed=3)
    block = torch.cat([block, block[:1]], dim=0)  # 13 candidates
    with pytest.raises(AssertionError, match="candidates"):
        with validating(op):
            op(layer, block, *weights, RMS_EPS, OUT_EPS)


def test_guard_refuses_a_strided_addend() -> None:
    prefix, _, block, weights = mix_inputs(dict(rows=4, candidates=3), seed=4)
    addend = torch.randn(4, 2 * HIDDEN, device="cuda").to(torch.bfloat16)[:, ::2]
    with pytest.raises(AssertionError, match="contiguous BF16"):
        with validating(add_op):
            add_op(prefix, addend, block, *weights, RMS_EPS, OUT_EPS)
