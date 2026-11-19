# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_rmsnorm and attn_res_add_rmsnorm entries: a sample of each entry's
certified cells against its reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm import (  # noqa: E501
    k31_attn_res_add_rmsnorm as add_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm import (  # noqa: E501
    k31_attn_res_rmsnorm as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

from ._k31_attn_res_inputs import OUT_EPS, RMS_EPS, assert_mix_close, mix_inputs  # noqa: E402

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res entries run a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "the attn_res entries require a CUDA device"

# A sample of attn_res_rmsnorm's cells and of attn_res_add_rmsnorm's.
CELLS = sample_cells(op.CELLS, size_key="rows")[:2]
ADD_CELLS = sample_cells(add_op.CELLS, size_key="rows")[:2]


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
