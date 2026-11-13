# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attn_res_add_rmsnorm_fwd family op.

A cell is (T, N): T tokens and N = K + 1 candidates, K snapshots plus the updated layer residual. Every cell checks
both outputs' shapes and dtypes, the bf16 sum bit for bit, bit-identical results from two identical calls, untouched
inputs, and closeness of the normalized output to an fp32 torch evaluation under the metric of
test_modeling_v2_attn_res_rmsnorm_fwd.py.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_add_rmsnorm_fwd import (
    attn_res_add_rmsnorm_fwd,
)

__extra_import_path__ = [".", "../../../.."]
from _native_attn_res_inputs import HIDDEN_SIZE, bits, make_inputs, similarity
from _validating import validating

assert torch.cuda.is_available(), "attn_res_add_rmsnorm_fwd requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 5e-3
INPUT_NAMES = (
    "layer_residual",
    "addend",
    "block_residual",
    "res_weight",
    "rms_weight",
    "output_rms_weight",
)


@pytest.mark.parametrize(
    "cell", attn_res_add_rmsnorm_fwd.CELLS, ids=[c.why[:40] for c in attn_res_add_rmsnorm_fwd.CELLS]
)
def test_certified_cells(cell) -> None:
    x = make_inputs(cell)
    inputs = tuple(x[name] for name in INPUT_NAMES)
    before = [t.clone() for t in inputs]
    eps = (cell.spec["eps"], cell.spec["output_eps"])
    with validating(attn_res_add_rmsnorm_fwd):
        updated, output = attn_res_add_rmsnorm_fwd(*inputs, *eps)
        updated_again, output_again = attn_res_add_rmsnorm_fwd(*inputs, *eps)
    expected_updated, expected = attn_res_add_rmsnorm_fwd.reference(*inputs, *eps)

    shape = (cell.spec["tokens"], 1, HIDDEN_SIZE)
    for name, t in (("updated", updated), ("output", output)):
        assert t.shape == shape and t.dtype == torch.bfloat16, f"{cell.why}: {name}"
        assert t.is_contiguous() and t.device == inputs[0].device, f"{cell.why}: {name}"
    assert torch.equal(bits(updated), bits(expected_updated)), f"{cell.why}: the bf16 sum differs"
    assert torch.equal(bits(updated), bits(updated_again)), f"{cell.why}: two calls disagree"
    assert torch.equal(bits(output), bits(output_again)), f"{cell.why}: two calls disagree"
    cosine, relative_l2 = similarity(output, expected)
    assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2, (
        f"{cell.why}: cosine {cosine:.6f}, relative L2 {relative_l2:.3e}"
    )
    for name, old, new in zip(INPUT_NAMES, before, inputs):
        assert torch.equal(bits(old), bits(new)), f"{cell.why}: {name} was mutated"
