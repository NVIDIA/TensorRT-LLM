# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_verify_rec entry of the Kimi K3.1 family: every cell, over two chained rounds (nothing
pending, then random accepted drafts), against the entry's FP64 reference, through the entry's guard."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_verify_rec import (  # noqa: E501
    k31_kda_verify_rec as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_verify_rec import (  # noqa: E501
    supported,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../../..", "../../../kernels/ssm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _kda_rec_layer import RecLayer, check_against_reference, metadata  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("kda_verify_rec runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "kda_verify_rec requires a CUDA device"

# The target's TEP generation-server verify is the only caller; its shapes are the entry's own cells.
TARGET_CELLS: dict = {}
CELLS = certified_cells(op, TARGET_CELLS)


@torch.no_grad()
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    H, num_spec, B = cell.spec["heads"], cell.spec["num_spec"], cell.spec["requests"]
    slots = B + 4
    layer = RecLayer(H, B, num_spec, slots, seed=H * 1000 + num_spec * 100 + B)
    for rnd in range(2):
        slot_idx, num_acc = metadata(B, slots, num_spec, rnd, seed=B + rnd)
        layer.new_tokens(seed=10 * rnd + B)
        args = layer.args(slot_idx, num_acc)
        ref = op.reference(**args)
        with validating(op):
            got = op(**args, use_pdl=cell.spec.get("use_pdl", False))
        torch.cuda.synchronize()
        assert got.data_ptr() == layer.out.data_ptr(), "the op must return the caller's out"
        check_against_reference(layer, slot_idx, ref)


def test_guard_refuses_a_misaligned_state_row() -> None:
    """The state is read in 16-byte rows: a view whose row stride is not a multiple of 4 floats is read wrong."""
    H, num_spec, B, slots = 6, 4, 2, 5
    layer = RecLayer(H, B, num_spec, slots, seed=1)
    layer.new_tokens(seed=1)
    slot_idx, num_acc = metadata(B, slots, num_spec, 0, seed=1)
    args = layer.args(slot_idx, num_acc)
    padded = torch.zeros(slots, H, 128, 130, device="cuda")
    args["recurrent_state"] = padded[..., :128]
    with pytest.raises(AssertionError, match="16-byte aligned rows"):
        with validating(op):
            op(**args)


def test_guard_refuses_misaligned_projections() -> None:
    H, num_spec, B, slots = 6, 4, 2, 5
    layer = RecLayer(H, B, num_spec, slots, seed=2)
    layer.new_tokens(seed=2)
    slot_idx, num_acc = metadata(B, slots, num_spec, 0, seed=2)
    args = layer.args(slot_idx, num_acc)
    flat = torch.zeros(layer.T * H * 128 + 1, dtype=torch.bfloat16, device="cuda")
    args["x_q"] = flat[1:].view(1, layer.T, H, 128)
    with pytest.raises(AssertionError, match="x_q must be 16-byte aligned"):
        with validating(op):
            op(**args)


@torch.no_grad()
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_target_predicate_admits_every_cell(cell) -> None:
    """The target takes the entry only where ``supported`` holds for the layer: every cell must pass it, and a state
    with misaligned rows, more drafts than the kernel's or another conv width must not."""
    H, num_spec, B = cell.spec["heads"], cell.spec["num_spec"], cell.spec["requests"]
    layer = RecLayer(H, B, num_spec, B + 4, seed=H * 1000 + num_spec * 100 + B)
    assert supported(layer.state, num_spec, 128, 128, 4)
    assert not supported(layer.state, 8, 128, 128, 4)
    assert not supported(layer.state, num_spec, 128, 128, 3)
    padded = torch.zeros(B + 4, H, 128, 130, device="cuda")
    assert not supported(padded[..., :128], num_spec, 128, 128, 4)
