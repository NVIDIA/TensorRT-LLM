# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_verify_rec entry of the Kimi K3.1 family: a sample of its cells (``sample_cells``), over two
chained rounds (nothing pending, then random accepted drafts), against the entry's FP64 reference, through the entry's
guard."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_verify_rec import (  # noqa: E501
    k31_kda_verify_rec as op,
)

__extra_import_path__ = ["../../../../..", "../../..", "../../../kernels/ssm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _kda_rec_layer import RecLayer, check_against_reference, metadata  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("kda_verify_rec runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "kda_verify_rec requires a CUDA device"

# The target's TEP generation-server verify is the only caller; its shapes are the entry's own cells.
TARGET_CELLS: dict = {}
CELLS = certified_cells(op, TARGET_CELLS)
SAMPLE = sample_cells(CELLS, size_key="requests")[:2]


@torch.no_grad()
@pytest.mark.parametrize("cell", SAMPLE, ids=[c.why[:44] for c in SAMPLE])
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
            got = op(**args)
        torch.cuda.synchronize()
        assert got.data_ptr() == layer.out.data_ptr(), "the op must return the caller's out"
        check_against_reference(layer, slot_idx, ref)
