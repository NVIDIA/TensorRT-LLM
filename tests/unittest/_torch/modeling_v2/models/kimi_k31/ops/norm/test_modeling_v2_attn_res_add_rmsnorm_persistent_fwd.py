# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_add_rmsnorm_persistent_fwd entry (engine persistent attn_res op)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm_persistent_fwd import (  # noqa: E501
    attn_res_add_rmsnorm_persistent_fwd as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

from ._k31_attn_res_inputs import OUT_EPS, RMS_EPS, assert_mix_close, mix_inputs  # noqa: E402

assert torch.cuda.is_available(), "the persistent attn_res op requires a CUDA device"
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("the engine attn_res ops run on the sm_100 family", allow_module_level=True)

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] + str(c.spec) for c in CELLS])
def test_certified_cells(cell) -> None:
    prefix, addend, block, weights = mix_inputs(
        cell.spec, seed=zlib.crc32(repr(cell.spec).encode()) % 2**31
    )
    args = (prefix, addend if cell.spec["add"] else None, block, *weights, RMS_EPS, OUT_EPS)
    with validating(op):
        updated, out = op(*args)
    ref_updated, ref_out = op.reference(*args)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    # Each element within one BF16 step per rounding; no bound on how many differ.
    assert_mix_close(out, ref_out, weights[2], max_fraction=None)
