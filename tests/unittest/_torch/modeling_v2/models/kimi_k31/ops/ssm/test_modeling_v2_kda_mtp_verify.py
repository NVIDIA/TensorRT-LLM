# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_mtp_verify entry of the Kimi K3.1 family.

A sample of the certified cells runs the op (through its guard) and the entry's plain-torch reference on
identical inputs and compares the returned core and every tensor the call writes in place.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_verify import (
    kda_mtp_verify as op,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _rubin_kda_verify_inputs import (  # noqa: E402 — needs the path declared above
    check_bf16_output,
    check_writes,
    kda_verify_args,
)
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "kda_mtp_verify requires a CUDA device"

# The target's fused verify is the only caller; its shapes are the entry's own cells.
TARGET_CELLS: dict = {}

CELLS = certified_cells(op, TARGET_CELLS)
#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(CELLS, size_key="N")[:2]


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:44] for c in _CELLS])
def test_certified_cells(cell) -> None:
    spec = dict(cell.spec)
    use_pdl = spec.pop("use_pdl", False)
    state_evict_last = spec.pop("state_evict_last", False)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    op_args = kda_verify_args(seed=seed, **spec)
    ref_args = kda_verify_args(seed=seed, **spec)
    untouched = [
        s
        for s in range(op_args["recurrent_state"].shape[0])
        if s not in op_args["ssm_state_indices"].tolist()
    ]
    before = op_args["recurrent_state"][untouched].clone()
    with validating(op):
        out = op(**op_args, use_pdl=use_pdl, state_evict_last=state_evict_last)
    ref = op.reference(**ref_args)
    torch.cuda.synchronize()
    check_bf16_output(out, ref)
    check_writes(op_args, ref_args, untouched, before)
