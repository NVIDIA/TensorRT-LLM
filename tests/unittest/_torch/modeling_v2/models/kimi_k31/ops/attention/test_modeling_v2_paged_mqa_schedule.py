# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 paged_mqa_schedule entry: integers, compared exactly with the
reference."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.paged_mqa_schedule import (  # noqa: E501
    paged_mqa_schedule as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_topk_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = inputs.schedule_args(cell.spec, inputs.seed_of(cell))
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref), f"schedule differs:\n{out.tolist()}\nexpected\n{ref.tolist()}"
