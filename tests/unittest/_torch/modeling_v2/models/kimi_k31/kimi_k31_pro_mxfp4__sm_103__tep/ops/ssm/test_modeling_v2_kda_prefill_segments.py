# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_prefill_segments entry on a sample of its cells (``sample_cells``).

Cells are built like the kda_prefill cells (packed raw inputs, strided beta, padded-slot V-first pool, planned
metadata). Output and the selected state rows are checked against the fp32 token-by-token recurrence with
kda_prefill's bands; every other pool row must be untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_prefill_segments import (  # noqa: E501
    k31_kda_prefill_segments as op,
)

from ....ops.ssm._kda_ops_util import (
    assert_scaled_close,
    bits,
    clone_strided,
    generator,
    requires_sm103,
)
from ....ops.ssm.test_modeling_v2_kda_prefill import POOL_SLOTS, TOL_OUT, TOL_STATE, _inputs

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103


CELLS = sample_cells(op.CELLS)[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:60] for c in CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _inputs(cell.spec, generator(cell))
    pool = args["state_pool"]
    ref_args = dict(args, state_pool=clone_strided(pool))
    out = op(**args)
    expected = op.reference(**ref_args)
    assert out.shape == args["v"].shape and out.dtype == torch.bfloat16
    assert_scaled_close(out, expected, TOL_OUT, "output")
    selected = set(cell.spec["slots"])
    for slot in range(POOL_SLOTS):
        got = pool[slot]
        if slot in selected:
            assert_scaled_close(got, ref_args["state_pool"][slot], TOL_STATE, f"state slot {slot}")
        else:
            assert torch.equal(bits(got), bits(ref_args["state_pool"][slot])), (
                f"slot {slot} changed"
            )
