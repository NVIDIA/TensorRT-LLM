# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 reset_recurrent_state_rows entry: every pool row, bit for bit, against the
reference on a same-layout copy."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.reset_recurrent_state_rows import (
    reset_recurrent_state_rows,
)

from ._kda_ops_util import (
    DTYPES,
    HEAD_DIM,
    bits,
    clone_strided,
    generator,
    padded_slots,
    requires_sm103,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

#: A boundary and a typical cell of ``reset_recurrent_state_rows.CELLS``.
_CELLS = sample_cells(reset_recurrent_state_rows.CELLS, size_key="pool_slots")[:2]


def _pools(spec, gen):
    heads, slots = spec["heads"], spec["pool_slots"]
    state = padded_slots(slots, (heads, HEAD_DIM, HEAD_DIM), DTYPES[spec["state_dtype"]])
    state.copy_(torch.randn(state.shape, generator=gen, device="cuda"))
    conv = None
    if spec["with_conv"]:
        conv = padded_slots(slots, (3 * heads * HEAD_DIM, 3), torch.bfloat16)
        conv.copy_(torch.randn(conv.shape, generator=gen, device="cuda"))
    return state, conv


@pytest.mark.parametrize(
    "cell",
    _CELLS,
    ids=[c.why[:60] for c in _CELLS],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(reset_recurrent_state_rows)
    spec = cell.spec
    state, conv = _pools(spec, generator(cell))
    indices = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    has_init = torch.tensor([not f for f in spec["fresh"]], dtype=torch.bool, device="cuda")
    ref_state = clone_strided(state)
    ref_conv = clone_strided(conv) if conv is not None else None
    reset_recurrent_state_rows(state, indices, has_init, conv)
    reset_recurrent_state_rows.reference(ref_state, indices, has_init, ref_conv)
    assert torch.equal(bits(state), bits(ref_state)), "recurrent pool differs"
    if conv is not None:
        assert torch.equal(bits(conv), bits(ref_conv)), "convolution pool differs"
