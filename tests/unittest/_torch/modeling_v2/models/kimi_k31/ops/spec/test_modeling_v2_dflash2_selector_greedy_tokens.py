# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_dflash2_selector_greedy_tokens entry: a sample of its cells bitwise against the argmax
of the k31_dflash2_selector_greedy_path entry's reference block (the selector's edge scores and greedy
walk)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_tokens import (  # noqa: E501
    k31_dflash2_selector_greedy_tokens as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import selector_walk_inputs  # noqa: E402
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(103, reason="the K3.1 DFlash2 selector entry is certified on sm_103")


CELLS = sample_cells(op.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = selector_walk_inputs(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31, block_tensor=False
    )
    certifying(op)
    assert torch.equal(op(*args), op.reference(*args))
