# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_compute_probs_split entry.

Every row must be bit for bit its own one-row sampler call (the multi-CTA family) on a sample of the entry's cells
(``sample_cells``).

The sampler's top-p cut is not run-to-run reproducible for a token sitting on it (its radix histogram sums masses
with float atomics), in either family. Random logits put a top-p 0.95 cut deep in a dense tail, so there the cells
compare top-p rows within one boundary token (row L1) and every other row bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.fused_compute_probs_split import (  # noqa: E501
    fused_compute_probs_split as op,
)

__extra_import_path__ = ["../../../../..", "../../..", "../../../ops/spec"]
from _k31_spec_inputs import on_sm103, row_l1, sampling_inputs, seed_of  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

#: One top-p boundary token of a random-logit row (see the fused_compute_probs_from_logits test).
_ROW_L1 = 5e-5


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[rows]`` bool: rows whose float32 words differ anywhere."""
    return (a.view(torch.int32) != b.view(torch.int32)).any(-1)


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = sampling_inputs(cell.spec, seed_of(cell), fused=True)
    got = op(*args)
    expected = op.reference(*args)
    top_p_rows = args[3][: args[0].shape[0]] < 1.0
    differ = _bits_differ(got, expected)
    strict = (differ & ~top_p_rows).nonzero().flatten()
    assert strict.numel() == 0, f"rows {strict.tolist()} differ from their own one-row call"
    l1 = row_l1(got, expected)[differ & top_p_rows]
    assert l1.numel() == 0 or l1.max().item() <= _ROW_L1, f"top-p row L1 {l1.max().item():.3e}"
