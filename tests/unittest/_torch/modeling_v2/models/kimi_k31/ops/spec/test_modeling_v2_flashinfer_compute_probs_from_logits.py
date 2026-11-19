# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 flashinfer_compute_probs_from_logits entry (bounded per row in L1,
see the fused entry's test)."""

import pytest

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.flashinfer_compute_probs_from_logits import (  # noqa: E501
    flashinfer_compute_probs_from_logits as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_spec_inputs import on_sm103, row_l1, sampling_inputs, seed_of  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

#: The fused entry's band and its derivation (same reference, same cells' shapes).
_ROW_L1 = 5e-5


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    logits, temperatures, top_k, top_p, _ = sampling_inputs(cell.spec, seed_of(cell), fused=False)
    probs = op(logits, temperatures, top_k, top_p)
    l1 = row_l1(probs, op.reference(logits, temperatures, top_k, top_p))
    assert l1.max().item() <= _ROW_L1, f"max row L1 {l1.max().item():.3e}"
