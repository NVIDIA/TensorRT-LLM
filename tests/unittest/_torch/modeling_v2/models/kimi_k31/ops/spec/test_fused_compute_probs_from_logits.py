# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_compute_probs_from_logits entry.

Bounded per row in L1 rather than elementwise: a token sitting exactly on a top-p cut (or tied
at a top-k cut) may land on either side in the kernel and in the reference, which moves that
token's mass and nothing else; a wrong filter moves O(0.1).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    fused_compute_probs_from_logits as op,
)

__extra_import_path__ = ["../../../..", "."]
from _k31_spec_inputs import on_sm103, row_l1, sampling_inputs, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

#: Measured on GB300 over the fused and flashinfer cells: max row L1 1.3e-5 (fp32 softmax over
#: 163840 entries, the kernel's and torch's reductions in different orders, renormalized up to
#: three times); 5e-5 leaves room for one boundary flip of a ~2e-5 token besides.
_ROW_L1 = 5e-5


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = sampling_inputs(cell.spec, seed_of(cell), fused=True)
    probs = op(*args)
    expected = op.reference(*args)
    assert probs.shape == expected.shape and probs.dtype == torch.float32
    l1 = row_l1(probs, expected)
    assert l1.max().item() <= _ROW_L1, f"max row L1 {l1.max().item():.3e}"
    torch.testing.assert_close(
        probs.sum(-1), torch.ones(probs.shape[0], device="cuda"), atol=1e-5, rtol=0
    )
    greedy = args[1] <= 1e-4
    assert torch.equal(probs[greedy].argmax(-1), args[0][greedy].float().argmax(-1))
