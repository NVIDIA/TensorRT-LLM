# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's own attention entry `csa2_q_norm` (the query latent's RMS norm, bit for bit stock's FULL-layer
norm). The bf16 projections and the consumers' schedule are the family's entries (`ops/attention/`), tested there.

Band: within one bf16 rounding of the float64 reference (element_ulp 2 in units of the row max's top-of-binade spacing,
well under one ulp RMS).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.attention.csa2_q_norm import (  # noqa: E402,E501
    csa2_q_norm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize("cell", csa2_q_norm.CELLS, ids=[c.why[:48] for c in csa2_q_norm.CELLS])
def test_q_norm_cells(cell, certifying):  # noqa: F811
    """The normalized latent within one bf16 rounding of the reference, bit for bit stock's FULL-layer norm
    (`rms_norm_mxfp8(quantize=False)`)."""
    certifying(csa2_q_norm)
    s, gen = cell.spec, gen_for(cell)
    x = (torch.randn(s["n"], s["k"], generator=gen, device=DEV) * 3).bfloat16()
    w = (1 + 0.1 * torch.randn(s["k"], generator=gen, device=DEV)).bfloat16()
    with validating(csa2_q_norm):
        got = csa2_q_norm(x, w, 1e-6)
    ref = csa2_q_norm.reference(x, w, 1e-6)
    assert_within_ulp(got.float(), ref.float(), element_ulp=2.0, rms_ulp=0.5, ulp=2**-8)
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

    assert torch.equal(got, rms_norm_mxfp8(x, w, 1e-6, quantize=False)[0])
