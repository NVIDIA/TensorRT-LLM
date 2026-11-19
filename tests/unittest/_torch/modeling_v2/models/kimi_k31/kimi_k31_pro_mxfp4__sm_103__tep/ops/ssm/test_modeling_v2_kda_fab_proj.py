# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kda_fab_proj entry of the Kimi K3.1 family: a sample of its cells (``sample_cells``) against the
true-FP32 reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_fab_proj import (  # noqa: E501
    FA_DIM,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_fab_proj import (  # noqa: E501
    k31_kda_fab_proj as op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("kda_fab_proj runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "kda_fab_proj requires a CUDA device"

# The target's TEP generation-server verify is the only caller; its shapes are the entry's own cells.
TARGET_CELLS: dict = {}
CELLS = certified_cells(op, TARGET_CELLS)
SAMPLE = sample_cells(CELLS, size_key="rows")[:2]


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    heads, rows, hidden = spec["heads"], spec["rows"], spec["hidden"]
    n1 = (FA_DIM + heads + 7) // 8 * 8
    w_bfa = torch.randn(n1, hidden, device="cuda", generator=g) * 0.02
    w_bfa[FA_DIM + heads :] = 0
    w_fb = torch.randn(heads * 128, FA_DIM, device="cuda", generator=g) * 0.05
    x = torch.randn(rows, hidden, device="cuda", generator=g)
    return x.bfloat16(), w_bfa.bfloat16(), w_fb.bfloat16(), heads


@pytest.mark.parametrize("cell", SAMPLE, ids=[c.why[:44] for c in SAMPLE])
def test_certified_cells(cell) -> None:
    x, w_bfa, w_fb, heads = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        beta, fg = op(x, w_bfa, w_fb, heads)
    ref_beta, ref_fg = op.reference(x, w_bfa, w_fb, heads)
    assert beta.dtype == fg.dtype == torch.bfloat16
    assert beta.shape == ref_beta.shape and fg.shape == ref_fg.shape
    # beta: the same FP32 sum in another order, rounded once to BF16 -> within one BF16 step.
    torch.testing.assert_close(beta.float(), ref_beta.float(), rtol=2**-7, atol=1e-3)
    # forget gate: f_a may round to the neighbouring BF16 value, which moves f_b by at most
    # 2^-8 |f_a| @ |W_fb|, plus its own final rounding.
    with torch.no_grad():
        fa = (x.float() @ w_bfa.float().t()).to(torch.bfloat16)[:, :FA_DIM].float()
        bound = (fa.abs() @ w_fb.float().abs().t()) * 2**-8
    err = (fg.float() - ref_fg.float()).abs()
    assert bool((err <= bound + ref_fg.float().abs() * 2**-7 + 1e-4).all()), err.max()
    assert (err <= ref_fg.float().abs() * 2**-7 + 1e-4).float().mean().item() > 0.99
