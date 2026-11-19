# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_add_rmsnorm_fwd entry.

A cell is (T, N) with T == 1, the only token count the target fuses the add and the norm at, and
N = K + 1 candidates. Each sampled cell checks the updated residual bit for bit (a plain bf16 add) and the
output's closeness to the entry's reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm_fwd import (
    attn_res_add_rmsnorm_fwd,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)

HIDDEN_SIZE = 7168
# The reference rounds at the kernel's three rounding points, so what remains is the fast-math
# softmax weights and the summation order inside each candidate's norm. Measured on GB300 over every
# cell: relative L2 at most 2.3e-5, at most 0.14 bf16 ulp of a row's largest magnitude per element,
# cosine >= 0.9999998. A dropped candidate moves the output by tens of percent.
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 1e-3


CELLS = sample_cells(attn_res_add_rmsnorm_fwd.CELLS, size_key="candidates")[:2]


@pytest.mark.parametrize(
    "cell",
    CELLS,
    ids=[c.why[:48] for c in CELLS],
)
def test_certified_cells_add_rmsnorm(cell, certifying) -> None:  # noqa: F811
    certifying(attn_res_add_rmsnorm_fwd)
    spec = cell.spec
    t, n, h = spec["tokens"], spec["candidates"], spec["hidden"]
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)

    def randn(*shape, scale=0.05, shift=0.0):
        return (shift + torch.randn(*shape, generator=gen, device="cuda") * scale).bfloat16()

    layer, layer_add = randn(t, 1, h), randn(t, 1, h)
    inputs = (layer, layer_add, randn(n - 1, t, 1, h), randn(h, scale=0.02))
    inputs += (randn(h, scale=0.02, shift=1.0), randn(h, scale=0.02, shift=1.0))
    args = (*inputs, spec["eps"], spec["output_eps"])
    updated, out = attn_res_add_rmsnorm_fwd(*args)
    _, ref_out = attn_res_add_rmsnorm_fwd.reference(*args)
    assert torch.equal(updated.view(torch.int16), (layer + layer_add).view(torch.int16))
    a, e = out.float().flatten(), ref_out.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    relative_l2 = ((a - e).norm() / (e.norm() + 1e-12)).item()
    assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2, (cosine, relative_l2)
