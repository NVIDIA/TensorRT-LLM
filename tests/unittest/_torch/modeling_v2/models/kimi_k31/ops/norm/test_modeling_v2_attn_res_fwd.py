# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_fwd entry.

A cell is (T, N): T tokens and N = K + 1 candidates, K snapshots plus the layer residual. Each sampled
cell checks the four outputs' closeness to the entry's fp32 reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_fwd import attn_res_fwd

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(103, 107)

# Measured on GB300 over every cell: the mixture within relative L2 7.5e-5 of the fp32 reference (at
# most 1.7 bf16 ulp of a row's largest magnitude per element: one bf16 rounding on each side plus
# fast-math softmax), the fp32 statistics within 2.5e-7; cosine >= 0.9999998 everywhere. The bands
# leave more than an order of magnitude; a dropped or doubled candidate moves the mixture by tens of
# percent.
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = (1e-3, 1e-5, 1e-5, 1e-5)
OUTPUT_NAMES = ("output", "rsigma", "probs", "logits")


CELLS = sample_cells(attn_res_fwd.CELLS, size_key="tokens")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(attn_res_fwd)
    spec = cell.spec
    t, n, h = spec["tokens"], spec["candidates"], spec["hidden"]
    # Residual-stream scale values, with a scoring weight that spreads the softmax over candidates.
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)

    def randn(*shape, scale=0.05, shift=0.0):
        return (shift + torch.randn(*shape, generator=gen, device="cuda") * scale).bfloat16()

    inputs = (randn(t, 1, h), randn(n - 1, t, 1, h), randn(h, scale=0.02))
    inputs += (randn(h, scale=0.02, shift=1.0),)
    outputs = attn_res_fwd(*inputs, spec["eps"])
    expected = attn_res_fwd.reference(*inputs, spec["eps"])
    for name, actual, ref, max_l2 in zip(OUTPUT_NAMES, outputs, expected, MAX_RELATIVE_L2):
        a, e = actual.float().flatten(), ref.float().flatten()
        cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
        relative_l2 = ((a - e).norm() / (e.norm() + 1e-12)).item()
        assert cosine > MIN_COSINE and relative_l2 < max_l2, (name, cosine, relative_l2)
