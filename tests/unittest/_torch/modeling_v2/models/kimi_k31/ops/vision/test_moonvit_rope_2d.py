# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_rope_2d entry: each certified cell against its reference within the band
stated below, and bit for bit against the engine MoonViT3d code it replaces."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.vision.moonvit_rope_2d import (
    TABLE_SIDE,
    THETA,
    moonvit_rope_2d,
    moonvit_rope_angles,
    moonvit_rope_table,
)
from tensorrt_llm._torch.models import modeling_kimi_k25 as engine_k25

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 (sm_107 driven separately)",
)

#: bf16 results against an fp32 / fp64 reference rounded once: the entries round intermediate bf16 values, each half
#: an ulp; four ulps of the row's scale covers them.
_BF16_BAND = dict(rtol=4 * 2.0**-8, atol=4 * 2.0**-8)


def _gen(cell) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)


def _ids(op):
    return [c.why[:48] for c in op.CELLS]


@pytest.mark.parametrize("cell", moonvit_rope_2d.CELLS, ids=_ids(moonvit_rope_2d))
def test_rope_2d_cells(cell, certifying) -> None:  # noqa: F811
    certifying(moonvit_rope_2d)
    gen = _gen(cell)
    grids, heads, head_dim = cell.spec["grids"], cell.spec["heads"], cell.spec["head_dim"]
    rows = sum(t * h * w for t, h, w in grids)
    qkv = torch.randn(rows, 3 * heads * head_dim, generator=gen, device="cuda").bfloat16()
    q, k, _ = qkv.split(heads * head_dim, dim=-1)
    q, k = q.view(rows, heads, head_dim), k.view(rows, heads, head_dim)

    table = moonvit_rope_table(head_dim, TABLE_SIDE, TABLE_SIDE, THETA, torch.device("cuda"))
    engine_rope = engine_k25.Rope2D(head_dim)
    grid_thws = torch.tensor(grids, dtype=torch.int64, device="cuda")
    freqs_cis = moonvit_rope_angles(table, grids)
    assert torch.equal(freqs_cis, engine_rope.get_freqs_cis(grid_thws, torch.device("cuda")))

    q_out, k_out = moonvit_rope_2d(q, k, freqs_cis)
    q_engine, k_engine = engine_k25._apply_rope_2d(q, k, freqs_cis)
    assert torch.equal(q_out, q_engine) and torch.equal(k_out, k_engine)
    q_ref, k_ref = moonvit_rope_2d.reference(q, k, freqs_cis)
    torch.testing.assert_close(q_out, q_ref, **_BF16_BAND)
    torch.testing.assert_close(k_out, k_ref, **_BF16_BAND)
