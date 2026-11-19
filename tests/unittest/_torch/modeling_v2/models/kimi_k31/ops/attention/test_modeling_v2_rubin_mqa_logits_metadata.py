# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_mqa_logits_metadata entry: DeepGEMM's MQA-logits schedule changes where the
work runs, not the logits. Each cell scores FP8 queries against gathered FP8 keys with causal
windows, with and without the schedule, and the logits inside every row's window must be equal bit
for bit; the reference's per-block key range must lie inside the keys.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_mqa_logits_metadata import (  # noqa: E501
    rubin_mqa_logits_metadata as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(107)

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _windows(rows: int, keys: int, packed: int):
    """Causal windows of ``packed`` equal requests whose rows end the keys of their request."""
    per = rows // packed
    kv_per = keys // packed
    ks, ke = [], []
    for r in range(packed):
        n = per if r < packed - 1 else rows - per * (packed - 1)
        start = r * kv_per
        for i in range(n):
            ks.append(start)
            ke.append(start + kv_per - n + i + 1)
    return torch.tensor(ks, dtype=torch.int32).cuda(), torch.tensor(ke, dtype=torch.int32).cuda()


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    from tensorrt_llm.deep_gemm import fp8_fp4_mqa_logits

    certifying(op)
    spec = cell.spec
    rows, keys = spec["rows"], spec["keys"]
    gen = torch.Generator().manual_seed(inputs.seed_of(cell))
    q = inputs.queries(rows, "int", gen)
    k, k_scale = inputs.keys(keys, "int", gen)
    w = inputs.weights(rows, "int", gen)
    ks, ke = _windows(rows, keys, spec.get("packed", 1))
    meta = op(ks, ke, keys, spec["heads"])
    ref = op.reference(ks, ke, keys, spec["heads"])
    assert bool((ref[:, 0] <= ref[:, 1]).all()) and int(ref[:, 1].max()) <= keys
    kv = (k.cuda(), k_scale.cuda())
    plain = fp8_fp4_mqa_logits(
        (q, None), kv, w, ks, ke, clean_logits=False, logits_dtype=torch.float32
    )
    scheduled = fp8_fp4_mqa_logits(
        (q, None), kv, w, ks, ke, clean_logits=False, logits_dtype=torch.float32, schedule_meta=meta
    )
    cols = torch.arange(plain.shape[1], device="cuda").unsqueeze(0)
    inside = (cols >= ks.long().unsqueeze(1)) & (cols < ke.long().unsqueeze(1))
    assert torch.equal(plain[inside], scheduled[inside])
