# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of rubin_gvr_topk_decode, the self-sampling GVR Top-K of the decode rows (a deterministic
kernel).

GPU (sm_107): a sample of the cells against the reference as sets (the kernel's output order is its own).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.rubin_gvr_topk_decode import (  # noqa: E501
    rubin_gvr_topk_decode as decode_op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

#: A boundary and a typical cell of ``decode_op.CELLS``.
_CELLS = sample_cells(decode_op.CELLS, size_key="batch")[:2]


on_sm107 = on_sm(107)


def _sets(rows: torch.Tensor):
    return [set(r[r >= 0].tolist()) for r in rows.cpu()]


def _logits(rows, width, gen, ties=False, align=4):
    stride = -(-width // align) * align
    base = torch.empty(rows, stride)
    if ties:
        base.copy_(torch.randint(-3, 4, (rows, stride), generator=gen).float())
    else:
        base.copy_(torch.randn(rows, stride, generator=gen))
    return base[:, :width]


@on_sm107
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_decode_certified_cells(cell, certifying):  # noqa: F811
    certifying(decode_op)
    s = cell.spec
    gen = torch.Generator().manual_seed(s["batch"] * 31 + s["next_n"])
    lo, hi = s["kv_lens"]
    kv_lens = torch.randint(lo, hi + 1, (s["batch"],), generator=gen, dtype=torch.int32)
    rows = s["batch"] * s["next_n"]
    width = -(-s["max_seq_len"] // 256) * 256
    logits = _logits(rows, width, gen, s.get("data") == "ties", 256)[:, : s["max_seq_len"]].cuda()
    out = torch.full((rows, s["top_k"]), -7, dtype=torch.int32, device="cuda")
    decode_op(logits, kv_lens.cuda(), out, s["next_n"], s["max_seq_len"])
    ref = decode_op.reference(logits, kv_lens, out, s["next_n"], s["max_seq_len"])
    assert _sets(out) == _sets(ref)
