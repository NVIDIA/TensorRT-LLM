# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""merge_chunked_attention_for_mla (family op): the merged attention and softmax statistics against the fp32 merge of
two partials, per request's merge op (copy, merge, skip); rows a skipped request owns are untouched."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.merge_chunked_attention_for_mla import (  # noqa: E501
    merge_chunked_attention_for_mla as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="certified on sm_100",
)


def _stats(rows, heads, gen):
    m = torch.randn(rows, heads, generator=gen, device="cuda") * 3
    s = torch.rand(rows, heads, generator=gen, device="cuda") * 50 + 1
    return torch.stack([m, s], -1).contiguous()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_cells(cell):
    spec = cell.spec
    heads, size = spec["num_heads"], spec["head_size"]
    q_lens = list(spec["q_lens"])
    rows = sum(q_lens)
    gen = torch.Generator(device="cuda").manual_seed(sum(q_lens))
    merged = torch.randn(rows, heads * size, generator=gen, device="cuda").bfloat16()
    temp = torch.randn(rows, heads * size, generator=gen, device="cuda").bfloat16()
    stats, temp_stats = _stats(rows, heads, gen), _stats(rows, heads, gen)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int64)
    cu[1:] = torch.tensor(q_lens).cumsum(0)
    merge_op = torch.tensor(spec["merge_op"], dtype=torch.int64, device="cuda")
    want_attn, want_stats = op.reference(
        merged, temp, stats, temp_stats, len(q_lens), cu, max(q_lens), merge_op, heads, size
    )
    before = merged.clone()
    with validating(op):
        op(
            merged,
            temp,
            stats,
            temp_stats,
            len(q_lens),
            cu.cuda(),
            max(q_lens),
            merge_op,
            heads,
            size,
        )
    torch.testing.assert_close(merged.float(), want_attn.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(stats, want_stats, rtol=1e-5, atol=1e-5)
    for b, o in enumerate(spec["merge_op"]):
        if o == 0:
            r = slice(int(cu[b]), int(cu[b + 1]))
            assert torch.equal(merged[r], before[r]), "a skipped request's rows moved"
