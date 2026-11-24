# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 moonvit_attention entry: each certified cell against its reference within the band
stated below; that the call capacities leave the rows unchanged, also when a call holds more images than its request
capacity and runs in groups. The tower A/B in ``test_vision.py`` pins the op
against the engine tower's attention."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.vision.moonvit_attention import (
    moonvit_attention,
)

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


@pytest.mark.parametrize("cell", moonvit_attention.CELLS, ids=_ids(moonvit_attention))
def test_attention_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(moonvit_attention)
    gen = _gen(cell)
    seg_lens, heads, head_dim = cell.spec["seg_lens"], cell.spec["heads"], cell.spec["head_dim"]
    capacity = cell.spec["capacity"]
    qkv = torch.randn(sum(seg_lens), 3 * heads * head_dim, generator=gen, device="cuda").bfloat16()
    out = torch.empty(sum(seg_lens), heads * head_dim, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(0, dtype=torch.int8, device="cuda")

    moonvit_attention(qkv, out, seg_lens, capacity, capacity // 4, workspace, heads, head_dim)
    ref = moonvit_attention.reference(
        qkv, out, seg_lens, capacity, capacity // 4, workspace, heads, head_dim
    )
    torch.testing.assert_close(out, ref, **_BF16_BAND)


def test_attention_capacities_do_not_change_the_rows() -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    seg_lens = (16, 1024, 400)
    qkv = torch.randn(sum(seg_lens), 3 * 1536, generator=gen, device="cuda").bfloat16()
    workspace = torch.empty(0, dtype=torch.int8, device="cuda")
    outs = []
    for capacity in (1024, 67320):
        out = torch.empty(sum(seg_lens), 1536, dtype=torch.bfloat16, device="cuda")
        outs.append(
            moonvit_attention(qkv, out, seg_lens, capacity, capacity // 4, workspace, 12, 128)
        )
    assert torch.equal(outs[0], outs[1])


def test_more_images_than_the_request_capacity_run_in_groups() -> None:
    """A call with more images than ``max_num_requests`` runs as native calls of at most that many images; the
    attention is per image, so the rows are those of one call that holds them all."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    seg_lens = (16, 64, 4, 400, 36)
    qkv = torch.randn(sum(seg_lens), 3 * 1536, generator=gen, device="cuda").bfloat16()
    workspace = torch.empty(0, dtype=torch.int8, device="cuda")
    outs = []
    for max_num_requests in (2, 256):
        out = torch.empty(sum(seg_lens), 1536, dtype=torch.bfloat16, device="cuda")
        outs.append(
            moonvit_attention(qkv, out, seg_lens, 1024, max_num_requests, workspace, 12, 128)
        )
    assert torch.equal(outs[0], outs[1])
