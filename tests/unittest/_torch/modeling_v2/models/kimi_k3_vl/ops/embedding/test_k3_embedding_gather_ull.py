# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ULL target's token-embedding gather (``ops/embedding/k3_embedding_gather_ull``) on every
cell of its ``.CELLS``, under ``validating``: bitwise the entry's reference (``F.embedding`` with
out-of-range ids as zero rows) and bitwise the family sibling ``k3_embedding_gather``."""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

_OPS = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.embedding"
k3_embedding_gather_ull = importlib.import_module(
    f"{_OPS}.k3_embedding_gather_ull"
).k3_embedding_gather

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or _ARCH.get(torch.cuda.get_device_capability()) not in k3_embedding_gather_ull.ARCHS,
    reason="k3_embedding_gather_ull is certified on its ARCHS",
)


@pytest.fixture(scope="module")
def k3_table():
    """The replicated K3 table shape, [163840, 7168] BF16."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    spec = k3_embedding_gather_ull.CELLS[0].spec
    return torch.randn(spec["vocab"], spec["hidden"], device="cuda", generator=gen).to(
        torch.bfloat16
    )


@pytest.mark.parametrize(
    "cell", k3_embedding_gather_ull.CELLS, ids=[c.why[:48] for c in k3_embedding_gather_ull.CELLS]
)
def test_k3_embedding_gather_ull_cells(cell, k3_table):
    sibling = importlib.import_module(f"{_OPS}.k3_embedding_gather").k3_embedding_gather
    op = k3_embedding_gather_ull
    spec = cell.spec
    assert tuple(k3_table.shape) == (spec["vocab"], spec["hidden"])
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows = spec["rows"]
    ids = torch.randint(0, spec["vocab"], (rows,), device="cuda", generator=gen).to(torch.int32)
    ids[0] = spec["vocab"] - 1
    if spec["padded"]:
        ids[1::3] = -1
    with validating(op):
        out = op(ids, k3_table)
    ref = op.reference(ids, k3_table)
    assert out.shape == (rows, spec["hidden"]) and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), cell.why
    if spec["padded"]:
        assert not out[1::3].view(torch.int16).any(), f"{cell.why}: padded ids are not zero rows"
    want = sibling(ids, k3_table)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), want.view(torch.int16)), cell.why
