# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""K3's token-embedding gather (``ops/embedding/k3_embedding_gather``, ``spec_head_fusion``)
is bitwise ``F.embedding``: every cell of ``k3_embedding_gather.CELLS`` under ``validating`` against
the entry's reference (out-of-range ids give zero rows), and small tables at 0..32 ids."""

import importlib
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

k3_embedding_gather = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "ops.embedding.k3_embedding_gather"
).k3_embedding_gather

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
certified = pytest.mark.skipif(
    not torch.cuda.is_available()
    or _ARCH.get(torch.cuda.get_device_capability()) not in k3_embedding_gather.ARCHS,
    reason="k3_embedding_gather is certified on its ARCHS",
)


@pytest.fixture(scope="module")
def k3_table():
    """The replicated K3 table shape, [163840, 7168] BF16."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    spec = k3_embedding_gather.CELLS[0].spec
    return torch.randn(spec["vocab"], spec["hidden"], device="cuda", generator=gen).to(
        torch.bfloat16
    )


@certified
@pytest.mark.parametrize(
    "cell", k3_embedding_gather.CELLS, ids=[c.why[:48] for c in k3_embedding_gather.CELLS]
)
def test_certified_cells(cell, k3_table):
    spec = cell.spec
    assert tuple(k3_table.shape) == (spec["vocab"], spec["hidden"])
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows = spec["rows"]
    ids = torch.randint(0, spec["vocab"], (rows,), device="cuda", generator=gen).to(torch.int32)
    ids[0] = spec["vocab"] - 1
    if spec["padded"]:
        ids[1::3] = -1
    with validating(k3_embedding_gather):
        out = k3_embedding_gather(ids, k3_table)
    ref = k3_embedding_gather.reference(ids, k3_table)
    assert out.shape == (rows, spec["hidden"]) and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), cell.why
    if spec["padded"]:
        assert not out[1::3].view(torch.int16).any(), f"{cell.why}: padded ids are not zero rows"


@pytest.mark.parametrize("num_ids", [0, 1, 7, 14, 32])
def test_embedding_gather_matches_embedding(num_ids):
    gen = torch.Generator(device="cuda").manual_seed(num_ids)
    weight = torch.randn(5000, 7168, device="cuda", generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, 5000, (num_ids,), device="cuda", generator=gen).to(torch.int32)
    if num_ids:
        ids[0] = 4999
    out = k3_embedding_gather(ids, weight)
    torch.cuda.synchronize()
    assert out.shape == (num_ids, 7168)
    assert torch.equal(out.view(torch.int16), F.embedding(ids, weight).view(torch.int16))
