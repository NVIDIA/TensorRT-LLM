# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 copy_kda_replay_conv_window entry.

The live conv pool and the replay conv caches are padded-slot views laid out as the V2 manager carves them. The
caches, bit for bit, against the reference on same-layout copies (the pending-draft tail columns and every other
slot untouched).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.copy_kda_replay_conv_window import (
    copy_kda_replay_conv_window as op,
)

from ._kda_ops_util import (
    COMMITTED,
    HEAD_DIM,
    bits,
    clone_strided,
    generator,
    padded_slots,
    replay_caches,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a Triton kernel: needs a GPU"
)

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="pool_slots")[:2]


CACHES = ("cs_q", "cs_k", "cs_v")


def _inputs(spec, gen):
    heads, num_slots = spec["heads"], spec["pool_slots"]
    conv = padded_slots(num_slots, (3 * heads * HEAD_DIM, COMMITTED), torch.bfloat16)
    conv.copy_(torch.randn(conv.shape, generator=gen, device="cuda"))
    caches = replay_caches(num_slots, heads, spec["num_spec"], gen)
    dtype = torch.int64 if spec.get("index_dtype") == "int64" else torch.int32
    indices = torch.tensor(spec["slots"], dtype=dtype, device="cuda")
    return conv, [caches[name] for name in CACHES], indices


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    conv, caches, indices = _inputs(cell.spec, generator(cell))
    expected = [clone_strided(cache) for cache in caches]
    op(conv, *caches, indices)
    op.reference(conv, *expected, indices)
    # Every slot of each cache: the selected ones copied, the others untouched.
    for name, got, ref in zip(CACHES, caches, expected):
        assert torch.equal(bits(got), bits(ref)), name
