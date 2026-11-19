# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 MXFP4 routed-expert packing: the two kernel entries over a sample of
their cells."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import mxfp4_expert_pack as pack
from tensorrt_llm._utils import is_sm_100f

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or not is_sm_100f():
    pytest.skip("the TRTLLM-Gen MXFP4 layout is SM10x-only", allow_module_level=True)

shuffle_op = pack.k31_shuffle_matrix
interleave_op = pack.k31_block_scale_interleave


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


SHUFFLE_CELLS = sample_cells(shuffle_op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", SHUFFLE_CELLS, ids=[c.why[:60] for c in SHUFFLE_CELLS])
def test_shuffle_matrix_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x = torch.randint(
        0, 256, (spec["rows"], spec["cols"]), generator=g, dtype=torch.uint8, device="cuda"
    )
    if spec["kind"] == "w31":
        indices = pack._w3_w1_permute_indices(x, num_elts_per_sf=spec["sf"])
    else:
        indices = pack._w2_permute_indices(x, num_elts_per_sf=spec["sf"])
    with validating(shuffle_op):
        out = shuffle_op(x, indices)
    assert torch.equal(out, shuffle_op.reference(x, indices))
    # The permutation moves every row: identity indices would pass the comparison above too.
    assert not torch.equal(indices.cpu(), torch.arange(spec["rows"]))


INTERLEAVE_CELLS = sample_cells(interleave_op.CELLS, size_key="rows")[:2]


@pytest.mark.parametrize("cell", INTERLEAVE_CELLS, ids=[c.why[:60] for c in INTERLEAVE_CELLS])
def test_block_scale_interleave_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x = torch.randint(
        0, 256, (spec["rows"], spec["cols"]), generator=g, dtype=torch.uint8, device="cuda"
    )
    with validating(interleave_op):
        out = interleave_op(x)
    assert torch.equal(out, interleave_op.reference(x))
