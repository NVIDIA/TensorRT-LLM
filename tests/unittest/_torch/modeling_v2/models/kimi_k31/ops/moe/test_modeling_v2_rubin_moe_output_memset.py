# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_moe_output_memset, the CuteDSL context path's zero of the output rows FC2 finalizes into.

On SM103 and SM107: a sample of the cells against the reference, on the ``rubin_moe_sort`` tables.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_output_memset import (  # noqa: E501
    rubin_moe_output_memset,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_sort import rubin_moe_sort

__extra_import_path__ = [".", "../../../..", "../.."]
import _rubin_nvfp4_moe_inputs as moe_inputs  # noqa: E402 — needs the path declared above
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401 -- path above

pytestmark = cpu_or_sm(103, 107)
gpu_10x = on_sm(103, 107)


CELLS = sample_cells(rubin_moe_output_memset.CELLS, size_key="tokens")[:2]


@gpu_10x
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] for c in CELLS])
def test_memset_cells(cell, certifying):  # noqa: F811
    certifying(rubin_moe_output_memset)
    spec = cell.spec
    ids, weights = moe_inputs.route(
        spec["tokens"], 896, spec["top_k"], zlib.crc32(cell.why.encode()), "cuda"
    )
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort(
        ids, weights, 896, spec["top_k"], 0, spec["local_num_experts"], spec["tile_tokens_dim"]
    )
    out = torch.ones(spec["tokens"], spec["hidden"], dtype=torch.bfloat16, device="cuda")
    args = (
        t2l,
        e2p,
        p2e,
        n,
        spec["tile_tokens_dim"],
        spec["top_k"],
        spec["ep_size"],
        spec["enable_alltoall"],
    )
    ref = rubin_moe_output_memset.reference(out, *args)
    rubin_moe_output_memset(out, *args)
    assert torch.equal(out, ref)
