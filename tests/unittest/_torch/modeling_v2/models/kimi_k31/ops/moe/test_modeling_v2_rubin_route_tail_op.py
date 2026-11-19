# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_route_tail entry: a sample of its cells against the plain-torch reference and noaux_tc_op's bits
on SM107."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import TOP_K
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_route_tail import (  # noqa: E501
    rubin_route_tail,
)

__extra_import_path__ = ["../../../..", "../../kernels/moe", "../.."]
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _rubin_route_tail_inputs import request_logits  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

# Certified on SM107 only; other GPUs skip the file.
pytestmark = cpu_or_sm(107)
gpu = on_sm(107)


def _check(out, ref):
    ids, weights, tile_expert, tile_limit, e2p, p2e, total, num_tiles = out
    r_ids, r_weights, r_expert, r_limit, r_e2p, r_p2e, r_total, r_tiles = ref
    assert torch.equal(ids, r_ids), "selected experts differ"
    # The FP32 total of the 16 picks is a reduction whose order the reference does not model:
    # the renormalized weights may sit two ulp apart. The kernel's bits are noaux_tc_op's
    # (test_certified_cells checks that exactly).
    torch.testing.assert_close(weights, r_weights, rtol=2**-22, atol=0)
    n = int(r_tiles.item())
    assert int(num_tiles.item()) == n and int(total.item()) == int(r_total.item())
    assert torch.equal(tile_expert[:n], r_expert[:n]) and torch.equal(tile_limit[:n], r_limit[:n])
    assert torch.equal(e2p, r_e2p), "expanded_idx_to_permuted_idx differs"
    valid = r_p2e >= 0
    assert torch.equal(p2e[valid], r_p2e[valid]), "permuted_idx_to_expanded_idx differs"


def _kwargs(spec):
    return dict(route_tail_bands=spec.get("route_tail_bands", ""))


CELLS = sample_cells(rubin_route_tail.CELLS, size_key="rows")[:2]


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_route_tail)
    spec = cell.spec
    logits, bias = request_logits(
        spec["rows"], zlib.crc32(cell.why.encode()), spec.get("mode", "realistic"), "cuda"
    )
    args = (logits, bias, spec.get("scale", 1.0), spec["offset"], spec["local"])
    out = rubin_route_tail(*args, **_kwargs(spec))
    _check(out, rubin_route_tail.reference(*args, **_kwargs(spec)))
    # The kernel's own claim: noaux_tc_op's bits.
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers noaux_tc_op

    weights, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, spec.get("scale", 1.0))
    assert torch.equal(out[0], ids.to(torch.int32))
    assert torch.equal(out[1].view(torch.int32), weights.float().view(torch.int32))
