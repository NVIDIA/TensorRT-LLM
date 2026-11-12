# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the k3_l2_prefetch entry: every cell of ``k3_l2_prefetch.CELLS`` runs (eagerly and
in a CUDA graph) and leaves the prefetched bytes and their neighbours unchanged, which is all its
no-op reference states."""

import pytest
import torch

pytest.importorskip("cutlass")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_kda.k3_l2_prefetch import (  # noqa: E501, E402
    tensor_range,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_l2_prefetch import (  # noqa: E501, E402
    k3_l2_prefetch,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating

GUARD = 4096  # bytes on each side of every range


def _driven_here() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in k3_l2_prefetch.ARCHS}


pytestmark = pytest.mark.skipif(not _driven_here(), reason="k3_l2_prefetch.ARCHS")


@pytest.mark.parametrize(
    "cell", k3_l2_prefetch.CELLS, ids=[c.why[:60] for c in k3_l2_prefetch.CELLS]
)
@torch.no_grad()
def test_cells(cell) -> None:
    gen = torch.Generator(device="cuda").manual_seed(7)
    buffers = [
        torch.randint(
            0, 256, (nbytes + 2 * GUARD,), generator=gen, device="cuda", dtype=torch.uint8
        )
        for nbytes in cell.spec["nbytes"]
    ]
    before = [b.clone() for b in buffers]
    ranges = [tensor_range(b[GUARD : GUARD + n]) for b, n in zip(buffers, cell.spec["nbytes"])]
    assert [n for _, n in ranges] == list(cell.spec["nbytes"]), "ranges must stay whole"

    with validating(k3_l2_prefetch):
        assert k3_l2_prefetch(ranges) is None
    torch.cuda.synchronize()
    assert k3_l2_prefetch.reference(ranges) is None

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        k3_l2_prefetch(ranges)
    graph.replay()
    torch.cuda.synchronize()
    for b, want in zip(buffers, before):
        assert torch.equal(b, want), cell.why
