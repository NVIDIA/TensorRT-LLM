# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the Kimi K3.1 one-sided all-to-all workspace layout entry (``ep_dispatch_combine``
``MoeA2AGetWorkspaceLayout``, a host op): every cell equals the reference's restatement of the native planner, for the
region sizes the K3.1 exchange plans."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import ep_dispatch_combine as exchange

pytestmark = pytest.mark.cpu_only

_HIDDEN, _TOP_K = 3584, 16


def _region_bytes(ep: int, max_tokens: int) -> tuple:
    """The dispatch and combine-input bytes the K3.1 exchange plans (BF16, MXFP8 payloads)."""

    def pad128(n: int) -> int:
        return (n + 127) // 128 * 128

    tokens = ep * max_tokens
    activations = pad128(tokens * _HIDDEN * 2)
    quantized = pad128(tokens * _HIDDEN) + pad128(tokens * ((_HIDDEN + 15) // 16) * 4)
    dispatch = max(activations, quantized) + 2 * pad128(tokens * _TOP_K * 4)
    combine = tokens * _HIDDEN * 2
    return dispatch, combine


op = exchange.moe_a2a_get_workspace_layout


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_layout_cells(cell) -> None:
    spec = cell.spec
    args = (spec["ep"], spec["max_tokens"], _TOP_K, *_region_bytes(**spec))
    got = op(*args)
    assert torch.equal(got, op.reference(*args)), (got.tolist(), op.reference(*args).tolist())
