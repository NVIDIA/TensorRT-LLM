# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family's ``attn_res.py``, host-side (no GPU): the most tokens of a classified step the fused attn_res
kernels take (``_attn_res_max_tokens``)."""

import pytest

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl import attn_res, step
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

#: (step, ceiling): a ``DecodeStep``'s fields and the most tokens of the step the fused attn_res kernels take.
ATTN_RES_CEILINGS = [
    ((5,), 8),  # a short prefill or mixed step
    ((8, 8, 1), 8),  # a decode step of one token tile
    ((6, 1, 6), 8),
    ((16, 2, 8), 32),  # a wide decode step
    ((64, 8, 8), 32),
]


@pytest.mark.parametrize("fields,ceiling", ATTN_RES_CEILINGS)
def test_attn_res_epilogue_ceiling(fields, ceiling):
    """The fused attn_res kernels take a classified step up to one token tile and a wide decode step up to 32
    tokens."""
    assert attn_res._attn_res_max_tokens(step.DecodeStep(*fields)) == ceiling
