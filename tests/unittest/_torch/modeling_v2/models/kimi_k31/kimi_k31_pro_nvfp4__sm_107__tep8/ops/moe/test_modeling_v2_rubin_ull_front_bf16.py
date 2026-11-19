# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_ull_front_bf16, the BF16 low-latency MoE front: the router logits and the down-projection
block rows of one rank of the TP group.

On the CPU: the block rows a rank reads and the entry's refusals. The cells
(``rubin_ull_front_bf16.CELLS``) need two or four SM107 ranks of one tray, which no CI stage has.
"""

from types import SimpleNamespace

__extra_import_path__ = ["../../.."]
import pytest
import torch
from _k31_utils import cpu_or_sm

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.ops.moe import (
    _rubin_ull_common as c,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.ops.moe.rubin_ull_front_bf16 import (  # noqa: E501
    rubin_ull_front_bf16,
)


def _ws(world, rank, nbytes=1 << 23):
    return SimpleNamespace(world=world, rank=rank, bytes_per_buffer=nbytes)


@pytest.mark.cpu_only
@cpu_or_sm(107)
def test_front_guard_and_rows():
    down = torch.zeros(c.LATENT, c.HIDDEN, dtype=torch.bfloat16)
    r0, n = c.block_slice(c.LATENT, 8, 5)
    rows = down[r0 : r0 + n]
    assert rows.shape == (384, c.HIDDEN) and rows.data_ptr() == down[2432].data_ptr()
    x = torch.zeros(8, c.HIDDEN, dtype=torch.bfloat16)
    logits = torch.zeros(8, c.NUM_EXPERTS)
    views = (torch.zeros(32, c.NUM_EXPERTS), torch.zeros(32, c.NUM_EXPERTS))
    args = (x, logits, rows, views, torch.zeros(c.NUM_EXPERTS), 1.0, torch.ones(1), 560, 112)
    zero = torch.zeros(16, dtype=torch.int32)
    rubin_ull_front_bf16.is_valid(*args, _ws(8, 5), (64, 8, 4, 12), zero=zero)
    with pytest.raises(AssertionError):
        rubin_ull_front_bf16.is_valid(*args, _ws(8, 1), (64, 8, 4, 12), zero=zero)
    with pytest.raises(AssertionError):
        rubin_ull_front_bf16.is_valid(
            x, logits[:4], *args[2:], _ws(8, 5), (64, 8, 4, 12), zero=zero
        )
