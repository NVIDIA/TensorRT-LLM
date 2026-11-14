# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_PROB1 (TP8): mv2_minimax_m3::minimax_m3_sm107tp8_draft_prob1 on one sm_107 GPU (the collective draft itself
is the multi-rank test in comm/): the fake is registered, max_grid is a co-resident grid at 1..4 rows (the exchange
world 8 kernel takes 4 rows) and refuses 0 / 5, and the op refuses what its kernel refuses before any launch."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import fake_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7), reason="sm_107"
)
OP = "minimax_m3_sm107tp8_draft_prob1"
K, V, VS, R = 7, 32768, 4096, 256
ROWS = 4


def _op():
    return getattr(family_ops(), OP)


def test_fake_is_registered():
    _op()
    assert torch._C._dispatch_has_kernel_for_dispatch_key(f"{fake_ops._NS}::{OP}", "Meta")


def test_max_grid():
    _op()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for c in range(1, ROWS + 1):
        g = int(family_ops().minimax_m3_sm107tp8_draft_prob1_max_grid(c))
        assert 1 <= g <= 512 and g >= min(sms, 512) // 2, (c, g, sms)
    for c in (0, ROWS + 1):
        with pytest.raises(RuntimeError):
            family_ops().minimax_m3_sm107tp8_draft_prob1_max_grid(c)


def _args(rows=2, world=8, k=K, grid=None):
    d = "cuda"
    z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=d)  # noqa: E731
    cache = torch.zeros(8, max(k, 1), V, dtype=torch.bfloat16, device=d)
    fake_ptr = 0x10000
    ws = [fake_ptr] * 7
    if grid is None:
        grid = int(family_ops().minimax_m3_sm107tp8_draft_prob1_max_grid(max(1, min(rows, ROWS))))
    caches = [int(cache.data_ptr())] + [fake_ptr] * (world - 1)
    return [
        torch.zeros(rows * k, VS, dtype=torch.bfloat16, device=d),
        torch.zeros(V, R, dtype=torch.bfloat16, device=d),
        torch.zeros(VS, R, dtype=torch.bfloat16, device=d),
        0,
        k,
        z(8),
        z(8),
        z(8),
        z(8, max(k, 1)),
        ws,
        [fake_ptr] * world,
        caches,
        0,
        world,
        grid,
        True,
        z(8),
        0,
        z(8),
        0,
        32,
        z(8 * max(k, 1), dt=torch.int32),
        z(8 * max(k, 1)),
        z(8, dt=torch.float32),
        z(8),
        cache,
        z(1, dt=torch.int32),
        z(1),
        z(1),
        [fake_ptr] * world,
        1 << 30,
        z(8 * max(k, 1) * 16, dt=torch.float32),
        torch.zeros(16, max(k, 1), V, dtype=torch.float32, device=d),
        z(8 * max(k, 1), dt=torch.int32),
        V,
    ]


@pytest.mark.parametrize(
    "kw",
    [
        dict(world=3),
        dict(world=1),
        dict(world=6),
        dict(world=16),
        dict(k=17),
        dict(rows=5),
        dict(grid=100000),
    ],
)
def test_refuses_before_launch(kw):
    op = _op()
    with pytest.raises(RuntimeError):
        op(*_args(**kw))
