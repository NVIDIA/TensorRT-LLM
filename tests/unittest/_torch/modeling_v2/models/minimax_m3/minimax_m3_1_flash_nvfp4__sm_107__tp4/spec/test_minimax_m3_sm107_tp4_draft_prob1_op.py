# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 draft: mv2_minimax_m3::minimax_m3_sm107tp4_draft_prob1 on one sm_107 GPU (the collective draft itself is the
4-rank test): the fake is registered (a chain op: the regfake gate is hard), max_grid is a co-resident grid at 1..8
rows and refuses 0 / 9, and the op refuses what the reference steps() refuses before any launch."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7), reason="sm_107"
)
OP = "minimax_m3_sm107tp4_draft_prob1"
K, V, VS, R = 7, 32768, 8192, 256


def _op():
    return getattr(family_ops(), OP)


def test_fake_is_registered():
    _op()
    # The family fakes register when family_ops() loads the library; importing them before that fails.
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import fake_ops

    assert torch._C._dispatch_has_kernel_for_dispatch_key(f"{fake_ops._NS}::{OP}", "Meta")


def test_max_grid():
    _op()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for c in range(1, 9):
        g = int(family_ops().minimax_m3_sm107tp4_draft_prob1_max_grid(c))
        assert 1 <= g <= 512 and g >= min(sms, 512) // 2, (c, g, sms)
    for c in (0, 9):
        with pytest.raises(RuntimeError):
            family_ops().minimax_m3_sm107tp4_draft_prob1_max_grid(c)


def _args(rows=2, world=4, k=K, grid=None):
    d = "cuda"
    z = lambda *s, dt=torch.int64: torch.zeros(*s, dtype=dt, device=d)  # noqa: E731
    cache = torch.zeros(8, max(k, 1), V, dtype=torch.bfloat16, device=d)
    fake_ptr = 0x10000
    ws = [fake_ptr] * 7
    if grid is None:
        grid = int(family_ops().minimax_m3_sm107tp4_draft_prob1_max_grid(max(1, min(rows, 8))))
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
    "kw", [dict(world=3), dict(world=1), dict(k=17), dict(rows=9), dict(grid=100000)]
)
def test_refuses_before_launch(kw):
    op = _op()
    with pytest.raises(RuntimeError):
        op(*_args(**kw))


@pytest.mark.parametrize(
    "index", [1, 2, 5, 6, 7, 8, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
)
def test_refuses_cpu_operands_before_launch(index):
    op = _op()
    args = _args()
    args[index] = args[index].cpu()
    with pytest.raises(RuntimeError, match="every tensor must be CUDA"):
        op(*args)


@pytest.mark.parametrize("index", [5, 6, 7])
def test_refuses_strided_prev_before_launch(index):
    op = _op()
    args = _args()
    args[index] = torch.zeros(16, dtype=torch.int64, device="cuda")[::2]
    with pytest.raises(RuntimeError, match="prev"):
        op(*args)


@pytest.mark.parametrize("bad", ["negative_voff", "shard_past_vocab", "short_w1"])
def test_refuses_global_vocab_contract_before_launch(bad):
    op = _op()
    args = _args()
    if bad == "negative_voff":
        args[3] = -1
    elif bad == "shard_past_vocab":
        args[3] = V
    else:
        args[1] = args[1][:-1].contiguous()
    with pytest.raises(RuntimeError):
        op(*args)
