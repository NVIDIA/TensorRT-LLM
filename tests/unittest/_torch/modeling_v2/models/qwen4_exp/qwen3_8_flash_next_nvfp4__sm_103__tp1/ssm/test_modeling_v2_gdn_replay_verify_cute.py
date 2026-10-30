# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's cached-replay verify dispatch over a bf16 pool (the CuTe kernel).

`_replay_verify` as the decode target calls it -- `a`/`b` as column views of the input
projection, the work items the metadata builds over every decode row (`_replay_work`), PDL
after the fused in_proj GEMV -- against the call it replaces (contiguous `a`/`b`, no work
items, no PDL), each on its own copy of the pool and the history, over consecutive verify
steps with random partial acceptance and `_advance_replay_state` in between: outputs, the
checkpoint (committed in-kernel on every wrap) and the history bitwise equal at every step.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.modules.fla.cache_manager import (
    ReplayStateUpdateMetadata,
    _advance_replay_state,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import _build_replay_work_items_torch

assert torch.cuda.is_available(), "the GDN verify dispatch test requires a CUDA device"

DEV = "cuda"
BF = torch.bfloat16
H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
HIST = 16
SLOTS = 40


class _Side:
    """One copy of a layer's replay state and the `core` stand-in `_replay_verify` reads.

    The state: the pool, the history and the bookkeeping (pending count, half).
    """

    def __init__(self, pool, old, pnat, buf, t, gating):
        self.pool = pool.clone()
        self.old = [x.clone() for x in old]
        self.pnat, self.buf = pnat.clone(), buf.clone()
        self.meta = ReplayStateUpdateMetadata(self.pnat, self.buf, t, HIST)
        replay = M._Replay(
            old={0: tuple(self.old)},
            cache_buf_idx=self.buf,
            prev_num_accepted=self.pnat,
            history=HIST,
            width=t,
            all_layer_commit=False,
            cute=True,
        )
        self.core = SimpleNamespace(
            conv_dim=CONV_DIM,
            key_dim=KEY_DIM,
            value_dim=VALUE_DIM,
            num_k_heads=H,
            head_k_dim=K,
            num_v_heads=HV,
            head_v_dim=V,
            _bound=SimpleNamespace(replay=replay),
        )
        self.layer = SimpleNamespace(a_log=gating[0], dt_bias=gating[1])

    def state(self):
        return [self.pool, *self.old[:3]]


@pytest.mark.parametrize("n,t", [(1, 4), (4, 4), (32, 4), (5, 3)])
def test_cute_dispatch_matches_the_plain_call(n, t):
    gen = torch.Generator(device=DEV).manual_seed(77 + n)
    cpu = torch.Generator().manual_seed(77 + n)
    slots = torch.randperm(SLOTS, generator=cpu)[:n].to(DEV, torch.int32)
    pool = (0.3 * torch.randn(SLOTS, HV, V, K, generator=gen, device=DEV)).to(BF)
    old = (
        (0.5 * torch.randn(SLOTS, 2, HIST, HV, V, generator=gen, device=DEV)).to(BF),
        torch.nn.functional.normalize(
            torch.randn(SLOTS, 2, HIST, H, K, generator=gen, device=DEV), dim=-1
        ).to(BF),
        -(0.2 * torch.rand(SLOTS, 2, HV, HIST, generator=gen, device=DEV)).cumsum(-1),
        torch.zeros(SLOTS, 2, HV, HIST, device=DEV),
    )
    pnat = torch.randint(0, HIST + 1, (SLOTS,), generator=cpu, dtype=torch.int32).to(DEV)
    buf = torch.randint(0, 2, (SLOTS,), generator=cpu, dtype=torch.int32).to(DEV)
    gating = (
        (torch.randn(HV, generator=gen, device=DEV) * 1.5 + 1.0).clamp(-7.0, 5.0),
        (torch.randn(HV, generator=gen, device=DEV) * 1.7 - 4.4).clamp(-11.0, 2.5),
    )
    plain = _Side(pool, old, pnat, buf, t, gating)
    disp = _Side(pool, old, pnat, buf, t, gating)
    mm = SimpleNamespace(
        replay_work_items=torch.zeros(n, 4, dtype=torch.int32, device=DEV),
        replay_n_writes=torch.zeros(1, dtype=torch.int32, device=DEV),
        replay_num_decodes=n,
    )
    wraps = 0
    for step in range(HIST // t + 3):
        wraps += int((disp.pnat[slots.long()] + t > HIST).sum())
        # what the metadata's prepare builds before the forward
        _build_replay_work_items_torch(
            slots, disp.pnat, disp.buf, mm.replay_work_items, mm.replay_n_writes, t, HIST
        )
        replay = M._replay_work(disp.core, mm, n)
        assert replay[0] is not None and not replay[2], "the CuTe kernel takes the work items"
        mq_out = (0.5 * torch.randn(n * t, CONV_DIM, generator=gen, device=DEV)).to(BF)
        # z | a | b: column views of one projection output, as the decode target holds them
        zab = torch.randn(n * t, VALUE_DIM + 2 * HV, generator=gen, device=DEV).to(BF)
        _, a, b = torch.split(zab, [VALUE_DIM, HV, HV], 1)
        want = M._replay_verify(
            plain.core,
            mq_out,
            a.contiguous(),
            b.contiguous(),
            plain.pool,
            0,
            plain.layer,
            slots,
            n,
            t,
            (None, None, False),
        )
        got = M._replay_verify(
            disp.core, mq_out, a, b, disp.pool, 0, disp.layer, slots, n, t, replay, pdl=True
        )
        torch.cuda.synchronize()
        assert torch.equal(got, want), f"step {step}: outputs differ"
        for name, x, y in zip(("pool", "u", "k", "G"), disp.state(), plain.state()):
            assert torch.equal(x, y), f"step {step}: {name} differs"
        accepted = torch.randint(1, t + 1, (n,), generator=cpu, dtype=torch.int32).to(DEV)
        for side in (plain, disp):
            _advance_replay_state(side.meta, slots, accepted)
    assert wraps > 0, "no history wrapped: the in-kernel commit was never exercised"


def test_no_work_items_without_the_metadata():
    """A metadata without the build (or built for another batch) hands no work items."""
    side = _Side(
        torch.zeros(2, HV, V, K, dtype=BF, device=DEV),
        (torch.zeros(1),) * 4,
        torch.zeros(2, dtype=torch.int32, device=DEV),
        torch.zeros(2, dtype=torch.int32, device=DEV),
        4,
        (None, None),
    )
    assert M._replay_work(side.core, SimpleNamespace(), 3) == (None, None, False)
    mm = SimpleNamespace(
        replay_work_items=torch.zeros(8, 4, dtype=torch.int32, device=DEV),
        replay_n_writes=torch.zeros(1, dtype=torch.int32, device=DEV),
        replay_num_decodes=5,
    )
    assert M._replay_work(side.core, mm, 3) == (None, None, False)
