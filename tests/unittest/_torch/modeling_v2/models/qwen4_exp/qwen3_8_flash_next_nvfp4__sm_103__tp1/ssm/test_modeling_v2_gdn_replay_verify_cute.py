# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's cached-replay verify dispatch over a bf16 pool (the CuTe kernel).

`_replay_verify` as the decode target calls it -- `a`/`b` as column views of the input
projection, PDL after the fused in_proj GEMV -- against the call it replaces (contiguous
`a`/`b`, no PDL), each on its own copy of the pool and the history, over consecutive verify
steps with random partial acceptance and `_advance_replay_state` in between: outputs, the
checkpoint (committed in-kernel on every wrap, as without the manager's all-layer commit) and
the history bitwise equal at every step, also with the dispatched side's history and bookkeeping
at replay slots of their own (the manager's PRIVATE replay pages). The partitioned replay (work
items and the manager's all-layer commit) is the CuTe entry's own test's.
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

assert torch.cuda.is_available(), "the GDN verify dispatch test requires a CUDA device"

DEV = "cuda"
BF = torch.bfloat16
H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
HIST = 16
SLOTS = 40
REPLAY_SLOTS = 2 * SLOTS


class _Side:
    """One copy of a layer's replay state and the `core` stand-in `_replay_verify` reads.

    The state: the pool, the history and the bookkeeping (pending count, half). With
    `replay_slots` the history and the bookkeeping move to those rows of larger buffers (the
    bookkeeping as strided columns of one metadata row per slot), NaN elsewhere; slot `s`'s
    history is then at row `replay_slots[s]`.
    """

    def __init__(self, pool, old, pnat, buf, t, gating, replay_slots=None):
        self.pool = pool.clone()
        self.replay_slots = replay_slots
        if replay_slots is None:
            self.old = [x.clone() for x in old]
            self.pnat, self.buf = pnat.clone(), buf.clone()
        else:
            rows = replay_slots.long()
            self.old = []
            for x in old:
                y = torch.full((REPLAY_SLOTS, *x.shape[1:]), float("nan"), device=DEV)
                y = y.to(x.dtype)
                y[rows] = x
                self.old.append(y)
            meta = torch.full((REPLAY_SLOTS, 4), 13, dtype=torch.int32, device=DEV)
            self.pnat, self.buf = meta[:, 0], meta[:, 1]
            self.pnat[rows], self.buf[rows] = pnat, buf
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

    def rows(self, slots):
        """The history / bookkeeping rows of checkpoint slots `slots`."""
        return slots if self.replay_slots is None else self.replay_slots[slots.long()]

    def state(self, slots):
        rows = self.rows(slots).long()
        return [self.pool, *(x[rows] for x in self.old), self.pnat[rows], self.buf[rows]]


@pytest.mark.parametrize("separate", [False, True], ids=["coupled", "replay_slots"])
@pytest.mark.parametrize("n,t", [(1, 4), (4, 4), (32, 4), (5, 3)])
def test_cute_dispatch_matches_the_plain_call(n, t, separate):
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
    )
    pnat = torch.randint(0, HIST + 1, (SLOTS,), generator=cpu, dtype=torch.int32).to(DEV)
    buf = torch.randint(0, 2, (SLOTS,), generator=cpu, dtype=torch.int32).to(DEV)
    gating = (
        (torch.randn(HV, generator=gen, device=DEV) * 1.5 + 1.0).clamp(-7.0, 5.0),
        (torch.randn(HV, generator=gen, device=DEV) * 1.7 - 4.4).clamp(-11.0, 2.5),
    )
    # every pool slot's history at a replay slot of its own, none of them a pool slot
    replay_slots = (torch.randperm(REPLAY_SLOTS - SLOTS, generator=cpu)[:SLOTS] + SLOTS).to(
        DEV, torch.int32
    )
    plain = _Side(pool, old, pnat, buf, t, gating)
    disp = _Side(pool, old, pnat, buf, t, gating, replay_slots if separate else None)
    wraps = 0
    for step in range(HIST // t + 3):
        wraps += int((plain.pnat[slots.long()] + t > HIST).sum())
        replay = M._replay_work(
            disp.core, SimpleNamespace(replay_state_indices=disp.rows(slots)), n
        )
        assert replay[:3] == (None, None, False), "no all-layer commit: the kernel commits itself"
        assert torch.equal(replay[3], disp.rows(slots))
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
            (None, None, False, None),
        )
        got = M._replay_verify(
            disp.core, mq_out, a, b, disp.pool, 0, disp.layer, slots, n, t, replay, pdl=True
        )
        torch.cuda.synchronize()
        assert torch.equal(got, want), f"step {step}: outputs differ"
        names = ("pool", "u", "k", "G", "pnat", "buf")
        for name, x, y in zip(names, disp.state(slots), plain.state(slots)):
            assert torch.equal(x, y), f"step {step}: {name} differs"
        accepted = torch.randint(1, t + 1, (n,), generator=cpu, dtype=torch.int32).to(DEV)
        for side in (plain, disp):
            _advance_replay_state(side.meta, side.rows(slots), accepted)
    assert wraps > 0, "no history wrapped: the in-kernel commit was never exercised"
