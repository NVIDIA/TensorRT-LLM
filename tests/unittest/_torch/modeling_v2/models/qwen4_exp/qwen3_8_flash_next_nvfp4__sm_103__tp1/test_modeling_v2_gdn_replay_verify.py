# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's Gated-DeltaNet verify step in both recurrent modes.

Drives the two targets' own `_gdn_verify` -- the prefill target's (generation rows
behind a context prefix) and the decode target's -- over consecutive verify steps of
this checkpoint's GDN geometry, with the cache manager's bookkeeping run in between as
`update_mamba_states` runs it: the conv window promotion in both modes, then either the
legacy per-step state promotion, or the cached replay's all-layer commit (partitioned
batches only) and `_advance_replay_state`.

Three things are checked at every step, with a random partial acceptance (or, in the
full-acceptance cases, every draft accepted):

* the replay mode's output against the legacy mode's on identical inputs and identical
  accepted histories: the two modes compute one recurrence with different kernels and
  state disciplines, so a mis-bound history buffer, a wrong partition condition or a
  skipped commit drifts off within a step;
* the prefill and the decode target's replay calls bitwise equal to each other (each on
  its own copy of the state);
* the conv pool identical across the modes (it is promoted the same way).
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
from tensorrt_llm._torch.modules.fla.cached_replay import (
    CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
    commit_gdn_cached_replay_history_layers,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import _build_replay_work_items_torch

assert torch.cuda.is_available(), "the GDN verify test requires a CUDA device"

DEV = "cuda"
H, HV, K, V = 16, 48, 128, 128
KEY_DIM, VALUE_DIM = H * K, HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CONV_KERNEL = 4
HIST = 16
LAYERS = 2
SLOTS = 40
STEPS = 7


def _steps(t):
    """Enough steps that every slot fills its history at least once: each step accepts
    at least one token, and a slot wraps once `pnat + t > HIST`."""
    return HIST - t + 3


BF16_ULP = 2.0**-8
#: Replay against legacy, per token block: the replay kernel rounds the history's update
#: vectors and keys to bf16 and rebuilds the state from them, where the legacy kernel
#: carries an fp32 state, so the two agree to a few bf16 ulps of the block.
ELEMENT_GATE = 6 * BF16_ULP
RMS_GATE = 1.5 * BF16_ULP


class _Mode:
    """One cache manager's worth of state for `LAYERS` GDN layers, in one mode, plus the
    `core` / `L` stand-ins `_gdn_verify` reads."""

    def __init__(self, replay, t, conv0, ssm0, weights, gating, max_batch):
        self.conv = [c.clone() for c in conv0]
        self.int_conv = [
            torch.zeros(max_batch, t, CONV_DIM, CONV_KERNEL - 1, dtype=torch.bfloat16, device=DEV)
            for _ in range(LAYERS)
        ]
        # V2 lays the layers out inside each slot; the all-layer commit is handed that
        # affine layout.
        self.backing = torch.stack(ssm0, 1).contiguous()
        self.ssm = [self.backing[:, i] for i in range(LAYERS)]
        bound = dict(
            conv=dict(enumerate(self.conv)),
            int_conv=dict(enumerate(self.int_conv)),
            int_ssm=None,
            replay=None,
        )
        if replay:
            self.old_x = torch.zeros(
                LAYERS, SLOTS, 2, HIST, HV, V, dtype=torch.bfloat16, device=DEV
            )
            self.old_B = torch.zeros(LAYERS, SLOTS, 2, HIST, H, K, dtype=torch.bfloat16, device=DEV)
            self.old_dt = torch.zeros(LAYERS, SLOTS, 2, HV, HIST, device=DEV)
            self.old_dA = torch.zeros(LAYERS, SLOTS, 2, HV, HIST, device=DEV)
            self.pnat = torch.zeros(SLOTS, dtype=torch.int32, device=DEV)
            self.buf = torch.zeros(SLOTS, dtype=torch.int32, device=DEV)
            bound["replay"] = M._Replay(
                old={
                    i: (self.old_x[i], self.old_B[i], self.old_dt[i], self.old_dA[i])
                    for i in range(LAYERS)
                },
                cache_buf_idx=self.buf,
                prev_num_accepted=self.pnat,
                history=HIST,
                width=t,
                all_layer_commit=True,
            )
            self.meta = ReplayStateUpdateMetadata(self.pnat, self.buf, t, HIST)
            self.mm = SimpleNamespace(
                replay_work_items=torch.zeros(max_batch, 4, dtype=torch.int32, device=DEV),
                replay_n_writes=torch.zeros(1, dtype=torch.int32, device=DEV),
            )
        else:
            self.int_ssm = [torch.zeros(max_batch, t, HV, V, K, device=DEV) for _ in range(LAYERS)]
            bound["int_ssm"] = dict(enumerate(self.int_ssm))
        self.core = SimpleNamespace(
            conv_dim=CONV_DIM,
            key_dim=KEY_DIM,
            value_dim=VALUE_DIM,
            num_k_heads=H,
            head_k_dim=K,
            num_v_heads=HV,
            head_v_dim=V,
            _bound=SimpleNamespace(**bound),
        )
        self.layers = [
            SimpleNamespace(conv_w=weights[i], a_log=gating[i][0], dt_bias=gating[i][1])
            for i in range(LAYERS)
        ]

    def work(self, slots, n, t):
        """What the forward hands `_gdn_verify` as `replay`: the metadata's prepare builds
        the work items, then `_replay_work` applies the manager's partition condition."""
        partitioned = n >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
        if partitioned:
            _build_replay_work_items_torch(
                slots,
                self.pnat,
                self.buf,
                self.mm.replay_work_items,
                self.mm.replay_n_writes,
                t,
                HIST,
            )
        replay = M._replay_work(self.core, self.mm, n)
        assert replay[2] == partitioned
        return replay

    def verify(self, target_cls, layer, mq, a, b, slots, int_idx, n, t, replay):
        return target_cls._gdn_verify(
            SimpleNamespace(core=self.core),
            mq,
            a,
            b,
            self.ssm[layer],
            layer,
            self.layers[layer],
            slots,
            int_idx,
            n,
            t,
            replay,
        )

    def commit(self, slots, n):
        """The manager's all-layer commit, then the replay bookkeeping advance is the
        caller's (it needs the accepted counts)."""
        commit_gdn_cached_replay_history_layers(
            ssm_states=self.ssm[0],
            ssm_state_num_layers=LAYERS,
            ssm_state_layer_stride=self.backing.stride(1),
            ssm_state_slot_stride=self.backing.stride(0),
            old_u=self.old_x,
            old_k=self.old_B,
            old_G=self.old_dt,
            replay_work_items=self.mm.replay_work_items[:n],
            n_writes=self.mm.replay_n_writes,
            history_size=HIST,
        )


def _rows(gen, rows):
    """One layer's verify-row activations: the pre-conv `[q|k|v]` projection and the raw
    `a` and `b` projections, each its own contiguous GEMM output."""
    return tuple(
        torch.randn(rows, width, generator=gen, device=DEV).to(torch.bfloat16)
        for width in (CONV_DIM, HV, HV)
    )


def _behind_prefix(x, prefix=5):
    """`x` as the trailing rows of a buffer that starts with `prefix` context rows."""
    buf = torch.zeros(prefix + x.shape[0], x.shape[1], dtype=x.dtype, device=DEV)
    buf[prefix:] = x
    return torch.split(buf, [prefix, x.shape[0]], 0)[1]


def _promote(pool, inter, slots, accepted):
    """`update_mamba_states`' copy: step `accepted - 1` of each batch row's per-step
    buffer into the row's slot."""
    rows = torch.arange(slots.shape[0], device=DEV)
    pool[slots.long()] = inter[rows, accepted.long() - 1]


def _gate(got, want, what):
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    scale = w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    worst = ((g - w).abs() / scale).max().item()
    rms = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()
    assert worst <= ELEMENT_GATE and rms <= RMS_GATE, (
        f"{what}: replay vs legacy {worst / BF16_ULP:.2f} ulps of the block, "
        f"RMS {rms / BF16_ULP:.2f} ulps"
    )


def _state(gen, cpu, n, t, gating=None):
    """Random weights, gating parameters and pre-step pools shared by the modes, and a
    random slot assignment of `n` requests."""
    slots = torch.randperm(SLOTS, generator=cpu)[:n].to(DEV, torch.int32)
    weights = [
        (0.3 * torch.randn(CONV_DIM, CONV_KERNEL, generator=gen, device=DEV)).to(torch.bfloat16)
        for _ in range(LAYERS)
    ]
    if gating is None:
        gating = [
            (
                (torch.randn(HV, generator=gen, device=DEV) * 1.5 + 1.0).clamp(-7.0, 5.0),
                (torch.randn(HV, generator=gen, device=DEV) * 1.7 - 4.4).clamp(-11.0, 2.5),
            )
            for _ in range(LAYERS)
        ]
    conv0 = [
        torch.randn(SLOTS, CONV_DIM, CONV_KERNEL - 1, generator=gen, device=DEV).to(torch.bfloat16)
        for _ in range(LAYERS)
    ]
    ssm0 = [0.3 * torch.randn(SLOTS, HV, V, K, generator=gen, device=DEV) for _ in range(LAYERS)]
    return slots, weights, gating, conv0, ssm0


@pytest.mark.parametrize(
    "n,t,accept",
    [
        # At the partition threshold, at max_draft_len 7 (T = 8, the replay kernel's
        # ceiling): the history wraps through the all-layer commit.
        (CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, 8, "random"),
        # Below the threshold, every draft accepted: from a fresh slot at T = 8 the second
        # step appends exactly up to the history's end (`pnat + T == 16`) and every second
        # step after it wraps in-kernel.
        (4, 8, "full"),
    ],
    ids=lambda v: str(v),
)
def test_replay_verify_tracks_legacy_in_both_targets(n, t, accept):
    gen = torch.Generator(device=DEV).manual_seed(1000 * n + t)
    cpu = torch.Generator().manual_seed(1000 * n + t)
    slots, weights, gating, conv0, ssm0 = _state(gen, cpu, n, t)
    int_idx = torch.arange(n, dtype=torch.int32, device=DEV)
    legacy = _Mode(False, t, conv0, ssm0, weights, gating, n)
    prefill = _Mode(True, t, conv0, ssm0, weights, gating, n)
    decode = _Mode(True, t, conv0, ssm0, weights, gating, n)
    partitioned = n >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE

    wraps = to_the_end = 0
    for step in range(_steps(t)):
        pending = prefill.pnat[slots.long()]
        wraps += int((pending + t > HIST).sum())
        to_the_end += int((pending + t == HIST).sum())
        replays = {id(mode): mode.work(slots, n, t) for mode in (prefill, decode)}
        for layer in range(LAYERS):
            mq, a, b = _rows(gen, n * t)
            # The prefill target's generation rows sit behind a context prefix: row views
            # at an offset into the batch's projections.
            mq_p, a_p, b_p = (_behind_prefix(x) for x in (mq, a, b))
            want = legacy.verify(M.DecodeTarget, layer, mq, a, b, slots, int_idx, n, t, None)
            got_p = prefill.verify(
                M.PrefillTarget, layer, mq_p, a_p, b_p, slots, int_idx, n, t, replays[id(prefill)]
            )
            got_d = decode.verify(
                M.DecodeTarget, layer, mq, a, b, slots, int_idx, n, t, replays[id(decode)]
            )
            assert got_p.shape == got_d.shape == want.shape == (1, n * t, HV, V)
            assert torch.equal(got_p, got_d), (
                f"step {step} layer {layer}: the prefill and decode targets' replay calls disagree"
            )
            _gate(got_d, want, f"step {step} layer {layer}")
        if accept == "full":
            accepted = torch.full((n,), t, dtype=torch.int32, device=DEV)
        else:
            accepted = torch.randint(1, t + 1, (n,), generator=cpu, dtype=torch.int32).to(DEV)
        for layer in range(LAYERS):
            for mode in (legacy, prefill, decode):
                _promote(mode.conv[layer], mode.int_conv[layer], slots, accepted)
            _promote(legacy.ssm[layer], legacy.int_ssm[layer], slots, accepted)
        for mode in (prefill, decode):
            if partitioned:
                mode.commit(slots, n)
            _advance_replay_state(mode.meta, slots, accepted)
        for layer in range(LAYERS):
            assert torch.equal(prefill.conv[layer], legacy.conv[layer])
            assert torch.equal(decode.conv[layer], legacy.conv[layer])
    assert wraps > 0, "the steps never filled a history, so the checkpoint never advanced"
    if accept == "full":
        assert wraps >= 3 * n and to_the_end >= 3 * n, (
            f"full acceptance at T={t} should wrap every row at least 3 times and append up to "
            f"the history's end as often ({wraps}, {to_the_end})"
        )


def test_a_wrong_partition_drifts_off_legacy():
    """The control: the work items withheld from a partitioned step while the manager
    still runs its all-layer commit -- the checkpoint committed twice -- must fail the
    same gate."""
    n, t = CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, 4
    gen = torch.Generator(device=DEV).manual_seed(7)
    cpu = torch.Generator().manual_seed(7)
    gating = [
        (torch.full((HV,), 0.5, device=DEV), torch.full((HV,), -4.0, device=DEV))
        for _ in range(LAYERS)
    ]
    slots, weights, gating, conv0, ssm0 = _state(gen, cpu, n, t, gating)
    int_idx = torch.arange(n, dtype=torch.int32, device=DEV)
    legacy = _Mode(False, t, conv0, ssm0, weights, gating, n)
    wrong = _Mode(True, t, conv0, ssm0, weights, gating, n)
    failed = False
    for _ in range(STEPS):
        # The manager's prepare builds the work items; the step does not receive them.
        _build_replay_work_items_torch(
            slots,
            wrong.pnat,
            wrong.buf,
            wrong.mm.replay_work_items,
            wrong.mm.replay_n_writes,
            t,
            HIST,
        )
        for layer in range(LAYERS):
            mq, a, b = _rows(gen, n * t)
            want = legacy.verify(M.DecodeTarget, layer, mq, a, b, slots, int_idx, n, t, None)
            got = wrong.verify(
                M.DecodeTarget, layer, mq, a, b, slots, int_idx, n, t, (None, None, False)
            )
            try:
                _gate(got, want, "control")
            except AssertionError:
                failed = True
        accepted = torch.randint(1, t + 1, (n,), generator=cpu, dtype=torch.int32).to(DEV)
        for layer in range(LAYERS):
            _promote(legacy.conv[layer], legacy.int_conv[layer], slots, accepted)
            _promote(wrong.conv[layer], wrong.int_conv[layer], slots, accepted)
            _promote(legacy.ssm[layer], legacy.int_ssm[layer], slots, accepted)
        wrong.commit(slots, n)
        _advance_replay_state(wrong.meta, slots, accepted)
    assert failed, "a doubled checkpoint commit passed the replay-vs-legacy gate"
