# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An attention-DP rank's lean verify conv (`DP_GDN_CONV_LEAN_ENV`, CPU).

The build's gating, the per-step decision (rows, the `DP_LOWROWS_ENV` floor), the wrapped
promotion (the lean flag consumed once, the stock method otherwise, a mismatch raised), that a
lean step's verify runs `gdn_conv_lean` and never the fused GEMV + conv, and that the gated norm
is the PDL launch. The kernels' bits against the stock pair are the GPU test
`ops/ssm/test_modeling_v2_gdn_conv_lean.py`; the bind's own self-check runs them on GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "env, dp, mtp, pdl, width, expected",
    [
        (None, True, True, True, 4, True),  # default on
        ("0", True, True, True, 4, False),
        (None, False, True, True, 4, False),  # tep4 / tp1 / dep2 / not SM103
        (None, True, False, True, 4, False),  # no MTP: no verify step
        (None, True, True, False, 4, False),  # PDL off
        (None, True, True, True, 3, False),  # another conv width
    ],
)
def test_build_gating(monkeypatch, env, dp, mtp, pdl, width, expected):
    if env is None:
        monkeypatch.delenv(M.DP_GDN_CONV_LEAN_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_GDN_CONV_LEAN_ENV, env)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp)
    monkeypatch.setattr(M, "_GDN_REPLAY_PDL", pdl)
    core = SimpleNamespace(mtp_enabled=mtp, conv_kernel=width)
    assert M._gdn_lean_build(core, None) is expected


def _lean_core(max_batch=32, t=4):
    return SimpleNamespace(
        dp_rank=0,
        _gdn_step=M._GdnStep(),
        _gdn_xsave_all=torch.zeros(2, max_batch, t, 8),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "lowrows, gens, p, expected",
    [
        (None, 5, 4, True),  # 20 rows
        (None, 4, 4, False),  # 16 rows: below the floor
        (None, 64, 4, True),  # 256 rows
        (None, 65, 4, False),  # past the ceiling (and past xsave)
        (None, 8, 2, False),  # rows a request other than xsave's
        ("2", 2, 4, True),  # DP_LOWROWS=2: from 8 rows
        ("2", 1, 4, False),
        ("3", 2, 4, False),
    ],
)
def test_step_decision(monkeypatch, lowrows, gens, p, expected):
    if lowrows is None:
        monkeypatch.delenv(M.DP_LOWROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_LOWROWS_ENV, lowrows)
    monkeypatch.setattr(M, "_gdn_lean_seen", lambda *a: None)  # its capture probe needs CUDA
    core = _lean_core(max_batch=64)
    M._gdn_lean_step(core, gens, gens * p, p)
    assert core._gdn_step.lean is expected
    assert core._gdn_step.gens == (gens if expected else 0)


class _State:
    """A stand-in for `GDNReplayState`: what the wrap reads, and the stock promotion it keeps."""

    def __init__(self):
        self.calls = []
        self._stacked_conv = torch.zeros(2, 4, 8, 3)

    def _promote_conv_states_and_advance_replay(self, batch, metadata, replay_indices):
        self.calls.append(("stock", batch.num_generations))


def _batch(gens):
    return SimpleNamespace(
        num_generations=gens,
        source_state_indices=torch.arange(gens),
        destination_state_indices=torch.arange(gens),
        num_accepted_tokens=torch.ones(gens, dtype=torch.int32),
        is_dummy_request=torch.zeros(gens, dtype=torch.bool),
    )


def _meta(width=4):
    return SimpleNamespace(
        prev_num_accepted_tokens=torch.zeros(4, dtype=torch.int32),
        cache_buf_idx=torch.zeros(4, dtype=torch.int32),
        replay_step_width=width,
        replay_history_size=16,
    )


@pytest.mark.cpu_only
def test_wrapped_promotion_consumes_the_lean_flag_once(monkeypatch):
    seen = []
    monkeypatch.setattr(M, "gdn_promote_x", lambda *a, **kw: seen.append((a, kw)))
    core, state = _lean_core(), _State()
    M._gdn_lean_wrap(core, state)
    core._gdn_lean_state_ref = state
    replay = torch.tensor([3, 1, 0, 2, 5])
    # Not lean: the stock promotion.
    state._promote_conv_states_and_advance_replay(_batch(5), _meta(), replay)
    assert state.calls == [("stock", 5)] and not seen
    # Lean: promote_x over the promotion's replay slots, then the flag is gone.
    core._gdn_step.lean, core._gdn_step.gens = True, 5
    state._promote_conv_states_and_advance_replay(_batch(5), _meta(), replay)
    assert len(seen) == 1 and seen[0][1]["replay_slots"] is replay
    assert seen[0][0][0] is state._stacked_conv and seen[0][0][1] is core._gdn_xsave_all
    assert not core._gdn_step.lean
    state._promote_conv_states_and_advance_replay(_batch(5), _meta(), replay)
    assert state.calls == [("stock", 5), ("stock", 5)] and len(seen) == 1
    assert getattr(state, M._GDN_LEAN_WRAPPED) is core


@pytest.mark.cpu_only
@pytest.mark.parametrize("gens, width", [(6, 4), (5, 2)])
def test_wrapped_promotion_raises_on_a_mismatch(monkeypatch, gens, width):
    monkeypatch.setattr(M, "gdn_promote_x", lambda *a, **kw: None)
    core, state = _lean_core(), _State()
    M._gdn_lean_wrap(core, state)
    core._gdn_lean_state_ref = state
    core._gdn_step.lean, core._gdn_step.gens = True, 5
    with pytest.raises(RuntimeError, match="the lean step carried 5 generation requests"):
        state._promote_conv_states_and_advance_replay(_batch(gens), _meta(width), None)


@pytest.mark.cpu_only
def test_a_state_this_core_no_longer_binds_takes_the_stock_promotion(monkeypatch):
    """Capacity estimation binds a throwaway manager first: its wrapped state must not take a
    lean step of the manager bound after it."""
    monkeypatch.setattr(M, "gdn_promote_x", lambda *a, **kw: pytest.fail("promote_x"))
    core, old, new = _lean_core(), _State(), _State()
    M._gdn_lean_wrap(core, old)
    core._gdn_lean_state_ref = new
    core._gdn_step.lean, core._gdn_step.gens = True, 5
    old._promote_conv_states_and_advance_replay(_batch(5), _meta(), None)
    assert old.calls == [("stock", 5)] and not core._gdn_step.lean


def _verify_core(lean):
    n, p = 6, 4
    pools = SimpleNamespace(ssm={0: torch.zeros(1, dtype=torch.bfloat16)}, conv={0: "pool0"})
    replay = SimpleNamespace(
        ring=None, old={0: (None, None, None, None)}, cache_buf_idx=None, prev_num_accepted=None,
        history=16,
    )  # fmt: skip
    step = M._GdnStep()
    step.lean, step.gens = lean, n if lean else 0
    core = SimpleNamespace(
        _bound=SimpleNamespace(pools=pools, replay=replay, int_conv={0: "snaps0"}),
        gdn_replay_map=False,
        conv_dim=48,
        key_dim=16,
        value_dim=16,
        num_k_heads=1,
        head_k_dim=16,
        num_v_heads=1,
        head_v_dim=16,
        _gdn_step=step,
        _gdn_xsave={0: "xsave0"},
    )
    return core, torch.zeros(n * p, 48), torch.zeros(n * p, 1), n, p


@pytest.mark.cpu_only
@pytest.mark.parametrize("lean", [True, False])
def test_verify_runs_the_lean_conv_on_a_lean_step(monkeypatch, lean):
    convs = []
    monkeypatch.setattr(M, "_GDN_REPLAY_PDL", True)
    monkeypatch.setattr(
        M,
        "gdn_conv_lean",
        lambda mq, w, pool, slots, rows, xsave, p, **kw: convs.append(("lean", pool, xsave))
        or torch.zeros(mq.shape[0], 48),
    )
    monkeypatch.setattr(
        M,
        "causal_conv1d_triton_update",
        lambda **kw: convs.append(("stock", kw["conv_state"], kw["intermediate_conv_window"]))
        or kw["x"].clone(),
    )
    monkeypatch.setattr(
        M,
        "gdn_cached_replay_cute",
        lambda **kw: torch.zeros(kw["q"].shape[0] * kw["q"].shape[1] * 16),
    )
    core, mq, a, n, p = _verify_core(lean)
    verify = ("int_idx", None, None, False, None)
    L = SimpleNamespace(conv_w="taps", a_log=None, dt_bias=None)
    M._gdn_verify(core, mq, a, a, 0, L, "slots", n, p, verify)
    assert convs == ([("lean", "pool0", "xsave0")] if lean else [("stock", "pool0", "snaps0")])


@pytest.mark.cpu_only
def test_step_clear_tolerates_a_core_without_the_form():
    core = SimpleNamespace()
    M._gdn_step_clear(core)
    assert M._gdn_lean_now(core) is False
    core._gdn_step = M._GdnStep()
    core._gdn_step.lean = True
    assert M._gdn_lean_now(core) is True
    M._gdn_step_clear(core)
    assert M._gdn_lean_now(core) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("want, expected", [(False, None), (True, "absent")])
def test_bind_without_the_wish_touches_nothing(want, expected):
    """A bind with the switch off leaves the core as it was (a core built before this form
    has none of its attributes)."""
    core = SimpleNamespace(_gdn_lean_want=want)
    if want:
        core._gdn_step = M._GdnStep()
        core.gdn_layers = [0]
        core._bound = SimpleNamespace(replay=None, int_conv=None, int_idx=None)
        core.dp_rank = 0
        M._gdn_lean_bind(core, SimpleNamespace(_speculative_state=None))
        assert core._gdn_lean_on is False and core._gdn_xsave_all is None
    else:
        M._gdn_lean_bind(core, None)
        assert not hasattr(core, "_gdn_lean_on")


class _ReplayState:
    """Laid out as the bind requires: 2 GDN layers, a stacked conv pool and snapshot buffer the
    pool views alias, linear chains, step width 4."""

    def __init__(self, c=8):
        self._linear_steps = True
        bf16 = torch.bfloat16
        self._stacked_conv = torch.zeros(6, 2, c, 3, dtype=bf16).permute(1, 0, 2, 3)
        self.intermediate_conv = torch.zeros(2, 4, 4, c, 3, dtype=bf16)

    def get_replay_metadata(self):
        return SimpleNamespace(replay_step_width=4, replay_history_size=16)


def _state_core(state, ring=None, taps=None):
    c = 8
    pools = SimpleNamespace(
        conv={0: state._stacked_conv[0], 1: state._stacked_conv[1]},
        ssm={0: torch.zeros(1, dtype=torch.bfloat16)},
    )
    bound = SimpleNamespace(
        replay=SimpleNamespace(ring=ring),
        int_conv={0: state.intermediate_conv[0], 1: state.intermediate_conv[1]},
        int_idx=torch.zeros(4, dtype=torch.int32),
        pools=pools,
    )
    w = torch.zeros(c, 4, dtype=torch.bfloat16) if taps is None else taps
    layers = {0: SimpleNamespace(conv_w=w), 1: SimpleNamespace(conv_w=w)}
    return SimpleNamespace(
        _bound=bound, gdn_layers=[0, 1], conv_dim=c, _decode=SimpleNamespace(_layers=layers)
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case, reason",
    [
        ("ok", ""),
        ("ring", "FlashInfer's ring replay"),
        ("taps", "conv taps"),
        ("chains", "linear draft chains"),
        ("alias", "is not layer"),
    ],
)
def test_bind_checks_the_managers_layout(monkeypatch, case, reason):
    import sys

    stub = sys.modules["tensorrt_llm._torch.modules.fla.cache_manager"]
    monkeypatch.setattr(stub, "GDNReplayState", _ReplayState, raising=False)
    state = _ReplayState()
    core = _state_core(
        state,
        ring="ring" if case == "ring" else None,
        taps=torch.zeros(8, 4, dtype=torch.float32) if case == "taps" else None,
    )
    if case == "chains":
        state._linear_steps = False
    if case == "alias":
        core._bound.pools.conv[1] = torch.zeros(6, 8, 3, dtype=torch.bfloat16)
    got, why = M._gdn_lean_state(core, SimpleNamespace(_speculative_state=state))
    if case == "ok":
        assert got is state and why == ""
    else:
        assert got is None and reason in why
