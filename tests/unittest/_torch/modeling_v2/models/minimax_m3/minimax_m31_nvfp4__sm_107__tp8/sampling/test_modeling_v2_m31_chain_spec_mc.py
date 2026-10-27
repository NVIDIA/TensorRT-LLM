# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for REJ_MC: the spec sampler's chain speculative sampling as
``torch.ops.trtllm.minimax_m31_chain_speculative_sampling_mc`` through ``ops/sampling/m31_chain_spec_mc``, installed as
the rejection sampler's override.

It must equal the stock ``torch.ops.trtllm.chain_speculative_sampling`` bit for bit, tokens and counts (the microbench
generators: realistic / same / far / ties / onehot drafts and targets, K7). The shared hook in
``speculative/interface.rejection_sampling_one_model`` must be stock while no override is set, and fall back to the
stock op whenever the override declines. TRTLLM_M31_DISABLE_REJ_MC=1 installs nothing.
"""

import os
import subprocess
import sys
import types
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_chain_spec_mc as entry,
)
from tensorrt_llm._torch.speculative import interface as spec_interface

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8"
)
CASES = ("realistic", "same", "far", "ties", "onehot")
op = entry.m31_chain_spec_mc


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("REJ_MC is built for sm_107 here")
    if entry.chain_spec_mc_op() is None:
        pytest.skip("minimax_m31_chain_speculative_sampling_mc is not built")
    if not hasattr(torch.ops.trtllm, "chain_speculative_sampling"):
        pytest.skip("the stock chain_speculative_sampling op is not built")
    pytest.importorskip("flashinfer.sampling")


def _probs(logits, top_p=0.95):
    """softmax then FlashInfer's top-p renorm, as the sampler feeds the rejection step."""
    import flashinfer.sampling

    v = logits.shape[-1]
    p = torch.softmax(logits, -1)
    renormed = flashinfer.sampling.top_p_renorm_probs(p.view(-1, v).contiguous(), top_p)
    return renormed.view(logits.shape).contiguous()


def _make(case, batch, k, v, seed):
    """The microbench generators."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rl(shape, scale=3.0):
        return torch.randn(*shape, v, generator=g, device="cuda") * scale

    dl = rl((batch, k))
    if case == "realistic":  # independent draft / target
        tl = torch.cat([dl, rl((batch, 1))], 1) + rl((batch, k + 1), 1.0)
    elif case == "same":  # target == draft: all-accept and zero-residual paths
        tl = torch.cat([dl, rl((batch, 1))], 1)
    elif case == "far":  # target peaked elsewhere: early reject
        tl = rl((batch, k + 1))
        tl[:, :, :8] += 20
    elif case == "ties":
        dl = torch.round(dl * 2) / 2
        tl = torch.round(torch.cat([dl, rl((batch, 1))], 1) * 2) / 2
    elif case == "onehot":
        tl = torch.full((batch, k + 1, v), -1e4, device="cuda")
        tl[:, :, 1234] = 0
    else:
        raise ValueError(case)
    dp, tp = _probs(dl), _probs(tl)
    ids = torch.multinomial(dp.view(-1, v), 1, generator=g).view(batch, k).to(torch.int32)
    seeds = torch.randint(0, 2**31, (batch,), generator=g, device="cuda", dtype=torch.int64)
    offsets = torch.randint(0, 2**20, (batch,), generator=g, device="cuda", dtype=torch.int64)
    return dp, ids.contiguous(), tp, seeds, offsets


def _stock(*args):
    return torch.ops.trtllm.chain_speculative_sampling(*args)


def _same(got, ref, what):
    assert torch.equal(got[0], ref[0]), f"{what}: accepted tokens differ"
    assert torch.equal(got[1], ref[1]), f"{what}: token counts differ"


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_bitwise_stock_op(cell, case) -> None:
    """At each certified (batch, K, vocab): the override equals the stock op (tokens and counts), and the explicit
    entry call at ctas_per_row 1 / 4 / 8 / 16 / 32 / 64 gives the same result again (REJ v2: the grid size never changes
    the result)."""
    _gpu()
    batch, k, v = cell.spec["batch"], cell.spec["k"], cell.spec["v"]
    for seed in range(4):
        args = _make(case, batch, k, v, 1000 * seed + batch)
        got = entry.chain_spec_mc(*args)
        assert got is not None, "an in-contract input must be served"
        _same(got, _stock(*args), f"{case} batch={batch} seed={seed}")
        for ctas in (1, 4, 8, 16, 32, 64):
            if ctas > 1 and batch * ctas > entry.sm_count(args[0].device):
                continue
            _same(op(*args, entry.prebuild(args[0].device), ctas), got, f"ctas_per_row={ctas}")


def test_graph_replay_equals_eager() -> None:
    """Captured in a CUDA graph (workspace prebuilt outside the capture), the replay gives the eager result."""
    _gpu()
    args = _make("realistic", 8, 7, 200064, 7)
    entry.prebuild(args[0].device)
    ref = entry.chain_spec_mc(*args)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        entry.chain_spec_mc(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.chain_spec_mc(*args)
    assert out is not None, "a prebuilt workspace must serve a captured call"
    for _ in range(3):
        out[0].fill_(-7)
        out[1].fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        _same(out, ref, "graph replay")


def test_outside_the_contract_the_override_declines() -> None:
    """Non-fp32 probs, non-int32 draft tokens, non-int64 or mis-sized seed / offset, a vocab not divisible by 4, a
    misaligned or non-contiguous probs, mismatched shapes, CPU inputs or more requests than MAX_BATCH return None (the
    sampler runs the stock op)."""
    _gpu()
    dp, ids, tp, seeds, offsets = _make("realistic", 2, 3, 32768, 1)
    assert entry.chain_spec_mc(dp, ids, tp, seeds, offsets) is not None
    assert entry.chain_spec_mc(dp.double(), ids, tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp.half(), seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids.long(), tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp, seeds.int(), offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp, seeds[:1], offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp, seeds, offsets[:1]) is None
    assert entry.chain_spec_mc(dp, ids[:, :2].contiguous(), tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp[:, :3].contiguous(), seeds, offsets) is None
    assert entry.chain_spec_mc(dp[..., :-4], ids, tp[..., :-4], seeds, offsets) is None
    assert (
        entry.chain_spec_mc(
            dp[..., :-2].contiguous(), ids, tp[..., :-2].contiguous(), seeds, offsets
        )
        is None
    )
    flat = torch.empty(dp.numel() + 1, device="cuda")
    assert entry.chain_spec_mc(flat[1:].view(dp.shape), ids, tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp.cpu(), ids.cpu(), tp.cpu(), seeds.cpu(), offsets.cpu()) is None
    b = entry.MAX_BATCH + 1
    wide = _make("realistic", b, 1, 32768, 2)
    assert entry.chain_spec_mc(*wide) is None
    # More draft tokens than the kernel's shared acceptance probabilities hold.
    deep = _make("realistic", 1, entry.MAX_DRAFT + 1, 32768, 3)
    assert entry.chain_spec_mc(*deep) is None
    edge = _make("realistic", 1, entry.MAX_DRAFT, 32768, 4)
    _same(entry.chain_spec_mc(*edge), _stock(*edge), "k = MAX_DRAFT")
    assert entry.ctas_per_row(1, 148) == 64 and entry.ctas_per_row(2, 148) == 64
    assert entry.ctas_per_row(4, 148) == 32 and entry.ctas_per_row(8, 148) == 16
    assert entry.ctas_per_row(16, 148) == 8 and entry.ctas_per_row(64, 148) == 2
    assert entry.ctas_per_row(200, 148) == 1
    # This device's own SM count keeps the largest grid co-resident.
    sms = entry.sm_count(torch.device("cuda"))
    assert entry.MAX_BATCH * entry.ctas_per_row(entry.MAX_BATCH, sms) <= sms


@pytest.fixture
def _fake_stock_op(monkeypatch):
    """Record the feature FlashInfer fallback; retain real torch tensor semantics."""
    stock = (torch.zeros(2, 4, dtype=torch.int32), torch.ones(2, dtype=torch.int32))
    fake_op = mock.Mock(return_value=(stock[0], None, stock[1] - 1))
    fake_fi = types.SimpleNamespace(
        __version__="0.6.4", sampling=types.SimpleNamespace(chain_speculative_sampling=fake_op)
    )
    monkeypatch.setattr(spec_interface, "flashinfer", fake_fi, raising=False)
    monkeypatch.setattr(spec_interface, "IS_FLASHINFER_AVAILABLE", True)
    spec_interface.set_chain_speculative_sampling_override(None)
    yield fake_op, stock
    spec_interface.set_chain_speculative_sampling_override(None)


def _cpu_args():
    return (
        torch.rand(2, 3, 8),
        torch.zeros(2, 3, dtype=torch.int32),
        torch.rand(2, 4, 8),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
    )


def test_hook_unset_is_stock(_fake_stock_op) -> None:
    """No override: rejection_sampling_one_model makes exactly the stock op call and returns its result."""
    fake_op, stock = _fake_stock_op
    args = _cpu_args()
    assert spec_interface._CHAIN_SPEC_OVERRIDE is None
    result = spec_interface.rejection_sampling_one_model(*args[:3], seed=args[3], offset=args[4])
    assert result[0] is stock[0]
    assert torch.equal(result[1], stock[1])
    fake_op.assert_called_once()
    assert all(a is b for a, b in zip(fake_op.call_args.args, args[:3]))
    assert fake_op.call_args.kwargs["seed"] is args[3]
    assert fake_op.call_args.kwargs["offset"] is args[4]
    assert fake_op.call_args.kwargs["deterministic"] is True
    assert fake_op.call_args.kwargs["generator"] is None


def test_hook_override_result_is_used(_fake_stock_op) -> None:
    """An override's result replaces the stock op; the stock op does not run."""
    fake_op, _ = _fake_stock_op
    mine = (torch.ones(2, 4, dtype=torch.int32), torch.full((2,), 4, dtype=torch.int32))
    override = mock.Mock(return_value=mine)
    spec_interface.set_chain_speculative_sampling_override(override)
    args = _cpu_args()
    assert (
        spec_interface.rejection_sampling_one_model(*args[:3], seed=args[3], offset=args[4]) is mine
    )
    override.assert_called_once_with(*args)
    fake_op.assert_not_called()


def test_hook_override_declining_falls_back(_fake_stock_op) -> None:
    """An override that returns None leaves the stock op call in place."""
    fake_op, stock = _fake_stock_op
    override = mock.Mock(return_value=None)
    spec_interface.set_chain_speculative_sampling_override(override)
    args = _cpu_args()
    result = spec_interface.rejection_sampling_one_model(*args[:3], seed=args[3], offset=args[4])
    assert result[0] is stock[0]
    assert torch.equal(result[1], stock[1])
    assert override.call_count == 1 and fake_op.call_count == 1


def test_the_target_installs_the_override() -> None:
    """With REJ_MC on (the default) and the op built, the target installs ``chain_spec_mc`` once per process."""
    _gpu()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
        modeling,
    )

    if not modeling.REJ_MC:
        pytest.skip("REJ_MC is switched off in this environment")
    saved = dict(modeling._REJ_MC_STATE)
    modeling._REJ_MC_STATE.clear()
    try:
        modeling._install_rej_mc()
        assert spec_interface._CHAIN_SPEC_OVERRIDE is entry.chain_spec_mc
        assert modeling._REJ_MC_STATE["installed"] is True
        if modeling.REJ_SLOTS and entry.chain_spec_mc_slots_op() is not None:
            assert spec_interface._CHAIN_SPEC_SLOT_OVERRIDE is entry.chain_spec_mc_slots
    finally:
        spec_interface.set_chain_speculative_sampling_override(None)
        spec_interface.set_chain_speculative_sampling_slot_override(None)
        modeling._REJ_MC_STATE.clear()
        modeling._REJ_MC_STATE.update(saved)


def test_the_kill_switch_installs_nothing() -> None:
    """TRTLLM_M31_DISABLE_REJ_MC=1 (read at import): REJ_MC is off and the target leaves the rejection hook unset."""
    probe = (
        f"import {TARGET}.modeling as m\n"
        "from tensorrt_llm._torch.speculative import interface as i\n"
        "m._install_rej_mc()\n"
        "print(m.REJ_MC, i._CHAIN_SPEC_OVERRIDE is None, m._REJ_MC_STATE['installed'])"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_REJ_MC="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False True False"


def _slot_case(case, batch, k, v, seed, slots=16, max_draft=8, v_pad=64):
    """A spec-metadata-like slot buffer [slots, max_draft, v + v_pad] with the request rows' draft probs at random
    distinct slots (the other cells hold junk), plus the gathered contiguous copy the sampler would pass."""
    dp, ids, tp, seeds, offs = _make(case, batch, k, v, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    buf = torch.rand(slots, max_draft, v + v_pad, generator=g, device="cuda")
    slot_ids = (
        torch.randperm(slots, generator=g, device="cuda")[:batch].to(torch.int64).contiguous()
    )
    buf[slot_ids, :k, :v] = dp
    gathered = buf[slot_ids, :k, :v].contiguous()
    assert torch.equal(gathered, dp)
    return buf, slot_ids, gathered, ids, tp, seeds, offs


def _slots_gpu():
    _gpu()
    if entry.chain_spec_mc_slots_op() is None:
        pytest.skip("minimax_m31_chain_speculative_sampling_mc_slots is not built")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_slots_bitwise_the_gathered_call(case, batch) -> None:
    """REJ_SLOTS: reading the draft rows in place by slot (padded draft length and vocab stride, junk in the other
    cells) gives exactly the gathered call's tokens and counts, and the stock op's."""
    _slots_gpu()
    k, v = 7, 200064
    for seed in range(3):
        buf, slot_ids, gathered, ids, tp, seeds, offs = _slot_case(
            case, batch, k, v, 100 * seed + batch
        )
        got = entry.chain_spec_mc_slots(buf, slot_ids, k, ids, tp, seeds, offs)
        assert got is not None, "an in-contract slot call must be served"
        _same(
            got,
            entry.chain_spec_mc(gathered, ids, tp, seeds, offs),
            f"{case} batch={batch} seed={seed} vs gathered",
        )
        _same(
            got,
            _stock(gathered, ids, tp, seeds, offs),
            f"{case} batch={batch} seed={seed} vs stock",
        )


def test_slots_graph_replay_with_new_slots() -> None:
    """Captured with the slot ids as a device tensor: a replay after the slots change reads the new rows."""
    _slots_gpu()
    batch, k, v = 2, 7, 200064
    buf, slot_ids, gathered, ids, tp, seeds, offs = _slot_case("realistic", batch, k, v, 5)
    entry.prebuild(buf.device)
    entry.chain_spec_mc_slots(buf, slot_ids, k, ids, tp, seeds, offs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.chain_spec_mc_slots(buf, slot_ids, k, ids, tp, seeds, offs)
    for rep in range(3):
        g = torch.Generator(device="cuda").manual_seed(50 + rep)
        new_slots = torch.randperm(buf.shape[0], generator=g, device="cuda")[:batch].to(torch.int64)
        slot_ids.copy_(new_slots)
        graph.replay()
        torch.cuda.synchronize()
        ref = _stock(buf[slot_ids, :k, :v].contiguous(), ids, tp, seeds, offs)
        _same(out, ref, f"graph replay {rep}")


def test_slots_outside_the_contract_decline() -> None:
    _slots_gpu()
    buf, slot_ids, gathered, ids, tp, seeds, offs = _slot_case("realistic", 2, 7, 200064, 9)
    assert (
        entry.chain_spec_mc_slots(buf, slot_ids.int(), 7, ids, tp, seeds, offs) is None
    )  # int32 slots
    assert entry.chain_spec_mc_slots(buf.double(), slot_ids, 7, ids, tp, seeds, offs) is None
    assert (
        entry.chain_spec_mc_slots(buf, slot_ids, 9, ids, tp, seeds, offs) is None
    )  # num_draft > max draft
    assert (
        entry.chain_spec_mc_slots(buf[:, :, :1000], slot_ids, 7, ids, tp, seeds, offs) is None
    )  # narrow / strided


def test_the_slots_kill_switch_installs_no_slot_hook() -> None:
    """TRTLLM_M31_DISABLE_REJ_SLOTS=1 (read at import): REJ_MC still installs, the slot-indexed hook stays unset."""
    probe = (
        f"import {TARGET}.modeling as m\n"
        "from tensorrt_llm._torch.speculative import interface as i\n"
        "m._install_rej_mc()\n"
        "print(m.REJ_SLOTS, i._CHAIN_SPEC_SLOT_OVERRIDE is None)"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_REJ_SLOTS="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False True"


def test_the_slots_hook_is_off_by_default_and_opt_in() -> None:
    """REJ_SLOTS is off by default (read at import); TRTLLM_M31_ENABLE_REJ_SLOTS=1 turns it on; the disable switch
    wins."""
    probe = f"import {TARGET}.modeling as m\nprint(m.REJ_SLOTS)"
    base = {k: v for k, v in os.environ.items() if not k.endswith("_REJ_SLOTS")}
    for extra, want in (
        ({}, "False"),
        ({"TRTLLM_M31_ENABLE_REJ_SLOTS": "1"}, "True"),
        ({"TRTLLM_M31_ENABLE_REJ_SLOTS": "1", "TRTLLM_M31_DISABLE_REJ_SLOTS": "1"}, "False"),
    ):
        out = subprocess.run(
            [sys.executable, "-c", probe],
            env=dict(base, **extra),
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip().splitlines()[-1] == want, (extra, out.stdout[-200:])


def test_rej_v2_taken_line_once_per_grid(monkeypatch) -> None:
    """REJ_V2's taken line names the grid a served call launched with, once per distinct ctas_per_row."""
    _gpu()
    from tensorrt_llm.logger import logger

    lines = []
    monkeypatch.setattr(logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(entry, "_V2_TAKEN", set())
    args = _make("realistic", 1, 7, 32768, 5)
    g = entry.ctas_per_row(1, entry.sm_count(args[0].device))
    for _ in range(2):
        assert entry.chain_spec_mc(*args) is not None
    taken = [m for m in lines if "REJ_V2 taken" in m]
    assert taken == [
        f"[M3 dispatch] REJ_V2 taken (max_ctas_per_row {entry.MAX_CTAS_PER_ROW}, ctas_per_row {g} at batch 1)"
    ], taken


def test_rej_v3_taken_line_once() -> None:
    """REJ v3 (the default kernel) logs one REJ_V3 taken line per process, with the lib's workspace bytes."""
    _gpu()
    from tensorrt_llm.logger import logger

    lines = []
    with (
        mock.patch.object(logger, "info", lambda msg, *a, **k: lines.append(msg)),
        mock.patch.object(entry, "_V2_TAKEN", set()),
    ):
        args = _make("realistic", 1, 7, 32768, 6)
        for _ in range(2):
            assert entry.chain_spec_mc(*args) is not None
    if entry.kernel_version() != 3:
        pytest.skip("this lib launches the v2 kernel")
    assert [m for m in lines if "REJ_V3 taken" in m] == [
        f"[M3 dispatch] REJ_V3 taken (ws_bytes_per_row {entry.ws_bytes_per_row()})"
    ], lines
    assert not [m for m in lines if "REJ_V2_KERNEL" in m], lines


def _v3_adversarial(case, batch, k, v, seed):
    """Rows aimed at v3's moved steps: the staged chain tails, both tile variants and the leader's tile walk."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dp = torch.softmax(torch.randn(batch, k, v, generator=g, device="cuda") * 3.0, -1)
    bonus = torch.softmax(torch.randn(batch, 1, v, generator=g, device="cuda") * 3.0, -1)
    if (
        case == "allaccept"
    ):  # target == draft: every draft token accepted, the target-only (bonus) path
        tp = torch.cat([dp, bonus], 1)
    elif case == "lasttile":  # the residual has mass only in the last tile
        tp = torch.cat([dp, bonus], 1)
        tp[:, :, (v - 1) // 4096 * 4096 :] += 1e-3
        tp = tp / tp.sum(-1, keepdim=True)
    elif (
        case == "dyadic"
    ):  # power-of-two masses: tile sums and prefixes land exactly on the threshold grid
        tp = torch.zeros(batch, k + 1, v, device="cuda")
        tp[:, :, : 4 * 4096] = 1.0 / (4 * 4096)
        dp = torch.zeros(batch, k, v, device="cuda")
        dp[:, :, 4096 : 3 * 4096] = 1.0 / (2 * 4096)
    elif case == "zerotail":  # the target has no mass past the third tile
        tp = torch.softmax(torch.randn(batch, k + 1, v, generator=g, device="cuda") * 3.0, -1)
        tp[:, :, 3 * 4096 :] = 0.0
        tp = tp / tp.sum(-1, keepdim=True)
    else:
        raise ValueError(case)
    dp, tp = dp.contiguous(), tp.contiguous()
    ids = (
        torch.multinomial(dp.clamp_min(1e-12).view(-1, v), 1, generator=g)
        .view(batch, k)
        .to(torch.int32)
    )
    seeds = torch.randint(0, 2**31, (batch,), generator=g, device="cuda", dtype=torch.int64)
    offsets = torch.randint(0, 2**20, (batch,), generator=g, device="cuda", dtype=torch.int64)
    return dp, ids.contiguous(), tp, seeds, offsets


@pytest.mark.parametrize("case", ["allaccept", "lasttile", "dyadic", "zerotail"])
@pytest.mark.parametrize("v", [200064, 32768 + 4, 4096 * 3 + 8])
def test_v3_adversarial_rows_bitwise(case, v) -> None:
    """REJ v3 equals the stock op on rows aimed at its moved steps, at every grid (1..64 CTAs per request), batch 1 / 2
    / 8, K 7."""
    _gpu()
    if case == "dyadic" and v < 4 * 4096:
        pytest.skip("the dyadic rows need 4 tiles")
    for batch in (1, 2, 8):
        for seed in range(8):
            args = _v3_adversarial(case, batch, 7, v, 100 * seed + batch)
            ref = _stock(*args)
            for ctas in (1, 2, 8, 16, 32, 64):
                if ctas > 1 and batch * ctas > entry.sm_count(args[0].device):
                    continue
                _same(
                    op(*args, entry.prebuild(args[0].device), ctas),
                    ref,
                    f"{case} v={v} batch={batch} seed={seed} ctas={ctas}",
                )


def test_rej_v3_kill_switch_runs_v2_bitwise() -> None:
    """TRTLLM_M31_DISABLE_REJ_V3=1 (read once by the lib): the op launches the v2 kernel, still bitwise the stock op."""
    _gpu()
    probe = (
        "import sys, torch\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "import test_modeling_v2_m31_chain_spec_mc as t\n"
        "ok = True\n"
        "for case in t.CASES:\n"
        "    args = t._make(case, 2, 7, 200064, 77)\n"
        "    got = t.op(*args, t.entry.prebuild(args[0].device), 16)\n"
        "    ref = t._stock(*args)\n"
        "    ok &= bool(torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]))\n"
        "print(t.entry.kernel_version(), ok)"
    )
    for extra, want in (({"TRTLLM_M31_DISABLE_REJ_V3": "1"}, "2 True"), ({}, "3 True")):
        env = {kk: vv for kk, vv in os.environ.items() if kk != "TRTLLM_M31_DISABLE_REJ_V3"}
        out = subprocess.run(
            [sys.executable, "-c", probe],
            env=dict(env, **extra),
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip().splitlines()[-1] == want, (
            extra,
            out.stdout[-300:],
            out.stderr[-500:],
        )
