# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for REJ_MC: the spec sampler's chain speculative sampling as
``torch.ops.trtllm.minimax_m31_chain_speculative_sampling_mc`` through ``ops/sampling/m31_chain_spec_mc``, installed as
the rejection sampler's override.

It must equal the stock ``torch.ops.trtllm.chain_speculative_sampling`` bit for bit, tokens and counts
(the microbenchmark generators: realistic / same / far / ties / onehot drafts and targets, K7). The shared hook in
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

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.sampling import (  # noqa: E501
    m31_chain_spec_mc as entry,
)
from tensorrt_llm._torch.speculative import interface as spec_interface

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4"
)
CASES = ("realistic", "same", "far", "ties", "onehot")
op = entry.m31_chain_spec_mc


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("REJ_MC is an sm_103 kernel")
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
    """The microbenchmark input generators."""
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
    entry call at ctas_per_row 1 / 4 / 8 gives the same result again."""
    _gpu()
    batch, k, v = cell.spec["batch"], cell.spec["k"], cell.spec["v"]
    for seed in range(4):
        args = _make(case, batch, k, v, 1000 * seed + batch)
        got = entry.chain_spec_mc(*args)
        assert got is not None, "an in-contract input must be served"
        _same(got, _stock(*args), f"{case} batch={batch} seed={seed}")
        for ctas in (1, 4, 8):
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
    assert entry.ctas_per_row(1, 148) == 16 and entry.ctas_per_row(8, 148) == 16
    assert entry.ctas_per_row(16, 148) == 8 and entry.ctas_per_row(64, 148) == 2
    assert entry.ctas_per_row(200, 148) == 1


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


@pytest.mark.parametrize("state", ["none", "scalar", "mixed", "nondeterministic"])
def test_ineligible_rng_state_stays_stock(_fake_stock_op, state) -> None:
    fake_op, stock = _fake_stock_op
    args = _cpu_args()
    override = mock.Mock()
    spec_interface.set_chain_speculative_sampling_override(override)
    seed, offset = args[3:]
    if state == "none":
        seed = offset = None
    elif state == "scalar":
        seed, offset = 17, 0
    elif state == "mixed":
        offset = 0
    result = spec_interface.rejection_sampling_one_model(
        *args[:3], deterministic=state != "nondeterministic", seed=seed, offset=offset
    )
    override.assert_not_called()
    fake_op.assert_called_once()
    assert result[0] is stock[0]
    assert torch.equal(result[1], stock[1])


def test_the_target_installs_the_override() -> None:
    """With REJ_MC on (the default) and the op built, the target installs ``chain_spec_mc`` once per process."""
    _gpu()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (  # noqa: E501
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
    finally:
        spec_interface.set_chain_speculative_sampling_override(None)
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
