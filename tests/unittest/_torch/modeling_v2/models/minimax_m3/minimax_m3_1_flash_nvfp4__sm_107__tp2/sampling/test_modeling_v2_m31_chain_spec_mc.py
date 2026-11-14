# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for REJ_MC: the spec sampler's chain speculative sampling as
``mv2_minimax_m3::minimax_m31_chain_speculative_sampling_mc`` through ``ops/sampling/m31_chain_spec_mc``, installed as
the rejection sampler's override where the engine provides one.

It must equal the stock ``torch.ops.trtllm.chain_speculative_sampling`` bit for bit, tokens and counts (realistic /
same / far / ties / onehot drafts and targets, K7). Without the engine's override hook the target installs nothing.
TRTLLM_M31_DISABLE_REJ_MC=1 installs nothing.
"""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.sampling import (  # noqa: E501
        m31_chain_spec_mc as entry,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2"
CASES = ("realistic", "same", "far", "ties", "onehot")
op = entry.m31_chain_spec_mc


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("REJ_MC is built for sm_107 here")
    assert entry.chain_spec_mc_op() is not None, (
        "minimax_m31_chain_speculative_sampling_mc is not built"
    )
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
    """The microbenchmark generators."""
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
            if ctas * batch <= entry.sm_count(args[0].device):
                _same(op(*args, entry.prebuild(args[0].device), ctas), got, f"ctas_per_row={ctas}")


def test_graph_replay_equals_eager() -> None:
    """Captured in a CUDA graph (workspace prebuilt outside the capture), the replay gives the eager result."""
    _gpu()
    args = _make("realistic", 8, 7, 200064, 7)
    entry.prebuild(args[0].device)
    ref = entry.chain_spec_mc(*args)
    stream = torch.cuda.Stream()
    # The eager warm-up on the side stream shares the persistent workspace with the default-stream
    # calls above (and reads their inputs): order it after them.
    stream.wait_stream(torch.cuda.current_stream())
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
    b = entry.max_batch() + 1
    wide = _make("realistic", b, 1, 32768, 2)
    assert entry.chain_spec_mc(*wide) is None
    assert entry.ctas_per_row(1, 148) == 16 and entry.ctas_per_row(8, 148) == 16
    assert entry.ctas_per_row(16, 148) == 8 and entry.ctas_per_row(64, 148) == 2
    assert entry.ctas_per_row(200, 148) == 1
    # This device's own SM count keeps the largest grid co-resident.
    sms = entry.sm_count(torch.device("cuda"))
    assert entry.MAX_BATCH * entry.ctas_per_row(entry.MAX_BATCH, sms) <= sms


def test_the_target_declines_without_the_override_hook(monkeypatch) -> None:
    """Without the rejection sampler's override hook in the engine the target installs nothing and keeps the stock
    op, even with REJ_MC on and the op built."""
    _gpu()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling,
    )

    monkeypatch.setattr(modeling, "_SET_CHAIN_SPEC_OVERRIDE", None)
    saved = dict(modeling._REJ_MC_STATE)
    modeling._REJ_MC_STATE.clear()
    try:
        modeling._install_rej_mc()
        assert modeling._REJ_MC_STATE["installed"] is False
    finally:
        modeling._REJ_MC_STATE.clear()
        modeling._REJ_MC_STATE.update(saved)


def test_the_kill_switch_installs_nothing() -> None:
    """TRTLLM_M31_DISABLE_REJ_MC=1 (read at import): REJ_MC is off and the target installs nothing."""
    probe = (
        f"import {TARGET}.modeling as m\n"
        "m._install_rej_mc()\n"
        "print(m.REJ_MC, m._REJ_MC_STATE['installed'])"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_REJ_MC="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False False"


def test_the_wide_kill_switch_restores_the_c8_cap() -> None:
    """MAX_BATCH 48 by default; TRTLLM_M31_DISABLE_REJ_MC_WIDE=1 restores the 8-request cap (a 9-request call keeps
    the stock op)."""
    _gpu()
    assert entry.WIDE and entry.max_batch() == entry.MAX_BATCH == 48
    probe = (
        f"import importlib, torch; m = importlib.import_module('{TARGET}.ops.sampling.m31_chain_spec_mc'); "
        "d = torch.softmax(torch.randn(9, 1, 32768, device='cuda'), -1); "
        "t = torch.softmax(torch.randn(9, 2, 32768, device='cuda'), -1); "
        "i = torch.zeros(9, 1, dtype=torch.int32, device='cuda'); "
        "s = torch.zeros(9, dtype=torch.int64, device='cuda'); "
        "print('CAP', m.max_batch(), m.chain_spec_mc(d, i, t, s, s.clone()) is None)"
    )
    for flag, want in (("1", "CAP 8 True"), ("0", "CAP 48 False")):
        out = subprocess.run(
            [sys.executable, "-c", probe],
            env=dict(os.environ, TRTLLM_M31_DISABLE_REJ_MC_WIDE=flag),
            capture_output=True,
            text=True,
            check=True,
        )
        assert [ln for ln in out.stdout.splitlines() if ln.startswith("CAP ")][-1] == want
