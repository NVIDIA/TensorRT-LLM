# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOPP_DET: the spec sampler's top-p renorm as
``torch.ops.trtllm.minimax_m31_topp_renorm_det`` through ``ops/sampling/m31_topp_det``, installed as the sampler's
top-p renorm override.

On rows with top_p < 1 it must equal FlashInfer's ``top_p_renorm_probs(probs, top_p, is_deterministic=True)`` bit for
bit (the microbench generators: randn / peaked / ties / near_uniform / all_equal rows); rows with top_p >= 1 keep every
token (an approved exception). The shared sampler hook must be byte-for-byte stock while no override is set, and
fall back to the stock call whenever the override declines. TRTLLM_M31_DISABLE_TOPP_DET=1 installs nothing.
"""

import os
import subprocess
import sys
import types
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_topp_det as entry,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as fi_ops

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8"
)
CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal")
FLT_MIN = torch.finfo(torch.float32).tiny
op = entry.m31_topp_det


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("TOPP_DET is built for sm_107 here")
    if entry.topp_det_op() is None:
        pytest.skip("minimax_m31_topp_renorm_det is not built")
    pytest.importorskip("flashinfer.sampling")


def _make(cls, rows, d, seed):
    """The microbench row generators: softmax of the class's logits."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if cls == "randn":
        logits = torch.randn(rows, d, generator=g, device="cuda") * 3
    elif cls == "peaked":  # few dominant tokens, long tail (real-like draft rows)
        logits = torch.randn(rows, d, generator=g, device="cuda") * 2
        idx = torch.randint(0, d, (rows, 8), generator=g, device="cuda")
        logits.scatter_(1, idx, 14 + 3 * torch.rand(rows, 8, generator=g, device="cuda"))
    elif cls == "ties":
        logits = torch.round(torch.randn(rows, d, generator=g, device="cuda") * 2) / 2
    elif cls == "near_uniform":
        logits = torch.randn(rows, d, generator=g, device="cuda") * 1e-3
    elif cls == "all_equal":
        logits = torch.zeros(rows, d, device="cuda")
    else:
        raise ValueError(cls)
    return torch.softmax(logits, -1).contiguous()


def _top_p(mode, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    if mode == "p95":
        return torch.full((rows,), 0.95, device="cuda")
    if mode == "mixed":
        return (0.05 + 0.95 * torch.rand(rows, generator=g, device="cuda")).contiguous()
    raise ValueError(mode)


def _fi_det(probs, top_p):
    import flashinfer.sampling

    return flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)


def _same(a, b, what):
    """Bit for bit."""
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"{what}: fp32 bits differ"


@pytest.mark.parametrize("mode", ["p95", "mixed"])
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_bitwise_flashinfer_det_below_one(cell, cls, mode) -> None:
    """Rows with top_p < 1 at each certified (rows, d): the override equals FlashInfer's det renorm bit for bit, and
    the explicit entry call gives the same bits again (deterministic)."""
    _gpu()
    rows, d = cell.spec["rows"], cell.spec["d"]
    for seed in range(2):
        probs = _make(cls, rows, d, 1000 * rows + seed)
        top_p = _top_p(mode, rows, seed)
        got = entry.top_p_renorm_det(probs, top_p)
        assert got is not None, "an in-contract input must be served"
        ref = _fi_det(probs, top_p)
        _same(got, ref, f"{cls} {mode} rows={rows} seed={seed}")
        again = op(
            probs,
            top_p,
            entry.prebuild(probs.device),
            entry.ctas_per_row(rows, entry.sm_count(probs.device)),
        )
        _same(again, got, "repeat")


@pytest.mark.parametrize("cls", CLASSES)
def test_keep_all_rows_in_a_mixed_batch(cls) -> None:
    """A batch of 8 rows, a third at top_p 1: the top_p 1 rows keep every element whose renormed value is a normal float
    (output nonzero <=> ftz(p * norm) != 0, the --use_fast_math semantics FlashInfer's apply shares) and sum to 1 within
    1e-5; the top_p < 1 rows are FlashInfer's det renorm bit for bit."""
    _gpu()
    rows, d = 8, 200064
    probs = _make(cls, rows, d, 77)
    top_p = _top_p("mixed", rows, 3)
    keep = torch.zeros(rows, dtype=torch.bool, device="cuda")
    keep[::3] = True
    top_p[keep] = 1.0
    got = entry.top_p_renorm_det(probs, top_p)
    assert got is not None
    ref = _fi_det(probs, top_p)
    _same(got[~keep], ref[~keep], f"{cls}: top_p < 1 rows")
    for r in keep.nonzero().flatten().tolist():
        p64 = probs[r].double()
        norm = 1.0 / float(p64[probs[r] >= FLT_MIN].sum())
        # kept <=> ftz(p * norm) != 0; elements within 1e-6 of the FLT_MIN boundary are left to the rounding
        live = p64 * norm >= FLT_MIN * (1 + 1e-6)
        dead = probs[r] < FLT_MIN
        assert bool((got[r][live] != 0).all()), f"{cls} row {r}: a kept token was dropped"
        assert bool((got[r][dead] == 0).all()), f"{cls} row {r}: a sub-FLT_MIN token is nonzero"
        total = float(got[r].double().sum())
        assert abs(total - 1.0) <= 1e-5, f"{cls} row {r}: sum {total}"


def test_graph_replay_equals_eager() -> None:
    """Captured in a CUDA graph (workspace prebuilt outside the capture), the replay gives the eager bits."""
    _gpu()
    probs = _make("peaked", 8, 200064, 5)
    top_p = _top_p("mixed", 8, 5)
    entry.prebuild(probs.device)
    ref = entry.top_p_renorm_det(probs, top_p)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        entry.top_p_renorm_det(probs, top_p)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.top_p_renorm_det(probs, top_p)
    assert out is not None, "a prebuilt workspace must serve a captured call"
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _same(out, ref, "graph replay")


def test_outside_the_contract_the_override_declines() -> None:
    """A float top_p, a non-fp32 / misaligned / non-contiguous probs, a d % 4 width, a mis-sized top_p or more rows
    than the workspace holds return None (the sampler runs stock FlashInfer)."""
    _gpu()
    d = 200064
    probs = _make("randn", 4, d, 1)
    top_p = torch.full((4,), 0.9, device="cuda")
    assert entry.top_p_renorm_det(probs, 0.9) is None
    assert entry.top_p_renorm_det(probs.half(), top_p) is None
    assert entry.top_p_renorm_det(probs, top_p.double()) is None
    assert entry.top_p_renorm_det(probs, top_p[:3]) is None
    assert entry.top_p_renorm_det(probs[:, : d - 2], top_p) is None
    assert entry.top_p_renorm_det(probs[:, : d - 2].contiguous(), top_p) is None
    flat = torch.empty(4 * d + 1, device="cuda")
    assert entry.top_p_renorm_det(flat[1:].view(4, d), top_p) is None
    wide = _make("randn", entry.MAX_ROWS + 1, d, 2)
    assert (
        entry.top_p_renorm_det(wide, torch.full((entry.MAX_ROWS + 1,), 0.9, device="cuda")) is None
    )
    assert entry.ctas_per_row(1, 148) == 16 and entry.ctas_per_row(8, 148) == 16
    assert entry.ctas_per_row(16, 148) == 8 and entry.ctas_per_row(64, 148) == 2
    assert entry.ctas_per_row(200, 148) == 1


@pytest.fixture
def _fake_flashinfer(monkeypatch):
    """The sampler ops module with a stand-in flashinfer whose calls are recorded, and no override afterwards."""
    stock = torch.full((2, 8), 0.125)
    sampling = types.SimpleNamespace(
        top_p_renorm_probs=mock.Mock(return_value=stock),
        softmax=mock.Mock(side_effect=lambda logits, temps, enable_pdl=False: logits.softmax(-1)),
        top_k_mask_logits=mock.Mock(side_effect=lambda logits, top_k: logits),
    )
    monkeypatch.setattr(fi_ops, "flashinfer", types.SimpleNamespace(sampling=sampling))
    fi_ops.set_top_p_renorm_override(None)
    yield sampling, stock
    fi_ops.set_top_p_renorm_override(None)


def test_hook_unset_is_stock(_fake_flashinfer) -> None:
    """No override: both ops make exactly the stock flashinfer call and return its result."""
    sampling, stock = _fake_flashinfer
    probs, top_p = torch.rand(2, 8), torch.full((2,), 0.9)
    assert fi_ops._TOP_P_RENORM_OVERRIDE is None
    assert fi_ops.top_p_renorm_probs_op(probs, top_p) is stock
    sampling.top_p_renorm_probs.assert_called_once_with(probs, top_p)
    assert (
        fi_ops.compute_probs_from_logits_op(torch.rand(2, 8), torch.ones(2), None, top_p) is stock
    )
    assert sampling.top_p_renorm_probs.call_count == 2


def test_hook_override_result_is_used(_fake_flashinfer) -> None:
    """An override's tensor replaces the stock call in both ops; the stock kernel does not run."""
    sampling, _ = _fake_flashinfer
    mine = torch.zeros(2, 8)
    override = mock.Mock(return_value=mine)
    fi_ops.set_top_p_renorm_override(override)
    probs, top_p = torch.rand(2, 8), torch.full((2,), 0.9)
    assert fi_ops.top_p_renorm_probs_op(probs, top_p) is mine
    override.assert_called_once_with(probs, top_p)
    assert fi_ops.compute_probs_from_logits_op(torch.rand(2, 8), torch.ones(2), None, top_p) is mine
    assert override.call_count == 2
    sampling.top_p_renorm_probs.assert_not_called()


def test_hook_override_declining_falls_back(_fake_flashinfer) -> None:
    """An override that returns None leaves the stock call in place; without top_p the override is never asked."""
    sampling, stock = _fake_flashinfer
    override = mock.Mock(return_value=None)
    fi_ops.set_top_p_renorm_override(override)
    probs, top_p = torch.rand(2, 8), torch.full((2,), 0.9)
    assert fi_ops.top_p_renorm_probs_op(probs, top_p) is stock
    assert (
        fi_ops.compute_probs_from_logits_op(torch.rand(2, 8), torch.ones(2), None, top_p) is stock
    )
    assert override.call_count == 2 and sampling.top_p_renorm_probs.call_count == 2
    fi_ops.compute_probs_from_logits_op(torch.rand(2, 8), torch.ones(2), None, None)
    assert override.call_count == 2


def test_the_target_installs_the_override() -> None:
    """With TOPP_DET on (the default) and the op built, the target installs ``top_p_renorm_det`` once per process."""
    _gpu()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
        modeling,
    )

    if not modeling.TOPP_DET:
        pytest.skip("TOPP_DET is switched off in this environment")
    saved = dict(modeling._TOPP_DET_STATE)
    modeling._TOPP_DET_STATE.clear()
    try:
        modeling._install_topp_det()
        assert fi_ops._TOP_P_RENORM_OVERRIDE is entry.top_p_renorm_det
        assert modeling._TOPP_DET_STATE["installed"] is True
    finally:
        fi_ops.set_top_p_renorm_override(None)
        modeling._TOPP_DET_STATE.clear()
        modeling._TOPP_DET_STATE.update(saved)


def test_the_kill_switch_installs_nothing() -> None:
    """TRTLLM_M31_DISABLE_TOPP_DET=1 (read at import): TOPP_DET is off and the target leaves the sampler hook unset."""
    probe = (
        f"import {TARGET}.modeling as m\n"
        "from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as f\n"
        "m._install_topp_det()\n"
        "print(m.TOPP_DET, f._TOP_P_RENORM_OVERRIDE is None, m._TOPP_DET_STATE['installed'])"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_TOPP_DET="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False True False"
