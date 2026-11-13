# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F3's routing in the tp8 target's DSpark worker (TAILBK R1), without GPUs or a model: the accept call leaves
its Philox window to the rejection launch (an unwritten buffer, registered with its operands) and returns its accept
row; the draft block's call writes the window first unless that launch did, so it always reads base + add; with the
fold off (or REJ_MC not the override) the window is the add, as before. The launch's values are certified by
sampling/test_modeling_v2_m31_fold_philox_in_rejection_gpu.py."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)
from tensorrt_llm._torch.speculative import interface

pytestmark = pytest.mark.cpu_only

K = 7
SPAN = dspark._RNG_SLOT_SPAN


def _meta():
    return SimpleNamespace(
        rng_draft_slot=lambda i: K + 1 + i,
        rng_accept_slot=2 * K + 1,
        offsets=torch.arange(K + 1, dtype=torch.int64) * 1000 + 5,
        request_seeds=torch.tensor([42], dtype=torch.int64),
        seeds=torch.full((K,), 42, dtype=torch.int64),
    )


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(dspark, "tail_bookkeep_draft_len", lambda *a: K)
    monkeypatch.setattr(dspark, "tail_bookkeep_log_once", lambda *a: None)
    dspark._chain_spec_mc._WINDOW[0] = None
    wk = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    wk._tailbk_win = None
    return wk


def test_the_accept_defers_its_window_and_the_block_reads_it_written(worker, monkeypatch):
    monkeypatch.setattr(dspark, "_f3_defers", lambda offsets: True)
    meta = _meta()
    seeds, offset = worker._tailbk_rng_window(meta, 0, 1, 1, 2 * K + 1)
    win = dspark._chain_spec_mc._WINDOW[0][0]
    assert seeds.data_ptr() == meta.request_seeds.data_ptr() and offset.numel() == 1
    assert offset.data_ptr() == win.data_ptr() + K * win.element_size()
    base, add = dspark._chain_spec_mc._WINDOW[0][1:]
    assert torch.equal(base, meta.offsets) and add == (K + 1) * SPAN
    seeds, offsets = worker._tailbk_rng_window(meta, 0, 1, K, K + 1)
    assert dspark._chain_spec_mc._WINDOW[0] is None
    assert torch.equal(win, meta.offsets + (K + 1) * SPAN)
    assert offsets.data_ptr() == win.data_ptr() and offsets.numel() == K


def test_a_window_the_launch_wrote_is_not_rewritten(worker, monkeypatch):
    monkeypatch.setattr(dspark, "_f3_defers", lambda offsets: True)
    meta = _meta()
    worker._tailbk_rng_window(meta, 0, 1, 1, 2 * K + 1)
    win = dspark._chain_spec_mc._WINDOW[0][0]
    win.fill_(-9)  # what the launch wrote; it took the window
    dspark._chain_spec_mc._WINDOW[0] = None
    worker._tailbk_rng_window(meta, 0, 1, K, K + 1)
    assert torch.all(win == -9)


def test_off_the_window_is_the_add(worker, monkeypatch):
    monkeypatch.setattr(dspark, "_f3_defers", lambda offsets: False)
    meta = _meta()
    _, offset = worker._tailbk_rng_window(meta, 0, 1, 1, 2 * K + 1)
    assert dspark._chain_spec_mc._WINDOW[0] is None
    assert torch.equal(offset, (meta.offsets + (K + 1) * SPAN)[K:])


def test_the_precondition(monkeypatch):
    cuda_like = SimpleNamespace()
    monkeypatch.setattr(dspark._chain_spec_mc, "window_ops_built", lambda: True)
    monkeypatch.setattr(
        interface, "_CHAIN_SPEC_OVERRIDE", dspark._chain_spec_mc.chain_spec_mc, raising=False
    )
    offsets = torch.zeros(K + 1, dtype=torch.int64)
    assert not dspark._f3_defers(offsets)  # a CPU tensor: no REJ_MC launch reads it
    assert not dspark._f3_defers(cuda_like)
    monkeypatch.setattr(interface, "_CHAIN_SPEC_OVERRIDE", None, raising=False)
    monkeypatch.setattr(dspark, "FOLD_F3", True)
    assert not dspark._f3_defers(offsets)


def _fold_f3_in_subprocess(value):
    name = "TRTLLM_M31_DISABLE_FOLD_PHILOX_IN_REJECTION"
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__tp8 import dspark as m; print('FOLD_F3=' + str(m.FOLD_F3))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("FOLD_F3=")][-1]


def test_fold_f3_kill_switch_subprocess():
    """FOLD_F3 is on by default; TRTLLM_M31_DISABLE_FOLD_PHILOX_IN_REJECTION=1 (read at import) turns it off."""
    assert _fold_f3_in_subprocess(None) == "FOLD_F3=True"
    assert _fold_f3_in_subprocess("0") == "FOLD_F3=True"
    assert _fold_f3_in_subprocess("1") == "FOLD_F3=False"
