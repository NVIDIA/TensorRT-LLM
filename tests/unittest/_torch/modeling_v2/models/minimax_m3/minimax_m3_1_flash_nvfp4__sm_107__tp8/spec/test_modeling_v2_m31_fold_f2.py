# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F2's routing in the tp8 target's DSpark worker, without GPUs or a model: on a step whose forward tail runs,
N3 A2's next_new_tokens is left to that tail (an unwritten int32 buffer, its operands and N3 A2 itself as the
fallback, handed over once); every other step (no tail armed, the arm already used, non-int32 operands, N3 A2 not
taken, the fold off) runs as before. The tail's values are certified by test_modeling_v2_m31_fold_f2_gpu.py."""

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
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.cpu_only

B, D = 2, 7


def _worker(monkeypatch, supported=True):
    wk = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    calls = []

    def stock(*operands):
        calls.append(("n3a2", operands))
        return "n3a2"

    def parent(self, *args):
        calls.append(("in_tree", args))
        return "in_tree"

    monkeypatch.setattr(dspark, "N3A_DISABLED", False)
    monkeypatch.setattr(dspark._strict_accept, "supports_next_new_tokens", lambda *a: supported)
    monkeypatch.setattr(dspark._strict_accept, "is_compiled_next_new_tokens", lambda *a: True)
    monkeypatch.setattr(dspark._strict_accept, "spec_next_new_tokens", stock)
    monkeypatch.setattr(DFlashWorker, "_prepare_next_new_tokens", parent)
    return wk, calls


def _tail():
    seen = []
    return SimpleNamespace(defer_next_new_tokens=lambda *a: seen.append(a)), seen


def _operands(dtype=torch.int32):
    accepted = torch.zeros(B, D + 1, dtype=dtype)
    nacc = torch.ones(B, dtype=torch.int32)
    batch = torch.arange(B + 2, dtype=torch.int32)
    draft = torch.zeros(B, D, dtype=torch.int32)
    return accepted, draft, batch, B, nacc


def test_an_armed_step_leaves_n3a2_to_the_tail(monkeypatch):
    wk, calls = _worker(monkeypatch)
    tail, seen = _tail()
    wk.arm_next_new_tokens_tail(tail)
    accepted, draft, batch, n, nacc = _operands()
    out = wk._prepare_next_new_tokens(accepted, draft, batch, n, nacc)
    assert calls == []
    assert out.dtype == torch.int32 and tuple(out.shape) == (B, D + 1)
    ((buf, operands, flush),) = seen
    assert buf is out
    assert operands[0] is accepted and operands[1] is nacc and operands[3] is draft
    assert operands[2].data_ptr() == batch.data_ptr() and operands[2].numel() == B
    assert flush() == "n3a2" and calls == [("n3a2", operands)]


def test_other_steps_run_as_before(monkeypatch):
    for case in ("unarmed", "used", "int64", "unsupported", "off"):
        wk, calls = _worker(monkeypatch, supported=case != "unsupported")
        tail, seen = _tail()
        monkeypatch.setattr(dspark, "FOLD_F2", case != "off")
        if case != "unarmed":
            wk.arm_next_new_tokens_tail(tail)
        if case == "used":
            wk._prepare_next_new_tokens(*_operands())
            seen.clear()
        accepted, draft, batch, n, nacc = _operands(torch.int64 if case == "int64" else torch.int32)
        got = wk._prepare_next_new_tokens(accepted, draft, batch, n, nacc)
        assert seen == [], case
        assert got == ("in_tree" if case == "unsupported" else "n3a2"), case
        assert calls[-1][0] == got, case


def _fold_f2_in_subprocess(value):
    name = "TRTLLM_M31_DISABLE_FOLD_NEXT_TOKENS_IN_TAIL"
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__tp8 import dspark as m; print('FOLD_F2=' + str(m.FOLD_F2))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("FOLD_F2=")][-1]


def test_fold_f2_kill_switch_subprocess():
    """FOLD_F2 is on by default; TRTLLM_M31_DISABLE_FOLD_NEXT_TOKENS_IN_TAIL=1 (read at import) turns it off."""
    assert _fold_f2_in_subprocess(None) == "FOLD_F2=True"
    assert _fold_f2_in_subprocess("0") == "FOLD_F2=True"
    assert _fold_f2_in_subprocess("1") == "FOLD_F2=False"
