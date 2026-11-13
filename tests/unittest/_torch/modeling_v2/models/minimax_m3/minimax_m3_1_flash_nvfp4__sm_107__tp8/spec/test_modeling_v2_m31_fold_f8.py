# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F8's routing in the tp8 target's DSpark worker, without GPUs or a model: a pure-decode step hands its
block-table refresh to the step-inputs launch (one combined launch, the refresh never runs on its own), a declined
combined launch still refreshes before the stock inputs, and any other step (contexts in the batch, the fold off, no
offsets) refreshes in place as before. The combined launch's values are certified by
spec_decode/test_modeling_v2_dspark_inputs_refresh.py."""

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

_ARGS = ("embed", "accepted", "nacc", "slots", "ctx_len", 128, 5, None, 8, 8)


def _worker(monkeypatch, on=True):
    wk = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    wk._f8_on = on
    wk._f8_pending = None
    wk._ctx_block_tables = torch.zeros(4, 6, dtype=torch.int32)
    wk._ctx_block_counts = torch.zeros(4, dtype=torch.int64)
    wk._ctx_pool_idx = 0
    wk._ctx_block_divisor = 2
    calls = []

    def parent_refresh(self, meta, num_seqs):
        calls.append(("refresh", meta.draft_kv_cache_block_offsets, num_seqs))
        return True

    monkeypatch.setattr(DFlashWorker, "_refresh_ctx_block_tables", parent_refresh)
    wk._f8_inputs = lambda *a: calls.append(("inputs", a)) or "inputs"
    return wk, calls


def _meta(num_contexts=0, offsets=True):
    return SimpleNamespace(
        num_contexts=num_contexts,
        draft_kv_cache_block_offsets=torch.zeros(1, 4, 2, 6, dtype=torch.int32)
        if offsets
        else None,
    )


def test_pure_decode_step_takes_the_combined_launch(monkeypatch):
    wk, calls = _worker(monkeypatch)
    seen = []
    monkeypatch.setattr(
        dspark, "s10c_c_drafter_inputs_and_refresh", lambda *a: seen.append(a) or "combined"
    )
    meta = _meta()
    assert wk._refresh_ctx_block_tables(meta, 3) is True
    assert calls == [] and wk._f8_pending is not None and wk._ctx_table_width == 6
    assert wk._f8_drafter_inputs(*_ARGS) == "combined"
    assert calls == [] and wk._f8_pending is None
    (args,) = seen
    assert args[: len(_ARGS)] == _ARGS
    src, pool_idx, num_seqs, divisor, tables, counts = args[-1]
    assert src is meta.draft_kv_cache_block_offsets and (pool_idx, num_seqs, divisor) == (0, 3, 2)
    assert tables is wk._ctx_block_tables and counts is wk._ctx_block_counts


def test_declined_combined_launch_refreshes_before_the_stock_inputs(monkeypatch):
    wk, calls = _worker(monkeypatch)
    monkeypatch.setattr(dspark, "s10c_c_drafter_inputs_and_refresh", lambda *a: None)
    meta = _meta()
    wk._refresh_ctx_block_tables(meta, 3)
    assert wk._f8_drafter_inputs(*_ARGS) == "inputs"
    assert [c[0] for c in calls] == ["refresh", "inputs"]
    assert calls[0][1] is meta.draft_kv_cache_block_offsets and calls[0][2] == 3
    assert wk._f8_pending is None


def test_other_steps_refresh_in_place(monkeypatch):
    for case in ("contexts", "off", "no_offsets"):
        wk, calls = _worker(monkeypatch, on=case != "off")
        monkeypatch.setattr(dspark, "s10c_c_drafter_inputs_and_refresh", lambda *a: "combined")
        meta = _meta(num_contexts=1 if case == "contexts" else 0, offsets=case != "no_offsets")
        assert wk._refresh_ctx_block_tables(meta, 3) is True
        assert [c[0] for c in calls] == ["refresh"] and wk._f8_pending is None, case
        assert wk._f8_drafter_inputs(*_ARGS) == "inputs", case
        assert [c[0] for c in calls] == ["refresh", "inputs"], case


def test_each_step_starts_without_a_pending_refresh(monkeypatch):
    """A pending refresh never outlives its step: the next step's refresh call replaces or clears it."""
    wk, calls = _worker(monkeypatch)
    wk._refresh_ctx_block_tables(_meta(), 3)
    assert wk._f8_pending is not None
    wk._refresh_ctx_block_tables(_meta(num_contexts=2), 3)
    assert wk._f8_pending is None and [c[0] for c in calls] == ["refresh"]


def _fold_f8_in_subprocess(value):
    name = "TRTLLM_M31_DISABLE_FOLD_BLOCK_TABLE_IN_INPUTS"
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__tp8 import dspark as m; print('FOLD_F8=' + str(m.FOLD_F8))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("FOLD_F8=")][-1]


def test_fold_f8_kill_switch_subprocess():
    """FOLD_F8 is on by default; TRTLLM_M31_DISABLE_FOLD_BLOCK_TABLE_IN_INPUTS=1 (read at import) turns it off."""
    assert _fold_f8_in_subprocess(None) == "FOLD_F8=True"
    assert _fold_f8_in_subprocess("0") == "FOLD_F8=True"
    assert _fold_f8_in_subprocess("1") == "FOLD_F8=False"
