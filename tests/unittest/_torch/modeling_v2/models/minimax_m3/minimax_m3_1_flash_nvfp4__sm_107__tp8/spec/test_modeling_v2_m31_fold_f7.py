# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F7's routing in the tp8 target, without GPUs or a model: FC_SHARD's gather takes the installed one-launch
gather when it returns the rows and the NCCL all-gather otherwise (declined, or uninstalled); the worker's gather
hands the bf16 rows to allgather_f32 as fp32 words of half the width and returns its result as the same bytes in
bf16, and declines every layout or shape it was not compiled for. The bitwise gather itself is certified by
comm/_allgather_cast_op_matrix.py (check_bf16_rows_as_fp32_words)."""

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
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.gemm import (  # noqa: E501
        m31_fc_shard,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ROWS, WIDTH, WORLD = 8, 768, 8


def _shard(monkeypatch):
    shard = m31_fc_shard.FcShard.__new__(m31_fc_shard.FcShard)
    shard.mapping = "the tp mapping"
    shard._fast_gather = None
    calls = []
    monkeypatch.setattr(
        m31_fc_shard, "allgather", lambda y, mapping, dim: calls.append((y, mapping, dim)) or "nccl"
    )
    return shard, calls


def test_fc_shard_gather_routes_by_the_installed_gather(monkeypatch):
    shard, calls = _shard(monkeypatch)
    y = torch.zeros(ROWS, WIDTH, dtype=torch.bfloat16)
    assert shard.gather(y) == "nccl" and calls == [(y, "the tp mapping", -1)]
    shard.set_fast_gather(lambda t: "one launch")
    assert shard.gather(y) == "one launch" and len(calls) == 1
    shard.set_fast_gather(lambda t: None)
    assert shard.gather(y) == "nccl" and len(calls) == 2
    shard.set_fast_gather(None)
    assert shard.gather(y) == "nccl" and len(calls) == 3


def _worker(monkeypatch, compiled=True):
    wk = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    wk._f7_ws = (WIDTH // 2, {"world": WORLD, "ws_uc": "uc"})
    seen = []

    def gather(x, **ws):
        seen.append((x, ws))
        return (
            torch.arange(x.shape[0] * WORLD * x.shape[1], dtype=torch.int32)
            .view(torch.float32)
            .view(x.shape[0], WORLD * x.shape[1])
        )

    monkeypatch.setattr(dspark._allgather_cast, "is_compiled", lambda *a, **k: compiled)
    monkeypatch.setattr(dspark._allgather_cast, "gather_last_dim_f32", gather)
    return wk, seen


def test_the_worker_gathers_bf16_rows_as_fp32_words(monkeypatch):
    wk, seen = _worker(monkeypatch)
    y = torch.randn(ROWS, WIDTH).bfloat16()
    out = wk._f7_gather(y)
    ((x, ws),) = seen
    assert (
        x.dtype == torch.float32
        and tuple(x.shape) == (ROWS, WIDTH // 2)
        and x.data_ptr() == y.data_ptr()
    )
    assert ws == {"world": WORLD, "ws_uc": "uc"}
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (ROWS, WORLD * WIDTH)
    want = (
        torch.arange(ROWS * WORLD * WIDTH // 2, dtype=torch.int32)
        .view(torch.bfloat16)
        .view(ROWS, WORLD * WIDTH)
    )
    assert torch.equal(out.view(torch.int16), want.view(torch.int16))


def test_the_worker_declines_what_it_was_not_compiled_for(monkeypatch):
    y = torch.zeros(ROWS, WIDTH, dtype=torch.bfloat16)
    cases = {
        "fp32 rows": (y.float(), True),
        "strided rows": (torch.zeros(ROWS, 2 * WIDTH, dtype=torch.bfloat16)[:, ::2], True),
        "another width": (torch.zeros(ROWS, WIDTH + 8, dtype=torch.bfloat16), True),
        "3-D rows": (y.view(1, ROWS, WIDTH), True),
        "not compiled": (y, False),
    }
    for name, (t, compiled) in cases.items():
        wk, seen = _worker(monkeypatch, compiled=compiled)
        assert wk._f7_gather(t) is None and seen == [], name


def test_prebuild_requires_whole_16_byte_vectors(monkeypatch):
    wk = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    calls = []
    monkeypatch.setattr(
        dspark._allgather_cast, "precompile_f32", lambda *a, **k: calls.append(a) or True
    )
    monkeypatch.setattr(
        dspark._allgather_cast, "workspace_args", lambda mapping, cols: {"cols": cols}
    )
    wk._a1_all_ok = lambda ok: ok
    odd = SimpleNamespace(w_shard=torch.zeros(WIDTH + 4, 2), mapping="m")
    assert wk._prebuild_f7(odd) is None and calls == []
    shard = SimpleNamespace(w_shard=torch.zeros(WIDTH, 2), mapping="m")
    assert wk._prebuild_f7(shard) == (WIDTH // 2, {"cols": WIDTH // 2})
    ((mapping, cols, rows, device),) = calls
    assert (mapping, cols) == ("m", WIDTH // 2) and rows == tuple(
        range(1, dspark.FC_SHARD_MAX_ROWS + 1)
    )


def _fold_f7_in_subprocess(value):
    name = "TRTLLM_M31_DISABLE_FOLD_FC_SHARD_GATHER_MC"
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__tp8 import dspark as m; print('FOLD_F7=' + str(m.FOLD_F7))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("FOLD_F7=")][-1]


def test_fold_f7_kill_switch_subprocess():
    """FOLD_F7 is on by default; TRTLLM_M31_DISABLE_FOLD_FC_SHARD_GATHER_MC=1 (read at import) turns it off."""
    assert _fold_f7_in_subprocess(None) == "FOLD_F7=True"
    assert _fold_f7_in_subprocess("0") == "FOLD_F7=True"
    assert _fold_f7_in_subprocess("1") == "FOLD_F7=False"
