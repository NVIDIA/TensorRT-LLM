# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP8_FLAGFREE_GATHER on CPU: the switch, the TP8-only attach and the two call sites in the TP8 DSpark worker. The
bitwise check against the stock gather runs at set-up on every rank of a TP8 decode worker (8 GPUs)."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

ffg = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.flagfree_gather"
)
DSPARK = (
    Path(ffg.__file__).resolve().parents[1] / "minimax_m3_1_flash_nvfp4__sm_107__tp8" / "dspark.py"
)


def test_default_on_with_kill_switch(monkeypatch):
    monkeypatch.delenv(ffg.SWITCH, raising=False)
    assert ffg.SWITCH == "TRTLLM_M31_DISABLE_TP8_FLAGFREE_GATHER"
    assert importlib.reload(ffg).ENABLED is True
    monkeypatch.setenv(ffg.SWITCH, "1")
    assert importlib.reload(ffg).ENABLED is False
    monkeypatch.setenv(ffg.SWITCH, "maybe")
    with pytest.raises(ValueError, match=ffg.SWITCH):
        importlib.reload(ffg)
    monkeypatch.delenv(ffg.SWITCH)
    importlib.reload(ffg)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_attach_is_tp8_only(tp):
    worker = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=tp, tp_rank=0), _a1={}, _f7_ws=(384, {})
    )
    assert ffg.attach(worker, SimpleNamespace(out_features=25008)) is None


def test_off_when_neither_stock_gather_is_set_up(monkeypatch):
    monkeypatch.setattr(ffg, "ENABLED", True)
    worker = SimpleNamespace(mapping=SimpleNamespace(tp_size=8, tp_rank=3), _a1=None, _f7_ws=None)
    assert ffg.attach(worker, SimpleNamespace(out_features=25008)) is None


def test_worker_hooks():
    src = DSPARK.read_text()
    assert "from ..ops import flagfree_gather as _ffg" in src
    assert "self._ffg = _ffg.attach(self, lm_head) if lm_head is not None else None" in src
    # Each call site asks the flag-free gather first, inside the stock gather's own taken branch.
    assert src.index("self._ffg.fc_output(") < src.index(
        "_allgather_cast.gather_last_dim_f32(y.view(torch.float32)"
    )
    assert src.index("self._ffg.verify_logits(shard)") < src.index(
        "return _allgather_cast.allgather_cast(shard"
    )
