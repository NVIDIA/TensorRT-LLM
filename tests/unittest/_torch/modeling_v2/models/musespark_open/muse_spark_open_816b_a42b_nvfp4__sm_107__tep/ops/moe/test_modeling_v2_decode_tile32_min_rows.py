# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MUSESPARK_OPEN_DECODE_TILE32_MIN_ROWS`` of the sm107 decode expert tactics, on the host only.

``ops/moe/decode_expert_tactics.py`` is loaded from its file once per test, with ``torch`` and the
logger stubbed for the load and the call (no torch, no device); the TRTLLM-gen runner's valid
configs (``_valid``) are stubbed per test. The tests pin:

* unset, the 68-expert window's pin covers 256..384 rows, as before the switch;
* a multiple of 8 from 128 to 256 moves the first pinned row count down, and the first call that
  reaches a listed row count logs once per call shape how many listed row counts take the pin;
* a row count where the pinned tactic is not a valid config keeps the autotuner, and the log
  line is a warning naming it;
* any other value refuses at import;
* ``pin_half_batch_steps`` (``batch_half_window_experts``) pins the 96-expert half window's
  40..64-row calls to the same tile, and nothing else; without it they keep the autotuner.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_FILE = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/moe/decode_expert_tactics.py"
)
_ENV = "MUSESPARK_OPEN_DECODE_TILE32_MIN_ROWS"
_TILE32 = (32, 236)
#: The stacked call's shapes: the 68-expert hot-replica window, the expert's sizes.
_WINDOW = 68
_TOP_K = 8
_HIDDEN = 7168
_INTERMEDIATE = 4096
_LOADS = itertools.count()


class _Log:
    def __init__(self) -> None:
        self.records = []

    def info(self, message, *args, **kwargs) -> None:
        self.records.append(("info", str(message)))

    def warning(self, message, *args, **kwargs) -> None:
        self.records.append(("warning", str(message)))


def _load(monkeypatch, value=None):
    """The module as a process with ``value`` in the switch (unset for None) imports it."""
    if value is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, str(value))
    torch = types.ModuleType("torch")
    torch.device = object
    torch.Tensor = object
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda device=None: (10, 7))
    log = _Log()
    package = types.ModuleType("tensorrt_llm")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", package)
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm.logger", types.SimpleNamespace(logger=log, __name__="logger")
    )
    name = f"_musespark_open_decode_tactics_under_test_{next(_LOADS)}"
    spec = importlib.util.spec_from_file_location(name, _FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, log


def _valid_everywhere(module, monkeypatch, invalid_rows=()):
    def valid(act_type, top_k, hidden, intermediate, window, rows):
        return frozenset() if rows in invalid_rows else frozenset({_TILE32})

    monkeypatch.setattr(module, "_valid", valid)


def _call(module, rows):
    return module.fixed_tactic(None, 0, _TOP_K, _HIDDEN, _INTERMEDIATE, _WINDOW, rows)


def test_unset_keeps_the_256_to_384_row_pin(monkeypatch):
    module, log = _load(monkeypatch)
    _valid_everywhere(module, monkeypatch)
    assert module.TILE32_MIN_ROWS == 256
    assert sorted(rows for _, rows in module.SM107_TACTICS) == list(range(256, 385, 8))
    assert _call(module, 248) is None
    assert _call(module, 256) == list(_TILE32)
    # No switch, no report.
    assert log.records == []


@pytest.mark.parametrize("value", [128, 192, 256])
def test_the_switch_moves_the_first_pinned_row_count(monkeypatch, value):
    module, log = _load(monkeypatch, value)
    _valid_everywhere(module, monkeypatch)
    assert module.TILE32_MIN_ROWS == value
    assert sorted(rows for _, rows in module.SM107_TACTICS) == list(range(value, 385, 8))
    assert _call(module, value - 8) is None
    assert _call(module, value) == list(_TILE32)
    assert _call(module, 384) == list(_TILE32)
    # The other windows and the non-SwiGLU expert keep the autotuner.
    assert module.fixed_tactic(None, 0, _TOP_K, _HIDDEN, _INTERMEDIATE, 64, value) is None
    assert module.fixed_tactic(None, 1, _TOP_K, _HIDDEN, _INTERMEDIATE, _WINDOW, value) is None
    infos = [message for level, message in log.records if level == "info"]
    if value < 256:
        assert len(infos) == 1, log.records
        assert f"{_ENV}={value}" in infos[0]
        assert f"{value}..384-row calls" in infos[0]
        count = len(range(value, 385, 8))
        assert f"a valid config at {count} of {count} row counts" in infos[0]
    else:
        assert infos == []


def test_an_invalid_row_count_keeps_the_autotuner_and_warns(monkeypatch):
    module, log = _load(monkeypatch, 128)
    _valid_everywhere(module, monkeypatch, invalid_rows={136})
    assert _call(module, 128) == list(_TILE32)
    assert _call(module, 136) is None
    warnings = [message for level, message in log.records if level == "warning"]
    assert len(warnings) == 1 and "the autotuner keeps [136]" in warnings[0]


@pytest.mark.parametrize("value", ["120", "130", "264", "512", "x", "-128", "128.0"])
def test_any_other_value_refuses_at_import(monkeypatch, value):
    with pytest.raises(ValueError, match=_ENV):
        _load(monkeypatch, value)


@pytest.mark.parametrize("value", [None, 128])
def test_the_half_window_batch_steps_are_pinned_only_when_asked(monkeypatch, value):
    module, log = _load(monkeypatch, value)
    _valid_everywhere(module, monkeypatch)

    def half(rows):
        return module.fixed_tactic(None, 0, _TOP_K, _HIDDEN, _INTERMEDIATE, 96, rows)

    assert all(half(rows) is None for rows in range(8, 129, 8))
    module.pin_half_batch_steps()
    module.pin_half_batch_steps()
    for rows in range(8, 129, 8):
        assert half(rows) == (list(_TILE32) if 40 <= rows <= 64 else None), rows
    # The hot-replica window keeps its own table.
    assert _call(module, 64) is None
    assert _call(module, 256) == list(_TILE32)
    infos = [message for level, message in log.records if level == "info"]
    assert sum("96-expert half window's 40..64-row decode steps" in m for m in infos) == 1
