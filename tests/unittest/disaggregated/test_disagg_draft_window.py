# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The disagg DFlash draft-window contract on the unified KV cache (``disaggregation/draft_kv.py``).

CPU only. The module imports neither torch nor the bindings, so it is loaded straight from its file:
the test runs in any Python with pytest, inside or outside the TensorRT-LLM container.
"""

import importlib.util
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PATH = os.path.join(_ROOT, "tensorrt_llm", "_torch", "disaggregation", "draft_kv.py")


def _load():
    name = "_draft_kv_under_test"
    spec = importlib.util.spec_from_file_location(name, _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dk = _load()


def test_module_imports_neither_torch_nor_tensorrt_llm():
    with open(_PATH) as f:
        source = f.read()
    top_level_imports = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not [line for line in top_level_imports if "torch" in line or "tensorrt_llm" in line]


# --- strict switches -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("OFF", False),
        ("no", False),
        ("1", True),
        ("true", True),
        (" On ", True),
        ("YES", True),
    ],
)
def test_strict_bool_env(raw, expected):
    env = {} if raw is None else {"X": raw}
    assert dk.parse_strict_bool_env("X", env) is expected


@pytest.mark.parametrize("raw", ["2", "enable", "y", "tru"])
def test_strict_bool_env_rejects_typos(raw):
    with pytest.raises(ValueError, match="X must be one of"):
        dk.parse_strict_bool_env("X", {"X": raw})


# --- the generation-side seed --------------------------------------------------------------------


def test_unified_cache_seeds_the_window_to_the_prompt_end():
    # A 140k-token prompt: the drafter's context is every prompt position, read from the
    # request's own pages that arrived with the target's KV.
    assert dk.gen_window_seed(True, 140_000, 300_000) == (140_000, dk.SEED_WINDOW)


def test_unified_seed_is_clamped_to_the_logical_capacity():
    assert dk.gen_window_seed(True, 5000, 4096) == (4096, dk.SEED_WINDOW)


def test_private_draft_storage_starts_empty():
    # The pre-port behaviour of every non-unified layout: the window is worker-local.
    assert dk.gen_window_seed(False, 140_000, 300_000) == (0, dk.SEED_EMPTY)


def test_seed_rejects_a_negative_position():
    with pytest.raises(ValueError):
        dk.gen_window_seed(True, -1, 10)


@pytest.mark.parametrize(
    "disagg,unified,reuse,mode",
    [
        (False, True, True, None),
        (True, True, True, "shipped"),
        (True, False, False, "worker_local"),
        (True, True, False, "worker_local"),
    ],
)
def test_draft_window_mode(disagg, unified, reuse, mode):
    assert dk.draft_window_mode(disagg, unified, reuse) == mode


# --- counters ------------------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _counters(interval_s=60.0):
    infos, warnings, clock = [], [], _Clock()
    counters = dk.DraftWindowCounters(
        role="unified",
        interval_s=interval_s,
        info=infos.append,
        warning=warnings.append,
        clock=clock,
    )
    return counters, infos, warnings, clock


def test_shipped_windows_log_a_summary_but_no_warning():
    counters, infos, warnings, clock = _counters()
    counters.record_seed(dk.SEED_WINDOW, 1000)
    clock.now = 10.0
    counters.record_seed(dk.SEED_WINDOW, 2000)  # inside the interval: no new line
    assert warnings == []
    assert len(infos) == 1
    clock.now = 61.0
    counters.record_seed(dk.SEED_WINDOW, 500)
    assert len(infos) == 2
    assert infos[-1] == (
        "DFlash draft window (unified): seeded={window=3} seeded_tokens=3500 "
        "ctx_store_fallback=0 ctx_store_fallback_by_reason={}"
    )


def test_an_empty_window_warns_once():
    counters, infos, warnings, _ = _counters()
    counters.record_seed(dk.SEED_EMPTY)
    counters.record_seed(dk.SEED_EMPTY)
    counters.record_seed(dk.SEED_DUMMY)
    assert len(warnings) == 2
    assert "seed=empty" in warnings[0] and "seed=dummy_slot" in warnings[1]
    assert "seeded={dummy_slot=1,empty=2}" in counters.summary()


def test_context_fallbacks_are_counted_by_reason():
    counters, _, warnings, _ = _counters()
    counters.record_fallback(dk.FALLBACK_NO_FREE_SLOT, "request=7")
    counters.record_fallback(dk.FALLBACK_NO_FREE_SLOT, "request=8")
    counters.record_fallback(dk.FALLBACK_CTX_OVERFLOW)
    assert len(warnings) == 2
    assert "reason=no_free_slot request=7" in warnings[0]
    summary = counters.summary()
    assert "ctx_store_fallback=3" in summary
    assert "ctx_store_fallback_by_reason={ctx_overflow=1,no_free_slot=2}" in summary


def test_zero_interval_logs_every_event():
    counters, infos, _, _ = _counters(interval_s=0.0)
    for _ in range(3):
        counters.record_seed(dk.SEED_WINDOW, 1)
    assert len(infos) == 3


def test_log_interval_env(monkeypatch):
    assert dk._default_log_interval({}) == 60.0
    assert dk._default_log_interval({dk.DRAFT_WINDOW_LOG_INTERVAL_ENV: "5"}) == 5.0
    for bad in ("soon", "-1"):
        with pytest.raises(ValueError, match=dk.DRAFT_WINDOW_LOG_INTERVAL_ENV):
            dk._default_log_interval({dk.DRAFT_WINDOW_LOG_INTERVAL_ENV: bad})


def test_summary_marker_is_greppable():
    # The G2 gate greps gen-worker logs for this prefix; keep it stable.
    assert dk.DraftWindowCounters.PREFIX == "DFlash draft window"


# --- context-state carrier -----------------------------------------------------------------------


def test_ctx_state_round_trip():
    assert dk.decode_draft_ctx_state(dk.encode_draft_ctx_state(4096, 0, True)) == (4096, 0, True)
    assert dk.decode_draft_ctx_state(dk.encode_draft_ctx_state(17, 2, False)) == (17, 2, False)


@pytest.mark.parametrize(
    "state", [None, b"\x01", "1,2,3", [1, 2], [1, 2, 3, 4], [-1, 0, 1], [1, 0, 2], ["a", 0, 1]]
)
def test_ctx_state_rejects_malformed(state):
    assert dk.decode_draft_ctx_state(state) is None
