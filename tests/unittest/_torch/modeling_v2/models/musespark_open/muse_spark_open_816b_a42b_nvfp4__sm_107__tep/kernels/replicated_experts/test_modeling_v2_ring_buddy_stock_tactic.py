# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TEP8 ``prerouted_experts`` tactic of ``ring_buddy``, on the host only (no torch, no device).

``kernels/replicated_experts/ring_buddy.py`` is loaded from its file with every module it imports
stubbed for the load; the autotuner, the MoE runner and its dummy-routing hook, and the prerouted
backend are stubbed per test. The tests pin ``_stock_tactic`` on the 64-expert two-window stack at
9..64 rows, where ``ring_buddy_assign`` writes routing tables with 8-token tiles:

* a tuner cache miss runs the runner's default config at tile 8;
* a cached tactic at tile 8 is kept;
* a cached tactic at another tile (what a disaggregated generation server's warmup saves for the
  stacked window) is treated as a miss, the default config at tile 8, logged once per
  (window, cached tile), instead of sending the call back to the stock call;
* while tuning there is no tactic (the stock call runs and tunes).

And the install's refusals of ``batch_half_window_experts`` (``half_max_rows``): the half flow
places up to ``HALF_BATCH_MAX_ROWS`` rows only on the half window, within one
``ring_buddy_assign`` program, and a refresh keeps a layer's row limit.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
_PACKAGE = "_musespark_open_tep_ring_buddy_under_test"
_NAME = f"{_PACKAGE}.kernels.replicated_experts.ring_buddy"

#: The stacked call's shapes: the 64-expert two-window stack of tep8, the expert's sizes.
_WINDOW = 64
_HIDDEN = 7168
_INTERMEDIATE = 4096
_ACT_TYPE = 0
_ROUTING = 1
_TUNE_MAX_NUM_TOKENS = 8192
#: What the stub backend's ``default_config`` returns.
_DEFAULT_CONFIG = 152


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _package(name: str, **attrs) -> types.ModuleType:
    module = _module(name, **attrs)
    module.__path__ = []
    return module


def _assign_constants() -> dict:
    """``ring_buddy_assign``'s geometry constants, read from its file (it imports triton)."""
    tree = ast.parse((_TARGET / "ops/moe/ring_buddy_assign.py").read_text())
    names = {"EXPERTS", "RANKS", "BLOCK", "TOP_K", "HALF_SPAN"}
    namespace: dict = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "<constants>", "exec"), namespace
            )
    namespace.pop("__builtins__", None)
    assert set(namespace) == names, namespace
    return namespace


class _Log:
    """The logger stub: every emitted call as (level, message); ``*_once``: the first per key."""

    def __init__(self) -> None:
        self.records: list = []
        self.keys: set = set()

    def _record(self, level: str, message) -> None:
        self.records.append((level, str(message)))

    def _once(self, level: str, message, key) -> None:
        assert key is not None
        if key not in self.keys:
            self.keys.add(key)
            self._record(level, message)

    def info(self, message, *args, **kwargs) -> None:
        self._record("info", message)

    def warning(self, message, *args, **kwargs) -> None:
        self._record("warning", message)

    def debug(self, message, *args, **kwargs) -> None:
        self._record("debug", message)

    def info_once(self, message, *args, key=None) -> None:
        self._once("info", message, key)

    def warning_once(self, message, *args, key=None) -> None:
        self._once("warning", message, key)

    def messages(self, level: str) -> list:
        return [message for kind, message in self.records if kind == level]


def _load() -> types.ModuleType:
    constants = _assign_constants()
    ops = _package(f"{_PACKAGE}.ops")
    moe = _package(f"{_PACKAGE}.ops.moe")
    prerouted = _module(f"{_PACKAGE}.ops.moe.prerouted_experts")
    assign = _module(
        f"{_PACKAGE}.ops.moe.ring_buddy_assign",
        block_of=lambda *args, **kwargs: None,
        ring_buddy_assign=lambda *args, **kwargs: None,
        **constants,
    )
    moe.prerouted_experts = prerouted
    moe.ring_buddy_assign = assign
    ops.moe = moe
    dispatch = _module(f"{_PACKAGE}.dispatch")
    package = _package(_PACKAGE, dispatch=dispatch, ops=ops)
    stubs = {
        "torch": _module(
            "torch",
            Tensor=object,
            float8_e4m3fn="float8_e4m3fn",
            int32="int32",
            zeros=lambda *args, **kwargs: ("zeros", args, kwargs),
        ),
        "tensorrt_llm._torch.distributed": _module(
            "distributed", allgather=lambda *args, **kwargs: None
        ),
        "tensorrt_llm._torch.utils": _module("utils", Fp4QuantizedTensor=type("Fp4", (), {})),
        "tensorrt_llm.logger": _module("logger", logger=_Log()),
        "tensorrt_llm.mapping": _module("mapping", Mapping=type("Mapping", (), {})),
        _PACKAGE: package,
        dispatch.__name__: dispatch,
        ops.__name__: ops,
        moe.__name__: moe,
        prerouted.__name__: prerouted,
        assign.__name__: assign,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            _NAME, _TARGET / "kernels/replicated_experts/ring_buddy.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


@pytest.fixture(scope="module")
def ring_buddy():
    return _load()


class _Tuner:
    """``AutoTuner.get()``: ``choose_one`` returns ``tactic`` (-1: the cache-miss fallback)."""

    def __init__(self) -> None:
        self.tactic = -1
        self.is_tuning_mode = False
        self.calls: list = []

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        self.calls.append((custom_op, tuning_config))
        return runners[0], self.tactic


class _Runner:
    """``FP4BlockScaleMoERunner``: records its arguments."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.tuning_config = "base_tuning_config"


class _Ext:
    """The prerouted backend: ``default_config`` and ``moe`` record their arguments."""

    def __init__(self) -> None:
        self.default_calls: list = []
        self.moe_calls: list = []

    def default_config(self, tile, top_k, hidden, intermediate, window, rows, act_type) -> int:
        self.default_calls.append((tile, top_k, hidden, intermediate, window, rows, act_type))
        return _DEFAULT_CONFIG

    def moe(self, *args):
        self.moe_calls.append(args)
        return ["outputs"]


class _Device:
    def __init__(self, index: int) -> None:
        self.index = index


class _Tensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape
        self.device = _Device(0)

    def view(self, dtype):
        return self


class _Env:
    def __init__(self, ring_buddy, tuner: _Tuner, ext: _Ext, log: _Log) -> None:
        self.rb = ring_buddy
        self.tuner = tuner
        self.ext = ext
        self.log = log

    @staticmethod
    def x(rows: int):
        return types.SimpleNamespace(fp4_tensor=_Tensor(rows, _HIDDEN // 2))

    @staticmethod
    def stacks() -> dict:
        names = (
            "w3_w1_weight",
            "w3_w1_weight_scale",
            "w2_weight",
            "w2_weight_scale",
            "fc31_scale_c",
            "fc31_alpha",
            "fc2_alpha",
        )
        return {name: _Tensor(_WINDOW) for name in names}

    def stock_tactic(self, rows: int):
        return self.rb._stock_tactic(
            self.x(rows),
            _Tensor(rows),
            self.stacks(),
            _Tensor(rows, self.rb.TOP_K),
            _Tensor(rows, self.rb.TOP_K),
            _INTERMEDIATE,
            _WINDOW,
            _ROUTING,
            _ACT_TYPE,
            _TUNE_MAX_NUM_TOKENS,
        )

    def default_call(self, rows: int) -> tuple:
        return (8, self.rb.TOP_K, _HIDDEN, _INTERMEDIATE, _WINDOW, rows, _ACT_TYPE)


@pytest.fixture
def env(ring_buddy, monkeypatch):
    tuner = _Tuner()
    ext = _Ext()
    log = _Log()

    def prepare_dummy_topk_and_hook(**kwargs):
        return None, kwargs["topk_weights"], kwargs["topk_ids"], "tuning_config"

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.autotuner",
        _module("autotuner", AutoTuner=types.SimpleNamespace(get=lambda: tuner)),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops",
        _module(
            "trtllm_gen_custom_ops",
            FP4BlockScaleMoERunner=_Runner,
            prepare_dummy_topk_and_hook=prepare_dummy_topk_and_hook,
        ),
    )
    monkeypatch.setattr(ring_buddy, "_prerouted_ext", lambda tag="tep4": ext)
    monkeypatch.setattr(ring_buddy, "_prerouted_state", {})
    monkeypatch.setattr(ring_buddy, "logger", log)
    return _Env(ring_buddy, tuner, ext, log)


def test_the_geometry(ring_buddy):
    assert 2 * ring_buddy.BLOCK == _WINDOW
    assert ring_buddy._PREROUTED_TILE == 8
    assert ring_buddy._TEP8_PREROUTED_ROWS == range(9, 65)


@pytest.mark.parametrize("miss", [-1, None, [-1, -1]])
@pytest.mark.parametrize("rows", [9, 16, 40, 64])
def test_a_cache_miss_runs_the_default_config_at_the_table_tile(env, rows, miss):
    env.tuner.tactic = miss
    assert env.stock_tactic(rows) == (8, _DEFAULT_CONFIG)
    assert env.ext.default_calls == [env.default_call(rows)]
    assert env.tuner.calls == [("trtllm::fp4_block_scale_moe_runner", "tuning_config")]
    assert env.log.records == []


@pytest.mark.parametrize("rows", [9, 64])
def test_a_cached_tactic_at_the_table_tile_is_kept(env, rows):
    env.tuner.tactic = [8, 77]
    assert env.stock_tactic(rows) == (8, 77)
    assert env.ext.default_calls == []
    assert env.log.records == []


@pytest.mark.parametrize("cached_tile", [16, 32, 64])
def test_a_cached_tactic_at_another_tile_runs_the_default_config_at_the_table_tile(
    env, cached_tile
):
    env.tuner.tactic = [cached_tile, 99]
    assert env.stock_tactic(64) == (8, _DEFAULT_CONFIG)
    assert env.ext.default_calls == [env.default_call(64)]
    (message,) = env.log.messages("info")
    assert f"tile {cached_tile}" in message and "default config at tile 8" in message
    assert f"{_WINDOW}-expert stacked-window tactic" in message


def test_the_cached_tile_override_is_logged_once_per_window_and_tile(env):
    env.tuner.tactic = [32, 99]
    for rows in (64, 40, 16):
        assert env.stock_tactic(rows) == (8, _DEFAULT_CONFIG)
    env.tuner.tactic = [16, 5]
    assert env.stock_tactic(9) == (8, _DEFAULT_CONFIG)
    messages = env.log.messages("info")
    assert len(messages) == 2
    assert "tile 32" in messages[0] and "tile 16" in messages[1]


def test_the_choice_is_memoized_per_shape(env):
    env.tuner.tactic = [32, 99]
    assert env.stock_tactic(64) == (8, _DEFAULT_CONFIG)
    env.tuner.tactic = [8, 77]
    assert env.stock_tactic(64) == (8, _DEFAULT_CONFIG)
    assert len(env.tuner.calls) == 1
    assert env.stock_tactic(40) == (8, 77)


@pytest.mark.parametrize("tactic", [-1, [8, 77], [32, 99]])
def test_tuning_has_no_tactic(env, tactic):
    env.tuner.tactic = tactic
    env.tuner.is_tuning_mode = True
    assert env.stock_tactic(64) is None
    assert env.tuner.calls == []
    assert env.ext.default_calls == []


def test_no_backend_keeps_the_stock_call(env, monkeypatch):
    monkeypatch.setattr(env.rb, "_prerouted_ext", lambda tag="tep4": None)
    env.tuner.tactic = [32, 99]
    assert env.stock_tactic(64) is None
    assert env.log.records == []


def test_the_prerouted_call_runs_where_the_cache_holds_another_tile(env):
    """The measured case: a 64-row call, the cache holding the stacked window's tile-32 tactic."""
    env.tuner.tactic = [32, 99]
    rows = 64
    meta = {
        "rows": rows,
        "tile": 8,
        **{name: _Tensor(1) for name in ("e2p", "p2t", "c2b", "c2m", "nnc", "tnp")},
    }
    outputs = env.rb._tep8_prerouted_call(
        env.stacks(),
        env.x(rows),
        _Tensor(rows),
        _Tensor(rows, env.rb.TOP_K),
        _Tensor(rows, env.rb.TOP_K),
        meta,
        _INTERMEDIATE,
        _WINDOW,
        _ACT_TYPE,
        _ROUTING,
        _TUNE_MAX_NUM_TOKENS,
    )
    assert outputs == ["outputs"]
    (args,) = env.ext.moe_calls
    # ..., window, tile_n, config, act_type, False
    assert args[-5:] == (_WINDOW, 8, _DEFAULT_CONFIG, _ACT_TYPE, False)


def _assign_max_slots() -> int:
    """``ring_buddy_assign.MAX_SLOTS``, read from its file (it imports triton)."""
    tree = ast.parse((_TARGET / "ops/moe/ring_buddy_assign.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "MAX_SLOTS"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ring_buddy_assign.py has no MAX_SLOTS")


def _refusal(ring_buddy, layers=(), half=True, half_max_rows=None):
    extra = {} if half_max_rows is None else {"half_max_rows": half_max_rows}
    return ring_buddy._install_refusal(
        types.SimpleNamespace(layers=list(layers)), 1, True, False, False, 4, None, half, **extra
    )


def test_the_batch_half_window_rows_fit_one_assign_program(ring_buddy):
    assert ring_buddy.HALF_MAX_ROWS == 32
    assert ring_buddy.HALF_BATCH_MAX_ROWS == 64
    assert ring_buddy.HALF_BATCH_MAX_ROWS * ring_buddy.TOP_K <= _assign_max_slots()


def test_the_batch_half_window_install_needs_the_half_window(ring_buddy):
    assert _refusal(ring_buddy) is None
    assert _refusal(ring_buddy, half_max_rows=ring_buddy.HALF_BATCH_MAX_ROWS) is None
    refusal = _refusal(ring_buddy, half=False, half_max_rows=ring_buddy.HALF_BATCH_MAX_ROWS)
    assert refusal is not None and "it needs half=True" in refusal
    for rows in (16, 48, 128):
        refusal = _refusal(ring_buddy, half_max_rows=rows)
        assert refusal is not None and f"half_max_rows={rows}" in refusal


def test_a_refresh_keeps_the_half_window_row_limit(ring_buddy):
    def layer(max_rows):
        state = types.SimpleNamespace(half=True, kind="nvfp4", max_rows=max_rows, windows=1)
        return types.SimpleNamespace(feed_forward=types.SimpleNamespace(_replicated_experts=state))

    refusal = _refusal(
        ring_buddy, [layer(ring_buddy.HALF_MAX_ROWS)], half_max_rows=ring_buddy.HALF_BATCH_MAX_ROWS
    )
    assert refusal is not None and "a refresh cannot" in refusal
    refusal = _refusal(ring_buddy, [layer(ring_buddy.HALF_BATCH_MAX_ROWS)])
    assert refusal is not None and "a refresh cannot" in refusal
