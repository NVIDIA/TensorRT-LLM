# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PP stage's fixed FlashInfer tile for the BF16 expert layers, on the host only.

``kernels/bf16_expert_tile/install.py`` is loaded from its file into a stand-in package whose
``dispatch`` / ``_tep`` modules, ``tensorrt_llm.logger`` and ``flashinfer.autotuner`` are
recorders. Claims checked here:

* the pinned tactic is ``[128, 0]``;
* the tuner's ``choose_one`` is wrapped once; outside a pinned forward, for another op, or for a
  tactic the op does not offer at the call's shape, the tuner's own choice is returned;
  inside a pinned forward the pinned ``[tile_N, config]`` is returned with the first runner;
* the wrapped expert forward pins only 2-D BF16 calls of 4096..8192 rows, restores the outer
  pin afterwards (also on an exception);
* only the stage's BF16 leaf layers that hold all 256 experts from slot 0 are wrapped.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_FILE = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__pp/kernels/bf16_expert_tile/install.py"
)
_PKG = "_mso_pp_bf16_tile_under_test"


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(("info", text))

    def warning(self, text):
        self.lines.append(("warning", text))


class _Tuner:
    instance = None

    def __init__(self):
        self.own_calls = []

    @classmethod
    def get(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        self.own_calls.append(custom_op)
        return runners[0] if runners else None, -1


class _Runner:
    def __init__(self, tactics):
        self.tactics = tactics

    def get_valid_tactics(self, inputs, profile):
        return self.tactics


LOG = _Log()
DISPATCH = types.ModuleType(f"{_PKG}.dispatch")
DISPATCH.TAG = "[musespark_open pp]"
DISPATCH.gathers = []
DISPATCH.peers = None
DISPATCH.layers = []


def _group_size(mapping):
    return int(getattr(mapping, "pp_size", 1) or 1) if mapping is not None else 1


def _allgather(mapping, local):
    DISPATCH.gathers.append(local)
    return [local] + list(DISPATCH.peers or [])


def _agreed_install(mapping, build, refusals=(ValueError,)):
    return build()


DISPATCH.group_size = _group_size
DISPATCH._allgather = _allgather
DISPATCH.agreed_install = _agreed_install
DISPATCH.owned_layers = lambda text_model: list(DISPATCH.layers)

RING_BUDDY = types.ModuleType("ring_buddy")
RING_BUDDY.expert_format = lambda moe: getattr(moe, "fmt", None)
RING_BUDDY._backend = lambda moe: moe.backend
TEP = types.ModuleType(f"{_PKG}._tep")
TEP.module = lambda name: {"kernels.replicated_experts.ring_buddy": RING_BUDDY}[name]

FLASHINFER = types.ModuleType("flashinfer")
FLASHINFER.__path__ = []
AUTOTUNER = types.ModuleType("flashinfer.autotuner")
AUTOTUNER.AutoTuner = _Tuner


def _load():
    stubs = {
        "tensorrt_llm": types.ModuleType("tensorrt_llm"),
        "tensorrt_llm.logger": types.ModuleType("tensorrt_llm.logger"),
    }
    stubs["tensorrt_llm"].__path__ = []
    stubs["tensorrt_llm.logger"].logger = LOG
    for name in (_PKG, f"{_PKG}.kernels", f"{_PKG}.kernels.bf16_expert_tile"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules[f"{_PKG}.dispatch"] = DISPATCH
    sys.modules[f"{_PKG}._tep"] = TEP
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        name = f"{_PKG}.kernels.bf16_expert_tile.install"
        spec = importlib.util.spec_from_file_location(name, _FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


TILE = _load()


@pytest.fixture
def flashinfer_stub(monkeypatch):
    _Tuner.instance = None
    monkeypatch.setitem(sys.modules, "flashinfer", FLASHINFER)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", AUTOTUNER)
    yield _Tuner
    _Tuner.instance = None


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # The host-only run has no CUDA stream to capture.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    DISPATCH.gathers.clear()
    DISPATCH.peers = None
    DISPATCH.layers = []
    LOG.lines.clear()
    TILE._LOGGED.clear()
    TILE._PIN.tactic = None


def test_the_pinned_tactic():
    assert TILE.PINNED_TACTIC == (128, 0)


def test_chooser_routes_only_the_pinned_op(flashinfer_stub):
    TILE._install_chooser()
    TILE._install_chooser()
    tuner = flashinfer_stub.get()
    assert tuner._musespark_open_pp_tile_pin is True
    runner = _Runner([[64, 0], [128, 0], [128, 3]])
    assert tuner.choose_one(TILE.OP, [runner], None, []) == (runner, -1)
    TILE._PIN.tactic = (128, 0)
    assert tuner.choose_one(TILE.OP, [runner], None, []) == (runner, [128, 0])
    assert tuner.choose_one("flashinfer::other_op", [runner], None, []) == (runner, -1)
    assert tuner.own_calls == [TILE.OP, "flashinfer::other_op"]


def test_chooser_keeps_the_own_choice_for_an_invalid_tactic(flashinfer_stub):
    TILE._install_chooser()
    tuner = flashinfer_stub.get()
    TILE._PIN.tactic = (128, 0)
    runner = _Runner([[64, 0], [64, 1]])
    shape = types.SimpleNamespace(shape=(4096, 8192))
    assert tuner.choose_one(TILE.OP, [runner], None, [shape]) == (runner, -1)
    assert tuner.choose_one(TILE.OP, [runner], None, [shape]) == (runner, -1)
    warnings = [text for level, text in LOG.lines if level == "warning"]
    assert len(warnings) == 1 and "not among the op's valid tactics" in warnings[0]


class _Experts:
    def __init__(self):
        self.seen = []

    def forward(self, x, router_logits):
        self.seen.append(TILE._PIN.tactic)
        if x.shape[0] == 4097:
            raise RuntimeError("boom")
        return x * 2


def _moe():
    return types.SimpleNamespace(experts=_Experts())


@pytest.mark.parametrize(
    "rows, dtype, pinned",
    [
        (4096, torch.bfloat16, True),
        (6144, torch.bfloat16, True),
        (8192, torch.bfloat16, True),
        (4095, torch.bfloat16, False),
        (8193, torch.bfloat16, False),
        (4096, torch.float16, False),
    ],
)
def test_wrapped_forward_pins_only_the_row_window(rows, dtype, pinned):
    moe = _moe()
    TILE._wrap(moe, 0, (128, 0))
    x = torch.ones(rows, 2, dtype=dtype)
    out = moe.experts.forward(x, None)
    assert torch.equal(out, x * 2)
    assert moe.experts.seen == [(128, 0) if pinned else None]
    assert TILE._PIN.tactic is None


def test_wrapped_forward_skips_3d_and_restores_on_error():
    moe = _moe()
    TILE._wrap(moe, 61, (128, 0))
    moe.experts.forward(torch.ones(1, 4096, 2, dtype=torch.bfloat16), None)
    assert moe.experts.seen == [None]
    TILE._PIN.tactic = (64, 9)
    with pytest.raises(RuntimeError, match="boom"):
        moe.experts.forward(torch.ones(4097, 2, dtype=torch.bfloat16), None)
    assert moe.experts.seen[-1] == (128, 0)
    assert TILE._PIN.tactic == (64, 9)


def _layer(index, fmt="bf16", slots=256, size=256, start=0, comm=None):
    backend = types.SimpleNamespace(
        num_slots=slots, expert_size_per_partition=size, slot_start=start
    )
    moe = types.SimpleNamespace(
        experts=types.SimpleNamespace(comm=comm, forward=lambda x, r: x),
        fmt=fmt,
        backend=backend,
    )
    return types.SimpleNamespace(layer_idx=index, feed_forward=moe)


def test_install_pins_only_the_bf16_leaf_layers(flashinfer_stub):
    DISPATCH.layers = [
        _layer(0),
        _layer(1, fmt="nvfp4"),
        _layer(2, slots=64),
        _layer(3, comm=object()),
    ]
    text_model = types.SimpleNamespace(_pp_stage=0)
    assert TILE.install(text_model, types.SimpleNamespace(pp_size=4)) == [0]
    assert flashinfer_stub.get()._musespark_open_pp_tile_pin is True
