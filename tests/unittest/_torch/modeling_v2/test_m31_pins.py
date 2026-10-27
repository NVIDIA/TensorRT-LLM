# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MiniMax-M3.1 bundled tactic pins: what a name-keyed entry serves on its first AutoTuner lookup."""

import json

import pytest
import torch

from tensorrt_llm import _bootstrap
from tensorrt_llm._torch import flashinfer_autotune_cache as fac
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_pins
from tensorrt_llm._torch.autotuner import AutoTuner, TunableRunner, TuningConfig

pytestmark = pytest.mark.cpu_only

UNIFIED_CACHE_ENV = (
    "TRTLLM_CACHE_DIR"  # tensorrt_llm defaults the AutoTuner cache path from it at import
)
OP = "m31_pins_test::gemm"


class _NamedRunner(TunableRunner):
    """Tactics are indices into ``names``; a rebuild may reorder that list."""

    def __init__(self, names):
        self.names = list(names)

    def get_valid_tactics(self, inputs, profile, **kwargs):
        return list(range(len(self.names)))

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0]

    def tactic_name(self, tactic):
        return "fallback" if tactic == -1 else self.names[tactic]

    def unique_id(self):
        return ("bf16",)


@pytest.fixture
def tuner(monkeypatch):
    t = AutoTuner.get()
    t.clear_cache()
    monkeypatch.setattr(t, "bundle_reporter", None)
    monkeypatch.setattr(t, "is_tuning_mode", False)
    for env in (
        m31_pins.DISABLE_ENV,
        m31_pins.DIR_ENV,
        m31_pins.AUTOTUNER_CACHE_ENV,
        UNIFIED_CACHE_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    # atexit is one module for the whole process: the stub must take whatever other callers pass.
    monkeypatch.setattr(m31_pins.atexit, "register", lambda *args, **kwargs: None)
    yield t
    t.clear_cache()


def _inputs():
    return [torch.empty(4, 8), torch.empty(8, 16)]


def _write_bundle(
    tmp_path, monkeypatch, tuner, runner, tactic, name, flashinfer=None, native_guard=None
):
    """A tp2 rank-0 bundle holding one entry for ``runner`` at ``_inputs()``'s shapes."""
    config = TuningConfig()
    key = tuner.profiling_cache.get_cache_key(
        OP, runner, tuple(tuner._get_input_sizes(_inputs())), config
    )
    bundle = {
        "format": m31_pins.FORMAT,
        "autotuner": [
            {
                "op": key[0],
                "runner": key[1],
                "runner_key": key[2],
                "shapes": [list(dims) for dims in key[3]],
                "runner_id": 0,
                "tactic": tactic,
                "tactic_name": name,
                "min_time": 0.01,
            }
        ],
        "flashinfer": flashinfer,
    }
    if native_guard is not None:
        bundle["native_guard"] = native_guard
    (tmp_path / "tp2").mkdir()
    (tmp_path / "tp2" / "rank_0.json").write_text(json.dumps(bundle))
    monkeypatch.setenv(m31_pins.DIR_ENV, str(tmp_path))
    return key, config


def test_entry_whose_tactic_keeps_its_name_is_served(tuner, tmp_path, monkeypatch):
    runner = _NamedRunner(["a", "b", "c"])
    key, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    reporter = m31_pins.install("tp2", 0, tuner)
    _, tactic = tuner.choose_one(OP, [runner], config, _inputs())
    assert tactic == 1
    assert reporter.unresolved_count == 0
    assert key not in tuner.profiling_cache.bundled_names


def test_reordered_build_serves_the_tactic_of_the_same_name(tuner, tmp_path, monkeypatch):
    runner = _NamedRunner(["c", "a", "b"])
    key, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    reporter = m31_pins.install("tp2", 0, tuner)
    _, tactic = tuner.choose_one(OP, [runner], config, _inputs())
    assert tactic == 2
    assert tuner.profiling_cache.cache[key][1] == 2
    assert reporter.unresolved_count == 0


def test_renamed_tactic_is_unresolved_and_dropped(tuner, tmp_path, monkeypatch):
    runner = _NamedRunner(["a", "b2", "c"])
    key, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    reporter = m31_pins.install("tp2", 0, tuner)
    _, tactic = tuner.choose_one(OP, [runner], config, _inputs())
    assert tactic == -1
    assert key not in tuner.profiling_cache.cache
    assert reporter.unresolved_count == 1


def test_kill_switch_loads_nothing(tuner, tmp_path, monkeypatch):
    _write_bundle(tmp_path, monkeypatch, tuner, _NamedRunner(["a", "b"]), 1, "b")
    monkeypatch.setenv(m31_pins.DISABLE_ENV, "1")
    assert m31_pins.install("tp2", 0, tuner) is None
    assert not tuner.profiling_cache.cache


def test_explicit_autotuner_cache_takes_precedence(tuner, tmp_path, monkeypatch):
    _write_bundle(tmp_path, monkeypatch, tuner, _NamedRunner(["a", "b"]), 1, "b")
    monkeypatch.setenv(m31_pins.AUTOTUNER_CACHE_ENV, str(tmp_path / "cache.json"))
    assert m31_pins.install("tp2", 0, tuner) is None
    assert not tuner.profiling_cache.cache


@pytest.mark.parametrize(
    "unified,cache,served",
    [
        (False, None, True),  # no AutoTuner cache: the bundle is served
        (False, "explicit", False),  # an explicit cache wins
        (True, "default", True),  # TRTLLM_CACHE_DIR's default path is not an explicit cache
        (True, "explicit", False),  # an explicit cache wins under TRTLLM_CACHE_DIR too
    ],
)
def test_bundle_vs_autotuner_cache_path(tuner, tmp_path, monkeypatch, unified, cache, served):
    runner = _NamedRunner(["a", "b"])
    _, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    root = str(tmp_path / "trtllm_cache")
    if unified:
        monkeypatch.setenv(UNIFIED_CACHE_ENV, root)
    if cache == "default":
        # The value tensorrt_llm sets at import from TRTLLM_CACHE_DIR.
        path = _bootstrap._unified_cache_defaults(root)[m31_pins.AUTOTUNER_CACHE_ENV]
        monkeypatch.setenv(m31_pins.AUTOTUNER_CACHE_ENV, path)
    elif cache == "explicit":
        monkeypatch.setenv(m31_pins.AUTOTUNER_CACHE_ENV, str(tmp_path / "elsewhere.json"))
    assert (m31_pins.install("tp2", 0, tuner) is not None) is served
    if served:
        _, tactic = tuner.choose_one(OP, [runner], config, _inputs())
        assert tactic == 1
    else:
        assert not tuner.profiling_cache.cache


def test_missing_bundle_loads_nothing(tuner, tmp_path, monkeypatch):
    monkeypatch.setenv(m31_pins.DIR_ENV, str(tmp_path))
    assert m31_pins.install("tp8", 3, tuner) is None


def test_flashinfer_section_needs_its_guard(tuner, tmp_path, monkeypatch):
    monkeypatch.setattr(fac, "_STATE", {"loaded": False, "logged": False, "bundled": None})
    have = {"flashinfer-python": "0.6.18", "nvidia-cutlass-dsl": "4.8.0"}
    monkeypatch.setattr(m31_pins, "_flashinfer_versions", lambda: have)
    fi_cache = {"_metadata": {}, "('mxfp8_gemm', (32, 4096))": ["CutlassMxfp8GemmRunner", 6]}
    _write_bundle(
        tmp_path,
        monkeypatch,
        tuner,
        _NamedRunner(["a"]),
        0,
        "a",
        flashinfer={"guard": dict(have, **{"flashinfer-python": "0.6.17"}), "cache": fi_cache},
    )
    reporter = m31_pins.install("tp2", 0, tuner)
    assert reporter.unresolved_count == 1
    assert fac._STATE["bundled"] is None


def test_flashinfer_section_with_matching_guard_is_served_load_only(tuner, tmp_path, monkeypatch):
    monkeypatch.setattr(fac, "_STATE", {"loaded": False, "logged": False, "bundled": None})
    monkeypatch.delenv(fac.CACHE_DIR_ENV, raising=False)
    have = {"flashinfer-python": "0.6.18", "nvidia-cutlass-dsl": "4.8.0"}
    monkeypatch.setattr(m31_pins, "_flashinfer_versions", lambda: have)
    fi_cache = {"_metadata": {}, "('mxfp8_gemm', (32, 4096))": ["CutlassMxfp8GemmRunner", 6]}
    _write_bundle(
        tmp_path,
        monkeypatch,
        tuner,
        _NamedRunner(["a"]),
        0,
        "a",
        flashinfer={"guard": have, "cache": fi_cache},
    )
    reporter = m31_pins.install("tp2", 0, tuner)
    assert reporter.unresolved_count == 0
    path = fac.cache_file()
    assert path is not None and json.loads(open(path).read()) == fi_cache
    assert fac.frozen()


GUARD = {"libtensorrt_llm.so": "aa", "libth_common.so": "bb"}


def test_converter_keeps_an_unnamed_runner_under_the_native_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(m31_pins, "_native_guard", lambda: dict(GUARD))
    cache = {
        "shared": {},
        "rank_0": {"('op', 'SomeRunner', '()', ((1, 2),))": {"runner_id": 0, "tactic": "3"}},
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    (written,) = m31_pins.convert(path, None, tmp_path / "out", 1)
    bundle = json.loads(written.read_text())
    assert bundle["autotuner"][0]["tactic_name"] is None and bundle["autotuner"][0]["tactic"] == 3
    assert bundle["native_guard"] == GUARD


@pytest.mark.parametrize("same_build", [True, False])
def test_unnamed_entry_is_served_only_on_the_same_native_build(
    tuner, tmp_path, monkeypatch, same_build
):
    monkeypatch.setattr(
        m31_pins,
        "_native_guard",
        lambda: dict(GUARD) if same_build else {"libtensorrt_llm.so": "cc"},
    )
    runner = _NamedRunner(["a", "b"])
    key, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, None, native_guard=GUARD)
    reporter = m31_pins.install("tp2", 0, tuner)
    _, tactic = tuner.choose_one(OP, [runner], config, _inputs())
    assert tactic == (1 if same_build else -1)
    assert reporter.unresolved_count == (0 if same_build else 1)
    assert key not in tuner.profiling_cache.bundled_names
