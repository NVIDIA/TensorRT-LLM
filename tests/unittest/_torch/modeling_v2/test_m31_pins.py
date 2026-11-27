# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1-Flash bundled tactic pins: what a name-keyed entry serves on its first AutoTuner lookup."""

import functools
import json
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm import _bootstrap
from tensorrt_llm._torch import flashinfer_autotune_cache as fac
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    AutoTunerProfilingCache,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import m31_pins, m31_switches

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
    # The profiling cache's metadata is the only device read; keep the host-only tests off the device.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "cpu-test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    t = AutoTuner.get()
    t.clear_cache()
    monkeypatch.setattr(t, "bundle_reporter", None)
    monkeypatch.setattr(t, "is_tuning_mode", False)
    monkeypatch.setattr(m31_switches, "_DISABLED", frozenset())
    for env in (
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


def _write_bundle(tmp_path, monkeypatch, tuner, runner, tactic, name, flashinfer=None):
    """A tp2 bundle (an AutoTuner cache file) holding one rank-0 entry for ``runner`` at ``_inputs()``'s shapes."""
    config = TuningConfig()
    key = tuner.profiling_cache.get_cache_key(
        OP, runner, tuple(tuner._get_input_sizes(_inputs())), config
    )
    value = {"runner_id": 0, "tactic": repr(tactic), "min_time": 0.01}
    if name is not None:
        value["tactic_name"] = name
    (tmp_path / "tp2").mkdir()
    (tmp_path / "tp2" / m31_pins.AUTOTUNER_FILE).write_text(
        json.dumps({"shared": {}, "rank_0": {str(key): value}})
    )
    if flashinfer is not None:
        m31_pins.flashinfer_file(tmp_path / "tp2", 0).write_text(json.dumps(flashinfer))
    monkeypatch.setenv(m31_pins.DIR_ENV, str(tmp_path))
    return key, config


@pytest.mark.parametrize(
    "names, bundled_name, expected",
    [
        (["a", "b", "c"], "b", 1),  # the tactic keeps its name: served
        (["c", "a", "b"], "b", 2),  # a reordered build serves the tactic of the same name
        (["a", "b2", "c"], "b", -1),  # a renamed tactic is unresolved and dropped
        (["a", "b"], None, -1),  # an unnamed entry is dropped and reported
    ],
)
def test_entry_is_served_by_tactic_name(
    tuner, tmp_path, monkeypatch, names, bundled_name, expected
):
    runner = _NamedRunner(names)
    key, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, bundled_name)
    reporter = m31_pins.install("tp2", 0, tuner)
    _, tactic = tuner.choose_one(OP, [runner], config, _inputs())
    assert tactic == expected
    assert reporter.unresolved_count == (0 if expected >= 0 else 1)
    if expected >= 0:
        assert tuner.profiling_cache.cache[key][1] == expected
        assert key not in tuner.profiling_cache.bundled_names
    else:
        assert key not in tuner.profiling_cache.cache


def test_kill_switch_or_missing_bundle_loads_nothing(tuner, tmp_path, monkeypatch):
    monkeypatch.setenv(m31_pins.DIR_ENV, str(tmp_path))
    assert m31_pins.install("tp8", 3, tuner) is None
    _write_bundle(tmp_path, monkeypatch, tuner, _NamedRunner(["a", "b"]), 1, "b")
    # The family switch registry reads TRTLLM_M31_DISABLE_BUNDLED_PINS once, at import.
    monkeypatch.setattr(m31_switches, "_DISABLED", frozenset({m31_pins.DISABLE_ID}))
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


@pytest.mark.parametrize("guard_matches", [False, True])
def test_flashinfer_section_needs_its_guard(tuner, tmp_path, monkeypatch, guard_matches):
    """A matching version guard serves the FlashInfer cache load-only; a mismatch is unresolved."""
    monkeypatch.setattr(fac, "_STATE", {"loaded": False, "logged": False, "bundled": None})
    monkeypatch.delenv(fac.CACHE_DIR_ENV, raising=False)
    # The bundled FlashInfer file lands under tmp_path, not the system temporary directory.
    monkeypatch.setattr(
        m31_pins.tempfile, "mkdtemp", functools.partial(tempfile.mkdtemp, dir=str(tmp_path))
    )
    have = {"flashinfer-python": "1.2.0", "nvidia-cutlass-dsl": "4.8.0"}
    monkeypatch.setattr(m31_pins, "_flashinfer_versions", lambda: have)
    fi_cache = {"_metadata": {}, "('mxfp8_gemm', (32, 4096))": ["CutlassMxfp8GemmRunner", 6]}
    guard = have if guard_matches else dict(have, **{"flashinfer-python": "1.1.0"})
    flashinfer = {"guard": guard, "cache": fi_cache}
    _write_bundle(tmp_path, monkeypatch, tuner, _NamedRunner(["a"]), 0, "a", flashinfer=flashinfer)
    reporter = m31_pins.install("tp2", 0, tuner)
    if not guard_matches:
        assert reporter.unresolved_count == 1
        assert fac._STATE["bundled"] is None
        return
    assert reporter.unresolved_count == 0
    path = fac.cache_file()
    assert path is not None and json.loads(open(path).read()) == fi_cache
    assert path.startswith(str(tmp_path))
    assert fac.frozen()


def test_bundled_flashinfer_file_is_served_under_the_unified_cache(tuner, tmp_path, monkeypatch):
    monkeypatch.setattr(fac, "_STATE", {"loaded": False, "logged": False, "bundled": None})
    monkeypatch.delenv(fac.CACHE_DIR_ENV, raising=False)
    have = {"flashinfer-python": "1.2.0", "nvidia-cutlass-dsl": "4.8.0"}
    monkeypatch.setattr(m31_pins, "_flashinfer_versions", lambda: have)
    root = tmp_path / "trtllm_cache"
    monkeypatch.setenv(UNIFIED_CACHE_ENV, str(root))
    flashinfer = {"guard": have, "cache": {"_metadata": {}}}
    _write_bundle(tmp_path, monkeypatch, tuner, _NamedRunner(["a"]), 0, "a", flashinfer=flashinfer)
    assert m31_pins.install("tp2", 0, tuner) is not None
    assert fac.cache_file().startswith(str(root / "flashinfer_autotune" / "bundled"))


def test_the_autotuner_records_tactic_names_and_a_bundle_is_its_cache_file(
    tuner, tmp_path, monkeypatch
):
    """A tuned entry's tactic name is saved beside it; the converter keeps the named entries, and the result loads
    back through the AutoTuner and serves the tactic of that name."""
    runner = _NamedRunner(["a", "b", "c"])
    config = TuningConfig()
    key = tuner.profiling_cache.get_cache_key(
        OP, runner, tuple(tuner._get_input_sizes(_inputs())), config
    )
    # What the profiling pass stores for its winner.
    tuner.profiling_cache[key] = (0, 1, 0.01)
    tuner._record_tactic_name(key, runner, 1)
    assert tuner.profiling_cache.tactic_names[key] == "b"
    unnamed = ("op", "SomeRunner", "()", ((1, 2),))
    tuner.profiling_cache.cache[unnamed] = (0, 3, 0.5)
    run = tmp_path / "run.json"
    tuner.profiling_cache.save_cache(run, rank=0)
    saved = json.loads(run.read_text())
    assert saved["rank_0"][str(key)]["tactic_name"] == "b"

    (bundle,) = m31_pins.convert(run, None, tmp_path / "tp2", 1)
    data = json.loads(bundle.read_text())
    assert list(data["rank_0"]) == [str(key)]

    tuner.clear_cache()
    monkeypatch.setenv(m31_pins.DIR_ENV, str(tmp_path))
    reordered = _NamedRunner(["c", "a", "b"])
    reporter = m31_pins.install("tp2", 0, tuner)
    assert reporter.unresolved_count == 0
    _, tactic = tuner.choose_one(OP, [reordered], config, _inputs())
    assert tactic == 2


def test_a_merged_entry_drops_its_recorded_name(tuner):
    key = ("op", "R", "()", ((1,),))
    tuner.profiling_cache.cache[key] = (0, 1, 0.5)
    tuner.profiling_cache.tactic_names[key] = "b"
    tuner.profiling_cache.merge_cache_data({key: (0, 2, 0.4)})
    assert key not in tuner.profiling_cache.tactic_names


@pytest.mark.parametrize("rank", [0, 1])
def test_shipped_tp2_bundle_is_name_keyed(rank):
    data = json.loads((m31_pins.BUNDLE_ROOT / "tp2" / m31_pins.AUTOTUNER_FILE).read_text())
    entries = data[f"rank_{rank}"]
    assert entries
    assert all(value["tactic_name"] for value in entries.values())
    cache = AutoTunerProfilingCache.__new__(AutoTunerProfilingCache)
    cache.cache, cache.bundled_names, cache.tactic_names = {}, {}, {}
    assert cache.load_bundle(m31_pins.BUNDLE_ROOT / "tp2" / m31_pins.AUTOTUNER_FILE, rank) == []
    assert len(cache.bundled_names) == len(entries)


def _bundle_entry(tuner, key, tactic, name):
    tuner.profiling_cache.cache[key] = (0, tactic, 0.01)
    tuner.profiling_cache.bundled_names[key] = name


def test_pending_bundle_entries_are_not_saved(tuner, tmp_path):
    config = TuningConfig()
    pending = tuner.profiling_cache.get_cache_key(
        OP, _NamedRunner(["a"]), tuple(tuner._get_input_sizes(_inputs())), config
    )
    resolved = tuner.profiling_cache.get_cache_key(
        OP + "_resolved", _NamedRunner(["a"]), tuple(tuner._get_input_sizes(_inputs())), config
    )
    _bundle_entry(tuner, pending, 1, "b")
    _bundle_entry(tuner, resolved, 0, "a")
    tuner.profiling_cache.bundled_names.pop(resolved)
    path = tmp_path / "c.json"
    tuner.profiling_cache.save_cache(path, rank=0)

    loaded = AutoTunerProfilingCache()
    loaded.load_cache(path, rank=0)
    assert pending not in loaded.cache
    assert resolved in loaded.cache


def test_loading_a_cache_keeps_the_bundled_entries_it_does_not_cover(tuner, tmp_path):
    config = TuningConfig()
    sizes = tuple(tuner._get_input_sizes(_inputs()))
    covered = tuner.profiling_cache.get_cache_key(OP, _NamedRunner(["a"]), sizes, config)
    other = tuner.profiling_cache.get_cache_key(OP + "_other", _NamedRunner(["a"]), sizes, config)
    persisted = AutoTunerProfilingCache()
    persisted.cache[covered] = (0, 2, 0.01)
    path = tmp_path / "c.json"
    persisted.save_cache(path, rank=0)
    _bundle_entry(tuner, covered, 1, "b")
    _bundle_entry(tuner, other, 0, "a")

    tuner.profiling_cache.load_cache(path, rank=0)

    from_file = AutoTunerProfilingCache()
    from_file.load_cache(path, rank=0)
    assert tuner.profiling_cache.cache[covered] == from_file.cache[covered]  # the file wins
    assert covered not in tuner.profiling_cache.bundled_names
    assert tuner.profiling_cache.cache[other] == (0, 0, 0.01)
    assert tuner.profiling_cache.bundled_names[other] == "a"


def test_mxfp8_sync_skips_pending_bundle_entries(tuner, monkeypatch):
    runner = SimpleNamespace(register_tactic=mock.Mock())
    monkeypatch.setattr(torch_custom_ops.MXFP8GemmRunner, "runner_dict", {("bf16",): runner})
    op = torch_custom_ops._MXFP8_AUTOTUNED_OP

    def key(m):
        return (op, "MXFP8GemmRunner", str(("bf16",)), ((m, 64), (m, 2), (128, 64), (128, 2)))

    _bundle_entry(tuner, key(96), 7, "stale")
    _bundle_entry(tuner, key(32), 3, "resolved")
    tuner.profiling_cache.bundled_names.pop(key(32))

    torch_custom_ops.MXFP8GemmRunner.sync_all_tactic_caches(tuner)

    runner.register_tactic.assert_called_once_with(32, 128, 64, 3)


class _JitNamedRunner(_NamedRunner):
    """A JIT-style runner (class-level ``kernel_cache``) that records the tactics it runs."""

    kernel_cache = {}

    def __init__(self, names):
        super().__init__(names)
        self.ran = []

    def forward(self, inputs, tactic=-1, **kwargs):
        self.ran.append(tactic)
        return inputs[0]


def test_priming_resolves_each_bundled_profile(tuner, monkeypatch):
    monkeypatch.setattr(tuner, "is_tuning_mode", True)
    monkeypatch.setattr(tuner, "_primed_cached_tactics", set())
    monkeypatch.delenv("TLLM_AUTOTUNER_PRIME_CACHED_TACTICS", raising=False)
    runner = _JitNamedRunner(["a", "b", "c"])
    config = TuningConfig(dynamic_tensor_specs=(DynamicTensorSpec(0, 0, (4, 8)),))
    inputs = _inputs()
    first, second = tuner._optimization_profiles(config, inputs)
    for profile, tactic, name in ((first, 1, "b"), (second, 0, "c")):
        key = tuner.profiling_cache.get_cache_key(
            OP, runner, profile.get_opt_shapes(), config, apply_map_to_tuning_buckets=False
        )
        _bundle_entry(tuner, key, tactic, name)

    _, tactic = tuner.choose_one(OP, [runner], config, inputs)

    assert tactic == 1
    # The second profile's stored index 0 names "a" in this build; its bundled name "c" is tactic 2.
    assert 0 not in runner.ran
    assert 2 in runner.ran


class _RaisingRunner(_NamedRunner):
    def get_valid_tactics(self, inputs, profile, **kwargs):
        raise RuntimeError("tactic enumeration failed")


def test_a_failed_name_check_leaves_no_unchecked_entry(tuner, tmp_path, monkeypatch):
    """The stored index names another tactic and enumeration raises: the entry is dropped too."""
    runner = _RaisingRunner(["a", "b"])
    key, config = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "a")
    m31_pins.install("tp2", 0, tuner)
    shapes = tuple(tuner._get_input_sizes(_inputs()))
    with pytest.raises(RuntimeError, match="enumeration failed"):
        tuner._search_cache_resolving_bundles(OP, [runner], shapes, config, True, _inputs(), None)
    assert key not in tuner.profiling_cache.cache
    assert key not in tuner.profiling_cache.bundled_names


def test_a_received_entry_drops_the_pending_bundled_name(tuner, tmp_path, monkeypatch):
    """A peer's value was tuned or checked there; this rank must not check it against its own name."""
    runner = _NamedRunner(["a", "b"])
    key, _ = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    m31_pins.install("tp2", 0, tuner)
    assert key in tuner.profiling_cache.bundled_names
    tuner.profiling_cache.merge_cache_data({key: (0, 0, 0.02)})
    assert key not in tuner.profiling_cache.bundled_names
    assert tuner.profiling_cache.cache[key] == (0, 0, 0.02)


def test_pending_bundled_entries_are_not_shared(tuner, tmp_path, monkeypatch):
    runner = _NamedRunner(["a", "b"])
    key, _ = _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    m31_pins.install("tp2", 0, tuner)
    other = ("other_op", "R", "()", ((1,),))
    tuner.profiling_cache.cache[other] = (0, 3, 0.5)
    gathered = []
    monkeypatch.setattr(tuner, "_is_distributed", lambda: True)
    monkeypatch.setattr(
        tuner,
        "_dist",
        SimpleNamespace(tp_cp_allgather=lambda obj: gathered.append(obj) or [obj]),
        raising=False,
    )
    tuner.post_tune_merge_tactics()
    assert key not in gathered[0] and other in gathered[0]
    assert key in tuner.profiling_cache.bundled_names  # still checked on its first lookup


def test_a_hit_on_another_key_builds_the_key_once(tuner, tmp_path, monkeypatch):
    """A pending name for one key does not send every other hit through the bundle path."""
    runner = _NamedRunner(["a", "b"])
    _write_bundle(tmp_path, monkeypatch, tuner, runner, 1, "b")
    m31_pins.install("tp2", 0, tuner)
    config = TuningConfig()
    other_inputs = [torch.empty(8, 8), torch.empty(8, 16)]
    other_key = tuner.profiling_cache.get_cache_key(
        OP, runner, tuple(tuner._get_input_sizes(other_inputs)), config
    )
    tuner.profiling_cache.cache[other_key] = (0, 0, 0.01)
    calls = []
    real = tuner.profiling_cache.get_cache_key

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tuner.profiling_cache, "get_cache_key", counting)
    shapes = tuple(tuner._get_input_sizes(other_inputs))
    hit, _, tactic, _ = tuner._search_cache_resolving_bundles(
        OP, [runner], shapes, config, True, other_inputs, None
    )
    assert hit and tactic == 0
    assert len(calls) == 1
