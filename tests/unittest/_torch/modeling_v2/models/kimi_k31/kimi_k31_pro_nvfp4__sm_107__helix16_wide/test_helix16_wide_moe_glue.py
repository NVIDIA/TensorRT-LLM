# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE glue of ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` around FC12 (CPU only, extracted
from ``modeling.py`` with stand-ins for the kernels): the ULL front's routing release gate; the
bank's EP window refusal; ``Fc12Experts.prepare`` (tuning widest first, the plain launches, the
tactic lookup, its refusals); routing's record of the switches is the target's.
"""

from __future__ import annotations

import contextlib
import enum
import importlib.util
import sys
import types
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
from _helix16_wide_cpu import (
    family_dir,
    literal,
    load_weights_module,
    modeling_path,
    source_objects,
)

pytestmark = pytest.mark.cpu_only

VENDOR = "kernels/vendor/vllm_mr8/specialized_decode/cutedsl"
# The package the extracted code's relative imports resolve in (``__package__`` FAKE.target).
FAKE = "_k31_helix16_wide_fake"
CONSTANTS = (
    "_FC12_MMA_N", "_FC12_DIE_SCHED", "_FC12_DIE_CLAIM", "_FC12_DIE_MIN_TOKENS",
    "_FC12_LOOKAHEAD", "_FC12_SM_RESERVE", "_FC12_L2_PREFETCH", "_FC12_LEVERS", "_MOE_EXPERTS",
    "_MOE_ACT_SITU", "_MOE_SITU", "_MOE_SWIGLU_LIMIT", "_MOE_EP", "_MOE_ALLTOALL",
)  # fmt: skip
ROWS = sorted({*range(1, 17), *range(8, 129, 8)})
TACTIC = ((128, 256, 256), (128, 256, 128), (1, 1))


def _fake_packages(monkeypatch, leaves: dict) -> None:
    """``FAKE`` and its subpackages holding ``leaves`` (dotted name under ``FAKE`` -> module)."""
    names = {FAKE}
    for leaf in leaves:
        parts = leaf.split(".")
        names.update(".".join([FAKE, *parts[:i]]) for i in range(1, len(parts)))
    for name in sorted(names):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for leaf, module in leaves.items():
        name = f"{FAKE}.{leaf}"
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)


@pytest.fixture(scope="module")
def policy():
    """The vendored ``common/policy.py`` (``batch_table``, pure Python), loaded by path."""
    path = family_dir() / VENDOR / "common/policy.py"
    spec = importlib.util.spec_from_file_location("_k31_helix16_wide_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Front:
    def __init__(self, serialize_experts=False):
        self.serialize_experts, self.released = serialize_experts, None

    def release_early(self, batches):
        self.released = sorted(batches)


def test_routing_releases_where_the_ull_target_does(monkeypatch, policy):
    early = literal(family_dir() / VENDOR / "moe/prepared.py", "RELEASE_EARLY_WIDE")
    vendored = {
        "cutedsl.moe.prepared": SimpleNamespace(RELEASE_EARLY_WIDE=early),
        "cutedsl.common.policy": policy,
    }
    loader = types.ModuleType(f"{FAKE}.kernels.vendor.vllm_mr8")
    loader.module = vendored.__getitem__
    _fake_packages(monkeypatch, {"kernels.vendor.vllm_mr8": loader})
    ns = source_objects(
        modeling_path(),
        ("NARROW_SMS", "release_routing_early"),
        {"__package__": f"{FAKE}.target", "Iterable": Iterable},
    )
    release, table = ns["release_routing_early"], policy.batch_table(early)
    small = [r for r in ROWS if r <= 64]
    rest = sorted({r - 64 for r in ROWS if r > 64})
    for rows, want in ((small, [3, 4, 5, 6, 7, 8, 64]), ([64], [64]), (rest, [8, 64])):
        front = _Front()
        release(front, rows, sms=200, tp=16)
        # ``PreparedMoE.__init__``'s rows: ``b for b in state.batches if early.get(b)``.
        assert front.released == want == sorted(b for b in rows if table.get(b))
    # Its condition: Rubin-class SM counts, TP16, no native shared experts.
    for front, sms, tp in ((_Front(), 152, 16), (_Front(True), 200, 16), (_Front(), 200, 8)):
        release(front, small, sms=sms, tp=tp)
        assert front.released is None


@pytest.fixture(scope="module")
def weights():
    return load_weights_module()


def _bank(weights, rank, **extra):
    u8 = dict(dtype=torch.uint8)
    bank = {
        "fc1_w": torch.zeros((weights.LOCAL, 2, 2), **u8),
        "fc2_w": torch.zeros((weights.LOCAL, 2, 2), **u8),
        "fc1_s": torch.zeros(4, **u8),
        "fc2_s": torch.zeros(4, **u8),
        "fc31_in_scale": torch.ones(1),
        "fc2_in_scale": torch.ones(1),
        "fc31_alpha": torch.ones(weights.LOCAL),
        "fc2_alpha": torch.ones(weights.LOCAL),
        "window": weights.expert_window(rank),
    }
    bank.update(extra)
    return bank


class _ActivationType(enum.IntEnum):
    Swiglu = 5
    SiTu = 9


class _Runner:
    """``RubinFc12Runner`` stood in: its constructor arguments recorded."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def get_tuning_config(self):
        return "tuning config"


@pytest.fixture
def fc12(monkeypatch, weights):
    """``Fc12Experts`` from the source, on stand-ins: the kernels record their launches (in or out
    of tuning mode), the tuner's cache holds ``state.tuned`` row counts."""
    state = SimpleNamespace(
        tuning=False, capturing=False, tuned=set(), launches=[], lookups=[], warnings=[],
        fail_lookup=False,
    )  # fmt: skip

    @contextlib.contextmanager
    def autotune():
        state.tuning = True
        try:
            yield
        finally:
            state.tuning = False

    def search_cache(key, runners, shapes, config):
        if state.fail_lookup:
            raise RuntimeError("no cache")
        state.lookups.append((key, runners[0], shapes[0][0], config))
        rows = shapes[0][0]
        return (True, 0, TACTIC, 0.0) if rows in state.tuned else (False, 0, -1, float("inf"))

    tuner = SimpleNamespace(profiling_cache=SimpleNamespace(search_cache=search_cache))
    autotuner = types.ModuleType("tensorrt_llm._torch.autotuner")
    autotuner.autotune, autotuner.AutoTuner = autotune, SimpleNamespace(get=lambda: tuner)
    utils = types.ModuleType("tensorrt_llm._torch.utils")
    utils.ActivationType = _ActivationType
    monkeypatch.setitem(sys.modules, autotuner.__name__, autotuner)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    launch = types.ModuleType(f"{FAKE}.kernels.moe.rubin_fc12_launch")
    launch.TUNER_KEY, launch.RubinFc12Runner = "kimi_k31::rubin_fc12", _Runner
    _fake_packages(monkeypatch, {"kernels.moe.rubin_fc12_launch": launch})

    class _Torch(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    proxy = _Torch("torch")
    proxy.cuda = SimpleNamespace(
        is_current_stream_capturing=lambda: state.capturing, synchronize=lambda *device: None
    )

    def rubin_fc12(*args, **kwargs):
        state.launches.append((state.tuning, args[0].shape[0], args[16:], kwargs))

    def rubin_moe_sort(ids, scales, *args):
        return tuple(torch.zeros(ids.shape[0], dtype=torch.int32) for _ in range(6))

    def fp4_quantize(latent, scale, vec, *flags):
        rows, width = latent.shape
        u8 = dict(dtype=torch.uint8)
        return torch.zeros((rows, width // 2), **u8), torch.zeros(rows * width // vec, **u8)

    ns = {
        "__package__": f"{FAKE}.target",
        "torch": proxy,
        "Dict": Dict,
        "Any": Any,
        "Sequence": Sequence,
        "expert_split": weights.expert_split,
        "expert_window": weights.expert_window,
        "EXPERTS_PER_RANK": weights.LOCAL,
        "rubin_fc12": rubin_fc12,
        "rubin_moe_sort": rubin_moe_sort,
        "fp4_quantize": fp4_quantize,
        "logger": SimpleNamespace(warning=state.warnings.append),
    }
    source_objects(modeling_path(), (*CONSTANTS, "Fc12Experts"), ns)
    return ns, state


def test_the_bank_window_must_be_the_ranks(fc12, weights):
    """The loader cuts the bank at the engine rank, the chain routes at the plain TP16 rank: a
    bank of another EP window is refused, as is a bank without its window."""
    ns, _ = fc12
    for rank in range(16):
        experts = ns["Fc12Experts"](_bank(weights, rank), rank=rank)
        assert (experts.offset, experts.local) == (weights.LOCAL * (rank % 8), weights.LOCAL)
    # Rank 11 holds window 3 at the other TP half: the same window, so the ids route right.
    ns["Fc12Experts"](_bank(weights, 3), rank=11)
    with pytest.raises(ValueError, match="cut at different ranks"):
        ns["Fc12Experts"](_bank(weights, 3), rank=5)
    bank = _bank(weights, 2)
    del bank["window"]
    with pytest.raises(ValueError, match="cut at different ranks"):
        ns["Fc12Experts"](bank, rank=2)


def test_prepare_tunes_widest_first_then_launches_every_row_count(fc12, weights):
    ns, state = fc12
    rank = 13
    experts = ns["Fc12Experts"](_bank(weights, rank), rank=rank)
    out = torch.zeros((128, 64), dtype=torch.bfloat16)
    state.tuned = {r for r in ROWS if r != 7}
    tactics = experts.prepare(ROWS, out, tune=True)
    assert [r for tuning, r, *_ in state.launches if tuning] == sorted(ROWS, reverse=True)
    assert [r for tuning, r, *_ in state.launches if not tuning] == ROWS
    assert state.launches[-1] == (False, 128, state.launches[-1][2], ns["_FC12_LEVERS"])
    offset = weights.LOCAL * (rank % 8)
    scalars = (896, 16, weights.LOCAL, offset, 128, float("inf"), 8, False, 16, 9, 4.0, 25.0, None)
    levers = ns["_FC12_LEVERS"]
    assert all(args == scalars and kwargs == levers for *_, args, kwargs in state.launches)
    assert ns["_FC12_LEVERS"]["mma_n"] == 256
    # The tactic of each row count, from the runner the launcher builds from those arguments.
    assert tactics == experts.tactics == {r: (TACTIC if r != 7 else -1) for r in ROWS}
    keys = {(key, rows) for key, _, rows, _ in state.lookups}
    assert keys == {("kimi_k31::rubin_fc12", r) for r in ROWS}
    runner = state.lookups[0][1]
    assert runner.args == (896, 16, weights.LOCAL, offset, 128, 16)
    assert runner.kwargs == dict(
        swiglu_limit=float("inf"), ep_size=8, enable_alltoall=False,
        activation_type=_ActivationType.SiTu, situ_beta=4.0, situ_linear_beta=25.0,
        zeroed_counters=None, **ns["_FC12_LEVERS"],
    )  # fmt: skip


def test_prepare_without_tuning_and_its_refusals(fc12, weights):
    ns, state = fc12
    experts = ns["Fc12Experts"](_bank(weights, 0), rank=0)
    out = torch.zeros((128, 64), dtype=torch.bfloat16)
    assert experts.prepare(ROWS, out, tune=False) == {r: -1 for r in ROWS}
    assert [tuning for tuning, *_ in state.launches] == [False] * len(ROWS)
    # A failed lookup is a warning (the launches ran), the tactics unknown.
    state.launches.clear()
    state.fail_lookup = True
    assert experts.prepare([8], out, tune=False) == {8: "unknown"}
    assert len(state.launches) == 1 and "tactic lookup failed" in state.warnings[-1]
    state.capturing = True
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        experts.prepare(ROWS, out, tune=True)


def test_routing_records_what_the_target_reads():
    """routing's ``HELIX16_WIDE_ATTR`` record (``helix16_wide_options``) is the target's
    ``_WIDE_ATTR`` with the keys of ``_WIDE_DEFAULTS``; routing accepts the target's switch values."""
    routing = family_dir() / "routing.py"
    assert literal(modeling_path(), "_WIDE_ATTR") == literal(routing, "HELIX16_WIDE_ATTR")
    import os

    names = ("HELIX16_WIDE_TAIL_ENV", "HELIX16_WIDE_TAIL", "HELIX16_WIDE_FRONT_ENV",
             "HELIX16_WIDE_FRONT", "helix16_wide_options")  # fmt: skip
    options = source_objects(routing, names, {"os": os})["helix16_wide_options"]
    defaults = literal(modeling_path(), "_WIDE_DEFAULTS")
    assert options() == defaults
    # The tails and fronts routing accepts are the target's, its default first.
    for ours, theirs, key in (("WIDE_TAILS", "HELIX16_WIDE_TAIL", "tail"),
                              ("WIDE_FRONTS", "HELIX16_WIDE_FRONT", "front")):  # fmt: skip
        assert set(literal(routing, theirs)) == set(literal(modeling_path(), ours))
        assert literal(routing, theirs)[0] == defaults[key]
