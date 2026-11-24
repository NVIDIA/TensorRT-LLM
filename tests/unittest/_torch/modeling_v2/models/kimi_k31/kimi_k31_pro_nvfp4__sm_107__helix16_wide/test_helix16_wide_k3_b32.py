# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K2 (k3 B32's q1 step at 32-64 requests), on the CPU, on the sources: ``compile_kda_decode``
builds the ``KdaSplit`` of 17-31 requests but the request count and raises outside 32-64 requests
or on a variant / repeated MR8 entry; ``kda_q1_split`` takes 32-64 at six heads only;
``prepare_decode`` runs twice on one layer and rebuilds an earlier 32 through K2 after a grow.
k3 B32's limits, N6 requests and select / accept: ``test_helix16_wide_draft_lens``."""

from __future__ import annotations

import ast
import contextlib
import functools
import sys
import types
from types import SimpleNamespace

import pytest
from _helix16_wide_cpu import family_dir, function_node, literal, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

K2_NAMES = (
    "KDA_Q1_PIN_REQUESTS", "KDA_Q1_PIN", "KDA_Q1_SCHEDULE", "kda_q1_split", "kda_decode_specs",
    "compile_kda_decode", "prepare_kda_q1_pinned",
)  # fmt: skip
KDA = "kernels/vendor/vllm_mr8/specialized_decode/cutedsl/kda"
MR8_PREPARED = f"{KDA}/prepared.py"
PINNED_LINE = "K31_WIDE_K2 q1 requests={} heads=6 base_schedule='16:1.0.0' dropped='32-64:3.16.3.2'"


def _specs(batch, state="f32"):
    """Stand-in operand layouts: (shape, stride, dtype) a key's ``executable`` reads (``specs[0]``'s
    rows, ``specs[9]``'s dtype: the recurrent state)."""
    rows = [((batch, 3216), (3216, 1), "bf16")] + [((i,), (1,), "f32") for i in range(1, 9)]
    rows += [((batch + 1, 6, 128, 128), (98304, 16384, 128, 1), state)]
    return tuple(rows + [((i,), (1,), "f32") for i in range(10, 15)])


def _compile_namespace(monkeypatch, schedule):
    """K2's names from the source, MR8's ``kda.prepared`` / ``common.host``, CuTe and torch stood
    in by recorders (MR8's ``SCHEDULE`` as given)."""
    runtime = types.ModuleType("cutlass.cute.runtime")
    runtime.make_fake_stream = lambda: "stream"
    for name, mod in (
        ("cutlass", types.ModuleType("cutlass")), ("cutlass.cute", types.ModuleType("cutlass.cute")),
        ("cutlass.cute.runtime", runtime),
    ):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, mod)
    built, compiled, logged, carved = [], [], [], []

    class KdaSplit:
        def __init__(self, *options):
            built.append(options)
            self.shared_carveout = 50 if options[1] == 48 else None

    prepared = types.ModuleType("_k31_helix16_wide_k2_kda_prepared")
    prepared.SCHEDULE, prepared.KdaSplit = {6: schedule}, KdaSplit
    host = SimpleNamespace(
        fake=lambda shape, stride, dtype: ("fake", shape),
        prepare_shared_carveout=lambda code, device, value: carved.append(value) or ("carved", code),
        layout=lambda t: t,
    )  # fmt: skip
    ns = {
        "cache": functools.cache, "logger": SimpleNamespace(info=logged.append),
        "module": {"cutedsl.kda.prepared": prepared, "cutedsl.common.host": host}.__getitem__,
        "cutlass": SimpleNamespace(Float32=lambda v: ("f32", v)),
        "cute": SimpleNamespace(compile=lambda *a: compiled.append(a) or f"code{len(compiled)}"),
        "torch": SimpleNamespace(
            bfloat16="bf16", int32="i32", accelerator=SimpleNamespace(
                device_index=lambda d: contextlib.nullcontext())),
    }  # fmt: skip
    source_objects(modeling_path(), K2_NAMES, ns)
    return (
        ns,
        prepared,
        SimpleNamespace(built=built, compiled=compiled, logged=logged, carved=carved),
    )


# --- K2 ------------------------------------------------------------------------------------------


def test_k2_builds_17_31_requests_kdasplit_at_32_64(monkeypatch):
    """``compile_kda_decode`` at 32 / 33 / 48 / 64 requests constructs ``KdaSplit`` with 17-31
    requests' options but the request count (schedule (0, 0, 0), q1, FP32 or BF16 state, not lean,
    no load norm), compiles it as ``executable`` does (13 operands, the three scalars, the rest, a
    fake stream; the carve-out where the kernel has one), logs ``K31_WIDE_K2`` once per build and
    leaves MR8's module as it was; outside 32-64 requests, or without the exact entry once, it
    raises."""
    schedule = literal(family_dir() / MR8_PREPARED, "SCHEDULE")[6]
    ns, prepared, rec = _compile_namespace(monkeypatch, schedule)
    assert schedule == "16:1.0.0,32-64:3.16.3.2" and ns["KDA_Q1_PIN"] == {6: "32-64:3.16.3.2"}
    assert list(ns["KDA_Q1_PIN_REQUESTS"]) == list(range(32, 65))
    assert ns["KDA_Q1_SCHEDULE"] == (0, 0, 0)
    before = dict(vars(prepared))
    build = ns["compile_kda_decode"]
    codes = [build(_specs(b), 6, 0, False, 0) for b in (32, 33, 48, 64)]
    assert codes == ["code1", "code2", ("carved", "code3"), "code4"]
    assert build(_specs(32), 6, 0, False, 0) == "code1"  # cached: no second build or line
    assert build(_specs(40, state="bf16"), 6, 0, False, 0) == "code5"
    assert rec.built == [
        (6, b, 0, False, (0, 0, 0), 1, False, False, False) for b in (32, 33, 48, 64)
    ] + [(6, 40, 0, False, (0, 0, 0), 1, True, False, False)]
    first = rec.compiled[0]
    assert first[1:14] == tuple(("fake", s[0]) for s in _specs(32)[:13])
    assert first[14:17] == (("f32", -5), ("f32", 128**-0.5), ("f32", 1e-6))
    assert first[17:] == (("fake", (13,)), ("fake", (14,)), "stream")
    assert rec.carved == [50]
    assert rec.logged == [PINNED_LINE.format(b) for b in (32, 33, 48, 64, 40)]
    assert dict(vars(prepared)) == before and prepared.SCHEDULE == {6: schedule}
    for b in (31, 65):
        with pytest.raises(ValueError, match=f"32-64 requests, not {b}"):
            build(_specs(b), 6, 0, False, 0)
    # The exact entry, once, or a raise: a variant or a repeat is never taken silently.
    for entries, count in (("16:1.0.0,32-64:3.16.3.2.1", 0), ("32-64:3.16.3.2,32-64:3.16.3.2", 2)):
        ns, prepared, rec = _compile_namespace(monkeypatch, entries)
        with pytest.raises(RuntimeError, match=f"holds '32-64:3.16.3.2' {count} times, not once"):
            ns["compile_kda_decode"](_specs(33), 6, 0, False, 0)
        assert rec.built == [] and prepared.SCHEDULE == {6: entries}


def _prepare_decode_namespace(built):
    """K2's split and build and ``FlatKda.prepare_decode`` from the source; ``kda_decode_specs`` /
    ``compile_kda_decode`` stood in by recorders of what they are given."""
    path = modeling_path()
    ns = {"cache": functools.cache}
    source_objects(
        path, ("KDA_Q1_PIN_REQUESTS", "KDA_Q1_PIN", "kda_q1_split", "prepare_kda_q1_pinned"), ns
    )
    ns["kda_decode_specs"] = lambda core, caches, batch, pitch: ((batch, pitch), repr(caches))
    ns["compile_kda_decode"] = lambda specs, *options: built.append((specs, options)) or (
        f"k2 {specs[0][0]}"
    )
    fn = function_node(path, "FlatKda.prepare_decode")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    ns["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        empty=lambda *a, **k: ("empty", a), bfloat16="bf16", float32="f32",
    )  # fmt: skip
    return ns


class _Core:
    """MR8's ``PreparedKda`` as ``prepare_decode`` sees it: ``prepare_caches`` builds every key of
    ``policies`` (MR8's ``executable``: recorded)."""

    def __init__(self, seen, policies=None, kernels=None):
        self.heads, self.has_bias, self.device = 6, False, SimpleNamespace(index=0)
        self.policies, self.kernels, self.seen = policies or {}, kernels or {}, seen

    def prepare_caches(self, caches):
        self.seen.append(sorted(self.policies))
        for key in self.policies:
            self.kernels[key] = f"mr8 {key} {caches!r}"


def _layer(core, capacity, caches):
    me = SimpleNamespace(
        decode_core=core, decode_capacity=capacity, decode_caches=caches,
        weights=SimpleNamespace(inputs=SimpleNamespace(pitch=3216, local_heads=6)),
        output=SimpleNamespace(device="cpu"), calls=[],
    )  # fmt: skip
    me._grow_output = lambda r: me.calls.append(("grow", r))
    me._prepare_projection = lambda r: me.calls.append(("proj", tuple(r)))
    return me


def test_prepare_decode_runs_twice_on_one_layer():
    """The second and third call (``decode_core`` built: the first-call branch, which binds
    ``module``, skipped) prepare 24 and then 32 requests: 24 is MR8's, 32 K2's (MR8's
    ``prepare_caches`` never sees it); every key stays known to the core."""
    seen, built = [], []
    ns = _prepare_decode_namespace(built)
    me = _layer(_Core(seen, {8: (3216, 0)}, {8: "old"}), 32, ("conv", "state"))
    ns["prepare_decode"](me, (24,))
    ns["prepare_decode"](me, (32,))
    assert seen == [[24], []]
    assert built == [(((32, 3216), "('conv', 'state')"), (6, 0, False, 0))]
    assert me.decode_core.kernels == {8: "old", 24: "mr8 24 ('conv', 'state')", 32: "k2 32"}
    assert set(me.decode_core.policies) == {8, 24, 32}
    assert me.calls == [("grow", 24), ("proj", (24,)), ("grow", 32), ("proj", (32,))]
    ns["prepare_decode"](me, (24, 32))  # both built: nothing new
    assert seen == [[24], [], []] and len(built) == 1


def test_a_grow_rebuilds_earlier_k2_rows_through_k2():
    """A later ``prepare_decode`` whose rows grow the scratch rebuilds every known key
    (``core.kernels`` cleared); a 32-64 key among them is built by K2 again (on the new scratch)
    although the new rows hold none, and MR8 builds the rest."""
    seen, built = [], []
    ns = _prepare_decode_namespace(built)
    me = _layer(_Core(seen), 0, None)
    ns["prepare_decode"](me, (16, 32))
    ns["prepare_decode"](me, (70,))  # grows: 16 and 70 MR8's, 32 K2's (outside 32-64: 70)
    assert seen == [[16], [16, 70]]
    assert [s[0] for s, _ in built] == [(32, 3216), (32, 3216)]
    assert built[0][0][1] != built[1][0][1]  # the second on the grown scratch
    assert me.decode_core.kernels[32] == "k2 32" and set(me.decode_core.kernels) == {16, 32, 70}


def test_k2_split_takes_32_64_at_six_heads_only():
    ns = source_objects(modeling_path(), ("KDA_Q1_PIN_REQUESTS", "KDA_Q1_PIN", "kda_q1_split"), {})
    policies = {b: (3216, 0) for b in (1, 16, 31, 32, 33, 48, 64, 65, 128)}
    mr8, k2 = ns["kda_q1_split"](6, policies)
    assert sorted(k2) == [32, 33, 48, 64] and sorted(mr8) == [1, 16, 31, 65, 128]
    assert ns["kda_q1_split"](12, policies) == (policies, {})  # TP8: MR8's own
