# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's ``kda_verify_pdl``: the KDA verify launched as the f_b GEMM's PDL dependent, and
``kda_verify_fused_fb``: the forget gate computed in the verify kernel from f_a (CPU, from source).

The verify kernel's PDL variant loads the request's state, replay entries, conv windows and
weights before ``griddepcontrol.wait`` and this layer's projections after it; its arithmetic and
epilogue are those of the plain launch (``ops/ssm/test_rubin_kda_verify_pdl.py`` compares the
bytes on the GPU behind a real f_b GEMM). Only the v2 kernel has that prologue, and it serves the
``mxfp8_exact`` output within one wave of (head, request) tiles. The same kernel computes the
forget gate from f_a (``ops/ssm/test_rubin_kda_fused_gate_linear.py`` and
``ops/ssm/test_rubin_kda_fused_gate_rows.py`` compare its bytes with the f_b GEMM path's).

helix8's ``_kda_verify_core`` runs from its source over recording stand-ins:

* the constants are logged generation constants: ``kda_verify_pdl`` on, read by
  ``_KDA_VERIFY_PDL``; ``kda_verify_fused_fb`` off, read by ``_KDA_VERIFY_FUSED_FB``;
* ``kda_verify_pdl`` off, no verify launches with PDL and nothing is logged;
* on, ``use_pdl`` is set exactly when the rank's 12 heads times the requests fit the SM count, on
  the MXFP8 and on the BF16 output, and the INFO line is logged exactly then; the early releases
  stay as they were (``kda_decode_early_trigger``, on; no ``mxfp8_pdl_early_trigger``) and the call
  takes ``_kda_gen_slots``'s slots;
* ``kda_verify_fused_fb`` on: the first eager verify at a row count within one wave runs the fused
  form and then today's on the same pools, restores the pools in between (the pools end as today's
  call alone leaves them) and returns today's output; equal output and pool words let every later
  verify at those rows skip the f_b GEMM and pass f_a / w_fb, unequal ones (or a refused call) keep
  the f_b GEMM with a WARNING; a capture before the check keeps the f_b GEMM and records nothing;
  past one wave, or with ``kda_verify_pdl`` off, nothing changes.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_HELIX8 = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__helix8"
    / "modeling.py"
)

_NAMES = (
    "_TARGET",
    "_CONSTANTS",
    "_DETAILS",
    "_K31",
    "_K31_KDA",
    "_KDA_TP",
    "_KDA_H",
    "_KDA_D",
    "_KDA_P",
    "_KDA_G0",
    "_KDA_FA0",
    "_KDA_B0",
    "_KDA_B1",
    "_KDA_LOWER_BOUND",
    "_KDA_EPS",
    "_KDA_SCALE",
    "_KDA_EARLY_TRIGGER",
    "_KDA_PREFETCH_ROWS",
    "_KDA_VERIFY_PDL",
    "_KDA_VERIFY_PDL_LINE",
    "_KDA_VERIFY_FUSED_FB",
    "_KDA_VERIFY_FUSED_FB_LINE",
    "_KDA_FUSED_FB_LINE",
    "_KDA_FUSED_FB_OFF",
    "_KDA_FUSED_FB_CHECKED",
    "_KDA_VERIFY_POOLS",
    "_kda_verify_pdl",
    "_kda_verify_fused_fb",
    "_kda_words",
    "_kda_fused_fb_check",
    "_kda_fb",
    "_kda_heads",
    "_kda_beta",
    "_kda_gen_slots",
    "_kda_verify_core",
    "constants_line",
)
#: helix8's verify rows per request (k5) and the fused projection's row pitch.
_ROWS_PER_GEN = 6
_FAB_N = 6288


def _definitions(names) -> list:
    """The module-level definitions and assignments of ``names`` in helix8's modeling.py, in source
    order; each name must be defined exactly once."""
    wanted, nodes, seen = set(names), [], {}
    for node in ast.parse(_HELIX8.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if defined & wanted:
            nodes.append(node)
            for name in defined & wanted:
                seen[name] = seen.get(name, 0) + 1
    assert seen == dict.fromkeys(wanted, 1), seen
    return nodes


class _Log:
    def __init__(self):
        self.lines: list = []

    def info_once(self, msg, key=None):
        self.lines.append(("info_once", msg))

    def info(self, msg):
        self.lines.append(("info", msg))

    def warning(self, msg):
        self.lines.append(("warning", msg))


def _ns(sms: int, on: bool, fused_fb: bool = False, verify=None) -> tuple[dict, list, _Log]:
    """The definitions over stand-ins of the projection, the verify entry, the SM count and the
    logger, with ``kda_verify_pdl`` set to ``on`` and ``kda_verify_fused_fb`` to ``fused_fb``;
    returns the namespace, the calls (verify keyword arguments; ``("project", gate)``;
    ``("f_b",)``) and the log."""
    calls: list = []
    log = _Log()

    def project(L, x, mx=None, gate=True):
        calls.append(("project", gate))
        rows = x.shape[0]
        g = torch.zeros(rows, ns["_KDA_P"], dtype=torch.bfloat16) if gate else None
        return torch.zeros(rows, _FAB_N, dtype=torch.bfloat16), g

    def record(**kwargs):
        calls.append(kwargs)
        return kwargs["x_q"]

    def linear(a, w, tf32):
        calls.append(("f_b",))
        return torch.zeros(a.shape[0], ns["_KDA_P"], dtype=torch.bfloat16)

    ns: dict[str, Any] = dict(
        torch=torch,
        functools=functools,
        Any=Any,
        Callable=Callable,
        Optional=Optional,
        MappingProxyType=MappingProxyType,
        logger=log,
        rubin_kda_mtp_verify=(lambda **kw: verify(calls, **kw)) if verify else record,
        _kda_project=project,
        _device_sms=lambda index: sms,
        rubin_linear=linear,
        # The written scale bytes of the stand-in outputs: the rows' leading bytes.
        mxfp8_scales_from_r128c4=lambda flat, rows, k: flat[: rows * (k // 32)],
    )
    module = ast.fix_missing_locations(ast.Module(_definitions(_NAMES), []))
    exec(compile(module, _HELIX8.name, "exec"), ns)
    ns["_KDA_VERIFY_PDL"] = on
    ns["_KDA_VERIFY_FUSED_FB"] = fused_fb
    ns["_KDA_PREFETCH_ROWS"] = 0  # the aux-stream prefetch fork needs a CUDA stream
    return ns, calls, log


def _pools(pool: int, p: int) -> SimpleNamespace:
    """A layer view whose written pools are small CPU tensors (state FP32, caches BF16 / FP32)."""
    gen = torch.Generator().manual_seed(pool)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=gen).to(dtype)

    m = _ROWS_PER_GEN - 1
    return SimpleNamespace(
        temporal=rand(pool, 2, 4),
        kda_conv_q=rand(pool, p, 3 + m, dtype=torch.bfloat16),
        kda_conv_k=rand(pool, p, 3 + m, dtype=torch.bfloat16),
        kda_conv_v=rand(pool, p, 3 + m, dtype=torch.bfloat16),
        kda_k_cache=rand(pool, m, p, dtype=torch.bfloat16),
        kda_g_cache=rand(pool, m, p),
        kda_v_cache=rand(pool, m, p, dtype=torch.bfloat16),
        kda_beta_cache=rand(pool, m, 16, dtype=torch.bfloat16)[..., :12],
        prev_num_accepted_tokens="accepted",
    )


def _verify(ns: dict, requests: int, mx_out: bool, lc=None):
    rows = requests * _ROWS_PER_GEN
    if lc is None:
        lc = SimpleNamespace(
            kda_conv_q="cs_q",
            kda_conv_k="cs_k",
            kda_conv_v="cs_v",
            temporal="state",
            kda_k_cache="k_cache",
            kda_g_cache="g_cache",
            kda_v_cache="v_cache",
            kda_beta_cache="beta_cache",
            prev_num_accepted_tokens="accepted",
        )
    st = SimpleNamespace(
        cache=SimpleNamespace(mamba_layer_cache=lambda layer: lc),
        kda_md=SimpleNamespace(
            generation_state_indices=torch.arange(1, requests + 1, dtype=torch.int32)
        ),
        num_contexts=0,
        num_generations=requests,
        rows_per_gen=_ROWS_PER_GEN,
    )
    L = SimpleNamespace(
        layer=3,
        fb="w_fb",
        conv_f32=torch.zeros(3 * ns["_KDA_P"], 4),
        a_log="a_log",
        dt_bias="dt_bias",
        o_norm_f32="o_norm",
        l2_next=(),
    )
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    return ns["_kda_verify_core"](L, x, st, SimpleNamespace(), mx_out=mx_out)


def test_the_constants_are_logged_generation_constants():
    ns, _, _ = _ns(sms=212, on=False)
    generation = ns["_CONSTANTS"]["generation"]
    assert generation["kda_verify_pdl"] is True
    # On in the shipped stack (the first-use device check keeps today's path on a mismatch).
    assert generation["kda_verify_fused_fb"] is True
    line = ns["constants_line"]("generation")
    assert "kda_verify_pdl=True" in line and "kda_verify_fused_fb=True" in line
    assigns = {
        t.id: ast.unparse(n.value)
        for n in ast.parse(_HELIX8.read_text()).body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert assigns["_KDA_VERIFY_PDL"] == "_CONSTANTS['generation']['kda_verify_pdl']"
    assert assigns["_KDA_VERIFY_FUSED_FB"] == "_CONSTANTS['generation']['kda_verify_fused_fb']"


@pytest.mark.parametrize("mx_out", (True, False), ids=("mxfp8_exact", "bf16"))
def test_off_no_verify_launches_with_pdl(mx_out):
    ns, calls, log = _ns(sms=212, on=False)
    for requests in range(1, 22):
        _verify(ns, requests, mx_out)
    verifies = [c for c in calls if isinstance(c, dict)]
    assert [c["use_pdl"] for c in verifies] == [False] * 21
    assert log.lines == []


@pytest.mark.parametrize("sms", (212, 148))
@pytest.mark.parametrize("mx_out", (True, False), ids=("mxfp8_exact", "bf16"))
def test_on_the_verify_launches_with_pdl_within_one_wave(sms, mx_out):
    ns, calls, log = _ns(sms=sms, on=True)
    one_wave = sms // ns["_KDA_H"]
    for requests in range(1, 22):
        log.lines.clear()
        _verify(ns, requests, mx_out)
        call = calls[-1]
        pdl = requests <= one_wave
        assert call["use_pdl"] is pdl, requests
        line = ns["_KDA_VERIFY_PDL_LINE"]
        assert log.lines == ([("info_once", line)] if pdl else []), requests
        assert call["early_trigger"] is ns["_KDA_EARLY_TRIGGER"] is True
        assert call["mxfp8_pdl_early_trigger"] is False
        assert call["quantize_output"] is mx_out and call["mxfp8_exact"] is mx_out
        assert call["fa"] is None and call["w_fb"] is None
        assert call["num_spec"] == _ROWS_PER_GEN - 1
        assert torch.equal(
            call["ssm_state_indices"], torch.arange(1, requests + 1, dtype=torch.int32)
        )


def _writer(differ: str = ""):
    """A verify stand-in that writes every pool row at its slots and returns an output: the
    fused form (``fa`` given) writes other output words (``differ="out"``), other g_cache words
    (``differ="g_cache"``) or is refused (``differ="refused"``)."""

    def verify(calls, **kw):
        calls.append(kw)
        fused = kw["fa"] is not None
        if fused:
            assert kw["g"] is None and kw["w_fb"] == "w_fb" and kw["fa"].shape[-1] == 128
        else:
            assert kw["fa"] is None and kw["w_fb"] is None and kw["g"].dim() == 4
        if fused and differ == "refused":
            raise ValueError("refused")
        idx = kw["ssm_state_indices"].long()
        for name in (
            "recurrent_state",
            "cs_q",
            "cs_k",
            "cs_v",
            "k_cache",
            "g_cache",
            "v_cache",
            "beta_cache",
        ):
            kw[name][idx] = kw[name][idx] * 0.5 + 1
        if fused and differ == "g_cache":
            kw["g_cache"][idx] += 1
        rows = kw["x_q"].shape[1]
        value = 2.0 if fused and differ == "out" else 1.0
        if kw["quantize_output"]:
            codes = torch.full((rows, 1536), value).to(torch.float8_e4m3fn)
            return codes, torch.full((rows * 48 + 512,), 127, dtype=torch.uint8)
        return torch.full((1, rows, 12, 128), value, dtype=torch.bfloat16)

    return verify


def _snapshot(lc) -> list:
    return [getattr(lc, n).clone() for n in ("temporal", "kda_g_cache", "kda_beta_cache")]


def _verifies(calls) -> list:
    return [("fused" if c["fa"] is not None else "f_b") for c in calls if isinstance(c, dict)]


@pytest.mark.parametrize("mx_out", (True, False), ids=("mxfp8_exact", "bf16"))
def test_fused_fb_runs_after_a_passing_device_check(monkeypatch, mx_out):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ns, calls, log = _ns(sms=212, on=True, fused_fb=True, verify=_writer())
    lc = _pools(24, ns["_KDA_P"])
    # The pools the f_b GEMM path alone leaves, for the check's call.
    ref_ns, _, _ = _ns(sms=212, on=True, verify=_writer())
    ref = _pools(24, ns["_KDA_P"])
    _verify(ref_ns, 9, mx_out, ref)
    out = _verify(ns, 9, mx_out, lc)
    # The check: the projection without f_b, the fused call, then today's [f_b GEMM -> verify].
    assert [c if not isinstance(c, dict) else _verifies([c])[0] for c in calls] == [
        ("project", False),
        "fused",
        ("f_b",),
        "f_b",
    ]
    for got, want in zip(_snapshot(lc), _snapshot(ref)):
        assert torch.equal(got, want)
    first = out[0] if mx_out else out
    assert bool((first.float() == 1.0).all())
    rows = 9 * _ROWS_PER_GEN
    assert ns["_KDA_FUSED_FB_CHECKED"] == {rows: True}
    assert log.lines == [
        ("info_once", ns["_KDA_VERIFY_PDL_LINE"]),
        ("info", ns["_KDA_FUSED_FB_LINE"].format(rows=rows)),
    ]
    # From then on these rows skip the f_b GEMM and pass f_a / w_fb with the PDL launch.
    calls.clear()
    log.lines.clear()
    _verify(ns, 9, mx_out, lc)
    _verify(ns, 9, mx_out, lc)
    assert [c for c in calls if not isinstance(c, dict)] == [("project", False)] * 2
    assert _verifies(calls) == ["fused", "fused"]
    call = calls[-1]
    assert call["use_pdl"] is True and call["early_trigger"] is True
    assert call["quantize_output"] is mx_out and call["mxfp8_exact"] is mx_out
    assert log.lines == [("info_once", ns["_KDA_VERIFY_FUSED_FB_LINE"])] * 2
    # Another row count is checked on its own; past one wave nothing changes.
    calls.clear()
    _verify(ns, 4, mx_out, lc)
    assert _verifies(calls) == ["fused", "f_b"]
    calls.clear()
    _verify(ns, 18, mx_out, lc)
    assert _verifies(calls) == ["f_b"] and calls[-1]["use_pdl"] is False
    assert sorted(ns["_KDA_FUSED_FB_CHECKED"]) == [24, 54]


@pytest.mark.parametrize("differ", ("out", "g_cache", "refused"))
def test_a_failing_device_check_keeps_the_f_b_gemm(monkeypatch, differ):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ns, calls, log = _ns(sms=212, on=True, fused_fb=True, verify=_writer(differ))
    lc = _pools(24, ns["_KDA_P"])
    ref_ns, _, _ = _ns(sms=212, on=True, verify=_writer())
    ref = _pools(24, ns["_KDA_P"])
    _verify(ref_ns, 2, True, ref)
    out = _verify(ns, 2, True, lc)
    assert bool((out[0].float() == 1.0).all())
    for got, want in zip(_snapshot(lc), _snapshot(ref)):
        assert torch.equal(got, want)
    rows = 2 * _ROWS_PER_GEN
    assert ns["_KDA_FUSED_FB_CHECKED"] == {rows: False}
    (warning,) = [m for kind, m in log.lines if kind == "warning"]
    assert warning.startswith(ns["_KDA_FUSED_FB_OFF"].split("{why}")[0].format(rows=rows))
    assert ("g_cache" in warning) is (differ == "g_cache")
    calls.clear()
    _verify(ns, 2, True, lc)
    assert [c for c in calls if not isinstance(c, dict)] == [("project", True)]
    assert _verifies(calls) == ["f_b"]
    # The checked row count is not checked again.
    assert ns["_KDA_FUSED_FB_CHECKED"] == {rows: False}


def test_a_capture_before_the_check_keeps_the_f_b_gemm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    ns, calls, log = _ns(sms=212, on=True, fused_fb=True, verify=_writer())
    _verify(ns, 9, True, _pools(24, ns["_KDA_P"]))
    assert _verifies(calls) == ["f_b"]
    assert ns["_KDA_FUSED_FB_CHECKED"] == {}
    assert log.lines == [("info_once", ns["_KDA_VERIFY_PDL_LINE"])]


def test_fused_fb_acts_with_kda_verify_pdl_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ns, calls, log = _ns(sms=212, on=False, fused_fb=True, verify=_writer())
    _verify(ns, 9, True, _pools(24, ns["_KDA_P"]))
    assert _verifies(calls) == ["f_b"] and calls[-1]["use_pdl"] is False
    assert ns["_KDA_FUSED_FB_CHECKED"] == {} and log.lines == []
