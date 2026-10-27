# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_ITEMS_IN_BCG: which tp2ctx items may run inside a breakable prefill CUDA graph capture.

Under any stream capture the items keep the stock path, except a breakable prefill graph capture with
TP2CTX_ITEMS_IN_BCG on (the default; TRTLLM_M31_DISABLE_TP2CTX_ITEMS_IN_BCG=1 turns it off), where NORMQ
(+ NF32 / N1b), MXFP8_CUBLASLT, SWIGLU_MXFP8, UNPERMUTE_ADD and GATE_CAST run (their hand-offs stay inside one
captured segment). C1's combine hand-off crosses the attention break and stays eager-only. In-model correctness is
the graph-path ids / GSM gates.
"""

from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    modeling as m,
)

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


@pytest.mark.parametrize(
    "capturing, in_bcg, opt_in, blocked",
    [
        (False, False, False, False),  # eager: never blocked
        (False, True, True, False),  # BCG replay of an eager break: not capturing
        (True, False, False, True),  # plain stream capture
        (True, False, True, True),  # plain stream capture, opt-in on: still blocked
        (True, True, False, True),  # BCG capture, opt-in off
        (True, True, True, False),  # BCG capture, opt-in on
    ],
)
def test_capture_gate(monkeypatch, capturing, in_bcg, opt_in, blocked):
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(m, "is_in_breakable_cuda_graph", lambda: in_bcg)
    monkeypatch.setattr(m, "_ITEMS_IN_BCG", opt_in)
    assert m._tp2ctx_capture_blocked() is blocked


def test_items_use_the_gate_and_c1_stays_eager():
    src = (_TARGET / "modeling.py").read_text()
    assert '_ITEMS_IN_BCG = not m31_disabled("TP2CTX_ITEMS_IN_BCG")' in src  # default ON
    # GATE_CAST, NORMQ boundary, MXFP8_CUBLASLT, SWIGLU_MXFP8, UNPERMUTE_ADD.
    assert src.count("_tp2ctx_capture_blocked()") == 5 + 1  # five guards + the def line
    for item in ("GATE_CAST", "NORMQ", "MXFP8_CUBLASLT", "SWIGLU_MXFP8", "UNPERMUTE_ADD"):
        assert f'_bcg_note("{item}"' in src
    sink = src[src.index("def _combine_mxfp8_sink(") :]
    sink = sink[: sink.index("\ndef ")]
    assert "is_in_breakable_cuda_graph()" in sink
    assert "torch.cuda.is_current_stream_capturing()" in sink
    assert "_tp2ctx_capture_blocked" not in sink


def test_bcg_note_once_per_item_and_bucket(monkeypatch):
    lines = []
    monkeypatch.setattr(m.logger, "info", lambda msg: lines.append(msg))
    monkeypatch.setattr(m, "_BCG_NOTED", set())
    x2k = m.torch.empty(2048, 6144, dtype=m.torch.bfloat16)
    x4k = m.torch.empty(4096, 6144, dtype=m.torch.bfloat16)
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: False)
    m._bcg_note("NORMQ", x2k)
    assert lines == []  # eager: no line
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: True)
    m._bcg_note("NORMQ", x2k)
    m._bcg_note("NORMQ", x2k)
    m._bcg_note("NORMQ", x4k)
    m._bcg_note("UNPERMUTE_ADD", (x2k,))  # not a tensor: the probe never raises
    assert lines == [
        "[M3.1 tp2ctx] TP2CTX_ITEMS_IN_BCG captured NORMQ (rows 2048) input torch.bfloat16 (2048, 6144)",
        "[M3.1 tp2ctx] TP2CTX_ITEMS_IN_BCG captured NORMQ (rows 4096) input torch.bfloat16 (4096, 6144)",
        "[M3.1 tp2ctx] TP2CTX_ITEMS_IN_BCG captured UNPERMUTE_ADD (rows -1) input tuple ()",
    ]


def test_a1_in_bcg_runs_only_inside_a_bcg_capture_with_a_warmed_handle(monkeypatch):
    attn = object()  # never dereferenced: every gate fails before the projection is read
    x = m.torch.empty(2048, 6144, dtype=m.torch.bfloat16)
    monkeypatch.setattr(m, "_ITEMS_IN_BCG", True)
    monkeypatch.setattr(m, "_A1_IN_BCG", True)
    monkeypatch.setattr(m, "is_in_breakable_cuda_graph", lambda: True)
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: False)
    assert m._a1_bcg_gemm(attn, x) is None  # not capturing: eager A1 / stock path
    monkeypatch.setattr(m, "_ITEMS_IN_BCG", False)
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: True)
    assert m._a1_bcg_gemm(attn, x) is None  # ITEMS_IN_BCG off
    monkeypatch.setattr(m, "_ITEMS_IN_BCG", True)
    monkeypatch.setattr(m, "_A1_IN_BCG", False)
    assert m._a1_bcg_gemm(attn, x) is None  # kill switch


def test_a1_in_bcg_wiring():
    src = (_TARGET / "modeling.py").read_text()
    assert '_A1_IN_BCG = not m31_disabled("TP2CTX_A1_IN_BCG")' in src
    # The captured projection tries A1's GEMM-only handle first; A1's eager take warms it.
    assert "packed = _a1_bcg_gemm(attn_layer, hidden_states)" in src
    assert "_a1_bcg_warm(proj, act, act_sf, num_tokens, n_total)" in src
    assert (
        "get_gemm(n_total, 4)" in src
    )  # the fused kernel's mainloop, heads_per_tile 4 as A1's runner
