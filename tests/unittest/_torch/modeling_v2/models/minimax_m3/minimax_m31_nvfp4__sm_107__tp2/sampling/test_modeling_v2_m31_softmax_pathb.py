# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 QA gates for the Path-B catalog, routing and capture lifecycle."""

import importlib
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.sampling import (  # noqa: E501
    m31_softmax_mc as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.sampling import (  # noqa: E501
    m31_softmax_pathb as entry,
)


def _gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 required")
    assert entry.softmax_pathb_op() is not None, "Path-B native library must be installed"


def _same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize(
    "cell", entry.m31_softmax_pathb.CELLS, ids=lambda cell: str(cell.spec["rows"])
)
def test_every_integer_row(cell) -> None:
    """Each catalog cell runs through the actual sampler override and stock oracle."""
    _gpu()
    rows = cell.spec["rows"]
    g = torch.Generator(device="cuda").manual_seed(1700 + rows)
    logits = torch.randn(rows, cell.spec["d"], device="cuda", generator=g)
    temperature = torch.linspace(0.1, 2.0, rows, device="cuda")
    with mock.patch.object(dispatch, "m31_softmax_mc", side_effect=AssertionError("Path A called")):
        actual = dispatch.softmax_mc(logits, temperature)
    assert actual is not None
    _same(actual, entry.m31_softmax_pathb.reference(logits, temperature))


@pytest.mark.parametrize("rows", [129, 224, 448])
def test_graph_changed_inputs_without_loading(rows) -> None:
    """A preloaded operation captures without loading libraries or building a workspace."""
    _gpu()
    logits = torch.randn(rows, entry.VOCAB, device="cuda")
    temperature = torch.ones(rows, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dispatch.softmax_mc(logits, temperature)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        mock.patch.object(
            torch.ops, "load_library", side_effect=AssertionError("load inside capture")
        ),
        mock.patch.object(
            dispatch, "prebuild", side_effect=AssertionError("Path-A workspace requested")
        ),
    ):
        with torch.cuda.graph(graph, stream=stream):
            actual = dispatch.softmax_mc(logits, temperature)
    assert actual is not None
    for seed in (17, 83):
        g = torch.Generator(device="cuda").manual_seed(seed)
        logits.normal_(generator=g)
        temperature.uniform_(0.1, 3.0, generator=g)
        actual.fill_(float("nan"))
        graph.replay()
        _same(actual, entry.m31_softmax_pathb.reference(logits, temperature))


def test_capture_first_use_does_not_load_or_poison_state(monkeypatch) -> None:
    """An unprepared captured call declines; a later uncaptured load remains possible."""
    monkeypatch.setattr(entry, "_STATE", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with mock.patch.object(torch.ops, "load_library", side_effect=AssertionError("capture load")):
        assert entry.softmax_pathb_op() is None
    assert entry._STATE == {}


@pytest.mark.parametrize(
    "kill", ["SOFTMAX_PATHB", "SOFTMAX_MC_WIDE", "SOFTMAX_MC", "SOFTMAX_MC_128"]
)
def test_import_time_kill_hierarchy(kill, monkeypatch) -> None:
    _gpu()
    logits = torch.zeros(129, entry.VOCAB, device="cuda")
    switches = importlib.import_module(
        entry.__package__.rsplit(".ops.sampling", 1)[0] + ".m31_switches"
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TRTLLM_M31_DISABLE_" + kill, "1")
            importlib.reload(switches)
            importlib.reload(entry)
            assert entry.ENABLED == (kill == "SOFTMAX_MC_128")
            assert (dispatch.softmax_mc(logits, None) is not None) == (kill == "SOFTMAX_MC_128")
    finally:
        importlib.reload(switches)
        importlib.reload(entry)


def test_invalid_inputs_and_native_alias_guards() -> None:
    _gpu()
    x = torch.zeros(129, entry.VOCAB, device="cuda")
    op = entry.m31_softmax_pathb
    assert op(x[:128], None) is None
    assert op(torch.empty(449, entry.VOCAB, device="cuda"), None) is None
    assert op(x.half(), None) is None
    assert op(x[:, :-4], None) is None
    assert op(x[:, :-4].contiguous(), None) is None
    flat = torch.empty(x.numel() + 1, device="cuda")
    assert op(flat[1:].view_as(x), None) is None
    for temp in (
        True,
        "1.0",
        torch.ones(128, device="cuda"),
        torch.ones(129),
        torch.ones(129, device="cuda", dtype=torch.float64),
    ):
        assert op(x, temp) is None
    native = entry.softmax_pathb_op()
    with pytest.raises(RuntimeError, match="not overlap"):
        native(x, None, 1.0, x)
    out = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="nonoverlapping"):
        native(x, out.view(-1)[:129], 0.0, out)


def test_native_fake_schema() -> None:
    _gpu()
    x = torch.randn(129, entry.VOCAB, device="cuda")
    out = torch.empty_like(x)
    torch.library.opcheck(
        entry.softmax_pathb_op(),
        (x, None, 1.0, out),
        test_utils=("test_schema", "test_faketensor"),
    )
