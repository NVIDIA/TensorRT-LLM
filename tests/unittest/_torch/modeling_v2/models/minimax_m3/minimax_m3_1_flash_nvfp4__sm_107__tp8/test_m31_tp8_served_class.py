# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The imported TP8 worker and decode target classes.

The dispatch cases call complete, bound methods on real M3DSparkWorker and
DecodeTarget instances. Unrelated setup/collectives are dependency doubles.
The GPU cases keep the real FSEL / MoE / drafter GEMV providers through those entry points.
Single-rank identity reductions do not certify eight-rank serving or acceptance.
"""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm  # noqa: F401 -- register ops before checking CUDA or required ops
from tensorrt_llm._torch.modeling_v2._native import NoTarget

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"
PKG = f"tensorrt_llm._torch.modeling_v2.models.minimax_m3.{TARGET}"
NEEDS_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="native provider/replay test requires SM107",
)


@pytest.fixture(scope="module")
def served():
    try:
        return SimpleNamespace(
            dspark=importlib.import_module(f"{PKG}.dspark"),
            modeling=importlib.import_module(f"{PKG}.modeling"),
            drafter_mxfp8=importlib.import_module(f"{PKG}.drafter_mxfp8"),
            moe=importlib.import_module(f"{PKG}.kernels.m3_moe.op"),
        )
    except NoTarget as no_target:
        pytest.skip(str(no_target))


def _selector(served, monkeypatch):
    model = served.modeling
    config = SimpleNamespace(**model._fsel._M3)
    config.num_index_heads = config.num_kv_heads
    monkeypatch.setattr(model, "step_contract_enabled", lambda: False)
    return model.DecodeTarget(SimpleNamespace(attn=[SimpleNamespace(m3_config=config)])), config


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [8, 9])
@pytest.mark.parametrize("mode", ["native", "killed", "declined"])
@pytest.mark.parametrize("early", [False, True])
def test_imported_decode_selector_keeps_native_and_stock_paths(
    served, monkeypatch, rows, mode, early
):
    model = served.modeling
    target, config = _selector(served, monkeypatch)
    scores = torch.zeros(config.num_kv_heads, model.LONG_SELECT_MIN_BLOCKS, rows)
    valid = torch.full((rows,), scores.shape[1], dtype=torch.int32)
    native = torch.full((rows, config.num_kv_heads, model._fsel.TOPK), 17, dtype=torch.int32)
    stock = torch.full_like(native, -1)
    select = Mock(return_value=None if mode == "declined" else native)
    fallback = Mock(return_value=stock)
    taken = Mock()
    monkeypatch.setattr(model._fsel, "fsel_select_blocks", select)
    monkeypatch.setattr(model, "long_select_blocks", fallback)
    monkeypatch.setattr(model, "LONG_SELECT_DSL", SimpleNamespace(keys=frozenset(), path="dsl"))
    monkeypatch.setattr(
        model,
        "FSEL_SELECT",
        SimpleNamespace(keys={"fsel"} if mode != "killed" else set(), path="tp8_fsel"),
    )
    monkeypatch.setattr(model, "SELECT_EARLY_MAX_ROWS", model.FSEL_MAX_ROWS if early else 0)
    monkeypatch.setattr(model, "FSEL_TRIGGER_PHASE", 0 if early else 1)
    monkeypatch.setattr(model, "taken", taken)
    result = target._select(0, scores, {"msa_n_valid_blocks": valid}, rows)
    eligible = mode != "killed" and rows <= model.FSEL_MAX_ROWS
    uses_native = eligible and mode == "native"
    assert result is (native if uses_native else stock)
    assert select.call_count == int(eligible)
    assert fallback.call_count == int(not uses_native)
    if eligible:
        assert select.call_args.kwargs["trigger_phase"] == (0 if early else 1)
    if not uses_native:
        assert fallback.call_args.kwargs["early_trigger"] == (early and rows <= model.FSEL_MAX_ROWS)
    assert any(c.args[0] == "SELECT_EARLY" for c in taken.call_args_list) == (
        early and rows <= model.FSEL_MAX_ROWS
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("pushed", [False, True])
@pytest.mark.parametrize("return_parts", [False, True])
def test_imported_decode_moe_forwards_served_engine_and_push(
    served, monkeypatch, rows, pushed, return_parts
):
    model = served.modeling
    monkeypatch.setattr(model, "step_contract_enabled", lambda: False)
    monkeypatch.setattr(model, "MOE_ENGINE_MAX_TOKENS", 8)
    engine = object()
    core = SimpleNamespace(engine=Mock(return_value=engine))
    target = model.DecodeTarget(core)
    hidden = torch.zeros(rows, served.moe.HIDDEN, dtype=torch.bfloat16)
    output = torch.ones_like(hidden)
    call = Mock(return_value=output)
    monkeypatch.setattr(model, "m3_moe_engine", call)
    monkeypatch.setattr(model, "taken", Mock())
    descriptor = object() if pushed else None
    result = target._moe(
        0, hidden, return_parts=return_parts, push=(descriptor, None) if pushed else None
    )
    core.engine.assert_called_once_with(0, rows, shared=True)
    call.assert_called_once_with(hidden, engine, None, descriptor)
    assert result[0] is output and result[1:] == (None, pushed, False)


def _replay(run, change_inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for seed in (191, 192, 193):
        change_inputs(seed)
        expected = run()
        for output in outputs:
            output.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(outputs, expected):
            assert torch.equal(got, want)


@pytest.fixture(scope="module")
def moe_operands():
    path = Path(__file__).parents[1] / "ops/moe/_m31_flash_moe_operands.py"
    spec = importlib.util.spec_from_file_location("_tp8_served_moe_operands", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@NEEDS_SM107
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("case", ["random", "concentrated"])
@pytest.mark.parametrize("disabled", [False, True])
def test_imported_decode_moe_engine_native_and_graph(
    served, monkeypatch, moe_operands, rows, case, disabled
):
    model, moe = served.modeling, served.moe
    monkeypatch.setattr(model, "step_contract_enabled", lambda: False)
    monkeypatch.setattr(model, "MOE_ENGINE_MAX_TOKENS", 8)
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    def state(killed):
        monkeypatch.setattr(moe, "m31_disabled", killed.__contains__)
        return moe.M3MoeState(
            device, rows, num_ctas=sms - moe.shared_ctas_for(rows, True), shared=True
        )

    baseline = state({"MOE_SFA_BULK", "MOE_K128"})
    baseline_layer = moe_operands.layer_for(baseline, moe.INTER, case)
    current = baseline if disabled else state(set())
    layer = baseline_layer if disabled else moe_operands.layer_for(current, moe.INTER, case)
    assert current.config.get("sfa_bulk", 0) == int(not disabled)
    assert current.config.get("k128", 0) == int(not disabled)
    assert current.mod.SFA_BULK == (not disabled) and current.mod.K128 == (not disabled)
    core = SimpleNamespace(engine=lambda i, n, shared: layer)
    target = model.DecodeTarget(core)
    hidden = moe_operands.hidden_states(rows, seed=190)
    reference = baseline.run(baseline_layer, hidden).clone()
    # Compile/warm the real provider outside capture and keep it bound through DecodeTarget._moe.
    provider = Mock(wraps=model.m3_moe_engine)
    monkeypatch.setattr(model, "m3_moe_engine", provider)

    def run():
        result, routed, engine_pushed, shared_pushed = target._moe(0, hidden, return_parts=False)
        assert routed is None and not engine_pushed and not shared_pushed
        return (result.clone(),)

    assert torch.equal(run()[0].view(torch.int16), reference.view(torch.int16))
    provider.assert_called_once()
    _replay(run, lambda seed: hidden.copy_(moe_operands.hidden_states(rows, seed=seed)))
