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
from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"
PKG = f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}"
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


def _worker(monkeypatch, served, rank, killed, *, real_layout=False):
    dspark = served.dspark
    worker = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    torch.nn.Module.__init__(worker)
    worker.mapping = SimpleNamespace(tp_size=8, tp_rank=rank, enable_attention_dp=False)
    worker.spec_config = SimpleNamespace(max_draft_len=7)
    worker._fused_drafter_inputs = Mock()
    # Keep the complete prebuild method while isolating unrelated setup and IPC. The worker hooks are doubles (they are
    # engine-side: a DSpark worker without them makes prebuild set up nothing).
    for name, result in {
        "enable_lean_glue": True,
        "enable_window_capped_decode": False,
        "enable_full_slot_arena": None,
        "enable_fc_tp_split": False,
        "enable_fused_ar_norm": True,
        "enable_fused_ar_norm_op": False,
        "enable_fused_ctx_store": True,
        "enable_fused_drafter_inputs": True,
        "enable_fused_ctx_advance": True,
        "enable_fused_block_refresh": True,
        "enable_int32_accept_counts": True,
        "enable_fused_forward_indices": True,
        "enable_fused_qk_norm_rope": True,
        "enable_fused_qk_norm_rope_append": False,
        "enable_attention_pdl": True,
        "enable_draft_logits_gather_op": False,
        "enable_fused_kv_rewind": False,
        "prebuild_s4": None,
    }.items():
        monkeypatch.setattr(worker, name, Mock(return_value=result), raising=False)
    for name, value in {
        "FC_SHARD": False,
        "N3A_DISABLED": True,
        "N3A_KV_DISABLED": True,
        "FOLD_F7": False,
    }.items():
        monkeypatch.setattr(dspark, name, value)
    monkeypatch.setattr(dspark, "m31_disabled", killed.__contains__)
    monkeypatch.setattr(dspark, "fold_k_norm_weight", lambda value: value)
    monkeypatch.setattr(dspark, "enable_parallel_refresh", lambda value: value)
    log = Mock()
    monkeypatch.setattr(dspark, "logger", log)
    routes = Mock()
    routes.attach_mock(
        Mock(wraps=dspark.route_his_layout) if real_layout else Mock(return_value=(15, 5)),
        "his",
    )
    for name in ("mx", "gu"):
        routes.attach_mock(Mock(return_value=0), name)
    for name, callback in {
        "route_his_layout": routes.his,
        "route_mx_linears": routes.mx,
        "route_gu_swiglu": routes.gu,
    }.items():
        monkeypatch.setattr(dspark, name, callback)
    if not real_layout:
        # The device-side setup the prebuild ends with (the context-length resume compile, the causal draft).
        monkeypatch.setattr(dspark, "install_draft_swiglu", Mock(return_value=5))
        monkeypatch.setattr(dspark, "install_causal_draft", Mock(return_value=0))
        monkeypatch.setattr(dspark.dspark_ctxlen, "prepare", Mock(return_value=False))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return worker, routes, log


def _registration(log):
    return [c.args[0] for c in log.info.call_args_list if "[M31_DRAFTER_HISLAYOUT]" in c.args[0]]


@pytest.mark.cpu_only
@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("disabled", [False, True])
def test_imported_dspark_prebuild_preserves_provider_order_and_kill(
    served, monkeypatch, rank, disabled
):
    killed = {"DRAFTER_HISLAYOUT"} if disabled else set()
    worker, routes, log = _worker(monkeypatch, served, rank, killed)
    draft = SimpleNamespace()
    worker.prebuild(None, draft)
    assert isinstance(worker, served.dspark.M3DSparkWorker)
    assert routes.his.call_count == int(not disabled)
    if not disabled:
        routes.his.assert_called_once_with(
            draft, [b * 7 for b in range(1, served.dspark.MAX_FAST_BATCH + 1)]
        )
    # The his layout's fused MLPs take the gate_up SwiGLU; without the layout the epilogue route does.
    assert [c[0] for c in routes.mock_calls] == (["mx", "gu"] if disabled else ["his", "mx"])
    count = "0 projections + 0 fused MLPs" if disabled else "15 projections + 5 fused MLPs"
    assert _registration(log) == [f"[M31_DRAFTER_HISLAYOUT] TP8 registered {count} on rank {rank}"]


def _selector(served, monkeypatch):
    model = served.modeling
    config = SimpleNamespace(**model._fsel._M3)
    config.num_index_heads = config.num_kv_heads
    monkeypatch.setattr(model, "step_contract_enabled", lambda: False)
    return model.DecodeTarget(SimpleNamespace(attn=[SimpleNamespace(m3_config=config)])), config


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 8, 9])
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


@NEEDS_SM107
@pytest.mark.parametrize("blocks", [4097, 8192, 8193, 9216])
@pytest.mark.parametrize("early", [False, True])
def test_imported_decode_fsel_native_and_graph(served, monkeypatch, blocks, early):
    model = served.modeling
    assert model._fsel.fsel_op() is not None, "required native FSEL op missing from build"
    target, config = _selector(served, monkeypatch)
    rows = model.FSEL_MAX_ROWS
    scores = torch.empty(config.num_kv_heads, blocks, rows, device="cuda")
    valid = torch.full((rows,), blocks, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(model, "FSEL_SELECT", SimpleNamespace(keys={"fsel"}, path="tp8_fsel"))
    monkeypatch.setattr(model, "SELECT_EARLY_MAX_ROWS", rows if early else 0)
    monkeypatch.setattr(model, "FSEL_TRIGGER_PHASE", 0 if early else 1)
    model._fsel.prepare(scores.device)
    adapter = Mock(wraps=model._fsel.fsel_select_blocks)
    monkeypatch.setattr(model._fsel, "fsel_select_blocks", adapter)

    def change(seed):
        scores.normal_(generator=torch.Generator(device="cuda").manual_seed(seed))

    def run():
        return (target._select(0, scores, {"msa_n_valid_blocks": valid}, rows).clone(),)

    change(190)
    actual = run()[0]
    reference = model._fsel.m3_fsel_topk.reference(
        scores, valid, torch.empty_like(actual), config.init_blocks, config.local_blocks
    )
    assert adapter.call_count == 1 and torch.equal(actual, reference)
    _replay(run, change)


@pytest.fixture(scope="module")
def moe_operands():
    path = Path(__file__).parent / "moe/_m3_moe_operands.py"
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
    baseline_layer = moe_operands.layer_for(baseline, case)
    current = baseline if disabled else state(set())
    layer = baseline_layer if disabled else moe_operands.layer_for(current, case)
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


class IdentityAR(torch.nn.Module):
    def uses_nccl_symmetric_memory_window(self):
        return True

    def forward(self, x, all_reduce_params=None):
        return x


def _drafter(served):
    shapes = served.drafter_mxfp8.HIS_SHAPES
    hidden = shapes["gate_up_proj"][1]
    intermediate = shapes["down_proj"][1]
    layers = []
    for index in range(5):
        mlp = GatedMLP(
            hidden_size=hidden,
            intermediate_size=intermediate,
            bias=False,
            dtype=torch.bfloat16,
            swiglu_limit=7.0,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
        ).cuda()
        attn = SimpleNamespace()
        for name in ("qkv_proj", "o_proj"):
            n, k = shapes[name]
            setattr(attn, name, Linear(k, n, bias=False, dtype=torch.bfloat16).cuda())
        linears = [attn.qkv_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj]
        for position, linear in enumerate(linears):
            generator = torch.Generator(device="cuda").manual_seed(
                900 + index * len(linears) + position
            )
            codes = (torch.randn(linear.weight.shape, device="cuda", generator=generator) * 16).to(
                torch.float8_e4m3fn
            )
            with torch.no_grad():
                linear.weight.copy_(codes.to(torch.bfloat16))
        for linear in (attn.o_proj, mlp.down_proj):
            linear.tp_mode, linear.reduce_output, linear.tp_size = TensorParallelMode.ROW, True, 8
            linear.all_reduce = IdentityAR()
        layers.append(SimpleNamespace(self_attn=attn, mlp=mlp))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


@NEEDS_SM107
@pytest.mark.parametrize("disabled", [False, True])
def test_imported_dspark_prebuild_real_layout_and_drafter_graph(served, monkeypatch, disabled):
    with torch.inference_mode():
        killed = {"DRAFTER_HISLAYOUT"} if disabled else set()
        worker, routes, log = _worker(monkeypatch, served, 0, killed, real_layout=True)
        draft = _drafter(served)
        worker.prebuild(None, draft)
        expected = "0 projections + 0 fused MLPs" if disabled else "15 projections + 5 fused MLPs"
        assert _registration(log) == [
            f"[M31_DRAFTER_HISLAYOUT] TP8 registered {expected} on rank 0"
        ]
        assert routes.his.call_count == int(not disabled)
        inputs = []
        for layer in draft.model.layers:
            inputs.append(
                [
                    torch.empty(7, linear.weight.shape[1], dtype=torch.bfloat16, device="cuda")
                    for linear in (
                        layer.self_attn.qkv_proj,
                        layer.self_attn.o_proj,
                        layer.mlp.gate_up_proj,
                    )
                ]
            )
            assert hasattr(layer.mlp, "_m31_hislayout_mlp") == (not disabled)
            assert hasattr(layer.self_attn.o_proj, "_m31_hislayout_row_forward") == (not disabled)
            # Unsupported stock ROW calls need no real NCCL window in this one-rank harness.
            monkeypatch.setattr(
                layer.self_attn.o_proj.all_reduce,
                "uses_nccl_symmetric_memory_window",
                lambda: False,
            )
            monkeypatch.setattr(
                layer.mlp.down_proj.all_reduce, "uses_nccl_symmetric_memory_window", lambda: False
            )
        params = SimpleNamespace(enable_allreduce=False)

        def change(seed):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            for group in inputs:
                for value in group:
                    value.normal_(generator=generator)

        def run():
            result = []
            for layer, (q, o, x) in zip(draft.model.layers, inputs):
                result.extend(
                    [
                        layer.self_attn.qkv_proj(q).clone(),
                        layer.self_attn.o_proj(o, all_reduce_params=params).clone(),
                        layer.mlp(x, final_all_reduce_params=params).clone(),
                    ]
                )
            return tuple(result)

        change(190)
        _replay(run, change)
