# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 dep target's construction, adoption, decode loop and step-loop hooks.

Host-only. Adoption runs on the stand-in built-in modules and kernel factories of the tp8
target's layer test (``test_modeling_v2_glm_layers.py``, loaded by path), with the fast paths
of a dep cell: the attention modules and the speculative worker are re-classed, and the
decoder layers, the MoE modules and the draft layer stay the built-in classes. The shell's
construction runs with the built-in constructor stubbed and a recorder in place of
``adopt_model``, as the tp8 target's shell test does.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__dep import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    attention,
    dispatch,
    layers,
    runtime,
    spec_worker,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import DSAHooks
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3Model,
    Deepseekv3MoE,
    DeepseekV3MTP,
    DeepseekV32Attention,
    GlmMoeDsaForCausalLM,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

PREFILL, DECODE = dispatch.Phase.PREFILL, dispatch.Phase.DECODE
REQUIRE, AUTO = _router_index.ModelingV2Mode.REQUIRE, _router_index.ModelingV2Mode.AUTO


def _load_tp8_layer_test():
    path = (
        Path(__file__).resolve().parents[1]
        / "glm_5_2_nvfp4__sm_103__tp8"
        / "test_modeling_v2_glm_layers.py"
    )
    spec = importlib.util.spec_from_file_location("_glm_tp8_layer_test_for_dep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = _load_tp8_layer_test()

DEP16, DEP8, DEP8_LM_HEAD_TP = (16, False, 3), (8, False, 3), (8, True, 5)
CELLS = [DEP16, DEP8, DEP8_LM_HEAD_TP]
IDS = ["dep16", "dep8", "dep8-lm-head-tp"]


def _served(cell):
    return modeling.CELLS[cell].paths.served


@pytest.fixture
def fast_paths(monkeypatch):
    def install(enabled):
        paths = T._FastPaths(frozenset(enabled))
        monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
        monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
        monkeypatch.setattr(dispatch, "log_capability", paths.log_capability)
        return paths

    return install


def _state(model):
    return dict(modeling.adopted_state(types.SimpleNamespace(model=model.model)))


# --- adoption -------------------------------------------------------------------------------


@pytest.mark.parametrize("cell", [DEP16, DEP8], ids=IDS[:2])
def test_with_an_fp8_kv_cache_the_attention_and_the_worker_are_adopted(fast_paths, cell):
    paths = fast_paths(_served(cell))
    model = T._built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)

    # No fast-path object is built: the attention tail, the hops, the router GEMM and the draft
    # front are not served.
    assert paths.events == []
    dense, moe, last, draft = model.model.layers
    assert [type(m) for m in (dense, moe, last)] == [DeepseekV3DecoderLayer] * 3
    assert type(draft) is DeepseekV3MTP
    assert [type(m.mlp) for m in (moe, last, draft)] == [Deepseekv3MoE] * 3
    for layer in (dense, moe, last, draft):
        mla = layer.self_attn
        assert type(mla) is attention.GlmMLA
        assert type(mla.sparse_attn_hooks) is attention.GlmDsaHooks
        assert type(mla.mqa) is attention.GlmDsaBackend
        assert mla._attn_tail is None and mla._tail_prefetch is False
        # The FMHA libraries stay the registry's.
        assert "prepended_fmha_lib_classes" not in vars(mla.mqa)
    assert [type(m.self_attn.mqa.indexer) for m in (dense, last)] == [attention.GlmIndexer] * 2
    assert type(model.spec_worker) is spec_worker.GlmMtpWorker


def test_with_an_nvfp4_kv_cache_only_the_worker_is_adopted(fast_paths):
    paths = fast_paths(_served(DEP8_LM_HEAD_TP))
    model = T._built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)

    assert paths.events == []
    for layer in model.model.layers:
        assert type(layer.self_attn) is DeepseekV32Attention
        assert type(layer.self_attn.sparse_attn_hooks) is DSAHooks
        assert type(layer.self_attn.mqa) is DSATrtllmAttention
    assert type(model.spec_worker) is spec_worker.GlmMtpWorker


@pytest.mark.parametrize(
    "disabled, attention_adopted, worker_adopted",
    [
        ({"rope_kv_append", "indexer_layer_front"}, False, True),
        ({"indexer_layer_front"}, True, True),
        ({"draft_loop", "decode_prologue"}, True, False),
        (set(), True, True),
    ],
    ids=["no rope", "no indexer front", "no draft loop", "all"],
)
def test_each_adoption_follows_its_fast_path(
    fast_paths, disabled, attention_adopted, worker_adopted
):
    fast_paths(_served(DEP8) - disabled)
    model = T._built_in_model([])
    modeling.adopt_model(model, plan=None)
    expected = attention.GlmMLA if attention_adopted else DeepseekV32Attention
    assert [type(layer.self_attn) for layer in model.model.layers] == [expected] * 4
    worker = spec_worker.GlmMtpWorker if worker_adopted else MTPEagleWorker
    assert type(model.spec_worker) is worker


def test_without_a_plan_nothing_is_adopted(fast_paths):
    fast_paths(modeling.NO_PATHS.served)
    model = T._built_in_model([])
    modeling.adopt_model(model, plan=None)
    assert [type(layer.self_attn) for layer in model.model.layers] == [DeepseekV32Attention] * 4
    assert type(model.spec_worker) is MTPEagleWorker


def test_without_a_worker_the_attention_is_adopted_alone(fast_paths):
    fast_paths(_served(DEP8))
    model = T._built_in_model([])
    model.spec_worker = None
    modeling.adopt_model(model, plan=None)
    assert type(model.model.layers[0].self_attn) is attention.GlmMLA


@pytest.mark.parametrize("cell", CELLS, ids=IDS)
def test_adoption_attaches_no_layer_state(fast_paths, cell):
    """Every rank checks the same empty per-layer state: no fast-path object is attached to the
    layers, their attention or their MoE."""
    fast_paths(_served(cell))
    model = T._built_in_model([])
    modeling.adopt_model(model, plan=None)
    state = _state(model)
    assert list(state) == [f"model.layers.{i}" for i in range(4)]
    assert all(attached == () for attached in state.values())


@pytest.mark.parametrize("cell", CELLS, ids=IDS)
def test_adoption_adds_no_parameter_buffer_or_module(fast_paths, cell):
    fast_paths(_served(cell))
    model = T._built_in_model([])
    before = [T._keys(layer) for layer in model.model.layers]
    modeling.adopt_model(model, plan=None)
    assert [T._keys(layer) for layer in model.model.layers] == before


def test_adoption_refuses_an_attention_module_that_is_not_the_builtin_class(fast_paths):
    fast_paths(_served(DEP8))
    model = T._built_in_model([])
    attention.GlmMLA.adopt(model.model.layers[1].self_attn)
    with pytest.raises(TypeError, match="GlmMLA adopts the built-in DeepseekV32Attention"):
        modeling.adopt_model(model, plan=None)


def test_the_target_uses_no_layer_class_of_the_family():
    """The decoder layers, the MoE modules and the draft layer stay the built-in classes: the
    target names none of the family's layer classes or their adoption."""
    source = inspect.getsource(modeling)
    for name in ("GlmDecoderLayer", "GlmMtpLayer", "GlmMoE", "layers.adopt_model"):
        assert name not in source, name
    assert modeling.adopted_state is layers.adopted_state


# --- the core and its two targets -----------------------------------------------------------


def _builtin_text_model() -> DeepseekV3Model:
    model = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(model)
    return model


def test_adopt_reclasses_the_builtin_text_model_in_place():
    model = _builtin_text_model()
    core = modeling.GlmMoeDsaDepModelingV2.adopt(model)
    assert core is model and type(core) is modeling.GlmMoeDsaDepModelingV2
    assert isinstance(core._prefill, modeling.PrefillTarget) and core._prefill.core is core
    assert isinstance(core._decode, modeling.DecodeTarget) and core._decode.core is core
    assert list(core.named_modules()) == [("", core)]
    with pytest.raises(TypeError, match="GlmMoeDsaDepModelingV2"):
        modeling.GlmMoeDsaDepModelingV2.adopt(core)


@pytest.mark.parametrize("num_contexts, target", [(0, "_decode"), (1, "_prefill"), (4, "_prefill")])
def test_each_step_goes_to_the_target_of_its_phase(num_contexts, target):
    core = modeling.GlmMoeDsaDepModelingV2.adopt(_builtin_text_model())
    md = types.SimpleNamespace(num_contexts=num_contexts)
    assert core._select_target(md) is getattr(core, target)


@pytest.fixture
def decode_core():
    calls = []
    core = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(core)
    core.embed_tokens = T._Embedding(calls)
    core.layers = nn.ModuleList([T._RecordingLayer(i, calls) for i in range(T.NUM_LAYERS + 1)])
    core.num_hidden_layers = T.NUM_LAYERS
    core.mapping_with_cp = None
    core = modeling.GlmMoeDsaDepModelingV2.adopt(core)
    for layer in core.layers:
        layer.register_forward_pre_hook(
            lambda m, a, k: calls.append(("hook", m.index)), with_kwargs=True
        )
    return core, calls


@pytest.mark.parametrize("prologue_rows", [None, "rows"], ids=["no packet", "packet"])
def test_the_decode_loop_runs_the_decoder_layers_as_modules(decode_core, prologue_rows):
    core, calls = decode_core
    rows = torch.full((2, T.HIDDEN), 7.0) if prologue_rows else None
    seen = []

    def take_embedding(attn_metadata, input_ids, embed_tokens):
        seen.append((attn_metadata, input_ids, embed_tokens))
        return rows

    core.decode_prologue = types.SimpleNamespace(take_embedding=take_embedding)
    md = T._NoContexts()
    with dispatch.step(DECODE):
        out = core._decode.forward(md, input_ids="ids", position_ids="pos", spec_metadata="spec")

    assert seen == [(md, "ids", core.embed_tokens)]
    embedded = [] if prologue_rows else [("embed", "ids")]
    start = rows if prologue_rows else torch.zeros(2, T.HIDDEN)
    assert torch.equal(out, start + T.NUM_LAYERS)
    layer_calls = []
    for i in range(T.NUM_LAYERS):
        kwargs = calls[len(embedded) + 2 * i + 1][1]
        assert kwargs["residual"] == (None if i == 0 else f"residual {i - 1}")
        assert kwargs["attn_metadata"] is md and kwargs["spec_metadata"] == "spec"
        layer_calls += [("hook", i), (i, kwargs, DECODE)]
    # The draft layer is not run, and nothing follows the last decoder layer.
    assert calls == embedded + layer_calls


def test_without_the_decode_prologue_the_decode_loop_is_the_builtin_loop(decode_core):
    core, calls = decode_core
    with dispatch.step(DECODE):
        out = core._decode.forward(T._NoContexts(), input_ids="ids")
    assert torch.equal(out, torch.zeros(2, T.HIDDEN) + T.NUM_LAYERS)
    assert [c[0] for c in calls] == ["embed"] + ["hook", 0, "hook", 1, "hook", 2]


def test_the_decode_loop_reads_no_fast_path_and_never_the_step_kind():
    source = inspect.getsource(modeling.DecodeTarget.forward)
    assert "dispatch." not in source
    assert "num_contexts" not in source and "num_ctx" not in source


# --- construction ---------------------------------------------------------------------------


def _model_config(cell, **facts):
    ranks, lm_head_tp, draft = cell
    defaults = {
        DEP16: (QuantAlgo.FP8, "TRTLLM"),
        DEP8: (QuantAlgo.FP8, "CUTEDSL"),
        DEP8_LM_HEAD_TP: (QuantAlgo.NVFP4, "CUTEDSL"),
    }[cell]
    return types.SimpleNamespace(
        mapping=types.SimpleNamespace(
            world_size=ranks,
            enable_attention_dp=True,
            enable_lm_head_tp_in_adp=lm_head_tp,
        ),
        spec_config=types.SimpleNamespace(max_draft_len=draft),
        quant_config=types.SimpleNamespace(kv_cache_quant_algo=facts.get("kv_cache", defaults[0])),
        moe_backend=facts.get("moe_backend", defaults[1]),
    )


RANK_UNIFORM_ENV = dispatch.SHARED_ENV + (_router_index.MODELING_V2_VALIDATE_ENV,)


@pytest.fixture
def shell_construction(monkeypatch):
    """The shell class with the built-in constructor stubbed, and the events of a construction."""
    events = []

    def adopt_model(model, plan):
        events.append(("adopt", type(model.model).__name__, plan))

    def builtin_init(self, model_config):
        events.append("built-in model")
        events.append(("installed", dispatch.installed()))
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = _builtin_text_model()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])

    def allgather(value, world_size):
        events.append(("allgather", value, world_size))
        return [value] * world_size

    shell_cls = modeling.ModelingV2Glm52Nvfp4Sm103Dep
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__init__", builtin_init)
    monkeypatch.setattr(modeling, "adopt_model", adopt_model)
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__post_init__", lambda self: None)
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__pp_init__", lambda self: None)
    monkeypatch.setattr(dispatch, "_require", [False])
    monkeypatch.setattr(dispatch, "_world_allgather", allgather)
    monkeypatch.setattr(dispatch, "_INSTALLED", [None])
    monkeypatch.setattr(dispatch, "_TARGET", [dispatch.DEFAULT_TARGET])
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(dispatch, "_log_info", lambda message: events.append(message))
    monkeypatch.setattr(modeling.logger, "warning", lambda message: events.append(message))
    for name in list(os.environ):
        if name.startswith(dispatch.KNOB_PREFIX) or name in RANK_UNIFORM_ENV:
            monkeypatch.delenv(name)

    def construct(mode, model_config):
        monkeypatch.setitem(_router_index._RESOLVED_MODES, shell_cls, mode)
        return shell_cls(model_config)

    return construct, events


@pytest.mark.parametrize("cell", CELLS, ids=IDS)
def test_the_shell_checks_the_cell_the_ranks_and_the_plan_then_adopts(shell_construction, cell):
    construct, events = shell_construction
    shell = construct(REQUIRE, _model_config(cell))

    allgather, record, built_in, installed, adopt, adopted = events
    assert allgather == ("allgather", ((dispatch.MODE_KEY, "require"),), cell[0])
    served = modeling.CELLS[cell].paths.served
    assert shell.plan.describe().startswith(
        f"modeling_v2=require, {len(served)}/{len(served)} fast paths"
    )
    assert record.startswith("[glm_moe_dsa] dispatch: record paths=")
    assert built_in == "built-in model"
    assert installed == ("installed", served)
    assert adopt == ("adopt", "GlmMoeDsaDepModelingV2", shell.plan)
    assert adopted == ("allgather", modeling.adopted_state(shell), cell[0])
    assert shell.plan.target is modeling.CELLS[cell].paths
    assert type(shell.model) is modeling.GlmMoeDsaDepModelingV2
    assert "_runtime_hooks" not in vars(shell)


@pytest.mark.parametrize(
    "cell, facts",
    [
        (DEP16, {"moe_backend": "CUTEDSL"}),
        (DEP8, {"kv_cache": QuantAlgo.NVFP4}),
        (DEP8_LM_HEAD_TP, {"kv_cache": QuantAlgo.FP8}),
    ],
    ids=IDS,
)
def test_outside_the_cells_require_raises_before_any_collective(shell_construction, cell, facts):
    construct, events = shell_construction
    with pytest.raises(ValueError, match="is not a deployment this target serves"):
        construct(REQUIRE, _model_config(cell, **facts))
    assert events == []


def test_outside_the_cells_auto_builds_the_builtin_model_without_fast_paths(shell_construction):
    construct, events = shell_construction
    shell = construct(AUTO, _model_config(DEP8, moe_backend="TRTLLM"))

    warning, allgather, record = events[:3]
    assert "is not a deployment this target serves" in warning
    assert allgather == ("allgather", ((dispatch.MODE_KEY, "auto"),), 8)
    assert record.startswith("[glm_moe_dsa] dispatch: record paths=<none> ")
    assert shell.plan.describe().startswith("modeling_v2=auto, 0/0 fast paths")
    assert ("installed", frozenset()) in events
    assert shell.plan.target is modeling.NO_PATHS
    assert not dispatch.require_mode()


def test_a_disable_knob_turns_its_path_and_dependents_off(monkeypatch, shell_construction):
    construct, _ = shell_construction
    monkeypatch.setenv(dispatch.DISABLE_ENV, "draft_loop")
    shell = construct(REQUIRE, _model_config(DEP8))
    off = {name for name in dispatch.FAST_PATH_NAMES if not dispatch.is_enabled(name)}
    assert off >= {"draft_loop", "decode_prologue"}
    assert dispatch.installed() == modeling.CELLS[DEP8].paths.served - {
        "draft_loop",
        "decode_prologue",
    }
    assert shell.plan.describe().startswith("modeling_v2=require, 8/10 fast paths")


# --- the step-loop hooks --------------------------------------------------------------------


def _model_of_cell(cell):
    paths = modeling.CELLS[cell].paths
    return types.SimpleNamespace(
        plan=dispatch.resolve(REQUIRE, {}, paths),
        model=types.SimpleNamespace(decode_prologue=None),
        model_config=types.SimpleNamespace(mapping=types.SimpleNamespace(enable_attention_dp=True)),
    )


@pytest.fixture
def quiet_dispatch(monkeypatch):
    monkeypatch.setattr(dispatch, "_require", [False])
    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)


@pytest.mark.usefixtures("quiet_dispatch")
@pytest.mark.parametrize("cell", CELLS, ids=IDS)
def test_the_hooks_declare_the_decode_prologue_and_no_step_loop_part(cell):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.decode_prologue import (  # noqa: E501
        DecodePrologueHook,
    )

    model = _model_of_cell(cell)
    hooks = runtime.make_runtime_hooks(model)
    assert isinstance(hooks, RuntimeHooks)
    assert not hooks.loop_on_execution_stream
    assert hooks.step_input_stager is None and hooks.sampler_store_writer is None
    assert isinstance(hooks.decode_prologue, DecodePrologueHook)
    assert model.model.decode_prologue is hooks.decode_prologue
    assert isinstance(hooks.executor_events, runtime.GlmExecutorEvents)


@pytest.mark.usefixtures("quiet_dispatch")
@pytest.mark.parametrize("cell", CELLS, ids=IDS)
def test_the_executor_events_install_only_the_host_policy(monkeypatch, cell):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (  # noqa: E501
        host_page_retention,
    )

    calls = []
    monkeypatch.setattr(host_page_retention, "install_host_policy", calls.append)
    events = runtime.make_runtime_hooks(_model_of_cell(cell)).executor_events
    events.after_warmup()
    events.before_overlap_loop()
    events.at_shutdown()
    assert calls == ["after_warmup"]


def test_the_shell_builds_its_hooks_on_the_first_read(monkeypatch, shell_construction):
    construct, _ = shell_construction
    shell = construct(REQUIRE, _model_config(DEP8))
    built = []

    def make_runtime_hooks(model):
        built.append(model)
        return RuntimeHooks()

    monkeypatch.setattr(runtime, "make_runtime_hooks", make_runtime_hooks)
    hooks = runtime_hooks_of(shell)
    assert built == [shell] and runtime_hooks_of(shell) is hooks

    def broken(model):
        raise AttributeError("plan")

    shell2 = construct(REQUIRE, _model_config(DEP8))
    monkeypatch.setattr(runtime, "make_runtime_hooks", broken)
    with pytest.raises(RuntimeError, match="step-loop hooks"):
        runtime_hooks_of(shell2)
