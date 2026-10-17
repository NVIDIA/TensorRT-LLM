# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 tp8 core and its two targets, and what the shell keeps from the built-in model.

Host-only: the core is adopted on a built-in text model created without its constructor
(which builds CUDA streams and the decoder layers), and the built-in forward is replaced by
a recorder, so each test sees exactly which call a step makes. The shell's construction runs
with the built-in constructor stubbed, stand-in decoder layers, a recorder in place of
``adopt_model`` (``test_modeling_v2_glm_layers.py`` covers it) and a stand-in for the
sparse-MLA decode FMHA library.
"""

from __future__ import annotations

import inspect
import os
import sys
import types

import pytest
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
    runtime,
)
from tensorrt_llm._torch.attention.backends.fmha import manager as fmha_manager
from tensorrt_llm._torch.attention.backends.fmha import registry as fmha_registry
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Model, GlmMoeDsaForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of

pytestmark = pytest.mark.cpu_only


def _builtin_text_model() -> DeepseekV3Model:
    """A built-in text model as far as class and module state go, without its layers."""
    model = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(model)
    return model


class _Hop:
    """Stands in for a post-MoE hop object a decoder layer carries."""

    def __init__(self, engaged: bool = True) -> None:
        self.engaged = engaged
        self.impl = "cluster"


class _Layer(nn.Module):
    """Stands in for a decoder layer with the fast-path attributes the constructors set."""

    def __init__(self, post_moe_hop=None, router_gemm=None) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn._attn_tail = None
        self.mlp = nn.Module()
        self.mlp.router_gemm = router_gemm
        self.post_attn_hop = None
        self.post_moe_hop = post_moe_hop


class _Metadata:
    def __init__(self, num_contexts: int) -> None:
        self.num_contexts = num_contexts


def test_adopt_reclasses_the_builtin_text_model_in_place():
    model = _builtin_text_model()
    core = modeling.GlmMoeDsaTp8ModelingV2.adopt(model)
    assert core is model
    assert type(core) is modeling.GlmMoeDsaTp8ModelingV2
    assert isinstance(core, DeepseekV3Model)
    assert isinstance(core._prefill, modeling.PrefillTarget) and core._prefill.core is core
    assert isinstance(core._decode, modeling.DecodeTarget) and core._decode.core is core
    # Plain attributes only: no module or parameter joins the built-in tree.
    assert list(core.named_modules()) == [("", core)]
    assert list(core.named_parameters()) == []


def test_adopt_refuses_anything_but_the_builtin_text_model():
    core = modeling.GlmMoeDsaTp8ModelingV2.adopt(_builtin_text_model())
    with pytest.raises(TypeError, match="GlmMoeDsaTp8ModelingV2"):
        modeling.GlmMoeDsaTp8ModelingV2.adopt(core)
    with pytest.raises(TypeError, match="Module"):
        modeling.GlmMoeDsaTp8ModelingV2.adopt(nn.Module())


def _recorder(calls):
    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        spec_metadata=None,
        **kwargs,
    ):
        calls.append((self, attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata))
        calls.append(kwargs)
        return "hidden states"

    return forward


def _run_step_both_ways(core, md, calls, receiver):
    # Called the way the speculative causal LM calls its text model: keywords throughout.
    out = core(
        input_ids="ids",
        attn_metadata=md,
        position_ids="positions",
        inputs_embeds=None,
        spec_metadata="spec",
        lora_params="lora",
    )
    assert out == "hidden states"
    assert calls == [(receiver, md, "ids", "positions", None, "spec"), {"lora_params": "lora"}]

    # And positionally, in the built-in forward's parameter order.
    calls.clear()
    assert core(md, "ids", "positions", None, "spec") == "hidden states"
    assert calls == [(receiver, md, "ids", "positions", None, "spec"), {}]


@pytest.mark.parametrize("num_contexts", [1, 4], ids=["prefill", "mixed"])
def test_each_step_runs_the_builtin_forward_through_its_target(monkeypatch, num_contexts):
    calls = []
    monkeypatch.setattr(DeepseekV3Model, "forward", _recorder(calls))
    core = modeling.GlmMoeDsaTp8ModelingV2.adopt(_builtin_text_model())
    md = _Metadata(num_contexts)

    assert type(core._select_target(md)) is modeling.PrefillTarget
    _run_step_both_ways(core, md, calls, core)


def test_a_decode_step_runs_the_decode_targets_loop(monkeypatch):
    calls, builtin = [], []
    monkeypatch.setattr(DeepseekV3Model, "forward", _recorder(builtin))
    monkeypatch.setattr(modeling.DecodeTarget, "forward", _recorder(calls))
    core = modeling.GlmMoeDsaTp8ModelingV2.adopt(_builtin_text_model())
    md = _Metadata(0)

    assert type(core._select_target(md)) is modeling.DecodeTarget
    _run_step_both_ways(core, md, calls, core._decode)
    assert builtin == []


def test_the_forwards_take_the_builtin_parameters():
    builtin = inspect.signature(DeepseekV3Model.forward)
    for forward in (
        modeling.GlmMoeDsaTp8ModelingV2.forward,
        modeling.PrefillTarget.forward,
        modeling.DecodeTarget.forward,
    ):
        assert list(inspect.signature(forward).parameters) == list(builtin.parameters), forward


def test_the_shell_is_the_builtin_causal_lm():
    shell = modeling.ModelingV2Glm52Nvfp4Sm103Tp8
    assert issubclass(shell, GlmMoeDsaForCausalLM)
    # The model loader inspects load_weights' parameters to decide what it passes.
    assert (
        inspect.getfullargspec(shell.load_weights).args
        == inspect.getfullargspec(GlmMoeDsaForCausalLM.load_weights).args
    )


def _hooks_of_plan(disable: str = "") -> RuntimeHooks:
    """``make_runtime_hooks`` of a model whose plan disables ``disable``."""
    env = {dispatch.DISABLE_ENV: disable} if disable else {}
    model = types.SimpleNamespace(
        plan=dispatch.resolve(_router_index.ModelingV2Mode.AUTO, env),
        model=types.SimpleNamespace(decode_prologue=None),
        model_config=types.SimpleNamespace(
            mapping=types.SimpleNamespace(enable_attention_dp=False)
        ),
    )
    return runtime.make_runtime_hooks(model)


class _StoreWriter:
    """Stands in for ``SamplerStoreWriter``, whose constructor allocates pinned host memory."""


@pytest.fixture
def host_only_store_writer(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (  # noqa: E501
        deferred_input_prep,
    )

    monkeypatch.setattr(deferred_input_prep, "SamplerStoreWriter", _StoreWriter)


@pytest.mark.usefixtures("host_only_store_writer")
def test_the_target_declares_the_step_input_stager_of_its_plan(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        PrepStager,
    )

    validate_env = _router_index.MODELING_V2_VALIDATE_ENV
    monkeypatch.delenv(validate_env, raising=False)
    hooks = _hooks_of_plan()
    assert isinstance(hooks.step_input_stager, PrepStager)
    assert hooks.step_input_stager.debug is None

    monkeypatch.setenv(validate_env, "1")
    assert _hooks_of_plan().step_input_stager.debug.level == 1
    monkeypatch.delenv(validate_env)

    assert _hooks_of_plan("deferred_input_prep").step_input_stager is None


@pytest.mark.usefixtures("host_only_store_writer")
def test_the_target_declares_the_decode_prologue_of_its_plan():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.decode_prologue import (  # noqa: E501
        DecodePrologueHook,
    )

    hooks = _hooks_of_plan()
    assert isinstance(hooks.decode_prologue, DecodePrologueHook)
    assert _hooks_of_plan("decode_prologue").decode_prologue is None


@pytest.mark.usefixtures("host_only_store_writer")
def test_the_target_declares_the_sampler_store_writer_of_its_plan():
    assert isinstance(_hooks_of_plan().sampler_store_writer, _StoreWriter)
    hooks = _hooks_of_plan("sampler_store")
    assert hooks.sampler_store_writer is None
    assert hooks.step_input_stager is not None


@pytest.mark.usefixtures("host_only_store_writer")
@pytest.mark.parametrize(
    "disabled, off",
    [
        ("", ()),
        ("deferred_input_prep", ("step_input_stager",)),
        ("sampler_store", ("sampler_store_writer",)),
        ("execution_stream_loop", ("loop_on_execution_stream",)),
        (
            "deferred_input_prep,sampler_store,execution_stream_loop",
            ("step_input_stager", "sampler_store_writer", "loop_on_execution_stream"),
        ),
    ],
)
def test_each_part_of_deferred_input_prep_is_a_hook_of_its_own(disabled, off):
    """The three parts are separate fast paths: disabling one leaves the other two declared."""
    hooks = _hooks_of_plan(disabled)
    for field in ("step_input_stager", "sampler_store_writer", "loop_on_execution_stream"):
        assert bool(getattr(hooks, field)) is (field not in off), field
    assert isinstance(hooks.executor_events, runtime.GlmExecutorEvents)


@pytest.fixture
def event_calls(monkeypatch):
    """The kernel entry points the executor events call, recorded in order."""
    calls = []
    kernels = runtime.__name__.rsplit(".", 1)[0] + ".kernels"
    monkeypatch.setitem(
        sys.modules,
        kernels + ".post_moe_hop",
        types.SimpleNamespace(
            check_health=lambda where, **kwargs: calls.append(("check_health", where, kwargs))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        kernels + ".host_page_retention",
        types.SimpleNamespace(
            install_host_policy=lambda where: calls.append(("install_host_policy", where))
        ),
    )
    compat = types.SimpleNamespace(
        declare_overlap_loop=lambda adp: calls.append(("declare_overlap_loop", adp)),
        loop_domain=lambda: calls.append(("loop_domain",)),
    )
    monkeypatch.setitem(
        sys.modules,
        kernels + ".deferred_input_prep",
        types.SimpleNamespace(compat=compat),
    )
    return calls


def _events(paths, enable_attention_dp=False):
    plan = types.SimpleNamespace(enabled=lambda name: name in paths)
    mapping = types.SimpleNamespace(enable_attention_dp=enable_attention_dp)
    return runtime.GlmExecutorEvents(plan, mapping)


def test_the_executor_events_act_for_the_fast_paths_of_the_plan(event_calls):
    events = _events({"post_moe_hop", "host_page_retention", "sampler_store"})

    events.after_warmup()
    events.before_overlap_loop()
    events.at_shutdown()
    assert event_calls == [
        ("check_health", "after_warmup", {"collective": True}),
        ("install_host_policy", "after_warmup"),
        ("declare_overlap_loop", False),
        ("loop_domain",),
        ("check_health", "end_of_serving", {"raise_on_error": False}),
    ]


@pytest.mark.parametrize("path", ["sampler_store", "execution_stream_loop"])
def test_the_overlap_loop_declares_its_domain_for_either_part(event_calls, path):
    _events({path}, enable_attention_dp=True).before_overlap_loop()
    assert event_calls == [("declare_overlap_loop", True), ("loop_domain",)]


def test_the_executor_events_do_nothing_without_their_fast_paths(event_calls):
    events = _events(set())

    events.after_warmup()
    events.before_overlap_loop()
    events.at_shutdown()
    assert event_calls == []


def test_the_stager_attaches_the_attribute_every_engine_call_site_reads():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
        PrepStager,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    for cls in (DecoderRunner, TrtllmAttentionMetadata, SpecMetadata, KVCacheManagerV2):
        assert PrepStager._ATTACH_ATTR in vars(cls), cls
        assert getattr(cls, PrepStager._ATTACH_ATTR) is None, cls


class _ModelConfig:
    """What the shell and its hooks read off the model config."""

    class mapping:
        world_size = 8
        enable_attention_dp = False


class _SparseMlaDecodeFmha:
    """Stands in for the FMHA library the ``sparse_mla_decode`` kernel package registers."""


SPARSE_MLA_LIB = "glm_moe_dsa_sparse_mla_decode"
RANK_UNIFORM_ENV = dispatch.SHARED_ENV + (_router_index.MODELING_V2_VALIDATE_ENV,)


@pytest.fixture
def shell_construction(monkeypatch):
    """The shell class with the built-in constructor stubbed, and the events of a construction.

    The built-in constructor records the installed fast paths and the FMHA library list it
    would build its attention layers from; ``adopt_model`` records the plan it attaches. The
    selector state and the FMHA registry module are restored afterwards. The store writer the
    hooks build is a host-only stand-in.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (  # noqa: E501
        deferred_input_prep,
    )

    monkeypatch.setattr(deferred_input_prep, "SamplerStoreWriter", _StoreWriter)
    events = []

    def adopt_model(model, plan):
        events.append(("adopt", type(model.model).__name__, plan))

    def builtin_init(self, model_config):
        events.append("built-in model")
        events.append(("installed", dispatch.installed()))
        events.append(("fmha", list(fmha_registry.FMHA_LIBS)))
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = _builtin_text_model()
        self.model.layers = nn.ModuleList([_Layer(), _Layer(post_moe_hop=_Hop())])

    def allgather(value, world_size):
        events.append(("allgather", value, world_size))
        return [value] * world_size

    shell_cls = modeling.ModelingV2Glm52Nvfp4Sm103Tp8
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__init__", builtin_init)
    monkeypatch.setattr(modeling, "adopt_model", adopt_model)
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__post_init__", lambda self: None)
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__pp_init__", lambda self: None)
    monkeypatch.setitem(
        _router_index._RESOLVED_MODES, shell_cls, _router_index.ModelingV2Mode.REQUIRE
    )
    monkeypatch.setattr(dispatch, "_require", [False])
    monkeypatch.setattr(dispatch, "_world_allgather", allgather)
    monkeypatch.setattr(dispatch, "_INSTALLED", [None])
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(dispatch, "_log_info", lambda message: events.append(message))
    monkeypatch.setattr(
        dispatch, "_REGISTRY", {"sparse_mla_decode": {"fmha_lib": _SparseMlaDecodeFmha}}
    )
    for name in list(os.environ):
        if name.startswith(dispatch.KNOB_PREFIX) or name in RANK_UNIFORM_ENV:
            monkeypatch.delenv(name)
    registry_namespace = dict(vars(fmha_registry))
    yield shell_cls, events
    vars(fmha_registry).update(registry_namespace)


def test_the_shell_checks_the_ranks_resolves_the_plan_and_adopts_the_core(
    monkeypatch, shell_construction
):
    """The knobs are checked across the ranks before any other construction step, and the
    attached fast-path state after the model is built."""
    shell_cls, events = shell_construction
    monkeypatch.setenv(dispatch.DISABLE_ENV, "draft_tail")

    shell = shell_cls(_ModelConfig())

    allgather, record, built_in, installed, fmha, adopt, adopted = events
    assert allgather == (
        "allgather",
        ((dispatch.DISABLE_ENV, "draft_tail"), (dispatch.MODE_KEY, "require")),
        8,
    )
    # The layers are adopted after the core and before their state is checked.
    assert adopt == ("adopt", "GlmMoeDsaTp8ModelingV2", shell.plan)
    assert adopted == ("allgather", modeling.adopted_state(shell), 8)
    assert record.startswith("[glm_moe_dsa] dispatch: record paths=")
    assert built_in == "built-in model"
    assert installed == ("installed", frozenset(dispatch.FAST_PATH_NAMES) - {"draft_tail"})
    assert fmha[0] == "fmha"
    assert type(shell.model) is modeling.GlmMoeDsaTp8ModelingV2
    assert shell.plan.require and dispatch.require_mode()
    assert not shell.plan.enabled("draft_tail") and shell.plan.enabled("routed_moe")
    # The hooks are built on the engine's first read, not during construction.
    assert "_runtime_hooks" not in vars(shell)
    hooks = runtime_hooks_of(shell)
    assert hooks.step_input_stager is not None
    assert isinstance(hooks.sampler_store_writer, _StoreWriter)
    assert hooks.loop_on_execution_stream
    assert isinstance(hooks.executor_events, runtime.GlmExecutorEvents)
    assert runtime_hooks_of(shell) is hooks


def test_an_attribute_error_while_building_the_hooks_is_not_read_as_no_hooks(
    monkeypatch, shell_construction
):
    shell_cls, _ = shell_construction
    shell = shell_cls(_ModelConfig())

    def broken(model):
        raise AttributeError("plan")

    monkeypatch.setattr(runtime, "make_runtime_hooks", broken)
    with pytest.raises(RuntimeError, match="step-loop hooks"):
        runtime_hooks_of(shell)


def test_the_fast_paths_read_the_installed_plan(monkeypatch, shell_construction):
    """The plan is installed before the built-in model is built; DISABLE turns paths off."""
    shell_cls, events = shell_construction
    monkeypatch.setenv(dispatch.DISABLE_ENV, "post_attention_hop")

    shell = shell_cls(_ModelConfig())

    off = {name for name in dispatch.FAST_PATH_NAMES if not dispatch.is_enabled(name)}
    assert off == shell.plan.disabled | shell.plan.disabled_dependents
    assert "post_attention_hop" in off and "hop_routing" in off
    assert ("installed", frozenset(dispatch.FAST_PATH_NAMES) - off) in events
    # A knob the target does not read, set after construction, changes nothing.
    monkeypatch.setenv(dispatch.FEATURE_ENV, "post_attention_hop")
    assert not dispatch.is_enabled("post_attention_hop")
    record = next(e for e in events if isinstance(e, str) and "dispatch: record" in e)
    assert "post_attention_hop" not in record.split(" side_selectors=")[0]


def test_the_fmha_registry_stays_the_builtin_one(monkeypatch, shell_construction):
    """sparse_mla_decode joins no process-wide FMHA library list: adoption declares it on the
    GLM attention backends (``test_modeling_v2_glm_attention.py``)."""
    shell_cls, events = shell_construction
    builtin_libs = list(fmha_registry.FMHA_LIBS)
    builtin_enabled = fmha_manager.get_enabled_fmha_lib_classes()

    shell_cls(_ModelConfig())
    monkeypatch.setenv(dispatch.DISABLE_ENV, "sparse_mla_decode")
    shell_cls(_ModelConfig())

    assert SPARSE_MLA_LIB not in builtin_libs
    assert [e for e in events if isinstance(e, tuple) and e[0] == "fmha"] == [
        ("fmha", builtin_libs)
    ] * 2
    assert list(fmha_registry.FMHA_LIBS) == builtin_libs
    assert fmha_manager.get_enabled_fmha_lib_classes() == builtin_enabled


@pytest.mark.parametrize(
    "knob",
    sorted(dispatch.REMOVED_KNOBS),
    ids=lambda knob: knob.removeprefix(dispatch.KNOB_PREFIX),
)
def test_a_removed_knob_fails_before_the_builtin_model_is_built(
    monkeypatch, shell_construction, knob
):
    shell_cls, events = shell_construction
    monkeypatch.setenv(knob, "deferred_input_prep")

    with pytest.raises(ValueError, match=f"this target does not read {knob} "):
        shell_cls(_ModelConfig())
    assert [event[0] for event in events] == ["allgather"]
    assert dispatch.installed() is None


def test_a_knob_value_no_rank_can_parse_fails_before_the_builtin_model_is_built(
    monkeypatch, shell_construction
):
    shell_cls, events = shell_construction
    monkeypatch.setenv(dispatch.DISABLE_ENV, "not_a_fast_path")

    with pytest.raises(ValueError, match="not_a_fast_path"):
        shell_cls(_ModelConfig())
    assert [event[0] for event in events] == ["allgather"]


def test_the_ranks_check_the_shared_switches_and_the_mode_too(monkeypatch, shell_construction):
    shell_cls, events = shell_construction
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    monkeypatch.setenv(_router_index.MODELING_V2_VALIDATE_ENV, "1")

    shell_cls(_ModelConfig())

    assert events[0] == (
        "allgather",
        (
            ("TRTLLM_ENABLE_PDL", "1"),
            (_router_index.MODELING_V2_VALIDATE_ENV, "1"),
            (dispatch.MODE_KEY, "require"),
        ),
        8,
    )


def test_ranks_that_disagree_fail_before_the_builtin_model_is_built(
    monkeypatch, shell_construction
):
    shell_cls, events = shell_construction

    def allgather(value, world_size):
        events.append(("allgather", value, world_size))
        return [value, value + (("TRTLLM_FORCE_MNNVL_AR", "1"),)]

    monkeypatch.setattr(dispatch, "_world_allgather", allgather)
    with pytest.raises(RuntimeError, match="TRTLLM_FORCE_MNNVL_AR='1'"):
        shell_cls(_ModelConfig())
    assert [event[0] for event in events] == ["allgather"]
    assert dispatch.installed() is None


def test_the_adopted_state_names_each_attached_fast_path_object():
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[_Layer(), _Layer()]))
    model.model.layers[1].post_moe_hop = _Hop()
    model.model.layers[1].mlp.router_gemm = _Hop(engaged=False)
    model.model.layers[1].self_attn._attn_tail = object()

    assert modeling.adopted_state(model) == (
        ("model.layers.0", ()),
        (
            "model.layers.1",
            (
                ("post_moe_hop", "_Hop", "engaged=True", "impl='cluster'"),
                ("self_attn._attn_tail", "object"),
                ("mlp.router_gemm", "_Hop", "engaged=False", "impl='cluster'"),
            ),
        ),
    )


def test_ranks_that_attached_different_state_fail_after_the_model_is_built(
    monkeypatch, shell_construction
):
    shell_cls, events = shell_construction
    hooks = []
    monkeypatch.setattr(runtime, "make_runtime_hooks", lambda model: hooks.append(model))

    def allgather(value, world_size):
        events.append(("allgather", value, world_size))
        if value and value[0][0] == "model.layers.0":
            # A peer whose post-MoE hop on layer 1 did not engage.
            return [value, (value[0], ("model.layers.1", ()))]
        return [value] * world_size

    monkeypatch.setattr(dispatch, "_world_allgather", allgather)
    with pytest.raises(RuntimeError, match=r"rank 1: \('model.layers.1', \(\)\)") as excinfo:
        shell_cls(_ModelConfig())
    assert "rank 0: ('model.layers.1', (('post_moe_hop', '_Hop'" in str(excinfo.value)
    assert "built-in model" in events and hooks == []
