# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 tp8 target's decoder layers, draft layer and MoE modules: their adoption, the
phase gate of every forward override, the decode loop of ``DecodeTarget`` and the shell's
aliases.

Host-only: the built-in modules are created without their constructors and carry only the
attributes that adoption and the phase checks read. The kernel factories are stand-ins that
record their calls, and the built-in methods an override falls back to are recorders, so each
test sees which call a step makes and in which order adoption builds the fast-path objects.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    attention,
    dispatch,
    layers,
    spec_worker,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import DSAHooks
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3Gate,
    DeepseekV3Model,
    Deepseekv3MoE,
    DeepseekV3MTP,
    DeepseekV32Attention,
    GlmMoeDsaForCausalLM,
)
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker

pytestmark = pytest.mark.cpu_only

PREFILL, DECODE = dispatch.Phase.PREFILL, dispatch.Phase.DECODE
NUM_LAYERS = 3  # layer 0 dense, layers 1 and 2 MoE, layer 3 the draft layer
HIDDEN = 16


def _load_scope():
    """``kernels/deferred_norm/scope.py`` without importing the deferred-norm kernels."""
    path = Path(dispatch.__file__).parent / "kernels" / "deferred_norm" / "scope.py"
    spec = importlib.util.spec_from_file_location(
        f"{dispatch.__package__}.kernels.deferred_norm.scope", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCOPE = _load_scope()


# --- built-in modules without their constructors ------------------------------------------


def _bare(cls, **attributes):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


class _Gate(nn.Module):
    """Stands in for the router gate: a weight, a bias and logits of its own."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, HIDDEN))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(4))
        self.register_buffer("routing_table", torch.zeros(4))

    def forward(self, hidden_states):
        return "gate logits"


class _Experts(nn.Module):
    def __init__(self, calls=None) -> None:
        super().__init__()
        self.calls = calls if calls is not None else []
        self.weight = nn.Parameter(torch.zeros(2, 2))

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "routed"


class _Tail:
    """Stands in for an attention tail."""

    def __init__(self, events) -> None:
        self.events = events

    def set_successor_weights(self, weights) -> None:
        self.events.append(("set_successor_weights", tuple(weights)))


class _SparseMlaDecodeFmha:
    """Stands in for the sparse-MLA decode FMHA library."""


def _attention(layer_idx):
    """A layer's built-in attention modules: the MLA, its DSA hooks, its DSA backend and, on a
    full-indexer layer (even index here), its indexer."""
    backend = DSATrtllmAttention.__new__(DSATrtllmAttention)
    backend.indexer = _bare(Indexer) if layer_idx % 2 == 0 else None
    return _bare(
        DeepseekV32Attention,
        layer_idx=layer_idx,
        sparse_attn_hooks=DSAHooks.__new__(DSAHooks),
        mqa=backend,
        mha=None,
        indexer=backend.indexer,
    )


class _SparseParams:
    def to_sparse_params(self, pretrained_config, layer_idx):
        return types.SimpleNamespace(is_full_indexer_layer=layer_idx % 2 == 0)


def _config():
    return types.SimpleNamespace(
        n_routed_experts=4,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        torch_dtype=torch.bfloat16,
    )


def _model_config(config):
    mapping = types.SimpleNamespace(enable_attention_dp=False, tp_size=8, tp_rank=0)
    return types.SimpleNamespace(
        pretrained_config=config,
        mapping=mapping,
        moe_backend="TRTLLM",
        sparse_attention_config=_SparseParams(),
    )


def _moe(experts_calls=None):
    return _bare(
        Deepseekv3MoE,
        gate=_Gate(),
        experts=_Experts(experts_calls),
        use_dp=False,
        mapping=types.SimpleNamespace(tp_size=8),
        shared_experts=None,
    )


def _layer(layer_idx, events, model_config, cls=DeepseekV3DecoderLayer, nvfp4=True):
    config = model_config.pretrained_config
    moe = layer_idx >= config.first_k_dense_replace
    return _bare(
        cls,
        model_config=model_config,
        layer_idx=layer_idx,
        mapping=model_config.mapping,
        allreduce=object(),
        self_attn=_attention(layer_idx),
        mlp=_moe() if moe else nn.Linear(2, 2),
        fusion_config=types.SimpleNamespace(PRE_MOE_FUSION=moe, POST_MOE_FUSION=moe),
        is_nvfp4=nvfp4,
        top_k=2,
    )


def _mtp(events, model_config):
    layer = _layer(NUM_LAYERS, events, model_config, cls=DeepseekV3MTP, nvfp4=False)
    layer.enorm = nn.LayerNorm(2)
    layer.hnorm = nn.LayerNorm(2)
    return layer


def _keys(module):
    return (
        [name for name, _ in module.named_parameters()],
        [name for name, _ in module.named_buffers()],
        [name for name, _ in module.named_modules()],
    )


# --- stand-in kernel factories ------------------------------------------------------------


class _Made:
    """A fast-path object a factory made, with the call that made it."""

    def __init__(self, events, name, *args, engaged=True, **kwargs):
        self.events, self.name, self.args, self.kwargs = events, name, args, kwargs
        self.engaged = engaged
        events.append(("make", name))

    def enable_routed_input(self) -> bool:
        self.events.append(("enable_routed_input", self.name))
        return True


class _FastPaths:
    """``dispatch.is_enabled`` / ``get_impl`` / ``log_capability`` for the adoption tests."""

    def __init__(self, enabled, post_moe_engaged=True):
        self.events = []
        self.enabled = enabled
        self.post_moe_engaged = post_moe_engaged
        events = self.events

        class DeferredNormHop(_Made):
            def __init__(self, post_moe_hop, name="deferred_norm_hop"):
                super().__init__(events, name, post_moe_hop)
                # As the real hop: it names its post-MoE hop too, so the search for the
                # constructing layer must skip it.
                self.post_moe_hop = post_moe_hop
                _, next_layer, how = SCOPE.next_layer_kind(post_moe_hop)
                events.append(("deferred hop serves", next_layer, how))

        class EvictHintsDeferredHop(DeferredNormHop):
            def __init__(self, post_moe_hop):
                super().__init__(post_moe_hop, name="deferred_norm_evict_hints_hop")

        def draft_front(*args):
            return None

        def make_attn_tail(mla):
            # As the real factory: the draft layer keeps the native tail.
            if mla.layer_idx >= NUM_LAYERS:
                return None
            events.append(("make", "attn_tail"))
            return _Tail(events)

        def wrap_attn_tail(mla, tail):
            if tail is not None:
                events.append(("make", "staged_attn_tail"))
            return tail

        self.draft_front = draft_front
        self.entries = {
            ("attention_output_tail", "make_attn_tail"): make_attn_tail,
            ("oproj_weight_staging", "wrap_attn_tail"): wrap_attn_tail,
            ("sparse_mla_decode", "fmha_lib"): _SparseMlaDecodeFmha,
            ("post_attention_hop", "make_post_attn_hop"): lambda *a, **k: _Made(
                events, "post_attn_hop", *a, **k
            ),
            ("post_attention_hop", "make_into_hop_transport"): lambda hop: _Made(
                events, "into_hop", hop
            ),
            ("hop_routing", "make_transport"): lambda hop: _Made(events, "hop_routing", hop),
            ("router_gemm", "make_moe_router"): lambda **k: _Made(events, "router_gemm", **k),
            ("post_moe_hop", "post_moe_hop"): lambda **k: _Made(
                events, "post_moe_hop", engaged=self.post_moe_engaged, **k
            ),
            ("deferred_norm", "deferred_hop"): DeferredNormHop,
            ("deferred_norm_evict_hints", "deferred_hop"): EvictHintsDeferredHop,
            ("draft_front", "native_norm_reason"): lambda enorm, hnorm: "",
            ("draft_front", "draft_front"): draft_front,
            ("draft_live_rows", "make_live_row_hop"): lambda hop: _Made(
                events, "draft_live_row_hop", hop
            ),
        }

    def is_enabled(self, path):
        assert path in dispatch.FAST_PATH_NAMES, path
        return path in self.enabled

    def get_impl(self, path, name):
        assert path in self.enabled, path
        return self.entries[(path, name)]

    def log_capability(self, path, key, eligible, reason=""):
        self.events.append(("log_capability", path, key, eligible, reason))
        return eligible


@pytest.fixture
def fast_paths(monkeypatch):
    def install(enabled=dispatch.FAST_PATH_NAMES, post_moe_engaged=True):
        paths = _FastPaths(frozenset(enabled), post_moe_engaged)
        monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
        monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
        monkeypatch.setattr(dispatch, "log_capability", paths.log_capability)
        return paths

    return install


def _built_in_model(events):
    """The built-in model as ``adopt_model`` sees it: decoder layers, then the draft layer, and
    the speculative worker."""
    model_config = _model_config(_config())
    core = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(core)
    core.num_hidden_layers = NUM_LAYERS
    core.layers = nn.ModuleList(
        [_layer(i, events, model_config) for i in range(NUM_LAYERS)] + [_mtp(events, model_config)]
    )
    return types.SimpleNamespace(model=core, spec_worker=_bare(MTPEagleWorker))


# --- adoption -----------------------------------------------------------------------------


def test_adoption_builds_each_object_in_the_builtin_constructor_order(fast_paths):
    """Layer by layer: the attention tail, the post-attention hop and its transports, the router
    GEMM, the post-MoE hops, the tail's successor weights; on the draft layer then its front and
    live-row hop."""
    paths = fast_paths()
    model = _built_in_model(paths.events)

    modeling.adopt_model(model, plan=None)

    tail = [("make", "attn_tail"), ("make", "staged_attn_tail")]
    pre_moe = tail + [("make", "post_attn_hop"), ("make", "into_hop"), ("make", "hop_routing")]
    post_moe = [
        ("make", "router_gemm"),
        ("make", "post_moe_hop"),
        ("enable_routed_input", "post_moe_hop"),
    ]
    layers_0 = pre_moe
    layers_1 = (
        pre_moe
        + post_moe
        + [
            ("make", "deferred_norm_evict_hints_hop"),
            ("deferred hop serves", 2, "model config"),
            ("enable_routed_input", "deferred_norm_evict_hints_hop"),
        ]
    )
    layers_2 = pre_moe + post_moe  # the last decoder layer: no next layer to defer into
    tails = [e for e in paths.events if e[0] == "set_successor_weights"]
    draft = [
        ("make", "post_attn_hop"),
        ("make", "router_gemm"),
        ("log_capability", "draft_front", "draft_front", True, ""),
        ("make", "draft_live_row_hop"),
    ]
    events = [e for e in paths.events if e[0] != "set_successor_weights"]
    assert events == layers_0 + layers_1 + layers_2 + draft
    # The tail stages its MoE layer's router weight, after that layer's post-MoE hops.
    core = model.model
    assert tails == [("set_successor_weights", (core.layers[i].mlp.gate.weight,)) for i in (1, 2)]
    assert paths.events.index(tails[0]) == len(layers_0 + layers_1)


def test_adoption_attaches_what_the_constructors_attached(fast_paths):
    paths = fast_paths()
    model = _built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)
    dense, moe, last, draft = model.model.layers

    assert [type(m) for m in (dense, moe, last)] == [layers.GlmDecoderLayer] * 3
    assert type(draft) is layers.GlmMtpLayer
    assert [type(m.mlp) for m in (moe, last, draft)] == [layers.GlmMoE] * 3
    assert type(dense.mlp) is nn.Linear
    for layer in (dense, moe, last, draft):
        mla = layer.self_attn
        assert type(mla) is attention.GlmMLA
        assert type(mla.sparse_attn_hooks) is attention.GlmDsaHooks
        assert type(mla.mqa) is attention.GlmDsaBackend
        assert mla.mqa.prepended_fmha_lib_classes == (_SparseMlaDecodeFmha,)
        assert mla._tail_prefetch is True
    # Layers 0 and 2 run a full indexer, layers 1 and 3 reuse the shared top-k.
    assert [type(m.self_attn.mqa.indexer) for m in (dense, last)] == [attention.GlmIndexer] * 2
    assert moe.self_attn.mqa.indexer is None and draft.self_attn.mqa.indexer is None
    assert isinstance(moe.self_attn._attn_tail, _Tail) and draft.self_attn._attn_tail is None

    hop = moe.post_attn_hop
    assert hop.args == (moe.mapping, HIDDEN, torch.bfloat16) and hop.kwargs == {
        "native_allreduce": moe.allreduce
    }
    assert moe._into_hop.args == (hop,) and moe._hop_routing.args == (moe._into_hop,)
    assert moe.mlp.router_gemm.kwargs == {
        "gate": moe.mlp.gate,
        "layer_idx": 1,
        "model_config": moe.model_config,
    }
    assert moe.post_moe_hop.kwargs == {
        "mapping": moe.mapping,
        "hidden_size": HIDDEN,
        "top_k": 2,
        "dtype": torch.bfloat16,
        "raw_producer": moe.mlp.experts,
    }
    assert moe.mlp._routed_into_hop is moe.post_moe_hop
    assert moe.deferred_norm_hop.post_moe_hop is moe.post_moe_hop
    assert last.deferred_norm_hop is None and last.post_moe_hop is not None
    assert dense.post_moe_hop is None and dense.post_attn_hop is not None
    assert draft._into_hop is None and draft.post_moe_hop is None
    assert draft.draft_front is paths.draft_front
    assert draft.draft_live_row_hop.args == (draft.post_attn_hop,)
    assert draft._draft_live_rows is None and draft._draft_live_rows_compacted is False
    assert type(model.spec_worker) is spec_worker.GlmMtpWorker
    # The adopted state names every object, layer by layer.
    state = dict(modeling.adopted_state(types.SimpleNamespace(model=model.model)))
    assert [entry[0] for entry in state["model.layers.1"]] == [
        "post_attn_hop",
        "_into_hop",
        "_hop_routing",
        "post_moe_hop",
        "deferred_norm_hop",
        "self_attn._attn_tail",
        "mlp.router_gemm",
        "mlp._routed_into_hop",
    ]
    assert [entry[0] for entry in state["model.layers.3"]] == [
        "post_attn_hop",
        "draft_front",
        "draft_live_row_hop",
        "mlp.router_gemm",
    ]


def test_the_deferred_norm_variant_without_evict_hints(fast_paths):
    paths = fast_paths(set(dispatch.FAST_PATH_NAMES) - {"deferred_norm_evict_hints"})
    model = _built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)
    assert ("make", "deferred_norm_hop") in paths.events
    assert ("deferred hop serves", 2, "model config") in paths.events


def test_without_the_fast_paths_adoption_attaches_nothing(fast_paths):
    paths = fast_paths(enabled=())
    model = _built_in_model(paths.events)
    before = [_keys(layer) for layer in model.model.layers]

    modeling.adopt_model(model, plan=None)

    assert paths.events == []
    for layer in model.model.layers:
        assert type(layer.self_attn) is attention.GlmMLA and layer.self_attn._attn_tail is None
        assert layer.self_attn.mqa.prepended_fmha_lib_classes == ()
        assert "prepended_fmha_lib_classes" not in vars(layer.self_attn.mqa)
        for name in ("post_attn_hop", "_into_hop", "_hop_routing", "post_moe_hop"):
            assert getattr(layer, name) is None and name not in vars(layer)
        assert layer.deferred_norm_hop is None and layer._expert_prefetch_link is None
    draft = model.model.layers[NUM_LAYERS]
    assert draft.draft_front is None and draft.draft_live_row_hop is None
    assert model.model.layers[1].mlp.router_gemm is None
    assert model.model.layers[1].mlp._routed_into_hop is None
    assert [_keys(layer) for layer in model.model.layers] == before


def test_adoption_adds_no_parameter_buffer_or_module(fast_paths):
    paths = fast_paths()
    model = _built_in_model(paths.events)
    before = [_keys(layer) for layer in model.model.layers]
    modeling.adopt_model(model, plan=None)
    assert [_keys(layer) for layer in model.model.layers] == before
    assert "mlp.gate.weight" in before[1][0] and before[1][1] == ["mlp.gate.routing_table"]


def test_a_post_moe_hop_that_does_not_engage_is_not_attached(fast_paths):
    paths = fast_paths(post_moe_engaged=False)
    model = _built_in_model(paths.events)
    modeling.adopt_model(model, plan=None)
    moe = model.model.layers[1]
    assert moe.post_moe_hop is None and moe.deferred_norm_hop is None
    assert moe.mlp._routed_into_hop is None
    assert ("enable_routed_input", "post_moe_hop") not in paths.events


def test_the_draft_front_needs_non_adp_tensor_parallelism(fast_paths):
    paths = fast_paths()
    events = paths.events
    model_config = _model_config(_config())
    model_config.mapping.enable_attention_dp = True
    draft = layers.GlmMtpLayer.adopt(_mtp(events, model_config))
    assert draft.draft_front is None
    assert (
        "log_capability",
        "draft_front",
        "draft_front",
        False,
        "requires non-ADP tensor parallelism",
    ) in events


class _OtherDecoderLayer(DeepseekV3DecoderLayer):
    pass


class _OtherMoE(Deepseekv3MoE):
    pass


class _OtherAttention(DeepseekV32Attention):
    pass


def _refusal(case):
    """``(adopt, module, the class the refusal names)``."""
    events = []
    model_config = _model_config(_config())
    if case == "the draft layer as a decoder layer":
        return layers.GlmDecoderLayer.adopt, _mtp(events, model_config), "DeepseekV3MTP"
    if case == "a decoder layer subclass":
        module = _layer(1, events, model_config, cls=_OtherDecoderLayer)
        return layers.GlmDecoderLayer.adopt, module, "_OtherDecoderLayer"
    if case == "a module as a decoder layer":
        return layers.GlmDecoderLayer.adopt, nn.Module(), "Module"
    if case == "a decoder layer as the draft layer":
        return layers.GlmMtpLayer.adopt, _layer(0, events, model_config), "DeepseekV3DecoderLayer"
    if case == "a decoder layer with an attention subclass":
        module = _layer(1, events, model_config)
        module.self_attn.__class__ = _OtherAttention
        return layers.GlmDecoderLayer.adopt, module, "_OtherAttention"
    module = _layer(1, events, model_config)
    module.mlp = _bare(_OtherMoE, gate=_Gate())
    return layers.GlmDecoderLayer.adopt, module, "_OtherMoE"


@pytest.mark.parametrize(
    "case",
    [
        "the draft layer as a decoder layer",
        "a decoder layer subclass",
        "a module as a decoder layer",
        "a decoder layer as the draft layer",
        "a decoder layer with an attention subclass",
        "a decoder layer with a MoE subclass",
    ],
)
def test_adoption_refuses_anything_but_the_exact_builtin_class(fast_paths, case):
    fast_paths(enabled=())
    adopt, module, got = _refusal(case)
    with pytest.raises(TypeError, match=f"adopts the built-in .*, got {got}$"):
        adopt(module)


def test_the_moe_is_adopted_once(fast_paths):
    fast_paths(enabled=())
    model_config = _model_config(_config())
    moe = layers.GlmMoE.adopt(_moe(), model_config, 1)
    with pytest.raises(TypeError, match="GlmMoE adopts the built-in Deepseekv3MoE, got GlmMoE"):
        layers.GlmMoE.adopt(moe, model_config, 1)


def test_the_classes_declare_no_state_of_their_own():
    for cls in (layers.GlmDecoderLayer, layers.GlmMtpLayer, layers.GlmMoE):
        for base in cls.__mro__[: cls.__mro__.index(nn.Module)]:
            if base.__module__ == layers.__name__:
                assert "__slots__" not in vars(base) and "__init__" not in vars(base), base


# --- the phase gate -----------------------------------------------------------------------

# Each forward override and the built-in method it falls back to.
OVERRIDES = [
    (layers.GlmDecoderLayer, DeepseekV3DecoderLayer, "forward"),
    (layers.GlmDecoderLayer, DeepseekV3DecoderLayer, "forward_MoE"),
    (layers.GlmDecoderLayer, DeepseekV3DecoderLayer, "forward_mlp"),
    (layers.GlmMtpLayer, DeepseekV3MTP, "forward"),
    (layers.GlmMoE, Deepseekv3MoE, "forward"),
    (layers.GlmMoE, Deepseekv3MoE, "compute_routed_output"),
]


def _gate_of(cls, name):
    tree = ast.parse(Path(layers.__file__).read_text())
    node = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == cls.__name__
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return node, node.body[0]


@pytest.mark.parametrize("cls, builtin, name", OVERRIDES, ids=lambda x: getattr(x, "__name__", x))
def test_each_override_starts_with_the_phase_gate(cls, builtin, name):
    """``if dispatch.phase() is not DECODE: return super().<name>(<every built-in parameter by
    name>)``; the parameters only the decode composition takes are dropped."""
    node, gate = _gate_of(cls, name)
    assert ast.unparse(gate.test) == "dispatch.phase() is not dispatch.Phase.DECODE"
    assert len(gate.body) == 1 and not gate.orelse
    call = gate.body[0].value
    assert ast.unparse(call.func) == f"super().{name}" and not call.args
    passed = [k.arg for k in call.keywords]
    parameters = inspect.signature(getattr(builtin, name)).parameters
    expected = [
        p.name if p.kind is not p.VAR_KEYWORD else None
        for p in parameters.values()
        if p.name != "self"
    ]
    assert passed == expected
    # The override keeps the built-in parameters, in order, and may add keywords after them.
    own = list(inspect.signature(getattr(cls, name)).parameters)
    assert own[: len(parameters)] == list(parameters)


def _decode_layer():
    model_config = _model_config(_config())
    layer = _layer(1, [], model_config)
    layer.__class__ = layers.GlmDecoderLayer
    layer.mlp.__class__ = layers.GlmMoE
    return layer


def _calls_through(monkeypatch, owner, name, calls):
    def recorder(self, *args, **kwargs):
        calls.append((name, self, args, kwargs))
        return f"{owner.__name__}.{name}"

    monkeypatch.setattr(owner, name, recorder)


LAYER_ARGS = dict(position_ids="pos", hidden_states="h", attn_metadata="md", residual="r")


@pytest.mark.parametrize("phase", [PREFILL, None], ids=["prefill", "outside a step"])
def test_off_decode_every_override_runs_the_builtin_method(monkeypatch, phase):
    calls = []
    for owner, name in [
        (DeepseekV3DecoderLayer, "forward"),
        (DeepseekV3DecoderLayer, "forward_MoE"),
        (DeepseekV3DecoderLayer, "forward_mlp"),
        (DeepseekV3MTP, "forward"),
        (Deepseekv3MoE, "forward"),
        (Deepseekv3MoE, "compute_routed_output"),
    ]:
        _calls_through(monkeypatch, owner, name, calls)
    monkeypatch.setattr(layers.GlmDecoderLayer, "forward_decode", lambda *a, **k: pytest.fail())
    layer = _decode_layer()
    mtp = _bare(layers.GlmMtpLayer)

    def run():
        out = [
            layer.forward(**LAYER_ARGS, spec_metadata="spec", lora_params="lora"),
            layer.forward_MoE("h", "md", "r", "spec", into_hop_published=True),
            layer.forward_mlp("h", "r"),
            mtp.forward("ids", "pos", "h", "embed", "md", [2], "spec", lora_params="lora"),
            layer.mlp.forward("h", None, [2], "params", False, "logits", "routing"),
            layer.mlp.compute_routed_output("h", None, [2], True, "logits", "routing"),
        ]
        assert out == [
            "DeepseekV3DecoderLayer.forward",
            "DeepseekV3DecoderLayer.forward_MoE",
            "DeepseekV3DecoderLayer.forward_mlp",
            "DeepseekV3MTP.forward",
            "Deepseekv3MoE.forward",
            "Deepseekv3MoE.compute_routed_output",
        ]

    if phase is None:
        run()
    else:
        with dispatch.step(phase):
            run()
    assert [(c[0], c[2], c[3]) for c in calls] == [
        ("forward", (), {**LAYER_ARGS, "spec_metadata": "spec", "lora_params": "lora"}),
        (
            "forward_MoE",
            (),
            dict(hidden_states="h", attn_metadata="md", residual="r", spec_metadata="spec"),
        ),
        ("forward_mlp", (), dict(hidden_states="h", residual="r", spec_metadata=None)),
        (
            "forward",
            (),
            dict(
                input_ids="ids",
                position_ids="pos",
                hidden_states="h",
                embed_tokens="embed",
                attn_metadata="md",
                all_rank_num_tokens=[2],
                spec_metadata="spec",
                lora_params="lora",
            ),
        ),
        (
            "forward",
            (),
            dict(
                hidden_states="h",
                hidden_states_fp4=None,
                all_rank_num_tokens=[2],
                final_all_reduce_params="params",
                do_finalize=False,
            ),
        ),
        (
            "compute_routed_output",
            (),
            dict(
                hidden_states="h", hidden_states_fp4=None, all_rank_num_tokens=[2], do_finalize=True
            ),
        ),
    ]


def test_on_a_decode_step_the_layer_runs_its_decode_composition(monkeypatch):
    calls = []
    _calls_through(monkeypatch, DeepseekV3DecoderLayer, "forward", calls)
    _calls_through(monkeypatch, layers.GlmDecoderLayer, "forward_decode", calls)
    layer = _decode_layer()
    with dispatch.step(DECODE):
        assert layer.forward(**LAYER_ARGS, lora_params="lora") == "GlmDecoderLayer.forward_decode"
    assert calls == [
        ("forward_decode", layer, ("pos", "h", "md", "r", None), {"lora_params": "lora"})
    ]


@pytest.mark.parametrize("phase", [DECODE, PREFILL], ids=["decode", "prefill"])
def test_the_router_gemm_replaces_the_gate_on_decode_steps_only(fast_paths, phase):
    paths = fast_paths(enabled={"router_gemm"})
    paths.entries[("router_gemm", "make_moe_router")] = lambda **k: (lambda h: "router logits")
    experts = []
    moe = layers.GlmMoE.adopt(_moe(experts), _model_config(_config()), 1)
    hidden = torch.zeros(2, HIDDEN)

    with dispatch.step(phase):
        assert moe.compute_routed_output(hidden, None, [2], True) == "routed"

    logits = "router logits" if phase is DECODE else "gate logits"
    assert experts == [
        (
            (hidden, logits),
            dict(
                do_finalize=True,
                output_dtype=hidden.dtype,
                all_rank_num_tokens=[2],
                use_dp_padding=False,
            ),
        )
    ]


def test_the_post_attention_hop_reads_the_builtin_gates_weight_only():
    layer = _decode_layer()
    assert layer._post_attn_router_weight() is None  # a stand-in gate
    layer.mlp.gate = _bare(DeepseekV3Gate, weight=nn.Parameter(torch.zeros(4, HIDDEN)))
    assert layer._post_attn_router_weight() is layer.mlp.gate.weight


# --- DecodeTarget -------------------------------------------------------------------------


class _NoContexts:
    """Attention metadata a decode step passes; reading the context count fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"DecodeTarget read attn_metadata.{name}")


class _RecordingLayer(nn.Module):
    def __init__(self, index, calls) -> None:
        super().__init__()
        self.index, self.calls = index, calls

    def forward(self, **kwargs):
        self.calls.append((self.index, kwargs, dispatch.phase()))
        return kwargs["hidden_states"] + 1, f"residual {self.index}"


class _Embedding(nn.Module):
    def __init__(self, calls) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, input_ids):
        self.calls.append(("embed", input_ids))
        return torch.zeros(2, HIDDEN)


@pytest.fixture
def decode_core(monkeypatch):
    calls = []
    core = DeepseekV3Model.__new__(DeepseekV3Model)
    nn.Module.__init__(core)
    core.embed_tokens = _Embedding(calls)
    core.layers = nn.ModuleList([_RecordingLayer(i, calls) for i in range(NUM_LAYERS + 1)])
    core.num_hidden_layers = NUM_LAYERS
    core.mapping_with_cp = None
    core = modeling.GlmMoeDsaTp8ModelingV2.adopt(core)
    for layer in core.layers:
        layer.register_forward_pre_hook(
            lambda m, a, k: calls.append(("hook", m.index)), with_kwargs=True
        )
    return core, calls


@pytest.mark.parametrize("prologue_rows", [None, "rows"], ids=["no packet", "packet"])
def test_the_decode_loop_runs_the_decoder_layers_as_modules(fast_paths, decode_core, prologue_rows):
    core, calls = decode_core
    paths = fast_paths(enabled={"decode_prologue", "cross_layer_expert_prefetch"})
    rows = torch.full((2, HIDDEN), 7.0) if prologue_rows else None
    seen = []

    def take_embedding(attn_metadata, input_ids, embed_tokens):
        seen.append((attn_metadata, input_ids, embed_tokens))
        return rows

    core.decode_prologue = types.SimpleNamespace(take_embedding=take_embedding)
    paths.entries[("cross_layer_expert_prefetch", "join")] = lambda: calls.append(("join",))
    md = _NoContexts()

    with dispatch.step(DECODE):
        out = core._decode.forward(
            md, input_ids="ids", position_ids="pos", spec_metadata="spec", lora_params="x"
        )

    assert seen == [(md, "ids", core.embed_tokens)]
    embedded = [] if prologue_rows else [("embed", "ids")]
    expected_start = (rows if prologue_rows else torch.zeros(2, HIDDEN)) + NUM_LAYERS
    assert torch.equal(out, expected_start)
    layer_calls = []
    for i in range(NUM_LAYERS):
        kwargs = calls[len(embedded) + 2 * i + 1][1]
        assert set(kwargs) == {
            "position_ids",
            "hidden_states",
            "attn_metadata",
            "residual",
            "spec_metadata",
        }
        assert kwargs["residual"] == (None if i == 0 else f"residual {i - 1}")
        assert kwargs["attn_metadata"] is md and kwargs["spec_metadata"] == "spec"
        layer_calls += [("hook", i), (i, kwargs, DECODE)]
    assert calls == embedded + layer_calls + [("join",)]


def test_the_decode_loop_without_its_fast_paths_is_the_builtin_loop(fast_paths, decode_core):
    core, calls = decode_core
    fast_paths(enabled=())
    with dispatch.step(DECODE):
        out = core._decode.forward(_NoContexts(), input_ids="ids")
    assert torch.equal(out, torch.zeros(2, HIDDEN) + NUM_LAYERS)
    assert [c[0] for c in calls] == ["embed"] + ["hook", 0, "hook", 1, "hook", 2]


def test_the_decode_loop_takes_ids_or_embeddings(fast_paths, decode_core):
    core, _ = decode_core
    fast_paths(enabled=())
    with pytest.raises(ValueError, match="cannot specify both input_ids and inputs_embeds"):
        core._decode.forward(_NoContexts(), input_ids="ids", inputs_embeds=torch.zeros(1))
    with pytest.raises(ValueError, match="must specify either one"):
        core._decode.forward(_NoContexts())


def test_the_decode_loop_never_reads_the_step_kind():
    source = inspect.getsource(modeling.DecodeTarget.forward)
    assert "num_contexts" not in source and "num_ctx" not in source


# --- the shell's aliases ------------------------------------------------------------------


@pytest.mark.parametrize("linked", [True, False], ids=["prefetch", "no prefetch"])
def test_the_shell_links_the_expert_prefetch_after_the_builtin_aliases(
    monkeypatch, fast_paths, linked
):
    paths = fast_paths(enabled={"cross_layer_expert_prefetch"} if linked else ())
    events = paths.events
    paths.entries[("cross_layer_expert_prefetch", "link_layers")] = (
        lambda layers_, n: events.append(("link_layers", layers_, n))
    )
    monkeypatch.setattr(
        GlmMoeDsaForCausalLM, "setup_aliases", lambda self: events.append("built-in aliases")
    )
    shell = _bare(modeling.ModelingV2Glm52Nvfp4Sm103Tp8)
    shell.model = types.SimpleNamespace(layers=["layer"])
    shell.model_config = types.SimpleNamespace(
        pretrained_config=types.SimpleNamespace(num_hidden_layers=78)
    )

    shell.setup_aliases()

    assert events == ["built-in aliases"] + ([("link_layers", ["layer"], 78)] if linked else [])
