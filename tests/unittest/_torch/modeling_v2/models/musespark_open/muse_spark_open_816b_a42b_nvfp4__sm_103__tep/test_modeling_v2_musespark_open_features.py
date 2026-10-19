# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's fast paths: what each one of a plan changes, and how they combine.

Host-only. For ``preattn_prep``: a stand-in text model with built-in
``MuseSparkOpenAttention`` objects created without their constructor (which
builds the projections, the attention backend and CUDA streams). What
``preattn_prep`` computes is the GPU test's business; here: which modules are
re-classed, which calls carry the entry, and that nothing changes under
``NO_PATHS``. For the drafter paths: the shell decides its plan once, at
construction. Under ``NO_PATHS`` its drafter load is the built-in one and
leaves every drafter module as built; with ``drafter_block_conv`` it
re-classes the drafter's DFlash 2 block convolutions and nothing else; with
``drafter_block_decode`` too one drafter load installs both (the block decode
alone is covered by ``test_modeling_v2_musespark_open_drafter_block_decode.py``), and
``draft_head`` is installed by the same load after them (alone:
``test_modeling_v2_musespark_open_draft_head_install.py``).
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime as _runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    weights as target_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    preattn_prep,
)
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenAttention,
    MuseSparkOpenModel,
)
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_runtime_hooks(monkeypatch):
    """The executor hooks pin host staging blocks; these shells are built without them."""
    monkeypatch.setattr(_runtime, "make_runtime_hooks", lambda model: None)


_CORE = modeling.MuseSparkOpenTepModelingV2
_ATTN = target_attention.MuseSparkOpenTepAttention
_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
_HIDDEN, _TAPS, _GROUP = 32, 2, 16
_PLAN = dispatch.plan_of
_PREATTN_PREP = dispatch.plan_of("preattn_prep")


# --------------------------------------------------------------------------
# Pre-attention prep stand-ins: the text model
# --------------------------------------------------------------------------
def _builtin_attention(layer_idx: int, rope: bool, **overrides) -> MuseSparkOpenAttention:
    """A built-in attention as far as the attributes ``adopt`` reads go."""
    attention = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(attention)
    attention.layer_idx = layer_idx
    attention._fuse_qk_norm_rope_gate = False
    attention.fuse_qk_norm_rope = False
    attention.is_qk_norm = False
    attention.q_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.k_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.rope_fusion = False
    attention.support_fused_qkv = True
    attention.head_dim = 64
    attention.skip_rope = not rope
    rotary = None
    if rope:
        rotary = RotaryEmbedding.__new__(RotaryEmbedding)
        nn.Module.__init__(rotary)
        rotary.is_neox = False
        rotary.inverse = False
        rotary.rotary_cos_sin = torch.zeros(16, 2, 32)
    attention.rotary_emb = rotary
    for name, value in overrides.items():
        setattr(attention, name, value)
    return attention


class _Layer(nn.Module):
    """Records each call's extra keywords; owns a built-in attention."""

    def __init__(self, index: int, calls: list) -> None:
        super().__init__()
        self.index = index
        self.pre_qkv_proj_norm = nn.Identity()
        self.self_attn = _builtin_attention(index, rope=(4 - index - 1) % 4 != 0)
        self.calls = calls

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        sp_num_tokens=None,
        **kwargs,
    ):
        self.calls.append((self.index, dict(kwargs)))
        out = hidden_states + 1.0
        if next_layer_norm is None:
            return out
        return out, out.to(torch.bfloat16)


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.bfloat16).unsqueeze(-1).expand(-1, 4).contiguous()


class _Metadata:
    def __init__(self, num_contexts: int) -> None:
        self.num_contexts = num_contexts


def _builtin_text_model(num_layers: int = 4) -> MuseSparkOpenModel:
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    calls: list = []
    model.embed_tokens = _Embedding()
    model.embed_norm = nn.Identity()
    model.norm = nn.Identity()
    model.layers = nn.ModuleList([_Layer(i, calls) for i in range(num_layers)])
    model.dtype = torch.bfloat16
    model.m_emb = 1.0
    model._prefill_sp_enabled = False
    model._sp_mapping = None
    model.calls = calls
    return model


@pytest.fixture
def preattn_prep_on(monkeypatch):
    """The plan with the fused pre-attention prep; the built-in prep kernels agree."""
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    return _PREATTN_PREP


@pytest.fixture
def no_features():
    return dispatch.NO_PATHS


# --------------------------------------------------------------------------
# Pre-attention prep wiring
# --------------------------------------------------------------------------
def test_no_feature_leaves_the_attention_and_the_layer_calls_alone(no_features):
    core = _CORE.adopt(_builtin_text_model())
    assert core._plan == dispatch.NO_PATHS
    assert core._decode.preattn_prep is None
    assert all(type(layer.self_attn) is MuseSparkOpenAttention for layer in core.layers)
    core(_Metadata(0), torch.tensor([3, 5], dtype=torch.int32), torch.tensor([7, 8]))
    assert [kwargs for _, kwargs in core.calls] == [{}] * 4


def test_preattn_prep_reclasses_every_attention_in_place(preattn_prep_on):
    model = _builtin_text_model()
    attentions = [layer.self_attn for layer in model.layers]
    modules_before = [name for name, _ in model.named_modules()]
    core = _CORE.adopt(model, preattn_prep_on)
    assert core._plan == _PREATTN_PREP
    for layer, attention in zip(core.layers, attentions):
        assert layer.self_attn is attention
        assert type(attention) is _ATTN
        assert isinstance(attention, MuseSparkOpenAttention)
        assert attention._preattn_prep is None
    assert [name for name, _ in core.named_modules()] == modules_before


def test_preattn_prep_hands_the_entry_to_every_decode_layer_call_and_no_prefill_call(
    preattn_prep_on, monkeypatch
):
    core = _CORE.adopt(_builtin_text_model(), preattn_prep_on)
    assert core._decode.preattn_prep is preattn_prep
    ids, positions = torch.tensor([3, 5], dtype=torch.int32), torch.tensor([7, 8])

    core(_Metadata(0), ids, positions, spec_metadata="spec")
    assert core.calls == [
        (i, {"spec_metadata": "spec", "preattn_prep": preattn_prep}) for i in range(4)
    ]

    core.calls.clear()
    core(_Metadata(1), ids, positions, spec_metadata="spec")
    assert core.calls == [(i, {"spec_metadata": "spec"}) for i in range(4)]


@pytest.mark.parametrize(
    "override, reason",
    [
        (dict(_fuse_qk_norm_rope_gate=True), "MUSESPARK_OPEN_FUSED_PREATTN"),
        (dict(fuse_qk_norm_rope=True), "C\\+\\+ fused qk-norm"),
        (dict(is_qk_norm=True), "carry weights"),
        (dict(rope_fusion=True), "inside the attention op"),
        (dict(support_fused_qkv=False), "split Q/K/V"),
        (dict(head_dim=128), "head_dim is not 64"),
    ],
)
def test_preattn_prep_refuses_a_prep_it_does_not_reproduce(preattn_prep_on, override, reason):
    model = _builtin_text_model()
    attention = model.layers[2].self_attn
    for name, value in override.items():
        setattr(attention, name, value)
    with pytest.raises(ValueError, match=f"layer 2: .*{reason}"):
        _CORE.adopt(model, preattn_prep_on)


@pytest.mark.parametrize("attribute", ("is_neox", "inverse"))
def test_preattn_prep_refuses_a_rotation_other_than_gptj(preattn_prep_on, attribute):
    model = _builtin_text_model()
    setattr(model.layers[0].self_attn.rotary_emb, attribute, True)
    with pytest.raises(ValueError, match="layer 0: .*GPT-J"):
        _CORE.adopt(model, preattn_prep_on)


@pytest.mark.parametrize("attribute", ("use_cuda_tile", "is_nvfp4"))
@pytest.mark.parametrize("norm", ("q_norm", "k_norm"))
def test_preattn_prep_refuses_a_cuda_tile_or_nvfp4_norm(preattn_prep_on, norm, attribute):
    model = _builtin_text_model()
    setattr(getattr(model.layers[3].self_attn, norm), attribute, True)
    with pytest.raises(ValueError, match="layer 3: .*cuda-tile or NVFP4"):
        _CORE.adopt(model, preattn_prep_on)
    # Layers 0-2 passed their checks, and still nothing was re-classed.
    assert all(type(layer.self_attn) is MuseSparkOpenAttention for layer in model.layers)


@pytest.mark.parametrize(
    "table",
    [
        torch.zeros(16, 2, 32, dtype=torch.bfloat16),
        torch.zeros(16, 64),
        torch.zeros(16, 2, 64),
        torch.zeros(16, 32, 2).transpose(1, 2),
    ],
    ids=["bf16", "flat", "wide", "non-contiguous"],
)
def test_preattn_prep_refuses_a_rope_table_the_entry_does_not_read_at_adoption(
    preattn_prep_on, table
):
    """At adoption, not at the first decode step or CUDA-graph warmup."""
    model = _builtin_text_model()
    model.layers[2].self_attn.rotary_emb.rotary_cos_sin = table
    with pytest.raises(ValueError, match=r"layer 2: .*RoPE table is not a contiguous fp32"):
        _CORE.adopt(model, preattn_prep_on)
    assert all(type(layer.self_attn) is MuseSparkOpenAttention for layer in model.layers)


def test_preattn_prep_refuses_when_the_builtin_prep_runs_other_kernels(
    preattn_prep_on, monkeypatch
):
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: "some reason")
    with pytest.raises(ValueError, match="CuTe-DSL RMSNorm and cos/sin-cache RoPE.*some reason"):
        _CORE.adopt(_builtin_text_model(), preattn_prep_on)


def _norm_module(**attrs) -> types.ModuleType:
    module = types.ModuleType("flashinfer.norm")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


@pytest.mark.parametrize(
    "available, norm_module, expected",
    [
        (False, _norm_module(_use_cuda_norm=lambda device: False), "flashinfer is not available"),
        (True, None, "flashinfer.norm does not import"),
        (True, _norm_module(), "kernel is unknown"),
        (True, _norm_module(_use_cuda_norm=lambda device: True), "CUDA-JIT kernel"),
        (True, _norm_module(_use_cuda_norm=lambda device: False), None),
    ],
    ids=["no-flashinfer", "import-error", "no-dispatch-probe", "cuda-jit-norm", "cute-dsl-norm"],
)
def test_the_kernel_check_names_each_mismatch_and_adoption_raises_valueerror(
    monkeypatch, available, norm_module, expected
):
    """Every mismatch -- flashinfer missing or unimportable included -- is a
    ``ValueError`` from adoption, never an ``ImportError``."""
    monkeypatch.setattr(target_attention._flashinfer_utils, "IS_FLASHINFER_AVAILABLE", available)
    monkeypatch.setitem(sys.modules, "flashinfer.norm", norm_module)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mismatch = target_attention._builtin_prep_kernel_mismatch()
    if expected is None:
        assert mismatch is None
        assert type(_CORE.adopt(_builtin_text_model(), _PREATTN_PREP).layers[0].self_attn) is _ATTN
        return
    assert expected in mismatch
    with pytest.raises(ValueError, match=r"\[musespark_open\] preattn_prep reproduces"):
        _CORE.adopt(_builtin_text_model(), _PREATTN_PREP)


def test_attention_adopt_refuses_anything_but_the_builtin_attention(preattn_prep_on):
    adopted = _ATTN.adopt(_builtin_attention(0, rope=True))
    with pytest.raises(TypeError, match="MuseSparkOpenTepAttention"):
        _ATTN.adopt(adopted)
    with pytest.raises(TypeError, match="Module"):
        _ATTN.adopt(nn.Module())


def test_the_entry_is_scoped_to_one_forward_call(monkeypatch):
    """``forward`` holds the entry for exactly its own call, and does not pass
    it on to the built-in forward, even when that raises."""
    seen = []

    def builtin_forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        seen.append((self._preattn_prep, args, kwargs))
        if kwargs.get("fail"):
            raise RuntimeError("boom")
        return "out"

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", builtin_forward)
    attention = _ATTN.adopt(_builtin_attention(0, rope=True))
    out = attention(
        position_ids="p",
        hidden_states="h",
        attn_metadata="md",
        overlap_sp_output_projection=False,
        preattn_prep=preattn_prep,
        spec_metadata="spec",
    )
    assert out == "out"
    assert seen == [
        (preattn_prep, (), {"overlap_sp_output_projection": False, "spec_metadata": "spec"})
    ]
    assert attention._preattn_prep is None

    seen.clear()
    attention(position_ids="p", hidden_states="h", attn_metadata="md")
    assert seen == [(None, (), {})]

    with pytest.raises(RuntimeError, match="boom"):
        attention("p", "h", "md", preattn_prep=preattn_prep, fail=True)
    assert attention._preattn_prep is None


def test_rows_above_the_cap_take_the_builtin_prep_without_a_warning(monkeypatch):
    attention = _ATTN.adopt(_builtin_attention(0, rope=True))
    attention._preattn_prep = preattn_prep
    sentinel = object()
    monkeypatch.setattr(
        MuseSparkOpenAttention, "preprocess_qkv", lambda self, qkv, position_ids: sentinel
    )
    warnings = []
    monkeypatch.setattr(target_attention.logger, "warning_once", lambda *a, **k: warnings.append(a))
    rows = target_attention._PREATTN_PREP_MAX_ROWS + 1
    qkv = torch.zeros(rows, 2304, dtype=torch.bfloat16)
    assert attention.preprocess_qkv(qkv, torch.zeros(1, rows, dtype=torch.int32)) is sentinel
    assert warnings == []
    # A decode projection the entry cannot take (here: on the host) falls back, loudly.
    assert attention.preprocess_qkv(qkv[:8], torch.zeros(1, 8, dtype=torch.int32)) is sentinel
    assert len(warnings) == 1


# --------------------------------------------------------------------------
# Drafter block stand-ins: the shell and its drafter
# --------------------------------------------------------------------------
def _conv() -> DFlash2BlockConv:
    groups = _HIDDEN // _GROUP
    return DFlash2BlockConv(
        base_kernel=torch.zeros(2, _TAPS, _HIDDEN, dtype=torch.bfloat16),
        kernel_projection_weight=torch.zeros(2 * _TAPS * groups, _HIDDEN, dtype=torch.bfloat16),
        taps=_TAPS,
        group_size=_GROUP,
        hidden_size=_HIDDEN,
    )


def _flash_attn_with_kvcache(**kwargs):
    return "flash-attn"


class _Drafter(nn.Module):
    """Block convolutions, and the VANILLA block attention attribute the windowed block attention wraps."""

    def __init__(self, layers: int = 2, convs: bool = True) -> None:
        super().__init__()
        self.attention_convs = nn.ModuleList(_conv() for _ in range(layers)) if convs else None
        self.mlp_convs = nn.ModuleList(_conv() for _ in range(layers)) if convs else None
        self.dflash_attention_backend = "VANILLA"
        self._dflash_flash_attention = _flash_attn_with_kvcache


class _Llm(nn.Module):
    def __init__(self, drafter) -> None:
        super().__init__()
        self.model = nn.Module()
        self.draft_model = drafter


@pytest.fixture
def block_decode(monkeypatch):
    """The written-out block decode bound as ``draft.install_block_decode`` binds it; the
    stand-in drafter is not a ``DFlashForCausalLM``, which the real check refuses."""

    def install(draft_model, enable_pdl):
        draft_model.dflash_forward = types.MethodType(draft.windowed_block_decode, draft_model)

    monkeypatch.setattr(draft, "install_block_decode", install)


@pytest.fixture
def build(monkeypatch):
    """Build shells through their own constructor, the built-in one and the load stubbed.

    Returns ``build(*paths)``: constructs a shell over a stand-in drafter whose
    composition is the plan of ``paths`` (``NO_PATHS`` without any). ``events``
    records the built-in constructor, text-model adoption and drafter loads.
    """
    events = []

    def builtin_init(self, model_config, *args, **kwargs):
        events.append("built-in model")
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = _Llm(_Drafter())

    def builtin_load_draft_weights(self, *args, **kwargs):
        events.append(("built-in draft load", args, kwargs))
        return "loaded"

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration, "load_draft_weights", builtin_load_draft_weights
    )
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(
            lambda cls, model, plan=dispatch.NO_PATHS, compute_capability=None: events.append(
                "adopt"
            )
        ),
    )

    def _build(*paths):
        plan = _PLAN(*paths)
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        return _SHELL(model_config=None)

    _build.events = events
    return _build


def _conv_types(drafter):
    return [type(c) for convs in (drafter.attention_convs, drafter.mlp_convs) for c in convs]


def test_the_drafter_load_keeps_the_builtin_signature():
    # The model loader inspects the parameters to decide what it passes.
    assert (
        inspect.getfullargspec(_SHELL.load_draft_weights).args
        == inspect.getfullargspec(MuseSparkOpenOmniForConditionalGeneration.load_draft_weights).args
    )


def test_no_feature_is_the_builtin_drafter_load(build):
    shell = build()
    assert shell._plan == dispatch.NO_PATHS
    params = {n: p.data_ptr() for n, p in shell.llm.draft_model.named_parameters()}
    assert shell.load_draft_weights("weights", "mapper") == "loaded"
    assert build.events == [
        "built-in model",
        "adopt",
        ("built-in draft load", ("weights", "mapper"), {}),
    ]
    assert _conv_types(shell.llm.draft_model) == [DFlash2BlockConv] * 4
    assert {n: p.data_ptr() for n, p in shell.llm.draft_model.named_parameters()} == params


def test_block_conv_reclasses_every_block_convolution_in_place(build):
    shell = build("drafter_block_conv")
    assert shell._plan == _PLAN("drafter_block_conv")
    drafter = shell.llm.draft_model
    modules = list(drafter.modules())
    params = {n: p.data_ptr() for n, p in drafter.named_parameters()}
    assert shell.load_draft_weights("weights") == "loaded"
    assert build.events[-1] == ("built-in draft load", ("weights",), {})
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 4
    assert list(drafter.modules()) == modules
    assert {n: p.data_ptr() for n, p in drafter.named_parameters()} == params
    # A second load (the engine may reload draft weights) leaves them installed.
    shell.load_draft_weights("weights")
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 4


def test_the_plan_is_decided_once_at_construction(monkeypatch, build):
    shell = build("drafter_block_conv")
    monkeypatch.setattr(modeling._dispatch, "compose", lambda *a, **k: dispatch.NO_PATHS)
    shell.load_draft_weights("weights")
    assert _conv_types(shell.llm.draft_model) == [draft.FusedDFlash2BlockConv] * 4

    shell = build()
    monkeypatch.setattr(modeling._dispatch, "compose", lambda *a, **k: _PLAN("drafter_block_conv"))
    assert shell.load_draft_weights("weights") == "loaded"
    assert _conv_types(shell.llm.draft_model) == [DFlash2BlockConv] * 4


def test_install_counts_and_is_idempotent():
    drafter = _Drafter(layers=3)
    assert draft.install_fused_block_conv(drafter) == 6
    assert draft.install_fused_block_conv(drafter) == 6
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 6


def test_block_conv_refuses_a_drafter_without_block_convolutions():
    """A plain DFlash drafter has no DFlash 2 block convolution to fuse: a planned
    block convolution raises instead of running the built-in drafter under its name."""
    plain = _Drafter(convs=False)
    assert draft.install_fused_block_conv(plain) == 0
    for model in (plain, None):
        with pytest.raises(
            ValueError, match="drafter_block_conv cannot run here: .*no DFlash 2 block conv"
        ):
            draft.install(model, True, False)


def test_a_repeated_install_logs_nothing(monkeypatch, block_decode):
    """The shell installs from every load hook; only the call that changed something logs."""
    logged = []
    # Only this target's lines: the shared logger also says "PDL enabled" on first use.
    monkeypatch.setattr(
        draft.logger, "info", lambda msg: msg.startswith("[musespark_open]") and logged.append(msg)
    )
    drafter = _Drafter(layers=2)
    draft.install(drafter, True, True)
    assert [msg.split(":")[0] for msg in logged] == [
        "[musespark_open] drafter_block_conv",
        "[musespark_open] drafter_block_decode",
        "[musespark_open] drafter_block_decode",
    ]
    draft.install(drafter, True, True)
    assert len(logged) == 3
    # A reload builds new block convolutions: the fused block convolution changes them, and says so.
    drafter.mlp_convs[0].__class__ = DFlash2BlockConv
    draft.install(drafter, True, True)
    assert logged[3:] == [
        "[musespark_open] drafter_block_conv: fused DFlash 2 block convolution on 4 modules"
    ]


def test_install_refuses_a_foreign_block_convolution():
    class Other(DFlash2BlockConv):
        pass

    drafter = _Drafter(layers=1)
    drafter.mlp_convs[0].__class__ = Other
    with pytest.raises(TypeError, match="Other"):
        draft.install_fused_block_conv(drafter)


def test_the_fused_module_falls_back_on_operands_the_kernel_does_not_take(monkeypatch):
    calls = []
    monkeypatch.setattr(draft, "dflash2_block_conv", lambda *a: calls.append(a))
    conv = _conv()
    conv.__class__ = draft.FusedDFlash2BlockConv
    x = torch.ones(4, _HIDDEN, dtype=torch.bfloat16)
    delta = torch.zeros(4, _TAPS, _HIDDEN // _GROUP, dtype=torch.bfloat16)
    # fp32 activations: torch promotes the built-in chain, so the built-in runs.
    out = conv._convolve(x.float(), delta, 0, 4)
    assert out.dtype == torch.float32 and not calls
    # A transposed (non-unit column stride) input also goes to the built-in.
    conv._convolve(torch.ones(_HIDDEN, 4, dtype=torch.bfloat16).t(), delta, 0, 4)
    assert not calls
    conv._convolve(x, delta, 1, 4)
    assert len(calls) == 1 and calls[0][2] is not None and calls[0][3:] == (4, _GROUP)


# --------------------------------------------------------------------------
# The features together
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "paths",
    [
        ("draft_head",),
        ("drafter_block_conv", "drafter_block_decode", "draft_head"),
        (
            "preattn_prep",
            "drafter_block_conv",
            "drafter_block_decode",
            "draft_head",
            "projection_gathers",
        ),
    ],
)
def test_one_drafter_load_installs_draft_head_after_the_drafter_modules(build, monkeypatch, paths):
    """``draft.install`` (the drafter's modules) runs before ``draft_head.install``
    (the DFlash worker and the logits processor), once each per load, with the
    plan decided at construction; the draft head alone does not call ``draft.install``."""
    calls = []
    monkeypatch.setattr(
        draft, "install", lambda model, conv, decode: calls.append(("draft", model, conv, decode))
    )
    monkeypatch.setattr(draft_head, "install", lambda llm: calls.append(("draft_head", llm)))
    shell = build(*paths)
    assert shell.load_draft_weights("weights") == "loaded"
    conv, decode = "drafter_block_conv" in paths, "drafter_block_decode" in paths
    expected = [("draft", shell.llm.draft_model, conv, decode)] if conv or decode else []
    expected.append(("draft_head", shell.llm))
    assert calls == expected


@pytest.mark.parametrize(
    "paths",
    [
        ("drafter_fc_shard", "fused_accept", "drafter_fixed_gemms"),
        ("draft_head", "fused_accept"),
        (
            "drafter_block_conv",
            "drafter_block_decode",
            "draft_head",
            "drafter_fc_shard",
            "fused_accept",
            "drafter_fixed_gemms",
        ),
        ("preattn_prep", "draft_head", "projection_gathers", "drafter_fixed_gemms"),
    ],
)
def test_one_drafter_load_installs_drafter_worker_after_draft_head(build, monkeypatch, paths):
    """``drafter_worker.install`` runs after ``draft_head.install``: the fused acceptance
    derives its worker class from the draft head's, and the draft head refuses a worker that
    is not the built-in class."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )

    calls = []
    monkeypatch.setattr(draft, "install", lambda model, conv, decode: calls.append("draft"))
    monkeypatch.setattr(draft_head, "install", lambda llm: calls.append("draft_head"))
    monkeypatch.setattr(
        drafter_worker,
        "install",
        lambda model, worker, mapping, *switches: calls.append(("drafter_worker", switches)),
    )
    shell = build(*paths)
    assert shell.load_draft_weights("weights") == "loaded"
    expected = ["draft"] if {"drafter_block_conv", "drafter_block_decode"} & set(paths) else []
    expected += ["draft_head"] if "draft_head" in paths else []
    switches = tuple(
        name in paths for name in ("drafter_fc_shard", "fused_accept", "drafter_fixed_gemms")
    )
    expected.append(("drafter_worker", switches))
    assert calls == expected


@pytest.mark.parametrize("hook", ["post_load_weights", "cache_derived_state"])
def test_the_post_load_walk_after_the_drafter_load_changes_nothing(
    build, monkeypatch, block_decode, hook
):
    """The loader's post-load walk calls the shell's hook after ``load_draft_weights``;
    the second install of the block convolution and the block attention is a no-op
    (and installs them alone on a load path that never calls ``load_draft_weights``)."""
    heads = []
    monkeypatch.setattr(draft_head, "install", lambda llm: heads.append(llm))
    shell = build("drafter_block_conv", "drafter_block_decode", "draft_head")
    drafter = shell.llm.draft_model
    getattr(shell, hook)()
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 4
    attention = drafter._dflash_flash_attention
    assert isinstance(attention, draft.WindowedBlockAttention)
    shell.load_draft_weights("weights")
    getattr(shell, hook)()
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 4
    assert drafter._dflash_flash_attention is attention
    assert heads == [shell.llm] * 3


@pytest.mark.parametrize(
    "paths",
    [(), ("preattn_prep",), ("projection_gathers",), ("preattn_prep", "projection_gathers")],
)
def test_target_only_selections_install_nothing_on_the_drafter(build, monkeypatch, paths):
    calls = []
    monkeypatch.setattr(draft, "install", lambda *args: calls.append("draft"))
    monkeypatch.setattr(draft_head, "install", lambda *args: calls.append("draft_head"))
    shell = build(*paths)
    assert shell.load_draft_weights("weights") == "loaded"
    assert calls == []


def test_preattn_prep_alone_leaves_the_drafter_alone(build):
    shell = build("preattn_prep")
    assert shell._plan == _PREATTN_PREP
    drafter = shell.llm.draft_model
    assert shell.load_draft_weights("weights") == "loaded"
    assert _conv_types(drafter) == [DFlash2BlockConv] * 4
    assert drafter._dflash_flash_attention is _flash_attn_with_kvcache


@pytest.mark.parametrize(
    "paths",
    [
        ("drafter_block_conv", "drafter_block_decode"),
        ("preattn_prep", "drafter_block_conv", "drafter_block_decode"),
    ],
)
def test_one_drafter_load_installs_every_selected_drafter_feature(build, block_decode, paths):
    shell = build(*paths)
    drafter = shell.llm.draft_model
    modules = list(drafter.modules())
    params = {n: p.data_ptr() for n, p in drafter.named_parameters()}
    assert shell.load_draft_weights("weights") == "loaded"
    assert build.events[-1] == ("built-in draft load", ("weights",), {})
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 4
    attention = drafter._dflash_flash_attention
    assert isinstance(attention, draft.WindowedBlockAttention)
    assert attention.builtin is _flash_attn_with_kvcache
    assert list(drafter.modules()) == modules
    assert {n: p.data_ptr() for n, p in drafter.named_parameters()} == params
    # A second load (the engine may reload draft weights) changes nothing.
    shell.load_draft_weights("weights")
    assert _conv_types(drafter) == [draft.FusedDFlash2BlockConv] * 4
    assert drafter._dflash_flash_attention is attention


def test_the_written_out_block_decode_reaches_the_convolutions_through_their_modules():
    """``drafter_block_conv`` replaces ``DFlash2BlockConv._convolve``; ``windowed_block_decode``
    must call the modules' ``prepare`` / ``finish`` (which call it), never the
    built-in convolution function, or the fused convolution would not apply under the block decode. The
    GPU test checks the same on the published drafter, bit for bit."""
    names = draft.windowed_block_decode.__code__.co_names
    assert {"attention_convs", "mlp_convs", "prepare", "finish"} <= set(names)
    assert "dflash2_grouped_conv" not in names and "_convolve" not in names


# --------------------------------------------------------------------------
# Ring-buddy experts, installed by the shell's load_weights
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "paths, expected",
    [
        ((), None),
        (("preattn_prep", "projection_gathers"), None),
        (("replicated_experts",), (True, 32)),
        (
            (
                "preattn_prep",
                "post_attention_hop",
                "projection_gathers",
                "replicated_experts",
                "swa_paged_decode",
                "dense_gemms",
            ),
            (True, 32),
        ),
    ],
)
def test_load_weights_installs_ring_buddy_after_the_builtin_load(monkeypatch, paths, expected):
    """The built-in load always runs; ``ring_buddy.install`` runs after it only when the
    plan decided at construction has the replicated experts, balanced."""
    calls = []
    fake = types.ModuleType("ring_buddy")
    fake.BLOCK, fake.STATIC_FROM = 32, 28
    fake.install = lambda text_model, mapping, balance=True, static_from=32: (
        calls.append(("install", balance, static_from))
    )
    package = modeling.__name__.rsplit(".", 1)[0]
    monkeypatch.setitem(sys.modules, f"{package}.kernels.replicated_experts.ring_buddy", fake)
    monkeypatch.setattr(
        importlib.import_module(f"{package}.kernels.replicated_experts"),
        "ring_buddy",
        fake,
        raising=False,
    )
    monkeypatch.setattr(
        modeling._weights, "load", lambda model, weights, mapper: calls.append("load") or "loaded"
    )
    # Only the hook is under test: an instance without the constructor's layers.
    model = object.__new__(modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep)
    model.__dict__["_plan"] = _PLAN(*paths)
    model.__dict__["llm"] = types.SimpleNamespace(model=object())
    model.__dict__["model_config"] = types.SimpleNamespace(mapping=object())
    result = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep.load_weights(model, {})
    assert result == "loaded"
    assert calls == (["load"] if expected is None else ["load", ("install", *expected)])


# --------------------------------------------------------------------------
# The persistent NVFP4 expert kernel, a modifier of the ring-buddy install
# --------------------------------------------------------------------------
_TP4 = types.SimpleNamespace(tp_size=4)
_TP8 = types.SimpleNamespace(tp_size=8)


@pytest.mark.parametrize(
    "paths, mapping, expected",
    [
        (("replicated_experts",), _TP8, ("install", True, 32, {})),
        (
            ("replicated_experts", "persistent_experts"),
            _TP8,
            ("install", True, 32, {"persistent": True}),
        ),
        (
            (
                "preattn_prep",
                "drafter_block_conv",
                "drafter_block_decode",
                "draft_head",
                "post_attention_hop",
                "projection_gathers",
                "swa_paged_decode",
                "dense_gemm_prefetch",
                "drafter_fc_shard",
                "fused_accept",
                "drafter_fixed_gemms",
                "replicated_experts",
                "persistent_experts",
            ),
            _TP8,
            ("install", True, 32, {"persistent": True}),
        ),
        # Without replicated_experts: the owner-only window, at tp 4 and tp 8.
        (
            ("persistent_experts",),
            _TP4,
            ("install", False, None, {"persistent": True, "windows": 1}),
        ),
        (
            ("persistent_experts",),
            _TP8,
            ("install", False, None, {"persistent": True, "windows": 1}),
        ),
        (
            ("persistent_experts", "bf16_routed_experts"),
            _TP4,
            ("install", False, None, {"persistent": True, "bf16_unpacked": True, "windows": 1}),
        ),
        (
            ("bf16_routed_experts",),
            _TP4,
            ("install", False, None, {"bf16_unpacked": True, "windows": 1}),
        ),
    ],
)
def test_persistent_experts_reaches_ring_buddy_install_only_when_selected(
    monkeypatch, paths, mapping, expected
):
    """With ``persistent_experts`` the install gets ``persistent=True``; without it the
    call is exactly the replicated experts' one (no ``persistent`` keyword at all).
    Without ``replicated_experts`` the expert paths take the owner-only window
    (``windows=1``, owner placement, nothing replicated)."""
    calls = []
    fake = types.ModuleType("ring_buddy")
    fake.BLOCK, fake.STATIC_FROM = 32, 28

    def install(text_model, mapping, balance=True, static_from=None, **kwargs):
        assert set(kwargs) <= {"persistent", "bf16_unpacked", "windows"}
        calls.append(("install", balance, static_from, kwargs))

    fake.install = install
    package = modeling.__name__.rsplit(".", 1)[0]
    monkeypatch.setitem(sys.modules, f"{package}.kernels.replicated_experts.ring_buddy", fake)
    monkeypatch.setattr(
        importlib.import_module(f"{package}.kernels.replicated_experts"),
        "ring_buddy",
        fake,
        raising=False,
    )
    target_weights.install_loaded_features(object(), mapping, _PLAN(*paths))
    assert calls == [expected]


def test_owner_window_refuses_other_tp_sizes(monkeypatch):
    """The owner-only window is written for tp 4 and tp 8 only."""
    with pytest.raises(ValueError, match="tp 4 and tp 8, not tp 2"):
        target_weights.install_loaded_features(
            object(), types.SimpleNamespace(tp_size=2), _PLAN("persistent_experts")
        )


class _Fork:
    """Records what ``DecodeTarget`` hands the context-K/V fork."""

    def __init__(self, layer):
        self.layer, self.events = layer, []

    def arm(self, spec_metadata, attn_metadata):
        self.events.append(("arm", spec_metadata, attn_metadata.num_contexts))

    def fire(self, point):
        self.events.append(("fire", point))


def test_decode_target_arms_the_fork_and_fires_it_after_the_tapped_layer(no_features):
    model = _builtin_text_model()
    for module in model.layers:
        module.layer_idx = module.index
    core = _CORE.adopt(model)
    fork = _Fork(None)
    core._context_kv_fork = fork
    calls = core.calls

    def recording_fire(point):
        fork.events.append(("fire", point, [index for index, _ in calls]))

    fork.fire = recording_fire
    ids, positions = torch.tensor([3, 5], dtype=torch.int32), torch.tensor([7, 8])
    core(_Metadata(0), ids, positions, spec_metadata="spec")
    assert fork.events == [("arm", "spec", 0)]


@pytest.mark.parametrize(
    "paths",
    [
        ("context_kv_fork",),
        ("drafter_fc_shard", "fused_accept", "drafter_fixed_gemms", "context_kv_fork"),
        ("draft_head", "fused_accept", "context_kv_fork"),
    ],
)
def test_context_kv_fork_installs_after_the_other_drafter_features(monkeypatch, paths):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
        fork as context_kv_fork,
    )

    calls = []
    monkeypatch.setattr(draft, "install", lambda model, conv, decode: calls.append("draft"))
    monkeypatch.setattr(draft_head, "install", lambda llm: calls.append("draft_head"))
    monkeypatch.setattr(
        drafter_worker,
        "install",
        lambda model, worker, mapping, *switches: calls.append("drafter_worker"),
    )
    monkeypatch.setattr(
        context_kv_fork, "install", lambda llm, core: calls.append(("context_kv_fork", llm, core))
    )
    shell = object.__new__(_SHELL)
    llm = types.SimpleNamespace(model="core", draft_model="drafter", spec_worker="worker")
    shell.__dict__.update(_plan=_PLAN(*paths), llm=llm, _mapping="mapping")
    _SHELL._install_draft_features(shell)
    expected = ["draft_head"] if "draft_head" in paths else []
    expected += (
        ["drafter_worker"]
        if {"drafter_fc_shard", "fused_accept", "drafter_fixed_gemms"} & set(paths)
        else []
    )
    expected.append(("context_kv_fork", llm, "core"))
    assert calls == expected
