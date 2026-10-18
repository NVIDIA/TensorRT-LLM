# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark Open tep8 core and its two targets, and what the shell keeps from the built-in model.

Host-only: the core is adopted on a built-in text model created without its
constructor (which builds CUDA streams, communication workspaces and the decoder
layers), with small CPU stand-ins for the embedding, the norms and the layers.

The decode target writes the built-in forward's decode path out over the
built-in modules, so the tests here run that path and the built-in forward on
the same stand-ins and require the same layer calls and the same output. A
change to the built-in forward's decode path fails here until the target
follows it.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
_WIDTH = 4


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = input_ids.to(torch.float32).unsqueeze(-1)
        return (rows + torch.arange(_WIDTH, dtype=torch.float32)).to(torch.bfloat16)


class _Layer(nn.Module):
    """Records what it is called with and hands the next layer an attention input."""

    def __init__(self, index: int, calls: list) -> None:
        super().__init__()
        self.index = index
        self.pre_qkv_proj_norm = _Scale(1.0 + 0.25 * index)
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
        self.calls.append(
            dict(
                index=self.index,
                position_ids=position_ids,
                hidden_states=hidden_states.clone(),
                attn_metadata=attn_metadata,
                precomputed_attn_input=None
                if precomputed_attn_input is None
                else precomputed_attn_input.clone(),
                next_layer_norm=next_layer_norm,
                sp_num_tokens=sp_num_tokens,
                kwargs=dict(kwargs),
            )
        )
        out = hidden_states * 0.5 + (self.index + 1)
        if next_layer_norm is None:
            return out
        return out, next_layer_norm(out.to(torch.bfloat16))


class _Metadata:
    def __init__(self, num_contexts: int) -> None:
        self.num_contexts = num_contexts


def _builtin_text_model(
    num_layers: int = 3, m_emb: float = 1.0, prefill_sp: bool = False
) -> MuseSparkOpenModel:
    """A built-in text model as far as class and forward state go, with CPU stand-in modules."""
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    calls: list = []
    model.embed_tokens = _Embedding()
    model.embed_norm = _Scale(2.0)
    model.norm = _Scale(3.0)
    model.layers = nn.ModuleList([_Layer(i, calls) for i in range(num_layers)])
    model.dtype = torch.bfloat16
    model.m_emb = m_emb
    model._prefill_sp_enabled = prefill_sp
    model._sp_mapping = None
    model.calls = calls
    return model


def _assert_same_layer_calls(model_a, model_b) -> None:
    assert [c["index"] for c in model_a.calls] == [c["index"] for c in model_b.calls]
    for a, b in zip(model_a.calls, model_b.calls):
        assert a["position_ids"] is b["position_ids"]
        assert a["attn_metadata"] is b["attn_metadata"]
        assert a["hidden_states"].dtype == b["hidden_states"].dtype == torch.float32
        assert torch.equal(a["hidden_states"], b["hidden_states"])
        if a["precomputed_attn_input"] is None:
            assert b["precomputed_attn_input"] is None
        else:
            assert torch.equal(a["precomputed_attn_input"], b["precomputed_attn_input"])
        # Each model hands every layer its own next layer's pre-QKV norm.
        index = a["index"]
        for model, call in ((model_a, a), (model_b, b)):
            expected = (
                model.layers[index + 1].pre_qkv_proj_norm if index + 1 < len(model.layers) else None
            )
            assert call["next_layer_norm"] is expected
        assert a["sp_num_tokens"] is None and b["sp_num_tokens"] is None
        assert a["kwargs"] == b["kwargs"]


def test_adopt_reclasses_the_builtin_text_model_in_place():
    model = _builtin_text_model()
    modules_before = [name for name, _ in model.named_modules()]
    core = _CORE.adopt(model)
    assert core is model
    assert type(core) is _CORE
    assert isinstance(core, MuseSparkOpenModel)
    assert isinstance(core._prefill, modeling.PrefillTarget) and core._prefill.core is core
    assert isinstance(core._decode, modeling.DecodeTarget) and core._decode.core is core
    # Plain attributes only: no module or parameter joins the built-in tree.
    assert [name for name, _ in core.named_modules()] == modules_before
    assert "_prefill" not in dict(core.named_modules())


def test_adopt_refuses_anything_but_the_builtin_text_model():
    core = _CORE.adopt(_builtin_text_model())
    with pytest.raises(TypeError, match="MuseSparkOpenTepModelingV2"):
        _CORE.adopt(core)
    with pytest.raises(TypeError, match="Module"):
        _CORE.adopt(nn.Module())


@pytest.mark.parametrize(
    "num_contexts, target",
    [(0, modeling.DecodeTarget), (1, modeling.PrefillTarget), (4, modeling.PrefillTarget)],
    ids=["decode", "prefill", "mixed"],
)
def test_each_step_selects_its_target(num_contexts, target):
    core = _CORE.adopt(_builtin_text_model())
    assert type(core._select_target(_Metadata(num_contexts))) is target


def test_prefill_runs_the_builtin_forward_and_decode_does_not(monkeypatch):
    calls = []

    def builtin_forward(
        self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs
    ):
        calls.append((self, attn_metadata, input_ids, position_ids, inputs_embeds, kwargs))
        return "hidden states"

    monkeypatch.setattr(MuseSparkOpenModel, "forward", builtin_forward)
    core = _CORE.adopt(_builtin_text_model())

    # Called the way the speculative causal LM calls its text model: keywords throughout.
    prefill = _Metadata(2)
    out = core(
        input_ids="ids",
        attn_metadata=prefill,
        position_ids="positions",
        inputs_embeds=None,
        spec_metadata="spec",
        lora_params="lora",
    )
    assert out == "hidden states"
    assert calls == [
        (core, prefill, "ids", "positions", None, {"spec_metadata": "spec", "lora_params": "lora"})
    ]

    calls.clear()
    ids = torch.tensor([3, 5], dtype=torch.int32)
    core(_Metadata(0), ids, torch.tensor([7, 8]))
    assert calls == []
    assert [c["index"] for c in core.calls] == [0, 1, 2]


@pytest.mark.parametrize("num_tokens", [1, 8, 4096], ids=lambda n: f"rows{n}")
@pytest.mark.parametrize("m_emb", [1.0, 0.5], ids=["m_emb1", "m_emb0.5"])
@pytest.mark.parametrize("prefill_sp", [False, True], ids=["sp_off", "sp_on"])
@pytest.mark.parametrize("embeds", [False, True], ids=["input_ids", "inputs_embeds"])
def test_decode_matches_the_builtin_forward(num_tokens, m_emb, prefill_sp, embeds):
    """Same layer calls, same output, as the built-in forward on a step with no context rows.

    The prefill sequence-parallel switch is on in half the cases: the built-in
    forward takes that split only with context rows, so a decode step never does.
    """
    builtin = _builtin_text_model(m_emb=m_emb, prefill_sp=prefill_sp)
    core = _CORE.adopt(_builtin_text_model(m_emb=m_emb, prefill_sp=prefill_sp))
    md = _Metadata(0)
    ids = torch.arange(num_tokens, dtype=torch.int32) % 97
    positions = torch.arange(num_tokens, dtype=torch.int32)
    inputs = (
        dict(inputs_embeds=_Embedding()(ids).to(torch.float32)) if embeds else dict(input_ids=ids)
    )
    extra = dict(spec_metadata=object(), lora_params=None)

    expected = MuseSparkOpenModel.forward(builtin, md, position_ids=positions, **inputs, **extra)
    got = core(attn_metadata=md, position_ids=positions, **inputs, **extra)

    assert got.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(got, expected)
    assert len(core.calls) == 3
    _assert_same_layer_calls(builtin, core)


@pytest.mark.parametrize(
    "inputs",
    [{}, {"input_ids": torch.zeros(1, dtype=torch.int32), "inputs_embeds": torch.zeros(1, 4)}],
    ids=["neither", "both"],
)
def test_decode_requires_exactly_one_input(inputs):
    core = _CORE.adopt(_builtin_text_model())
    with pytest.raises(ValueError, match="exactly one of input_ids and inputs_embeds"):
        core(_Metadata(0), **inputs)


def test_the_forwards_take_the_builtin_parameters():
    builtin = inspect.signature(MuseSparkOpenModel.forward)
    for forward in (_CORE.forward, modeling.PrefillTarget.forward, modeling.DecodeTarget.forward):
        assert list(inspect.signature(forward).parameters) == list(builtin.parameters), forward


def test_the_shell_is_the_builtin_omni_model():
    assert issubclass(_SHELL, MuseSparkOpenOmniForConditionalGeneration)
    # The model loader inspects load_weights' parameters to decide what it passes.
    assert (
        inspect.getfullargspec(_SHELL.load_weights).args
        == inspect.getfullargspec(MuseSparkOpenOmniForConditionalGeneration.load_weights).args
    )
    # The engine reads these off the class; the shell answers as the built-in class does.
    for hook in (
        "get_preferred_kv_cache_manager_version",
        "requires_kv_cache_manager_v2",
        "get_preferred_transceiver_runtime",
    ):
        assert getattr(_SHELL, hook)(None) == getattr(
            MuseSparkOpenOmniForConditionalGeneration, hook
        )(None), hook


class _Llm(nn.Module):
    def __init__(self, model: MuseSparkOpenModel) -> None:
        super().__init__()
        self.model = model


@pytest.fixture
def shell(monkeypatch):
    """A shell built with the built-in constructor stubbed, re-entry guard included."""
    events = []

    def builtin_init(self, model_config, *args, **kwargs):
        events.append("built-in model")
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = _Llm(_builtin_text_model())

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    shell = _SHELL(model_config=None)
    shell.events = events
    return shell


def test_the_shell_adopts_the_text_model_once(shell):
    assert shell.events == ["built-in model"]
    core = shell.llm.model
    assert type(core) is _CORE
    decode = core._decode

    # The built-in constructor returns early when re-entered; so does the adoption.
    _SHELL.__init__(shell, None)
    assert shell.events == ["built-in model", "built-in model"]
    assert shell.llm.model is core and core._decode is decode


def test_the_shell_loads_through_the_builtin_loader(monkeypatch, shell):
    seen = []

    def builtin_load_weights(self, weights, weight_mapper=None):
        seen.append((self, weights, weight_mapper))
        return {"loaded": 1}

    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration, "load_weights", builtin_load_weights
    )
    assert shell.load_weights("weights", "mapper") == {"loaded": 1}
    assert shell.load_weights("weights") == {"loaded": 1}
    assert seen == [(shell, "weights", "mapper"), (shell, "weights", None)]
