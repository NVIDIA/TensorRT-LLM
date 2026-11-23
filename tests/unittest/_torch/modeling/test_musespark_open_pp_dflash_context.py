# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A DFlash drafter on the pipeline stages of a MuseSparkOpen context server, on CPU.

What the built-in decoder does inside ``pp_dflash_context_server()`` (the
marker the pipelined context-server target sets): admit a DFlash
``speculative_config`` with ``attention_backend`` TRTLLM at
``pipeline_parallel_size > 1``, drop the drafter on every stage but the last
before its weights exist, and skip its checkpoint there. Without the marker
every refusal is the one the decoder gives on any other server;
``_refusal_off_the_context_server`` below is a verbatim copy of that refusal
logic, kept as the reference.

Runs with or without a built package (``_musespark_open_cpu_import``). The
drafters here are module trees of meta tensors in the released drafter's shapes,
so the 8.6 GiB of weights cost nothing.
"""

import inspect
import os
import threading
from types import SimpleNamespace

import pytest
import torch
from _musespark_open_cpu_import import load
from torch import nn

B, MU, Mapping = load()

pytestmark = pytest.mark.cpu_only

OLD_SPEC_REFUSAL = (
    "speculative decoding is not supported; serve the stages without a speculative_config"
)
PARTITION = [15, 16, 16, 15]
NUM_LAYERS = 62

# Released drafter geometry: 5 layers, hidden 8192, 64 query heads and 8 KV heads
# of width 128, MLP 28672, fc over 5 captures; DFlash 2 convolutions of 2 taps
# over groups of 16 channels. The selector rank is not pinned by any test here.
HIDDEN, HEAD_DIM, Q_HEADS, KV_HEADS, INTER = 8192, 128, 64, 8, 28672
DRAFT_LAYERS, CAPTURES, TAPS, GROUP, VOCAB, RANK = 5, 5, 2, 16, 202048, 256


def _refusal_off_the_context_server(model_config):
    """Reference copy of ``pipeline_parallel_refusal`` off the pipelined context server."""
    mapping = model_config.mapping
    if mapping.pp_size == 1:
        return None
    if getattr(model_config, "spec_config", None) is not None:
        return (
            "speculative decoding is not supported; serve the stages without a speculative_config"
        )
    if mapping.cp_size != 1:
        return f"context parallelism (cp_size={mapping.cp_size}) is not supported"
    if mapping.enable_attention_dp:
        return "attention data parallelism is not supported"
    num_layers = model_config.pretrained_config.num_hidden_layers
    if num_layers < mapping.pp_size:
        return f"{num_layers} decoder layers cannot fill {mapping.pp_size} stages"
    return None


# --------------------------------------------------------------------------
# Speculative configs
# --------------------------------------------------------------------------
class _Mode:
    """The ``SpeculativeDecodingMode`` predicates the decoder reads."""

    def __init__(self, name):
        self.name = name

    def is_dflash(self):
        return self.name == "DFLASH"

    def is_sa(self):
        return self.name == "SA"

    def use_one_engine(self):
        return True


def _spec(kind):
    """A speculative_config as the decoder reads it: ``spec_dec_mode`` and ``attention_backend``."""
    if kind is None:
        return None
    if kind == "opaque":
        return object()
    mode, backend = kind.split("-")
    return SimpleNamespace(spec_dec_mode=_Mode(mode), attention_backend=backend)


SPEC_KINDS = [
    None,
    "DFLASH-TRTLLM",
    "DFLASH-VANILLA",
    "DFLASH-FA4",
    "DFLASH-AUTO",
    "DSPARK-TRTLLM",
    "SA-TRTLLM",
    "opaque",
]


def _refusal_config(
    pp_size=4, cp_size=1, attention_dp=False, spec=None, layers=NUM_LAYERS, tp_size=1
):
    mapping = SimpleNamespace(
        pp_size=pp_size, cp_size=cp_size, enable_attention_dp=attention_dp, tp_size=tp_size
    )
    return SimpleNamespace(
        mapping=mapping,
        spec_config=spec,
        pretrained_config=SimpleNamespace(num_hidden_layers=layers),
    )


@pytest.fixture(autouse=True)
def _no_recorded_server():
    """The decoder records whether it built a pipelined DFlash context server
    (``set_pp_dflash_context_server``); no test sees another test's record."""
    B.set_pp_dflash_context_server(False)
    yield
    B.set_pp_dflash_context_server(False)


# --------------------------------------------------------------------------
# The marker
# --------------------------------------------------------------------------
def test_the_marker_is_scoped_to_the_with_block_and_nests():
    assert not B.in_pp_dflash_context_server()
    with B.pp_dflash_context_server():
        assert B.in_pp_dflash_context_server()
        with B.pp_dflash_context_server():
            assert B.in_pp_dflash_context_server()
        assert B.in_pp_dflash_context_server()
    assert not B.in_pp_dflash_context_server()


def test_the_marker_is_cleared_when_the_construction_raises():
    with pytest.raises(RuntimeError, match="boom"):
        with B.pp_dflash_context_server():
            raise RuntimeError("boom")
    assert not B.in_pp_dflash_context_server()


def test_the_marker_does_not_leak_to_another_thread():
    seen = []
    with B.pp_dflash_context_server():
        thread = threading.Thread(target=lambda: seen.append(B.in_pp_dflash_context_server()))
        thread.start()
        thread.join()
    assert seen == [False]


# --------------------------------------------------------------------------
# Refusal table
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pp_size", [1, 4])
@pytest.mark.parametrize("marker", [False, True], ids=["no-marker", "marker"])
@pytest.mark.parametrize("kind", SPEC_KINDS, ids=str)
def test_refusal_table(pp_size, marker, kind):
    config = _refusal_config(pp_size=pp_size, spec=_spec(kind))
    if marker:
        with B.pp_dflash_context_server():
            refusal = B.pipeline_parallel_refusal(config)
            admitted = B.pipeline_parallel_dflash_context(config)
    else:
        refusal = B.pipeline_parallel_refusal(config)
        admitted = B.pipeline_parallel_dflash_context(config)
    expected_admitted = pp_size > 1 and marker and kind == "DFLASH-TRTLLM"
    assert admitted == expected_admitted
    if pp_size == 1 or kind is None or expected_admitted:
        assert refusal is None
    elif marker:
        assert refusal == (
            "a pipelined context server serves speculative decoding only with DFlash and "
            "attention_backend TRTLLM"
        )
    else:
        assert refusal == OLD_SPEC_REFUSAL


@pytest.mark.parametrize("kind", SPEC_KINDS, ids=str)
@pytest.mark.parametrize(
    "deployment",
    [
        dict(pp_size=1),
        dict(),
        dict(cp_size=2),
        dict(attention_dp=True),
        dict(layers=3),
        dict(pp_size=2, cp_size=2, attention_dp=True),
        dict(tp_size=2),
    ],
    ids=["pp1", "pp4", "cp2", "adp", "few-layers", "pp2-cp2-adp", "pp4-tp2"],
)
def test_without_the_marker_every_refusal_is_the_old_one(kind, deployment):
    """Without the marker: the reference refusal, string for string."""
    config = _refusal_config(spec=_spec(kind), **deployment)
    assert B.pipeline_parallel_refusal(config) == _refusal_off_the_context_server(config)


@pytest.mark.parametrize(
    "deployment, expected",
    [
        (dict(cp_size=2), "context parallelism (cp_size=2) is not supported"),
        (dict(attention_dp=True), "attention data parallelism is not supported"),
        (dict(layers=3), "3 decoder layers cannot fill 4 stages"),
        (
            dict(tp_size=2),
            "a pipelined DFlash context server runs one GPU per stage; "
            "tensor_parallel_size 2 is not supported",
        ),
    ],
    ids=["cp2", "adp", "few-layers", "tp2"],
)
def test_an_admitted_drafter_keeps_the_other_refusals(deployment, expected):
    config = _refusal_config(spec=_spec("DFLASH-TRTLLM"), **deployment)
    with B.pp_dflash_context_server():
        assert B.pipeline_parallel_refusal(config) == expected


# --------------------------------------------------------------------------
# Construction: the real constructor and stage hooks over light stand-ins
# --------------------------------------------------------------------------
def _meta(*shape):
    return nn.Parameter(torch.empty(*shape, dtype=torch.bfloat16, device="meta"), False)


class _Linear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = _meta(out_features, in_features)


class _Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = _meta(width)


class _Backend(B.AttentionBackend):
    """An attention backend object as an attention module holds it: not a module."""

    def __init__(self):
        self.rotary_cos_sin = torch.zeros(4, 8)
        self.layer_idx = 0


class _Rope(nn.Module):
    """A RoPE table kept as a plain attribute, as ``RotaryEmbedding`` keeps it."""

    def __init__(self):
        super().__init__()
        self.rotary_cos_sin = torch.ones(4, 2, 8)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _Linear(Q_HEADS * HEAD_DIM, HIDDEN)
        self.k_proj = _Linear(KV_HEADS * HEAD_DIM, HIDDEN)
        self.v_proj = _Linear(KV_HEADS * HEAD_DIM, HIDDEN)
        self.o_proj = _Linear(HIDDEN, Q_HEADS * HEAD_DIM)
        self.q_norm = _Norm(HEAD_DIM)
        self.k_norm = _Norm(HEAD_DIM)
        self.rotary_emb = _Rope()
        self.attn = _Backend()


class _DraftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.input_layernorm = _Norm(HIDDEN)
        self.post_attention_layernorm = _Norm(HIDDEN)
        self.mlp = nn.Module()
        self.mlp.gate_proj = _Linear(INTER, HIDDEN)
        self.mlp.up_proj = _Linear(INTER, HIDDEN)
        self.mlp.down_proj = _Linear(HIDDEN, INTER)
        self.attention_conv = nn.Module()
        self.attention_conv.base_kernel = _meta(2, TAPS, HIDDEN)
        self.attention_conv.kernel_projection = _Linear(2 * TAPS * (HIDDEN // GROUP), HIDDEN)


class _Drafter(nn.Module):
    """The released DFlash 2 drafter's module tree in meta tensors.

    ``load_weights`` and ``load_weights_from_target_model`` record their calls:
    a stage that dropped the drafter must reach neither.
    """

    def __init__(self, shared=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_DraftLayer() for _ in range(DRAFT_LAYERS))
        self.model.norm = _Norm(HIDDEN)
        self.fc = _Linear(HIDDEN, HIDDEN * CAPTURES)
        self.hidden_norm = _Norm(HIDDEN)
        self._rotary_cos_sin = torch.ones(4, 2, 8)
        if shared is not None:
            self.shared = shared
        self.calls = []

    def forward(self, *args, **kwargs):
        return "drafted"

    def load_weights(self, weights, weight_mapper=None):
        self.calls.append("load_weights")

    def load_weights_from_target_model(self, target_model):
        self.calls.append("load_weights_from_target_model")


def _num_params(module):
    """Parameters a module owns; a drafter's reference to a target module (``shared``) is not its own."""
    return sum(
        param.numel() for name, param in module.named_parameters() if not name.startswith("shared.")
    )


class _Skippable(nn.Module):
    """LM head / speculative worker stand-in with the ``skip_forward`` the generic hook uses."""

    def __init__(self, *shape):
        super().__init__()
        self.weight = _meta(*shape) if shape else None

    def skip_forward(self, *args, **kwargs):
        return None


class _StageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = _Skippable(VOCAB, HIDDEN)
        self.pp_init_calls = 0

    def __pp_init__(self):
        self.pp_init_calls += 1


def _mapping(rank, pp_size=4):
    return Mapping(
        world_size=pp_size,
        rank=rank,
        gpus_per_node=pp_size,
        pp_size=pp_size,
        pp_partition=PARTITION if pp_size == 4 else None,
    )


def _decoder_config(rank, kind="DFLASH-TRTLLM", pp_size=4):
    return SimpleNamespace(
        quant_config=None,
        pretrained_config=SimpleNamespace(
            num_hidden_layers=NUM_LAYERS,
            hidden_size=HIDDEN,
            vocab_size=VOCAB,
            metap_multiplier=1.0,
            output_soft_cap_temp=None,
        ),
        mapping=_mapping(rank, pp_size),
        spec_config=_spec(kind),
    )


@pytest.fixture
def light_decoder(monkeypatch):
    """``MuseSparkOpenForCausalLM(config)`` with the real constructor and stage hooks.

    The decoder, LM head, drafter and worker are the stand-ins above, attached
    in the order ``SpecDecOneEngineForCausalLM`` attaches them; the quantization
    pass of ``__post_init__`` is skipped. Returns ``build(config)`` and the
    generic hook's warnings.
    """
    warnings = []
    built = {}

    def light_init(self, model, model_config, hidden_size=None, vocab_size=None):
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = model
        self.lm_head = _Skippable(VOCAB, HIDDEN)
        self.draft_model = _Drafter(shared=model.embed_tokens) if model_config.spec_config else None
        self.spec_worker = _Skippable() if model_config.spec_config else None
        self.spec_config = model_config.spec_config
        self.prologue = []
        self.epilogue = [self.lm_head]
        if self.draft_model is not None:
            self.epilogue.append(self.draft_model)
        if self.spec_worker is not None:
            self.epilogue.append(self.spec_worker)
        built["drafter_params"] = _num_params(self.draft_model) if self.draft_model else 0
        # SpecDecOneEngineForCausalLM builds an external drafter's config with this mapping.
        built["drafter_mapping"] = (
            self.external_drafter_mapping(model_config) if model_config.spec_config else None
        )

    spec_engine = B.SpecDecOneEngineForCausalLM
    monkeypatch.setattr(spec_engine, "__init__", light_init)
    monkeypatch.setattr(B, "MuseSparkOpenModel", lambda config: _StageModel())
    monkeypatch.setattr(B, "MuseSparkOpenLogitsProcessor", lambda **kwargs: None)
    monkeypatch.setattr(B.MuseSparkOpenForCausalLM, "__post_init__", lambda self: None)
    monkeypatch.setattr(MU.logger, "warning", lambda message, *a, **k: warnings.append(message))

    def build(config):
        return B.MuseSparkOpenForCausalLM(config)

    return SimpleNamespace(build=build, warnings=warnings, built=built)


def test_the_released_drafter_holds_8_6_gib_of_weights():
    """The stand-in drafter is the released one by parameter count.

    Attention, MLP and fc alone are 4.61 B BF16 parameters (8.59 GiB); the norms
    and the DFlash 2 convolutions add the rest.
    """
    gib = _num_params(_Drafter()) * 2 / 2**30
    assert 8.5 < gib < 8.9, gib


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_a_context_server_keeps_the_drafter_on_the_last_stage_only(
    monkeypatch, light_decoder, rank
):
    with B.pp_dflash_context_server():
        decoder = light_decoder.build(_decoder_config(rank))
    drafter = decoder.draft_model
    assert decoder._pp_dflash_context
    assert decoder.model.pp_init_calls == 1
    assert light_decoder.built["drafter_params"] == _num_params(_Drafter())
    # The target embedding the drafter also references is the stage's own: it is
    # kept on stage 0 and dropped by the stage hook elsewhere, never by the drafter drop.
    assert drafter.shared is decoder.model.embed_tokens
    if rank == 3:
        assert not decoder._pp_stage_drafter_removed
        assert _num_params(drafter) == _num_params(_Drafter())
        assert drafter in decoder.epilogue
        assert drafter(None) == "drafted"
        assert "weight" in decoder.lm_head._parameters
        assert drafter.model.layers[0].self_attn.rotary_emb.rotary_cos_sin.device.type == "cpu"
        return
    assert decoder._pp_stage_drafter_removed
    assert _num_params(drafter) == 0
    assert not list(drafter.buffers())
    assert all(
        getattr(module, "_weights_removed", False)
        for name, module in drafter.named_modules()
        if not name.startswith("shared")
    )
    assert drafter not in decoder.epilogue
    # Plain-attribute tensors are swapped for meta tensors of the same shape.
    for owner, name, shape in [
        (drafter, "_rotary_cos_sin", (4, 2, 8)),
        (drafter.model.layers[2].self_attn.rotary_emb, "rotary_cos_sin", (4, 2, 8)),
        (drafter.model.layers[4].self_attn.attn, "rotary_cos_sin", (4, 8)),
    ]:
        value = getattr(owner, name)
        assert value.device.type == "meta" and tuple(value.shape) == shape
    with pytest.raises(RuntimeError, match=f"pipeline stage {rank} holds no drafter"):
        drafter(None)
    # The generic hook still drops the LM head and the worker here, and has no
    # drafter left to warn about.
    assert "weight" not in decoder.lm_head._parameters
    assert decoder.spec_worker.forward == decoder.spec_worker.skip_forward
    assert not [w for w in light_decoder.warnings if "Fail to skip forward" in w]
    # The stand-in stage model keeps its embedding, so only the drafter drop could
    # have removed it, and it must not have.
    assert "weight" in decoder.model.embed_tokens._parameters


class _DraftBackbone(MU.DecoderModel):
    """The drafter's backbone as the generic stage hook sees it: ``DRAFT_LAYERS`` layers."""

    def __init__(self, mapping):
        config = SimpleNamespace(
            pretrained_config=SimpleNamespace(
                torch_dtype=torch.bfloat16, num_hidden_layers=DRAFT_LAYERS
            ),
            mapping=mapping,
            spec_config=None,
        )
        super().__init__(config)
        self.layers = nn.ModuleList(_Skippable(4, 4) for _ in range(DRAFT_LAYERS))


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_a_context_server_builds_the_drafter_without_pipeline_stages(
    monkeypatch, light_decoder, rank
):
    """The drafter's own generic stage hook sees one stage, so all of its layers stay.

    With the stage's pipeline mapping that hook splits the drafter's layers by the
    target's ``pp_partition``, which cannot hold them, and raises.
    """
    with B.pp_dflash_context_server():
        decoder = light_decoder.build(_decoder_config(rank))
    stage_mapping = decoder.model_config.mapping
    mapping = light_decoder.built["drafter_mapping"]
    assert mapping is not stage_mapping
    assert (mapping.world_size, mapping.rank, mapping.tp_size, mapping.pp_size) == (1, 0, 1, 1)
    assert mapping.pp_partition is None and not mapping.has_pp()
    assert mapping.gpus_per_node == stage_mapping.gpus_per_node

    backbone = _DraftBackbone(mapping)
    backbone.__pp_init__()
    assert all("weight" in layer._parameters for layer in backbone.layers)
    assert all(layer.forward != layer.skip_forward for layer in backbone.layers)
    with pytest.raises(ValueError, match="does not match num_layers=5"):
        _DraftBackbone(stage_mapping).__pp_init__()


@pytest.mark.parametrize(
    "marker, rank, pp_size",
    [(False, 0, 1), (True, 0, 1), (False, 0, 4)],
    ids=["pp1", "pp1-marker", "pp4-no-spec"],
)
def test_every_other_decoder_builds_the_drafter_with_its_own_mapping(
    light_decoder, marker, rank, pp_size
):
    kind = None if pp_size == 4 else "DFLASH-TRTLLM"
    config = _decoder_config(rank, kind, pp_size)
    if marker:
        with B.pp_dflash_context_server():
            decoder = light_decoder.build(config)
    else:
        decoder = light_decoder.build(config)
    if kind is not None:
        assert light_decoder.built["drafter_mapping"] is config.mapping
    with B.pp_dflash_context_server():
        assert decoder.external_drafter_mapping(config) is config.mapping


@pytest.mark.parametrize("rank", [0, 3])
def test_without_the_marker_the_constructor_refuses_as_before(light_decoder, rank):
    with pytest.raises(ValueError) as error:
        light_decoder.build(_decoder_config(rank))
    assert str(error.value) == f"MuseSparkOpen pipeline parallelism: {OLD_SPEC_REFUSAL}"


@pytest.mark.parametrize("kind", ["DFLASH-VANILLA", "DFLASH-AUTO", "DSPARK-TRTLLM", "SA-TRTLLM"])
def test_a_context_server_refuses_every_other_speculative_config(monkeypatch, light_decoder, kind):
    with B.pp_dflash_context_server():
        with pytest.raises(ValueError, match="only with DFlash and attention_backend TRTLLM"):
            light_decoder.build(_decoder_config(1, kind))


def test_one_stage_with_the_marker_is_the_old_decoder(light_decoder):
    """pp 1 is never a pipelined context server: the drafter stays and the generic hook is a no-op."""
    with B.pp_dflash_context_server():
        decoder = light_decoder.build(_decoder_config(0, pp_size=1))
    assert not decoder._pp_dflash_context and not decoder._pp_stage_drafter_removed
    assert _num_params(decoder.draft_model) == _num_params(_Drafter())


@pytest.mark.parametrize("rank", [0, 3])
def test_the_server_is_decided_at_construction(light_decoder, rank):
    """A stage built as a context server stays one after the marker is gone, and the
    construction records it for the KV cache and executor setup."""
    with B.pp_dflash_context_server():
        decoder = light_decoder.build(_decoder_config(rank))
    assert B.set_pp_dflash_context_server.__globals__["pp_dflash_context_enabled"]() is True
    assert decoder._pp_dflash_context
    assert decoder._pp_stage_drafter_removed == (rank != 3)


# --------------------------------------------------------------------------
# Drafter checkpoint load
# --------------------------------------------------------------------------
def _released_drafter_checkpoint():
    """Key -> meta tensor of every tensor a released DFlash 2 drafter checkpoint ships."""
    shapes = {
        "fc.weight": (HIDDEN, HIDDEN * CAPTURES),
        "hidden_norm.weight": (HIDDEN,),
        "norm.weight": (HIDDEN,),
        "candidate_selector.predecessor_codebook": (VOCAB, RANK),
        "candidate_selector.successor_codebook": (VOCAB, RANK),
        "candidate_selector.hidden_projection.weight": (RANK, HIDDEN),
    }
    for layer in range(DRAFT_LAYERS):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "self_attn.q_proj.weight": (Q_HEADS * HEAD_DIM, HIDDEN),
                prefix + "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
                prefix + "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
                prefix + "self_attn.o_proj.weight": (HIDDEN, Q_HEADS * HEAD_DIM),
                prefix + "self_attn.q_norm.weight": (HEAD_DIM,),
                prefix + "self_attn.k_norm.weight": (HEAD_DIM,),
                prefix + "input_layernorm.weight": (HIDDEN,),
                prefix + "post_attention_layernorm.weight": (HIDDEN,),
                prefix + "mlp.gate_proj.weight": (INTER, HIDDEN),
                prefix + "mlp.up_proj.weight": (INTER, HIDDEN),
                prefix + "mlp.down_proj.weight": (HIDDEN, INTER),
            }
        )
        for conv in ("attention_conv", "mlp_conv"):
            shapes[prefix + f"{conv}.base_kernel"] = (2, TAPS, HIDDEN)
            shapes[prefix + f"{conv}.kernel_projection.weight"] = (
                2 * TAPS * (HIDDEN // GROUP),
                HIDDEN,
            )
    return {
        key: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        for key, shape in shapes.items()
    }


class _UntouchableCheckpoint(dict):
    """A checkpoint dict whose tensors must not be read: any value access raises."""

    def __getitem__(self, key):
        raise AssertionError(f"read {key}")

    def get(self, key, default=None):
        raise AssertionError(f"read {key}")

    def values(self):
        raise AssertionError("read the values")

    def items(self):
        raise AssertionError("read the items")


def _call_load_weights(load_method, weights, weight_mapper):
    """The model loader's ``_call_load_weights``: keyword arguments by the method's signature."""
    args = inspect.getfullargspec(load_method).args
    kwargs = {"weight_mapper": weight_mapper} if "weight_mapper" in args else {}
    load_method(weights, **kwargs)


@pytest.fixture
def base_draft_load(monkeypatch):
    calls = []

    def load_draft_weights(self, weights, weight_mapper=None):
        calls.append((self, weights, weight_mapper))
        self.draft_model.load_weights(weights, weight_mapper=weight_mapper)
        self.draft_model.load_weights_from_target_model(self)

    monkeypatch.setattr(B.SpecDecOneEngineForCausalLM, "load_draft_weights", load_draft_weights)
    return calls


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_a_stage_without_the_drafter_skips_the_released_checkpoint(
    monkeypatch, light_decoder, base_draft_load, rank
):
    with B.pp_dflash_context_server():
        decoder = light_decoder.build(_decoder_config(rank))
    checkpoint = _released_drafter_checkpoint()
    mapper = object()
    _call_load_weights(decoder.load_draft_weights, checkpoint, mapper)
    _call_load_weights(decoder.load_draft_weights, _UntouchableCheckpoint(checkpoint), mapper)
    assert base_draft_load == []
    assert decoder.draft_model.calls == []
    assert _num_params(decoder.draft_model) == 0
    assert not hasattr(decoder.draft_model, "attention_convs")


@pytest.mark.parametrize(
    "marker, rank, pp_size",
    [(False, 0, 1), (True, 0, 1), (True, 3, 4), (False, 0, 4)],
    ids=["pp1", "pp1-marker", "last-stage", "pp4-no-spec"],
)
def test_every_other_decoder_loads_the_drafter_as_before(
    light_decoder, base_draft_load, marker, rank, pp_size
):
    kind = None if (pp_size == 4 and rank == 0) else "DFLASH-TRTLLM"
    if marker:
        with B.pp_dflash_context_server():
            decoder = light_decoder.build(_decoder_config(rank, kind, pp_size))
    else:
        decoder = light_decoder.build(_decoder_config(rank, kind, pp_size))
    if kind is None:
        decoder.draft_model = _Drafter()
    checkpoint = _released_drafter_checkpoint()
    mapper = object()
    _call_load_weights(decoder.load_draft_weights, checkpoint, mapper)
    assert base_draft_load == [(decoder, checkpoint, mapper)]
    assert decoder.draft_model.calls == ["load_weights", "load_weights_from_target_model"]


def test_the_draft_load_keeps_the_loader_signature():
    """The model loader passes ``weight_mapper`` by keyword when the method names it."""
    override = inspect.signature(B.MuseSparkOpenForCausalLM.load_draft_weights)
    base = inspect.signature(B.SpecDecOneEngineForCausalLM.load_draft_weights)
    assert (
        list(override.parameters) == list(base.parameters) == ["self", "weights", "weight_mapper"]
    )


_DRAFTER_DIR_NAME = "Muse-Spark-Open-DFlash2"


def _released_drafter_dir():
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        return None
    path = os.path.join(root, _DRAFTER_DIR_NAME)
    return path if os.path.isfile(os.path.join(path, "model.safetensors.index.json")) else None


@pytest.mark.skipif(_released_drafter_dir() is None, reason="released drafter checkpoint not found")
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_a_stage_without_the_drafter_reads_nothing_of_the_released_checkpoint(
    monkeypatch, light_decoder, base_draft_load, rank
):
    """Every key the released checkpoint lists, and not one tensor read."""
    import json

    with open(os.path.join(_released_drafter_dir(), "model.safetensors.index.json")) as index:
        keys = json.load(index)["weight_map"]
    with B.pp_dflash_context_server():
        decoder = light_decoder.build(_decoder_config(rank))
    checkpoint = _UntouchableCheckpoint.fromkeys(keys)
    _call_load_weights(decoder.load_draft_weights, checkpoint, object())
    assert base_draft_load == [] and decoder.draft_model.calls == []
