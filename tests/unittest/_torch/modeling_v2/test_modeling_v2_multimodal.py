# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared base of Modeling V2 vision-language targets, and the router's multimodal check.

A tiny language model and a tiny tower stand in for a target's: construction, the runtime
contract, admission (also through the executor), the forward's hand-off in both row modes
through the real multimodal mixin, the compile scope and the weight split. Then the check the
router runs on every class a route returns.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2 import NoTarget
from tensorrt_llm._torch.modeling_v2 import _router_index as router
from tensorrt_llm._torch.modeling_v2.multimodal import ModelingV2ForConditionalGeneration
from tensorrt_llm._torch.models.checkpoints.model_hooks import call_load_weights
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor.engine import model_setup
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.inputs.multimodal import MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only

VOCAB, HIDDEN = 32, 4
IMAGE_TOKEN = VOCAB - 1  # in vocabulary, so the fused path embeds the whole prompt once
MAX_ITEM = 16  # the most rows the tiny tower takes in one item
ARCH = "TinyForConditionalGeneration"


# ------------------------------------------------------------------ stubs ---


class _TinyLlm(nn.Module):
    """A language model target in miniature: it adopts the text config on its own config
    copy, as a target does, and records what it is handed."""

    moe_impls_honored = frozenset({"tiny_moe"})

    def __init__(self, model_config):
        super().__init__()
        self.got = model_config
        model_config.pretrained_config = model_config.pretrained_config.text_config
        self.model_config = model_config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Embedding(VOCAB, HIDDEN)
        self.draft_config = None
        self.draft_model = object()
        self.spec_worker = object()
        self.multimodal_token_ids = torch.tensor([IMAGE_TOKEN], dtype=torch.int32)
        self.calls = []
        self.loaded = None

    @property
    def config(self):
        return self.model_config.pretrained_config

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return "V2"

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits=False,
        spec_metadata=None,
        **kwargs,
    ):
        """Records the call: the first six arguments in the order a language model takes them
        (`DecoderModelForCausalLM.forward`), and the others by name."""
        args = (
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            spec_metadata,
        )
        self.calls.append((args, kwargs))
        return "logits"

    def load_weights(self, weights):
        self.loaded = weights


class _TinyTower(nn.Module):
    """Each row it returns holds its input pixel's value; it refuses an item of more than
    `MAX_ITEM` rows, as a tower with a per-call bound does."""

    def __init__(self, vision_config, dtype):
        super().__init__()
        self.vision_config = vision_config
        self.scale = nn.Parameter(torch.ones((), dtype=dtype))
        self.calls = 0
        self.loaded = None

    def forward(self, pixels):
        self.calls += 1
        assert pixels.shape[0] <= MAX_ITEM, f"an item of {pixels.shape[0]} rows"
        return pixels.expand(-1, HIDDEN) * self.scale

    def load_weights(self, weights):
        self.loaded = weights


class _TinyForConditionalGeneration(ModelingV2ForConditionalGeneration):
    language_model_class = _TinyLlm
    mm_encoder_class = _TinyTower
    mm_encoder_weight_prefix = "model.visual."

    def validate_media_request(self, request):
        image = (request.py_multimodal_data or {}).get("image")
        if image is not None and image["pixel_values"].shape[0] > MAX_ITEM:
            raise ValueError("an image larger than the tower takes")

    def encode_multimodal_inputs(self, multimodal_params):
        if self.mm_encoder is None:
            raise ValueError("no tower")
        return torch.cat(
            [self.mm_encoder(p.multimodal_data["image"]["pixel_values"]) for p in multimodal_params]
        )


class _TinyRowsForConditionalGeneration(_TinyForConditionalGeneration):
    language_model_takes_rows = True


def _model_config(disable=False, side_stream=0, encoder_only=False):
    composite = PretrainedConfig(
        architectures=[ARCH],
        text_config=PretrainedConfig(hidden_size=HIDDEN, torch_dtype=torch.float32),
        vision_config=PretrainedConfig(),
        torch_dtype=torch.float32,
    )
    return ModelConfig(
        pretrained_config=composite,
        disable_mm_encoder=disable,
        mm_encoder_only=encoder_only,
        multimodal_config=MultimodalConfig(encoder_side_stream_max_ahead=side_stream),
    )


def _model(cls=_TinyForConditionalGeneration, disable=False):
    model = cls(_model_config(disable=disable))
    with torch.no_grad():
        # Token `i` embeds to `i` in every column, so a fused row says where it came from.
        model.llm.model.embed_tokens.weight.copy_(
            torch.arange(VOCAB, dtype=torch.float32)[:, None].expand(VOCAB, HIDDEN)
        )
    return model


def _image(*values):
    """One image whose rows hold `values`."""
    pixels = torch.tensor(values, dtype=torch.float32)[:, None]
    return {"image": {"pixel_values": pixels}}


def _params(*values, runtime=None, mrope=False):
    data = _image(*values) if values else {}
    if mrope:
        data["mrope_config"] = {"mrope_position_deltas": torch.zeros(1, 1, dtype=torch.int32)}
    return MultimodalParams(multimodal_data=data, multimodal_runtime=runtime)


# ----------------------------------------------------------- construction ---


@pytest.mark.parametrize("disable", [False, True], ids=["tower", "disable_mm_encoder"])
def test_construction_keeps_the_engine_config_and_builds_the_tower_only_when_asked(
    monkeypatch, disable
):
    model_config = _model_config(disable=disable)
    composite = model_config.pretrained_config
    # The tower takes the activation dtype: the text config's, not the composite's.
    composite.text_config.torch_dtype = torch.bfloat16
    built = []

    class Tower(_TinyTower):
        def __init__(self, vision_config, dtype):
            built.append(dtype)
            super().__init__(vision_config, dtype)

    monkeypatch.setattr(_TinyForConditionalGeneration, "mm_encoder_class", Tower)
    model = _TinyForConditionalGeneration(model_config)

    # The loader's own ModelConfig, now carrying the text config the pools are sized from.
    assert model.model_config is model_config
    assert model_config.pretrained_config is composite.text_config
    assert model.config is composite.text_config
    # The language model was built on its own copy, as the engine builds a decoder model.
    assert model.llm.got is not model_config
    assert model.llm.got.skip_create_weights_in_init is True
    assert model.language_model is model.llm
    if disable:
        # Never built, not built and dropped.
        assert built == []
        assert model.mm_encoder is None
        assert not any(name.startswith("mm_encoder.") for name, _ in model.named_parameters())
    else:
        assert built == [torch.bfloat16]
        assert isinstance(model.mm_encoder, Tower)
        assert model.mm_encoder.vision_config is composite.vision_config
        assert model.mm_encoder.scale.dtype == torch.bfloat16


def test_build_mm_encoder_overrides_mm_encoder_class():
    """A tower not built as `mm_encoder_class(vision_config, dtype)` comes from an override of
    `build_mm_encoder`. With neither, the model builds no tower and serves text only."""

    class Built(_TinyForConditionalGeneration):
        def build_mm_encoder(self, pretrained_config):
            return nn.Identity()

    class NoTower(_TinyForConditionalGeneration):
        mm_encoder_class = None

    assert isinstance(_model(Built).mm_encoder, nn.Identity)
    assert _model(NoTower).mm_encoder is None


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "disable, side_stream, encoder_only",
    [(False, 2, False), (False, 0, False), (True, 2, False), (False, 2, True)],
    ids=["tower", "tower_encoder_inline", "disable_mm_encoder", "mm_encoder_only"],
)
def test_the_route_declines_the_encoder_side_stream_while_a_tower_is_built(
    monkeypatch, disable, side_stream, encoder_only, mode
):
    """A `NoTarget` from the router, not the constructor: `auto` falls back to the built-in
    model and `require` fails. Nothing is declined without a tower, with the encoder in the
    step (`encoder_side_stream_max_ahead=0`, the default), or on an encoder-only worker, which
    builds the architecture's encoder model and not this one."""
    monkeypatch.setattr(
        router, "registered_routes", lambda: {ARCH: lambda *args: _TinyForConditionalGeneration}
    )
    monkeypatch.setattr(router, "_device_compute_capability", lambda: (10, 0))
    config = _model_config(disable=disable, side_stream=side_stream, encoder_only=encoder_only)
    llm_args = SimpleNamespace(modeling_v2=mode)
    if disable or side_stream == 0 or encoder_only:
        assert router.modeling_v2_resolve(config, llm_args) is _TinyForConditionalGeneration
    elif mode == "auto":
        assert router.modeling_v2_resolve(config, llm_args) is None
    else:
        with pytest.raises(ValueError, match="encoder_side_stream_max_ahead=2"):
            router.modeling_v2_resolve(config, llm_args)


# --------------------------------------------------------------- contract ---


def test_the_model_declares_the_runtime_contract():
    model = _model()
    assert isinstance(model, MultimodalModelMixin)
    assert model.dtype == torch.float32
    assert (model.embedding_dim, model.embedding_dtype) == (HIDDEN, torch.float32)
    assert model.text_embedding_layer is model.llm.model.embed_tokens
    assert model.samples_from_vocab_shards is False
    # What the engine reads off the routed model is the language model's.
    llm = model.llm
    assert model.lm_head is llm.lm_head
    assert model.draft_config is None
    assert (model.draft_model, model.spec_worker) == (llm.draft_model, llm.spec_worker)
    assert model.multimodal_token_ids is llm.multimodal_token_ids
    # So are the class-level hooks the router and the loader read off the routed class.
    assert _TinyForConditionalGeneration.get_preferred_kv_cache_manager_version() == "V2"
    assert _TinyForConditionalGeneration.moe_impls_honored == frozenset({"tiny_moe"})
    assert _TinyRowsForConditionalGeneration.moe_impls_honored == frozenset({"tiny_moe"})
    # A flag the language model does not set takes the value its reader takes for none.
    assert _TinyForConditionalGeneration.draft_moe_impls_honored == frozenset()
    assert _TinyForConditionalGeneration.supports_nvfp4_kv_cache is False

    class Own(_TinyForConditionalGeneration):
        language_model_class = _TinyLlm
        moe_impls_honored = frozenset({"own"})

    class FlaglessLlm(nn.Module):
        pass

    class Swapped(_TinyForConditionalGeneration):
        language_model_class = FlaglessLlm

    assert Own.moe_impls_honored == frozenset({"own"})
    # A subclass that swaps in another language model does not keep its parent's flags.
    assert Swapped.moe_impls_honored == frozenset()


# -------------------------------------------------------------- admission ---


def _request(tokens=(1, 2, 3), bound=None, **data):
    return SimpleNamespace(
        py_multimodal_data=data or None,
        py_prompt_token_bound=bound,
        check_token_id_range=lambda vocab: max(tokens) < vocab,
    )


def _admit(model, request):
    """Both admission hooks, in the executor's order."""
    model.validate_request(request)
    if request.py_multimodal_data:
        model.validate_multimodal_request_data(request.py_multimodal_data)


MROPE_ONLY = {"mrope_config": {"mrope_position_deltas": torch.zeros(1, 1)}}


def test_admission_checks_the_token_range_and_the_towers_media():
    model = _model()
    _admit(model, _request())
    _admit(model, _request(tokens=(1, IMAGE_TOKEN), **_image(1.0, 2.0)))
    _admit(model, _request(bound=VOCAB - 1))
    # M-RoPE positions alone (a disaggregated generation request) are not media.
    _admit(model, _request(**MROPE_ONLY))
    with pytest.raises(ValueError, match="Token ID out of range"):
        _admit(model, _request(tokens=(1, VOCAB)))
    with pytest.raises(ValueError, match="Token ID out of range"):
        _admit(model, _request(bound=VOCAB))
    with pytest.raises(ValueError, match="larger than the tower takes"):
        _admit(model, _request(**_image(*range(MAX_ITEM + 1))))
    with pytest.raises(ValueError, match="precomputed multimodal embeddings"):
        _admit(model, _request(multimodal_embedding=torch.zeros(2, HIDDEN)))


def test_without_a_tower_media_is_refused_and_mrope_positions_pass():
    model = _model(disable=True)
    _admit(model, _request())
    _admit(model, _request(**MROPE_ONLY))
    # No tower check without a tower; the media check refuses the request instead.
    model.validate_request(_request(**_image(*range(MAX_ITEM + 1))))
    for data in (_image(1.0), {"multimodal_embedding": torch.zeros(2, HIDDEN)}):
        with pytest.raises(ValueError, match="disable_mm_encoder.*serves text only"):
            _admit(model, _request(**data))


def test_the_executor_fails_only_the_request_the_model_refuses():
    """Through the executor's own admission (`_fetch_and_activate_new_requests`): the bad
    request fails with its own error and no budget charge; the others are admitted."""
    model = _model()
    requests = [
        _request(),
        _request(**_image(*range(MAX_ITEM + 1))),
        _request(**MROPE_ONLY),
        _request(tokens=(1, VOCAB)),
        _request(**_image(1.0)),
        _request(multimodal_embedding=torch.zeros(2, HIDDEN)),
    ]
    for request_id, request in enumerate(requests):
        request.request_id = request_id
        request.py_disaggregated_params = None
        request.sampling_config = None
        request.kv_cache_retention_config = None
    executor = SimpleNamespace(
        model_engine=SimpleNamespace(model=model),
        _is_kv_manager_v2=False,
        kv_cache_manager=None,
        kv_cache_transceiver=None,
        max_beam_width=1,
        sampler=Mock(),
        waiting_queue=object(),
        active_requests=[],
        _mm_encoder_item_scheduling_enabled=False,
        _fetch_new_requests=Mock(return_value=requests),
        _configure_csa2_remote_tail=Mock(),
        _validate_token_id_range=Mock(),
        _validate_request_budget=Mock(),
        _handle_errors=Mock(),
    )
    executor._validate_request = PyExecutor._validate_request.__get__(executor)

    admitted = PyExecutor._fetch_and_activate_new_requests(executor)

    assert admitted == [requests[0], requests[2], requests[4]]
    failed = [
        (call.kwargs["requests"], call.kwargs["charge_budget"])
        for call in executor._handle_errors.call_args_list
    ]
    assert failed == [([requests[1]], False), ([requests[3]], False), ([requests[5]], False)]


# ---------------------------------------------------------------- forward ---


def _forward(model, params, num_contexts=1, mm_rows=None, inputs_embeds=None):
    """One step over a 6-token prompt whose tokens 1..`mm_rows` are placeholders."""
    input_ids = torch.tensor([1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 4, 5], dtype=torch.int32)
    kwargs = {"multimodal_params": params, "resource_manager": "rm"}
    if mm_rows is not None:
        kwargs["mm_token_indices"] = torch.arange(1, 1 + mm_rows, dtype=torch.int64)
        kwargs["text_token_indices"] = torch.tensor([0, *range(1 + mm_rows, 6)], dtype=torch.int64)
    out = model.forward(
        SimpleNamespace(num_contexts=num_contexts, num_generations=0),
        input_ids,
        "position_ids",
        inputs_embeds,
        False,
        "spec_metadata",
        **kwargs,
    )
    assert out == "logits"
    ((args, llm_kwargs),) = model.llm.calls
    assert args[0].num_contexts == num_contexts
    assert args[2] == "position_ids" and args[4:] == (False, "spec_metadata")
    assert "multimodal_params" not in llm_kwargs and llm_kwargs["resource_manager"] == "rm"
    return input_ids, args, llm_kwargs


def test_rows_mode_hands_the_rows_and_their_placeholder_rows_to_the_language_model():
    model = _model(_TinyRowsForConditionalGeneration)
    input_ids, args, llm_kwargs = _forward(model, [_params(7.0, 8.0, 9.0)], mm_rows=3)
    # `input_ids` reach the language model unchanged, and `inputs_embeds` stays None.
    assert args[1] is input_ids and args[3] is None
    rows, mm_token_indices = llm_kwargs["multimodal_rows"]
    assert rows[:, 0].tolist() == [7.0, 8.0, 9.0]
    assert mm_token_indices.tolist() == [1, 2, 3]


def test_fused_mode_hands_fused_embeddings_to_the_language_model():
    model = _model()
    _, args, llm_kwargs = _forward(model, [_params(7.0, 8.0, 9.0)], mm_rows=3)
    assert args[1] is None
    assert args[3][:, 0].tolist() == [1.0, 7.0, 8.0, 9.0, 4.0, 5.0]
    assert "multimodal_rows" not in llm_kwargs


@pytest.mark.parametrize(
    "cls", [_TinyForConditionalGeneration, _TinyRowsForConditionalGeneration], ids=["fused", "rows"]
)
def test_a_chunk_hands_over_its_slice_of_the_rows(cls):
    """The mixin keeps the chunk's rows: 2 of 5 reused, 3 in this chunk."""
    model = _model(cls)
    runtime = MultimodalRuntimeData(
        embed_mask_cumsum=torch.arange(1, 6, dtype=torch.int64),
        past_seen_token_num=2,
        chunk_end_pos=5,
    )
    _, args, llm_kwargs = _forward(
        model, [_params(5.0, 6.0, 7.0, 8.0, 9.0, runtime=runtime)], mm_rows=3
    )
    if cls is _TinyRowsForConditionalGeneration:
        assert llm_kwargs["multimodal_rows"][0][:, 0].tolist() == [7.0, 8.0, 9.0]
    else:
        assert args[3][1:4, 0].tolist() == [7.0, 8.0, 9.0]


@pytest.mark.parametrize(
    "cls", [_TinyForConditionalGeneration, _TinyRowsForConditionalGeneration], ids=["fused", "rows"]
)
@pytest.mark.parametrize(
    "params, num_contexts",
    [
        ([], 1),
        ([MultimodalParams(multimodal_data={})], 1),
        # A text context mixed with a generation request's position deltas.
        ([_params(mrope=True)], 1),
        # A generation-only step.
        ([_params(7.0)], 0),
        # A chunk after the image: its 3 placeholders were all in earlier chunks.
        (
            [
                _params(
                    7.0,
                    8.0,
                    9.0,
                    runtime=MultimodalRuntimeData(
                        embed_mask_cumsum=torch.tensor([0, 1, 2, 3, 3, 3]),
                        past_seen_token_num=4,
                        chunk_end_pos=6,
                    ),
                )
            ],
            1,
        ),
    ],
    ids=["no_params", "empty", "mrope_only", "generation", "chunk_after_the_placeholders"],
)
def test_a_step_without_media_runs_no_tower(params, num_contexts, cls):
    model = _model(cls)
    input_ids, args, llm_kwargs = _forward(model, params, num_contexts=num_contexts)
    assert args[1] is input_ids and args[3] is None
    assert llm_kwargs.get("multimodal_rows") is None
    assert model.mm_encoder.calls == 0


def test_the_rows_must_match_the_placeholder_rows():
    with pytest.raises(AssertionError, match="3 multimodal rows and 2 placeholder rows"):
        _forward(_model(_TinyRowsForConditionalGeneration), [_params(7.0, 8.0, 9.0)], mm_rows=2)


class _DeepStack:
    """V1 Qwen3-VL's DeepStack pattern in miniature: an extra feature for each of the chunk's
    rows (10 times the row), handed to the language model by the mixin's hook."""

    def after_active_multimodal_embeddings(self, *, active_embeddings, **kwargs):
        return active_embeddings, (torch.cat(active_embeddings) * 10,)

    def get_language_model_extra_forward_kwargs(self, *, mm_inputs, spec_metadata, **kwargs):
        self.hook_got = dict(kwargs, mm_inputs=mm_inputs)
        # A V1 hook also returns `spec_metadata`; the language model still takes it once.
        return {"deepstack_embeds": list(mm_inputs.extra_embeds), "spec_metadata": spec_metadata}


@pytest.mark.parametrize(
    "cls", [_TinyForConditionalGeneration, _TinyRowsForConditionalGeneration], ids=["fused", "rows"]
)
def test_what_the_extra_forward_kwargs_hook_returns_reaches_the_language_model(cls):
    model = _model(type("DeepStackForConditionalGeneration", (_DeepStack, cls), {}))
    params = [_params(7.0, 8.0, 9.0)]
    input_ids, args, llm_kwargs = _forward(model, params, mm_rows=3)
    (deepstack,) = llm_kwargs["deepstack_embeds"]
    got = model.hook_got
    # The hook gets the same arguments as in the mixin's own forward.
    assert got["raw_input_ids"] is input_ids and got["position_ids"] == "position_ids"
    assert got["multimodal_params"] is params and got["num_generation_requests"] == 0
    assert got["resource_manager"] == "rm"
    if cls is _TinyRowsForConditionalGeneration:
        # The rows mode: the unchanged input_ids, no inputs_embeds, one extra row per row.
        assert got["mm_inputs"].input_ids is input_ids and got["mm_inputs"].inputs_embeds is None
        assert got["multimodal_rows"] is llm_kwargs["multimodal_rows"]
        assert deepstack[:, 0].tolist() == [70.0, 80.0, 90.0]
    else:
        # The fused mode: the fused inputs, the extra rows spread over the tokens.
        assert got["mm_inputs"].input_ids is None and got["mm_inputs"].inputs_embeds is args[3]
        assert deepstack[:, 0].tolist() == [0.0, 70.0, 80.0, 90.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "cls", [_TinyForConditionalGeneration, _TinyRowsForConditionalGeneration], ids=["fused", "rows"]
)
def test_by_default_the_language_model_takes_no_extra_argument(cls):
    _, _, llm_kwargs = _forward(_model(cls), [_params(7.0, 8.0, 9.0)], mm_rows=3)
    expected = {"resource_manager", "mm_token_indices", "text_token_indices"}
    if cls is _TinyRowsForConditionalGeneration:
        expected.add("multimodal_rows")
    assert set(llm_kwargs) == expected


class _ExtraRowsOnly(_DeepStack):
    """A hook that keeps no multimodal row of the chunk and returns extra rows."""

    def after_active_multimodal_embeddings(self, *, active_embeddings, **kwargs):
        return [], (torch.cat(active_embeddings) * 10,)


@pytest.mark.parametrize(
    "cls", [_TinyForConditionalGeneration, _TinyRowsForConditionalGeneration], ids=["fused", "rows"]
)
def test_extra_rows_reach_the_hook_when_the_chunk_has_no_multimodal_row(cls):
    model = _model(type("ExtraRowsOnlyForConditionalGeneration", (_ExtraRowsOnly, cls), {}))
    input_ids, args, llm_kwargs = _forward(model, [_params(7.0, 8.0, 9.0)])
    # Nothing is fused or handed over as rows, and the extra rows still reach the hook.
    assert args[1] is input_ids and args[3] is None
    assert llm_kwargs.get("multimodal_rows") is None
    (extra,) = llm_kwargs["deepstack_embeds"]
    assert extra[:, 0].tolist() == [70.0, 80.0, 90.0]


def test_in_the_fused_mode_caller_inputs_embeds_replace_input_ids_as_in_the_mixin():
    model = _model(
        type("DeepStackForConditionalGeneration", (_DeepStack, _TinyForConditionalGeneration), {})
    )
    inputs_embeds = torch.ones(6, HIDDEN)
    _, args, _ = _forward(model, [], inputs_embeds=inputs_embeds)
    assert args[1] is None and args[3] is inputs_embeds
    mm_inputs = model.hook_got["mm_inputs"]
    assert mm_inputs.input_ids is None and mm_inputs.inputs_embeds is inputs_embeds


# ---------------------------------------------------- compile and weights ---


def test_torch_compile_reaches_only_the_language_models_core(monkeypatch):
    compiled = []

    def spy(module, **kwargs):
        compiled.append(module)
        return module

    monkeypatch.setattr(torch, "compile", spy)
    model = _model()
    core = model.llm.model
    returned, prefill_only = model_setup.apply_torch_compile(
        model, "inductor", llm_args=SimpleNamespace(), fullgraph=False
    )
    assert returned is model and prefill_only is False
    assert compiled == [core]


class _Checkpoint(Mapping):
    """Checkpoint weights that record which keys are read."""

    def __init__(self, weights):
        self.weights, self.read = weights, set()

    def __getitem__(self, key):
        self.read.add(key)
        return self.weights[key]

    def __iter__(self):
        return iter(self.weights)

    def __len__(self):
        return len(self.weights)


@pytest.mark.parametrize("disable", [False, True], ids=["tower", "disable_mm_encoder"])
def test_the_tower_and_the_language_model_each_load_their_own_keys(disable):
    model = _model(disable=disable)
    weights = _Checkpoint(
        {"model.visual.patch.weight": 1, "model.layers.0.weight": 2, "lm_head.weight": 3}
    )
    model.load_weights(weights)
    assert model.llm.loaded == {"model.layers.0.weight": 2, "lm_head.weight": 3}
    if not disable:
        assert model.mm_encoder.loaded == {"model.visual.patch.weight": 1}
    # Without a tower, the tower's keys stay unread.
    tower_keys = set() if disable else {"model.visual.patch.weight"}
    assert weights.read == {"model.layers.0.weight", "lm_head.weight"} | tower_keys


class _MappedLlm(_TinyLlm):
    def load_weights(self, weights, weight_mapper=None):
        self.loaded = weights
        self.weight_mapper = weight_mapper


class _MappedForConditionalGeneration(_TinyForConditionalGeneration):
    language_model_class = _MappedLlm


class _Mapper:
    """The model a weight mapper is bound to, as `BaseWeightMapper` binds it."""

    def init_model_and_config(self, model, config):
        self.model, self.config = model, config


def _bound_mapper(model):
    """A mapper the engine bound to the routed model."""
    mapper = _Mapper()
    mapper.init_model_and_config(model, model.model_config)
    return mapper


def test_the_engines_weight_mapper_reaches_a_language_model_that_takes_one():
    """The engine passes `weight_mapper` only to a `load_weights` that names it
    (`call_load_weights`); the base names it and hands it on the same way, bound to the
    language model, whose modules the keys name."""
    model = _model(_MappedForConditionalGeneration)
    mapper = _bound_mapper(model)
    call_load_weights(model.load_weights, {"model.layers.0.weight": 2}, mapper)
    assert model.llm.weight_mapper is mapper
    assert (mapper.model, mapper.config) == (model.llm, model.llm.model_config)
    # Without a mapper, the language model gets none.
    model = _model(_MappedForConditionalGeneration)
    call_load_weights(model.load_weights, {"model.layers.0.weight": 2}, None)
    assert model.llm.weight_mapper is None
    # A language model whose `load_weights` takes no mapper is called without one, and the
    # mapper stays bound to the routed model.
    model = _model()
    mapper = _bound_mapper(model)
    call_load_weights(model.load_weights, {"model.layers.0.weight": 2}, mapper)
    assert model.llm.loaded == {"model.layers.0.weight": 2}
    assert mapper.model is model


# ---------------------------------------------------- the router's check ---


class _Decoder:
    """A language model target that neither serves media nor refuses it."""


class _TextOnly:
    def validate_multimodal_request_data(self, mm_data):
        raise ValueError("text only")


class _OwnEncoder:
    """A target that encodes media in its own forward and refuses at admission what its
    tower cannot take."""

    def validate_request(self, request):
        raise ValueError("a request the tower cannot take")


class _TokenRangeCheck:
    def validate_request(self, request):
        """Checks the token range only."""


class _InheritedValidator(_TokenRangeCheck):
    """A language model target whose `validate_request` is a base class's: it refuses no media
    of its own."""


class _Marked:
    supports_multimodal = True


class _MarkedFalse:
    supports_multimodal = False


def test_a_vision_language_architecture_routes_to_a_multimodal_or_text_only_target(monkeypatch):
    monkeypatch.setattr(router, "_architecture_takes_media", lambda arch: arch == ARCH)
    for target in (_TinyForConditionalGeneration, _TextOnly, _OwnEncoder, _Marked):
        router._check_multimodal_contract(target, ARCH)
    router._check_multimodal_contract(_Decoder, "TinyForCausalLM")
    # The marker counts only when it is True; `validate_request` only in the class's own body.
    for target in (_Decoder, _MarkedFalse, _InheritedValidator):
        with pytest.raises(
            ValueError,
            match=f"{target.__name__} is the target for '{ARCH}'.*"
            "ModelingV2ForConditionalGeneration.*validate_multimodal",
        ) as error:
            router._check_multimodal_contract(target, ARCH)
        # Not a `NoTarget`, which `auto` turns into a fallback to the built-in model.
        assert not isinstance(error.value, NoTarget)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_a_target_that_breaks_the_contract_fails_at_startup(monkeypatch, mode):
    """Not a `NoTarget`: `auto` does not fall back, as the target is wrong, not the
    deployment."""
    monkeypatch.setattr(router, "registered_routes", lambda: {ARCH: lambda *args: _Decoder})
    monkeypatch.setattr(router, "_device_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(router, "_architecture_takes_media", lambda arch: True)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=[ARCH]), moe_tactic=None, spec_config=None
    )
    with pytest.raises(ValueError, match=f"_Decoder is the target for '{ARCH}'"):
        router.modeling_v2_resolve(config, SimpleNamespace(modeling_v2=mode))


def test_media_architectures_are_known_by_their_input_processor():
    assert router._architecture_takes_media("Qwen4ExpForConditionalGeneration")
    assert not router._architecture_takes_media("DeepseekV3ForCausalLM")
    assert not router._architecture_takes_media("NoSuchArchitectureForCausalLM")
