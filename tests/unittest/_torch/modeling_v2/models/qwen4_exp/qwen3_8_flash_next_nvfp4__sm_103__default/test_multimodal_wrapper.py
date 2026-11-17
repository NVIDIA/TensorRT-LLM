# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default target's multimodal wrapper on the CPU.

Construction, the multimodal runtime contract, admission (also through the executor), the
encoder hook, the forward's hand-off to the language model, the compile scope and the weight
split. The language model is never built (its core asserts sm_103): construction swaps in a
stub class, and the other tests call the wrapper's real methods on stubs, with the real
multimodal mixin in between. Towers are stubs over the real `VisionTower.forward`, or real
ones on the meta device.
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.configs import Qwen4ExpVisionConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.multimodal import is_multimodal_model
from tensorrt_llm._torch.pyexecutor.engine import model_setup
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
    MultimodalRuntimeData,
)

pytestmark = pytest.mark.cpu_only

Wrapper = M.ModelingV2Qwen38FlashNextNvfp4Sm103DefaultForConditionalGeneration

# Qwen3.8-Flash-Next's vision config.
_VISION = dict(
    deepstack_visual_indexes=[],
    depth=27,
    dtype="bfloat16",
    hidden_act="gelu_pytorch_tanh",
    hidden_size=1152,
    in_channels=3,
    intermediate_size=4304,
    num_heads=16,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)
HIDDEN = 2560
MERGE = 2
PIXEL_WIDTH = 4
VOCAB = 100


# ------------------------------------------------------------------ stubs ---


def _tower(dtype=torch.float32, width=1):
    """A `VisionTower` whose per-call body is a stub: merged row `j` of item `i` of modality
    `m` holds `100 * m + i` in every column, so a row says where it came from. The real
    `forward` (grid checks, frame calls) runs around it."""
    tower = object.__new__(V.VisionTower)
    nn.Module.__init__(tower)
    tower.dtype, tower.pixel_width, tower.merge = dtype, PIXEL_WIDTH, MERGE
    tower.calls = 0

    def run(pixel_values, grid):
        tower.calls += 1
        return pixel_values[:: MERGE * MERGE, :1].expand(-1, width).to(dtype).clone()

    tower._run = run
    return tower


IMAGE, VIDEO = 1, 2
IMAGES = [(1, 4, 4), (1, 2, 2)]  # 4 and 1 rows
VIDEOS = [(2, 2, 4), (2, 4, 4)]  # 4 and 8 rows
MIXED_ORDER = [("video", 0), ("image", 0), ("video", 1), ("image", 1)]
MIXED_ROWS = [200.0] * 4 + [100.0] * 4 + [201.0] * 8 + [101.0]


def _media(modality, grids):
    """One modality's payload; each item's patch rows hold `100 * modality + item`."""
    pixels = torch.cat(
        [
            torch.full((t * h * w, PIXEL_WIDTH), float(100 * modality + item))
            for item, (t, h, w) in enumerate(grids)
        ]
    )
    if modality == IMAGE:
        return {"pixel_values": pixels, "image_grid_thw": torch.tensor(grids)}
    return {"pixel_values_videos": pixels, "video_grid_thw": torch.tensor(grids)}


def _params(image=None, video=None, order=None, runtime=None, mrope=True, delta_only=False):
    data = {}
    if mrope:
        data["mrope_config"] = {"mrope_position_ids": torch.zeros(3, 1, 8, dtype=torch.int32)}
    if delta_only:
        data["mrope_config"] = {"mrope_position_deltas": torch.zeros(1, 1, dtype=torch.int32)}
    if image is not None:
        data["image"] = _media(IMAGE, image)
    if video is not None:
        data["video"] = _media(VIDEO, video)
    if order is not None:
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(item_refs=order)
    return MultimodalParams(multimodal_data=data, multimodal_runtime=runtime)


def _runtime(total: int, cached: int, chunk_end: int) -> MultimodalRuntimeData:
    """A prompt of `total` placeholder tokens, `cached` of them reused, up to `chunk_end`."""
    return MultimodalRuntimeData(
        embed_mask_cumsum=torch.arange(1, total + 1, dtype=torch.int64),
        past_seen_token_num=cached,
        chunk_end_pos=chunk_end,
    )


class _Llm(nn.Module):
    """Records what the wrapper hands the language model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.calls = []
        self.lm_head = SimpleNamespace(num_embeddings=VOCAB)
        self.multimodal_token_ids = torch.tensor([90, 91], dtype=torch.int32)

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "logits"


def _wrapper(tower=None, llm=None):
    """A real wrapper instance around stubs, without the constructor."""
    wrapper = Wrapper.__new__(Wrapper)
    nn.Module.__init__(wrapper)
    wrapper.llm = llm if llm is not None else _Llm()
    wrapper.mm_encoder = tower
    wrapper.model_config = SimpleNamespace(multimodal_config=None, torch_dtype=torch.bfloat16)
    return wrapper


# ----------------------------------------------------------- construction ---


class _StubLlm(nn.Module):
    """Stands in for the language model target: takes its config copy and adopts the text
    config on it, as the target does."""

    def __init__(self, model_config):
        super().__init__()
        self.got = model_config
        model_config.pretrained_config = model_config.pretrained_config.text_config
        self.model_config = model_config

    @property
    def config(self):
        return self.model_config.pretrained_config


def _composite():
    return PretrainedConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=PretrainedConfig(hidden_size=HIDDEN, torch_dtype=torch.bfloat16),
        vision_config=Qwen4ExpVisionConfig(**_VISION),
        torch_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("disable", [False, True], ids=["tower", "disable_mm_encoder"])
def test_construction_keeps_the_engine_config_and_builds_the_tower_only_when_asked(
    monkeypatch, disable
):
    monkeypatch.setattr(M, "ModelingV2Qwen38FlashNextNvfp4Sm103Default", _StubLlm)
    composite = _composite()
    model_config = ModelConfig(pretrained_config=composite, disable_mm_encoder=disable)
    with torch.device("meta"):
        wrapper = Wrapper(model_config)

    # The loader's own ModelConfig, now carrying the text config the pools are sized from.
    assert wrapper.model_config is model_config
    assert model_config.pretrained_config is composite.text_config
    assert wrapper.config is composite.text_config
    # The language model was built on its own copy, as the engine builds a decoder model.
    assert wrapper.llm.got is not model_config
    assert wrapper.llm.got.skip_create_weights_in_init is True
    assert wrapper.language_model is wrapper.llm
    assert is_multimodal_model(wrapper)
    assert wrapper.dtype == torch.bfloat16
    params = {name.split(".")[0] for name, _ in wrapper.named_parameters()}
    if disable:
        assert wrapper.mm_encoder is None and not params
    else:
        assert isinstance(wrapper.mm_encoder, V.VisionTower)
        assert isinstance(wrapper.mm_encoder, MultimodalEncoderMixin)
        assert wrapper.mm_encoder.dtype == torch.bfloat16
        # Every vision weight is the tower's.
        assert params == {"mm_encoder"}
        assert all(".w.vis_" in name or ".w.l" in name for name, _ in wrapper.named_parameters())


# --------------------------------------------------------------- contract ---


def test_the_class_declares_the_runtime_contract():
    assert Wrapper.supports_multimodal is True
    assert is_multimodal_model(Wrapper.__new__(Wrapper))
    assert Wrapper.supports_encoder_cache is True
    assert Wrapper.supports_mm_encoder_item_scheduling is False
    assert Wrapper.supported_encoder_attention_backends == {}
    assert tuple(Wrapper.mm_encoder_groups) == ()
    assert Wrapper.get_preferred_kv_cache_manager_version() == "V2"
    for name in ("language_model", "dtype", "embedding_dim", "embedding_dtype"):
        assert isinstance(Wrapper.__dict__[name], property), name
    wrapper = _wrapper()
    wrapper.config = SimpleNamespace(hidden_size=HIDDEN)
    assert (wrapper.embedding_dim, wrapper.embedding_dtype) == (HIDDEN, torch.bfloat16)
    assert wrapper.samples_from_vocab_shards is False
    # What the engine reads off the routed model is the language model's.
    llm = wrapper.llm
    llm.draft_config, llm.draft_model, llm.spec_worker = None, object(), object()
    assert wrapper.lm_head is llm.lm_head
    assert (wrapper.draft_model, wrapper.spec_worker) == (llm.draft_model, llm.spec_worker)
    assert wrapper.draft_config is None
    assert torch.equal(wrapper.multimodal_token_ids, llm.multimodal_token_ids)


def test_the_tower_is_an_encoder_without_engine_metadata():
    assert issubclass(V.VisionTower, MultimodalEncoderMixin)
    assert V.VisionTower.encoder_cuda_graph_modality is None
    tower = _tower()
    tower.setup_attn_metadata(16, {"attention": 16})
    tower.set_attn_max_seq_len(16)
    assert tower.attn_metadata is None


# -------------------------------------------------------------- admission ---

_GEOMETRY = SimpleNamespace(merge=MERGE, patch=16, pixel_width=PIXEL_WIDTH)


def _request(tokens=(1, 2, 3), image=None, mrope_len=None, bound=None, **data):
    """A request whose prompt is `tokens`, optionally around one image's placeholders."""
    prompt_len = len(tokens)
    if image is not None:
        rows = sum(t * h * w for t, h, w in image) // (MERGE * MERGE)
        prompt_len += rows
        data["image"] = _media(IMAGE, image)
        data["mrope_config"] = {
            "mrope_position_ids": torch.zeros(3, 1, mrope_len or prompt_len, dtype=torch.int32)
        }
    return SimpleNamespace(
        py_multimodal_data=data or None,
        prompt_len=prompt_len,
        py_prompt_token_bound=bound,
        check_token_id_range=lambda vocab: max(tokens) < vocab,
    )


def _admit(wrapper, request):
    """Both admission hooks, in the executor's order."""
    Wrapper.validate_request(wrapper, request)
    if request.py_multimodal_data:
        Wrapper.validate_multimodal_request_data(wrapper, request.py_multimodal_data)


def test_admission_checks_the_tokens_and_the_media():
    wrapper = SimpleNamespace(mm_encoder=_GEOMETRY, llm=SimpleNamespace(lm_head=_Llm().lm_head))
    _admit(wrapper, _request())
    _admit(wrapper, _request(image=IMAGES))
    _admit(wrapper, _request(bound=VOCAB - 1))
    # M-RoPE positions alone (a disaggregated generation request) are not media.
    _admit(wrapper, _request(mrope_config={"mrope_position_deltas": torch.zeros(1, 1)}))
    with pytest.raises(ValueError, match="one position per prompt token"):
        _admit(wrapper, _request(image=IMAGES, mrope_len=4))
    with pytest.raises(ValueError, match="Token ID out of range"):
        _admit(wrapper, _request(tokens=(1, VOCAB)))
    with pytest.raises(ValueError, match="Token ID out of range"):
        _admit(wrapper, _request(bound=VOCAB))
    with pytest.raises(ValueError, match="precomputed multimodal embeddings"):
        _admit(wrapper, _request(multimodal_embedding=torch.zeros(4, HIDDEN)))


def test_without_a_tower_media_is_refused_and_mrope_positions_pass():
    wrapper = SimpleNamespace(mm_encoder=None, llm=SimpleNamespace(lm_head=_Llm().lm_head))
    _admit(wrapper, _request())
    _admit(wrapper, _request(mrope_config={"mrope_position_deltas": torch.zeros(1, 1)}))
    for request in (
        _request(image=IMAGES),
        _request(multimodal_embedding=torch.zeros(4, HIDDEN)),
    ):
        with pytest.raises(ValueError, match="disable_mm_encoder and serves text only"):
            _admit(wrapper, request)


def test_the_executor_fails_only_the_request_the_wrapper_refuses():
    """Through the executor's own admission (`_fetch_and_activate_new_requests`): the bad
    request fails with its own error and no budget charge; the others are admitted."""
    model = SimpleNamespace(mm_encoder=_GEOMETRY, llm=SimpleNamespace(lm_head=_Llm().lm_head))
    model.validate_request = types.MethodType(Wrapper.validate_request, model)
    model.validate_multimodal_request_data = types.MethodType(
        Wrapper.validate_multimodal_request_data, model
    )
    requests = [
        _request(),
        _request(image=IMAGES, mrope_len=4),
        _request(mrope_config={"mrope_position_deltas": torch.zeros(1, 1)}),
        _request(tokens=(1, VOCAB)),
        _request(image=IMAGES),
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
    executor._validate_request = types.MethodType(PyExecutor._validate_request, executor)

    admitted = PyExecutor._fetch_and_activate_new_requests(executor)

    assert admitted == [requests[0], requests[2], requests[4]]
    failed = [
        (call.kwargs["requests"], call.kwargs["charge_budget"])
        for call in executor._handle_errors.call_args_list
    ]
    assert failed == [([requests[1]], False), ([requests[3]], False)]


# ----------------------------------------------------------- encoder hook ---


def test_the_encoder_hook_takes_a_profiling_dummy():
    """The engine's memory profiling encodes one dummy image with no position ids and no item
    metadata, at the model's dtype: rows of the text hidden width, in that dtype."""
    tower = _tower(dtype=torch.bfloat16, width=HIDDEN)
    wrapper = _wrapper(tower)
    dummy = MultimodalParams(multimodal_data={"image": _media(IMAGE, [(1, 8, 8)])})
    dummy.multimodal_data["image"]["pixel_values"] = dummy.multimodal_data["image"][
        "pixel_values"
    ].to(wrapper.dtype)
    rows = wrapper.encode_multimodal_inputs([dummy])
    assert rows.shape == (16, HIDDEN) and rows.dtype == torch.bfloat16


def test_the_encoder_hook_returns_each_request_whole_in_prompt_order():
    tower = _tower()
    wrapper = _wrapper(tower)
    mixed = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    rows = wrapper.encode_multimodal_inputs([_params(image=IMAGES), mixed])
    assert rows[:, 0].tolist() == [100.0] * 4 + [101.0] + MIXED_ROWS
    with pytest.raises(ValueError, match="require a local multimodal encoder"):
        _wrapper(None).encode_multimodal_inputs([mixed])


# ---------------------------------------------------------------- forward ---


def _forward(wrapper, params, num_contexts=1, num_rows=None, input_ids=None):
    input_ids = input_ids if input_ids is not None else torch.arange(24, dtype=torch.int32)
    kwargs = {"multimodal_params": params, "resource_manager": "rm"}
    if num_rows is not None:
        kwargs["mm_token_indices"] = torch.arange(1, 1 + num_rows, dtype=torch.int64)
    out = wrapper.forward(
        SimpleNamespace(num_contexts=num_contexts),
        input_ids,
        "position_ids",
        None,
        False,
        "spec_metadata",
        **kwargs,
    )
    assert out == "logits"
    ((args, llm_kwargs),) = wrapper.llm.calls
    # `input_ids` reach the language model unchanged, and `inputs_embeds` stays None.
    assert args[1] is input_ids
    assert args[2:] == ("position_ids", None, False, "spec_metadata")
    assert "multimodal_params" not in llm_kwargs and llm_kwargs["resource_manager"] == "rm"
    return llm_kwargs["vision"], kwargs.get("mm_token_indices")


def test_a_media_step_hands_rows_and_placeholder_rows_to_the_language_model():
    wrapper = _wrapper(_tower())
    vision, indices = _forward(wrapper, [_params(image=IMAGES)], num_rows=5)
    rows, mm_token_indices = vision
    assert rows[:, 0].tolist() == [100.0] * 4 + [101.0]
    assert mm_token_indices is indices


def test_a_chunk_hands_over_its_slice_of_the_prompt_order_rows():
    """The mixin keeps the chunk's rows: 3 reused, 6 in this chunk."""
    tower = _tower()
    wrapper = _wrapper(tower)
    runtime = _runtime(len(MIXED_ROWS), cached=3, chunk_end=9)
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime)
    (rows, _), _ = _forward(wrapper, [params], num_rows=6)
    assert rows[:, 0].tolist() == MIXED_ROWS[3:9]


@pytest.mark.parametrize("with_tower", [True, False], ids=["tower", "no_tower"])
@pytest.mark.parametrize(
    "params, num_contexts",
    [
        ([], 1),
        ([MultimodalParams(multimodal_data={})], 1),
        # A text prompt through the multimodal processor: M-RoPE positions only.
        ([_params()], 1),
        # A text context mixed with a generation request's position deltas.
        ([_params(mrope=False, delta_only=True)], 1),
        # A generation-only step.
        ([_params(image=IMAGES)], 0),
    ],
    ids=["no_params", "empty", "mrope_only", "delta_only", "generation"],
)
def test_a_step_without_media_runs_no_tower(params, num_contexts, with_tower):
    tower = _tower() if with_tower else None
    wrapper = _wrapper(tower)
    vision, _ = _forward(wrapper, params, num_contexts=num_contexts)
    assert vision is None
    assert tower is None or tower.calls == 0


def test_the_rows_must_match_the_placeholder_rows():
    with pytest.raises(AssertionError, match="5 vision rows and 4 placeholder rows"):
        _forward(_wrapper(_tower()), [_params(image=IMAGES)], num_rows=4)


# ---------------------------------------------------------- compile scope ---


def test_torch_compile_reaches_only_the_language_models_core(monkeypatch):
    compiled = []

    def spy(module, **kwargs):
        compiled.append(module)
        return module

    monkeypatch.setattr(torch, "compile", spy)
    wrapper = _wrapper(_tower())
    core = wrapper.llm.model
    model, prefill_only = model_setup.apply_torch_compile(
        wrapper, "inductor", llm_args=SimpleNamespace(), fullgraph=False
    )
    assert model is wrapper and prefill_only is False
    assert compiled == [core]


# ---------------------------------------------------------------- weights ---


def test_the_tower_and_the_language_model_each_load_their_own_keys():
    order = []
    llm, tower = _Llm(), _tower()
    llm.load_weights = lambda weights: order.append(("llm", weights))
    tower.load_weights = lambda weights: order.append(("tower", weights))
    weights = {"model.visual.pos_embed.weight": None}
    Wrapper.load_weights(_wrapper(tower, llm), weights)
    assert order == [("tower", weights), ("llm", weights)]
    order.clear()
    Wrapper.load_weights(_wrapper(None, llm), weights)
    assert order == [("llm", weights)]
