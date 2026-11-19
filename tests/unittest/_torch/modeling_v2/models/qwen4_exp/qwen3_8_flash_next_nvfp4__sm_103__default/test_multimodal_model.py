# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default target's vision-language model on the CPU: what it declares to the shared base
(`ModelingV2ForConditionalGeneration`), its tower, the tower's admission check, its encoder hook
and the rows it hands the language model. The base itself is tested in
`tests/unittest/_torch/modeling_v2/test_modeling_v2_multimodal.py`.

The language model is never built (its core asserts sm_103): construction swaps in a stub
class, and the other tests call the model's real methods on stubs, with the real multimodal
mixin in between. Towers are stubs over the real `VisionTower.forward`, or real ones on the
meta device.
"""

from __future__ import annotations

from types import SimpleNamespace

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
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    weights as _weights,
)
from tensorrt_llm._torch.modeling_v2.multimodal import ModelingV2ForConditionalGeneration
from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
    MultimodalRuntimeData,
)

pytestmark = pytest.mark.cpu_only

Model = M.ModelingV2Qwen38FlashNextNvfp4Sm103DefaultForConditionalGeneration

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


def _params(image=None, video=None, order=None, runtime=None, mrope=True):
    data = {}
    if mrope:
        data["mrope_config"] = {"mrope_position_ids": torch.zeros(3, 1, 8, dtype=torch.int32)}
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
    """Records what the vision-language model hands its language model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.calls = []
        self.lm_head = SimpleNamespace(num_embeddings=VOCAB)
        self.multimodal_token_ids = torch.tensor([90, 91], dtype=torch.int32)

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


def _model(tower=None, llm=None):
    """A real instance around stubs, without the constructor."""
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.llm = llm if llm is not None else _Llm()
    model.mm_encoder = tower
    model.model_config = SimpleNamespace(multimodal_config=None, torch_dtype=torch.bfloat16)
    return model


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
    monkeypatch.setattr(Model, "language_model_class", _StubLlm)
    composite = _composite()
    model_config = ModelConfig(pretrained_config=composite, disable_mm_encoder=disable)
    with torch.device("meta"):
        model = Model(model_config)

    assert model.model_config is model_config
    assert model.config is composite.text_config
    assert isinstance(model.llm, _StubLlm)
    assert isinstance(model, MultimodalModelMixin)
    assert model.dtype == torch.bfloat16
    params = {name.split(".")[0] for name, _ in model.named_parameters()}
    if disable:
        assert model.mm_encoder is None and not params
    else:
        assert isinstance(model.mm_encoder, V.VisionTower)
        assert isinstance(model.mm_encoder, MultimodalEncoderMixin)
        assert model.mm_encoder.dtype == torch.bfloat16
        # Every vision weight is the tower's, and every one is read from under the prefix the
        # model hands the tower.
        assert params == {"mm_encoder"}
        assert all(".w.vis_" in name or ".w.l" in name for name, _ in model.named_parameters())
        sources = _weights.VISION_WEIGHTS.manifest(model.mm_encoder.dims).values()
        keys = {key for rows in sources for key, *_ in rows}
        assert keys and all(key.startswith(Model.mm_encoder_weight_prefix) for key in keys)


# --------------------------------------------------------------- contract ---


def test_the_class_declares_what_the_shared_base_needs():
    assert issubclass(Model, ModelingV2ForConditionalGeneration)
    assert Model.language_model_class is M.ModelingV2Qwen38FlashNextNvfp4Sm103Default
    # The base builds it as `VisionTower(vision_config, dtype)`.
    assert Model.mm_encoder_class is V.VisionTower
    assert Model.mm_encoder_weight_prefix == "model.visual."
    # The PLE layer hashes `input_ids`, so the language model places the rows itself.
    assert Model.language_model_takes_rows is True
    assert Model.supports_encoder_cache is True
    assert Model.supports_mm_encoder_item_scheduling is False
    assert Model.supported_encoder_attention_backends == {}
    assert Model.get_preferred_kv_cache_manager_version() == "V2"


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


def test_admission_runs_the_towers_media_check():
    """The base's `validate_request` checks the token range, then the tower's media check
    (`vision.validate_request`); without a tower, media is refused by the base."""
    model = _model(_GEOMETRY)
    model.validate_request(_request())
    model.validate_request(_request(image=IMAGES))
    # M-RoPE positions alone (a disaggregated generation request) are not media.
    model.validate_request(_request(mrope_config={"mrope_position_deltas": torch.zeros(1, 1)}))
    with pytest.raises(ValueError, match="one position per prompt token"):
        model.validate_request(_request(image=IMAGES, mrope_len=4))
    with pytest.raises(ValueError, match="Token ID out of range"):
        model.validate_request(_request(tokens=(1, VOCAB)))
    _model(None).validate_request(_request(image=IMAGES, mrope_len=4))


# ----------------------------------------------------------- encoder hook ---


def test_the_encoder_hook_takes_a_profiling_dummy():
    """The engine's memory profiling encodes one dummy image with no position ids and no item
    metadata, at the model's dtype: rows of the text hidden width, in that dtype."""
    tower = _tower(dtype=torch.bfloat16, width=HIDDEN)
    model = _model(tower)
    dummy = MultimodalParams(multimodal_data={"image": _media(IMAGE, [(1, 8, 8)])})
    dummy.multimodal_data["image"]["pixel_values"] = dummy.multimodal_data["image"][
        "pixel_values"
    ].to(model.dtype)
    rows = model.encode_multimodal_inputs([dummy])
    assert rows.shape == (16, HIDDEN) and rows.dtype == torch.bfloat16


def test_the_encoder_hook_returns_each_request_whole_in_prompt_order():
    tower = _tower()
    model = _model(tower)
    mixed = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    rows = model.encode_multimodal_inputs([_params(image=IMAGES), mixed])
    assert rows[:, 0].tolist() == [100.0] * 4 + [101.0] + MIXED_ROWS
    with pytest.raises(ValueError, match="require a local multimodal encoder"):
        _model(None).encode_multimodal_inputs([mixed])


# ---------------------------------------------------------------- forward ---


def _forward(model, params, num_contexts=1, num_rows=None, input_ids=None):
    input_ids = input_ids if input_ids is not None else torch.arange(24, dtype=torch.int32)
    kwargs = {"multimodal_params": params, "resource_manager": "rm"}
    if num_rows is not None:
        kwargs["mm_token_indices"] = torch.arange(1, 1 + num_rows, dtype=torch.int64)
    out = model.forward(
        SimpleNamespace(num_contexts=num_contexts, num_generations=0),
        input_ids,
        "position_ids",
        None,
        False,
        "spec_metadata",
        **kwargs,
    )
    assert out == "logits"
    ((args, llm_kwargs),) = model.llm.calls
    # `input_ids` reach the language model unchanged, and `inputs_embeds` stays None.
    assert args[1] is input_ids
    assert args[2:] == ("position_ids", None, False, "spec_metadata")
    assert "multimodal_params" not in llm_kwargs and llm_kwargs["resource_manager"] == "rm"
    return llm_kwargs["multimodal_rows"], kwargs.get("mm_token_indices")


def test_a_media_step_hands_rows_and_placeholder_rows_to_the_language_model():
    model = _model(_tower())
    multimodal_rows, indices = _forward(model, [_params(image=IMAGES)], num_rows=5)
    rows, mm_token_indices = multimodal_rows
    assert rows[:, 0].tolist() == [100.0] * 4 + [101.0]
    assert mm_token_indices is indices


def test_a_chunk_hands_over_its_slice_of_the_prompt_order_rows():
    """The mixin keeps the chunk's rows: 3 reused, 6 in this chunk."""
    tower = _tower()
    model = _model(tower)
    runtime = _runtime(len(MIXED_ROWS), cached=3, chunk_end=9)
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime)
    (rows, _), _ = _forward(model, [params], num_rows=6)
    assert rows[:, 0].tolist() == MIXED_ROWS[3:9]
