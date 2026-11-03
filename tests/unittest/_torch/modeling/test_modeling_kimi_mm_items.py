# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoonViT encoder items of Kimi K2.5 and K3: geometry, item metadata, dummy data, selection.

The processor tests stub the HF image processor's ``media_proc_cfg``, and the model tests use
unbuilt model instances, so all but the last two tests run on CPU without a checkpoint. The
last two run the real K3 and K2.5 processors through the LLM API's hashing wrapper when
``LLM_MODELS_ROOT`` holds their checkpoints.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.modeling_kimi_k3_vl import (
    KimiK3ForConditionalGeneration,
    KimiK3InputProcessor,
    KimiK31ForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_kimi_k25 import (
    KimiK25ForConditionalGeneration,
    KimiK25InputProcessor,
    KimiK25VisionModel,
    Learnable2DPosEmb,
    MoonViT3dEncoder,
    Rope2D,
    _navit_max_patches,
    _tpool_patch_merger,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalEncoderContractError
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    mm_item_scheduling_enabled,
    setup_mm_encoder_attn_metadata,
)
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy

pytestmark = pytest.mark.cpu_only

_PATCH = 14


def _processor(cls=KimiK3InputProcessor, *, in_patch_limit=65536, frame_patch_limit=16384):
    """Build an input processor around a stub HF image processor."""
    processor = cls.__new__(cls)
    processor._config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=_PATCH, merge_kernel_size=(2, 2))
    )
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            media_proc_cfg={
                "in_patch_limit": in_patch_limit,
                "patch_size": _PATCH,
                "merge_kernel_size": 2,
                "patch_limit_on_one_side": 512,
                "in_patch_limit_each_frame": frame_patch_limit,
                "temporal_merge_kernel_size": 4,
            }
        )
    )
    processor._dtype = torch.bfloat16
    return processor


def _k25_processor():
    return _processor(KimiK25InputProcessor, in_patch_limit=16384, frame_patch_limit=4096)


def _model(cls=KimiK3ForConditionalGeneration):
    """Return an unbuilt model; the item hooks read only their arguments and ``mm_encoder``."""
    return cls.__new__(cls)


def _navit_patches(width, height, *, in_patch_limit, side=512, merge=2):
    """Patches of the grid the HF ``navit_resize_image`` gives an image."""
    s1 = math.sqrt(in_patch_limit / (max(1.0, width // _PATCH) * max(1.0, height // _PATCH)))
    scale = min(1.0, s1, side * _PATCH / width, side * _PATCH / height)
    new_w = min(max(1, int(width * scale)), side * _PATCH)
    new_h = min(max(1, int(height * scale)), side * _PATCH)
    factor = merge * _PATCH
    return merge * math.ceil(new_h / factor) * merge * math.ceil(new_w / factor)


def _payload(*grids, pixel_rows=None):
    """Packed MoonViT media whose pixel rows count up, so selections can be compared."""
    if pixel_rows is None:
        pixel_rows = sum(t * h * w for t, h, w in grids)
    return {
        "pixel_values": torch.arange(pixel_rows, dtype=torch.float32)
        .view(-1, 1, 1, 1)
        .expand(-1, 3, _PATCH, _PATCH),
        "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
    }


def _params(processor, payload):
    mm_data = {"image": payload}
    mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = processor.get_mm_item_metadata([], mm_data)
    return MultimodalParams(multimodal_data=mm_data)


# ---------------------------------------------------------------------------
# Processor geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_patch_limit, max_patches", [(65536, 67320), (16384, 17856)])
def test_max_image_follows_the_loaded_patch_limit(in_patch_limit, max_patches):
    processor = _processor(in_patch_limit=in_patch_limit)

    assert processor.get_mm_max_tokens_per_item() == {"image": max_patches}


def test_padding_takes_real_images_past_the_patch_limit():
    assert _navit_patches(3600, 3600, in_patch_limit=65536) == 66564
    assert _navit_patches(1889, 7377, in_patch_limit=65536) == 67320
    assert _navit_max_patches(_PATCH, 2, 65536, 512) == 67320


def test_max_image_bounds_resized_images():
    bound = _navit_max_patches(_PATCH, 2, 65536, 512)
    sizes = list(range(1, 400, 7)) + list(range(400, 30000, 97))

    assert max(_navit_patches(w, h, in_patch_limit=65536) for w in sizes for h in sizes) <= bound


def test_budget_selects_the_largest_legal_image():
    processor = _processor()

    assert processor.get_mm_max_tokens_per_item(32768) == {"image": 32768}
    assert processor.get_mm_max_tokens_per_item(10**6) == {"image": 67320}
    # Grid sides are multiples of the merge kernel.
    assert processor.get_mm_max_tokens_per_item(4099) == {"image": 4096}
    assert processor.get_mm_max_tokens_per_item(3) == {}


def test_only_k25_reports_video_as_whole_chunks_of_the_largest_frames():
    processor = _k25_processor()

    assert processor.get_mm_max_tokens_per_item() == {"image": 17856, "video": 4 * 5352}
    assert processor.get_mm_max_tokens_per_item(12000) == {"image": 12000, "video": 2 * 5352}
    assert "video" not in _processor().get_mm_max_tokens_per_item()


def test_output_and_attention_capacity_follow_the_merge_kernel():
    processor = _processor()

    assert processor.get_max_mm_encoder_output_embeddings(67320) == 16830
    assert processor.get_mm_encoder_attention_metadata_capacity(67320) == {"attention": 16830}


# ---------------------------------------------------------------------------
# Item metadata
# ---------------------------------------------------------------------------


def test_item_metadata_for_one_and_two_images():
    processor = _processor()

    one = processor.get_mm_item_metadata([], {"image": _payload((1, 400, 128))})
    assert one.item_refs == (("image", 0),)
    assert one.encoder_token_lengths == (51200,)
    assert one.output_embedding_lengths == (12800,)

    two = processor.get_mm_item_metadata([], {"image": _payload((1, 400, 128), (1, 72, 102))})
    two.validate()
    assert two.item_refs == (("image", 0), ("image", 1))
    assert two.encoder_token_lengths == (51200, 7344)
    assert two.output_embedding_lengths == (12800, 1836)


def test_item_metadata_follows_the_prompt_item_order():
    processor = _processor()
    mm_data = {"image": _payload((1, 4, 6), (1, 8, 4))}
    order = [{"modality": "image", "index": 1}, {"modality": "image", "index": 0}]

    metadata = processor.get_mm_item_metadata([], mm_data, order)
    assert metadata.item_refs == (("image", 1), ("image", 0))
    assert metadata.encoder_token_lengths == (32, 24)
    assert metadata.output_embedding_lengths == (8, 6)
    with pytest.raises(ValueError, match="do not match the processed Kimi media"):
        processor.get_mm_item_metadata([], mm_data, order[:1])


def test_item_metadata_leaves_attached_embeddings_alone():
    processor = _processor()

    assert processor.get_mm_item_metadata([], {"multimodal_embedding": torch.zeros(4, 8)}) is None


@pytest.mark.parametrize("grid", [(1, 514, 2), (1, 6, 5), (2, 4, 4), (1, 512, 512)])
def test_item_metadata_rejects_grids_outside_the_processor_geometry(grid):
    with pytest.raises(ValueError, match="outside the startup processor geometry"):
        _processor().get_mm_item_metadata([], {"image": _payload(grid)})


def test_item_metadata_rejects_pixels_that_do_not_match_the_grids():
    with pytest.raises(ValueError, match="pixel_values holds 23 patches"):
        _processor().get_mm_item_metadata([], {"image": _payload((1, 4, 6), pixel_rows=23)})


def test_k25_video_item_spans_its_chunks():
    payload = _payload((1, 32, 32), (4, 16, 16), (2, 16, 16), (1, 8, 8))
    payload["video_chunk_counts"] = [2, 1]

    metadata = _k25_processor().get_mm_item_metadata([], {"image": payload})
    assert metadata.item_refs == (("image", 0), ("video", 0), ("video", 1))
    assert metadata.encoder_token_lengths == (1024, 6 * 256, 64)
    # The merger pools the frames of a chunk, so each chunk yields one frame of embeddings.
    assert metadata.output_embedding_lengths == (256, 2 * 64, 16)


# ---------------------------------------------------------------------------
# Profiling dummy data
# ---------------------------------------------------------------------------


def test_dummy_data_matches_its_declared_item_lengths():
    processor = _processor()

    mm_data = processor.get_dummy_mm_data(
        max_num_encoder_tokens=67320, mm_counts={"image": 1}, dtype=torch.bfloat16
    )
    image = mm_data["image"]
    metadata = mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    [(t, h, w)] = image["image_grid_thw"].tolist()
    assert metadata.item_refs == (("image", 0),)
    assert metadata.encoder_token_lengths == (t * h * w,) == (67320,)
    assert metadata.output_embedding_lengths == ((h // 2) * (w // 2),)
    assert image["pixel_values"].shape == (67320, 3, _PATCH, _PATCH)
    assert image["pixel_values"].dtype == torch.bfloat16
    assert processor.get_mm_item_metadata([], mm_data) == metadata


def test_dummy_data_holds_the_requested_items_within_the_budget():
    processor = _processor(in_patch_limit=4096)

    mm_data = processor.get_dummy_mm_data(max_num_encoder_tokens=11000, mm_counts={"image": 2})
    assert mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY].encoder_token_lengths == (5352, 5352)
    assert mm_data["image"]["pixel_values"].shape[0] == 2 * 5352
    with pytest.raises(ValueError, match="exceed max_num_encoder_tokens"):
        processor.get_dummy_mm_data(max_num_encoder_tokens=10000, mm_counts={"image": 2})
    with pytest.raises(ValueError, match="nonnegative"):
        processor.get_dummy_mm_data(max_num_encoder_tokens=1024, mm_counts={"image": -1})
    with pytest.raises(ValueError, match="cannot build dummy data"):
        processor.get_dummy_mm_data(max_num_encoder_tokens=1024, mm_counts={"video": 1})


def test_k25_dummy_video_is_chunked_like_the_processor():
    mm_data = _k25_processor().get_dummy_mm_data(
        max_num_encoder_tokens=6 * 5352, mm_counts={"video": 1}
    )
    image = mm_data["image"]
    metadata = mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert image["video_chunk_counts"] == [2]
    assert [row[0] for row in image["image_grid_thw"].tolist()] == [4, 2]
    assert metadata.item_refs == (("video", 0),)
    assert metadata.encoder_token_lengths == (6 * 5352,)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_dummy_grids_pass_moonvit_position_and_merge_steps(modality):
    mm_data = _k25_processor().get_dummy_mm_data(
        max_num_encoder_tokens=4 * 5352, mm_counts={modality: 1}
    )
    grid_thws = mm_data["image"]["image_grid_thw"]
    hidden = torch.zeros(mm_data["image"]["pixel_values"].shape[0], 8)

    assert Learnable2DPosEmb(64, 64, 4, 8)(hidden, grid_thws).shape == hidden.shape
    assert Rope2D(8).get_freqs_cis(grid_thws, hidden.device).shape == (hidden.shape[0], 4)
    merged = _tpool_patch_merger(hidden, grid_thws, (2, 2))
    assert sum(cells.shape[0] for cells in merged) == sum(
        mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY].output_embedding_lengths
    )


# ---------------------------------------------------------------------------
# Model item hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls, enabled",
    [
        (KimiK3ForConditionalGeneration, True),
        (KimiK31ForConditionalGeneration, True),
        (KimiK25ForConditionalGeneration, False),
    ],
)
def test_item_scheduling_capability(model_cls, enabled):
    llm_args = SimpleNamespace(
        disable_mm_encoder=False,
        pipeline_parallel_size=1,
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT
        ),
    )

    assert mm_item_scheduling_enabled(llm_args, _model(model_cls)) is enabled


def test_selecting_one_of_two_images_yields_its_patches_and_grid():
    payload = _payload((1, 4, 6), (1, 8, 4))
    param = _params(_processor(), payload)

    [(residual, lengths, modality)] = _model().prepare_multimodal_encoder_inputs([(param, 1)])
    image = residual.multimodal_data["image"]
    assert modality == "image"
    assert lengths == (8,)
    assert image["image_grid_thw"].tolist() == [[1, 8, 4]]
    assert torch.equal(image["pixel_values"], payload["pixel_values"][24:56])
    assert residual.item_metadata.item_refs == (("image", 0),)
    assert residual.item_metadata.encoder_token_lengths == (32,)

    [(residual, lengths, _)] = _model().prepare_multimodal_encoder_inputs([(param, 1), (param, 0)])
    image = residual.multimodal_data["image"]
    assert lengths == (8, 6)
    assert image["image_grid_thw"].tolist() == [[1, 8, 4], [1, 4, 6]]
    assert torch.equal(
        image["pixel_values"],
        torch.cat([payload["pixel_values"][24:56], payload["pixel_values"][:24]]),
    )


def test_selecting_a_video_keeps_all_of_its_chunks():
    payload = _payload((1, 4, 4), (4, 2, 2), (2, 2, 2), (1, 2, 4))
    payload["video_chunk_counts"] = [2, 1]
    param = _params(_k25_processor(), payload)

    [(residual, lengths, modality)] = _model(
        KimiK25ForConditionalGeneration
    ).prepare_multimodal_encoder_inputs([(param, 1)])
    image = residual.multimodal_data["image"]
    assert modality == "video"
    assert lengths == (2,)
    assert image["image_grid_thw"].tolist() == [[4, 2, 2], [2, 2, 2]]
    assert image["video_chunk_counts"] == [2]
    assert torch.equal(image["pixel_values"], payload["pixel_values"][16:40])
    assert residual.item_metadata.item_refs == (("video", 0),)


class _FakeMoonViT:
    """Emit one row per merged cell, valued ``100 * h + w`` of its grid row."""

    def forward(self, multimodal_params):
        rows = [
            torch.full(((h // 2) * (w // 2), 4), float(100 * h + w))
            for param in multimodal_params
            for _, h, w in param.multimodal_data["image"]["image_grid_thw"].tolist()
        ]
        return [torch.cat(rows)] if rows else []


def test_item_encoder_outputs_follow_the_selected_items():
    model = _model()
    model.mm_encoder = _FakeMoonViT()
    param = _params(_processor(), _payload((1, 4, 6), (1, 8, 4)))

    outputs = model.forward_multimodal_encoder_items(
        model.prepare_multimodal_encoder_inputs([(param, 1), (param, 0)])
    )
    assert [output.shape[0] for output in outputs] == [8, 6]
    assert outputs[0].unique().tolist() == [804.0]
    assert outputs[1].unique().tolist() == [406.0]

    model.mm_encoder = SimpleNamespace(forward=lambda multimodal_params: [])
    with pytest.raises(MultimodalEncoderContractError):
        model.encode_multimodal_inputs([param])


def _tower():
    """The parts of a built MoonViT tower that ``validate_multimodal_data`` reads."""
    return SimpleNamespace(
        merge_kernel_size=(2, 2),
        patch_embed=SimpleNamespace(
            pos_emb=SimpleNamespace(num_frames=4),
            proj=torch.nn.Conv2d(3, 8, kernel_size=_PATCH, stride=_PATCH),
        ),
        encoder=SimpleNamespace(rope_2d=Rope2D(8)),
    )


@pytest.mark.parametrize("grid", [(5, 2, 2), (1, 514, 2), (1, 2, 3)])
def test_tower_rejects_grids_it_cannot_encode(grid):
    tower = _tower()
    KimiK25VisionModel.validate_multimodal_data(tower, {"image": _payload((4, 512, 2))})

    with pytest.raises(ValueError, match="MoonViT cannot encode"):
        KimiK25VisionModel.validate_multimodal_data(tower, {"image": _payload(grid)})


def test_tower_rejects_pixels_that_do_not_match_the_grids():
    with pytest.raises(ValueError, match="pixel_values must have shape"):
        KimiK25VisionModel.validate_multimodal_data(
            _tower(), {"image": _payload((1, 4, 6), pixel_rows=25)}
        )


class _Metadata:
    """Stand-in for the backend metadata, which needs CUDA to build."""

    def __init__(self, *, max_num_requests, max_num_tokens, kv_cache_manager):
        self.max_num_requests = max_num_requests
        self.max_num_tokens = max_num_tokens

    def prepare(self):
        pass


def _moonvit_encoder():
    """A MoonViT encoder without blocks; only its attention metadata handling runs."""
    encoder = MoonViT3dEncoder.__new__(MoonViT3dEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.metadata_cls = _Metadata
    return encoder


def test_moonvit_attention_keeps_one_max_seq_len_across_batches():
    # The native attention op is cached by max_seq_len, so every batch must present the same one.
    model = torch.nn.Module()
    model.encoder = _moonvit_encoder()
    processor = _processor()
    setup_mm_encoder_attn_metadata(
        model, processor, 67320, processor.get_mm_encoder_attention_metadata_capacity(67320)
    )

    metadata = model.encoder.attn_metadata
    assert (metadata.max_num_requests, metadata.max_num_tokens) == (16830, 67320)
    for seq_lens in ([84], [4096, 7344], [67320]):
        assert model.encoder.prepare_attn_metadata(seq_lens, metadata).max_seq_len == 67320
    with pytest.raises(ValueError, match="must be positive"):
        model.encoder.set_attn_max_seq_len(0)


def test_moonvit_attention_without_engine_setup_uses_the_longest_sequence():
    encoder = _moonvit_encoder()
    metadata = _Metadata(max_num_requests=8, max_num_tokens=1024, kv_cache_manager=None)

    assert encoder.prepare_attn_metadata([84, 300], metadata).max_seq_len == 300


class _PilOnlyImageProcessor:
    """Count tokens like the HF processor, which rejects anything but PIL images."""

    media_proc_cfg = {"temporal_merge_kernel_size": 4}

    def media_tokens_calculator(self, media):
        images = [media["image"]] if media["type"] == "image" else media["video_chunk"]
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported data type: {type(image)}")
        width, height = images[0].size
        return (height // 28) * (width // 28)


def test_tensor_images_and_frames_count_like_pil_images():
    # The serving media loader yields CHW float tensors.
    processor = _k25_processor()
    processor._processor = SimpleNamespace(image_processor=_PilOnlyImageProcessor())
    image = torch.rand(3, 56, 84)

    assert processor.get_num_tokens_per_image(image=image) == 6
    assert processor.get_num_tokens_per_video(video=[image] * 6) == 2 * 6


# ---------------------------------------------------------------------------
# Real processors through the LLM API's hashing wrapper
# ---------------------------------------------------------------------------

_MODELS_ROOT = llm_models_root()


def _checkpoint(name):
    path = None if _MODELS_ROOT is None else _MODELS_ROOT / name
    return pytest.mark.skipif(
        path is None or not path.is_dir(), reason=f"needs the {name} checkpoint"
    ), path


_NEEDS_K3, _KIMI_K3_DIR = _checkpoint("Kimi-K3")
_NEEDS_K25, _KIMI_K25_DIR = _checkpoint("Kimi-K2.5-NVFP4")


def _process(model_dir, prompt, multi_modal_data):
    """Run the checkpoint's processor the way the LLM API does, with multimodal hashing."""
    from transformers import AutoTokenizer

    from tensorrt_llm.inputs import create_input_processor, create_input_processor_with_hash
    from tensorrt_llm.sampling_params import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = create_input_processor(str(model_dir), tokenizer=tokenizer)
    token_ids, extra = create_input_processor_with_hash(processor)(
        {"prompt": prompt, "multi_modal_data": multi_modal_data}, SamplingParams()
    )
    # Hashing falls back silently on an error, so require its output.
    assert extra.get("multimodal_input") is not None
    mm_data = extra["multimodal_data"]
    metadata = mm_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    grids = mm_data["image"]["image_grid_thw"].tolist()
    assert sum(metadata.output_embedding_lengths) == token_ids.count(
        int(processor.get_mm_token_ids()[0])
    )
    return processor, mm_data, metadata, grids


@_NEEDS_K3
@pytest.mark.parametrize("image_format", ["pil", "pt"])
def test_k3_processor_items_match_the_prompt_and_the_reported_maximum(image_format):
    images = [Image.new("RGB", (224, 448)), Image.new("RGB", (3600, 3600))]
    if image_format == "pt":
        # trtllm-serve's default media format: CHW float tensors in [0, 1].
        images = [pil_to_tensor(image).float() / 255 for image in images]
    processor, _, metadata, grids = _process(
        _KIMI_K3_DIR,
        "<|kimi_image_placeholder|><|kimi_image_placeholder|>Describe both.",
        {"image": images},
    )

    assert metadata.item_refs == (("image", 0), ("image", 1))
    assert metadata.encoder_token_lengths == tuple(t * h * w for t, h, w in grids)
    assert metadata.output_embedding_lengths == tuple(
        processor.get_num_tokens_per_image(image=image) for image in images
    )
    # Padding takes the large image past in_patch_limit, but not past the reported maximum.
    max_image = processor.get_mm_max_tokens_per_item()["image"]
    assert 65536 < metadata.encoder_token_lengths[1] <= max_image


@_NEEDS_K25
def test_k25_processor_items_cover_an_image_and_a_chunked_video():
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    frames = [Image.new("RGB", (224, 224), color=(40 * i, 0, 0)) for i in range(6)]
    processor, mm_data, metadata, grids = _process(
        _KIMI_K25_DIR,
        "<|media_begin|><|media_pad|><|media_end|><|kimi_k25_video_placeholder|>Describe both.",
        {"image": [image], "video": [frames]},
    )

    # The 6 frames form chunks of 4 and 2 frames; the video is one item over both.
    assert mm_data["image"]["video_chunk_counts"] == [2]
    assert [t for t, _, _ in grids] == [1, 4, 2]
    assert metadata.item_refs == (("image", 0), ("video", 0))
    assert metadata.encoder_token_lengths == (
        grids[0][1] * grids[0][2],
        sum(t * h * w for t, h, w in grids[1:]),
    )
    assert metadata.output_embedding_lengths[0] == processor.get_num_tokens_per_image(image=image)
