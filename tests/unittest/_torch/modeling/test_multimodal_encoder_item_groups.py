# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-request and selected-item encoding share prompt-order item metadata.

Group outputs follow modality-local payload order. Selecting items must slice
the tensors and their metadata together, so output assembly returns the same
rows and order as encoding those items in the full request.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderContractError,
    MultimodalModelMixin,
    encode_multimodal_by_groups,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLModelBase
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalEncoderRequestState,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalItemPartition,
    MultimodalParams,
    MultimodalRuntimeData,
)

HIDDEN = 4
pytestmark = pytest.mark.cpu_only
# Rows the encoder emits per patch-grid item. Distinct values catch splits that
# happen to line up only for uniform sizes.
ITEM_ROWS = {"image": [2, 3], "video": [4]}


def test_qwen_processor_produces_prompt_order_and_packed_partitions():
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    data = {
        "image": {"pixel_values": torch.zeros(8, 2), "image_grid_thw": torch.tensor([[1, 2, 4]])},
        "video": {
            "pixel_values_videos": torch.zeros(16, 2),
            "video_grid_thw": torch.tensor([[2, 2, 4]]),
        },
    }
    metadata = processor.get_mm_item_metadata([12, 11, 13, 12, 10, 13], data)
    metadata.validate()
    assert metadata.item_refs == (("video", 0), ("image", 0))
    assert metadata.encoder_token_lengths == (16, 8)
    assert metadata.output_embedding_lengths == (4, 2)
    assert metadata.input_partitions["image"]["pixel_values"].item_lengths == (8,)
    assert metadata.input_partitions["video"]["pixel_values_videos"].item_lengths == (16,)


def test_qwen_common_metadata_preserves_positive_output_requirement():
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    data = {"image": {"image_grid_thw": torch.tensor([[1, 1, 1]])}}
    with pytest.raises(ValueError, match="positive embedding lengths"):
        processor.get_mm_item_metadata([12, 10, 13], data)


@pytest.mark.parametrize("modality", ["image", "video"])
def test_qwen_precomputed_embeddings_skip_payloads_without_grids(modality: str) -> None:
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase

    processor = object.__new__(Qwen2VLInputProcessorBase)
    metadata = MultimodalItemMetadata([(modality, 0)], output_embedding_lengths=[3])
    data = {
        modality: {},
        "multimodal_embedding": torch.zeros(3, HIDDEN),
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
    }

    result = processor.get_mm_item_metadata([], data)

    assert result.item_refs == metadata.item_refs
    assert result.output_embedding_lengths == (3,)
    assert result.input_partitions == {}


def test_qwen_partition_uses_metadata_costs_without_reading_grid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.models.modeling_qwen2vl import _qwen_vision_item_partitions

    metadata = MultimodalItemMetadata([("image", 1), ("image", 0)], [32, 16], [8, 4])
    params = MultimodalParams(
        multimodal_data={
            "image": {"image_grid_thw": torch.tensor([[1, 4, 4], [2, 4, 4]])},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        }
    )

    def unexpected_grid_read(tensor: torch.Tensor) -> list:
        raise AssertionError("Known encoder costs must not be recomputed from grid values")

    monkeypatch.setattr(torch.Tensor, "tolist", unexpected_grid_read)
    partitions = _qwen_vision_item_partitions(params, "image")

    assert partitions["pixel_values"].item_lengths == (16, 32)


@pytest.mark.parametrize("model", ["qwen2", "qwen2_5", "qwen3", "mistral"])
@pytest.mark.parametrize("custom", [False, True])
def test_processors_use_the_common_metadata_hook(model, custom):
    from dataclasses import replace

    from tensorrt_llm._torch.models.modeling_mistral import MistralHFInputProcessor
    from tensorrt_llm._torch.models.modeling_qwen2vl import (
        Qwen2_5VLInputProcessorBase,
        Qwen2VLInputProcessorBase,
    )
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor_cls = {
        "qwen2": Qwen2VLInputProcessorBase,
        "qwen2_5": Qwen2_5VLInputProcessorBase,
        "qwen3": Qwen3VLInputProcessorBase,
        "mistral": MistralHFInputProcessor,
    }[model]
    calls = []

    class CustomProcessor(processor_cls):
        def get_mm_item_metadata(self, prompt_token_ids, multimodal_data, mm_item_order=None):
            calls.append(prompt_token_ids)
            metadata = super().get_mm_item_metadata(
                prompt_token_ids, multimodal_data, mm_item_order
            )
            return replace(metadata, encoder_token_lengths=[80])

    processor = object.__new__(CustomProcessor if custom else processor_cls)
    processor._config = SimpleNamespace(
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        vision_config=SimpleNamespace(
            spatial_merge_size=2, patch_size=2, image_size=8, num_channels=3
        ),
    )
    processor._processor = SimpleNamespace()
    prompt = [12, 10, 10, 13]
    image = {"pixel_values": torch.zeros(8, 2), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    if model == "mistral":
        image = {"pixel_values": torch.zeros(1, 3, 4, 8), "image_sizes": [[4, 8]]}
    metadata = processor.get_mm_item_metadata(prompt, {"image": image})
    assert calls == ([prompt] if custom else [])
    assert metadata.item_refs == (("image", 0),)
    assert metadata.output_embedding_lengths == (2,)
    assert metadata.encoder_token_lengths == ((80,) if custom else (8,))
    metadata.validate()
    if model != "mistral":
        partition = metadata.input_partitions["image"]["pixel_values"]
        assert partition.item_lengths == (8,)
        torch.testing.assert_close(
            partition.select(image["pixel_values"], [0]), image["pixel_values"]
        )


def test_gemma_unified_processor_describes_image_and_audio_without_encoder_costs():
    from tensorrt_llm._torch.models.modeling_gemma4_unified import Gemma4UnifiedInputProcessor

    processor = object.__new__(Gemma4UnifiedInputProcessor)
    data = {
        "image": {
            "pixel_values": torch.zeros(1, 4, 6),
            "image_position_ids": torch.zeros(1, 4, 2),
            "image_seq_lens": [4],
        },
        "audio": {"audio_features": torch.zeros(1, 5, 8), "audio_features_mask": torch.ones(1, 5)},
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            [("audio", 0), ("image", 0)], output_embedding_lengths=[2, 4]
        ),
    }
    order = [{"modality": "audio", "index": 0}, {"modality": "image", "index": 0}]
    metadata = processor.get_mm_item_metadata([], data, order)
    metadata.validate()
    assert metadata.item_refs == (("audio", 0), ("image", 0))
    assert metadata.encoder_token_lengths is None
    assert metadata.input_partitions["audio"]["audio_features_mask"].item_axis == 0
    assert metadata.input_partitions["image"]["image_position_ids"].item_axis == 0


@pytest.mark.parametrize(
    "modality,field,value",
    [
        ("image", "image_position_ids", torch.zeros(2, 4, 2)),
        ("image", "image_seq_lens", [4, 4]),
        ("audio", "audio_features_mask", torch.ones(2, 5)),
    ],
)
def test_gemma_unified_processor_rejects_misaligned_sibling_items(
    modality: str, field: str, value: torch.Tensor | list[int]
) -> None:
    from tensorrt_llm._torch.models.modeling_gemma4_unified import Gemma4UnifiedInputProcessor

    processor = object.__new__(Gemma4UnifiedInputProcessor)
    primary = "pixel_values" if modality == "image" else "audio_features"
    data = {
        modality: {primary: torch.zeros(1, 4, 6), field: value},
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            [(modality, 0)], output_embedding_lengths=[4]
        ),
    }

    with pytest.raises(ValueError, match=rf"{modality}\.{field} must have 1 "):
        processor.get_mm_item_metadata([], data)


def test_nemotron_audio_keeps_clip_groups_and_video_audio_relationship():
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import (
        NemotronHMultimodalInputProcessor,
    )

    processor = object.__new__(NemotronHMultimodalInputProcessor)
    audio = {
        "input_audio_features": torch.arange(30).reshape(3, 5, 2),
        "feature_attention_mask": torch.ones(3, 5),
        "audio_num_clips": torch.tensor([2, 1]),
        "audio_output_lengths": torch.tensor([1, 3, 2]),
    }
    data = {
        "audio": audio,
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            [("audio", 0), ("audio", 1)], output_embedding_lengths=[4, 2]
        ),
    }
    metadata = processor.get_mm_item_metadata([], data)
    metadata.validate()
    params = MultimodalParams(multimodal_data=data)
    selected = MultimodalModelMixin().build_multimodal_encoder_input(params, [1, 0], "audio")
    MultimodalModelMixin._apply_metadata_slice(selected, params, [1, 0])
    assert selected.item_metadata.encoder_token_lengths is None
    partition = selected.item_metadata.input_partitions["audio"]["input_audio_features"]
    assert partition.item_lengths == (1, 2)
    torch.testing.assert_close(
        selected.multimodal_data["audio"]["input_audio_features"],
        audio["input_audio_features"][[2, 0, 1]],
    )
    assert selected.multimodal_data["audio"]["audio_num_clips"].tolist() == [1, 2]
    assert selected.multimodal_data["audio"]["audio_output_lengths"].tolist() == [2, 1, 3]
    assert selected.encoder_state is None
    video = {"audio": audio, "video_size": [[2, 1, 32, 32]]}
    video_data = {
        "video": video,
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            [("video", 0)], output_embedding_lengths=[6]
        ),
    }
    video_metadata = processor.get_mm_item_metadata([], video_data)
    assert video_metadata.item_refs == (("video", 0),)
    assert "audio" not in video_metadata.input_partitions
    assert video_data["video"]["audio"] is audio


@pytest.mark.parametrize("with_mask", [False, True])
def test_gemma_audio_compacts_valid_rows_in_input_order(with_mask: bool):
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

    hidden = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    mask = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    ).repeat_interleave(4, dim=1)[:, :17]

    class Projection(torch.nn.Identity):
        embedding_projection = SimpleNamespace(weight=torch.zeros(1))

    def tower(features, attention_mask=None):
        del features
        return SimpleNamespace(
            last_hidden_state=hidden,
            attention_mask=(
                attention_mask[:, ::4]
                if attention_mask is not None
                else torch.ones(2, 5, dtype=torch.bool)
            ),
        )

    model = SimpleNamespace(model_dtype=torch.bfloat16, audio_tower=tower, embed_audio=Projection())
    actual = Gemma4ForConditionalGeneration._get_audio_features(
        model, torch.zeros(2, 17, 4), mask if with_mask else None
    )
    expected = hidden[mask[:, ::4]] if with_mask else hidden.reshape(-1, 4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("full_prompt_ids", [False, True])
def test_nemotron_audio_skips_incomplete_or_prompt_coupled_partitions(
    full_prompt_ids: bool,
) -> None:
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import (
        NemotronHMultimodalInputProcessor,
    )

    processor = object.__new__(NemotronHMultimodalInputProcessor)
    audio = {"input_audio_features": torch.zeros(1, 5, 2)}
    if full_prompt_ids:
        audio.update(audio_num_clips=torch.tensor([1]), evs_ids=torch.tensor([1, 2, 3]))
    data = {
        "audio": audio,
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            [("audio", 0)], output_embedding_lengths=[2]
        ),
    }

    metadata = processor.get_mm_item_metadata([], data)

    assert metadata.item_refs == (("audio", 0),)
    assert "audio" not in metadata.input_partitions


def _grid_for_rows(rows: int) -> list[int]:
    """A (t, h, w) grid whose product is the item's patch count."""
    return [1, 1, rows]


class _GroupedEncoderModel(MultimodalModelMixin):
    """Minimal model whose encoder serves image and video in one call.

    Mirrors the Qwen3-VL grouped encoder, including its prompt-order output
    assembly from the common item metadata.
    """

    supports_mm_encoder_item_scheduling = True

    def get_multimodal_item_partitions(self, param, modality):
        if modality == "audio":
            return {"input_features": MultimodalItemPartition()}
        return Qwen3VLModelBase.get_multimodal_item_partitions(self, param, modality)

    def __init__(self):
        self.encoder_calls: list[int] = []
        self.mm_encoder_groups = (
            EncoderGroup(
                modalities=("image", "video"),
                encoder_fn=self._encode,
                build_batched_input=self._build_batched_input,
            ),
        )

    @staticmethod
    def _build_batched_input(params: list[MultimodalParams]) -> dict:
        # Per the EncoderGroup contract: all items of the first modality across
        # requests, then all items of the second.
        pixels = []
        for modality, pixel_key in (("image", "pixel_values"), ("video", "pixel_values_videos")):
            for mp in params:
                data = mp.multimodal_data.get(modality)
                if data is not None:
                    pixels.append(data[pixel_key])
        return {"pixel_values": torch.cat(pixels, dim=0)}

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.encoder_calls.append(pixel_values.shape[0])
        # One output row per input patch row, carrying the input value so the
        # test can assert *which* rows came back, not just how many.
        return pixel_values.clone()

    def encode_multimodal_inputs(self, multimodal_params) -> torch.Tensor:
        return encode_multimodal_by_groups(self.mm_encoder_groups, list(multimodal_params))


@pytest.mark.parametrize("empty_record", [False, True])
def test_grouped_encoder_rejects_raw_payload_without_item_layout(empty_record: bool) -> None:
    model = _GroupedEncoderModel()
    data = {"image": {"pixel_values": torch.ones(2, HIDDEN)}}
    if empty_record:
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
            [], output_embedding_lengths=[]
        )
    params = MultimodalParams(multimodal_data=data)

    with pytest.raises(MultimodalEncoderContractError, match="metadata.*grouped encoding"):
        model.encode_multimodal_inputs([params])

    assert model.encoder_calls == []


@pytest.mark.parametrize(
    "mm_counts,expected_refs,expected_costs,expected_rows",
    [
        ({"image": 2}, (("image", 0), ("image", 1)), (8, 8), (2, 2)),
        ({"video": 2}, (("video", 0), ("video", 1)), (16, 16), (4, 4)),
        (
            {"video": 1, "image": 2},
            (("video", 0), ("image", 0), ("image", 1)),
            (16, 8, 8),
            (4, 2, 2),
        ),
    ],
    ids=["images", "videos", "mixed"],
)
def test_qwen_profiling_dummy_supports_grouped_encoding(
    mm_counts: dict[str, int],
    expected_refs: tuple[tuple[str, int], ...],
    expected_costs: tuple[int, ...],
    expected_rows: tuple[int, ...],
) -> None:
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    class Processor(Qwen3VLInputProcessorBase):
        def _get_dummy_grid_for_modality(
            self, modality: str, max_num_encoder_tokens: int
        ) -> tuple[int, int, int]:
            return {"image": (1, 2, 4), "video": (2, 2, 4)}[modality]

    processor = object.__new__(Processor)
    processor._config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, patch_size=1, in_channels=HIDDEN)
    )
    data = processor.get_dummy_mm_data(
        max_num_encoder_tokens=64, mm_counts=mm_counts, dtype=torch.float16
    )
    params = MultimodalParams(multimodal_data=data)
    metadata = params.item_metadata
    assert metadata is not None
    metadata.validate()
    assert metadata.item_refs == expected_refs
    assert metadata.encoder_token_lengths == expected_costs
    assert metadata.output_embedding_lengths == expected_rows

    group = EncoderGroup(
        modalities=("image", "video"),
        encoder_fn=lambda pixel_values: pixel_values[::4],
        build_batched_input=_GroupedEncoderModel._build_batched_input,
    )
    output = encode_multimodal_by_groups([group], [params])
    assert output.shape == (sum(expected_rows), HIDDEN)
    assert output.dtype == torch.float16


def test_common_metadata_shares_executor_and_encoder_layout():
    from tensorrt_llm._torch.pyexecutor.llm_request import get_multimodal_embedding_lengths

    pixels = torch.arange(5 * HIDDEN).reshape(5, HIDDEN)
    params = MultimodalParams(
        multimodal_data={
            "image": {"pixel_values": pixels},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", 1), ("image", 0)], output_embedding_lengths=[3, 2]
            ),
        }
    )
    request = SimpleNamespace(py_multimodal_data=params.multimodal_data, multimodal_lengths=[3, 2])

    assert get_multimodal_embedding_lengths(request) == (3, 2)
    actual = _GroupedEncoderModel().encode_multimodal_inputs([params])

    torch.testing.assert_close(actual, torch.cat((pixels[2:], pixels[:2])))
    assert params.item_metadata.encoder_token_lengths is None
    assert "multimodal_embedding_lengths" not in params.multimodal_data


def _make_request(modalities: list[str], *, value_start: float = 1.0) -> MultimodalParams:
    """Build a request whose items carry unique, checkable patch values."""
    per_modality_rows: dict[str, list[int]] = {}
    item_refs: list[tuple[str, int]] = []
    lengths: list[int] = []
    for modality in modalities:
        local_idx = len(per_modality_rows.setdefault(modality, []))
        rows = ITEM_ROWS[modality][local_idx % len(ITEM_ROWS[modality])]
        per_modality_rows[modality].append(rows)
        item_refs.append((modality, local_idx))
        lengths.append(rows)

    multimodal_data: dict = {}
    value = value_start
    for modality, rows_per_item in per_modality_rows.items():
        pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
        blocks = []
        for rows in rows_per_item:
            blocks.append(torch.full((rows, HIDDEN), value))
            value += 1.0
        multimodal_data[modality] = {
            pixel_key: torch.cat(blocks, dim=0),
            grid_key: torch.tensor([_grid_for_rows(r) for r in rows_per_item]),
        }
    multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        item_refs=item_refs,
        encoder_token_lengths=list(lengths),
        output_embedding_lengths=list(lengths),
    )
    return MultimodalParams(multimodal_data=multimodal_data)


def _expected_rows(request: MultimodalParams, item_idx: int) -> torch.Tensor:
    """The patch rows the encoder should return for one item."""
    metadata = request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    modality, local_idx = metadata.item_refs[item_idx]
    data = request.multimodal_data[modality]
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grids = data["image_grid_thw" if modality == "image" else "video_grid_thw"]
    counts = torch.prod(grids, dim=1).tolist()
    start = sum(int(c) for c in counts[:local_idx])
    return data[pixel_key][start : start + int(counts[local_idx])]


def _run_items(model, selected):
    encoder_inputs = model.prepare_multimodal_encoder_inputs(selected)
    return model.forward_multimodal_encoder_items(encoder_inputs), encoder_inputs


@pytest.mark.parametrize("indices", [[2, 0], [1, 2]])
def test_declared_qwen_layout_keeps_shared_fields(indices):
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image", "image"])
    shared = torch.arange(3)
    request.multimodal_data["image"]["shared_stats"] = shared
    expected = torch.cat([_expected_rows(request, i) for i in indices])

    encoder_inputs = model.prepare_multimodal_encoder_inputs([(request, i) for i in indices])
    residual = encoder_inputs[0][0]
    outputs = model.forward_multimodal_encoder_items(encoder_inputs)

    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
    assert residual.multimodal_data["image"]["shared_stats"] is shared
    if indices == [1, 2]:
        source_pixels = request.multimodal_data["image"]["pixel_values"]
        selected_pixels = residual.multimodal_data["image"]["pixel_values"]
        assert (
            selected_pixels.untyped_storage().data_ptr()
            == source_pixels.untyped_storage().data_ptr()
        )


def test_declared_audio_layout_selects_packed_frames_on_a_nonzero_axis():
    class PackedAudioModel(_GroupedEncoderModel):
        def get_multimodal_item_partitions(self, param, modality):
            return {
                "features": MultimodalItemPartition(item_axis=1, item_lengths=(2, 3, 1)),
                "mask": MultimodalItemPartition(item_lengths=(2, 3, 1)),
                "frame_counts": MultimodalItemPartition(),
                "shared_stats": MultimodalItemPartition(item_axis=None),
            }

    features = torch.arange(12).reshape(2, 6)
    mask = torch.tensor([1, 1, 1, 1, 0, 1])
    shared = torch.arange(3)
    request = MultimodalParams(
        multimodal_data={
            "audio": {
                "features": features,
                "mask": mask,
                "frame_counts": [2, 3, 1],
                "shared_stats": shared,
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("audio", 0), ("audio", 1), ("audio", 2)], [2, 3, 1], [1, 2, 1]
            ),
        }
    )

    prepared = PackedAudioModel().prepare_multimodal_encoder_inputs([(request, 2), (request, 0)])
    residual = prepared[0][0]

    torch.testing.assert_close(
        residual.multimodal_data["audio"]["features"], features[:, [5, 0, 1]]
    )
    torch.testing.assert_close(residual.multimodal_data["audio"]["mask"], mask[[5, 0, 1]])
    assert residual.multimodal_data["audio"]["frame_counts"] == [1, 2]
    assert residual.multimodal_data["audio"]["shared_stats"] is shared
    assert residual.item_metadata == MultimodalItemMetadata(
        [("audio", 0), ("audio", 1)], [1, 2], [1, 1]
    )
    assert request.item_metadata.item_refs == (("audio", 0), ("audio", 1), ("audio", 2))


def test_sliced_items_survive_the_grouped_encoder_path():
    """Selected views preserve output rows without acquiring request cache ownership."""
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image"])
    owner = MultimodalEncoderRequestState(
        item_metadata=request.item_metadata, item_ready=[False, False]
    )
    owner.set_item_cache_key(0, ("owned", 0), ready=True)
    request.encoder_state = owner
    selected = [(request, 0), (request, 1)]

    outputs, encoder_inputs = _run_items(model, selected)

    assert len(outputs) == 2
    assert request.encoder_state is owner
    assert owner.item_cache_keys == [("owned", 0), None]
    assert all(param.encoder_state is None for param, _, _ in encoder_inputs)
    for item_idx, output in enumerate(outputs):
        expected = _expected_rows(request, item_idx)
        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "modalities,item_order,expected_runs",
    [
        pytest.param(["image", "image"], [0, 1], [("image", 5)], id="adjacent-images"),
        pytest.param(
            ["video", "image", "image"],
            [0, 1, 2],
            [("video", 4), ("image", 5)],
            id="video-then-images",
        ),
        pytest.param(
            ["image", "video", "image"],
            [0, 1, 2],
            [("image", 2), ("video", 4), ("image", 3)],
            id="interleaved-vision",
        ),
        pytest.param(["image", "image"], [1, 0], [("image", 5)], id="reordered-images"),
    ],
)
def test_full_and_item_encoder_paths_share_item_metadata(
    modalities: list[str], item_order: list[int], expected_runs: list[tuple[str, int]]
) -> None:
    model = _GroupedEncoderModel()
    request = _make_request(modalities)
    original = request.item_metadata
    # Reorder prompt refs while retaining the original payload-local indices.
    request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        [original.item_refs[i] for i in item_order],
        [original.encoder_token_lengths[i] for i in item_order],
        [original.output_embedding_lengths[i] for i in item_order],
    )
    expected = [_expected_rows(request, i) for i in range(len(item_order))]
    torch.testing.assert_close(
        model.encode_multimodal_inputs([request]), torch.cat(expected), rtol=0, atol=0
    )

    model.encoder_calls.clear()
    item_outputs, encoder_inputs = _run_items(model, [(request, i) for i in range(len(item_order))])

    assert [modality for _, _, modality in encoder_inputs] == [m for m, _ in expected_runs]
    assert model.encoder_calls == [rows for _, rows in expected_runs]
    assert len(item_outputs) == len(expected)
    for actual, rows in zip(item_outputs, expected, strict=True):
        torch.testing.assert_close(actual, rows, rtol=0, atol=0)


def test_metadata_is_rebased_for_partial_and_reordered_items() -> None:
    model = _GroupedEncoderModel()
    request = _make_request(["image", "video", "image"])
    original = request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]

    outputs, encoder_inputs = _run_items(model, [(request, 2), (request, 0)])

    assert len(encoder_inputs) == 1
    residual = encoder_inputs[0][0]
    metadata = residual.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert metadata == MultimodalItemMetadata([("image", 0), ("image", 1)], [3, 2], [3, 2])
    assert residual.item_metadata.output_embedding_lengths == (3, 2)
    assert "multimodal_embedding_lengths" not in residual.multimodal_data
    assert request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] is original
    torch.testing.assert_close(outputs[0], _expected_rows(request, 2), rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], _expected_rows(request, 0), rtol=0, atol=0)


def _attach_cache_metadata(request, item_lengths):
    request.multimodal_input = MultimodalInput.from_components(
        [[i + 1] * 8 for i in range(len(item_lengths))],
        [sum(item_lengths[:i]) for i in range(len(item_lengths))],
        list(item_lengths),
    )
    request.multimodal_runtime = MultimodalRuntimeData(
        0, sum(item_lengths), torch.arange(1, sum(item_lengths) + 1, dtype=torch.int64)
    )
    request.multimodal_data["mm_processor_kwargs_hash"] = "test"


@pytest.mark.parametrize("reversed_order", [False, True])
def test_partial_cache_uses_prompt_refs_and_lengths(reversed_order):
    model = _GroupedEncoderModel()
    model._multimodal_encoder_cache = TensorLRUCache(4096)
    request = _make_request(["image", "image"])
    metadata = request.item_metadata
    if reversed_order:
        metadata = MultimodalItemMetadata([("image", 1), ("image", 0)], [3, 2], [3, 2])
        request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    lengths = metadata.output_embedding_lengths
    _attach_cache_metadata(request, lengths)
    expected = model.encode_multimodal_inputs([request])
    keys = model._encoder_cache_keys(request)
    model._multimodal_encoder_cache.put(keys[0], expected[: lengths[0]])

    actual = model._get_or_encode_multimodal_embeddings([request])

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        model._multimodal_encoder_cache.get(keys[1]), expected[lengths[0] :], rtol=0, atol=0
    )
    assert model.encoder_calls[-1] == lengths[1]
    assert request.item_metadata is metadata


def test_full_cache_write_uses_canonical_lengths_without_legacy_fields():
    model = _GroupedEncoderModel()
    model._multimodal_encoder_cache = TensorLRUCache(4096)
    request = _make_request(["image", "image"])
    lengths = request.item_metadata.output_embedding_lengths
    _attach_cache_metadata(request, lengths)
    expected = model.encode_multimodal_inputs([request])
    calls_before = len(model.encoder_calls)

    actual = model._get_or_encode_multimodal_embeddings([request])

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(model.encoder_calls) == calls_before + 1
    for key, rows in zip(
        model._encoder_cache_keys(request), torch.split(expected, lengths), strict=True
    ):
        torch.testing.assert_close(model._multimodal_encoder_cache.get(key), rows, rtol=0, atol=0)


def test_partial_cache_batches_mixed_vision_and_audio_misses():
    model = _GroupedEncoderModel()
    model.mm_encoder_groups += (
        EncoderGroup(
            ("audio",),
            model._encode,
            lambda params: {
                "pixel_values": torch.cat(
                    [p.multimodal_data["audio"]["input_features"].flatten(0, 1) for p in params]
                )
            },
        ),
    )
    model._multimodal_encoder_cache = TensorLRUCache(4096)
    request = _make_request(["image", "video", "image"])
    request.multimodal_data["audio"] = {
        "input_features": torch.stack([torch.full((1, HIDDEN), value) for value in (10.0, 20.0)])
    }
    request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        [("video", 0), ("audio", 0), ("image", 1), ("image", 0), ("audio", 1)],
        [4, 1, 3, 2, 1],
        [4, 1, 3, 2, 1],
    )
    lengths = request.item_metadata.output_embedding_lengths
    _attach_cache_metadata(request, lengths)
    expected = model.encode_multimodal_inputs([request])
    expected_items = torch.split(expected, lengths)
    keys = model._encoder_cache_keys(request)
    for i in (0, 3):
        model._multimodal_encoder_cache.put(keys[i], expected_items[i])
    calls_before = len(model.encoder_calls)

    actual = model._get_or_encode_multimodal_embeddings([request])

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Image misses share one vision call; audio misses share one audio call.
    assert model.encoder_calls[calls_before:] == [3, 2]
    for key, rows in zip(keys, expected_items, strict=True):
        torch.testing.assert_close(model._multimodal_encoder_cache.get(key), rows, rtol=0, atol=0)


def test_alternating_item_selection_reuses_validated_request_metadata(monkeypatch):
    model = _GroupedEncoderModel()
    requests = [_make_request(["image", "video"] * 4) for _ in range(2)]
    validated = []
    validate = MultimodalItemMetadata.validate

    def record_validation(metadata):
        validated.append(metadata)
        validate(metadata)

    monkeypatch.setattr(MultimodalItemMetadata, "validate", record_validation)

    model.prepare_multimodal_encoder_inputs(
        [(request, i) for request in requests for i in range(8)]
    )

    assert not validated


@pytest.mark.parametrize("shared_vision_encoder", [False, True])
def test_vision_and_audio_groups_share_full_and_item_ordering(shared_vision_encoder: bool) -> None:
    model = _GroupedEncoderModel()

    def build_vision_input(
        params: list[MultimodalParams], modality: str
    ) -> dict[str, torch.Tensor]:
        pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        return {
            "pixel_values": torch.cat([mp.multimodal_data[modality][pixel_key] for mp in params])
        }

    if not shared_vision_encoder:
        model.mm_encoder_groups = tuple(
            EncoderGroup((modality,), model._encode, partial(build_vision_input, modality=modality))
            for modality in ("image", "video")
        )

    def build_audio_input(params: list[MultimodalParams]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.cat(
                [mp.multimodal_data["audio"]["input_features"].flatten(0, 1) for mp in params]
            )
        }

    model.mm_encoder_groups += (EncoderGroup(("audio",), model._encode, build_audio_input),)
    request = _make_request(["video", "image"])
    audio = torch.stack([torch.full((2, HIDDEN), value) for value in (10.0, 20.0)])
    request.multimodal_data["audio"] = {"input_features": audio}
    request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        [("video", 0), ("audio", 0), ("image", 0), ("audio", 1)], [4, 2, 2, 2], [4, 2, 2, 2]
    )
    expected = torch.cat(
        [_expected_rows(request, 0), audio[0], _expected_rows(request, 2), audio[1]]
    )

    full_output = model.encode_multimodal_inputs([request])
    item_outputs, _ = _run_items(model, [(request, i) for i in range(4)])

    torch.testing.assert_close(full_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(item_outputs), expected, rtol=0, atol=0)


def test_partial_selection_only_encodes_selected_items():
    """The scheduler may pick a subset (the rest being cache hits or deferred);
    unselected items must not leak into the encoder input."""
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image"])

    outputs, _ = _run_items(model, [(request, 1)])

    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], _expected_rows(request, 1))
    assert model.encoder_calls == [ITEM_ROWS["image"][1]]


def test_wrong_encoder_output_rows_are_request_contract_error():
    class _WrongRowsModel(_GroupedEncoderModel):
        def encode_multimodal_inputs(self, multimodal_params):
            output = super().encode_multimodal_inputs(multimodal_params)
            return output[:-1]

    model = _WrongRowsModel()
    request = _make_request(["image"])
    encoder_inputs = model.prepare_multimodal_encoder_inputs([(request, 0)])

    with pytest.raises(MultimodalEncoderContractError, match="rows declared"):
        model.forward_multimodal_encoder_items(encoder_inputs)


def test_items_from_multiple_requests_stay_separated():
    model = _GroupedEncoderModel()
    first = _make_request(["image"])
    second = _make_request(["image"], value_start=9.0)

    outputs, encoder_inputs = _run_items(model, [(first, 0), (second, 0)])

    assert len(encoder_inputs) == 2
    assert model.encoder_calls == [2 * ITEM_ROWS["image"][0]]
    torch.testing.assert_close(outputs[0], _expected_rows(first, 0))
    torch.testing.assert_close(outputs[1], _expected_rows(second, 0))


def test_request_cache_input_is_built_once_and_shared_with_forward_views():
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import _build_request_multimodal_input

    class Request:
        def __init__(self):
            self.py_multimodal_params = _make_request(["image", "image"])
            self.hash_reads = 0
            self.multimodal_positions = [1, 5]
            self.multimodal_lengths = [2, 3]
            self.multimodal_item_run_cu_offsets = None
            self.multimodal_run_positions = None
            self.multimodal_run_lengths = None

        @property
        def multimodal_hashes(self):
            self.hash_reads += 1
            return [[1] * 8, [2] * 8]

    request = Request()
    assert _build_request_multimodal_input(request, False) is None
    assert request.hash_reads == 0

    first = _build_request_multimodal_input(request, True)
    second = _build_request_multimodal_input(request, True)
    assert first is second is request.py_multimodal_params.multimodal_input
    assert request.hash_reads == 1

    model = _GroupedEncoderModel()
    residual = model.build_multimodal_encoder_input(request.py_multimodal_params, [1], "image")
    model._apply_metadata_slice(residual, request.py_multimodal_params, [1])
    assert residual.multimodal_input.multimodal_hashes == [[2] * 8]
    assert first.multimodal_hashes == [[1] * 8, [2] * 8]
