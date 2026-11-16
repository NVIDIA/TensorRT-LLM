# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VLInputProcessorBase,
    _expand_prompt_token_ids_for_mm_handoff,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)
from tensorrt_llm.inputs.registry import (
    BaseMultimodalInputProcessor,
    create_input_processor_with_hash,
)
from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


_VISION_START_TOKEN_ID = 151652
_VISION_END_TOKEN_ID = 151653
_IMAGE_TOKEN_ID = 151655
_VIDEO_TOKEN_ID = 151656


class _FakeTokenizer:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __call__(self, prompt, return_tensors=None):
        assert prompt == "video prompt"
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.input_ids]))


def test_qwen3vl_disagg_video_prompt_expands_video_placeholder():
    """Video placeholder expands to one token per encoder embedding."""
    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        text_config=SimpleNamespace(hidden_size=16),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
    )
    processor._tokenizer = _FakeTokenizer(
        [11, _VISION_START_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_END_TOKEN_ID, 12]
    )

    prompt_token_ids, params = processor.build_disagg_prefill_multimodal_inputs(
        {"prompt": "video prompt"},
        [{"tensor_size": (3, 16)}],
        mm_hashes=[[1] * 8],
    )

    # In-vocab contract: the video placeholder expands to the real
    # _VIDEO_TOKEN_ID (no legacy OOV remap).
    assert prompt_token_ids == [
        11,
        _VISION_START_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        12,
    ]
    assert params.multimodal_input.multimodal_lengths == [4]
    assert params.multimodal_input.multimodal_positions == [1]
    assert params.item_metadata.item_refs == (("video", 0),)
    assert params.item_metadata.output_embedding_lengths == (3,)
    assert params.multimodal_input.multimodal_item_run_cu_offsets == [0, 1]
    assert params.multimodal_input.multimodal_run_positions == [1]
    assert params.multimodal_input.multimodal_run_lengths == [4]
    assert params.multimodal_data["special_token_offsets"] == [0]


def test_qwen3vl_disagg_mixed_prompt_layout_preserves_item_order():
    """Image and video handoff metadata follows prompt order."""
    # NOTE: This locks in helper behavior—but it doesn't map to a supported runtime path today
    # because Qwen3-VL EPD is single-modality only as of adding this test.
    input_ids = torch.tensor(
        [
            _VISION_START_TOKEN_ID,
            _IMAGE_TOKEN_ID,
            _VISION_END_TOKEN_ID,
            42,
            _VISION_START_TOKEN_ID,
            _VIDEO_TOKEN_ID,
            _VISION_END_TOKEN_ID,
        ]
    )

    prompt_token_ids, params = _expand_prompt_token_ids_for_mm_handoff(
        input_ids,
        [{"tensor_size": (2, 16)}, {"tensor_size": (3, 16)}],
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        mm_hashes=[[1] * 8, [2] * 8],
    )

    # In-vocab contract: placeholders expand to the real image / video token ids.
    assert prompt_token_ids == [
        _VISION_START_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        42,
        _VISION_START_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VISION_END_TOKEN_ID,
    ]
    assert params.multimodal_input.multimodal_lengths == [3, 4]
    assert params.multimodal_input.multimodal_positions == [0, 5]
    assert params.multimodal_input.multimodal_item_run_cu_offsets == [0, 1, 2]
    assert params.multimodal_input.multimodal_run_positions == [0, 5]
    assert params.multimodal_input.multimodal_run_lengths == [3, 4]
    assert params.item_metadata.item_refs == (("image", 0), ("video", 0))
    assert params.item_metadata.output_embedding_lengths == (2, 3)
    assert params.multimodal_data["special_token_offsets"] == [0, 3]


def test_qwen3vl_disagg_prompt_layout_requires_handle_per_placeholder():
    """Each prompt placeholder must have a matching MM handle."""
    with pytest.raises(ValueError, match="placeholders=1, mm_handles=0"):
        _expand_prompt_token_ids_for_mm_handoff(
            torch.tensor([1, 2, 3]),
            [],
            image_token_id=2,
            video_token_id=4,
            vision_start_token_id=1,
        )


@pytest.mark.parametrize("model", ["qwen2.5", "llava", "kimi"])
def test_disagg_image_processors_return_common_params(model):
    from tensorrt_llm._torch.models.modeling_kimi_k25 import KimiK25InputProcessor
    from tensorrt_llm._torch.models.modeling_llava_next import LlavaNextInputProcessor
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5VLInputProcessorBase

    processor_cls = {
        "qwen2.5": Qwen2_5VLInputProcessorBase,
        "llava": LlavaNextInputProcessor,
        "kimi": KimiK25InputProcessor,
    }[model]
    processor = object.__new__(processor_cls)
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_TOKEN_ID,
        image_token_index=_IMAGE_TOKEN_ID,
        text_config=SimpleNamespace(hidden_size=16),
    )
    processor._media_placeholder_token_id = _IMAGE_TOKEN_ID
    processor._tokenizer = _FakeTokenizer([11, _IMAGE_TOKEN_ID, 12, _IMAGE_TOKEN_ID, 13])
    handles = [{"tensor_size": (2, 16)}, {"tensor_size": (3, 16)}]
    hashes = [[1] * 8, [2] * 8]

    tokens, params = processor.build_disagg_prefill_multimodal_inputs(
        {"prompt": "video prompt"}, handles, mm_hashes=hashes
    )

    assert tokens == [11, _IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, 12, *([_IMAGE_TOKEN_ID] * 3), 13]
    assert isinstance(params, MultimodalParams)
    assert params.multimodal_data["multimodal_embedding"] is handles
    assert params.get_item_layout() == ((("image", 0), ("image", 1)), (2, 3))
    assert params.item_metadata.encoder_token_lengths is None
    assert params.multimodal_input.multimodal_hashes is hashes
    assert params.multimodal_input.multimodal_positions == [1, 4]
    assert params.multimodal_input.multimodal_lengths == [2, 3]


@pytest.mark.parametrize("deepstack_levels", [0, 3])
def test_disagg_preprocess_keeps_processor_params_and_mrope(monkeypatch, deepstack_levels):
    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._processor = SimpleNamespace()
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        text_config=SimpleNamespace(hidden_size=16, vocab_size=151936),
        vision_config=SimpleNamespace(deepstack_visual_indexes=list(range(deepstack_levels))),
    )
    processor._tokenizer = _FakeTokenizer(
        [11, _VISION_START_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_END_TOKEN_ID, 12]
    )
    llm = object.__new__(BaseLLM)
    llm.input_processor = processor
    llm._hf_model_config = SimpleNamespace(is_encoder_decoder=False)
    handles = [{"tensor_size": (3, 16 * (1 + deepstack_levels))}]
    mrope_ids = {"position_ids": "handle", "tensor_size": [3, 1, 7]}
    mrope_deltas = {"position_deltas": "handle", "tensor_size": [1, 1]}
    disagg = DisaggregatedParams(
        request_type="context_and_generation",
        multimodal_embedding_handles=handles,
        multimodal_hashes=[[1] * 8],
        mrope_position_ids_handle=mrope_ids,
        mrope_position_deltas_handle=mrope_deltas,
    )
    build = processor.build_disagg_prefill_multimodal_inputs
    built = []

    def capture(*args, **kwargs):
        result = build(*args, **kwargs)
        built.append(result[1])
        return result

    monkeypatch.setattr(processor, "build_disagg_prefill_multimodal_inputs", capture)
    tokens, prompt, params, encoder_tokens = llm._preprocess(
        {"prompt": "video prompt"}, SamplingParams(), disagg
    )

    assert params is built[0]
    assert prompt == "video prompt"
    assert encoder_tokens is None
    assert params.get_item_layout() == ((("video", 0),), (3,))
    assert params.multimodal_data["multimodal_embedding"] is handles
    assert params.multimodal_data["mrope_config"]["mrope_position_ids"] is mrope_ids
    assert params.multimodal_data["mrope_config"]["mrope_position_deltas"] is mrope_deltas
    cumsum = params.multimodal_data["multimodal_embed_mask_cumsum"]
    assert cumsum.device.type == "cpu"
    # Only video embedding positions advance the mask; framing and text do not.
    assert cumsum.tolist() == [0, 0, 1, 2, 3, 3, 3]
    assert cumsum.numel() == len(tokens)
    assert params.encoder_state is None
    assert params.encoder_event is None


def test_disagg_decode_rebuilds_only_mrope_params():
    llm = object.__new__(BaseLLM)
    llm._hf_model_config = SimpleNamespace(is_encoder_decoder=False)
    mrope_ids, mrope_deltas = {"position_ids": "handle"}, {"position_deltas": "handle"}
    disagg = DisaggregatedParams(
        request_type="generation_only",
        mrope_position_ids_handle=mrope_ids,
        mrope_position_deltas_handle=mrope_deltas,
    )

    _, _, params, _ = llm._preprocess({"prompt_token_ids": [10, 11]}, SamplingParams(), disagg)

    assert params.item_metadata is None
    assert params.multimodal_input is None
    assert params.multimodal_data == {
        "mrope_config": {"mrope_position_ids": mrope_ids, "mrope_position_deltas": mrope_deltas}
    }


@pytest.mark.parametrize("model", ["qwen2.5", "qwen3"])
def test_external_embeddings_reuse_common_item_metadata(model):
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2_5VLInputProcessorBase

    cls = Qwen3VLInputProcessorBase if model == "qwen3" else Qwen2_5VLInputProcessorBase
    processor = object.__new__(cls)
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        text_config=SimpleNamespace(hidden_size=16),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[], spatial_merge_size=2),
    )
    processor._tokenizer = _FakeTokenizer([11, _IMAGE_TOKEN_ID, 12, _IMAGE_TOKEN_ID, 13])
    mrope_config = {"mrope_position_deltas": torch.zeros(1, dtype=torch.int32)}
    processor.get_mrope_config = lambda *args: mrope_config
    embeddings = [torch.ones(4, 16), torch.ones(9, 16)]

    tokens, extra = processor.attach_multimodal_embeddings(
        {"prompt": "video prompt"}, {"image": embeddings}, SamplingParams()
    )
    params = MultimodalParams(multimodal_data=extra["multimodal_data"])

    assert tokens == [11, *([_IMAGE_TOKEN_ID] * 4), 12, *([_IMAGE_TOKEN_ID] * 9), 13]
    assert params.get_item_layout() == ((("image", 0), ("image", 1)), (4, 9))
    assert params.multimodal_data["multimodal_embedding"] is embeddings
    assert params.multimodal_data["mrope_config"] is mrope_config
    assert params.multimodal_input is None


def test_qwen3vl_video_token_count_sums_single_video_grid_rows():
    """One logical video may be represented by multiple processor grid rows."""
    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )

    assert (
        processor.get_num_tokens_per_video(
            video=[],
            video_grid_thw=torch.tensor([[1, 8, 10], [1, 8, 10]]),
        )
        == 40
    )


@pytest.fixture(params=[False, True], ids=["no-special-ids", "special-ids"])
def video_handoff(request):
    """Build the hashing path's layout with and without optional special-token IDs."""
    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_TOKEN_ID,
        video_token_id=_VIDEO_TOKEN_ID,
        vision_start_token_id=_VISION_START_TOKEN_ID,
        text_config=SimpleNamespace(hidden_size=16, vocab_size=200000),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
    )
    processor._processor = SimpleNamespace()
    if request.param:
        processor._processor.mm_special_token_ids = torch.tensor([_VISION_START_TOKEN_ID])
    processor.multimodal_hashing_supported = True
    processor.get_num_tokens_per_video = Mock(return_value=6 if request.param else 4)
    processor._tokenizer = _FakeTokenizer(
        [11, _VISION_START_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_END_TOKEN_ID, 12]
    )
    # Two frame groups, separated by timestamp text (70/71 and 72/73).
    token_ids = [
        11,
        70,
        71,
        _VISION_START_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        72,
        73,
        _VISION_START_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VIDEO_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        12,
    ]
    processed = {
        "multimodal_data": {
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                item_refs=[("video", 0)],
                encoder_token_lengths=[16],
                output_embedding_lengths=[4],
            ),
        }
    }
    with patch.object(Qwen3VLInputProcessorBase, "__call__", return_value=(token_ids, processed)):
        _, hashed = create_input_processor_with_hash(processor)(
            {"prompt": "video prompt", "multi_modal_data": {"video": [[torch.zeros(2, 2, 3)]]}},
            SamplingParams(max_tokens=1),
        )
    return processor, token_ids, MultimodalParams(**hashed)


@pytest.mark.parametrize("hashing_enabled", [True, False])
def test_encoder_handoff_preserves_video_timestamps(video_handoff, hashing_enabled):
    """Keep timestamps and sparse visual runs with or without hashing metadata."""
    processor, token_ids, params = video_handoff
    expected = params.multimodal_input
    expected_offsets = params.multimodal_data.get("special_token_offsets")
    if not hashing_enabled:
        params.multimodal_input = None
    layout = processor.get_disagg_prefill_inputs(token_ids, params)
    handles = [{"tensor_size": [4, 16]}]
    result_ids, result_params = processor.build_disagg_prefill_multimodal_inputs(
        {"prompt": "video prompt"},
        handles,
        mm_hashes=[[1] * 8],
        encoder_inputs=layout,
    )
    result = result_params.multimodal_input
    assert result_ids == token_ids
    assert result.multimodal_positions == expected.multimodal_positions
    assert result.multimodal_lengths == expected.multimodal_lengths
    assert result_params.item_metadata.output_embedding_lengths == (4,)
    assert result.multimodal_item_run_cu_offsets == expected.multimodal_item_run_cu_offsets
    assert result.multimodal_run_positions == expected.multimodal_run_positions
    assert result.multimodal_run_lengths == expected.multimodal_run_lengths
    assert result_params.multimodal_data.get("special_token_offsets") == expected_offsets
    assert result_params.item_metadata.item_refs == (("video", 0),)
    if processor.get_mm_special_token_ids() is None:
        assert result.multimodal_lengths == [4]
        assert result.multimodal_run_positions == [4, 10]
        assert "special_token_offsets" not in result_params.multimodal_data


@pytest.mark.parametrize("round_trip", ["json", "pickle"])
def test_encoder_layout_survives_handoff_serialization(video_handoff, round_trip):
    """Restore a typed encoder layout after JSON or pickle transport."""
    processor, token_ids, params = video_handoff
    original = DisaggregatedParams(
        multimodal_embedding_handles=[{"tensor_size": [4, 16]}],
        multimodal_hashes=[[1] * 8],
        multimodal_input_layout=processor.get_disagg_prefill_inputs(token_ids, params),
    )
    if round_trip == "json":
        restored = DisaggregatedParams(**json.loads(json.dumps(asdict(original))))
    else:
        restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert restored.multimodal_input_layout.prompt_token_ids == token_ids


def test_prefill_consumes_encoder_layout_and_rejects_stale_positions(video_handoff):
    """Use exact encoder tokens and reject mRoPE positions for a different prompt."""
    processor, token_ids, params = video_handoff
    llm = object.__new__(BaseLLM)
    llm.input_processor = processor
    llm._is_encoder_decoder_model = Mock(return_value=False)
    handoff = DisaggregatedParams(
        request_type="context_and_generation",
        multimodal_embedding_handles=[{"tensor_size": [4, 16]}],
        multimodal_hashes=[[1] * 8],
        multimodal_input_layout=processor.get_disagg_prefill_inputs(token_ids, params),
        mrope_position_ids_handle={"tensor_size": [3, 1, len(token_ids)]},
        mrope_position_deltas_handle={"tensor_size": [1, 1]},
    )
    result_ids, _, result_mm, _ = llm._preprocess(
        {"prompt": "video prompt"},
        SamplingParams(max_tokens=1),
        handoff,
    )
    assert result_ids == token_ids
    assert result_mm.multimodal_input.multimodal_hashes == [[1] * 8]
    assert (
        result_mm.multimodal_input.multimodal_run_positions
        == params.multimodal_input.multimodal_run_positions
    )
    assert result_mm.multimodal_data.get("special_token_offsets") == params.multimodal_data.get(
        "special_token_offsets"
    )
    assert ("special_token_offsets" in result_mm.multimodal_data) == (
        "special_token_offsets" in params.multimodal_data
    )
    assert result_mm.multimodal_data["multimodal_embed_mask_cumsum"].tolist() == [
        0,
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        2,
        2,
        3,
        4,
        4,
        4,
    ]
    # An older encoder omits layout; rebuilding this video cannot match its mRoPE.
    handoff.multimodal_input_layout = None
    with pytest.raises(ValueError, match="mRoPE.*prompt"):
        llm._preprocess({"prompt": "video prompt"}, SamplingParams(max_tokens=1), handoff)


@pytest.mark.parametrize(
    "shape,error,match",
    [
        ([4], ValueError, "rank 2"),
        ([4, 8], RuntimeError, "hidden size 16"),
        ([3, 16], ValueError, "embedding lengths"),
    ],
)
def test_encoder_layout_does_not_bypass_handle_validation(video_handoff, shape, error, match):
    """Reject embedding handles with incompatible rank, width, or row count."""
    processor, token_ids, params = video_handoff
    layout = processor.get_disagg_prefill_inputs(token_ids, params)
    with pytest.raises(error, match=match):
        processor.build_disagg_prefill_multimodal_inputs(
            {},
            [{"tensor_size": shape}],
            encoder_inputs=layout,
        )


def test_encoder_layout_owns_its_metadata(video_handoff):
    """Keep exported and consumed layouts independent of caller-owned metadata."""
    processor, token_ids, params = video_handoff
    expected_positions = list(params.multimodal_input.multimodal_run_positions)
    layout = processor.get_disagg_prefill_inputs(token_ids, params)
    restored_ids, restored_params = processor.build_disagg_prefill_multimodal_inputs(
        {},
        [{"tensor_size": [4, 16]}],
        mm_hashes=[[1] * 8],
        encoder_inputs=layout,
    )
    restored_ids.append(99)
    restored_params.multimodal_input.multimodal_run_positions[0] = 0
    assert layout.prompt_token_ids == token_ids
    assert layout.multimodal_run_positions == expected_positions
    layout.prompt_token_ids.append(100)
    layout.multimodal_run_positions[0] = 1
    assert token_ids[-1] == 12
    assert params.multimodal_input.multimodal_run_positions == expected_positions


@pytest.mark.parametrize("modality", ["image", "video"])
def test_multiple_items_without_hashing_keep_prompt_order(video_handoff, modality):
    """Recover ordered per-item runs from token masks when hashing is unavailable."""
    processor, video_ids, _ = video_handoff
    image_ids = [
        11,
        _VISION_START_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        12,
    ]
    item_ids = image_ids if modality == "image" else video_ids
    item_embeddings = 2 if modality == "image" else 4
    params = MultimodalParams(
        multimodal_data={
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [(modality, 0), (modality, 1)],
                output_embedding_lengths=[item_embeddings, item_embeddings],
            ),
        }
    )
    layout = processor.get_disagg_prefill_inputs(item_ids + item_ids, params)
    assert layout.prompt_token_ids == item_ids + item_ids
    has_special_ids = processor.get_mm_special_token_ids() is not None
    if modality == "image":
        positions = [1, 7] if has_special_ids else [2, 8]
        assert layout.multimodal_positions == positions
        assert layout.multimodal_lengths == ([3, 3] if has_special_ids else [2, 2])
        assert layout.multimodal_item_run_cu_offsets == [0, 1, 2]
        assert layout.multimodal_run_positions == positions
        assert layout.special_token_offsets == ([0, 3] if has_special_ids else None)
        assert layout.item_types == [0, 0]
    else:
        assert layout.multimodal_positions == ([3, 17] if has_special_ids else [4, 18])
        assert layout.multimodal_lengths == ([6, 6] if has_special_ids else [4, 4])
        assert layout.multimodal_item_run_cu_offsets == [0, 2, 4]
        assert layout.multimodal_run_positions == (
            [3, 9, 17, 23] if has_special_ids else [4, 10, 18, 24]
        )
        assert layout.special_token_offsets == ([0, 3, 6, 9] if has_special_ids else None)
        assert layout.item_types == [1, 1]

    restored_ids, restored = processor.build_disagg_prefill_multimodal_inputs(
        {},
        [{"tensor_size": [item_embeddings, 16]}] * 2,
        mm_hashes=[[1] * 8, [2] * 8],
        encoder_inputs=layout,
    )
    assert restored_ids == item_ids + item_ids
    assert restored.item_metadata.item_refs == ((modality, 0), (modality, 1))
    assert restored.item_metadata.output_embedding_lengths == (item_embeddings, item_embeddings)
    assert restored.multimodal_input.multimodal_run_positions == layout.multimodal_run_positions


@pytest.mark.parametrize("item_types", [None, [2]])
def test_encoder_layout_rejects_unknown_item_types(video_handoff, item_types):
    """Reject handoffs that cannot be mapped to Qwen image/video items."""
    processor, token_ids, params = video_handoff
    layout = processor.get_disagg_prefill_inputs(token_ids, params)
    layout.item_types = item_types
    with pytest.raises(ValueError, match="identify each image or video item"):
        processor.build_disagg_prefill_multimodal_inputs(
            {}, [{"tensor_size": [4, 16]}], encoder_inputs=layout
        )


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [], output_embedding_lengths=[]
            )
        },
    ],
)
def test_no_layout_for_text_only_inputs(video_handoff, data):
    """Leave prompt reconstruction unchanged when there are no visual embeddings."""
    processor, _, _ = video_handoff
    params = MultimodalParams(multimodal_data=data)
    assert processor.get_disagg_prefill_inputs([1, 2, 3], params) is None
    assert (
        BaseMultimodalInputProcessor.get_disagg_prefill_inputs(processor, [1, 2, 3], params) is None
    )


def test_legacy_image_handoff_still_works(video_handoff):
    """Accept image handles without a layout while requiring paired mRoPE metadata."""
    processor, _, _ = video_handoff
    processor._tokenizer = _FakeTokenizer(
        [11, _VISION_START_TOKEN_ID, _IMAGE_TOKEN_ID, _VISION_END_TOKEN_ID, 12]
    )
    llm = object.__new__(BaseLLM)
    llm.input_processor = processor
    llm._is_encoder_decoder_model = Mock(return_value=False)
    handoff = DisaggregatedParams(
        request_type="context_and_generation",
        multimodal_embedding_handles=[{"tensor_size": [2, 16]}],
        multimodal_hashes=[[1] * 8],
        mrope_position_ids_handle={"tensor_size": [3, 1, 6]},
        mrope_position_deltas_handle={"tensor_size": [1, 1]},
    )
    ids, _, mm, _ = llm._preprocess(
        {"prompt": "video prompt"}, SamplingParams(max_tokens=1), handoff
    )
    assert ids == [
        11,
        _VISION_START_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _IMAGE_TOKEN_ID,
        _VISION_END_TOKEN_ID,
        12,
    ]
    assert mm.multimodal_input.multimodal_lengths == [3]
    handoff.mrope_position_deltas_handle = None
    with pytest.raises(ValueError, match="position IDs require position deltas"):
        llm._preprocess({"prompt": "video prompt"}, SamplingParams(max_tokens=1), handoff)


@pytest.mark.parametrize("invalid_layout", [[], "layout", {"prompt_token_ids": [1, 2]}])
def test_invalid_layout_is_rejected_at_deserialization(invalid_layout):
    """Reject unsupported payload types and incomplete layout dictionaries."""
    with pytest.raises(TypeError):
        DisaggregatedParams(multimodal_input_layout=invalid_layout)


@pytest.mark.parametrize("existing_params", [False, True])
def test_encoder_submits_layout_without_mutating_caller_params(video_handoff, existing_params):
    """Attach the layout to submitted parameters without modifying the caller's copy."""
    processor, token_ids, params = video_handoff
    encoder = object.__new__(BaseLLM)
    encoder.shutdown = Mock()
    encoder._encode_only = False
    encoder.args = SimpleNamespace(mm_encoder_only=True, return_perf_metrics=False)
    encoder.input_processor = processor
    encoder._executor = Mock()
    encoder._executor.is_shutdown.return_value = False
    encoder._prepare_sampling_params = Mock(return_value=SamplingParams(max_tokens=1))
    encoder._preprocess = Mock(return_value=(token_ids, "video prompt", params, None))
    encoder._check_arguments = Mock()
    encoder._post_processor_hook = None
    original = (
        DisaggregatedParams(request_type="context_and_generation", disagg_request_id=7)
        if existing_params
        else None
    )
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        encoder.generate_async({"prompt": "video prompt"}, disaggregated_params=original)
    submitted = encoder._executor.generate_async.call_args.kwargs["disaggregated_params"]
    assert submitted.request_type == "context_and_generation"
    assert submitted.multimodal_input_layout.prompt_token_ids == token_ids
    if original is not None:
        assert submitted.disagg_request_id == 7
        assert submitted is not original
        assert original.multimodal_input_layout is None


@pytest.mark.parametrize("preprocessed", [False, True])
@pytest.mark.parametrize("request_type", ["context_only", "context_and_generation"])
def test_prefill_drops_consumed_layout_before_executor(video_handoff, preprocessed, request_type):
    """Submit only engine-needed metadata while preserving the caller's handoff."""
    processor, token_ids, params = video_handoff
    llm = object.__new__(BaseLLM)
    llm.shutdown = Mock()
    llm._encode_only = False
    llm.args = SimpleNamespace(mm_encoder_only=False, return_perf_metrics=False)
    llm.input_processor = processor
    llm._executor = Mock()
    llm._executor.is_shutdown.return_value = False
    llm._is_encoder_decoder_model = Mock(return_value=False)
    llm._prepare_sampling_params = Mock(return_value=SamplingParams(max_tokens=1))
    llm._configure_bart_decoder_prefix = Mock()
    llm._check_arguments = Mock()
    llm._post_processor_hook = None
    original = DisaggregatedParams(
        request_type=request_type,
        disagg_request_id=7,
        multimodal_embedding_handles=[{"tensor_size": [4, 16]}],
        multimodal_hashes=[[1] * 8],
        multimodal_input_layout=processor.get_disagg_prefill_inputs(token_ids, params),
        mrope_position_ids_handle={"tensor_size": [3, 1, len(token_ids)]},
        mrope_position_deltas_handle={"tensor_size": [1, 1]},
    )
    expected = asdict(original)
    inputs = {"prompt": "video prompt"}
    if preprocessed:
        ids, _, mm_params, encoder_ids = llm._preprocess(
            inputs, SamplingParams(max_tokens=1), original
        )
        inputs = PreprocessedInputs(ids, mm_params, encoder_ids)
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(inputs, disaggregated_params=original)
    submitted = llm._executor.generate_async.call_args
    assert submitted.args[0] == token_ids
    assert submitted.kwargs["multimodal_params"].multimodal_data["multimodal_embedding"] == (
        original.multimodal_embedding_handles
    )
    assert asdict(original) == expected
    expected["multimodal_input_layout"] = None
    assert asdict(submitted.kwargs["disaggregated_params"]) == expected
