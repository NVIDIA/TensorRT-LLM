# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)
from tensorrt_llm.inputs.registry import create_input_processor_with_hash
from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs, _encoder_disaggregated_params
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
    processor._processor = SimpleNamespace(
        mm_special_token_ids=torch.tensor([_VISION_START_TOKEN_ID])
    )
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
    processor.image_token_index = _IMAGE_TOKEN_ID
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
    llm.input_processor = SimpleNamespace()
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
    processor._processor = SimpleNamespace()
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
    items = [
        {"mm_embeddings": embeddings[0], "image_grid_thw": [1, 4, 4]},
        {"mm_embeddings": embeddings[1], "image_grid_thw": [1, 6, 6]},
    ]

    tokens, extra = processor.attach_multimodal_embeddings(
        {"prompt": "video prompt"}, {"image": items}, SamplingParams()
    )
    params = MultimodalParams(multimodal_data=extra["multimodal_data"])

    assert tokens == [11, *([_IMAGE_TOKEN_ID] * 4), 12, *([_IMAGE_TOKEN_ID] * 9), 13]
    assert params.get_item_layout() == ((("image", 0), ("image", 1)), (4, 9))
    assert params.multimodal_data["multimodal_embedding"] == embeddings
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


def _legacy_layout(token_ids, params):
    return _encoder_disaggregated_params(token_ids, params, None).multimodal_input_layout


@pytest.mark.parametrize("round_trip", ["json", "pickle"])
def test_encoder_layout_survives_handoff_serialization(video_handoff, round_trip):
    """Restore a typed encoder layout after JSON or pickle transport."""
    processor, token_ids, params = video_handoff
    original = DisaggregatedParams(
        multimodal_embedding_handles=[{"tensor_size": [4, 16]}],
        multimodal_hashes=[[1] * 8],
        multimodal_input_layout=_legacy_layout(token_ids, params),
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
        multimodal_input_layout=_legacy_layout(token_ids, params),
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
        ([4, 8], ValueError, "hidden size 16"),
        ([3, 16], ValueError, "embedding lengths"),
    ],
)
def test_encoder_layout_does_not_bypass_handle_validation(video_handoff, shape, error, match):
    """Reject embedding handles with incompatible rank, width, or row count."""
    processor, token_ids, params = video_handoff
    layout = _legacy_layout(token_ids, params)
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
    layout = _legacy_layout(token_ids, params)
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


@pytest.mark.parametrize("item_types", [None, [2]])
def test_encoder_layout_rejects_unknown_item_types(video_handoff, item_types):
    """Reject handoffs that cannot be mapped to Qwen image/video items."""
    processor, token_ids, params = video_handoff
    layout = _legacy_layout(token_ids, params)
    layout.item_types = item_types
    with pytest.raises(ValueError, match="identify each image or video item"):
        processor.build_disagg_prefill_multimodal_inputs(
            {}, [{"tensor_size": [4, 16]}], encoder_inputs=layout
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
    includes_start = processor.get_mm_special_token_ids() is not None
    assert mm.multimodal_input.multimodal_lengths == [2 + int(includes_start)]
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
    assert submitted.multimodal_input == asdict(params.multimodal_input)
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
        multimodal_input_layout=_legacy_layout(token_ids, params),
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
