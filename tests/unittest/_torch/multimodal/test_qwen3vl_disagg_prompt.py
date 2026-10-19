# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VLInputProcessorBase,
    _expand_prompt_token_ids_for_mm_handoff,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm import BaseLLM
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
    mrope_ids, mrope_deltas = {"position_ids": "handle"}, {"position_deltas": "handle"}
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
    assert cumsum.numel() == len(tokens)
    assert cumsum[-1].item() == 3
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

    assert len(tokens) == 16
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
