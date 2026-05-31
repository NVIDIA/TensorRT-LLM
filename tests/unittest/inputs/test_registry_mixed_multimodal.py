# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalPromptOrder, find_mm_token_lengths
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData
from tensorrt_llm.inputs.registry import create_input_processor_with_hash
from tensorrt_llm.sampling_params import SamplingParams

# TODO(claude) : add concise docstrings/comments explaining intended behavior for each test.
# TODO(claude) : If there are some constants/macros declare it globally.


class _FakeMixedProcessor:
    def __init__(self):
        self.multimodal_hashing_supported = None
        self.lengths_seen_by_expander = None
        self.video_call_kwargs = None

    def __call__(self, inputs, sampling_params):
        assert sampling_params is not None
        multimodal_data = {
            "image": {},
            "video": {},
            "audio": {},
        }
        prompt_token_ids = inputs.get("prompt_token_ids", [])
        mm_data = inputs.get("multi_modal_data")
        # Model the post-merge `call_with_token_ids` contract: when called with
        # tokenized input + multimodal data, the processor itself expands the
        # placeholder tokens via `expand_prompt_token_ids_for_mm` (the merge
        # relocated this out of the hashing wrapper closure and into the
        # processor `__call__` path). The hashing wrapper then consumes the
        # already-expanded ids. A real processor resolves per-item token counts
        # in prompt order; mirror that here so the test exercises the real flow.
        if prompt_token_ids and mm_data:
            lengths_by_key = find_mm_token_lengths(mm_data, self)
            item_order = MultimodalPromptOrder.resolve(
                mm_data, self, prompt_token_ids=prompt_token_ids
            )
            num_mm_tokens = item_order.flatten(lengths_by_key)
            prompt_token_ids, _ = self.expand_prompt_token_ids_for_mm(
                prompt_token_ids,
                num_mm_tokens,
                hf_processor_mm_kwargs=inputs.get("mm_processor_kwargs"),
                mm_data=mm_data,
            )
        return prompt_token_ids, {"multimodal_data": multimodal_data}

    def get_text_with_mm_placeholders(self, mm_counts):
        return "".join(f"<{modality}>" * count for modality, count in mm_counts.items())

    # Modality membership for both placeholder (pre-expansion) and feature
    # (post-expansion) token ids, so order parsing is expansion-tolerant.
    _MODALITY_TOKEN_IDS = {
        "image": {100, 101},
        "video": {200, 201},
        "audio": {300, 301},
    }

    def get_mm_item_order(self, prompt_token_ids, mm_data):
        # Derive prompt-item order from the first occurrence of each modality's
        # tokens. A real processor parses placeholder/vision-pad tokens that
        # survive `expand_prompt_token_ids_for_mm`, so this hook must work on
        # BOTH the original prompt (the processor's own call_with_token_ids
        # path) AND the expanded prompt (the hashing wrapper, which passes the
        # processor's already-expanded output). Asserting one exact id list
        # broke once the merge relocated expansion into the processor.
        first_pos = {}
        for pos, tok in enumerate(prompt_token_ids):
            for modality, ids in self._MODALITY_TOKEN_IDS.items():
                if tok in ids and modality not in first_pos:
                    first_pos[modality] = pos
        ordered = sorted(first_pos, key=first_pos.get)
        return [(modality, 0) for modality in ordered]

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids,
        num_mm_tokens_per_placeholder,
        hf_processor_mm_kwargs=None,
        mm_data=None,
    ):
        assert hf_processor_mm_kwargs is None
        assert mm_data is not None
        self.lengths_seen_by_expander = list(num_mm_tokens_per_placeholder)
        assert self.lengths_seen_by_expander == [2, 4, 3]
        return [
            10,
            101,
            101,
            11,
            201,
            201,
            201,
            201,
            12,
            301,
            301,
            301,
            13,
        ], None

    def get_num_tokens_per_image(self, image):
        assert image.tolist() == [1]
        return 2

    def get_num_tokens_per_video(self, **kwargs):
        self.video_call_kwargs = kwargs
        return 4

    def get_num_tokens_per_audio(self, audio):
        assert audio.tolist() == [3]
        return 3

    def get_vocab_size(self):
        return None

    def get_mm_token_ids(self):
        return torch.tensor([101, 201, 301])

    def get_mm_special_token_ids(self):
        return None


def test_normalize_mm_item_order_decodes_layout_item_types():
    assert MultimodalPromptOrder.from_raw_entries(
        [0, 1, 2], source="layout_metadata.item_types"
    ) == [
        ("image", 0),
        ("video", 0),
        ("audio", 0),
    ]


def test_normalize_mm_item_order_rejects_unknown_layout_item_type():
    with pytest.raises(ValueError, match="unknown item type: 7"):
        MultimodalPromptOrder.from_raw_entries([7], source="layout_metadata.item_types")


def test_find_mm_token_lengths_preserves_all_modalities_and_video_audio():
    processor = _FakeMixedProcessor()
    video_audio = AudioData(samples=torch.ones(16000).numpy(), sample_rate=16000)
    mm_data = {
        "image": [torch.tensor([1])],
        "video": [
            VideoData(
                frames=[torch.tensor([2])],
                metadata={"fps": 30.0},
                audio=video_audio,
            )
        ],
        "audio": [torch.tensor([3])],
    }

    lengths = find_mm_token_lengths(mm_data, processor)

    assert lengths == {"image": [2], "video": [4], "audio": [3]}
    assert torch.equal(processor.video_call_kwargs["video"][0], torch.tensor([2]))
    assert processor.video_call_kwargs["video_metadata"] == {"fps": 30.0}
    assert processor.video_call_kwargs["video_audio"] is video_audio


def test_tokenized_wrapper_mixed_modalities_builds_metadata_for_all_items():
    processor = _FakeMixedProcessor()
    wrapper = create_input_processor_with_hash(processor)
    mm_data = {
        "image": [torch.tensor([1])],
        "video": [[torch.tensor([2])]],
        "audio": [torch.tensor([3])],
    }

    prompt_token_ids, extra = wrapper(
        {
            "prompt_token_ids": [10, 100, 11, 200, 12, 300, 13],
            "multi_modal_data": mm_data,
            "multi_modal_uuids": {
                "image": ["img-0"],
                "video": ["vid-0"],
                "audio": ["aud-0"],
            },
        },
        SamplingParams(),
    )

    assert processor.lengths_seen_by_expander == [2, 4, 3]
    assert prompt_token_ids == [
        10,
        101,
        101,
        11,
        201,
        201,
        201,
        201,
        12,
        301,
        301,
        301,
        13,
    ]
    mm_input = extra["multimodal_input"]
    assert mm_input.multimodal_positions == [1, 4, 9]
    assert mm_input.multimodal_lengths == [2, 4, 3]
    assert mm_input.multimodal_uuids == ["img-0", "vid-0", "aud-0"]
    assert mm_input.multimodal_item_run_cu_offsets == [0, 1, 2, 3]
    assert mm_input.multimodal_run_positions == [1, 4, 9]
    assert mm_input.multimodal_run_lengths == [2, 4, 3]
    assert extra["multimodal_data"]["multimodal_embedding_lengths"] == [2, 4, 3]


def test_text_wrapper_single_video_uses_default_item_order():
    class _SingleVideoProcessor(_FakeMixedProcessor):
        def __call__(self, inputs, sampling_params):
            return [10, 201, 201, 201, 201, 13], {"multimodal_data": {"video": {}}}

        def get_mm_item_order(self, prompt_token_ids, mm_data):
            raise AssertionError("single-modality requests should not need prompt-order hooks")

        def get_mm_token_ids(self):
            return torch.tensor([201])

    processor = _SingleVideoProcessor()
    wrapper = create_input_processor_with_hash(processor)
    prompt_token_ids, extra = wrapper(
        {"prompt": "video prompt", "multi_modal_data": {"video": [[torch.tensor([2])]]}},
        SamplingParams(),
    )

    assert prompt_token_ids == [10, 201, 201, 201, 201, 13]
    assert extra["multimodal_input"].multimodal_positions == [1]
    assert extra["multimodal_input"].multimodal_lengths == [4]
    assert extra["multimodal_data"]["multimodal_item_order"] == [{"modality": "video", "index": 0}]
