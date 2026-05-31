# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalPromptOrder, find_mm_token_lengths
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData
from tensorrt_llm.inputs.registry import (
    BaseMultimodalInputProcessor,
    create_input_processor_with_hash,
)
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


# Placeholder token IDs the fake processor emits for each modality in the
# tokenized fast path. The interleaved prompt below carries image, audio, image
# in that exact prompt order, so the resolved order is NOT modality-major.
_IMAGE_PLACEHOLDER_ID = 100
_AUDIO_PLACEHOLDER_ID = 300

# Distinct per-item token counts so the flattened list is order-sensitive: the
# only way to get [2, 5, 3] (img0, audio0, img1) is to flatten in PROMPT order
# across modalities. The superseded single-modality next(iter(...)) path could
# only ever return one modality's list (e.g. images -> [2, 3]).
_IMG0_TOKENS = 2
_IMG1_TOKENS = 3
_AUDIO0_TOKENS = 5

# The prompt-item-ordered per-item token counts the real
# MultimodalPromptOrder.resolve().flatten() must hand to the expand hook.
_EXPECTED_PROMPT_ORDER_TOKENS = [_IMG0_TOKENS, _AUDIO0_TOKENS, _IMG1_TOKENS]

# What the superseded single-modality `_get_single_mm_token_lengths`
# (`next(iter(num_mm_tokens_by_key.values()))`) path would have produced: only
# the first modality bucket (image), dropping audio and any interleaving.
_OLD_SINGLE_MODALITY_TOKENS = [_IMG0_TOKENS, _IMG1_TOKENS]


class _FakeTokenIdMMProcessor(BaseMultimodalInputProcessor):
    """Minimal `BaseMultimodalInputProcessor` opting into the tokenized fast path.

    Exercises `call_with_token_ids` directly (not the hashing wrapper) with a
    MIXED, interleaved prompt (image, audio, image). The dummy-placeholder step
    and the real `expand_prompt_token_ids_for_mm` hook are stubbed so the test
    drives the REAL `find_mm_token_lengths` + `MultimodalPromptOrder.resolve()`
    + `.flatten()` without a GPU or HF model.
    """

    # Opt in to the tokenized+MM fast path so `call_with_token_ids` runs.
    supports_token_id_mm_expansion = True

    def __init__(self):
        # Bypass BaseMultimodalInputProcessor.__init__ (it wires HF artifacts we
        # do not need); set only the attributes the exercised path reads.
        self._tokenizer = None
        self._config = None
        self._model_path = None
        self._use_fast = True
        self._trust_remote_code = True
        self._multimodal_hashing_supported = None
        # Captures the per-item token counts handed to the expand hook, i.e. the
        # output of `MultimodalPromptOrder.resolve(...).flatten(...)`.
        self.lengths_seen_by_expander = None

    # --- abstract members (unused by the exercised path, stubbed minimally) ---
    @property
    def processor(self):
        return None

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return torch.float32

    def call_with_text_prompt(self, inputs, sampling_params):
        # Invoked by `_process_multimodal_with_dummy_placeholders` on the
        # synthetic placeholder PROMPT (text, not token ids), so it does not
        # re-enter the token-id path. Returns an empty token list plus the
        # processed multimodal payload the rest of `call_with_token_ids` reuses.
        assert inputs.get("prompt") is not None
        assert inputs.get("prompt_token_ids") is None
        return [], {"multimodal_data": {"image": {}, "audio": {}}}

    # --- tokenized+MM fast-path hooks ---
    def get_text_with_mm_placeholders(self, mm_counts):
        return "".join(f"<{modality}>" * count for modality, count in mm_counts.items())

    def get_mm_item_order(self, prompt_token_ids, mm_data):
        # Parse the interleaved prompt to recover (modality, index) in PROMPT
        # order. The prompt is image, audio, image, so this yields
        # [(image,0), (audio,0), (image,1)] -- an order modality-major grouping
        # cannot produce.
        order = []
        counters = {"image": 0, "audio": 0}
        for tok in prompt_token_ids:
            if tok == _IMAGE_PLACEHOLDER_ID:
                order.append(("image", counters["image"]))
                counters["image"] += 1
            elif tok == _AUDIO_PLACEHOLDER_ID:
                order.append(("audio", counters["audio"]))
                counters["audio"] += 1
        return order

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids,
        num_mm_tokens_per_placeholder,
        hf_processor_mm_kwargs=None,
        mm_data=None,
    ):
        # Capture exactly what the resolve().flatten() step produced; this is
        # the assertion target. Return a sentinel expanded list (the expansion
        # arithmetic itself is out of scope here).
        self.lengths_seen_by_expander = list(num_mm_tokens_per_placeholder)
        assert mm_data is not None
        return [-1], None

    def get_num_tokens_per_image(self, *, image, **kwargs):
        # img0 -> 2 tokens, img1 -> 3 tokens (distinct so order is observable).
        return _IMG0_TOKENS if image.tolist() == [1] else _IMG1_TOKENS

    def get_num_tokens_per_audio(self, *, audio, **kwargs):
        return _AUDIO0_TOKENS


def _mixed_interleaved_inputs():
    """Tokenized inputs with two images straddling one audio (image, audio, image)."""
    # mm_data lists are modality-major (image[0], image[1], audio[0]); the
    # prompt token order is image, audio, image -- so flattening in prompt order
    # must interleave across the two modalities.
    mm_data = {
        "image": [torch.tensor([1]), torch.tensor([2])],
        "audio": [torch.tensor([3])],
    }
    prompt_token_ids = [
        10,
        _IMAGE_PLACEHOLDER_ID,  # image[0]
        11,
        _AUDIO_PLACEHOLDER_ID,  # audio[0]
        12,
        _IMAGE_PLACEHOLDER_ID,  # image[1]
        13,
    ]
    return {
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": mm_data,
    }


def test_call_with_token_ids_flattens_mixed_modalities_in_prompt_order():
    # Drives BaseMultimodalInputProcessor.call_with_token_ids directly (the
    # disaggregated/tokenized fast path authored by PR #14419). With an
    # interleaved image, audio, image prompt, the per-item counts passed to
    # expand_prompt_token_ids_for_mm must be flattened in PROMPT-ITEM order
    # across modalities, i.e. [img0, audio0, img1] = [2, 5, 3]. The superseded
    # single-modality next(iter(...)) path could only yield one modality's list.
    processor = _FakeTokenIdMMProcessor()
    inputs = _mixed_interleaved_inputs()

    expanded_ids, extra = processor.call_with_token_ids(inputs, SamplingParams())

    # The expand hook ran on the real resolve().flatten() output, in prompt order.
    assert processor.lengths_seen_by_expander == _EXPECTED_PROMPT_ORDER_TOKENS
    # Sanity: this is provably NOT the order the old single-modality path gave.
    assert processor.lengths_seen_by_expander != _OLD_SINGLE_MODALITY_TOKENS
    # Result is the (sentinel) expanded ids the hook returned.
    assert expanded_ids == [-1]
    assert "multimodal_data" in extra


def test_call_with_token_ids_red_when_flatten_regresses_to_single_modality(monkeypatch):
    # RED guard: prove the assertion above actually catches a regression. Stub
    # MultimodalPromptOrder.flatten to emulate the superseded single-modality
    # `_get_single_mm_token_lengths` (`next(iter(num_mm_tokens_by_key.values()))`)
    # behavior -- only the first modality bucket. Under that stub the expand hook
    # sees [2, 3] (images only), which the prompt-order assertion rejects.
    processor = _FakeTokenIdMMProcessor()
    inputs = _mixed_interleaved_inputs()

    def _single_modality_flatten(self, values_by_key):
        # Mirror the old next(iter(...)) assumption: first modality only.
        return list(next(iter(values_by_key.values())))

    monkeypatch.setattr(MultimodalPromptOrder, "flatten", _single_modality_flatten)

    processor.call_with_token_ids(inputs, SamplingParams())

    # Under the regressed flatten, the expander sees the WRONG (image-only,
    # un-interleaved) list -- exactly what the prompt-order assertion guards.
    assert processor.lengths_seen_by_expander == _OLD_SINGLE_MODALITY_TOKENS
    assert processor.lengths_seen_by_expander != _EXPECTED_PROMPT_ORDER_TOKENS
    # monkeypatch auto-restores MultimodalPromptOrder.flatten at teardown.
