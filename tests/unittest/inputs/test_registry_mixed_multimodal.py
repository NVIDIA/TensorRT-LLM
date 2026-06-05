# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import ClassVar

import torch

from tensorrt_llm.inputs.multimodal import MixedModalItemOrder, find_mm_token_lengths
from tensorrt_llm.inputs.multimodal_data import AudioData, VideoData
from tensorrt_llm.inputs.registry import (
    BaseMultimodalInputProcessor,
    create_input_processor_with_hash,
)
from tensorrt_llm.sampling_params import SamplingParams

# Synthetic token-id scheme for the fake processors: per modality a placeholder
# id (pre-expansion) and a feature id (post-expansion). Per-item token counts
# are distinct per modality so a flatten in the wrong order is observable.
_IMAGE_PLACEHOLDER_ID = 100
_VIDEO_PLACEHOLDER_ID = 200
_AUDIO_PLACEHOLDER_ID = 300
_IMAGE_FEATURE_ID = 101
_VIDEO_FEATURE_ID = 201
_AUDIO_FEATURE_ID = 301
_IMAGE_TOKENS = 2
_VIDEO_TOKENS = 4
_AUDIO_TOKENS = 3
# Per-item token counts in image, video, audio prompt order.
_MIXED_PROMPT_ORDER_TOKENS = [_IMAGE_TOKENS, _VIDEO_TOKENS, _AUDIO_TOKENS]


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
        # When called with tokenized input + multimodal data, the processor
        # itself expands placeholder tokens via `expand_prompt_token_ids_for_mm`,
        # resolving per-item token counts in prompt order; the hashing wrapper
        # then consumes the already-expanded ids.
        if prompt_token_ids and mm_data:
            lengths_by_key = find_mm_token_lengths(mm_data, self)
            # The processor computes its own prompt-item order from the tokens
            # (via its model-internal `get_mm_item_order` hook), mirroring how the
            # real token-id fast path resolves order before expansion.
            item_order = self.get_mm_item_order(prompt_token_ids, mm_data)
            num_mm_tokens = MixedModalItemOrder.project_by_order(item_order, lengths_by_key)
            prompt_token_ids, _ = self.expand_prompt_token_ids_for_mm(
                prompt_token_ids,
                num_mm_tokens,
                hf_processor_mm_kwargs=inputs.get("mm_processor_kwargs"),
                mm_data=mm_data,
            )
        return prompt_token_ids, {"multimodal_data": multimodal_data}

    # Modality membership for placeholder and feature ids, so order parsing is
    # tolerant of either pre- or post-expansion token ids.
    _MODALITY_TOKEN_IDS: ClassVar[dict[str, set[int]]] = {
        "image": {_IMAGE_PLACEHOLDER_ID, _IMAGE_FEATURE_ID},
        "video": {_VIDEO_PLACEHOLDER_ID, _VIDEO_FEATURE_ID},
        "audio": {_AUDIO_PLACEHOLDER_ID, _AUDIO_FEATURE_ID},
    }

    def get_mm_item_order(self, prompt_token_ids, mm_data):
        # Derive prompt-item order from the first occurrence of each modality's
        # tokens. Must work on both the original prompt (call_with_token_ids
        # path) and the expanded prompt (hashing wrapper), so it keys off
        # placeholder and feature ids alike.
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
        assert self.lengths_seen_by_expander == _MIXED_PROMPT_ORDER_TOKENS
        return [
            10,
            _IMAGE_FEATURE_ID,
            _IMAGE_FEATURE_ID,
            11,
            _VIDEO_FEATURE_ID,
            _VIDEO_FEATURE_ID,
            _VIDEO_FEATURE_ID,
            _VIDEO_FEATURE_ID,
            12,
            _AUDIO_FEATURE_ID,
            _AUDIO_FEATURE_ID,
            _AUDIO_FEATURE_ID,
            13,
        ], None

    def get_num_tokens_per_image(self, image):
        return _IMAGE_TOKENS

    def get_num_tokens_per_video(self, **kwargs):
        self.video_call_kwargs = kwargs
        return _VIDEO_TOKENS

    def get_num_tokens_per_audio(self, audio):
        return _AUDIO_TOKENS

    def get_vocab_size(self):
        return None

    def get_mm_token_ids(self):
        return torch.tensor([_IMAGE_FEATURE_ID, _VIDEO_FEATURE_ID, _AUDIO_FEATURE_ID])

    def get_mm_special_token_ids(self):
        return None


def test_find_mm_token_lengths_preserves_all_modalities_and_video_audio():
    """find_mm_token_lengths keeps every modality and threads VideoData audio to the per-video hook."""
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

    assert lengths == {
        "image": [_IMAGE_TOKENS],
        "video": [_VIDEO_TOKENS],
        "audio": [_AUDIO_TOKENS],
    }
    assert torch.equal(processor.video_call_kwargs["video"][0], torch.tensor([2]))
    assert processor.video_call_kwargs["video_metadata"] == {"fps": 30.0}
    assert processor.video_call_kwargs["video_audio"] is video_audio


def test_tokenized_wrapper_mixed_modalities_builds_metadata_for_all_items():
    """Hashing wrapper expands every placeholder run and builds per-item metadata across all three modalities."""
    processor = _FakeMixedProcessor()
    wrapper = create_input_processor_with_hash(processor)
    mm_data = {
        "image": [torch.tensor([1])],
        "video": [[torch.tensor([2])]],
        "audio": [torch.tensor([3])],
    }

    prompt_token_ids, extra = wrapper(
        {
            "prompt_token_ids": [
                10,
                _IMAGE_PLACEHOLDER_ID,
                11,
                _VIDEO_PLACEHOLDER_ID,
                12,
                _AUDIO_PLACEHOLDER_ID,
                13,
            ],
            "multi_modal_data": mm_data,
            "multi_modal_uuids": {
                "image": ["img-0"],
                "video": ["vid-0"],
                "audio": ["aud-0"],
            },
        },
        SamplingParams(),
    )

    assert processor.lengths_seen_by_expander == _MIXED_PROMPT_ORDER_TOKENS
    assert prompt_token_ids == [
        10,
        _IMAGE_FEATURE_ID,
        _IMAGE_FEATURE_ID,
        11,
        _VIDEO_FEATURE_ID,
        _VIDEO_FEATURE_ID,
        _VIDEO_FEATURE_ID,
        _VIDEO_FEATURE_ID,
        12,
        _AUDIO_FEATURE_ID,
        _AUDIO_FEATURE_ID,
        _AUDIO_FEATURE_ID,
        13,
    ]
    mm_input = extra["multimodal_input"]
    assert mm_input.multimodal_positions == [1, 4, 9]
    assert mm_input.multimodal_lengths == _MIXED_PROMPT_ORDER_TOKENS
    assert mm_input.multimodal_uuids == ["img-0", "vid-0", "aud-0"]
    assert mm_input.multimodal_item_run_cu_offsets == [0, 1, 2, 3]
    assert mm_input.multimodal_run_positions == [1, 4, 9]
    assert mm_input.multimodal_run_lengths == _MIXED_PROMPT_ORDER_TOKENS
    assert extra["multimodal_data"]["multimodal_embedding_lengths"] == _MIXED_PROMPT_ORDER_TOKENS


def test_hashing_path_preserves_processor_baked_embedding_lengths():
    """A processor that pre-bakes multimodal_embedding_lengths keeps its value.

    Mirrors Nano, which computes per-item embedding lengths in its preprocess
    (`compute_mm_embedding_lengths`) and writes them into multimodal_data. The
    registry's missing-key guard must NOT recompute/overwrite them. We pre-bake a
    deliberately distinct sentinel so an overwrite would be observable: the real
    mask-derived value here is `_MIXED_PROMPT_ORDER_TOKENS` ([2, 4, 3]).
    """
    _SENTINEL_LENGTHS = [99, 98, 97]

    class _PrebakedLengthsProcessor(_FakeMixedProcessor):
        def __call__(self, inputs, sampling_params):
            prompt_token_ids, extra = super().__call__(inputs, sampling_params)
            extra["multimodal_data"]["multimodal_embedding_lengths"] = list(_SENTINEL_LENGTHS)
            return prompt_token_ids, extra

    processor = _PrebakedLengthsProcessor()
    wrapper = create_input_processor_with_hash(processor)
    mm_data = {
        "image": [torch.tensor([1])],
        "video": [[torch.tensor([2])]],
        "audio": [torch.tensor([3])],
    }

    _, extra = wrapper(
        {
            "prompt_token_ids": [
                10,
                _IMAGE_PLACEHOLDER_ID,
                11,
                _VIDEO_PLACEHOLDER_ID,
                12,
                _AUDIO_PLACEHOLDER_ID,
                13,
            ],
            "multi_modal_data": mm_data,
        },
        SamplingParams(),
    )

    # The processor-baked value survives unchanged (guard skips the recompute).
    assert extra["multimodal_data"]["multimodal_embedding_lengths"] == _SENTINEL_LENGTHS
    # Sanity: the registry would otherwise have derived the real mask value.
    assert _SENTINEL_LENGTHS != _MIXED_PROMPT_ORDER_TOKENS


def test_text_wrapper_single_video_uses_default_item_order():
    """A single-modality (video-only) request bypasses get_mm_item_order and uses the default order."""

    class _SingleVideoProcessor(_FakeMixedProcessor):
        def __call__(self, inputs, sampling_params):
            return [10] + [_VIDEO_FEATURE_ID] * _VIDEO_TOKENS + [13], {
                "multimodal_data": {"video": {}}
            }

        def get_mm_item_order(self, prompt_token_ids, mm_data):
            raise AssertionError("single-modality requests should not need prompt-order hooks")

        def get_mm_token_ids(self):
            return torch.tensor([_VIDEO_FEATURE_ID])

    processor = _SingleVideoProcessor()
    wrapper = create_input_processor_with_hash(processor)
    prompt_token_ids, extra = wrapper(
        {"prompt": "video prompt", "multi_modal_data": {"video": [[torch.tensor([2])]]}},
        SamplingParams(),
    )

    assert prompt_token_ids == [10] + [_VIDEO_FEATURE_ID] * _VIDEO_TOKENS + [13]
    assert extra["multimodal_input"].multimodal_positions == [1]
    assert extra["multimodal_input"].multimodal_lengths == [_VIDEO_TOKENS]
    assert extra["multimodal_data"]["multimodal_item_order"] == [{"modality": "video", "index": 0}]


# Interleaved image, audio, image prompt with distinct per-item token counts, so
# the flattened list is order-sensitive: only a prompt-order flatten yields
# [2, 5, 3] (img0, audio0, img1); the old next(iter(...)) path returned just one
# modality's bucket (images -> [2, 3]).
_IMG0_TOKENS = 2
_IMG1_TOKENS = 3
_AUDIO0_TOKENS = 5

# Prompt-item-ordered counts the real resolve().flatten() must hand the expand hook.
_EXPECTED_PROMPT_ORDER_TOKENS = [_IMG0_TOKENS, _AUDIO0_TOKENS, _IMG1_TOKENS]

# Image-only bucket the old single-modality path produced (dropping audio).
_OLD_SINGLE_MODALITY_TOKENS = [_IMG0_TOKENS, _IMG1_TOKENS]


class _FakeTokenIdMMProcessor(BaseMultimodalInputProcessor):
    """Minimal `BaseMultimodalInputProcessor` for the tokenized fast path.

    Drives `call_with_token_ids` on a mixed interleaved prompt (image, audio,
    image), with the expand hook stubbed so the real find_mm_token_lengths +
    resolve().flatten() run without a GPU/HF model.
    """

    # Opt in to the tokenized+MM fast path so `call_with_token_ids` runs.
    supports_token_id_mm_expansion = True

    def __init__(self):
        # Bypass BaseMultimodalInputProcessor.__init__; set only the attributes
        # the exercised path reads.
        self._tokenizer = None
        self._config = None
        self._model_path = None
        self._use_fast = True
        self._trust_remote_code = True
        self._multimodal_hashing_supported = None
        # Captures the per-item token counts handed to the expand hook.
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
        # Invoked on the synthetic placeholder text prompt, so it must not
        # re-enter the token-id path.
        assert inputs.get("prompt") is not None
        assert inputs.get("prompt_token_ids") is None
        return [], {"multimodal_data": {"image": {}, "audio": {}}}

    # --- tokenized+MM fast-path hooks ---
    def get_text_with_mm_placeholders(self, mm_counts):
        return "".join(f"<{modality}>" * count for modality, count in mm_counts.items())

    def get_mm_item_order(self, prompt_token_ids, mm_data):
        # Parse the interleaved prompt into (modality, index) in prompt order:
        # image, audio, image -> [(image,0), (audio,0), (image,1)].
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
    # mm_data is modality-major (image[0], image[1], audio[0]) but the prompt
    # interleaves, so a prompt-order flatten must cross modalities.
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
    """call_with_token_ids flattens mixed-modality counts in prompt-item order."""
    # Interleaved image, audio, image: the per-item counts handed to the expand
    # hook must be prompt-ordered across modalities, i.e. [img0, audio0, img1] =
    # [2, 5, 3].
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
    # The token-id path bakes the resolved prompt order into metadata as the
    # top-level `multimodal_item_order` key (same wire format the text path
    # uses), so the hashing wrapper inherits the correct order instead of
    # re-resolving it from a stale dummy-text scan.
    assert extra["multimodal_data"]["multimodal_item_order"] == [
        {"modality": "image", "index": 0},
        {"modality": "audio", "index": 0},
        {"modality": "image", "index": 1},
    ]


# Embedding lengths the dummy-placeholder pass bakes in MODALITY-MAJOR order
# (image[0], image[1], audio[0]) — mirroring Nano's `_process_mixed_modalities`,
# which runs `compute_mm_embedding_lengths` on the synthetic modality-major
# placeholder text. Deliberately distinct per item so an order mismatch is
# observable, and deliberately != num_mm_tokens so a stale value is not silently
# the same list as the count flatten.
_STALE_MODALITY_MAJOR_EMBED_LENGTHS = [_IMG0_TOKENS, _IMG1_TOKENS, _AUDIO0_TOKENS]
# The same per-item embedding lengths re-ordered to the REAL interleaved prompt
# order (image[0], audio[0], image[1]) the consumer indexes by.
_EXPECTED_PROMPT_ORDER_EMBED_LENGTHS = [_IMG0_TOKENS, _AUDIO0_TOKENS, _IMG1_TOKENS]


class _FakeStaleLengthsProcessor(_FakeTokenIdMMProcessor):
    """`_FakeTokenIdMMProcessor` whose dummy pass bakes stale lengths.

    Mirrors Nano: the dummy-placeholder pass (driven by the modality-major
    `get_text_with_mm_placeholders` text) bakes per-item embedding lengths in
    modality-major order. For an interleaved real prompt the consumer indexes
    those lengths by the interleaved item order, so a stale modality-major list
    yields the wrong item's row count.
    """

    def call_with_text_prompt(self, inputs, sampling_params):
        prompt_token_ids, extra = super().call_with_text_prompt(inputs, sampling_params)
        # The dummy pass bakes lengths in modality-major order (image, image,
        # audio) — stale w.r.t. the interleaved real prompt resolved later.
        extra["multimodal_data"]["multimodal_embedding_lengths"] = list(
            _STALE_MODALITY_MAJOR_EMBED_LENGTHS
        )
        return prompt_token_ids, extra

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids,
        num_mm_tokens_per_placeholder,
        hf_processor_mm_kwargs=None,
        mm_data=None,
    ):
        # Build a realistic expanded prompt: each item's placeholder becomes the
        # matching number of that modality's feature tokens, in the resolved
        # (interleaved) prompt order. This lets the registry's mask-based
        # re-bake of `multimodal_embedding_lengths` count real embed slots.
        self.lengths_seen_by_expander = list(num_mm_tokens_per_placeholder)
        assert mm_data is not None
        feature_id = {"image": _IMAGE_FEATURE_ID, "audio": _AUDIO_FEATURE_ID}
        # `item_order` for this prompt is image, audio, image (see
        # `get_mm_item_order`); zip it with the resolved per-item counts.
        order = self.get_mm_item_order(prompt_token_ids, mm_data)
        expanded = [10]
        for (modality, _), count in zip(order, num_mm_tokens_per_placeholder):
            expanded.extend([feature_id[modality]] * count)
            expanded.append(13)
        return expanded, None

    def get_vocab_size(self):
        return None

    def get_mm_token_ids(self):
        return torch.tensor([_IMAGE_FEATURE_ID, _AUDIO_FEATURE_ID])

    def get_mm_special_token_ids(self):
        return None


def test_call_with_token_ids_emits_embedding_lengths_in_prompt_order():
    """call_with_token_ids re-bakes multimodal_embedding_lengths in prompt order.

    Lengths-shaped twin of the item_order fix: the dummy-placeholder pass bakes a
    stale MODALITY-MAJOR `multimodal_embedding_lengths` (image, image, audio).
    `call_with_token_ids` re-bakes `multimodal_item_order` in the real interleaved
    order (image, audio, image) but must ALSO emit `multimodal_embedding_lengths`
    in that same real order, so a consumer indexing lengths by the interleaved
    item order reads each item's true row count — not the stale modality-major
    list.
    """
    processor = _FakeStaleLengthsProcessor()
    inputs = _mixed_interleaved_inputs()

    _, extra = processor.call_with_token_ids(inputs, SamplingParams())

    embed_lengths = extra["multimodal_data"]["multimodal_embedding_lengths"]
    # Lengths track the REAL interleaved prompt order (image, audio, image).
    assert embed_lengths == _EXPECTED_PROMPT_ORDER_EMBED_LENGTHS
    # Sanity: the stale modality-major value would have been observably wrong.
    assert _STALE_MODALITY_MAJOR_EMBED_LENGTHS != _EXPECTED_PROMPT_ORDER_EMBED_LENGTHS
    # And lengths align position-for-position with the re-baked item order.
    item_order = extra["multimodal_data"]["multimodal_item_order"]
    assert item_order == [
        {"modality": "image", "index": 0},
        {"modality": "audio", "index": 0},
        {"modality": "image", "index": 1},
    ]
    assert len(embed_lengths) == len(item_order)


# --- Single-modality token-id path: stale dummy-text lengths get re-baked ----
#
# Companion to the mixed-interleaved `test_call_with_token_ids_emits_embedding_
# lengths_in_prompt_order`. A SINGLE-modality (multi-image) token-id request
# never reorders across modalities -- `call_with_token_ids` takes the
# `len(mm_items) <= 1` default-order branch -- but the producer-side bake (the
# Nano text path's `multimodal_embedding_lengths`) runs on the synthetic
# placeholder text, whose tiler can plan a DIFFERENT per-image row count than the
# real prompt (Nano's dynamic-resolution image tiling is budget-adaptive). So the
# dummy pass can leave per-item lengths whose COUNTS (not order) are stale w.r.t.
# the real prompt. `call_with_token_ids` unconditionally re-bakes
# `multimodal_embedding_lengths` from the REAL expanded prompt ids
# (`compute_mm_embedding_lengths`, registry.py), so no separate pop of the stale
# lengths is needed -- the re-bake overwrites them. This guards that re-bake for
# the single-modality shape the B-lite producer bake newly emits.
_IMG0_REAL_TOKENS = 2
_IMG1_REAL_TOKENS = 4
# Real per-image embed rows, in prompt order (the encoder's actual output rows).
_REAL_IMAGE_EMBED_LENGTHS = [_IMG0_REAL_TOKENS, _IMG1_REAL_TOKENS]
# Stale per-image lengths the dummy-placeholder pass baked (the dummy tiler
# planned more rows than the real prompt). Deliberately distinct so an un-rebaked
# stale value is observably wrong, and same length (2 items) so a wrong-COUNT
# (not wrong-shape) regression is what is caught.
_STALE_DUMMY_IMAGE_EMBED_LENGTHS = [7, 9]


class _FakeSingleModalityStaleLengthsProcessor(BaseMultimodalInputProcessor):
    """Single-modality (two-image) token-id processor with a stale dummy bake.

    `call_with_text_prompt` (the dummy-placeholder pass) bakes a per-image
    `multimodal_embedding_lengths` with the WRONG counts, mimicking Nano's
    dummy-vs-real tiling divergence. `call_with_token_ids` must re-bake the real
    counts from the expanded prompt ids.
    """

    supports_token_id_mm_expansion = True

    def __init__(self):
        # Bypass BaseMultimodalInputProcessor.__init__; set only what the
        # exercised path reads.
        self._tokenizer = None
        self._config = None
        self._model_path = None
        self._use_fast = True
        self._trust_remote_code = True
        self._multimodal_hashing_supported = None

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
        # The dummy pass runs on synthetic placeholder text; it bakes stale
        # per-image lengths (wrong counts) the token-id path must overwrite.
        assert inputs.get("prompt") is not None
        assert inputs.get("prompt_token_ids") is None
        return [], {
            "multimodal_data": {
                "image": {},
                "multimodal_embedding_lengths": list(_STALE_DUMMY_IMAGE_EMBED_LENGTHS),
            }
        }

    def get_text_with_mm_placeholders(self, mm_counts):
        return "".join(f"<{modality}>" * count for modality, count in mm_counts.items())

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids,
        num_mm_tokens_per_placeholder,
        hf_processor_mm_kwargs=None,
        mm_data=None,
    ):
        # Expand each image placeholder into its REAL per-image feature-token run
        # (img0 -> 2, img1 -> 4), so the registry's mask-based re-bake counts the
        # real embed slots.
        assert mm_data is not None
        expanded = [10]
        for count in num_mm_tokens_per_placeholder:
            expanded.extend([_IMAGE_FEATURE_ID] * count)
            expanded.append(13)
        return expanded, None

    def get_num_tokens_per_image(self, *, image, **kwargs):
        # img0 -> 2 tokens, img1 -> 4 tokens (distinct so order/count is
        # observable). Keyed off the per-image payload tensor value.
        return _IMG0_REAL_TOKENS if image.tolist() == [1] else _IMG1_REAL_TOKENS

    def get_vocab_size(self):
        return None

    def get_mm_token_ids(self):
        return torch.tensor([_IMAGE_FEATURE_ID])

    def get_mm_special_token_ids(self):
        return None


def test_call_with_token_ids_single_modality_rebakes_embedding_lengths():
    """Single-modality token-id path re-bakes lengths from the real prompt.

    The dummy-placeholder pass baked stale per-image lengths ([7, 9]).
    `call_with_token_ids` must emit `multimodal_embedding_lengths` matching the
    REAL per-image embed rows ([2, 4]) computed from the expanded prompt ids, not
    the stale dummy bake -- and without a separate pop (the re-bake overwrites).
    """
    processor = _FakeSingleModalityStaleLengthsProcessor()
    inputs = {
        "prompt_token_ids": [
            10,
            _IMAGE_PLACEHOLDER_ID,  # image[0]
            11,
            _IMAGE_PLACEHOLDER_ID,  # image[1]
            13,
        ],
        "multi_modal_data": {"image": [torch.tensor([1]), torch.tensor([2])]},
    }

    _, extra = processor.call_with_token_ids(inputs, SamplingParams())

    embed_lengths = extra["multimodal_data"]["multimodal_embedding_lengths"]
    # Lengths track the REAL per-image embed rows, not the stale dummy bake.
    assert embed_lengths == _REAL_IMAGE_EMBED_LENGTHS
    # Sanity: the stale dummy value would have been observably wrong.
    assert _STALE_DUMMY_IMAGE_EMBED_LENGTHS != _REAL_IMAGE_EMBED_LENGTHS
    # The single-modality order is the deterministic per-image default order.
    item_order = extra["multimodal_data"]["multimodal_item_order"]
    assert item_order == [
        {"modality": "image", "index": 0},
        {"modality": "image", "index": 1},
    ]
    # Lengths align position-for-position with the re-baked item order.
    assert len(embed_lengths) == len(item_order)
