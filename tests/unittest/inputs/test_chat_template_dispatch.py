# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for content-format-driven chat template dispatch and placeholder handling."""

import threading

import pytest

from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.inputs.utils import (
    ConversationMessage,
    _build_openai_content,
    _resolve_content_format,
    add_multimodal_placeholders,
    async_apply_chat_template,
    interleave_mm_placeholders,
)


@pytest.fixture(autouse=True, scope="module")
def _register_test_models():
    """Register temporary test models for each test, then clean up."""
    # STRING format model (explicit)
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "test_string_model",
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<image>", "video": "<video>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            placeholders_separator="\n",
            content_format=ContentFormat.STRING,
        ),
    )
    # OPENAI format model (explicit)
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "test_openai_model",
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<image>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            placeholders_separator="\n",
            content_format=ContentFormat.OPENAI,
        ),
    )
    # Auto-detect model (content_format=None)
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "test_auto_model",
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<img>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            placeholders_separator="\n",
        ),
    )
    # PASSTHROUGH format model (skip chat template entirely)
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "test_passthrough_model",
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<image>"},
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
            content_format=ContentFormat.PASSTHROUGH,
        ),
    )

    yield

    for mt in (
        "test_string_model",
        "test_openai_model",
        "test_auto_model",
        "test_passthrough_model",
    ):
        try:
            MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata(mt)
        except ValueError:
            pass


class TestResolveContentFormat:
    def test_explicit_string(self):
        fmt = _resolve_content_format("test_string_model", None)
        assert fmt == ContentFormat.STRING

    def test_explicit_openai(self):
        fmt = _resolve_content_format("test_openai_model", None)
        assert fmt == ContentFormat.OPENAI

    def test_auto_detect_openai_template(self):
        template = (
            "{% for m in messages %}"
            "{% for c in m['content'] %}"
            "{% if c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
            "{% endfor %}{% endfor %}"
        )
        fmt = _resolve_content_format("test_auto_model", template)
        assert fmt == ContentFormat.OPENAI

    def test_auto_detect_string_template(self):
        template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        fmt = _resolve_content_format("test_auto_model", template)
        assert fmt == ContentFormat.STRING

    def test_explicit_passthrough(self):
        fmt = _resolve_content_format("test_passthrough_model", None)
        assert fmt == ContentFormat.PASSTHROUGH

    def test_unknown_model_defaults_to_string(self):
        fmt = _resolve_content_format("unknown_model", None)
        assert fmt == ContentFormat.STRING


class TestBuildOpenAIContent:
    def test_with_content_parts(self):
        conv = ConversationMessage(
            role="user",
            content="Hello",
            media=[],
            content_parts=[
                "Hello",
                {"type": "image", "media_index": 0},
                " world",
            ],
        )
        result = _build_openai_content(conv, {"<image>": 1})
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "image"},
            {"type": "text", "text": " world"},
        ]

    def test_without_content_parts_fallback(self):
        conv = ConversationMessage(
            role="user",
            content="Hello",
            media=[],
        )
        result = _build_openai_content(conv, {"<image>": 2})
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "image"},
            {"type": "image"},
        ]

    def test_video_modality_inference(self):
        conv = ConversationMessage(role="user", content="Describe", media=[])
        result = _build_openai_content(conv, {"<video>": 1})
        assert result == [
            {"type": "text", "text": "Describe"},
            {"type": "video"},
        ]


class TestInterleaveMmPlaceholders:
    def test_basic_interleaving(self):
        content_parts = [
            "Hello",
            {"type": "image", "media_index": 0},
            " what is this?",
        ]
        result = interleave_mm_placeholders(
            "test_string_model", content_parts, {"<image>": 1}, {"<image>": "image"}
        )
        assert result == "Hello\n<image>\n what is this?"

    def test_multiple_images(self):
        content_parts = [
            {"type": "image", "media_index": 0},
            "Compare these:",
            {"type": "image", "media_index": 1},
        ]
        result = interleave_mm_placeholders(
            "test_string_model", content_parts, {"<image>": 2}, {"<image>": "image"}
        )
        assert result == "<image>\nCompare these:\n<image>"

    def test_empty_content_parts_fallback(self):
        result = interleave_mm_placeholders(
            "test_string_model", [], {"<image>": 1}, {"<image>": "image"}
        )
        # Falls back to add_multimodal_placeholders
        assert "<image>" in result

    def test_mixed_modalities(self):
        content_parts = [
            {"type": "image", "media_index": 0},
            "Describe the image and video",
            {"type": "video", "media_index": 1},
        ]
        result = interleave_mm_placeholders(
            "test_string_model",
            content_parts,
            {"<image>": 1, "<video>": 1},
            {"<image>": "image", "<video>": "video"},
        )
        assert result == "<image>\nDescribe the image and video\n<video>"

    def test_numbered_placeholders(self):
        """Models like Phi4MM use unique per-item placeholders."""
        content_parts = [
            {"type": "image", "media_index": 0},
            "Compare these two images:",
            {"type": "image", "media_index": 1},
        ]
        result = interleave_mm_placeholders(
            "test_string_model",
            content_parts,
            {"<|image_1|>": 1, "<|image_2|>": 1},
            {"<|image_1|>": "image", "<|image_2|>": "image"},
        )
        assert result == "<|image_1|>\nCompare these two images:\n<|image_2|>"

    def test_numbered_placeholders_three_images(self):
        """Three unique image placeholders interleaved with text."""
        content_parts = [
            "First:",
            {"type": "image", "media_index": 0},
            "Second:",
            {"type": "image", "media_index": 1},
            "Third:",
            {"type": "image", "media_index": 2},
        ]
        result = interleave_mm_placeholders(
            "test_string_model",
            content_parts,
            {"<|image_1|>": 1, "<|image_2|>": 1, "<|image_3|>": 1},
            {"<|image_1|>": "image", "<|image_2|>": "image", "<|image_3|>": "image"},
        )
        assert result == "First:\n<|image_1|>\nSecond:\n<|image_2|>\nThird:\n<|image_3|>"

    def test_shared_placeholder_with_count(self):
        """Shared placeholder repeated by count (non-numbered models)."""
        content_parts = [
            {"type": "image", "media_index": 0},
            "text",
            {"type": "image", "media_index": 1},
        ]
        result = interleave_mm_placeholders(
            "test_string_model",
            content_parts,
            {"<image>": 2},
            {"<image>": "image"},
        )
        assert result == "<image>\ntext\n<image>"

    def test_ambiguous_placeholder_resolved_by_mapping(self):
        """Explicit mapping resolves placeholders that have no modality hint."""
        content_parts = [
            {"type": "video", "media_index": 0},
            "text",
        ]
        result = interleave_mm_placeholders(
            "test_string_model",
            content_parts,
            {"<media_token>": 1},
            {"<media_token>": "video"},
        )
        assert result == "<media_token>\ntext"

    def test_missing_placeholder_in_mapping_raises(self):
        """KeyError when a placeholder is not in the mapping."""
        content_parts = [
            {"type": "image", "media_index": 0},
        ]
        with pytest.raises(KeyError, match="not found in placeholder_modalities"):
            interleave_mm_placeholders(
                "test_string_model",
                content_parts,
                {"<image>": 1},
                {},
            )


class TestAddMultimodalPlaceholdersDedup:
    """Tests for placeholder deduplication in add_multimodal_placeholders.

    When a client (e.g. VLMEvalKit) already embeds <image> in the prompt text
    AND sends image data via image_url, TRT-LLM must not insert a duplicate.
    """

    def test_no_duplicate_when_placeholder_already_in_text(self):
        result = add_multimodal_placeholders(
            "test_string_model",
            "<image>\nWhat is shown?",
            {"<image>": 1},
        )
        assert result.count("<image>") == 1
        assert result == "<image>\nWhat is shown?"

    def test_adds_placeholder_when_not_in_text(self):
        result = add_multimodal_placeholders(
            "test_string_model",
            "What is shown?",
            {"<image>": 1},
        )
        assert result.count("<image>") == 1
        assert result == "<image>\nWhat is shown?"

    def test_partial_dedup_multiple_images(self):
        """Text has 1 placeholder but 3 images — should add 2 more."""
        result = add_multimodal_placeholders(
            "test_string_model",
            "<image>\nCompare these images",
            {"<image>": 3},
        )
        assert result.count("<image>") == 3

    def test_no_extra_when_all_present(self):
        """Text already has all placeholders — nothing to add."""
        text = "<image>\n<image>\nCompare"
        result = add_multimodal_placeholders(
            "test_string_model",
            text,
            {"<image>": 2},
        )
        assert result == text
        assert result.count("<image>") == 2

    def test_excess_existing_placeholders_preserved(self):
        """Text has more placeholders than expected — don't remove any."""
        text = "<image>\n<image>\n<image>\nDescribe"
        result = add_multimodal_placeholders(
            "test_string_model",
            text,
            {"<image>": 2},
        )
        assert result == text
        assert result.count("<image>") == 3


class TestAsyncApplyChatTemplate:
    @pytest.mark.asyncio
    async def test_runs_in_worker_thread(self):
        event_loop_thread_id = threading.current_thread().ident

        class TrackingTokenizer:
            def __init__(self):
                self.worker_thread_id = None

            def apply_chat_template(self, **_):
                self.worker_thread_id = threading.current_thread().ident
                return "rendered"

        tokenizer = TrackingTokenizer()

        result = await async_apply_chat_template(
            model_type="test_string_model",
            tokenizer=tokenizer,
            processor=None,
            conversation=[ConversationMessage(role="user", content="hello", media=[])],
            add_generation_prompt=True,
            mm_placeholder_counts=[{}],
            chat_template="{{ messages }}",
        )

        assert result == "rendered"
        assert tokenizer.worker_thread_id is not None
        assert tokenizer.worker_thread_id != event_loop_thread_id


class TestServingChatTemplateGather:
    """Cover the asyncio.gather integration in the serving chat-template paths."""

    @pytest.mark.asyncio
    async def test_resource_governor_convert_messages(self, monkeypatch):
        from unittest.mock import Mock

        import tensorrt_llm.serve.resource_governor as rg

        governor = object.__new__(rg.ResourceGovernor)
        governor.model_config = Mock()
        governor.tokenizer = Mock()
        governor.processor = None

        async def fake_mm_coroutine():
            # parse_chat_messages_coroutines' coroutine yields
            # (mm_data, mm_embeddings).
            return ({"image": ["data"]}, None)

        monkeypatch.setattr(
            rg,
            "parse_chat_messages_coroutines",
            lambda messages, model_config, _: ([], fake_mm_coroutine(), [{}]),
        )
        # Must resolve the top-level model type, matching the serving call
        # sites (not the raw model_config.model_type).
        monkeypatch.setattr(rg, "resolve_top_level_model_type", lambda cfg: "resolved-model-type")

        captured = {}

        async def fake_async_apply(**kwargs):
            captured.update(kwargs)
            return [1, 2, 3]

        monkeypatch.setattr(rg, "async_apply_chat_template", fake_async_apply)

        token_ids = await governor._convert_messages(
            messages=[{"role": "user", "content": "hi"}],
            tool_dicts=None,
            add_generation_prompt=True,
            documents=None,
            chat_template=None,
            chat_template_kwargs=None,
        )

        # Returns only token_ids, not the (mm_data, mm_embeddings) tuple.
        assert token_ids == [1, 2, 3]
        # Uses the top-level resolver and forwards the real placeholder counts.
        assert captured["model_type"] == "resolved-model-type"
        assert captured["mm_placeholder_counts"] == [{}]

    @pytest.mark.asyncio
    async def test_responses_create_input_tokens_unpacks_mm_tuple(self, monkeypatch):
        """_create_input_tokens must return mm_data, not the whole gather tuple."""
        from unittest.mock import Mock

        import tensorrt_llm.serve.responses_utils as ru

        async def fake_create_input_messages(request, prev_msgs):
            return [{"role": "user", "content": "hi"}]

        async def fake_mm_coroutine():
            return ({"image": ["data"]}, {"image": ["embed"]})

        monkeypatch.setattr(ru, "_create_input_messages", fake_create_input_messages)
        monkeypatch.setattr(
            ru,
            "parse_chat_messages_coroutines",
            lambda messages, model_config: ([], fake_mm_coroutine(), [{}]),
        )
        monkeypatch.setattr(ru, "resolve_top_level_model_type", lambda cfg: "resolved-model-type")
        monkeypatch.setattr(ru, "_get_chat_completion_function_tools", lambda tools: [])

        async def fake_async_apply(**kwargs):
            return [1, 2, 3]

        monkeypatch.setattr(ru, "async_apply_chat_template", fake_async_apply)

        request = Mock()
        request.tools = None
        request.store = False

        token_ids, mm_data = await ru._create_input_tokens(
            request=request,
            prev_response=None,
            prev_msgs=None,
            conversation_store=None,
            enable_store=False,
            tokenizer=Mock(),
            model_config=Mock(),
            processor=None,
        )

        assert token_ids == [1, 2, 3]
        # mm_data is the data dict, not the (mm_data, mm_embeddings) tuple.
        assert mm_data == {"image": ["data"]}
