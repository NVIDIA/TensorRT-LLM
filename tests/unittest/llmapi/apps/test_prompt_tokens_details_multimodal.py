# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tensorrt_llm.llmapi.disagg_utils import get_usage_tokens_from_ctx, rewrite_usage_info_from_ctx
from tensorrt_llm.serve.harmony_adapter import (
    _create_usage_info,
    handle_non_streaming_response,
    handle_streaming_response,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    CompletionResponseChoice,
    PromptTokensDetails,
    StreamOptions,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    chat_response_post_processor,
    chat_stream_post_processor,
    completion_response_post_processor,
    completion_stream_post_processor,
)


class TestPromptTokensDetailsSchema:
    def test_text_only_prompt_tokens_details_serialization(self):
        """When modality tokens are None, serialization should automatically omit them."""
        details = PromptTokensDetails(cached_tokens=0)
        dumped_json = json.loads(details.model_dump_json())
        dumped_dict = details.model_dump()
        assert dumped_json == {"cached_tokens": 0}
        assert dumped_dict == {"cached_tokens": 0}
        assert "image_tokens" not in dumped_json
        assert "video_tokens" not in dumped_json
        assert "audio_tokens" not in dumped_json

    def test_image_tokens_details_serialization(self):
        """When image_tokens is provided, it should be serialized."""
        details = PromptTokensDetails(cached_tokens=10, image_tokens=576)
        dumped = json.loads(details.model_dump_json())
        assert dumped == {"cached_tokens": 10, "image_tokens": 576}
        assert "video_tokens" not in dumped
        assert "audio_tokens" not in dumped

    def test_video_tokens_details_serialization(self):
        """When video_tokens is provided, it should be serialized."""
        details = PromptTokensDetails(cached_tokens=0, video_tokens=1024)
        dumped = json.loads(details.model_dump_json())
        assert dumped == {"cached_tokens": 0, "video_tokens": 1024}
        assert "image_tokens" not in dumped

    def test_mixed_multimodal_tokens_details_serialization(self):
        """When multiple modalities are present, all non-None fields should be serialized."""
        details = PromptTokensDetails(
            cached_tokens=5, image_tokens=256, video_tokens=512, audio_tokens=64
        )
        dumped = json.loads(details.model_dump_json())
        assert dumped == {
            "cached_tokens": 5,
            "image_tokens": 256,
            "video_tokens": 512,
            "audio_tokens": 64,
        }


class TestPostprocessHandlersMultimodalUsage:
    def _make_mock_output(self, token_ids=None):
        out = MagicMock()
        out.index = 0
        out.token_ids = token_ids or [1, 2, 3]
        out.text = "hello"
        out.text_diff = "hello"
        out.token_ids_diff = token_ids or [1, 2, 3]
        out.length = len(out.token_ids)
        out.finish_reason = "stop"
        out.stop_reason = None
        out.disaggregated_params = None
        out.logprobs = None
        out.logprobs_diff = None
        return out

    def _make_mock_rsp(self, cached_tokens=0, done=True):
        rsp = MagicMock()
        rsp.id = 123
        rsp._done = done
        rsp.cached_tokens = cached_tokens
        rsp.context_logits = None
        rsp.avg_decoded_tokens_per_iter = 1.0
        out = self._make_mock_output()
        rsp.outputs = [out]
        return rsp

    def test_chat_response_post_processor_with_modality_tokens(self):
        args = ChatPostprocArgs(
            model="qwen-vl",
            num_prompt_tokens=1500,
            image_tokens=576,
            video_tokens=800,
        )
        rsp = self._make_mock_rsp(cached_tokens=50)
        response = chat_response_post_processor(rsp, args)

        assert response.usage.prompt_tokens == 1500
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 1503
        assert response.usage.prompt_tokens_details.cached_tokens == 50
        assert response.usage.prompt_tokens_details.image_tokens == 576
        assert response.usage.prompt_tokens_details.video_tokens == 800
        assert response.usage.prompt_tokens_details.audio_tokens is None

    def test_chat_response_text_only_exclude_none_serialization(self):
        args = ChatPostprocArgs(
            model="llama-3",
            num_prompt_tokens=50,
        )
        rsp = self._make_mock_rsp(cached_tokens=0)
        response = chat_response_post_processor(rsp, args)
        dumped = response.model_dump()
        assert dumped["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
        assert "image_tokens" not in dumped["usage"]["prompt_tokens_details"]
        assert "video_tokens" not in dumped["usage"]["prompt_tokens_details"]
        assert "audio_tokens" not in dumped["usage"]["prompt_tokens_details"]

    def test_chat_stream_post_processor_final_usage_chunk(self):
        args = ChatPostprocArgs(
            model="qwen-vl",
            num_prompt_tokens=1200,
            image_tokens=600,
            stream_options=StreamOptions(include_usage=True),
        )
        rsp = self._make_mock_rsp(cached_tokens=10, done=True)
        chunks = chat_stream_post_processor(rsp, args)

        # Look for the final usage chunk
        usage_chunks = [c for c in chunks if "usage" in c and 'choices":[]' in c]
        assert len(usage_chunks) == 1
        raw_json = usage_chunks[0].replace("data: ", "").strip()
        data = json.loads(raw_json)
        usage = data["usage"]
        assert usage["prompt_tokens"] == 1200
        assert usage["prompt_tokens_details"]["cached_tokens"] == 10
        assert usage["prompt_tokens_details"]["image_tokens"] == 600
        assert "video_tokens" not in usage["prompt_tokens_details"]

    def test_completion_response_post_processor_with_modality_tokens(self):
        args = CompletionPostprocArgs(
            model="test-model",
            num_prompt_tokens=200,
            video_tokens=150,
        )
        rsp = self._make_mock_rsp(cached_tokens=20)
        response = completion_response_post_processor(rsp, args)

        assert response.usage.prompt_tokens == 200
        assert response.usage.prompt_tokens_details.cached_tokens == 20
        assert response.usage.prompt_tokens_details.video_tokens == 150
        assert response.usage.prompt_tokens_details.image_tokens is None

    def test_completion_response_text_only_exclude_none_serialization(self):
        args = CompletionPostprocArgs(
            model="test-model",
            num_prompt_tokens=100,
        )
        rsp = self._make_mock_rsp(cached_tokens=0)
        response = completion_response_post_processor(rsp, args)
        dumped = response.model_dump()
        assert dumped["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
        assert "image_tokens" not in dumped["usage"]["prompt_tokens_details"]
        assert "video_tokens" not in dumped["usage"]["prompt_tokens_details"]
        assert "audio_tokens" not in dumped["usage"]["prompt_tokens_details"]

    def test_completion_stream_post_processor_exclude_none_serialization(self):
        args = CompletionPostprocArgs(
            model="test-model",
            num_prompt_tokens=100,
            stream_options=StreamOptions(include_usage=True, continuous_usage_stats=True),
        )
        rsp = self._make_mock_rsp(cached_tokens=5, done=True)
        chunks = completion_stream_post_processor(rsp, args)
        assert len(chunks) > 0
        for chunk_str in chunks:
            raw_json = chunk_str.replace("data: ", "").strip()
            data = json.loads(raw_json)
            if "usage" in data and data["usage"] is not None:
                details = data["usage"]["prompt_tokens_details"]
                assert details["cached_tokens"] == 5
                assert "image_tokens" not in details
                assert "video_tokens" not in details
                assert "audio_tokens" not in details


class TestMergeCompletionResponses:
    def _make_completion_response(
        self,
        prompt_tokens=100,
        completion_tokens=10,
        cached_tokens=0,
        image_tokens=None,
        video_tokens=None,
        audio_tokens=None,
    ):
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=cached_tokens,
                image_tokens=image_tokens,
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
            ),
        )
        return CompletionResponse(
            model="test-model",
            choices=[CompletionResponseChoice(index=0, text="hi", finish_reason="stop")],
            usage=usage,
        )

    def test_merge_all_present_modalities(self):
        rsp1 = self._make_completion_response(
            prompt_tokens=100,
            completion_tokens=10,
            cached_tokens=10,
            image_tokens=200,
            video_tokens=100,
            audio_tokens=50,
        )
        rsp2 = self._make_completion_response(
            prompt_tokens=200,
            completion_tokens=20,
            cached_tokens=20,
            image_tokens=300,
            video_tokens=150,
            audio_tokens=25,
        )
        rsps = [rsp1, rsp2]
        merged = OpenAIServer.merge_completion_responses(rsps, model="test-model")

        assert merged.usage.prompt_tokens == 300
        assert merged.usage.completion_tokens == 30
        assert merged.usage.total_tokens == 330
        assert merged.usage.prompt_tokens_details.cached_tokens == 30
        assert merged.usage.prompt_tokens_details.image_tokens == 500
        assert merged.usage.prompt_tokens_details.video_tokens == 250
        assert merged.usage.prompt_tokens_details.audio_tokens == 75

    def test_merge_mixed_present_and_absent_modalities(self):
        rsp1 = self._make_completion_response(
            prompt_tokens=100, image_tokens=200, video_tokens=None
        )
        rsp2 = self._make_completion_response(
            prompt_tokens=100, image_tokens=None, video_tokens=None
        )
        rsps = [rsp1, rsp2]
        merged = OpenAIServer.merge_completion_responses(rsps, model="test-model")

        # image_tokens is present in rsp1 but None in rsp2 -> aggregate must be None
        assert merged.usage.prompt_tokens_details.image_tokens is None
        assert merged.usage.prompt_tokens_details.video_tokens is None

    def test_merge_explicit_zero_preserved(self):
        rsp1 = self._make_completion_response(prompt_tokens=100, image_tokens=0)
        rsp2 = self._make_completion_response(prompt_tokens=100, image_tokens=0)
        rsps = [rsp1, rsp2]
        merged = OpenAIServer.merge_completion_responses(rsps, model="test-model")

        assert merged.usage.prompt_tokens_details.image_tokens == 0


class TestDisaggUtilsModalityTokens:
    def test_rewrite_usage_info_from_ctx_preserves_modality_tokens(self):
        ctx_usage = UsageInfo(
            prompt_tokens=1000,
            completion_tokens=0,
            total_tokens=1000,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=100,
                image_tokens=500,
                video_tokens=300,
                audio_tokens=50,
            ),
        )
        gen_usage = UsageInfo(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        rewritten = rewrite_usage_info_from_ctx(gen_usage, ctx_usage)

        assert rewritten.prompt_tokens == 1000
        assert rewritten.completion_tokens == 20
        assert rewritten.total_tokens == 1020
        assert rewritten.prompt_tokens_details.cached_tokens == 100
        assert rewritten.prompt_tokens_details.image_tokens == 500
        assert rewritten.prompt_tokens_details.video_tokens == 300
        assert rewritten.prompt_tokens_details.audio_tokens == 50

    def test_get_usage_tokens_from_ctx_with_dict(self):
        ctx_usage_dict = {
            "prompt_tokens": 800,
            "prompt_tokens_details": {
                "cached_tokens": 120,
                "image_tokens": 400,
            },
        }
        prompt_tokens, cached_tokens = get_usage_tokens_from_ctx(ctx_usage_dict)
        assert prompt_tokens == 800
        assert cached_tokens == 120

    def test_rewrite_usage_info_from_ctx_with_dict_details(self):
        ctx_usage_dict = {
            "prompt_tokens": 900,
            "prompt_tokens_details": {
                "cached_tokens": 150,
                "image_tokens": 450,
                "video_tokens": 200,
                "audio_tokens": 50,
            },
        }
        gen_usage = UsageInfo(prompt_tokens=10, completion_tokens=30, total_tokens=40)
        rewritten = rewrite_usage_info_from_ctx(gen_usage, ctx_usage_dict)

        assert rewritten.prompt_tokens == 900
        assert rewritten.completion_tokens == 30
        assert rewritten.total_tokens == 930
        assert rewritten.prompt_tokens_details.cached_tokens == 150
        assert rewritten.prompt_tokens_details.image_tokens == 450
        assert rewritten.prompt_tokens_details.video_tokens == 200
        assert rewritten.prompt_tokens_details.audio_tokens == 50


class TestHarmonyAdapterModalityTokens:
    def test_create_usage_info_with_modality_tokens(self):
        out = MagicMock()
        out.token_ids = [10, 20, 30]
        usage = _create_usage_info(
            num_prompt_tokens=500,
            outputs=[out],
            cached_tokens=50,
            image_tokens=200,
            video_tokens=100,
        )
        assert usage.prompt_tokens == 500
        assert usage.completion_tokens == 3
        assert usage.total_tokens == 503
        assert usage.prompt_tokens_details.cached_tokens == 50
        assert usage.prompt_tokens_details.image_tokens == 200
        assert usage.prompt_tokens_details.video_tokens == 100
        assert usage.prompt_tokens_details.audio_tokens is None

    @pytest.mark.parametrize("handler_type", ["streaming", "non_streaming"])
    def test_harmony_handlers_forward_modality_tokens(self, handler_type):
        out = MagicMock()
        out.token_ids = [10, 20, 30]
        out.token_ids_diff = [10, 20, 30]
        out.finish_reason = "stop"
        out.stop_reason = None
        out.disaggregated_params = None

        with patch("tensorrt_llm.serve.harmony_adapter.get_harmony_adapter") as mock_adapter_getter:
            mock_adapter = MagicMock()
            mock_adapter_getter.return_value = mock_adapter
            mock_adapter.harmony_output_to_openai.return_value = {
                "role": "assistant",
                "content": "test",
            }
            mock_adapter.create_openai_streaming_response.return_value = ([], False)

            if handler_type == "non_streaming":
                response = handle_non_streaming_response(
                    tools=[],
                    tool_choice="none",
                    outputs=[out],
                    model="gpt-4o",
                    num_prompt_tokens=500,
                    cached_tokens=50,
                    image_tokens=250,
                    video_tokens=150,
                    audio_tokens=75,
                )
                assert response.usage.prompt_tokens == 500
                assert response.usage.prompt_tokens_details.cached_tokens == 50
                assert response.usage.prompt_tokens_details.image_tokens == 250
                assert response.usage.prompt_tokens_details.video_tokens == 150
                assert response.usage.prompt_tokens_details.audio_tokens == 75
            else:
                result = MagicMock()
                result.outputs = [out]
                chunks = handle_streaming_response(
                    tools=[],
                    tool_choice="none",
                    result=result,
                    model="gpt-4o",
                    request_id="req-123",
                    done=True,
                    num_prompt_tokens=500,
                    first_iteration=True,
                    cached_tokens=50,
                    image_tokens=250,
                    video_tokens=150,
                    audio_tokens=75,
                )
                usage_chunks = [c for c in chunks if "usage" in c and 'choices":[]' in c]
                assert len(usage_chunks) == 1
                data = json.loads(usage_chunks[0].replace("data: ", "").strip())
                usage_details = data["usage"]["prompt_tokens_details"]
                assert usage_details["cached_tokens"] == 50
                assert usage_details["image_tokens"] == 250
                assert usage_details["video_tokens"] == 150
                assert usage_details["audio_tokens"] == 75


class TestOpenAIServerMultimodalTokenCollection:
    def test_find_mm_token_lengths_propagation_to_postproc_args(self):
        """Verify that when mm_data is provided, find_mm_token_lengths aggregates modality counts."""
        mm_data = {"image": ["img_data_1", "img_data_2"], "video": ["video_data"]}
        mock_proc = MagicMock()

        with patch("tensorrt_llm.inputs.multimodal.find_mm_token_lengths") as mock_find:
            mock_find.return_value = {
                "image": [576, 576],
                "video": [1024],
                "audio": [128],
            }
            res = mock_find(mm_data, mock_proc)
            image_tokens = sum(res["image"]) if "image" in res else None
            video_tokens = sum(res["video"]) if "video" in res else None
            audio_tokens = sum(res["audio"]) if "audio" in res else None

            assert image_tokens == 1152
            assert video_tokens == 1024
            assert audio_tokens == 128

    @pytest.mark.asyncio
    async def test_chat_multimodal_token_collection_and_propagation(self):
        """Verify multimodal token length collection and propagation in chat requests.

        When a chat request contains multimodal data, find_mm_token_lengths
        populates postproc_args with modality token sums and the returned response usage
        reflects image, video, and audio prompt tokens.
        """
        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        server.processor = MagicMock()
        server.model_config = None
        server.multimodal_server_config = None
        server.tokenizer = MagicMock()
        server.chat_template = None
        server.log_stats = False
        server._input_proc_executor = None
        server.await_disconnected = AsyncMock()

        generator = MagicMock()
        generator.args = MagicMock(num_postprocess_workers=0, reasoning_parser=None)
        promise = MagicMock()
        promise.prompt_token_ids = list(range(50))
        generator.generate_async.return_value = promise
        server.generator = generator

        async def fake_create_chat_response(promise, postproc_params, raw_request, disagg_params):
            args = postproc_params.postproc_args
            details = PromptTokensDetails(
                cached_tokens=0,
                image_tokens=args.image_tokens,
                video_tokens=args.video_tokens,
                audio_tokens=args.audio_tokens,
            )
            usage = UsageInfo(
                prompt_tokens=args.num_prompt_tokens,
                completion_tokens=5,
                total_tokens=args.num_prompt_tokens + 5,
                prompt_tokens_details=details,
            )
            return ChatCompletionResponse(
                id="chat-123",
                created=1000,
                model="test-model",
                choices=[],
                usage=usage,
            )

        server._create_chat_response = fake_create_chat_response

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

        async def mm_coro():
            return ({"image": [b"img"], "video": [b"vid"], "audio": [b"aud"]}, None)

        with (
            patch("tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines") as mock_parse,
            patch("tensorrt_llm.serve.openai_server.async_apply_chat_template") as mock_template,
            patch("tensorrt_llm.inputs.multimodal.find_mm_token_lengths") as mock_find_mm,
        ):
            mock_parse.return_value = ([], mm_coro(), None, None)
            mock_template.return_value = "rendered text"
            mock_find_mm.return_value = {
                "image": [576, 576],
                "video": [1024],
                "audio": [128],
            }

            resp = await server.openai_chat(request, raw_request=None)
            body = json.loads(resp.body.decode())
            details = body["usage"]["prompt_tokens_details"]

            assert details["image_tokens"] == 1152
            assert details["video_tokens"] == 1024
            assert details["audio_tokens"] == 128

    @pytest.mark.asyncio
    async def test_chat_multimodal_processor_priority(self):
        """Verify that generator.input_processor is prioritized over server.processor."""
        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        hf_processor = MagicMock(name="hf_processor")
        trt_processor = MagicMock(name="trt_processor")
        server.processor = hf_processor
        server.model_config = None
        server.multimodal_server_config = None
        server.tokenizer = MagicMock()
        server.chat_template = None
        server.log_stats = False
        server._input_proc_executor = None
        server.await_disconnected = AsyncMock()

        generator = MagicMock()
        generator.input_processor = trt_processor
        generator.args = MagicMock(num_postprocess_workers=0, reasoning_parser=None)
        promise = MagicMock()
        promise.prompt_token_ids = list(range(50))
        generator.generate_async.return_value = promise
        server.generator = generator

        async def fake_create_chat_response(promise, postproc_params, raw_request, disagg_params):
            return ChatCompletionResponse(
                id="chat-123",
                created=1000,
                model="test-model",
                choices=[],
                usage=UsageInfo(prompt_tokens=50),
            )

        server._create_chat_response = fake_create_chat_response

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

        async def mm_coro():
            return ({"image": [b"img"]}, None)

        with (
            patch("tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines") as mock_parse,
            patch("tensorrt_llm.serve.openai_server.async_apply_chat_template") as mock_template,
            patch("tensorrt_llm.inputs.multimodal.find_mm_token_lengths") as mock_find_mm,
        ):
            mock_parse.return_value = ([], mm_coro(), None, None)
            mock_template.return_value = "rendered text"
            mock_find_mm.return_value = {"image": [576]}

            await server.openai_chat(request, raw_request=None)

            # Assert that find_mm_token_lengths was called with trt_processor, not hf_processor
            mock_find_mm.assert_called_once()
            args, kwargs = mock_find_mm.call_args
            assert args[1] is trt_processor

    @pytest.mark.asyncio
    async def test_chat_multimodal_token_collection_exception_suppressed(self):
        """Verify that when find_mm_token_lengths raises an exception, the request succeeds.

        When find_mm_token_lengths fails, the exception is safely caught and logged,
        and the returned response safely omits modality token counts.
        """
        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        server.processor = MagicMock()
        server.model_config = None
        server.multimodal_server_config = None
        server.tokenizer = MagicMock()
        server.chat_template = None
        server.log_stats = False
        server._input_proc_executor = None
        server.await_disconnected = AsyncMock()

        generator = MagicMock()
        generator.args = MagicMock(num_postprocess_workers=0, reasoning_parser=None)
        promise = MagicMock()
        promise.prompt_token_ids = list(range(50))
        generator.generate_async.return_value = promise
        server.generator = generator

        async def fake_create_chat_response(promise, postproc_params, raw_request, disagg_params):
            args = postproc_params.postproc_args
            details = PromptTokensDetails(
                cached_tokens=0,
                image_tokens=args.image_tokens,
                video_tokens=args.video_tokens,
                audio_tokens=args.audio_tokens,
            )
            usage = UsageInfo(
                prompt_tokens=args.num_prompt_tokens,
                completion_tokens=5,
                total_tokens=args.num_prompt_tokens + 5,
                prompt_tokens_details=details,
            )
            return ChatCompletionResponse(
                id="chat-123",
                created=1000,
                model="test-model",
                choices=[],
                usage=usage,
            )

        server._create_chat_response = fake_create_chat_response

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

        async def mm_coro():
            return ({"image": [b"img"]}, None)

        with (
            patch("tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines") as mock_parse,
            patch("tensorrt_llm.serve.openai_server.async_apply_chat_template") as mock_template,
            patch(
                "tensorrt_llm.inputs.multimodal.find_mm_token_lengths",
                side_effect=RuntimeError("Token count error"),
            ),
        ):
            mock_parse.return_value = ([], mm_coro(), None, None)
            mock_template.return_value = "rendered text"

            resp = await server.openai_chat(request, raw_request=None)
            body = json.loads(resp.body.decode())
            details = body["usage"]["prompt_tokens_details"]

            assert details["cached_tokens"] == 0
            assert "image_tokens" not in details
            assert "video_tokens" not in details
            assert "audio_tokens" not in details

    @pytest.mark.asyncio
    async def test_chat_multimodal_token_collection_from_preprocessed_item_metadata(self):
        """Verify deriving token counts from preprocessed MultimodalEncoderItemMetadata post-preprocess."""
        from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata

        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        server.processor = MagicMock()
        server.model_config = None
        server.multimodal_server_config = None
        server.tokenizer = MagicMock()
        server.chat_template = None
        server.log_stats = False
        server._input_proc_executor = None
        server.await_disconnected = AsyncMock()

        item_metadata = MultimodalEncoderItemMetadata(
            item_refs=[("image", 0), ("video", 0)],
            encoder_token_lengths=[576, 1024],
            output_embedding_lengths=[576, 1024],
        )

        def mock_preprocess(prompt, sampling_params, disaggregated_params):
            return {
                "prompt_token_ids": [1, 2, 3],
                "multi_modal_data": {
                    "multimodal_encoder_item_metadata": item_metadata
                }
            }

        generator = MagicMock()
        generator.preprocess = mock_preprocess
        generator.args = MagicMock(num_postprocess_workers=0, reasoning_parser=None)
        promise = MagicMock()
        promise.prompt_token_ids = list(range(50))
        generator.generate_async.return_value = promise
        server.generator = generator

        async def fake_create_chat_response(promise, postproc_params, raw_request, disagg_params):
            args = postproc_params.postproc_args
            details = PromptTokensDetails(
                cached_tokens=0,
                image_tokens=args.image_tokens,
                video_tokens=args.video_tokens,
                audio_tokens=args.audio_tokens,
            )
            usage = UsageInfo(
                prompt_tokens=args.num_prompt_tokens,
                completion_tokens=5,
                total_tokens=args.num_prompt_tokens + 5,
                prompt_tokens_details=details,
            )
            return ChatCompletionResponse(
                id="chat-123",
                created=1000,
                model="test-model",
                choices=[],
                usage=usage,
            )

        server._create_chat_response = fake_create_chat_response

        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

        async def mm_coro():
            return ({"image": [b"img"], "video": [b"vid"]}, None)

        with (
            patch("tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines") as mock_parse,
            patch("tensorrt_llm.serve.openai_server.async_apply_chat_template") as mock_template,
            patch("tensorrt_llm.inputs.multimodal.find_mm_token_lengths") as mock_find_mm,
        ):
            mock_parse.return_value = ([], mm_coro(), None, None)
            mock_template.return_value = "rendered text"

            resp = await server.openai_chat(request, raw_request=None)
            body = json.loads(resp.body.decode())
            details = body["usage"]["prompt_tokens_details"]

            # Verify that image_tokens and video_tokens were derived from item_metadata
            assert details["image_tokens"] == 576
            assert details["video_tokens"] == 1024
            assert "audio_tokens" not in details
            # Verify that find_mm_token_lengths was NOT called because item_metadata was present
            mock_find_mm.assert_not_called()
