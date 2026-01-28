# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for gRPC server components."""

import pytest

from tensorrt_llm.grpc import trtllm_service_pb2 as pb2
from tensorrt_llm.grpc.grpc_request_manager import (
    create_disaggregated_params_from_proto,
    create_lora_request_from_proto,
    create_sampling_params_from_proto,
)

pytestmark = pytest.mark.threadleak(enabled=False)


class TestSamplingParamsConversion:
    """Tests for proto to SamplingParams conversion."""

    def test_basic_sampling_config(self):
        """Test basic sampling config conversion."""
        proto_config = pb2.SamplingConfig(
            beam_width=1,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        output_config = pb2.OutputConfig()

        params = create_sampling_params_from_proto(
            proto_config=proto_config,
            output_config=output_config,
            max_tokens=100,
        )

        assert params.max_tokens == 100
        assert params.temperature == 0.7
        assert params.top_k == 50
        assert params.top_p == 0.9

    def test_beam_search_config(self):
        """Test beam search configuration."""
        proto_config = pb2.SamplingConfig(
            beam_width=4,
            num_return_sequences=2,
            length_penalty=1.2,
            early_stopping=1,
        )
        output_config = pb2.OutputConfig()

        params = create_sampling_params_from_proto(
            proto_config=proto_config,
            output_config=output_config,
            max_tokens=50,
        )

        assert params.beam_width == 4
        assert params.n == 2
        assert params.length_penalty == 1.2

    def test_penalties_config(self):
        """Test penalty parameters conversion."""
        proto_config = pb2.SamplingConfig(
            repetition_penalty=1.1,
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )
        output_config = pb2.OutputConfig()

        params = create_sampling_params_from_proto(
            proto_config=proto_config,
            output_config=output_config,
            max_tokens=100,
        )

        assert params.repetition_penalty == 1.1
        assert params.presence_penalty == 0.5
        assert params.frequency_penalty == 0.3

    def test_logprobs_config(self):
        """Test logprobs configuration."""
        proto_config = pb2.SamplingConfig()
        output_config = pb2.OutputConfig(
            logprobs=5,
            prompt_logprobs=3,
        )

        params = create_sampling_params_from_proto(
            proto_config=proto_config,
            output_config=output_config,
            max_tokens=100,
        )

        assert params.logprobs == 5
        assert params.prompt_logprobs == 3

    def test_guided_decoding_json_schema(self):
        """Test guided decoding with JSON schema."""
        proto_config = pb2.SamplingConfig()
        output_config = pb2.OutputConfig()
        guided_decoding = pb2.GuidedDecodingParams(
            guide_type=pb2.GuidedDecodingParams.GUIDE_TYPE_JSON_SCHEMA,
            guide='{"type": "object", "properties": {"name": {"type": "string"}}}',
        )

        params = create_sampling_params_from_proto(
            proto_config=proto_config,
            output_config=output_config,
            max_tokens=100,
            guided_decoding=guided_decoding,
        )

        assert params.guided_decoding_params is not None
        assert params.guided_decoding_params.json_schema is not None

    def test_guided_decoding_regex(self):
        """Test guided decoding with regex."""
        proto_config = pb2.SamplingConfig()
        output_config = pb2.OutputConfig()
        guided_decoding = pb2.GuidedDecodingParams(
            guide_type=pb2.GuidedDecodingParams.GUIDE_TYPE_REGEX,
            guide=r"\d{3}-\d{4}",
        )

        params = create_sampling_params_from_proto(
            proto_config=proto_config,
            output_config=output_config,
            max_tokens=100,
            guided_decoding=guided_decoding,
        )

        assert params.guided_decoding_params is not None
        assert params.guided_decoding_params.regex is not None


class TestLoraRequestConversion:
    """Tests for proto to LoRARequest conversion."""

    def test_basic_lora_config(self):
        """Test basic LoRA config conversion."""
        lora_config = pb2.LoraConfig(task_id=123)

        request = create_lora_request_from_proto(lora_config)

        assert request is not None
        assert request.task_id == 123

    def test_none_lora_config(self):
        """Test None LoRA config returns None."""
        request = create_lora_request_from_proto(None)
        assert request is None


class TestDisaggregatedParamsConversion:
    """Tests for proto to DisaggregatedParams conversion."""

    def test_context_only_request(self):
        """Test context-only disaggregated request."""
        proto_params = pb2.DisaggregatedParams(
            request_type=pb2.DisaggregatedParams.REQUEST_TYPE_CONTEXT_ONLY,
            ctx_request_id="ctx-123",
        )

        params = create_disaggregated_params_from_proto(proto_params)

        assert params is not None
        assert params.ctx_request_id == "ctx-123"

    def test_generation_only_request(self):
        """Test generation-only disaggregated request."""
        proto_params = pb2.DisaggregatedParams(
            request_type=pb2.DisaggregatedParams.REQUEST_TYPE_GENERATION_ONLY,
            ctx_request_id="gen-456",
        )

        params = create_disaggregated_params_from_proto(proto_params)

        assert params is not None

    def test_none_params(self):
        """Test None disaggregated params returns None."""
        params = create_disaggregated_params_from_proto(None)
        assert params is None


class TestProtoMessages:
    """Tests for proto message structure."""

    def test_generate_request_structure(self):
        """Test GenerateRequest message structure."""
        request = pb2.GenerateRequest(
            request_id="test-123",
            tokenized=pb2.TokenizedInput(
                input_token_ids=[1, 2, 3, 4, 5],
                original_text="Hello world",
            ),
            sampling_config=pb2.SamplingConfig(temperature=0.8),
            max_tokens=50,
            streaming=True,
        )

        assert request.request_id == "test-123"
        assert list(request.tokenized.input_token_ids) == [1, 2, 3, 4, 5]
        assert request.tokenized.original_text == "Hello world"
        assert request.sampling_config.temperature == 0.8
        assert request.max_tokens == 50
        assert request.streaming is True

    def test_generate_response_chunk(self):
        """Test GenerateResponse with chunk."""
        response = pb2.GenerateResponse(
            request_id="test-123",
            chunk=pb2.GenerateStreamChunk(
                token_ids=[10, 11, 12],
                sequence_index=0,
                prompt_tokens=5,
                completion_tokens=3,
            ),
        )

        assert response.request_id == "test-123"
        assert list(response.chunk.token_ids) == [10, 11, 12]
        assert response.chunk.prompt_tokens == 5
        assert response.chunk.completion_tokens == 3

    def test_generate_response_complete(self):
        """Test GenerateResponse with complete."""
        response = pb2.GenerateResponse(
            request_id="test-123",
            complete=pb2.GenerateComplete(
                output_token_ids=[10, 11, 12, 13],
                finish_reason="stop",
                prompt_tokens=5,
                completion_tokens=4,
            ),
        )

        assert response.request_id == "test-123"
        assert list(response.complete.output_token_ids) == [10, 11, 12, 13]
        assert response.complete.finish_reason == "stop"

    def test_health_check_messages(self):
        """Test HealthCheck messages."""
        _request = pb2.HealthCheckRequest()  # noqa: F841 - verify message construction
        response = pb2.HealthCheckResponse(status="healthy")

        assert response.status == "healthy"

    def test_model_info_response(self):
        """Test GetModelInfoResponse message."""
        response = pb2.GetModelInfoResponse(
            model_id="meta-llama/Llama-2-7b",
            max_input_len=4096,
            max_seq_len=8192,
            vocab_size=32000,
        )

        assert response.model_id == "meta-llama/Llama-2-7b"
        assert response.max_input_len == 4096
        assert response.max_seq_len == 8192
        assert response.vocab_size == 32000

    def test_server_info_response(self):
        """Test GetServerInfoResponse message."""
        response = pb2.GetServerInfoResponse(
            version="0.17.0",
            backend="tensorrt-llm",
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            world_size=2,
        )

        assert response.version == "0.17.0"
        assert response.backend == "tensorrt-llm"
        assert response.tensor_parallel_size == 2
        assert response.world_size == 2

    def test_embed_messages(self):
        """Test Embed request and response messages."""
        request = pb2.EmbedRequest(
            request_id="embed-123",
            tokenized=pb2.TokenizedInput(input_token_ids=[1, 2, 3]),
        )
        response = pb2.EmbedResponse(
            request_id="embed-123",
            embedding=[0.1, 0.2, 0.3, 0.4],
            prompt_tokens=3,
        )

        assert request.request_id == "embed-123"
        assert response.request_id == "embed-123"
        assert list(response.embedding) == [0.1, 0.2, 0.3, 0.4]
        assert response.prompt_tokens == 3

    def test_abort_messages(self):
        """Test Abort request and response messages."""
        request = pb2.AbortRequest(request_id="abort-123")
        response = pb2.AbortResponse(success=True, message="Request aborted")

        assert request.request_id == "abort-123"
        assert response.success is True
        assert response.message == "Request aborted"
