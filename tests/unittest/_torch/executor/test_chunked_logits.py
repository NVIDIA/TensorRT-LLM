#!/usr/bin/env python3
"""
Unit tests for chunked logits functionality in TensorRT-LLM.

This module tests the chunked logits storage system that provides memory-efficient
logits handling through device-side fragments and batched host transfers.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest, LogitsStorage, PyResult, executor_request_to_llm_request)
from tensorrt_llm.bindings import executor as tllm_executor


# Test fixtures
@pytest.fixture
def sample_logits():
    """Generate sample logits for testing"""
    return torch.randn(1, 1, 1000, device='cuda')


@pytest.fixture
def sample_logits_2d():
    """Generate 2D logits that should be expanded to 3D"""
    return torch.randn(1, 1000, device='cuda')


@pytest.fixture
def sample_logits_multi_beam():
    """Generate logits with multiple beams"""
    return torch.randn(2, 2, 1000, device='cuda')


@pytest.fixture
def chunked_request():
    """Create LlmRequest with chunked logits enabled"""
    return LlmRequest(input_token_ids=[1, 2, 3],
                      max_new_tokens=10,
                      return_generation_logits=True,
                      use_chunked_logits=True,
                      logits_chunk_size=4)


@pytest.fixture
def non_chunked_request():
    """Create LlmRequest with chunked logits disabled"""
    return LlmRequest(input_token_ids=[1, 2, 3],
                      max_new_tokens=10,
                      return_generation_logits=True,
                      use_chunked_logits=False)


@pytest.fixture
def streaming_chunked_request():
    """Create LlmRequest with streaming chunked logits"""
    return LlmRequest(input_token_ids=[1, 2, 3],
                      max_new_tokens=10,
                      return_generation_logits=True,
                      use_chunked_logits=True,
                      logits_chunk_size=4,
                      streaming=True)


# Test parameters
CHUNK_SIZES = [1, 2, 4, 8, 16]
STREAMING_MODES = [True, False]
DEVICE_MEMORY_MODES = [True, False]
LOGITS_SHAPES = [(1, 1, 1000), (2, 1, 1000), (1, 2, 1000)]


class TestLogitsStorage:
    """Unit tests for LogitsStorage class"""

    @pytest.mark.unit
    def test_initialization(self):
        """Test LogitsStorage initialization with different parameters"""
        # Test basic initialization
        storage = LogitsStorage(seq_length=10,
                                use_device_memory=True,
                                should_exclude_last=False,
                                use_chunked_logits=False,
                                streaming=False,
                                chunk_size=8)

        assert storage.seq_length == 10
        assert storage.use_device_memory is True
        assert storage._should_exclude_last is False
        assert storage.use_chunked_logits is False
        assert storage.streaming is False
        assert storage.chunk_size == 8
        assert storage._logits_indices == []
        assert storage.beam_width == -1
        assert storage.vocab_size == -1

    @pytest.mark.unit
    def test_initialization_chunked_mode(self):
        """Test LogitsStorage initialization in chunked mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_logits=True,
                                streaming=True,
                                chunk_size=4)

        assert storage.use_chunked_logits is True
        assert storage.streaming is True
        assert storage.chunk_size == 4
        assert hasattr(storage, '_device_fragments')
        assert hasattr(storage, '_current_position')
        assert storage._device_fragments == []
        assert storage._current_position == 0

    @pytest.mark.unit
    def test_initialization_with_exclude_last(self):
        """Test LogitsStorage initialization with should_exclude_last=True"""
        storage = LogitsStorage(seq_length=10, should_exclude_last=True)

        assert storage.seq_length == 11  # Should be incremented by 1
        assert storage._should_exclude_last is True

    @pytest.mark.unit
    def test_append_2d_logits(self, sample_logits_2d):
        """Test appending 2D logits (should be expanded to 3D)"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)
        storage.append(sample_logits_2d)

        # Should be expanded to 3D
        assert storage.beam_width == 1
        assert storage.vocab_size == 1000

    @pytest.mark.unit
    def test_append_3d_logits(self, sample_logits):
        """Test appending 3D logits"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)
        storage.append(sample_logits)

        assert storage.beam_width == 1
        assert storage.vocab_size == 1000

    @pytest.mark.unit
    def test_append_invalid_shape(self):
        """Test appending logits with invalid shape"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)

        with pytest.raises(AssertionError):
            storage.append(torch.randn(1000))  # 1D - should fail

    @pytest.mark.unit
    def test_append_non_chunked_mode(self, sample_logits):
        """Test append behavior in non-chunked mode"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)
        storage.append(sample_logits)

        # Should have storage allocated
        assert storage._storage is not None
        assert len(storage._logits_indices) == 1
        assert storage._logits_indices[0] == (0, 1)

    @pytest.mark.unit
    def test_append_chunked_mode(self, sample_logits):
        """Test append behavior in chunked mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_logits=True,
                                chunk_size=2)
        storage.append(sample_logits)

        # Should have fragment added
        assert len(storage._device_fragments) == 1
        assert storage._current_position == 0

    @pytest.mark.unit
    def test_append_chunked_mode_streaming(self, sample_logits):
        """Test append behavior in chunked streaming mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_logits=True,
                                streaming=True)
        storage.append(sample_logits)

        # Should transfer immediately in streaming mode
        assert len(storage._device_fragments) == 0
        assert storage._current_position == 1

    @pytest.mark.unit
    def test_append_chunked_mode_non_streaming(self, sample_logits):
        """Test append behavior in chunked non-streaming mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_logits=True,
                                chunk_size=2,
                                streaming=False)

        # Add first fragment
        storage.append(sample_logits)
        assert len(storage._device_fragments) == 1

        # Add second fragment - should trigger transfer
        storage.append(sample_logits)
        assert len(storage._device_fragments) == 0
        assert storage._current_position == 2

    @pytest.mark.unit
    def test_get_all_logits(self, sample_logits):
        """Test get method with all_logits=True"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)
        storage.append(sample_logits)

        result = storage.get(all_logits=True)
        assert result is not None
        assert result.shape == (1, 1, 1000)

    @pytest.mark.unit
    def test_get_last_logits(self, sample_logits):
        """Test get method with all_logits=False"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)
        storage.append(sample_logits)

        result = storage.get(all_logits=False)
        assert result is not None
        assert result.shape == (1, 1, 1000)

    @pytest.mark.unit
    def test_get_no_storage(self):
        """Test get method when no storage is allocated"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)

        result = storage.get(all_logits=True)
        assert result is None

    @pytest.mark.unit
    def test_finalize_transfer_chunked_mode(self, sample_logits):
        """Test finalize_transfer in chunked mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_logits=True,
                                chunk_size=5)
        storage.append(sample_logits)

        # Should have fragment pending
        assert len(storage._device_fragments) == 1

        storage.finalize_transfer()

        # Should transfer remaining fragments
        assert len(storage._device_fragments) == 0

    @pytest.mark.unit
    def test_finalize_transfer_non_chunked_mode(self):
        """Test finalize_transfer in non-chunked mode (should be no-op)"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)

        # Should not raise any errors
        storage.finalize_transfer()

    @pytest.mark.unit
    def test_set_exclude_last(self):
        """Test set_exclude_last method"""
        storage = LogitsStorage(seq_length=10)
        storage.set_exclude_last(True)
        assert storage._should_exclude_last is True

        storage.set_exclude_last(False)
        assert storage._should_exclude_last is False

    @pytest.mark.unit
    def test_storage_overflow(self, sample_logits):
        """Test storage overflow handling"""
        storage = LogitsStorage(seq_length=2, use_chunked_logits=False)
        storage.append(sample_logits)
        storage.append(sample_logits)

        # This should cause overflow
        with pytest.raises(ValueError, match="LogitsStorage overflow"):
            storage.append(sample_logits)

    @pytest.mark.unit
    def test_beam_width_mismatch(self, sample_logits, sample_logits_multi_beam):
        """Test beam width mismatch handling"""
        storage = LogitsStorage(seq_length=10, use_chunked_logits=False)
        storage.append(sample_logits)

        # This should cause beam width mismatch
        with pytest.raises(AssertionError, match="Beam width mismatch"):
            storage.append(sample_logits_multi_beam)


class TestPyResult:
    """Unit tests for PyResult class"""

    @pytest.mark.unit
    def test_initialization(self):
        """Test PyResult initialization"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          use_device_memory=True,
                          streaming=False,
                          return_log_probs=True,
                          return_context_logits=True,
                          return_generation_logits=True,
                          exclude_last_generation_logits=False,
                          use_chunked_logits=True,
                          chunk_size=4)

        assert result._streaming is False
        assert result._context_logits is not None
        assert result._generation_logits is not None
        assert result._log_probs is not None
        assert result._mm_embeddings is None

    @pytest.mark.unit
    def test_initialization_no_logits(self):
        """Test PyResult initialization without logits"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_log_probs=False,
                          return_context_logits=False,
                          return_generation_logits=False)

        assert result._context_logits is None
        assert result._generation_logits is None
        assert result._log_probs is None

    @pytest.mark.unit
    def test_append_context_logits(self, sample_logits):
        """Test append_context_logits method"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_context_logits=True,
                          use_chunked_logits=False)

        result.append_context_logits(sample_logits)

        # Should have logits stored
        assert result._context_logits._storage is not None

    @pytest.mark.unit
    def test_append_generation_logits(self, sample_logits):
        """Test append_generation_logits method"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_generation_logits=True,
                          use_chunked_logits=False)

        result.append_generation_logits(sample_logits)

        # Should have logits stored
        assert result._generation_logits._storage is not None

    @pytest.mark.unit
    def test_append_logits_no_storage(self, sample_logits):
        """Test append methods when storage is not enabled"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_context_logits=False,
                          return_generation_logits=False)

        # Should not raise errors
        result.append_context_logits(sample_logits)
        result.append_generation_logits(sample_logits)

    @pytest.mark.unit
    def test_post_processing_transfer(self, sample_logits):
        """Test post_processing_transfer method"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_generation_logits=True,
                          use_chunked_logits=True)

        result.append_generation_logits(sample_logits)
        result.post_processing_transfer()

        # Should not raise errors

    @pytest.mark.unit
    def test_context_logits_property(self, sample_logits):
        """Test context_logits property"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_context_logits=True,
                          use_chunked_logits=False)

        result.append_context_logits(sample_logits)
        context_logits = result.context_logits

        assert context_logits is not None
        assert context_logits.shape == (1, 1000)  # Should remove beam dimension

    @pytest.mark.unit
    def test_generation_logits_property(self, sample_logits):
        """Test generation_logits property"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_generation_logits=True,
                          use_chunked_logits=False)

        result.append_generation_logits(sample_logits)
        generation_logits = result.generation_logits

        assert generation_logits is not None
        assert generation_logits.shape == (1, 1, 1000
                                           )  # Should transpose dimensions

    @pytest.mark.unit
    def test_generation_logits_property_streaming(self, sample_logits):
        """Test generation_logits property in streaming mode"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_generation_logits=True,
                          use_chunked_logits=False,
                          streaming=True)

        result.append_generation_logits(sample_logits)
        generation_logits = result.generation_logits

        assert generation_logits is not None
        assert generation_logits.shape == (1, 1, 1000)

    @pytest.mark.unit
    def test_log_probs_property(self):
        """Test log_probs property"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_log_probs=True)

        log_probs = result.log_probs
        assert log_probs is not None

    @pytest.mark.unit
    def test_cum_log_probs_property(self):
        """Test cum_log_probs property"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_log_probs=True)

        cum_log_probs = result.cum_log_probs
        assert cum_log_probs is not None

    @pytest.mark.unit
    def test_mm_embedding_handle_property(self):
        """Test mm_embedding_handle property"""
        result = PyResult(prompt_len=5, max_new_tokens=10)

        handle = result.mm_embedding_handle
        assert handle is None


class TestLlmRequest:
    """Unit tests for LlmRequest class"""

    @pytest.mark.unit
    def test_initialization_chunked_logits(self):
        """Test LlmRequest initialization with chunked logits"""
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=10,
                             return_generation_logits=True,
                             use_chunked_logits=True,
                             logits_chunk_size=4)

        assert request.py_use_chunked_logits is True
        assert request.py_logits_chunk_size == 4
        assert request.py_result is not None

    @pytest.mark.unit
    def test_initialization_default_chunked_logits(self):
        """Test LlmRequest initialization with default chunked logits"""
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=10,
                             return_generation_logits=True)

        # Should use default values
        assert request.py_use_chunked_logits is True  # Default is True
        assert request.py_logits_chunk_size == 8  # Default is 8

    @pytest.mark.unit
    def test_py_result_creation(self):
        """Test PyResult creation with correct parameters"""
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=10,
                             return_generation_logits=True,
                             use_chunked_logits=True,
                             logits_chunk_size=4)

        assert request.py_result is not None
        assert request.py_result._generation_logits is not None
        assert request.py_result._generation_logits.use_chunked_logits is True
        assert request.py_result._generation_logits.chunk_size == 4

    @pytest.mark.unit
    def test_create_child_request(self, chunked_request):
        """Test child request creation with parameter inheritance"""
        child = chunked_request.create_child_request(999)

        assert child.py_use_chunked_logits == chunked_request.py_use_chunked_logits
        assert child.py_logits_chunk_size == chunked_request.py_logits_chunk_size
        assert child.is_child is True
        assert child.request_id == 999

    @pytest.mark.unit
    def test_create_child_request_py_result(self, chunked_request):
        """Test child request PyResult recreation"""
        child = chunked_request.create_child_request(999)

        assert child.py_result is not None
        assert child.py_result._generation_logits is not None
        assert child.py_result._generation_logits.use_chunked_logits is True
        assert child.py_result._generation_logits.chunk_size == 4

    @pytest.mark.unit
    def test_is_generation_only_request(self):
        """Test is_generation_only_request method"""
        request = LlmRequest(input_token_ids=[1, 2, 3], max_new_tokens=10)

        # Should return False by default (context and generation)
        assert request.is_generation_only_request() is False

    @pytest.mark.unit
    def test_create_response(self, chunked_request):
        """Test create_response method"""
        # Mock the parent method
        with patch.object(chunked_request.__class__.__bases__[0],
                          'create_serialized_result') as mock_create:
            mock_create.return_value = (b'serialized_result', True)

            response = chunked_request.create_response()

            assert response is not None
            assert response.request_id == chunked_request.py_request_id
            assert response.client_id == chunked_request.py_client_id

    @pytest.mark.unit
    def test_finish_by(self, chunked_request):
        """Test finish_by method"""
        from tensorrt_llm.bindings.executor import FinishReason

        chunked_request.finish_by(FinishReason.LENGTH, 0)

        assert chunked_request.state.value == 5  # GENERATION_COMPLETE

    @pytest.mark.unit
    def test_is_dummy_property(self, chunked_request):
        """Test is_dummy property"""
        assert chunked_request.is_dummy is False

        chunked_request.is_attention_dp_dummy = True
        assert chunked_request.is_dummy is True


class TestExecutorRequestConversion:
    """Unit tests for executor_request_to_llm_request function"""

    @pytest.mark.unit
    def test_executor_request_conversion(self):
        """Test conversion from executor request to LlmRequest"""
        # Create mock executor request
        executor_request = MagicMock()
        executor_request.sampling_config = MagicMock()
        executor_request.sampling_config.beam_width = 1
        executor_request.max_tokens = 10
        executor_request.input_token_ids = [1, 2, 3]
        executor_request.streaming = False
        executor_request.end_id = None
        executor_request.pad_id = None
        executor_request.embedding_bias = None
        executor_request.bad_words = None
        executor_request.stop_words = None
        executor_request.prompt_tuning_config = None
        executor_request.multimodal_input = None
        executor_request.mrope_config = None
        executor_request.multimodal_embedding = None
        executor_request.lora_config = None
        executor_request.guided_decoding_params = None
        executor_request.context_phase_params = None
        executor_request.cache_salt_id = None
        executor_request.client_id = None
        executor_request.request_type = tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION

        # Mock output config
        executor_request.output_config = MagicMock()
        executor_request.output_config.return_log_probs = False
        executor_request.output_config.return_context_logits = False
        executor_request.output_config.return_perf_metrics = False
        executor_request.output_config.return_generation_logits = False
        executor_request.output_config.exclude_input_from_output = False

        # Test conversion
        llm_request = executor_request_to_llm_request(
            req_id=1,
            executor_request=executor_request,
            child_req_ids=[],
            exclude_last_generation_logits=False)

        assert llm_request is not None
        assert llm_request.py_request_id == 1
        assert llm_request.py_use_chunked_logits is False  # Default from executor
        assert llm_request.py_logits_chunk_size == 8  # Default from executor

    @pytest.mark.unit
    def test_executor_request_conversion_with_chunked_params(self):
        """Test conversion with chunked logits parameters"""
        # Create mock executor request with chunked params
        executor_request = MagicMock()
        executor_request.sampling_config = MagicMock()
        executor_request.sampling_config.beam_width = 1
        executor_request.max_tokens = 10
        executor_request.input_token_ids = [1, 2, 3]
        executor_request.streaming = True
        executor_request.end_id = None
        executor_request.pad_id = None
        executor_request.embedding_bias = None
        executor_request.bad_words = None
        executor_request.stop_words = None
        executor_request.prompt_tuning_config = None
        executor_request.multimodal_input = None
        executor_request.mrope_config = None
        executor_request.multimodal_embedding = None
        executor_request.lora_config = None
        executor_request.guided_decoding_params = None
        executor_request.context_phase_params = None
        executor_request.cache_salt_id = None
        executor_request.client_id = None
        executor_request.request_type = tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION
        executor_request.use_chunked_logits = True
        executor_request.logits_chunk_size = 4

        # Mock output config
        executor_request.output_config = MagicMock()
        executor_request.output_config.return_log_probs = False
        executor_request.output_config.return_context_logits = False
        executor_request.output_config.return_perf_metrics = False
        executor_request.output_config.return_generation_logits = True
        executor_request.output_config.exclude_input_from_output = False

        # Test conversion
        llm_request = executor_request_to_llm_request(
            req_id=1,
            executor_request=executor_request,
            child_req_ids=[],
            exclude_last_generation_logits=False)

        assert llm_request is not None
        assert llm_request.py_use_chunked_logits is True
        assert llm_request.py_logits_chunk_size == 4


class TestChunkedLogitsIntegration:
    """Integration tests for chunked logits functionality"""

    @pytest.mark.integration
    def test_chunked_vs_non_chunked_equivalence(self, sample_logits):
        """Test that chunked and non-chunked modes produce equivalent results"""
        # Create chunked request
        chunked_request = LlmRequest(input_token_ids=[1, 2, 3],
                                     max_new_tokens=5,
                                     return_generation_logits=True,
                                     use_chunked_logits=True,
                                     logits_chunk_size=2)

        # Create non-chunked request
        non_chunked_request = LlmRequest(input_token_ids=[1, 2, 3],
                                         max_new_tokens=5,
                                         return_generation_logits=True,
                                         use_chunked_logits=False)

        # Add same logits to both
        for _ in range(5):
            chunked_request.py_result.append_generation_logits(sample_logits)
            non_chunked_request.py_result.append_generation_logits(
                sample_logits)

        # Finalize chunked request
        chunked_request.py_result.post_processing_transfer()

        # Get results
        chunked_logits = chunked_request.py_result.generation_logits
        non_chunked_logits = non_chunked_request.py_result.generation_logits

        # Should be equivalent
        assert chunked_logits is not None
        assert non_chunked_logits is not None
        assert chunked_logits.shape == non_chunked_logits.shape
        assert torch.allclose(chunked_logits, non_chunked_logits, atol=1e-6)

    @pytest.mark.integration
    def test_streaming_vs_non_streaming_behavior(self, sample_logits):
        """Test different behavior between streaming and non-streaming modes"""
        # Create streaming request
        streaming_request = LlmRequest(input_token_ids=[1, 2, 3],
                                       max_new_tokens=5,
                                       return_generation_logits=True,
                                       use_chunked_logits=True,
                                       logits_chunk_size=3,
                                       streaming=True)

        # Create non-streaming request
        non_streaming_request = LlmRequest(input_token_ids=[1, 2, 3],
                                           max_new_tokens=5,
                                           return_generation_logits=True,
                                           use_chunked_logits=True,
                                           logits_chunk_size=3,
                                           streaming=False)

        # Add logits one by one
        for i in range(5):
            streaming_request.py_result.append_generation_logits(sample_logits)
            non_streaming_request.py_result.append_generation_logits(
                sample_logits)

            # Check fragment counts
            streaming_fragments = len(streaming_request.py_result.
                                      _generation_logits._device_fragments)
            non_streaming_fragments = len(non_streaming_request.py_result.
                                          _generation_logits._device_fragments)

            if i < 4:  # Before final transfer
                # Streaming should have 0 fragments (immediate transfer)
                # Non-streaming should accumulate fragments
                assert streaming_fragments == 0
                assert non_streaming_fragments == (i +
                                                   1) % 3  # Modulo chunk_size

        # Finalize both
        streaming_request.py_result.post_processing_transfer()
        non_streaming_request.py_result.post_processing_transfer()

        # Both should have same final result
        streaming_logits = streaming_request.py_result.generation_logits
        non_streaming_logits = non_streaming_request.py_result.generation_logits

        assert streaming_logits is not None
        assert non_streaming_logits is not None
        assert streaming_logits.shape == non_streaming_logits.shape
        assert torch.allclose(streaming_logits, non_streaming_logits, atol=1e-6)

    @pytest.mark.integration
    def test_memory_management(self, sample_logits):
        """Test memory management in chunked mode"""
        request = LlmRequest(
            input_token_ids=[1, 2, 3],
            max_new_tokens=10,
            return_generation_logits=True,
            use_chunked_logits=True,
            logits_chunk_size=2,
            return_logits_device_memory=False  # Use host memory
        )

        # Add logits
        for _ in range(5):
            request.py_result.append_generation_logits(sample_logits)

        # Check that storage is on CPU (host memory)
        assert request.py_result._generation_logits._storage is not None
        assert request.py_result._generation_logits._storage.device.type == 'cpu'

        # Finalize
        request.py_result.post_processing_transfer()

        # Should have no pending fragments
        assert len(request.py_result._generation_logits._device_fragments) == 0

    @pytest.mark.integration
    def test_large_sequence_handling(self):
        """Test handling of large sequences"""
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=100,
                             return_generation_logits=True,
                             use_chunked_logits=True,
                             logits_chunk_size=10)

        # Add many logits
        for i in range(50):
            logits = torch.randn(1, 1, 1000, device='cuda')
            request.py_result.append_generation_logits(logits)

            # Check fragment management
            fragments = len(
                request.py_result._generation_logits._device_fragments)
            expected_fragments = (i + 1) % 10  # Modulo chunk_size
            assert fragments == expected_fragments

        # Finalize
        request.py_result.post_processing_transfer()

        # Should have final result
        final_logits = request.py_result.generation_logits
        assert final_logits is not None
        assert final_logits.shape == (1, 50, 1000)

    @pytest.mark.integration
    def test_child_request_inheritance(self, sample_logits):
        """Test that child requests properly inherit chunked logits settings"""
        parent = LlmRequest(input_token_ids=[1, 2, 3],
                            max_new_tokens=10,
                            return_generation_logits=True,
                            use_chunked_logits=True,
                            logits_chunk_size=4)

        child = parent.create_child_request(999)

        # Add logits to child
        for _ in range(3):
            child.py_result.append_generation_logits(sample_logits)

        # Should work with inherited settings
        child.py_result.post_processing_transfer()
        child_logits = child.py_result.generation_logits

        assert child_logits is not None
        assert child_logits.shape == (1, 3, 1000)


class TestChunkedLogitsPerformance:
    """Performance tests for chunked logits functionality"""

    @pytest.mark.performance
    def test_chunk_size_performance(self, sample_logits):
        """Test performance with different chunk sizes"""
        chunk_sizes = [1, 2, 4, 8, 16]
        results = {}

        for chunk_size in chunk_sizes:
            request = LlmRequest(input_token_ids=[1, 2, 3],
                                 max_new_tokens=20,
                                 return_generation_logits=True,
                                 use_chunked_logits=True,
                                 logits_chunk_size=chunk_size)

            # Time the append operations
            import time
            start_time = time.time()

            for _ in range(20):
                request.py_result.append_generation_logits(sample_logits)

            request.py_result.post_processing_transfer()

            end_time = time.time()
            results[chunk_size] = end_time - start_time

        # All should complete successfully
        for chunk_size, duration in results.items():
            assert duration > 0
            print(f"Chunk size {chunk_size}: {duration:.4f}s")

    @pytest.mark.performance
    def test_memory_usage_comparison(self, sample_logits):
        """Test memory usage comparison between chunked and non-chunked modes"""
        import os

        import psutil

        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        # Test chunked mode
        initial_memory = get_memory_usage()

        chunked_request = LlmRequest(input_token_ids=[1, 2, 3],
                                     max_new_tokens=50,
                                     return_generation_logits=True,
                                     use_chunked_logits=True,
                                     logits_chunk_size=5,
                                     return_logits_device_memory=False)

        for _ in range(50):
            chunked_request.py_result.append_generation_logits(sample_logits)

        chunked_request.py_result.post_processing_transfer()
        chunked_memory = get_memory_usage()

        # Test non-chunked mode
        non_chunked_request = LlmRequest(input_token_ids=[1, 2, 3],
                                         max_new_tokens=50,
                                         return_generation_logits=True,
                                         use_chunked_logits=False,
                                         return_logits_device_memory=False)

        for _ in range(50):
            non_chunked_request.py_result.append_generation_logits(
                sample_logits)

        non_chunked_memory = get_memory_usage()

        # Both should use reasonable memory
        chunked_delta = chunked_memory - initial_memory
        non_chunked_delta = non_chunked_memory - chunked_memory

        print(f"Chunked mode memory delta: {chunked_delta:.2f} MB")
        print(f"Non-chunked mode memory delta: {non_chunked_delta:.2f} MB")

        # Both should be reasonable (less than 100MB for this test)
        assert chunked_delta < 100
        assert non_chunked_delta < 100


class TestChunkedLogitsCompatibility:
    """Compatibility tests for chunked logits functionality"""

    @pytest.mark.compatibility
    def test_backward_compatibility(self):
        """Test that existing code continues to work with default parameters"""
        # Test with minimal parameters (should use defaults)
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=10,
                             return_generation_logits=True)

        # Should work with default chunked logits settings
        assert request.py_use_chunked_logits is True  # Default
        assert request.py_logits_chunk_size == 8  # Default
        assert request.py_result is not None

    @pytest.mark.compatibility
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=10,
                             return_generation_logits=True,
                             use_chunked_logits=True,
                             logits_chunk_size=4)

        assert request.py_use_chunked_logits is True
        assert request.py_logits_chunk_size == 4

    @pytest.mark.compatibility
    def test_api_stability(self):
        """Test that public APIs remain stable"""
        request = LlmRequest(input_token_ids=[1, 2, 3],
                             max_new_tokens=10,
                             return_generation_logits=True)

        # Test that existing methods still work
        assert hasattr(request, 'py_result')
        assert hasattr(request, 'create_response')
        assert hasattr(request, 'create_child_request')
        assert hasattr(request, 'is_generation_only_request')

        # Test that new attributes are available
        assert hasattr(request, 'py_use_chunked_logits')
        assert hasattr(request, 'py_logits_chunk_size')


if __name__ == "__main__":
    pytest.main([__file__])
