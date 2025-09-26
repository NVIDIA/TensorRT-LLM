"""
Unit tests for chunked logits functionality in TensorRT-LLM.

This module tests the chunked logits storage system that provides memory-efficient
logits handling through device-side fragments and batched host transfers.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LogitsStorage, PyResult)
from tensorrt_llm.bindings import SamplingConfig


# Test fixtures
@pytest.fixture
def sample_logits():
    """Generate sample logits for testing"""
    return torch.randn(1, 1, 1000, device='cuda')


class TestLogitsStorage:
    """Unit tests for LogitsStorage class"""

    def test_initialization_chunked_mode(self):
        """Test LogitsStorage initialization in chunked mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_generation_logits=True,
                                chunk_size=4)

        assert storage.use_chunked_generation_logits is True
        assert storage.chunk_size == 4
        assert hasattr(storage, '_device_fragments')
        assert hasattr(storage, '_current_position')
        assert storage._device_fragments == []
        assert storage._current_position == 0

    def test_append_chunked_mode_streaming(self, sample_logits):
        """Test append behavior in chunked streaming mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_generation_logits=True,
                                chunk_size=1)
        storage.append(sample_logits)

        # Should transfer immediately in streaming mode
        assert len(storage._device_fragments) == 0
        assert storage._current_position == 1

    def test_append_chunked_mode_non_streaming(self, sample_logits):
        """Test append behavior in chunked non-streaming mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_generation_logits=True,
                                chunk_size=2)

        # Add first fragment
        storage.append(sample_logits)
        assert len(storage._device_fragments) == 1

        # Add second fragment - should trigger transfer
        storage.append(sample_logits)
        assert len(storage._device_fragments) == 0
        assert storage._current_position == 2

    def test_finalize_chunked_transfer_chunked_mode(self, sample_logits):
        """Test finalize_chunked_transfer in chunked mode"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_generation_logits=True,
                                chunk_size=5)
        storage.append(sample_logits)

        # Should have fragment pending
        assert len(storage._device_fragments) == 1

        storage.finalize_chunked_transfer()

        # Should transfer remaining fragments
        assert len(storage._device_fragments) == 0

    def test_finalize_chunked_transfer_non_chunked_mode(self):
        """Test finalize_chunked_transfer in non-chunked mode (should be no-op)"""
        storage = LogitsStorage(seq_length=10,
                                use_chunked_generation_logits=False)

        # Should not raise any errors
        storage.finalize_chunked_transfer()


class TestPyResult:
    """Unit tests for PyResult class"""

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
                          use_chunked_generation_logits=True,
                          chunk_size=4)

        assert result._streaming is False
        assert result._context_logits is not None
        assert result._generation_logits is not None
        assert result._log_probs is not None
        assert result._mm_embeddings is None

    def test_append_logits_no_storage(self, sample_logits):
        """Test append methods when storage is not enabled"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_context_logits=False,
                          return_generation_logits=False)

        # Should not raise errors
        result.append_context_logits(sample_logits)
        result.append_generation_logits(sample_logits)

    def test_transfer_remaining_device_logits(self, sample_logits):
        """Test transfer_remaining_device_logits method"""
        result = PyResult(prompt_len=5,
                          max_new_tokens=10,
                          return_generation_logits=True,
                          use_chunked_generation_logits=True)

        result.append_generation_logits(sample_logits)
        result.transfer_remaining_device_logits()

        # Should not raise errors


class TestLlmRequest:
    """Unit tests for LlmRequest class"""

    def test_initialization_chunked_logits(self):
        """Test LlmRequest initialization with default chunked logits"""
        request_non_streaming = LlmRequest(request_id=2,
                                           max_new_tokens=10,
                                           input_tokens=[1, 2, 3],
                                           sampling_config=SamplingConfig(),
                                           is_streaming=False,
                                           return_generation_logits=True)

        # Should use default values
        assert request_non_streaming.py_use_chunked_generation_logits is True  # Default is True
        assert request_non_streaming.py_logits_chunk_size == 8  # Default is 8

        request_streaming = LlmRequest(request_id=3,
                                       max_new_tokens=10,
                                       input_tokens=[1, 2, 3],
                                       sampling_config=SamplingConfig(),
                                       is_streaming=True,
                                       return_generation_logits=True)

        assert request_streaming.py_use_chunked_generation_logits is True
        assert request_streaming.py_logits_chunk_size == 1  # 1 in streaming mode


class TestChunkedLogitsComplicated:
    """Integration tests for chunked logits functionality"""

    def test_chunked_vs_non_chunked_equivalence(self, sample_logits):
        """Test that chunked and non-chunked modes produce equivalent results"""
        # Create chunked request
        chunked_request = LlmRequest(request_id=5,
                                     max_new_tokens=5,
                                     input_tokens=[1, 2, 3],
                                     sampling_config=SamplingConfig(),
                                     is_streaming=False,
                                     return_generation_logits=True,
                                     use_chunked_generation_logits=True,
                                     logits_chunk_size=2)

        # Create non-chunked request
        non_chunked_request = LlmRequest(request_id=6,
                                         max_new_tokens=5,
                                         input_tokens=[1, 2, 3],
                                         sampling_config=SamplingConfig(),
                                         is_streaming=False,
                                         return_generation_logits=True,
                                         use_chunked_generation_logits=False)

        # Add same logits to both
        for _ in range(5):
            chunked_request.py_result.append_generation_logits(sample_logits)
            non_chunked_request.py_result.append_generation_logits(
                sample_logits)

        # Finalize chunked request
        chunked_request.py_result.transfer_remaining_device_logits()

        # Get results
        chunked_logits = chunked_request.py_result.generation_logits
        non_chunked_logits = non_chunked_request.py_result.generation_logits.cpu(
        )

        # Should be equivalent
        assert chunked_logits is not None
        assert non_chunked_logits is not None
        assert chunked_logits.shape == non_chunked_logits.shape
        assert torch.allclose(chunked_logits, non_chunked_logits, atol=1e-6)

    def test_streaming_vs_non_streaming_behavior(self, sample_logits):
        """Test different behavior between streaming and non-streaming modes"""
        # Create streaming request
        streaming_request = LlmRequest(request_id=7,
                                       max_new_tokens=5,
                                       input_tokens=[1, 2, 3],
                                       sampling_config=SamplingConfig(),
                                       is_streaming=True,
                                       return_generation_logits=True,
                                       use_chunked_generation_logits=True,
                                       logits_chunk_size=3)

        # Create non-streaming request
        non_streaming_request = LlmRequest(request_id=8,
                                           max_new_tokens=5,
                                           input_tokens=[1, 2, 3],
                                           sampling_config=SamplingConfig(),
                                           is_streaming=False,
                                           return_generation_logits=True,
                                           use_chunked_generation_logits=True,
                                           logits_chunk_size=3)

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
        streaming_request.py_result.transfer_remaining_device_logits()
        non_streaming_request.py_result.transfer_remaining_device_logits()

        # Both should have same final result
        streaming_logits = streaming_request.py_result.generation_logits
        non_streaming_logits = non_streaming_request.py_result.generation_logits

        assert streaming_logits is not None
        assert non_streaming_logits is not None
        # In non-streaming mode, logits are retrieved as a single tensor, where in streaming mode, logits is the last token.
        assert torch.allclose(streaming_logits,
                              non_streaming_logits[-1],
                              atol=1e-6)

    def test_memory_management(self, sample_logits):
        """Test memory management in chunked mode"""
        request = LlmRequest(
            request_id=9,
            max_new_tokens=10,
            input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(),
            is_streaming=False,
            return_generation_logits=True,
            use_chunked_generation_logits=True,
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
        request.py_result.transfer_remaining_device_logits()

        # Should have no pending fragments
        assert len(request.py_result._generation_logits._device_fragments) == 0

    def test_large_sequence_handling(self):
        """Test handling of large sequences"""
        request = LlmRequest(request_id=10,
                             max_new_tokens=100,
                             input_tokens=[1, 2, 3],
                             sampling_config=SamplingConfig(),
                             is_streaming=False,
                             return_generation_logits=True,
                             use_chunked_generation_logits=True,
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
        request.py_result.transfer_remaining_device_logits()

        # Should have final result
        final_logits = request.py_result.generation_logits
        assert final_logits is not None
        assert final_logits.shape == (1, 50, 1000)


if __name__ == "__main__":
    pytest.main([__file__])
