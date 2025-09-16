from typing import List
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds, get_multimodal_embeddings)
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData)


class TestMultimodalRuntimeData:
    """Test cases for MultimodalRuntimeData computation logic, testing both KV cache reuse and chunked prefill."""

    def test_fully_cached_multimodal_tokens(self):
        """Test when all multimodal tokens are cached (KV cache reuse scenario)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=20,
            mm_token_lengths=[5, 8, 7],  # Total: 20 tokens
            mm_token_positions=[0, 5, 13],  # Positions: 0-5, 5-13, 13-20
            chunk_end_pos=20,
            special_token_offsets=[])

        # All tokens should be cached since past_seen_token_num (20) >= all positions + lengths
        assert runtime.num_unseen_mm_tokens == 20
        assert runtime.num_mm_tokens_in_chunk == 0

    def test_no_cached_multimodal_tokens(self):
        """Test when no multimodal tokens are cached (KV cache reuse scenario)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=10,
            mm_token_lengths=[5, 8, 7],  # Total: 20 tokens
            mm_token_positions=[10, 18,
                                30],  # All positions > past_seen_token_num
            chunk_end_pos=40,
            special_token_offsets=[])

        # No multimodal tokens should be cached
        assert runtime.num_unseen_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 20

    def test_partial_caching_with_chunk_boundaries(self):
        """Test partial caching with chunk boundaries (chunked prefill scenario)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=15,
            mm_token_lengths=[5, 8, 7],  # Total: 20 tokens
            mm_token_positions=[10, 18, 25],  # Positions: 10-15, 18-26, 25-32
            chunk_end_pos=30,
            special_token_offsets=[])

        # Expected caching:
        # Chunk 0: [10-15] - 5 tokens fully cached, 0 tokens in current chunk
        # Chunk 1: [18-26] - 0 tokens cached, 8 tokens in current chunk (18-26)
        # Chunk 2: [25-32] - 0 tokens cached, 5 tokens in current chunk (25-30), 2 tokens beyond chunk
        assert runtime.num_unseen_mm_tokens == 5  # 5 tokens from chunk 0
        assert runtime.num_mm_tokens_in_chunk == 13  # 8 + 5 tokens in current chunk

    def test_chunk_boundary_case1(self):
        """Test case chunk around chunk boundaries."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=12,
            mm_token_lengths=[6, 4, 8],  # Total: 18 tokens
            mm_token_positions=[8, 16, 22],  # Positions: 8-14, 16-20, 22-30
            chunk_end_pos=20,
            special_token_offsets=[])

        # Expected caching:
        # Chunk 0: [8-14] - 4 tokens cached (8-12), 2 tokens in current chunk (12-14)
        # Chunk 1: [16-20] - 0 tokens cached, 4 tokens in current chunk (16-20)
        # Chunk 2: [22-30] - 0 tokens cached, 0 tokens in current chunk (beyond chunk_end_pos)
        assert runtime.num_unseen_mm_tokens == 4  # 4 tokens from chunk 0
        assert runtime.num_mm_tokens_in_chunk == 6  # 2 + 4 tokens in current chunk

    def test_chunk_boundary_case2(self):
        """Test test chunk end is very large."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=30,
            mm_token_lengths=[3, 4, 5, 6, 7, 8],  # Total: 33 tokens
            mm_token_positions=[
                0, 5, 10, 15, 25, 35
            ],  # Positions: 0-3, 5-9, 10-15, 15-21, 25-32, 35-43
            chunk_end_pos=100,
            special_token_offsets=[])

        expected_cached = 3 + 4 + 5 + 6 + 5  # 23 tokens
        expected_current_chunk = 2 + 8  # 10 tokens
        assert runtime.num_unseen_mm_tokens == expected_cached
        assert runtime.num_mm_tokens_in_chunk == expected_current_chunk

    def test_validation_errors(self):
        """Test validation logic for invalid inputs."""
        # Test mismatched lengths
        with pytest.raises(
                ValueError,
                match=
                "mm_token_positions \\(2\\) and mm_token_lengths \\(3\\) must have the same length"
        ):
            MultimodalRuntimeData(past_seen_token_num=10,
                                  mm_token_lengths=[5, 8, 7],
                                  mm_token_positions=[0, 5],
                                  chunk_end_pos=20,
                                  special_token_offsets=[])

        # Test negative past_seen_token_num
        with pytest.raises(ValueError,
                           match="past_seen_token_num must be non-negative"):
            MultimodalRuntimeData(past_seen_token_num=-1,
                                  mm_token_lengths=[5],
                                  mm_token_positions=[0],
                                  chunk_end_pos=10,
                                  special_token_offsets=[])

        # Test non-positive token lengths
        with pytest.raises(ValueError,
                           match="All mm_token_lengths must be positive"):
            MultimodalRuntimeData(past_seen_token_num=10,
                                  mm_token_lengths=[5, 0, 7],
                                  mm_token_positions=[0, 5, 10],
                                  chunk_end_pos=20,
                                  special_token_offsets=[])

        # Test negative positions
        with pytest.raises(ValueError,
                           match="All mm_token_positions must be non-negative"):
            MultimodalRuntimeData(past_seen_token_num=10,
                                  mm_token_lengths=[5, 8, 7],
                                  mm_token_positions=[0, -5, 10],
                                  chunk_end_pos=20,
                                  special_token_offsets=[])


class TestFindInputMmEmbed:
    """Focused test cases for find_input_mm_embeds function - testing both KV cache reuse and chunked prefill."""

    def create_mock_runtime(self,
                            num_unseen_mm_tokens: int,
                            num_mm_tokens_in_chunk: int,
                            mm_token_lengths: List[int],
                            num_unseen_special_tokens: int = 0,
                            num_special_tokens_in_chunk: int = 0,
                            total_special_tokens_in_request: int = 0):
        """Helper to create a mock MultimodalRuntimeData."""
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.num_unseen_mm_tokens = num_unseen_mm_tokens
        runtime.num_mm_tokens_in_chunk = num_mm_tokens_in_chunk
        runtime.total_mm_tokens_in_request = sum(mm_token_lengths)
        runtime.num_unseen_special_tokens = num_unseen_special_tokens
        runtime.num_special_tokens_in_chunk = num_special_tokens_in_chunk
        runtime.total_special_tokens_in_request = total_special_tokens_in_request

        return runtime

    def create_multimodal_params(self, num_unseen_mm_tokens: int,
                                 num_mm_tokens_in_chunk: int,
                                 mm_token_lengths: List[int]):
        """Helper to create MultimodalParams with runtime data."""
        runtime = self.create_mock_runtime(num_unseen_mm_tokens,
                                           num_mm_tokens_in_chunk,
                                           mm_token_lengths)
        return MultimodalParams(multimodal_runtime=runtime)

    def test_mm_embed_not_batched(self):
        """
        Test individual batching mode where each mm_embed corresponds to one param.
        This tests the case where len(mm_embeds) == len(multimodal_params) > 1.
        """
        mm_embeds = [
            torch.randn(10, 512),  # Batch 1: 10 tokens
            torch.randn(15, 512),  # Batch 2: 15 tokens
            torch.randn(8, 512)  # Batch 3: 8 tokens
        ]
        multimodal_params = [
            self.create_multimodal_params(
                3, 7, [5, 5]),  # 3 unseen, 7 in current chunk
            self.create_multimodal_params(8, 7,
                                          [15]),  # 8 unseen, 7 in current chunk
            self.create_multimodal_params(
                0, 8, [4, 4])  # 0 unseen, 8 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Should return individual slices for each batch
        assert len(result) == 3
        assert result[0].shape == (7, 512)  # 7 tokens in current chunk
        assert result[1].shape == (7, 512)  # 7 tokens in current chunk
        assert result[2].shape == (8, 512)  # 8 tokens in current chunk

        # Verify the slices are correct
        torch.testing.assert_close(result[0], mm_embeds[0][3:10])
        torch.testing.assert_close(result[1], mm_embeds[1][8:15])
        torch.testing.assert_close(result[2], mm_embeds[2][0:8])

    def test_mm_embed_batched(self):
        """
        Test batching (concatenated) mm_embeds with fused mm_embeds for each batch.
        This tests the case where len(mm_embeds) == 1
        """
        mm_embeds = [torch.randn(33,
                                 512)]  # Pre-concatenated: 10 + 13 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(4, 6,
                                          [10]),  # 4 cached, 6 in current chunk
            self.create_multimodal_params(
                7, 6, [6, 7]),  # 7 cached, 6 in current chunk
            self.create_multimodal_params(
                3, 7, [4, 6])  # 3 cached, 7 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices:
        # Batch 1: [4:10] = 6 tokens
        # Batch 2: [10+7:10+13] = [17:23] = 6 tokens
        # Batch 3: [23+3:23+10] = [26:33] = 7 tokens
        # Total: 6 + 6 + 7 = 19 tokens
        assert len(result) == 1
        assert result[0].shape == (19, 512)

        # Verify the slices are correct
        expected = torch.cat(
            [
                mm_embeds[0][4:10],  # Batch 1: 6 tokens
                mm_embeds[0][17:23],  # Batch 2: 6 tokens
                mm_embeds[0][26:33]  # Batch 3: 7 tokens
            ],
            dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_mixed_caching_with_fully_cached_batches(self):
        """
        Test mixed scenarios where some batches are fully cached (should be skipped).
        """
        mm_embeds = [torch.randn(25, 512)]  # Pre-concatenated: 8 + 9 + 8 tokens
        multimodal_params = [
            self.create_multimodal_params(
                8, 0, [8]),  # All unseen - should be skipped
            self.create_multimodal_params(
                3, 6, [6, 3]),  # 3 unseen, 6 in current chunk
            self.create_multimodal_params(8, 0,
                                          [8])  # All unseen - should be skipped
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Only batch 2 should contribute: [8+3:8+9] = [11:17] = 6 tokens
        assert len(result) == 1
        assert result[0].shape == (6, 512)

        # Verify the slice is correct
        torch.testing.assert_close(result[0], mm_embeds[0][11:17])

    def test_all_batches_fully_unseen(self):
        """
        Test edge case where all batches are fully unseen.
        """
        mm_embeds = [torch.randn(30,
                                 512)]  # Pre-concatenated: 10 + 10 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(10, 0, [10]),  # All unseen
            self.create_multimodal_params(10, 0, [10]),  # All unseen
            self.create_multimodal_params(10, 0, [10])  # All unseen
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Should return empty list
        assert result == []

    def test_no_batches_cached(self):
        """
        Test edge case where no batches have any cached tokens.
        """
        mm_embeds = [torch.randn(30,
                                 512)]  # Pre-concatenated: 10 + 10 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(
                0, 10, [10]),  # No unseen, 10 in current chunk
            self.create_multimodal_params(
                0, 10, [10]),  # No unseen, 10 in current chunk
            self.create_multimodal_params(
                0, 10, [10])  # No unseen, 10 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Should return the full embeddings
        assert len(result) == 1
        torch.testing.assert_close(result[0], mm_embeds[0])

    def test_chunked_prefill_scenario(self):
        """
        Test chunked prefill scenario where some tokens are cached and some are in current chunk.
        """
        mm_embeds = [torch.randn(25, 512)]  # Pre-concatenated: 8 + 9 + 8 tokens
        multimodal_params = [
            self.create_multimodal_params(5, 3,
                                          [8]),  # 5 unseen, 3 in current chunk
            self.create_multimodal_params(2, 7,
                                          [9]),  # 2 unseen, 7 in current chunk
            self.create_multimodal_params(6, 2,
                                          [8])  # 6 unseen, 2 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices:
        # Batch 1: [5:8] = 3 tokens
        # Batch 2: [8+2:8+9] = [10:17] = 7 tokens
        # Batch 3: [17+6:17+8] = [23:25] = 2 tokens
        # Total: 3 + 7 + 2 = 12 tokens
        assert len(result) == 1
        assert result[0].shape == (12, 512)

        # Verify the slices are correct
        expected = torch.cat(
            [
                mm_embeds[0][5:8],  # Batch 1: 3 tokens
                mm_embeds[0][10:17],  # Batch 2: 7 tokens
                mm_embeds[0][23:25]  # Batch 3: 2 tokens
            ],
            dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_error_handling_mismatched_counts(self):
        """
        Test error handling when mm_embeds and multimodal_params counts don't match
        in individual batching mode.
        """
        mm_embeds = [torch.randn(10, 512), torch.randn(15, 512)]  # 2 embeddings
        multimodal_params = [self.create_multimodal_params(0, 10, [10])
                             ]  # Only 1 param

        with pytest.raises(
                ValueError,
                match=
                "Number of mm_embeds \\(2\\) does not match number of multimodal params \\(1\\)"
        ):
            find_input_mm_embeds(mm_embeds, multimodal_params)

    def test_single_batch_scenarios(self):
        """
        Test various single batch scenarios.
        """
        # Single batch, no caching
        mm_embeds = [torch.randn(20, 512)]
        multimodal_params = [self.create_multimodal_params(0, 20, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert len(result) == 1
        torch.testing.assert_close(result[0], mm_embeds[0])

        # Single batch, partial caching
        multimodal_params = [self.create_multimodal_params(5, 15, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert len(result) == 1
        assert result[0].shape == (15, 512)
        torch.testing.assert_close(result[0], mm_embeds[0][5:20])

        # Single batch, all cached
        multimodal_params = [self.create_multimodal_params(20, 0, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result == []

    def test_different_devices(self):
        """
        Test with tensors on different devices (if CUDA is available).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test CPU tensors
        mm_embeds = [torch.randn(10, 512, device='cpu')]
        multimodal_params = [self.create_multimodal_params(3, 7, [10])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result[0].device == mm_embeds[0].device

        # Test CUDA tensors
        mm_embeds = [torch.randn(10, 512, device='cuda')]
        multimodal_params = [self.create_multimodal_params(3, 7, [10])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result[0].device == mm_embeds[0].device

    def test_special_tokens_in_batched_mode(self):
        """Test special token handling in batched mode."""
        mm_embeds = [torch.randn(12, 512)
                     ]  # Pre-concatenated: (8-2) + (10-4) = 6 + 6 = 12 tokens
        multimodal_params = [
            self.create_mock_runtime(num_unseen_mm_tokens=2,
                                     num_mm_tokens_in_chunk=6,
                                     mm_token_lengths=[8],
                                     num_unseen_special_tokens=1,
                                     num_special_tokens_in_chunk=1,
                                     total_special_tokens_in_request=2),
            self.create_mock_runtime(num_unseen_mm_tokens=4,
                                     num_mm_tokens_in_chunk=6,
                                     mm_token_lengths=[10],
                                     num_unseen_special_tokens=2,
                                     num_special_tokens_in_chunk=2,
                                     total_special_tokens_in_request=4)
        ]
        multimodal_params = [
            MultimodalParams(multimodal_runtime=runtime)
            for runtime in multimodal_params
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices accounting for special tokens:
        # Batch 1: local_start = 2-1=1, local_end = 1+(6-1)=6, slice [1:6] = 5 tokens
        # Batch 2: local_start = 4-2=2, local_end = 2+(6-2)=6, slice [6+2:6+6] = [8:12] = 4 tokens
        # Total: 5 + 4 = 9 tokens
        assert len(result) == 1
        assert result[0].shape == (9, 512)

        # Verify the slices are correct
        expected = torch.cat(
            [
                mm_embeds[0][1:6],  # Batch 1: 5 tokens
                mm_embeds[0][8:12]  # Batch 2: 4 tokens
            ],
            dim=0)
        torch.testing.assert_close(result[0], expected)


class TestGetMultimodalEmbeddings:
    """Test cases for get_multimodal_embeddings function - testing caching and encoder forward optimization."""

    def create_mock_runtime(self,
                            total_mm_tokens: int,
                            total_special_tokens: int = 0):
        """Helper to create a mock MultimodalRuntimeData with total_mm_tokens and special_tokens."""
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.total_mm_tokens_in_request = total_mm_tokens
        runtime.total_special_tokens_in_request = total_special_tokens
        return runtime

    def create_multimodal_params_with_data(self,
                                           has_cached_embedding: bool = False,
                                           total_mm_tokens: int = 10,
                                           total_special_tokens: int = 0,
                                           cached_embedding=None):
        """Helper to create MultimodalParams with optional cached embeddings."""
        runtime = self.create_mock_runtime(total_mm_tokens,
                                           total_special_tokens)

        multimodal_data = {
            # Add some dummy multimodal data to ensure has_content() returns True
            "image": {
                "pixel_values": torch.randn(3, 224, 224)
            }
        }
        if has_cached_embedding:
            if cached_embedding is None:
                cached_embedding = torch.randn(total_mm_tokens, 512)
            multimodal_data["multimodal_embedding"] = cached_embedding

        param = MultimodalParams(multimodal_data=multimodal_data,
                                 multimodal_runtime=runtime)
        return param

    def test_no_multimodal_params(self):
        """Test with empty multimodal_params list."""

        def mock_encoder(params):
            return [torch.randn(10, 512)]

        result = get_multimodal_embeddings(mock_encoder, [])
        assert result == []

    def test_all_params_need_processing(self):
        """Test when all params need encoder processing (no cached embeddings)."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            # Return concatenated embeddings for all params
            total_tokens = sum(
                param.multimodal_runtime.total_mm_tokens_in_request
                for param in params)
            return [torch.randn(total_tokens, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=5),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=8),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=7)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once
        assert encoder_call_count == 1

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (20, 512)  # 5 + 8 + 7 = 20 tokens

        # All params should now have cached embeddings
        for param in multimodal_params:
            assert "multimodal_embedding" in param.multimodal_data
            assert param.multimodal_data["multimodal_embedding"] is not None

    def test_all_params_already_cached(self):
        """Test when all params already have cached embeddings."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, 512)]

        # Create params with pre-cached embeddings
        cached_emb1 = torch.randn(5, 512)
        cached_emb2 = torch.randn(8, 512)
        cached_emb3 = torch.randn(7, 512)

        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=5,
                cached_embedding=cached_emb1),
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=8,
                cached_embedding=cached_emb2),
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=7,
                cached_embedding=cached_emb3)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should not be called
        assert encoder_call_count == 0

        # Should return concatenated cached embeddings
        assert len(result) == 1
        assert result[0].shape == (20, 512)  # 5 + 8 + 7 = 20 tokens

        # Verify the embeddings are correct
        expected = torch.cat([cached_emb1, cached_emb2, cached_emb3], dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_mixed_cached_and_uncached(self):
        """Test mix of cached and uncached params."""
        encoder_call_count = 0
        processed_params = []

        def mock_encoder(params):
            nonlocal encoder_call_count, processed_params
            encoder_call_count += 1
            processed_params = params
            # Return embeddings for uncached params only
            total_tokens = sum(
                param.multimodal_runtime.total_mm_tokens_in_request
                for param in params)
            return [torch.randn(total_tokens, 512)]

        # Mix: cached, uncached, cached
        cached_emb = torch.randn(5, 512)
        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=5,
                cached_embedding=cached_emb),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=8),
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=7,
                cached_embedding=torch.randn(7, 512))
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once, only for uncached param
        assert encoder_call_count == 1
        assert len(processed_params) == 1  # Only the middle param
        assert processed_params[0] == multimodal_params[1]

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (20, 512)  # 5 + 8 + 7 = 20 tokens

        # Uncached param should now have cached embedding
        assert "multimodal_embedding" in multimodal_params[1].multimodal_data
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"] is not None

    def test_missing_multimodal_runtime(self):
        """Test handling when multimodal_runtime is missing."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, 512)]

        # Create param without multimodal_runtime but with content
        param = MultimodalParams(multimodal_data={
            "image": {
                "pixel_values": torch.randn(3, 224, 224)
            }
        })

        result = get_multimodal_embeddings(mock_encoder, [param])

        # Should call encoder and return its output directly (no caching)
        assert encoder_call_count == 1
        assert len(result) == 1
        assert result[0].shape == (10, 512)

        # Should not have cached embedding due to missing runtime
        assert "multimodal_embedding" not in param.multimodal_data

    def test_missing_total_mm_tokens(self):
        """Test handling when total_mm_tokens is None."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, 512)]

        # Create runtime without total_mm_tokens
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.total_mm_tokens_in_request = None

        param = MultimodalParams(multimodal_data={
            "image": {
                "pixel_values": torch.randn(3, 224, 224)
            }
        },
                                 multimodal_runtime=runtime)

        result = get_multimodal_embeddings(mock_encoder, [param])

        # Should call encoder and return its output directly (no caching)
        assert encoder_call_count == 1
        assert len(result) == 1
        assert result[0].shape == (10, 512)

    def test_multiple_modalities_early_return(self):
        """Test early return when encoder outputs multiple modalities."""

        def mock_encoder(params):
            # Return multiple embeddings (multiple modalities)
            return [torch.randn(5, 512), torch.randn(8, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=5)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return encoder output directly without caching
        assert len(result) == 2
        assert result[0].shape == (5, 512)
        assert result[1].shape == (8, 512)

        # Should not have cached anything
        assert "multimodal_embedding" not in multimodal_params[
            0].multimodal_data

    def test_caching_with_torch_split(self):
        """Test that caching uses torch.split correctly for multiple params."""

        def mock_encoder(params):
            # Return single concatenated tensor for all params
            return [torch.randn(20, 512)]  # 5 + 8 + 7 = 20 tokens

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=5),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=8),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=7)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Check that embeddings were split correctly
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (5, 512)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (8, 512)
        assert multimodal_params[2].multimodal_data[
            "multimodal_embedding"].shape == (7, 512)

        # Verify the result is correct concatenation
        assert result[0].shape == (20, 512)
        expected = torch.cat([
            multimodal_params[0].multimodal_data["multimodal_embedding"],
            multimodal_params[1].multimodal_data["multimodal_embedding"],
            multimodal_params[2].multimodal_data["multimodal_embedding"]
        ],
                             dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_different_devices(self):
        """Test with tensors on different devices (if CUDA is available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        def mock_encoder(params):
            return [torch.randn(10, 512, device='cuda')]

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=10)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Result should be on CUDA
        assert result[0].device.type == 'cuda'
        # Cached embedding should also be on CUDA
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].device.type == 'cuda'

    def test_special_tokens_basic_caching(self):
        """Test caching behavior with special tokens present."""

        def mock_encoder(params):
            # Return embeddings for non-special tokens only
            # Total: (10-2) + (8-1) + (6-3) = 8 + 7 + 3 = 18 tokens
            return [torch.randn(18, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=10,
                total_special_tokens=2),  # 8 actual embedding tokens
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=8,
                total_special_tokens=1),  # 7 actual embedding tokens
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=6,
                total_special_tokens=3)  # 3 actual embedding tokens
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (18, 512)  # 8 + 7 + 3 = 18 tokens

        # Check that embeddings were split correctly based on non-special token counts
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (8, 512)  # 10 - 2
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (7, 512)  # 8 - 1
        assert multimodal_params[2].multimodal_data[
            "multimodal_embedding"].shape == (3, 512)  # 6 - 3

    def test_special_tokens_all_special(self):
        """Test edge case where all tokens are special tokens."""

        def mock_encoder(params):
            # Should return empty tensor when no actual embedding tokens
            return [torch.randn(0, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=5,
                total_special_tokens=5),  # All tokens are special
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=3,
                total_special_tokens=3)  # All tokens are special
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return empty embeddings
        assert len(result) == 1
        assert result[0].shape == (0, 512)

        # Cached embeddings should also be empty
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (0, 512)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (0, 512)

    def test_special_tokens_mixed_with_cached(self):
        """Test special tokens with mixed cached and uncached params."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            # Only process uncached param: 12 - 3 = 9 tokens
            return [torch.randn(9, 512)]

        # Mix: cached (with special tokens), uncached (with special tokens)
        cached_emb = torch.randn(4, 512)  # 6 - 2 = 4 actual tokens
        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=6,
                total_special_tokens=2,
                cached_embedding=cached_emb),
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=12,
                total_special_tokens=3)  # 9 actual embedding tokens
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once for uncached param
        assert encoder_call_count == 1

        # Should return concatenated embeddings: 4 + 9 = 13 tokens
        assert len(result) == 1
        assert result[0].shape == (13, 512)

        # Verify cached embedding is preserved and uncached is now cached
        torch.testing.assert_close(
            multimodal_params[0].multimodal_data["multimodal_embedding"],
            cached_emb)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (9, 512)


if __name__ == "__main__":
    pytest.main([__file__])
