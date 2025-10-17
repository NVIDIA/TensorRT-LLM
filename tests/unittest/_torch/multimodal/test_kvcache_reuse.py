from unittest.mock import Mock

import pytest
import torch

# Import the function to test
from tensorrt_llm._torch.models.modeling_multimodal_utils import \
    find_uncached_mm_embeds
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData)


class TestMultimodalRuntimeData:
    """Test cases for MultimodalRuntimeData computation logic, specifically num_cached_mm_tokens."""

    def test_fully_cached_multimodal_tokens(self):
        """Test when all multimodal tokens are cached."""
        runtime = MultimodalRuntimeData(
            num_cached_tokens=20,
            mm_token_lengths=[5, 8, 7],  # Total: 20 tokens
            mm_token_positions=[0, 5, 13]  # Positions: 0-5, 5-13, 13-20
        )

        # All tokens should be cached since num_cached_tokens (20) >= all positions + lengths
        assert runtime.num_cached_mm_tokens == 20
        assert runtime.total_mm_tokens == 20

    def test_no_cached_multimodal_tokens(self):
        """Test when no multimodal tokens are cached."""
        runtime = MultimodalRuntimeData(
            num_cached_tokens=10,
            mm_token_lengths=[5, 8, 7],  # Total: 20 tokens
            mm_token_positions=[10, 18, 30]  # All positions > num_cached_tokens
        )

        # No multimodal tokens should be cached
        assert runtime.num_cached_mm_tokens == 0
        assert runtime.total_mm_tokens == 20

    def test_complex_scenario_with_multiple_chunks(self):
        """Test a complex scenario with many chunks and various caching states."""
        runtime = MultimodalRuntimeData(
            num_cached_tokens=30,
            mm_token_lengths=[3, 4, 5, 6, 7, 8],  # Total: 33 tokens
            mm_token_positions=[
                0, 5, 10, 15, 25, 35
            ]  # Positions: 0-3, 5-9, 10-15, 15-21, 25-32, 35-43
        )

        # Expected caching:
        # Chunk 0: fully cached (3 tokens)
        # Chunk 1: fully cached (4 tokens)
        # Chunk 2: fully cached (5 tokens)
        # Chunk 3: fully cached (6 tokens)
        # Chunk 4: partially cached (30-25=5 out of 7 tokens)
        # Chunk 5: not cached
        expected_cached = 3 + 4 + 5 + 6 + 5  # 23 tokens
        assert runtime.num_cached_mm_tokens == expected_cached
        assert runtime.total_mm_tokens == 33


class TestFindUncachedMmEmbed:
    """Focused test cases for find_uncached_mm_embeds function - testing edge cases and potential bugs."""

    def create_mock_runtime(self, num_cached_mm_tokens: int,
                            total_mm_tokens: int):
        """Helper to create a mock MultimodalRuntimeData."""
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.num_cached_mm_tokens = num_cached_mm_tokens
        runtime.total_mm_tokens = total_mm_tokens
        return runtime

    def create_multimodal_params(self, num_cached_mm_tokens: int,
                                 total_mm_tokens: int):
        """Helper to create MultimodalParams with runtime data."""
        runtime = self.create_mock_runtime(num_cached_mm_tokens,
                                           total_mm_tokens)
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
            self.create_multimodal_params(3, 10),  # 3 cached, 7 uncached
            self.create_multimodal_params(8, 15),  # 8 cached, 7 uncached
            self.create_multimodal_params(0, 8)  # 0 cached, 8 uncached
        ]

        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)

        # Should return individual slices for each batch
        assert len(result) == 3
        assert result[0].shape == (7, 512)  # 10 - 3 = 7
        assert result[1].shape == (7, 512)  # 15 - 8 = 7
        assert result[2].shape == (8, 512)  # 8 - 0 = 8

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
            self.create_multimodal_params(4, 10),  # 4 cached, 6 uncached
            self.create_multimodal_params(7, 13),  # 7 cached, 6 uncached
            self.create_multimodal_params(3, 10)  # 3 cached, 7 uncached
        ]

        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)

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
            self.create_multimodal_params(8,
                                          8),  # All cached - should be skipped
            self.create_multimodal_params(3, 9),  # 3 cached, 6 uncached
            self.create_multimodal_params(8,
                                          8)  # All cached - should be skipped
        ]

        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)

        # Only batch 2 should contribute: [8+3:8+9] = [11:17] = 6 tokens
        assert len(result) == 1
        assert result[0].shape == (6, 512)

        # Verify the slice is correct
        torch.testing.assert_close(result[0], mm_embeds[0][11:17])

    def test_all_batches_fully_cached(self):
        """
        Test edge case where all batches are fully cached.
        """
        mm_embeds = [torch.randn(30,
                                 512)]  # Pre-concatenated: 10 + 10 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(10, 10),  # All cached
            self.create_multimodal_params(10, 10),  # All cached
            self.create_multimodal_params(10, 10)  # All cached
        ]

        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)

        # Should return empty list
        assert result == []

    def test_no_batches_cached(self):
        """
        Test edge case where no batches have any cached tokens.
        """
        mm_embeds = [torch.randn(30,
                                 512)]  # Pre-concatenated: 10 + 10 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(0, 10),  # No cached
            self.create_multimodal_params(0, 10),  # No cached
            self.create_multimodal_params(0, 10)  # No cached
        ]

        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)

        # Should return the full embeddings
        assert result == mm_embeds

    def test_error_handling_mismatched_counts(self):
        """
        Test error handling when mm_embeds and multimodal_params counts don't match
        in individual batching mode.
        """
        mm_embeds = [torch.randn(10, 512), torch.randn(15, 512)]  # 2 embeddings
        multimodal_params = [self.create_multimodal_params(0,
                                                           10)]  # Only 1 param

        with pytest.raises(
                ValueError,
                match=
                "Number of mm_embeds \\(2\\) does not match number of multimodal params \\(1\\)"
        ):
            find_uncached_mm_embeds(mm_embeds, multimodal_params)

    def test_single_batch_scenarios(self):
        """
        Test various single batch scenarios.
        """
        # Single batch, no caching
        mm_embeds = [torch.randn(20, 512)]
        multimodal_params = [self.create_multimodal_params(0, 20)]
        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)
        assert result == mm_embeds

        # Single batch, partial caching
        multimodal_params = [self.create_multimodal_params(5, 20)]
        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)
        assert len(result) == 1
        assert result[0].shape == (15, 512)
        torch.testing.assert_close(result[0], mm_embeds[0][5:20])

        # Single batch, all cached
        multimodal_params = [self.create_multimodal_params(20, 20)]
        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)
        assert result == []

    def test_different_devices(self):
        """
        Test with tensors on different devices (if CUDA is available).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test CPU tensors
        mm_embeds = [torch.randn(10, 512, device='cpu')]
        multimodal_params = [self.create_multimodal_params(3, 10)]
        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)
        assert result[0].device == mm_embeds[0].device

        # Test CUDA tensors
        mm_embeds = [torch.randn(10, 512, device='cuda')]
        multimodal_params = [self.create_multimodal_params(3, 10)]
        result = find_uncached_mm_embeds(mm_embeds, multimodal_params)
        assert result[0].device == mm_embeds[0].device


if __name__ == "__main__":
    pytest.main([__file__])
