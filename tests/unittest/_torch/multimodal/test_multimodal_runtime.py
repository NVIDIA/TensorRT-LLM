from typing import List
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds, get_multimodal_embeddings)
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData,
                                            _as_cpu_tensor, _compute_mm_masks,
                                            _find_mm_token_start_pos_from_masks)
from tensorrt_llm.inputs.registry import (BaseMultimodalInputProcessor,
                                          maybe_compute_mm_embed_cumsum)

# Embedding dim kept small — functions under test only index along dim 0.
_EMBED_DIM = 4


def _make_runtime(
    num_cached_mm_tokens: int,
    num_mm_tokens_in_chunk: int,
    mm_token_lengths: List[int],
) -> MultimodalRuntimeData:
    """Build real runtime data with dense MM-token positions."""
    total_embeds = sum(mm_token_lengths)
    return MultimodalRuntimeData(
        past_seen_token_num=num_cached_mm_tokens,
        chunk_end_pos=num_cached_mm_tokens + num_mm_tokens_in_chunk,
        embed_mask_cumsum=torch.arange(1, total_embeds + 1, dtype=torch.int64),
    )


def _make_multimodal_params(
    num_cached_mm_tokens: int,
    num_mm_tokens_in_chunk: int,
    mm_token_lengths: List[int],
) -> MultimodalParams:
    """Build a MultimodalParams wrapping runtime data."""
    runtime = _make_runtime(num_cached_mm_tokens, num_mm_tokens_in_chunk,
                            mm_token_lengths)
    return MultimodalParams(multimodal_runtime=runtime)


class TestFindInputMmEmbed:
    """Test cases for find_input_mm_embeds — slicing embeddings per chunk."""

    def test_mm_embed_not_batched(self):
        """Individual batching: len(mm_embeds) == len(multimodal_params) > 1."""
        mm_embeds = [
            torch.randn(10, _EMBED_DIM),  # Batch 1: 10 tokens
            torch.randn(15, _EMBED_DIM),  # Batch 2: 15 tokens
            torch.randn(8, _EMBED_DIM),  # Batch 3: 8 tokens
        ]
        multimodal_params = [
            _make_multimodal_params(3, 7, [5, 5]),
            _make_multimodal_params(8, 7, [15]),
            _make_multimodal_params(0, 8, [4, 4]),
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        assert len(result) == 3
        assert result[0].shape == (7, _EMBED_DIM)
        assert result[1].shape == (7, _EMBED_DIM)
        assert result[2].shape == (8, _EMBED_DIM)
        torch.testing.assert_close(result[0], mm_embeds[0][3:10])
        torch.testing.assert_close(result[1], mm_embeds[1][8:15])
        torch.testing.assert_close(result[2], mm_embeds[2][0:8])

    def test_mm_embed_batched(self):
        """Batched: len(mm_embeds) == 1, slicing from concatenated tensor."""
        mm_embeds = [
            torch.randn(33, _EMBED_DIM)  # Pre-concatenated: 10 + 13 + 10 tokens
        ]
        multimodal_params = [
            _make_multimodal_params(4, 6, [10]),  # 4 cached, 6 in current chunk
            _make_multimodal_params(7, 6,
                                    [6, 7]),  # 7 cached, 6 in current chunk
            _make_multimodal_params(3, 7,
                                    [4, 6])  # 3 cached, 7 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices:
        # Batch 1: [4:10] = 6 tokens
        # Batch 2: [10+7:10+13] = [17:23] = 6 tokens
        # Batch 3: [23+3:23+10] = [26:33] = 7 tokens
        # Total: 6 + 6 + 7 = 19 tokens
        assert len(result) == 1
        assert result[0].shape == (19, _EMBED_DIM)

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
        mm_embeds = [torch.randn(25, _EMBED_DIM)
                     ]  # Pre-concatenated: 8 + 9 + 8 tokens
        multimodal_params = [
            _make_multimodal_params(8, 0,
                                    [8]),  # All unseen - should be skipped
            _make_multimodal_params(3, 6,
                                    [6, 3]),  # 3 unseen, 6 in current chunk
            _make_multimodal_params(8, 0, [8])  # All unseen - should be skipped
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Only batch 2 should contribute: [8+3:8+9] = [11:17] = 6 tokens
        assert len(result) == 1
        assert result[0].shape == (6, _EMBED_DIM)

        # Verify the slice is correct
        torch.testing.assert_close(result[0], mm_embeds[0][11:17])

    def test_all_batches_fully_unseen(self):
        """All cached, 0 in chunk → empty result."""
        mm_embeds = [
            torch.randn(30, _EMBED_DIM)  # Pre-concatenated: 10 + 10 + 10 tokens
        ]
        multimodal_params = [
            _make_multimodal_params(10, 0, [10]),
            _make_multimodal_params(10, 0, [10]),
            _make_multimodal_params(10, 0, [10]),
        ]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result == []

    def test_no_batches_cached(self):
        """0 cached, all in chunk → returns full tensor."""
        mm_embeds = [
            torch.randn(30, _EMBED_DIM)  # Pre-concatenated: 10 + 10 + 10 tokens
        ]
        multimodal_params = [
            _make_multimodal_params(0, 10, [10]),
            _make_multimodal_params(0, 10, [10]),
            _make_multimodal_params(0, 10, [10]),
        ]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert len(result) == 1
        torch.testing.assert_close(result[0], mm_embeds[0])

    def test_chunked_prefill_scenario(self):
        """Mix of cached and in-chunk across three batched requests."""
        mm_embeds = [torch.randn(25, _EMBED_DIM)
                     ]  # Pre-concatenated: 8 + 9 + 8 tokens
        multimodal_params = [
            _make_multimodal_params(5, 3, [8]),  # 5 unseen, 3 in current chunk
            _make_multimodal_params(2, 7, [9]),  # 2 unseen, 7 in current chunk
            _make_multimodal_params(6, 2, [8])  # 6 unseen, 2 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices:
        # Batch 1: [5:8] = 3 tokens
        # Batch 2: [8+2:8+9] = [10:17] = 7 tokens
        # Batch 3: [17+6:17+8] = [23:25] = 2 tokens
        # Total: 3 + 7 + 2 = 12 tokens
        assert len(result) == 1
        assert result[0].shape == (12, _EMBED_DIM)

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
        mm_embeds = [torch.randn(10, _EMBED_DIM),
                     torch.randn(15, _EMBED_DIM)]  # 2 embeddings
        multimodal_params = [_make_multimodal_params(0, 10,
                                                     [10])]  # Only 1 param

        with pytest.raises(
                ValueError,
                match=
                "Number of mm_embeds \\(2\\) does not match number of multimodal params \\(1\\)"
        ):
            find_input_mm_embeds(mm_embeds, multimodal_params)

    @pytest.mark.parametrize("cached,in_chunk,expect_empty,expect_len", [
        (0, 20, False, 20),
        (5, 15, False, 15),
        (20, 0, True, 0),
    ],
                             ids=["no_caching", "partial", "all_cached"])
    def test_single_batch(self, cached, in_chunk, expect_empty, expect_len):
        """Single-request batched slicing: no caching, partial, all cached."""
        mm_embeds = [torch.randn(20, _EMBED_DIM)]
        multimodal_params = [_make_multimodal_params(cached, in_chunk, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        if expect_empty:
            assert result == []
        else:
            assert len(result) == 1
            assert result[0].shape == (expect_len, _EMBED_DIM)
            torch.testing.assert_close(result[0],
                                       mm_embeds[0][cached:cached + in_chunk])

    def test_noncontiguous_two_requests_batched_chunk_in_gap(self):
        """
        Non-contiguous: two requests in a batch, where one request's chunk
        falls entirely in a text gap between MM regions.
        Request 1: 10 MM tokens, 5 cached, 5 in chunk.
        Request 2: 20 MM tokens (two regions of 10 each), but chunk is in the
                   text gap — 10 cached, 0 in chunk.
        Pre-concatenated: 10 + 20 = 30 tokens.
        """
        mm_embeds = [torch.randn(30, _EMBED_DIM)]
        multimodal_params = [
            _make_multimodal_params(5, 5, [10]),  # 5 cached, 5 in chunk
            _make_multimodal_params(10, 0,
                                    [10, 10]),  # 10 cached, 0 in chunk (gap)
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Only request 1 contributes: [5:10] = 5 tokens
        # Request 2 contributes nothing (0 in chunk)
        assert len(result) == 1
        assert result[0].shape == (5, _EMBED_DIM)
        torch.testing.assert_close(result[0], mm_embeds[0][5:10])

    def test_noncontiguous_individual_batching_mixed_gaps(self):
        """
        Non-contiguous: individual batching mode, three requests with
        different non-contiguous patterns.
        """
        mm_embeds = [
            torch.randn(20,
                        _EMBED_DIM),  # Request 1: 20 tokens (two regions of 10)
            torch.randn(15, _EMBED_DIM),  # Request 2: 15 tokens (one region)
            torch.randn(20,
                        _EMBED_DIM),  # Request 3: 20 tokens (two regions of 10)
        ]
        multimodal_params = [
            _make_multimodal_params(
                10, 10,
                [10, 10
                 ]),  # 10 cached (first region), 10 in chunk (second region)
            _make_multimodal_params(0, 15,
                                    [15]),  # nothing cached, all in chunk
            _make_multimodal_params(20, 0,
                                    [10, 10]),  # all cached, nothing in chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        assert len(result) == 3
        assert result[0].shape == (10, _EMBED_DIM)  # second region of request 1
        assert result[1].shape == (15, _EMBED_DIM)  # all of request 2
        assert result[2].shape == (0, _EMBED_DIM)  # nothing from request 3

        torch.testing.assert_close(result[0], mm_embeds[0][10:20])
        torch.testing.assert_close(result[1], mm_embeds[1][0:15])


def _make_mm_params_with_data(
    has_cached_embedding: bool = False,
    total_mm_tokens: int = 10,
    total_special_tokens: int = 0,
    cached_embedding=None,
) -> MultimodalParams:
    """Build a MultimodalParams with dummy pixel data (for has_content()) and optional cached embeddings."""
    runtime = Mock(spec=MultimodalRuntimeData)
    # total_embeds_in_request is the new name; it counts embed slots only,
    # so the "special" portion is subtracted at construction time here.
    runtime.total_embeds_in_request = total_mm_tokens - total_special_tokens

    multimodal_data = {"image": {"pixel_values": torch.randn(1, 1, 1)}}
    if has_cached_embedding:
        if cached_embedding is None:
            cached_embedding = torch.randn(runtime.total_embeds_in_request,
                                           _EMBED_DIM)
        multimodal_data["multimodal_embedding"] = cached_embedding

    return MultimodalParams(multimodal_data=multimodal_data,
                            multimodal_runtime=runtime)


class TestGetMultimodalEmbeddings:
    """Test cases for get_multimodal_embeddings — caching and encoder forward optimization."""

    def test_no_multimodal_params(self):
        """Test with empty multimodal_params list."""

        def mock_encoder(params):
            return [torch.randn(10, _EMBED_DIM)]

        result = get_multimodal_embeddings(mock_encoder, [])
        assert result == []

    def test_all_params_need_processing(self):
        """Test when all params need encoder processing (no cached embeddings)."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            # Return concatenated embeddings for all params
            total_tokens = sum(param.multimodal_runtime.total_embeds_in_request
                               for param in params)
            return [torch.randn(total_tokens, _EMBED_DIM)]

        multimodal_params = [
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=5),
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=8),
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=7)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once
        assert encoder_call_count == 1

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (20, _EMBED_DIM)  # 5 + 8 + 7 = 20 tokens

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
            return [torch.randn(10, _EMBED_DIM)]

        # Create params with pre-cached embeddings
        cached_emb1 = torch.randn(5, _EMBED_DIM)
        cached_emb2 = torch.randn(8, _EMBED_DIM)
        cached_emb3 = torch.randn(7, _EMBED_DIM)

        multimodal_params = [
            _make_mm_params_with_data(has_cached_embedding=True,
                                      total_mm_tokens=5,
                                      cached_embedding=cached_emb1),
            _make_mm_params_with_data(has_cached_embedding=True,
                                      total_mm_tokens=8,
                                      cached_embedding=cached_emb2),
            _make_mm_params_with_data(has_cached_embedding=True,
                                      total_mm_tokens=7,
                                      cached_embedding=cached_emb3)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should not be called
        assert encoder_call_count == 0

        # Should return concatenated cached embeddings
        assert len(result) == 1
        assert result[0].shape == (20, _EMBED_DIM)  # 5 + 8 + 7 = 20 tokens

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
            total_tokens = sum(param.multimodal_runtime.total_embeds_in_request
                               for param in params)
            return [torch.randn(total_tokens, _EMBED_DIM)]

        # Mix: cached, uncached, cached
        cached_emb = torch.randn(5, _EMBED_DIM)
        multimodal_params = [
            _make_mm_params_with_data(has_cached_embedding=True,
                                      total_mm_tokens=5,
                                      cached_embedding=cached_emb),
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=8),
            _make_mm_params_with_data(has_cached_embedding=True,
                                      total_mm_tokens=7,
                                      cached_embedding=torch.randn(
                                          7, _EMBED_DIM))
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once, only for uncached param
        assert encoder_call_count == 1
        assert len(processed_params) == 1  # Only the middle param
        assert processed_params[0] == multimodal_params[1]

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (20, _EMBED_DIM)  # 5 + 8 + 7 = 20 tokens

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
            return [torch.randn(10, _EMBED_DIM)]

        # Create param without multimodal_runtime but with content
        param = MultimodalParams(
            multimodal_data={"image": {
                "pixel_values": torch.randn(1, 1, 1)
            }})

        result = get_multimodal_embeddings(mock_encoder, [param])

        # Should call encoder and return its output directly (no caching)
        assert encoder_call_count == 1
        assert len(result) == 1
        assert result[0].shape == (10, _EMBED_DIM)

        # Should not have cached embedding due to missing runtime
        assert "multimodal_embedding" not in param.multimodal_data

    def test_missing_total_mm_tokens(self):
        """Test handling when total_mm_tokens is None."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, _EMBED_DIM)]

        # Create runtime without total_mm_tokens
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.total_embeds_in_request = None

        param = MultimodalParams(
            multimodal_data={"image": {
                "pixel_values": torch.randn(1, 1, 1)
            }},
            multimodal_runtime=runtime)

        result = get_multimodal_embeddings(mock_encoder, [param])

        # Should call encoder and return its output directly (no caching)
        assert encoder_call_count == 1
        assert len(result) == 1
        assert result[0].shape == (10, _EMBED_DIM)

    def test_multiple_modalities_early_return(self):
        """Test early return when encoder outputs multiple modalities."""

        def mock_encoder(params):
            # Return multiple embeddings (multiple modalities)
            return [torch.randn(5, _EMBED_DIM), torch.randn(8, _EMBED_DIM)]

        multimodal_params = [
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=5)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return encoder output directly without caching
        assert len(result) == 2
        assert result[0].shape == (5, _EMBED_DIM)
        assert result[1].shape == (8, _EMBED_DIM)

        # Should not have cached anything
        assert "multimodal_embedding" not in multimodal_params[
            0].multimodal_data

    def test_caching_with_torch_split(self):
        """Test that caching uses torch.split correctly for multiple params."""

        def mock_encoder(params):
            # Return single concatenated tensor for all params
            return [torch.randn(20, _EMBED_DIM)]  # 5 + 8 + 7 = 20 tokens

        multimodal_params = [
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=5),
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=8),
            _make_mm_params_with_data(has_cached_embedding=False,
                                      total_mm_tokens=7)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Check that embeddings were split correctly
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (5, _EMBED_DIM)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (8, _EMBED_DIM)
        assert multimodal_params[2].multimodal_data[
            "multimodal_embedding"].shape == (7, _EMBED_DIM)

        # Verify the result is correct concatenation
        assert result[0].shape == (20, _EMBED_DIM)
        expected = torch.cat([
            multimodal_params[0].multimodal_data["multimodal_embedding"],
            multimodal_params[1].multimodal_data["multimodal_embedding"],
            multimodal_params[2].multimodal_data["multimodal_embedding"]
        ],
                             dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_special_tokens_basic_caching(self):
        """Test caching behavior with special tokens present."""

        def mock_encoder(params):
            # Return embeddings for non-special tokens only
            # Total: (10-2) + (8-1) + (6-3) = 8 + 7 + 3 = 18 tokens
            return [torch.randn(18, _EMBED_DIM)]

        multimodal_params = [
            _make_mm_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=10,
                total_special_tokens=2),  # 8 actual embedding tokens
            _make_mm_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=8,
                total_special_tokens=1),  # 7 actual embedding tokens
            _make_mm_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=6,
                total_special_tokens=3)  # 3 actual embedding tokens
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (18, _EMBED_DIM)  # 8 + 7 + 3 = 18 tokens

        # Check that embeddings were split correctly based on non-special token counts
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (8, _EMBED_DIM)  # 10 - 2
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (7, _EMBED_DIM)  # 8 - 1
        assert multimodal_params[2].multimodal_data[
            "multimodal_embedding"].shape == (3, _EMBED_DIM)  # 6 - 3

    def test_special_tokens_all_special(self):
        """Test edge case where all tokens are special tokens."""

        def mock_encoder(params):
            # Should return empty tensor when no actual embedding tokens
            return [torch.randn(0, _EMBED_DIM)]

        multimodal_params = [
            _make_mm_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=5,
                total_special_tokens=5),  # All tokens are special
            _make_mm_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=3,
                total_special_tokens=3)  # All tokens are special
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return empty embeddings
        assert len(result) == 1
        assert result[0].shape == (0, _EMBED_DIM)

        # Cached embeddings should also be empty
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (0, _EMBED_DIM)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (0, _EMBED_DIM)

    def test_special_tokens_mixed_with_cached(self):
        """Test special tokens with mixed cached and uncached params."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            # Only process uncached param: 12 - 3 = 9 tokens
            return [torch.randn(9, _EMBED_DIM)]

        # Mix: cached (with special tokens), uncached (with special tokens)
        cached_emb = torch.randn(4, _EMBED_DIM)  # 6 - 2 = 4 actual tokens
        multimodal_params = [
            _make_mm_params_with_data(has_cached_embedding=True,
                                      total_mm_tokens=6,
                                      total_special_tokens=2,
                                      cached_embedding=cached_emb),
            _make_mm_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=12,
                total_special_tokens=3)  # 9 actual embedding tokens
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once for uncached param
        assert encoder_call_count == 1

        # Should return concatenated embeddings: 4 + 9 = 13 tokens
        assert len(result) == 1
        assert result[0].shape == (13, _EMBED_DIM)

        # Verify cached embedding is preserved and uncached is now cached
        torch.testing.assert_close(
            multimodal_params[0].multimodal_data["multimodal_embedding"],
            cached_emb)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (9, _EMBED_DIM)


def _find_mm_token_start_positions(input_ids,
                                   num_mm_tokens,
                                   vocab_size=None,
                                   mm_token_ids=None,
                                   mm_special_token_ids=None):
    """Compose the two intake helpers into the 2-tuple the tests assert on.

    Kept as a local test-file helper rather than a production wrapper since
    the composition is not used outside tests — production call sites in
    `multimodal_hashing_process` use the masks emitted by `_compute_mm_masks`
    for purposes other than just position-finding (e.g., stashing the embed
    mask), so a single-purpose wrapper would be strictly worse for them.
    """
    ids = _as_cpu_tensor(input_ids)
    if ids.numel() == 0:
        return [], []
    mm_mask, _, special_mask = _compute_mm_masks(ids, vocab_size, mm_token_ids,
                                                 mm_special_token_ids)
    return _find_mm_token_start_pos_from_masks(mm_mask, special_mask,
                                               num_mm_tokens)


class TestFindMmTokenStartPositions:
    """Integration tests for the intake position-finding composition:
    `_compute_mm_masks` + `_find_mm_token_start_pos_from_masks`.
    Verifies the 2-tuple `(start_positions, special_positions)` returned
    by composing the two helpers via `_find_mm_token_start_positions`."""

    def test_early_return_no_mm_tokens(self):
        """When input has no MM tokens, should return two empty lists."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        result = _find_mm_token_start_positions(
            input_ids=input_ids,
            num_mm_tokens=[],
            vocab_size=100,
        )
        assert result == ([], [])

    def test_early_return_no_match(self):
        """When mm_token_ids don't match anything in input_ids."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        result = _find_mm_token_start_positions(
            input_ids=input_ids,
            num_mm_tokens=[2],
            mm_token_ids=torch.tensor([99]),
        )
        assert result == ([], [])

    def test_basic_contiguous_tokens(self):
        """Basic case: contiguous MM tokens identified by out-of-vocab IDs."""
        input_ids = torch.tensor([1, 2, 10, 11, 12, 3, 4, 10, 11, 5])
        start_pos, special_pos = _find_mm_token_start_positions(
            input_ids=input_ids,
            num_mm_tokens=[3, 2],
            vocab_size=10,
        )
        assert start_pos == [2, 7]
        assert special_pos == []

    def test_with_mm_token_ids(self):
        """MM tokens identified by explicit token IDs."""
        input_ids = torch.tensor([1, 5, 5, 5, 2, 3, 5, 5, 4])
        start_pos, _ = _find_mm_token_start_positions(
            input_ids=input_ids,
            num_mm_tokens=[3, 2],
            mm_token_ids=torch.tensor([5]),
        )
        assert start_pos == [1, 6]

    def test_with_special_tokens(self):
        """Special tokens (e.g. image_break) detected within MM region."""
        input_ids = torch.tensor([1, 5, 5, 6, 5, 7, 2])
        start_pos, special_pos = _find_mm_token_start_positions(
            input_ids=input_ids,
            num_mm_tokens=[5],
            mm_token_ids=torch.tensor([5]),
            mm_special_token_ids=torch.tensor([6, 7]),
        )
        assert start_pos == [1]
        assert special_pos == [2, 4]

    @pytest.mark.parametrize(
        "input_ids,num_mm_tokens,expected_start_positions",
        [
            ([1, 100, 100, 2, 3, 100, 100, 100, 4], [5], [1]),
            ([0, 100, 100, 0, 0, 100, 0, 0, 100, 100, 0], [3, 2], [1, 8]),
        ],
        ids=["single_unit", "multiple_items"],
    )
    def test_non_contiguous_tokens(self, input_ids, num_mm_tokens,
                                   expected_start_positions):
        """Non-contiguous MM positions still start at each item's first MM token."""
        start_pos, _ = _find_mm_token_start_positions(
            input_ids=torch.tensor(input_ids),
            num_mm_tokens=num_mm_tokens,
            vocab_size=10,
        )
        assert start_pos == expected_start_positions

    def test_raises_without_vocab_size_or_mm_token_ids(self):
        """Should raise ValueError when neither vocab_size nor mm_token_ids provided."""
        with pytest.raises(ValueError,
                           match="Provide either mm_token_ids or vocab_size"):
            _find_mm_token_start_positions(
                input_ids=torch.tensor([1, 2, 3]),
                num_mm_tokens=[1],
            )


class _FakeMultimodalInputProcessor(BaseMultimodalInputProcessor):
    """Concrete test fake for maybe_compute_mm_embed_cumsum."""

    def __init__(self,
                 vocab_size=100,
                 mm_token_ids=None,
                 mm_special_token_ids=None):
        self._vocab_size = vocab_size
        self._mm_token_ids = mm_token_ids
        self._mm_special_token_ids = mm_special_token_ids

    @property
    def processor(self):
        return None

    @property
    def tokenizer(self):
        return None

    @property
    def config(self):
        return None

    @property
    def dtype(self):
        return torch.float32

    def __call__(self, inputs, sampling_params):
        raise NotImplementedError("This fake only supports token queries.")

    def get_vocab_size(self):
        return self._vocab_size

    def get_mm_token_ids(self):
        return self._mm_token_ids

    def get_mm_special_token_ids(self):
        return self._mm_special_token_ids


class TestMaybeComputeMmEmbedCumsum:
    """Test cases for maybe_compute_mm_embed_cumsum — emits a flat int64
    cumsum tensor at `extra["multimodal_data"]["multimodal_embed_mask_cumsum"]`."""

    def test_none_extra_is_noop(self):
        """No crash when extra_processed_inputs is None."""
        maybe_compute_mm_embed_cumsum([1, 2, 3], None,
                                      _FakeMultimodalInputProcessor())

    def test_no_multimodal_data_key_is_noop(self):
        """No crash when multimodal_data key is absent."""
        extra = {"some_other_key": {}}
        maybe_compute_mm_embed_cumsum([1, 2, 3], extra,
                                      _FakeMultimodalInputProcessor())
        assert "multimodal_embed_mask_cumsum" not in extra

    def test_already_present_is_idempotent(self):
        """Existing cumsum is NOT overwritten."""
        original_cumsum = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        extra = {
            "multimodal_data": {
                "multimodal_embed_mask_cumsum": original_cumsum
            }
        }
        maybe_compute_mm_embed_cumsum(
            [100, 101, 102, 103, 104], extra,
            _FakeMultimodalInputProcessor(vocab_size=100))
        assert (extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
                is original_cumsum)

    def test_computes_mask_when_absent(self):
        """Flat int64 cumsum computed from token IDs when not already present."""
        extra = {"multimodal_data": {"multimodal_embedding": "placeholder"}}
        # input: [1, 100, 101, 2, 102] → ids >= vocab_size=100 are mm.
        # bool mask: [F, T, T, F, T] → cumsum: [0, 1, 2, 2, 3]
        maybe_compute_mm_embed_cumsum(
            [1, 100, 101, 2, 102], extra,
            _FakeMultimodalInputProcessor(vocab_size=100))
        cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
        torch.testing.assert_close(
            cumsum,
            torch.tensor([0, 1, 2, 2, 3], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_no_mm_tokens_stores_all_false(self):
        """When no MM tokens match, stores an all-zero flat cumsum."""
        extra = {"multimodal_data": {"some_key": "value"}}
        maybe_compute_mm_embed_cumsum(
            [1, 2, 3], extra, _FakeMultimodalInputProcessor(vocab_size=100))
        cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
        torch.testing.assert_close(
            cumsum,
            torch.tensor([0, 0, 0], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_mask_excludes_special_tokens(self):
        """Specials do not increment the cumsum."""
        proc = _FakeMultimodalInputProcessor(
            vocab_size=None,
            mm_token_ids=torch.tensor([50, 60]),
            mm_special_token_ids=torch.tensor([60]))
        extra = {"multimodal_data": {"embed": "x"}}
        # input: [1, 50, 60, 50, 2] → mm at positions 1,3; special at 2.
        # bool mask: [F, T, F, T, F] → cumsum: [0, 1, 1, 2, 2]
        maybe_compute_mm_embed_cumsum([1, 50, 60, 50, 2], extra, proc)
        cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
        torch.testing.assert_close(
            cumsum,
            torch.tensor([0, 1, 1, 2, 2], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_no_vocab_and_no_mm_ids_is_noop(self):
        """When processor provides neither vocab_size nor mm_token_ids, no crash."""
        proc = _FakeMultimodalInputProcessor(vocab_size=None, mm_token_ids=None)
        extra = {"multimodal_data": {"embed": "x"}}
        maybe_compute_mm_embed_cumsum([100, 101], extra, proc)
        # Should not have set the cumsum since we can't identify MM tokens
        assert "multimodal_embed_mask_cumsum" not in extra["multimodal_data"]


if __name__ == "__main__":
    pytest.main([__file__])
