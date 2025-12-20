"""Unit tests for FlashInfer VLM mask generation custom op."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_gemma3_mask import (
    _get_context_mask,
    flashinfer_vlm_mask_gen,
)


class TestGetContextMask:
    """Tests for _get_context_mask function (core mask logic).

    This tests the core masking behavior:
    - Text tokens: Causal attention (lower triangular)
    - Image tokens: Bidirectional attention (can attend to each other)
    """

    def test_causal_only_no_images(self):
        """No image tokens -> standard causal (lower triangular) mask."""
        image_token_mask = torch.tensor([False, False, False, False])

        result = _get_context_mask(image_token_mask)

        # Standard causal mask: position i can attend to positions 0..i
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        assert torch.equal(result, expected)

    def test_image_bidirectional(self):
        """Image tokens should attend to each other bidirectionally."""
        # All image tokens
        image_token_mask = torch.tensor([True, True, True])

        result = _get_context_mask(image_token_mask)

        # Image-to-image is fully bidirectional -> all True
        expected = torch.ones(3, 3, dtype=torch.bool)
        assert torch.equal(result, expected)

    def test_mixed_text_image(self):
        """Mixed text+image: text causal, image bidirectional."""
        # [text, image, image, text]
        image_token_mask = torch.tensor([False, True, True, False])

        result = _get_context_mask(image_token_mask)

        # Position 0 (text): can attend to [0]
        # Position 1 (image): can attend to [0, 1, 2] (causal + bidir with pos 2)
        # Position 2 (image): can attend to [0, 1, 2] (causal + bidir with pos 1)
        # Position 3 (text): can attend to [0, 1, 2, 3] (standard causal)
        expected = torch.tensor(
            [
                [True, False, False, False],  # text at 0
                [True, True, True, False],  # image at 1 (bidir with 2)
                [True, True, True, False],  # image at 2 (bidir with 1)
                [True, True, True, True],  # text at 3
            ]
        )
        assert torch.equal(result, expected)

    def test_single_token(self):
        """Single token should produce 1x1 True mask."""
        image_token_mask = torch.tensor([False])

        result = _get_context_mask(image_token_mask)

        expected = torch.tensor([[True]])
        assert torch.equal(result, expected)

    def test_image_at_end(self):
        """Image tokens at end of sequence."""
        image_token_mask = torch.tensor([False, False, True, True])

        result = _get_context_mask(image_token_mask)

        # Images at 2 and 3 should be bidirectional
        assert result[2, 3]  # image at 2 can attend to 3
        assert result[3, 2]  # image at 3 can attend to 2

    def test_image_at_start(self):
        """Image tokens at start of sequence."""
        image_token_mask = torch.tensor([True, True, False, False])

        result = _get_context_mask(image_token_mask)

        # Images at 0 and 1 should be bidirectional
        assert result[0, 1]  # image at 0 can attend to 1
        assert result[1, 0]  # image at 1 can attend to 0

    def test_multiple_image_blocks(self):
        """Multiple image blocks with text in between.

        Note: ALL image tokens can attend to ALL other image tokens bidirectionally,
        regardless of which image block they belong to. This is the correct behavior
        for Gemma3 VLM.
        """
        # [text, image, image, text, text, image, image, text]
        image_token_mask = torch.tensor([False, True, True, False, False, True, True, False])

        result = _get_context_mask(image_token_mask)

        # First image block (positions 1, 2) should be bidirectional with each other
        assert result[1, 2]  # image at 1 can attend to image at 2
        assert result[2, 1]  # image at 2 can attend to image at 1

        # Second image block (positions 5, 6) should be bidirectional with each other
        assert result[5, 6]  # image at 5 can attend to image at 6
        assert result[6, 5]  # image at 6 can attend to image at 5

        # ALL image tokens attend to ALL other image tokens bidirectionally
        # (this is the expected Gemma3 behavior)
        assert result[1, 5]  # image at 1 CAN see image at 5 (bidirectional)
        assert result[1, 6]  # image at 1 CAN see image at 6 (bidirectional)
        assert result[2, 5]  # image at 2 CAN see image at 5 (bidirectional)
        assert result[2, 6]  # image at 2 CAN see image at 6 (bidirectional)
        assert result[5, 1]  # image at 5 CAN see image at 1 (bidirectional)
        assert result[6, 2]  # image at 6 CAN see image at 2 (bidirectional)

        # Text tokens should remain causal
        assert not result[0, 3]  # text at 0 cannot see future text at 3
        assert result[3, 0]  # text at 3 can see past text at 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFlashinferVlmMaskGen:
    """Tests for flashinfer_vlm_mask_gen custom op."""

    def test_single_context_sequence(self):
        """Single context sequence generates correct mask."""
        device = "cuda"
        # [text, image, image, text]
        image_token_mask = torch.tensor([False, True, True, False], device=device)
        qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
        seq_len = torch.tensor([4], dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        # Mask should be flattened 4x4 = 16 elements
        assert mask.shape == (16,)
        assert mask.dtype == torch.bool

    def test_multiple_context_sequences(self):
        """Multiple context sequences concatenate correctly."""
        device = "cuda"
        # Seq 1: 3 tokens, Seq 2: 2 tokens
        image_token_mask = torch.tensor([False, True, False, True, True], device=device)
        qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
        seq_len = torch.tensor([3, 2], dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        # Total: 3*3 + 2*2 = 9 + 4 = 13 elements
        assert mask.shape == (13,)

    def test_no_context_sequences(self):
        """No context sequences (all generate) -> empty mask."""
        device = "cuda"
        image_token_mask = torch.tensor([False, False], device=device)
        qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        seq_len = torch.tensor([1, 1], dtype=torch.int32, device=device)  # All generate

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        # No context requests -> empty
        assert mask.shape == (0,)

    def test_output_shapes_match_flattened_masks(self):
        """Output shapes match sum(seq_len^2) for context sequences."""
        device = "cuda"
        seq_lens = [5, 3, 7]  # All context
        total_tokens = sum(seq_lens)
        image_token_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)

        cumsum = [0] + list(torch.tensor(seq_lens).cumsum(0).tolist())
        qo_indptr = torch.tensor(cumsum, dtype=torch.int32, device=device)
        seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        expected_size = sum(s * s for s in seq_lens)
        assert mask.shape == (expected_size,)

    def test_mask_is_contiguous(self):
        """Output mask should be a contiguous tensor."""
        device = "cuda"
        image_token_mask = torch.tensor([False, True, False, True], device=device)
        qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
        seq_len = torch.tensor([4], dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        assert mask.is_contiguous()

    def test_mask_values_match_expected(self):
        """Verify actual mask values for a known input."""
        device = "cuda"
        # [text, image, image, text]
        image_token_mask = torch.tensor([False, True, True, False], device=device)
        qo_indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
        seq_len = torch.tensor([4], dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        # Expected 2D mask:
        # [True,  False, False, False]  # text at 0
        # [True,  True,  True,  False]  # image at 1 (bidir with 2)
        # [True,  True,  True,  False]  # image at 2 (bidir with 1)
        # [True,  True,  True,  True]   # text at 3
        expected_2d = torch.tensor(
            [
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
            device=device,
        )
        expected_flat = expected_2d.flatten()

        assert torch.equal(mask, expected_flat)

    def test_all_text_produces_causal_mask(self):
        """All text tokens should produce a standard causal mask."""
        device = "cuda"
        image_token_mask = torch.tensor([False, False, False], device=device)
        qo_indptr = torch.tensor([0, 3], dtype=torch.int32, device=device)
        seq_len = torch.tensor([3], dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        # Expected: lower triangular (causal)
        expected_2d = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            device=device,
        )
        expected_flat = expected_2d.flatten()

        assert torch.equal(mask, expected_flat)

    def test_all_image_produces_bidirectional_mask(self):
        """All image tokens should produce a fully bidirectional mask."""
        device = "cuda"
        image_token_mask = torch.tensor([True, True, True], device=device)
        qo_indptr = torch.tensor([0, 3], dtype=torch.int32, device=device)
        seq_len = torch.tensor([3], dtype=torch.int32, device=device)

        mask = flashinfer_vlm_mask_gen(image_token_mask, qo_indptr, seq_len)

        # Expected: all True (bidirectional)
        expected = torch.ones(9, dtype=torch.bool, device=device)

        assert torch.equal(mask, expected)
