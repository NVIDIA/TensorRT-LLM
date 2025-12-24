"""Unit tests for VLM utilities (get_image_token_mask, has_image_tokens)."""

import torch

from tensorrt_llm._torch.auto_deploy.utils.vlm_utils import get_image_token_mask, has_image_tokens


class TestGetImageTokenMask:
    """Tests for get_image_token_mask function."""

    def test_gemma3_with_image_tokens(self):
        """token_type_ids with 1=image should return correct bool mask."""
        # token_type_ids: 0=text, 1=image
        token_type_ids = torch.tensor([0, 0, 1, 1, 1, 0, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = get_image_token_mask("gemma3", named_args)

        assert result is not None
        expected = torch.tensor([False, False, True, True, True, False, False])
        assert torch.equal(result, expected)

    def test_gemma3_no_image_tokens(self):
        """token_type_ids all zeros should return all-False mask."""
        token_type_ids = torch.tensor([0, 0, 0, 0, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = get_image_token_mask("gemma3", named_args)

        assert result is not None
        assert not result.any()  # All False

    def test_gemma3_all_image_tokens(self):
        """token_type_ids all ones should return all-True mask."""
        token_type_ids = torch.tensor([1, 1, 1, 1])
        named_args = {"token_type_ids": token_type_ids}

        result = get_image_token_mask("gemma3", named_args)

        assert result is not None
        assert result.all()  # All True

    def test_gemma3_missing_token_type_ids(self):
        """Missing token_type_ids should return None."""
        named_args = {"input_ids": torch.tensor([1, 2, 3])}

        result = get_image_token_mask("gemma3", named_args)

        assert result is None

    def test_gemma3_empty_named_args(self):
        """Empty named_args should return None."""
        result = get_image_token_mask("gemma3", {})

        assert result is None

    def test_qwen2_vl_with_mm_token_type_ids(self):
        """mm_token_type_ids should work for qwen2_vl."""
        mm_token_type_ids = torch.tensor([0, 1, 1, 0, 0, 1])
        named_args = {"mm_token_type_ids": mm_token_type_ids}

        result = get_image_token_mask("qwen2_vl", named_args)

        assert result is not None
        expected = torch.tensor([False, True, True, False, False, True])
        assert torch.equal(result, expected)

    def test_qwen2_5_vl_with_mm_token_type_ids(self):
        """mm_token_type_ids should work for qwen2.5_vl."""
        mm_token_type_ids = torch.tensor([1, 0, 0, 1])
        named_args = {"mm_token_type_ids": mm_token_type_ids}

        result = get_image_token_mask("qwen2.5_vl", named_args)

        assert result is not None
        expected = torch.tensor([True, False, False, True])
        assert torch.equal(result, expected)

    def test_qwen3_vl_with_mm_token_type_ids(self):
        """mm_token_type_ids should work for qwen3_vl."""
        mm_token_type_ids = torch.tensor([0, 1, 1, 1, 0])
        named_args = {"mm_token_type_ids": mm_token_type_ids}

        result = get_image_token_mask("qwen3_vl", named_args)

        assert result is not None
        expected = torch.tensor([False, True, True, True, False])
        assert torch.equal(result, expected)

    def test_llava_with_mm_token_type_ids(self):
        """mm_token_type_ids should work for llava."""
        mm_token_type_ids = torch.tensor([0, 0, 1, 1, 0])
        named_args = {"mm_token_type_ids": mm_token_type_ids}

        result = get_image_token_mask("llava", named_args)

        assert result is not None
        expected = torch.tensor([False, False, True, True, False])
        assert torch.equal(result, expected)

    def test_llava_next_with_mm_token_type_ids(self):
        """mm_token_type_ids should work for llava_next."""
        mm_token_type_ids = torch.tensor([1, 1, 0, 0])
        named_args = {"mm_token_type_ids": mm_token_type_ids}

        result = get_image_token_mask("llava_next", named_args)

        assert result is not None
        expected = torch.tensor([True, True, False, False])
        assert torch.equal(result, expected)

    def test_unknown_model_type(self):
        """Unknown model type should return None."""
        token_type_ids = torch.tensor([0, 1, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = get_image_token_mask("unknown_model", named_args)

        assert result is None

    def test_empty_model_type(self):
        """Empty model type should return None."""
        token_type_ids = torch.tensor([0, 1, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = get_image_token_mask("", named_args)

        assert result is None

    def test_gemma3_wrong_field_name(self):
        """Gemma3 with mm_token_type_ids (wrong field) should return None."""
        mm_token_type_ids = torch.tensor([0, 1, 0])
        named_args = {"mm_token_type_ids": mm_token_type_ids}

        result = get_image_token_mask("gemma3", named_args)

        assert result is None

    def test_qwen2_vl_wrong_field_name(self):
        """Qwen2-VL with token_type_ids (wrong field) should return None."""
        token_type_ids = torch.tensor([0, 1, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = get_image_token_mask("qwen2_vl", named_args)

        assert result is None


class TestHasImageTokens:
    """Tests for has_image_tokens function."""

    def test_has_image_tokens_true(self):
        """Should return True when image tokens present."""
        token_type_ids = torch.tensor([0, 0, 1, 1, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = has_image_tokens("gemma3", named_args)

        assert result is True

    def test_has_image_tokens_false_all_text(self):
        """Should return False when no image tokens."""
        token_type_ids = torch.tensor([0, 0, 0, 0, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = has_image_tokens("gemma3", named_args)

        assert result is False

    def test_has_image_tokens_false_missing_field(self):
        """Should return False when field is missing."""
        named_args = {"input_ids": torch.tensor([1, 2, 3])}

        result = has_image_tokens("gemma3", named_args)

        assert result is False

    def test_has_image_tokens_false_unknown_model(self):
        """Should return False for unknown model type."""
        token_type_ids = torch.tensor([0, 1, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = has_image_tokens("unknown_model", named_args)

        assert result is False

    def test_has_image_tokens_single_image(self):
        """Should return True even with single image token."""
        token_type_ids = torch.tensor([0, 0, 0, 1, 0, 0, 0])
        named_args = {"token_type_ids": token_type_ids}

        result = has_image_tokens("gemma3", named_args)

        assert result is True
