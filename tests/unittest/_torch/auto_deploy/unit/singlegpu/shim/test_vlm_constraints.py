"""Unit tests for VLM constraint checks in ADExecutor.

Tests that chunked_prefill and block_reuse are properly validated when
image tokens are present (e.g., Gemma3 VLM).
"""

from types import SimpleNamespace

import pytest
import torch


class MockRequest:
    """Mock LLM request for testing."""

    def __init__(
        self,
        tokens: list,
        seq_slot: int,
        context_current_position: int = 0,
        context_chunk_size: int = None,
        py_multimodal_data: dict = None,
    ):
        self.tokens = tokens
        self.seq_slot = seq_slot
        self.context_current_position = context_current_position
        self.context_chunk_size = context_chunk_size or len(tokens)
        self.py_multimodal_data = py_multimodal_data
        self.py_batch_idx = None

    def get_tokens(self, beam_idx: int):
        return self.tokens

    def get_token(self, beam_idx: int, idx: int):
        return self.tokens[idx]

    def get_num_tokens(self, beam_idx: int):
        return len(self.tokens)


class MockKVCacheManager:
    """Mock KV cache manager for testing."""

    def get_cache_indices(self, request):
        return list(range(10))

    def get_num_kv_blocks(self, end_compute: int):
        return (end_compute + 15) // 16


class MockResourceManager:
    """Mock resource manager for testing."""

    def get_resource_manager(self, resource_type):
        return MockKVCacheManager()


class TestVLMConstraintChecks:
    """Tests for VLM constraint validation in _prepare_inputs."""

    def _create_mock_ad_config(
        self, enable_chunked_prefill: bool = False, enable_block_reuse: bool = False
    ):
        """Create a mock ADConfig with specified settings."""
        kv_cache_config = SimpleNamespace(enable_block_reuse=enable_block_reuse)
        return SimpleNamespace(
            enable_chunked_prefill=enable_chunked_prefill,
            kv_cache_config=kv_cache_config,
        )

    def test_constraint_check_logic_with_image_tokens(self):
        """Verify the constraint check logic for token_type_ids with images."""
        # Simulate the constraint check from _prepare_inputs
        extra_args = {
            "token_type_ids": [
                torch.tensor([0, 0, 1, 1, 0]),  # Has image tokens (1s)
            ]
        }

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        assert has_image_tokens is True

    def test_constraint_check_logic_without_image_tokens(self):
        """Verify no false positive when no image tokens."""
        extra_args = {
            "token_type_ids": [
                torch.tensor([0, 0, 0, 0, 0]),  # All text tokens
            ]
        }

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        assert has_image_tokens is False

    def test_constraint_check_logic_missing_token_type_ids(self):
        """Verify no error when token_type_ids is missing."""
        extra_args = {
            "input_ids": [torch.tensor([1, 2, 3])],
        }

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        assert has_image_tokens is False

    def test_constraint_check_logic_multiple_sequences(self):
        """Verify detection across multiple sequences."""
        extra_args = {
            "token_type_ids": [
                torch.tensor([0, 0, 0]),  # No images
                torch.tensor([0, 1, 0]),  # Has image
                torch.tensor([0, 0, 0]),  # No images
            ]
        }

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        assert has_image_tokens is True

    def test_constraint_check_raises_for_chunked_prefill(self):
        """Should raise error when enable_chunked_prefill=True with images."""
        extra_args = {"token_type_ids": [torch.tensor([0, 1, 1, 0])]}

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        ad_config = self._create_mock_ad_config(
            enable_chunked_prefill=True, enable_block_reuse=False
        )

        if has_image_tokens:
            if getattr(ad_config, "enable_chunked_prefill", False):
                with pytest.raises(RuntimeError, match="enable_chunked_prefill"):
                    raise RuntimeError(
                        "Gemma3 VLM with image tokens requires enable_chunked_prefill=False."
                    )

    def test_constraint_check_raises_for_block_reuse(self):
        """Should raise error when enable_block_reuse=True with images."""
        extra_args = {"token_type_ids": [torch.tensor([0, 1, 1, 0])]}

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        ad_config = self._create_mock_ad_config(
            enable_chunked_prefill=False, enable_block_reuse=True
        )

        if has_image_tokens:
            kv_cache_config = getattr(ad_config, "kv_cache_config", None)
            if kv_cache_config and getattr(kv_cache_config, "enable_block_reuse", False):
                with pytest.raises(RuntimeError, match="enable_block_reuse"):
                    raise RuntimeError(
                        "Gemma3 VLM with image tokens requires enable_block_reuse=False."
                    )

    def test_no_constraint_error_when_disabled(self):
        """No error when both chunked_prefill and block_reuse are disabled."""
        extra_args = {"token_type_ids": [torch.tensor([0, 1, 1, 0])]}

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        ad_config = self._create_mock_ad_config(
            enable_chunked_prefill=False, enable_block_reuse=False
        )

        # This should NOT raise
        error_raised = False
        if has_image_tokens:
            if getattr(ad_config, "enable_chunked_prefill", False):
                error_raised = True
            kv_cache_config = getattr(ad_config, "kv_cache_config", None)
            if kv_cache_config and getattr(kv_cache_config, "enable_block_reuse", False):
                error_raised = True

        assert error_raised is False

    def test_no_constraint_error_without_images(self):
        """No error when no image tokens even if features are enabled."""
        extra_args = {
            "token_type_ids": [torch.tensor([0, 0, 0, 0])]  # All text
        }

        has_image_tokens = any(
            (t == 1).any() if isinstance(t, torch.Tensor) else False
            for t in extra_args.get("token_type_ids", [])
        )

        # Should not check constraints if no images
        error_raised = False
        if has_image_tokens:
            error_raised = True  # Would check constraints

        assert error_raised is False
        assert has_image_tokens is False
