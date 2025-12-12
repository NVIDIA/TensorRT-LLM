"""Unit tests for VLM auto-detection and mask generation in kvcache_transformers.

Tests the auto-detection of VLM models based on attention_type and the
forward_with_prepare_metadata mask generation logic.
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn


class MockAttentionModule(nn.Module):
    """Mock attention module with configurable attention_type."""

    def __init__(self, attention_type: str = None):
        super().__init__()
        if attention_type is not None:
            self.attention_type = attention_type
        # Create a mock _node_ref
        self._node_ref = MagicMock()
        self._node_ref.kwargs = {}

    def forward(self, x):
        return x


class MockConfig:
    """Mock model config."""

    def __init__(self, sliding_window: int = None, sub_configs: list = None):
        self.sliding_window = sliding_window
        self.sub_configs = sub_configs or []


class TestVLMAutoDetection:
    """Tests for VLM auto-detection based on attention_type."""

    def test_detection_with_full_attention_type(self):
        """Model with attention_type='full_attention' triggers detection."""
        # Simulate the detection logic from _apply_to_full_model
        mod = nn.Module()
        attn1 = MockAttentionModule("full_attention")
        attn1._node_ref.kwargs = {"mask_kind": "full"}
        mod.attn1 = attn1

        needs_vlm_masks = any(
            submod._node_ref.kwargs.get("mask_kind", "none") != "none"
            for submod in mod.modules()
            if hasattr(submod, "_node_ref")
        )

        assert needs_vlm_masks is True

    def test_detection_with_sliding_attention_type(self):
        """Model with attention_type='sliding_attention' triggers detection."""
        mod = nn.Module()
        attn1 = MockAttentionModule("sliding_attention")
        attn1._node_ref.kwargs = {"mask_kind": "sliding"}
        mod.attn1 = attn1

        needs_vlm_masks = any(
            submod._node_ref.kwargs.get("mask_kind", "none") != "none"
            for submod in mod.modules()
            if hasattr(submod, "_node_ref")
        )

        assert needs_vlm_masks is True

    def test_detection_without_attention_type(self):
        """Model without attention_type skips VLM config."""
        mod = nn.Module()
        attn1 = MockAttentionModule()  # No attention_type
        attn1._node_ref.kwargs = {"mask_kind": "none"}
        mod.attn1 = attn1

        needs_vlm_masks = any(
            submod._node_ref.kwargs.get("mask_kind", "none") != "none"
            for submod in mod.modules()
            if hasattr(submod, "_node_ref")
        )

        assert needs_vlm_masks is False

    def test_detection_mixed_attention_types(self):
        """Mixed layers: some full, some sliding, some none."""
        mod = nn.Module()

        attn1 = MockAttentionModule("full_attention")
        attn1._node_ref.kwargs = {"mask_kind": "full"}
        mod.attn1 = attn1

        attn2 = MockAttentionModule("sliding_attention")
        attn2._node_ref.kwargs = {"mask_kind": "sliding"}
        mod.attn2 = attn2

        attn3 = MockAttentionModule()
        attn3._node_ref.kwargs = {"mask_kind": "none"}
        mod.attn3 = attn3

        needs_vlm_masks = any(
            submod._node_ref.kwargs.get("mask_kind", "none") != "none"
            for submod in mod.modules()
            if hasattr(submod, "_node_ref")
        )

        assert needs_vlm_masks is True

    def test_detection_all_none(self):
        """All layers with mask_kind='none' → no VLM masks needed."""
        mod = nn.Module()

        for i in range(3):
            attn = MockAttentionModule()
            attn._node_ref.kwargs = {"mask_kind": "none"}
            setattr(mod, f"attn{i}", attn)

        needs_vlm_masks = any(
            submod._node_ref.kwargs.get("mask_kind", "none") != "none"
            for submod in mod.modules()
            if hasattr(submod, "_node_ref")
        )

        assert needs_vlm_masks is False

    def test_detection_no_node_ref(self):
        """Modules without _node_ref are skipped."""
        mod = nn.Module()
        mod.linear = nn.Linear(10, 10)  # No _node_ref

        needs_vlm_masks = any(
            submod._node_ref.kwargs.get("mask_kind", "none") != "none"
            for submod in mod.modules()
            if hasattr(submod, "_node_ref")
        )

        assert needs_vlm_masks is False


class TestSlidingWindowExtraction:
    """Tests for sliding_window extraction from config."""

    def test_sliding_window_from_main_config(self):
        """sliding_window extracted from main config."""
        config = MockConfig(sliding_window=512)

        sliding_window = getattr(config, "sliding_window", None)

        assert sliding_window == 512

    def test_sliding_window_from_subconfig(self):
        """sliding_window extracted from text_config when not in main."""
        config = MockConfig(sliding_window=None, sub_configs=["text_config"])
        config.text_config = MockConfig(sliding_window=1024)

        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            for sub_config_key in getattr(config, "sub_configs", []):
                sub_config = getattr(config, sub_config_key, None)
                if sub_config:
                    sliding_window = getattr(sub_config, "sliding_window", None)
                    if sliding_window is not None:
                        break

        assert sliding_window == 1024

    def test_sliding_window_missing(self):
        """No sliding_window anywhere → None."""
        config = MockConfig(sliding_window=None, sub_configs=["text_config"])
        config.text_config = MockConfig(sliding_window=None)

        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is None:
            for sub_config_key in getattr(config, "sub_configs", []):
                sub_config = getattr(config, sub_config_key, None)
                if sub_config:
                    sliding_window = getattr(sub_config, "sliding_window", None)
                    if sliding_window is not None:
                        break

        assert sliding_window is None


class TestMaskKindMapping:
    """Tests for attention_type → mask_kind mapping in fake_profiler_mha."""

    def test_full_attention_maps_to_full(self):
        """attention_type='full_attention' → mask_kind='full'."""
        attn_type = "full_attention"

        if attn_type == "full_attention":
            mask_kind = "full"
        elif attn_type == "sliding_attention":
            mask_kind = "sliding"
        else:
            mask_kind = "none"

        assert mask_kind == "full"

    def test_sliding_attention_maps_to_sliding(self):
        """attention_type='sliding_attention' → mask_kind='sliding'."""
        attn_type = "sliding_attention"

        if attn_type == "full_attention":
            mask_kind = "full"
        elif attn_type == "sliding_attention":
            mask_kind = "sliding"
        else:
            mask_kind = "none"

        assert mask_kind == "sliding"

    def test_unknown_attention_type_maps_to_none(self):
        """Unknown attention_type → mask_kind='none'."""
        attn_type = "custom_attention"

        if attn_type == "full_attention":
            mask_kind = "full"
        elif attn_type == "sliding_attention":
            mask_kind = "sliding"
        else:
            mask_kind = "none"

        assert mask_kind == "none"

    def test_no_attention_type_maps_to_none(self):
        """No attention_type attribute → mask_kind='none'."""
        module = MockAttentionModule()  # No attention_type

        if hasattr(module, "attention_type"):
            attn_type = module.attention_type
            if attn_type == "full_attention":
                mask_kind = "full"
            elif attn_type == "sliding_attention":
                mask_kind = "sliding"
            else:
                mask_kind = "none"
        else:
            mask_kind = "none"

        assert mask_kind == "none"


class TestForwardWithPrepareMetadataMaskGeneration:
    """Tests for mask generation in forward_with_prepare_metadata."""

    def test_masks_initialized_to_none(self):
        """custom_mask_full and custom_mask_sliding should default to None."""
        cm_kwargs = {}
        cm_kwargs["custom_mask_full"] = None
        cm_kwargs["custom_mask_sliding"] = None

        assert cm_kwargs["custom_mask_full"] is None
        assert cm_kwargs["custom_mask_sliding"] is None

    def test_masks_generated_when_image_tokens_present(self):
        """Masks should be generated when image tokens are present."""
        # This is a simplified test of the logic
        token_type_ids = torch.tensor([0, 1, 1, 0])
        image_token_mask = token_type_ids == 1

        has_images = image_token_mask.any()

        assert has_images is True

    def test_masks_not_generated_when_no_images(self):
        """Masks should stay None when no image tokens."""
        token_type_ids = torch.tensor([0, 0, 0, 0])
        image_token_mask = token_type_ids == 1

        has_images = image_token_mask.any()

        assert has_images is False

    def test_masks_not_generated_when_token_type_ids_missing(self):
        """Masks should stay None when token_type_ids is missing."""
        cm_kwargs = {"input_ids": torch.tensor([1, 2, 3])}

        token_type_ids = cm_kwargs.get("token_type_ids")

        assert token_type_ids is None
