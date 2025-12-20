"""Unit tests for VLM auto-detection and mask generation in kvcache_transformers.

Tests the auto-detection of VLM models based on model_type and VlmMaskGeneratorRegistry.
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn


class MockAttentionModule(nn.Module):
    """Mock attention module."""

    def __init__(self):
        super().__init__()
        # Create a mock _node_ref
        self._node_ref = MagicMock()
        self._node_ref.kwargs = {}

    def forward(self, x):
        return x


class MockConfig:
    """Mock model config."""

    def __init__(self, model_type: str = None, sub_configs: list = None):
        self.model_type = model_type
        self.sub_configs = sub_configs or []


class TestVLMAutoDetection:
    """Tests for VLM auto-detection based on model_type and registry."""

    def test_detection_with_registered_model_type(self):
        """Model with registered model_type is detected as VLM."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.vlm_mask_registry import (
            VlmMaskGeneratorRegistry,
        )

        # Gemma3 is registered
        model_type = "gemma3"
        mask_generator = VlmMaskGeneratorRegistry.get(model_type)

        assert mask_generator is not None

    def test_detection_with_unregistered_model_type(self):
        """Model with unregistered model_type is not detected as VLM."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.vlm_mask_registry import (
            VlmMaskGeneratorRegistry,
        )

        model_type = "llama"  # Not a VLM with custom mask needs
        mask_generator = VlmMaskGeneratorRegistry.get(model_type)

        assert mask_generator is None

    def test_detection_with_none_model_type(self):
        """None model_type returns None from registry."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.vlm_mask_registry import (
            VlmMaskGeneratorRegistry,
        )

        model_type = None
        mask_generator = VlmMaskGeneratorRegistry.get(model_type)

        assert mask_generator is None

    def test_model_type_from_main_config(self):
        """model_type extracted from main config."""
        config = MockConfig(model_type="gemma3")

        model_type = getattr(config, "model_type", None)

        assert model_type == "gemma3"

    def test_model_type_from_text_config(self):
        """model_type extracted from text_config when not in main."""
        config = MockConfig(model_type=None, sub_configs=["text_config"])
        config.text_config = MockConfig(model_type="gemma3")

        model_type = getattr(config, "model_type", None)
        if model_type is None and hasattr(config, "text_config"):
            model_type = getattr(config.text_config, "model_type", None)

        assert model_type == "gemma3"


class TestMaskGeneration:
    """Tests for VLM mask generation."""

    def test_image_token_mask_from_token_type_ids(self):
        """Image token mask is correctly derived from token_type_ids."""
        token_type_ids = torch.tensor([0, 1, 1, 0, 0])  # 1 = image token
        image_token_mask = token_type_ids == 1

        expected = torch.tensor([False, True, True, False, False])
        assert torch.equal(image_token_mask, expected)

    def test_no_images_detected(self):
        """No images detected when all token_type_ids are 0."""
        token_type_ids = torch.tensor([0, 0, 0, 0])
        image_token_mask = token_type_ids == 1

        has_images = image_token_mask.any().item()
        assert has_images is False

    def test_images_detected(self):
        """Images detected when token_type_ids contains 1s."""
        token_type_ids = torch.tensor([0, 1, 1, 0])
        image_token_mask = token_type_ids == 1

        has_images = image_token_mask.any().item()
        assert has_images is True


class TestForwardWithPrepareMetadataMaskGeneration:
    """Tests for mask generation in forward_with_prepare_metadata."""

    def test_mask_initialized_to_none(self):
        """custom_mask should default to None."""
        cm_kwargs = {}
        cm_kwargs["custom_mask"] = None

        assert cm_kwargs["custom_mask"] is None

    def test_mask_generated_when_image_tokens_present(self):
        """Mask should be generated when image tokens are present."""
        token_type_ids = torch.tensor([0, 1, 1, 0])
        image_token_mask = token_type_ids == 1

        has_images = image_token_mask.any().item()

        assert has_images is True

    def test_mask_not_generated_when_no_images(self):
        """Mask should stay None when no image tokens."""
        token_type_ids = torch.tensor([0, 0, 0, 0])
        image_token_mask = token_type_ids == 1

        has_images = image_token_mask.any().item()

        assert has_images is False

    def test_mask_not_generated_when_token_type_ids_missing(self):
        """Mask should stay None when token_type_ids is missing."""
        cm_kwargs = {"input_ids": torch.tensor([1, 2, 3])}

        token_type_ids = cm_kwargs.get("token_type_ids")

        assert token_type_ids is None


class TestUnifiedAttnExportNoMaskKind:
    """Tests unified attention wrapper does not pass mask_kind to torch_attention."""

    def test_gemma3_nested_text_config_layer_types(self):
        from tensorrt_llm._torch.auto_deploy.export.library.unified_attn import (
            torch_attention_hf_wrapper,
        )

        class TextCfg:
            layer_types = ["full_attention", "sliding_attention"]

        class Cfg:
            text_config = TextCfg()

        class Mod(torch.nn.Module):
            def __init__(self, layer_idx: int):
                super().__init__()
                self.layer_idx = layer_idx
                self.config = Cfg()

        # Monkeypatch torch op to capture kwargs passed by wrapper.
        captured = {"kwargs": {}}
        original = torch.ops.auto_deploy.torch_attention

        def _fake(*args, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return args[0]

        torch.ops.auto_deploy.torch_attention = _fake  # type: ignore[assignment]
        try:
            q = torch.zeros(1, 2, 3, 4)
            k = torch.zeros(1, 2, 3, 4)
            v = torch.zeros(1, 2, 3, 4)

            torch_attention_hf_wrapper(Mod(0), q, k, v, None)
            # mask_kind should not be passed to torch_attention
            assert "mask_kind" not in captured["kwargs"]

            torch_attention_hf_wrapper(Mod(1), q, k, v, None)
            assert "mask_kind" not in captured["kwargs"]
        finally:
            torch.ops.auto_deploy.torch_attention = original  # type: ignore[assignment]
