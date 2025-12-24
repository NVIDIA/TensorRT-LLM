"""Unit tests for FlashInfer attention op with VLM support."""

# ruff: noqa: I001

from tensorrt_llm._torch.auto_deploy.custom_ops import torch_attention as _torch_attention  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_attention import (
    FlashInferAttention as FlashInferAttentionDescriptor,
)
import torch
from torch.fx import Graph


class TestFlashInferConstantsExtraction:
    """Tests for FlashInferAttentionDescriptor.get_constants() method."""

    def _make_torch_attention_node(self, *, sliding_window, logit_cap):
        g = Graph()
        q = g.placeholder("q")
        k = g.placeholder("k")
        v = g.placeholder("v")
        attn = g.call_function(
            torch.ops.auto_deploy.torch_attention,
            args=(q, k, v),
            kwargs={
                "attn_mask": None,
                "dropout_p": 0.0,
                "is_causal": True,
                "scale": None,
                "sinks": None,
                "sliding_window": sliding_window,
                "logit_cap": logit_cap,
                "layout": "bsnd",
            },
        )
        return attn

    def test_sliding_window_converts_to_window_left(self):
        # sliding_window is exclusive; FlashInfer window_left is inclusive.
        node = self._make_torch_attention_node(sliding_window=128, logit_cap=None)
        constants = FlashInferAttentionDescriptor.get_constants(node)
        assert constants[-2] == 127  # window_left

    def test_logit_cap_converts_to_logits_soft_cap(self):
        node = self._make_torch_attention_node(sliding_window=None, logit_cap=30.0)
        constants = FlashInferAttentionDescriptor.get_constants(node)
        assert constants[-1] == 30.0  # logits_soft_cap

    def test_defaults_disable_features(self):
        node = self._make_torch_attention_node(sliding_window=None, logit_cap=None)
        constants = FlashInferAttentionDescriptor.get_constants(node)
        assert constants[-2] == -1  # window_left disabled
        assert constants[-1] == 0.0  # logits_soft_cap disabled
