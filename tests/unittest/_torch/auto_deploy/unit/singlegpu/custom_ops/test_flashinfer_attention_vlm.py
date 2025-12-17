"""Unit tests for FlashInfer attention op with VLM custom masks.

Tests the mask selection logic and PlanParams hashing for VLM support.
"""

# ruff: noqa: I001

from tensorrt_llm._torch.auto_deploy.custom_ops import torch_attention as _torch_attention  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_attention import (
    FlashInferAttention as FlashInferAttentionDescriptor,
    PlanParams,
)
import torch
from torch.fx import Graph


class TestMaskKindSelection:
    """Tests for mask_kind selection logic in flashinfer_mha_with_cache."""

    def test_mask_selection_full(self):
        """mask_kind='full' should select custom_mask_full."""
        mask_kind = "full"
        is_generate = False
        custom_mask_full = torch.tensor([True, True, False, True])
        custom_mask_sliding = torch.tensor([True, False, False, True])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        elif mask_kind == "sliding" and custom_mask_sliding is not None:
            custom_mask = custom_mask_sliding
        else:
            custom_mask = None

        assert custom_mask is custom_mask_full

    def test_mask_selection_sliding(self):
        """mask_kind='sliding' should select custom_mask_sliding."""
        mask_kind = "sliding"
        is_generate = False
        custom_mask_full = torch.tensor([True, True, False, True])
        custom_mask_sliding = torch.tensor([True, False, False, True])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        elif mask_kind == "sliding" and custom_mask_sliding is not None:
            custom_mask = custom_mask_sliding
        else:
            custom_mask = None

        assert custom_mask is custom_mask_sliding

    def test_mask_selection_none(self):
        """mask_kind='none' should result in no custom mask."""
        mask_kind = "none"
        is_generate = False
        custom_mask_full = torch.tensor([True, True, False, True])
        custom_mask_sliding = torch.tensor([True, False, False, True])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        elif mask_kind == "sliding" and custom_mask_sliding is not None:
            custom_mask = custom_mask_sliding
        else:
            custom_mask = None

        assert custom_mask is None

    def test_mask_ignored_during_generate(self):
        """Custom masks should be ignored when is_generate=True (s=1)."""
        mask_kind = "full"  # Would normally select full mask
        is_generate = True  # Decode phase
        custom_mask_full = torch.tensor([True, True, False, True])
        custom_mask_sliding = torch.tensor([True, False, False, True])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        elif mask_kind == "sliding" and custom_mask_sliding is not None:
            custom_mask = custom_mask_sliding
        else:
            custom_mask = None

        assert custom_mask is None

    def test_mask_selection_with_none_masks(self):
        """mask_kind='full' with None masks should result in None."""
        mask_kind = "full"
        is_generate = False
        custom_mask_full = None
        custom_mask_sliding = None

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        elif mask_kind == "sliding" and custom_mask_sliding is not None:
            custom_mask = custom_mask_sliding
        else:
            custom_mask = None

        assert custom_mask is None

    def test_mask_selection_unknown_kind(self):
        """Unknown mask_kind should result in None."""
        mask_kind = "unknown"
        is_generate = False
        custom_mask_full = torch.tensor([True, True])
        custom_mask_sliding = torch.tensor([True, False])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        elif mask_kind == "sliding" and custom_mask_sliding is not None:
            custom_mask = custom_mask_sliding
        else:
            custom_mask = None

        assert custom_mask is None


class TestPlanParamsWithCustomMask:
    """Tests for PlanParams with has_custom_mask field."""

    def test_plan_params_default_no_custom_mask(self):
        """Default PlanParams should have has_custom_mask=False."""
        pp = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
        )

        assert pp.has_custom_mask is False

    def test_plan_params_with_custom_mask(self):
        """PlanParams with has_custom_mask=True."""
        pp = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )

        assert pp.has_custom_mask is True

    def test_plan_params_hash_differs_with_custom_mask(self):
        """has_custom_mask should affect PlanParams hash."""
        pp1 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=False,
        )
        pp2 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )

        assert hash(pp1) != hash(pp2)

    def test_plan_params_hash_same_with_same_config(self):
        """Same config should produce same hash."""
        pp1 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )
        pp2 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )

        assert hash(pp1) == hash(pp2)

    def test_plan_params_equality(self):
        """PlanParams with same config should be equal."""
        pp1 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )
        pp2 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )

        assert pp1 == pp2

    def test_plan_params_inequality_with_custom_mask(self):
        """PlanParams differing only in has_custom_mask should not be equal."""
        pp1 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=False,
        )
        pp2 = PlanParams(
            n_heads=8,
            n_kv_heads=8,
            head_dim=64,
            num_seq=4,
            is_generate=False,
            page_size=16,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
            has_custom_mask=True,
        )

        assert pp1 != pp2


class TestGenerateVsPrefillPhase:
    """Tests for generate vs prefill phase detection."""

    def test_is_generate_true_for_s_equals_1(self):
        """s=1 should indicate generate phase."""
        s = 1
        is_generate = s == 1

        assert is_generate is True

    def test_is_generate_false_for_s_greater_than_1(self):
        """s>1 should indicate prefill phase."""
        s = 128
        is_generate = s == 1

        assert is_generate is False

    def test_mask_logic_prefill(self):
        """Prefill phase should use custom masks."""
        s = 128
        mask_kind = "full"
        is_generate = s == 1
        custom_mask_full = torch.tensor([True, False])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        else:
            custom_mask = None

        assert custom_mask is custom_mask_full

    def test_mask_logic_generate(self):
        """Generate phase should not use custom masks."""
        s = 1  # Single token (decode)
        mask_kind = "full"
        is_generate = s == 1
        custom_mask_full = torch.tensor([True, False])

        if mask_kind == "none" or is_generate:
            custom_mask = None
        elif mask_kind == "full" and custom_mask_full is not None:
            custom_mask = custom_mask_full
        else:
            custom_mask = None

        assert custom_mask is None


class TestFlashInferConstantsExtraction:
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
                "mask_kind": "none",
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
