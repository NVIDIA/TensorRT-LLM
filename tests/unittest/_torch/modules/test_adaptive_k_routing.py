"""
Unit tests for AdaptiveKMoeRoutingMethod.

Tests cover:
- Basic functionality with different configurations
- Entropy-based K selection behavior
- Output shape and dtype compatibility
- Statistics tracking accuracy
- Edge cases

Author: Gabriele Balsamo (gabriele.balsamo30@gmail.com)
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fused_moe.routing import (
    AdaptiveKMoeRoutingMethod, RoutingMethodType)


class TestAdaptiveKMoeRoutingBasic:
    """Basic functionality tests for AdaptiveKMoeRoutingMethod."""

    @pytest.mark.parametrize("k_min,k_max", [(2, 8), (1, 4), (2, 6)])
    def test_init_parameters(self, k_min, k_max):
        """Test initialization with different k_min/k_max values."""
        routing = AdaptiveKMoeRoutingMethod(k_min=k_min, k_max=k_max)
        assert routing.k_min == k_min
        assert routing.k_max == k_max
        assert routing.top_k == k_max  # For compatibility

    def test_routing_method_type(self):
        """Test routing method type enum."""
        routing = AdaptiveKMoeRoutingMethod()
        assert routing.routing_method_type == RoutingMethodType.AdaptiveK

    def test_experts_per_token(self):
        """Test experts_per_token property returns k_max."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
        assert routing.experts_per_token == 8


class TestAdaptiveKMoeRoutingApply:
    """Tests for the apply method."""

    @pytest.fixture
    def routing(self):
        return AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)

    def test_output_shape(self, routing):
        """Test output shapes match expectations."""
        router_logits = torch.randn(100, 64, device='cuda')
        indices, scales = routing.apply(router_logits)

        assert indices.shape == (100, 8)  # (num_tokens, k_max)
        assert scales.shape == (100, 8)

    def test_output_dtype(self, routing):
        """Test output dtypes are correct."""
        router_logits = torch.randn(50, 32, device='cuda')
        indices, scales = routing.apply(router_logits)

        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32

    def test_weights_sum_to_one(self, routing):
        """Test that non-zero weights sum to ~1 per token."""
        router_logits = torch.randn(100, 64, device='cuda') * 3
        indices, scales = routing.apply(router_logits)

        # Each row should sum to 1 (considering all slots)
        weight_sums = scales.sum(dim=-1)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_experts_are_valid_indices(self, routing):
        """Test expert indices are valid."""
        num_experts = 64
        router_logits = torch.randn(100, num_experts, device='cuda')
        indices, scales = routing.apply(router_logits)

        assert (indices >= 0).all()
        assert (indices < num_experts).all()


class TestAdaptiveKMoeRoutingEntropy:
    """Tests for entropy-based K selection."""

    def test_low_entropy_uses_fewer_experts(self):
        """Test that low entropy (confident) routing uses fewer experts."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=2,
            k_max=8,
            entropy_thresholds=[1.0, 2.0]
        )

        # Create very peaked distribution (low entropy)
        logits = torch.zeros(10, 64, device='cuda')
        logits[:, 0] = 10.0  # Expert 0 dominates

        indices, scales = routing.apply(logits)
        stats = routing.get_stats()

        # Should use k_min for most tokens due to low entropy
        assert stats['avg_k'] < 4, f"Expected low avg_k, got {stats['avg_k']}"

    def test_high_entropy_uses_more_experts(self):
        """Test that high entropy (uncertain) routing uses more experts."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=2,
            k_max=8,
            entropy_thresholds=[0.5, 1.0]  # Very low thresholds
        )

        # Create uniform distribution (high entropy)
        logits = torch.zeros(10, 8, device='cuda')  # All equal -> max entropy

        indices, scales = routing.apply(logits)
        stats = routing.get_stats()

        # Should use k_max for all tokens (uniform = max entropy)
        assert stats['avg_k'] == 8, f"Expected k_max=8, got {stats['avg_k']}"


class TestAdaptiveKMoeRoutingStatistics:
    """Tests for statistics tracking."""

    def test_stats_initial(self):
        """Test initial statistics."""
        routing = AdaptiveKMoeRoutingMethod()
        stats = routing.get_stats()

        assert stats['total_tokens'] == 0
        assert stats['mean_entropy'] == 0.0

    def test_stats_after_apply(self):
        """Test statistics update after apply."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)

        logits = torch.randn(100, 64, device='cuda')
        routing.apply(logits)

        stats = routing.get_stats()
        assert stats['total_tokens'] == 100
        assert stats['mean_entropy'] > 0
        assert 'k_distribution' in stats
        assert 'compute_savings_pct' in stats

    def test_stats_accumulate(self):
        """Test statistics accumulate across multiple calls."""
        routing = AdaptiveKMoeRoutingMethod()

        routing.apply(torch.randn(50, 32, device='cuda'))
        routing.apply(torch.randn(50, 32, device='cuda'))

        stats = routing.get_stats()
        assert stats['total_tokens'] == 100

    def test_stats_reset(self):
        """Test statistics reset."""
        routing = AdaptiveKMoeRoutingMethod()
        routing.apply(torch.randn(100, 32, device='cuda'))

        routing.reset_stats()
        stats = routing.get_stats()

        assert stats['total_tokens'] == 0


class TestAdaptiveKMoeRoutingEdgeCases:
    """Edge case tests."""

    def test_single_token(self):
        """Test with single token."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
        indices, scales = routing.apply(torch.randn(1, 64, device='cuda'))

        assert indices.shape == (1, 8)
        assert scales.shape == (1, 8)

    def test_k_max_equals_num_experts(self):
        """Test when k_max equals number of experts."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)
        indices, scales = routing.apply(torch.randn(10, 8, device='cuda'))

        assert indices.shape == (10, 8)


class TestComputeSavings:
    """Tests for compute savings calculation."""

    def test_savings_with_sparse_logits(self):
        """Test realistic compute savings with sparse logits."""
        routing = AdaptiveKMoeRoutingMethod(k_min=2, k_max=8)

        # Simulate sparse logits (peaked distributions)
        torch.manual_seed(42)
        logits = torch.randn(1000, 8, device='cuda') * 3
        logits[:, 0] += 2  # Expert 0 preferred

        routing.apply(logits)
        stats = routing.get_stats()

        # Should achieve some savings with peaked distributions
        assert stats['compute_savings_pct'] > 0, \
            "Expected compute savings with sparse logits"

    def test_no_savings_with_uniform(self):
        """Test no savings with uniform distribution."""
        routing = AdaptiveKMoeRoutingMethod(
            k_min=2,
            k_max=8,
            entropy_thresholds=[0.1, 0.2]  # Very low thresholds
        )

        # Uniform logits -> max entropy -> use k_max
        logits = torch.zeros(100, 8, device='cuda')

        routing.apply(logits)
        stats = routing.get_stats()

        # With uniform distribution and low thresholds, should use k_max
        assert stats['avg_k'] == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
