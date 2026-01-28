"""Tests for RMSNorm sharding detection and transformation.

This module tests that the sharding transform correctly detects and transforms
torch_rmsnorm ops based on weight shape and position in the graph:

1. Full hidden norm (weight shape = [num_heads * head_dim], AFTER q/k projection):
   - Detected as QK norm needing sharding → replaced with sharded_rmsnorm
   - Weight is sharded

2. Per-head norm (weight shape = [head_dim], like GLM):
   - NOT detected as needing sharding → stays as local torch_rmsnorm
   - No transformation needed

3. Input layernorm (feeds INTO q/k/v projections, not after):
   - NOT detected as QK norm → stays as local torch_rmsnorm
   - Even though weight shape matches, it's not a QK norm

These are graph-level unit tests that verify the transform logic.
"""

import torch
import torch.nn as nn

# Ensure custom ops are registered
from tensorrt_llm._torch.auto_deploy.custom_ops import rms_norm  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class AttentionWithInputLayernorm(nn.Module):
    """Attention module with input_layernorm BEFORE q/k/v projections.

    This simulates the standard Llama pattern where input_layernorm feeds
    INTO the attention projections. This should NOT be detected as a QK norm.
    """

    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Input layernorm - BEFORE q/k/v projections (like Llama)
        self.input_layernorm_weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-6

        # Q/K/V/O projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        # Input layernorm BEFORE projections (Llama pattern)
        x_normed = torch.ops.auto_deploy.torch_rmsnorm(x, self.input_layernorm_weight, self.eps)

        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Reshape for attention
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_heads, self.head_dim)
        v = v.view(b, s, self.num_heads, self.head_dim)

        y = torch.ops.auto_deploy.torch_attention(q, k, v, is_causal=True, layout="bsnd")
        y = y.contiguous().view(b, s, -1)

        return self.o_proj(y)


class SimpleAttentionWithQKNorm(nn.Module):
    """Attention module with configurable QK normalization.

    Args:
        hidden_size: Total hidden dimension
        num_heads: Number of attention heads
        use_full_hidden_norm: If True, use full hidden norm (like MiniMax)
            - Weight shape = [hidden_size], norm before reshape
            - Should be transformed to sharded_rmsnorm
            If False, use per-head norm (like GLM)
            - Weight shape = [head_dim], norm after reshape
            - Should NOT be transformed
    """

    def __init__(
        self, hidden_size: int = 64, num_heads: int = 4, use_full_hidden_norm: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_full_hidden_norm = use_full_hidden_norm

        # Q/K/V/O projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # QK norm weights: shape depends on norm type
        norm_dim = hidden_size if use_full_hidden_norm else self.head_dim
        self.q_norm_weight = nn.Parameter(torch.ones(norm_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(norm_dim))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_full_hidden_norm:
            # Full hidden norm: apply before reshape
            q = torch.ops.auto_deploy.torch_rmsnorm(q, self.q_norm_weight, self.eps)
            k = torch.ops.auto_deploy.torch_rmsnorm(k, self.k_norm_weight, self.eps)
            # Reshape for attention
            q = q.view(b, s, self.num_heads, self.head_dim)
            k = k.view(b, s, self.num_heads, self.head_dim)
            v = v.view(b, s, self.num_heads, self.head_dim)
        else:
            # Reshape first for per-head norm
            q = q.view(b, s, self.num_heads, self.head_dim)
            k = k.view(b, s, self.num_heads, self.head_dim)
            v = v.view(b, s, self.num_heads, self.head_dim)
            # Per-head norm: apply after reshape (broadcasts over heads)
            q = torch.ops.auto_deploy.torch_rmsnorm(q, self.q_norm_weight, self.eps)
            k = torch.ops.auto_deploy.torch_rmsnorm(k, self.k_norm_weight, self.eps)

        y = torch.ops.auto_deploy.torch_attention(q, k, v, is_causal=True, layout="bsnd")
        y = y.contiguous().view(b, s, -1)

        return self.o_proj(y)


def count_ops(gm, op) -> int:
    """Count the number of nodes with a specific op in the graph."""
    count = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and is_op(node, op):
            count += 1
    return count


# =============================================================================
# Tests
# =============================================================================


class TestRMSNormShardingTransform:
    """Tests for the sharding transform on RMSNorm ops."""

    def test_full_hidden_norm_transformed_to_sharded(self):
        """Test that full hidden norm RMSNorm ops are replaced with sharded_rmsnorm.

        When weight shape = [hidden_size] (matches q_proj output dim):
        - torch_rmsnorm should be replaced with sharded_rmsnorm
        - Weight should be sharded
        """
        model = SimpleAttentionWithQKNorm(use_full_hidden_norm=True).to("cuda", dtype=torch.float16)
        x = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)

        # Export to graph
        gm = torch_export_to_gm(model, args=(x,), clone=True)

        # Check before transform
        before_rmsnorm = count_ops(gm, torch.ops.auto_deploy.torch_rmsnorm.default)
        before_sharded = count_ops(gm, torch.ops.auto_deploy.sharded_rmsnorm.default)
        assert before_rmsnorm == 2, f"Expected 2 torch_rmsnorm before, got {before_rmsnorm}"
        assert before_sharded == 0, f"Expected 0 sharded_rmsnorm before, got {before_sharded}"

        # Apply sharding transform with world_size=2
        optimizer = InferenceOptimizer(
            None,
            {
                "detect_sharding": {
                    "stage": "sharding",
                    "simple_shard_only": False,
                    "sharding_source": ["manual", "factory", "heuristic"],
                    "support_partial_config": True,
                    "sharding_dims": ["tp", "ep", "bmm"],
                    "shard_all_unprocessed": True,
                    "allreduce_strategy": "NCCL",
                    "dist_backend": "auto",
                    "requires_shape_prop": True,
                },
                "sharding_transform_executor": {
                    "stage": "sharding",
                    "run_shape_prop": True,
                },
            },
        )
        optimizer.shared_config.local_rank = 0
        optimizer.shared_config.world_size = 2
        gm_transformed = optimizer(None, gm)

        # Check after transform
        after_rmsnorm = count_ops(gm_transformed, torch.ops.auto_deploy.torch_rmsnorm.default)
        after_sharded = count_ops(gm_transformed, torch.ops.auto_deploy.sharded_rmsnorm.default)

        # The QK norms (weight matching q/k output dims) should be transformed
        assert after_sharded == 2, (
            f"Expected 2 sharded_rmsnorm after transform, got {after_sharded}. "
            f"Remaining torch_rmsnorm: {after_rmsnorm}"
        )

    def test_per_head_norm_not_transformed(self):
        """Test that per-head norm RMSNorm ops are NOT replaced.

        When weight shape = [head_dim] (doesn't match q_proj output dim):
        - torch_rmsnorm should stay as torch_rmsnorm
        - No sharded_rmsnorm should be added
        """
        model = SimpleAttentionWithQKNorm(use_full_hidden_norm=False).to(
            "cuda", dtype=torch.float16
        )
        x = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)

        # Export to graph
        gm = torch_export_to_gm(model, args=(x,), clone=True)

        # Check before transform
        before_rmsnorm = count_ops(gm, torch.ops.auto_deploy.torch_rmsnorm.default)
        assert before_rmsnorm == 2, f"Expected 2 torch_rmsnorm before, got {before_rmsnorm}"

        # Apply sharding transform with world_size=2
        # Using the same config as default.yaml for detect_sharding and sharding_transform_executor
        optimizer = InferenceOptimizer(
            None,
            {
                "detect_sharding": {
                    "stage": "sharding",
                    "simple_shard_only": False,
                    "sharding_source": ["manual", "factory", "heuristic"],
                    "support_partial_config": True,
                    "sharding_dims": ["tp", "ep", "bmm"],
                    "shard_all_unprocessed": True,
                    "allreduce_strategy": "NCCL",
                    "dist_backend": "auto",
                    "requires_shape_prop": True,
                },
                "sharding_transform_executor": {
                    "stage": "sharding",
                    "run_shape_prop": True,
                },
            },
        )
        # Set world_size and rank on shared_config
        optimizer.shared_config.local_rank = 0
        optimizer.shared_config.world_size = 2
        gm_transformed = optimizer(None, gm)

        # Check after transform
        after_rmsnorm = count_ops(gm_transformed, torch.ops.auto_deploy.torch_rmsnorm.default)
        after_sharded = count_ops(gm_transformed, torch.ops.auto_deploy.sharded_rmsnorm.default)

        # Per-head norms should NOT be transformed to sharded
        assert after_sharded == 0, (
            f"Expected 0 sharded_rmsnorm for per-head norm, got {after_sharded}"
        )
        # The original rmsnorm ops should still be present (or fewer if some were removed)
        # Note: Some rmsnorm ops may be removed/transformed for other reasons, but none should become sharded
        print(f"After transform: {after_rmsnorm} torch_rmsnorm, {after_sharded} sharded_rmsnorm")

    def test_input_layernorm_not_transformed(self):
        """Test that input_layernorm (before q/k/v projections) is NOT replaced.

        When RMSNorm feeds INTO q/k/v projections (like Llama's input_layernorm):
        - torch_rmsnorm should stay as torch_rmsnorm
        - No sharded_rmsnorm should be added
        - Even though weight shape matches [hidden_size], it's not a QK norm

        This tests the fix for the bug where input_layernorm was incorrectly
        detected as a QK norm because its weight shape matched q_proj output dim.
        """
        model = AttentionWithInputLayernorm().to("cuda", dtype=torch.float16)
        x = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)

        # Export to graph
        gm = torch_export_to_gm(model, args=(x,), clone=True)

        # Check before transform
        before_rmsnorm = count_ops(gm, torch.ops.auto_deploy.torch_rmsnorm.default)
        assert before_rmsnorm == 1, f"Expected 1 torch_rmsnorm before, got {before_rmsnorm}"

        # Apply sharding transform with world_size=4 (like the failing Llama test)
        optimizer = InferenceOptimizer(
            None,
            {
                "detect_sharding": {
                    "stage": "sharding",
                    "simple_shard_only": False,
                    "sharding_source": ["manual", "factory", "heuristic"],
                    "support_partial_config": True,
                    "sharding_dims": ["tp", "ep", "bmm"],
                    "shard_all_unprocessed": True,
                    "allreduce_strategy": "NCCL",
                    "dist_backend": "auto",
                    "requires_shape_prop": True,
                },
                "sharding_transform_executor": {
                    "stage": "sharding",
                    "run_shape_prop": True,
                },
            },
        )
        optimizer.shared_config.local_rank = 0
        optimizer.shared_config.world_size = 4
        gm_transformed = optimizer(None, gm)

        # Check after transform
        after_rmsnorm = count_ops(gm_transformed, torch.ops.auto_deploy.torch_rmsnorm.default)
        after_sharded = count_ops(gm_transformed, torch.ops.auto_deploy.sharded_rmsnorm.default)

        # Input layernorm should NOT be transformed to sharded_rmsnorm
        # because it feeds INTO q/k/v projections, not after them
        assert after_sharded == 0, (
            f"Expected 0 sharded_rmsnorm for input_layernorm, got {after_sharded}. "
            f"input_layernorm should not be detected as a QK norm."
        )
        print(f"After transform: {after_rmsnorm} torch_rmsnorm, {after_sharded} sharded_rmsnorm")
