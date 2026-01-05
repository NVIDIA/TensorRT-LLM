"""Tests for BMM MoE checkpoint loading hooks."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.transform.library.fused_moe import (
    _bmm_moe_down_split_hook,
    _bmm_moe_gate_up_split_hook,
)


@pytest.fixture
def gate_up_stacked_weight():
    """Fixture for stacked gate_up weight in Llama4 format (E, H, 2*I)."""
    num_experts = 4
    hidden_size = 64
    intermediate_size = 32
    return torch.randn(num_experts, hidden_size, intermediate_size * 2)


@pytest.fixture
def down_stacked_weight():
    """Fixture for stacked down weight in Llama4 format (E, I, H)."""
    num_experts = 4
    hidden_size = 64
    intermediate_size = 32
    return torch.randn(num_experts, intermediate_size, hidden_size)


class TestBmmMoeGateUpSplitHook:
    """Tests for _bmm_moe_gate_up_split_hook."""

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size",
        [
            (4, 64, 32),
            (8, 128, 64),
            (2, 32, 16),
        ],
    )
    def test_splits_stacked_weights_into_per_expert_w1_w3(
        self, num_experts, hidden_size, intermediate_size
    ):
        """Verify gate_up hook splits stacked weights into w1/w3 per expert."""
        # Llama4 format: (E, H, 2*I)
        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        state_dict = {"gate_up_weight": stacked}
        w1_keys = [f"w1_{i}" for i in range(num_experts)]
        w3_keys = [f"w3_{i}" for i in range(num_experts)]

        _bmm_moe_gate_up_split_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            intermediate_size=intermediate_size,
            w1_keys=w1_keys,
            w3_keys=w3_keys,
        )

        for i in range(num_experts):
            assert w1_keys[i] in state_dict
            assert w3_keys[i] in state_dict
            # After transpose: (I, H)
            assert state_dict[w1_keys[i]].shape == (intermediate_size, hidden_size)
            assert state_dict[w3_keys[i]].shape == (intermediate_size, hidden_size)

    def test_w1_w3_content_matches_original_stacked(self):
        """Verify split w1/w3 tensors match the original stacked content."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        state_dict = {"gate_up_weight": stacked.clone()}
        w1_keys = [f"w1_{i}" for i in range(num_experts)]
        w3_keys = [f"w3_{i}" for i in range(num_experts)]

        _bmm_moe_gate_up_split_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            intermediate_size=intermediate_size,
            w1_keys=w1_keys,
            w3_keys=w3_keys,
        )

        for i in range(num_experts):
            # w1 is first half: stacked[i, :, :intermediate_size].T
            expected_w1 = stacked[i, :, :intermediate_size].transpose(0, 1).contiguous()
            # w3 is second half: stacked[i, :, intermediate_size:].T
            expected_w3 = stacked[i, :, intermediate_size:].transpose(0, 1).contiguous()

            torch.testing.assert_close(state_dict[w1_keys[i]], expected_w1)
            torch.testing.assert_close(state_dict[w3_keys[i]], expected_w3)

    def test_handles_missing_source_key(self):
        """Verify hook does nothing when source key is missing."""
        state_dict = {}

        # Should not raise
        _bmm_moe_gate_up_split_hook(
            state_dict,
            "",
            source_key="missing_key",
            intermediate_size=32,
            w1_keys=["w1"],
            w3_keys=["w3"],
        )

        assert len(state_dict) == 0

    @pytest.mark.parametrize("prefix", ["", "model.layers.0.moe."])
    def test_works_with_module_prefix(self, prefix):
        """Verify hook works correctly with module path prefix."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        state_dict = {f"{prefix}gate_up_weight": stacked}
        w1_keys = [f"w1_{i}" for i in range(num_experts)]
        w3_keys = [f"w3_{i}" for i in range(num_experts)]

        _bmm_moe_gate_up_split_hook(
            state_dict,
            prefix,
            source_key="gate_up_weight",
            intermediate_size=intermediate_size,
            w1_keys=w1_keys,
            w3_keys=w3_keys,
        )

        for i in range(num_experts):
            assert f"{prefix}{w1_keys[i]}" in state_dict
            assert f"{prefix}{w3_keys[i]}" in state_dict


class TestBmmMoeDownSplitHook:
    """Tests for _bmm_moe_down_split_hook."""

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size",
        [
            (4, 64, 32),
            (8, 128, 64),
            (2, 32, 16),
        ],
    )
    def test_splits_stacked_weights_into_per_expert_w2(
        self, num_experts, hidden_size, intermediate_size
    ):
        """Verify down hook splits stacked weights into w2 per expert."""
        # Llama4 format: (E, I, H)
        stacked = torch.randn(num_experts, intermediate_size, hidden_size)
        state_dict = {"down_weight": stacked}
        w2_keys = [f"w2_{i}" for i in range(num_experts)]

        _bmm_moe_down_split_hook(
            state_dict,
            "",
            source_key="down_weight",
            w2_keys=w2_keys,
        )

        for i in range(num_experts):
            assert w2_keys[i] in state_dict
            # After transpose: (H, I)
            assert state_dict[w2_keys[i]].shape == (hidden_size, intermediate_size)

    def test_w2_content_matches_original_stacked(self):
        """Verify split w2 tensors match the original stacked content."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, intermediate_size, hidden_size)
        state_dict = {"down_weight": stacked.clone()}
        w2_keys = [f"w2_{i}" for i in range(num_experts)]

        _bmm_moe_down_split_hook(
            state_dict,
            "",
            source_key="down_weight",
            w2_keys=w2_keys,
        )

        for i in range(num_experts):
            expected_w2 = stacked[i].transpose(0, 1).contiguous()
            torch.testing.assert_close(state_dict[w2_keys[i]], expected_w2)

    def test_handles_missing_source_key(self):
        """Verify hook does nothing when source key is missing."""
        state_dict = {}

        _bmm_moe_down_split_hook(
            state_dict,
            "",
            source_key="missing_key",
            w2_keys=["w2"],
        )

        assert len(state_dict) == 0


class TestBmmMoeHooksIntegration:
    """Integration tests for BMM MoE hooks working together."""

    def test_full_checkpoint_loading_flow(self):
        """Test the full flow: split gate_up and down into per-expert weights."""
        num_experts = 4
        hidden_size = 64
        intermediate_size = 32

        # Simulate a checkpoint with stacked weights
        gate_up_stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        down_stacked = torch.randn(num_experts, intermediate_size, hidden_size)

        state_dict = {
            "gate_up_weight": gate_up_stacked.clone(),
            "down_weight": down_stacked.clone(),
        }

        w1_keys = [f"w1_{i}" for i in range(num_experts)]
        w2_keys = [f"w2_{i}" for i in range(num_experts)]
        w3_keys = [f"w3_{i}" for i in range(num_experts)]

        # Step 1: Split gate_up into w1 and w3
        _bmm_moe_gate_up_split_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            intermediate_size=intermediate_size,
            w1_keys=w1_keys,
            w3_keys=w3_keys,
        )

        # Step 2: Split down into w2
        _bmm_moe_down_split_hook(
            state_dict,
            "",
            source_key="down_weight",
            w2_keys=w2_keys,
        )

        # Verify: all per-expert weights present with correct shapes
        for i in range(num_experts):
            assert state_dict[w1_keys[i]].shape == (intermediate_size, hidden_size)
            assert state_dict[w2_keys[i]].shape == (hidden_size, intermediate_size)
            assert state_dict[w3_keys[i]].shape == (intermediate_size, hidden_size)
