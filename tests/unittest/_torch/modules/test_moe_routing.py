import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fused_moe import (
    DefaultMoeRoutingMethod, LoadBalancedMoeRoutingMethod,
    RenormalizeMoeRoutingMethod, SparseMixerMoeRoutingMethod,
    StaticMoeRoutingMethod, create_renormalize_expert_load_balanced_logits)


# Test DefaultMoeRoutingMethod with different top_k values
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_default_moe_routing(top_k):
    routing = DefaultMoeRoutingMethod(top_k=top_k)
    assert routing.experts_per_token == top_k

    logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.2, 0.3]],
        dtype=torch.float32).cuda()
    indices, scales = routing.apply(logits)
    indices = indices.cpu()
    scales = scales.cpu()

    assert indices.shape == (3, top_k)
    assert scales.shape == (3, top_k)

    assert indices.dtype == torch.int32
    assert scales.dtype == torch.float32
    reference_indices = torch.tensor([[3, 2, 1], [0, 1, 2], [1, 3, 2]],
                                     dtype=torch.int32)
    reference_scales = F.softmax(logits, dim=1).cpu()

    # Check that the selected experts are the largest top_k values
    for i in range(top_k):
        assert indices[0, i] == reference_indices[0, i]
        assert indices[1, i] == reference_indices[1, i]
        assert indices[2, i] == reference_indices[2, i]

    for i in range(top_k):
        assert scales[0, i] == pytest.approx(
            reference_scales[0, reference_indices[0, i]])
        assert scales[1, i] == pytest.approx(
            reference_scales[1, reference_indices[1, i]])
        assert scales[2, i] == pytest.approx(
            reference_scales[2, reference_indices[2, i]])


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_renormalize_moe_routing(top_k):
    routing = RenormalizeMoeRoutingMethod(top_k=top_k)
    assert routing.experts_per_token == top_k

    logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.2, 0.3]],
        dtype=torch.float32).cuda()
    indices, scales = routing.apply(logits)
    assert indices.shape == (3, top_k)
    assert scales.shape == (3, top_k)

    assert indices.dtype == torch.int32
    assert scales.dtype == torch.float32

    reference_indices, reference_scales = DefaultMoeRoutingMethod(
        top_k=top_k).apply(logits)
    reference_scales /= reference_scales.sum(dim=1, keepdim=True)

    assert torch.equal(indices, reference_indices)
    assert torch.allclose(scales, reference_scales)


def gen_unique_logits(num_tokens, num_experts, dtype):
    unique_logits = torch.rand((num_tokens, num_experts), dtype=dtype)

    for i in range(unique_logits.size(0)):
        torch.manual_seed(42 * i)
        unique_row = torch.randperm(num_experts)
        unique_logits[i] = unique_row.to(dtype)

    return unique_logits.cuda()


@pytest.mark.parametrize("num_tokens", [30])
@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("num_experts", [8, 67, 128])
def test_customized_renormalize_moe_routing(num_tokens, top_k, num_experts,
                                            dtype):

    #Because the order of equal elements is unpredictable, we use unique data to prevent any ambiguity.
    router_logits = gen_unique_logits(num_tokens, num_experts, dtype)

    routing = RenormalizeMoeRoutingMethod(top_k=top_k,
                                          force_enable_pytorch_op=False)
    indices, scales = routing.apply(router_logits)

    ref_routing = RenormalizeMoeRoutingMethod(top_k=top_k,
                                              force_enable_pytorch_op=True)
    reference_indices, reference_scales = ref_routing.apply(router_logits)

    assert torch.equal(indices, reference_indices)
    assert torch.allclose(scales, reference_scales)


def test_sparse_mixer_reference():
    routing = SparseMixerMoeRoutingMethod(top_k=2, eps=0.2)
    assert routing.experts_per_token == 2

    # Test case 1: Equal logits for first two experts
    logits = torch.tensor([[1.0, 1.0, -float('inf'), -float('inf')],
                           [2.0, 0.0, -float('inf'), -float('inf')],
                           [0.0, 2.0, -float('inf'), -float('inf')],
                           [1.0, 1.0, 1.0, -float('inf')]],
                          dtype=torch.float32).cuda()
    indices, scales = routing.apply(logits.clone())
    indices = indices.cpu()
    scales = scales.cpu()

    assert indices.shape == (4, routing.experts_per_token)
    assert scales.shape == (4, routing.experts_per_token)
    assert indices.dtype == torch.int32
    assert scales.dtype == torch.float32

    # Test case 1: Equal logits for first two experts
    assert scales[0, 0] == pytest.approx(0.5)
    assert scales[0, 1] == pytest.approx(1.0)
    assert torch.all(indices[0, :] < 2)

    # Test case 2: First expert has higher logit
    assert scales[1, 0] == pytest.approx(1.0)
    assert scales[1, 1] == pytest.approx(1.0)
    assert indices[1, 0] == 0
    assert indices[1, 1] == 1

    # Test case 3: Second expert has higher logit
    assert scales[2, 0] == pytest.approx(1.0)
    assert scales[2, 1] == pytest.approx(1.0)
    assert indices[2, 0] == 1
    assert indices[2, 1] == 0

    # Test case 4: Equal logits for first three experts
    assert scales[3, 0] == pytest.approx(1 / 3)
    assert scales[3, 1] == pytest.approx(0.5)
    assert torch.all(indices[3, :] < 3)


def test_load_balanced_moe_routing():
    for k, tokens in [(2, 2), (2, 32), (3, 100)]:
        routing = LoadBalancedMoeRoutingMethod(top_k=k)
        assert routing.experts_per_token == k

        # Values don't matter for load balanced routing
        logits = torch.empty((tokens, 4), dtype=torch.float32).cuda()

        indices, scales = routing.apply(logits)
        assert indices.shape == (tokens, k)
        assert scales.shape == (tokens, k)
        assert indices.dtype == torch.int32
        assert scales.dtype == torch.float32

        indices_flat = indices.view(-1)
        assert torch.sum((indices_flat == 0).int()) == tokens * k // 4
        assert torch.sum((indices_flat == 1).int()) == tokens * k // 4
        assert torch.sum((indices_flat == 2).int()) == tokens * k // 4
        assert torch.sum((indices_flat == 3).int()) == tokens * k // 4


def test_static_moe_routing():
    routing = StaticMoeRoutingMethod(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32).cuda())
    with torch.device('cpu'):
        assert routing.experts_per_token == 4

        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                              dtype=torch.float32).cuda()
        indices, scales = routing.apply(logits)
        indices = indices.cpu()

        assert scales is None
        assert indices.shape == (2, 4)
        assert indices.dtype == torch.int32

        assert torch.equal(
            indices,
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32))

        routing = StaticMoeRoutingMethod(
            torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]],
                         dtype=torch.int32).cuda(),
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                         dtype=torch.float32).cuda())
        indices, scales = routing.apply(logits)
        scales = scales.cpu()

        assert scales is not None
        assert scales.shape == (2, 4)
        assert scales.dtype == torch.float32
        assert torch.equal(
            scales,
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                         dtype=torch.float32))


@pytest.mark.parametrize(
    "num_tokens,expected_assignments,description",
    [(3, [2, 2, 1, 1],
      "3 tokens - slight imbalance due to total work not divisible by EP size"),
     (4, [2, 2, 2, 2], "4 tokens - perfect balance"),
     (32, [16, 16, 16, 16], "32 tokens - large batch with perfect balance")])
def test_renormalize_expert_load_balanced_logits(num_tokens,
                                                 expected_assignments,
                                                 description):
    """Test GPU load balancing with RenormalizeMoeRoutingMethod across different token counts."""
    # Test parameters (consistent across all test cases)
    num_experts = 8
    experts_per_token = 2
    moe_ep_size = 4
    device = torch.device('cuda')

    # Generate expert load balanced logits using the utility function directly
    logits = create_renormalize_expert_load_balanced_logits(
        num_tokens=num_tokens,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        moe_ep_size=moe_ep_size,
        device=device,
        dtype=torch.float32)

    # Use RenormalizeMoeRoutingMethod to get expert assignments
    routing = RenormalizeMoeRoutingMethod(top_k=experts_per_token)
    indices, scales = routing.apply(logits)

    # Verify shapes
    assert indices.shape == (num_tokens, experts_per_token)
    assert scales.shape == (num_tokens, experts_per_token)

    # Count expert assignments per GPU
    # GPU 0: experts [0, 1], GPU 1: experts [2, 3], GPU 2: experts [4, 5], GPU 3: experts [6, 7]
    gpu_assignments = [0, 0, 0, 0]  # Count for each GPU
    experts_per_gpu = num_experts // moe_ep_size  # = 2

    indices_flat = indices.view(-1).cpu()
    for expert_idx in indices_flat:
        gpu_id = expert_idx.item() // experts_per_gpu
        gpu_assignments[gpu_id] += 1

    # Verify total assignments and expected distribution
    total_expected = num_tokens * experts_per_token
    assert sum(
        gpu_assignments
    ) == total_expected, f"Total assignments mismatch for {description}"

    # For cases where perfect balance isn't possible, check sorted equality
    # For perfect balance cases, check exact equality
    if len(set(expected_assignments)
           ) == 1:  # All values are the same (perfect balance)
        assert gpu_assignments == expected_assignments, f"Perfect balance expected for {description}"
    else:  # Slight imbalance expected
        assert sorted(gpu_assignments) == sorted(
            expected_assignments), f"Load balance failed for {description}"


if __name__ == '__main__':
    pytest.main()
