import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fused_moe import (DefaultMoeRoutingMethod,
                                                   LoadBalancedMoeRoutingMethod,
                                                   RenormalizeMoeRoutingMethod,
                                                   SparseMixerMoeRoutingMethod,
                                                   StaticMoeRoutingMethod)


# Test DefaultMoeRoutingMethod with different top_k values
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_default_moe_routing(top_k):
    routing = DefaultMoeRoutingMethod(top_k=top_k)
    assert routing.experts_per_token == top_k

    logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.2, 0.3]],
        dtype=torch.float32)
    indices, scales = routing.apply(logits)
    assert indices.shape == (3, top_k)
    assert scales.shape == (3, top_k)

    assert indices.dtype == torch.int32
    assert scales.dtype == torch.float32
    reference_indices = torch.tensor([[3, 2, 1], [0, 1, 2], [1, 3, 2]],
                                     dtype=torch.int32)
    reference_scales = F.softmax(logits, dim=1)

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
        dtype=torch.float32)
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


def test_sparse_mixer_reference():
    routing = SparseMixerMoeRoutingMethod(top_k=2, eps=0.2)
    assert routing.experts_per_token == 2

    # Test case 1: Equal logits for first two experts
    logits = torch.tensor([[1.0, 1.0, -float('inf'), -float('inf')],
                           [2.0, 0.0, -float('inf'), -float('inf')],
                           [0.0, 2.0, -float('inf'), -float('inf')],
                           [1.0, 1.0, 1.0, -float('inf')]],
                          dtype=torch.float32)
    indices, scales = routing.apply(logits.clone())

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
        logits = torch.empty((tokens, 4), dtype=torch.float32)

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
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32))
    assert routing.experts_per_token == 4

    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                          dtype=torch.float32)
    indices, scales = routing.apply(logits)
    assert scales is None
    assert indices.shape == (2, 4)
    assert indices.dtype == torch.int32

    assert torch.equal(
        indices, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32))

    routing = StaticMoeRoutingMethod(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32),
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                     dtype=torch.float32))
    indices, scales = routing.apply(logits)
    assert scales is not None
    assert scales.shape == (2, 4)
    assert scales.dtype == torch.float32
    assert torch.equal(
        scales,
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                     dtype=torch.float32))


if __name__ == '__main__':
    pytest.main()
