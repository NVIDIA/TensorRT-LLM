import os
import pickle
import sys

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (
    BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod,
    DeepSeekV4MoeRoutingMethod, DefaultMoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod, LoadBalancedMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod, RenormalizeMoeRoutingMethod,
    RenormalizeNaiveMoeRoutingMethod, SparseMixerMoeRoutingMethod,
    StaticMoeRoutingMethod, create_load_balanced_logits, create_moe)
from tensorrt_llm._torch.modules.fused_moe import routing as moe_routing
from tensorrt_llm._torch.modules.fused_moe.routing import \
    get_cached_perfect_router_logits
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping


def verify_load_balanced_logits(
    logits: torch.Tensor,
    routing_method: BaseMoeRoutingMethod,
    num_experts: int,
    moe_ep_size: int,
) -> tuple[bool, list[int]]:
    """
    Verify that the logits produce load balanced expert assignments.

    Args:
        logits: Router logits [num_tokens, num_experts]
        routing_method: The routing method to apply
        num_experts: Total number of experts
        moe_ep_size: Number of GPUs for expert parallelism

    Returns:
        Tuple of (is_balanced, gpu_counts):
            is_balanced: True if max diff between ranks is <= 1
            gpu_counts: List of expert assignments per GPU
    """
    indices, _ = routing_method.apply(logits)
    experts_per_gpu = num_experts // moe_ep_size

    gpu_counts = [0] * moe_ep_size
    for expert_idx in indices.view(-1).cpu().tolist():
        gpu_id = expert_idx // experts_per_gpu
        gpu_counts[gpu_id] += 1

    max_count = max(gpu_counts)
    min_count = min(gpu_counts)
    is_balanced = (max_count - min_count) <= 1

    return is_balanced, gpu_counts


def verify_flat_router_timeline_hotspot_balance(
    logits: torch.Tensor,
    routing_method: BaseMoeRoutingMethod,
    num_experts: int,
    moe_ep_size: int,
    ep_rank: int,
) -> tuple[bool, list[str]]:
    """Verify the order-insensitive no-hotspot receiver timeline for flat routers.

    Flat perfect-router logits intentionally give the selected experts the same
    score, so ``routing_method.apply`` may permute the top-k slot order for a
    token. The no-hotspot invariant can still be recovered from the final
    receiver-GPU multiset: token ``t`` on rank ``r`` should target the cyclic
    receiver set ``{(t * top_k + slot + r) % moe_ep_size}`` for all
    ``slot in [0, top_k)``.

    Returns:
        Tuple of ``(is_balanced, errors)`` where ``errors`` contains one message
        per token that violates the expected receiver set or GPU coverage.
    """
    indices, _ = routing_method.apply(logits)
    experts_per_gpu = num_experts // moe_ep_size
    token_receiver_gpus = (indices.to(torch.int64) //
                           experts_per_gpu).cpu().tolist()
    top_k = indices.shape[1]
    expected_gpu_coverage = min(top_k, moe_ep_size)

    errors = []
    for token_idx, receiver_gpus in enumerate(token_receiver_gpus):
        expected_receivers = sorted(
            ((token_idx * top_k) + slot_idx + ep_rank) % moe_ep_size
            for slot_idx in range(top_k))
        observed_receivers = sorted(receiver_gpus)
        if observed_receivers != expected_receivers:
            errors.append(
                f"rank {ep_rank}: token {token_idx} expected receiver GPUs "
                f"{expected_receivers}, got {observed_receivers} from "
                f"experts {indices[token_idx, :].cpu().tolist()}")

        observed_coverage = len(set(receiver_gpus))
        if observed_coverage != expected_gpu_coverage:
            errors.append(
                f"rank {ep_rank}: token {token_idx} expected coverage of "
                f"{expected_gpu_coverage} GPUs, got {receiver_gpus}")

    return not errors, errors


cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


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
@pytest.mark.parametrize("num_experts", [8, 67, 128, 512])
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

    # Compare top-k as a set: when router_logits contain ties (e.g. bf16 +
    # num_experts>256 collapses many randperm values to the same bf16 number),
    # the kernel and the reference may legitimately break ties in different
    # orders. Sort each row by indices before comparing, and align scales by
    # the same permutation.
    idx_sorted, perm = torch.sort(indices, dim=1)
    ref_idx_sorted, ref_perm = torch.sort(reference_indices, dim=1)
    assert torch.equal(idx_sorted, ref_idx_sorted)

    scales_sorted = torch.gather(scales, 1, perm)
    ref_scales_sorted = torch.gather(reference_scales, 1, ref_perm)
    assert torch.allclose(scales_sorted, ref_scales_sorted)


@pytest.mark.parametrize("routing_cls",
                         [DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod])
@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float32, torch.float16])
def test_customized_moe_routing_512_experts_topk10_middle_block(
        routing_cls, dtype):
    num_tokens = 4
    num_experts = 512
    top_k = 10
    router_logits = torch.full((num_tokens, num_experts),
                               -20,
                               dtype=torch.float32,
                               device="cuda")
    expected_topk_indices = torch.arange(256,
                                         256 + top_k,
                                         device="cuda",
                                         dtype=torch.int32)

    for token_idx in range(num_tokens):
        router_logits[token_idx, expected_topk_indices.long()] = torch.arange(
            top_k, 0, -1, dtype=torch.float32, device="cuda") + token_idx
    router_logits = router_logits.to(dtype)

    routing = routing_cls(top_k=top_k, force_enable_pytorch_op=False)
    indices, scales = routing.apply(router_logits)

    ref_routing = routing_cls(top_k=top_k, force_enable_pytorch_op=True)
    reference_indices, reference_scales = ref_routing.apply(router_logits)

    assert torch.equal(indices, reference_indices)
    assert torch.allclose(scales, reference_scales, atol=1e-3, rtol=1e-3)


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


def _make_v3_routing(top_k, n_group, topk_group, num_experts):
    """Build a DSv3 routing method (bias stored on routing_impl)."""
    bias = torch.zeros(num_experts, dtype=torch.float32)
    return DeepSeekV3MoeRoutingMethod(
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=False,
    )


def _make_v4_routing(top_k, n_group, topk_group, num_experts, is_hashed):
    """Build a DSv4 routing method.

    ``is_hashed=True`` mirrors ``DeepseekV4Gate`` hashed gates, where the
    bias callable returns ``None``. ``is_hashed=False`` returns a zero-bias
    tensor, mirroring the dense gate path.
    """
    if is_hashed:
        bias_callable = lambda: None  # noqa: E731
    else:
        bias = torch.zeros(num_experts, dtype=torch.float32)
        bias_callable = lambda: bias  # noqa: E731
    return DeepSeekV4MoeRoutingMethod(
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=bias_callable,
        callable_tid2eid=lambda: torch.zeros(0, dtype=torch.int32),
        is_hashed=is_hashed,
    )


# DSv4-Pro production: num_experts_per_tok=6, n_routed_experts=384, n_group=8,
# topk_group=4 (verified against the deployed checkpoint config). n_group and
# topk_group are not explicitly in the HF config; they fall back to the DSv3
# defaults that ``DeepseekV4Gate`` consumes via ``config.n_group`` /
# ``config.topk_group``.
_DSV4_PRO_TOP_K = 6
_DSV4_PRO_N_GROUP = 8
_DSV4_PRO_TOPK_GROUP = 4


@pytest.mark.parametrize(
    "routing_cls", [DeepSeekV3MoeRoutingMethod, DeepSeekV4MoeRoutingMethod])
def test_grouped_routing_exposes_n_group_topk_group(routing_cls):
    """Unit regression: V3 and V4 routing both expose n_group/topk_group.

    The DeepSeekV3 perfect-router planner reads
    ``routing_method.n_group`` and ``routing_method.topk_group`` directly.
    DeepSeekV3MoeRoutingMethod nests the settings on ``routing_impl`` and
    exposes them via @property; DeepSeekV4MoeRoutingMethod stores them
    directly on the instance. Both layouts must surface the same values.
    """
    n_group, topk_group = _DSV4_PRO_N_GROUP, _DSV4_PRO_TOPK_GROUP
    num_experts = 64
    if routing_cls is DeepSeekV4MoeRoutingMethod:
        routing = _make_v4_routing(top_k=_DSV4_PRO_TOP_K,
                                   n_group=n_group,
                                   topk_group=topk_group,
                                   num_experts=num_experts,
                                   is_hashed=False)
    else:
        routing = _make_v3_routing(top_k=_DSV4_PRO_TOP_K,
                                   n_group=n_group,
                                   topk_group=topk_group,
                                   num_experts=num_experts)

    assert routing.n_group == n_group
    assert routing.topk_group == topk_group


@pytest.mark.parametrize(
    "shape",
    [
        # (num_experts, moe_ep_size) — small case for CI speed, plus the
        # production DSv4-Pro shape (DEP=16 disagg-serving benchmark).
        pytest.param((64, 1), id="small"),
        pytest.param((384, 16), id="dsv4_pro_dep16"),
    ],
)
@pytest.mark.parametrize(
    "routing_name",
    ["v3", "v4_dense", "v4_hashed"],
)
def test_perfect_router_get_cached_logits_e2e(routing_name, shape):
    """End-to-end regression: synthesize perfect-router logits on CPU.

    Exercises the full pipeline through ``get_cached_perfect_router_logits``
    at ``top_k=6`` (DSv4-Pro ``num_experts_per_tok``):
      - Group-aware planner reads ``n_group`` / ``topk_group`` correctly
        (V3 nests them on ``routing_impl``, V4 stores them directly).
      - ``_force_zero_routing_bias_in_place`` handles both Tensor and
        ``None`` bias values. DSv4 hashed gates return ``None`` from their
        bias callable, which previously crashed with ``zeros_like(None)``.
      - Production DSv4-Pro shape (384 experts, EP=16) is covered alongside
        a small case so CI remains fast while still exercising the
        DEP=16 disagg-serving topology.
    """
    num_experts, moe_ep_size = shape
    n_group, topk_group = _DSV4_PRO_N_GROUP, _DSV4_PRO_TOPK_GROUP
    top_k = _DSV4_PRO_TOP_K
    num_tokens = 16
    ep_rank = 0
    dtype = torch.float32
    device = torch.device("cpu")

    if routing_name == "v3":
        routing = _make_v3_routing(top_k=top_k,
                                   n_group=n_group,
                                   topk_group=topk_group,
                                   num_experts=num_experts)
    elif routing_name == "v4_dense":
        routing = _make_v4_routing(top_k=top_k,
                                   n_group=n_group,
                                   topk_group=topk_group,
                                   num_experts=num_experts,
                                   is_hashed=False)
    else:  # v4_hashed
        routing = _make_v4_routing(top_k=top_k,
                                   n_group=n_group,
                                   topk_group=topk_group,
                                   num_experts=num_experts,
                                   is_hashed=True)

    # Isolate this case from the module-level cache populated by other tests.
    moe_routing._PERFECT_ROUTER_LOGITS_CACHE.clear()

    logits = get_cached_perfect_router_logits(num_tokens=num_tokens,
                                              num_experts=num_experts,
                                              experts_per_token=top_k,
                                              moe_ep_size=moe_ep_size,
                                              ep_rank=ep_rank,
                                              device=device,
                                              dtype=dtype,
                                              routing_method=routing)

    assert logits.shape == (num_tokens, num_experts)
    assert logits.dtype == dtype
    assert logits.device.type == "cpu"
    # Cache hit on the second call must not crash either (covers the case
    # where _force_zero_routing_bias_in_place runs again on an already-prepared
    # routing method).
    logits_2 = get_cached_perfect_router_logits(num_tokens=num_tokens,
                                                num_experts=num_experts,
                                                experts_per_token=top_k,
                                                moe_ep_size=moe_ep_size,
                                                ep_rank=ep_rank,
                                                device=device,
                                                dtype=dtype,
                                                routing_method=routing)
    assert torch.equal(logits, logits_2)


# -----------------------------------------------------------------
# Unified routing-method builder. Covers every RoutingMethodType in
# tensorrt_llm._torch.modules.fused_moe.routing:
#   Default, Renormalize, RenormalizeNaive, Llama4, MiniMax2, DeepSeekV3.
# Bias-aware routers seed a non-zero bias so tests can verify the
# perfect-router path resets it to zeros.
# -----------------------------------------------------------------


def _build_routing_method_for_perfect_router(routing_name,
                                             top_k,
                                             num_experts,
                                             device,
                                             n_group=1,
                                             topk_group=1):
    if routing_name == "Default":
        return DefaultMoeRoutingMethod(top_k=top_k)
    if routing_name == "Renormalize":
        return RenormalizeMoeRoutingMethod(top_k=top_k)
    if routing_name == "RenormalizeNaive":
        return RenormalizeNaiveMoeRoutingMethod(top_k=top_k)
    if routing_name == "Llama4":
        return Llama4RenormalizeMoeRoutingMethod(top_k=top_k)
    if routing_name == "MiniMax2":
        bias = torch.arange(1,
                            num_experts + 1,
                            device=device,
                            dtype=torch.float32)
        return MiniMaxM2MoeRoutingMethod(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: bias)
    if routing_name == "DeepSeekV3":
        bias = torch.arange(1,
                            num_experts + 1,
                            device=device,
                            dtype=torch.float32)
        return DeepSeekV3MoeRoutingMethod(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=1.0,
            callable_e_score_correction_bias=lambda: bias,
            is_fused=False)
    raise ValueError(f"Unknown routing method: {routing_name}")


# (id, routing_name, num_experts, experts_per_token, has_bias, extra)
_ROUTING_CONFIGS = [
    ("default_8e_k2", "Default", 8, 2, False, {}),
    ("renormalize_8e_k2", "Renormalize", 8, 2, False, {}),
    ("renormalize_16e_k4", "Renormalize", 16, 4, False, {}),
    ("renormalize_64e_k8", "Renormalize", 64, 8, False, {}),
    ("renormalize_naive_8e_k2", "RenormalizeNaive", 8, 2, False, {}),
    ("llama4_8e_k2", "Llama4", 8, 2, False, {}),
    ("minimax2_8e_k2", "MiniMax2", 8, 2, True, {}),
    (
        "deepseekv3_64e_k8_g8_tg4",
        "DeepSeekV3",
        64,
        8,
        True,
        {
            "n_group": 8,
            "topk_group": 4,
            "check_group": True,
            # Group constraint (topk_group=4) conflicts with per-gpu balance
            # when moe_ep_size >= n_group, so cap the EP size axis.
            "max_moe_ep_size": 4,
        }),
]


# num_tokens axis mixes typical values (1, 16, 100) with prime/odd
# counts (3, 7) to exercise edge cases previously covered by
# test_edge_case_token_counts.
@pytest.mark.parametrize("config",
                         _ROUTING_CONFIGS,
                         ids=[c[0] for c in _ROUTING_CONFIGS])
@pytest.mark.parametrize("num_tokens", [1, 3, 7, 16, 100])
@pytest.mark.parametrize("moe_ep_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
def test_load_balanced_logits(config, num_tokens, moe_ep_size, dtype):
    """Unified load-balance test covering all supported routing methods.

    Subsumes the previous per-routing-method, dtype sweep, configuration
    sweep, and edge-case token-count tests. Each parametrized case verifies:
      1. Returned logits have the requested dtype.
      2. For bias-aware routers, the bias callable is reset to zeros.
      3. Resulting expert assignments are load balanced across GPUs
         (max-min GPU count difference <= 1).
      4. Total number of assignments equals num_tokens * experts_per_token.
      5. For DeepSeekV3, the group constraint (<= topk_group distinct
         groups per token) is respected.
    """
    (config_id, routing_name, num_experts, experts_per_token, has_bias,
     extra) = config
    device = torch.device('cuda')

    if num_experts % moe_ep_size != 0:
        pytest.skip(
            f"num_experts ({num_experts}) not divisible by moe_ep_size ({moe_ep_size})"
        )

    max_moe_ep_size = extra.get("max_moe_ep_size")
    if max_moe_ep_size is not None and moe_ep_size > max_moe_ep_size:
        pytest.skip(
            f"{config_id}: moe_ep_size ({moe_ep_size}) exceeds supported "
            f"cap ({max_moe_ep_size}) for this routing method")

    n_group = extra.get("n_group", 1)
    topk_group = extra.get("topk_group", 1)
    routing = _build_routing_method_for_perfect_router(routing_name,
                                                       experts_per_token,
                                                       num_experts,
                                                       device,
                                                       n_group=n_group,
                                                       topk_group=topk_group)

    logits = create_load_balanced_logits(num_tokens=num_tokens,
                                         num_experts=num_experts,
                                         experts_per_token=experts_per_token,
                                         moe_ep_size=moe_ep_size,
                                         ep_rank=0,
                                         device=device,
                                         dtype=dtype,
                                         routing_method=routing)

    assert logits.dtype == dtype, (
        f"{config_id}: expected dtype {dtype}, got {logits.dtype}")

    if has_bias:
        assert torch.count_nonzero(routing.e_score_correction_bias) == 0, (
            f"{config_id}: bias was not reset to zero")

    is_balanced, gpu_counts = verify_load_balanced_logits(
        logits, routing, num_experts, moe_ep_size)

    assert is_balanced, (
        f"{config_id} not load balanced: gpu_counts={gpu_counts}")
    assert sum(gpu_counts) == num_tokens * experts_per_token, (
        f"{config_id}: total assignments mismatch "
        f"({sum(gpu_counts)} != {num_tokens * experts_per_token})")

    if extra.get("check_group"):
        indices, _ = routing.apply(logits)
        experts_per_group = num_experts // extra["n_group"]
        indices_2d = indices.view(num_tokens, experts_per_token)
        groups_2d = indices_2d // experts_per_group
        group_membership = F.one_hot(groups_2d.to(torch.int64),
                                     num_classes=extra["n_group"])
        per_token_group_count = group_membership.amax(dim=1).sum(dim=1)
        max_groups = int(per_token_group_count.max().item())
        assert max_groups <= extra["topk_group"], (
            f"{config_id}: some token selected experts from "
            f"{max_groups} groups (max allowed: {extra['topk_group']})")


# -----------------------------------------------------------------
# Multi-GPU load balance test driven by ENABLE_PERFECT_ROUTER.
#
# Each MPI worker sets ENABLE_PERFECT_ROUTER=1, builds a real CUTLASS
# MoE in DEP/TEP mode, runs a forward pass (internally the MoE swaps
# the input router logits for the cached load-balanced logits), then
# verifies that those cached logits indeed produce balanced expert
# dispatch across the EP ranks.
# -----------------------------------------------------------------


def _mapping_for_parallel_mode(world_size, parallel_mode):
    if parallel_mode == "DEP":
        enable_attention_dp = True
    elif parallel_mode == "TEP":
        enable_attention_dp = False
    else:
        raise ValueError(f"Unknown parallel_mode: {parallel_mode}")
    return Mapping(world_size=world_size,
                   tp_size=world_size,
                   moe_ep_size=world_size,
                   moe_tp_size=1,
                   enable_attention_dp=enable_attention_dp)


def _perfect_router_worker(parallel_mode, routing_name, num_tokens, dtype,
                           world_size):
    """Worker body executed in each MPI rank for the perfect-router test."""
    # Enable the perfect router path before any MoE module is constructed.
    os.environ["ENABLE_PERFECT_ROUTER"] = "1"

    # Local import inside the worker so module-level _PERFECT_ROUTER_LOGITS_CACHE
    # is the copy living in the child process. Clear any stale state.
    from tensorrt_llm._torch.modules.fused_moe import routing as moe_routing
    moe_routing._PERFECT_ROUTER_LOGITS_CACHE.clear()

    spec = _PERFECT_ROUTER_ROUTING_SPECS[routing_name]
    num_experts = spec["num_experts"]
    top_k = spec["top_k"]
    n_group = spec["n_group"]
    topk_group = spec["topk_group"]
    hidden_size = 256
    intermediate_size = 256

    mapping = _mapping_for_parallel_mode(world_size, parallel_mode)
    mapping.rank = mpi_rank()
    torch.cuda.set_device(mapping.rank)

    device = torch.device(f"cuda:{mapping.rank}")
    with torch.device(device):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        routing_method = _build_routing_method_for_perfect_router(
            routing_name,
            top_k,
            num_experts,
            device,
            n_group=n_group,
            topk_group=topk_group)

        # Minimal unquantized weights expected by CutlassFusedMoE.
        weights = {}
        for expert_id in range(num_experts):
            weights[f"{expert_id}.w1.weight"] = torch.randn(
                (intermediate_size, hidden_size), dtype=dtype, device=device)
            weights[f"{expert_id}.w2.weight"] = torch.randn(
                (hidden_size, intermediate_size), dtype=dtype, device=device)
            weights[f"{expert_id}.w3.weight"] = torch.randn(
                (intermediate_size, hidden_size), dtype=dtype, device=device)

        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        # Simulate per-rank token-count skew so each rank exercises a
        # different cache key (num_tokens, routing_method, dtype, ep_size).
        # DEP processes its own share of tokens per rank; TEP replicates
        # input across ranks so keep a uniform list there.
        if parallel_mode == "DEP":
            all_rank_num_tokens = [
                num_tokens + i for i in range(mapping.world_size)
            ]
        else:
            all_rank_num_tokens = [num_tokens] * mapping.world_size
        my_num_tokens = all_rank_num_tokens[mapping.rank]

        model_config = ModelConfig(pretrained_config=pretrained_config,
                                   mapping=mapping,
                                   moe_backend="CUTLASS",
                                   max_num_tokens=max(256,
                                                      max(all_rank_num_tokens)))

        fused_moe = create_moe(routing_method=routing_method,
                               reduce_results=True,
                               model_config=model_config)
        fused_moe.load_weights([weights])
        if hasattr(fused_moe, "post_load_weights"):
            fused_moe.post_load_weights()
        fused_moe.cuda(device)

        assert getattr(fused_moe, "_enable_perfect_router", False), (
            f"rank {mapping.rank}: ENABLE_PERFECT_ROUTER=1 was set but "
            "fused_moe did not enable the perfect router path")

        x = torch.randn((my_num_tokens, hidden_size),
                        dtype=dtype,
                        device=device)
        dummy_logits = torch.randn((my_num_tokens, num_experts),
                                   dtype=dtype,
                                   device=device)

        with torch.inference_mode():
            _ = fused_moe.forward(x,
                                  dummy_logits,
                                  all_rank_num_tokens=all_rank_num_tokens)
        torch.cuda.synchronize()

        # The MoE module routes on logits fetched from this cache; fetching
        # them here and verifying balance mirrors the production routing path.
        # Use mapping.rank as ep_rank to match what the MoE module uses.
        perfect_logits = get_cached_perfect_router_logits(
            num_tokens=my_num_tokens,
            num_experts=num_experts,
            experts_per_token=top_k,
            moe_ep_size=mapping.moe_ep_size,
            ep_rank=mapping.rank,
            device=device,
            dtype=dummy_logits.dtype,
            routing_method=routing_method)

        is_balanced, gpu_counts = verify_load_balanced_logits(
            perfect_logits, routing_method, num_experts, mapping.moe_ep_size)
        assert is_balanced, (
            f"rank {mapping.rank}: cached perfect router logits are not "
            f"load balanced across moe_ep_size={mapping.moe_ep_size}: "
            f"gpu_counts={gpu_counts}")
        assert sum(gpu_counts) == my_num_tokens * top_k, (
            f"rank {mapping.rank}: total assignments mismatch "
            f"({sum(gpu_counts)} != {my_num_tokens * top_k})")

        # Grouped routing methods such as DeepSeekV3 can legitimately perturb
        # the flat-router timeline while still preserving the batch-level load
        # balance checks above, so only run the dedicated hotspot helper on
        # flat routers.
        if n_group == 1:
            is_hotspot_balanced, hotspot_errors = (
                verify_flat_router_timeline_hotspot_balance(
                    perfect_logits,
                    routing_method,
                    num_experts,
                    mapping.moe_ep_size,
                    mapping.rank,
                ))
            assert is_hotspot_balanced, hotspot_errors[0]


# For DeepSeekV3 the per-GPU load balance solver requires
# moe_ep_size < n_group, so use a larger expert count with a matching
# group configuration. Other routing methods have no such constraint.
_PERFECT_ROUTER_ROUTING_SPECS = {
    "Default": dict(num_experts=8, top_k=2, n_group=1, topk_group=1),
    "Renormalize": dict(num_experts=8, top_k=2, n_group=1, topk_group=1),
    "RenormalizeNaive": dict(num_experts=8, top_k=2, n_group=1, topk_group=1),
    "Llama4": dict(num_experts=8, top_k=2, n_group=1, topk_group=1),
    "MiniMax2": dict(num_experts=8, top_k=2, n_group=1, topk_group=1),
    "DeepSeekV3": dict(num_experts=64, top_k=8, n_group=8, topk_group=4),
}
_PERFECT_ROUTER_ROUTING_NAMES = list(_PERFECT_ROUTER_ROUTING_SPECS)


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize("parallel_mode", ["DEP", "TEP"])
@pytest.mark.parametrize("routing_name", _PERFECT_ROUTER_ROUTING_NAMES)
@pytest.mark.parametrize("num_tokens", [8, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_perfect_router_load_balanced_multi_gpu(parallel_mode, routing_name,
                                                num_tokens, dtype):
    """Verify ENABLE_PERFECT_ROUTER produces balanced EP dispatch on 4 GPUs.

    Covers both DEP (attention DP + MoE EP) and TEP (attention TP + MoE EP)
    parallel modes against the CUTLASS MoE backend. For every parametrized
    combination each of the 4 MPI ranks:
      1. Sets ENABLE_PERFECT_ROUTER=1 before touching any MoE module.
      2. Builds an unquantized CUTLASS MoE with moe_ep_size=world_size=4.
      3. Runs a forward pass so the MoE module's perfect-router path exercises
         the cache for the given (num_tokens, routing_method, dtype, ep_size).
      4. Re-fetches the cached logits and asserts that, when routed through
         ``routing_method``, their expert assignments are load balanced
         (max-min GPU count difference <= 1) across ``moe_ep_size`` ranks.
      5. For flat routers, verifies the final receiver-GPU coverage without
         depending on a stable intra-token top-k order.
    """
    world_size = 4
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            _perfect_router_worker,
            *zip(*[(parallel_mode, routing_name, num_tokens, dtype,
                    world_size)] * world_size),
        )
        for r in results:
            assert r is None


if __name__ == '__main__':
    pytest.main()
