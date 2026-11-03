# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP must reconstruct each expert before its nonlinear post-expert norm."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE


@pytest.mark.parametrize(
    "tp,ep,attention_dp,supported",
    [
        (1, 1, False, True),
        (4, 4, False, True),
        (4, 4, True, True),
        (4, 1, False, True),
        (8, 1, False, True),
        (16, 1, False, True),
        (4, 1, True, False),
        (4, 2, False, False),
    ],
)
def test_parallelism_admission(monkeypatch, tp, ep, attention_dp, supported):
    from tensorrt_llm._torch.models import modeling_musespark_open
    from tensorrt_llm.mapping import Mapping

    class Admitted(Exception):
        pass

    def stop_before_weights(*args, **kwargs):
        raise Admitted

    monkeypatch.setattr(modeling_musespark_open, "Linear", stop_before_weights)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            hidden_size=16,
            expert_hidden_size=8,
            num_experts=16,
            moe_topk=8,
            metap_multiplier=1.0,
        ),
        mapping=Mapping(
            world_size=tp,
            tp_size=tp,
            moe_ep_size=ep,
            enable_attention_dp=attention_dp,
        ),
    )
    if supported:
        with pytest.raises(Admitted):
            MuseSparkOpenMoE(config, layer_idx=0)
    else:
        with pytest.raises(ValueError, match="MuseSparkOpen requires"):
            MuseSparkOpenMoE(config, layer_idx=0)


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tp_finalize_matches_unsharded_expert_math(tp_size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    tokens, top_k, hidden, intermediate = 3, 2, 16, 64
    # FC2 is row parallel: its input and contraction dimension are sharded.
    activation = torch.randn(tokens * top_k, intermediate, device=device)
    weight = torch.randn(tokens * top_k, intermediate, hidden, device=device)
    shards = [
        torch.einsum("si,sih->sh", a, w)
        for a, w in zip(activation.chunk(tp_size, 1), weight.chunk(tp_size, 1))
    ]
    complete = torch.einsum("si,sih->sh", activation, weight)
    gain = torch.randn(hidden, device=device) * 0.2
    routing = torch.rand(tokens, top_k, device=device)
    scaled = complete.double() * (1 + gain.double())
    normed = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + 1e-6)
    expected = (normed.reshape(tokens, top_k, hidden) * routing.double()[..., None]).sum(1)

    for rank in range(tp_size):
        mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
        mod.mapping = SimpleNamespace(moe_tp_size=tp_size)
        mod.expert_hidden_size = hidden
        mod.top_k = top_k
        mod.post_expert_norm_weight = gain
        mod.post_expert_norm_eps = 1e-6
        mod.nvtx = lambda _: nullcontext()
        # Every rank may have a different GEMM permutation and padded columns.
        index = torch.randperm(tokens * top_k, device=device).int()
        scratch = torch.full((tokens * top_k, hidden + 8), float("nan"), device=device)
        scratch[index.long(), :hidden] = shards[rank]
        calls = []

        def reduce(rows):
            torch.testing.assert_close(rows, shards[rank])
            assert rows.dtype == torch.float32
            calls.append(rows)
            return torch.stack(shards).sum(0)

        mod.all_reduce = reduce
        actual = mod.finalize(
            gemm2_output=scratch,
            expanded_idx_to_permuted_idx=index,
            token_selected_slots=None,
            routing_weights=routing,
            num_tokens=tokens,
        )
        assert len(calls) == 1
        torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)

    # The former EP ordering applied to TP partials computes a different model.
    partial_scaled = torch.stack(shards).double() * (1 + gain.double())
    partial_normed = partial_scaled * torch.rsqrt(
        partial_scaled.square().mean(-1, keepdim=True) + 1e-6
    )
    wrong = (partial_normed.sum(0).reshape(tokens, top_k, hidden) * routing[..., None]).sum(1)
    assert not torch.allclose(wrong, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("tokens", [1, 16, 128])
@torch.inference_mode()
def test_tp_forward_distributed_and_graph(tokens):
    """Real model finalize/tail and NCCL, with rank-local FC2 partials.

    Launch under MPI with 4, 8 or 16 ranks. Synthetic expert outputs isolate
    the reduction contract from quantized GEMM rounding and weight loading.
    """
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    world = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if world not in (4, 8, 16):
        pytest.skip("Launch with 4, 8 or 16 MPI ranks")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    hidden, top_k = 4096, 8
    torch.manual_seed(23)
    complete = torch.randn(tokens * top_k, hidden, device="cuda")
    # Opposing rank contributions exercise cancellation before normalization.
    partial = complete / world + (1 if rank % 2 else -1) * 4
    gain = torch.randn(hidden, device="cuda") * 0.1
    routing = torch.softmax(torch.randn(tokens, top_k, device="cuda"), -1)
    torch.manual_seed(31 + rank)
    index = torch.randperm(tokens * top_k, device="cuda").int()
    scratch = torch.empty_like(partial)
    scratch[index.long()] = partial

    mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(mod)
    mod.mapping = Mapping(
        world_size=world,
        rank=rank,
        tp_size=world,
        moe_ep_size=1,
        gpus_per_node=torch.cuda.device_count(),
    )
    mod.expert_hidden_size = hidden
    mod.top_k = top_k
    mod.post_expert_norm_weight = gain
    mod.post_expert_norm_eps = 1e-6
    mod.all_reduce = AllReduce(mapping=mod.mapping, strategy=AllReduceStrategy.NCCL)
    mod.nvtx = lambda _: nullcontext()
    mod.router_logits = lambda x: x
    mod.pooler = lambda x: x
    mod.pre_expert_norm = lambda x: x
    mod.unpooler = lambda x: x

    def experts(*args, **kwargs):
        # With the finalize registered, ConfigurableMoE runs it on the backend's
        # unfinalized triple and returns the dense result.
        return mod.finalize(
            gemm2_output=scratch,
            expanded_idx_to_permuted_idx=index,
            token_selected_slots=None,
            routing_weights=routing,
            num_tokens=tokens,
        )

    mod.experts = experts
    x = torch.zeros(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    scaled = complete.double() * (1 + gain.double())
    rows = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + 1e-6)
    expected = (rows.reshape(tokens, top_k, hidden) * routing.double()[..., None]).sum(1)
    for _ in range(3):
        actual = mod.forward(x, None)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.005, atol=0.005)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mod.forward(x, None)
    original = captured.clone()
    # Uniform scaling nearly cancels through RMSNorm and would not detect a
    # stale replay. Perturb one feature instead.
    scratch[:, 0].add_(1.25)
    graph.replay()
    eager = mod.forward(x, None)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    assert not torch.allclose(captured, original)


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "nvfp4"])
@pytest.mark.parametrize("tokens", [1, 16])
@torch.inference_mode()
def test_tp_backend_matches_complete_experts(quantized, tokens):
    """Production loader, GEMMs and finalize versus complete expert weights."""
    from mpi4py import MPI
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_musespark_open import (
        MuseSparkOpenRoutingMethod,
        post_expert_norm_combine,
    )
    from tensorrt_llm._torch.moe.fused_moe import create_moe
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    world, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    if world not in (4, 8, 16):
        pytest.skip("Launch with 4, 8 or 16 MPI ranks")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(101)
    hidden, intermediate, experts, top_k = 4096, 4096, 16, 8
    weights = {}
    for expert in range(experts):
        for name, rows, cols in (
            ("w1", intermediate, hidden),
            ("w3", intermediate, hidden),
            ("w2", hidden, intermediate),
        ):
            key = f"{expert}.{name}"
            if quantized:
                weights[key + ".weight"] = torch.randint(
                    0, 256, (rows, cols // 2), dtype=torch.uint8
                )
                weights[key + ".weight_scale"] = torch.full(
                    (rows, cols // 16), 0.5, dtype=torch.float8_e4m3fn
                )
                weights[key + ".weight_scale_2"] = torch.tensor(0.005)
                weights[key + ".input_scale"] = torch.tensor(1.0)
            else:
                weights[key + ".weight"] = (torch.randn(rows, cols) * 0.01).to(torch.bfloat16)
    bias = torch.zeros(experts, device="cuda")

    def make(mapping):
        config = ModelConfig(
            pretrained_config=PretrainedConfig(
                hidden_size=hidden,
                intermediate_size=intermediate,
                num_experts=experts,
                torch_dtype=torch.bfloat16,
            ),
            mapping=mapping,
            moe_backend="TRTLLM",
        )
        routing = MuseSparkOpenRoutingMethod(
            top_k=top_k,
            num_experts=experts,
            callable_e_score_correction_bias=lambda: bias,
            renorm_eps=1e-15,
        )
        moe = create_moe(
            routing_method=routing,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            dtype=torch.bfloat16,
            reduce_results=False,
            model_config=config,
            override_quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4) if quantized else None,
            layer_idx=0,
        ).cuda()
        moe.load_weights([weights])
        moe.post_load_weights()
        return moe

    whole = make(Mapping())
    mapping = Mapping(
        world_size=world,
        rank=rank,
        tp_size=world,
        moe_ep_size=1,
        gpus_per_node=torch.cuda.device_count(),
    )
    shard = make(mapping)
    torch.manual_seed(202)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(tokens, experts, device="cuda")
    gain = torch.randn(hidden, device="cuda") * 0.1
    complete, _, complete_index = whole(x, logits, do_finalize=False)
    _, routing = whole.routing_method.apply(logits)
    expected = post_expert_norm_combine(
        complete,
        complete_index,
        routing,
        gain,
        1e-6,
        num_tokens=tokens,
        top_k=top_k,
        hidden_size=hidden,
    )
    partial, _, index = shard(x, logits, do_finalize=False)
    mod = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(mod)
    mod.mapping = mapping
    mod.expert_hidden_size, mod.top_k = hidden, top_k
    mod.post_expert_norm_weight, mod.post_expert_norm_eps = gain, 1e-6
    mod.all_reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
    mod.nvtx = lambda _: nullcontext()
    actual = mod.finalize(
        gemm2_output=partial,
        expanded_idx_to_permuted_idx=index,
        token_selected_slots=None,
        routing_weights=routing,
        num_tokens=tokens,
    )
    relative_l2 = (actual - expected).norm() / expected.norm()
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    print(
        f"TP{world} rank={rank} quantized={quantized} tokens={tokens} "
        f"relative_l2={relative_l2.item():.6g} cosine={cosine.item():.8f}"
    )
    assert torch.isfinite(actual).all()
    assert relative_l2 < 0.03
    assert cosine > 0.999
