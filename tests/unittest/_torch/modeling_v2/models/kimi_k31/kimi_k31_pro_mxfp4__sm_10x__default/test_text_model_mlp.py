# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 shared-expert and dense-layer MLP construction.

* shared-expert sharding and reduction follow the selected parallel mode;
* the shared expert's all-reduce is ordered before the join with the routed output;
* the first dense layer selects a block-aligned, node-local MLP TP group.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import allreduce_module
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
    K31SituMLP,
)
from tensorrt_llm._torch.utils import AuxStreamType


def _make_aux_stream_dict() -> dict[AuxStreamType, object]:
    return {
        AuxStreamType.Attention: object(),
        AuxStreamType.MoeShared: object(),
        AuxStreamType.MoeChunkingOverlap: object(),
        AuxStreamType.MoeBalancer: object(),
        AuxStreamType.MoeOutputMemset: object(),
    }


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=512,
        num_hidden_layers=4,
        first_k_dense_replace=1,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=256,
        num_shared_experts=2,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0,
        num_expert_group=1,
        topk_group=1,
    )


@pytest.mark.parametrize(
    "attention_dp,tp_size,rank,expected_shared_tp,expected_shared_rank,moe_backend",
    [
        (True, 8, 7, 1, 0, "TRTLLM"),
        (False, 1, 0, 1, 0, "TRTLLM"),
        (False, 8, 7, 8, 7, "TRTLLM"),
    ],
    ids=[
        "attention_dp",
        "single_rank",
        "direct_tp",
    ],
)
def test_kimi_k3_shared_expert_parallel_construction(
    monkeypatch,
    attention_dp,
    tp_size,
    rank,
    expected_shared_tp,
    expected_shared_rank,
    moe_backend,
):
    """Shared experts are replicated or sharded for the selected parallel mode."""
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
        modeling as modeling_kimi_linear,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _FakeAllReduce(nn.Module):
        def __init__(self, mapping=None, *args, **kwargs):
            super().__init__()
            # ``AllReduce`` keeps its mapping; the runtime reads the routed
            # all-reduce's to place the fused latent-MoE tail.
            self.mapping = mapping

    routed_kwargs = {}

    class _FakeRoutedExperts(nn.Module):
        """The routed experts: records what the runtime builds them with."""

        def __init__(self, model_config, mapping, **kwargs):
            super().__init__()
            routed_kwargs.update(kwargs, mapping=mapping)
            self.mapping = mapping
            self.initial_local_expert_ids = [0, 1, 2, 3]

    monkeypatch.setattr(modeling_kimi_linear, "K31TargetRoutedExperts", _FakeRoutedExperts)
    monkeypatch.setattr(distributed, "AllReduce", _FakeAllReduce)
    # The family projections build their all-reduce through the comm op's tp_allreduce.
    monkeypatch.setattr(allreduce_module, "AllReduce", _FakeAllReduce)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())

    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        tp_size=tp_size,
        enable_attention_dp=attention_dp,
    )
    model_config = ModelConfig(
        mapping=mapping,
        quant_config=QuantConfig(),
        moe_backend=moe_backend,
    )
    config = _runtime_config()
    aux_stream_dict = _make_aux_stream_dict()
    runtime = modeling_kimi_linear.K31MoE(
        model_config,
        config,
        layer_idx=1,
        aux_stream_dict=aux_stream_dict,
    )

    shared = runtime.shared_experts
    # Direct MoE TP reduces the routed latent partial in this module.
    assert runtime._reduce_routed_output is (not attention_dp and tp_size > 1)
    # The routed experts take the EP-only split of the TP ranks and the model's SiTU caps.
    assert (
        routed_kwargs["mapping"].moe_ep_size == tp_size
        and routed_kwargs["mapping"].moe_tp_size == 1
    )
    assert (routed_kwargs["gate_cap"], routed_kwargs["linear_cap"]) == (4.0, 25.0)
    # The chunk-overlap stream of a call split into chunks is the model's.
    assert routed_kwargs["aux_stream_dict"] is aux_stream_dict
    assert runtime.shared_expert_stream is aux_stream_dict[AuxStreamType.MoeShared]
    assert isinstance(shared, K31SituMLP)
    assert shared.gate_up_proj.tp_size == expected_shared_tp
    assert shared.gate_up_proj.tp_rank == expected_shared_rank
    assert shared.down_proj.tp_size == expected_shared_tp
    assert shared.down_proj.tp_rank == expected_shared_rank
    # The shared branch carries its own all-reduce whenever it is TP-sharded:
    # ``K31MoE`` passes ``reduce_output=use_shared_tp``, and the
    # forward comment is explicit that "the shared MLP includes its output
    # all-reduce on the auxiliary stream", joined before the separate routed
    # all-reduce. Only attention-DP (shared expert replicated at TP 1) and the
    # single-rank case leave it unreduced.
    assert shared.down_proj.reduce_output is (not attention_dp and tp_size > 1)
    local_intermediate = (
        config.moe_intermediate_size * config.num_shared_experts // expected_shared_tp
    )
    assert shared.gate_up_proj.weight.shape == (2 * local_intermediate, config.hidden_size)
    assert shared.down_proj.weight.shape == (config.hidden_size, local_intermediate)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "defer,multi_stream,is_compiling,has_aux_stream",
    [
        (False, True, False, True),
        (False, False, False, True),
    ],
    ids=["parallel", "serial"],
)
def test_kimi_k3_shared_allreduce_ordering(
    monkeypatch: pytest.MonkeyPatch,
    defer: bool,
    multi_stream: bool,
    is_compiling: bool,
    has_aux_stream: bool,
) -> None:
    """Check shared-allreduce dependencies and result propagation without CUDA work."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
        modeling as modeling_kimi_linear,
    )
    from tensorrt_llm._torch.distributed import AllReduceParams
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    calls = []
    current_stream = "main"
    aux_stream = object() if has_aux_stream else None
    parallel = multi_stream and has_aux_stream and not is_compiling

    class _Event:
        def __init__(self, name: str, enabled: bool) -> None:
            self.name = name
            self.enabled = enabled
            self.recorded = False

        def record(self) -> None:
            assert self.enabled
            assert not self.recorded
            self.recorded = True
            calls.append((f"{self.name}_record", current_stream))

        def wait(self) -> None:
            assert self.enabled
            assert self.recorded
            self.recorded = False
            calls.append((f"{self.name}_wait", current_stream))

    @contextmanager
    def use_stream(stream: object) -> Iterator[None]:
        nonlocal current_stream
        assert parallel and stream is aux_stream
        previous_stream = current_stream
        current_stream = "aux"
        try:
            yield
        finally:
            current_stream = previous_stream

    class _Projection(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            calls.append((self.name, current_stream))
            return value

    def routed_experts(value: torch.Tensor, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        calls.append(("routed_moe", current_stream))
        return value * 2

    def shared_allreduce(value: torch.Tensor) -> torch.Tensor:
        calls.append(("shared_allreduce", current_stream))
        return value + 5

    class _Shared(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = SimpleNamespace(all_reduce=shared_allreduce)

        def forward(
            self, value: torch.Tensor, final_all_reduce_params: AllReduceParams | None = None
        ) -> torch.Tensor:
            calls.append(("shared_local", current_stream))
            partial = value * 3
            if defer:
                assert final_all_reduce_params.enable_allreduce is False
                return partial
            assert final_all_reduce_params is None
            return self.down_proj.all_reduce(partial)

    runtime = modeling_kimi_linear.K31MoE.__new__(modeling_kimi_linear.K31MoE)
    nn.Module.__init__(runtime)
    runtime.gate = SimpleNamespace(compute_logits=lambda value: value)
    runtime.routed_experts = routed_experts
    runtime.routed_expert_down_proj = _Projection("routed_down")
    # The routed latent norm runs through the catalog RMSNorm op with the module's weight and epsilon.
    runtime.routed_expert_norm = SimpleNamespace(weight=None, variance_epsilon=0.0)

    def routed_norm(value: torch.Tensor, weight: object, eps: float) -> torch.Tensor:
        calls.append(("routed_norm", current_stream))
        return value

    monkeypatch.setattr(modeling_kimi_linear, "flashinfer_rmsnorm", routed_norm)
    runtime.routed_expert_up_proj = _Projection("routed_up")
    runtime.shared_experts = _Shared()
    runtime._reduce_routed_output = False
    runtime.shared_expert_stream = aux_stream
    runtime.moe_main_event = _Event("fork", parallel)
    runtime.moe_shared_event = _Event("shared_done", parallel)

    monkeypatch.setattr(torch.cuda, "stream", use_stream)
    # maybe_execute_in_parallel records the aux outputs on the calling stream
    # after the join; keep that query and lookup off the CUDA runtime too.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: pytest.fail("Events must be initialized once"))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: is_compiling)

    shared_stream = "aux" if parallel else "main"

    # Reusing the same owner must record fresh dependencies without replacing events.
    for _ in range(2):
        calls.clear()
        with with_multi_stream(multi_stream):
            output = runtime(torch.ones(2, 4))
        assert sum(name == "shared_allreduce" for name, _ in calls) == 1
        allreduce = calls.index(("shared_allreduce", shared_stream))
        assert calls.index(("shared_local", shared_stream)) < allreduce
        if parallel:
            # The final join must cover the shared all-reduce.
            completed = calls.index(("shared_done_record", "aux"))
            assert allreduce < completed < calls.index(("shared_done_wait", "main"))
        torch.testing.assert_close(output, torch.full((2, 4), 10.0))


@pytest.mark.parametrize(
    "attention_dp,tp_size,rank,gpus_per_node,intermediate_size,expected_tp_size,expected_tp_rank",
    [
        (True, 8, 7, 8, 516, 1, 0),
        (False, 8, 7, 8, 512, 8, 7),
        (False, 8, 7, 8, 516, 4, 3),
        (False, 8, 7, 8, 515, 1, 0),
        (False, 16, 15, 8, 512, 8, 7),
    ],
    ids=["attention_dp", "full_tp", "gcd_subgroup", "replicated", "single_node_cap"],
)
def test_kimi_k3_dense_layer_mlp_tp_group(
    monkeypatch,
    attention_dp,
    tp_size,
    rank,
    gpus_per_node,
    intermediate_size,
    expected_tp_size,
    expected_tp_rank,
):
    """The first dense layer selects a block-aligned, node-local MLP TP group."""
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
        modeling as modeling_kimi_linear,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig

    class _IdentityAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.aux_stream = kwargs["aux_stream"]

        def forward(self, hidden_states, attn_metadata):
            return hidden_states

    class _IdentityAllReduce(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states

    monkeypatch.setattr(modeling_kimi_linear, "K31KdaLayer", _IdentityAttention)
    monkeypatch.setattr(distributed, "AllReduce", _IdentityAllReduce)
    monkeypatch.setattr(allreduce_module, "AllReduce", _IdentityAllReduce)

    mapping = Mapping(
        world_size=tp_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=tp_size,
        enable_attention_dp=attention_dp,
    )
    model_config = ModelConfig(mapping=mapping, quant_config=QuantConfig())
    config = SimpleNamespace(
        hidden_size=512,
        intermediate_size=intermediate_size,
        num_experts=8,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        linear_attn_config={"kda_layers": [1], "full_attn_layers": []},
        rms_norm_eps=1e-5,
        attn_res_block_size=1,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    aux_stream_dict = _make_aux_stream_dict()
    layer = modeling_kimi_linear.K31DecoderLayer(
        model_config,
        config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
    )

    assert not layer.is_moe
    assert layer.linear_attn.aux_stream is aux_stream_dict[AuxStreamType.Attention]
    assert isinstance(layer.mlp, K31SituMLP)
    assert layer.mlp_tp_size == expected_tp_size
    assert layer.mlp.gate_up_proj.tp_size == expected_tp_size
    assert layer.mlp.gate_up_proj.tp_rank == expected_tp_rank
    assert layer.mlp.down_proj.tp_size == expected_tp_size
    assert layer.mlp.down_proj.tp_rank == expected_tp_rank
    assert layer.mlp.down_proj.reduce_output is (expected_tp_size > 1)
    local_intermediate = config.intermediate_size // expected_tp_size
    assert layer.mlp.gate_up_proj.weight.shape == (
        2 * local_intermediate,
        config.hidden_size,
    )
    assert layer.mlp.down_proj.weight.shape == (
        config.hidden_size,
        local_intermediate,
    )
