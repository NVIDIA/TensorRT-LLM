# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 routed experts (``ops/moe/routed_experts``) against the engine's ConfigurableMoE.

The engine reference is ``create_moe`` on the TRTLLM-Gen W4A8_MXFP4_MXFP8 backend with DeepSeek-V3 one-group routing
and SiTU, with the same packed MXFP4 weights (loaded through the engine's packed-expert loader, copied into the family
module). Outputs are compared bit for bit, for calls in one chunk and for calls split into chunks by a MoE token
budget (``moe_max_num_tokens``) below the call's rows.

One GPU, attention TP (one rank of TEP8 emulated, MoE EP8 or MoE TP4 x EP2, no collective inside the routed
experts): the TEP target's experts (the MoE prologue with the runner's weights, the CuTe DSL grouped GEMMs) against
the engine's routing inside the runner, with a budget below the call's rows: chunks alternate between the
chunk-overlap stream and the caller's (at MoE TP4 with chunks below 512 rows, two 1-CTA kernels, each stream on its
own workspace).
"""

import pytest
import torch

__extra_import_path__ = ["../.."]
from _k31_utils import on_sm

NUM_EXPERTS, TOP_K, HIDDEN, INTER = 896, 16, 3584, 3072
CAPS = (4.0, 25.0)

_sm10x = on_sm(
    100, 103, reason="the TRTLLM-Gen W4A8 MXFP4 runner and the CuTe DSL experts: sm_100 / sm_103"
)


# ---------------------------------------------------------------------------------------------------------------------
# Shared builders (GPU)
# ---------------------------------------------------------------------------------------------------------------------


def _model_config(mapping, max_num_tokens: int, budget):
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig

    pretrained = PretrainedConfig()
    pretrained.torch_dtype = torch.bfloat16
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        moe_max_num_tokens=budget,
        moe_backend="TRTLLM",
    )


def _aux_streams():
    from tensorrt_llm._torch.utils import AuxStreamType

    return {t: torch.cuda.Stream() for t in AuxStreamType}


def _bias():
    g = torch.Generator(device="cuda").manual_seed(7)
    return (torch.randn(NUM_EXPERTS, generator=g, device="cuda") * 0.1).float().contiguous()


def _expert(e: int) -> dict:
    """Checkpoint tensors of expert ``e`` (deterministic per expert): packed MXFP4 codes and UE8M0 scales."""
    g = torch.Generator(device="cuda").manual_seed(1000 + e)

    def codes(*shape):
        return torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.uint8)

    def scales(*shape):
        return torch.randint(117, 125, shape, generator=g, device="cuda", dtype=torch.uint8)

    return dict(
        w1_weight=codes(INTER, HIDDEN // 2),
        w1_weight_scale=scales(INTER, HIDDEN // 32),
        w3_weight=codes(INTER, HIDDEN // 2),
        w3_weight_scale=scales(INTER, HIDDEN // 32),
        w2_weight=codes(HIDDEN, INTER // 2),
        w2_weight_scale=scales(HIDDEN, INTER // 32),
    )


def _engine(cfg, bias):
    """The engine MoE the targets built: create_moe, TRTLLM-Gen W4A8_MXFP4_MXFP8, DeepSeek-V3 routing, SiTU."""
    from tensorrt_llm._torch.moe.fused_moe import SiTuActivation, create_moe
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    routing = DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    moe = create_moe(
        routing_method=routing,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=cfg,
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
        layer_idx=1,
        aux_stream_dict=_aux_streams(),
        activation=SiTuActivation(gate_softcap=CAPS[0], linear_softcap=CAPS[1]),
    )
    backend = moe.backend
    for slot, e in enumerate(backend.initial_local_expert_ids):
        backend.quant_method.load_packed_mxfp4_expert(
            backend, global_expert_id=e, local_slot_id=slot, **_expert(e)
        )
    backend.post_load_weights()
    return moe


def _family(cfg, bias, engine):
    """The family's routed experts on the engine's packed weights."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.routed_experts import (
        K31RoutedExperts,
    )

    experts = K31RoutedExperts(
        cfg,
        cfg.mapping,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        gate_cap=CAPS[0],
        linear_cap=CAPS[1],
        layer_idx=1,
        e_score_correction_bias=lambda: bias,
        routed_scaling_factor=1.0,
        aux_stream_dict=_aux_streams(),
    )
    for name in ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale"):
        getattr(experts, name).data = getattr(engine.backend, name).detach().clone()
    experts.post_load_weights()
    return experts


def _inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    logits = torch.randn(rows, NUM_EXPERTS, generator=g, device="cuda") * 2.0
    return x, logits


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


class _NoAllReduce(torch.nn.Module):
    """The MoE-TP all-reduce handle stand-in: the routed experts return the rank's partial, which the caller reduces;
    one process emulates one rank, so the TP group's handle cannot be built."""

    def __init__(self, *args, **kwargs):
        super().__init__()


# ---------------------------------------------------------------------------------------------------------------------
# Attention TP (one GPU)
# ---------------------------------------------------------------------------------------------------------------------


#: (MoE TP, MoE EP, MoE token budget) of one TEP8 rank. Full-width experts (MoE TP 1) take the CTA-pair kernel; MoE
#: TP 4 below 512 rows per chunk takes the 1-CTA kernel, whose workspace each chunk stream has its own of.
_TP_SPLITS = {
    "ep8_budget512": (1, 8, 512),
    "tp4ep2_budget256": (4, 2, 256),
}


@_sm10x
@pytest.mark.parametrize("split", list(_TP_SPLITS))
def test_attention_tp_matches_engine(monkeypatch, split: str) -> None:
    import tensorrt_llm._torch.distributed as distributed
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm import allreduce_module
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setattr(distributed, "AllReduce", _NoAllReduce)
    monkeypatch.setattr(allreduce_module, "AllReduce", _NoAllReduce)
    moe_tp, moe_ep, budget = _TP_SPLITS[split]
    mapping = Mapping(world_size=8, rank=3, tp_size=8, moe_tp_size=moe_tp, moe_ep_size=moe_ep)
    cfg = _model_config(mapping, 8192, budget)
    bias = _bias()
    with torch.device("cuda"):
        engine = _engine(cfg, bias)
        family = _family(cfg, bias, engine)
    for rows in (1, 72, 513, 2048):
        x, logits = _inputs(rows, seed=rows)
        expected = engine(x, logits)
        got = family(x, logits)
        torch.cuda.synchronize()
        assert _same(got, expected), f"{split} rows {rows}"
