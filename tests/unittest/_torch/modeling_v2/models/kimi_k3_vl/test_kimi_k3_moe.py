# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family's MoE block (``moe.py``) at both targets' expert splits, on one GPU.

* Host-side: the gate admits only the routing the routed experts compute; the routed experts' quantization is checked
  at construction.
* ``pack_mxfp4_expert`` gives the bytes the built-in trtllm-gen MXFP4 expert loader
  (``W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.load_packed_mxfp4_expert``) writes, at several ranks of each MoE TP split,
  on the GPU and on the host.
* One rank's block of each target's TP16 mapping, built in one process (``Mapping(world_size=16, rank=r, ...)``, each
  ``K3AllReduce`` replaced by the sum over a group of one, since building it is collective): it holds the rank's
  expert window and intermediate share, every slot loaded once; its routed experts above ``moe_max_num_tokens`` are
  the concatenation of their chunks' results, bit for bit; its generic forward routes a chunk of 1 to 64 tokens with
  ``moe/kimi_k3_noaux_tc_mxfp8_quant`` and a larger one with ``noaux_tc_op``, and is finite and repeats bit for bit.
"""

from __future__ import annotations

import copy
import types
from unittest import mock

import pytest
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.moe.fused_moe.quantization import W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

try:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl import moe as family_moe
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

__extra_import_path__ = ["."]
from _kimi_k3_parity import assert_bitwise  # noqa: E402

HIDDEN, LATENT, NUM_EXPERTS, TOP_K = 7168, 3584, 896, 16
# The checkpoint's routed and shared intermediate widths (two shared experts).
MOE_INTERMEDIATE, NUM_SHARED = 3072, 2
SITU_CAPS = (4.0, 25.0)

#: Each target's MoE split, the TP16 rank the block check builds, and what that rank holds, from ``Mapping``'s
#: definitions (no MoE cluster): ``moe_ep_rank = tp_rank % moe_ep``, ``moe_tp_rank = tp_rank // moe_ep``; the expert
#: window starts at ``moe_ep_rank * (896 // moe_ep)`` and holds ``896 // moe_ep`` experts; the intermediate share is
#: ``3072 // moe_tp`` checkpoint columns padded to whole 128-row tiles. ``pack_ranks``: the MoE TP ranks the pack check
#: compares.
#:   moe TP 4 x EP 4, rank 9: ep rank 9 % 4 = 1, window [224, 448); tp rank 9 // 4 = 2; 768 columns, padded 768.
#:   moe TP 16 x EP 1, rank 5: ep rank 0, window [0, 896); tp rank 5; 192 columns, padded 256.
LAYOUTS = {
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4": dict(
        moe_tp=4,
        moe_ep=4,
        rank=9,
        experts=(224, 224),
        moe_tp_rank=2,
        intermediate=(768, 768),
        pack_ranks=(0, 1, 2, 3),
    ),
    "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1": dict(
        moe_tp=16,
        moe_ep=1,
        rank=5,
        experts=(0, 896),
        moe_tp_rank=5,
        intermediate=(256, 192),
        pack_ranks=(0, 7, 15),
    ),
}

# One engine token budget, with MoE chunks of 1024 tokens: 3000 tokens run 3 chunks of 1000, alternating with the
# chunking stream.
MAX_NUM_TOKENS, MOE_MAX_NUM_TOKENS = 4096, 1024
# Decode-size, the fused-routing boundary, prefill-size, and chunked.
TOKEN_COUNTS = (1, 8, 64, 65, 300, 3000)


def sm_100f() -> bool:
    """Whether the GPU takes the trtllm-gen MXFP4 SiTU cubins and the fused route + quant."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


needs_sm_100f = pytest.mark.skipif(
    not sm_100f(),
    reason="the trtllm-gen MXFP4 SiTU cubins and the fused route + quant need SM100 or SM103",
)


class _OneRankReduce(nn.Module):
    """The all-reduce of a group of one (the identity), with ``K3AllReduce``'s constructor and call signature."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, all_reduce_params=None):
        return x


def _cfg():
    cfg = PretrainedConfig()
    cfg.hidden_size = HIDDEN
    cfg.num_experts = NUM_EXPERTS
    cfg.num_experts_per_token = TOP_K
    cfg.routed_expert_hidden_size = LATENT
    cfg.latent_moe_use_norm = True
    cfg.activation_situ_beta, cfg.activation_situ_linear_beta = SITU_CAPS
    cfg.routed_scaling_factor = 2.827
    cfg.moe_router_activation_func = "sigmoid"
    cfg.num_expert_group = 1
    cfg.topk_group = 1
    cfg.moe_renormalize = True
    cfg.moe_intermediate_size = MOE_INTERMEDIATE
    cfg.num_shared_experts = NUM_SHARED
    cfg.rms_norm_eps = 1e-5
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _model_config(layout: dict) -> ModelConfig:
    mapping = Mapping(
        world_size=16,
        rank=layout["rank"],
        tp_size=16,
        moe_tp_size=layout["moe_tp"],
        moe_ep_size=layout["moe_ep"],
    )
    return ModelConfig(
        pretrained_config=_cfg(),
        mapping=mapping,
        moe_backend="TRTLLM",
        max_num_tokens=MAX_NUM_TOKENS,
        moe_max_num_tokens=MOE_MAX_NUM_TOKENS,
    )


def _aux_streams():
    stream = torch.cuda.Stream()
    return {
        AuxStreamType.MoeShared: stream,
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
        AuxStreamType.Attention: stream,
    }


def _expert_tensors(expert: int):
    """One random checkpoint expert: packed MXFP4 w1 / w3 ``[3072, 1792]`` and w2 ``[3584, 1536]``, E8M0 scales in a
    narrow band, drawn on the GPU from a seed of the expert id."""
    g = torch.Generator(device="cuda").manual_seed(1000 + expert)

    def packed(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device="cuda")

    def scales(*shape):
        return torch.randint(118, 124, shape, generator=g, dtype=torch.uint8, device="cuda")

    return dict(
        w1_weight=packed(MOE_INTERMEDIATE, LATENT // 2),
        w1_weight_scale=scales(MOE_INTERMEDIATE, LATENT // 32),
        w2_weight=packed(LATENT, MOE_INTERMEDIATE // 2),
        w2_weight_scale=scales(LATENT, MOE_INTERMEDIATE // 32),
        w3_weight=packed(MOE_INTERMEDIATE, LATENT // 2),
        w3_weight_scale=scales(MOE_INTERMEDIATE, LATENT // 32),
    )


def _build(layout: dict):
    """The block at ``layout``'s rank: random non-expert weights, every local expert loaded from a random checkpoint
    expert (the checkpoint tensors on the host, as the loader hands them), each all-reduce over a group of one."""
    model_config = _model_config(layout)
    with torch.device("cuda"), mock.patch.object(family_moe, "K3AllReduce", _OneRankReduce):
        moe = family_moe.KimiK3MoERuntime(
            model_config, model_config.pretrained_config, 3, _aux_streams()
        )
    moe.cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        for name, param in moe.named_parameters():
            if name.startswith("routed_experts."):
                continue
            noise = torch.randn(param.shape, generator=g, device="cuda")
            if name == "routed_expert_norm.weight":
                param.copy_(1.0 + 0.1 * noise)
            elif name == "gate.e_score_correction_bias":
                param.copy_(0.05 * noise)
            else:
                param.copy_(0.02 * noise)
    backend = moe.routed_experts.backend
    for slot, expert in enumerate(backend.initial_local_expert_ids):
        tensors = {k: v.cpu() for k, v in _expert_tensors(expert).items()}
        backend.load_packed_mxfp4_expert(global_expert_id=expert, local_slot_id=slot, **tensors)
    backend.check_loaded()
    backend.post_load_weights()
    return moe


def test_gate_checks_routing():
    """Host-side: the gate admits only the routing the routed experts compute (sigmoid scores, renormalized)."""
    cfg = _cfg()
    gate = family_moe.KimiK3MoEGate(cfg)
    gate.check_routing()
    for field, value in (("moe_router_activation_func", "softmax"), ("moe_renormalize", False)):
        bad = copy.copy(cfg)
        setattr(bad, field, value)
        with pytest.raises(ValueError):
            family_moe.KimiK3MoEGate(bad).check_routing()


def test_routed_quant_is_checked():
    """Host-side: the block serves the MXFP4 checkpoint's routed-expert quantization (W4A8_MXFP4_MXFP8) and rejects
    at construction any other, unquantized included."""
    family_moe._check_routed_quant(QuantAlgo.W4A8_MXFP4_MXFP8)
    for other in (None, QuantAlgo.NVFP4, QuantAlgo.W4A16_MXFP4):
        with pytest.raises(NotImplementedError, match="no per-expert checkpoint layout"):
            family_moe._check_routed_quant(other)


@needs_sm_100f
@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_pack_matches_builtin_loader(layout_name):
    """``pack_mxfp4_expert`` gives the built-in trtllm-gen MXFP4 expert loader's bytes for a few checkpoint experts at
    several ranks of the layout's MoE TP split, and the same bytes on the host."""
    layout_spec = LAYOUTS[layout_name]
    moe_tp, experts = layout_spec["moe_tp"], (0, 1, 2)
    method = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
    for rank in layout_spec["pack_ranks"]:
        layout = family_moe.Mxfp4ExpertLayout(
            hidden=LATENT, intermediate=MOE_INTERMEDIATE, moe_tp_size=moe_tp, moe_tp_rank=rank
        )
        shapes = layout.slot_shapes()
        module = types.SimpleNamespace(
            expert_size_per_partition=len(experts),
            initial_local_expert_ids=list(experts),
            scaling_vector_size=32,
            intermediate_size=MOE_INTERMEDIATE,
            intermediate_size_per_partition=MOE_INTERMEDIATE // moe_tp,
            tp_size=moe_tp,
            tp_rank=rank,
            hidden_size=LATENT,
            **{
                name: torch.zeros((len(experts), *shape), dtype=torch.uint8, device="cuda")
                for name, shape in shapes.items()
            },
        )
        for slot, expert in enumerate(experts):
            tensors = _expert_tensors(expert)
            method.load_packed_mxfp4_expert(
                module, global_expert_id=expert, local_slot_id=slot, **tensors
            )
            packed = family_moe.pack_mxfp4_expert(layout, **tensors)
            for name in shapes:
                assert_bitwise(
                    packed[name],
                    getattr(module, name)[slot],
                    f"{name}, moe tp {moe_tp} rank {rank}",
                )
            host = family_moe.pack_mxfp4_expert(layout, **{k: v.cpu() for k, v in tensors.items()})
            for name in shapes:
                assert_bitwise(host[name].cuda(), packed[name], f"{name} on the host")


def _routing_spy():
    """Record which routing each ``K3RoutedExperts`` chunk takes: ``("fused", rows)`` for
    ``moe/kimi_k3_noaux_tc_mxfp8_quant``, ``("unfused", rows)`` for ``noaux_tc_op`` (then ``mxfp8_quantize``)."""
    calls = []
    fused, unfused = family_moe.kimi_k3_noaux_tc_mxfp8_quant, family_moe.noaux_tc_op

    def spy_fused(router_logits, bias, hidden_states, routed_scaling_factor):
        calls.append(("fused", hidden_states.shape[0]))
        return fused(router_logits, bias, hidden_states, routed_scaling_factor)

    def spy_unfused(router_logits, *args):
        calls.append(("unfused", router_logits.shape[0]))
        return unfused(router_logits, *args)

    patch = mock.patch.multiple(
        family_moe, kimi_k3_noaux_tc_mxfp8_quant=spy_fused, noaux_tc_op=spy_unfused
    )
    return calls, patch


@needs_sm_100f
@pytest.mark.parametrize("layout_name", sorted(LAYOUTS))
def test_block(layout_name):
    """One rank's block: its expert window and intermediate share; a second load of a slot refused; the routed
    experts above ``moe_max_num_tokens`` equal to their chunks' results concatenated; the generic forward's routing
    per chunk (the fused route + quant for 1 to 64 tokens, ``noaux_tc_op`` above); and the generic forward finite,
    nonzero and the same bits when repeated."""
    layout = LAYOUTS[layout_name]
    moe = _build(layout)
    routed = moe.routed_experts
    backend = routed.backend
    assert (backend.slot_start, backend.expert_size_per_partition) == layout["experts"]
    assert (backend.layout.moe_tp_size, backend.layout.moe_tp_rank) == (
        layout["moe_tp"],
        layout["moe_tp_rank"],
    )
    assert (backend.layout.inter_pad, backend.intermediate_size_per_partition) == layout[
        "intermediate"
    ]
    assert backend.loaded_slots == set(range(backend.expert_size_per_partition))
    with pytest.raises(ValueError, match="loaded twice"):
        backend.load_packed_mxfp4_expert(
            global_expert_id=backend.slot_start,
            local_slot_id=0,
            **_expert_tensors(backend.slot_start),
        )
    assert moe._reduce_routed_output and routed.all_reduce is not None
    assert moe.shared_experts.down_proj.all_reduce is not None
    assert routed.moe_max_num_tokens == MOE_MAX_NUM_TOKENS and routed.chunk_stream is not None
    assert routed.route_quant_fused
    assert torch.equal(backend.act_alpha, torch.full_like(backend.act_alpha, SITU_CAPS[0]))
    assert torch.equal(backend.act_beta, torch.full_like(backend.act_beta, SITU_CAPS[1]))

    g = torch.Generator(device="cuda").manual_seed(11)
    gate = moe.gate
    calls, spy = _routing_spy()
    with torch.inference_mode(), spy:
        # 3000 tokens in chunks of 1024: three chunks of 1000, the even ones on the chunking stream.
        x = torch.randn(3000, LATENT, generator=g, device="cuda").to(torch.bfloat16)
        logits = gate.compute_logits(
            torch.randn(3000, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
        )
        args = (gate.e_score_correction_bias, gate.routed_scaling_factor)
        chunked = routed(x, logits, *args)
        pieces = [routed._run(xc, lc, *args) for xc, lc in zip(x.split(1000), logits.split(1000))]
        assert_bitwise(chunked, torch.cat(pieces), "routed experts in 3 chunks")

        for rows in TOKEN_COUNTS:
            h = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
            calls.clear()
            first = moe.forward_generic(h)
            # One chunk up to MOE_MAX_NUM_TOKENS (3000 tokens: three of 1000).
            chunk = rows if rows <= MOE_MAX_NUM_TOKENS else 1000
            expected = [("fused" if chunk <= 64 else "unfused", chunk)] * (rows // chunk)
            assert calls == expected, (rows, calls)
            assert torch.isfinite(first.float()).all(), rows
            assert first.float().abs().max() > 0, rows
            assert_bitwise(moe.forward_generic(h), first, f"repeated forward at {rows} tokens")
