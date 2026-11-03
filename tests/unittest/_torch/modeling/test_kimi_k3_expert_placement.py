# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 static expert placement (router relabelling)."""

import json

import pytest
import torch

from tensorrt_llm._torch.modules import kimi_k3_expert_placement as placement
from tensorrt_llm._torch.modules.kimi_k3_expert_placement import (
    expert_placement_for_layer,
    load_expert_placement,
    permute_router_to_slots,
)
from tensorrt_llm.mapping import Mapping

NUM_EXPERTS = 896
TOP_K = 16

_needs_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100/SM103 GPU",
)


def _write_placement(path, tables, num_experts, ep_size=16, fmt="kimi_k3_expert_placement/v1"):
    data = {
        "format": fmt,
        "num_experts": num_experts,
        "ep_size": ep_size,
        "expert_of_slot": {str(k): list(v) for k, v in tables.items()},
    }
    path.write_text(json.dumps(data))
    return str(path)


def _permutation(num_experts, seed):
    return tuple(
        torch.randperm(num_experts, generator=torch.Generator().manual_seed(seed)).tolist()
    )


def _assert_same_routing_up_to_ties(logits, bias, perm_t, ids, weights, ids_p, weights_p):
    """The permuted router's routing (slots ``ids_p``, weights ``weights_p``) picks, token by
    token, the experts ``ids`` with the weights ``weights`` of the original router, except where
    equal fp32 scores meet: the routing kernels break such ties by routed-expert id, i.e. by slot
    on the permuted router (see ``kimi_k3_expert_placement``), so the tied experts may come in
    the other order, or the other one is picked at the top-k boundary. Such a token must hold
    experts of equal score at every rank in both routings; its weights follow the pick and are
    not compared."""
    ids_c = perm_t[ids_p]
    tied = (ids_c != ids).any(dim=1)
    same = ~tied
    torch.testing.assert_close(weights_p[same], weights[same], rtol=0, atol=0)
    if not bool(tied.any()):
        return
    # The kernels' score is sigmoid(logit) + bias with the sigmoid as 0.5 * tanh(0.5 * logit) +
    # 0.5 in fp32. torch's tanh may differ from theirs in the last bits, hence a tolerance of a
    # few ulp; scores that are not tied differ by far more near the top-k boundary.
    scores = 0.5 * torch.tanh(0.5 * logits[tied]) + 0.5 + bias
    torch.testing.assert_close(
        scores.gather(1, ids_c[tied]), scores.gather(1, ids[tied]), rtol=0, atol=1e-6
    )
    # Ties are rare (the real K3 routers: about 0.1% of the token-layers).
    num_tied = int(tied.sum())
    assert num_tied <= max(1, ids.shape[0] // 100), f"{num_tied} of {ids.shape[0]} tokens differ"


def test_load_expert_placement_drops_identity_layers(tmp_path):
    perm = _permutation(32, 0)
    path = _write_placement(
        tmp_path / "p.json", {1: perm, 2: list(range(32))}, num_experts=32, ep_size=4
    )
    load_expert_placement.cache_clear()
    tables = load_expert_placement(path, 32)
    assert tables == {1: perm}


@pytest.mark.parametrize(
    "tables,num_experts,fmt",
    [
        ({1: [0, 1, 2, 2]}, 4, "kimi_k3_expert_placement/v1"),  # not a permutation
        ({1: [0, 1, 2]}, 4, "kimi_k3_expert_placement/v1"),  # wrong length
        ({1: [1, 0, 2, 3]}, 4, "something_else"),  # wrong format
    ],
)
def test_load_expert_placement_rejects_bad_files(tmp_path, tables, num_experts, fmt):
    path = _write_placement(tmp_path / "bad.json", tables, num_experts=num_experts, fmt=fmt)
    load_expert_placement.cache_clear()
    with pytest.raises(ValueError):
        load_expert_placement(path, num_experts)


def test_load_expert_placement_rejects_other_expert_count(tmp_path):
    path = _write_placement(tmp_path / "p.json", {1: [1, 0, 2, 3]}, num_experts=4)
    load_expert_placement.cache_clear()
    with pytest.raises(ValueError):
        load_expert_placement(path, 8)


def test_expert_placement_for_layer_follows_env(tmp_path, monkeypatch):
    perm = _permutation(16, 1)
    path = _write_placement(tmp_path / "p.json", {3: perm}, num_experts=16, ep_size=4)
    load_expert_placement.cache_clear()
    monkeypatch.delenv(placement.EXPERT_PLACEMENT_ENV, raising=False)
    assert expert_placement_for_layer(3, 16, ep_size=4) is None
    monkeypatch.setenv(placement.EXPERT_PLACEMENT_ENV, path)
    assert expert_placement_for_layer(3, 16, ep_size=4) == perm
    assert expert_placement_for_layer(4, 16, ep_size=4) is None
    # Another EP size still gets the (correct, unbalanced) relabelling.
    assert expert_placement_for_layer(3, 16, ep_size=2) == perm


def test_permute_router_to_slots_moves_rows():
    perm = _permutation(8, 2)
    weight = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)
    bias = torch.arange(8, dtype=torch.float32) * 0.5
    ref_weight, ref_bias = weight.clone(), bias.clone()
    permute_router_to_slots(weight, bias, perm)
    for slot, expert in enumerate(perm):
        assert torch.equal(weight[slot], ref_weight[expert])
        assert bias[slot] == ref_bias[expert]


@_needs_sm100
@pytest.mark.parametrize("num_tokens", [1, 8, 32, 256])
def test_relabelled_router_routes_to_the_same_experts(num_tokens):
    """The production router GEMM + CuTe routing tail on the permuted router pick, per token,
    the slots holding exactly the experts (and weights) the original router picks, up to fp32
    score ties."""
    import tensorrt_llm  # noqa: F401  (registers the ops)
    from tensorrt_llm._torch.moe.kernels.blackwell import k3_route_tail  # noqa: F401

    hidden = 7168
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    weight = (torch.randn(NUM_EXPERTS, hidden, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    bias = torch.randn(NUM_EXPERTS, device="cuda", generator=gen) * 0.01
    x = torch.randn(num_tokens, hidden, device="cuda", generator=gen).to(torch.bfloat16)
    perm = _permutation(NUM_EXPERTS, 3)
    perm_t = torch.tensor(perm, device="cuda")

    def route(w, b):
        logits = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
        ids, weights = torch.ops.trtllm.k3_route_tail(logits, b, 1.0, 0, 56, 128)[:2]
        return logits, ids.long(), weights

    logits, ids, weights = route(weight, bias)
    weight_p, bias_p = weight.clone(), bias.clone()
    permute_router_to_slots(weight_p, bias_p, perm)
    logits_p, ids_p, weights_p = route(weight_p, bias_p)

    torch.testing.assert_close(logits_p, logits[:, perm_t], rtol=0, atol=0)
    _assert_same_routing_up_to_ties(logits, bias, perm_t, ids, weights, ids_p, weights_p)


@_needs_sm100
@pytest.mark.parametrize("num_tokens", [384, 896, 1792])
def test_relabelled_router_noaux_routes_to_the_same_experts(num_tokens):
    """Above the routing tail's token limit (b >= 64 decode batches) the router GEMM feeds the
    DeepSeek-V3 ``noaux_tc`` routing; on the permuted router it picks, per token, the slots holding
    exactly the experts (and weights) the original router picks, up to fp32 score ties."""
    import tensorrt_llm  # noqa: F401  (registers the ops)
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

    hidden = 7168
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    weight = (torch.randn(NUM_EXPERTS, hidden, device="cuda", generator=gen) * 0.02).to(
        torch.bfloat16
    )
    bias = torch.randn(NUM_EXPERTS, device="cuda", generator=gen) * 0.01
    x = torch.randn(num_tokens, hidden, device="cuda", generator=gen).to(torch.bfloat16)
    perm = _permutation(NUM_EXPERTS, 5)
    perm_t = torch.tensor(perm, device="cuda")

    def route(w, b):
        logits = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
        routing = DeepSeekV3MoeRoutingMethod(
            top_k=TOP_K,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=2.5,
            callable_e_score_correction_bias=lambda: b,
        )
        ids, weights = routing.apply(logits)
        return logits, ids.long(), weights

    logits, ids, weights = route(weight, bias)
    weight_p, bias_p = weight.clone(), bias.clone()
    permute_router_to_slots(weight_p, bias_p, perm)
    logits_p, ids_p, weights_p = route(weight_p, bias_p)

    torch.testing.assert_close(logits_p, logits[:, perm_t], rtol=0, atol=0)
    # Same experts in the same order with the same weights, up to fp32 score ties.
    _assert_same_routing_up_to_ties(logits, bias, perm_t, ids, weights, ids_p, weights_p)


# ---------------------------------------------------------------------------
# Model level: a two-layer K3 (dense layer 0, latent-MoE layer 1) loaded from the same synthetic
# checkpoint with and without a placement must hold the same expert in each relabelled slot and
# produce the same MoE output.
# ---------------------------------------------------------------------------

_HIDDEN, _LATENT, _INTER, _EXPERTS, _TOPK = 1024, 512, 256, 32, 4


def _tiny_k3_model(moe_backend: str, packed_mxfp4: bool = False):
    """``packed_mxfp4``: K3.1 routed experts (group-32 MXFP4 ``weight_packed`` / ``weight_scale``,
    W4A8_MXFP4_MXFP8), else NVFP4."""
    pytest.importorskip("fla")
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    config = ModelConfig(
        pretrained_config=KimiLinearConfig(
            vocab_size=128,
            hidden_size=_HIDDEN,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            rms_norm_eps=1e-6,
            attn_res_block_size=1,
            mla_use_output_gate=True,
            max_position_embeddings=128,
            torch_dtype=torch.bfloat16,
            linear_attn_config={"kda_layers": [], "full_attn_layers": [1, 2], "num_heads": 8},
            num_experts=_EXPERTS,
            num_experts_per_token=_TOPK,
            moe_intermediate_size=_INTER,
            num_shared_experts=1,
            first_k_dense_replace=1,
            routed_expert_hidden_size=_LATENT,
            latent_moe_use_norm=True,
            hidden_act="situ",
            activation_situ_linear_beta=25.0,
        ),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        quant_config=QuantConfig(),
        quant_config_dict={
            "model.layers.1.block_sparse_moe.experts": (
                QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8, group_size=32)
                if packed_mxfp4
                else QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
            )
        },
        moe_backend=moe_backend,
        allreduce_strategy=AllReduceStrategy.NCCL,
    )
    with torch.device("cuda"):
        return KimiLinearForCausalLM(config)


def _synthetic_k3_checkpoint(model) -> dict:
    gen = torch.Generator().manual_seed(2026)
    name_map, expected_keys, _ = model.checkpoint_name_plan("")
    params = dict(model.named_parameters())
    weights = {}
    for name, key in name_map.items():
        param = params[name]
        moe_side = ".block_sparse_moe." in name
        shape = tuple(param.shape)
        if name == "lm_head.weight":
            shape = (model.config.vocab_size, _HIDDEN)
        if name.endswith("gate_up_proj.weight"):
            half = (shape[0] // 2, shape[1])
            for source in ("gate_proj", "up_proj"):
                weights[key.replace("gate_up_proj", source)] = (
                    torch.randn(half, generator=gen) * 0.02
                ).to(param.dtype)
            continue
        if name.endswith("mlp.down_proj.weight") and not moe_side:
            shape = (_HIDDEN, 512)
        value = torch.randn(shape, generator=gen) * 0.02 if moe_side else torch.zeros(shape)
        if name.endswith("e_score_correction_bias"):
            value = torch.randn(shape, generator=gen) * 0.01
        if name.endswith("norm.weight"):
            value = torch.ones(shape)
        weights[key] = value.to(param.dtype)
    # MLA projections in checkpoint layout (zeros: only the MoE block runs here).
    mla_shapes = {
        "q_a_proj": (1536, _HIDDEN),
        "kv_a_proj_with_mqa": (576, _HIDDEN),
        "q_b_proj": (8 * 192, 1536),
        "kv_b_proj": (8 * 256, 512),
        "g_proj": (8 * 128, _HIDDEN),
        "o_proj": (_HIDDEN, 8 * 128),
    }
    for layer in range(2):
        for name, shape in mla_shapes.items():
            weights[f"model.layers.{layer}.self_attn.{name}.weight"] = torch.zeros(
                shape, dtype=torch.bfloat16
            )
    packed = any(key.endswith(".weight_packed") for key in expected_keys)
    for key in expected_keys:
        if ".block_sparse_moe.experts." not in key or key in weights:
            continue
        suffix = key.rsplit(".", 2)
        w, kind = suffix[-2], suffix[-1]
        # Routed experts work in the latent width: w1 / w3 are [inter, latent], w2 [latent, inter].
        rows, cols = (_LATENT, _INTER) if w == "w2" else (_INTER, _LATENT)
        if kind in ("weight", "weight_packed"):
            weights[key] = torch.randint(
                0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8
            )
        elif kind == "weight_scale" and packed:
            # Group-32 UE8M0 exponents around 2^0.
            weights[key] = torch.randint(
                120, 132, (rows, cols // 32), generator=gen, dtype=torch.uint8
            )
        elif kind == "weight_scale":
            weights[key] = (
                torch.randint(120, 132, (rows, cols // 16), generator=gen, dtype=torch.int32)
                .float()
                .div(126.0)
                .to(torch.float8_e4m3fn)
            )
        elif kind == "weight_scale_2":
            weights[key] = torch.tensor(0.00012207, dtype=torch.float32)
        else:
            weights[key] = torch.tensor(1.0, dtype=torch.float32)
    return weights


def _load_like_model_loader(model, weights: dict) -> None:
    """``load_weights`` followed by every module's ``post_load_weights``, as the model loader runs them (the MXFP4
    backends derive kernel state there, e.g. the unpadded intermediate size)."""
    model.load_weights(dict(weights))
    for module in model.modules():
        post_load_weights = getattr(module, "post_load_weights", None)
        if post_load_weights is not None and not getattr(module, "_weights_removed", False):
            post_load_weights()


@_needs_sm100
@pytest.mark.parametrize(
    "moe_backend,packed_mxfp4",
    [
        # K3.1 group-32 MXFP4 experts: the TRTLLM-Gen W4A8 backend through the packed-MXFP4
        # expert loader.
        ("TRTLLM", True),
    ],
    ids=["trtllm-mxfp4"],
)
def test_model_load_with_placement_relabels_experts(
    tmp_path, monkeypatch, moe_backend, packed_mxfp4
):
    perm = _permutation(_EXPERTS, 9)
    path = _write_placement(tmp_path / "p.json", {1: perm}, num_experts=_EXPERTS, ep_size=1)
    load_expert_placement.cache_clear()

    monkeypatch.delenv(placement.EXPERT_PLACEMENT_ENV, raising=False)
    base = _tiny_k3_model(moe_backend, packed_mxfp4)
    weights = _synthetic_k3_checkpoint(base)
    _load_like_model_loader(base, weights)
    monkeypatch.setenv(placement.EXPERT_PLACEMENT_ENV, path)
    placed = _tiny_k3_model(moe_backend, packed_mxfp4)
    _load_like_model_loader(placed, weights)

    moe_base = base.model.layers[1].block_sparse_moe
    moe_placed = placed.model.layers[1].block_sparse_moe
    assert moe_base.expert_of_slot is None and moe_placed.expert_of_slot == perm
    perm_t = torch.tensor(perm, device="cuda")
    torch.testing.assert_close(moe_placed.gate.weight, moe_base.gate.weight[perm_t], rtol=0, atol=0)
    torch.testing.assert_close(
        moe_placed.gate.e_score_correction_bias,
        moe_base.gate.e_score_correction_bias[perm_t],
        rtol=0,
        atol=0,
    )
    for attr in ("w3_w1_weight", "w2_weight"):
        a = getattr(moe_placed.routed_experts.backend, attr).data.view(torch.uint8)
        b = getattr(moe_base.routed_experts.backend, attr).data.view(torch.uint8)
        assert torch.equal(a, b[perm_t]), attr

    torch.manual_seed(0)
    hidden = torch.randn(64, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        out_base = moe_base(hidden.clone())
        out_placed = moe_placed(hidden.clone())
    torch.testing.assert_close(out_placed.float(), out_base.float(), rtol=2e-2, atol=2e-3)
    print(
        f"{moe_backend}: bitwise equal {torch.equal(out_placed, out_base)}, "
        f"max abs diff {(out_placed.float() - out_base.float()).abs().max().item():.3e}"
    )
