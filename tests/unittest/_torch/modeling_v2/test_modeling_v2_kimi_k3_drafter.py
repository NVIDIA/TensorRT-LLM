# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's DSpark drafter (``K3DSparkDrafter`` of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``) on one GPU,
at one TP16 rank's drafter shapes: 6 query heads and 1 KV head of 64, hidden 7168, MLP width 896, context pages of
64 rows, two layers.

* A decode block of every split ``attention/k3_drafter_attn_qknorm`` certifies runs the drafter entries (the
  attention once per layer) and matches the stock block forward (``DFlashForCausalLM.dflash_forward`` on the same
  module and context), with the torch GEMMs and with the decode GEMV sites (which take the blocks of at most 8 rows).
  The entries leave the cache untouched.
* A split the entry does not certify runs the stock forward, bit for bit, without the entry.
* Under CUDA-graph capture, the entries take a block only once its attention compile key has run eagerly.
* Negative control: a weight changed between the two forwards fails the comparison.
"""

import math

import pytest
import torch
from transformers import Qwen3Config

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    decode_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as target,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="the K3 drafter entries run on sm_100 only",
)

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM, LAYERS, PAGE = 7168, 896, 6, 1, 64, 2, 64
VOCAB, RANK = 1024, 16
# Context lengths per request: inside a page, on a page boundary, across pages, empty.
CTX = (37, 64, 130, 0, 5, 200, 63, 1)
REL_L2 = 1e-2
SITES = ("drafter_qkv", "drafter_o", "drafter_gate_up", "drafter_down")


def _config():
    return Qwen3Config.from_dict(
        dict(
            architectures=["Qwen3ForCausalLM"],
            model_type="qwen3",
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_hidden_layers=LAYERS,
            num_attention_heads=HEADS,
            num_key_value_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            hidden_act="silu",
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            torch_dtype="bfloat16",
            tie_word_embeddings=False,
            markov_rank=RANK,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            dflash_config={"mask_token_id": VOCAB - 2, "target_layer_ids": [0, 1]},
        )
    )


def _weights(seed=11):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {
        "fc.weight": rnd(HIDDEN, 2 * HIDDEN),
        "hidden_norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "norm.weight": rnd(HIDDEN, scale=0.05) + 1.0,
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
        "confidence_head.proj.weight": rnd(1, HIDDEN + RANK),
        "confidence_head.proj.bias": rnd(1),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = rnd(HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.k_proj.weight"] = rnd(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.v_proj.weight"] = rnd(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = rnd(HEAD_DIM, scale=0.05) + 1.0
        w[p + "self_attn.k_norm.weight"] = rnd(HEAD_DIM, scale=0.05) + 1.0
        w[p + "input_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN, scale=0.05) + 1.0
        w[p + "mlp.gate_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTER)
    return w


@pytest.fixture(scope="module")
def drafter():
    model_config = ModelConfig(pretrained_config=_config(), attn_backend="TRTLLM")
    module = target.K3DSparkDrafter(model_config, dflash_attention_backend="TRTLLM").to("cuda")
    module.load_weights(_weights())
    assert module._k3_layers_take(), "the test drafter must be one the drafter entries take"
    return module


@pytest.fixture(scope="module")
def gemvs():
    return decode_gemv.K3DecodeGemvs.create(None, sites=SITES)


def _block(batch, block, seed):
    """A block's inputs: noise rows, query positions after each request's context, and a paged context cache
    (random rows, the requests' pages scattered) with its page table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor(CTX[:batch], dtype=torch.int32, device="cuda")
    width = math.ceil((int(ctx.max()) + block) / PAGE) + 1
    pages = batch * width + 3
    order = torch.randperm(pages, generator=g, device="cuda")[: batch * width]
    caches = [
        (torch.randn(pages, 2, KV_HEADS, PAGE, HEAD_DIM, generator=g, device="cuda")).to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    return dict(
        noise_embedding=torch.randn(batch, block, HIDDEN, generator=g, device="cuda").to(
            torch.bfloat16
        ),
        query_positions=(ctx.long()[:, None] + torch.arange(block, device="cuda")),
        num_ctx_per_req=ctx,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, dtype=torch.int32, device="cuda"),
        ctx_kv_cache=caches,
        ctx_page_table=order.view(batch, width).to(torch.int32),
    )


def _copy(inputs):
    out = dict(inputs)
    out["ctx_kv_cache"] = [c.clone() for c in inputs["ctx_kv_cache"]]
    return out


def _rel_l2(y, ref):
    return ((y.double() - ref.double()).norm() / ref.double().norm()).item()


@pytest.fixture
def attn_calls(monkeypatch):
    """Counts the drafter's calls of the attention entry."""
    calls = []
    entry = target.k3_drafter_attn_qknorm

    def counted(*args):
        calls.append(args[0].shape[0])
        return entry(*args)

    monkeypatch.setattr(target, "k3_drafter_attn_qknorm", counted)
    return calls


@pytest.mark.parametrize("use_gemvs", [False, True], ids=["torch", "gemv"])
@pytest.mark.parametrize("split", sorted(target.DRAFTER_ATTN_SPLITS))
def test_block_matches_the_stock_forward(drafter, gemvs, attn_calls, monkeypatch, use_gemvs, split):
    batch, block = split
    projected = []
    project = gemvs.project

    def recorded(site, x, weight):
        y = project(site, x, weight)
        projected.append((site, y is not None))
        return y

    monkeypatch.setattr(gemvs, "project", recorded)
    drafter.decode_gemvs = gemvs if use_gemvs else None
    inputs = _block(batch, block, seed=batch * 10 + block)
    stock_inputs = _copy(inputs)
    before = [c.clone() for c in inputs["ctx_kv_cache"]]
    out = drafter.dflash_forward(**inputs)
    ref = DFlashForCausalLM.dflash_forward(drafter, **stock_inputs)
    torch.cuda.synchronize()
    assert attn_calls == [batch * block] * LAYERS
    assert out.shape == ref.shape == (batch * block, HIDDEN)
    err = _rel_l2(out, ref)
    assert err <= REL_L2, err
    assert all(torch.equal(c, b) for c, b in zip(inputs["ctx_kv_cache"], before))
    if use_gemvs:
        # At most 8 rows every site takes its projection; above, each declines and its module runs (the down
        # projection's site is not asked once gate / up's declined).
        takes = batch * block <= decode_gemv.MAX_ROWS
        asked = SITES if takes else SITES[:3]
        assert sorted(projected) == sorted(
            (site, takes) for site in asked for _ in range(LAYERS)
        ), projected
    else:
        assert projected == []


def test_an_uncertified_split_runs_the_stock_forward(drafter, attn_calls):
    drafter.decode_gemvs = None
    batch, block = 4, 4
    assert (batch, block) not in target.DRAFTER_ATTN_SPLITS
    inputs = _block(batch, block, seed=44)
    stock_inputs = _copy(inputs)
    out = drafter.dflash_forward(**inputs)
    ref = DFlashForCausalLM.dflash_forward(drafter, **stock_inputs)
    assert attn_calls == []
    assert torch.equal(out, ref)


def test_capture_takes_only_compiled_keys(drafter):
    inputs = _block(1, 8, seed=18)
    args = (inputs["noise_embedding"], inputs["ctx_kv_cache"], inputs["ctx_page_table"])
    drafter._k3_attn_ran.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refused = drafter._k3_block_keys(*args)
    assert refused is None
    drafter.dflash_forward(**inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        taken = drafter._k3_block_keys(*args)
    assert taken is not None and taken <= drafter._k3_attn_ran


def test_negative_control_a_changed_weight_fails(drafter):
    drafter.decode_gemvs = None
    inputs = _block(1, 8, seed=81)
    stock_inputs = _copy(inputs)
    weight = drafter.model.layers[0].self_attn.o_proj.weight
    saved = weight.detach().clone()
    try:
        with torch.no_grad():
            weight.add_(torch.randn_like(weight) * 0.02)
        out = drafter.dflash_forward(**inputs)
    finally:
        with torch.no_grad():
            weight.copy_(saved)
    ref = DFlashForCausalLM.dflash_forward(drafter, **stock_inputs)
    assert _rel_l2(out, ref) > REL_L2
