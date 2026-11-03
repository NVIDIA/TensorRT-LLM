# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MLA DSpark drafter's helix layer with the dspark_ull2 paths (one GPU).

* ``scatter_rows_to_pool(rows_mask=...)`` writes exactly what the eager
  ``torch.where(mask, rows, pool[pages, offsets])`` round trip writes.
* ``draft_helix_store_enabled`` (drafting Kimi K3.1 engines without attention DP, with the fused tail):
  under helix the fused block RoPE + owner-masked store gives
  the same drafter output, the same exchange inputs and the same pool bytes as the separate
  launches (BF16 and FP8 pools; the exchange itself is a stand-in here, the multi-GPU test covers
  it).
"""

import pytest
import torch

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import MLADSparkForCausalLM
from tensorrt_llm._torch.pyexecutor import input_commit
from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard, to_draft_fp8
from tensorrt_llm._torch.speculative.dspark_ull2 import to_draft_fp8_1k
from tensorrt_llm._torch.speculative.spec_tail_kernels import scatter_rows_to_pool
from tensorrt_llm.mapping import CpType, Mapping

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _sm100f() -> bool:
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._utils import is_sm_100f

    return is_sm_100f()


needs_sm100f = pytest.mark.skipif(not _sm100f(), reason="CuTe DSL MLA decode needs SM100f")

# A tiny MLA drafter with the real latent geometry (the kernels are specialised on
# kv_lora_rank 512 / rope 64).
HIDDEN = 64
INTERMEDIATE = 32
HEADS = 8
Q_LORA = 16
KV_LORA = 512
NOPE = 128
ROPE = 64
V_DIM = 128
LAYERS = 2
BLOCK = 7
LATENT = KV_LORA + ROPE
VOCAB = 512
RANK = 16
NUM_CAPTURE = 2
PAGE = 32
MAX_SEQ_LEN = 2048


def _tiny_mla_config():
    from transformers import PretrainedConfig

    cfg = PretrainedConfig(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        vocab_size=VOCAB,
        rms_norm_eps=1e-6,
        max_position_embeddings=MAX_SEQ_LEN,
        rope_theta=50000.0,
        block_size=BLOCK,
        mask_token_id=VOCAB - 3,
        target_layer_ids=[0, 1],
        markov_rank=RANK,
        markov_head_type="vanilla",
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": MAX_SEQ_LEN // 4,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    cfg.dflash_config = {"mask_token_id": VOCAB - 3, "target_layer_ids": [0, 1]}
    cfg.torch_dtype = torch.bfloat16
    return cfg


def _tiny_mla_weights(seed=11):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    w = {
        "context_proj.weight": rnd(HIDDEN, HIDDEN * NUM_CAPTURE),
        "context_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "final_norm.weight": rnd(HIDDEN).abs() + 1.0,
        "embed_tokens.weight": rnd(VOCAB, HIDDEN),
        "markov_head.markov_w1.weight": rnd(VOCAB, RANK),
        "markov_head.markov_w2.weight": rnd(VOCAB, RANK),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "post_attention_layernorm.weight"] = rnd(HIDDEN).abs() + 1.0
        w[p + "self_attn.q_a_proj.weight"] = rnd(Q_LORA, HIDDEN)
        w[p + "self_attn.q_a_layernorm.weight"] = rnd(Q_LORA).abs() + 1.0
        w[p + "self_attn.q_b_proj.weight"] = rnd(HEADS * (NOPE + ROPE), Q_LORA)
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rnd(LATENT, HIDDEN)
        w[p + "self_attn.kv_a_layernorm.weight"] = rnd(KV_LORA).abs() + 1.0
        w[p + "self_attn.kv_b_proj.weight"] = rnd(HEADS * (NOPE + V_DIM), KV_LORA)
        w[p + "self_attn.o_proj.weight"] = rnd(HIDDEN, HEADS * V_DIM)
        w[p + "mlp.gate_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.up_proj.weight"] = rnd(INTERMEDIATE, HIDDEN)
        w[p + "mlp.down_proj.weight"] = rnd(HIDDEN, INTERMEDIATE)
    return w


def _build_drafter(weights):
    from tensorrt_llm._torch.model_config import ModelConfig

    model_config = ModelConfig(pretrained_config=_tiny_mla_config(), attn_backend="TRTLLM")
    drafter = MLADSparkForCausalLM(model_config, dflash_attention_backend="CUTEDSL").to("cuda")
    drafter.load_weights(dict(weights))
    drafter._build_fused_kv_buffers()
    return drafter


# ---------------------------------------------------------------------------
# Owner-masked scatter.
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("rows", [1, 7, 56, 130])
def test_scatter_rows_to_pool_rows_mask_matches_the_eager_round_trip(dtype, rows):
    torch.manual_seed(rows)
    pages, page_size = 40, 16
    pool = (torch.randn(pages, page_size, LATENT, device="cuda") * 300).to(dtype)
    slots = torch.randperm(pages * page_size, device="cuda")[:rows]
    page_idx, offsets = slots // page_size, slots % page_size
    src = (torch.randn(rows, LATENT, device="cuda") * 300).bfloat16()
    src[0, :5] = torch.tensor([-0.0, 1e5, -1e5, float("inf"), 448.0])
    mask = torch.rand(rows, device="cuda") < 0.5
    mask[0] = True

    expected = pool.clone()
    view = expected.view(torch.uint8) if dtype != torch.bfloat16 else expected
    blk = to_draft_fp8(src).view(torch.uint8) if dtype != torch.bfloat16 else src
    view[page_idx, offsets] = torch.where(mask.view(-1, 1), blk, view[page_idx, offsets])

    for rows_mask in (mask, mask.view(rows, 1, 1).to(torch.int8)):
        got = pool.clone()
        scatter_rows_to_pool(src, got, page_idx, offsets, rows_mask=rows_mask)
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    # Without a mask every row is stored, as before.
    got = pool.clone()
    scatter_rows_to_pool(src, got, page_idx, offsets)
    full = pool.clone()
    full_view = full.view(torch.uint8) if dtype != torch.bfloat16 else full
    full_view[page_idx, offsets] = blk
    assert torch.equal(got.view(torch.uint8), full.view(torch.uint8))


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1,), (7, 64, 576), (3, 5, 4097)])
def test_to_draft_fp8_1k_matches_to_draft_fp8(dtype, shape):
    torch.manual_seed(1)
    x = (torch.randn(shape, device="cuda") * 200).to(dtype)
    flat = x.view(-1)
    specials = [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 448.0, 449.0, -464.0, 1e-9]
    flat[: min(flat.numel(), len(specials))] = torch.tensor(specials[: flat.numel()], dtype=dtype)
    got = to_draft_fp8_1k(x)
    assert got.shape == x.shape and got.dtype == torch.float8_e4m3fn
    assert torch.equal(got.view(torch.uint8), to_draft_fp8(x).view(torch.uint8))


# ---------------------------------------------------------------------------
# Helix drafter forward: fused block store vs the separate launches.
# ---------------------------------------------------------------------------


def _helix_case(drafter, cp_size, cp_rank, fp8, ctx_lens, seed):
    """One rank's sharded paged pools (garbage in every slot it does not own) and the forward
    arguments."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(ctx_lens)
    shard = HelixDraftShard(cp_size, cp_rank, PAGE)
    max_pages = -(-(max(ctx_lens) + BLOCK) // (PAGE * cp_size)) + 1
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    pools = [
        (torch.randn(batch * max_pages, 1, 1, PAGE, LATENT, device="cuda", generator=gen) * 3).to(
            dtype
        )
        for _ in range(LAYERS)
    ]
    tables = torch.arange(batch * max_pages, device="cuda", dtype=torch.int32).view(
        batch, max_pages
    )
    for b, n in enumerate(ctx_lens):
        captured = torch.randn(n, HIDDEN * NUM_CAPTURE, device="cuda", generator=gen).bfloat16()
        latent, _ = drafter.precompute_context_kv(
            drafter.project_target_hidden(captured), torch.arange(n, device="cuda")
        )
        pos, owned = shard.local_slots(torch.arange(n, device="cuda"))
        pos, latent = pos[owned], latent[owned]
        for layer in range(LAYERS):
            rows = latent[:, layer, 0]
            # Byte views: index_put support for FP8 varies across builds.
            dst = pools[layer].view(torch.uint8) if fp8 else pools[layer]
            dst[tables[b].long()[pos // PAGE], 0, 0, pos % PAGE] = (
                to_draft_fp8(rows).view(torch.uint8) if fp8 else rows
            )
    kwargs = dict(
        noise_embedding=torch.randn(batch, BLOCK, HIDDEN, device="cuda", generator=gen).bfloat16(),
        query_positions=torch.stack([torch.arange(n, n + BLOCK, device="cuda") for n in ctx_lens]),
        num_ctx_per_req=torch.tensor(ctx_lens, device="cuda"),
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(batch, device="cuda"),
        ctx_page_table=tables,
    )
    return pools, kwargs


@needs_sm100f
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16_pool", "fp8_pool"])
@pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 2), (16, 5)])
def test_helix_fused_block_store_matches_the_separate_launches(monkeypatch, cp_size, cp_rank, fp8):
    monkeypatch.setattr(input_commit, "_K31_ATTENTION_DP", False)
    monkeypatch.setattr(input_commit, "_K31_GENERATION", False)
    drafter = _build_drafter(_tiny_mla_weights())
    drafter._helix_mapping = Mapping(
        world_size=cp_size,
        rank=cp_rank,
        cp_size=cp_size,
        tp_size=1,
        cp_config={"cp_type": CpType.HELIX},
    )
    # Contexts that end on, before and after a page boundary of this rank, and a block that
    # straddles two ranks' pages.
    ctx_lens = [PAGE * cp_size + 3, 37, PAGE - 2, 5 * PAGE + 30]
    exchanged = []

    def exchange_standin(partial_o, stats, mapping, num_heads, value_dim, zero_kv_mask=None, **_):
        # Records what the exchange is given; returns this rank's partials with the rows it holds
        # no key of zeroed, as the exchange's sender does.
        assert mapping is drafter._helix_mapping
        exchanged.append((partial_o.clone(), stats.clone(), zero_kv_mask.clone()))
        return partial_o.masked_fill(zero_kv_mask.view(-1, 1), 0.0)

    monkeypatch.setattr(modeling_dspark, "_helix_post_process", exchange_standin)
    fused_calls = []
    fused_rope = modeling_dspark.dspark_mla_block_rope

    def counted_rope(*args, **kwargs):
        fused_calls.append(1)
        return fused_rope(*args, **kwargs)

    monkeypatch.setattr(modeling_dspark, "dspark_mla_block_rope", counted_rope)

    results = []
    for store in ("0", "1"):
        # "1": the fused tail with the fused draft store; "0": the separate launches.
        monkeypatch.setattr(input_commit, "_K31_DRAFTS", store == "1")
        pools, kwargs = _helix_case(drafter, cp_size, cp_rank, fp8, ctx_lens, seed=cp_rank)
        exchanged.clear()
        fused_calls.clear()
        with torch.no_grad():
            out = drafter.dflash_forward(ctx_k_cache=pools[0], ctx_kv_cache=pools, **kwargs)
        torch.cuda.synchronize()
        # The switch is what selects the fused kernel under helix.
        assert len(fused_calls) == (LAYERS if store == "1" else 0)
        results.append((out, [p.clone() for p in pools], list(exchanged)))

    (out_a, pools_a, ex_a), (out_b, pools_b, ex_b) = results
    assert torch.equal(out_a.view(torch.int16), out_b.view(torch.int16))
    for pa, pb in zip(pools_a, pools_b):
        assert torch.equal(pa.view(torch.uint8), pb.view(torch.uint8))
    assert len(ex_a) == len(ex_b) == LAYERS
    for (oa, sa, ma), (ob, sb, mb) in zip(ex_a, ex_b):
        assert torch.equal(oa.view(torch.int16), ob.view(torch.int16))
        assert torch.equal(sa.view(torch.int32), sb.view(torch.int32))
        assert torch.equal(ma, mb)
