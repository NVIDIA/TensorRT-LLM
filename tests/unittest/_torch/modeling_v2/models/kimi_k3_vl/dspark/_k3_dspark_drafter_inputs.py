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
"""Shared inputs of the family DSpark drafter's tests: a synthetic drafter checkpoint of the real
drafter's shapes (two layers, 64 heads, MLA 576, hidden 7168, a 4096 vocab), one draft step's inputs
over four requests with 517 / 1203 / 64 / 333 context tokens (block 7, page 32), and the drafter's
block forward over a latent pool. Every value is a hash of its index, the same on every rank and run.
"""

import torch

_K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
NEW = f"{_K3}.dspark"

HIDDEN, INTER, LAYERS, HEADS, VOCAB, RANK = 7168, 2048, 2, 64, 4096, 16
Q_LORA, KV_LORA, NOPE, ROPE, V_DIM = 1536, 512, 128, 64, 128
TARGET_LAYERS = (0, 1)
CTX_LENS = (517, 1203, 64, 333)
BLOCK, PAGE, MAX_SEQ = 7, 32, 16384


def drafter_config_dict():
    """The drafter checkpoint's config.json (Inferact/Kimi-K3-DSpark) at two layers, a 4096 vocab."""
    return dict(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_DIM,
        mla_use_nope=False,
        mla_use_output_gate=False,
        vocab_size=VOCAB,
        rms_norm_eps=1e-5,
        max_position_embeddings=1048576,
        rope_theta=50000.0,
        num_target_layers=len(TARGET_LAYERS),
        target_hidden_size=HIDDEN,
        target_layer_ids=list(TARGET_LAYERS),
        mask_token_id=VOCAB - 3,
        markov_rank=RANK,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        tie_word_embeddings=False,
        draft_vocab_size=VOCAB,
        torch_dtype="bfloat16",
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def hashed(n, seed, device):
    x = torch.arange(n, device=device, dtype=torch.int64)
    x = (x * 2654435761 + seed * 40503 + 12345) & 0xFFFFFFFF
    for _ in range(2):
        x = x ^ (x >> 16)
        x = (x * 0x45D9F3B) & 0xFFFFFFFF
    return x ^ (x >> 16)


def normalish(shape, seed, scale, device="cuda", offset=0.0):
    """Sum of four hashed uniforms in [-1, 1): bell-shaped, std ~1.15, times ``scale``; BF16. The
    same values on every rank and in every run."""
    n = torch.Size(shape).numel()
    acc = torch.zeros(n, device=device, dtype=torch.float32)
    for i in range(4):
        acc += (hashed(n, seed * 4 + i, device) % 65536).to(torch.float32) / 32768.0 - 1.0
    return (acc * scale + offset).to(torch.bfloat16).view(shape)


def drafter_weights():
    """A checkpoint of the drafter (HF MLA names under ``model.``, the TorchSpec head names)."""
    seed = iter(range(1000))
    w = {}

    def put(name, *shape, scale=0.02, offset=0.0):
        w[name] = normalish(shape, next(seed), scale, offset=offset)

    put("fc.weight", HIDDEN, len(TARGET_LAYERS) * HIDDEN, scale=0.01)
    put("hidden_norm.weight", HIDDEN, scale=0.05, offset=1.0)
    put("model.embed_tokens.weight", VOCAB, HIDDEN, scale=0.5)
    put("model.norm.weight", HIDDEN, scale=0.05, offset=1.0)
    put("markov_head.markov_w1.weight", VOCAB, RANK, scale=0.5)
    put("markov_head.markov_w2.weight", VOCAB, RANK, scale=0.5)
    put("confidence_head.proj.weight", 1, HIDDEN + RANK)
    put("confidence_head.proj.bias", 1)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        put(p + "self_attn.q_a_proj.weight", Q_LORA, HIDDEN)
        put(p + "self_attn.q_a_layernorm.weight", Q_LORA, scale=0.05, offset=1.0)
        put(p + "self_attn.q_b_proj.weight", HEADS * (NOPE + ROPE), Q_LORA, scale=0.03)
        put(p + "self_attn.kv_a_proj_with_mqa.weight", KV_LORA + ROPE, HIDDEN)
        put(p + "self_attn.kv_a_layernorm.weight", KV_LORA, scale=0.05, offset=1.0)
        put(p + "self_attn.kv_b_proj.weight", HEADS * (NOPE + V_DIM), KV_LORA, scale=0.04)
        put(p + "self_attn.o_proj.weight", HIDDEN, HEADS * V_DIM, scale=0.01)
        put(p + "mlp.gate_proj.weight", INTER, HIDDEN)
        put(p + "mlp.up_proj.weight", INTER, HIDDEN)
        put(p + "mlp.down_proj.weight", HIDDEN, INTER, scale=0.02)
        put(p + "input_layernorm.weight", HIDDEN, scale=0.05, offset=1.0)
        put(p + "post_attention_layernorm.weight", HIDDEN, scale=0.05, offset=1.0)
    return w


class Inputs:
    """One draft step's inputs and a latent pool holding the context latents, per drafter."""

    def __init__(self, drafter, dtype=torch.bfloat16):
        dev = torch.device("cuda")
        self.batch = len(CTX_LENS)
        self.ctx = torch.tensor(CTX_LENS, dtype=torch.int32, device=dev)
        n_ctx = sum(CTX_LENS)
        self.captured = normalish((n_ctx, len(TARGET_LAYERS) * HIDDEN), 1, 1.0)
        self.positions = torch.cat(
            [torch.arange(c, device=dev, dtype=torch.int32) for c in CTX_LENS]
        )
        helix = drafter._helix_mapping
        self.cp, self.cp_rank = (helix.cp_size, helix.cp_rank) if helix is not None else (1, 0)
        pages_global = [-(-(c + BLOCK) // PAGE) for c in CTX_LENS]
        local = [max(0, -(-(g - self.cp_rank) // self.cp)) for g in pages_global]
        table = torch.zeros(self.batch, max(local) + 1, dtype=torch.int32, device=dev)
        phys = list(range(1, 1 + sum(local)))[::-1]  # reversed: a non-trivial page table
        i = 0
        for b, n in enumerate(local):
            table[b, :n] = torch.tensor(phys[i : i + n], dtype=torch.int32, device=dev)
            i += n
        self.table = table
        self.num_pages = 1 + sum(local)
        self.dtype = dtype
        mask_id = int(drafter.mask_token_id)
        self.bonus = (hashed(self.batch, 21, dev) % (VOCAB - 3)).to(torch.int64)
        ids = torch.full((self.batch, BLOCK), mask_id, dtype=torch.int64, device=dev)
        ids[:, 0] = self.bonus
        self.ids = ids
        self.query_positions = self.ctx.view(-1, 1) + torch.arange(
            BLOCK, device=dev, dtype=torch.int32
        )

    def pools(self, latent):
        """Per-layer ``[pages, 1, 1, page, 576]`` pools: hashed filler, the context latents stored
        on this rank's pages (``to_draft_fp8`` of them in an FP8 pool)."""
        from importlib import import_module

        to_fp8 = import_module(f"{NEW}.primitives").to_draft_fp8
        dev = latent.device
        req = torch.repeat_interleave(torch.arange(self.batch, device=dev), self.ctx.long())
        pos = self.positions.long()
        page_g = pos // PAGE
        owned = page_g % self.cp == self.cp_rank
        phys = self.table.long()[req, page_g // self.cp]
        pools = []
        for layer in range(latent.shape[1]):
            pool = normalish((self.num_pages, 1, 1, PAGE, KV_LORA + ROPE), 100 + layer, 1.0)
            pool[phys[owned], 0, 0, (pos % PAGE)[owned]] = latent[owned, layer, 0]
            if self.dtype == torch.float8_e4m3fn:
                pool = to_fp8(pool)
            pools.append(pool)
        return pools


def forward(drafter, inp, pools, noise, variant, fixup, placement):
    kwargs = {}
    if variant == "placement":
        kwargs["block_placement"] = placement
    elif variant == "fixup":
        kwargs["block_fixup"] = fixup
    return drafter.dflash_forward(
        noise_embedding=noise,
        query_positions=inp.query_positions,
        num_ctx_per_req=inp.ctx,
        ctx_k_cache=torch.empty(0, device="cuda"),
        ctx_v_cache=None,
        ctx_cache_batch_idx=torch.arange(inp.batch, device="cuda"),
        ctx_kv_cache=pools,
        ctx_page_table=inp.table,
        **kwargs,
    )
