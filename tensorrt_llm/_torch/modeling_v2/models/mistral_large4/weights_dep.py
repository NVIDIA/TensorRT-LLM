# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: mistral-large-4-nvfp4 / dep, the language model.

The checkpoint is Mistral-native (`consolidated-*.safetensors`, `layers.N.attention.wq`, ...)
and ModelOpt `MIXED_PRECISION`: the routed experts NVFP4, the other big linears FP8 128x128
block (`FP8_PB_WO`), everything else bf16. Every rank is handed the whole of it, and under
attention data parallelism the topology divides the requests, not the weights, so the table
makes one split only:

* **expert windows** -- rank `r` (its `moe_ep_rank`) loads routed experts `[Er, Er + E)`,
  `E = n_routed_experts / moe_ep_size` whole experts (32 at 4 ranks, 16 at 8 ranks over two
  nodes of four), the windows tiling the 128 once;
* **replicated, whole** -- everything else: `wq`, `wkv_b`, `wo` and the MLA SSMax `s`/`b` over
  all 64 heads; `wkv_a_with_mqa` and `kv_a_norm`; the whole indexer; the dense layer's and the
  shared expert's `w1`/`w3`/`w2` at full width, which the forward runs on each rank's own
  tokens; the router, the embedding, every norm and `lerope.alpha`; and the shell's `lm_head`,
  which under attention DP is the full `[vocab, hidden]` on every rank.

So no row slices one of its checkpoint tensors: no row has a `shard`, each source is read whole,
and only the expert rows choose, by rank, which tensors they read.

Layer sets come from the config: `wkv_a_with_mqa` and `kv_a_norm` exist on the KV-owning layers
only (`kv_source_layers[i] == i`; the sharers read their source's latent), the indexer on the
`index_sharing == 'full'` layers only, the dense MLP on the first `first_k_dense_replace`
layers and the router, shared expert and routed experts on the rest.

Load-time layouts, each the one its consumer reads:

* `wkv_b` regrouped per head `[k_nope | v]` -> `[all k_nope ; all v]` over all heads (k_nope is
  64 wide on RoPE layers and 128 on NoPE layers);
* the dense MLP and the shared expert `[w1 ; w3]`, gate rows first, the half order
  `flashinfer_silu_and_mul` reads;
* FP8 linears (`wq`, `wo`, the indexer's `wq_b`, the dense and shared `w1`/`w3`/`w2`): e4m3
  `[N, K]` as stored, and their block scales in DeepGEMM's packed UE8M0 operand (see
  `weight_table`); at full width every K is a whole number of words, so there is no padding;
* the indexer's `wk` and `weights_proj` stacked into one `[128 + 32, hidden]` bf16 GEMM operand;
  its `k_norm` and `lerope.alpha` widened to fp32;
* the routed experts, this rank's window, in the NVFP4 MoE runner's layout (see
  `weight_table`).

The per-tensor NVFP4 scalars are checked over all of a layer's experts, the other ranks'
included, so one NVFP4 quantization of the gathered activations is right for every expert on
every rank and the windows' partial outputs sum.

The BF16 vision encoder, pre-projector norm, patch merger and language adapter are
replicated on every rank when the configuration declares a vision encoder.

`expected_unconsumed` accounts for off-rank routed experts and disabled DSpark draft weights.
Off-rank expert scalars are still read to check shared scales; their codes and block scales
are never read.
At `moe_ep_size` 1 every expert belongs to the rank and is read, sliced.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from .weight_table import (
    ALL_LAYERS,
    E2M1_PER_BYTE,
    F8,
    F32,
    I32,
    NVFP4_BLOCK,
    U8,
    VISION_WEIGHTS,
    MistralLarge4Weights,
    W,
    block_sf,
    checkpoint_dims,
    dense_layers,
    draft_layers,
    draft_source,
    expert_window,
    fc1_codes,
    fc1_scales,
    fc2_codes,
    fc2_scales,
    fp8_linear,
    indexer_layers,
    moe_layers,
    nope_layers,
    nope_owners,
    owners,
    rope_layers,
    rope_owners,
    sf_words,
    shared_scalar,
    stack,
    to_fp32,
    window_scalar,
)


def _regroup_kv_b(t: torch.Tensor, core) -> torch.Tensor:
    """All heads' `[heads * (k_nope + v), kv_lora]` rows, per-head `[k_nope | v]` ->
    `[heads * k_nope ; heads * v]`: the K-B operand, then the V-B one. k_nope is read off the
    tensor: 64 on RoPE layers, 128 on NoPE layers."""
    cfg = core.model_config.pretrained_config
    heads = cfg.num_attention_heads
    v = cfg.v_head_dim
    t3 = t.reshape(heads, t.shape[0] // heads, t.shape[1])
    nope = t3.shape[1] - v
    return torch.cat(
        [t3[:, :nope, :].reshape(heads * nope, -1), t3[:, nope:, :].reshape(heads * v, -1)], dim=0
    ).contiguous()


def _mlp(module, inter, layers):
    """A SwiGLU MLP at full width: `[w1 ; w3]` stacked by output rows, gate first (the half
    order `flashinfer_silu_and_mul` reads), and `w2`. The scales follow; in the scale storage
    the stacked axis is the second one."""
    p = "{p}." + module
    return (
        W(
            "mlp_gu",
            shape=lambda d: (2 * inter(d), d.hidden),
            dtype=F8,
            src=stack(
                (f"{p}.w1.weight", lambda d: (slice(0, inter(d)),)),
                (f"{p}.w3.weight", lambda d: (slice(inter(d), 2 * inter(d)),)),
            ),
            layers=layers,
        ),
        W(
            "mlp_gu_s",
            shape=lambda d: (sf_words(d.hidden), 2 * inter(d)),
            dtype=I32,
            src=stack(
                (f"{p}.w1.weight_scale", lambda d: (slice(None), slice(0, inter(d)))),
                (f"{p}.w3.weight_scale", lambda d: (slice(None), slice(inter(d), 2 * inter(d)))),
            ),
            transform=block_sf,
            layers=layers,
        ),
        *fp8_linear("mlp_dn", f"{p}.w2.weight", lambda d: d.hidden, inter, layers),
    )


def _kv_b(nope, layers):
    """`wkv_b`, all heads, regrouped. Per-head rows are `nope(d) + v` wide."""
    return W(
        "wkv_b",
        shape=lambda d: (d.heads * (nope(d) + d.v_dim), d.kv_lora),
        src="{p}.attention.wkv_b.weight",
        transform=_regroup_kv_b,
        layers=layers,
    )


class MistralLarge4DepWeights(MistralLarge4Weights):
    """mistral-large-4-nvfp4 / dep.

    When `load` streams from the shard files, it reads this rank's expert window and the
    replicated rest only; reading through the engine's mappings instead would fault the expert
    windows of a node's four ranks -- 539 GiB, the whole expert stack, at 4 ranks; half of it on
    each of the two nodes at 8 -- and the replicated rest into that node's host memory.
    """

    WEIGHTS: tuple[W, ...] = (
        # --- MLA: every layer, KV owners only where marked ---
        W("attn_norm", shape=lambda d: (d.hidden,), src="{p}.attention_norm.weight"),
        # Per head `[nope 64 | rope 64]` on RoPE layers, 128 nope on NoPE layers: 128 either way.
        *fp8_linear(
            "wq",
            "{p}.attention.wq.weight",
            lambda d: d.heads * d.qk_dim,
            lambda d: d.hidden,
            ALL_LAYERS,
        ),
        W("ssmax_s", shape=lambda d: (d.heads,), src="{p}.attention.ssmax_scaling.s"),
        W("ssmax_b", shape=lambda d: (d.heads,), src="{p}.attention.ssmax_scaling.b"),
        # `[c_kv 512 | k_pe 64]` on RoPE owners; NoPE owners project the 512 latent alone.
        W(
            "wkv_a",
            shape=lambda d: (d.lat_dim, d.hidden),
            src="{p}.attention.wkv_a_with_mqa.weight",
            layers=rope_owners,
        ),
        W(
            "wkv_a",
            shape=lambda d: (d.kv_lora, d.hidden),
            src="{p}.attention.wkv_a_with_mqa.weight",
            layers=nope_owners,
        ),
        W(
            "kv_a_norm",
            shape=lambda d: (d.kv_lora,),
            src="{p}.attention.kv_a_norm.weight",
            layers=owners,
        ),
        # Every layer absorbs through its own `wkv_b`, KV sharers included.
        _kv_b(lambda d: d.nope, rope_layers),
        _kv_b(lambda d: d.qk_dim, nope_layers),
        *fp8_linear(
            "wo",
            "{p}.attention.wo.weight",
            lambda d: d.hidden,
            lambda d: d.heads * d.v_dim,
            ALL_LAYERS,
        ),
        W("ffn_norm", shape=lambda d: (d.hidden,), src="{p}.ffn_norm.weight"),
        # --- the DSA indexer, on the 'full' layers ---
        W(
            "ix_q_norm_w",
            shape=lambda d: (d.hidden,),
            src="{p}.attention.indexer.q_input_norm.weight",
            layers=indexer_layers,
        ),
        W(
            "ix_q_norm_b",
            shape=lambda d: (d.hidden,),
            src="{p}.attention.indexer.q_input_norm.bias",
            layers=indexer_layers,
        ),
        # Per head `[rope 64 | nope 64]`, the reverse of MLA's.
        *fp8_linear(
            "ix_wq_b",
            "{p}.attention.indexer.wq_b.weight",
            lambda d: d.idx_heads * d.idx_dim,
            lambda d: d.hidden,
            indexer_layers,
        ),
        # `[wk ; weights_proj]`: the key and the head weights, one GEMM.
        W(
            "ix_wk_wp",
            shape=lambda d: (d.idx_dim + d.idx_heads, d.hidden),
            src=stack(
                ("{p}.attention.indexer.wk.weight", lambda d: (slice(0, d.idx_dim),)),
                (
                    "{p}.attention.indexer.weights_proj.weight",
                    lambda d: (slice(d.idx_dim, d.idx_dim + d.idx_heads),),
                ),
            ),
            layers=indexer_layers,
        ),
        # fp32, as the fused key entry (LayerNorm in fp32) takes them.
        W(
            "ix_k_norm_w",
            shape=lambda d: (d.idx_dim,),
            dtype=F32,
            src="{p}.attention.indexer.k_norm.weight",
            transform=to_fp32,
            layers=indexer_layers,
        ),
        W(
            "ix_k_norm_b",
            shape=lambda d: (d.idx_dim,),
            dtype=F32,
            src="{p}.attention.indexer.k_norm.bias",
            transform=to_fp32,
            layers=indexer_layers,
        ),
        W(
            "ix_ssmax_s",
            shape=lambda d: (d.idx_heads,),
            src="{p}.attention.indexer.ssmax_scaling.s",
            layers=indexer_layers,
        ),
        W(
            "ix_ssmax_b",
            shape=lambda d: (d.idx_heads,),
            src="{p}.attention.indexer.ssmax_scaling.b",
            layers=indexer_layers,
        ),
        # --- the SwiGLU MLP at full width: the dense layer's own, then each MoE layer's shared
        # expert, one key ---
        *_mlp("feed_forward", lambda d: d.dense_inter, dense_layers),
        *_mlp("shared_experts", lambda d: d.shared_inter, moe_layers),
        # --- routing ---
        W(
            "router",
            shape=lambda d: (d.num_experts, d.hidden),
            src="{p}.gate.weight",
            layers=moe_layers,
        ),
        # --- this rank's routed experts, NVFP4: `[w3 ; w1]` as stored, then each expert in the
        # MoE runner's layout ---
        W(
            "fc1_w",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // E2M1_PER_BYTE),
            dtype=U8,
            src=expert_window(
                ("w3.weight", lambda d: (slice(0, d.moe_inter),)),
                ("w1.weight", lambda d: (slice(d.moe_inter, 2 * d.moe_inter),)),
            ),
            layers=moe_layers,
            layout=fc1_codes,
        ),
        W(
            "fc1_s",
            shape=lambda d: (d.local_experts, 2 * d.moe_inter, d.hidden // NVFP4_BLOCK),
            dtype=F8,
            src=expert_window(
                ("w3.weight_scale", lambda d: (slice(0, d.moe_inter),)),
                ("w1.weight_scale", lambda d: (slice(d.moe_inter, 2 * d.moe_inter),)),
            ),
            layers=moe_layers,
            layout=fc1_scales,
        ),
        W(
            "fc2_w",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // E2M1_PER_BYTE),
            dtype=U8,
            src=expert_window(("w2.weight", lambda d: ())),
            layers=moe_layers,
            layout=fc2_codes,
        ),
        W(
            "fc2_s",
            shape=lambda d: (d.local_experts, d.hidden, d.moe_inter // NVFP4_BLOCK),
            dtype=F8,
            src=expert_window(("w2.weight_scale", lambda d: ())),
            layers=moe_layers,
            layout=fc2_scales,
        ),
        # The per-tensor scalars: the layer's one FC1 and one FC2 activation scale, and each
        # local expert's global weight scales (`w3`'s are `w1`'s).
        W(
            "isc1",
            shape=lambda d: (1,),
            dtype=F32,
            src=window_scalar("w1.input_scale", "w3.input_scale"),
            transform=shared_scalar,
            layers=moe_layers,
        ),
        W(
            "isc2",
            shape=lambda d: (1,),
            dtype=F32,
            src=window_scalar("w2.input_scale"),
            transform=shared_scalar,
            layers=moe_layers,
        ),
        W(
            "ws2_13",
            shape=lambda d: (d.local_experts,),
            dtype=F32,
            src=expert_window((("w1.weight_scale_2", "w3.weight_scale_2"), lambda d: ())),
            transform=shared_scalar,
            layers=moe_layers,
        ),
        W(
            "ws2_2",
            shape=lambda d: (d.local_experts,),
            dtype=F32,
            src=expert_window(("w2.weight_scale_2", lambda d: ())),
            layers=moe_layers,
        ),
        # --- not per-layer ---
        W("final_norm", shape=lambda d: (d.hidden,), src="norm.weight", layers=None),
        W("embed", shape=lambda d: (d.vocab, d.hidden), src="tok_embeddings.weight", layers=None),
        # The LeRoPE per-frequency log-scales the rope tables are built from, fp32 as the
        # reference widens them.
        W(
            "lerope_alpha",
            shape=lambda d: (d.rope_dim // 2,),
            dtype=F32,
            src="lerope.alpha",
            transform=to_fp32,
            layers=None,
        ),
        *VISION_WEIGHTS,
    )

    # BF16 draft weights, replicated whole like the rest of the attention and the dense MLP.
    DRAFT_WEIGHTS: tuple[W, ...] = (
        W(
            "draft_attn_norm",
            lambda d: (d.hidden,),
            draft_source("attention_norm.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_wq",
            lambda d: (d.heads * d.qk_dim, d.hidden),
            draft_source("attention.wq.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_wkv_a",
            lambda d: (d.lat_dim, d.hidden),
            draft_source("attention.wkv_a_with_mqa.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_kv_a_norm",
            lambda d: (d.kv_lora,),
            draft_source("attention.kv_a_norm.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_wkv_b",
            lambda d: (d.heads * (d.nope + d.v_dim), d.kv_lora),
            draft_source("attention.wkv_b.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_wo",
            lambda d: (d.hidden, d.heads * d.v_dim),
            draft_source("attention.wo.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_ffn_norm",
            lambda d: (d.hidden,),
            draft_source("ffn_norm.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_w1",
            lambda d: (d.dense_inter, d.hidden),
            draft_source("feed_forward.w1.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_w3",
            lambda d: (d.dense_inter, d.hidden),
            draft_source("feed_forward.w3.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_w2",
            lambda d: (d.hidden, d.dense_inter),
            draft_source("feed_forward.w2.weight"),
            layers=draft_layers,
        ),
        W(
            "draft_final_norm",
            lambda d: (d.hidden,),
            "mtp_dspark.backbone.final_norm.weight",
            layers=None,
        ),
        W(
            "draft_target_norm",
            lambda d: (d.hidden,),
            "mtp_dspark.backbone.target_norm.weight",
            layers=None,
        ),
        W(
            "draft_target_projection",
            lambda d: (d.hidden, d.hidden * d.draft_capture_layers),
            "mtp_dspark.backbone.target_projection.weight",
            layers=None,
        ),
        W(
            "draft_mask",
            lambda d: (d.draft_depth - 1, d.hidden),
            "mtp_dspark.backbone.mask_embedding",
            layers=None,
        ),
        W(
            "draft_markov_embedding",
            lambda d: (d.vocab, d.draft_rank),
            "mtp_dspark.autoregressive_head.embedding.weight",
            layers=None,
        ),
        W(
            "draft_markov_output",
            lambda d: (d.vocab, d.draft_rank),
            "mtp_dspark.autoregressive_head.output.weight",
            layers=None,
        ),
        W(
            "draft_joint",
            lambda d: (3 * d.draft_rank, d.hidden + 2 * d.draft_rank),
            "mtp_dspark.autoregressive_head.joint_proj.weight",
            layers=None,
        ),
        W(
            "draft_joint_bias",
            lambda d: (3 * d.draft_rank,),
            "mtp_dspark.autoregressive_head.joint_proj.bias",
            layers=None,
        ),
    )

    def lm_head_shard(self, d: SimpleNamespace):
        """Under attention DP the shell builds `lm_head` whole, `[vocab, hidden]` on every rank:
        the checkpoint's whole `output.weight`."""
        return None

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        return SimpleNamespace(
            **checkpoint_dims(core),
            draft_tp_rank=0,
            heads=cfg.num_attention_heads,
            moe_inter=cfg.moe_intermediate_size,
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            shared_inter=cfg.moe_intermediate_size * cfg.n_shared_experts,
            dense_inter=cfg.intermediate_size,
        )


MODEL_WEIGHTS = MistralLarge4DepWeights()
