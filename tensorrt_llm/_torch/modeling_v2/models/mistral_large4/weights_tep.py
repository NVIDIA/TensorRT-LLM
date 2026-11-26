# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight table: mistral-large-4-nvfp4 / tep, the language model.

The checkpoint is Mistral-native (`consolidated-*.safetensors`, `layers.N.attention.wq`, ...)
and ModelOpt `MIXED_PRECISION`: the routed experts NVFP4, the other big linears FP8 128x128
block (`FP8_PB_WO`), everything else bf16. Every rank is handed the whole of it, so every split
is this table's, every width read off the mapping (`tp_size` and `moe_ep_size` 4 on one node, or
8 over two nodes of four):

* **tensor-parallel slices** -- `wq` by output rows and `wo` by input columns (this rank's 16
  heads at 4 ranks, 8 heads -- 1024 rows or columns -- at 8); `wkv_b` by heads; the MLA SSMax
  `s`/`b` by heads; the dense layer's and the shared expert's `w1`/`w3` by rows and `w2` by
  columns (6144 and 1536 per rank at 4 ranks, 3072 and 768 at 8); the shell's `lm_head` by
  vocabulary rows (51200 or 25600 per rank). Every slice is whole 128-blocks, and a block scale
  is sliced on the same boundary. The slice is taken on the checkpoint entry before the copy to
  the device, so a rank moves only its own bytes. That is the `shard` column of `W`.
* **expert windows** -- rank `r` loads routed experts `[Er, Er + E)`, `E = n_routed_experts /
  moe_ep_size`: 32 at 4 ranks, 16 at 8.
* **replicated** -- `wkv_a_with_mqa` and `kv_a_norm`, the whole indexer (all 32 of its heads),
  the router, the embedding, every norm and `lerope.alpha`.

Layer sets come from the config: `wkv_a_with_mqa` and `kv_a_norm` exist on the KV-owning layers
only (`kv_source_layers[i] == i`; the sharers read their source's latent), the indexer on the
`index_sharing == 'full'` layers only, the dense MLP on the first `first_k_dense_replace`
layers and the router, shared expert and routed experts on the rest.

Load-time layouts, each the one its consumer reads:

* `wkv_b` regrouped per head `[k_nope | v]` -> `[all k_nope ; all v]` over this rank's heads
  (k_nope is 64 wide on RoPE layers and 128 on NoPE layers);
* the dense MLP and the shared expert `[w1 ; w3]`, gate rows first, the half order
  `flashinfer_silu_and_mul` reads;
* FP8 linears (`wq`, `wo`, the indexer's `wq_b`, the dense and shared `w1`/`w3`/`w2`): e4m3
  `[N, K]` as stored, and their block scales in DeepGEMM's packed UE8M0 operand (see
  `weight_table`). The shared expert's `w2` at K 768 (8 ranks) has 6 blocks in 2 words, and
  bytes 2 and 3 of the second word are zero;
* the indexer's `wk` and `weights_proj` stacked into one `[128 + 32, hidden]` bf16 GEMM operand;
  its `k_norm` and `lerope.alpha` widened to fp32;
* the routed experts, this rank's window, in the NVFP4 MoE runner's layout (see
  `weight_table`).

The per-tensor NVFP4 scalars are checked over all of a layer's experts, the other ranks'
included, so one NVFP4 quantization of the replicated activations is right for every expert on
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
    BLOCK,
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
    shard_cols,
    shard_rows,
    shard_sf_cols,
    shared_scalar,
    stack,
    to_fp32,
    window_scalar,
)


def _regroup_kv_b(t: torch.Tensor, core) -> torch.Tensor:
    """This rank's `[heads * (k_nope + v), kv_lora]` rows, per-head `[k_nope | v]` ->
    `[heads * k_nope ; heads * v]`: the K-B operand, then the V-B one. k_nope is read off the
    slice: 64 on RoPE layers, 128 on NoPE layers."""
    cfg = core.model_config.pretrained_config
    heads = cfg.num_attention_heads // core.model_config.mapping.tp_size
    v = cfg.v_head_dim
    t3 = t.reshape(heads, t.shape[0] // heads, t.shape[1])
    nope = t3.shape[1] - v
    return torch.cat(
        [t3[:, :nope, :].reshape(heads * nope, -1), t3[:, nope:, :].reshape(heads * v, -1)], dim=0
    ).contiguous()


def _tp_mlp(module, inter, layers):
    """A SwiGLU MLP split over its intermediate dimension: `[w1 ; w3]` by output rows, gate
    first (the half order `flashinfer_silu_and_mul` reads), `w2` by input columns. The same
    row slice applies to `w1` and `w3` before they are stacked, and the scales follow in
    128-blocks; in the scale storage the stacked axis is the second one."""
    blocks = lambda d: inter(d) // BLOCK  # noqa: E731
    start = lambda d: d.tp_rank * inter(d)  # noqa: E731
    b_start = lambda d: d.tp_rank * blocks(d)  # noqa: E731
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
            shard=shard_rows(start, inter),
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
            shard=shard_rows(b_start, blocks),
        ),
        *fp8_linear(
            "mlp_dn",
            f"{p}.w2.weight",
            lambda d: d.hidden,
            inter,
            layers,
            w_shard=shard_cols(start, inter),
            s_shard=shard_sf_cols(b_start, blocks),
        ),
    )


def _kv_b(nope, layers):
    """`wkv_b`, this rank's heads, regrouped. Per-head rows are `nope(d) + v` wide."""
    width = lambda d: d.heads * (nope(d) + d.v_dim)  # noqa: E731
    return W(
        "wkv_b",
        shape=lambda d: (width(d), d.kv_lora),
        src="{p}.attention.wkv_b.weight",
        transform=_regroup_kv_b,
        layers=layers,
        shard=shard_rows(lambda d: d.tp_rank * width(d), width),
    )


def _heads(width):
    """This rank's slice of a per-head axis of the given per-head width."""
    return shard_rows(lambda d: d.tp_rank * d.heads * width(d), lambda d: d.heads * width(d))


class MistralLarge4TepWeights(MistralLarge4Weights):
    """mistral-large-4-nvfp4 / tep.

    When `load` streams from the shard files, reading through the engine's mappings instead
    would fault a node's four ranks' expert windows -- 539 GiB, the whole expert stack, on one
    node of four; 270 GiB, half of it, on each node of two -- into that node's host memory.
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
            w_shard=_heads(lambda d: d.qk_dim),
            s_shard=_heads(lambda d: d.qk_dim // BLOCK),
        ),
        W(
            "ssmax_s",
            shape=lambda d: (d.heads,),
            src="{p}.attention.ssmax_scaling.s",
            shard=_heads(lambda d: 1),
        ),
        W(
            "ssmax_b",
            shape=lambda d: (d.heads,),
            src="{p}.attention.ssmax_scaling.b",
            shard=_heads(lambda d: 1),
        ),
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
            w_shard=shard_cols(
                lambda d: d.tp_rank * d.heads * d.v_dim, lambda d: d.heads * d.v_dim
            ),
            s_shard=shard_sf_cols(
                lambda d: d.tp_rank * d.heads * d.v_dim // BLOCK,
                lambda d: d.heads * d.v_dim // BLOCK,
            ),
        ),
        W("ffn_norm", shape=lambda d: (d.hidden,), src="{p}.ffn_norm.weight"),
        # --- the DSA indexer: replicated, on the 'full' layers ---
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
        # --- the TP-sliced SwiGLU MLP: the dense layer's own, then each MoE layer's shared
        # expert, one key ---
        *_tp_mlp("feed_forward", lambda d: d.dense_inter, dense_layers),
        *_tp_mlp("shared_experts", lambda d: d.shared_inter, moe_layers),
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

    # BF16 draft weights use the same attention and dense-MLP partition as
    # this target. The RNN and target-feature projection are replicated.
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
            shard=shard_rows(
                lambda d: d.draft_tp_rank * d.heads * d.qk_dim, lambda d: d.heads * d.qk_dim
            ),
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
            shard=shard_rows(
                lambda d: d.draft_tp_rank * d.heads * (d.nope + d.v_dim),
                lambda d: d.heads * (d.nope + d.v_dim),
            ),
        ),
        W(
            "draft_wo",
            lambda d: (d.hidden, d.heads * d.v_dim),
            draft_source("attention.wo.weight"),
            layers=draft_layers,
            shard=shard_cols(
                lambda d: d.draft_tp_rank * d.heads * d.v_dim, lambda d: d.heads * d.v_dim
            ),
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
            shard=shard_rows(lambda d: d.draft_tp_rank * d.dense_inter, lambda d: d.dense_inter),
        ),
        W(
            "draft_w3",
            lambda d: (d.dense_inter, d.hidden),
            draft_source("feed_forward.w3.weight"),
            layers=draft_layers,
            shard=shard_rows(lambda d: d.draft_tp_rank * d.dense_inter, lambda d: d.dense_inter),
        ),
        W(
            "draft_w2",
            lambda d: (d.hidden, d.dense_inter),
            draft_source("feed_forward.w2.weight"),
            layers=draft_layers,
            shard=shard_cols(lambda d: d.draft_tp_rank * d.dense_inter, lambda d: d.dense_inter),
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
        """The shell builds `lm_head` vocab-parallel under TP: this rank's block of rows."""
        rows = d.vocab // d.tp_size
        return shard_rows(lambda _: d.tp_rank * rows, lambda _: rows)(d)

    def dims(self, core) -> SimpleNamespace:
        cfg = core.model_config.pretrained_config
        mapping = core.model_config.mapping
        tp = mapping.tp_size
        local_experts = cfg.n_routed_experts // mapping.moe_ep_size
        return SimpleNamespace(
            **checkpoint_dims(core),
            draft_tp_rank=mapping.tp_rank,
            heads=cfg.num_attention_heads // tp,
            moe_inter=cfg.moe_intermediate_size,
            local_experts=local_experts,
            expert_offset=local_experts * mapping.moe_ep_rank,
            shared_inter=cfg.moe_intermediate_size * cfg.n_shared_experts // tp,
            dense_inter=cfg.intermediate_size // tp,
            tp_rank=mapping.tp_rank,
            tp_size=tp,
        )


MODEL_WEIGHTS = MistralLarge4TepWeights()
