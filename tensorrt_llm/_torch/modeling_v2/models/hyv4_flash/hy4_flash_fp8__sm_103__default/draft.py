# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFly2 drafter: five block-parallel MLA layers over the target's own sparse pools.

Draft layer `i` attends the picks of target layer `dims.target_kv_layers[i]` plus the block's own rows,
which it writes into that target layer's pool, so the drafter keeps no cache of its own. Column `s` of
a block is corrected with the token sampled at `s - 1`; the worker walks that.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn as nn

from . import weights as _weights
from .dims import dims, draft_dims
from .modeling import _INERT, MlaLayer, _rms
from .ops.activation.flashinfer_silu_and_mul__r18 import flashinfer_silu_and_mul
from .ops.attention.mla_rope_generation__mla_decode_prologue import mla_rope_generation
from .ops.attention.thop_attention__mla_windowed import MLA_GENERATION, thop_attention
from .ops.gemm.block_fp8_gemm import block_fp8_gemm
from .ops.gemm.bmm_out__r16 import bmm_out
from .ops.gemm.cublas_mm__r11 import cublas_mm
from .ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128
from .ops.spec.dfly_block_prepare import BLOCK_WIDTH, MASK_TOKEN_ID
from .ops.spec.dfly_fusion import DflyFusion, DflyHiddenCorrection

#: The checkpoint namespace the draft read keeps.
DRAFT_KEY_PREFIX = "model.dfly_layers."


class DflyBlockSelection(NamedTuple):
    """Where one block sits in the target's pool and what its rows attend (the worker builds it)."""

    #: int32 `[layers, rows, index_width]`: pool rows per draft layer, picks then the block's own
    #: slots, `-1` padded.
    packed: torch.Tensor
    #: int32 `[layers, rows]`: live columns of each row of `packed`.
    lengths: torch.Tensor
    #: int32 `[gens, width]`: the block's pool slots, `PAD_SLOT_ID` if inactive.
    slots: torch.Tensor
    #: int64 `[rows]`: the block's positions.
    positions: torch.Tensor
    #: bool `[gens]`: requests that draft this step.
    active: torch.Tensor
    #: The target's attention metadata for this step.
    attn_metadata: object


def draft_bundle(core) -> SimpleNamespace:
    """`draft_dims` plus the per-layer geometry the reused MLA layer reads."""
    d = draft_dims(core)
    c = dims(core)
    # The target sparse layer's geometry without the indexer: the draft selects with its picks.
    geom = c.layers[d.target_kv_layers[0]]._replace(sparse=False)
    return SimpleNamespace(
        **vars(d),
        layers=(geom,) * d.num_layers,
        tokens_per_block=c.tokens_per_block,
        max_seq_len=c.max_seq_len,
        max_batch=c.max_batch,
        max_context_len=c.max_context_len,
    )


class Hyv4FlashDraft(nn.Module):
    """The DFly2 drafter, at one rank's share of one deployment."""

    block_size = BLOCK_WIDTH
    mask_token_id = MASK_TOKEN_ID
    dflash_attention_backend = "VANILLA"
    # One coherent path with a hidden correction: the engine must take the plain block path.
    has_candidate_selector = False
    is_dflash2 = False
    lilicorr = None
    checkpoint_key_prefixes = (DRAFT_KEY_PREFIX,)

    def __init__(self, core):
        super().__init__()
        # Only the core, never the shell: module walks recurse `children()` without a memo, so a
        # reference back to the shell is an unending cycle.
        self.core = core
        self.model_config = core.model_config
        self.dims = draft_bundle(core)
        self.w = _weights.DRAFT_WEIGHTS.declare(self.dims)
        # Rows per request the reused MLA layer reads off its host: the whole block.
        self.gen_width = self.dims.block_width
        self.layers = [MlaLayer(self, i) for i in range(self.dims.num_layers)]
        self._fusion = DflyFusion()
        self._correction = DflyHiddenCorrection()
        self._rope: dict | None = None
        self._rope_positions = 0
        # The target's logits processor and LM head, bound by the shell.
        self.logits_processor = None
        self.lm_head = None

    @property
    def embed_tokens_weight(self) -> torch.Tensor:
        """The target's embedding table: the drafter owns none."""
        return self.core.w["embed"]

    def _pool_row(self, md, layer: int) -> int:
        """A draft layer's pool row: its target layer's."""
        return self.dims.target_kv_layers[layer]

    def _allreduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.core._allreduce(x)

    def load_weights(self, weights, weight_mapper=None, allow_partial_loading=False) -> None:
        del weight_mapper, allow_partial_loading
        _weights.DRAFT_WEIGHTS.load(self, weights)

    def post_load_weights(self) -> None:
        return None

    def _rope_table(self, md, device) -> dict:
        """The target sparse layers' RoPE table, borrowed from the core and grown to fit."""
        wanted = int(getattr(md, "max_seq_len", 0) or 0) or self.dims.max_seq_len
        wanted = min(wanted, int(self.core.model_config.pretrained_config.max_position_embeddings))
        if self._rope is None or wanted > self._rope_positions:
            self._rope_positions = max(wanted, 8192)
            self._rope = self.core._rope_tables(device, self._rope_positions, self.dims.theta)
        return self._rope

    def project_target_hidden(
        self, anchors: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """The column-0 fusion of the anchor embedding with the target's hidden state."""
        d = self.dims
        return self._fusion(
            anchors,
            target_hidden,
            self.w["fusion_enorm"],
            self.w["fusion_hnorm"],
            self.w["fusion_eh_proj"],
            self.w["fusion_gate"],
            d.rms_eps,
        )

    def apply_draft_logit_transform(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _expand_q(self, q_resid: torch.Tensor, i: int) -> torch.Tensor:
        w = self.w
        q, s = fp8_quantize_1x128(q_resid)
        flat = block_fp8_gemm(
            q, w[f"l{i}_q_b_proj"], s, w[f"l{i}_q_b_proj_scale"], output_dtype=q_resid.dtype
        )
        return flat.view(-1, self.dims.heads_local, self.dims.qk_dim)

    def _linear(self, x, weight, scale) -> torch.Tensor:
        """Block-FP8 where the table declares scales, bf16 otherwise."""
        w = self.w
        if scale in w:
            q, s = fp8_quantize_1x128(x)
            return block_fp8_gemm(q, w[weight], s, w[scale], output_dtype=x.dtype)
        return cublas_mm(x, w[weight].t())

    def _mlp(self, x: torch.Tensor, i: int) -> torch.Tensor:
        d = self.dims
        up = flashinfer_silu_and_mul(
            torch.cat(
                [
                    self._linear(x, f"l{i}_mlp_gate", f"l{i}_mlp_gate_scale"),
                    self._linear(x, f"l{i}_mlp_up", f"l{i}_mlp_up_scale"),
                ],
                dim=-1,
            )
        )
        out = self._linear(up, f"l{i}_mlp_down", f"l{i}_mlp_down_scale")
        return self._allreduce(out) if d.mlp_tp > 1 else out

    def _block_attention(
        self, layer: MlaLayer, q, latent, md, rope, packed, lengths
    ) -> torch.Tensor:
        """`MlaLayer.generation_partial` over `block_width` rows per request.

        The sparse kernel derives row `j`'s key count as `sequence_length - width + 1 + j`, so each
        generating request gets the live count of its block's last row from `lengths`; the rope
        prologue keeps the step's device lengths, which place the block's rows.
        """
        d, i = self.dims, layer.layer
        width = d.block_width
        num_tokens = q.shape[0]
        q_pe = q[..., d.nope :]
        absorbed = layer._absorbed_q(q, q_pe)
        device = q.device
        cu_q = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        cu_kv = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        counter = torch.empty(1, dtype=torch.uint32, device=device)
        mla_rope_generation(
            absorbed.view(num_tokens, -1),
            q_pe,
            latent,
            rope["rotary_cos_sin"],
            cu_q,
            cu_kv,
            counter,
            None,
            None,
            None,
            md.kv_lens_cuda_runtime,
            md.kv_lens_runtime,
            md.prompt_lens_cpu_runtime,
            int(md.num_contexts),
            md.kv_cache_block_offsets,
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            None,
            None,
            None,
            None,
            None,
            [None, None],
            width,  # predicted_tokens_per_seq
            self._pool_row(md, i),
            d.heads_local,
            1,
            d.lat_dim,
            0,
            d.tokens_per_block,
            layer._window(md),
            1,
            0,
            d.q_scaling,
            d.q_lora,
            d.kv_lora,
            d.nope,
            d.rope_dim,
            d.v_dim,
            True,
            None,
            d.rms_eps,
            False,
            False,
            False,
            False,
            None,
        )
        lat = torch.empty(num_tokens, d.heads_local * d.kv_lora, dtype=q.dtype, device=device)
        args = layer._generation_args(md, rope, packed)
        args["predicted_tokens_per_seq"] = width
        seq_lens = md.kv_lens_cuda_runtime.clone()
        first = int(md.num_contexts)
        seq_lens[first : first + lengths.numel() // width] = lengths[width - 1 :: width]
        args["sequence_length"] = seq_lens
        thop_attention(
            q=absorbed.contiguous(),
            k=None,
            v=None,
            output=lat,
            is_fused_qkv=True,
            num_kv_heads=1,
            head_size=d.lat_dim,
            v_head_dim=d.kv_lora,
            attention_input_type=MLA_GENERATION,
            num_contexts=int(md.num_contexts),
            num_ctx_tokens=0,
            sparse_attn_indices=packed,
            num_sparse_topk=None if packed is None else int(packed.shape[1]),
            latent_cache=latent,
            q_pe=q_pe,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            **args,
            **_INERT,
        )
        out = torch.empty(num_tokens, d.heads_local, d.v_dim, dtype=q.dtype, device=device)
        bmm_out(
            lat.view(num_tokens, d.heads_local, d.kv_lora).transpose(0, 1),
            self.w[f"l{i}_v_b_proj"],
            out.transpose(0, 1),
        )
        return out.view(num_tokens, d.heads_local * d.v_dim)

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: int,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache,
        ctx_page_table,
        block_placement: DflyBlockSelection,
    ) -> torch.Tensor:
        """The block's hidden states after `final_norm`; the engine's `ctx_*` arguments are unused."""
        del num_ctx_per_req, ctx_k_cache, ctx_v_cache, ctx_cache_batch_idx
        del ctx_kv_cache, ctx_page_table
        md = block_placement.attn_metadata
        d = self.dims
        rope = self._rope_table(md, noise_embedding.device)
        x = noise_embedding.reshape(-1, d.hidden).contiguous()
        for i in range(d.num_layers):
            layer = self.layers[i]
            residual = x
            h = _rms(x, self.w[f"l{i}_input_layernorm"], d.rms_eps)
            q_resid = _rms(
                cublas_mm(h, self.w[f"l{i}_q_a_proj"].t()),
                self.w[f"l{i}_q_a_layernorm"],
                d.rms_eps,
            )
            kv = cublas_mm(h, self.w[f"l{i}_kv_a_proj_with_mqa"].t())
            latent = kv[:, : d.kv_lora + d.rope_dim]
            latent[:, : d.kv_lora] = _rms(
                latent[:, : d.kv_lora], self.w[f"l{i}_kv_a_layernorm"], d.rms_eps
            )
            q = self._expand_q(q_resid, i)
            attn = self._block_attention(
                layer,
                q,
                latent,
                md,
                rope,
                block_placement.packed[i],
                block_placement.lengths[i],
            )
            x = residual + layer.epilogue(attn, h)
            residual = x
            h = _rms(x, self.w[f"l{i}_post_attention_layernorm"], d.rms_eps)
            x = residual + self._mlp(h, i)
        return _rms(x, self.w["final_norm"], d.rms_eps)

    def hidden_correction(
        self, hidden: torch.Tensor, previous_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(hidden + mlp([norm(h) | norm(embed(prev))]), confidence)` for one block column."""
        d = self.dims
        w = self.w
        previous = torch.nn.functional.embedding(previous_tokens.long(), self.embed_tokens_weight)
        return self._correction(
            hidden,
            previous,
            w["corr_hidden_norm"],
            w["corr_embed_norm"],
            w["corr_gate"],
            w["corr_up"],
            w["corr_down"],
            w["conf_proj"],
            w["conf_bias"],
            d.rms_eps,
        )

    def block_logits(self, hidden: torch.Tensor, attn_metadata) -> torch.Tensor:
        """The target's LM head, through its logits processor, on a corrected (already normed) row."""
        return self.logits_processor(hidden, self.lm_head, attn_metadata, True)


__all__ = [
    "DRAFT_KEY_PREFIX",
    "DflyBlockSelection",
    "Hyv4FlashDraft",
    "draft_bundle",
]
