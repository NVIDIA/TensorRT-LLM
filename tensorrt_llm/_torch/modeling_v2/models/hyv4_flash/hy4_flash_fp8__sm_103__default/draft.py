# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFly2 drafter: five block-parallel MLA layers over the target's own sparse pools.

One draft block is `dims.block_width` rows per generating request -- column 0 is
the anchor (the last token the target has already accepted) and columns 1..W-1
are mask-token rows. A block row is a *query*: the five draft layers run once
over all `W` columns of a request in parallel (block-parallel), and each draft
layer `i` attends

* the top-k picks target sparse layer ``dims.target_kv_layers[i]``
  (25/29/33/37/41) published through the family's
  ``Hyv4FlashCacheManager.draft_picks`` while the target ran its own step, and
* the block's own `W` rows, which the layer writes into that same target layer's
  pool at the slots ``DflyBlockPrepare`` computed for this block.

The drafter keeps no cache: its rows live in the target layer's pool, at the
slots the target's next verify step reads them back from, so the two agree by
construction and nothing draft-specific has to be evicted, reused or
transferred. That is also why the forward takes the *target's* attention
metadata: the pool, its block table and the sequence lengths are the target's.

The block is not sampled column by column off the logits alone: column `s`'s
hidden state is corrected with the embedding of the token sampled at column
`s - 1` (`DflyHiddenCorrection`), so the sampler walks the block left to right.
That walk lives in the worker's ``_sample_block_draft_tokens`` -- which the
engine's ``DFlashWorker`` says a family whose positions condition on the
previous draft overrides -- and this module exposes the two pieces it needs:
:meth:`Hyv4FlashDraft.hidden_correction` (one step) and
:meth:`Hyv4FlashDraft.block_logits` (the target's LM head on the corrected row).

The engine reuses the target's LM head and embedding, so the drafter owns only
its five layers and four heads (`weights.DRAFT_WEIGHTS`), and the shell wires it
to the worker: ``draft_model``, ``draft_config = None`` (an embedded draft has
no separate architecture, so the target's weight mapper is reused),
``checkpoint_key_prefixes`` (the draft read is the target's directory narrowed
to ``model.dfly_layers.``), and ``load_draft_weights``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import torch
import torch.nn as nn

from . import weights as _weights
from .dims import DEFAULT_BLOCK_WIDTH, dims, draft_dims
from .modeling import _INERT, MlaLayer, _rms
from .ops.attention.mla_rope_generation__mla_decode_prologue import mla_rope_generation
from .ops.attention.thop_attention__mla_windowed import MLA_GENERATION, thop_attention
from .ops.gemm.block_fp8_gemm import block_fp8_gemm
from .ops.gemm.bmm_out__r16 import bmm_out
from .ops.gemm.cublas_mm__r11 import cublas_mm
from .ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128
from .ops.spec.dfly_block_prepare import BLOCK_WIDTH, MASK_TOKEN_ID
from .ops.spec.dfly_fusion import DflyFusion, DflyHiddenCorrection

#: The checkpoint's `model.dfly_layers.` namespace: what a draft read keeps and
#: what marks the draft as living in the target's own checkpoint.
DRAFT_KEY_PREFIX = "model.dfly_layers."


class DflyBlockSelection(NamedTuple):
    """What the drafter reads per layer for one block, plus the target's step.

    Built by the worker (`DflyBlockIndices` fills `packed`/`lengths` from the
    picks state, `DflyBlockPrepare` gives `slots`/`positions`/`active`) and
    handed to `dflash_forward` in place of the engine's `block_placement`: the
    DFly2 block has no cache of its own to place, so what the drafter needs is
    where the block sits in the *target's* pool and which rows it attends.
    """

    #: int32 `[layers, rows, index_width]`: primary-pool rows per draft layer,
    #: the picked rows first and this row's own block slots after them, the
    #: tail `-1` (the sparse entry's padding).
    packed: torch.Tensor
    #: int32 `[layers, rows]`: how many of each row's columns are live.
    lengths: torch.Tensor
    #: int32 `[gens, width]`: the block's pool slots, `PAD_SLOT_ID` if inactive.
    slots: torch.Tensor
    #: int64 `[rows]`: the block's positions, one per row.
    positions: torch.Tensor
    #: bool `[gens]`: requests that draft this step.
    active: torch.Tensor
    #: The target's attention metadata for this step: the pools, block table and
    #: sequence lengths the block is written into and read from.
    attn_metadata: object


def draft_bundle(core) -> SimpleNamespace:
    """`draft_dims` plus the per-layer geometry the reused MLA layer reads.

    The drafter's five layers share one geometry -- the target sparse layer's,
    without the indexer the drafter does not own -- so `layers` is one geom
    repeated, and the widths a layer reads off its host (`tokens_per_block`,
    `max_seq_len`, the batch bounds) come from the core's bundle.
    """
    d = draft_dims(core)
    c = dims(core)
    # A sparse layer's heads and RoPE base; `sparse=False` keeps the reused
    # layer from building an indexer, and the draft selects with the target
    # indexer's picks instead.
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

    #: What the engine reads off a drafter before it runs one.
    block_size = BLOCK_WIDTH
    mask_token_id = MASK_TOKEN_ID
    #: The draft layers read the target's primary pool through the sparse MLA
    #: entry, so the backend is the plain one -- there is no private arena for a
    #: paged drafter cache to live in, and the engine adopts this value when the
    #: config leaves `attention_backend` at AUTO.
    dflash_attention_backend = "VANILLA"
    #: DFly2 draws one coherent path with a hidden correction, not a candidate
    #: selector or an lili-correction; the engine must take the plain block path.
    has_candidate_selector = False
    is_dflash2 = False
    lilicorr = None
    #: The draft read is the target's checkpoint narrowed to this prefix.
    checkpoint_key_prefixes = (DRAFT_KEY_PREFIX,)

    def __init__(self, core):
        super().__init__()
        #: The text core: pool rows, the plain allreduce, the RoPE tables, the
        #: embedding and LM head the drafter borrows, and the rope table cache.
        #: Nothing above it: the shell holds this drafter as a submodule, and
        #: the engine's weight materialization walks the module tree
        #: (`memory_init.init_meta_tensors_on_cuda` through `model._apply`,
        #: then `model.to("cuda")`), which recurses over `children()` with no
        #: memo -- a reference back up to the shell is an unending module cycle
        #: (`RecursionError: maximum recursion depth exceeded`).
        self.core = core
        self.model_config = core.model_config
        self.dims = draft_bundle(core)
        self.w = _weights.DRAFT_WEIGHTS.declare(self.dims)
        #: The rows a draft layer's host contributes per request: the whole
        #: block, which is what the reused MLA block reads off its host.
        self.gen_width = self.dims.block_width
        #: The five draft layers. `MlaLayer` is the target's own MLA block: the
        #: absorb, the two prologue/attention entries and the epilogue are the
        #: target's math, driven here over a draft host (`self`), whose pool row
        #: is the *target* layer's and whose weights are the draft table's.
        self.layers = [MlaLayer(self, i) for i in range(self.dims.num_layers)]
        self._fusion = DflyFusion()
        self._correction = DflyHiddenCorrection()
        self._rope: dict | None = None
        self._rope_positions = 0
        #: The worker binds the target's logits processor and LM head here, so
        #: `block_logits` runs the same path the target's own head does.
        self.logits_processor = None
        self.lm_head = None
        self.draft_model_full = None

    # -- the host interface the reused MLA layer reads -------------------------

    @property
    def embed_tokens_weight(self) -> torch.Tensor:
        """The target's embedding table `[vocab, hidden]`.

        The drafter owns no embedding: the checkpoint's five `model.dfly_layers.*`
        layers decode the target's own token ids, so the block's token embeddings
        and the correction's previous-token embedding are the core's table.
        """
        return self.core.w["embed"]

    def _pool_row(self, md, layer: int) -> int:
        """A draft layer's row inside the primary pool: its target layer's row.

        The block's rows are appended to the pool of the sparse target layer
        this draft layer reuses, so the native prologue and the attention read
        and write that layer's pool, not a row of the drafter's own -- the
        drafter has none.
        """
        return self.dims.target_kv_layers[layer]

    def _allreduce(self, x: torch.Tensor) -> torch.Tensor:
        """The core's own plain sum, over the same attention group."""
        return self.core._allreduce(x)

    # -- lifecycle ------------------------------------------------------------

    def load_weights(self, weights, weight_mapper=None, allow_partial_loading=False) -> None:
        """Fill the drafter's declared weights from the checkpoint."""
        del weight_mapper, allow_partial_loading
        _weights.DRAFT_WEIGHTS.load(self, weights)

    def post_load_weights(self) -> None:
        """Nothing to build: the drafter's tables are the core's, built already."""
        return None

    def _rope_table(self, md, device) -> dict:
        """The draft layers' RoPE table: the target sparse layer's, grown to fit.

        A draft layer ropes at the target sparse layer's base (1e7) and at the
        block's positions, so it reads the same table the core's sparse layers
        do; the core builds and grows that table, and this borrows it.
        """
        wanted = int(getattr(md, "max_seq_len", 0) or 0) or self.dims.max_seq_len
        wanted = min(wanted, int(self.core.model_config.pretrained_config.max_position_embeddings))
        if self._rope is None or wanted > self._rope_positions:
            self._rope_positions = max(wanted, 8192)
            self._rope = self.core._rope_tables(device, self._rope_positions, self.dims.theta)
        return self._rope

    # -- the block -------------------------------------------------------------

    def project_target_hidden(
        self, anchors: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """The slot-0 fusion: the anchor embedding with the target's hidden state.

        This is what the drafter does with the target's captured hidden state --
        the equivalent of another family's `project_target_hidden` -- and the
        worker applies it to the block's column 0 before the block runs
        (`DflyFusion`: `anchors + gate * eh_proj(cat(enorm(anchors), hnorm(t)))`).
        """
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

    def precompute_context_kv(self, *args, **kwargs) -> None:
        """Nothing: a DFly2 drafter has no context of its own to precompute.

        The engine's generic DFlash drafter caches a context KV during the
        target's prefill; this drafter attends the target's own pool instead, so
        there is nothing to build, nothing to keep across steps and nothing to
        transfer. The worker skips the engine's context-input preparation.
        """
        return None

    def apply_draft_logit_transform(self, logits: torch.Tensor) -> torch.Tensor:
        """The identity: this drafter's config declares no logit transform."""
        return logits

    def _expand_q(self, q_resid: torch.Tensor, i: int) -> torch.Tensor:
        """`[T, heads_local, qk_dim]`: the draft FFN's `q_b` is block-FP8."""
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
        """The draft FFN: block-FP8 SwiGLU, intermediate split over the MLP group."""
        from .ops.activation.flashinfer_silu_and_mul__r18 import flashinfer_silu_and_mul

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
        """One draft layer's attention over the target pool, for a whole block.

        Adapted from `MlaLayer.generation_partial`: the same absorbed-query
        prologue, the same sparse latent attention and the same `v_b_proj`
        expansion, with the one difference a block makes -- the prologue appends
        and the attention reads `block_width` rows per request instead of one,
        which is the layout the engine's own verify step uses
        (`tokens_per_gen_step`).

        `lengths` is the row's own count of live columns in `packed`
        (`DflyBlockIndices`: the request's picks, then the block's slots up to
        and including the row's own), one list per draft layer. The entry's
        `sequence_length` is the bound its sparse kernel derives row `j`'s key
        count from (`bound - width + 1 + j`: the dense causal bound read as a
        count), so a request is handed the writer's count through the block's
        *last* row -- the one place the block's `width` rows and the selection's
        live prefix agree. The step's device length is the pool's end, which
        parts from that count once the request's picks fill the selection
        (`anchors > index_topk`) and sends the kernel into the `-1` tail the
        writer left. The rope prologue keeps the step's device lengths, which
        are what place the block's rows.
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
            width,  # predicted_tokens_per_seq: the block
            self._pool_row(md, i),  # layer_idx: the target layer's pool
            d.heads_local,
            1,
            d.lat_dim,
            0,
            d.tokens_per_block,
            layer._window(md),  # no window: the pool's max length
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
        # The entry's per-request key bound. The context rows ahead of the block
        # keep the step's own lengths; each generating request gets the writer's
        # count through its block's last row, which the kernel's derivation
        # turns into that row's own count and, row by row, into the block's.
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
        """The block's hidden states: five draft layers over `[gens * W, hidden]`.

        The engine's `DFlashWorker` calls this with its own keyword names; for
        DFly2 the `ctx_*` arguments are all unused (the drafter has no context
        of its own) and `block_placement` is a `DflyBlockSelection` carrying
        this block's slots, positions and per-layer selections, because the pool
        the block is written into is the target's. Returns the block's rows after
        `final_norm`, which is the state the hidden correction is applied to.
        """
        del num_ctx_per_req, ctx_k_cache, ctx_v_cache, ctx_cache_batch_idx
        del ctx_kv_cache, ctx_page_table
        md = block_placement.attn_metadata
        d = self.dims
        rope = self._rope_table(md, noise_embedding.device)
        # eq. `HYV4DFlyModel.forward`: the block's rows go through the five
        # layers in parallel; the fusion already happened on column 0.
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
        # `HYV4DFlyModel.forward` ends on the drafter's own `final_norm`, and the
        # correction walk reads its output directly.
        return _rms(x, self.w["final_norm"], d.rms_eps)

    # -- what the worker's sampling walk reads --------------------------------

    def hidden_correction(
        self, hidden: torch.Tensor, previous_tokens: torch.Tensor, embed_tokens=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One column's correction: `hidden + mlp([norm(h) | norm(embed(prev))])`.

        `previous_tokens` are the tokens sampled at the *previous* column (the
        anchor's token for the block's first draft), so this is the sequential
        half of the block and the reason the walk is the worker's. `embed_tokens`
        defaults to the target's own table (`embed_tokens_weight`), which is the
        table the engine's vocabulary is in.

        The second return is the row's confidence (`conf_proj`/`conf_bias` on the
        same correction input), the head the checkpoint trains beside the
        corrected hidden; the walk drops it unless a recorder asks for it.
        """
        d = self.dims
        w = self.w
        weight = self.embed_tokens_weight if embed_tokens is None else embed_tokens.weight
        previous = torch.nn.functional.embedding(previous_tokens.long(), weight)
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

    def block_logits(self, hidden: torch.Tensor, attn_metadata, embed_tokens=None) -> torch.Tensor:
        """The target's LM head on a corrected row.

        The row is already normed (`dflash_forward` applies the drafter's
        `model.dfly_layers.final_norm`), so the head runs on it as it is; it is
        the target's head, through the same logits processor the target uses, so
        a draft token is drawn from the distribution the drafter's own row scores.
        """
        del embed_tokens
        return self.logits_processor(hidden, self.lm_head, attn_metadata, True)

    def draft_weights_manifest(self):
        """The drafter's weight table, for the shell's own checks."""
        return _weights.DRAFT_WEIGHTS.manifest(self)


__all__ = [
    "DRAFT_KEY_PREFIX",
    "DEFAULT_BLOCK_WIDTH",
    "DflyBlockSelection",
    "Hyv4FlashDraft",
    "draft_bundle",
]
