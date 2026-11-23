# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash's text core: 42 layers, both phases, no speculation, no media of its own.

One forward for both phases and all four members. A layer is

    iHC pre-mapping -> input_layernorm -> gated MLA -> sigmoid gate -> o_proj
    -> post_proj_norm -> iHC post -> post_attention_layernorm
    -> [Engram, on layers 1/11/21/31] -> MLP (dense at layer 0, latent MoE
    elsewhere) -> sandwich norm -> iHC post

and the model is `hc_head` -> `model.norm` -> the shell's vocab-parallel
lm_head. Every width comes from `dims`, every kernel from the target's `ops/`
entries, and every value from this checkpoint.

Two things the engine fixes for us, stated here because they are conventions and
not arithmetic:

* **`q_scaling` is 1.0.** The native MLA path divides the softmax by
  `sqrt(kv_lora + rope)` times `q_scaling` and corrects the sparse branch by
  `sqrt(qk_nope + qk_rope)/sqrt(kv_lora + rope)`, so 1.0 is exactly the
  checkpoint's `1/sqrt(256)`. The vocabulary's `q_scaling: 0.0625` in the
  prologue cells is that effective scale, not the argument.
* **The interleaved mRoPE table equals the plain duplicated one for text.** With
  no image or video tokens every mRoPE plane carries the token's own position,
  so the interleaved table and the plain one agree row for row; the op reads
  only `rotary_cos_sin`.

The two phases take the two shapes this MLA entry has: **the context phase is
unabsorbed** (explicit K and V at `[T, heads_local * 256]` and the split view of
a packed `[T, heads_local * 448]` buffer, `head_size` at the qk width) and **the
decode phase is absorbed** (latent MQA over the paged pool at `head_size` 576,
V expanded by `v_b_proj` after the kernel). Only the decode shape exists in the
fused-query form; a context call that passes no K/V leaves its output
unwritten, so the context phase must not take it.

The **context phase is dense over the keys it is given**: it passes no
`sparse_attn_indices`, and `attention_window_size` is what masks a windowed
layer there, so a sparse layer's context over more than `index_topk` keys
attends the whole context where the checkpoint attends its top-k.

The **decode phase takes one of two shapes, and the layer's geometry picks it**
(`_decode_selection`). A **sparse** layer passes its indexer's selection as
`sparse_attn_indices` and the kernel attends exactly the pool rows it names -- a
row of nothing but `-1` yields an exactly zero output, which is what pins the
selection's layout. Selections are built in request-local key positions as a
**valid prefix followed by `-1` padding** (the layout `indexer_select` writes
and the one the engine's own sparse indexers emit), and `_pool_rows` resolves
them to rows of the primary pool. A **windowed** layer passes none and decodes
through the dense windowed entry, whose mask is the constant
`attention_window_size` and whose per-step key range comes from the device
sequence lengths: the sparse entry bounds a row by the *host* KV length, a
scalar a captured CUDA graph freezes at its capture value, so a static
selection there replayed an empty window.

**Media.** The image and video tower (`vision.py`) is a separate module the
shell keeps outside the core: before each step it runs the tower over the
step's media and parks the `[R, hidden]` rows with the `[R]` placeholder rows
they replace, which the core's `forward` writes over the token embedding before
the streams are built. The core itself never sees a pixel. Text-only steps park
nothing and run exactly as before.
"""

from __future__ import annotations

import os

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.catalog.attention.load_paged_kv_cache_for_mla import (
    load_paged_kv_cache_for_mla,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger

from .. import cache_manager
from ..engram_host import EngramHostTable
from . import vision as _vision
from . import weights as _weights
from .dims import LayerGeom, dims
from .ops.activation.flashinfer_silu_and_mul__r18 import flashinfer_silu_and_mul
from .ops.activation.gated_attn_output import gated_attn_output
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.indexer_pool import indexer_pool
from .ops.attention.indexer_select import indexer_select
from .ops.attention.mla_rope_append_paged_kv_assign_q__r14 import mla_rope_append_paged_kv_assign_q
from .ops.attention.mla_rope_generation__mla_decode_prologue import mla_rope_generation
from .ops.attention.thop_attention__mla_windowed import MLA_CONTEXT, MLA_GENERATION, thop_attention
from .ops.comm.allgather__r20 import allgather
from .ops.comm.allreduce__r19 import PLAIN, allreduce, allreduce_group_args
from .ops.comm.reducescatter__r21 import reducescatter
from .ops.engram.engram_branch import engram_branch
from .ops.engram.engram_fp8_roundtrip import engram_fp8_roundtrip
from .ops.gemm.block_fp8_gemm import block_fp8_gemm
from .ops.gemm.bmm_out__r16 import bmm_out
from .ops.gemm.cublas_mm__r11 import cublas_mm
from .ops.mhc.ihc_mapping import ihc_mapping
from .ops.moe.fp8_block_scale_moe_runner import fp8_block_scale_moe_runner
from .ops.moe.noaux_tc_op__r17 import noaux_tc_op  # noqa: F401
from .ops.norm.flashinfer_rmsnorm__r12 import flashinfer_rmsnorm
from .ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128

#: Set `TRTLLM_HYV4_DRAFT_PICKS_LOG` to have every eager step report what the
#: draft layers' picks state holds (`_log_picks`) -- a probe for the fill, not a
#: feature.
_PICKS_DEBUG = os.environ.get("TRTLLM_HYV4_DRAFT_PICKS_LOG", "") not in ("", "0")

#: The engine's inert attention arguments: everything this target does not drive.
_INERT = dict(
    output_sf=None,
    max_context_q_len_override=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    rotary_inv_freq=None,
    block_ids_per_seq=None,
    beam_width=1,
    mask_type=1,
    position_embedding_type=8,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    use_paged_context_fmha=False,
    chunked_prefill_buffer_batch_size=None,
    rope_append=True,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    sparse_attn_kv_lens=None,
)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return flashinfer_rmsnorm(x, weight, eps)


def _hc_post(gates: torch.Tensor, hc: int) -> torch.Tensor:
    """A connection's block-output weights: the last `hc` of its `2 * hc` gate rows."""
    return gates[..., hc:]


def _hc_residual(post: torch.Tensor, block: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """HF's `post * block + residual`, carried in fp32 and rounded once."""
    return (post.float().unsqueeze(-1) * block.float().unsqueeze(-2) + residual.float()).to(
        residual.dtype
    )


def _phase_rows(t: torch.Tensor | None, lo: int, hi: int | None) -> torch.Tensor | None:
    """One phase's rows of a per-row selection tensor, or None when there is none."""
    return None if t is None else t[lo:hi]


def _recent_rows(positions: torch.Tensor, width: int) -> torch.Tensor:
    """`[T, width]` int32: each row's own key position and the `width - 1` before it.

    The per-row selection in request-local key positions, the row's own first and
    descending, `-1` in the slots whose position falls before the document's start.
    A sparse selection is read as a **valid prefix** followed by `-1` padding: the
    layout `indexer_select` writes (ascending ids, `-1` past the row's length), the
    one the engine's own indexers emit, and the one the sparse decode kernel is
    promised (`prims_ts_sparse_mla`'s `assume_valid_prefix`: the producers "emit
    valid prefixes ... with -1 only in the tail"). Padding in front of the live
    entries instead leaves the kernel gathering `-1` rows for every slot up to the
    live length. A sparse layer below `index_topk` passes every visible key this
    way, and so does the indexer's fallback for a row whose top-k came back short;
    `_pool_rows` resolves it to the pool rows the sparse MLA entry reads. The
    width is fixed per layer so a captured CUDA graph keeps its shape while
    positions grow.
    """
    back = torch.arange(width, dtype=torch.int32, device=positions.device)
    return (positions.to(torch.int32)[:, None] - back[None, :]).clamp(min=-1)


def _decode_selection(geom: LayerGeom, selected: torch.Tensor | None) -> torch.Tensor | None:
    """The key selection a layer's decode hands the MLA generation entry, if it owns one.

    Only a sparse layer selects keys at decode: it passes its indexer's top-k as
    `sparse_attn_indices` and the kernel attends exactly those pool rows. A
    windowed layer selects nothing -- its decode is the dense windowed entry,
    masked by the constant `attention_window_size` over a per-step key range the
    entry reads from the device sequence lengths. A selection there would route
    the step through the sparse entry, which bounds a row by the host KV length,
    a scalar a captured CUDA graph freezes at its capture value.
    """
    return selected if geom.sparse else None


def _row_batch(md) -> torch.Tensor:
    """`[T]`: the request each row of the step belongs to.

    A step's rows are the context requests' new tokens, request by request, then
    one generation row per decoding request -- the order both phases' prologues
    read. The sparse backend builds that map itself for its index conversion
    (`build_req_idx_per_token`, over the same per-request step lengths), and it
    exists whether or not the backend keeps cached-context MLA buffers, so read
    it instead of re-deriving it from the per-request length indptrs.
    """
    return md.req_idx_per_token[: int(md.num_tokens)].to(torch.int64)


def _picks_rows(md) -> torch.Tensor:
    """`[N]` int64: the step's rows whose selections the draft layers read.

    DFly2's drafter attends over the picks of the anchor's *source* token, the
    last token the target accepted for the request, named by the row that token
    occupied in the step the target last processed. A generation step's rows are
    its verify block's, and which of them turns out to be the accepted one is a
    sampling result the step's own forward does not know, so every generation
    row is published. A context step verifies nothing: only its last row per
    request can be an anchor's source -- the prompt tail a request's first block
    is built from -- so a chunk of prompt tokens publishes one row per request,
    not one per token.
    """
    rows = _row_batch(md)
    total = int(md.num_tokens)
    num_ctx = int(md.num_contexts)
    tc = int(md.num_ctx_tokens) if num_ctx else 0
    picked = []
    if tc:
        # `_row_batch` keeps a context request's rows together, so its last row
        # is where the running per-request count ends.
        counts = torch.zeros(num_ctx, dtype=torch.int64, device=rows.device)
        counts.scatter_add_(0, rows[:tc], torch.ones(tc, dtype=torch.int64, device=rows.device))
        picked.append(counts.cumsum(0) - 1)
    if total > tc:
        picked.append(torch.arange(tc, total, dtype=torch.int64, device=rows.device))
    if not picked:
        return torch.empty(0, dtype=torch.int64, device=rows.device)
    return torch.cat(picked)


def _log_picks(picks, md, rows: torch.Tensor) -> None:
    """One line per eager step describing what the draft layers' picks state holds.

    A probe, off unless `TRTLLM_HYV4_DRAFT_PICKS_LOG` is set: it reads the step's
    written rows back to the host -- a synchronisation a captured graph may not
    do -- so it is skipped while the stream is capturing. What it prints is the
    fill a run left in the state: which rows of the step carry picks, per draft
    layer, and the pool rows of one of them.
    """
    if not _PICKS_DEBUG or torch.cuda.is_current_stream_capturing():
        return
    history = picks.history
    written = history.index_select(1, rows)
    counts = (written >= 0).sum(-1)
    layers = ", ".join(
        f"L{layer}:{int(counts[k].min())}..{int(counts[k].max())}"
        for k, layer in enumerate(picks.layers)
    )
    sample = history[0, int(rows[0])][:8].tolist()
    logger.warning(
        "[hyv4-draft-picks] rows=%d of %d (T=%d, contexts=%d, ctx_tokens=%d) picks/row %s; "
        "row %d of layer %d starts %s",
        rows.numel(),
        history.shape[1],
        int(md.num_tokens),
        int(md.num_contexts),
        int(md.num_ctx_tokens) if int(md.num_contexts) else 0,
        layers,
        int(rows[0]),
        picks.layers[0],
        sample,
    )


def _ngram_tokens(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    row_batch: torch.Tensor,
    max_ngram: int,
    pad_id: int,
) -> torch.Tensor:
    """`[T, max_ngram]`: each row's own token and the `max_ngram - 1` before it.

    HF's `build_ngram_tokens`: column `d` holds the token at position `p - d`,
    and a position reaching past its document's start is the pad id, so an
    n-gram never spans two documents. A predecessor counts only when it is the
    previous row of this forward and of the same request at position `p - d`;
    the ones before the step's first row come from `EngramHistory`.
    """
    tokens = input_ids.to(torch.int64).view(-1)
    positions = position_ids.to(torch.int64).view(-1)
    rows = row_batch.to(torch.int64)
    columns = [tokens]
    for offset in range(1, max_ngram):
        previous = torch.full_like(tokens, pad_id)
        previous_position = torch.full_like(positions, -1)
        previous_row = torch.full_like(rows, -1)
        if tokens.numel() > offset:
            previous[offset:] = tokens[:-offset]
            previous_position[offset:] = positions[:-offset]
            previous_row[offset:] = rows[:-offset]
        adjacent = (
            (positions >= offset)
            & (previous_position == positions - offset)
            & (previous_row == rows)
        )
        columns.append(torch.where(adjacent, previous, pad_id))
    return torch.stack(columns, dim=-1)


class EngramHistory:
    """The tokens before each row of a step, across forward boundaries.

    HF's `build_ngram_tokens` reads the `max_ngram - 1` tokens before every
    position out of the whole sequence, and `HYV4FlashEngramLayer` hashes those
    n-grams. A step's `input_ids` carry only the step's own tokens, so a
    generation step -- one token per sequence -- would give every n-gram `pad`
    predecessors and hash a memory the checkpoint never chose. The engine's own
    Engram keeps the same per-request lookback for the same reason
    (`_torch/modules/engram/engram.py`: "a decode forward carries exactly one
    token per sequence, so every n-gram would degenerate to (pad, ..., pad,
    tok)").

    A request's row is its **first KV block**: `kv_cache_block_offsets` is the
    C++ encoded block table (`attention/backends/utils.py`: an entry holds
    `pool_block_index * num_pool_layers * kv_factor`), so its first entry is the
    request's own, stable for its lifetime and unique among live requests --
    where the step's row index and the metadata's request ids are neither (a
    captured decode graph replays `request_ids = range(padded_batch_size)`).
    Rows are that key modulo a prime, and each row records the key it was
    written for, so two requests that collide read `pad` instead of each other's
    tokens. Columns are the position modulo the step's token budget, which no
    step exceeds, so one request's rows never share a column; a cell is read
    only for the exact position it stores, which makes a recycled row, a
    colliding key or an overwritten ring cell degrade to `pad` -- HF's own
    answer for a position with no predecessor -- rather than to a wrong token.
    A request writes every position it visits and reads only the ones before
    them, so it never reads a cell it did not write.

    Device ops only, no host branch and no data-dependent shape, so a captured
    decode graph replays the same launches.
    """

    #: The row count of the lookback ring: prime, so the encoded block stride
    #: (`num_pool_layers * kv_factor`) cannot alias it. Collisions cost one
    #: request its lookback while both are live, never a wrong token.
    ROWS = 1021

    def __init__(self, core: "Hyv4FlashCore") -> None:
        self.core = core
        self._tok: torch.Tensor | None = None
        self._pos: torch.Tensor | None = None
        self._key: torch.Tensor | None = None

    def _state(
        self, device: torch.device, width: int, pad_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The ring, allocated once on the first step and kept across replays."""
        tok = self._tok
        if tok is None or tok.device != device or tok.shape[1] < width:
            self._tok = torch.full((self.ROWS, width), pad_id, dtype=torch.int64, device=device)
            self._pos = torch.full((self.ROWS, width), -1, dtype=torch.int64, device=device)
            self._key = torch.full((self.ROWS,), -1, dtype=torch.int64, device=device)
        return self._tok, self._pos, self._key

    def lookback(
        self,
        md,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        row_batch: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor | None:
        """`[T, max_ngram - 1]`: the tokens before each row, `pad` where none is known.

        Reads the ring before writing this step's tokens into it, so a row never
        sees a token of its own step -- those come from `_ngram_tokens`' shift,
        which the caller prefers where it applies.
        """
        d = self.core.dims
        depth = d.engram_max_ngram - 1
        offsets = getattr(md, "kv_cache_block_offsets", None)
        if depth <= 0 or offsets is None:
            return None
        tokens = input_ids.to(torch.int64).view(-1)
        positions = position_ids.to(torch.int64).view(-1)
        rows = row_batch.to(torch.int64)
        tok, pos, key_of = self._state(tokens.device, int(d.max_num_tokens), pad_id)
        width = tok.shape[1]
        # The request's own first block, and whether this row is a real request
        # rather than a graph's padding row (whose block entry is a placeholder).
        own = offsets[0][rows, 0, 0].to(torch.int64)
        live = md.kv_lens_cuda_runtime[rows] > 0
        own = torch.where(live, own, torch.full_like(own, -1))
        key = own.clamp(min=0) % self.ROWS
        columns = []
        for step in range(1, depth + 1):
            target = positions - step
            slot = target.clamp(min=0) % width
            cell = tok[key, slot]
            valid = (pos[key, slot] == target) & (key_of[key] == own) & (target >= 0)
            columns.append(torch.where(valid, cell, torch.full_like(cell, pad_id)))
        # Write this step's tokens at their own positions: unique cells per
        # request (no step holds more rows than the ring's width), and a
        # padding row writes its cells back unchanged.
        slot = positions.clamp(min=0) % width
        tok[key, slot] = torch.where(live, tokens, tok[key, slot])
        pos[key, slot] = torch.where(live, positions, pos[key, slot])
        key_of.scatter_(0, key, torch.where(live, own, key_of[key]))
        return torch.stack(columns, dim=-1)


class Indexer:
    """The compressed indexer of one sparse layer: pooled keys, then the top-k groups.

    HF's `HYV4FlashCompressedIndexer`, in this target's terms. `pool_and_select`
    is the whole of it: project the chunk's token keys, pool them four at a time
    through the learned value/gate pair with `intra_group_ape`, score the groups
    with the sixteen query heads and their head weights, take the top
    `index_group_topk - 1` complete groups plus each row's own, and expand back
    to at most `index_topk` token ids. Below `index_topk` tokens the indexer is
    skipped entirely and every key is attended -- HF's `kv_length <= index_topk`
    short circuit.
    """

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims

    def pool_and_select(
        self,
        q_resid: torch.Tensor,
        x: torch.Tensor,
        md,
        row_batch: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """`(packed ids, lengths, visible_mask)` for this chunk.

        Below `index_topk` no row needs a selection: every key is visible, so the
        ids are the identity over the row's own keys, `-1` past its start -- the
        layout the sparse attention entry reads in both phases.
        """
        d, w, i = self.d, self.core.w, self.layer
        num_seqs = int(md.num_seqs)
        # HF's short circuit: below `index_topk` every key is visible, so the
        # selection is the identity and the row's visible count is its own
        # position plus one. The step's KV lengths are read on the host, where
        # the metadata keeps them: a device read here would sync, which a
        # CUDA-graph capture does not allow.
        if int(md.kv_lens_runtime[:num_seqs].max()) <= d.index_topk:
            packed = _recent_rows(positions, d.index_topk)
            lengths = (positions.to(torch.int32) + 1).clamp(max=d.index_topk)
            return packed, lengths, torch.ones(num_seqs, dtype=torch.bool, device=positions.device)
        token_keys = _rms(
            cublas_mm(x, w[f"l{i}_idx_token_key"].t()), w[f"l{i}_idx_token_key_norm"], d.rms_eps
        )
        # `indexer_pool` reads one request per batch row (`[B, tokens, head_dim]`),
        # so lay this step's token keys out per request: each request's own rows in
        # their order, zero-padded to the longest. A row never scores a group at or
        # past its own position, so neither the padding nor a shorter request's
        # empty slots can be recalled. The pool holds only the keys this step
        # carries; HF caches token keys so a decode row or a chunked prefill can
        # pool its cached prefix, which this target does not do yet.
        rows = row_batch.to(torch.int64)
        # Rows past the step's requests are the engine's graph padding. They are
        # redirected to one trash slot rather than dropped with a mask: a boolean
        # index is a device->host read, and a CUDA-graph capture does not allow
        # one, so every step here stays on the device.
        live = (rows >= 0) & (rows < num_seqs)
        batch = rows.clamp(0, num_seqs - 1)
        per_seq = torch.zeros(num_seqs, dtype=torch.int64, device=token_keys.device)
        per_seq.scatter_add_(0, batch, live.to(torch.int64))
        starts = per_seq.cumsum(0) - per_seq
        slot = torch.arange(token_keys.shape[0], device=token_keys.device) - starts[batch]
        # The longest request's row count, from the step's host metadata: the
        # context requests' longest chunk (a decode-only step carries one row per
        # request). Rows past it are the engine's graph padding, which no real
        # query can recall.
        width = max(int(md.max_ctx_seq_len), 1)
        flat = token_keys.new_zeros(num_seqs * width + 1, token_keys.shape[-1])
        trash = torch.full_like(batch, num_seqs * width)
        dest = torch.where(live & (slot >= 0) & (slot < width), batch * width + slot, trash)
        flat.scatter_(0, dest.unsqueeze(1).expand(-1, token_keys.shape[-1]), token_keys)
        keys = flat[: num_seqs * width].view(num_seqs, width, token_keys.shape[-1])
        group_keys = indexer_pool(
            keys,
            w[f"l{i}_idx_kv_proj"],
            w[f"l{i}_idx_gate_proj"],
            w[f"l{i}_idx_ape"],
            w[f"l{i}_idx_key_norm"],
            d.rms_eps,
            kpool=d.index_kpool,
        )
        query = cublas_mm(q_resid, w[f"l{i}_idx_query"].t()).view(
            -1, d.index_n_heads, d.index_head_dim
        )
        head_weights = cublas_mm(x, w[f"l{i}_idx_head_weight"].t()).float() * d.index_scale
        packed, lengths, nonempty = indexer_select(
            query,
            group_keys,
            head_weights,
            batch,
            positions,
            ratio=d.index_kpool,
            index_topk=d.index_topk,
        )
        # A row whose own prefix is only partly in this step's pool cannot be
        # scored by an indexer that pools the step's rows: every group past them
        # is pooled from zeros, scores `relu(0)` = 0, and loses only to the
        # groups that are actually visible, so the recall degenerates to the
        # lowest group ids plus the row's own group. That is every decode row
        # (its predecessors are cached) and every chunk after the first. Those
        # rows attend their most recent `index_topk` keys instead -- the dense
        # short circuit above, extended past its own length: causal, and it keeps
        # the recent context in view. HF scores the cached prefix through a
        # token-key cache this target does not keep yet.
        recent = _recent_rows(positions, d.index_topk)
        full = (slot + 1) >= (positions.to(torch.int64) + 1)
        packed = torch.where(full[:, None], packed, recent)
        lengths = torch.where(
            full, lengths, (positions.to(torch.int32) + 1).clamp(max=d.index_topk)
        )
        return packed, lengths, nonempty


class EngramLayer:
    """One Engram layer: a host-resident hash table read by a gated per-head branch.

    The table is 31.7 GB and lives in pinned host memory (`EngramHostTable`);
    the gather is a lookup through the device-visible mapping, never a copy onto
    the device. HF's `HYV4FlashEngramLayer`, with this checkpoint's own moduli
    and per-head row offsets.
    """

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims
        self.table: EngramHostTable = core.engram_tables[layer]

    def hash_ngrams(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """`[T, num_heads]` global table rows for this layer's sixteen heads.

        HF's `hash_ngrams`: compress each n-gram column through the identity
        lookup table (ids past it read as the pad row), multiply column `d` by
        the layer's own odd multiplier `M[d]`, XOR the columns of one order
        together and reduce modulo that head's modulus, then shift the result
        into the head's row block.
        """
        d, w = self.d, self.core.w
        i = self.layer
        table = w[f"l{i}_engram_lookup"]
        ids = ngram_tokens.to(torch.int64)
        compressed = table[ids.clamp(max=d.engram_table_size - 1)]
        compressed = torch.where(
            ids < d.engram_table_size, compressed, table[d.engram_pad_id]
        ).long()
        mixed = compressed * self.core.hash_multipliers[i]
        orders = d.engram_max_ngram - 1
        per_order = d.engram_num_heads // orders
        rows = []
        for k in range(d.engram_num_heads):
            order = k // per_order
            h = mixed[:, 0]
            for offset in range(1, order + 2):
                h = torch.bitwise_xor(h, mixed[:, offset])
            rows.append((h % self.core.head_moduli[i][k]) + d.engram_head_offsets[i][k])
        return torch.stack(rows, dim=-1)

    def forward(
        self, streams: torch.Tensor, ngram_tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """`[T, hc, hidden] -> [T, hc, hidden]`: the branch added to each stream."""
        d, w, i = self.d, self.core.w, self.layer
        rows = self.hash_ngrams(ngram_tokens)
        head_dim = d.engram_head_dim
        gathered = self.table.gather(rows.reshape(-1)).view(-1, d.engram_num_heads, head_dim)
        if d.engram_fp8:
            gathered = engram_fp8_roundtrip(gathered)
        # One RMSNorm per hash head, each with its own `[head_dim]` weight: HF's
        # `lookup_per_head_norms`, a `ModuleList` of sixteen norms rather than
        # one norm over the concatenation.
        norms = w[f"l{i}_engram_head_norm"]
        gathered = torch.stack(
            [_rms(gathered[:, k], norms[k], d.rms_eps) for k in range(d.engram_num_heads)],
            dim=1,
        ).view(-1, d.engram_num_heads * head_dim)
        return engram_branch(
            streams,
            gathered,
            w[f"l{i}_engram_key_w"],
            w[f"l{i}_engram_key_b"],
            w[f"l{i}_engram_norm_key"],
            w[f"l{i}_engram_norm_query"],
            w[f"l{i}_engram_value_w"],
            w[f"l{i}_engram_value_b"],
            w[f"l{i}_engram_pre_gamma"],
            w[f"l{i}_engram_hc_scale"],
            mask,
        )


class LatentMoe:
    """HY4-Flash's MoE: route on the hidden, run the experts on the latent, add the shared one.

    `pre_expert` maps 3072 -> 1536, the block-scaled FP8 runner routes each row
    to its top-16 of the checkpoint's 512 experts and runs the ones this rank
    holds -- its own 128-id window -- with the clamped SwiGLU, `post_expert`
    maps back, the shared expert is a plain SwiGLU branch, and
    `combine_output_norm` closes the block.

    Which rows the window sees, and what completes its partial sum, follow from
    the rows the rank holds. Under attention TP every rank holds every row and
    its own share of the experts, so the plain sum over the group's partials is
    the complete routed sum -- and the same sum completes the shared expert,
    whose intermediate is split over that group. Under attention DP a rank
    holds only its own requests' rows: the group's windows then run over the
    union of the ranks' rows, gathered in rank order, and the completed sum
    comes back to the rows' own ranks by reduce-scatter, while the shared
    expert is whole on every rank and stays on the rows that own it.
    """

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims

    def _routed(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's expert window over `x`, mapped back to the hidden width.

        The partial is over the window alone, so it is still missing the other
        windows' contributions -- and, under `moe_tp`, the other intermediate
        slices of its own experts. `x` is whatever rows the caller holds: the
        rank's own rows under attention TP, the group's union under attention
        DP, so the caller decides which rows the window runs on.
        """
        d, w, i = self.d, self.core.w, self.layer
        latent = cublas_mm(x, w[f"l{i}_pre_expert"].t())
        latent_q, latent_s = fp8_quantize_1x128(latent)
        routed = fp8_block_scale_moe_runner(
            routing_logits=cublas_mm(x, w[f"l{i}_router"].t()),
            routing_bias=w[f"l{i}_router_bias"],
            hidden_states=latent_q,
            hidden_states_scale=latent_s,
            gemm1_weights=w[f"l{i}_experts_gate_up"],
            gemm1_weights_scale=w[f"l{i}_experts_gate_up_scale"],
            gemm2_weights=w[f"l{i}_experts_down"],
            gemm2_weights_scale=w[f"l{i}_experts_down_scale"],
            num_experts=d.num_experts,
            top_k=d.top_k,
            num_fused_shared_experts=0,
            n_group=d.n_group,
            topk_group=d.topk_group,
            intermediate_size=d.moe_inter,
            local_expert_offset=d.expert_offset,
            local_num_experts=d.local_experts,
            routed_scaling_factor=d.routed_scale,
            routing_method_type=2,  # DeepSeekV3: sigmoid, bias add, top-k, renorm
            topk_weights=None,
            topk_ids=None,
            act_type=0,  # SwiGLU is the only activation this op computes
            gemm1_clamp_limit=d.swiglu_limit,
            output=None,
            tune_max_num_tokens=d.max_num_tokens,
            use_dp=d.attention_dp,
        )
        return cublas_mm(routed, w[f"l{i}_post_expert"].t())

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's share of the shared expert, mapped to the hidden width.

        Whole under attention DP -- `mlp_tp` is 1 there, so the intermediate is
        not split and this is the complete contribution -- and this rank's slice
        of the intermediate under attention TP, where the group's sum completes
        it.
        """
        w, i = self.core.w, self.layer
        return cublas_mm(
            flashinfer_silu_and_mul(
                cublas_mm(
                    x,
                    torch.cat([w[f"l{i}_shared_gate"], w[f"l{i}_shared_up"]], dim=0).t(),
                )
            ),
            w[f"l{i}_shared_down"].t(),
        )

    def forward(self, x: torch.Tensor, md) -> torch.Tensor:
        d, w, i = self.d, self.core.w, self.layer
        if self.core._dp_exchange():
            # Attention DP: no rank can complete a row's routed sum alone, and
            # the shared expert is whole here -- the exchange completes the one
            # and the other is added on the rows that own it, once.
            sizes = self.core._dp_row_sizes(md, x.shape[0])
            union = allgather(x, sizes, self.core._dp_group, self.core._dp_pg)
            out = reducescatter(
                self._routed(union), sizes, self.core._dp_group, self.core._dp_pg
            ) + self._shared(x)
        else:
            # Attention TP: every rank holds every row and its own share of the
            # experts and of the shared intermediate, so one sum over the group
            # completes both partials.
            out = self._routed(x) + self._shared(x)
            if d.moe_ep > 1 or d.mlp_tp > 1:
                out = self.core._allreduce(out)
        return _rms(out, w[f"l{i}_combine_output_norm"], d.rms_eps)


class MlaLayer:
    """One gated-MLA layer: prologue, attention core, epilogue.

    `context_rows` is the prefill phase (unabsorbed, explicit K/V) and
    `generation_partial` the decode one (absorbed, latent MQA); both return a
    `v_dim`-wide output, and `epilogue` applies the sigmoid gate, `o_proj` and
    `post_proj_norm`.
    """

    def __init__(self, core: "Hyv4FlashCore", layer: int) -> None:
        self.core = core
        self.layer = layer
        self.d = core.dims
        self.geom = core.dims.layers[layer]
        self.indexer = Indexer(core, layer) if self.geom.sparse else None
        #: Rows this layer's host contributes per generating request in one
        #: step: 1 for the target's own decode, `block_width` for the drafter's
        #: block and for the target's verify step under speculation (the
        #: engine's `tokens_per_gen_step`). The generation entry and its
        #: prologue lay the generation rows out by it, so a verify step that
        #: processes six rows for one request must say so.
        self.gen_width = core.gen_width

    def _absorbed_q(self, q: torch.Tensor, q_roped_tail: torch.Tensor) -> torch.Tensor:
        """`[T, heads_local, 512 + 64]`: `q_nope @ k_b` with the roped tail appended."""
        d, w, i = self.d, self.core.w, self.layer
        q_nope = q[..., : d.nope]
        k_b = w[f"l{i}_k_b_proj"]
        out = torch.empty(
            q.shape[0],
            self.geom.heads_local,
            d.kv_lora + d.rope_dim,
            dtype=q.dtype,
            device=q.device,
        )
        bmm_out(q_nope.transpose(0, 1), k_b, out[:, :, : d.kv_lora].transpose(0, 1))
        out[:, :, d.kv_lora :] = q_roped_tail
        return out

    def _window(self, md) -> int:
        """The window the prologue writes into the block array and the reader remaps by.

        A windowed layer passes its own window, the value the prologue cells are
        certified at; a layer with no window passes the pool's max sequence
        length, which is what the engine's own backend resolves a missing
        per-layer window to. Zero is not "no window" -- it is a window below
        every KV length, which the generation prologue's note calls uncertified.
        """
        return self.geom.window or self._pool_max(md)

    def _pool_max(self, md) -> int:
        """The pool's maximum sequence length: the bound a graph-stable scalar uses."""
        return int(getattr(md, "max_seq_len", self.d.max_seq_len) or self.d.max_seq_len)

    def context_rows(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        md,
        rope: dict,
    ) -> torch.Tensor:
        """Prefill: rope and append the chunk's latent rows, then attend over them.

        The context entry is the unabsorbed one: it reads K and V the caller
        built, at `[T, heads_local * qk_dim]` and the `[.., H*nope:]` split view
        of a packed `[T, H*(nope + v)]` buffer, with `latent_cache=None` so it
        ropes nothing and appends nothing -- the prologue above did both. The
        absorbed form (`is_fused_qkv`, K/V absent) is a generation-only shape in
        this entry, so a context call in it leaves the output unwritten.
        """
        d, w, i = self.d, self.core.w, self.layer
        hl = self.geom.heads_local
        num_tokens = q.shape[0]
        # The context gather copies positions [0, L_s) of every context sequence,
        # so the K/V it produces spans each request's cached prefix as well as the
        # rows this step appends. `roped`, `q` and `out` hold this step's rows only.
        num_kv = int(md.host_total_kv_lens[0])
        roped = torch.empty(num_tokens, hl, d.qk_dim, dtype=q.dtype, device=q.device)
        roped.copy_(q)
        mla_rope_append_paged_kv_assign_q(
            # The op takes the query as `[T, head_num * (nope | rope)]` and ropes
            # its tail in place; this view shares `roped`'s storage.
            roped.view(num_tokens, hl * d.qk_dim),
            latent,  # latent_cache rows to append
            md.num_contexts,
            md.ctx_cached_token_indptr,
            md.ctx_kv_indptr,
            int(md.max_ctx_seq_len),
            rope["rotary_cos_sin"],
            hl,  # head_num
            d.nope,  # nope_size
            d.rope_dim,  # rope_size
            d.kv_lora,  # lora_size
            md.kv_cache_block_offsets,
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            0,  # residual_dim
            self.core._pool_row(md, i),  # layer_idx
            d.tokens_per_block,
            self._window(md),
            1,  # beam_width
            0,  # quant_mode
        )
        # Read the rows back: the pool holds `[normed compressed_kv | rope(k_pe)]`
        # per token, so this is where the context's roped K rope half comes from
        # without a second rope implementation beside the prologue's.
        ckv, k_pe = load_paged_kv_cache_for_mla(
            out_dtype=q.dtype,
            num_contexts=md.num_contexts,
            num_ctx_kv_tokens=num_kv,
            max_ctx_kv_len=int(md.max_ctx_kv_len),
            cu_ctx_kv_lens=md.ctx_kv_indptr,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            kv_scale_quant_orig=None,
            layer_idx=self.core._pool_row(md, i),
            kv_lora_rank=d.kv_lora,
            qk_rope_head_dim=d.rope_dim,
            tokens_per_block=d.tokens_per_block,
            attention_window_size=self._window(md),
            beam_width=1,
            quant_mode=0,
        )
        # One GEMM over the rank's head block, then both operands are slices of
        # it: K's no-PE half per head, and V as the split view the context FMHA
        # reads (its V row stride is the packed width, not `heads_local * v`).
        kv = cublas_mm(ckv, w[f"l{i}_kv_b_proj"].t())
        k = torch.empty(num_kv, hl, d.qk_dim, dtype=q.dtype, device=q.device)
        k[..., : d.nope].copy_(kv[:, : hl * d.nope].view(num_kv, hl, d.nope))
        k[..., d.nope :].copy_(k_pe.unsqueeze(1))
        out = torch.empty(num_tokens, hl * d.v_dim, dtype=q.dtype, device=q.device)
        self._attend_context(
            roped.view(num_tokens, hl * d.qk_dim),
            k.view(num_kv, hl * d.qk_dim),
            kv[:, hl * d.nope :],
            out,
            md,
            rope,
        )
        return out

    def generation_partial(
        self,
        q: torch.Tensor,
        latent: torch.Tensor,
        md,
        rope: dict,
        packed: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode: rope the absorbed query and append the row in one prologue call."""
        d, w, i = self.d, self.core.w, self.layer
        num_tokens = q.shape[0]
        q_pe = q[..., d.nope :]
        absorbed = self._absorbed_q(q, q_pe)
        device = q.device
        # The prologue writes the two cu-seqlens runs and the FMHA tile counter
        # (its `precomputed_*` flags are off), and the attention entry the same
        # step makes reads them back, so both calls share these buffers.
        cu_q = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        cu_kv = torch.empty(num_tokens + 1, dtype=torch.int32, device=device)
        counter = torch.empty(1, dtype=torch.uint32, device=device)
        mla_rope_generation(
            absorbed.view(num_tokens, -1),  # fused_q, roped in place
            q_pe,  # q_pe, the unroped rope half
            latent,  # latent_cache rows to append
            rope["rotary_cos_sin"],
            cu_q,  # cu_q_seqlens
            cu_kv,  # cu_kv_seqlens
            counter,  # fmha_scheduler_counter
            None,  # mla_bmm1_scale
            None,  # mla_bmm2_scale
            None,  # quant_q_buffer
            md.kv_lens_cuda_runtime,  # sequence_length
            md.kv_lens_runtime,  # host_past_key_value_lengths
            md.prompt_lens_cpu_runtime,  # host_context_lengths
            md.num_contexts,  # num_contexts: the sequences' offset
            md.kv_cache_block_offsets,
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            None,  # kv_cache_scale_orig_quant
            None,  # out_scale
            None,  # block_ids_per_seq
            [None, None],  # helix_tensor_params
            self.gen_width,  # predicted_tokens_per_seq
            self.core._pool_row(md, self.layer),  # layer_idx
            self.geom.heads_local,  # num_heads
            1,  # num_kv_heads
            d.lat_dim,  # head_size
            0,  # residual_dim: the plain row layout
            d.tokens_per_block,
            self._window(md),
            1,  # beam_width
            0,  # quant_mode
            d.q_scaling,
            d.q_lora,
            d.kv_lora,
            d.nope,
            d.rope_dim,
            d.v_dim,  # v_head_dim
            True,  # rope_append
            None,  # kv_norm_weight
            d.rms_eps,  # kv_norm_eps: unused without a weight
            False,  # precomputed_cu_seqlens
            False,  # precomputed_fmha_scheduler
            False,  # kv_only
            False,  # kv_done_elsewhere
            None,  # quant_scale_qkv
        )
        lat = torch.empty(
            num_tokens,
            self.geom.heads_local * d.kv_lora,
            dtype=q.dtype,
            device=q.device,
        )
        self._attend(
            absorbed,
            lat,
            md,
            rope,
            packed,
            # The generation entry checks for all five: the latent rows it has
            # already had appended, the unroped rope half, and the prologue's
            # scheduler scratch.
            latent_cache=latent,
            q_pe=q_pe,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
        )
        # The absorbed form attends in latent space, so its value expansion is
        # the last step here; both phases hand `epilogue` a `v_dim`-wide output.
        hl = self.geom.heads_local
        out = torch.empty(num_tokens, hl, d.v_dim, dtype=q.dtype, device=q.device)
        bmm_out(
            lat.view(num_tokens, hl, d.kv_lora).transpose(0, 1),
            w[f"l{i}_v_b_proj"],
            out.transpose(0, 1),
        )
        return out.view(num_tokens, hl * d.v_dim)

    def _step_args(self, md, rope: dict) -> dict:
        """The step-level arguments both MLA calls share."""
        d = self.d
        return dict(
            # The engine's attention workspace, grown by the kernel and switched
            # to the graph's own copy under capture; `FallbackFmha.attention`
            # refuses a None one.
            workspace=md.effective_workspace,
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            rotary_cos_sin=rope["rotary_cos_sin"],
            attention_sinks=self.core.w[f"l{self.layer}_sink"],
            update_kv_cache=True,
            # The context call's layout: one row per request. `_generation_args`
            # replaces it with the host's `gen_width` for the decode rows.
            predicted_tokens_per_seq=1,
            local_layer_idx=self.core._pool_row(md, self.layer),
            num_heads=self.geom.heads_local,
            tokens_per_block=d.tokens_per_block,
            max_num_requests=int(getattr(md, "max_num_requests", d.max_batch) or d.max_batch),
            max_context_length=int(
                getattr(md, "max_context_length", d.max_context_len) or d.max_context_len
            ),
            max_seq_len=int(getattr(md, "max_seq_len", d.max_seq_len) or d.max_seq_len),
            attention_window_size=self._window(md),
            quant_mode=0,
            q_scaling=d.q_scaling,
            rope_dim=d.rope_dim,
            rope_base=self.geom.theta,
            rope_max_positions=d.max_seq_len,
            rope_original_max_positions=d.max_seq_len,
            is_mla_enable=True,
            q_lora_rank=d.q_lora,
            kv_lora_rank=d.kv_lora,
            qk_nope_head_dim=d.nope,
            qk_rope_head_dim=d.rope_dim,
        )

    def _attend_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        md,
        rope: dict,
    ) -> None:
        """The unabsorbed context call: explicit K/V, `head_size` at the qk width.

        `head_size` only separates a context op from a generation one here; the
        context geometry's widths come from `qk_nope + qk_rope` (K, and the
        softmax's `1/sqrt(256)`) and `v_head_dim` (V, and the output's width).
        """
        d = self.d
        thop_attention(
            q=q,
            k=k,
            v=v,
            output=out,
            is_fused_qkv=False,
            num_kv_heads=self.geom.heads_local,  # context runs as MHA
            head_size=d.qk_dim,
            v_head_dim=d.v_dim,
            attention_input_type=MLA_CONTEXT,
            num_contexts=int(md.num_contexts),
            num_ctx_tokens=q.shape[0],
            # The prologue appended the rows and roped both tails, so the entry
            # neither ropes nor appends; the sparse selection addresses pool
            # rows, which this dense K/V is not.
            latent_cache=None,
            q_pe=None,
            sparse_attn_indices=None,
            num_sparse_topk=None,
            **self._step_args(md, rope),
            **_INERT,
        )

    def _attend(
        self,
        absorbed: torch.Tensor,
        out: torch.Tensor,
        md,
        rope: dict,
        packed: torch.Tensor | None,
        *,
        latent_cache: torch.Tensor,
        q_pe: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
    ) -> None:
        """The absorbed generation call: latent MQA over the paged pool.

        The entry requires the latent rows (already appended by the prologue),
        the unroped rope half, and the prologue's cu-seqlens runs and tile
        counter, which it reads instead of building its own.
        """
        d = self.d
        thop_attention(
            q=absorbed.contiguous(),
            k=None,
            v=None,
            output=out,
            is_fused_qkv=True,
            num_kv_heads=1,
            head_size=d.lat_dim,
            v_head_dim=d.kv_lora,
            attention_input_type=MLA_GENERATION,
            num_contexts=int(md.num_contexts),
            num_ctx_tokens=0,
            sparse_attn_indices=packed,
            num_sparse_topk=None if packed is None else int(packed.shape[1]),
            latent_cache=latent_cache,
            q_pe=q_pe,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            **self._generation_args(md, rope, packed),
            **_INERT,
        )

    def _generation_args(self, md, rope: dict, packed: torch.Tensor | None) -> dict:
        """`_step_args` for the generation entry, with a step-invariant KV bound.

        The entry's host KV length is a launch scalar, not the step's key bound:
        a dense row is bounded by the device sequence lengths and a sparse row
        by the width of its own selection, so any value at or above that bound
        selects the same work. A captured CUDA graph bakes the scalar it reads
        at capture and replays it at every step, and the request a graph is
        captured on need not be the one it replayed for -- so the entry is
        handed a bound that does not move with the step rather than the step's
        own length. On a sparse layer that bound is the selection's width, the
        rows the entry reads; on a windowed layer it is the pool's maximum
        sequence length, which keeps a history that has outgrown the window on
        the sliding mask, at capture and at every replay alike. A smaller value
        would let a capture take the dense mask over a history longer than the
        window, where the layer would attend past its own window.
        """
        args = self._step_args(md, rope)
        # `_step_args` carries the context call's one-row-per-request layout;
        # the generation rows are this host's `gen_width` per request.
        args["predicted_tokens_per_seq"] = self.gen_width
        bound = int(packed.shape[1]) if packed is not None else self._pool_max(md)
        args["host_past_key_value_lengths"] = torch.full_like(md.kv_lens_runtime, bound)
        return args

    def epilogue(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid gate, `o_proj` and `post_proj_norm`, over the `v_dim`-wide output."""
        d, w, i = self.d, self.core.w, self.layer
        gated = gated_attn_output(attn, self._gate(x))
        out = self._linear(gated, f"l{i}_o_proj", f"l{i}_o_proj_scale")
        if d.attn_tp > 1:
            out = self.core._allreduce(out)
        return _rms(out, w[f"l{i}_post_proj_norm"], d.rms_eps)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x, f"l{self.layer}_linear_gate", f"l{self.layer}_linear_gate_scale")

    def _linear(self, x: torch.Tensor, weight: str, scale: str) -> torch.Tensor:
        """Block-FP8 on the quantized layers, bf16 on layer 0."""
        w = self.core.w
        if f"{weight}_scale" in w or scale in w:
            q, s = fp8_quantize_1x128(x)
            return block_fp8_gemm(q, w[weight], s, w[scale], output_dtype=x.dtype)
        return cublas_mm(x, w[weight].t())


class Hyv4FlashCore(DecoderModel):
    """The 42-layer text core."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        m = model_config.mapping
        assert m.pp_size == 1, "pipeline parallelism is not implemented"
        assert m.cp_size == 1, "context parallelism is not implemented"
        assert m.moe_cluster_size == 1, "moe_cluster_size is not implemented"
        assert m.enable_attention_dp or m.attn_tp_size == m.tp_size, (
            f"attention TP {m.attn_tp_size} differs from TP {m.tp_size}"
        )
        assert not m.enable_attention_dp or m.moe_tp_size * m.moe_ep_size == m.tp_size, (
            "attention DP gathers the ranks' rows over the TP group, so the expert split "
            f"must tile that group: moe_tp {m.moe_tp_size} x moe_ep {m.moe_ep_size} "
            f"does not cover TP {m.tp_size}"
        )

        self.dims = dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(self.dims)
        self.engram_tables = _weights.MODEL_WEIGHTS.engram_tables(self)
        self._loading_layer = 0
        # Rows this core contributes per generating request in one step. Under
        # speculation the engine's verify step processes a whole draft block --
        # `tokens_per_gen_step` = `max_draft_len + 1` rows per request, the
        # block width the drafter's own layers read -- and the generation
        # prologue and entry lay those rows out by it. Without speculation a
        # step is one row per request.
        spec_config = getattr(model_config, "spec_config", None)
        self.gen_width = int(spec_config.tokens_per_gen_step) if spec_config is not None else 1
        # The plain sum: no residual or norm fused in, so the fusion operands
        # are None (`eps` is unread for NONE) and nothing may complete early.
        self._ar_args = dict(
            allreduce_group_args(m),
            op=PLAIN,
            residual=None,
            norm_weight=None,
            scale=None,
            bias=None,
            eps=self.dims.rms_eps,
            trigger_completion_at_end=False,
        )
        # The attention-DP row exchange, the one collective pair whose operands
        # are the step's own rows: the group whose rows a step gathers, and the
        # communicator flavor the engine's other collectives run on. `mpi_disabled`
        # decides between the MPI-session symbols and a boxed c10d process group;
        # a one-rank group exchanges nothing.
        self._dp_group = list(m.tp_group)
        self._dp_pg = m.tp_group_pg.boxed() if mpi_disabled() else None
        self._rope: dict[float, dict] | None = None
        self._rope_positions = 0
        self._engram_history = EngramHistory(self)
        self._prefill: PrefillTarget | None = None
        self._decode: DecodeTarget | None = None
        # This step's vision rows and the placeholder rows they replace, parked by the
        # shell before it calls the inherited forward (which drops `multimodal_params`).
        self._vision: tuple[torch.Tensor, torch.Tensor] | None = None

    # -- lifecycle ---------------------------------------------------------

    def _rope_tables(self, device, positions: int, theta: float) -> dict:
        """The duplicated float2 GPT-J rope table the MLA ops read.

        Pairs are consecutive channels (`2p`, `2p + 1`), rotated by
        `theta ** (-2p / rope_dim)`; the table duplicates the first half into the
        second, which is the layout the kernels read. `positions` is a row count:
        every row depends only on its own index.
        """
        rope_dim = self.dims.rope_dim
        half = rope_dim // 2
        inv = theta ** (-2.0 * torch.arange(half, dtype=torch.float64) / rope_dim)
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos(), ang.sin()
        table = torch.empty(positions, rope_dim, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        return {
            "rotary_cos_sin": table.reshape(1, positions * rope_dim * 2)
            .float()
            .to(device)
            .contiguous(),
            "rotary_inv_freq": inv.float().to(device).contiguous(),
            "rope_dim": rope_dim,
            "rope_base": theta,
        }

    def post_load_weights(self) -> None:
        """Build the rope tables, the Engram moduli and the two phase targets."""
        cfg = self.model_config.pretrained_config
        device = self.w["final_norm"].device
        # A row per position; the table is grown on demand in `forward` when a
        # step asks for more, never built for the config's 1M maximum up front.
        self._rope_positions = min(int(cfg.max_position_embeddings), 8192)
        self._rope = {
            theta: self._rope_tables(device, self._rope_positions, theta)
            for theta in sorted({g.theta for g in self.dims.layers})
        }
        self._engram_constants()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _engram_constants(self) -> None:
        """The hash multipliers and per-head moduli, re-derived from the config.

        HF's `HYV4FlashEngramLayer.draw_hash_multipliers`: one odd multiplier
        per n-gram offset, drawn from `np.random.default_rng(engram_seed +
        10007 * layer_id)` over `[0, int64_max // compressed_vocab // 2)`. The
        checkpoint stores no multiplier, so it is re-drawn here; the bound is
        the loaded map's compressed vocabulary, which is how HF re-draws it
        after loading (`_init_weights`).
        """
        import numpy as np

        d = self.dims
        self.hash_multipliers: dict[int, torch.Tensor] = {}
        self.head_moduli: dict[int, list[int]] = {}
        for i in d.engram_layers:
            sizes = d.engram_head_vocab_sizes[d.engram_ids[d.engram_layers.index(i)]]
            self.head_moduli[i] = list(sizes)
            lookup = self.w[f"l{i}_engram_lookup"]
            compressed_vocab = int(lookup.max()) + 1
            half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
            rng = np.random.default_rng(
                d.engram_seed + 10007 * d.engram_ids[d.engram_layers.index(i)]
            )
            drawn = rng.integers(0, half_bound, size=d.engram_max_ngram, dtype=np.int64)
            self.hash_multipliers[i] = torch.tensor(
                (drawn * 2 + 1).tolist(), dtype=torch.int64, device=lookup.device
            )

    # -- collectives -------------------------------------------------------

    def _allreduce(self, x: torch.Tensor) -> torch.Tensor:
        """The plain sum of a row-parallel partial over the attention group."""
        if self.dims.world == 1:
            return x
        return allreduce(x, **self._ar_args)

    def _dp_exchange(self) -> bool:
        """Whether one step's rows are split over the group and must be exchanged.

        Attention DP is the mode that gives each rank its own requests' rows;
        attention TP instead gives every rank every row, which is the allreduce
        path. A group of one rank has nothing to exchange either way.
        """
        return self.dims.attention_dp and len(self._dp_group) > 1

    def _dp_row_sizes(self, md, rows: int) -> list[int] | None:
        """The group's per-rank row counts of this step, for the MoE's round trip.

        `None` when every rank holds the same number of rows -- the even split
        the collectives take without a vector, which is the form a graph-eligible
        decode step is padded to. Otherwise the counts the engine gathered for
        this step, in group order, which is also the order the allgather
        concatenates and the reduce-scatter splits in.
        """
        counts = getattr(md, "all_rank_num_tokens", None)
        assert counts is not None, (
            "an attention-DP step carries the group's row counts on the attention metadata "
            "(`all_rank_num_tokens`, written by the runner's token padding); this step has "
            "none, so the MoE cannot know which split to gather"
        )
        counts = [int(n) for n in counts]
        assert len(counts) == len(self._dp_group), (counts, self._dp_group)
        rank = self._dp_group.index(self.dims.rank)
        assert counts[rank] == rows, (
            f"rank {self.dims.rank} holds {rows} rows but all_rank_num_tokens says "
            f"{counts[rank]} ({counts}); the MoE's round trip would gather the wrong split"
        )
        return None if min(counts) == max(counts) else counts

    # -- forward -----------------------------------------------------------

    def _pool_row(self, md, layer: int) -> int:
        """The decoder layer's index inside the primary pool.

        One primary pool holds the text layers in decoder order, so the pool's
        local layer index is the decoder layer index; a layer with no row of its
        own (a windowed layer whose latent lives in the same pool) still reads
        its own rows there.
        """
        return layer

    def _pool_rows(self, md, layer: int, packed: torch.Tensor) -> torch.Tensor:
        """One layer's selection as primary-pool token rows, from request-local positions.

        The sparse MLA entry addresses the pool the prologue appended to as a base
        pointer plus a row index, so a request-local key position must be resolved
        through the step's block table to `physical_page * (pool_pages_per_layer *
        tokens_per_block) + layer_in_pool * tokens_per_block + token_in_page`. The
        block table and the per-token request map are step-level and shared by the
        context and generation phases, so one conversion serves a mixed step; the
        pool's page scale and this layer's offset come from the manager that built
        the pool. Unresolvable positions stay -1, the padding the kernel skips.
        """
        manager = md.kv_cache_manager
        scale, layer_offset = manager.get_primary_pool_page_index_params(self._pool_row(md, layer))
        # The block table this resolves against was built with the manager's own
        # page size, so the conversion takes that size rather than the dims'.
        tokens_per_block = int(manager.tokens_per_block)
        return convert_req_index_to_global(
            md.req_idx_per_token[: packed.shape[0]],
            md.block_table,
            packed,
            tokens_per_block,
            int(packed.shape[1]),
            int(scale) * tokens_per_block,
            int(layer_offset),
        )

    def _layer_forward(
        self,
        md,
        i: int,
        streams: torch.Tensor,
        row_batch: torch.Tensor,
        position_ids: torch.Tensor,
        ngram_tokens: torch.Tensor | None,
        engram_mask: torch.Tensor | None,
        rope: dict,
        picks: cache_manager.DraftPicks | None = None,
        picks_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One decoder layer over `[T, hc, hidden]` residual streams."""
        d, w = self.dims, self.w
        geom = d.layers[i]

        # HF adds the memory to the streams before the layer's hyper-connection
        # reads them, so the attention and the FFN both see it.
        if i in d.engram_layers:
            streams = streams + EngramLayer(self, i).forward(streams, ngram_tokens, engram_mask)
        residual = streams
        h, gates = ihc_mapping(
            streams,
            w[f"l{i}_hc_attn_fn"],
            w[f"l{i}_hc_attn_base"],
            w[f"l{i}_hc_attn_scale"],
            d.hc_magnitude,
            d.hc_eps,
            d.rms_eps,
        )
        h = _rms(h, w[f"l{i}_input_layernorm"], d.rms_eps)

        q_resid = _rms(cublas_mm(h, w[f"l{i}_q_a_proj"].t()), w[f"l{i}_q_a_layernorm"], d.rms_eps)
        kv = cublas_mm(h, w[f"l{i}_kv_a_proj_with_mqa"].t())
        latent = kv[:, : d.kv_lora + d.rope_dim]
        latent[:, : d.kv_lora] = _rms(latent[:, : d.kv_lora], w[f"l{i}_kv_a_layernorm"], d.rms_eps)
        q = self._expand_q(q_resid, i)
        layer = MlaLayer(self, i)
        selected = None
        if geom.sparse:
            selected, _, _ = layer.indexer.pool_and_select(q_resid, h, md, row_batch, position_ids)
        packed = _decode_selection(geom, selected)
        if packed is not None:
            # The selection is in request-local key positions; the sparse entry
            # reads rows of the primary KV pool, so resolve it to those rows.
            packed = self._pool_rows(md, i, packed)
            if picks is not None:
                # Published for DFly2's draft layers, which attend over this
                # layer's picks. The same rows the sparse entry reads, so the
                # write is a side effect and takes no decision of its own; a
                # layer outside the checkpoint's draft-source list writes none.
                picks.write(i, packed, picks_rows)
        # A step may carry both phases: the context rows first, then one
        # generation row per decoding request. Each phase's prologue takes only
        # its own rows; the sequence offset `num_contexts` tells both native
        # entries which sequences those rows belong to.
        num_ctx = int(md.num_contexts)
        tc = int(md.num_ctx_tokens) if num_ctx else 0
        gen = h.shape[0] - tc
        if tc and gen:
            attn = torch.cat(
                [
                    layer.context_rows(q[:tc], latent[:tc], md, rope),
                    layer.generation_partial(
                        q[tc:],
                        latent[tc:],
                        md,
                        rope,
                        _phase_rows(packed, tc, None),
                    ),
                ],
                dim=0,
            )
        elif tc:
            attn = layer.context_rows(q, latent, md, rope)
        else:
            attn = layer.generation_partial(q, latent, md, rope, packed)
        attn = layer.epilogue(attn, h)
        streams = _hc_residual(_hc_post(gates, d.hc_mult), attn, residual)

        residual = streams
        h, gates = ihc_mapping(
            streams,
            w[f"l{i}_hc_mlp_fn"],
            w[f"l{i}_hc_mlp_base"],
            w[f"l{i}_hc_mlp_scale"],
            d.hc_magnitude,
            d.hc_eps,
            d.rms_eps,
        )
        h = _rms(h, w[f"l{i}_post_attention_layernorm"], d.rms_eps)
        if i < d.first_dense:
            h = self._dense_mlp(h, i)
            h = _rms(h, w[f"l{i}_post_fc2_norm"], d.rms_eps)
        else:
            h = LatentMoe(self, i).forward(h, md)
        return _hc_residual(_hc_post(gates, d.hc_mult), h, residual)

    def _expand_q(self, q_resid: torch.Tensor, i: int) -> torch.Tensor:
        """`[T, heads_local, qk_dim]`: the unabsorbed query, `q_b` on its quantized layers."""
        d, w = self.dims, self.w
        geom = d.layers[i]
        if i >= d.first_dense:
            q, s = fp8_quantize_1x128(q_resid)
            flat = block_fp8_gemm(
                q, w[f"l{i}_q_b_proj"], s, w[f"l{i}_q_b_proj_scale"], output_dtype=q_resid.dtype
            )
        else:
            flat = cublas_mm(q_resid, w[f"l{i}_q_b_proj"].t())
        return flat.view(-1, geom.heads_local, d.qk_dim)

    def _dense_mlp(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Layer 0's plain SwiGLU, its intermediate split over the attention group."""
        w, d = self.w, self.dims
        up = flashinfer_silu_and_mul(
            cublas_mm(x, torch.cat([w[f"l{i}_dense_gate"], w[f"l{i}_dense_up"]], dim=0).t())
        )
        out = cublas_mm(up, w[f"l{i}_dense_down"].t())
        if d.mlp_tp > 1:
            out = self._allreduce(out)
        return out

    def _head(self, streams: torch.Tensor) -> torch.Tensor:
        """`hc_head` then `model.norm`: the four streams collapse to one hidden state."""
        d, w = self.dims, self.w
        h, _ = ihc_mapping(
            streams,
            w["hc_head_fn"],
            w["hc_head_base"],
            w["hc_head_scale"],
            d.hc_magnitude,
            d.hc_eps,
            d.rms_eps,
        )
        return _rms(h, w["final_norm"], d.rms_eps)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        md = attn_metadata
        # The engine hands a non-MRoPE model `[1, T]` position ids and an MRoPE
        # one `[3, 1, N]` (`token_model_inputs` in the decoder's common
        # runner); the core reads one scalar key position per token, so take a
        # single axis and flatten it.
        if position_ids is not None and position_ids.dim() > 1:
            position_ids = position_ids[0].reshape(-1)
        wanted = int(getattr(md, "max_seq_len", 0) or 0)
        if wanted > self._rope_positions:
            self._rope_positions = min(
                wanted, int(self.model_config.pretrained_config.max_position_embeddings)
            )
            self._rope = {
                theta: self._rope_tables(self.w["final_norm"].device, self._rope_positions, theta)
                for theta in self._rope
            }
        if inputs_embeds is None:
            inputs_embeds = torch.nn.functional.embedding(input_ids, self.w["embed"])
        # This step's media, if any: the shell ran the tower and parked the rows
        # with the placeholder rows they replace. Overwrite the placeholders in
        # place, so the streams below start from the image/video rows exactly as
        # they start from an embedded token.
        vision, self._vision = self._vision, None
        if vision is not None:
            rows, mm_token_indices = vision
            inputs_embeds = inputs_embeds.index_copy(
                0, mm_token_indices, rows.to(inputs_embeds.dtype)
            )
        d = self.dims
        # HF `HYV4FlashModel.forward`: every hyper-connection stream starts from
        # the same embedding, `[T, hc, hidden]`.
        streams = inputs_embeds.unsqueeze(1).expand(-1, d.hc_mult, -1).contiguous()
        row_batch = _row_batch(md)
        ngram_tokens = engram_mask = None
        if d.engram_layers:
            # HF `build_engram_inputs`: the n-gram window over the step's tokens,
            # and the positions that own one (the pad token does not). The
            # tokens before the step's first row come from the per-request
            # history, so a decode row hashes the n-gram the checkpoint means.
            ngram_tokens = _ngram_tokens(
                input_ids, position_ids, row_batch, d.engram_max_ngram, d.engram_pad_id
            )
            history = self._engram_history.lookback(
                md, input_ids, position_ids, row_batch, d.engram_pad_id
            )
            if history is not None:
                ngram_tokens[:, 1:] = torch.where(
                    ngram_tokens[:, 1:] == d.engram_pad_id, history, ngram_tokens[:, 1:]
                )
            engram_mask = input_ids.view(-1) != d.engram_pad_id
        picks = None
        picks_rows = None
        if d.draft_source_layers:
            # DFly2's draft layers read the sparse layers' picks back out of the
            # manager, which is the one object a draft model that shares this
            # cache reaches. One line per step is all the write costs; the
            # allocation is the size of an eager step's worth of rows.
            picks = md.kv_cache_manager.draft_picks(
                d.draft_source_layers,
                d.index_topk,
                rows=d.max_num_tokens,
                device=streams.device,
            )
            picks_rows = _picks_rows(md)
            if int(md.num_tokens) > picks.history.shape[1]:
                raise ValueError(
                    f"this step carries {int(md.num_tokens)} rows and the picks state holds "
                    f"{picks.history.shape[1]}; its row count is the engine's per-step token "
                    f"budget, and a row outside it must be published through a larger state"
                )
        for i in range(d.num_layers):
            streams = self._layer_forward(
                md,
                i,
                streams,
                row_batch,
                position_ids,
                ngram_tokens,
                engram_mask,
                self._rope[d.layers[i].theta],
                picks,
                picks_rows,
            )
        if picks is not None:
            _log_picks(picks, md, picks_rows)
        return self._head(streams)

    def _select_target(self, md) -> "PrefillTarget | DecodeTarget":
        return self._decode if md.num_contexts == 0 else self._prefill


class PrefillTarget:
    """The context step: one forward, no cached-context specialization yet."""

    def __init__(self, core: Hyv4FlashCore) -> None:
        self.core = core

    def forward(self, md, **kwargs) -> torch.Tensor:
        return self.core.forward(md, **kwargs)


class DecodeTarget:
    """The generation step: one forward, one token (or one draft block) per sequence."""

    def __init__(self, core: Hyv4FlashCore) -> None:
        self.core = core

    def forward(self, md, **kwargs) -> torch.Tensor:
        return self.core.forward(md, **kwargs)


class ModelingV2Hyv4FlashForConditionalGeneration(DecoderModelForCausalLM):
    """The shell: the engine's lm_head, logits processor and lifecycle over the text core.

    With a one-engine DFly2 `speculative_config` this class is also the model the
    engine's speculative path reads: it builds the target's own checkpoint drafter
    (:class:`~.draft.Hyv4FlashDraft`, five block-parallel MLA layers over the
    target's sparse pools) and the worker that runs it
    (:class:`~.spec_worker.Hyv4FlashDFlashWorker`) in place of the engine's
    :class:`~tensorrt_llm._torch.speculative.dflash.DFlashWorker`, and its
    `forward` is the one-engine step: target hidden states, the target logits the
    worker's acceptance reads, then the worker (acceptance, drafting).

    The engine's separate-draft read is what these attributes serve: `draft_config`
    is None because the drafter has no architecture of its own -- it is five
    `model.dfly_layers.*` layers of this same checkpoint -- so the read reuses the
    target's weight mapper with the drafter's `checkpoint_key_prefixes`, and
    `load_draft_weights` hands that read's tensors to the drafter.
    """

    runtime_hooks = RuntimeHooks(kv_cache_manager_cls=cache_manager.Hyv4FlashCacheManager)

    #: This target does not take the one-engine speculative fast paths (the
    #: vocabulary-shard key exchange and the graph variants that come with it);
    #: its worker verifies through the gathered logits.
    spec_decode_fast_paths: bool = False

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        text = getattr(cfg, "text_config", cfg)
        super().__init__(
            Hyv4FlashCore(model_config),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # The engine's speculative-decoding reads. `spec_config` and `spec_worker`
        # are None without speculation, so the forward below is the plain one;
        # `layer_idx` is the id the engine captures the final hidden state under.
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        self.layer_idx = -1
        if self.spec_config is not None:
            self._build_drafter(model_config)
        # The two placeholder ids the tower's rows replace: the checkpoint's
        # `image_token_id` / `video_token_id`, the pad tokens its own processor
        # expands to one placeholder row per merged patch row (the `start` and
        # `end` tokens stay text). Built lazily; `__init__` runs under meta init.
        self._mm_token_ids_value = (int(cfg.image_token_id), int(cfg.video_token_id))
        self._mm_token_ids_cache: torch.Tensor | None = None
        # The tower is replicated on every rank and built once the weights are real
        # (`post_load_weights`); `None` until then and for a config without a tower.
        self.vision_tower: _vision.VisionTower | None = None

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """The placeholder ids whose rows the tower's embeddings replace.

        The executor finds the placeholder rows by these ids
        (`prepare_multimodal_indices`), which is what pairs them with the tower's
        rows without a host-side scan of the prompt."""
        if self._mm_token_ids_cache is None:
            self._mm_token_ids_cache = torch.tensor(self._mm_token_ids_value, dtype=torch.int32)
        return self._mm_token_ids_cache

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """The `multimodal_data` tensors moved to the device; the `(t, h, w)` grids stay host-side."""
        return list(_vision.DEVICE_PATHS)

    def validate_request(self, request) -> None:
        """Reject at admission a media request the vision tower would assert on."""
        _vision.validate_request(self.vision_tower, request)

    @classmethod
    def get_model_defaults(cls, llm_args):
        return {"lm_head_output_dtype": "float32"}

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config):
        return "V2"

    def _build_drafter(self, model_config: ModelConfig) -> None:
        """Build the checkpoint's own DFly2 drafter and the worker that runs it.

        The drafter is five `model.dfly_layers.*` layers of this same
        checkpoint over the target's sparse layers' latent pools
        (:class:`~.draft.Hyv4FlashDraft`), so it is built from the core the
        shell already holds and reads the target's head and logits processor
        from it. The worker (:class:`~.spec_worker.Hyv4FlashDFlashWorker`)
        replaces the engine's `DFlashWorker` for the same reason: its
        preparation talks about a private draft cache this drafter does not
        have.
        """
        spec_config = self.spec_config
        if not (
            spec_config.spec_dec_mode.is_dflash() and spec_config.spec_dec_mode.use_one_engine()
        ):
            raise ValueError(
                "This target carries DFly2's drafter in the target checkpoint; it "
                f"serves it in one engine, got {spec_config.spec_dec_mode=} with "
                f"{spec_config.uses_external_draft_model=}"
            )
        # The drafter's modules import this one (`MlaLayer`), so they load here.
        from .draft import Hyv4FlashDraft
        from .spec_worker import Hyv4FlashDFlashWorker

        # The drafter owns no head and no KV cache of its own: its block is
        # written into the target sparse layers' pools, and its drafts are drawn
        # through the target's own head and logits processor
        # (`Hyv4FlashDraft.block_logits`).
        drafter = Hyv4FlashDraft(self.model)
        drafter.logits_processor = self.logits_processor
        drafter.lm_head = self.lm_head
        self.draft_model = drafter
        self.epilogue.append(drafter)

        # The engine's helper reads the config the same way; a config that asked
        # for a second, worker-owned draft cache under TRTLLM/FA4 would be one
        # this drafter does not keep.
        from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache

        if should_use_separate_draft_kv_cache(spec_config):
            raise ValueError(
                "The DFly2 drafter keeps no KV cache of its own: its block lives in "
                "the target sparse layers' pools. Serve it with the drafter's own "
                "attention backend (dflash attention_backend VANILLA)."
            )
        self.use_separate_draft_kv_cache = False

        worker = Hyv4FlashDFlashWorker(
            spec_config, model_config.mapping, self.use_separate_draft_kv_cache
        )
        worker.set_draft_model(drafter)
        worker.set_target_model(self)
        self.spec_worker = worker
        self.epilogue.append(worker)

    def load_draft_weights(self, weights, weight_mapper=None) -> None:
        """Fill the checkpoint's DFly2 draft from the draft read's tensors.

        The second read is the target checkpoint narrowed by the drafter's
        `checkpoint_key_prefixes` (`model.dfly_layers.`); the mapper is the
        target's own, since the draft has no architecture of its own to resolve
        one from (`draft_config` is None).
        """
        self.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)

    def set_guided_decoder(self, guided_decoder) -> bool:
        """Hand the guided decoder to the worker, which masks the target logits."""
        if self.spec_worker is None:
            return False
        return self.spec_worker.set_guided_decoder(guided_decoder)

    def load_weights(self, weights, *args, **kwargs):
        # A reload rebuilds everything `post_load_weights` derives from the weights.
        self.model._prefill = self.model._decode = None
        self.vision_tower = None
        _weights.MODEL_WEIGHTS.load(self, weights)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """One engine step: the media first, then the target, then the drafter.

        Without a speculative config this is the inherited forward. With one it
        is the one-engine step the engine's speculative path reads: the target's
        hidden states for every row of the step, the target logits the worker's
        acceptance reads (the rows `spec_metadata.gather_ids` names), then the
        worker, which accepts the drafts and drafts the next block.

        The inherited `DecoderModelForCausalLM.forward` drops `multimodal_params`,
        so the tower runs here and the core picks the rows up on its way in."""
        self.model._vision = _vision.multimodal_embeds(self.vision_tower, kwargs)
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )

        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden)
        if attn_metadata.padded_num_tokens is not None:
            # A captured batch is padded to a graph's width; the step's own rows
            # are `num_tokens` of them, and every row index the drafter reads
            # (the gather ids, the block's source rows) is in that layout.
            num_tokens = attn_metadata.num_tokens
            hidden = hidden[:num_tokens]
            input_ids = input_ids[:num_tokens] if input_ids is not None else None
            position_ids = position_ids[..., :num_tokens] if position_ids is not None else None

        # The target logits the worker's acceptance reads, in the gathered
        # layout: this worker declines the vocabulary-shard selection, so these
        # are full fp32 logits on every rank.
        logits = self.spec_worker.target_logits(
            hidden[spec_metadata.gather_ids],
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        if self.model.dims.vision is None:
            return
        self.vision_tower = _vision.VisionTower(self.model.w, self.model.dims.vision)
        # The CPU-only load checks build the shell over host weights; they launch
        # no kernel and have no device to warm up on.
        if self.lm_head.weight.is_cuda:
            self.vision_tower.warm_up()
