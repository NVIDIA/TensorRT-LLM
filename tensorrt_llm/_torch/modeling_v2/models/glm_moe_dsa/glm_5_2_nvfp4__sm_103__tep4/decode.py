# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DecodeTarget: a pure decode step (no context rows) of up to 64 rows.

Above 64 rows a decode step is `WideDecodeTarget`'s; only an engine built without the wide
decode cells (`TRTLLM_GLM52_WIDE_DECODE=0` at MTP1 or without speculation) keeps those steps
here, on the context-sized cells the ops pick for them. The forward is the layer loop written
out: the ops it calls pick their kernels by the row count, and the `Phase` it passes says which
cells those rows map onto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2._router_index import step_contract_enabled

from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.cute_dsl_sparse_mla_decode import INDEX_KIND_GLOBAL
from .ops.attention.indexer import (
    indexer_decode_logits,
    indexer_decode_topk,
    indexer_q_side,
    indexer_topk_buf,
    indexer_write_k,
)
from .ops.attention.mla_rope import mla_rope
from .ops.attention.mtp_index_share import mtp_share_load, mtp_share_reuses, mtp_share_store
from .ops.attention.sparse_mla import decode_slots, sparse_mla_decode
from .ops.comm.cute_dsl_allreduce import allreduce
from .ops.comm.cute_dsl_allreduce_add_rmsnorm import allreduce_add_rmsnorm
from .ops.comm.cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 import allreduce_add_rmsnorm_quant
from .ops.elementwise.cute_dsl_hidden_add import hidden_add
from .ops.embedding.cute_dsl_embedding import embed
from .ops.gemm.absorbed_bmm import k_b_q_base, v_b
from .ops.gemm.dense_mlp import dense_down, dense_gate_up_swiglu
from .ops.gemm.indexer_fold import q_b_ix_wq_b, qkva_ix_fold
from .ops.gemm.indexer_pre_proj import indexer_pre_proj
from .ops.gemm.mtp_eh_proj import mtp_eh_norm, mtp_eh_proj
from .ops.gemm.projection import DECODE_MAX_ROWS, projection
from .ops.gemm.shared_mlp import shared_down, shared_gate_up_swiglu
from .ops.moe.routed_moe import (
    moe_combine,
    moe_experts,
    moe_l2_prefetch,
    moe_open,
    moe_quantize_first,
    moe_router,
    moe_topk,
    mtp_moe_open,
)
from .ops.norm.cute_dsl_hidden_add_rmsnorm import hidden_add_rmsnorm
from .ops.norm.cute_dsl_hidden_rmsnorm import hidden_rmsnorm
from .ops.norm.qkv_a_rmsnorm import qkv_a_rmsnorm
from .step import DRAFT, PREFILL, Phase, build_step_args

if TYPE_CHECKING:
    from .modeling import GlmMoeDsaTep4ModelingV2


class DecodeTarget:
    """Routed to on a step without context rows, unless it carries more than 64 rows and the
    engine built the wide decode cells (`WideDecodeTarget`).

    Two paths only the decode Targets take, both at the MTP widths they are built for
    (`mtp_widths`: 2, and 4 / 6 at MTP3 / MTP5): the specialised MLA rope kernel, and pool rows
    computed once per selection rather than once per layer (the decode kernel adds
    `layer * 32` itself).
    """

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # An MTP3 / MTP5 engine's decode rows without a cell of their own run on the next cell
        # up (`TRTLLM_GLM52_DEC_CELL_UP`); MTP1 and no speculation keep the stock op there.
        self.phase = Phase(up="decode") if core.dec_cell_up else PREFILL
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: their `step_args` projections differ, so
        # probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def on_load(self) -> None:
        """This Target's cross-op handoffs, decided once from the core's handles after load
        (an upstream op then writes its downstream op's operand where the table says so):

        - `_qb_pre[i]`: layer `i`'s q_a norm writes its decode q_b MXF8 GEMM's operand
          (`qkv_a_rmsnorm`), where the norm's q8 kernel and that GEMM built.
        - `_vb_pre[i]`: layer `i`'s V-B writes its decode o_proj MXF8 GEMM's operand
          (`absorbed_bmm.v_b`): trunk layers whose o_proj has that GEMM.
        - `_quant_rows`: the row counts up to which a trunk MoE layer's pre-MoE collective
          also writes the routed experts' NVFP4 input: the decode widths.

        No collective hands this Target an MXF8 operand: the all-reduce writes one only
        two-shot (above 512 rows) and never under capture."""
        core = self.core
        qb8 = core._attn_h.norm.q8 is not None and core._proj_h.q_b.dec8 is not None
        n = core.num_layers
        self._qb_pre = [qb8 and a.qb_dec8 is not None for a in core._attn]
        self._vb_pre = [i < n and a.o_dec8 is not None for i, a in enumerate(core._attn)]
        self._quant_rows = DECODE_MAX_ROWS

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes. Everything checked is fixed at engine construction, so once per target
        instance is sound, and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """States the two context counts instead of reading them back: this target runs only
        when `num_contexts` is zero."""
        step = build_step_args(md)
        step["num_contexts"] = 0
        step["num_ctx_tokens"] = 0
        return step

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        proj, comm, dense, shared_h = core._proj_h, core._comm, core._dense_h, core._shared_h
        ah = core._attn_h
        moe_h = core._moe_h
        ph = self.phase
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        h = (
            embed(input_ids, core.w["embed"], core._embedding)
            if inputs_embeds is None
            else inputs_embeds
        )
        rows = h.shape[0]
        assert rows == md.num_tokens, (rows, md.num_tokens)
        gen_p = rows // md.num_seqs
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        ixh, ix_state = core._ix_h, core._ix_state
        topk_buf = indexer_topk_buf(md, rows, 0, True, skip_gen, ixh)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        # The step's selection kind and base rows (`decode_slots`), and the MTP5 atom split's
        # indexer metadata, built by the first indexer layer for all.
        slot_memo: dict = {}
        split_memo: dict = {}
        x = hidden_rmsnorm(h, core._attn[0].norm1, core.eps, core._rows.norm)
        residual = h
        for i in range(core.num_layers):
            a = core._attn[i]
            ix = core._idx[i]
            # Attention.
            qkva, kw = qkva_ix_fold(ph, x, a.qkva, a.qkva8, None, a.ix_fold_w, ah.qkva_fold)
            q_a, latent, _, qb_pre = qkv_a_rmsnorm(
                qkva, a.q_norm, a.kv_norm, ah.norm, self._qb_pre[i]
            )
            q, _ = q_b_ix_wq_b(ph, q_a, a.qb, None, a.qb_dec8, qb_pre, ah.q_b_fold)
            # The indexer layers' selection (the others reuse the last one's).
            if ix is not None:
                k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
                kb, wq = indexer_pre_proj(ph, x, ix.fused, ix.fused_bf16, kw, ixh.pre_proj)
                if not skip_gen:
                    ixq = projection(ph, q_a, ix.qb, proj.ix_qb)
                    q_fp8, weights, kb = indexer_q_side(
                        ph, q_a, ixq, ix.qb, pos, wq, kb, ix.kn_w, ix.kn_b, md, k_pool, ix_state,
                        ixh,
                    )  # fmt: skip
                indexer_write_k(md, k_pool, kb, ix.kn_w, ix.kn_b, pos, ix_state, ixh)
                if not skip_gen:
                    logits = indexer_decode_logits(
                        md, k_pool, q_fp8, weights, 0, gen_p, split_memo, ixh
                    )
                    indexer_decode_topk(md, logits, topk_buf, 0, gen_p, ixh)
            sel = decode_slots(
                md, req_idx, topk_buf, ix is not None, core._pool_layer_id[i], gen_p, slot_memo,
                ah.fmha,
            )  # fmt: skip
            query = k_b_q_base(ph, q, a.k_b, a.k_b_nk, gen_p, "trunk", ah.k_b)
            quant_q, cu_q, bmm1, bmm2 = mla_rope(query, latent, md, i, gen_p, 0, ah.rope)
            lat_out = torch.empty([rows, core.heads * core.kv_lora], dtype=h.dtype, device=h.device)
            sparse_mla_decode(quant_q, md, sel, lat_out, cu_q, bmm1, bmm2, 0, gen_p, ah.fmha)
            o_pre = v_b(ph, lat_out, attn_out, a.v_b_t, a.v_b_nk, self._vb_pre[i], ah.v_b)
            # o_proj, the MLP and the closing collective.
            partial = projection(
                ph, attn_out, a.o, proj.o_proj, w8=a.o8, wd8=a.o_dec8, dec_pre=o_pre
            )
            gu, dn = core._mlp[i]
            m = core._moe[i]
            if m is None:
                w8 = core._dense_w8[i]
                o, _ = allreduce_add_rmsnorm(partial, residual, a.norm2, comm)
                act, act8 = dense_gate_up_swiglu(ph, o, gu, w8, None, dense)
                mlp_out = dense_down(ph, act, act8, dn, w8, dense)
            else:
                # The decode L2 prefetch of this layer's routed FC1 weights.
                pf = moe_l2_prefetch(i, m, rows, False, moe_h)
                o, _, pre = allreduce_add_rmsnorm_quant(
                    partial, residual, a.norm2, m.g1, core._ar_quant, self._quant_rows
                )  # fmt: skip
                slot, acc = moe_open(rows, False, moe_h, core._moe_parity)
                main = torch.cuda.current_stream()
                side = core._side_stream
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    act, on_pre = shared_gate_up_swiglu(
                        ph, o, gu, core._shared_gu_il[i], core._shared_w8[i], None, shared_h, True
                    )
                    shared = shared_down(ph, act, dn, shared_h, acc, on_pre)
                quantized = moe_quantize_first(o, m, pre is not None, moe_h)
                logits, in_slot = moe_router(ph, o, m, slot, False, moe_h)
                routing = moe_topk(logits, in_slot, m, slot, i, False, moe_h)
                routed = moe_experts(o, routing, m, quantized, slot, acc, side, pf, i, False, moe_h)
                mlp_out = moe_combine(routed, shared, acc, rows, main, side, moe_h)
            x, _ = allreduce_add_rmsnorm(mlp_out, residual, core._next_norm[i], comm)
        return x

    def mtp_forward(
        self,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        hidden_states,
        attn_metadata,
        position_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step, decode rows only. Pool rows go through the stock conversion against
        the draft pool's live mapping. Run eagerly on an engine drafting more than one token (a
        context or mixed step's draft loop), its row counts without a decode cell run on the
        next cell up (`DRAFT`)."""
        core = self.core
        eager = core.max_draft_len > 1 and not torch.cuda.is_current_stream_capturing()
        ph = DRAFT if eager else self.phase
        md = attn_metadata
        rows = hidden_states.shape[0]
        gen_p = rows // md.num_seqs
        j = core.num_layers
        a = core._attn[j]
        ix = core._idx[j]
        e = embed(input_ids, core.w["embed"], core._embedding)
        eh = mtp_eh_norm(ph, e, hidden_states, core._mtp_eh_h)
        x = mtp_eh_proj(ph, eh, core._mtp_eh_h)
        residual = x
        xn = hidden_rmsnorm(x, a.norm1, core.eps, core._rows.norm)
        pos = torch.reshape(position_ids, [-1])
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        ixh, ix_state = core._ix_h, core._ix_state
        topk_buf = indexer_topk_buf(md, rows, 0, True, skip_gen, ixh)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        # With the engine's index share on, a draft step after the first takes draft step 0's
        # selection (`mtp_share_load`): the indexer's q side, scorer and top-k do not run; its
        # keys still project and append. Draft step 0 stashes.
        share = ix is not None and core.mtp_index_share
        share_reuse = share and mtp_share_reuses(md)
        select = not skip_gen and not share_reuse
        ah = core._attn_h
        qkva, kw = qkva_ix_fold(ph, xn, a.qkva, a.qkva8, None, a.ix_fold_w, ah.qkva_fold)
        q_a, latent, _, qb_pre = qkv_a_rmsnorm(qkva, a.q_norm, a.kv_norm, ah.norm, self._qb_pre[j])
        q, _ = q_b_ix_wq_b(ph, q_a, a.qb, None, a.qb_dec8, qb_pre, ah.q_b_fold)
        if ix is not None:
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(j)
            kb, wq = indexer_pre_proj(ph, xn, ix.fused, ix.fused_bf16, kw, ixh.pre_proj)
            if select:
                ixq = projection(ph, q_a, ix.qb, core._proj_h.ix_qb)
                q_fp8, weights, kb = indexer_q_side(
                    ph, q_a, ixq, ix.qb, pos, wq, kb, ix.kn_w, ix.kn_b, md, k_pool, ix_state, ixh
                )
            indexer_write_k(md, k_pool, kb, ix.kn_w, ix.kn_b, pos, ix_state, ixh)
            if select:
                logits = indexer_decode_logits(md, k_pool, q_fp8, weights, 0, gen_p, None, ixh)
                indexer_decode_topk(md, logits, topk_buf, 0, gen_p, ixh)
            elif share_reuse and not skip_gen:
                mtp_share_load(md, topk_buf, rows, ixh.count)
            if share and not share_reuse and core._mtp_share_stash and md.in_mtp_draft_loop:
                mtp_share_store(md, topk_buf, 0, 0, rows // gen_p, gen_p)
        # The selection as global rows of the MTP layer's own pool (the separate draft pool's
        # live mapping), on the stock conversion.
        pool_rows = convert_req_index_to_global(
            req_idx, md.block_table, topk_buf, md.tokens_per_block, core.idx_topk, stride,
            pool_layer,
        )  # fmt: skip
        query = k_b_q_base(ph, q, a.k_b, a.k_b_nk, gen_p, "mtp", ah.k_b)
        quant_q, cu_q, bmm1, bmm2 = mla_rope(query, latent, md, local, gen_p, 0, ah.rope)
        lat_out = torch.empty([rows, core.heads * core.kv_lora], dtype=x.dtype, device=x.device)
        sel = (pool_rows, INDEX_KIND_GLOBAL, None)
        sparse_mla_decode(quant_q, md, sel, lat_out, cu_q, bmm1, bmm2, 0, gen_p, ah.fmha)
        v_b(ph, lat_out, attn_out, a.v_b_t, a.v_b_nk, self._vb_pre[j], ah.v_b)
        # The MTP layer's o_proj and MoE.
        partial = projection(ph, attn_out, a.o, core._proj_h.o_proj)
        o, _ = allreduce_add_rmsnorm(partial, residual, a.norm2, core._comm)
        gu, dn = core._mlp[j]
        mtp_h, mtp_m = core._mtp_moe_h, core._mtp_moe
        exact = mtp_moe_open(rows, False, mtp_h)
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            act, on_pre = shared_gate_up_swiglu(
                ph, o, gu, core._shared_gu_il[j], None, None, core._shared_h
            )
            shared = shared_down(ph, act, dn, core._shared_h, None, on_pre)
        slot = core._mtp_parity.take()
        quantized = moe_quantize_first(o, mtp_m, False, mtp_h)
        logits, in_slot = moe_router(ph, o, mtp_m, slot, exact, mtp_h)
        routing = moe_topk(logits, in_slot, mtp_m, slot, j, exact, mtp_h)
        routed = moe_experts(o, routing, mtp_m, quantized, slot, None, None, None, j, exact, mtp_h)
        main.wait_stream(side)
        summed = hidden_add(routed, shared, core._rows.add)
        reduced = allreduce(summed, core._comm)
        # Model structure: an engine drafting more than one token feeds the output back into
        # the next draft step head-normalized (`_mtp_out_normed`); MTP1's `shared_head` norms.
        if core._mtp_out_normed:
            head_norm = core._mtp["head_norm"]
            return hidden_add_rmsnorm(residual, reduced, head_norm, core.eps, core._rows.add_norm)
        return hidden_add(residual, reduced, core._rows.add)
