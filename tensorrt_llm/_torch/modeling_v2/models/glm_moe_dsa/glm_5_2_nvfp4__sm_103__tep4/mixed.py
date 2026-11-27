# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MixedTarget: a step with context rows and generation rows beside them.

The context rows (the first `tc`) take the sm103 prefill kernels, as on `PrefillTarget`: the
context rope/append, the dense-causal or index-path sparse MLA, or the non-absorbed MHA on a
dense-causal step. The generation rows after them take the decode indexer and the sparse MLA
decode on the stock rope op, at the offsets (`ctx`, `tc`) that only this Target's call sites
carry; on the fp8 MHA route they run on their own stream beside the context rows' MHA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2._router_index import step_contract_enabled

from . import context_plan as _plan
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.ctx_mha import mha_kv_expand, mha_kv_gather, mha_prefill, o_operand
from .ops.attention.ctx_rope import ctx_rope_append, ctx_rope_q_in_place
from .ops.attention.ctx_sparse_mla import (
    sparse_mla_prefill_dense_causal,
    sparse_mla_prefill_index,
    sparse_mla_prefill_selected,
)
from .ops.attention.cute_dsl_sparse_mla_decode import INDEX_KIND_GLOBAL
from .ops.attention.indexer import (
    indexer_decode_logits,
    indexer_decode_topk,
    indexer_prefill_select,
    indexer_q_side,
    indexer_topk_buf,
    indexer_write_k,
)
from .ops.attention.mla_rope import mla_rope
from .ops.attention.mtp_index_share import mtp_share_reuses, mtp_share_store
from .ops.attention.sparse_mla import sparse_mla_decode
from .ops.comm.cute_dsl_allreduce import allreduce
from .ops.comm.cute_dsl_allreduce_add_rmsnorm import allreduce_add_rmsnorm
from .ops.comm.cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 import allreduce_add_rmsnorm_quant
from .ops.elementwise.cute_dsl_copy import row_copy
from .ops.elementwise.cute_dsl_hidden_add import hidden_add
from .ops.elementwise.cute_dsl_rows_select import cute_dsl_rows_select
from .ops.embedding.cute_dsl_embedding import embed
from .ops.gemm.absorbed_bmm import absorbed_bmm, k_b_q_base, v_b
from .ops.gemm.ctx_k_b import ctx_k_b
from .ops.gemm.dense_mlp import dense_down, dense_gate_up_swiglu
from .ops.gemm.indexer_pre_proj import indexer_pre_proj
from .ops.gemm.mtp_eh_proj import mtp_eh_norm, mtp_eh_proj
from .ops.gemm.projection import DECODE_MAX_ROWS, projection, projection_tail
from .ops.gemm.q_b_e4m3 import q_b_e4m3
from .ops.gemm.shared_mlp import shared_down, shared_gate_up_swiglu
from .ops.moe.routed_moe import chunk_sizes as moe_chunk_sizes
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
from .ops.norm.cute_dsl_hidden_rmsnorm import hidden_rmsnorm
from .ops.norm.qkv_a_rmsnorm import qkv_a_rmsnorm
from .ops.quantization.cute_dsl_act_quant_mxf8_rows import cute_dsl_act_quant_mxf8_rows
from .step import PREFILL, build_step_args

if TYPE_CHECKING:
    from .modeling import GlmMoeDsaTep4ModelingV2


_STEP_MAX_TOKENS = 2**31 // (32 * 128)


class MixedTarget:
    """Routed to on a step with both context rows and generation rows."""

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the `step_args` projections differ, so
        # probing one does not prove the others' fields exist.
        self._contract_pending = step_contract_enabled()
        # The dense-causal group table's sync-free upload, kept across context steps.
        self._dc_uploader = _plan.DcGroupsUploader()

    def on_load(self) -> None:
        """This Target's cross-op handoffs, decided once from the core's handles after load
        (an upstream op then writes its downstream op's operand where the table says so, on
        the rows it can; elsewhere the downstream op quantizes itself):

        - `_mlp_x8[i]`: layer `i`'s pre-MLP all-reduce + norm writes the MXF8 operand of its
          MLP's first GEMM: a dense layer's where the dense MXF8 trio built for it, a MoE
          layer's where the shared expert's MXF8 gate|up did (two-shot rows only).
        - `_close_x8[i]`: layer `i`'s closing all-reduce + norm writes layer `i + 1`'s qkva
          MXF8 operand, where that qkva has the MXF8 GEMM (never after the last layer).
        - `_qb_pre[i]`: layer `i`'s q_a norm writes its decode q_b MXF8 GEMM's operand
          (`qkv_a_rmsnorm`, on a step of up to 64 rows), where the norm's q8 kernel and that
          GEMM built.
        - `_quant_rows`: the row counts up to which a trunk MoE layer's pre-MoE collective
          also writes the routed experts' NVFP4 input: the decode widths.

        The generation rows' V-B never writes o_proj's operand here (`_attn_gen`)."""
        core = self.core
        n = core.num_layers
        self._mlp_x8 = [
            (core._dense_w8[i] if core._moe[i] is None else core._shared_w8[i]) is not None
            for i in range(n)
        ]
        qkva8 = core._proj_h.qkva.mxf8 is not None
        self._close_x8 = [qkva8 and core._attn[i + 1].qkva8 is not None for i in range(n - 1)]
        self._close_x8.append(False)
        qb8 = core._attn_h.norm.q8 is not None and core._proj_h.q_b.dec8 is not None
        self._qb_pre = [qb8 and a.qb_dec8 is not None for a in core._attn]
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
        return build_step_args(md)

    def _layer_attention(
        self,
        md,
        step,
        x,
        pos,
        attn_out,
        i: int,
        pool_layer: int,
        slots_of,
        topk_buf,
        tc,
        gen,
        gen_p,
        ctx,
        need_q,
        skip_ctx,
        skip_gen,
        ctx_rope=None,
        dc=None,
        x8=None,
        ctx_last=None,
        split_memo=None,
    ):
        """One layer's attention block, both phases, into `attn_out`. `slots_of(topk_buf, fresh)`
        maps the request-local selection to this layer's pool rows, `(context, generation,
        generation index kind, its layer)` (`fresh`: this layer selected anew); the context half
        is None under the dense-causal context attention (`dc`). `x8` is `x`'s
        MXF8 operand (the closing all-reduce of the layer before wrote it). `split_memo` is the
        step's store for the decode indexer's MTP5 atom split metadata.

        Returns o_proj's MXF8 operand `(q8, sf)` when the fp8 context attention wrote the
        context rows' attention into it (their bf16 rows of `attn_out` are then not written:
        o_proj must take the operand), else None."""
        core = self.core
        proj, ah, ch = core._proj_h, core._attn_h, core._ctx_h
        a = core._attn[i]
        # The context rows' route on this step (`context_plan.layer_route`) and its per-step
        # operands. On the fp8 MHA route (`mha8`) the context rows read their query from q_b's
        # e4m3 output and can write o_proj's MXF8 operand; the q_a norm then writes the MXF8
        # q_b's operand.
        slot = _plan.pool_slot(step, pool_layer)
        route, plan = _plan.layer_route(dc, ctx_last, slot, ctx_rope, a, ch.mha)
        mha8 = route == _plan.MHA and ch.mha.fp8
        q8_rows, store_from = _plan.q8_rows(mha8 and ch.qa8, tc, need_q)
        qkva = projection(PREFILL, x, a.qkva, proj.qkva, w8=a.qkva8, pre=x8)
        q_a, latent, qa8, qb_pre = qkv_a_rmsnorm(
            qkva, a.q_norm, a.kv_norm, ah.norm, self._qb_pre[i] and not mha8, q8_rows, store_from
        )
        # On the fp8 MHA route the queries come from q_a (below): no bf16 q_b here.
        q = (
            None
            if mha8
            else projection(PREFILL, q_a, a.qb, proj.q_b, wd8=a.qb_dec8, dec_pre=qb_pre)
        )
        # The indexer layers' selection, both phases (the others reuse the last one's).
        ix = core._idx[i]
        if ix is not None:
            ixh, ix_state = core._ix_h, core._ix_state
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
            kb, wq = indexer_pre_proj(PREFILL, x, ix.fused, ix.fused_bf16, None, ixh.pre_proj)
            if need_q:
                ixq = projection(PREFILL, q_a, ix.qb, proj.ix_qb)
                q_fp8, weights, kb = indexer_q_side(
                    PREFILL, q_a, ixq, ix.qb, pos, wq, kb, ix.kn_w, ix.kn_b, md, k_pool, ix_state,
                    ixh,
                )  # fmt: skip
            indexer_write_k(md, k_pool, kb, ix.kn_w, ix.kn_b, pos, ix_state, ixh)
            if not skip_ctx:
                indexer_prefill_select(md, k_pool, q_fp8, weights, topk_buf, ixh)
            if not skip_gen:
                logits = indexer_decode_logits(
                    md, k_pool, q_fp8[tc:], weights[tc:], ctx, gen_p, split_memo, ixh
                )
                indexer_decode_topk(md, logits, topk_buf[tc:], ctx, gen_p, ixh)
            # Draft step 0 of a mixed step (the later steps have no context rows and take
            # `DecodeTarget`): the MTP layer stashes its selection for them.
            if (
                i == core.num_layers
                and core._mtp_share_stash
                and md.in_mtp_draft_loop
                and not mtp_share_reuses(md)
            ):
                mtp_share_store(md, topk_buf, ctx, tc, gen // gen_p, gen_p)
        slots_ctx, slots_gen, gen_kind, gen_layer = slots_of(topk_buf, ix is not None)
        sel_gen = (slots_gen, gen_kind, gen_layer)
        if mha8:
            # The context rows' e4m3 query from q_b (reading the norm's MXF8 operand `qa8`),
            # the generation rows' bf16 query from the q_b tail.
            q_ctx = q_b_e4m3(q_a[:tc], a.qb8, qa8, ah.q_b_e4m3)
            q_gen = projection_tail(PREFILL, q_a, tc, a.qb, proj.q_b)
        else:
            q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        lat_ctx, lat_gen = torch.split(latent, [tc, gen], 0)
        # With `ctx_last` the context part of `attn_out` holds the plan's `n_pad` rows.
        n_ctx_out = tc if ctx_last is None else ctx_last.n_pad
        attn_ctx, attn_gen = torch.split(attn_out, [n_ctx_out, gen], 0)
        # o_proj's MXF8 operand for the fp8 MHA's O epilogue (trunk layers only: the MTP
        # layer's close reads the bf16 `attn_out`), or None.
        o8 = (
            o_operand(PREFILL, attn_out, a.o8, proj.o_proj, ch.mha)
            if mha8 and i < core.num_layers
            else None
        )
        # On the fp8 MHA route the generation rows' attention (absorbed K-B, RoPE
        # append, sparse MLA decode, V-B) reads only their query, latent, slots and pool pages
        # and writes only their attention rows and those pages, so it runs on its own stream
        # beside the context rows' rope, K/V expansion and MHA, joined before o_proj reads it
        # (bit for bit the serial outputs). With `gen_side_quant` their quantize into `o8` runs
        # there too (`side_quant`). The absorbed routes share K-B / V-B with these rows and
        # keep the serial order.
        gen_stream = ch.gen_stream if mha8 else None
        side_quant = gen_stream is not None and o8 is not None and ch.gen_side_quant
        if gen_stream is not None:
            core._count("ctx_gen_side", True)
            core._count("ctx_gen_side_quant", side_quant)
            main = step.get("diet_main")
            gen_stream.wait_stream(torch.cuda.current_stream() if main is None else main)
            with torch.cuda.stream(gen_stream):
                self._attn_gen(a, pool_layer, q_gen, lat_gen, attn_gen, sel_gen, gen_p, ctx, md)
                if side_quant:
                    cute_dsl_act_quant_mxf8_rows(attn_gen, *o8, tc, compiled=ch.mha.q8_rows)
        o8_done = False
        if route == _plan.MHA:
            # The non-absorbed MHA: rope back into the query, the K/V source rows, the K / V
            # expansion, the causal MHA (into `o8` where it is given).
            q3 = torch.reshape(q_ctx, [tc, core.heads, core.qk_dim])
            lat = ctx_rope_q_in_place(q3, lat_ctx, step, slot, core.nope, ctx_rope, ch.rope)
            src = mha_kv_gather(lat, plan, slot, ch.mha)
            mha_kv_expand(src, a.kvb8, a.kvb_aug, plan, ch.mha)
            o8_done = mha_prefill(q3, plan, attn_ctx, ctx_rope[0], a.mha_hs, o8, ch.mha)
        else:
            # The absorbed sparse MLA: K-B, the rope / KV append (which mutates the query and
            # the pool, so nothing after it may fall back), the attention by the step's plan,
            # V-B.
            on_kernel = ctx_rope is not None and slot.primary
            quant_q = plan.quant_q if route == _plan.DENSE_CAUSAL else None
            query = ctx_k_b(PREFILL, q_ctx, a.k_b, a.k_b_nk, quant_q, on_kernel, ch.k_b)
            quant_q = ctx_rope_append(query, lat_ctx, step, slot, pool_layer, ctx_rope, ch.rope)
            pool = step["primary_pool"]
            heads, lat_w, dev = core.heads, core.heads * core.kv_lora, attn_out.device
            if route == _plan.SELECTED:
                assert on_kernel, "the selected-rows attention needs the rope kernel"
                n = plan.n_pad
                q_sel = torch.empty([n, heads, core.lat_dim], dtype=quant_q.dtype, device=dev)
                cute_dsl_rows_select(
                    quant_q.view(tc, -1), plan.ctx_rows, q_sel.view(n, -1), ch.rows_select
                )
                lat_out = torch.empty([n, lat_w], dtype=attn_out.dtype, device=dev)
                sparse_mla_prefill_selected(q_sel, pool, lat_out, plan, ch.fmha)
            elif route == _plan.DENSE_CAUSAL:
                lat_out = torch.empty([tc, lat_w], dtype=attn_out.dtype, device=dev)
                layer_offset = slot.layer_in_pool * slot.tpb
                sparse_mla_prefill_dense_causal(quant_q, pool, lat_out, plan, layer_offset, ch.fmha)
            else:
                lat_out = torch.empty([tc, lat_w], dtype=attn_out.dtype, device=dev)
                sparse_mla_prefill_index(quant_q, pool, slots_ctx, lat_out, step, ch.fmha)
            n = int(lat_out.shape[0])
            absorbed_bmm(
                PREFILL,
                torch.transpose(torch.reshape(lat_out, [n, heads, core.kv_lora]), 0, 1),
                a.v_b_t,
                a.v_b_nk,
                torch.transpose(torch.reshape(attn_ctx, [n, heads, core.v_dim]), 0, 1),
                ah.absorb.v_b,
            )
        if gen_stream is None:
            self._attn_gen(a, pool_layer, q_gen, lat_gen, attn_gen, sel_gen, gen_p, ctx, md)
        else:
            (torch.cuda.current_stream() if main is None else main).wait_stream(gen_stream)
        if not o8_done:
            return None
        if not side_quant:
            cute_dsl_act_quant_mxf8_rows(attn_gen, *o8, tc, compiled=ch.mha.q8_rows)
        return o8

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
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        step = self.step_args(md)
        ch = core._ctx_h
        _plan.step_tables(step, ch.diet)
        core._count("target_mixed", True)
        h = (
            embed(input_ids, core.w["embed"], core._embedding)
            if inputs_embeds is None
            else inputs_embeds
        )
        rows = h.shape[0]
        # The slot mappings and window vectors are indexed by the prepared token order; a
        # padded input would scatter into stale slots.
        assert rows == md.num_tokens, (rows, md.num_tokens)
        assert rows < _STEP_MAX_TOKENS, (
            f"{rows} tokens in one step overflow the indexer kernel's int32 offsets; "
            f"set max_num_tokens below {_STEP_MAX_TOKENS}"
        )
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_p = gen // (md.num_seqs - ctx)
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs)
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = indexer_topk_buf(md, rows, tc, skip_ctx, skip_gen, core._ix_h)
        need_q = not (skip_ctx and skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        # Built once per step and shared by every layer: the context rope kernel's operands
        # (None: the stock rope/append op), the dense-causal context attention's plan (None:
        # the index path), and the selection's conversion to pool rows.
        ctx_rope = _plan.rope_step(step, tc, ch.rope)
        dc = _plan.dense_causal_step(
            md, step, tc, skip_ctx, req_idx, h.device, ctx_rope is not None, ch.mha,
            self._dc_uploader, core,
        )  # fmt: skip
        slots = _plan.ctx_slots(
            md, req_idx, rows, tc, gen, gen_p, dc, ch.mixed_base, core._attn_h.fmha
        )
        # The generation rows' MTP5 atom split metadata, built by the first indexer layer.
        split_memo: dict = {}

        x = hidden_rmsnorm(h, core._attn[0].norm1, core.eps, core._rows.norm)
        x8 = None
        residual = h
        for i in range(core.num_layers):
            layer_id = core._pool_layer_id[i]

            def slots_of(buf, fresh, layer_id=layer_id):
                return slots.layer(buf, fresh, layer_id)

            o8 = self._layer_attention(
                md, step, x, pos, attn_out, i, i, slots_of, topk_buf, tc, gen, gen_p, ctx,
                need_q, skip_ctx, skip_gen, ctx_rope, dc, x8, split_memo=split_memo,
            )  # fmt: skip
            x, x8 = self._mlp_and_close(attn_out, residual, i, rows, o8, step.get("diet_main"))
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
        """One draft step on the MTP module. Its KV lives in the separate draft pool: the
        pool-local layer index and the pool stride are read live, not from the trunk's.
        Attention-to-output runs on the draft head's rows where it can
        (`context_plan.mtp_last_plan`); the output's other rows are then not written."""
        core = self.core
        ch = core._ctx_h
        md = attn_metadata
        step = self.step_args(md)
        rows = hidden_states.shape[0]
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_p = gen // (md.num_seqs - ctx)
        j = core.num_layers
        emb = embed(input_ids, core.w["embed"], core._embedding)
        eh = mtp_eh_norm(PREFILL, emb, hidden_states, core._mtp_eh_h)
        x = mtp_eh_proj(PREFILL, eh, core._mtp_eh_h)
        residual = x
        xn = hidden_rmsnorm(x, core._attn[j].norm1, core.eps, core._rows.norm)
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs)
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = indexer_topk_buf(md, rows, tc, skip_ctx, skip_gen, core._ix_h)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]
        # The MTP layer's context rows take the trunk's rope/append kernel, at its own pool's
        # layer (`ctx_rope_append` falls back to the stock op on any other pool layout), and
        # the index path or the selected rows.
        ctx_rope = _plan.rope_step(step, tc, ch.rope)
        last = _plan.mtp_last_plan(
            md, step, ctx_rope, ctx, tc, gen, local, stride, ch.rows_select is not None,
            core.max_draft_len, core.idx_topk, core._count, core.mtp_last_declines,
        )  # fmt: skip

        def slots_of(buf, fresh):
            if last is not None:
                # The context queries read the block table: only the generation rows convert.
                slots = convert_req_index_to_global(
                    req_idx[tc:], md.block_table, buf[tc:], md.tokens_per_block, core.idx_topk,
                    stride, pool_layer,
                )  # fmt: skip
                return None, slots, INDEX_KIND_GLOBAL, None
            slots = convert_req_index_to_global(
                req_idx, md.block_table, buf, md.tokens_per_block, core.idx_topk, stride, pool_layer
            )
            ctx_s, gen_s = torch.split(slots, [tc, gen], 0)
            return ctx_s, gen_s, INDEX_KIND_GLOBAL, None

        on_kernel = core.route_hits.get("ctx_rope", 0)
        out_rows = rows if last is None else last.r_pad
        attn_out = torch.empty([out_rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        self._layer_attention(
            md, step, xn, torch.reshape(position_ids, [-1]), attn_out, j, local, slots_of,
            topk_buf, tc, gen, gen_p, ctx, not (skip_ctx and skip_gen), skip_ctx, skip_gen,
            ctx_rope, ctx_last=last,
        )  # fmt: skip
        core._count("mtp_ctx_rope", core.route_hits.get("ctx_rope", 0) > on_kernel)
        if last is None:
            return self._mtp_close(attn_out, residual, rows)
        # The selected rows' residual, the close on those rows, then the rows back in place.
        res_sel = torch.empty(
            [last.r_pad, residual.shape[1]], dtype=residual.dtype, device=x.device
        )
        cute_dsl_rows_select(residual, last.rows, res_sel, ch.rows_select)
        out = self._mtp_close(attn_out, res_sel, last.r_pad)
        full = torch.empty([rows, out.shape[1]], dtype=out.dtype, device=out.device)
        return cute_dsl_rows_select(out, last.rows, full, ch.rows_select, scatter=True)

    def _attn_gen(self, a, layer: int, q, latent, out, slots, gen_p: int, ctx: int, md) -> None:
        """The generation rows: sparse MLA over the selection `slots`, absorbed
        form, into `out`, on the stock RoPE / append op (the `"stock"` caller of
        `k_b_q_base`)."""
        core = self.core
        ah = core._attn_h
        gen = int(q.shape[0])
        query = k_b_q_base(PREFILL, q, a.k_b, a.k_b_nk, gen_p, "stock", ah.k_b)
        quant_q, cu_q, bmm1, bmm2 = mla_rope(query, latent, md, layer, gen_p, ctx, ah.rope)
        lat_out = torch.empty([gen, core.heads * core.kv_lora], dtype=q.dtype, device=q.device)
        sparse_mla_decode(quant_q, md, slots, lat_out, cu_q, bmm1, bmm2, ctx, gen_p, ah.fmha)
        v_b(PREFILL, lat_out, out, a.v_b_t, a.v_b_nk, False, ah.v_b)

    def _mlp_and_close(self, attn_out, residual, i: int, rows: int, o8, main):
        """o_proj, the layer's MLP, and the closing collective normed for the next layer.
        `o8` is `attn_out`'s MXF8 operand when the context attention wrote it (o_proj then
        reads it instead of `attn_out`). Returns `(x, x8)`: the next layer's input and, on a
        two-shot step, its MXF8 operand for the next layer's qkva (else None). `main`:
        the stream the step runs on, when the caller already holds it (else read here)."""
        core = self.core
        comm = core._comm
        a = core._attn[i]
        partial = projection(
            PREFILL, attn_out, a.o, core._proj_h.o_proj, w8=a.o8, pre=o8, wd8=a.o_dec8
        )
        gu, dn = core._mlp[i]
        m = core._moe[i]
        if m is None:
            w8 = core._dense_w8[i]
            o, x8 = allreduce_add_rmsnorm(partial, residual, a.norm2, comm, self._mlp_x8[i])
            act, act8 = dense_gate_up_swiglu(PREFILL, o, gu, w8, x8, core._dense_h)
            mlp_out = dense_down(PREFILL, act, act8, dn, w8, core._dense_h)
        else:
            # The decode L2 prefetch of this layer's routed FC1 weights (decode widths only).
            pf = moe_l2_prefetch(i, m, rows, False, core._moe_h)
            o, x8, pre = allreduce_add_rmsnorm_quant(
                partial, residual, a.norm2, m.g1, core._ar_quant, self._quant_rows,
                self._mlp_x8[i],
            )  # fmt: skip
            slot, acc = moe_open(rows, False, core._moe_h, core._moe_parity)
            if main is None:
                main = torch.cuda.current_stream()
            side = core._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                act, on_pre = shared_gate_up_swiglu(
                    PREFILL, o, gu, core._shared_gu_il[i], core._shared_w8[i], x8,
                    core._shared_h, True,
                )  # fmt: skip
                shared = shared_down(PREFILL, act, dn, core._shared_h, acc, on_pre)
            routed = self._routed(
                o, m, pre is not None, slot, acc, side, pf, i, core._moe_h, core._moe_parity
            )
            mlp_out = moe_combine(routed, shared, acc, rows, main, side, core._moe_h)
        return allreduce_add_rmsnorm(mlp_out, residual, core._next_norm[i], comm, self._close_x8[i])

    def _routed(self, o, m, pre: bool, slot, acc, side, pf, layer: int, h, parity):
        """A MoE layer's routed experts on `o`, in chunks of the token budget
        (`moe_chunk_sizes`; one chunk on every step at or below it). `slot`: the slot
        `moe_open` opened for the layer, or None (each chunk takes its own from `parity`).
        Each chunk's result is a view of its slot's output buffer, and the slots alternate per
        chunk, so with more than one chunk each result is copied out before a later chunk
        reuses its slot."""
        sizes = moe_chunk_sizes(int(o.shape[0]))
        routed = o.new_empty(o.shape) if len(sizes) > 1 else None
        start = 0
        for n in sizes:
            chunk = o[start : start + n]
            s = slot if slot is not None else parity.take()
            quantized = moe_quantize_first(chunk, m, pre, h)
            logits, in_slot = moe_router(PREFILL, chunk, m, s, False, h)
            routing = moe_topk(logits, in_slot, m, s, layer, False, h)
            part = moe_experts(chunk, routing, m, quantized, s, acc, side, pf, layer, False, h)
            if routed is None:
                return part
            row_copy(routed[start : start + n], part, self.core._rows.copy)
            start += n
        return routed

    def _mtp_close(self, attn_out, residual, rows: int):
        """The MTP layer's o_proj and MoE (its routed ops' handle `_mtp_moe_h`). Returns the
        hidden state after the module's own head norm when the engine drafts more than one
        token (`_mtp_out_normed`: it is fed back into the next draft step, as stock's), else
        un-normalized with the norm applied by `shared_head`."""
        core = self.core
        j = core.num_layers
        a = core._attn[j]
        partial = projection(PREFILL, attn_out, a.o, core._proj_h.o_proj)
        o, _ = allreduce_add_rmsnorm(partial, residual, a.norm2, core._comm)
        gu, dn = core._mlp[j]
        mtp_h = core._mtp_moe_h
        mtp_moe_open(rows, False, mtp_h)
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            act, on_pre = shared_gate_up_swiglu(
                PREFILL, o, gu, core._shared_gu_il[j], None, None, core._shared_h
            )
            shared = shared_down(PREFILL, act, dn, core._shared_h, None, on_pre)
        routed = self._routed(
            o, core._mtp_moe, False, None, None, None, None, j, mtp_h, core._mtp_parity
        )
        main.wait_stream(side)
        summed = hidden_add(routed, shared, core._rows.add)
        reduced = allreduce(summed, core._comm)
        out = hidden_add(residual, reduced, core._rows.add)
        if core._mtp_out_normed:
            return hidden_rmsnorm(out, core._mtp["head_norm"], core.eps, core._rows.norm)
        return out
