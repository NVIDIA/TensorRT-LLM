# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrefillTarget: a step with context rows, and possibly generation rows beside them.

The general case. The context rows take the sm103 prefill kernels (the context rope/append,
the dense-causal or index-path sparse MLA, or the non-absorbed MHA on a dense-causal step);
a mixed step's generation rows take the sparse MLA decode on the stock rope op, at the
offsets (`ctx`, `tc`) that only this Target's call sites carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2._router_index import step_contract_enabled

from .forward_ops import PREFILL, _build_step_args
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.cute_dsl_sparse_mla_decode import INDEX_KIND_GLOBAL

if TYPE_CHECKING:
    from .modeling import GlmMoeDsaTep4ModelingV2


_STEP_MAX_TOKENS = 2**31 // (32 * 128)


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

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
        return _build_step_args(md)

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
        rows,
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
        op = core.op
        a = core._attn[i]
        # The fp8 non-absorbed MHA's context rows read their query from q_b's e4m3 output and
        # can write o_proj's MXF8 operand; the q_a norm then writes the MXF8 q_b's operand.
        mha8 = op.ctx.mha8(a, pool_layer, step, ctx_rope, dc, tc)
        q8_rows, store_from = op.ctx.q8_rows(mha8, tc, need_q)
        q, q_a, latent, qa8 = op.attn.qkv(
            PREFILL, x, a, q_b=not mha8, x8=x8, q8_rows=q8_rows, q_store_from=store_from
        )
        ix = core._idx[i]
        if ix is not None:
            q_fp8, weights, k_pre = op.indexer.project(PREFILL, x, q_a, pos, ix, rows, need_q)
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
            op.indexer.write_k(md, k_pool, k_pre, rows)
            if tc and not skip_ctx:
                op.indexer.prefill_select(md, k_pool, q_fp8, weights, topk_buf)
            if gen and not skip_gen:
                op.indexer.decode(
                    md, k_pool, q_fp8[tc:], weights[tc:], topk_buf[tc:], ctx, gen // gen_p, gen_p,
                    split_memo,
                )  # fmt: skip
            # Draft step 0 of a context or mixed step (the later steps have no context rows
            # and take `DecodeTarget`): the MTP layer stashes its selection for them.
            if (
                i == core.num_layers
                and core._mtp_share_stash
                and md.in_mtp_draft_loop
                and not op.indexer.share_reuses(md)
            ):
                op.indexer.share_store(md, topk_buf, ctx, tc, gen // gen_p if gen else 0, gen_p)
        slots_ctx, slots_gen, gen_kind, gen_layer = slots_of(topk_buf, ix is not None)
        sel_gen = (slots_gen, gen_kind, gen_layer)
        q_ctx, q_gen = op.ctx.queries(mha8, q, q_a, qa8, a, tc, gen)
        lat_ctx, lat_gen = torch.split(latent, [tc, gen], 0)
        # With `ctx_last` the context part of `attn_out` holds the plan's `n_pad` rows.
        n_ctx_out = tc if ctx_last is None else ctx_last.n_pad
        attn_ctx, attn_gen = torch.split(attn_out, [n_ctx_out, gen], 0)
        o8 = op.ctx.o_operand(mha8, i < core.num_layers, attn_out, a)
        # The generation rows' attention beside the context rows' on their own stream, or after.
        gen_stream, side_quant = op.ctx.gen_side(gen, tc, mha8, ctx_last, o8)
        if gen_stream is not None:
            main = step.get("diet_main")
            gen_stream.wait_stream(torch.cuda.current_stream() if main is None else main)
            with torch.cuda.stream(gen_stream):
                self._attn_gen(a, pool_layer, q_gen, lat_gen, attn_gen, sel_gen, gen_p, ctx, md)
                if side_quant:
                    op.ctx.quant_gen_rows(attn_gen, o8, tc)
        o8_done = False
        if tc:
            o8_done = op.ctx.attn(
                a, pool_layer, q_ctx, lat_ctx, attn_ctx, slots_ctx, tc, step, ctx_rope, dc, o8,
                ctx_last,
            )  # fmt: skip
        if gen_stream is None and gen:
            self._attn_gen(a, pool_layer, q_gen, lat_gen, attn_gen, sel_gen, gen_p, ctx, md)
        if gen_stream is not None:
            (torch.cuda.current_stream() if main is None else main).wait_stream(gen_stream)
        if not o8_done:
            return None
        if not side_quant:
            op.ctx.quant_gen_rows(attn_gen, o8, tc)
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
        op = core.op
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        step = self.step_args(md)
        op.ctx.step_tables(step)
        h = op.rows.embed(input_ids) if inputs_embeds is None else inputs_embeds
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
        gen_seqs = md.num_seqs - ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs) if tc else True
        skip_gen = bool(md.skip_indexer_for_gen_reqs) if gen else True
        topk_buf = op.indexer.topk_buf(md, rows, tc, skip_ctx, skip_gen)
        need_q = not (skip_ctx and skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        # Built once per step and shared by every layer: the context rope kernel's operands
        # (None: the stock rope/append op), the dense-causal context attention's plan (None:
        # the index path), and the selection's conversion to pool rows.
        ctx_rope = op.ctx.rope_step(step, tc)
        dc = op.ctx.dense_causal_step(
            md, step, tc, skip_ctx, req_idx, h.device, rope_ok=ctx_rope is not None
        )
        slots = op.ctx.slots(md, req_idx, rows, tc, gen, gen_p, dc)
        # The generation rows' MTP5 atom split metadata, built by the first indexer layer.
        split_memo: dict = {}

        x = op.rows.rmsnorm(h, core._attn[0].norm1)
        x8 = None
        residual = h
        for i in range(core.num_layers):
            layer_id = core._pool_layer_id[i]

            def slots_of(buf, fresh, layer_id=layer_id):
                return slots.layer(buf, fresh, layer_id)

            o8 = self._layer_attention(
                md, step, x, pos, attn_out, i, i, slots_of, topk_buf, rows, tc, gen, gen_p, ctx,
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
        pool-local layer index and the pool stride are read live, not from the trunk's. On a
        context step attention-to-output runs on the draft head's rows where it can
        (`Ctx.mtp_last_plan`); the output's other rows are then not written."""
        core = self.core
        op = core.op
        md = attn_metadata
        step = self.step_args(md)
        rows = hidden_states.shape[0]
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        j = core.num_layers
        emb = op.rows.embed(input_ids)
        x = op.mtp.front(PREFILL, emb, hidden_states, rows)
        residual = x
        xn = op.rows.rmsnorm(x, core._attn[j].norm1)
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs) if tc else True
        skip_gen = bool(md.skip_indexer_for_gen_reqs) if gen else True
        topk_buf = op.indexer.topk_buf(md, rows, tc, skip_ctx, skip_gen)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]
        # The MTP layer's context rows take the trunk's rope/append kernel, at its own pool's
        # layer (`Ctx.attn` falls back to the stock op on any other pool layout), and the
        # index path.
        ctx_rope = op.ctx.rope_step(step, tc)
        last = op.ctx.mtp_last_plan(md, step, ctx_rope, ctx, tc, gen, local, stride)

        def slots_of(buf, fresh):
            if last is not None:
                # The context queries read the block table: only the generation rows convert.
                if not gen:
                    return None, None, INDEX_KIND_GLOBAL, None
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
            topk_buf, rows, tc, gen, gen_p, ctx, not (skip_ctx and skip_gen), skip_ctx, skip_gen,
            ctx_rope, ctx_last=last,
        )  # fmt: skip
        if tc:
            core._count("mtp_ctx_rope", core.route_hits.get("ctx_rope", 0) > on_kernel)
        if last is None:
            return self._mtp_close(attn_out, residual, rows)
        # The selected rows' residual, the close on those rows, then the rows back in place.
        res_sel = torch.empty(
            [last.r_pad, residual.shape[1]], dtype=residual.dtype, device=x.device
        )
        op.ctx.rows_select(residual, last.rows, res_sel)
        out = self._mtp_close(attn_out, res_sel, last.r_pad)
        full = torch.empty([rows, out.shape[1]], dtype=out.dtype, device=out.device)
        return op.ctx.rows_select(out, last.rows, full, scatter=True)

    def _attn_gen(self, a, layer: int, q, latent, out, slots, gen_p: int, ctx: int, md) -> None:
        """A mixed step's generation rows: sparse MLA over the selection `slots`, absorbed
        form, into `out`, on the stock RoPE / append op (`Attn.gen_query`'s `"stock"`)."""
        core = self.core
        op = core.op
        gen = int(q.shape[0])
        quant_q, cu_q, bmm1, bmm2 = op.attn.gen_query(
            PREFILL, a, layer, q, latent, gen, gen_p, ctx, md, "stock"
        )
        lat_out = torch.empty([gen, core.heads * core.kv_lora], dtype=q.dtype, device=q.device)
        op.attn.sparse_decode(quant_q, md, slots, lat_out, cu_q, bmm1, bmm2, ctx, gen_p)
        op.attn.v_b(PREFILL, a, lat_out, out, False)

    def _mlp_and_close(self, attn_out, residual, i: int, rows: int, o8, main):
        """o_proj, the layer's MLP, and the closing collective normed for the next layer.
        `o8` is `attn_out`'s MXF8 operand when the context attention wrote it (o_proj then
        reads it instead of `attn_out`). Returns `(x, x8)`: the next layer's input and, on a
        two-shot context step, its MXF8 operand for the next layer's qkva (else None). `main`:
        the stream the step runs on, when the caller already holds it (else read here)."""
        core = self.core
        op = core.op
        a = core._attn[i]
        partial = op.gemm.project(PREFILL, "o_proj", attn_out, a.o, w8=a.o8, pre=o8, wd8=a.o_dec8)
        gu, dn = core._mlp[i]
        m = core._moe[i]
        if m is None:
            w8 = core._dense_w8[i]
            o, x8 = op.comm.reduce_norm_x8(partial, residual, a.norm2, w8 is not None)
            mlp_out = op.mlp.dense(PREFILL, o, gu, dn, w8, x8)
        else:
            # The decode L2 prefetch of this layer's routed FC1 weights (decode widths only).
            op.moe.pf_fork(i, m, rows)
            w8 = core._shared_w8[i]
            o, x8, prequant = op.comm.reduce_norm_moe(
                partial, residual, a.norm2, m.g1, rows, w8 is not None, False
            )
            began, acc = op.moe.open_layer(rows)
            if main is None:
                main = torch.cuda.current_stream()
            side = core._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                shared = op.mlp.shared(
                    PREFILL, o, gu, core._shared_gu_il[i], dn, out=acc, trunk=True, x8=x8, w8=w8
                )
            routed = op.moe.routed(
                PREFILL, o, m, rows, prequant, side if acc is not None else None, began,
                acc is not None, i,
            )  # fmt: skip
            mlp_out = op.moe.combine(routed, shared, acc, rows, main, side)
        return op.comm.close(mlp_out, residual, i)

    def _mtp_close(self, attn_out, residual, rows: int):
        """The MTP layer's o_proj and MoE (`Mtp.moe`). Returns the hidden state after the
        module's own head norm when the engine drafts more than one token (`_mtp_out_normed`:
        it is fed back into the next draft step, as stock's), else un-normalized with the norm
        applied by `shared_head`."""
        core = self.core
        op = core.op
        j = core.num_layers
        a = core._attn[j]
        o = op.comm.reduce_norm(
            op.gemm.project(PREFILL, "o_proj", attn_out, a.o), residual, a.norm2
        )
        gu, dn = core._mlp[j]
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared = op.mlp.shared(PREFILL, o, gu, core._shared_gu_il[j], dn)
        routed = op.mtp.moe(PREFILL, o, rows)
        main.wait_stream(side)
        reduced = op.comm.reduce(op.rows.add(routed, shared))
        return op.ctx.mtp_out(residual, reduced)
