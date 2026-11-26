# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WideDecodeTarget: a pure decode step above 64 rows.

With three or five draft tokens a decode step carries 4 / 6 query rows per sequence, so 32
sequences make 128 / 192 rows; MTP1 passes 64 rows above 32 sequences and no speculation
above 64: past the 64 the decode cells serve, but a decode step all the same. Its rows take
the MTP decode cells (`Phase(wide=True)`): the projections, absorbed BMMs and the shared
expert at 128 / 192 or on the next cell up at the exact rows; o_proj on the wide decode MXF8
GEMM, whose operand the V-B writes; the qkva projection with the indexer's
key / head-weight projection folded in, q_b with the indexer's wq_b, the indexer's q side
with its K-cache update; and the routed experts on their exact rows on the prefill top-k and
pair (`Moe.decode_exact`), behind a fence that keeps the shared expert off the router's early
launch. Each of those is an op that takes the rows where its cell built; the forward is the
same layer loop as `DecodeTarget`'s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.modeling_v2._router_index import step_contract_enabled

from .forward_ops import Phase, _build_step_args

if TYPE_CHECKING:
    from .modeling import GlmMoeDsaTep4ModelingV2

#: Every row this Target sees is a wide decode step's, above 64; an eager draft step's rows
#: are too (`up` only serves rows up to 64).
_WIDE = Phase(wide=True)
_WIDE_DRAFT = Phase(wide=True, up="draft")


class WideDecodeTarget:
    """Routed to on a step without context rows that carries more than 64 rows, on an engine
    that built the wide decode cells (`wide_cells`)."""

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast (TRTLLM_MODELING_V2_VALIDATE): the metadata fields
        this target consumes must exist. Calling `step_args` is the check."""
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
        step = _build_step_args(md)
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
        op = core.op
        ph = _WIDE
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        h = op.rows.embed(input_ids) if inputs_embeds is None else inputs_embeds
        rows = h.shape[0]
        assert rows == md.num_tokens, (rows, md.num_tokens)
        gen_p = rows // md.num_seqs
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = op.indexer.topk_buf(md, rows, 0, True, skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        slots = op.attn.decode_slots(md, req_idx, gen_p)
        # The MTP5 atom split's indexer metadata, built by the first indexer layer for all.
        split_memo: dict = {}
        x = op.rows.rmsnorm(h, core._attn[0].norm1)
        residual = h
        for i in range(core.num_layers):
            a = core._attn[i]
            ix = core._idx[i]
            # Attention.
            q, q_a, latent, kw, ixq = op.attn.qkv_gen(ph, x, a, i, not skip_gen)
            if ix is not None:
                k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
                q_fp8, weights, k_pre = op.indexer.project(
                    ph, x, q_a, pos, ix, rows, not skip_gen, kw=kw, ixq=ixq, kc=(md, k_pool)
                )
                if k_pre is not None:  # else written with the q side (`Indexer.qk_handle`)
                    op.indexer.write_k(md, k_pool, k_pre, rows)
                if not skip_gen:
                    op.indexer.decode(
                        md, k_pool, q_fp8, weights, topk_buf, 0, rows // gen_p, gen_p, split_memo
                    )
            sel = slots.layer(topk_buf, ix is not None, core._pool_layer_id[i])
            quant_q, cu_q, bmm1, bmm2 = op.attn.gen_query(
                ph, a, i, q, latent, rows, gen_p, 0, md, "trunk"
            )
            lat_out = torch.empty([rows, core.heads * core.kv_lora], dtype=h.dtype, device=h.device)
            op.attn.sparse_decode(quant_q, md, sel, lat_out, cu_q, bmm1, bmm2, 0, gen_p)
            o_pre = op.attn.v_b(ph, a, lat_out, attn_out, True)
            # o_proj, the MLP and the closing collective.
            partial = op.gemm.project(
                ph, "o_proj", attn_out, a.o, w8=a.o8, wd8=a.o_dec8, dec_pre=o_pre
            )
            gu, dn = core._mlp[i]
            m = core._moe[i]
            if m is None:
                w8 = core._dense_w8[i]
                o, x8 = op.comm.reduce_norm_x8(partial, residual, a.norm2, w8 is not None)
                mlp_out = op.mlp.dense(ph, o, gu, dn, w8, x8)
            else:
                exact = op.moe.wide_exact(rows)
                # No L2 prefetch on the exact rows (`Moe.pf_fork`).
                op.moe.pf_fork(i, m, rows, exact)
                w8 = core._shared_w8[i]
                o, x8, prequant = op.comm.reduce_norm_moe(
                    partial, residual, a.norm2, m.g1, rows, w8 is not None, exact
                )
                began, acc = op.moe.open_layer(rows, exact)
                main = torch.cuda.current_stream()
                side = core._side_stream
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    op.mlp.shared_after_collective(exact)
                    shared = op.mlp.shared(
                        ph, o, gu, core._shared_gu_il[i], dn, out=acc, trunk=True, x8=x8, w8=w8
                    )
                routed = op.moe.routed(
                    ph, o, m, rows, prequant, side if acc is not None else None, began,
                    acc is not None, i, exact,
                )  # fmt: skip
                mlp_out = op.moe.combine(routed, shared, acc, rows, main, side)
            x, _ = op.comm.close(mlp_out, residual, i)
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
        """A draft step of more than 64 rows: draft step 0 of an MTP1 / MTP3 / MTP5 decode step
        (the later draft steps carry one row per sequence). Pool rows go through the stock
        conversion against the draft pool's live mapping."""
        core = self.core
        op = core.op
        eager = not torch.cuda.is_current_stream_capturing()
        ph = _WIDE_DRAFT if eager else _WIDE
        md = attn_metadata
        rows = hidden_states.shape[0]
        gen_p = rows // md.num_seqs
        j = core.num_layers
        a = core._attn[j]
        ix = core._idx[j]
        x = op.mtp.front(ph, op.rows.embed(input_ids), hidden_states, rows)
        residual = x
        xn = op.rows.rmsnorm(x, a.norm1)
        pos = torch.reshape(position_ids, [-1])
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = op.indexer.topk_buf(md, rows, 0, True, skip_gen)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        # With the engine's index share on, a draft step after the first takes draft step 0's
        # selection (`share_load`); draft step 0 stashes.
        share = ix is not None and core.mtp_index_share
        share_reuse = share and op.indexer.share_reuses(md)
        select = not skip_gen and not share_reuse
        q, q_a, latent, kw, ixq = op.attn.qkv_gen(ph, xn, a, j, select)
        if ix is not None:
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(j)
            q_fp8, weights, k_pre = op.indexer.project(
                ph, xn, q_a, pos, ix, rows, select, kw=kw, ixq=ixq, kc=(md, k_pool)
            )
            if k_pre is not None:
                op.indexer.write_k(md, k_pool, k_pre, rows)
            if select:
                op.indexer.decode(md, k_pool, q_fp8, weights, topk_buf, 0, rows // gen_p, gen_p)
            elif share_reuse and not skip_gen:
                core._count("mtp_ix_share", True)
                op.indexer.share_load(md, topk_buf, rows)
            if share and not share_reuse and core._mtp_share_stash and md.in_mtp_draft_loop:
                op.indexer.share_store(md, topk_buf, 0, 0, rows // gen_p, gen_p)
        sel = op.attn.draft_slots(md, req_idx, topk_buf, stride, pool_layer)
        quant_q, cu_q, bmm1, bmm2 = op.attn.gen_query(
            ph, a, local, q, latent, rows, gen_p, 0, md, "mtp"
        )
        lat_out = torch.empty([rows, core.heads * core.kv_lora], dtype=x.dtype, device=x.device)
        op.attn.sparse_decode(quant_q, md, sel, lat_out, cu_q, bmm1, bmm2, 0, gen_p)
        op.attn.v_b(ph, a, lat_out, attn_out, False)
        # The MTP layer's o_proj and MoE.
        o = op.comm.reduce_norm(op.gemm.project(ph, "o_proj", attn_out, a.o), residual, a.norm2)
        gu, dn = core._mlp[j]
        exact = op.mtp.wide_exact(rows)
        main = torch.cuda.current_stream()
        side = core._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            op.mlp.shared_after_collective(exact)
            shared = op.mlp.shared(ph, o, gu, core._shared_gu_il[j], dn)
        routed = op.mtp.moe(ph, o, rows, exact)
        main.wait_stream(side)
        reduced = op.comm.reduce(op.rows.add(routed, shared))
        return op.mtp.out(residual, reduced)
