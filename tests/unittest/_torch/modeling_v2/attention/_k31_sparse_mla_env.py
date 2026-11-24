# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's sparse MLA call of `thop_attention`, with explicit state.

The K3.1 target's MLA layers run DSA: an indexer picks up to 2048 cached keys per query row, and the
attention itself is the TRTLLM-Gen sparse MLA kernel behind `thop_attention`, in absorption mode in
both phases. What the op receives is what the engine's TRTLLM backend forwards for that layer:

* q: the fused query `[T, H * (512 + 64)]` -- per head the absorbed `q_nope @ W_UK` (512) then the
  rope slot (64). H is the head count the MLA hands the backend: 96 per rank pads to 128.
* one latent KV head of 576 over a paged pool (kv_factor 1), bf16 or fp8-e4m3 at kv scale 1.0;
  V is the first 512 of each latent row (`v_head_dim = kv_lora_rank`, rope_append on).
* sparse_attn_indices `[T, 2048]` int32: per query row the global pool rows it attends
  (`page * tokens_per_block + slot`, ascending, -1 past the row's last entry), as the target's
  `convert_req_index_to_global` produces them; sparse_attn_offsets None, block size 1,
  num_sparse_topk 2048.
* context (attention_input_type 1): `latent_cache` (the new rows) and `q_pe` are consumed -- the op
  ropes them, appends the rows to the pool and writes the rope slot of q, then attends over the
  pool through the indices.
* generation (attention_input_type 2): the new rows are already in the pool (the preceding
  `mla_rope_generation` appended them; here the test writes them), and the call carries that op's
  products: cu_q_seqlens / cu_kv_seqlens / a zeroed scheduler counter, and over an fp8 pool the
  e4m3 query in quant_q_buffer with mla_bmm1_scale / mla_bmm2_scale.

K3.1 has no position encoding on the MLA rope slot (NoPE): the target installs an identity rotary
table (cos 1, sin 0), and so does this environment, so a rope slot carries its values unchanged.

The reference side gathers each row's selected latent rows from a mirror of what the pool holds
(test-computed, never read back) and hands them to the entry's own `reference` one row at a time:
each row attends its own key set, which is exactly a one-query call with those keys.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    MLA_CONTEXT,
    MLA_GENERATION,
    QM_FP8_KV_CACHE,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention as op,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
Q_LORA_RANK = 1536
TOPK = 2048
#: Both phases scale QK^T by 1 / (q_scaling * sqrt(nope + rope)); K3.1's MLA has q_scaling 1.0.
SOFTMAX_SCALE = 1.0 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
#: PositionEmbeddingType.rope_gpt_neox, what K3.1's MLA backend is built with.
POSITION_EMBEDDING_TYPE_ROPE_GPT_NEOX = 2
#: The block-size argument the DSA backend passes (token-granular indices).
DSA_INDICES_BLOCK_SIZE = 1


class K31SparseMlaEnv:
    """One MLA layer's paged latent pool and the batch state of one call.

    Pages are handed out in a shuffled order, so a row's global index differs from its position
    and an addressing slip reads another request's (or no request's) rows.
    """

    def __init__(
        self,
        num_heads: int,
        tokens_per_block: int,
        fp8_pool: bool,
        max_seq_len: int,
        max_batch: int,
        seed: int,
    ) -> None:
        self.num_heads = num_heads
        self.tokens_per_block = tokens_per_block
        self.fp8_pool = fp8_pool
        self.max_batch = max_batch
        self.gen = torch.Generator().manual_seed(seed)
        blocks_per_seq = -(-max_seq_len // tokens_per_block)
        self.max_seq_len = blocks_per_seq * tokens_per_block
        num_blocks = blocks_per_seq * max_batch + 2
        dtype = torch.float8_e4m3fn if fp8_pool else torch.bfloat16
        self.pool = torch.zeros(
            num_blocks, tokens_per_block, LATENT_DIM, dtype=dtype, device="cuda"
        )
        self.pool_pointers = torch.zeros(1, 2, dtype=torch.int64)
        self.pool_pointers[0, 0] = self.pool.data_ptr()
        self.pool_mapping = torch.zeros(1, 2, dtype=torch.int32)
        self.block_offsets = torch.zeros(
            1, max_batch, 2, blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        self.workspace = torch.empty(0, dtype=torch.int8, device="cuda")
        rope = RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=10000.0,
            max_positions=self.max_seq_len,
            original_max_positions=self.max_seq_len,
            duplicate_data=True,
        )
        self.rotary_inv_freq, self.rotary_cos_sin = rope.create_rope_const_params()
        flat = self.rotary_cos_sin.view(-1)
        flat[0::2] = 1.0
        flat[1::2] = 0.0
        # The engine passes unit kv scales for every MLA layer, fp8 pool or not.
        self.kv_scale_orig_quant = torch.ones(1, dtype=torch.float32, device="cuda")
        self.kv_scale_quant_orig = torch.ones(1, dtype=torch.float32, device="cuda")
        self.quant_mode = QM_FP8_KV_CACHE if fp8_pool else 0
        self._free_pages = torch.randperm(num_blocks, generator=self.gen).tolist()
        self.pages: Dict[int, List[int]] = {}
        self.prompt_lens: Dict[int, int] = {}
        self.rows: Dict[int, torch.Tensor] = {}  # bf16 [L, 576] per request, as appended
        self.last_kwargs: Dict[str, object] = {}

    # -- pool -------------------------------------------------------------------------------

    def add_request(self, request_id: int) -> None:
        self.pages[request_id] = []
        self.rows[request_id] = torch.empty(0, LATENT_DIM, dtype=torch.bfloat16, device="cuda")

    def cached_len(self, request_id: int) -> int:
        return self.rows[request_id].shape[0]

    def _ensure_pages(self, request_id: int, total: int) -> None:
        pages = self.pages[request_id]
        while len(pages) * self.tokens_per_block < total:
            pages.append(self._free_pages.pop())

    def _fill_offsets(self, request_ids: Sequence[int]) -> None:
        self.block_offsets.zero_()
        for s, rid in enumerate(request_ids):
            ids = torch.tensor(self.pages[rid], dtype=torch.int32, device="cuda")
            self.block_offsets[0, s, 0, : ids.numel()] = ids
            self.block_offsets[0, s, 1, : ids.numel()] = ids

    def to_pool(self, rows: torch.Tensor) -> torch.Tensor:
        """The bytes a latent row is stored as: e4m3 at kv scale 1.0, or the bf16 row."""
        return rows.float().to(torch.float8_e4m3fn) if self.fp8_pool else rows

    def write_rows(self, request_id: int, rows: torch.Tensor) -> None:
        """Append rows to the pool the way the generation preprocessing does (rope is identity)."""
        start = self.cached_len(request_id)
        self._ensure_pages(request_id, start + rows.shape[0])
        tpb = self.tokens_per_block
        stored = self.to_pool(rows)
        for i in range(rows.shape[0]):
            pos = start + i
            self.pool[self.pages[request_id][pos // tpb], pos % tpb] = stored[i]
        self.rows[request_id] = torch.cat([self.rows[request_id], rows])

    def pool_values(self, request_id: int) -> torch.Tensor:
        """fp32 [L, 576]: the request's rows as the pool holds them."""
        return self.to_pool(self.rows[request_id]).float()

    def check_pool(self, request_id: int) -> None:
        """The pool holds each appended row bitwise at its page and slot."""
        tpb = self.tokens_per_block
        expected = self.to_pool(self.rows[request_id])
        for j, page in enumerate(self.pages[request_id]):
            n = min(tpb, expected.shape[0] - j * tpb)
            if n <= 0:
                break
            got = self.pool[page, :n].view(torch.uint8)
            want = expected[j * tpb : j * tpb + n].view(torch.uint8)
            assert torch.equal(got, want), f"request {request_id} page {j}: pool rows differ"

    # -- selections -------------------------------------------------------------------------

    def select(self, visible: int, max_valid: int = TOPK) -> torch.Tensor:
        """Local positions one query row attends: all `visible` keys up to the top-k, else a
        random top-k subset; ascending, as the target hands them to the kernel."""
        n = min(visible, TOPK, max_valid)
        if n == visible:
            return torch.arange(visible, dtype=torch.int64)
        return torch.randperm(visible, generator=self.gen)[:n].sort().values

    def global_rows(self, request_id: int, local: torch.Tensor) -> torch.Tensor:
        pages = torch.tensor(self.pages[request_id], dtype=torch.int64)
        tpb = self.tokens_per_block
        return pages[local // tpb] * tpb + local % tpb

    def indices(self, selections: List[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        """`[rows, 2048]` int32 global rows, -1 padded, from (request, local positions) per row."""
        out = torch.full((len(selections), TOPK), -1, dtype=torch.int32)
        for r, (rid, local) in enumerate(selections):
            out[r, : local.numel()] = self.global_rows(rid, local).to(torch.int32)
        return out.cuda()

    # -- the call ---------------------------------------------------------------------------

    def _call(
        self,
        *,
        q: torch.Tensor,
        output: torch.Tensor,
        kv_lens: List[int],
        ctx_lens: List[int],
        num_contexts: int,
        num_ctx_tokens: int,
        attention_input_type: int,
        latent_cache: torch.Tensor,
        q_pe: torch.Tensor,
        sparse_attn_indices: torch.Tensor,
        predicted_tokens_per_seq: int,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        ns = len(kv_lens)
        kwargs = dict(
            q=q,
            k=None,
            v=None,
            output=output,
            output_sf=None,
            workspace_=self.workspace,
            sequence_length=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
            host_past_key_value_lengths=torch.tensor(kv_lens, dtype=torch.int32),
            host_total_kv_lens=torch.tensor(
                [sum(kv_lens[:num_contexts]), sum(kv_lens[num_contexts:])], dtype=torch.int32
            ),
            context_lengths=torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"),
            host_context_lengths=torch.tensor(ctx_lens, dtype=torch.int32),
            host_request_types=torch.tensor(
                [0 if i < num_contexts else 1 for i in range(ns)], dtype=torch.int32
            ),
            max_context_q_len_override=None,
            kv_cache_block_offsets=self.block_offsets,
            host_kv_cache_pool_pointers=self.pool_pointers,
            host_kv_cache_pool_mapping=self.pool_mapping,
            cache_indirection=None,
            kv_scale_orig_quant=self.kv_scale_orig_quant,
            kv_scale_quant_orig=self.kv_scale_quant_orig,
            out_scale=None,
            rotary_inv_freq=self.rotary_inv_freq,
            rotary_cos_sin=self.rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=False,
            update_kv_cache=True,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            local_layer_idx=0,
            num_heads=self.num_heads,
            num_kv_heads=1,
            head_size=LATENT_DIM,
            tokens_per_block=self.tokens_per_block,
            max_num_requests=self.max_batch,
            max_context_length=self.max_seq_len,
            max_seq_len=self.max_seq_len,
            attention_window_size=self.max_seq_len,
            beam_width=1,
            mask_type=1,
            quant_mode=self.quant_mode,
            q_scaling=1.0,
            position_embedding_type=POSITION_EMBEDDING_TYPE_ROPE_GPT_NEOX,
            rope_dim=QK_ROPE_HEAD_DIM,
            rope_base=10000.0,
            rope_scale_type=0,
            rope_scale=1.0,
            rope_short_m_scale=1.0,
            rope_long_m_scale=1.0,
            rope_max_positions=self.max_seq_len,
            rope_original_max_positions=self.max_seq_len,
            use_paged_context_fmha=True,
            attention_input_type=attention_input_type,
            is_mla_enable=True,
            chunked_prefill_buffer_batch_size=1,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=KV_LORA_RANK,
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
            spec_decoding_position_offsets_for_cpp=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=DSA_INDICES_BLOCK_SIZE,
            num_sparse_topk=TOPK,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
        )
        self.last_kwargs = kwargs
        with validating(op):
            op(**kwargs)
        torch.cuda.synchronize()

    def context(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        q_pe: torch.Tensor,
        gen_request_ids: Sequence[int] = (),
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """A context_only call over prompt chunks; returns (output, each row's selection).

        A request's chunk follows whatever rows it already holds (a fresh prompt holds none; a
        later chunk of a chunked prefill holds the earlier chunks', written beforehand). Row t of
        a chunk after L held rows sees positions [0, L + t] and attends its selection of them.
        As the engine does, the batch state counts a chunk's rows as the context length and
        L + chunk as the KV length. The op appends the chunk's latent rows itself, so the mirror
        records them after the call. `gen_request_ids` are generation requests of the same step:
        the batch state names them after the prompts (their rows of this step already counted).
        """
        selections: List[Tuple[int, torch.Tensor]] = []
        held = [self.cached_len(rid) for rid in request_ids]
        for rid, ln, past in zip(request_ids, seq_lens, held):
            self._ensure_pages(rid, past + ln)
            self.prompt_lens[rid] = past + ln
            selections += [(rid, self.select(past + t + 1)) for t in range(ln)]
        gen_request_ids = list(gen_request_ids)
        self._fill_offsets(list(request_ids) + gen_request_ids)
        total = sum(seq_lens)
        output = torch.empty(total, self.num_heads * KV_LORA_RANK, dtype=q.dtype, device="cuda")
        self._call(
            q=q,
            output=output,
            kv_lens=[past + ln for past, ln in zip(held, seq_lens)]
            + [self.cached_len(r) for r in gen_request_ids],
            ctx_lens=list(seq_lens) + [self.prompt_lens[r] for r in gen_request_ids],
            num_contexts=len(request_ids),
            num_ctx_tokens=total,
            attention_input_type=MLA_CONTEXT,
            latent_cache=latent_cache,
            q_pe=q_pe,
            sparse_attn_indices=self.indices(selections),
            predicted_tokens_per_seq=1,
        )
        start = 0
        for rid, ln in zip(request_ids, seq_lens):
            self.rows[rid] = torch.cat([self.rows[rid], latent_cache[start : start + ln]])
            start += ln
        return output, selections

    def generation(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        new_rows: torch.Tensor,
        q_pe: torch.Tensor,
        predicted_tokens_per_seq: int,
        max_valid: int = TOPK,
        ctx_requests: Sequence[Tuple[int, int]] = (),
        rows_written: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """A generation_only call: P query rows per request, token-major; row t of a request
        whose history is L sees positions [0, L + t]. `new_rows` are written to the pool first
        (unless `rows_written`), standing in for the rope-generation op the target runs before
        this call. `ctx_requests` are the (request, prompt length) context requests of the same
        step, which lead the batch state; the scheduler buffers cover generation requests only."""
        p = predicted_tokens_per_seq
        if not rows_written:
            for g, rid in enumerate(request_ids):
                self.write_rows(rid, new_rows[g * p : (g + 1) * p])
        ctx_ids = [rid for rid, _ in ctx_requests]
        ctx_lens = [ln for _, ln in ctx_requests]
        self._fill_offsets(ctx_ids + list(request_ids))
        kv_lens = [self.cached_len(r) for r in request_ids]
        selections = [
            (rid, self.select(kv_lens[g] - p + t + 1, max_valid))
            for g, rid in enumerate(request_ids)
            for t in range(p)
        ]
        num_gen = len(request_ids)
        cu_q = (torch.arange(num_gen + 1, dtype=torch.int32) * (self.num_heads * p)).cuda()
        cu_kv = torch.zeros(num_gen + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(kv_lens, dtype=torch.int32).cumsum(0)
        quant_q = bmm1 = bmm2 = None
        if self.fp8_pool:
            quant_q = (
                fused_q.view(-1, self.num_heads, LATENT_DIM)
                .float()
                .to(torch.float8_e4m3fn)
                .contiguous()
            )
            bmm1 = torch.tensor(
                [SOFTMAX_SCALE, SOFTMAX_SCALE * math.log2(math.e)],
                dtype=torch.float32,
                device="cuda",
            )
            bmm2 = torch.ones(1, dtype=torch.float32, device="cuda")
        rows = num_gen * p
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        self._call(
            q=fused_q,
            output=output,
            kv_lens=ctx_lens + kv_lens,
            ctx_lens=ctx_lens + [self.prompt_lens[r] for r in request_ids],
            num_contexts=len(ctx_ids),
            num_ctx_tokens=sum(ctx_lens),
            attention_input_type=MLA_GENERATION,
            latent_cache=new_rows,
            q_pe=q_pe,
            sparse_attn_indices=self.indices(selections),
            predicted_tokens_per_seq=p,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=torch.zeros(1, dtype=torch.uint32, device="cuda"),
            mla_bmm1_scale=bmm1,
            mla_bmm2_scale=bmm2,
            quant_q_buffer=quant_q,
        )
        return output, selections

    # -- reference --------------------------------------------------------------------------

    def reference_rows(
        self,
        q_rows: torch.Tensor,
        selections: List[Tuple[int, torch.Tensor]],
        num_heads: Optional[int] = None,
        rows: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """The entry's `reference`, one query row at a time over that row's selected latent rows
        (K = the 576-wide rows as the pool holds them, V = their first 512). `q_rows` is the fused
        query with its rope slot already holding what the op puts there. `num_heads` < the env's
        evaluates only the first heads of each row; `rows` restricts to a subset of rows."""
        h = self.num_heads if num_heads is None else num_heads
        rows = range(len(selections)) if rows is None else rows
        values = {rid: self.pool_values(rid) for rid in {rid for rid, _ in selections}}
        out = []
        for r in rows:
            rid, local = selections[r]
            k = values[rid][local.cuda()]
            q_row = q_rows[r].view(self.num_heads, LATENT_DIM)[:h].reshape(1, h * LATENT_DIM)
            out.append(
                op.reference(
                    q=q_row,
                    k=k,
                    v=k[:, :KV_LORA_RANK].contiguous(),
                    q_lens=[1],
                    kv_lens=[k.shape[0]],
                    num_heads=h,
                    num_kv_heads=1,
                    qk_head_dim=LATENT_DIM,
                    v_head_dim=KV_LORA_RANK,
                    softmax_scale=SOFTMAX_SCALE,
                    causal=False,
                    e4m3_inputs=self.fp8_pool,
                )
            )
        return torch.cat(out)

    def selected_value_mean(
        self, selections: List[Tuple[int, torch.Tensor]], rows: Sequence[int]
    ) -> torch.Tensor:
        """fp32 [len(rows), 512]: the plain mean of each row's selected V rows -- what a head whose
        query is all zeros computes (every score 0, uniform weights)."""
        values = {rid: self.pool_values(rid) for rid in {rid for rid, _ in selections}}
        return torch.stack(
            [values[selections[r][0]][selections[r][1].cuda(), :KV_LORA_RANK].mean(0) for r in rows]
        )
