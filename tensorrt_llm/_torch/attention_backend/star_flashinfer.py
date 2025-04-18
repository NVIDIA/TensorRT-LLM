from dataclasses import dataclass, field
from typing import Dict, Optional

import flashinfer
import numpy as np
import torch

from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..distributed import allgather
from .flashinfer import FlashInferAttentionMetadata, PlanParams
from .interface import AttentionBackend, AttentionMask, PredefinedAttentionMask


# Please sync with flashinfer's DISPATCH_GQA_GROUP_SIZE in include/flashinfer/utils.cuh
def _grouped_size_compiled_for_decode_kernels(num_qo_heads: int,
                                              num_kv_heads: int) -> bool:
    return (num_qo_heads // num_kv_heads) in [1, 2, 4, 8]


@dataclass(kw_only=True)
class StarAttentionWrappers:
    decode_wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper
    prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper]

    is_planned: bool


@dataclass(kw_only=True)
class StarAttentionMetadata(FlashInferAttentionMetadata):
    num_queries: int = 0

    _plan_params_to_wrappers: Dict[PlanParams,
                                   StarAttentionWrappers] = field(init=False)

    def needs_plan(self, plan_params: PlanParams) -> bool:
        if plan_params not in self._plan_params_to_wrappers:
            return True

        wrappers = self._plan_params_to_wrappers[plan_params]
        return not wrappers.is_planned

    @property
    def paged_kv_indices(self) -> torch.Tensor:
        return self._paged_kv_indices[:self.num_context_blocks +
                                      self.num_query_blocks +
                                      self.num_generation_blocks]

    @property
    def paged_kv_last_page_len(self) -> torch.Tensor:
        return self._paged_kv_last_page_len[:self.num_contexts +
                                            self.num_queries +
                                            self.num_generations]

    @property
    def qo_indptr(self) -> torch.Tensor:
        return self._qo_indptr[:self.num_contexts + self.num_queries +
                               self.num_generations + 1]

    @property
    def cached_token_lens(self) -> torch.Tensor:
        return self._cached_token_lens[:self.num_contexts + self.num_queries +
                                       self.num_generations]

    @property
    def query_lens(self) -> torch.Tensor:
        """
        The length of each query-phase sequence in the batch.
        The shape is (num_queries), where num_queries is the number of query-phase sequences in the batch.
        """
        return self.seq_lens[self.num_contexts:(self.num_contexts +
                                                self.num_queries)]

    @property
    def num_qry_tokens(self) -> int:
        """
        Number of tokens in sequences in the context phase
        """
        return int(self.query_lens.sum())

    @property
    def num_generations(self) -> int:
        """
        The number of generation-phase sequences in the batch.
        """
        return self.seq_lens.shape[0] - self.num_contexts - self.num_queries

    def prepare(self) -> None:
        context_lens = self.context_lens
        query_lens = self.query_lens
        # indices of used cache blocks for each sequence
        block_ids_per_seq = self.kv_cache_params.block_ids_per_seq
        paged_kv_indices = torch.tensor(
            [x for block_ids in block_ids_per_seq for x in block_ids],
            dtype=torch.int32)  # [batch, num_pages]
        self._paged_kv_indices[:paged_kv_indices.size(0)].copy_(
            paged_kv_indices, non_blocking=True)

        # lengths of each query for each sequence in the batch (1 for generation-phase sequences)
        qo_lens = torch.concat([
            context_lens, query_lens,
            torch.ones(self.num_generations, dtype=torch.int32)
        ])

        # start and end indices of each sequence in the ragged query
        qo_indptr = torch.concat([
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(qo_lens, dtype=torch.int32, dim=0),
        ])
        self._qo_indptr[:qo_indptr.size(0)].copy_(qo_indptr, non_blocking=True)

        # number of tokens in the kv cache for each sequence in the batch
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq, dtype=torch.int)

        self._cached_token_lens[:cached_token_lens.size(0)].copy_(
            cached_token_lens, non_blocking=True)

        # number of tokens needed in the kv cache for each sequence after the next pass
        if self.mapping.cp_rank == self.mapping.cp_size - 1:
            # we need to append new tokens at last rank bsx1, bsx1
            kv_lens = self.cached_token_lens + self.seq_lens_cuda
        else:
            mask = torch.tensor([1] * self.num_contexts + [0] *
                                (self.num_queries + self.num_generations))
            kv_lens = self.cached_token_lens + self.seq_lens_cuda * mask.to(
                self.seq_lens_cuda.device)

        # number of cache blocks used by each sequence in the cache
        self.num_blocks = [len(block_ids) for block_ids in block_ids_per_seq]

        self.num_context_blocks = sum(self.num_blocks[:self.num_contexts])
        self.num_query_blocks = sum(
            self.num_blocks[self.num_contexts:self.num_contexts +
                            self.num_queries])
        self.num_generation_blocks = sum(self.num_blocks[self.num_contexts +
                                                         self.num_queries:])

        # number of tokens in the last cache block used by each sequence
        paged_kv_last_page_len = kv_lens - (
            torch.Tensor(self.num_blocks).int().cuda() - 1) * self.page_size
        self._paged_kv_last_page_len[:paged_kv_last_page_len.size(0)].copy_(
            paged_kv_last_page_len, non_blocking=True)

        # Ragged page table, see https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout
        # For decoding, this MUST be allocated ahead of time (for CUDA graphs).
        # Prefill is prepared here as well just for the sake of consistency.
        paged_kv_indptr_decode = torch.cumsum(
            torch.Tensor([0] + self.num_blocks[self.num_contexts +
                                               self.num_queries:]).int(),
            dtype=torch.int32,
            dim=0,
        )
        self.paged_kv_indptr_decode[:paged_kv_indptr_decode.size(0)].copy_(
            paged_kv_indptr_decode, non_blocking=True)

        # We treat query requests as same as context requests for flashinfer kernel calls
        paged_kv_indptr_prefill = torch.cumsum(
            torch.tensor([0] +
                         self.num_blocks[:self.num_contexts + self.num_queries],
                         dtype=torch.int32),
            dtype=torch.int32,
            dim=0,
        )
        self.paged_kv_indptr_prefill[:paged_kv_indptr_prefill.size(0)].copy_(
            paged_kv_indptr_prefill, non_blocking=True)

        # This paged_kv_indptr attribute has prefill, query and decode information in it.
        # It's for the append_paged_kv_cache kernel.
        if self.num_contexts + self.num_queries == 0:
            self.paged_kv_indptr = self.paged_kv_indptr_decode[:
                                                               paged_kv_indptr_decode
                                                               .size(0)]
        elif self.num_generations == 0:
            assert not self.is_cuda_graph
            self.paged_kv_indptr = self.paged_kv_indptr_prefill[:
                                                                paged_kv_indptr_prefill
                                                                .size(0)]

        else:
            assert not self.is_cuda_graph
            self.paged_kv_indptr = torch.cumsum(
                torch.tensor([0] + self.num_blocks, dtype=torch.int32),
                dtype=torch.int32,
                dim=0,
            ).cuda()

        for plan_params in self._plan_params_to_wrappers:
            # Re-plan the cached wrappers for a new set of requests.
            self._plan_params_to_wrappers[plan_params].is_planned = False
            self._plan_with_params(plan_params)

    def _plan_with_params(self, plan_params: PlanParams) -> PlanParams:
        if not self.needs_plan(plan_params):
            return plan_params

        if self.is_cuda_graph and torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "Cannot plan() for flashinfer kernels while stream is capturing. "
                "Make sure you run a few warmup runs before capturing the graph!"
            )

        if not self.is_cuda_graph:
            if plan_params in self._plan_params_to_wrappers:
                prefill_wrapper = self._plan_params_to_wrappers[
                    plan_params].prefill_wrapper
            else:
                prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer, self.kv_layout, backend='fa2')
        else:
            prefill_wrapper = None

        is_causal = False
        if plan_params.attention_mask_type == AttentionMaskType.causal:
            is_causal = True
        elif plan_params.attention_mask_type == AttentionMaskType.padding:
            is_causal = False

        def prefill_plan():
            assert prefill_wrapper is not None, "Prefill not supported w/ CUDA graphs"
            prefill_wrapper.plan(
                self.qo_indptr[:self.num_contexts + self.num_queries + 1],
                self.paged_kv_indptr_prefill[:self.num_contexts +
                                             self.num_queries + 1],
                self._paged_kv_indices[:self.num_context_blocks +
                                       self.num_query_blocks],
                self._paged_kv_last_page_len[:self.num_contexts +
                                             self.num_queries],
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim,
                self.page_size,
                causal=is_causal,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype)

        if plan_params in self._plan_params_to_wrappers:
            decode_wrapper = self._plan_params_to_wrappers[
                plan_params].decode_wrapper
        else:
            use_tensor_cores = self._use_tensor_cores(plan_params)

            decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                self.kv_layout,
                use_cuda_graph=self.is_cuda_graph,
                paged_kv_indptr_buffer=self.paged_kv_indptr_decode,
                paged_kv_indices_buffer=self._paged_kv_indices,
                paged_kv_last_page_len_buffer=self._paged_kv_last_page_len,
                use_tensor_cores=use_tensor_cores,
            )

        def decode_plan():
            paged_kv_indptr = torch.cumsum(
                torch.Tensor([0] +
                             self.num_blocks[self.num_contexts +
                                             self.num_queries:]).int().cuda(),
                dtype=torch.int32,
                dim=0,
            )
            decode_wrapper.plan(
                paged_kv_indptr,
                self.paged_kv_indices[self.num_context_blocks +
                                      self.num_query_blocks:],
                self.paged_kv_last_page_len[self.num_contexts +
                                            self.num_queries:],
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim,
                self.page_size,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
            )

        # Must sync after append_paged_kv_cache and before plan
        torch.cuda.current_stream().synchronize()

        if self.num_contexts + self.num_queries > 0:
            prefill_plan()

        if self.num_generations > 0:
            decode_plan()

        self._plan_params_to_wrappers[plan_params] = StarAttentionWrappers(
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
            is_planned=True,
        )

        return plan_params


class StarAttention(AttentionBackend[StarAttentionMetadata]):

    Metadata = StarAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, **kwargs)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: StarAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                **kwargs) -> torch.Tensor:
        assert isinstance(
            metadata,
            StarAttentionMetadata,
        )
        assert not metadata.is_cross, "Star Attention does not support cross attention yet."

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        num_contexts = metadata.num_contexts
        num_queries = metadata.num_queries
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens
        num_qry_tokens = metadata.num_qry_tokens

        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx)
        if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
        ):
            qc = self.quant_config
            if qc.layer_quant_mode.has_fp8_kv_cache():
                assert kv_cache.dtype == torch.float8_e4m3fn, f"KV cache should have fp8 dtype, but get {kv_cache.dtype}"
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
        assert k.dtype == v.dtype == kv_cache.dtype, f"KV cache dtype {kv_cache.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"

        def append_kv_cache(append_key, append_value, begin_req_id, num_reqs):
            # calculate new ids
            qo_indptr_n = metadata.qo_indptr[begin_req_id:(
                begin_req_id + num_reqs + 1)] - metadata.qo_indptr[begin_req_id]

            paged_kv_indptr_n = metadata.paged_kv_indptr[begin_req_id:(
                begin_req_id + num_reqs +
                1)] - metadata.paged_kv_indptr[begin_req_id]
            paged_kv_indices_n = metadata.paged_kv_indices[
                metadata.paged_kv_indptr[begin_req_id]:metadata.
                paged_kv_indptr[begin_req_id + num_reqs]]
            paged_kv_last_page_len_n = metadata.paged_kv_last_page_len[
                begin_req_id:begin_req_id + num_reqs]

            seqlens = flashinfer.get_seq_lens(paged_kv_indptr_n,
                                              paged_kv_last_page_len_n,
                                              metadata.page_size)
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                qo_indptr_n, seqlens, append_key.shape[0])
            flashinfer.page.append_paged_kv_cache(
                append_key=append_key,
                append_value=append_value,
                batch_indices=batch_indices,
                positions=positions,
                paged_kv_cache=kv_cache,
                kv_indices=paged_kv_indices_n,
                kv_indptr=paged_kv_indptr_n,
                kv_last_page_len=paged_kv_last_page_len_n,
                kv_layout=metadata.kv_layout)

        if metadata.is_cuda_graph:
            # decode phase, only append at last rank
            if metadata.mapping.cp_rank == metadata.mapping.cp_size - 1:
                seqlens = flashinfer.get_seq_lens(
                    metadata.paged_kv_indptr, metadata.paged_kv_last_page_len,
                    metadata.page_size)
                batch_indices, positions = flashinfer.get_batch_indices_positions(
                    metadata.qo_indptr, seqlens, k.shape[0])
                flashinfer.page.append_paged_kv_cache(
                    append_key=k,
                    append_value=v,
                    batch_indices=batch_indices,
                    positions=positions,
                    paged_kv_cache=kv_cache,
                    kv_indices=metadata.paged_kv_indices,
                    kv_indptr=metadata.paged_kv_indptr,
                    kv_last_page_len=metadata.paged_kv_last_page_len,
                    kv_layout=metadata.kv_layout)
        else:
            if metadata.mapping.cp_rank != metadata.mapping.cp_size - 1:
                if num_contexts > 0:
                    append_kv_cache(k[:num_ctx_tokens], v[:num_ctx_tokens], 0,
                                    num_contexts)
            else:
                append_kv_cache(k, v, 0,
                                num_contexts + num_generations + num_queries)

        def prefill_forward(plan_params: PlanParams):
            wrapper = metadata.get_prefill_wrapper(plan_params)
            query = q[:num_ctx_tokens + num_qry_tokens]
            output, lse = wrapper.run(query, kv_cache, return_lse=True)
            return output, lse

        def decode_forward(plan_params: PlanParams):
            wrapper = metadata.get_decode_wrapper(plan_params)
            output, lse = wrapper.run(q[num_ctx_tokens + num_qry_tokens:],
                                      kv_cache,
                                      return_lse=True)
            return output, lse

        if attention_mask == PredefinedAttentionMask.CAUSAL:
            attention_mask_type = int(AttentionMaskType.causal)
            attention_mask_data = None
        elif attention_mask == PredefinedAttentionMask.FULL:
            attention_mask_type = int(AttentionMaskType.padding)
            attention_mask_data = None
        else:
            raise ValueError("Unexpected attention mask type")

        # this will do nothing if the last forward pass had the same parameters
        plan_params = metadata.plan(self.num_heads,
                                    self.num_kv_heads,
                                    self.head_dim,
                                    q_dtype=q.dtype,
                                    kv_dtype=kv_cache.dtype,
                                    attention_mask_type=attention_mask_type,
                                    attention_mask_data=attention_mask_data)

        def reduce_results(output, lse):
            out_tmp = output
            lse = lse.unsqueeze(-1) / np.log2(np.e)  # [b * s, nheads, 1]
            if metadata.mapping.cp_size != 1:
                output_tensor = allgather(output,
                                          metadata.mapping,
                                          gather_dim=0)
                lse_tensor = allgather(lse, metadata.mapping, gather_dim=0)
                output_tensor = output_tensor.to(torch.float32)
            else:
                lse_tensor = lse
                output_tensor = output.to(torch.float32)
            out_gather = torch.split(output_tensor, out_tmp.size(0))
            lse_gather = torch.split(lse_tensor, out_tmp.size(0))
            out, lse = out_gather[0], lse_gather[0]
            for i in range(1, len(out_gather)):
                block_out, block_lse = out_gather[i], lse_gather[i]
                new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
                out = torch.exp(lse - new_lse) * out + torch.exp(
                    block_lse - new_lse) * block_out
                lse = new_lse

            output = out.type_as(out_tmp)
            return output

        if num_contexts + num_queries > 0:
            ctx_output, ctx_lse = prefill_forward(plan_params)
            if num_contexts > 0:
                context_output = ctx_output[:num_ctx_tokens]
                context_output = context_output.view(num_ctx_tokens, -1)
            else:
                context_output = torch.tensor([], dtype=q.dtype).to(q.device)
            if num_queries > 0:
                query_output = ctx_output[num_ctx_tokens:]
                query_lse = ctx_lse[num_ctx_tokens:]
                query_output = reduce_results(query_output, query_lse)
                query_output = query_output.view(num_qry_tokens, -1)
            else:
                query_output = torch.tensor([], dtype=q.dtype).to(q.device)

            ctx_output = torch.concat([context_output, query_output])
        else:
            ctx_output = torch.tensor([], dtype=q.dtype).to(q.device)

        if num_generations > 0:
            gen_output, gen_lse = decode_forward(plan_params)
            gen_output = reduce_results(gen_output, gen_lse)
            gen_output = gen_output.view(num_generations, -1)
        else:
            gen_output = torch.tensor([], dtype=q.dtype).to(q.device)

        output = torch.cat([ctx_output, gen_output])
        return output
