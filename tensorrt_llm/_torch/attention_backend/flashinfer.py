import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import flashinfer
import torch
from flashinfer.jit.core import check_cuda_arch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionBackend, AttentionMask, AttentionMetadata, PredefinedAttentionMask)
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.models.modeling_utils import QuantConfig

try:
    check_cuda_arch()
except RuntimeError:
    # Override TORCH_CUDA_ARCH_LIST for JIT compilation of flashinfer kernels
    # since the existed TORCH_CUDA_ARCH_LIST may be too general and flashinfer requires sm75+.
    capability = torch.cuda.get_device_capability()
    arch_list = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list


@dataclass(kw_only=True, frozen=True)
class PlanParams:
    """
    Parameters that affect the flashinfer execution plan
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype

    attention_mask_type: AttentionMaskType
    attention_mask_data: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class FlashInferWrappers:
    decode_wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper
    prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper]

    is_planned: bool


@dataclass(kw_only=True)
class FlashInferAttentionMetadata(AttentionMetadata):
    workspace_buffer: Optional[torch.Tensor] = None

    kv_layout: Literal["NHD", "HND"] = "NHD"

    paged_kv_indptr_decode: torch.Tensor = field(init=False)
    paged_kv_indptr_prefill: torch.Tensor = field(init=False)
    _paged_kv_indices: torch.Tensor = field(init=False, repr=False)
    _paged_kv_last_page_len: torch.Tensor = field(init=False)
    _qo_indptr: torch.Tensor = field(init=False)
    _kv_indptr: torch.Tensor = field(init=False)
    _cached_token_lens: torch.Tensor = field(init=False)

    _plan_params_to_wrappers: Dict[PlanParams,
                                   FlashInferWrappers] = field(init=False)

    def needs_plan(self, plan_params: PlanParams) -> bool:
        if plan_params not in self._plan_params_to_wrappers:
            return True

        wrappers = self._plan_params_to_wrappers[plan_params]
        return not wrappers.is_planned

    def get_prefill_wrapper(
        self, plan_params: PlanParams
    ) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
        assert plan_params in self._plan_params_to_wrappers, "Plan params not found, make sure to call plan()"
        result = self._plan_params_to_wrappers[plan_params].prefill_wrapper
        assert result is not None, "Prefill wrapper was not created in plan()"
        return result

    def get_decode_wrapper(
        self, plan_params: PlanParams
    ) -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
        assert plan_params in self._plan_params_to_wrappers, "Plan params not found, make sure to call plan()"
        result = self._plan_params_to_wrappers[plan_params].decode_wrapper
        return result

    @property
    def paged_kv_indices(self) -> torch.Tensor:
        return self._paged_kv_indices[:self.num_generation_blocks +
                                      self.num_context_blocks]

    @property
    def paged_kv_last_page_len(self) -> torch.Tensor:
        return self._paged_kv_last_page_len[:self.num_contexts +
                                            self.num_generations]

    @property
    def qo_indptr(self) -> torch.Tensor:
        return self._qo_indptr[:self.num_contexts + self.num_generations + 1]

    @property
    def kv_indptr(self) -> torch.Tensor:
        return self._kv_indptr[:self.num_contexts + self.num_generations + 1]

    @property
    def cached_token_lens(self) -> torch.Tensor:
        return self._cached_token_lens[:self.num_contexts +
                                       self.num_generations]

    @property
    def batch_indices(self) -> torch.Tensor:
        return self._batch_indices[:self.num_tokens]

    @property
    def positions(self) -> torch.Tensor:
        return self._positions[:self.num_tokens]

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.workspace_buffer is None:
            self.workspace_buffer = torch.empty(256 * 1024 * 1024,
                                                dtype=torch.uint8,
                                                device="cuda")

        self.paged_kv_indptr_decode = torch.empty((self.max_num_requests + 1, ),
                                                  device='cuda',
                                                  dtype=torch.int)
        self.paged_kv_indptr_prefill = torch.empty(
            (self.max_num_requests + 1, ), device='cuda', dtype=torch.int)
        self._paged_kv_last_page_len = torch.empty((self.max_num_requests, ),
                                                   device='cuda',
                                                   dtype=torch.int)

        self._qo_indptr = torch.zeros(self.max_num_requests + 1,
                                      device='cuda',
                                      dtype=torch.int)

        self._kv_indptr = torch.zeros(
            self.max_num_requests + 1, device='cuda',
            dtype=torch.int) if self.is_cross else self._qo_indptr

        self._cached_token_lens = torch.empty((self.max_num_requests, ),
                                              dtype=torch.int,
                                              device='cuda')
        self._batch_indices = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self._positions = torch.empty((self.max_num_tokens, ),
                                      dtype=torch.int,
                                      device='cuda')
        self._plan_params_to_wrappers = {}

        if self.kv_cache_manager is not None:
            max_num_pages = self.kv_cache_manager.blocks_in_primary_pool
            self._paged_kv_indices = torch.empty((max_num_pages, ),
                                                 device='cuda',
                                                 dtype=torch.int)

    @property
    def page_size(self) -> int:
        """
        Number of tokens per cache page
        """
        return self.kv_cache_manager.tokens_per_block

    def prepare(self) -> None:
        _thread_local.metadata = self
        # start and end indices of each sequence in the ragged query
        torch.cumsum(self.seq_lens_cuda,
                     dim=0,
                     dtype=torch.int32,
                     out=self._qo_indptr[1:self.seq_lens_cuda.size(0) + 1])

        # indices of used cache blocks for each sequence
        assert self.request_ids is not None
        block_ids_per_seq = self.kv_cache_manager.get_batch_cache_indices(
            self.request_ids)
        paged_kv_indices = torch.tensor(
            [x for block_ids in block_ids_per_seq for x in block_ids],
            dtype=torch.int32)
        self._paged_kv_indices[:paged_kv_indices.size(0)].copy_(
            paged_kv_indices, non_blocking=True)

        # number of tokens in the kv cache for each sequence in the batch
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq, dtype=torch.int)
        self._cached_token_lens[:cached_token_lens.size(0)].copy_(
            cached_token_lens, non_blocking=True)

        # number of tokens needed in the kv cache for each sequence after the next pass
        kv_lens = self.cached_token_lens + self.seq_lens_kv_cuda

        # start and end indices of each sequence in the ragged key and value
        # for self attention it's the same as qo_indptr so avoid computing twice.
        if self.is_cross:
            # start and end indices of each sequence in the ragged key and value
            torch.cumsum(self.seq_lens_kv_cuda,
                         dim=0,
                         dtype=torch.int32,
                         out=self._kv_indptr[1:self.seq_lens_kv_cuda.size(0) +
                                             1])

        # number of cache blocks used by each sequence in the cache
        self.num_blocks = [len(block_ids) for block_ids in block_ids_per_seq]
        self.num_context_blocks = sum(self.num_blocks[:self.num_contexts])
        self.num_generation_blocks = sum(self.num_blocks[self.num_contexts:])

        # number of tokens in the last cache block used by each sequence
        paged_kv_last_page_len = kv_lens - (torch.Tensor(
            self.num_blocks).int().cuda(non_blocking=True) - 1) * self.page_size
        self._paged_kv_last_page_len[:paged_kv_last_page_len.size(0)].copy_(
            paged_kv_last_page_len, non_blocking=True)

        # Ragged page table, see https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout
        # For decoding, this MUST be allocated ahead of time (for CUDA graphs).
        # Prefill is prepared here as well just for the sake of consistency.
        paged_kv_indptr_decode = torch.cumsum(
            torch.Tensor([0] + self.num_blocks[self.num_contexts:]).int(),
            dtype=torch.int32,
            dim=0,
        )
        self.paged_kv_indptr_decode[:paged_kv_indptr_decode.size(0)].copy_(
            paged_kv_indptr_decode, non_blocking=True)

        paged_kv_indptr_prefill = torch.cumsum(
            torch.tensor([0] + self.num_blocks[:self.num_contexts],
                         dtype=torch.int32),
            dtype=torch.int32,
            dim=0,
        )
        self.paged_kv_indptr_prefill[:paged_kv_indptr_prefill.size(0)].copy_(
            paged_kv_indptr_prefill, non_blocking=True)

        # This paged_kv_indptr attribute has both prefill and decode information in it.
        # It's for the append_paged_kv_cache kernel.
        if self.num_contexts == 0:
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

        # For cross attention, num_tokens is 0 during decode, and we don't need to update kv cache.
        if self.num_tokens > 0:
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                self.kv_indptr,
                flashinfer.get_seq_lens(self.paged_kv_indptr,
                                        self.paged_kv_last_page_len,
                                        self.page_size),
                self.num_tokens,
            )
            self._batch_indices[:batch_indices.size(0)].copy_(batch_indices,
                                                              non_blocking=True)
            self._positions[:positions.size(0)].copy_(positions,
                                                      non_blocking=True)

        for plan_params in self._plan_params_to_wrappers:
            # Re-plan the cached wrappers for a new set of requests.
            self._plan_params_to_wrappers[plan_params].is_planned = False
            self._plan_with_params(plan_params)

        if self.cross is not None and self.cross is not self:
            self.cross.prepare()

    def plan(self,
             num_heads: int,
             num_kv_heads: int,
             head_dim: int,
             q_dtype: torch.dtype,
             kv_dtype: torch.dtype,
             attention_mask_type: int,
             attention_mask_data: Optional[torch.Tensor] = None) -> PlanParams:
        plan_params = PlanParams(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            attention_mask_type=AttentionMaskType(attention_mask_type),
            attention_mask_data=attention_mask_data)
        return self._plan_with_params(plan_params)

    def _use_tensor_cores(self, plan_params: PlanParams):
        return plan_params.kv_dtype in [
            torch.float8_e4m3fn, torch.float8_e5m2
        ] or (plan_params.num_heads // plan_params.num_kv_heads >= 4)

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
                # flashinfer fa3 backend has accuracy issue in H100 PCIe
                prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer, self.kv_layout, backend='fa2')
        else:
            prefill_wrapper = None

        is_causal = plan_params.attention_mask_type == AttentionMaskType.causal

        def prefill_plan():
            assert prefill_wrapper is not None, "Prefill not supported w/ CUDA graphs"
            prefill_wrapper.plan(
                self.qo_indptr[:self.num_contexts + 1],
                self.paged_kv_indptr_prefill[:self.num_contexts + 1],
                self._paged_kv_indices[:self.num_context_blocks],
                self._paged_kv_last_page_len[:self.num_contexts],
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim,
                self.page_size,
                causal=is_causal,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
            )

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
                             self.num_blocks[self.num_contexts:]).int().cuda(),
                dtype=torch.int32,
                dim=0,
            )
            decode_wrapper.plan(
                paged_kv_indptr,
                self.paged_kv_indices[self.num_context_blocks:],
                self.paged_kv_last_page_len[self.num_contexts:],
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim,
                self.page_size,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
            )

        # Must sync after append_paged_kv_cache and before plan
        torch.cuda.current_stream().synchronize()

        if self.num_contexts > 0:
            prefill_plan()

        if self.num_generations > 0:
            decode_plan()

        self._plan_params_to_wrappers[plan_params] = FlashInferWrappers(
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
            is_planned=True,
        )

        return plan_params


_thread_local = threading.local()


def get_metadata() -> FlashInferAttentionMetadata:
    try:
        return _thread_local.metadata
    except AttributeError:
        return None


class FlashInferAttention(AttentionBackend[FlashInferAttentionMetadata]):

    Metadata = FlashInferAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config)

        self.has_fp8_kv_cache = False
        if quant_config and quant_config.layer_quant_mode.has_any_quant():
            quant_mode = quant_config.layer_quant_mode
            if quant_mode.has_fp8_kv_cache():
                self.has_fp8_kv_cache = True

    @torch.library.custom_op("trtllm::flashinfer_forward", mutates_args=())
    @staticmethod
    def forward_pattern(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_idx: int,
        has_fp8_kv_cache: bool,
        attention_mask_type: int,
        attention_mask_data: Optional[torch.Tensor],
    ) -> torch.Tensor:
        '''
        Wrapping the flashinfer forward as a custom op is required to fix `torch.compile` graph breaks,
        otherwise it will graph break when calling `metadata.num_contexts` since it convert tensor's sum directly to int.
        '''
        # torch.compile does not support custom object as arguments, so we have to use global function to get the metadata.
        metadata = get_metadata()

        # This is only for memory estimation for now.
        # NOTE: this method is not accurate while it works for most scenario.
        if metadata is None or metadata.kv_cache_manager is None:
            q = q.view(1, -1, num_heads, head_dim)
            k = k.view(1, -1, num_kv_heads, head_dim)
            v = v.view(1, -1, num_kv_heads, head_dim)
            return VanillaAttention.dummy_forward(q, k, v)

        assert isinstance(
            metadata,
            FlashInferAttentionMetadata,
        )

        # Query
        q = q.view(-1, num_heads, head_dim)

        # Key and Value
        kv_cache = metadata.kv_cache_manager.get_buffers(layer_idx)

        if k is not None and v is not None:
            k = k.view(-1, num_kv_heads, head_dim)
            v = v.view(-1, num_kv_heads, head_dim)

            if has_fp8_kv_cache:
                assert kv_cache.dtype == torch.float8_e4m3fn, f"KV cache should have fp8 dtype, but get {kv_cache.dtype}"
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_cache.dtype, f"KV cache dtype {kv_cache.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"

            flashinfer.page.append_paged_kv_cache(
                append_key=k,
                append_value=v,
                batch_indices=metadata.batch_indices,
                positions=metadata.positions,
                paged_kv_cache=kv_cache,
                kv_indices=metadata.paged_kv_indices,
                kv_indptr=metadata.paged_kv_indptr,
                kv_last_page_len=metadata.paged_kv_last_page_len,
                kv_layout=metadata.kv_layout)

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens

        def prefill_forward(plan_params: PlanParams):
            wrapper = metadata.get_prefill_wrapper(plan_params)
            output = wrapper.run(q[:num_ctx_tokens], kv_cache)
            output = output.view(num_ctx_tokens, -1)
            return output

        def decode_forward(plan_params: PlanParams):
            wrapper = metadata.get_decode_wrapper(plan_params)
            output = wrapper.run(q[num_ctx_tokens:], kv_cache)
            output = output.view(num_generations, -1)
            return output

        # this will do nothing if the last forward pass had the same parameters
        plan_params = metadata.plan(num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    q_dtype=q.dtype,
                                    kv_dtype=kv_cache.dtype,
                                    attention_mask_type=attention_mask_type,
                                    attention_mask_data=attention_mask_data)

        if num_contexts > 0:
            ctx_output = prefill_forward(plan_params)

        if num_generations > 0:
            gen_output = decode_forward(plan_params)

        if num_contexts > 0 and num_generations > 0:
            output = torch.cat([ctx_output, gen_output], dim=0)
        elif num_contexts > 0:
            output = ctx_output
        elif num_generations > 0:
            output = gen_output

        return output

    @forward_pattern.register_fake
    @staticmethod
    def _(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_idx: int,
        has_fp8_kv_cache: bool,
        attention_mask_type: int,
        attention_mask_data: Optional[torch.Tensor],
    ):
        return torch.empty_like(q)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: FlashInferAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                **kwargs) -> torch.Tensor:
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            attention_mask_type = int(AttentionMaskType.causal)
            attention_mask_data = None
        elif attention_mask == PredefinedAttentionMask.FULL:
            attention_mask_type = int(AttentionMaskType.padding)
            attention_mask_data = None
        else:
            raise ValueError("Unexpected attention mask type")

        return FlashInferAttention.forward_pattern(
            q, k, v, self.num_heads, self.head_dim, self.num_kv_heads,
            self.layer_idx, self.has_fp8_kv_cache, attention_mask_type,
            attention_mask_data)
