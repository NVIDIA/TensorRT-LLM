import math
import os
import weakref
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import flashinfer
import torch
from flashinfer.jit.core import check_cuda_arch
from typing_extensions import Self

from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..utils import get_global_attrs, get_model_extra_attrs
from .interface import (AttentionBackend, AttentionMask, AttentionMetadata,
                        CustomAttentionMask, PredefinedAttentionMask)

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
    sm_scale: Optional[float] = None
    window_left: Optional[int] = None


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
            # Note: even though flashinfer only recommends 128 MB, we have to push it
            # a bit higher to cover all possible CUDA graph cases. If it's too small,
            # warmup will crash.
            self.workspace_buffer = torch.empty(320 * 1024 * 1024,
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

    def create_cuda_graph_metadata(self,
                                   max_batch_size: int,
                                   sub_cross_metadata: bool = False,
                                   max_draft_tokens: int = 0,
                                   buffers=None) -> Self:
        metadata = super().create_cuda_graph_metadata(max_batch_size,
                                                      sub_cross_metadata,
                                                      max_draft_tokens)
        metadata.max_num_requests = max_batch_size
        metadata.max_num_tokens = max_batch_size * (1 + max_draft_tokens)
        # Post init again to make sure all tensors are allocated
        metadata.__post_init__()
        return metadata

    @property
    def page_size(self) -> int:
        """
        Number of tokens per cache page
        """
        return self.kv_cache_manager.tokens_per_block

    def prepare(self) -> None:
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self)
        # start and end indices of each sequence in the ragged query
        torch.cumsum(self.seq_lens_cuda,
                     dim=0,
                     dtype=torch.int32,
                     out=self._qo_indptr[1:self.seq_lens_cuda.size(0) + 1])

        # indices of used cache blocks for each sequence
        assert self.request_ids is not None
        block_ids_per_seq = self.kv_cache_manager.get_batch_cache_indices(
            self.request_ids)

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
        # NOTE: do not use len(block_ids) - that will give you a number
        # that can be too big if using chunked prefill/kv cache reuse
        # since we allocate all blocks ahead of time.
        num_blocks = ((kv_lens + self.page_size - 1) // self.page_size)
        self.num_blocks = num_blocks.tolist()
        self.num_context_blocks = sum(self.num_blocks[:self.num_contexts])
        self.num_generation_blocks = sum(self.num_blocks[self.num_contexts:])

        paged_kv_indices_list = []
        for i, block_ids in enumerate(block_ids_per_seq):
            paged_kv_indices_list.extend(block_ids[:self.num_blocks[i]])

        paged_kv_indices = torch.tensor(paged_kv_indices_list,
                                        dtype=torch.int32)

        self._paged_kv_indices[:paged_kv_indices.size(0)].copy_(
            paged_kv_indices, non_blocking=True)

        # number of tokens in the last cache block used by each sequence
        paged_kv_last_page_len = kv_lens - (num_blocks - 1) * self.page_size
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
            self.paged_kv_indptr = self.paged_kv_indptr_prefill[:
                                                                paged_kv_indptr_prefill
                                                                .size(0)]
        else:
            assert not self.is_cuda_graph, "Cannot mix decode/prefill with CUDA graphs"
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

        # Generally, plan_params with non-trivial attention_mask_data are relevant only the
        # corresponding forward pass. So, flush them out here as they won't be relevant for
        # subsequent forward calls.
        for plan_params in list(self._plan_params_to_wrappers.keys()):
            if plan_params.attention_mask_data is None:
                # Re-plan the cached wrappers for a new set of requests.
                self._plan_params_to_wrappers[plan_params].is_planned = False
                self._plan_with_params(plan_params)
            else:
                del self._plan_params_to_wrappers[plan_params]

        if self.cross is not None and self.cross is not self:
            self.cross.prepare()

    def plan(self,
             num_heads: int,
             num_kv_heads: int,
             head_dim: int,
             q_dtype: torch.dtype,
             kv_dtype: torch.dtype,
             attention_mask_type: int,
             q_scaling: Optional[float] = None,
             attention_window_size: Optional[int] = None,
             attention_mask_data: Optional[torch.Tensor] = None) -> PlanParams:

        sm_scale = None
        if q_scaling is not None:
            sm_scale = 1 / (math.sqrt(head_dim) * q_scaling)

        plan_params = PlanParams(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            sm_scale=sm_scale,
            window_left=attention_window_size
            if attention_window_size is not None else -1,
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

        if plan_params in self._plan_params_to_wrappers:
            prefill_wrapper = self._plan_params_to_wrappers[
                plan_params].prefill_wrapper
        else:
            # flashinfer fa3 backend has accuracy issue in H100 PCIe
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                self.kv_layout,
                backend='fa2',
                qo_indptr_buf=self.qo_indptr,
                paged_kv_indptr_buf=self.paged_kv_indptr_prefill,
                paged_kv_indices_buf=self._paged_kv_indices,
                paged_kv_last_page_len_buf=self._paged_kv_last_page_len,
                use_cuda_graph=self.is_cuda_graph)

        is_causal = plan_params.attention_mask_type == AttentionMaskType.causal

        def prefill_plan():
            # Setting `window_left` to -1 for custom attention mask is important.
            # Else, FlashInfer proceeds to use SWA regardless of attention_mask_data.
            if plan_params.attention_mask_data is not None:
                window_left = -1
            else:
                window_left = plan_params.window_left
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
                sm_scale=plan_params.sm_scale,
                window_left=window_left,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
                custom_mask=plan_params.attention_mask_data,
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
                sm_scale=plan_params.sm_scale,
                window_left=plan_params.window_left,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
            )

        # Must sync after append_paged_kv_cache and before plan.
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


class FlashInferAttention(AttentionBackend[FlashInferAttentionMetadata]):

    Metadata = FlashInferAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        skip_create_weights_in_init: bool = False,
        **kwargs,
    ):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, **kwargs)
        if not skip_create_weights_in_init:
            self.update_quant_config(self.quant_config)
        self.q_scaling = q_scaling

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        self.quant_config = new_quant_config
        self.has_fp8_kv_cache = False
        if self.quant_config:
            self.has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache(
            )

    def forward_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: FlashInferAttentionMetadata,
        attention_mask_type: int,
        output: torch.Tensor,
        attention_mask_data: Optional[torch.Tensor] = None,
        attention_window_size: Optional[int] = None,
    ) -> None:
        # Query
        q = q.view(-1, self.num_heads, self.head_dim)

        # Key and Value
        kv_cache = metadata.kv_cache_manager.get_buffers(self.layer_idx)

        if k is not None and v is not None:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            if self.has_fp8_kv_cache:
                assert kv_cache.dtype == torch.float8_e4m3fn, (
                    f"KV cache should have fp8 dtype, but get {kv_cache.dtype}")
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_cache.dtype, (
                f"KV cache dtype {kv_cache.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"
            )

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

        def prefill_forward(plan_params: PlanParams, out: torch.Tensor):
            wrapper = metadata.get_prefill_wrapper(plan_params)
            wrapper.run(q[:num_ctx_tokens],
                        kv_cache,
                        out=out.view(-1, self.num_heads, self.head_dim))

        def decode_forward(plan_params: PlanParams, out: torch.Tensor):
            wrapper = metadata.get_decode_wrapper(plan_params)
            wrapper.run(q[num_ctx_tokens:],
                        kv_cache,
                        out=out.view(-1, self.num_heads, self.head_dim))

        # this will do nothing if the last forward pass had the same parameters
        plan_params = metadata.plan(self.num_heads,
                                    self.num_kv_heads,
                                    self.head_dim,
                                    q_dtype=q.dtype,
                                    kv_dtype=kv_cache.dtype,
                                    q_scaling=self.q_scaling,
                                    attention_window_size=attention_window_size,
                                    attention_mask_type=attention_mask_type,
                                    attention_mask_data=attention_mask_data)

        if num_contexts == 0:
            decode_forward(plan_params, output)
        elif num_generations == 0:
            prefill_forward(plan_params, output)
        else:
            prefill_forward(plan_params, output[:num_ctx_tokens, :])
            decode_forward(plan_params, output[num_ctx_tokens:, :])

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: FlashInferAttentionMetadata,
                *,
                attention_window_size: Optional[int] = None,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                attention_mask_data: Optional[torch.Tensor] = None,
                output: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if attention_mask == CustomAttentionMask.CUSTOM:
            assert attention_mask_data is not None, "attention_mask_data is required for custom attention mask."
            attention_mask_type = int(AttentionMaskType.custom_mask)
            attention_mask_data = attention_mask_data if attention_mask_data.ndim == 1 else attention_mask_data.flatten(
            )
        elif attention_mask == PredefinedAttentionMask.CAUSAL:
            attention_mask_type = int(AttentionMaskType.causal)
            attention_mask_data = None
        elif attention_mask == PredefinedAttentionMask.FULL:
            attention_mask_type = int(AttentionMaskType.padding)
            attention_mask_data = None
        else:
            raise ValueError("Unexpected attention mask type")

        if output is None:
            output = torch.empty_like(q)

        # FlashInfer's sliding window attention is inclusive, while the attention window size defined in TRTLLM is exclusive.
        # So we need to subtract 1 from the attention window size for a consistent behavior.
        if attention_window_size is not None:
            attention_window_size = attention_window_size - 1

        self.forward_impl(q=q,
                          k=k,
                          v=v,
                          metadata=metadata,
                          attention_mask_type=attention_mask_type,
                          attention_mask_data=attention_mask_data,
                          attention_window_size=attention_window_size,
                          output=output)
        return output
