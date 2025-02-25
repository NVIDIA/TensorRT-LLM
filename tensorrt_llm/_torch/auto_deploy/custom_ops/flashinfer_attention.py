from dataclasses import dataclass, fields
from typing import Dict, List, Literal, Optional, Tuple, Union

import flashinfer
import torch

from ..utils.cuda_graph import cuda_graph_state
from .attention_interface import (
    AttentionDescriptor,
    AttentionInfo,
    AttentionRegistry,
    BufferInitializerDict,
    CacheInitializerDict,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)


@dataclass
class PlanParams:
    """Parameters that affect the flashinfer execution plan."""

    n_heads: int
    n_kv_heads: int
    head_dim: int
    num_seq: int
    is_generate: bool
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype

    pos_embd_mode: Literal["NONE", "ROPE_LLAMA", "ALIBI"] = "NONE"
    rope_theta: Optional[float] = None
    rope_scale: Optional[float] = None
    causal: bool = True

    def __hash__(self):
        """Convert all fields to a string representation and concatenate them."""
        return hash("_".join([str(getattr(self, f.name)) for f in fields(self)]))


class _FlashInferPlanner:
    """A class interface to handle flashinfer-related planning/wrapping operations."""

    workspace_buffer: Optional[torch.Tensor]
    prefill_wrapper: Optional[flashinfer.BatchPrefillWithPagedKVCacheWrapper]
    decode_wrapper: Optional[flashinfer.BatchDecodeWithPagedKVCacheWrapper]
    cached_decode_wrappers: Dict[PlanParams, flashinfer.BatchDecodeWithPagedKVCacheWrapper]
    plan_params: Optional[PlanParams]

    def __init__(self):
        self.workspace_buffer = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.cached_decode_wrappers = {}
        self.plan_params = None

    def _init_decode_wrapper(self):
        assert self.workspace_buffer is not None
        return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD", use_tensor_cores=True
        )

    def init_workspace(self, workspace_buffer: torch.Tensor):
        self.__init__()  # reset all state

        self.workspace_buffer = workspace_buffer
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.decode_wrapper = self._init_decode_wrapper()

    def reset(self) -> None:
        self.plan_params = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        plan_params: PlanParams,
    ) -> Union[
        flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    ]:
        # plan decode helper function
        def _plan_decode(wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper):
            wrapper.plan(
                kv_page_indptr,
                kv_page_indices,
                kv_last_page_len,
                plan_params.n_heads,
                plan_params.n_kv_heads,
                plan_params.head_dim,
                plan_params.page_size,
                pos_encoding_mode=plan_params.pos_embd_mode,
                rope_theta=plan_params.rope_theta,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
            )

        # we want to plan during warm-up of cuda graph capture to ensure we have the plan cached
        if cuda_graph_state.in_warm_up() and plan_params not in self.cached_decode_wrappers:
            self.cached_decode_wrappers[plan_params] = self._init_decode_wrapper()
            _plan_decode(self.cached_decode_wrappers[plan_params])

        # check if we are in cuda graph capture and just return the pre-cached decode wrapper
        if torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up():
            assert plan_params.is_generate, "Only generate is supported during cuda graph capture."
            wrapper = self.cached_decode_wrappers[plan_params]
            # copy the metadata to the wrapper to ensure it is up-to-date for graph replay!
            wrapper._paged_kv_indptr_buf.copy_(kv_page_indptr)
            wrapper._paged_kv_indices_buf.copy_(kv_page_indices)
            wrapper._paged_kv_last_page_len_buf.copy_(kv_last_page_len)
            return wrapper

        # check for re-planning
        if plan_params != self.plan_params:
            if plan_params.is_generate:
                _plan_decode(self.decode_wrapper)
            else:
                # plan prefill
                self.prefill_wrapper.plan(
                    qo_indptr,
                    kv_page_indptr,
                    kv_page_indices,
                    kv_last_page_len,
                    plan_params.n_heads,  # Q heads
                    plan_params.n_kv_heads,  # KV heads
                    plan_params.head_dim,
                    plan_params.page_size,
                    causal=plan_params.causal,
                    pos_encoding_mode=plan_params.pos_embd_mode,
                    rope_theta=plan_params.rope_theta,
                    q_data_type=plan_params.q_dtype,
                    kv_data_type=plan_params.kv_dtype,
                )
            self.plan_params = plan_params

        # return desired wrapper
        return self.decode_wrapper if plan_params.is_generate else self.prefill_wrapper


_GlobalFlashInferPlanner = _FlashInferPlanner()


@torch.library.custom_op("attention::prepare_flashinfer_metadata", mutates_args=())
def prepare_flashinfer_metadata(
    input_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    page_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for flashinfer attention.

    Please refer to https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout and
    https://docs.flashinfer.ai/api/prefill.html#flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan
    to understand the convention.
    """
    # reset the planner
    _GlobalFlashInferPlanner.reset()

    # retrieve sanitzed metadata
    seq_len = SequenceInfo._get_sanitized_seq_len(input_ids, seq_len)
    num_seq = len(seq_len)

    # prepare flashinfer-style metadata
    offsets = input_pos[:num_seq].clone()

    qo_indptr = torch.zeros(num_seq + 1, dtype=torch.int, device=seq_len.device)
    qo_indptr[1:] = torch.cumsum(seq_len, 0)

    paged_kv_indptr = torch.zeros_like(qo_indptr)
    paged_kv_indptr[1:] = torch.cumsum(pages_per_seq[:num_seq], 0)

    # NOTE: it is okay to clone cache_loc here without truncation. paged_kv_indptr is already
    # truncated and will point to the correct sub range of cache_loc.
    paged_kv_indices = cache_loc.clone()

    paged_kv_last_page_len = ((offsets + seq_len - 1) % page_size) + 1

    # return metadata
    return (qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, offsets)


@prepare_flashinfer_metadata.register_fake
def prepare_flashinfer_metadata_fake(
    input_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
):
    qo_indptr = torch.empty(len(seq_len) + 1, dtype=seq_len.dtype, device=seq_len.device)
    return (
        qo_indptr,  # qo_indptr
        torch.empty_like(qo_indptr),  # paged_kv_indptr
        torch.empty_like(cache_loc),  # paged_kv_indices
        torch.empty_like(seq_len),  # paged_kv_last_page_len
        torch.empty_like(input_pos),  # offsets
    )


@torch.library.custom_op("attention::flashinfer_mha_with_cache", mutates_args=())
def flashinfer_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    offsets: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    fuse_rope: bool,
    rope_theta: Optional[float],
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    b, s, d = q.shape
    head_dim = k_cache.shape[-1]
    n_heads = q.shape[2] // head_dim
    n_kv_heads = k.shape[2] // head_dim

    bs_view = (b * s,)
    q = q.view(*bs_view, n_heads, head_dim)
    k = k.view(*bs_view, n_kv_heads, head_dim)
    v = v.view(*bs_view, n_kv_heads, head_dim)

    pp = PlanParams(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_seq=len(qo_indptr) - 1,
        is_generate=(s == 1),
        page_size=k_cache.shape[1],
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        pos_embd_mode="ROPE_LLAMA" if fuse_rope else "NONE",
        rope_theta=rope_theta,
    )

    # TODO: Get flashinfer fuse_rope working with fp8 kv cache (https://github.com/flashinfer-ai/flashinfer/issues/661)
    if rope_theta is not None:
        q, k = flashinfer.apply_rope(q, k, qo_indptr, offsets, rope_theta=rope_theta)

    # Assuming k_scale = v_scale = 1.0, we just have to cast k and v to fp8 before appending to kv cache
    k_scale, v_scale = 1.0, 1.0
    if k_cache.dtype == torch.float8_e4m3fn:
        k = (k / k_scale).to(torch.float8_e4m3fn)
        v = (v / v_scale).to(torch.float8_e4m3fn)

    # Append to kv cache
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, pp.page_size),
        bs_view[0],
    )

    flashinfer.page.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        (k_cache, v_cache),
        paged_kv_indices,
        paged_kv_indptr,
        paged_kv_last_page_len,
    )

    # run the flashinfer planner and obtain the correct wrapper
    wrapper = _GlobalFlashInferPlanner.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        pp,
    )
    y = wrapper.run(q, (k_cache, v_cache), k_scale=k_scale, v_scale=v_scale)

    return y.view(b, s, d)  # [b,s,n*h_d]


@flashinfer_mha_with_cache.register_fake
def flashinfer_mha_with_cache_fake(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    offsets: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    fuse_rope: bool,
    rope_theta: Optional[float],
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q.contiguous())


@AttentionRegistry.register("FlashInfer")
class FlashInferAttention(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return True

    @classmethod
    def get_attention_op(cls) -> MHACallable:
        return torch.ops.attention.flashinfer_mha_with_cache

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.attention.prepare_flashinfer_metadata, 5

    @classmethod
    def get_cache_initializers(cls, attention_info: AttentionInfo) -> CacheInitializerDict:
        def _get_cache(si: SequenceInfo):
            return torch.empty(
                si.num_pages,
                si.page_size,
                attention_info.num_kv_heads,
                attention_info.head_dim,
                device=si.device,
                dtype=attention_info.cache_dtype or attention_info.dtype,
            )

        return {"k_cache": _get_cache, "v_cache": _get_cache}

    @classmethod
    def get_global_buffer_initializers(cls, attention_info) -> BufferInitializerDict:
        def _init_workspace(si: SequenceInfo) -> torch.Tensor:
            buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=si.device)
            _GlobalFlashInferPlanner.init_workspace(buffer)
            return buffer

        return {"workspace_buffer": _init_workspace}

    @classmethod
    def get_constants(cls, attention_info) -> List[Constant]:
        return [
            False,  # fuse_rope
            attention_info.rope_theta,  # rope_theta
            1.0,  # k_scale
            1.0,  # v_scale
        ]
