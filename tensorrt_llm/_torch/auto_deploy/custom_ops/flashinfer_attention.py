from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple, Union

import flashinfer
import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ...flashinfer_utils import get_env_enable_pdl
from ..utils.cuda_graph import cuda_graph_state
from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BufferInitializerDict,
    CacheConfig,
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
    sm_scale: Optional[float] = None

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
        # NOTE (lucaslie): flashinfer fa3 backend has accuracy issue + illegal memory access issues
        # on H100 PCIe, see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend="fa2",
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
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
                sm_scale=plan_params.sm_scale,
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
                    q_data_type=plan_params.q_dtype,
                    kv_data_type=plan_params.kv_dtype,
                    sm_scale=plan_params.sm_scale,
                )
            self.plan_params = plan_params

        # return desired wrapper
        return self.decode_wrapper if plan_params.is_generate else self.prefill_wrapper


_GlobalFlashInferPlanner = _FlashInferPlanner()


@torch.library.custom_op("auto_deploy::flashinfer_attention_prepare_metadata", mutates_args=())
def prepare_flashinfer_metadata(
    position_ids: torch.Tensor,
    batch_info: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
) -> List[torch.Tensor]:
    """Prepare metadata for flashinfer attention.

    Please refer to https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout and
    https://docs.flashinfer.ai/api/prefill.html#flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan
    to understand the convention.
    """
    # reset the planner
    _GlobalFlashInferPlanner.reset()

    # retrieve host-side metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    qo_indptr = cu_seqlen[: num_seq + 1]

    # NOTE: in theory we could easily precompute batch_indices. And positions is just position_ids
    # so we could skip that as well. However, we still need a place for resetting the planner and
    # for now we keep it here since the kernel is fast
    # Compute batch_indices and positions so that they can be reused for kv cache appends
    # for all the layers
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr, seq_len_with_cache[:num_seq], num_tokens
    )
    # return extra metadata
    return batch_indices, positions


@prepare_flashinfer_metadata.register_fake
def prepare_flashinfer_metadata_fake(
    position_ids: torch.Tensor,
    batch_info: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
):
    num_tokens = position_ids.shape[0] * position_ids.shape[1]
    return (
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),  # batch_indices
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),  # positions
    )


@torch.library.custom_op("auto_deploy::flashinfer_attention_mha_with_cache", mutates_args=())
def flashinfer_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    # EXTRA METADATA
    flashinfer_batch_indices: torch.Tensor,
    flashinfer_positions: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    # reshape to standard [b*s, n_heads, head_dim] layout
    head_dim = k_cache.shape[-1]
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    q = q.reshape(b * s, -1, head_dim)
    k = k.reshape(b * s, -1, head_dim)
    v = v.reshape(b * s, -1, head_dim)

    # convert to flashinfer-style metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info.tolist()
    num_seq = num_prefill + num_decode

    qo_indptr = cu_seqlen[: num_seq + 1]
    paged_kv_indptr = cu_num_pages[: num_seq + 1]

    # NOTE: it is okay to have cache_loc here without truncation. paged_kv_indptr will be
    # truncated and will point to the correct sub range of cache_loc.
    paged_kv_indices = cache_loc
    paged_kv_last_page_len = last_page_len[:num_seq]

    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]

    pp = PlanParams(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_seq=len(qo_indptr) - 1,
        is_generate=(s == 1),
        page_size=k_cache.shape[1],
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        sm_scale=scale,
    )

    # Assuming k_scale = v_scale = 1.0
    k_scale, v_scale = 1.0, 1.0
    # k = (k / k_scale).to(torch.float8_e4m3fn) if k_scale != 1.0, same for v
    if k_cache.dtype == torch.float8_e4m3fn:
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    flashinfer.page.append_paged_kv_cache(
        k,
        v,
        flashinfer_batch_indices,
        flashinfer_positions,
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

    y = wrapper.run(
        q, (k_cache, v_cache), k_scale=k_scale, v_scale=v_scale, enable_pdl=get_env_enable_pdl()
    )

    return y.view(q_shape_og)  # [b,s,n*h_d] or [b,s, n, h_d]


@flashinfer_mha_with_cache.register_fake
def flashinfer_mha_with_cache_fake(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    # EXTRA METADATA
    flashinfer_batch_indices: torch.Tensor,
    flashinfer_positions: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    workspace_buffer: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q.contiguous())


@AttentionRegistry.register("flashinfer")
class FlashInferAttention(AttentionDescriptor):
    @classmethod
    def _get_planner(cls) -> _FlashInferPlanner:
        return _GlobalFlashInferPlanner

    @classmethod
    def is_paged(cls):
        """Return if the attention op is paged or not."""
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_attention_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info", "cu_seqlen", "cu_num_pages", "cache_loc", "last_page_len"]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        return (torch.ops.auto_deploy.flashinfer_attention_prepare_metadata.default, 2, [])

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        # source op is [bsnd] layout already
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]

        def _get_cache(si: SequenceInfo):
            return torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                head_dim,
                device=si.device,
                dtype=cache_config.dtype or k_fake.dtype,
            )

        return {"k_cache": _get_cache, "v_cache": _get_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        def _init_workspace(si: SequenceInfo) -> torch.Tensor:
            # NOTE (lucaslie): avoid OOM for many cudagraphs,
            # see https://github.com/NVIDIA/TensorRT-LLM/pull/3686
            buffer = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device=si.device)
            cls._get_planner().init_workspace(buffer)
            return buffer

        return {"workspace_buffer": _init_workspace}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        # Sanity check: layout == "bsnd"
        # Prefer kwargs; fall back to the final positional arg if it's a string.
        layout = source_attn_node.kwargs.get("layout", None)
        if (
            layout is None
            and len(source_attn_node.args) > 0
            and isinstance(source_attn_node.args[-1], str)
        ):
            layout = source_attn_node.args[-1]
        if layout != "bsnd":
            raise RuntimeError(
                f"Expected torch_attention layout='bsnd' but got {layout!r} "
                f"for node: {source_attn_node.format_node()}"
            )

        # Double check other arguments
        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                "Unsupported attention arguments for "
                f"{source_attn_node=}: {attn_mask=}, {dropout_p=}, {is_causal=}"
            )

        # Get scale from args or kwargs
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        return [
            scale,  # softmax scale
            1.0,  # k_scale
            1.0,  # v_scale
        ]
