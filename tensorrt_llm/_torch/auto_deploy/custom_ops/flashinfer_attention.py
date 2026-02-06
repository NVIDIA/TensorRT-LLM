from dataclasses import dataclass, fields
from typing import Dict, List, Literal, Optional, Tuple, Union

import flashinfer
import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ....llmapi.llm_args import KvCacheConfig
from ...flashinfer_utils import get_env_enable_pdl
from ..utils.cuda_graph import cuda_graph_state
from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    KVPagedResourceHandler,
    MHACallable,
    PrepareMetadataCallable,
    PrepareMetadataHostCallable,
    ResourceHandlerDict,
)


@dataclass
class PlanParams:
    """Parameters that affect the flashinfer execution plan."""

    n_heads: int
    n_kv_heads: int
    head_dim: int
    num_seq: int
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
    cached_cuda_graph_decode_wrappers: Dict[
        PlanParams, flashinfer.BatchDecodeWithPagedKVCacheWrapper
    ]
    plan_params_prefill: Optional[PlanParams]
    plan_params_decode: Optional[PlanParams]
    kv_layout: Literal["NHD", "HND"] = "HND"

    def __init__(self):
        self.workspace_buffer = None
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.cached_cuda_graph_decode_wrappers = {}
        self.plan_params_prefill = None
        self.plan_params_decode = None

    def _init_decode_wrapper(
        self,
        use_cuda_graph: bool = False,
        indptr: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        last_page_len: Optional[torch.Tensor] = None,
    ):
        assert self.workspace_buffer is not None
        if use_cuda_graph:
            return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                self.kv_layout,
                use_cuda_graph=True,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last_page_len,
                use_tensor_cores=True,
                backend="fa2" if torch.cuda.get_device_capability(0) == (9, 0) else "auto",
            )
        else:
            return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                self.kv_layout,
                use_tensor_cores=True,
                backend="fa2" if torch.cuda.get_device_capability(0) == (9, 0) else "auto",
            )

    def reset(self, device: torch.device) -> None:
        self.plan_params_prefill = None
        self.plan_params_decode = None

        if isinstance(self.workspace_buffer, torch.Tensor):
            return

        self.__init__()  # reset all state

        # NOTE (lucaslie): avoid OOM for many cudagraphs,
        # see https://github.com/NVIDIA/TensorRT-LLM/pull/3686
        self.workspace_buffer = torch.empty(320 * 1024 * 1024, device=device, dtype=torch.uint8)

        # NOTE (lucaslie): flashinfer fa3 backend has accuracy issue + illegal memory access issues
        # on H100 PCIe, see https://github.com/NVIDIA/TensorRT-LLM/issues/4504
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            self.kv_layout,
            backend="fa2" if torch.cuda.get_device_capability(0) == (9, 0) else "auto",
        )
        self.decode_wrapper = self._init_decode_wrapper()

    def plan_generate_only(
        self,
        num_seq: int,
        cu_num_pages: torch.Tensor,
        cache_loc: torch.Tensor,
        last_page_len: torch.Tensor,
    ):
        for plan_params in self.cached_cuda_graph_decode_wrappers:
            if plan_params.num_seq == num_seq:
                wrapper = self.cached_cuda_graph_decode_wrappers[plan_params]
                flashinfer.decode.fast_decode_plan(
                    wrapper,
                    cu_num_pages,
                    cache_loc,
                    last_page_len,
                    plan_params.n_heads,
                    plan_params.n_kv_heads,
                    plan_params.head_dim,
                    plan_params.page_size,
                    q_data_type=plan_params.q_dtype,
                    kv_data_type=plan_params.kv_dtype,
                    sm_scale=plan_params.sm_scale,
                )

    def plan_prefill(
        self,
        qo_indptr_host: torch.Tensor,
        kv_page_indptr_host: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len_host: torch.Tensor,
        kv_lens_arr_host: torch.Tensor,
        plan_params: PlanParams,
    ) -> None:
        # check for re-planning
        if plan_params != self.plan_params_prefill:
            # plan prefill
            # NOTE (lucaslie): we use host versions here. the plan actually needs both (host+device)
            # version. Unfortunately, there is no good way to access the plan API and provide both
            # although we have both available. I have decided to use the host versions here to
            # ensure non-blocking invocation of plan, whereas the other way around would trigger a
            # blocking copy to cpu. This way we trigger a non-blocking copy to device (note that
            # this is safe since we do have pinned CPU memory for all our host-side arguments).
            self.prefill_wrapper.plan(
                qo_indptr_host,
                kv_page_indptr_host,
                kv_page_indices,
                kv_last_page_len_host,
                plan_params.n_heads,  # Q heads
                plan_params.n_kv_heads,  # KV heads
                plan_params.head_dim,
                plan_params.page_size,
                causal=plan_params.causal,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
                sm_scale=plan_params.sm_scale,
                seq_lens=kv_lens_arr_host,
            )
            self.plan_params_prefill = plan_params

        # return prefill wrapper
        return self.prefill_wrapper

    def plan_decode(
        self,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        plan_params: PlanParams,
    ) -> Union[
        flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        flashinfer.BatchDecodeWithPagedKVCacheWrapper,
    ]:
        # plan decode helper function
        def _plan_decode(
            wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        ):
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
        if (
            cuda_graph_state.in_warm_up()
            and plan_params not in self.cached_cuda_graph_decode_wrappers
        ):
            # During CUDA graph capture, the metadata tensors provided by auto-deploy are stable.
            wrapper = self._init_decode_wrapper(
                use_cuda_graph=True,
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_len,
            )
            self.cached_cuda_graph_decode_wrappers[plan_params] = wrapper
            _plan_decode(self.cached_cuda_graph_decode_wrappers[plan_params])
        # check if we are in cuda graph capture and just return the pre-cached decode wrapper
        if torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up():
            wrapper = self.cached_cuda_graph_decode_wrappers[plan_params]
            return wrapper

        # check for re-planning
        if plan_params != self.plan_params_decode:
            _plan_decode(self.decode_wrapper)
            self.plan_params_decode = plan_params

        # return decode wrapper
        return self.decode_wrapper


_GlobalFlashInferPlanner = _FlashInferPlanner()


@torch.library.custom_op("auto_deploy::flashinfer_attention_prepare_metadata", mutates_args=())
def prepare_flashinfer_metadata(
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
) -> List[torch.Tensor]:
    """Prepare metadata for flashinfer attention.

    Please refer to https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout and
    https://docs.flashinfer.ai/api/prefill.html#flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.plan
    to understand the convention.
    """
    # retrieve host-side metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode

    _GlobalFlashInferPlanner.reset(position_ids.device)

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
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
):
    num_tokens = position_ids.shape[0] * position_ids.shape[1]
    return (
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),  # batch_indices
        torch.empty(num_tokens, dtype=torch.int32, device=position_ids.device),  # positions
    )


def prepare_flashinfer_metadata_host(
    batch_info_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    last_page_len_host: torch.Tensor,
) -> None:
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()

    if num_prefill == 0:
        _GlobalFlashInferPlanner.plan_generate_only(
            num_decode,
            cu_num_pages_host[: num_decode + 1],
            cache_loc_host,
            last_page_len_host[:num_decode],
        )


@torch.library.custom_op("auto_deploy::flashinfer_attention_mha_with_cache", mutates_args=())
def flashinfer_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    # EXTRA METADATA
    flashinfer_batch_indices: torch.Tensor,
    flashinfer_positions: torch.Tensor,
    # CACHES - combined KV cache with shape [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
    kv_cache: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    # kv_cache shape: [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim] (HND layout)
    head_dim = kv_cache.shape[-1]
    page_size = kv_cache.shape[3]  # tokens_per_block
    q_shape_og = q.shape
    b, s = q_shape_og[:2]

    q = q.reshape(b * s, -1, head_dim).contiguous()
    k = k.reshape(b * s, -1, head_dim).contiguous()
    v = v.reshape(b * s, -1, head_dim).contiguous()

    # convert to flashinfer-style metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]

    # Assuming k_scale = v_scale = 1.0
    k_scale, v_scale = 1.0, 1.0
    # k = (k / k_scale).to(torch.float8_e4m3fn) if k_scale != 1.0, same for v
    if kv_cache.dtype == torch.float8_e4m3fn:
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    flashinfer.page.append_paged_kv_cache(
        append_key=k,
        append_value=v,
        batch_indices=flashinfer_batch_indices,
        positions=flashinfer_positions,
        paged_kv_cache=kv_cache,
        kv_indices=cache_loc,
        kv_indptr=cu_num_pages[: num_seq + 1],
        kv_last_page_len=last_page_len[:num_seq],
        kv_layout=_GlobalFlashInferPlanner.kv_layout,
    )

    # check if we need to re-combine outputs
    if num_prefill > 0 and num_decode > 0:
        y = torch.empty_like(q)
    else:
        y = None

    # now run split prefill, decode
    if num_prefill > 0:
        q_prefill = q[:num_prefill_tokens]

        pp_prefill = PlanParams(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_seq=num_prefill,
            page_size=page_size,
            q_dtype=q_prefill.dtype,
            kv_dtype=kv_cache.dtype,
            sm_scale=scale,
        )

        wrapper_prefill = _GlobalFlashInferPlanner.plan_prefill(
            qo_indptr_host=cu_seqlen_host[: num_prefill + 1],
            kv_page_indptr_host=cu_num_pages_host[: num_prefill + 1],
            kv_page_indices=cache_loc,
            kv_last_page_len_host=last_page_len_host[:num_prefill],
            kv_lens_arr_host=seq_len_with_cache_host[:num_prefill],
            plan_params=pp_prefill,
        )

        y_prefill = wrapper_prefill.run(
            q_prefill,
            kv_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            enable_pdl=get_env_enable_pdl(),
        )
        if y is not None:
            y[:num_prefill_tokens] = y_prefill
        else:
            y = y_prefill

    if num_decode > 0:
        q_decode = q[num_prefill_tokens:num_total_tokens]

        pp_decode = PlanParams(
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_seq=num_decode,
            page_size=page_size,
            q_dtype=q_decode.dtype,
            kv_dtype=kv_cache.dtype,
            sm_scale=scale,
        )

        # run the flashinfer planner and obtain the correct wrapper
        wrapper_decode = _GlobalFlashInferPlanner.plan_decode(
            kv_page_indptr=cu_num_pages[num_prefill : num_seq + 1],
            kv_page_indices=cache_loc,
            kv_last_page_len=last_page_len[num_prefill:num_seq],
            plan_params=pp_decode,
        )

        y_decode = wrapper_decode.run(
            q_decode,
            kv_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            enable_pdl=get_env_enable_pdl(),
        )
        if y is not None:
            y[num_prefill_tokens:num_total_tokens] = y_decode
        else:
            y = y_decode

    return y.view(q_shape_og)  # [b,s,n*h_d] or [b,s, n, h_d]


@flashinfer_mha_with_cache.register_fake
def flashinfer_mha_with_cache_fake(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    # EXTRA METADATA
    flashinfer_batch_indices: torch.Tensor,
    flashinfer_positions: torch.Tensor,
    # CACHES - combined KV cache
    kv_cache: torch.Tensor,
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
        return [
            "batch_info_host",
            "cu_seqlen_host",
            "cu_num_pages",
            "cu_num_pages_host",
            "cache_loc",
            "last_page_len",
            "last_page_len_host",
            "seq_len_with_cache_host",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        return (torch.ops.auto_deploy.flashinfer_attention_prepare_metadata.default, 2, [])

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        # source op is [bsnd] layout already
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]

        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads,
                head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype),
                kv_factor=2,
                kv_layout=_GlobalFlashInferPlanner.kv_layout,
            )
        }

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        return prepare_flashinfer_metadata_host

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
