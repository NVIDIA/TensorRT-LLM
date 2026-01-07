from typing import List, Tuple

import flashinfer
import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

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


@torch.library.custom_op("auto_deploy::flashinfer_mla_prepare_metadata", mutates_args=())
def flashinfer_mla_prepare_metadata(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
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
    # _GlobalFlashInferPlanner.reset()

    # retrieve sanitized metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]

    # prepare flashinfer-style metadata
    offsets = input_pos[:num_seq].clone()

    qo_indptr = torch.zeros(num_seq + 1, dtype=torch.int, device=seq_len.device)
    qo_indptr[1:] = torch.cumsum(seq_len, 0)

    paged_kv_indptr = torch.zeros_like(qo_indptr)
    paged_kv_indptr[1:] = torch.cumsum(pages_per_seq[:num_seq], 0)

    paged_kv_indices = torch.arange(
        sum(pages_per_seq[:num_seq]), dtype=torch.int32, device=seq_len.device
    )

    paged_kv_last_page_len = ((offsets + seq_len - 1) % page_size) + 1

    # Compute batch_indices and positions so that they can be reused for kv cache appends
    # for all the layers
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, page_size),
        position_ids.numel(),
    )

    # kv_len_arr = flashinfer.get_seq_lens(paged_kv_indptr, paged_kv_last_page_len, page_size)
    kv_len_arr = seq_len.clone()

    return (
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        batch_indices,
        positions,
        kv_len_arr,
        torch.tensor(page_size, dtype=torch.int32),
        position_ids.clone(),
    )


# TODO: Move the truncation of seq_len out of this custom op
@flashinfer_mla_prepare_metadata.register_fake
def flashinfer_mla_prepare_metadata_fake(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    page_size: int,
):
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    qo_indptr = torch.empty(num_seq + 1, dtype=seq_len.dtype, device=seq_len.device)
    scalar = torch.empty(1, dtype=torch.int32, device=seq_len.device)
    batch_indices = torch.empty_like(input_ids).flatten()
    positions = torch.empty_like(input_ids).flatten()
    return (
        torch.empty_like(qo_indptr),  # qo_indptr
        torch.empty_like(qo_indptr),  # paged_kv_indptr
        torch.empty_like(seq_len),  # paged_kv_indices
        torch.empty_like(seq_len),  # paged_kv_last_page_len
        batch_indices,  # batch_indices
        positions,  # positions
        torch.empty_like(seq_len),  # kv_len_arr
        scalar,  # page_size
        torch.empty_like(position_ids),  # position_ids
    )


@AttentionRegistry.register("FlashinferMLABackend")
class FlashinferMLABackend(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        # (q_normed_dn, compressed_kv, k_pe), (sin_cache, cos_cache), (wkv_b, wq_b, w_uq_ukv, wo_proj, w_uv_o)
        return 10

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        # return torch.ops.auto_deploy.torch_attention_deepseek_fused_mla
        return torch.ops.auto_deploy.torch_deepseek_mla_no_cache

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        # return torch.ops.auto_deploy.triton_attention_fused_flattened_mla_with_cache
        # return torch.ops.auto_deploy.torch_deepseek_decode_only_absorb_attn
        return torch.ops.auto_deploy.flashinfer_deepseek_mla_with_kv_cache

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.auto_deploy.flashinfer_mla_prepare_metadata, 9

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        kv_fake = source_attn_node.args[2].meta["val"]
        head_dim_kpe = 64
        head_dim_ckv = 512  # AKA kv_lora_rank

        def _create_paged_cache(
            si: SequenceInfo,
            head_dim: int,
            dtype: torch.dtype,
            init_randn: bool = False,
        ):
            """
            The compressed KV and K positional encoding caches (CKV and KPE) with paged layout.
            Note that there is only one (shared) head.
            """

            tensor_init = torch.randn if init_randn else torch.zeros
            return tensor_init(
                si.num_pages,
                si.page_size,
                head_dim,
                device=si.device,
                dtype=dtype,
            )

        def _create_ckv_cache(si: SequenceInfo):
            return _create_paged_cache(
                si,
                head_dim_ckv,
                dtype=cache_config.dtype or kv_fake.dtype,
                init_randn=False,
            )

        def _create_kpe_cache(si: SequenceInfo):
            return _create_paged_cache(
                si,
                head_dim_kpe,
                dtype=cache_config.dtype or kv_fake.dtype,
                init_randn=False,
            )

        return {"ckv_cache": _create_ckv_cache, "kpe_cache": _create_kpe_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        softmax_scale = source_attn_node.args[-1]
        return [
            softmax_scale,
        ]
