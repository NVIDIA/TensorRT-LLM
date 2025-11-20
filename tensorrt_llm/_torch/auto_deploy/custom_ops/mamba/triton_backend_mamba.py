from typing import List, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

# Triton kernels
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import cu_seqlens_to_chunk_indices_offsets
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

from ...utils.node_utils import extract_op_args
from ..attention_interface import (
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


@torch.library.custom_op("auto_deploy::triton_ssm_prepare_metadata", mutates_args=())
def _triton_ssm_prepare_metadata(
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
    chunk_size: int,
) -> List[torch.Tensor]:
    """Prepare metadata for cached SSM transform.

    Returns a tuple of (seq_len_sanitized, seq_start, slot_idx_sanitized).
    """
    # Determine number of active sequences and compute seq_start boundaries
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)

    seq_start = torch.zeros_like(seq_len_sanitized)
    if num_seq > 1:
        seq_start[1:] = torch.cumsum(seq_len_sanitized[:-1], 0)

    # Truncate slot indices to match active sequences
    slot_idx_sanitized = slot_idx[:num_seq].clone().to(torch.long)
    # TODO(https://github.com/NVIDIA/TensorRT-LLM/issues/8170): update torch
    # reference implementation to support chunked prefill.
    use_initial_states = input_pos[:num_seq] > 0

    device = position_ids.device

    chunk_indices = torch.zeros(num_seq, dtype=torch.int32, device=device)
    chunk_offsets = torch.zeros(num_seq, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(num_seq + 1, dtype=torch.int32, device=device)
    _, s = position_ids.shape[:2]
    if s > 1:
        # only compute chunk indices and offsets for prefill.
        prefill_mask = seq_len_sanitized > 1
        num_prefill = int(prefill_mask.sum().item())
        num_prefill_tokens = int(seq_len_sanitized[:num_prefill].sum().item())
        num_decode = num_seq - num_prefill
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(seq_len_sanitized[:num_prefill].to(torch.int32), dim=0),
            ],
            dim=0,
        )
        chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets(cu_seqlens, chunk_size)
        seq_idx_prefill = torch.repeat_interleave(
            torch.arange(num_prefill, device=device, dtype=torch.int32),
            seq_len_sanitized[:num_prefill],
        ).view(1, -1)
    else:
        num_prefill = 0
        num_prefill_tokens = 0
        num_decode = num_seq
        seq_idx_prefill = torch.empty(1, 0, dtype=torch.int32, device=device)
    batch_info_tensor = torch.tensor(
        [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32
    )  # host tensor

    return (
        seq_len_sanitized,
        seq_start,
        slot_idx_sanitized,
        use_initial_states,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        batch_info_tensor,
    )


@_triton_ssm_prepare_metadata.register_fake
def _triton_ssm_prepare_metadata_fake(
    position_ids, seq_len, input_pos, cache_loc, pages_per_seq, slot_idx, page_size, chunk_size
):
    # Use the same sanitization logic to determine sizes in fake mode
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)
    device = slot_idx.device
    # Always-correct shapes
    seq_len_fake = torch.empty_like(seq_len_sanitized)
    seq_start_fake = torch.empty_like(seq_len_sanitized)
    slot_idx_fake = torch.empty(num_seq, dtype=torch.long, device=device)
    use_initial_states_fake = torch.empty(num_seq, dtype=torch.bool, device=device)
    cu_seqlens_fake = torch.empty(num_seq + 1, dtype=torch.int32, device=device)

    # Token-dependent shapes (prefill vs decode)
    _, s = position_ids.shape[:2]
    if s > 1:
        prefill_mask = seq_len_sanitized > 1
        num_prefill = int(prefill_mask.sum().item())
        num_prefill_tokens = int(seq_len_sanitized[:num_prefill].sum().item())
        cu_seqlens_runtime = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(seq_len_sanitized[:num_prefill].to(torch.int32), dim=0),
            ],
            dim=0,
        )
        chunk_indices_rt, chunk_offsets_rt = cu_seqlens_to_chunk_indices_offsets(
            cu_seqlens_runtime, chunk_size
        )
        chunk_indices_fake = torch.empty_like(chunk_indices_rt)
        chunk_offsets_fake = torch.empty_like(chunk_offsets_rt)
        seq_idx_prefill_fake = torch.empty(1, num_prefill_tokens, dtype=torch.int32, device=device)
    else:
        chunk_indices_fake = torch.empty(0, dtype=torch.int32, device=device)
        chunk_offsets_fake = torch.empty(0, dtype=torch.int32, device=device)
        seq_idx_prefill_fake = torch.empty(1, 0, dtype=torch.int32, device=device)

    batch_info_tensor_fake = torch.empty(3, dtype=torch.int32)

    return (
        seq_len_fake,
        seq_start_fake,
        slot_idx_fake,
        use_initial_states_fake,
        cu_seqlens_fake,
        chunk_indices_fake,
        chunk_offsets_fake,
        seq_idx_prefill_fake,
        batch_info_tensor_fake,
    )


@torch.library.custom_op("auto_deploy::triton_cached_ssm", mutates_args={})
def _triton_cached_ssm(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # METADATA
    seq_len: torch.Tensor,  # [num_seq]
    seq_start: torch.Tensor,  # [num_seq]
    slot_idx: torch.Tensor,  # [num_seq]
    use_initial_states: torch.Tensor,  # [num_seq]
    cu_seqlens: torch.Tensor,  # [num_seq + 1]
    chunk_indices: torch.Tensor,  # [num_seq + 1]
    chunk_offsets: torch.Tensor,  # [num_seq + 1]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill]
    batch_info_tensor: torch.Tensor,  # [3]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    """Flattened cached SSM transform op that respects slot-indexed state caches.

    Split mixed batches into prefill (seq_len>1) and decode (seq_len==1):
    - Prefill: run one varlen combined scan over concatenated prefill tokens and update final states per slot.
    - Decode: batch single-token updates with selective_state_update and update states per slot.
    """
    b, s, num_heads, head_dim = hidden_states.shape
    # Flatten tokens for indexing/scatter
    bs = b * s
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])  # [bs, H, D]
    B_flat = B.reshape(bs, *B.shape[2:])  # [bs, G, N]
    C_flat = C.reshape(bs, *C.shape[2:])  # [bs, G, N]
    dt_flat = dt.reshape(bs, dt.shape[2])  # [bs, H]

    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    y_flat = y.view(bs, *y.shape[2:])

    ssm_state_size = B.shape[3]

    num_prefill, num_prefill_tokens, num_decode = batch_info_tensor.tolist()

    # Prefill: concatenate tokens at the front and run combined scan
    if num_prefill > 0:
        hs_prefill = hs_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, H, D]
        B_prefill = B_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, G, N]
        C_prefill = C_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, G, N]
        dt_prefill = dt_flat[:num_prefill_tokens].unsqueeze(0)  # [1, S_p, H]

        initial_states = None
        if torch.any(use_initial_states[:num_prefill]):
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                ssm_state_cache[slot_idx[:num_prefill]],
                0,
            )
        else:
            chunk_indices = None
            chunk_offsets = None

        y_prefill, varlen_states = mamba_chunk_scan_combined(
            hs_prefill,
            dt_prefill,
            A,
            B_prefill,
            C_prefill,
            chunk_size=chunk_size,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx_prefill,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=True,
            dt_limit=(time_step_limit[0], time_step_limit[1]),
            return_final_states=False,
            return_varlen_states=True,
            mamba_ssm_cache_dtype=ssm_state_cache.dtype,
        )

        y_flat[:num_prefill_tokens] = y_prefill[0].to(y_flat.dtype)
        ssm_state_cache.index_copy_(
            0, slot_idx[:num_prefill], varlen_states.to(ssm_state_cache.dtype)
        )

    # Decode: batch single-token updates via selective_state_update
    if num_decode > 0:
        slot_idx_decode = slot_idx[num_prefill:]

        x_decode = hs_flat[num_prefill_tokens : num_prefill_tokens + num_decode]  # [nd, H, D]
        B_decode = B_flat[num_prefill_tokens : num_prefill_tokens + num_decode]  # [nd, G, N]
        C_decode = C_flat[num_prefill_tokens : num_prefill_tokens + num_decode]  # [nd, G, N]
        dt_decode = dt_flat[num_prefill_tokens : num_prefill_tokens + num_decode]  # [nd, H]

        dt_hp = dt_decode[:, :, None].expand(-1, num_heads, head_dim)
        dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
        A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
        D_full = D[..., None].expand(num_heads, head_dim)

        y_dec = selective_state_update(
            ssm_state_cache,
            x_decode,
            dt_hp,
            A_full,
            B_decode,
            C_decode,
            D=D_full,
            z=None,
            dt_bias=dt_bias_hp,
            dt_softplus=True,
            state_batch_indices=slot_idx_decode,
        )  # [nd, H, D]

        y_flat[num_prefill_tokens : num_prefill_tokens + num_decode].copy_(y_dec.to(y_flat.dtype))

    return y


@_triton_cached_ssm.register_fake
def _triton_cached_ssm_fake(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # METADATA
    seq_len: torch.Tensor,  # [num_seq]
    seq_start: torch.Tensor,  # [num_seq]
    slot_idx: torch.Tensor,  # [num_seq]
    use_initial_states: torch.Tensor,  # [num_seq]
    cu_seqlens: torch.Tensor,  # [num_seq + 1]
    chunk_indices: torch.Tensor,  # [num_seq + 1]
    chunk_offsets: torch.Tensor,  # [num_seq + 1]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill]
    batch_info_tensor: torch.Tensor,  # [3]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
):
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


# TODO: consider inheriting from TorchBackendSSM instead of redefining everything
@AttentionRegistry.register("triton_ssm")
class TritonBackendSSM(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        return True

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        # Hidden states follow [b, s, n, d]
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # torch_ssm_transform signature has 7 node/state arguments
        return 7

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        # Keep source op unchanged (used for uncached pre-export)
        return torch.ops.auto_deploy.torch_ssm

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_cached_ssm

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        # Returns: seq_len, seq_start, slot_idx, use_initial_states,
        # cu_seqlens, chunk_indices, chunk_offsets, seq_idx_prefill, batch_info_tensor
        return torch.ops.auto_deploy.triton_ssm_prepare_metadata, 9

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        # Shapes from fake tensors
        hs_fake: torch.Tensor = source_attn_node.args[0].meta["val"]
        B_fake: torch.Tensor = source_attn_node.args[2].meta["val"]

        num_heads = hs_fake.shape[-2]
        head_dim = hs_fake.shape[-1]

        if B_fake.ndim >= 4:
            ssm_state_size = B_fake.shape[-1]
        else:
            ssm_state_size = max(1, B_fake.shape[-1])

        # extract ssm_state_dtype from cache_config or hs_fake
        ssm_state_dtype = cache_config.mamba_dtype or hs_fake.dtype

        def _get_ssm_cache(si: SequenceInfo):
            return torch.empty(
                si.max_batch_size,
                num_heads,
                head_dim,
                ssm_state_size,
                device=si.device,
                dtype=ssm_state_dtype,
            )

        return {"ssm_state_cache": _get_ssm_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        time_step_limit, chunk_size = extract_op_args(
            source_attn_node, "time_step_limit", "chunk_size"
        )
        return [time_step_limit, chunk_size]
