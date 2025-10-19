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
    b, s = hidden_states.shape[:2]
    num_seq = seq_len.shape[0]
    # Flatten tokens for indexing/scatter
    bs = b * s
    device = hidden_states.device
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])  # [bs, H, D]
    B_flat = B.reshape(bs, *B.shape[2:])  # [bs, G, N]
    C_flat = C.reshape(bs, *C.shape[2:])  # [bs, G, N]
    dt_flat = dt.reshape(bs, dt.shape[2])  # [bs, H]

    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    y_flat = y.view(bs, *y.shape[2:])

    num_heads = hidden_states.shape[2]
    head_dim = hidden_states.shape[3]
    ssm_state_size = B.shape[3]

    if s == 1:
        num_prefill = 0
        num_decode = num_seq
    else:
        prefill_mask = seq_len > 1
        num_prefill = int(prefill_mask.sum().item())
        num_decode = num_seq - num_prefill

    # Prefill: concatenate tokens at the front and run combined scan
    if num_prefill > 0:
        seq_len_prefill = seq_len[:num_prefill].to(torch.int32)
        total_prefill_tokens = int(seq_len_prefill.sum().item())
        prefill_idx = torch.arange(total_prefill_tokens, device=device, dtype=torch.long)

        hs_prefill = hs_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, H, D]
        B_prefill = B_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, G, N]
        C_prefill = C_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, G, N]
        dt_prefill = dt_flat.index_select(0, prefill_idx).unsqueeze(0)  # [1, S_p, H]

        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(seq_len_prefill, dim=0),
            ],
            dim=0,
        )
        seq_ids = torch.arange(num_prefill, device=device, dtype=torch.int32)
        seq_idx_prefill = torch.repeat_interleave(seq_ids, seq_len_prefill).view(1, -1)

        initial_states = chunk_indices = chunk_offsets = None
        if torch.any(use_initial_states[:num_prefill]):
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                ssm_state_cache[slot_idx[:num_prefill]],
                0,
            )
            chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets(
                cu_seqlens, chunk_size
            )
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
        )

        y_flat.index_copy_(0, prefill_idx, y_prefill[0].to(y_flat.dtype))
        ssm_state_cache.index_copy_(
            0, slot_idx[:num_prefill].to(torch.long), varlen_states.to(ssm_state_cache.dtype)
        )

    # Decode: batch single-token updates via selective_state_update
    if num_decode > 0:
        decode_idx = seq_start[num_prefill:].to(torch.long)
        slot_idx_decode = slot_idx[num_prefill:].to(torch.long)

        x_decode = hs_flat.index_select(0, decode_idx)  # [nd, H, D]
        B_decode = B_flat.index_select(0, decode_idx)  # [nd, G, N]
        C_decode = C_flat.index_select(0, decode_idx)  # [nd, G, N]
        dt_decode = dt_flat.index_select(0, decode_idx)  # [nd, H]

        dt_hp = dt_decode[:, :, None].expand(-1, num_heads, head_dim)
        dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
        dt_pre = torch.nn.functional.softplus(dt_hp + dt_bias_hp.to(dtype=dt_hp.dtype))
        dt_pre = torch.clamp(dt_pre, time_step_limit[0], time_step_limit[1])
        A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
        D_full = D[..., None].expand(num_heads, head_dim)

        dt_bias_zero = torch.zeros_like(dt_bias_hp)
        y_dec = selective_state_update(
            ssm_state_cache,
            x_decode,
            dt_pre,
            A_full,
            B_decode,
            C_decode,
            D=D_full,
            z=None,
            dt_bias=dt_bias_zero,
            dt_softplus=False,
            state_batch_indices=slot_idx_decode,
        )  # [nd, H, D]

        y_flat.index_copy_(0, decode_idx, y_dec.to(y_flat.dtype))

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


## Note: we reuse the existing metadata custom op and its registered fake from torch backend.


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
        # Returns (seq_len, seq_start, slot_idx, use_initial_states)
        return torch.ops.auto_deploy.torch_ssm_prepare_metadata, 4

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

        def _get_ssm_cache(si: SequenceInfo):
            return torch.empty(
                si.max_batch_size,
                num_heads,
                head_dim,
                ssm_state_size,
                device=si.device,
                dtype=cache_config.dtype or hs_fake.dtype,
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
