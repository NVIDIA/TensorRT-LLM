from typing import List, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

# Triton kernels
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined

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


@torch.library.custom_op("auto_deploy::triton_cached_ssm_transform", mutates_args={})
def _triton_cached_ssm_transform(
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
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    """Flattened cached SSM transform op that respects slot-indexed state caches.

    Implements generate-only (s==1) via selective_state_update and prefill/mixed (s>1) via
    mamba_chunk_scan_combined, updating the slot-indexed cache in-op. Returns y only.
    """
    b, s = hidden_states.shape[:2]
    num_seq = seq_len.shape[0]

    if s == 1:
        # Generate-only batch: gather cache slices for slots (already sanitized by metadata)
        slot_idx_long = slot_idx.to(torch.long)
        ssm_batch = ssm_state_cache.index_select(dim=0, index=slot_idx_long)

        # Shapes
        batch_size = b
        num_heads = hidden_states.shape[2]
        head_dim = hidden_states.shape[3]
        n_groups = B.shape[2]
        ssm_state_size = B.shape[3]

        # Prepare per-head, per-dim tensors
        dt_hp = dt[:, 0, :][:, :, None].expand(batch_size, num_heads, head_dim)
        dt_bias_hp = dt_bias[..., None].expand(num_heads, head_dim)
        dt_pre = torch.nn.functional.softplus(dt_hp + dt_bias_hp.to(dtype=dt_hp.dtype))
        dt_pre = torch.clamp(dt_pre, time_step_limit[0], time_step_limit[1])
        A_full = A[..., None, None].expand(num_heads, head_dim, ssm_state_size)
        D_full = D[..., None].expand(num_heads, head_dim)
        B_grouped = B.reshape(batch_size, n_groups, ssm_state_size)
        C_grouped = C.reshape(batch_size, n_groups, ssm_state_size)
        x = hidden_states.reshape(batch_size, num_heads, head_dim)

        # compute new state; avoid mutating input cache slice
        updated_state = ssm_batch.clone()
        # Provide a zero dt_bias tensor to satisfy kernel arg expansion; we've already
        # applied dt_bias and softplus/clamp into dt_pre above.
        dt_bias_zero = torch.zeros_like(dt_bias_hp)
        y_hp = selective_state_update(
            updated_state,
            x,
            dt_pre,
            A_full,
            B_grouped,
            C_grouped,
            D=D_full,
            z=None,
            dt_bias=dt_bias_zero,
            dt_softplus=False,
        )
        y = y_hp.reshape(batch_size, 1, num_heads, head_dim)

        # Scatter updated states back to global cache
        ssm_state_cache.index_copy_(0, slot_idx_long, updated_state.to(ssm_state_cache.dtype))

        return y.to(hidden_states.dtype)

    # Context/mixed phase (flattened sequences). Expect b == 1, but handle general b robustly.
    bs = b * s
    flat_idx = torch.arange(bs, device=hidden_states.device, dtype=torch.long)

    # NOTE: use reshape to force contiguous format after reshape
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])
    B_flat = B.reshape(bs, *B.shape[2:])
    C_flat = C.reshape(bs, *C.shape[2:])
    dt_flat = dt.reshape(bs, *dt.shape[2:])

    # NOTE: need contiguous format to process it sequentially
    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    y_flat = y.view(bs, *y.shape[2:])

    for i in range(num_seq):
        length_i = seq_len[i]
        if length_i.eq(0):
            continue

        start_i = seq_start[i]
        end_i = start_i + length_i

        mask_i = (flat_idx >= start_i.to(torch.long)) & (flat_idx < end_i.to(torch.long))
        idx_i = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)

        hs_seq = hs_flat.index_select(0, idx_i).unsqueeze(0)
        B_seq = B_flat.index_select(0, idx_i).unsqueeze(0)
        C_seq = C_flat.index_select(0, idx_i).unsqueeze(0)
        dt_seq = dt_flat.index_select(0, idx_i).unsqueeze(0)

        y_seq, ssm_state_i = mamba_chunk_scan_combined(
            hs_seq,
            dt_seq,
            A,
            B_seq,
            C_seq,
            chunk_size=chunk_size,
            D=D,
            z=None,
            dt_bias=dt_bias,
            seq_idx=None,
            dt_softplus=True,
            dt_limit=(time_step_limit[0], time_step_limit[1]),
            return_final_states=True,
        )

        y_flat.index_copy_(0, idx_i, y_seq[0].to(y_flat.dtype))

        slot_i = slot_idx[i].to(torch.long).unsqueeze(0)
        ssm_state_cache.index_copy_(0, slot_i, ssm_state_i.to(ssm_state_cache.dtype))

    return y


@_triton_cached_ssm_transform.register_fake
def _triton_cached_ssm_transform_fake(
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
        return torch.ops.auto_deploy.torch_ssm_transform

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_cached_ssm_transform

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        # Returns (seq_len, seq_start, slot_idx)
        return torch.ops.auto_deploy.torch_ssm_prepare_metadata, 3

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
