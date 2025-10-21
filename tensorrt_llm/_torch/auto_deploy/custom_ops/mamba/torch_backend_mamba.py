"""Custom op collection for cached mamba2 ssm transform (linear attention) in pure PyTorch.

This file contains two kinds of functionality:
1) Low-level cached SSM ops (decode + prefill) that operate on dense [batch, seq_len] inputs.
2) AttentionDescriptor-compatible metadata/op wrappers that handle flattened sequences and slot
   indexed SSM state caches per the auto_deploy attention interface.
"""

from typing import List, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

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
from .torch_mamba import _torch_ssm_prefill


def _torch_cached_ssm_decode(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    chunk_size: int,
    ssm_state_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    # retrieve some shape information
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    n_groups, ssm_state_size = B.shape[2:]

    # Note: there is no need to pad parameter matrices here, as there is just one new token
    # for batched generation
    dt = dt[:, 0, :][:, None, ...]
    dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], head_dim)
    # [num_heads] -> [num_heads, head_dim]
    dt_bias = dt_bias[..., None].expand(dt_bias.shape[0], head_dim)

    dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])
    A = A[..., None, None].expand(num_heads, head_dim, ssm_state_size).to(dtype=torch.float32)
    # [bsz, num_heads, head_dim, state_size]
    dA = torch.exp(dt[..., None] * A)

    # Discretize B
    # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
    # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
    B = B.reshape(batch_size, n_groups, -1)[..., None, :]
    B = B.expand(batch_size, n_groups, num_heads // n_groups, B.shape[-1]).contiguous()
    B = B.reshape(batch_size, -1, B.shape[-1])
    # [bsz, num_heads, head_dim, state_size]
    dB = dt[..., None] * B[..., None, :]

    # Discretize x into dB
    # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
    hidden_states = hidden_states.reshape(batch_size, -1, head_dim)
    dBx = dB * hidden_states[..., None]

    # State calculation
    updated_ssm_state = ssm_state_cache * dA + dBx

    # Subsequent output
    # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
    C = C.reshape(batch_size, n_groups, -1)[..., None, :]
    C = C.expand(batch_size, n_groups, num_heads // n_groups, C.shape[-1]).contiguous()
    C = C.reshape(batch_size, -1, C.shape[-1])
    # [bsz, num_heads, head_dim]

    ssm_states = updated_ssm_state.to(dtype=C.dtype)  # Shape: [b, h, d, n]
    # Reshape ssm_states to merge the first two dimensions
    ssm_states_reshaped = ssm_states.view(
        batch_size * num_heads, head_dim, ssm_state_size
    )  # Shape: [b*h, d, n]
    C_reshaped = C.view(batch_size * num_heads, ssm_state_size, 1)  # Shape: [b*h, n, 1]
    y = torch.bmm(ssm_states_reshaped, C_reshaped)
    y = y.view(batch_size, num_heads, head_dim)

    # D skip connection
    # [num_heads] -> [num_heads, head_dim]
    D = D[..., None].expand(D.shape[0], head_dim)
    y = (y + hidden_states * D).to(y.dtype)

    # [bsz, num_heads, head_dim] -> [bsz, 1, num_heads, head_dim]
    y = y.reshape(batch_size, 1, num_heads, head_dim)
    return y, updated_ssm_state


def _update_ssm_state_cache(ssm_cache: torch.Tensor, ssm_state: torch.Tensor) -> None:
    ssm_cache.copy_(ssm_state)


# ---------------------------------------------------------------
# Metadata + flattened cached op that integrates with the AD i/f
# ---------------------------------------------------------------


@torch.library.custom_op("auto_deploy::torch_ssm_prepare_metadata", mutates_args=())
def _torch_ssm_prepare_metadata(
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
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
    use_initial_states = input_pos > 0
    return (seq_len_sanitized, seq_start, slot_idx_sanitized, use_initial_states)


@_torch_ssm_prepare_metadata.register_fake
def _torch_ssm_prepare_metadata_fake(
    position_ids, seq_len, input_pos, cache_loc, pages_per_seq, slot_idx, page_size
):
    # Use the same sanitization logic to determine sizes in fake mode
    seq_len_sanitized = SequenceInfo._get_sanitized_seq_len(position_ids, seq_len)
    num_seq = len(seq_len_sanitized)
    return (
        torch.empty_like(seq_len_sanitized),
        torch.empty_like(seq_len_sanitized),
        torch.empty(num_seq, dtype=torch.long, device=slot_idx.device),
        torch.empty(num_seq, dtype=torch.bool, device=slot_idx.device),
    )


@torch.library.custom_op("auto_deploy::torch_cached_ssm", mutates_args={})
def _torch_cached_ssm(
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

    This op supports two layouts from the attention interface:
    - Generate-only: hidden_states is [b, 1, H, D]. We'll gather caches using slot_idx[:b].
    - Flattened context/mixed: hidden_states is [1, total_s, H, D] and seq_len/seq_start
      describe per-sequence segments. We'll process each segment and scatter final states to caches.
    """
    b, s = hidden_states.shape[:2]
    num_seq = seq_len.shape[0]

    if s == 1:
        # Generate-only batch: gather cache slices for slots (already sanitized by metadata)
        slot_idx_long = slot_idx.to(torch.long)
        ssm_batch = ssm_state_cache.index_select(dim=0, index=slot_idx_long)

        y, updated_state = _torch_cached_ssm_decode(
            hidden_states,
            A,
            B,
            C,
            D,
            dt,
            dt_bias,
            time_step_limit,
            chunk_size,
            ssm_batch,
        )

        # Scatter updated states back to global cache
        ssm_state_cache.index_copy_(0, slot_idx_long, updated_state.to(ssm_state_cache.dtype))

        # return in the same dtype as the input
        return y.to(hidden_states.dtype)

    # Prefill
    if any(use_initial_states):
        # TODO(https://github.com/NVIDIA/TensorRT-LLM/issues/8170): update torch
        # reference implementation to support chunked prefill.
        raise ValueError(
            "torch mamba backend does not yet support chunked prefill "
            "and can not correctly handle initial states."
        )
    # Context/mixed phase (flattened sequences). Expect b == 1, but handle general b robustly.
    # We'll iterate over sequences defined by (seq_len, seq_start) and update state per slot.
    # Process across the flattened second dimension.
    # Precompute a device index for flattened positions
    bs = b * s
    flat_idx = torch.arange(bs, device=hidden_states.device, dtype=torch.long)

    # NOTE: use reshape to force contiguous format after reshape, needed to process it sequentially
    hs_flat = hidden_states.reshape(bs, *hidden_states.shape[2:])
    B_flat = B.reshape(bs, *B.shape[2:])
    C_flat = C.reshape(bs, *C.shape[2:])
    dt_flat = dt.reshape(bs, *dt.shape[2:])

    # NOTE: need contiguous format to process it sequentially
    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    y_flat = y.view(bs, *y.shape[2:])

    for i in range(num_seq):
        length_i = seq_len[i]
        # Skip empty sequences without synchronizing to host
        if length_i.eq(0):
            continue

        start_i = seq_start[i]
        end_i = start_i + length_i

        # Build device indices for this sequence's token range
        mask_i = (flat_idx >= start_i.to(torch.long)) & (flat_idx < end_i.to(torch.long))
        idx_i = torch.nonzero(mask_i, as_tuple=False).squeeze(-1)

        # Gather per-sequence views
        hs_seq = hs_flat.index_select(0, idx_i).unsqueeze(0)
        B_seq = B_flat.index_select(0, idx_i).unsqueeze(0)
        C_seq = C_flat.index_select(0, idx_i).unsqueeze(0)
        dt_seq = dt_flat.index_select(0, idx_i).unsqueeze(0)

        # Run prefill and obtain final SSM state for this sequence
        y_seq, ssm_state_i = _torch_ssm_prefill(
            hs_seq, A, B_seq, C_seq, D, dt_seq, dt_bias, time_step_limit, chunk_size
        )

        # Write outputs back using device indices
        y_flat.index_copy_(0, idx_i, y_seq[0].to(y_flat.dtype))

        # Scatter the final state to the slot-indexed cache using device index
        slot_i = slot_idx[i].to(torch.long).unsqueeze(0)
        ssm_state_cache.index_copy_(0, slot_i, ssm_state_i.to(ssm_state_cache.dtype))

    return y


@_torch_cached_ssm.register_fake
def _torch_cached_ssm_fake(
    # INPUTS
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # CACHES
    ssm_state_cache: torch.Tensor,
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
):
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=torch.float32,
    )


@AttentionRegistry.register("torch_ssm")
class TorchBackendSSM(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        # TODO: we should refine our notion of "is_paged" --> seems counterintuitive for ssm now
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
        return torch.ops.auto_deploy.torch_ssm

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.torch_cached_ssm

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        # Returns (seq_len, seq_start, slot_idx)
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

        # Infer state size by assuming B has shape [b, s, n_groups * ssm_state_size]
        # During runtime we pass [b, s, n_groups, ssm_state_size]; both give the same last dim product.
        if B_fake.ndim >= 4:
            ssm_state_size = B_fake.shape[-1]
        else:
            # Fallback: assume last dim is n_groups * state_size and choose a minimal positive size
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
        # time_step_limit, chunk_size should be extracted and passed in as constants
        time_step_limit, chunk_size = extract_op_args(
            source_attn_node, "time_step_limit", "chunk_size"
        )
        return [time_step_limit, chunk_size]
