"""Frozen Hopper kernel recipe."""

import cutlass

from .fwd import SolAttnSplitForwardSm90
from .mainloop import SolAttnMainloopSm90


def make_kernel(tokens: int, kv_splits: int):
    blocks = (tokens + 63) // 64
    full_groups, tail = divmod(blocks, 64)
    has_full_groups = tail == 0
    has_full_blocks = tokens % 64 == 0
    kernel = SolAttnMainloopSm90 if kv_splits == 1 else SolAttnSplitForwardSm90

    return kernel(
        cutlass.BFloat16,
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=1,
        is_causal=False,
        is_local=False,
        pack_gqa=False,
        tile_m=64,
        tile_n=64,
        num_stages=1,
        num_threads=128,
        sol_attn_assume_lane_group_route_reduce=(
            has_full_blocks and has_full_groups
        ),
        sol_attn_assume_full_k_exact_blocks=has_full_blocks,
        sol_attn_tail_exact_words1=0 < tail <= 8,
        sol_attn_assume_full_route_groups=has_full_groups,
        sol_attn_static_num_full_route_groups=(
            -1 if has_full_groups else full_groups
        ),
        sol_attn_static_tail_valid_count=(-1 if has_full_groups else tail),
        sol_attn_tail_physical_tile16=0 < tail <= 16,
        sol_attn_exact_mask_seqlen_last_only=(
            not has_full_blocks
        ),
        sol_attn_tail16_lane_group_route_reduce=tail == 16,
        sol_attn_num_splits=kv_splits,
    )


__all__ = ["make_kernel"]
