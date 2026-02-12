"""Inplace variants of dynamic custom ops for piecewise CUDA graph support.

For piecewise CUDA graphs, each static segment writes its output to a fixed memory
address. The next segment reads from that same address. To ensure address stability,
dynamic ops (attention, SSM, etc.) need inplace variants that write into a
pre-allocated output buffer instead of allocating a new one.

This module registers inplace variants for all dynamic cached ops. Each inplace
variant takes an extra `output` parameter, calls the original op, and copies the
result into `output`.

The graph transform in `piecewise_utils.py` swaps original ops → inplace ops
before graph splitting.

Mapping:
  Original Op                                     → Inplace Variant
  ────────────────────────────────────────────────────────────────────
  flashinfer_attention_mha_with_cache              → flashinfer_attention_mha_with_cache_inplace
  triton_attention_flattened_mha_with_cache         → triton_attention_flattened_mha_with_cache_inplace
  torch_cached_attention_with_cache                → torch_cached_attention_with_cache_inplace
  triton_cached_ssm                                → triton_cached_ssm_inplace
  torch_cached_ssm                                 → torch_cached_ssm_inplace
  flashinfer_cached_ssm                            → flashinfer_cached_ssm_inplace
  fla_cached_delta_rule                            → fla_cached_delta_rule_inplace
"""

from typing import Dict, List, Optional

import torch

# =============================================================================
# ATTENTION INPLACE OPS
# =============================================================================


# ── FlashInfer Attention ──
@torch.library.custom_op(
    "auto_deploy::flashinfer_attention_mha_with_cache_inplace", mutates_args=("output",)
)
def flashinfer_attention_mha_with_cache_inplace(
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
    # CACHES
    kv_cache: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
    k_scale: float,
    v_scale: float,
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.flashinfer_attention_mha_with_cache(
        q,
        k,
        v,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        flashinfer_batch_indices,
        flashinfer_positions,
        kv_cache,
        scale,
        k_scale,
        v_scale,
    )
    output.copy_(result)


@flashinfer_attention_mha_with_cache_inplace.register_fake
def flashinfer_attention_mha_with_cache_inplace_fake(
    q,
    k,
    v,
    batch_info_host,
    cu_seqlen_host,
    cu_num_pages,
    cu_num_pages_host,
    cache_loc,
    last_page_len,
    last_page_len_host,
    seq_len_with_cache_host,
    flashinfer_batch_indices,
    flashinfer_positions,
    kv_cache,
    scale,
    k_scale,
    v_scale,
    output,
) -> None:
    pass


# ── Triton Attention ──
@torch.library.custom_op(
    "auto_deploy::triton_attention_flattened_mha_with_cache_inplace", mutates_args=("output",)
)
def triton_attention_flattened_mha_with_cache_inplace(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
    sinks: Optional[torch.Tensor],
    sliding_window: Optional[int],
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache(
        q,
        k,
        v,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        k_cache,
        v_cache,
        scale,
        sinks,
        sliding_window,
    )
    output.copy_(result)


@triton_attention_flattened_mha_with_cache_inplace.register_fake
def triton_attention_flattened_mha_with_cache_inplace_fake(
    q,
    k,
    v,
    batch_info_host,
    seq_len,
    input_pos,
    slot_idx,
    cu_seqlen,
    k_cache,
    v_cache,
    scale,
    sinks,
    sliding_window,
    output,
) -> None:
    pass


# ── Torch Backend Attention ──
@torch.library.custom_op(
    "auto_deploy::torch_cached_attention_with_cache_inplace", mutates_args=("output",)
)
def torch_cached_attention_with_cache_inplace(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # CONSTANTS
    scale: Optional[float],
    sinks: Optional[torch.Tensor],
    sliding_window_size: Optional[int],
    logit_cap: Optional[float],
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.torch_cached_attention_with_cache(
        q,
        k,
        v,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        k_cache,
        v_cache,
        scale,
        sinks,
        sliding_window_size,
        logit_cap,
    )
    output.copy_(result)


@torch_cached_attention_with_cache_inplace.register_fake
def torch_cached_attention_with_cache_inplace_fake(
    q,
    k,
    v,
    batch_info_host,
    seq_len,
    input_pos,
    slot_idx,
    cu_seqlen,
    k_cache,
    v_cache,
    scale,
    sinks,
    sliding_window_size,
    logit_cap,
    output,
) -> None:
    pass


# =============================================================================
# SSM INPLACE OPS
# =============================================================================


# ── Triton SSM ──
@torch.library.custom_op("auto_deploy::triton_cached_ssm_inplace", mutates_args=("output",))
def triton_cached_ssm_inplace(
    # INPUTS
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    # CACHES
    ssm_state_cache: torch.Tensor,
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.triton_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
    )
    output.copy_(result)


@triton_cached_ssm_inplace.register_fake
def triton_cached_ssm_inplace_fake(
    hidden_states,
    A,
    B,
    C,
    D,
    dt,
    dt_bias,
    batch_info_host,
    cu_seqlen,
    slot_idx,
    use_initial_states,
    chunk_indices,
    chunk_offsets,
    seq_idx_prefill,
    ssm_state_cache,
    time_step_limit,
    chunk_size,
    output,
) -> None:
    pass


# ── Torch SSM ──
@torch.library.custom_op("auto_deploy::torch_cached_ssm_inplace", mutates_args=("output",))
def torch_cached_ssm_inplace(
    # INPUTS
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # CACHES
    ssm_state_cache: torch.Tensor,
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.torch_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
    )
    output.copy_(result)


@torch_cached_ssm_inplace.register_fake
def torch_cached_ssm_inplace_fake(
    hidden_states,
    A,
    B,
    C,
    D,
    dt,
    dt_bias,
    batch_info_host,
    seq_len,
    cu_seqlen,
    slot_idx,
    use_initial_states,
    ssm_state_cache,
    time_step_limit,
    chunk_size,
    output,
) -> None:
    pass


# ── FlashInfer SSM ──
@torch.library.custom_op("auto_deploy::flashinfer_cached_ssm_inplace", mutates_args=("output",))
def flashinfer_cached_ssm_inplace(
    # INPUTS
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    # CACHES
    ssm_state_cache: torch.Tensor,
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.flashinfer_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
    )
    output.copy_(result)


@flashinfer_cached_ssm_inplace.register_fake
def flashinfer_cached_ssm_inplace_fake(
    hidden_states,
    A,
    B,
    C,
    D,
    dt,
    dt_bias,
    batch_info_host,
    cu_seqlen,
    slot_idx,
    use_initial_states,
    chunk_indices,
    chunk_offsets,
    seq_idx_prefill,
    ssm_state_cache,
    time_step_limit,
    chunk_size,
    output,
) -> None:
    pass


# =============================================================================
# DELTA RULE INPLACE OP
# =============================================================================


@torch.library.custom_op("auto_deploy::fla_cached_delta_rule_inplace", mutates_args=("output",))
def fla_cached_delta_rule_inplace(
    # INPUTS
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # CACHES
    delta_cache: torch.Tensor,
    # CONSTANTS
    scale: float,
    # OUTPUT (inplace)
    output: torch.Tensor,
) -> None:
    result = torch.ops.auto_deploy.fla_cached_delta_rule(
        q,
        k,
        v,
        beta,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )
    output.copy_(result)


@fla_cached_delta_rule_inplace.register_fake
def fla_cached_delta_rule_inplace_fake(
    q,
    k,
    v,
    beta,
    batch_info_host,
    cu_seqlen,
    slot_idx,
    use_initial_states,
    delta_cache,
    scale,
    output,
) -> None:
    pass


# =============================================================================
# MAPPING: original op name → inplace op
# =============================================================================

# This mapping is used by the graph transform to swap original → inplace ops
ORIGINAL_TO_INPLACE: Dict[str, object] = {}


def _populate_mapping():
    """Populate the original → inplace op mapping.

    Must be called after all ops are registered (i.e., at import time, bottom of module).
    """
    global ORIGINAL_TO_INPLACE
    pairs = [
        (
            "auto_deploy::flashinfer_attention_mha_with_cache",
            torch.ops.auto_deploy.flashinfer_attention_mha_with_cache_inplace,
        ),
        (
            "auto_deploy::triton_attention_flattened_mha_with_cache",
            torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache_inplace,
        ),
        (
            "auto_deploy::torch_cached_attention_with_cache",
            torch.ops.auto_deploy.torch_cached_attention_with_cache_inplace,
        ),
        (
            "auto_deploy::triton_cached_ssm",
            torch.ops.auto_deploy.triton_cached_ssm_inplace,
        ),
        (
            "auto_deploy::torch_cached_ssm",
            torch.ops.auto_deploy.torch_cached_ssm_inplace,
        ),
        (
            "auto_deploy::flashinfer_cached_ssm",
            torch.ops.auto_deploy.flashinfer_cached_ssm_inplace,
        ),
        (
            "auto_deploy::fla_cached_delta_rule",
            torch.ops.auto_deploy.fla_cached_delta_rule_inplace,
        ),
    ]
    ORIGINAL_TO_INPLACE = {name: op for name, op in pairs}


_populate_mapping()
