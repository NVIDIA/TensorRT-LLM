# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN exact replay: one chunked delta-rule arithmetic for prefill and decode.

With the replay state bound to the cache (``GDNExactReplayState``),
``ssm_states`` holds each slot's FP32 recurrent state at its last chunk
boundary and the per-slot replay buffers hold the inputs of the partial chunk
after it: the normalized ``q`` and ``k``, ``v``, the log decay ``g`` and
``beta``. Every chunked delta-rule call starts at that boundary, in FP32.

A prefill row first runs its complete chunks, whose end state becomes the new
boundary, and then its partial chunk from that boundary. A decode step writes
its token into the buffered partial chunk and runs that chunk again, keeping
only the new token's output row; when the token completes the chunk, the
chunk's end state becomes the new boundary. Positions after the current token
hold stale inputs that the causal chunk kernels never read for the current
row.

Prefill rows run one sequence at a time, each on a chunk grid that starts at
its own boundary, so a sequence's bits do not depend on the other rows of the
batch. Decode rows run as one dense batch of one chunk per row.

The convolution and the q/k normalization before the delta rule also use one
kernel for every row: decode rows go through the token-major prefill
convolution as one-token sequences that continue their conv state, and
through the prefill branch of the post-convolution kernel with one token per
program.
"""

from __future__ import annotations

import os

import torch

from tensorrt_llm._utils import get_sm_version

from ..fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from ..fla.chunk_o import chunk_fwd_kernel_o, chunk_fwd_o
from ..fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from ..fla.cumsum import chunk_local_cumsum
from ..fla.solve_tril import solve_tril
from ..fla.wy_fast import recompute_w_u_fwd
from .causal_conv1d import causal_conv1d_fn
from .fuse_elementwise_ops import fused_gdn_post_conv

# The chunk size of the delta-rule kernels.
GDN_CHUNK_SIZE = 64

# One tile configuration per autotuned chunk kernel. Autotuning picks tiles by
# timing and the tiles set the accumulation order, so the deterministic mode
# pins them. Each entry must exist in the kernel's autotune list.
_PINNED_GDN_CONFIGS = (
    (chunk_gated_delta_rule_fwd_kernel_h_blockdim64, {"BV": 64}, 3, 4),
    (chunk_fwd_kernel_o, {"BK": 64, "BV": 64}, 3, 4),
)
_pinned = False


def exact_replay_requested() -> bool:
    """Whether FORCE_DETERMINISTIC asks for the replay semantics."""
    return os.environ.get("FORCE_DETERMINISTIC", "0") == "1"


def exact_replay_disabled_reason(
    spec_config: object | None, ssm_cache_dtype: torch.dtype, is_disagg: bool
) -> str | None:
    """Why FORCE_DETERMINISTIC keeps the default GDN kernels, or None."""
    if spec_config is not None:
        return "speculative decoding is enabled"
    if is_disagg:
        return "disaggregated serving transfers the recurrent state without the replay buffers"
    if ssm_cache_dtype != torch.float32:
        return f"mamba_ssm_cache_dtype is {ssm_cache_dtype}; set it to float32"
    if get_sm_version() not in (100, 103):
        return "it is implemented for SM100 and SM103 only"
    return None


def _autotuner(kernel):
    """The autotuner behind a kernel's heuristics wrapper."""
    return kernel if hasattr(kernel, "configs") else kernel.fn


def pin_gdn_autotune_configs() -> None:
    """Restrict every autotuned chunk kernel to its pinned tile configuration."""
    global _pinned
    if _pinned:
        return
    for kernel, kwargs, num_stages, num_warps in _PINNED_GDN_CONFIGS:
        tuner = _autotuner(kernel)
        matches = [
            config
            for config in tuner.configs
            if all(config.kwargs.get(key) == value for key, value in kwargs.items())
            and config.num_stages == num_stages
            and config.num_warps == num_warps
        ]
        if not matches:
            name = getattr(kernel, "__name__", repr(kernel))
            raise RuntimeError(f"{name} has no autotune configuration {kwargs}")
        tuner.configs = matches[:1]
        tuner.cache.clear()
    _pinned = True


def replay_gdn(q, k, v, g, beta, scale, initial_state, out):
    """Chunked delta rule of dense rows that start at their boundary states.

    ``q`` and ``k``: (b, L, Hk, K); ``v``: (b, L, Hv, V); ``g`` and ``beta``:
    (b, L, Hv) in FP32, ``g`` being the per-token log decay; ``L`` is a
    multiple of the chunk size; ``initial_state``: (b, Hv, V, K) FP32;
    ``out``: (b, L, Hv, V). Returns the FP32 state after the last chunk.
    """
    g_cumsum = chunk_local_cumsum(g, chunk_size=GDN_CHUNK_SIZE, cu_seqlens=None)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=None, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g_cumsum, cu_seqlens=None)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        initial_state_indices=None,
        output_final_state=True,
        inplace_indexed_state_update=False,
        cu_seqlens=None,
    )
    chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cumsum, scale=scale, cu_seqlens=None, output=out)
    return final_state


def gdn_exact_replay_forward_core(
    mixer, mixed_qkv, a, b, attn_metadata, mamba_metadata, layer_cache, output
):
    """Convolution and delta rule of one step under the replay semantics."""
    if getattr(mamba_metadata, "inforward_split", None) is not None:
        raise RuntimeError("GDN exact replay does not support in-forward state snapshots")
    num_prefills = attn_metadata.num_contexts
    batch_size = attn_metadata.seq_lens.shape[0]
    num_decodes = batch_size - num_prefills
    num_prefill_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    chunk = GDN_CHUNK_SIZE
    num_k_heads, head_k_dim = mixer.num_k_heads_per_tp, mixer.head_k_dim
    num_v_heads, head_v_dim = mixer.num_v_heads_per_tp, mixer.head_v_dim
    scale = head_k_dim**-0.5
    state_indices = mamba_metadata.state_indices[:batch_size]
    conv_states, ssm_states = layer_cache.conv, layer_cache.temporal
    replay_q, replay_k, replay_v = layer_cache.replay_q, layer_cache.replay_k, layer_cache.replay_v
    replay_g, replay_beta = layer_cache.replay_g, layer_cache.replay_beta

    if output is None:
        output = mixed_qkv.new_empty((1, mixed_qkv.shape[0], num_v_heads, head_v_dim))
    out_all = output[0]
    out_all[num_tokens:].zero_()

    # Every row goes through the token-major prefill convolution and the
    # prefill branch of the post-convolution kernel, one token per program;
    # decode rows are one-token sequences that continue their conv state.
    # Each of these computes a token from its own inputs only. The
    # channel-major convolution picks between two SiLU forms per warp of
    # tokens, and with several tokens per program the rounding of a token's
    # q/k norm depends on its row in the block, so a token's bits would
    # depend on its neighbors there.
    tokens = mixed_qkv[:num_tokens]
    conv_out = torch.empty((num_tokens, tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
    cu_seqlens = (
        mamba_metadata.query_start_loc[: batch_size + 1]
        if num_prefills
        else mamba_metadata._arange_buffer[: batch_size + 1]
    )
    causal_conv1d_fn(
        tokens.t(),
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        activation=mixer.activation,
        conv_states=conv_states,
        has_initial_state=mamba_metadata.replay_conv_has_initial[:batch_size],
        cache_indices=state_indices,
        query_start_loc=cu_seqlens,
        out=conv_out.t(),
    )
    q_all, k_all, v_all, g_all, beta_all = (
        tensor[0]
        for tensor in fused_gdn_post_conv(
            conv_out.t(),
            None,
            a[:num_tokens],
            b[:num_tokens],
            mixer.A_log,
            mixer.dt_bias,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            block_tokens=1,
        )
    )

    if num_prefills:
        seq_lens = attn_metadata.seq_lens[:num_prefills].tolist()
        start = 0
        for row, length in enumerate(seq_lens):
            end = start + length
            slot = state_indices[row : row + 1].long()
            n_pre = mamba_metadata.replay_prefill_n_pre[row]
            inputs = [t[start:end] for t in (q_all, k_all, v_all, g_all, beta_all)]
            buffers = (replay_q, replay_k, replay_v, replay_g, replay_beta)
            if n_pre:
                # Resume the buffered partial chunk ahead of the new tokens.
                inputs = [
                    torch.cat((buffer[slot][0, :n_pre], new))
                    for buffer, new in zip(buffers, inputs)
                ]
            aug_len = n_pre + length
            out_aug = torch.empty(
                aug_len, num_v_heads, head_v_dim, dtype=out_all.dtype, device=out_all.device
            )
            if mamba_metadata.replay_prefill_has_boundary[row]:
                state = ssm_states[slot]
            else:
                state = torch.zeros(
                    (1, *ssm_states.shape[1:]), dtype=torch.float32, device=ssm_states.device
                )
            full_len = (aug_len // chunk) * chunk
            if full_len:
                state = replay_gdn(
                    *(t[None, :full_len] for t in inputs), scale, state, out_aug[None, :full_len]
                )
                ssm_states[slot] = state
            tail = aug_len - full_len
            if tail:
                for buffer, t in zip(buffers, inputs):
                    buffer[slot, :tail] = t[None, full_len:]
                out_tail = torch.empty(
                    1, chunk, num_v_heads, head_v_dim, dtype=out_all.dtype, device=out_all.device
                )
                replay_gdn(*(buffer[slot] for buffer in buffers), scale, state, out_tail)
                out_aug[full_len:].copy_(out_tail[0, :tail])
            out_all[start:end].copy_(out_aug[n_pre:])
            start = end

    if num_decodes:
        rows = slice(num_prefills, batch_size)
        tokens = slice(num_prefill_tokens, num_tokens)
        slots = state_indices[rows].long()
        pos = mamba_metadata.replay_pos[rows]
        has_boundary = mamba_metadata.replay_has_boundary[rows]
        ar = mamba_metadata._arange_buffer_long[:num_decodes]
        # Dense rows: the buffered partial chunk with the new token at `pos`.
        dense = []
        for buffer, new in zip(
            (replay_q, replay_k, replay_v, replay_g, replay_beta),
            (q_all, k_all, v_all, g_all, beta_all),
        ):
            buffer[slots, pos] = new[tokens]
            dense.append(buffer.index_select(0, slots))
        current = ssm_states.index_select(0, slots)
        initial = torch.where(has_boundary[:, None, None, None], current, 0)
        out_dense = torch.empty(
            num_decodes, chunk, num_v_heads, head_v_dim, dtype=out_all.dtype, device=out_all.device
        )
        final = replay_gdn(*dense, scale, initial, out_dense)
        out_all[tokens].copy_(out_dense[ar, pos])
        # The new token completes the chunk: its end state is the new boundary.
        fold = (pos == chunk - 1)[:, None, None, None]
        ssm_states.index_copy_(0, slots, torch.where(fold, final, current))
    return output
