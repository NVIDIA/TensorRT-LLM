# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 exact replay: one chunked-scan arithmetic for prefill and decode.

With the replay state bound to the cache (``Mamba2ExactReplayState``),
``ssm_states`` holds each slot's FP32 state at its last chunk boundary and the
per-slot replay buffers hold the inputs of the partial chunk after it: the
post-convolution ``x`` and ``B`` and the raw ``dt``. Every SSD call starts at
that boundary. A decode step re-feeds the buffered partial chunk ahead of its
new token, so the kernels see the chunk grid of a single-shot prefill and only
the new token's output row is kept; ``C`` is not buffered because it only
affects its own row.

Prefill rows run one sequence at a time so each sequence's chunk grid starts
at its own first token, independent of the other rows of the batch. Decode
rows run as one dense batch of ``chunk_size`` positions per row: positions
after the current token hold stale inputs that the causal scan and the prefix
cumsum never read for the current row, and a chunk's state is only consumed
when the current token completes it.
"""

from __future__ import annotations

import torch
from einops import rearrange

from .causal_conv1d import causal_conv1d_fn
from .ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_fwd_kernel
from .ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_fwd_kernel
from .ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_cumsum_fwd_kernel,
    _chunk_state_fwd,
    _chunk_state_fwd_kernel,
)
from .ssd_state_passing import _state_passing_fwd, _state_passing_fwd_kernel

# One tile configuration per SSD kernel. Autotuning picks tiles by timing and
# the tiles set the FP32 accumulation order, so the deterministic mode pins
# them. Each entry must exist in the kernel's autotune list.
_PINNED_SSD_CONFIGS = (
    (_chunk_cumsum_fwd_kernel, {"BLOCK_SIZE_H": 8}, None, None),
    (_chunk_state_fwd_kernel, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 4, 4),
    (_state_passing_fwd_kernel, {"BLOCK_SIZE": 256}, None, None),
    (_bmm_chunk_fwd_kernel, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 4, 2),
    (_chunk_scan_fwd_kernel, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 2, 4),
)
_pinned = False


def pin_ssd_autotune_configs() -> None:
    """Restrict every SSD autotuner to its pinned tile configuration."""
    global _pinned
    if _pinned:
        return
    for kernel, kwargs, num_stages, num_warps in _PINNED_SSD_CONFIGS:
        matches = [
            config
            for config in kernel.configs
            if all(config.kwargs.get(key) == value for key, value in kwargs.items())
            and num_stages in (None, config.num_stages)
            and num_warps in (None, config.num_warps)
        ]
        if not matches:
            name = getattr(kernel, "__name__", repr(kernel))
            raise RuntimeError(f"{name} has no autotune configuration {kwargs}")
        kernel.configs = matches[:1]
        kernel.cache.clear()
    _pinned = True


def replay_ssd(x, dt, A, B, C, chunk_size, D, dt_bias, initial_states, dt_softplus, out):
    """Chunked scan of dense rows that start at their boundary states.

    ``x``: (b, L, H, P); ``dt``: (b, L, H) before bias and softplus; ``B`` and
    ``C``: (b, L, G, N); ``initial_states``: (b, H, P, N) FP32, or None for
    zero states; ``out``: (b, L, H, P). Returns ``(states, final_states)``:
    ``states[:, c]`` is the FP32 state at the start of chunk ``c`` and
    ``final_states`` the FP32 state after the last chunk.
    """
    dstate = B.shape[-1]
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=(0.0, float("inf"))
    )
    chunk_states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=True)
    states, final_states = _state_passing_fwd(
        rearrange(chunk_states, "... p n -> ... (p n)"),
        dA_cumsum,
        initial_states=(
            rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None
            else None
        ),
        chunk_size=chunk_size,
        out_dtype=torch.float32,
    )
    states, final_states = (
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in (states, final_states)
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
    _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, out=out)
    return states, final_states


def prepare_exact_replay_rows(
    metadata, counter, batch_size, num_contexts, context_lens, chunk_size=None
):
    """Snapshot this step's replay positions and advance the slot token counts.

    ``counter`` is the slot-shared int32 count of tokens each slot has run
    through the SSM; rows whose request has no cached tokens start over.
    Fills ``metadata.replay_pos`` (position of a row's new token inside its
    chunk), ``metadata.replay_has_boundary`` (a boundary state exists),
    ``metadata.replay_conv_has_initial`` (conv state exists; always true for
    decode rows) and, for prefill rows, the host lists
    ``replay_prefill_n_pre`` and ``replay_prefill_has_boundary``. The counts
    advance here, once per step, because every layer sees the same tokens.
    ``chunk_size`` defaults to the metadata's SSD chunk size.
    """
    chunk_size = chunk_size or metadata.chunk_size
    slots = metadata.state_indices[:batch_size].long()
    counts = counter.index_select(0, slots)
    conv_has_initial = metadata.replay_conv_has_initial[:batch_size]
    conv_has_initial.fill_(True)
    if num_contexts:
        has_initial = metadata.has_initial_states[:num_contexts]
        counts[:num_contexts] = torch.where(has_initial, counts[:num_contexts], 0)
        conv_has_initial[:num_contexts].copy_(has_initial)
    metadata.replay_pos[:batch_size].copy_(counts % chunk_size)
    metadata.replay_has_boundary[:batch_size].copy_(counts >= chunk_size)
    if num_contexts:
        host_counts = counts[:num_contexts].tolist()
        metadata.replay_prefill_n_pre = [count % chunk_size for count in host_counts]
        metadata.replay_prefill_has_boundary = [count >= chunk_size for count in host_counts]
        counts[:num_contexts] += context_lens
        counts[num_contexts:] += 1
    else:
        metadata.replay_prefill_n_pre = []
        metadata.replay_prefill_has_boundary = []
        counts += 1
    counter[slots] = counts


def exact_replay_forward_core(
    mixer, zxbcdt, attn_metadata, mamba_metadata, layer_cache, preallocated_ssm_out
):
    """Convolution and SSM of one step under the replay semantics."""
    num_prefills = attn_metadata.num_contexts
    batch_size = attn_metadata.seq_lens.shape[0]
    num_decodes = batch_size - num_prefills
    num_prefill_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    chunk_size = mixer.chunk_size
    nheads, head_dim = mixer.tp_nheads, mixer.head_dim
    ngroups, dstate = mixer.tp_ngroups, mixer.d_state
    d_inner, conv_dim = mixer.tp_d_inner, mixer.tp_conv_dim
    state_indices = mamba_metadata.state_indices[:batch_size]
    conv_states, ssm_states = layer_cache.conv, layer_cache.temporal
    replay_x, replay_dt, replay_B = (
        layer_cache.replay_x,
        layer_cache.replay_dt,
        layer_cache.replay_B,
    )

    preallocated_ssm_out[num_tokens:].zero_()

    # Every row goes through the prefill convolution kernel in the token-major
    # layout; decode rows are one-token sequences continuing their conv state.
    cu_seqlens = (
        mamba_metadata.cu_seqlens[: batch_size + 1]
        if num_prefills
        else mamba_metadata._arange_buffer[: batch_size + 1]
    )
    xbc = zxbcdt[:num_tokens, d_inner : d_inner + conv_dim]
    conv_out = torch.empty_like(xbc, memory_format=torch.contiguous_format)
    causal_conv1d_fn(
        xbc.t(),
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        query_start_loc=cu_seqlens,
        cache_indices=state_indices,
        has_initial_state=mamba_metadata.replay_conv_has_initial[:batch_size],
        conv_states=conv_states,
        activation="silu",
        out=conv_out.t(),
    )
    group_dim = ngroups * dstate
    x_all = conv_out[:, :d_inner].view(num_tokens, nheads, head_dim)
    B_all = conv_out[:, d_inner : d_inner + group_dim].view(num_tokens, ngroups, dstate)
    C_all = conv_out[:, d_inner + group_dim :].view(num_tokens, ngroups, dstate)
    dt_all = zxbcdt[:num_tokens, d_inner + conv_dim :]
    out_all = preallocated_ssm_out[:num_tokens].view(num_tokens, nheads, head_dim)

    if num_prefills:
        seq_lens = attn_metadata.seq_lens[:num_prefills].tolist()
        start = 0
        for row, length in enumerate(seq_lens):
            end = start + length
            slot = state_indices[row : row + 1].long()
            n_pre = mamba_metadata.replay_prefill_n_pre[row]
            x_new, dt_new = x_all[start:end], dt_all[start:end]
            B_new, C_new = B_all[start:end], C_all[start:end]
            if n_pre:
                x_aug = torch.cat((replay_x[slot][0, :n_pre], x_new))
                dt_aug = torch.cat((replay_dt[slot][0, :n_pre], dt_new))
                B_aug = torch.cat((replay_B[slot][0, :n_pre], B_new))
                C_aug = torch.cat((C_new.new_zeros(n_pre, ngroups, dstate), C_new))
                out_aug = torch.empty(
                    n_pre + length, nheads, head_dim, dtype=zxbcdt.dtype, device=zxbcdt.device
                )
            else:
                x_aug, dt_aug, B_aug, C_aug = x_new, dt_new, B_new, C_new
                out_aug = out_all[start:end]
            aug_len = n_pre + length
            initial = ssm_states[slot] if mamba_metadata.replay_prefill_has_boundary[row] else None
            states, final = replay_ssd(
                x_aug[None],
                dt_aug[None],
                mixer.A,
                B_aug[None],
                C_aug[None],
                chunk_size,
                mixer.D,
                mixer.dt_bias,
                initial,
                mixer.delta_softplus,
                out_aug[None],
            )
            if n_pre:
                out_all[start:end].copy_(out_aug[n_pre:])
            full_chunks = aug_len // chunk_size
            if full_chunks:
                ssm_states[slot] = (
                    states[:, full_chunks] if full_chunks < states.shape[1] else final
                )
            tail = aug_len - full_chunks * chunk_size
            if tail:
                replay_x[slot, :tail] = x_aug[None, aug_len - tail :]
                replay_dt[slot, :tail] = dt_aug[None, aug_len - tail :]
                replay_B[slot, :tail] = B_aug[None, aug_len - tail :]
            start = end

    if num_decodes:
        rows = slice(num_prefills, batch_size)
        tokens = slice(num_prefill_tokens, num_tokens)
        slots = state_indices[rows].long()
        pos = mamba_metadata.replay_pos[rows]
        has_boundary = mamba_metadata.replay_has_boundary[rows]
        ar = mamba_metadata._arange_buffer_long[:num_decodes]
        x_d, dt_d, B_d, C_d = x_all[tokens], dt_all[tokens], B_all[tokens], C_all[tokens]
        # Dense rows: the buffered partial chunk with the new token at `pos`.
        x_dense = replay_x.index_select(0, slots)
        x_dense[ar, pos] = x_d
        dt_dense = replay_dt.index_select(0, slots)
        dt_dense[ar, pos] = dt_d
        B_dense = replay_B.index_select(0, slots)
        B_dense[ar, pos] = B_d
        C_dense = torch.zeros_like(B_dense)
        C_dense[ar, pos] = C_d
        current = ssm_states.index_select(0, slots)
        initial = torch.where(has_boundary[:, None, None, None], current, 0)
        out_dense = torch.empty_like(x_dense)
        _, final = replay_ssd(
            x_dense,
            dt_dense,
            mixer.A,
            B_dense,
            C_dense,
            chunk_size,
            mixer.D,
            mixer.dt_bias,
            initial,
            mixer.delta_softplus,
            out_dense,
        )
        out_all[tokens].copy_(out_dense[ar, pos])
        replay_x[slots, pos] = x_d
        replay_dt[slots, pos] = dt_d
        replay_B[slots, pos] = B_d
        # The new token completes the chunk: its end state is the new boundary.
        fold = (pos == chunk_size - 1)[:, None, None, None]
        ssm_states.index_copy_(0, slots, torch.where(fold, final, current))
