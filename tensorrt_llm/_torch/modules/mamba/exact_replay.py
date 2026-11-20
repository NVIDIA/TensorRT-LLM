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

Every sequence's chunk grid starts at its own first token, independent of the
other rows of the batch: prefill rows are scanned alone, or as dense batches of
rows padded to the same number of chunks, and decode rows as dense batches of
``chunk_size`` positions per row. The scan loads
the positions after the current token against zero coefficients, so they must
be finite: they are zero after a prefill or a reset, and hold the finished
chunk of the same request after its chunk folds. A chunk's end state is only
consumed when the current token completes it.

A row's position in its chunk follows from the tokens its request has already
run (``num_cached_tokens_per_seq``), so no per-slot counter is kept.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

    from .mamba2_metadata import Mamba2Metadata
    from .mamba2_mixer import Mamba2Mixer

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

# Decode rows scanned per call. A row's dense chunk and its FP32 states are
# transient memory the KV-cache sizing does not reserve, so a decode batch is
# scanned in groups of this many rows; the rows are independent, so the group
# size does not change any bit.
DECODE_GROUP_ROWS = 32
# Tokens, padded to whole chunks, that one dense prefill call scans; it bounds
# the same transient memory for context rows.
PREFILL_GROUP_TOKENS = 16384


def exact_replay_requested() -> bool:
    """Whether the deterministic Mamba2 kernels are requested.

    ``FORCE_MAMBA_KERNEL_DETERMINISTIC=1`` asks for them alone;
    ``FORCE_DETERMINISTIC=1`` implies it, like the other component switches.
    """
    return (
        os.environ.get("FORCE_DETERMINISTIC", "0") == "1"
        or os.environ.get("FORCE_MAMBA_KERNEL_DETERMINISTIC", "0") == "1"
    )


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


def replay_ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_states: torch.Tensor | None,
    dt_softplus: bool,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked scan of dense rows that start at their boundary states.

    Args:
        x: Post-convolution inputs, (b, L, H, P).
        dt: Time steps before bias and softplus, (b, L, H).
        A: Per-head decay parameter, (H,).
        B: Post-convolution input projections, (b, L, G, N).
        C: Post-convolution output projections, (b, L, G, N).
        chunk_size: Tokens per chunk.
        D: Per-head skip weight, (H,).
        dt_bias: Per-head time-step bias, (H,).
        initial_states: FP32 states at the first token, (b, H, P, N), or None
            for zero states.
        dt_softplus: Whether ``dt`` goes through softplus.
        out: Output rows, (b, L, H, P), written in place.

    Returns:
        ``(states, final_states)``: ``states[:, c]`` is the FP32 state at the
        start of chunk ``c`` and ``final_states`` the FP32 state after the last
        chunk.
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
    del chunk_states
    states, final_states = (
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in (states, final_states)
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, output_dtype=torch.float32)
    _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, out=out)
    return states, final_states


def prepare_exact_replay_rows(
    metadata: "Mamba2Metadata",
    num_cached_tokens: Sequence[int] | torch.Tensor,
    batch_size: int,
    num_contexts: int,
    chunk_size: int,
) -> None:
    """Set this step's replay positions from the tokens each request has run.

    Fills the static device buffers ``metadata.replay_pos`` (position of a
    row's new token inside its chunk), ``metadata.replay_has_boundary`` (a
    boundary state exists) and ``metadata.replay_conv_has_initial`` (a
    convolution state exists) with one host-to-device copy each, and the host
    lists ``replay_prefill_n_pre`` and ``replay_prefill_has_boundary`` of the
    context rows.

    Args:
        metadata: The step's Mamba2 metadata (``ReplayMamba2Metadata``).
        num_cached_tokens: Tokens every row's request has run before this
            step, on the host.
        batch_size: Rows of the step, context rows first.
        num_contexts: Number of context rows.
        chunk_size: Tokens per chunk.
    """
    if isinstance(num_cached_tokens, torch.Tensor):
        cached = num_cached_tokens[:batch_size].to(dtype=torch.int64, device="cpu")
    else:
        cached = torch.tensor(list(num_cached_tokens[:batch_size]), dtype=torch.int64)
    pos_host = metadata.replay_pos_host[:batch_size]
    flags_host = metadata.replay_flags_host[:, :batch_size]
    torch.remainder(cached, chunk_size, out=pos_host)
    torch.ge(cached, chunk_size, out=flags_host[0])
    torch.gt(cached, 0, out=flags_host[1])
    metadata.replay_pos[:batch_size].copy_(pos_host, non_blocking=True)
    metadata.replay_has_boundary[:batch_size].copy_(flags_host[0], non_blocking=True)
    metadata.replay_conv_has_initial[:batch_size].copy_(flags_host[1], non_blocking=True)
    metadata.replay_prefill_n_pre = pos_host[:num_contexts].tolist()
    metadata.replay_prefill_has_boundary = flags_host[0, :num_contexts].tolist()


def advance_exact_replay_rows(metadata: "Mamba2Metadata", batch_size: int, chunk_size: int) -> None:
    """Move every row one token on, for a decode step that skips ``prepare()``.

    Args:
        metadata: The step's Mamba2 metadata (``ReplayMamba2Metadata``).
        batch_size: Rows of the step; all of them decode.
        chunk_size: Tokens per chunk.
    """
    pos = metadata.replay_pos[:batch_size]
    pos.add_(1)
    folded = pos == chunk_size
    metadata.replay_has_boundary[:batch_size].logical_or_(folded)
    pos.masked_fill_(folded, 0)
    metadata.replay_conv_has_initial[:batch_size].fill_(True)


def exact_replay_forward_core(
    mixer: "Mamba2Mixer",
    zxbcdt: torch.Tensor,
    attn_metadata: "AttentionMetadata",
    mamba_metadata: "Mamba2Metadata",
    layer_cache: object,
    preallocated_ssm_out: torch.Tensor,
) -> None:
    """Convolution and SSM of one step under the replay semantics.

    Args:
        mixer: The Mamba2 mixer whose parameters and geometry are used.
        zxbcdt: The step's projected tokens, (num_tokens, d_inner + conv_dim +
            nheads), bfloat16; context rows first.
        attn_metadata: The step's attention metadata (row and token counts).
        mamba_metadata: The step's Mamba2 metadata, with the replay rows set
            by ``prepare_exact_replay_rows`` or ``advance_exact_replay_rows``.
        layer_cache: This layer's cache views: ``conv`` and ``temporal`` (the
            FP32 boundary states) and the replay buffers ``replay_x``,
            ``replay_dt`` and ``replay_B``, all indexed by state slot. They
            are updated in place.
        preallocated_ssm_out: Output buffer, (>= num_tokens, d_inner),
            bfloat16; rows past ``num_tokens`` are zeroed.
    """
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
    buffers = (layer_cache.replay_x, layer_cache.replay_dt, layer_cache.replay_B)

    preallocated_ssm_out[num_tokens:].zero_()

    # Every row goes through the prefill convolution kernel in the token-major
    # layout; decode rows are one-token sequences continuing their conv state.
    # A kernel that computes a token from its own inputs only is needed here,
    # whatever TRTLLM_MAMBA_TOKEN_MAJOR_CONV selects for the default path.
    cu_seqlens = (
        mamba_metadata.cu_seqlens[: batch_size + 1]
        if num_prefills
        else mamba_metadata.replay_arange[: batch_size + 1]
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

    new_inputs = (x_all, dt_all, B_all)

    def scan(inputs, C, initial, out):
        x, dt, B = inputs
        return replay_ssd(
            x,
            dt,
            mixer.A,
            B,
            C,
            chunk_size,
            mixer.D,
            mixer.dt_bias,
            initial,
            mixer.delta_softplus,
            out,
        )

    def store(slot, source, tail, fresh, boundary):
        """Leave a row's new boundary state and its partial chunk in its slot.

        ``source`` holds, per buffer, the last ``fresh`` tokens of the partial
        chunk of ``tail`` tokens; the ones before them are already buffered.
        """
        if boundary is not None:
            ssm_states[slot] = boundary
        for buffer, values in zip(buffers, source):
            if fresh:
                buffer[slot, tail - fresh : tail] = values[None]
            # Everything after the partial chunk is zero: a decode step scans
            # the whole buffer row, zero times a non-finite value is not zero,
            # and a newly acquired slot holds arbitrary bytes.
            buffer[:, tail:].index_fill_(0, slot, 0)

    def prefill_row(start, end, slot, n_pre, has_boundary):
        """One sequence, scanned in place from its boundary state."""
        length = end - start
        boundary = ssm_states[slot] if has_boundary else None
        new_boundary = None
        if n_pre:
            # Only the chunk the buffered tokens belong to needs them ahead of
            # the new tokens; the chunks after it are scanned in place.
            head = min(length, chunk_size - n_pre)
            head_inputs = [
                torch.cat((buffer[slot, :n_pre][0], new[start : start + head]))[None]
                for buffer, new in zip(buffers, new_inputs)
            ]
            head_C = torch.cat(
                (C_all.new_zeros(n_pre, ngroups, dstate), C_all[start : start + head])
            )
            head_out = torch.empty(
                1, n_pre + head, nheads, head_dim, dtype=zxbcdt.dtype, device=zxbcdt.device
            )
            _, final = scan(head_inputs, head_C[None], boundary, head_out)
            out_all[start : start + head].copy_(head_out[0, n_pre:])
            if n_pre + head < chunk_size:
                store(slot, [new[start:end] for new in new_inputs], n_pre + length, length, None)
                return
            boundary = new_boundary = final
            start += head
        rest = end - start
        tail = 0
        if rest:
            states, final = scan(
                [new[None, start:end] for new in new_inputs],
                C_all[None, start:end],
                boundary,
                out_all[None, start:end],
            )
            full_chunks = rest // chunk_size
            if full_chunks:
                new_boundary = states[:, full_chunks] if full_chunks < states.shape[1] else final
            tail = rest - full_chunks * chunk_size
        store(slot, [new[end - tail : end] for new in new_inputs], tail, tail, new_boundary)

    def prefill_group(members, num_chunks):
        """Sequences with the same number of chunks, scanned as one dense batch.

        Each row holds its buffered tokens and its new ones from position 0,
        zero-padded to whole chunks, so its chunk grid is the one it has alone.
        """
        rows, padded = len(members), num_chunks * chunk_size
        dense = [new.new_zeros(rows, padded, *new.shape[1:]) for new in new_inputs]
        C_dense = C_all.new_zeros(rows, padded, ngroups, dstate)
        initial = ssm_states.new_zeros(rows, *ssm_states.shape[1:])
        for i, (start, end, slot, n_pre, has_boundary) in enumerate(members):
            used = n_pre + end - start
            for values, buffer, new in zip(dense, buffers, new_inputs):
                if n_pre:
                    values[i, :n_pre] = buffer[slot, :n_pre][0]
                values[i, n_pre:used] = new[start:end]
            C_dense[i, n_pre:used] = C_all[start:end]
            if has_boundary:
                initial[i] = ssm_states[slot][0]
        out_dense = torch.empty(
            rows, padded, nheads, head_dim, dtype=zxbcdt.dtype, device=zxbcdt.device
        )
        states, final = scan(dense, C_dense, initial, out_dense)
        for i, (start, end, slot, n_pre, has_boundary) in enumerate(members):
            used = n_pre + end - start
            out_all[start:end].copy_(out_dense[i, n_pre:used])
            full_chunks = used // chunk_size
            new_boundary = None
            if full_chunks:
                new_boundary = (
                    states[i : i + 1, full_chunks] if full_chunks < num_chunks else final[i : i + 1]
                )
            tail = used - full_chunks * chunk_size
            fresh = min(tail, end - start)
            store(
                slot,
                [values[i, used - fresh : used] for values in dense],
                tail,
                fresh,
                new_boundary,
            )

    if num_prefills:
        # Group the sequences by their number of chunks: a group runs as one
        # dense batch, which costs one set of kernel launches however many
        # short prompts the step holds. A group is capped by its token count,
        # which bounds the transient FP32 chunk states; a sequence alone in
        # its group is scanned in place, without the dense copy.
        groups: dict[int, list] = {}
        start = 0
        for row, length in enumerate(attn_metadata.seq_lens[:num_prefills].tolist()):
            n_pre = mamba_metadata.replay_prefill_n_pre[row]
            member = (
                start,
                start + length,
                state_indices[row : row + 1].long(),
                n_pre,
                mamba_metadata.replay_prefill_has_boundary[row],
            )
            groups.setdefault(-(-(n_pre + length) // chunk_size), []).append(member)
            start += length
        for num_chunks, members in groups.items():
            per_call = max(1, PREFILL_GROUP_TOKENS // (num_chunks * chunk_size))
            for begin in range(0, len(members), per_call):
                batch = members[begin : begin + per_call]
                if len(batch) == 1:
                    prefill_row(*batch[0])
                else:
                    prefill_group(batch, num_chunks)

    for begin in range(0, num_decodes, DECODE_GROUP_ROWS):
        rows = slice(
            num_prefills + begin, min(num_prefills + begin + DECODE_GROUP_ROWS, batch_size)
        )
        tokens = slice(
            num_prefill_tokens + begin, num_prefill_tokens + begin + rows.stop - rows.start
        )
        slots = state_indices[rows].long()
        pos = mamba_metadata.replay_pos[rows]
        ar = mamba_metadata.arange_long(rows.stop - rows.start)[:-1]
        new = (x_all[tokens], dt_all[tokens], B_all[tokens])
        # Dense rows: the buffered partial chunk with the new token at `pos`.
        dense = []
        for buffer, value in zip(buffers, new):
            buffer[slots, pos] = value
            dense.append(buffer.index_select(0, slots))
        C_dense = torch.zeros_like(dense[2])
        C_dense[ar, pos] = C_all[tokens]
        # A row without a boundary starts from the zero state; its slot holds
        # nothing to keep until the row's first chunk folds.
        initial = ssm_states.index_select(0, slots)
        initial.masked_fill_(
            mamba_metadata.replay_has_boundary[rows].logical_not()[:, None, None, None], 0
        )
        out_dense = torch.empty_like(dense[0])
        _, final = replay_ssd(
            dense[0],
            dense[1],
            mixer.A,
            dense[2],
            C_dense,
            chunk_size,
            mixer.D,
            mixer.dt_bias,
            initial,
            mixer.delta_softplus,
            out_dense,
        )
        out_all[tokens].copy_(out_dense[ar, pos])
        # The new token completes the chunk: its end state is the new boundary.
        fold = (pos == chunk_size - 1)[:, None, None, None]
        ssm_states.index_copy_(0, slots, torch.where(fold, final, initial, out=final))
