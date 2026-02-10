# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-torch cached backend for the Gated Delta Rule.

The Gated Delta Rule extends the basic Delta Rule with an exponential decay gate ``g``:
  S = S * exp(g) + k * (v - S^T @ k) * beta

This module provides:
  - ``_torch_gated_delta_step``:   single-token recurrence (decode)
  - ``_torch_gated_delta_prefill``: loop-based prefill over the sequence dimension
  - ``torch_cached_gated_delta_rule``: cached custom op dispatching prefill / decode
  - ``TorchGatedDeltaBackend``: AttentionDescriptor registered as ``"torch_gated_delta"``

Reference:
  - Gated Delta Networks paper: https://arxiv.org/abs/2412.06464
"""

from typing import List, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    StateResourceHandler,
)

# ---------------------------------------------------------------------------
# Core recurrence helpers
# ---------------------------------------------------------------------------


def _torch_gated_delta_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-token gated delta rule recurrence.

    All computation is performed in float32 for numerical stability.

    Args:
        q:     [B, H, K]
        k:     [B, H, K]
        v:     [B, H, V]
        g:     [B, H]          gating / decay values (negative, log-space)
        beta:  [B, H]          beta scaling values
        state: [B, H, K, V]    recurrent state
        scale: query scaling factor

    Returns:
        output:    [B, H, V]
        new_state: [B, H, K, V]
    """
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    state = state.float()

    # Apply decay gate to state
    state = state * torch.exp(g[..., None, None])  # [B, H, K, V]

    # Delta update: v' = v - S^T @ k
    v_prime = v - torch.einsum("bhk,bhkv->bhv", k, state)  # [B, H, V]
    v_prime = v_prime * beta[..., None]  # [B, H, V]

    # Update state: S = S + k outer v'
    state = state + torch.einsum("bhk,bhv->bhkv", k, v_prime)  # [B, H, K, V]

    # Output: o = (q * scale) @ S
    output = torch.einsum("bhk,bhkv->bhv", q * scale, state)  # [B, H, V]

    return output, state


def _torch_gated_delta_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loop-based prefill for the gated delta rule.

    Iterates ``_torch_gated_delta_step`` over the sequence dimension.

    Args:
        q:     [B, S, H, K]  (bsnd layout)
        k:     [B, S, H, K]
        v:     [B, S, H, V]
        g:     [B, S, H]
        beta:  [B, S, H]
        scale: query scaling factor
        initial_state: [B, H, K, V]

    Returns:
        output:      [B, S, H, V]
        final_state: [B, H, K, V]
    """
    B, S, H, V = v.shape
    state = initial_state.float()
    outputs = []

    for t in range(S):
        # Slice at time t from bsnd: q[:, t] gives [B, H, K]
        o_t, state = _torch_gated_delta_step(
            q[:, t],  # [B, H, K]
            k[:, t],  # [B, H, K]
            v[:, t],  # [B, H, V]
            g[:, t],  # [B, H]
            beta[:, t],  # [B, H]
            state,  # [B, H, K, V]
            scale,
        )
        outputs.append(o_t)  # [B, H, V]

    # Stack along seq dim: [B, S, H, V]
    output = torch.stack(outputs, dim=1)
    return output, state


# ---------------------------------------------------------------------------
# Cached custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::torch_cached_gated_delta_rule", mutates_args=())
def torch_cached_gated_delta_rule(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    """Cached gated delta rule using pure-torch recurrence.

    Handles mixed prefill + decode batches. Inputs use the autodeploy bsnd layout.

    Args:
        q:    [B, S, H, K]
        k:    [B, S, H, K]
        v:    [B, S, H, V]
        g:    [B, S, H]
        beta: [B, S, H]
        batch_info_host: [num_prefill, num_prefill_tokens, num_decode] on host
        cu_seqlen:       cumulative sequence lengths for prefill sequences
        slot_idx:        per-sequence slot indices into delta_cache
        use_initial_states: per-sequence bool (True if cache history exists)
        delta_cache:     [max_slots, H, K, V] recurrent state cache
        scale:           query scaling factor

    Returns:
        output: [B, S, H, V]
    """
    b, s, num_heads, _ = q.shape

    # Pre-allocate output
    y = torch.empty_like(v, memory_format=torch.contiguous_format)

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # Clean up metadata
    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    # Flatten for indexing: [B*S, H, D]
    q_flat = q.reshape(b * s, num_heads, -1)
    k_flat = k.reshape(b * s, num_heads, -1)
    v_flat = v.reshape(b * s, num_heads, -1)
    g_flat = g.reshape(b * s, num_heads)
    beta_flat = beta.reshape(b * s, num_heads)
    y_flat = y.reshape(b * s, num_heads, -1)

    key_dim = q.shape[-1]
    value_dim = v.shape[-1]

    # ---- PREFILL ----
    if num_prefill > 0:
        for seq_idx in range(num_prefill):
            start = cu_seqlen_prefill[seq_idx].item()
            end = cu_seqlen_prefill[seq_idx + 1].item()
            slot = slot_idx[seq_idx]

            # Gather per-sequence tensors: [1, seq_len, H, D]
            q_seq = q_flat[start:end].unsqueeze(0)  # [1, S, H, K]
            k_seq = k_flat[start:end].unsqueeze(0)  # [1, S, H, K]
            v_seq = v_flat[start:end].unsqueeze(0)  # [1, S, H, V]
            g_seq = g_flat[start:end].unsqueeze(0)  # [1, S, H]
            beta_seq = beta_flat[start:end].unsqueeze(0)  # [1, S, H]

            # Initial state for this sequence
            if use_initial_states[seq_idx]:
                init_state = delta_cache[slot].unsqueeze(0).clone()  # [1, H, K, V]
            else:
                init_state = torch.zeros(
                    1,
                    num_heads,
                    key_dim,
                    value_dim,
                    dtype=torch.float32,
                    device=q.device,
                )

            y_seq, final_state = _torch_gated_delta_prefill(
                q_seq,
                k_seq,
                v_seq,
                g_seq,
                beta_seq,
                scale,
                init_state,
            )

            # Write output
            y_flat[start:end] = y_seq.squeeze(0).to(y_flat.dtype)

            # Write final state back to cache
            delta_cache[slot] = final_state.squeeze(0).to(delta_cache.dtype)

    # ---- DECODE ----
    if num_decode > 0:
        for i in range(num_decode):
            token_idx = num_prefill_tokens + i
            seq_idx = num_prefill + i
            slot = slot_idx[seq_idx]

            # Single token: [H, D]
            q_tok = q_flat[token_idx]  # [H, K]
            k_tok = k_flat[token_idx]  # [H, K]
            v_tok = v_flat[token_idx]  # [H, V]
            g_tok = g_flat[token_idx]  # [H]
            beta_tok = beta_flat[token_idx]  # [H]

            # Load state from cache
            state = delta_cache[slot].unsqueeze(0).clone()  # [1, H, K, V]

            o_tok, new_state = _torch_gated_delta_step(
                q_tok.unsqueeze(0),  # [1, H, K]
                k_tok.unsqueeze(0),  # [1, H, K]
                v_tok.unsqueeze(0),  # [1, H, V]
                g_tok.unsqueeze(0),  # [1, H]
                beta_tok.unsqueeze(0),  # [1, H]
                state,  # [1, H, K, V]
                scale,
            )

            # Write output
            y_flat[token_idx] = o_tok.squeeze(0).to(y_flat.dtype)

            # Write state back to cache
            delta_cache[slot] = new_state.squeeze(0).to(delta_cache.dtype)

    return y


@torch_cached_gated_delta_rule.register_fake
def torch_cached_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    delta_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(v)


# ---------------------------------------------------------------------------
# AttentionDescriptor backend
# ---------------------------------------------------------------------------


@AttentionRegistry.register("torch_gated_delta")
class TorchGatedDeltaBackend(AttentionDescriptor):
    """Pure-torch cached backend for the Gated Delta Rule.

    Registered as ``"torch_gated_delta"`` in the AttentionRegistry.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q, k, v, g, beta
        return 5

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_gated_delta_rule

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.torch_cached_gated_delta_rule.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        key_node = source_attn_node.args[1]
        value_node = source_attn_node.args[2]
        num_heads = key_node.meta["val"].shape[-2]
        key_dim = key_node.meta["val"].shape[-1]
        value_dim = value_node.meta["val"].shape[-1]

        return {
            "delta_cache": StateResourceHandler(
                num_heads,
                key_dim,
                value_dim,
                # NOTE: torch backend uses float32 cache to avoid bfloat16 quantization
                # errors that accumulate across autoregressive decode steps.
                dtype=torch.float32,
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale = extract_op_args(source_attn_node, "scale")[0]
        if scale is None:
            key_node = source_attn_node.args[1]
            key_dim = key_node.meta["val"].shape[-1]
            scale = key_dim**-0.5
        return [scale]
