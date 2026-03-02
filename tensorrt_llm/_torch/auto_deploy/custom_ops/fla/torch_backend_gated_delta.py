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

This op accepts raw (un-normalized, un-expanded) q/k and raw gating projections
(a, b) together with per-head parameters (A_log, dt_bias). L2 normalization,
GQA repeat-interleave, and gating computation are performed internally.

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
import torch.nn.functional as F
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
from .fla_gated_delta import _l2norm

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
        q:     [B, H, K]       (already l2-normalized and GQA-expanded)
        k:     [B, H, K]       (already l2-normalized and GQA-expanded)
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

    state = state * torch.exp(g[..., None, None])
    v_prime = v - torch.einsum("bhk,bhkv->bhv", k, state)
    v_prime = v_prime * beta[..., None]
    state = state + torch.einsum("bhk,bhv->bhkv", k, v_prime)
    output = torch.einsum("bhk,bhkv->bhv", q * scale, state)

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
        q:     [B, S, H, K]  (bsnd layout, already l2-normalized and GQA-expanded)
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
        o_t, state = _torch_gated_delta_step(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            scale,
        )
        outputs.append(o_t)

    output = torch.stack(outputs, dim=1)
    return output, state


# ---------------------------------------------------------------------------
# Preprocessing: L2 norm + GQA expand + gating computation
# ---------------------------------------------------------------------------


def _preprocess_raw_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b_proj: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    HV: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply L2 normalization, GQA expansion, and gating computation.

    Args:
        q:       [..., H_k, K]
        k:       [..., H_k, K]
        a:       [..., HV]       raw gating projection
        b_proj:  [..., HV]       raw beta projection
        A_log:   [HV]            log of decay base
        dt_bias: [HV]            gating bias
        HV:      number of value heads

    Returns:
        q_out:  [..., HV, K] (l2-normed, expanded)
        k_out:  [..., HV, K] (l2-normed, expanded)
        g:      [..., HV]    (decay gate in log-space)
        beta:   [..., HV]    (sigmoid-activated scaling)
    """
    H_k = q.shape[-2]
    interleave = HV // H_k

    q_out = _l2norm(q.float()).to(q.dtype)
    k_out = _l2norm(k.float()).to(k.dtype)

    if interleave > 1:
        q_out = q_out.repeat_interleave(interleave, dim=-2)
        k_out = k_out.repeat_interleave(interleave, dim=-2)

    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = b_proj.float().sigmoid()

    return q_out, k_out, g, beta


# ---------------------------------------------------------------------------
# Cached custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "auto_deploy::torch_cached_gated_delta_rule", mutates_args=("delta_cache",)
)
def torch_cached_gated_delta_rule(
    # INPUTS (raw, un-normalized, un-expanded)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, HV, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    """Cached gated delta rule using pure-torch recurrence.

    Handles mixed prefill + decode batches. Inputs use the autodeploy bsnd layout.
    L2 normalization, GQA expansion, and gating (g/beta) are computed internally.
    """
    bsz, s, H_k, _ = q.shape
    HV = v.shape[2]

    y = torch.empty_like(v, memory_format=torch.contiguous_format)

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    # Flatten for indexing: [B*S, ...]
    q_flat = q.reshape(bsz * s, H_k, -1)
    k_flat = k.reshape(bsz * s, H_k, -1)
    v_flat = v.reshape(bsz * s, HV, -1)
    a_flat = a.reshape(bsz * s, HV)
    b_flat = b.reshape(bsz * s, HV)
    y_flat = y.reshape(bsz * s, HV, -1)

    key_dim = q.shape[-1]
    value_dim = v.shape[-1]

    # ---- PREFILL ----
    if num_prefill > 0:
        for seq_idx in range(num_prefill):
            start = cu_seqlen_prefill[seq_idx].item()
            end = cu_seqlen_prefill[seq_idx + 1].item()
            slot = slot_idx[seq_idx]

            q_seq = q_flat[start:end].unsqueeze(0)  # [1, S_i, H_k, K]
            k_seq = k_flat[start:end].unsqueeze(0)  # [1, S_i, H_k, K]
            v_seq = v_flat[start:end].unsqueeze(0)  # [1, S_i, HV, V]
            a_seq = a_flat[start:end].unsqueeze(0)  # [1, S_i, HV]
            b_seq = b_flat[start:end].unsqueeze(0)  # [1, S_i, HV]

            q_proc, k_proc, g_seq, beta_seq = _preprocess_raw_inputs(
                q_seq, k_seq, a_seq, b_seq, A_log, dt_bias, HV
            )

            if use_initial_states[seq_idx]:
                init_state = delta_cache[slot].unsqueeze(0).clone()
            else:
                init_state = torch.zeros(
                    1,
                    HV,
                    key_dim,
                    value_dim,
                    dtype=torch.float32,
                    device=q.device,
                )

            y_seq, final_state = _torch_gated_delta_prefill(
                q_proc,
                k_proc,
                v_seq,
                g_seq,
                beta_seq,
                scale,
                init_state,
            )

            y_flat[start:end] = y_seq.squeeze(0).to(y_flat.dtype)
            delta_cache[slot] = final_state.squeeze(0).to(delta_cache.dtype)

    # ---- DECODE ----
    if num_decode > 0:
        for i in range(num_decode):
            token_idx = num_prefill_tokens + i
            seq_idx = num_prefill + i
            slot = slot_idx[seq_idx]

            q_tok = q_flat[token_idx].unsqueeze(0)  # [1, H_k, K]
            k_tok = k_flat[token_idx].unsqueeze(0)  # [1, H_k, K]
            v_tok = v_flat[token_idx].unsqueeze(0)  # [1, HV, V]
            a_tok = a_flat[token_idx].unsqueeze(0)  # [1, HV]
            b_tok = b_flat[token_idx].unsqueeze(0)  # [1, HV]

            q_proc, k_proc, g_tok, beta_tok = _preprocess_raw_inputs(
                q_tok, k_tok, a_tok, b_tok, A_log, dt_bias, HV
            )

            state = delta_cache[slot].unsqueeze(0).clone()

            o_tok, new_state = _torch_gated_delta_step(
                q_proc,
                k_proc,
                v_tok,
                g_tok,
                beta_tok,
                state,
                scale,
            )

            y_flat[token_idx] = o_tok.squeeze(0).to(y_flat.dtype)
            delta_cache[slot] = new_state.squeeze(0).to(delta_cache.dtype)

    return y


@torch_cached_gated_delta_rule.register_fake
def torch_cached_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
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
        # q, k, v, a, b, A_log, dt_bias
        return 7

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
        # State is per value-head: [HV, K, V]
        num_heads = value_node.meta["val"].shape[-2]
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
