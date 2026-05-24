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

from typing import List, Optional

import torch
from flashinfer.mamba import selective_state_update as _flashinfer_ssm_update
from torch.fx import Node

from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import (
    replay_selective_state_update,
)
from tensorrt_llm._utils import get_sm_version

from ..._compat import KvCacheConfig
from ..attention_interface import (
    AttentionRegistry,
    BatchInfo,
    MHACallable,
    ReplayCacheBufIdxHandler,
    ReplayOldBHandler,
    ReplayOldDAcumsumHandler,
    ReplayOldDtHandler,
    ReplayOldXHandler,
    ReplayPrevNumAcceptedHandler,
    ResourceHandlerDict,
    SpecSSMResourceHandler,
)
from .mamba_backend_common import (
    BaseBackendSSM,
    _flatten_ssm_inputs,
    _prepare_ssm_decode_inputs,
    _prepare_ssm_grouped_state_update_inputs,
    _run_ssm_prefill,
)


def _fi_align(t: torch.Tensor) -> torch.Tensor:
    """Ensure 128-byte alignment required by FlashInfer kernels.

    - Contiguous + aligned: contiguous() is a no-op, returns t unchanged.
    - Non-contiguous: contiguous() allocates fresh aligned storage, returns it.
    - Contiguous + misaligned: contiguous() is a no-op, clone() forces a new aligned allocation.
    """
    t = t.contiguous()
    return t if t.data_ptr() % 128 == 0 else t.clone()


@torch.library.custom_op(
    "auto_deploy::flashinfer_cached_ssm",
    mutates_args=(
        "ssm_state_cache",
        "intermediate_ssm_state_cache",
        # replay buffers: written in-place by the precompute and main kernels
        # (double-buffered B/dt/dA_cumsum, single-buffered x); None in non-replay mode
        "replay_old_x",
        "replay_old_b",
        "replay_old_dt",
        "replay_old_da_cumsum",
    ),
)
def _flashinfer_cached_ssm(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    intermediate_ssm_state_cache: Optional[
        torch.Tensor
    ],  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]; None in replay mode
    replay_old_x: Optional[torch.Tensor],  # [max_batch, T, nheads, head_dim]; None in non-replay
    replay_old_b: Optional[torch.Tensor],  # [max_batch, 2, T, ngroups, dstate]; None in non-replay
    replay_old_dt: Optional[torch.Tensor],  # [max_batch, 2, nheads, T] fp32; None in non-replay
    replay_old_da_cumsum: Optional[
        torch.Tensor
    ],  # [max_batch, 2, nheads, T] fp32; None in non-replay
    replay_cache_buf_idx: Optional[torch.Tensor],  # [max_batch] int32; None in non-replay
    replay_prev_num_accepted: Optional[torch.Tensor],  # [max_batch] int32; None in non-replay
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat = _flatten_ssm_inputs(
        hidden_states, B, C, dt
    )
    ssm_state_size = B.shape[3]
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_extend, num_decode = batch_info.get_num_sequences()
    num_prefill_tokens, num_extend_tokens, num_decode_tokens = batch_info.get_num_tokens()
    num_total_tokens = num_prefill_tokens + num_extend_tokens + num_decode_tokens

    # Pre-hoist index type conversions so their kernels run before the extend
    # causal_conv1d_update kernel rather than landing in the gap between it
    # and the PDL-dependent replay precompute kernel.
    slot_idx_i32 = slot_idx.to(torch.int32)
    # NOTE: intermediate_state_indices_extend (torch.arange) is NOT hoisted here.
    # It is only needed for the non-replay FlashInfer extend path. Hoisting it
    # unconditionally would create an unnecessary arange kernel in the gap for
    # the replay path (where use_replay=True and the arange is unused).

    if out is not None:
        preallocated_ssm_out = out.view(bs, num_heads, head_dim)
    else:
        preallocated_ssm_out = torch.zeros(
            [bs, num_heads, head_dim],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # PREFILL
    _run_ssm_prefill(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
        preallocated_ssm_out[:num_prefill_tokens].unsqueeze(0),
    )

    # EXTEND: multi-token MTP verification path, writes intermediate SSM states
    extend_inputs = _prepare_ssm_grouped_state_update_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx_i32,
        seq_start=num_prefill,
        token_start=num_prefill_tokens,
        num_seq=num_extend,
        num_tokens=num_extend_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        ssm_state_size=ssm_state_size,
    )

    if extend_inputs is not None:
        tokens_per_extend = num_extend_tokens // num_extend
        (
            slot_idx_extend,
            x_extend,
            B_extend,
            C_extend,
            dt_extend,
            A_full,
            D_full,
            dt_bias_hp,
        ) = extend_inputs

        preallocated_ssm_out_e = preallocated_ssm_out[
            num_prefill_tokens : num_prefill_tokens + num_extend_tokens
        ].view(num_extend, tokens_per_extend, num_heads, head_dim)

        use_replay = batch_info.is_use_replay()
        if use_replay:
            # Replay path: fast-forward SSM state via tl.dot on cached values.
            # State is updated in-place; no disable_state_update needed.
            # x_extend/B_extend/C_extend are non-contiguous views from the CUDA graph's
            # preallocated buffer (stride_T > nheads × head_dim). The Triton kernel
            # handles arbitrary strides via pointer arithmetic; skipping .contiguous()
            # avoids a captured BF16 copy kernel and reduces GPU time by ~5%.
            replay_selective_state_update(
                ssm_state_cache,
                replay_old_x,
                replay_old_b,
                replay_old_dt,
                replay_old_da_cumsum,
                replay_cache_buf_idx,
                replay_prev_num_accepted,
                x_extend,
                dt_extend,
                A_full,
                B_extend,
                C_extend,
                out=preallocated_ssm_out_e,
                D=D_full,
                dt_bias=dt_bias_hp,
                dt_softplus=True,
                state_batch_indices=slot_idx_extend,
                launch_with_pdl=True,  # PDL chain: triton_causal_conv extend → precompute → main
            )
        else:
            if intermediate_ssm_state_cache.size(1) < tokens_per_extend:
                raise RuntimeError(
                    "flashinfer_cached_ssm: intermediate_ssm_state_cache is too small "
                    f"for extend branch (size1={intermediate_ssm_state_cache.size(1)}, "
                    f"tokens_per_extend={tokens_per_extend})"
                )
            # Only needed for the non-replay FlashInfer path; not hoisted to avoid
            # an unnecessary arange kernel in the stream gap for the replay path.
            intermediate_state_indices = torch.arange(
                num_extend, dtype=torch.int32, device=slot_idx_extend.device
            )
            _flashinfer_ssm_update(
                ssm_state_cache,
                x_extend.contiguous(),
                dt_extend,
                A_full,
                B_extend.contiguous(),
                C_extend.contiguous(),
                D_full,
                z=None,
                dt_bias=dt_bias_hp,
                dt_softplus=True,
                state_batch_indices=slot_idx_extend,
                out=preallocated_ssm_out_e,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_ssm_state_cache,
                cache_steps=tokens_per_extend,
                intermediate_state_indices=intermediate_state_indices,
            )

    # DECODE: single-token autoregressive path
    decode_inputs = _prepare_ssm_decode_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx_i32,
        num_prefill + num_extend,
        num_prefill_tokens + num_extend_tokens,
        num_decode,
        num_decode_tokens,
        num_heads,
        head_dim,
        ssm_state_size,
    )

    if decode_inputs is not None:
        (
            slot_idx_decode,
            x_decode,
            B_decode,
            C_decode,
            dt_hp,
            dt_bias_hp,
            A_full,
            D_full,
        ) = decode_inputs

        x_decode = _fi_align(x_decode)
        B_decode = _fi_align(B_decode)
        C_decode = _fi_align(C_decode)

        y_decode = _flashinfer_ssm_update(
            ssm_state_cache,
            x_decode,
            dt_hp,
            A_full,
            B_decode,
            C_decode,
            D_full,
            z=None,
            dt_bias=dt_bias_hp,
            dt_softplus=True,
            state_batch_indices=slot_idx_decode,
        )
        preallocated_ssm_out[num_prefill_tokens + num_extend_tokens : num_total_tokens].copy_(
            y_decode
        )

    if out is not None:
        # out is reused across CUDA graph replays with varying num_total_tokens,
        # so stale data from prior replays can linger in the padding region.
        if num_total_tokens < bs:
            preallocated_ssm_out[num_total_tokens:].zero_()
        return out.new_empty(0)

    return preallocated_ssm_out.view(b, s, num_heads, head_dim)


@_flashinfer_cached_ssm.register_fake
def _flashinfer_cached_ssm_fake(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    intermediate_ssm_state_cache: Optional[
        torch.Tensor
    ],  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]; None in replay mode
    replay_old_x: Optional[torch.Tensor],  # [max_batch, T, nheads, head_dim]; None in non-replay
    replay_old_b: Optional[torch.Tensor],  # [max_batch, 2, T, ngroups, dstate]; None in non-replay
    replay_old_dt: Optional[torch.Tensor],  # [max_batch, 2, nheads, T] fp32; None in non-replay
    replay_old_da_cumsum: Optional[
        torch.Tensor
    ],  # [max_batch, 2, nheads, T] fp32; None in non-replay
    replay_cache_buf_idx: Optional[torch.Tensor],  # [max_batch] int32; None in non-replay
    replay_prev_num_accepted: Optional[torch.Tensor],  # [max_batch] int32; None in non-replay
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    out: Optional[torch.Tensor] = None,
):
    if out is not None:
        return out.new_empty(0)
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


# Flashinfer's selective_state_update kernel only supports these head dimensions
FLASHINFER_SUPPORTED_HEAD_DIMS = [64, 128]


@AttentionRegistry.register("flashinfer_ssm")
class FlashinferBackendSSM(BaseBackendSSM):
    # ssm_replay=True allocates replay buffers at compile time (SM80+), which causes
    # BatchInfo.is_use_replay() to return True at runtime.  Never check ssm_replay at
    # runtime — use batch_info.is_use_replay() instead.
    ssm_replay: bool = False

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_cached_ssm.default

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        ret = super().get_cache_initializers(source_attn_node, cache_config)

        # Replay requires both ssm_replay=True and SM >= 80 (Ampere+).
        # Only when use_replay=True do we skip FlashInfer at runtime; otherwise
        # _flashinfer_ssm_update is called and head_dim must be supported.
        use_replay = cls.ssm_replay and get_sm_version() >= 80

        if not use_replay and ret["ssm_state_cache"].head_dim not in FLASHINFER_SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"flashinfer_ssm only supports head_dim in {FLASHINFER_SUPPORTED_HEAD_DIMS}. "
                f"Got head_dim={ret['ssm_state_cache'].head_dim}. "
                "Consider using 'triton_ssm' backend instead."
            )

        ssm_h = ret["ssm_state_cache"]

        # All 7 optional caches are always registered positionally (None = unused in this mode).
        # intermediate_ssm_state_cache: real in non-replay, None in replay.
        # replay_old_*: real in replay mode (SM80+), None otherwise.
        if use_replay:
            ret["intermediate_ssm_state_cache"] = None
            x_dtype = torch.bfloat16
            B_fake = source_attn_node.args[2].meta["val"]
            n_groups = B_fake.shape[-2] if B_fake.ndim >= 4 else 1
            ret["replay_old_x"] = ReplayOldXHandler(
                num_heads=ssm_h.num_heads, head_dim=ssm_h.head_dim, dtype=x_dtype
            )
            ret["replay_old_b"] = ReplayOldBHandler(
                n_groups=n_groups, d_state=ssm_h.d_state, dtype=x_dtype
            )
            ret["replay_old_dt"] = ReplayOldDtHandler(num_heads=ssm_h.num_heads)
            ret["replay_old_da_cumsum"] = ReplayOldDAcumsumHandler(num_heads=ssm_h.num_heads)
            ret["replay_cache_buf_idx"] = ReplayCacheBufIdxHandler()
            ret["replay_prev_num_accepted"] = ReplayPrevNumAcceptedHandler()
        else:
            ret["intermediate_ssm_state_cache"] = SpecSSMResourceHandler.from_base(ssm_h)
            ret["replay_old_x"] = None
            ret["replay_old_b"] = None
            ret["replay_old_dt"] = None
            ret["replay_old_da_cumsum"] = None
            ret["replay_cache_buf_idx"] = None
            ret["replay_prev_num_accepted"] = None
        return ret
