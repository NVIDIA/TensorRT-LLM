# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Sparse MLA generation attention for GLM DSA layers with an FP8 latent cache.

The sparse_attention contract: the native ``mla_rope_generation`` has already
rotated and FP8-quantized the query into ``quant_q_buffer`` and appended this
step's latent rows to the paged pool; the native DSA predictor has converted
the top-k selection into global pool rows. This module computes only the
attention over those rows and writes the latent output ``[R, H * 512]`` that
the native W_UV projection consumes.

``SparseMlaDecodeFmha`` is an FMHA library for ``TrtllmAttention``'s library
list. Every eligibility decision is made in ``is_available`` / ``is_supported``,
so an ineligible layer or request falls through to the native library before
any device work. A layer that is structurally outside the domain (not DSA, other
head counts, BF16 cache, another SM family) takes the native path; a selected sparse_mla_decode
whose kernel is missing from the build, or whose device cannot launch it, raises
instead of falling back on one rank.

Rows: uniform generation batches (``seq_len_q`` query rows per sequence) and the
ragged-verification layout, which ``TrtllmAttention.forward`` presents to the
FMHA library as one row per query token with that row's causal KV length
(``presented_token_major``); both give each row the same causal extent.

Index semantics: the native rules (positions at or past a row's causal length
are ignored, repeats count) with one deliberate difference: a negative entry at a
position below the causal length is skipped, while the native generation kernel
reads the pool row that ``-1`` addresses. Native producers of the global indices
(the indexer top-k and the local-to-global remap) never emit such an entry; the
tests check that premise for the native and grouped remaps.

Dispatch accounting: ``engagement_snapshot()`` returns the per-layer static
decisions, the number of host-side calls that ran sparse_mla_decode per layer (eager calls and
CUDA-graph captures; a replay runs the captured kernel without host code) and
the declined selections per reason. Each layer's first static decision and first
sparse_mla_decode call are logged once.
"""

from typing import TYPE_CHECKING, Dict, Optional

import torch

from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._utils import get_sm_version

from ... import dispatch

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

FEATURE_ID = "sparse_mla_decode"
NUM_HEADS = 8
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOP_K = 2048
SUPPORTED_SM_VERSIONS = (100, 103)


def op_available() -> bool:
    """Whether the loaded TensorRT-LLM build registers the sparse_mla_decode kernel."""
    return hasattr(torch.ops.trtllm, "glm_moe_dsa_sparse_mla_decode")


_DEVICE_SUPPORT: Dict[int, bool] = {}

# Dispatch accounting of this process (see the module docstring).
_STATIC_DECISIONS: Dict[int, str] = {}
_ENGAGED_CALLS: Dict[int, int] = {}
_DECLINED_SELECTIONS: Dict[str, int] = {}


def device_supported(device_index: int) -> bool:
    """Whether the device can launch the 16-CTA cluster kernel (cached per device)."""
    if device_index not in _DEVICE_SUPPORT:
        _DEVICE_SUPPORT[device_index] = bool(
            op_available()
            and get_sm_version() in SUPPORTED_SM_VERSIONS
            and torch.ops.trtllm.glm_moe_dsa_sparse_mla_decode_supported(device_index)
        )
    return _DEVICE_SUPPORT[device_index]


def engagement_snapshot() -> dict:
    """Copy of this process's dispatch accounting.

    Returns:
        ``static``: layer index -> "" when the layer may use sparse_mla_decode, else the reason;
        ``engaged_calls``: layer index -> host-side calls that ran sparse_mla_decode;
        ``declined_selections``: reason -> uncached library selections that declined.
    """
    return {
        "static": dict(_STATIC_DECISIONS),
        "engaged_calls": dict(_ENGAGED_CALLS),
        "declined_selections": dict(_DECLINED_SELECTIONS),
    }


def sparse_mla_decode(
    q_fp8: torch.Tensor,
    kv_pool_ptr: int,
    indices: torch.Tensor,
    bmm1_scale: torch.Tensor,
    bmm2_scale: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    seq_len_q: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run sparse MLA attention for ``R`` query rows; writes and returns ``out``.

    The rounding and masking rules are those of the native generation FMHA for
    this layout (see ``sparseMlaDecode.h``): index-list positions at or past a
    row's causal KV length are ignored, negative entries are skipped, repeated
    rows count per occurrence, probabilities enter P*V as ``e4m3(448 * p)``.

    Args:
        q_fp8: ``[R, 8, 576]`` e4m3 query bytes (``quant_q_buffer``).
        kv_pool_ptr: Device address of the FP8 latent pool base.
        indices: ``[R, 2048]`` int32 global pool rows.
        bmm1_scale: ``[2]`` float32 device scales from ``mla_rope_generation``.
        bmm2_scale: ``[1]`` float32 device scale from ``mla_rope_generation``.
        seq_lens_kv: ``[R / seq_len_q]`` int32 KV lengths including this step's rows.
        seq_len_q: Query rows per sequence.
        out: ``[R, 8 * 512]`` BF16 output.
    """
    torch.ops.trtllm.glm_moe_dsa_sparse_mla_decode(
        q_fp8, kv_pool_ptr, indices, bmm1_scale, bmm2_scale, seq_lens_kv, seq_len_q, out
    )
    return out


def pool_base_pointer(metadata: "TrtllmAttentionMetadata", local_layer_idx: int) -> int:
    """Pool base address the native sparse MLA path uses for this layer."""
    pool_index = int(metadata.host_kv_cache_pool_mapping[local_layer_idx, 0])
    pointers = metadata.host_kv_cache_pool_pointers
    base = pointers[pool_index, 0, 0] if pointers.dim() == 3 else pointers[pool_index, 0]
    return int(base)


def static_ineligibility(attn: "TrtllmAttention") -> Optional[str]:
    """Return why ``attn`` can never use sparse_mla_decode, or None if it may."""
    if getattr(attn.sparse_params, "algorithm", None) != "dsa":
        return "not a DSA layer"
    if not attn.is_mla_enable:
        return "not MLA"
    if attn.num_heads != NUM_HEADS or attn.num_kv_heads != 1:
        return f"local heads {attn.num_heads}/{attn.num_kv_heads}"
    if attn.kv_lora_rank != KV_LORA_RANK or attn.qk_rope_head_dim != QK_ROPE_HEAD_DIM:
        return "latent dims"
    if not attn.rope_append:
        return "rope_append disabled"
    if not getattr(attn, "has_fp8_kv_cache", False) or getattr(attn, "has_fp4_kv_cache", False):
        return "KV cache is not FP8"
    if getattr(attn, "kv_cache_dtype", None) == "fp8_ds_mla":
        return "fp8_ds_mla cache layout"
    if attn.attention_chunk_size is not None:
        return "chunked attention"
    if get_sm_version() not in SUPPORTED_SM_VERSIONS:
        return f"SM {get_sm_version()}"
    # The layer is in the domain from here on: a missing kernel or a failed device
    # query is an error of the selected candidate, not a per-rank fallback.
    if not op_available():
        raise RuntimeError(
            "the plan enables sparse_mla_decode, but this "
            "TensorRT-LLM build does not register "
            "torch.ops.trtllm.glm_moe_dsa_sparse_mla_decode"
        )
    device = torch.cuda.current_device()
    if not device_supported(device):
        raise RuntimeError(
            f"the plan enables sparse_mla_decode, but device "
            f"{device} (SM {get_sm_version()}) cannot launch its 16-CTA cluster kernel: the "
            "cluster occupancy query failed or returned no active cluster"
        )
    return None


def request_ineligibility(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata: "TrtllmAttentionMetadata",
    forward_args: AttentionForwardArgs,
) -> Optional[str]:
    """Return why this request must take the native path, or None."""
    if forward_args.attention_input_type != AttentionInputType.generation_only:
        return "not generation-only"
    if k is not None or v is not None:
        return "separate K/V"
    if metadata.num_generations <= 0:
        return "no generation rows"
    if metadata.effective_beam_width != 1:
        return "beam search"
    if metadata.helix_position_offsets is not None:
        return "helix context parallelism"
    if metadata.is_spec_dec_tree:
        return "tree speculative decoding"
    if metadata.num_sparse_topk != TOP_K:
        return f"top-k {metadata.num_sparse_topk}"
    if q.dim() != 2 or q.shape[1] != NUM_HEADS * HEAD_DIM_QK:
        return "query shape"
    num_rows = q.shape[0]
    if (
        forward_args.quant_q_buffer is None
        or forward_args.mla_bmm1_scale is None
        or forward_args.mla_bmm2_scale is None
    ):
        return "no FP8 query"
    if forward_args.quant_scale_qkv is not None:
        return "pre-quantized query path"
    if forward_args.quant_q_buffer.numel() != num_rows * NUM_HEADS * HEAD_DIM_QK:
        return "query buffer shape"
    output = forward_args.output
    if (
        output is None
        or output.dtype != torch.bfloat16
        or forward_args.output_sf is not None
        or not output.is_contiguous()
        or output.numel() != num_rows * NUM_HEADS * KV_LORA_RANK
    ):
        return "output format"
    if forward_args.softmax_stats_tensor is not None or forward_args.attention_sinks is not None:
        return "softmax statistics or sinks requested"
    if forward_args.enable_dsv4_epilogue_fusion:
        return "epilogue fusion"
    sparse = forward_args.sparse_runtime_params
    indices = sparse.sparse_attn_indices
    if indices is None or indices.dtype != torch.int32 or indices.dim() != 2:
        return "no sparse indices"
    if indices.shape[0] != num_rows or indices.shape[1] != TOP_K or indices.stride(1) != 1:
        return "sparse index layout"
    if (
        sparse.sparse_attn_kv_lens is not None
        or sparse.aux_kv_cache_pool_ptr is not None
        or sparse.sparse_kv_indices is not None
    ):
        return "dynamic or auxiliary-pool sparse attention"
    # KV lengths per generation sequence, this step's rows included. Under the
    # ragged-verification presentation every query token is its own sequence
    # (seq_len_q 1) carrying its causal length, so the same rule applies.
    kv_lens = metadata.kv_lens_cuda_runtime
    if kv_lens.dtype != torch.int32 or not kv_lens.is_contiguous():
        return "KV length layout"
    gen_seqs = kv_lens.shape[0] - metadata.num_contexts
    if gen_seqs <= 0 or num_rows % gen_seqs != 0:
        return "rows do not tile the generation sequences"
    return None


class SparseMlaDecodeFmha(Fmha):
    """Generation-only sparse MLA library for GLM DSA layers with an FP8 cache."""

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        reason = static_ineligibility(attn)
        _STATIC_DECISIONS[attn.layer_idx] = reason or ""
        return dispatch.log_capability(
            FEATURE_ID, f"layer{attn.layer_idx}", reason is None, reason or ""
        )

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase is not None and phase != FmhaPhase.GENERATION:
            return False
        reason = request_ineligibility(q, k, v, metadata, forward_args)
        if reason is not None:
            _DECLINED_SELECTIONS[reason] = _DECLINED_SELECTIONS.get(reason, 0) + 1
            dispatch.log_capability(
                FEATURE_ID,
                f"request:{reason}",
                False,
                f"{reason}; first seen at layer {self.attn.layer_idx} with {q.shape[0]} rows",
            )
            return False
        return True

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        attn = self.attn
        num_rows = q.shape[0]
        gen_lens = metadata.kv_lens_cuda_runtime[metadata.num_contexts :]
        seq_len_q = num_rows // gen_lens.shape[0]
        calls = _ENGAGED_CALLS.get(attn.layer_idx, 0)
        _ENGAGED_CALLS[attn.layer_idx] = calls + 1
        if calls == 0:
            ragged = metadata.token_major_gen_view() is not None
            dispatch.log_capability(
                FEATURE_ID,
                f"call:layer{attn.layer_idx}",
                True,
                f"first call: {num_rows} rows, seq_len_q {seq_len_q}, ragged {ragged}",
            )
        sparse_mla_decode(
            forward_args.quant_q_buffer.view(num_rows, NUM_HEADS, HEAD_DIM_QK),
            pool_base_pointer(metadata, attn.local_layer_idx),
            forward_args.sparse_runtime_params.sparse_attn_indices,
            forward_args.mla_bmm1_scale,
            forward_args.mla_bmm2_scale,
            gen_lens,
            seq_len_q,
            forward_args.output,
        )
