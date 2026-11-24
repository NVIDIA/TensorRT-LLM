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
"""Kimi K3 MLA cached-KV context: FP8 Q / K / V straight into the attention op.

The base ``MLA`` cached-KV context paths (``forward_context_with_cached_kv``,
``forward_context_with_chunked_prefill`` and, for an NVFP4 latent cache,
``forward_context_with_chunked_prefill_fp4``) expand every latent KV partition through
``kv_b_proj`` into a BF16 K ``[T, H, qk_nope + qk_rope]`` (a concat) and V, and the attention
op's FP8 context MLA FMHA converts Q, K and V to FP8 on every call, i.e. once per KV partition.
:class:`KimiK3MlaContextFp8Mixin` overrides the base hook ``_cached_kv_context_inputs``: one
Triton pass (:func:`..kernels.fp8_static_quant.mla_kv_to_fp8`) writes contiguous FP8 K and V
from the kv_b_proj output and k_pe, Q is quantized once per layer call, and the op reads them as
they are. The bytes are the op's own unit-scale conversion of the same BF16 values, so the
attention output is unchanged.

* ``TRTLLM_K3_MLA_CTX_FP8_KV_FUSED=1``: MLA layers with an FP8 KV cache (the K3 FP8-KV layers,
  ``TLLM_KIMI_K3_FP8_KV_LAYERS``).
* ``TRTLLM_K3_MLA_CTX_TRITON_QUANT=1``: MLA layers with an NVFP4 latent cache, whose cached prefix
  runs partition by partition through the FP8 context MLA FMHA. Where ``ctx_split_k`` runs that
  cached prefix on the Rubin split-K context FMHA instead (SM107), this hook is not called and
  the switch selects the split-K path's Triton Q / K-pe quantization.

Both default to 0. They need an attention op that accepts FP8 K / V
(``torch.ops.trtllm.attention_supports_mla_context_fp8_kv_input``); with a C++ library built
without it the switches log one warning and the layers keep the BF16 K / V path.
"""

from __future__ import annotations

import functools
import math
from typing import Callable, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..features import active as _features
from ..kernels import fp8_static_quant

# The K3Features fields of the two paths, as log lines name them.
_TRITON_QUANT_FEATURE = "mla_ctx_triton_quant"
_FP8_KV_FUSED_FEATURE = "mla_ctx_fp8_kv_fused"
# SM versions on which the attention op runs the FP8 context MLA FMHA (``mFP8ContextMLA``).
_FP8_CONTEXT_MLA_SMS = (90, 100, 103, 107, 120)
# Head dimensions of the op's FP8 context copy kernel and of mla_kv_to_fp8's callers.
_HEAD_DIMS = (128, 64, 128)
# Heads per mla_kv_to_fp8 program (its tuned default), reduced to a divisor of the rank's heads.
_MLA_KV_HEADS_PER_PROGRAM = 4


def _mla_ctx_triton_quant_enabled() -> bool:
    """features.mla_ctx_triton_quant: Triton FP8 Q / K / V for the NVFP4-cache MLA context.

    Every cached-prefix partition of FP4 MLA chunked prefill runs the FP8 context MLA FMHA; Q is
    quantized once per layer and each partition's K / V straight from the kv_b_proj output by
    the Triton kernels in ``kernels/fp8_static_quant.py`` instead of the attention op's per-call
    copy. The bytes are identical.
    """
    return _features().mla_ctx_triton_quant


def _mla_ctx_fp8_kv_fused_enabled() -> bool:
    """features.mla_ctx_fp8_kv_fused: FP8 Q / K / V into the attention op for cached-KV context.

    MLA layers with an FP8 KV cache (the K3 FP8-KV layers) run the attention op's FP8 context
    FMHA, which copies its BF16 Q, K and V to FP8 on every call, i.e. once per cached-KV
    partition. With the feature one Triton pass writes FP8 K and V from the kv_b_proj output and
    k_pe (no BF16 [T, H, qk_head_dim] concat), Q is quantized once per layer, and the op reads
    them as they are. The FP8 bytes, and so the attention output, are the ones of the default
    path.
    """
    return _features().mla_ctx_fp8_kv_fused


_unit_scales: dict[int, torch.Tensor] = {}


def _unit_scale(device: torch.device) -> torch.Tensor:
    """One-element FP32 1.0 on ``device``, cached (a static FP8 scale that leaves values as they are)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    scale = _unit_scales.get(index)
    if scale is None:
        scale = torch.ones(1, dtype=torch.float32, device=device)
        _unit_scales[index] = scale
    return scale


@functools.cache
def attention_op_accepts_fp8_kv() -> bool:
    """Whether the loaded attention op takes FP8 K / V (and an FP8 Q through ``quant_q_buffer``)
    in the non-absorption MLA context.

    An op without that support reads FP8 K / V as BF16, so the switches must not reach it.
    """
    probe = getattr(torch.ops.trtllm, "attention_supports_mla_context_fp8_kv_input", None)
    if probe is not None and bool(probe()):
        return True
    logger.warning(
        f"{_FP8_KV_FUSED_FEATURE} / {_TRITON_QUANT_FEATURE}: the loaded TensorRT-LLM C++ library has no "
        "FP8 K / V input for the MLA context attention op "
        "(torch.ops.trtllm.attention_supports_mla_context_fp8_kv_input is missing); rebuild it. "
        "MLA context attention keeps the BF16 K / V path."
    )
    return False


_fp8_kv_attention_inputs_warmed: set = set()


class KimiK3MlaContextFp8Mixin:
    """FP8 Q / K / V for the cached-KV context paths of the base ``MLA`` (see the module doc).

    Mixed in before ``MLA``; it only overrides ``_cached_kv_context_inputs`` and wraps
    ``forward_context`` to compile the Triton kernels during the context warmup.
    """

    def _use_fp8_kv_attention_inputs(self, q: torch.Tensor) -> bool:
        """Whether this layer call hands the attention op FP8 Q / K / V.

        Needs dense MLA on the op's FP8 context MLA FMHA (an FP8 KV cache under
        TRTLLM_K3_MLA_CTX_FP8_KV_FUSED, an NVFP4 latent cache under
        TRTLLM_K3_MLA_CTX_TRITON_QUANT) without Helix CP, a contiguous BF16 Q, the 128 / 64 / 128
        head dimensions of the op's FP8 copy kernel, and an op that accepts FP8 K / V.
        """
        mha = self.mha
        if not isinstance(mha, TrtllmAttention) or mha.sparse_params is not None:
            return False
        if mha.uses_fp4_mla_attention:
            enabled = _mla_ctx_triton_quant_enabled()
        elif mha.has_fp8_kv_cache:
            enabled = _mla_ctx_fp8_kv_fused_enabled()
        else:
            return False
        return (
            enabled
            and not self.mapping.has_cp_helix()
            and get_sm_version() in _FP8_CONTEXT_MLA_SMS
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
            and (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) == _HEAD_DIMS
            and attention_op_accepts_fp8_kv()
        )

    def _quantize_q_fp8(self, q: torch.Tensor) -> torch.Tensor:
        """Q in FP8 as the attention op's dense MLA context path quantizes it (unit scale)."""
        return fp8_static_quant.static_quantize_e4m3(q, _unit_scale(q.device))

    def _expand_kv_to_fp8_k_v(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_b_proj + RoPE concat + FP8 copy -> E4M3 (k [T, H*qk_head_dim], v [T, H*v_head_dim]).

        The kv_b_proj GEMM of the base ``_expand_kv_to_k_v``, then one Triton pass writes K (each
        head's k_nope and the shared k_pe) and V in FP8, contiguous, as the attention op's FP8
        context FMHA reads them.
        """
        num_tokens = compressed_kv.shape[0]
        return self._mla_kv_to_fp8(
            self.kv_b_proj(compressed_kv), k_pe.view(num_tokens, self.qk_rope_head_dim)
        )

    def _mla_kv_to_fp8(
        self, kv: torch.Tensor, k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fp8_static_quant.mla_kv_to_fp8(
            kv,
            k_pe,
            self.num_heads_tp,
            self.qk_nope_head_dim,
            self.v_head_dim,
            heads_per_program=math.gcd(self.num_heads_tp, _MLA_KV_HEADS_PER_PROGRAM),
        )

    def _cached_kv_context_inputs(
        self, q: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ]:
        if not self._use_fp8_kv_attention_inputs(q):
            return super()._cached_kv_context_inputs(q)
        if self.mha.uses_fp4_mla_attention:
            kind, switch = "NVFP4", _TRITON_QUANT_FEATURE
        else:
            kind, switch = "FP8", _FP8_KV_FUSED_FEATURE
        logger.info_once(
            f"Kimi K3 MLA cached-KV context: FP8 Q / K / V into the attention op for {kind} "
            f"KV-cache layers ({switch}).",
            key=("k3_mla_ctx_fp8_kv_inputs", kind),
        )
        # Q is final here (RoPE applied); every KV partition of this call reuses its FP8 copy.
        return self._quantize_q_fp8(q), self._expand_kv_to_fp8_k_v

    def _warmup_fp8_kv_attention_inputs(self, q: torch.Tensor) -> None:
        """Compile the FP8 K / V and Q kernels once per process, so the first context step that
        reads cached KV does not pay the Triton compile."""
        key = (self.num_heads_tp, q.device)
        if key in _fp8_kv_attention_inputs_warmed:
            return
        _fp8_kv_attention_inputs_warmed.add(key)
        num_tokens = 2
        latent = q.new_zeros(num_tokens, self.kv_lora_rank + self.qk_rope_head_dim)
        kv = q.new_zeros(num_tokens, self.num_heads_tp * (self.qk_nope_head_dim + self.v_head_dim))
        # Chunk loads give a contiguous k_pe; the current chunk slices latent_cache.
        for k_pe in (latent[:, self.kv_lora_rank :].contiguous(), latent[:, self.kv_lora_rank :]):
            self._mla_kv_to_fp8(kv, k_pe)
        self._quantize_q_fp8(q.new_zeros(num_tokens, self.num_heads_tp * self.qk_head_dim))

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mha = self.mha
        if (
            isinstance(mha, TrtllmAttention)
            and isinstance(attn_metadata, TrtllmAttentionMetadata)
            and mha.has_cached_kv_for_mla_context_warmup(attn_metadata)
            and self._use_fp8_kv_attention_inputs(q)
        ):
            self._warmup_fp8_kv_attention_inputs(q)
        return super().forward_context(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )


__all__ = ["KimiK3MlaContextFp8Mixin", "attention_op_accepts_fp8_kv"]
