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

"""TRT-LLM FMHA adapter for the vendored PrimTS native sparse MLA kernels.

Serves the DeepSeek-V4 generation phase on SM100 with the Blackwell
task-scheduled two-source sparse MLA decode kernel
(``BatchSparseMLADecodePagedTSWrapper``, flashinfer-ai/flashinfer#5434). The
sliding-window pool is the primary source and, for compressing layers, the
compressed pool is the extra source. Both lists feed one softmax, exactly like
the TRTLLM-Gen sparse MLA kernel the fallback library runs.

The adapter consumes the dual-pool global indices that
``DeepseekV4TrtllmAttention.sparse_attn_predict`` already produces: SWA slots
are storage rows of the SWA pool and compressed slots are storage rows of the
layer's compress pool, so no page-table conversion happens here. The kernel
takes the two index lists and their live lengths directly and masks ``-1``
holes itself. The fused DSv4 epilogue (conjugate RoPE on ``D[448:512]`` +
1x128 block-scaled E4M3 output in the group-major ``[G, T, (H/G)*512]`` layout
with FP32 scales ``[G, (H/G)*4, pad4(T)]``) maps one-to-one onto the kernel's
``output_quant_block_size=128`` / ``out_scale_format="fp32"`` mode, so the
``o_a_proj`` BMM consumes the attention output without a separate quant
kernel.

The library is enabled by default on SM100 and precedes the TRTLLM-Gen
fallback for the DeepSeek-V4 generation phase; ``TLLM_FMHA_LIBS=-prims_ts_sparse_mla``
restores the previous library. The context phase stays on the fallback library.
SM103 (B300/GB300) is admitted by the kernels' architecture guard but not yet
signoff-qualified upstream, so it keeps the fallback for now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Optional

import torch
from packaging.version import InvalidVersion, Version

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_sm_version
from tensorrt_llm.logger import logger

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
        DeepseekV4CacheManager,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_MIN_CUTLASS_DSL_VERSION = Version("4.7.0")
_LATENT_DIM = 512
_ROPE_DIM = 64
_FUSED_OUTPUT_BLOCK = 128


def get_prims_ts_toolchain_unsupported_reason() -> Optional[str]:
    """Return why the CuTe DSL task-scheduling toolchain cannot run PrimTS kernels."""
    try:
        installed_version = Version(version("nvidia-cutlass-dsl"))
    except (PackageNotFoundError, InvalidVersion):
        return "nvidia-cutlass-dsl>=4.7 is required."
    if installed_version < _MIN_CUTLASS_DSL_VERSION:
        return (
            f"nvidia-cutlass-dsl>={_MIN_CUTLASS_DSL_VERSION} is required, got {installed_version}."
        )
    try:
        import_module("cutlass.experimental.task_scheduling")
    except ImportError:
        return "CUTLASS task scheduling is missing."
    return None


@dataclass
class _SparsePlan:
    """One planned wrapper plus the live-length buffers it reads on every launch.

    The index lists come straight from ``sparse_attn_predict``'s metadata
    buffer; only the per-row live lengths are derived here, into buffers whose
    addresses stay stable for CUDA graph replay.
    """

    wrapper: "BatchSparseMLADecodePagedTSWrapper"
    lengths: torch.Tensor
    extra_lengths: Optional[torch.Tensor]
    validated: bool = False


class PrimsTSSparseMlaFmha(Fmha):
    """DeepSeek-V4 generation-phase sparse MLA on the PrimTS Blackwell kernels."""

    # The native kernel reads one KV head of 512 channels per selected token.
    MAX_HEADS = 128

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        # Plans are keyed by the execution profile that changes kernel
        # selection or buffer geometry: query rows, dtype, sinks, fusion.
        self._plans: dict[tuple, _SparsePlan] = {}
        # FP8 dequantization scales are fixed per layer; the wrapper takes host
        # scalars, so read them once during eager warmup, never under capture.
        self._kv_dequant_scale: Optional[float] = None
        # Pool-base-rooted views of the SWA and compress pools, per manager.
        self._pool_views: dict[tuple[int, int], torch.Tensor] = {}
        self._sinks_cache: Optional[tuple[int, torch.Tensor]] = None

    # ------------------------------------------------------------------ availability
    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if not attn.is_mla_enable:
            return False
        if getattr(getattr(attn, "sparse_params", None), "algorithm", None) != "deepseek_v4":
            return False
        reason = cls._unavailable_reason(attn)
        if reason is not None:
            # A DeepSeek-V4 layer that does not get this library falls back to
            # TRTLLM-Gen silently otherwise; say so once per layer at info.
            logger.info(
                f"PrimTS sparse MLA FMHA is unavailable for layer "
                f"{getattr(attn, 'layer_idx', None)}: {reason}"
            )
            return False
        return True

    @classmethod
    def _unavailable_reason(cls, attn: "TrtllmAttention") -> Optional[str]:
        sm = get_sm_version()
        if sm == 103:
            # The vendored kernels admit SM103 through their architecture guard
            # but flashinfer has only signed off accuracy and performance on
            # SM100 (prims_ts/kernels/mla_decode/README.md). TRT-LLM L0 #64329
            # hit an illegal memory access in the GB300 DeepSeek-V4-Pro perf
            # sanity warmup (TP4/ADP, FP8 KV, MTP 3) that the same configuration
            # at TP8 does not reproduce on B200, so B300/GB300 stay on the
            # TRTLLM-Gen fallback until the kernels are qualified there.
            return "SM103 is not yet qualified for the PrimTS sparse MLA kernels."
        if sm != 100:
            return f"requires SM100, got SM{sm}."
        if getattr(attn, "use_fp8_ds_mla", False) or getattr(attn, "kv_cache_dtype", "auto") == (
            "fp8_ds_mla"
        ):
            return "the footer-scale fp8_ds_mla cache layout is not a native E4M3 pool."
        if getattr(attn, "_uses_nvfp4_compress", False):
            return "NVFP4 compressed pools are not supported."
        if (
            int(attn.kv_lora_rank) + int(attn.qk_rope_head_dim) != _LATENT_DIM
            or int(attn.head_dim) != _LATENT_DIM
        ):
            return (
                "requires a 512-wide latent (kv_lora_rank + qk_rope_head_dim), got "
                f"{attn.kv_lora_rank} + {attn.qk_rope_head_dim} with head_dim {attn.head_dim}."
            )
        if int(attn.qk_rope_head_dim) != _ROPE_DIM:
            return f"requires a {_ROPE_DIM}-wide RoPE segment, got {attn.qk_rope_head_dim}."
        if not 1 <= int(attn.num_heads) <= cls.MAX_HEADS:
            return f"supports 1..{cls.MAX_HEADS} heads, got {attn.num_heads}."
        return get_prims_ts_toolchain_unsupported_reason()

    # ------------------------------------------------------------------ support
    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase=None,
    ) -> bool:
        reason = self._unsupported_reason(q, k, v, metadata, forward_args)
        if reason is not None:
            logger.debug(f"PrimTS sparse MLA FMHA does not support request: {reason}")
            return False
        return True

    def _unsupported_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Optional[str]:
        attn = self.attn
        if fwd.attention_input_type != AttentionInputType.generation_only:
            return "only generation-only requests are served; context stays on the fallback."
        if k is not None or v is not None:
            return "MLA generation takes a fused query only."
        if q.device.type != "cuda" or q.ndim != 2:
            return "the query must be a rank-2 CUDA tensor."
        if metadata.kv_cache_manager is None or not hasattr(
            metadata.kv_cache_manager, "swa_pool_ptr"
        ):
            return "a DeepSeek-V4 KV cache manager is required."
        if metadata.beam_width != 1:
            return "beam search is not supported."
        if getattr(metadata, "token_positions_cuda", None) is None:
            return "token positions are required for lengths and output RoPE."
        if fwd.attention_mask_data is not None or fwd.relative_attention_bias is not None:
            return "custom masks and relative bias are not supported."
        if metadata.helix_position_offsets is not None:
            return "Helix parallelism is not supported."
        rows = int(q.shape[0])
        if rows == 0:
            return "the request contains no generation tokens."
        if int(metadata.num_tokens) - int(metadata.num_ctx_tokens) != rows:
            return "the query rows do not match the generation token count."
        heads = int(attn.num_heads)
        if q.shape[1] != heads * _LATENT_DIM:
            return f"query width must be {heads * _LATENT_DIM}, got {q.shape[1]}."

        indices = fwd.sparse_runtime_params.sparse_attn_indices
        if not isinstance(indices, torch.Tensor):
            return "dual-pool sparse indices are required."
        if indices.ndim != 2 or indices.dtype != torch.int32 or indices.shape[0] != rows:
            return "sparse indices must be int32 [rows, window + compressed]."
        if indices.stride(-1) != 1:
            return "sparse index rows must be contiguous."
        window = int(attn.sparse_attention_config.window_size)
        extra = self._extra_capacity(metadata)
        if indices.shape[1] != window + extra:
            return (
                f"sparse index width {indices.shape[1]} does not match the dual-pool "
                f"layout {window} + {extra}."
            )

        is_fp8 = self._is_fp8_cache(metadata)
        if is_fp8:
            quant_q = fwd.quant_q_buffer
            if quant_q is None or quant_q.dtype not in (torch.uint8, torch.float8_e4m3fn):
                return "FP8 sparse MLA requires the quantized query from MLA preprocessing."
            if not quant_q.is_contiguous() or quant_q.numel() < rows * heads * _LATENT_DIM:
                return "the quantized query buffer is too small or not contiguous."
            if attn.kv_scale_quant_orig is None:
                return "FP8 sparse MLA requires the KV-cache dequantization scale."
        elif q.dtype != torch.bfloat16:
            return f"BF16 KV caches require a BF16 query, got {q.dtype}."

        output = fwd.output
        if output is None:
            return "a preallocated output is required."
        if fwd.enable_dsv4_epilogue_fusion:
            groups = output.shape[0] if output.ndim == 3 else 0
            if (
                output.ndim != 3
                or output.dtype != torch.float8_e4m3fn
                or groups <= 0
                or heads % groups
                or output.shape[1] != rows
                or output.shape[2] != heads // groups * _LATENT_DIM
                or not output.is_contiguous()
            ):
                return "the fused epilogue output must be a contiguous E4M3 [G, T, (H/G)*512]."
            sf = fwd.output_sf
            if (
                sf is None
                or sf.dtype != torch.float32
                or sf.ndim != 3
                or sf.shape[0] != groups
                or sf.shape[1] != heads // groups * (_LATENT_DIM // _FUSED_OUTPUT_BLOCK)
                or sf.shape[2] != (rows + 3) // 4 * 4
                or not sf.is_contiguous()
            ):
                return "the fused epilogue scales must be FP32 [G, (H/G)*4, pad4(T)]."
            cos_sin = fwd.dsv4_inv_rope_cos_sin_cache
            if cos_sin is None or cos_sin.dtype != torch.float32 or cos_sin.numel() % _ROPE_DIM:
                return "the fused epilogue needs the FP32 inverse-RoPE cos/sin table."
        else:
            if output.dtype != torch.bfloat16 or output.numel() != rows * heads * _LATENT_DIM:
                return "the output must be BF16 with one 512-wide row per query head."
            if fwd.out_scale is not None or fwd.output_sf is not None:
                return "unfused quantized attention output is not supported."
        return None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _is_fp8_cache(metadata: "TrtllmAttentionMetadata") -> bool:
        from tensorrt_llm.bindings import DataType

        return metadata.kv_cache_manager.dtype == DataType.FP8

    def _extra_capacity(self, metadata: "DeepseekV4TrtllmAttentionMetadata") -> int:
        ratio = int(self.attn.compress_ratio)
        if ratio <= 1:
            return 0
        return int(metadata.max_compressed_indices[ratio])

    def _pool_view(
        self,
        manager: "DeepseekV4CacheManager",
        attn_type,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Return a ``[rows, 1, 512]`` view of a whole pool rooted at its base pointer.

        ``sparse_attn_predict`` emits storage rows relative to the pool base, so
        the view must start there and span every layer sharing the pool.
        """
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
            DeepseekV4AttentionType,
        )

        key = (id(manager), int(attn_type.value), compress_ratio)
        view = self._pool_views.get(key)
        if view is not None:
            return view
        if attn_type == DeepseekV4AttentionType.SWA:
            base_ptr = manager.swa_pool_ptr
            layers = list(manager.pp_layers)
        else:
            base_ptr = manager.compress_pool_ptrs[compress_ratio]
            layers = [
                layer
                for layer in manager.pp_layers
                if manager._compress_ratios[layer] == compress_ratio
            ]
        total_rows = 0
        for layer in layers:
            buffer = manager.get_buffers(layer, attn_type)
            if buffer.shape[-1] != _LATENT_DIM:
                raise RuntimeError(
                    f"PrimTS sparse MLA expects native 512-wide pool rows, got {tuple(buffer.shape)}."
                )
            offset = buffer.data_ptr() - base_ptr
            row_bytes = _LATENT_DIM * buffer.element_size()
            if offset < 0 or offset % row_bytes:
                raise RuntimeError(
                    f"{attn_type.name} buffer for layer {layer} is not row-aligned to its pool base."
                )
            total_rows = max(total_rows, offset // row_bytes + buffer.shape[0] * buffer.shape[1])
        view = convert_to_torch_tensor(
            TensorWrapper(base_ptr, manager.dtype, (total_rows, 1, _LATENT_DIM))
        )
        self._pool_views[key] = view
        return view

    def _sinks(self, fwd: AttentionForwardArgs) -> Optional[torch.Tensor]:
        sinks = fwd.attention_sinks
        if sinks is None:
            return None
        cached = self._sinks_cache
        if (
            cached is not None
            and cached[0] == sinks.data_ptr()
            and cached[1].numel() == sinks.numel()
        ):
            return cached[1]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS sparse MLA attention sinks must be prepared before CUDA graph capture."
            )
        prepared = sinks.detach().to(torch.float32).reshape(-1).contiguous()
        self._sinks_cache = (sinks.data_ptr(), prepared)
        return prepared

    @staticmethod
    def _softmax_scale(attn: "TrtllmAttention") -> float:
        """TRT-LLM's MLA softmax scale, ``1 / (q_scaling * sqrt(qk_nope + qk_rope))``.

        The scale follows the un-absorbed QK head width (attentionOp.cpp and the
        DeepSeek-V4 fallback), not the 512-wide absorbed latent the kernel
        contracts over. DeepSeek-V4 checkpoints have 448 + 64 = 512, so the two
        only differ for configs such as the 128 + 64 unit-test scenario.
        """
        qk_head_dim = int(attn.qk_nope_head_dim) + int(attn.qk_rope_head_dim)
        return 1.0 / (float(attn.q_scaling) * math.sqrt(float(qk_head_dim)))

    def _kv_dequant(self) -> float:
        if self._kv_dequant_scale is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PrimTS sparse MLA FP8 scales must be cached before CUDA graph capture."
                )
            scale = float(self.attn.kv_scale_quant_orig.view(-1)[0].item())
            if not (math.isfinite(scale) and scale > 0):
                raise RuntimeError("PrimTS sparse MLA FP8 scales must be finite and positive.")
            self._kv_dequant_scale = scale
        return self._kv_dequant_scale

    def _get_plan(
        self,
        *,
        device: torch.device,
        rows: int,
        extra: int,
        q_dtype: torch.dtype,
        has_sinks: bool,
        fused_groups: Optional[int],
    ) -> _SparsePlan:
        from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
            BatchSparseMLADecodePagedTSWrapper,
        )

        attn = self.attn
        heads = int(attn.num_heads)
        window = int(attn.sparse_attention_config.window_size)
        key = (rows, extra, q_dtype, has_sinks, fused_groups)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS sparse MLA plans must be created during eager warmup, not under "
                f"CUDA graph capture (rows={rows}, extra={extra}, dtype={q_dtype})."
            )
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper.plan(
            device,
            rows,
            heads,
            max_topk=window,
            max_extra_topk=extra,
            max_seq_len_q=1,
            q_data_type=q_dtype,
            kv_layout="NHD",
            has_sinks=has_sinks,
            return_lse=False,
            # Window slots and compressed slots are valid prefixes in the
            # dual-pool layout, but the ratio-4 indexer top-k may carry holes;
            # keep the generic mask path.
            assume_valid_prefix=False,
            **(
                dict(
                    output_quant_block_size=_FUSED_OUTPUT_BLOCK,
                    out_scale_format="fp32",
                    num_output_groups=fused_groups,
                )
                if fused_groups
                else {}
            ),
        )
        plan = _SparsePlan(
            wrapper=wrapper,
            lengths=torch.empty((rows,), device=device, dtype=torch.int32),
            extra_lengths=(
                torch.empty((rows,), device=device, dtype=torch.int32) if extra else None
            ),
        )
        self._plans[key] = plan
        return plan

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
            DeepseekV4AttentionType,
        )

        reason = self._unsupported_reason(q, k, v, metadata, forward_args)
        if reason is not None:
            raise RuntimeError(f"PrimTS sparse MLA FMHA cannot run this request: {reason}")
        attn = self.attn
        manager = metadata.kv_cache_manager
        heads = int(attn.num_heads)
        rows = int(q.shape[0])
        start = int(metadata.num_ctx_tokens)
        end = start + rows
        ratio = int(attn.compress_ratio)
        window = int(attn.sparse_attention_config.window_size)
        extra = self._extra_capacity(metadata)
        is_fp8 = self._is_fp8_cache(metadata)
        fused = bool(forward_args.enable_dsv4_epilogue_fusion)
        output = forward_args.output
        groups = int(output.shape[0]) if fused else None

        if is_fp8:
            # The MLA preprocessing quantizes Q with the KV-cache scale, so both
            # descales equal the layer's kv_scale_quant_orig (see
            # DeepseekV4TrtllmAttention.sparse_attn_predict).
            dequant = self._kv_dequant()
            q_scale = kv_scale = dequant
            query = (
                forward_args.quant_q_buffer.view(-1)[: rows * heads * _LATENT_DIM]
                .view(torch.float8_e4m3fn)
                .view(rows, 1, heads, _LATENT_DIM)
            )
        else:
            q_scale = kv_scale = 1.0
            query = q.view(rows, 1, heads, _LATENT_DIM)
        # The kernel takes the combined scales: the softmax scale carries the Q
        # and KV descales, the output scale carries the KV descale (both pools
        # share it). They are per-layer constants, so the wrapper's static-scale
        # kernel serves them without a device scalar.
        softmax_scale = self._softmax_scale(attn) * q_scale * kv_scale
        output_scale = kv_scale

        sinks = self._sinks(forward_args)
        plan = self._get_plan(
            device=q.device,
            rows=rows,
            extra=extra,
            q_dtype=query.dtype,
            has_sinks=sinks is not None,
            fused_groups=groups,
        )

        # Storage-row indices straight from sparse_attn_predict: the window
        # slots address the SWA pool and the compressed slots address the
        # compress pool, both relative to their pool base. The live window
        # length is the token's position within the window; the compressed
        # length is what the indexer's top-k selected beyond the window.
        indices = forward_args.sparse_runtime_params.sparse_attn_indices
        positions = metadata.token_positions_cuda[start:end]
        torch.add(positions, 1, out=plan.lengths)
        plan.lengths.clamp_(max=window)
        swa_pool = self._pool_view(manager, DeepseekV4AttentionType.SWA, 1)
        extra_pool = None
        run_kwargs = {}
        if extra:
            topk_lens = metadata.sparse_mla_topk_lens[ratio][start:end]
            torch.sub(topk_lens, window, out=plan.extra_lengths)
            plan.extra_lengths.clamp_(min=0, max=extra)
            extra_pool = self._pool_view(manager, DeepseekV4AttentionType.COMPRESS, ratio)
            run_kwargs["extra_indices"] = indices[:, window:]
            run_kwargs["extra_lengths"] = plan.extra_lengths

        if fused:
            # TRT-LLM's fused epilogue buffers are group-major [G, T, C] with
            # FP32 scales [G, (H/G)*4, pad4(T)]; the kernel takes their
            # token-major logical views.
            run_kwargs["out"] = output.transpose(0, 1)
            run_kwargs["out_scale"] = forward_args.output_sf.permute(2, 0, 1)[:rows]
            # Positions carry the query prefix shape [rows, 1].
            run_kwargs["token_positions"] = positions.contiguous().view(rows, 1)
            cos_sin = forward_args.dsv4_inv_rope_cos_sin_cache
            run_kwargs["cos_sin_cache"] = cos_sin.reshape(-1, _ROPE_DIM).contiguous()
        else:
            run_kwargs["out"] = output.view(rows, 1, heads, _LATENT_DIM)

        # Validate the index contract once per plan while still eager; graph
        # replay must keep validation off because it synchronizes.
        validate = not plan.validated and not torch.cuda.is_current_stream_capturing()
        plan.wrapper.run(
            query,
            swa_pool,
            indices[:, :window],
            plan.lengths,
            extra_pool,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            sinks=sinks,
            validate=validate,
            **run_kwargs,
        )
        plan.validated = True


__all__ = ["PrimsTSSparseMlaFmha", "get_prims_ts_toolchain_unsupported_reason"]
