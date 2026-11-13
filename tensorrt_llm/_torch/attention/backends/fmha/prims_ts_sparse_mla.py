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
layer's compress pool, so no page-table conversion happens here. With KV-cache
offload, a ratio-4 layer's compressed slots are rows of its sparse cache arena
instead, and the arena is the extra source. The kernel
takes the two index lists and their live lengths directly; ``-1`` holes before a
live length are masked in the kernel only on the configurations that can produce
them (KV-cache offload, heuristic top-k), otherwise the plan promises valid
prefixes. The fused DSv4 epilogue (conjugate RoPE on ``D[448:512]`` +
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

import functools
import math
import time
import weakref
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
# Same compiler-target gate as fmha/prims_ts.py: the kernels share one CuTe DSL
# task-scheduling stack, so a toolchain the dense libraries refuse is refused here too.
_MIN_CUTLASS_COMPILER_VERSION = "13.3"
# Index contracts validated by an eager launch in this process.
_VALIDATED_CONTRACTS: set = set()
_LATENT_DIM = 512
# Prepared-sinks cache entries (one per layer; DeepSeek-V4 has 61).
_SINKS_CACHE_LIMIT = 512
_ROPE_DIM = 64
_FUSED_OUTPUT_BLOCK = 128


@functools.cache
def _workspace_size(device: torch.device, rows: int, heads: int, **plan_kwargs) -> int:
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        get_prims_ts_sparse_mla_decode_workspace_size,
    )

    return int(get_prims_ts_sparse_mla_decode_workspace_size(device, rows, heads, **plan_kwargs))


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
        cutlass = import_module("cutlass")
        compiler_version_supported = cutlass.target_version(
            min_version=_MIN_CUTLASS_COMPILER_VERSION
        )
    except Exception as error:  # noqa: BLE001 - availability probes must fail closed
        return f"could not query the active CUTLASS compiler version: {error}"
    if not compiler_version_supported:
        return f"the active CUTLASS compiler must target CUDA>={_MIN_CUTLASS_COMPILER_VERSION}."
    try:
        import_module("cutlass.experimental.task_scheduling")
    except ImportError:
        return "CUTLASS task scheduling is missing."
    return None


@dataclass
class _SparsePlan:
    """One planned wrapper for a static profile; every launch input comes from the metadata."""

    wrapper: "BatchSparseMLADecodePagedTSWrapper"


class PrimsTSSparseMlaFmha(Fmha):
    """DeepSeek-V4 generation-phase sparse MLA on the PrimTS Blackwell kernels."""

    # The native kernel reads one KV head of 512 channels per selected token.
    MAX_HEADS = 128
    # Plans live in ``TrtllmAttentionMetadata.fmha_plan_caches`` under this key,
    # like the block-sparse and FlashInfer wrappers: every layer that runs with
    # one metadata object plans each static profile once, and layers sharing a
    # metadata object execute serially. All plans of a metadata object share
    # one kernel workspace (under WORKSPACE_CACHE_KEY) that is grown to the
    # largest profile planned so far (a plan keeps the buffer it was planned
    # with alive until the metadata object is dropped); warmup loads the plan
    # with the largest workspace first so the metadata allocates it once.
    PLAN_CACHE_KEY = "prims_ts_sparse_mla"
    WORKSPACE_CACHE_KEY = "prims_ts_sparse_mla.workspace"

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        # FP8 dequantization scales are fixed per layer; the wrapper takes host
        # scalars, so read them once during eager warmup, never under capture.
        self._kv_dequant_scale: Optional[float] = None
        # Pool-base-rooted views of the SWA and compress pools, owned by the
        # manager they were built from. The owner is held weakly: a freed
        # manager (or a new one at the same address) never resolves to a
        # stale view of released pool memory.
        self._checked_key = None
        self._pool_views_owner: Optional[weakref.ReferenceType] = None
        self._pool_views: dict[tuple[int, int], torch.Tensor] = {}
        # One prepared FP32 sinks vector per source tensor (one per layer).
        self._sinks_cache: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}

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
            # TRTLLM-Gen silently otherwise; say so once per reason at info.
            logger.info_once(
                f"PrimTS sparse MLA FMHA is unavailable, DeepSeek-V4 generation stays on the "
                f"TRTLLM-Gen fallback: {reason}",
                key=f"prims_ts_sparse_mla_unavailable:{reason}",
            )
            return False
        return True

    @classmethod
    def _unavailable_reason(cls, attn: "TrtllmAttention") -> Optional[str]:
        sm = get_sm_version()
        if sm == 103:
            # The kernels' architecture guard admits SM103, but they are only
            # signed off on SM100 (prims_ts/kernels/mla_decode/README.md); keep
            # B300/GB300 on the TRTLLM-Gen fallback until they are qualified.
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
        if self._reads_sparse_arena(metadata) and (
            attn.layer_idx not in metadata.sparse_offload_state.layers
            or not hasattr(metadata.kv_cache_manager, "get_sparse_compress_buffers")
        ):
            return "the offloaded layer has no sparse cache arena."
        if metadata.beam_width != 1:
            return "beam search is not supported."
        if getattr(metadata, "token_positions_cuda", None) is None:
            return "token positions are required for the output RoPE."
        if getattr(metadata, "sparse_mla_swa_lens", None) is None:
            return "per-step live lengths are required (DeepSeek-V4 metadata)."
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

    def _reads_sparse_arena(self, metadata: "TrtllmAttentionMetadata") -> bool:
        """Whether this layer's compressed rows sit in the sparse cache arena.

        KV-cache offload covers the ratio-4 layers only; the others keep their
        compress pool. Only DeepSeek-V4 metadata defines ``sparse_offload_state``.
        """
        return (
            int(self.attn.compress_ratio) == 4
            and getattr(metadata, "sparse_offload_state", None) is not None
        )

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

        owner = self._pool_views_owner() if self._pool_views_owner is not None else None
        if owner is not manager:
            self._pool_views.clear()
            self._pool_views_owner = weakref.ref(manager)
        key = (int(attn_type.value), compress_ratio)
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
        if (
            sinks.dtype == torch.float32
            and sinks.dim() == 1
            and sinks.is_contiguous()
            and not sinks.requires_grad
        ):
            # DeepseekV4TrtllmAttention.attn_sink is already a per-head FP32
            # parameter; pass it through so its address stays stable per layer.
            return sinks
        key = (sinks.data_ptr(), sinks.numel(), sinks.dtype)
        prepared = self._sinks_cache.get(key)
        if prepared is not None:
            return prepared
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS sparse MLA attention sinks must be prepared before CUDA graph capture."
            )
        prepared = sinks.detach().to(torch.float32).reshape(-1).contiguous()
        if len(self._sinks_cache) >= _SINKS_CACHE_LIMIT:
            self._sinks_cache.clear()
        self._sinks_cache[key] = prepared
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
        metadata: "TrtllmAttentionMetadata",
        device: torch.device,
        rows: int,
        extra: int,
        q_dtype: torch.dtype,
        has_sinks: bool,
        fused_groups: Optional[int],
    ) -> _SparsePlan:
        """Return the metadata-cached plan for this static profile, planning it once.

        The key holds every value that changes kernel selection or buffer
        geometry: device, query rows, extra capacity, heads, window, dtype,
        sinks, fusion groups. One plan serves every layer with that profile.
        """
        from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

        attn = self.attn
        heads = int(attn.num_heads)
        window = int(attn.sparse_attention_config.window_size)
        key = (device, rows, extra, heads, window, q_dtype, has_sinks, fused_groups)
        cache = metadata.fmha_plan_caches.setdefault(self.PLAN_CACHE_KEY, {})
        plan = cache.get(key)
        if plan is not None:
            return plan
        t_plan = time.perf_counter()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS sparse MLA plans must be created during eager warmup, not under "
                f"CUDA graph capture (rows={rows}, extra={extra}, dtype={q_dtype})."
            )
        plan_kwargs, _ = self._plan_kwargs(extra, q_dtype, has_sinks, fused_groups)
        # One workspace per metadata object, shared by every plan and grown to
        # the largest profile: plans run serially, and a plan's scratch is only
        # live during its own launch.
        required_bytes = _workspace_size(device, rows, heads, **plan_kwargs)
        shared = metadata.fmha_plan_caches.setdefault(self.WORKSPACE_CACHE_KEY, {})
        workspace = shared.get("buffer")
        if workspace is None or workspace.numel() < required_bytes:
            workspace = torch.empty((required_bytes,), device=device, dtype=torch.uint8)
            shared["buffer"] = workspace
        wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper(workspace_buffer=workspace)
        wrapper.plan(device, rows, heads, **plan_kwargs)
        plan = _SparsePlan(wrapper=wrapper)
        cache[key] = plan
        logger.debug(
            f"PrimTS sparse MLA plan rows={rows} extra={extra} sinks={has_sinks} "
            f"fused={fused_groups}: {(time.perf_counter() - t_plan) * 1e3:.1f} ms "
            f"({len(cache)} plans on this metadata)"
        )
        return plan

    def _holes_possible(self) -> bool:
        """Whether -1 can appear inside a live index prefix with this layer's config
        (KV-cache offload masks non-resident pages in place; the heuristic GVR top-k
        pads from its own candidate count); read directly so a renamed field cannot
        silently fail open."""
        sparse_cfg = self.attn.sparse_attention_config
        return bool(sparse_cfg.enable_kv_cache_offload) or bool(sparse_cfg.enable_heuristic_topk)

    def _plan_kwargs(
        self, extra: int, q_dtype: torch.dtype, has_sinks: bool, fused_groups: Optional[int]
    ) -> tuple[dict, bool]:
        """Return the wrapper plan() kwargs for a static profile and whether -1 holes
        can appear inside a live index prefix.

        DeepSeek-V4's producers emit valid prefixes: the window slots are the
        first min(pos + 1, window) rows, the compressed slots the first
        `topk_lens - window` rows (ratio 128 by construction, ratio 4 from the
        indexer's top-k), with -1 only in the tail. Promising that lets the
        kernel derive its masks from the live lengths instead of loading and
        checking every index. KV-cache offload breaks the promise (non-resident
        pages are masked to -1 in place), and the opt-in heuristic (GVR) top-k
        engines pad from their own candidate count, so both keep the generic
        hole-checking path. The fields are read directly so a renamed config
        cannot silently fail open.
        """
        holes_possible = self._holes_possible()
        plan_kwargs = dict(
            max_topk=int(self.attn.sparse_attention_config.window_size),
            max_extra_topk=extra,
            max_q_len=1,
            q_dtype=q_dtype,
            kv_layout="NHD",
            use_attention_sinks=has_sinks,
            return_lse=False,
            assume_valid_prefix=not holes_possible,
        )
        if fused_groups:
            plan_kwargs.update(
                output_quant_block_size=_FUSED_OUTPUT_BLOCK,
                out_scale_format="fp32",
                num_output_groups=fused_groups,
            )
        return plan_kwargs, holes_possible

    def _kernel_variant_rows(
        self, metadata: "TrtllmAttentionMetadata", device: torch.device, plan_kwargs: dict
    ) -> list[int]:
        """Plan every reachable row count without compiling and retain unique specs."""
        from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

        requests = int(metadata.max_num_requests or 0)
        draft = int(metadata.max_total_draft_tokens or 0)
        tokens = int(metadata.max_num_tokens or 0)
        max_rows = requests * (1 + draft)
        if tokens:
            max_rows = min(max_rows, tokens) if max_rows else tokens
        if not max_rows:
            return []
        heads = int(self.attn.num_heads)
        sizes = {
            rows: _workspace_size(device, rows, heads, **plan_kwargs)
            for rows in range(1, max_rows + 1)
        }
        workspace = torch.empty(max(sizes.values()), device=device, dtype=torch.uint8)
        wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper(workspace_buffer=workspace)
        representatives: dict[object, int] = {}
        for rows in sizes:
            wrapper.plan(device, rows, heads, compile=False, **plan_kwargs)
            spec = wrapper._state["compile_spec"]
            previous = representatives.get(spec)
            # Split counts can change with rows. Keep the largest workspace,
            # breaking ties by rows, so later local loading allocates it once.
            if previous is None or (sizes[rows], rows) > (sizes[previous], previous):
                representatives[spec] = rows
        return sorted(representatives.values(), key=lambda rows: (sizes[rows], rows), reverse=True)

    @staticmethod
    def _claim_contract_validation(contract_key: tuple) -> bool:
        """Return True exactly once per process for an index contract (never under capture).

        The contract (valid prefixes, indices inside the pools) is a property
        of the index producers and the pools they address, so one validated
        eager launch per process and contract is enough; validating every
        plan's first launch re-synchronized the host with a GPU that was busy
        with prefill on the context-ramp steps that created new plans.
        """
        if torch.cuda.is_current_stream_capturing() or contract_key in _VALIDATED_CONTRACTS:
            return False
        _VALIDATED_CONTRACTS.add(contract_key)
        logger.debug(f"PrimTS sparse MLA validating the index contract {contract_key}")
        return True

    # ------------------------------------------------------------------ forward
    def _check_request(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> None:
        """Run the full request check once per request geometry.

        The manager selected this library through is_supported(); the re-check
        here guards a changed request, so it runs only when the per-step
        geometry (metadata object, query, indices, output, fusion mode, token
        counts) differs from the last accepted call.
        """
        indices = fwd.sparse_runtime_params.sparse_attn_indices
        output = fwd.output
        check_key = (
            id(metadata),
            int(metadata.num_tokens),
            int(metadata.num_ctx_tokens),
            q.shape,
            q.dtype,
            k is None,
            v is None,
            None
            if not isinstance(indices, torch.Tensor)
            else (indices.shape, indices.dtype, indices.stride(-1)),
            None if output is None else (output.shape, output.dtype, output.is_contiguous()),
            fwd.enable_dsv4_epilogue_fusion,
            fwd.quant_q_buffer is None,
            fwd.attention_sinks is None,
            fwd.output_sf is None,
            fwd.out_scale is None,
            fwd.dsv4_inv_rope_cos_sin_cache is None,
        )
        if check_key == self._checked_key:
            return
        reason = self._unsupported_reason(q, k, v, metadata, fwd)
        if reason is not None:
            raise RuntimeError(f"PrimTS sparse MLA FMHA cannot run this request: {reason}")
        self._checked_key = check_key

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

        self._check_request(q, k, v, metadata, forward_args)
        indices = forward_args.sparse_runtime_params.sparse_attn_indices
        output = forward_args.output
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
        groups = int(output.shape[0]) if fused else None

        if is_fp8:
            # Mirror the fallback's scheduler prologue, which sets both
            # dequant_scale_q and dequant_scale_kv to the layer's
            # kv_scale_quant_orig (DeepseekV4TrtllmAttention.sparse_attn_predict).
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
        profile = dict(
            metadata=metadata,
            device=q.device,
            extra=extra,
            q_dtype=query.dtype,
            has_sinks=sinks is not None,
            fused_groups=groups,
        )
        plan = self._get_plan(rows=rows, **profile)

        # Storage-row indices straight from sparse_attn_predict: the window
        # slots address the SWA pool and the compressed slots address the
        # compress pool, both relative to their pool base. The live lengths
        # come from the metadata too, derived once per step: the token's
        # position within the window, and the compressed rows selected beyond
        # the window.
        positions = metadata.token_positions_cuda[start:end]
        lengths = metadata.sparse_mla_swa_lens[start:end]
        swa_pool = self._pool_view(manager, DeepseekV4AttentionType.SWA, 1)
        extra_pool = None
        run_kwargs = {}
        if extra:
            if self._reads_sparse_arena(metadata):
                # sparse_attn_predict emits arena rows relative to the data arena's
                # base; the manager's cached view of it is rooted there.
                extra_pool = manager.get_sparse_compress_buffers(attn.layer_idx)[0]
            else:
                extra_pool = self._pool_view(manager, DeepseekV4AttentionType.COMPRESS, ratio)
            run_kwargs["extra_indices"] = indices[:, window:]
            run_kwargs["extra_lengths"] = metadata.sparse_mla_extra_lens[ratio][start:end]

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

        # One validated eager launch per process and contract; the key names
        # the producers (window, extra capacity, compress ratio, hole policy)
        # and the pools they address, so the KV-cache estimation executor's
        # temporary pools and the serving pools are validated separately.
        validate = self._claim_contract_validation(
            (
                str(q.device),
                window,
                extra,
                ratio,
                self._holes_possible(),
                str(query.dtype),
                swa_pool.data_ptr(),
                int(swa_pool.shape[0]),
                extra_pool.data_ptr() if extra_pool is not None else 0,
                int(extra_pool.shape[0]) if extra_pool is not None else 0,
            )
        )
        plan.wrapper.run(
            query,
            swa_pool,
            indices[:, :window],
            lengths,
            extra_pool,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            attention_sinks=sinks,
            validate=validate,
            **run_kwargs,
        )


@dataclass(frozen=True)
class _WarmupProfile:
    library: PrimsTSSparseMlaFmha
    device: torch.device
    extra: int
    q_dtype: torch.dtype
    has_sinks: bool
    fused_groups: Optional[int]

    def plan_kwargs(self) -> dict:
        return self.library._plan_kwargs(
            self.extra, self.q_dtype, self.has_sinks, self.fused_groups
        )[0]

    def load(self, metadata: "TrtllmAttentionMetadata", rows: int) -> None:
        self.library._get_plan(
            metadata=metadata,
            device=self.device,
            rows=rows,
            extra=self.extra,
            q_dtype=self.q_dtype,
            has_sinks=self.has_sinks,
            fused_groups=self.fused_groups,
        )


def _warmup_profiles(profiles: list[_WarmupProfile], metadata: "TrtllmAttentionMetadata") -> None:
    """Compile all layer profiles in one worker group, then load them locally."""
    if not profiles:
        return
    from . import prims_ts_precompile

    payloads = []
    plans = []
    seen = set()
    for profile in profiles:
        kwargs = profile.plan_kwargs()
        heads = int(profile.library.attn.num_heads)
        key = (profile.device, heads, tuple(kwargs.items()))
        if key in seen:
            continue
        seen.add(key)
        for rows in profile.library._kernel_variant_rows(metadata, profile.device, kwargs):
            device_index = profile.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            payloads.append(prims_ts_precompile.encode_payload(device_index, rows, heads, kwargs))
            workspace_bytes = _workspace_size(profile.device, rows, heads, **kwargs)
            plans.append((workspace_bytes, profile, rows))
    if not payloads:
        return
    start = time.perf_counter()
    prims_ts_precompile.precompile_payloads(
        payloads, local_ranks=prims_ts_precompile.local_rank_count(metadata.mapping)
    )
    # Allocate the largest workspace first so all profiles can share it. The
    # serving process must load every artifact even when the disk cache is warm.
    for _, profile, rows in sorted(plans, key=lambda plan: plan[0], reverse=True):
        profile.load(metadata, rows)
    logger.info(
        f"PrimTS sparse MLA warmup loaded {len(payloads)} variants across {len(seen)} profiles "
        f"in {time.perf_counter() - start:.1f} s"
    )


def warmup_prims_ts_sparse_mla(
    model: torch.nn.Module, metadata: "DeepseekV4TrtllmAttentionMetadata"
) -> None:
    """Warm enabled DeepSeek-V4 generation profiles before the first model forward."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
        DeepseekV4TrtllmAttention,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
        supports_dsv4_epilogue_fusion,
    )
    from tensorrt_llm._torch.attention.mla import MLA

    if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("PrimTS kernel warmup requires eager metadata outside graph capture")
    profiles = []
    device = metadata.token_positions_cuda.device
    is_fp8 = PrimsTSSparseMlaFmha._is_fp8_cache(metadata)
    for layer in model.modules():
        if not isinstance(layer, MLA) or not isinstance(layer.mqa, DeepseekV4TrtllmAttention):
            continue
        if layer.o_a_proj is None:
            continue
        if not is_fp8 and layer.dtype != torch.bfloat16:
            continue
        manager = layer.mqa._fmha_manager
        if manager is None:
            continue
        for library in manager.fmha_libs:
            if isinstance(library, PrimsTSSparseMlaFmha):
                fused = supports_dsv4_epilogue_fusion(layer)
                if not fused and layer.dtype != torch.bfloat16:
                    continue  # The unfused kernel only writes BF16 output.
                profiles.append(
                    _WarmupProfile(
                        library=library,
                        device=device,
                        extra=library._extra_capacity(metadata),
                        q_dtype=torch.float8_e4m3fn if is_fp8 else torch.bfloat16,
                        has_sinks=getattr(layer.mqa, "attn_sink", None) is not None,
                        fused_groups=layer.n_local_groups if fused else None,
                    )
                )
    _warmup_profiles(profiles, metadata)


__all__ = [
    "PrimsTSSparseMlaFmha",
    "get_prims_ts_toolchain_unsupported_reason",
    "warmup_prims_ts_sparse_mla",
]
