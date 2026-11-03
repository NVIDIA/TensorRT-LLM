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

"""FMHA library: DSA generation on the PrimTS D576 sparse MLA kernel.

Serves the generation phase of single-source DSA sparse MLA (latent 512 + RoPE 64,
one per-tensor E4M3 latent pool, Kimi K3.1) with FlashInfer's task-scheduled sparse
MLA decode (``BatchSparseMLADecodePagedTSWrapper`` of the ``prims_ts_next`` vendor
package, ``kv_cache_format="native"``). It reads exactly what the TRTLLM-Gen
token-sparse kernel reads:

* the rope'd E4M3 query ``quant_q_buffer`` ``[T, H, 576]`` that MLA generation
  preprocessing (``mla_rope_generation`` / the DSA decode prologue) writes,
* the layer's global pool rows from ``DSATrtllmAttention.sparse_attn_predict``
  (``convertReqIndexToGlobal`` output, rows of the unique primary pool viewed as
  ``[tokens, 576]``),
* the BMM scales of the same preprocessing (``bmm1[0]`` = softmax scale times the Q
  and KV descales, ``bmm2[0]`` = KV descale); they are per-layer constants, read once
  in an eager call and passed to the kernel as host scalars.

The DSA Top-K rows are ``-1`` padded beyond each row's valid entries and the kernel
masks ``-1``, so every row is launched with the full Top-K width as its live length.
Under Helix the rank keeps a front-packed share of each row and the metadata holds the
kept counts (``helix_row_kv_lens_cuda``); those are the live lengths, and the
natural-log LSE becomes the ``(max, sum) = (lse, 1)`` softmax statistics the Helix
combine weights partial outputs with.

Available only for ``kv_cache_config.dtype='nvfp4_ds_mla'`` layers, whose inline NVFP4
records no other library reads; other DSA layers stay on TRTLLM-Gen.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .interface import Fmha
from .prims_ts_sparse_mla import get_prims_ts_toolchain_unsupported_reason

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

_LATENT_DIM = 512
_ROPE_DIM = 64
_QUERY_DIM = _LATENT_DIM + _ROPE_DIM
_MAX_HEADS = 128


class PrimsTSDsaMlaFmha(Fmha):
    """DSA generation-phase sparse MLA (D576, per-tensor E4M3 latent pool) on PrimTS."""

    supports_nvfp4_ds_mla = True

    # Plans live in ``TrtllmAttentionMetadata.fmha_plan_caches`` under this key: every
    # layer that runs with one metadata object plans each row count once, and the
    # layers sharing a metadata object execute serially on one stream, so all plans of
    # a metadata object share one workspace grown to the largest plan.
    PLAN_CACHE_KEY = "prims_ts_dsa_mla"
    WORKSPACE_CACHE_KEY = "prims_ts_dsa_mla.workspace"
    LENGTHS_CACHE_KEY = "prims_ts_dsa_mla.lengths"

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        # Host copies of the per-layer BMM scales (softmax scale x Q/KV descales, KV
        # descale), read once in an eager call; CUDA graphs capture them.
        self._scales: Optional[tuple[float, float]] = None
        self._checked_key = None

    # ------------------------------------------------------------------ availability
    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        # Only DSA MLA layers are candidates; other attention layers stay silent.
        if not attn.is_mla_enable or (
            getattr(getattr(attn, "sparse_params", None), "algorithm", None) != "dsa"
        ):
            return False
        if not _nvfp4_ds_mla(attn):
            return False
        # The inline NVFP4 records have no other reader: this library serves them always.
        reason = cls._unavailable_reason(attn)
        if reason is not None:
            raise RuntimeError(f"kv_cache_config.dtype='nvfp4_ds_mla' cannot run: {reason}")
        return True

    @classmethod
    def _unavailable_reason(cls, attn: "TrtllmAttention") -> Optional[str]:
        if not attn.is_mla_enable:
            return "not an MLA layer."
        if getattr(getattr(attn, "sparse_params", None), "algorithm", None) != "dsa":
            return "not a DSA sparse layer."
        if get_sm_version() not in (100, 103):
            return f"requires SM100 or SM103, got SM{get_sm_version()}."
        if getattr(attn, "use_fp8_ds_mla", False):
            return "the fp8_ds_mla inline-scale cache is not a per-tensor E4M3 pool."
        if int(attn.kv_lora_rank) != _LATENT_DIM or int(attn.qk_rope_head_dim) != _ROPE_DIM:
            return (
                f"requires kv_lora_rank {_LATENT_DIM} and qk_rope_head_dim {_ROPE_DIM}, got "
                f"{attn.kv_lora_rank} and {attn.qk_rope_head_dim}."
            )
        if not 1 <= int(attn.num_heads) <= _MAX_HEADS:
            return f"serves 1..{_MAX_HEADS} heads per rank, got {attn.num_heads}."
        reason = get_prims_ts_toolchain_unsupported_reason()
        if reason is not None:
            return reason
        try:
            from tensorrt_llm._torch.attention.backends.prims_ts_next import (
                mla_decode,  # noqa: F401
            )
        except ImportError as error:
            return f"the prims_ts_next kernels cannot be imported: {error}"
        return None

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
            logger.debug(f"PrimTS D576 sparse MLA does not support request: {reason}")
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
        from tensorrt_llm.bindings import DataType

        attn = self.attn
        if fwd.attention_input_type != AttentionInputType.generation_only:
            return "only generation-only requests are served."
        if k is not None or v is not None:
            return "MLA generation takes a fused query only."
        if q.device.type != "cuda" or q.ndim != 2:
            return "the query must be a rank-2 CUDA tensor."
        manager = metadata.kv_cache_manager
        if manager is None or getattr(manager, "dtype", None) != DataType.FP8:
            return "requires a per-tensor E4M3 latent cache (or nvfp4_ds_mla records in its container)."
        if _nvfp4_ds_mla(attn):
            return self._nvfp4_unsupported_reason(q, metadata, fwd)
        if metadata.beam_width != 1:
            return "beam search is not supported."
        if fwd.attention_mask_data is not None or fwd.relative_attention_bias is not None:
            return "custom masks and relative bias are not supported."
        if fwd.attention_sinks is not None:
            return "attention sinks are not supported."
        rows = int(q.shape[0])
        if rows == 0:
            return "the request contains no generation tokens."
        if int(metadata.num_tokens) - int(metadata.num_ctx_tokens) != rows:
            return "the query rows do not match the generation token count."
        heads = int(attn.num_heads)
        quant_q = fwd.quant_q_buffer
        if quant_q is None or quant_q.dtype not in (torch.uint8, torch.float8_e4m3fn):
            return "requires the E4M3 query from MLA generation preprocessing."
        if not quant_q.is_contiguous() or quant_q.numel() < rows * heads * _QUERY_DIM:
            return "the quantized query buffer is too small or not contiguous."
        if fwd.mla_bmm1_scale is None or fwd.mla_bmm2_scale is None:
            return "requires the MLA BMM scales."
        indices = fwd.sparse_runtime_params.sparse_attn_indices
        if not isinstance(indices, torch.Tensor):
            return "token-sparse indices are required."
        if (
            indices.ndim != 2
            or indices.dtype != torch.int32
            or indices.shape[0] != rows
            or indices.stride(-1) != 1
        ):
            return "sparse indices must be int32 [rows, top_k] with contiguous rows."
        if fwd.sparse_runtime_params.aux_kv_cache_pool_ptr:
            return "indices into an auxiliary (gathered) pool are not supported."
        helix = metadata.helix_position_offsets is not None
        if helix:
            if getattr(metadata, "_helix_num_rows", 0) <= 0 or not getattr(
                metadata, "_helix_rows_set", False
            ):
                return "Helix requires the token-major kept-count rows."
            stats = fwd.softmax_stats_tensor
            if (
                stats is None
                or stats.dtype != torch.float32
                or stats.numel() < rows * heads * 2
                or not stats.is_contiguous()
            ):
                return "Helix requires a contiguous FP32 [rows, heads, 2] softmax-stats tensor."
        elif getattr(metadata, "is_ragged_verify", False):
            return "ragged verification rows are not supported."
        output = fwd.output
        if (
            output is None
            or output.dtype != torch.bfloat16
            or output.numel() != rows * heads * _LATENT_DIM
            or not output.is_contiguous()
        ):
            return "the output must be a contiguous BF16 [rows, heads * 512]."
        if fwd.out_scale is not None or fwd.output_sf is not None:
            return "quantized attention output is not supported."
        return None

    def _nvfp4_unsupported_reason(
        self, q: torch.Tensor, metadata: "TrtllmAttentionMetadata", fwd: AttentionForwardArgs
    ) -> Optional[str]:
        """Generation request checks of the inline NVFP4 record mode (the KV append and RoPE run here)."""
        heads = int(self.attn.num_heads)
        rows = int(q.shape[0])
        if q.dtype != torch.bfloat16 or q.shape[1] != heads * _QUERY_DIM or q.stride(-1) != 1:
            return "nvfp4_ds_mla takes the BF16 fused query [rows, heads * 576]."
        if getattr(metadata, "token_positions_cuda", None) is None:
            return "nvfp4_ds_mla needs the token positions of the DSA metadata."
        if metadata.helix_position_offsets is not None:
            return "nvfp4_ds_mla does not support Helix."
        if fwd.update_kv_cache:
            latent = fwd.latent_cache
            if (
                latent is None
                or latent.numel() != rows * _QUERY_DIM
                or latent.dtype != torch.bfloat16
            ):
                return "nvfp4_ds_mla appends the BF16 latent rows [rows, 576] of the step."
        indices = fwd.sparse_runtime_params.sparse_attn_indices
        if not isinstance(indices, torch.Tensor) or indices.ndim != 2 or indices.shape[0] != rows:
            return "token-sparse indices are required."
        if indices.dtype != torch.int32 or indices.stride(-1) != 1:
            return "sparse indices must be int32 with contiguous rows."
        output = fwd.output
        if (
            output is None
            or output.dtype != torch.bfloat16
            or output.numel() != rows * heads * _LATENT_DIM
            or not output.is_contiguous()
        ):
            return "the output must be a contiguous BF16 [rows, heads * 512]."
        return None

    # ------------------------------------------------------------------ helpers
    def _host_scales(self, fwd: AttentionForwardArgs) -> tuple[float, float]:
        if self._scales is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PrimTS D576 sparse MLA scales must be read in an eager call before "
                    "CUDA graph capture."
                )
            softmax_scale = float(fwd.mla_bmm1_scale.view(-1)[0].item())
            output_scale = float(fwd.mla_bmm2_scale.view(-1)[0].item())
            for name, value in (("bmm1", softmax_scale), ("bmm2", output_scale)):
                if not (math.isfinite(value) and value > 0):
                    raise RuntimeError(
                        f"PrimTS D576 sparse MLA {name} scale must be finite and positive, "
                        f"got {value}."
                    )
            self._scales = (softmax_scale, output_scale)
        return self._scales

    @classmethod
    def _plan(
        cls,
        metadata: "TrtllmAttentionMetadata",
        device: torch.device,
        rows: int,
        heads: int,
        top_k: int,
        return_lse: bool,
        kv_cache_format: str = "native",
    ):
        """The metadata-cached wrapper for one row count, planned (and compiled) once."""
        from tensorrt_llm._torch.attention.backends.prims_ts_next import mla_decode

        key = (device, rows, heads, top_k, return_lse, kv_cache_format)
        cache = metadata.fmha_plan_caches.setdefault(cls.PLAN_CACHE_KEY, {})
        plan = cache.get(key)
        if plan is not None:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS D576 sparse MLA plans must be created in an eager warmup call, not "
                f"under CUDA graph capture (rows={rows}, heads={heads})."
            )
        t_plan = time.perf_counter()
        kwargs = dict(
            max_topk=top_k,
            max_q_len=1,
            packed_query=True,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=_ROPE_DIM,
            kv_cache_format=kv_cache_format,
            return_lse=return_lse,
        )
        required = int(
            mla_decode.get_prims_ts_sparse_mla_decode_workspace_size(device, rows, heads, **kwargs)
        )
        shared = metadata.fmha_plan_caches.setdefault(cls.WORKSPACE_CACHE_KEY, {})
        workspace = shared.get("buffer")
        if workspace is None or workspace.numel() < required:
            workspace = torch.empty((required,), device=device, dtype=torch.uint8)
            shared["buffer"] = workspace
        wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper(workspace_buffer=workspace)
        wrapper.plan(device, rows, heads, **kwargs)
        qo_indptr = torch.arange(rows + 1, dtype=torch.int32, device=device)
        plan = (wrapper, qo_indptr)
        cache[key] = plan
        logger.debug(
            f"PrimTS D576 sparse MLA plan rows={rows} heads={heads} top_k={top_k} "
            f"lse={return_lse}: {(time.perf_counter() - t_plan) * 1e3:.0f} ms"
        )
        return plan

    @classmethod
    def _full_lengths(
        cls, metadata: "TrtllmAttentionMetadata", device: torch.device, rows: int, top_k: int
    ) -> torch.Tensor:
        """A stable int32 [rows] buffer of ``top_k`` (the live length of a -1 padded row)."""
        shared = metadata.fmha_plan_caches.setdefault(cls.LENGTHS_CACHE_KEY, {})
        buf = shared.get(top_k)
        if buf is None or buf.numel() < rows:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PrimTS D576 sparse MLA lengths must be allocated before capture."
                )
            capacity = max(rows, int(metadata.max_num_tokens or 0))
            buf = torch.full((capacity,), top_k, dtype=torch.int32, device=device)
            shared[top_k] = buf
        return buf[:rows]

    def _check_request(self, q, k, v, metadata, fwd) -> None:
        indices = fwd.sparse_runtime_params.sparse_attn_indices
        key = (
            id(metadata),
            int(metadata.num_tokens),
            int(metadata.num_ctx_tokens),
            q.shape,
            None if not isinstance(indices, torch.Tensor) else (indices.shape, indices.stride(-1)),
            metadata.helix_position_offsets is None,
            fwd.softmax_stats_tensor is None,
        )
        if key == self._checked_key:
            return
        reason = self._unsupported_reason(q, k, v, metadata, fwd)
        if reason is not None:
            raise RuntimeError(f"PrimTS D576 sparse MLA cannot run this request: {reason}")
        self._checked_key = key

    def _forward_nvfp4_ds_mla(self, q, metadata, forward_args) -> None:
        """Generation over the inline NVFP4 records: append the step's KV rows, quantize the query, attend."""
        _forward_nvfp4_ds_mla_impl(self, q, metadata, forward_args)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        self._check_request(q, k, v, metadata, forward_args)
        if _nvfp4_ds_mla(self.attn):
            self._forward_nvfp4_ds_mla(q, metadata, forward_args)
            return
        attn = self.attn
        heads = int(attn.num_heads)
        rows = int(q.shape[0])
        device = q.device
        indices = forward_args.sparse_runtime_params.sparse_attn_indices
        top_k = int(indices.shape[1])
        helix = metadata.helix_position_offsets is not None
        softmax_scale, output_scale = self._host_scales(forward_args)

        query = (
            forward_args.quant_q_buffer.view(-1)[: rows * heads * _QUERY_DIM]
            .view(torch.float8_e4m3fn)
            .view(rows, heads, _QUERY_DIM)
        )
        metadata._ensure_pool_view_cached()
        pool = metadata._cached_pool_view.view(torch.float8_e4m3fn).view(-1, _QUERY_DIM)
        out = forward_args.output.view(rows, heads, _LATENT_DIM)
        if helix:
            start = int(metadata.num_contexts)
            lengths = metadata.helix_row_kv_lens_cuda[start : start + rows]
        else:
            lengths = self._full_lengths(metadata, device, rows, top_k)

        wrapper, qo_indptr = self._plan(metadata, device, rows, heads, top_k, helix)
        if helix:
            stats = forward_args.softmax_stats_tensor.view(-1)[: rows * heads * 2].view(
                rows, heads, 2
            )
            lse = metadata.fmha_plan_caches.setdefault(self.PLAN_CACHE_KEY + ".lse", {})
            buf = lse.get("buffer")
            if buf is None or buf.numel() < rows * heads:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "PrimTS D576 sparse MLA LSE scratch must exist before capture."
                    )
                buf = torch.empty(
                    (max(rows, int(metadata.max_num_tokens or 0)) * heads,),
                    dtype=torch.float32,
                    device=device,
                )
                lse["buffer"] = buf
            lse_rows = buf[: rows * heads].view(rows, heads)
            wrapper.run(
                query,
                pool,
                indices,
                lengths,
                qo_indptr=qo_indptr,
                softmax_scale=softmax_scale,
                output_scale=output_scale,
                out=out,
                lse=lse_rows,
                validate=False,
            )
            # Helix combine weight of a partial output: sum * exp(max - global max).
            stats[..., 0].copy_(lse_rows)
            stats[..., 1].fill_(1.0)
        else:
            wrapper.run(
                query,
                pool,
                indices,
                lengths,
                qo_indptr=qo_indptr,
                softmax_scale=softmax_scale,
                output_scale=output_scale,
                out=out,
                validate=False,
            )


def _nvfp4_ds_mla(attn) -> bool:
    return bool(getattr(attn, "use_nvfp4_ds_mla", False))


def append_nvfp4_ds_mla_rows(
    attn, metadata, latent_rows: torch.Tensor, start: int, end: int, is_generation: bool
) -> None:
    """Quantize latent rows ``[end - start, 576]`` (RoPE applied) into the layer's ``nvfp4_ds_mla`` records."""
    from tensorrt_llm._torch.attention.backends.sparse.nvfp4_ds_mla_kv import (
        TOKEN_BYTES,
        quant_scatter,
        record_scales,
    )

    metadata._ensure_pool_view_cached()
    page_scale, layer_offset = metadata.kv_cache_manager.get_primary_pool_page_index_params(
        attn.get_local_layer_idx(metadata)
    )
    if is_generation:
        block_table, req_idx = metadata._cached_block_table_gen, metadata._cached_req_idx_gen
    else:
        block_table, req_idx = metadata._cached_block_table_ctx, metadata._cached_req_idx_ctx
    tpb = metadata._cached_tokens_per_block
    positions = metadata.token_positions_cuda[start:end]
    loc = torch.ops.trtllm.convert_req_index_to_global(
        req_idx.contiguous(),
        block_table,
        positions.unsqueeze(1).contiguous(),
        tpb,
        1,
        page_scale * tpb,
        layer_offset,
    ).view(-1)
    pool = metadata._cached_pool_view.view(torch.uint8).view(-1, TOKEN_BYTES)
    quant_scatter(pool, loc, latent_rows, record_scales())


def _require_identity_rope(attn) -> None:
    if not getattr(attn, "rope_is_identity", False):
        raise NotImplementedError(
            "kv_cache_config.dtype='nvfp4_ds_mla' supports the NoPE (identity-RoPE) MLA of Kimi K3 / K3.1 only."
        )


def _nvfp4_q_quant_scales(attn) -> tuple[torch.Tensor, torch.Tensor]:
    """Device FP32 [1] E4M3 quantization scales of the query's latent part (as MLA generation preprocessing uses)
    and of its RoPE part (times ``g_rope / g`` of the records)."""
    from tensorrt_llm._torch.attention.backends.sparse.nvfp4_ds_mla_kv import record_scales

    cached = getattr(attn, "_nvfp4_ds_mla_q_scales", None)
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "nvfp4_ds_mla query scale must be prepared before CUDA graph capture."
            )
        scales = record_scales()
        scale = attn.kv_scale_orig_quant.reshape(-1)[:1].to(torch.float32).contiguous()
        cached = (scale, (scale * (scales.rope / scales.nope)).contiguous())
        attn._nvfp4_ds_mla_q_scales = cached
        attn._nvfp4_ds_mla_q_dequant = 1.0 / float(scale[0].item())
    return cached


def _forward_nvfp4_ds_mla_impl(lib: "PrimsTSDsaMlaFmha", q, metadata, fwd) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.dsa.prims_ts_prefill import (
        quantize_query_e4m3,
    )
    from tensorrt_llm._torch.attention.backends.sparse.nvfp4_ds_mla_kv import (
        TOKEN_BYTES,
        record_scales,
    )

    attn = lib.attn
    _require_identity_rope(attn)
    heads = int(attn.num_heads)
    rows = int(q.shape[0])
    start = int(metadata.num_ctx_tokens)
    if fwd.update_kv_cache and fwd.latent_cache is not None:
        append_nvfp4_ds_mla_rows(
            attn, metadata, fwd.latent_cache.view(rows, _QUERY_DIM), start, start + rows, True
        )
    q_scale, pe_scale = _nvfp4_q_quant_scales(attn)
    q_heads = q.view(rows, heads, _QUERY_DIM)
    query = torch.empty(rows, heads, _QUERY_DIM, dtype=torch.float8_e4m3fn, device=q.device)
    quantize_query_e4m3(
        q_heads[..., :_LATENT_DIM], q_heads[..., _LATENT_DIM:], q_scale, query, pe_scale
    )
    indices = fwd.sparse_runtime_params.sparse_attn_indices
    top_k = int(indices.shape[1])
    metadata._ensure_pool_view_cached()
    pool = metadata._cached_pool_view.view(torch.uint8).view(-1, TOKEN_BYTES)
    lengths = lib._full_lengths(metadata, q.device, rows, top_k)
    wrapper, qo_indptr = lib._plan(
        metadata, q.device, rows, heads, top_k, False, kv_cache_format="nvfp4.fp8rope"
    )
    qk_dim = int(attn.qk_nope_head_dim) + int(attn.qk_rope_head_dim)
    g = record_scales().nope
    softmax_scale = attn._nvfp4_ds_mla_q_dequant * g / (float(attn.q_scaling) * math.sqrt(qk_dim))
    wrapper.run(
        query,
        pool,
        indices,
        lengths,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=g,
        out=fwd.output.view(rows, heads, _LATENT_DIM),
        validate=False,
    )


__all__ = [
    "PrimsTSDsaMlaFmha",
]
