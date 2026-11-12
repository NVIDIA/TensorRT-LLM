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

"""TRT-LLM FMHA adapter for DSA generation on the PrimTS D576 sparse MLA kernel over NVFP4 KV.

Serves the generation phase of DSA layers (512 + 64 latent, e.g. GLM-5) whose latent cache uses the
``fp8rope`` NVFP4 layout, which ``TRTLLM_NVFP4_MLA_KV_CACHE_LAYOUT`` selects by default for the
caches that can hold it. Each cache row is the 352-byte record that
``BatchSparseMLADecodePagedTSWrapper`` reads as
``kv_cache_format="nvfp4.fp8rope"``: the kernel loads the selected records, expands their E2M1 NoPE
values with the E4M3 block scales in its loader, and runs FP8 QK/PV MMAs. Generation therefore needs
neither the NVFP4-to-FP8 gather nor its scratch rows.

``DSATrtllmAttention.sparse_attn_predict`` skips that gather when
:func:`prims_ts_reads_nvfp4_mla_records` holds for the layer and passes the layer's global top-k
rows, which address the records relative to the primary pool base. DSA metadata asks the same
function before it leaves out the gather's FP8 rows. Where it does not hold (Helix, beam search, a
layer whose FMHA list lacks this library, e.g. with skip-correction), the layer gathers and
TRTLLM-Gen serves generation. The query is the E4M3 ``quant_q_buffer`` written by the MLA RoPE
kernel with a unit scale; the records carry real-valued block scales, so the kernel needs no
descale. The context phase keeps the gather and the TRTLLM-Gen kernels.
"""

from __future__ import annotations

import math
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .interface import Fmha
from .prims_ts_sparse_mla import PrimsTSSparseMlaFmha, get_prims_ts_toolchain_unsupported_reason

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        BatchSparseMLADecodePagedTSWrapper,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_LATENT_DIM = 512
_ROPE_DIM = 64
_QK_DIM = _LATENT_DIM + _ROPE_DIM
_RECORD_BYTES = 352
_KV_CACHE_FORMAT = "nvfp4.fp8rope"
# Process-level state shared by every library instance and metadata object: the static profiles
# whose kernel variants were warmed and the index contracts validated with one eager launch.
_WARMED_PROFILES: set = set()
_VALIDATED_CONTRACTS: set = set()


@dataclass
class _SparsePlan:
    """One planned wrapper for a row count, with its graph-stable live lengths."""

    wrapper: "BatchSparseMLADecodePagedTSWrapper"
    lengths: torch.Tensor


def may_use_nvfp4_mla_record_cache(attn: "TrtllmAttention") -> bool:
    """Whether the latent cache of a DSA layer may hold PrimTS ``nvfp4.fp8rope`` records.

    The cache manager settles the layout of its cache after the layers are built, so a layer can
    only rule records out: it has no NVFP4 KV cache, or the ``residual`` layout is requested.
    """
    if not getattr(attn, "has_fp4_kv_cache", False):
        return False
    from ..sparse.dsa.cache_manager import (
        NVFP4_MLA_RESIDUAL_LAYOUT,
        get_requested_nvfp4_mla_kv_cache_layout,
    )

    return get_requested_nvfp4_mla_kv_cache_layout() != NVFP4_MLA_RESIDUAL_LAYOUT


def _metadata_unsupported_reason(metadata) -> Optional[str]:
    """Why no generation with this metadata can read NVFP4 records in place, if so."""
    from ..sparse.dsa.cache_manager import NVFP4_MLA_RECORD_LAYOUT

    manager = getattr(metadata, "kv_cache_manager", None)
    if getattr(manager, "mla_kv_cache_layout", None) != NVFP4_MLA_RECORD_LAYOUT:
        return f"the KV cache manager does not use the {NVFP4_MLA_RECORD_LAYOUT!r} layout."
    if getattr(metadata, "beam_width", 1) != 1:
        return "beam search is not supported."
    mapping = getattr(metadata, "mapping", None)
    if (mapping is not None and mapping.has_cp_helix()) or getattr(
        metadata, "helix_position_offsets", None
    ) is not None:
        return "Helix parallelism is not supported."
    return None


def _layer_lists_library(attn: "TrtllmAttention") -> bool:
    """Whether the layer's FMHA list holds :class:`PrimsTSDsaSparseMlaFmha` (cached per list)."""
    manager = getattr(attn, "_fmha_manager", None)
    cached = getattr(attn, "_prims_ts_record_reader", None)
    if cached is None or cached[0] is not manager:
        listed = manager is not None and any(
            isinstance(fmha, PrimsTSDsaSparseMlaFmha) for fmha in manager.fmha_libs
        )
        cached = (manager, listed)
        attn._prims_ts_record_reader = cached
    return cached[1]


def prims_ts_reads_nvfp4_mla_records(metadata, attn: Optional["TrtllmAttention"] = None) -> bool:
    """Whether DSA generation reads the NVFP4 records in place on :class:`PrimsTSDsaSparseMlaFmha`.

    The one decision behind both gather skips, so they agree with each other and with what the
    library serves. ``DSATrtllmAttention.sparse_attn_predict`` passes its layer and skips the
    gather when this holds. DSA metadata asks without a layer, before the layers are known to
    it, and then leaves out the gather's FP8 rows; a layer that gathers nevertheless (its FMHA
    list lacks the library: skip-correction, an unsupported head count) has them allocated by
    ``DSAtrtllmAttentionMetadata.ensure_nvfp4_mla_generation_buffers``.

    Without a layer: the cache holds records, the metadata rules out neither beam search nor
    Helix, the library is enabled, and the GPU and the CuTe DSL toolchain can run it. With a
    layer: the same metadata conditions, and the layer's FMHA list holds the library.
    """
    if _metadata_unsupported_reason(metadata) is not None:
        return False
    if attn is not None:
        return _layer_lists_library(attn)
    from .registry import get_enabled_fmha_lib_classes

    if PrimsTSDsaSparseMlaFmha not in get_enabled_fmha_lib_classes():
        return False
    if get_sm_version() not in (100, 103):
        return False
    return get_prims_ts_toolchain_unsupported_reason() is None


class PrimsTSDsaSparseMlaFmha(Fmha):
    """DSA generation-phase sparse MLA on PrimTS, reading NVFP4 KV records in place."""

    MAX_HEADS = 128
    PLAN_CACHE_KEY = "prims_ts_dsa_sparse_mla"
    WORKSPACE_CACHE_KEY = "prims_ts_dsa_sparse_mla.workspace"
    # Generation row counts scanned when warming kernel variants: every count up to
    # WARM_ROW_SCAN, then a geometric tail up to the metadata's row budget.
    WARM_ROW_SCAN = 128
    WARM_ROW_TAIL = (192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._checked_key = None
        # Record view of the primary pool, owned (weakly) by the manager it was built from.
        self._records_owner: Optional[weakref.ReferenceType] = None
        self._records: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ availability
    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if not attn.is_mla_enable:
            return False
        if getattr(getattr(attn, "sparse_params", None), "algorithm", None) != "dsa":
            return False
        if not may_use_nvfp4_mla_record_cache(attn):
            return False
        reason = cls._unavailable_reason(attn)
        if reason is not None:
            logger.info_once(
                "PrimTS DSA sparse MLA FMHA is unavailable, DSA generation gathers the NVFP4 "
                f"records for TRTLLM-Gen instead: {reason}",
                key=f"prims_ts_dsa_sparse_mla_unavailable:{reason}",
            )
            return False
        return True

    @classmethod
    def _unavailable_reason(cls, attn: "TrtllmAttention") -> Optional[str]:
        sm = get_sm_version()
        if sm not in (100, 103):
            return f"requires SM100 or SM103, got SM{sm}."
        if int(attn.kv_lora_rank) != _LATENT_DIM or int(attn.qk_rope_head_dim) != _ROPE_DIM:
            return (
                f"requires a {_LATENT_DIM} + {_ROPE_DIM} latent, got "
                f"{attn.kv_lora_rank} + {attn.qk_rope_head_dim}."
            )
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
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            return False
        if forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr is not None:
            # sparse_attn_predict gathered FP8 rows for TRTLLM-Gen.
            return False
        reason = self._unsupported_reason(q, k, v, metadata, forward_args)
        if reason is not None:
            # The indices address NVFP4 records that no other library reads, so falling through
            # would attend to the raw record bytes.
            raise RuntimeError(f"PrimTS DSA sparse MLA FMHA cannot serve this generation: {reason}")
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
        if k is not None or v is not None:
            return "MLA generation takes a fused query only."
        if q.device.type != "cuda" or q.ndim != 2:
            return "the query must be a rank-2 CUDA tensor."
        reason = _metadata_unsupported_reason(metadata)
        if reason is not None:
            return reason
        if fwd.attention_mask_data is not None or fwd.relative_attention_bias is not None:
            return "custom masks and relative bias are not supported."
        if fwd.softmax_stats_tensor is not None:
            return "Helix parallelism is not supported."
        if fwd.attention_sinks is not None:
            return "attention sinks are not supported."
        rows = int(q.shape[0])
        if rows == 0:
            return "the request contains no generation tokens."
        if int(metadata.num_tokens) - int(metadata.num_ctx_tokens) != rows:
            return "the query rows do not match the generation token count."
        heads = int(attn.num_heads)
        if q.shape[1] != heads * _QK_DIM:
            return f"query width must be {heads * _QK_DIM}, got {q.shape[1]}."
        quant_q = fwd.quant_q_buffer
        if quant_q is None or quant_q.dtype not in (torch.uint8, torch.float8_e4m3fn):
            return "the E4M3 query from the MLA RoPE kernel is required."
        if not quant_q.is_contiguous() or quant_q.numel() < rows * heads * _QK_DIM:
            return "the quantized query buffer is too small or not contiguous."
        indices = fwd.sparse_runtime_params.sparse_attn_indices
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 2
            or indices.dtype != torch.int32
            or indices.shape[0] != rows
            or indices.stride(-1) != 1
        ):
            return "sparse indices must be int32 [rows, topk] with contiguous rows."
        output = fwd.output
        if (
            output is None
            or output.dtype != torch.bfloat16
            or output.numel() != rows * heads * _LATENT_DIM
            or not output.is_contiguous()
        ):
            return "a contiguous BF16 output with one 512-wide row per query head is required."
        if fwd.out_scale is not None or fwd.output_sf is not None:
            return "quantized attention output is not supported."
        return None

    def _check_request(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> None:
        """Re-run the request check whenever the per-step geometry changes."""
        indices = fwd.sparse_runtime_params.sparse_attn_indices
        output = fwd.output
        check_key = (
            id(metadata),
            int(metadata.num_tokens),
            int(metadata.num_ctx_tokens),
            q.shape,
            q.dtype,
            None
            if not isinstance(indices, torch.Tensor)
            else (indices.shape, indices.dtype, indices.stride(-1)),
            None if output is None else (output.shape, output.dtype, output.is_contiguous()),
            fwd.quant_q_buffer is None,
            fwd.out_scale is None,
        )
        if check_key == self._checked_key:
            return
        reason = self._unsupported_reason(q, k, v, metadata, fwd)
        if reason is not None:
            raise RuntimeError(f"PrimTS DSA sparse MLA FMHA cannot run this request: {reason}")
        self._checked_key = check_key

    # ------------------------------------------------------------------ helpers
    def _record_view(self, manager) -> torch.Tensor:
        """Return the primary pool as ``[storage_rows, 1, 352]`` records rooted at its base.

        The global top-k rows count pages of every local layer from the pool base, so the view
        spans the whole pool.
        """
        owner = self._records_owner() if self._records_owner is not None else None
        if owner is manager and self._records is not None:
            return self._records
        pool = manager.get_unique_primary_pool()
        if pool.element_size() != 1 or pool.numel() % _RECORD_BYTES:
            raise RuntimeError(
                f"PrimTS DSA sparse MLA expects {_RECORD_BYTES}-byte records, got a pool of "
                f"shape {tuple(pool.shape)} and dtype {pool.dtype}."
            )
        records = pool.view(torch.uint8).reshape(-1, 1, _RECORD_BYTES)
        if records.shape[0] != pool.shape[0] * pool.shape[1] * manager.tokens_per_block:
            raise RuntimeError("PrimTS DSA sparse MLA expects one record per pool token.")
        self._records_owner = weakref.ref(manager)
        self._records = records
        return records

    @staticmethod
    def _softmax_scale(attn: "TrtllmAttention") -> float:
        """TRT-LLM's MLA softmax scale, ``1 / (q_scaling * sqrt(qk_nope + qk_rope))``."""
        qk_head_dim = int(attn.qk_nope_head_dim) + int(attn.qk_rope_head_dim)
        return 1.0 / (float(attn.q_scaling) * math.sqrt(float(qk_head_dim)))

    @staticmethod
    def _plan_kwargs(topk: int) -> dict:
        # The indexer pads short selections with -1 and may interleave them, so the kernel keeps
        # its per-index hole checks over the whole width.
        return dict(
            max_topk=topk,
            max_q_len=1,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=_ROPE_DIM,
            kv_cache_format=_KV_CACHE_FORMAT,
            return_lse=False,
            assume_valid_prefix=False,
        )

    def _get_plan(
        self, *, metadata: "TrtllmAttentionMetadata", device: torch.device, rows: int, topk: int
    ) -> _SparsePlan:
        """Return the metadata-cached plan for a row count, planning it once."""
        from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

        heads = int(self.attn.num_heads)
        key = (device, rows, heads, topk)
        cache = metadata.fmha_plan_caches.setdefault(self.PLAN_CACHE_KEY, {})
        plan = cache.get(key)
        if plan is not None:
            return plan
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS DSA sparse MLA plans must be created during eager warmup, not under "
                f"CUDA graph capture (rows={rows}, topk={topk})."
            )
        t_plan = time.perf_counter()
        plan_kwargs = self._plan_kwargs(topk)
        # One workspace per metadata object, shared by every plan and grown to the largest one:
        # plans run serially, and a plan's scratch is only live during its own launch.
        required_bytes = int(
            mla_decode._resolve_sparse_mla_plan(device, rows, heads, **plan_kwargs)[4]
        )
        shared = metadata.fmha_plan_caches.setdefault(self.WORKSPACE_CACHE_KEY, {})
        workspace = shared.get("buffer")
        if workspace is None or workspace.numel() < required_bytes:
            workspace = torch.empty((required_bytes,), device=device, dtype=torch.uint8)
            shared["buffer"] = workspace
        wrapper = mla_decode.BatchSparseMLADecodePagedTSWrapper(workspace_buffer=workspace)
        wrapper.plan(device, rows, heads, **plan_kwargs)
        lengths = torch.full((rows,), topk, dtype=torch.int32, device=device)
        plan = _SparsePlan(wrapper=wrapper, lengths=lengths)
        cache[key] = plan
        logger.debug(
            f"PrimTS DSA sparse MLA plan rows={rows} topk={topk}: "
            f"{(time.perf_counter() - t_plan) * 1e3:.1f} ms ({len(cache)} plans on this metadata)"
        )
        return plan

    def _warm_kernel_variants(
        self, *, metadata: "TrtllmAttentionMetadata", device: torch.device, topk: int
    ) -> None:
        """Compile every kernel variant a step's row count can select, once per process.

        Runs on eager metadata only, never under capture; CUDA graph metadata objects replay
        one row count each. Keeps one representative per compile signature and plans them
        largest first so the shared workspace is allocated once at its final size.
        """
        if torch.cuda.is_current_stream_capturing() or getattr(metadata, "is_cuda_graph", False):
            return
        heads = int(self.attn.num_heads)
        profile_key = (str(device), heads, topk)
        if profile_key in _WARMED_PROFILES:
            return
        _WARMED_PROFILES.add(profile_key)
        from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

        from . import prims_ts_precompile

        plan_kwargs = self._plan_kwargs(topk)
        requests = int(metadata.max_num_requests or 0)
        draft = int(metadata.max_total_draft_tokens or 0)
        tokens = int(metadata.max_num_tokens or 0)
        max_rows = requests * (1 + draft)
        if tokens:
            max_rows = min(max_rows, tokens) if max_rows else tokens
        candidates = list(range(1, min(max_rows, self.WARM_ROW_SCAN) + 1))
        candidates += [r for r in self.WARM_ROW_TAIL if self.WARM_ROW_SCAN < r <= max_rows]
        t_warm = time.perf_counter()
        representative: dict = {}
        for rows in candidates:
            spec = mla_decode._resolve_sparse_mla_plan(device, rows, heads, **plan_kwargs)[2]
            signature = mla_decode._mla_kernel_compile_signature(spec.kernel)
            representative[signature] = rows
        rows_list = sorted(representative.values(), reverse=True)
        PrimsTSSparseMlaFmha._precompile_variants(
            device,
            heads,
            plan_kwargs,
            rows_list,
            local_ranks=prims_ts_precompile.local_rank_count(metadata.mapping),
        )
        for rows in rows_list:
            self._get_plan(metadata=metadata, device=device, rows=rows, topk=topk)
        logger.debug(
            f"PrimTS DSA sparse MLA warmed {len(representative)} kernel variants "
            f"(rows scanned up to {max_rows}): {(time.perf_counter() - t_warm) * 1e3:.0f} ms"
        )

    @staticmethod
    def _claim_contract_validation(contract_key: tuple) -> bool:
        """Return True exactly once per process for an index contract (never under capture)."""
        if torch.cuda.is_current_stream_capturing() or contract_key in _VALIDATED_CONTRACTS:
            return False
        _VALIDATED_CONTRACTS.add(contract_key)
        return True

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
        attn = self.attn
        heads = int(attn.num_heads)
        rows = int(q.shape[0])
        indices = forward_args.sparse_runtime_params.sparse_attn_indices
        topk = int(indices.shape[1])
        query = (
            forward_args.quant_q_buffer.view(-1)[: rows * heads * _QK_DIM]
            .view(torch.float8_e4m3fn)
            .view(rows, 1, heads, _QK_DIM)
        )
        records = self._record_view(metadata.kv_cache_manager)
        self._warm_kernel_variants(metadata=metadata, device=q.device, topk=topk)
        plan = self._get_plan(metadata=metadata, device=q.device, rows=rows, topk=topk)
        # One validated eager launch per process and pool: it checks every index against the
        # record count, which catches a global-row mismatch before serving.
        validate = self._claim_contract_validation(
            (str(q.device), topk, records.data_ptr(), int(records.shape[0]))
        )
        plan.wrapper.run(
            query,
            records,
            indices,
            plan.lengths,
            softmax_scale=self._softmax_scale(attn),
            output_scale=1.0,
            out=forward_args.output.view(rows, 1, heads, _LATENT_DIM),
            validate=validate,
        )


__all__ = [
    "PrimsTSDsaSparseMlaFmha",
    "may_use_nvfp4_mla_record_cache",
    "prims_ts_reads_nvfp4_mla_records",
]
