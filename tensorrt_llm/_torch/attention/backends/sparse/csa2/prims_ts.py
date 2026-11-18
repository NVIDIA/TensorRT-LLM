# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimTS sparse MLA reading persistent CSA2 FP8 SWA and NVFP4 MAIN records."""

from __future__ import annotations

import functools
import math
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from ...interface import AttentionForwardArgs, AttentionInputType
from ._prims_ts_bank import (
    main_bank_reserved_rows,
    main_record_owners_reserved_rows,
    main_records_rows_limit,
)
from .params import CSA2BackendForwardArgs, CSA2Mode, PrimsTSMainFormat, SwaCacheFormat

if TYPE_CHECKING:
    from .backend import CSA2TrtllmAttention
    from .metadata import CSA2TrtllmMetadata

_PLAN_CACHE_KEY = "csa2_prims_ts"
_ARENA_CACHE_KEY = "csa2_prims_ts_arenas"
_HEAD_DIM = 512
_SWA_TILE = 128
# Generation tiles read their selections as published up to this many heads per rank
# (_reads_raw_selections).
_RAW_SELECTION_HEADS = 32
_PACKED_ABI = "packed_e4m3_unit_scale_v4"
_MAX_SOFTMAX_SCALE = torch.finfo(torch.float32).max / math.log2(math.e)
# CSA2 side stream that prepares selection metadata beside a layer's projections.
_SELECTION_STREAM = 2
# Generation tiles whose selections are final before their projections keep the
# selection bank below this many rows (CSA2PrimsTS._uses_selection_bank).
_EARLY_SELECTION_BANK_ROWS = 1024
# Arena entries naming the selections the shared MAIN list and selection bank last held
# (CSA2PrimsTS._main_selection_group).
_MAIN_HOLDER_KEY = "main_holder"
_SELECTION_BANK_HOLDER_KEY = "csa2_prims_ts_selection_bank_holder"


def _validate_softmax_scale(scale: float) -> None:
    if not math.isfinite(scale) or not 0 < scale <= _MAX_SOFTMAX_SCALE:
        raise ValueError(
            "CSA2 PrimTS softmax scale must be positive, finite and representable "
            "after FP32 base-2 conversion"
        )


class _PrimsTSMetadata(NamedTuple):
    """Independent compact physical-row lists and live lengths for both sources."""

    indices: torch.Tensor
    lengths: torch.Tensor
    extra_indices: torch.Tensor
    extra_lengths: torch.Tensor


@functools.cache
def _workspace_capacity(device: torch.device, heads: int, width: int, queries: int) -> int:
    from ._prims_ts_packed import get_packed_sparse_mla_workspace_size

    # The packed producer supports Keep-AB only. Use its actual selector for
    # every admitted row count, including split-KV topology transitions.
    return max(
        get_packed_sparse_mla_workspace_size(
            device, rows, heads, max_topk=_SWA_TILE, max_extra_topk=width - _SWA_TILE
        )
        for rows in range(1, queries + 1)
    )


def workspace_reservation_bytes(
    device: torch.device,
    heads: int,
    width: int,
    queries: int,
    generation_queries: int = 0,
    main_pool_rows: int = 0,
) -> int:
    """Bound one serial arena shared by eager and graph-retained packed plans.

    All wrapper Q8/reduction/LSE/dummy storage is carved from the byte
    workspace (Q8 only for plans of up to ``_Q8_ARENA_ROWS`` rows). Plans own no
    additional defaults or cache tensors. Compact
    metadata is the only separate allocation, plus, for widths with MAIN rows,
    the reserved storage of the device's decoded context MAIN bank and its
    one-byte row marks (a phase that owns more MAIN rows grows them past the
    reservation), and the selection bank of generation tiles of up to
    ``generation_queries`` rows with its INT64 owner slot per row of a MAIN pool
    of up to ``main_pool_rows`` rows, of which it covers up to
    ``main_record_owners_reserved_rows()`` (a larger pool grows them past the reservation).
    """
    bank = 0
    if width > _SWA_TILE:
        bank = main_bank_reserved_rows() * (_HEAD_DIM + 1)
        if generation_queries > 0:
            selections = generation_queries * (width - _SWA_TILE)
            bank += _HEAD_DIM * min(
                max(selections, main_bank_reserved_rows()), main_records_rows_limit()
            )
            bank += 8 * (min(main_pool_rows, main_record_owners_reserved_rows()) + 1)
    return _workspace_capacity(device, heads, width, queries) + 4 * queries * (width + 2) + bank


def _allocate_metadata(device: torch.device, rows: int, width: int) -> _PrimsTSMetadata:
    return _PrimsTSMetadata(
        indices=torch.empty((rows, _SWA_TILE), device=device, dtype=torch.int32),
        lengths=torch.empty((rows,), device=device, dtype=torch.int32),
        extra_indices=torch.empty((rows, width - _SWA_TILE), device=device, dtype=torch.int32),
        extra_lengths=torch.empty((rows,), device=device, dtype=torch.int32),
    )


def _metadata_view(prepared: _PrimsTSMetadata, rows: int) -> _PrimsTSMetadata:
    return _PrimsTSMetadata(*(tensor[:rows] for tensor in prepared))


def _rotate_query(q: torch.Tensor, raw: CSA2BackendForwardArgs, heads: int) -> None:
    """Apply in place the GPT-J query RoPE ``raw.query_rope`` leaves to the kernel, for a plan
    that reads Q rotated; after the projection's event, as the module's own RoPE would run."""
    from .projection import SMALL_M_ROWS, gptj_rope_small_m

    positions, cos_sin = raw.query_rope
    if raw.query_ready is not None:
        torch.cuda.current_stream().wait_event(raw.query_ready)
    view = q.view(q.shape[0], heads, _HEAD_DIM)
    if q.shape[0] <= SMALL_M_ROWS:
        gptj_rope_small_m(view, positions, cos_sin)
    else:
        torch.ops.trtllm.mla_rope_inplace(
            view, positions, cos_sin, heads, _HEAD_DIM - 64, 64, False, False
        )


def _takes_raw_selections(inputs: CSA2BackendForwardArgs, width: int) -> bool:
    """Whether a tile's published selections fit a raw-selection plan of ``width`` columns.

    INT64 SWA slots of one window and, with MAIN rows, INT32 logical positions spanning the
    planned MAIN columns, mapped through a nonempty INT32 page table with INT64 requests and
    visible lengths; anything else is compacted first.
    """
    if inputs.swa_indices is None:
        return False
    rows = inputs.swa_indices.shape[0]

    def plain(indices: torch.Tensor | None, dtype: torch.dtype, columns: int) -> bool:
        return (
            isinstance(indices, torch.Tensor)
            and indices.dtype == dtype
            and indices.shape == (rows, columns)
            and indices.stride(1) == 1
            and indices.data_ptr() % indices.element_size() == 0
        )

    if not plain(inputs.swa_indices, torch.int64, _SWA_TILE):
        return False
    if width <= _SWA_TILE:
        return inputs.topk_indices is None
    mapping = inputs.main_mapping
    if (
        mapping is None
        or inputs.main_pool is None
        or not plain(inputs.topk_indices, torch.int32, width - _SWA_TILE)
    ):
        return False
    pages, page_size, max_positions, requests, visible = mapping
    return (
        pages.dtype == torch.int32
        and pages.ndim == 2
        and pages.numel() > 0
        and pages.stride(1) == 1
        and isinstance(page_size, int)
        and page_size > 0
        and isinstance(max_positions, int)
        and max_positions >= 0
        and inputs.main_pool.shape[0] <= 0x7FFFFFFF - page_size
        and all(
            values.dtype == torch.int64 and values.shape == (rows,) and values.is_contiguous()
            for values in (requests, visible)
        )
    )


def _runs_native_kernel(
    device: torch.device,
    rows: int,
    heads: int,
    width: int,
    swa_cache_format: SwaCacheFormat,
    main_cache_format: PrimsTSMainFormat,
    quantized_output: bool = False,
) -> bool:
    """Whether a compacted plan of this tile runs a native kernel rather than the packed producer.

    The native FP8+NVFP4 MAIN kernel (``decodes_nvfp4_main``) and the TMA gather over plain
    E4M3 rows (``_select_gather_profile``) read compacted lists; only the packed producer
    also reads selections as published (raw selections).
    """
    from ._prims_ts_packed import (
        _select_gather_profile,
        _validate_geometry,
        decodes_nvfp4_main,
        gathers_plain_rows,
    )

    extra = width - _SWA_TILE
    if rows <= 0:
        return False
    if (
        swa_cache_format == "swa_fp8"
        and main_cache_format == "main"
        and extra > 0
        and decodes_nvfp4_main(
            device, rows, heads, _SWA_TILE, extra, quantized_output=quantized_output
        )
    ):
        return True
    if not gathers_plain_rows(swa_cache_format, main_cache_format, extra):
        return False
    capacity = _validate_geometry(rows, heads, _SWA_TILE, extra)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return _select_gather_profile(rows, heads, capacity, sm_count) is not None


def _reads_raw_selections(
    device: torch.device,
    rows: int,
    heads: int,
    width: int,
    swa_cache_format: SwaCacheFormat,
    main_cache_format: PrimsTSMainFormat,
    quantized_output: bool = False,
) -> bool:
    """Whether a generation tile of this geometry reads its selections as published.

    Tiles of at most ``_RAW_SELECTION_HEADS`` heads per rank that no native kernel serves
    (``_runs_native_kernel``) do. Wider tiles, such as 64-head DEP tiles, compact them
    first, which measures faster end to end.
    """
    return heads <= _RAW_SELECTION_HEADS and not _runs_native_kernel(
        device, rows, heads, width, swa_cache_format, main_cache_format, quantized_output
    )


class CSA2PrimsTS:
    """Module compute helper; plans and serial scratch belong to query metadata."""

    # A selection bank converts each selected MAIN row once (dedup_main_records).
    dedup_selection_bank = True
    # Generation tiles that would otherwise run the packed producer over compacted lists
    # read the selections as published instead (PackedSparseMLA raw selections).
    raw_selections = True

    def __init__(self, attention: CSA2TrtllmAttention) -> None:
        self.attention = attention
        self._selection_done: torch.cuda.Event | None = None

    def forward_context(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        forward_args: AttentionForwardArgs,
    ) -> torch.Tensor:
        if forward_args.attention_input_type != AttentionInputType.context_only:
            raise ValueError("CSA2 context helper requires context_only inputs")
        return self._forward_packed(q, metadata, forward_args)

    def forward_generation(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        forward_args: AttentionForwardArgs,
    ) -> torch.Tensor:
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            raise ValueError("CSA2 generation helper requires generation_only inputs")
        return self._forward_packed(q, metadata, forward_args)

    def _get_plan(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        swa_cache_format: SwaCacheFormat = "swa",
        main_cache_format: PrimsTSMainFormat = "main",
        output_groups: int | None = None,
        raw_selections: bool = False,
        query_rope: bool = False,
        inputs_final: bool = False,
        main_ready: bool = True,
    ) -> dict:
        from ._prims_ts_packed import PackedSparseMLA, precompile_packed_sparse_mla

        rows = metadata.num_tokens
        heads = self.attention.num_heads
        width = metadata.num_sparse_topk
        # A kernel that does not run ahead of its preceding launch reads its MAIN part
        # after that launch whatever ``main_ready`` says: its plans are one plan.
        main_ready = main_ready or not inputs_final
        key = (
            q.device,
            rows,
            heads,
            width,
            swa_cache_format,
            main_cache_format,
            output_groups,
            raw_selections,
            query_rope,
            inputs_final,
            main_ready,
            _PACKED_ABI,
        )
        plans = metadata.fmha_plan_caches.setdefault(_PLAN_CACHE_KEY, {})
        if key in plans:
            record = plans[key]
            if not record["warmed"] and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the CSA2 PrimTS profile before CUDA Graph capture")
            return record
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the CSA2 PrimTS profile before CUDA Graph capture")
        arenas = metadata.fmha_plan_caches.setdefault(_ARENA_CACHE_KEY, {})
        arena_key = (q.device, heads, width, swa_cache_format, _PACKED_ABI)
        arena = arenas.get(arena_key)
        capacity = max(
            rows, getattr(metadata, "_csa2_prims_ts_max_queries", metadata.max_num_tokens)
        )
        if arena is None:
            arena = {
                "workspace": torch.empty(
                    _workspace_capacity(q.device, heads, width, capacity),
                    device=q.device,
                    dtype=torch.uint8,
                ),
                "metadata": _allocate_metadata(q.device, capacity, width),
                "capacity": capacity,
                "main_formats": set(),
            }
            arenas[arena_key] = arena
        variant = (
            main_cache_format,
            output_groups,
            raw_selections,
            query_rope,
            inputs_final,
            main_ready,
        )
        if variant not in arena["main_formats"]:
            # Row counts select compile-time kernel variants. Compile every
            # variant this arena can plan for the MAIN format and output now, so
            # no request compiles one. All of them share the arena's storage.
            precompile_packed_sparse_mla(
                q.device,
                arena["capacity"],
                heads,
                max_topk=_SWA_TILE,
                max_extra_topk=width - _SWA_TILE,
                swa_cache_format=swa_cache_format,
                main_cache_format=main_cache_format,
                valid_prefix=not raw_selections,
                output_groups=output_groups,
                raw_selections=raw_selections,
                query_rope=query_rope,
                inputs_final=inputs_final,
                main_ready=main_ready,
            )
            arena["main_formats"].add(variant)
        if rows > arena["capacity"]:
            raise ValueError("CSA2 PrimTS query count exceeds its reserved metadata capacity")
        wrapper = PackedSparseMLA(arena["workspace"])
        wrapper.plan(
            q.device,
            rows,
            heads,
            max_topk=_SWA_TILE,
            max_extra_topk=width - _SWA_TILE,
            has_sinks=True,
            swa_cache_format=swa_cache_format,
            main_cache_format=main_cache_format,
            # prepare_prims_ts_metadata() compacts each list's valid rows into
            # its live prefix, so plain E4M3 rows may be gathered by TMA; raw
            # selections keep their holes.
            valid_prefix=not raw_selections,
            output_groups=output_groups,
            raw_selections=raw_selections,
            query_rope=query_rope,
            inputs_final=inputs_final,
            main_ready=main_ready,
        )
        record = {
            "wrapper": wrapper,
            "warmed": False,
            "metadata": _metadata_view(arena["metadata"], rows),
            "arena": arena,
            # Only workspace views are plan-owned; persistent cache tensors
            # are live launch arguments and must not enter storage accounting.
            "owned_tensors": wrapper.workspace_tensors,
        }
        plans[key] = record
        return record

    @staticmethod
    def _context_query(
        q: torch.Tensor, raw: CSA2BackendForwardArgs, forward_args: AttentionForwardArgs
    ) -> torch.Tensor:
        """The Q the kernel reads: a projected unit-scale E4M3 Q, or BF16 Q.

        The fused Q projection rounds its RoPE'd BF16 Q RN/satfinite to E4M3 at unit scale,
        as the kernel's own quantization would. A context tile with
        ``projected_q_unit_scale`` gets no BF16 Q at all; a generation tile takes the
        projection's E4M3 Q when it carries no amax word (no live rescale).
        """
        projected = raw.projected_quant_q
        generation = (
            forward_args.attention_input_type == AttentionInputType.generation_only
            and raw.projected_q_amax is None
        )
        if projected is None or not (
            forward_args.attention_input_type == AttentionInputType.context_only or generation
        ):
            if raw.projected_q_unit_scale:
                raise ValueError("CSA2 PrimTS unit-scale context Q requires its projected E4M3 Q")
            return q
        if (
            projected.shape != q.shape
            or projected.dtype != torch.float8_e4m3fn
            or projected.device != q.device
            or not projected.is_contiguous()
        ):
            raise ValueError("CSA2 PrimTS projected Q must be contiguous E4M3 shaped like Q")
        return projected

    def _forward_packed(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        forward_args: AttentionForwardArgs,
    ) -> torch.Tensor:
        from ._prims_ts_bank import (
            decode_main_bank,
            decode_main_records,
            dedup_main_records,
            main_bank_marks,
            main_bank_storage,
            main_record_owners,
            main_records_storage,
        )
        from ._prims_ts_metadata import prepare_prims_ts_metadata
        from .metadata import csa2_side_stream

        raw = forward_args.sparse_backend_args
        if not isinstance(raw, CSA2BackendForwardArgs):
            raise TypeError("CSA2 requires CSA2BackendForwardArgs")
        if (
            q.shape != (metadata.num_tokens, self.attention.num_heads * _HEAD_DIM)
            or q.dtype != torch.bfloat16
            or not q.is_cuda
            # A unit-scale context Q is read only as its projected E4M3 copy (_context_query).
            or not (q.is_contiguous() or raw.projected_q_unit_scale)
        ):
            raise ValueError("CSA2 PrimTS requires contiguous CUDA BF16[T,H*512] queries")
        if metadata.kv_input_mode != "packed":
            raise ValueError("CSA2 PrimTS requires packed KV input metadata without staging")
        mxfp8 = self._mxfp8_output(q, metadata, forward_args)
        # A plan that writes the grouped MXFP8 O-A input leaves the BF16 output unwritten.
        self.attention._validate_helper_output(
            q, forward_args if mxfp8 is None else replace(forward_args, output=None)
        )
        sinks = forward_args.attention_sinks
        if (
            sinks.shape != (self.attention.num_heads,)
            or sinks.dtype != torch.float32
            or sinks.device != q.device
            or not sinks.is_contiguous()
        ):
            raise ValueError("CSA2 PrimTS sinks must be contiguous CUDA FP32[H]")
        scale = 1.0 / (self.attention.q_scaling * math.sqrt(_HEAD_DIM))
        _validate_softmax_scale(scale)
        swa_cache_format = raw.swa_cache_format or self.attention.swa_cache_format
        # An eager context phase reads its MAIN rows from a bank decoded once per
        # row, not per selection: all of its requests' rows, or only the selected
        # ones when its queries select fewer rows than the requests own.
        bank = metadata.prims_ts_main_bank
        output_groups = None if mxfp8 is None else mxfp8[0].shape[0]
        # A tile without a MAIN bank can still gather unit-scale E4M3 rows: one bank row
        # per selection, decoded after compaction (_uses_selection_bank).
        selection_bank = bank is None and self._uses_selection_bank(
            q,
            metadata,
            swa_cache_format,
            quantized_output=mxfp8 is not None,
            early_selection=raw.selection_ready is not None,
        )
        main_cache_format = "main" if bank is None and not selection_bank else "main_fp8"
        # A generation tile whose compacted plan would run the packed producer reads its
        # selections as published instead: the producer masks invalid selections and maps
        # MAIN positions itself. A tile that a native kernel serves, or a wide one, keeps its
        # compacted lists (_reads_raw_selections). A module tile's backend publishes INT64 SWA
        # slots and mapped INT32 logical MAIN positions; a direct caller's selections are
        # checked as given, before any plan exists.
        raw_selections = (
            self.raw_selections
            and forward_args.attention_input_type == AttentionInputType.generation_only
            and bank is None
            and not selection_bank
            and _reads_raw_selections(
                q.device,
                metadata.num_tokens,
                self.attention.num_heads,
                metadata.num_sparse_topk,
                swa_cache_format,
                main_cache_format,
                quantized_output=mxfp8 is not None,
            )
            and (raw.state is not None or _takes_raw_selections(raw, metadata.num_sparse_topk))
        )
        # A raw-selection kernel can take Q before its RoPE and rotate it itself; any other
        # plan reads it rotated.
        query_rope = raw.query_rope if raw_selections else None
        with torch.cuda.device(q.device):
            if raw.query_rope is not None and query_rope is None:
                _rotate_query(q, raw, self.attention.num_heads)
            # Reject cold capture before cache writes or indexer side effects.
            record = self._get_plan(
                q,
                metadata,
                swa_cache_format,
                main_cache_format,
                output_groups,
                raw_selections,
                query_rope is not None,
                self._inputs_final(raw, query_rope),
                self._main_ready(),
            )
            inputs = self.attention.prepare_sparse_inputs(q, metadata, forward_args)
            if (inputs.swa_cache_format or "swa") != swa_cache_format:
                raise ValueError("CSA2 PrimTS prepared SWA format must match its planned format")
            if raw_selections and not _takes_raw_selections(inputs, metadata.num_sparse_topk):
                if query_rope is not None:
                    _rotate_query(q, raw, self.attention.num_heads)
                    query_rope = None
                # Graph capture repeats the eager warm-up's choice, so mark the raw plan as
                # warm and compact; its compacting plan was warmed at the same step.
                record["warmed"] = True
                raw_selections = False
                record = self._get_plan(
                    q, metadata, swa_cache_format, main_cache_format, output_groups
                )
            prepared = record["metadata"]
            wrapper = record["wrapper"]
            has_extra = prepared.extra_indices.shape[1] > 0
            main_source = inputs.main_pool
            marks = None
            arenas = metadata.fmha_plan_caches.setdefault(_ARENA_CACHE_KEY, {})
            if bank is not None:
                if not has_extra or inputs.main_pool is None or inputs.main_mapping is None:
                    raise ValueError("CSA2 PrimTS MAIN banks need logical MAIN selections")
                main_source = main_bank_storage(arenas, q.device, bank.rows)
                if bank.selected_only:
                    marks = main_bank_marks(arenas, q.device, bank.rows)
                else:
                    # After this layer's MAIN publication, on the stream that wrote it.
                    decode_main_bank(inputs.main_pool, bank, main_source)
            extra_indices = prepared.extra_indices
            arena = record["arena"]
            group = None if bank is not None else self._main_selection_group(inputs, q.shape[0])
            dedup, claims, bank_holder = False, None, None
            if selection_bank:
                if not has_extra or inputs.main_pool is None:
                    raise ValueError("CSA2 PrimTS selection banks need MAIN selections")
                # Sized once for the largest tile this metadata plans, before capture.
                max_queries = getattr(
                    metadata, "_csa2_prims_ts_max_queries", metadata.max_num_tokens
                )
                main_source = main_records_storage(
                    arenas,
                    q.device,
                    extra_indices.numel(),
                    capacity=max(max_queries, q.shape[0]) * prepared.extra_indices.shape[1],
                )
                if "selection_indices" not in record:
                    # Both index sets exist before capture, whichever a call uses.
                    record["selection_indices"] = torch.arange(
                        prepared.extra_indices.numel(), device=q.device, dtype=torch.int32
                    ).view(prepared.extra_indices.shape)
                    record["selection_rows"] = torch.empty_like(prepared.extra_indices)
                dedup = self.dedup_selection_bank and self._shares_main_rows(raw, q.shape[0])
                # Selection c of query r reads bank row r * width + c, or with dedup the bank
                # row of the first selection of its MAIN row, the only selections decoded.
                extra_indices = record["selection_rows" if dedup else "selection_indices"]
                bank_holder = (group, dedup, main_source.data_ptr(), extra_indices.data_ptr())
            # The group's MAIN list, and the selection bank this tile reads, may hold this
            # layer's selections already; the layer then compacts only its own SWA rows.
            swa_only = (
                group is not None
                and arena.get(_MAIN_HOLDER_KEY) == group
                and (not selection_bank or arenas.get(_SELECTION_BANK_HOLDER_KEY) == bank_holder)
            )
            if dedup and not swa_only:
                # The compaction claims each kept MAIN row's owner slot for the decode.
                claims = main_record_owners(arenas, q.device, int(inputs.main_pool.shape[0]))
            sources = dict(
                swa_cache=inputs.swa_pool,
                swa_indices=prepared.indices,
                swa_lengths=prepared.lengths,
                extra_kv_cache=main_source if has_extra else None,
                extra_sparse_indices=extra_indices if has_extra else None,
                extra_sparse_lens=prepared.extra_lengths if has_extra else None,
            )

            def compact() -> None:
                if swa_only:
                    prepare_prims_ts_metadata(inputs, prepared, swa_only=True)
                    return
                prepare_prims_ts_metadata(inputs, prepared, bank, marks, claims)
                arena[_MAIN_HOLDER_KEY] = group

            def decode_selection_bank() -> None:
                if dedup:
                    dedup_main_records(
                        inputs.main_pool,
                        prepared.extra_indices,
                        extra_indices,
                        main_source,
                        arenas,
                        claimed=True,
                    )
                else:
                    decode_main_records(
                        inputs.main_pool, prepared.extra_indices, main_source, arenas
                    )
                arenas[_SELECTION_BANK_HOLDER_KEY] = bank_holder

            decodes = selection_bank and not swa_only
            early = raw.selection_ready is not None and marks is None
            if raw_selections:
                # The kernel reads the published lists; the shared compact metadata and
                # its group holders stay as they are.
                sources.update(
                    swa_indices=inputs.swa_indices,
                    swa_lengths=None,
                    extra_sparse_indices=inputs.topk_indices if has_extra else None,
                    extra_sparse_lens=None,
                    main_mapping=inputs.main_mapping if has_extra else None,
                )
            elif not early:
                # Every layer's bank shares the device's marks, so a selected-row
                # bank marks and decodes them in this stream's order.
                compact()
            else:
                # The selections and the MAIN rows they address were final before this
                # layer's projections, so compact them and decode their selection bank
                # beside the projections and this layer's KV write. The previous layer's
                # attention, which read the shared metadata and bank, and its bank decode,
                # which advanced the owner slots' epoch, precede ``selection_ready`` on the
                # model stream.
                side = csa2_side_stream(q.device, _SELECTION_STREAM)
                if self._selection_done is None:
                    self._selection_done = torch.cuda.Event()
                with torch.cuda.stream(side):
                    side.wait_event(raw.selection_ready)
                    compact()
                    if decodes:
                        decode_selection_bank()
                    self._selection_done.record()
                self._selection_done.wait()
            if marks is not None:
                # The rows the compaction kept, after this layer's MAIN publication.
                decode_main_bank(inputs.main_pool, bank, main_source, marks)
            if decodes and not early:
                # The compacted MAIN records, after this layer's MAIN publication.
                decode_selection_bank()
            output = forward_args.output
            if output is None:
                output = torch.empty_like(q, memory_format=torch.contiguous_format)
            rows, heads = metadata.num_tokens, self.attention.num_heads
            query = q if query_rope is not None else self._context_query(q, raw, forward_args)
            wrapper.run(
                query.view(rows, 1, heads, _HEAD_DIM),
                **sources,
                softmax_scale=scale,
                output_scale=1.0,
                sinks=sinks,
                out=None if mxfp8 is not None else output.view(rows, 1, heads, _HEAD_DIM),
                validate=False,
                query_ready=raw.query_ready,
                output_quant=mxfp8,
                **({} if query_rope is None else {"query_rope": query_rope}),
            )
            record["warmed"] = True
            if mxfp8 is not None:
                raw.state.output_mxfp8_written = True
            return output

    @staticmethod
    def _main_selection_group(inputs: CSA2BackendForwardArgs, rows: int) -> tuple | None:
        """Key of the MAIN list a compaction of this tile writes, or None if no layer shares it.

        Layers of one staging group (KV owner, index source and query start) read one MAIN
        pool through the same selections, page table and visible prefixes in a forward, so
        their compactions write the same MAIN list and the same selection bank. The key
        names the forward (its request metadata, index publication and staging epoch) and
        whether the stream captures, so a CUDA Graph records its own compaction.
        """
        source = None if inputs.state is None else inputs.state.metadata
        if (
            inputs.stage_group is None
            or source is None
            or inputs.topk_indices is None
            or inputs.main_pool is None
        ):
            return None
        mapping = inputs.main_mapping
        return (
            inputs.stage_group,
            id(source),
            id(getattr(source, "csa2_indices", None)),
            getattr(source, "_csa2_stage_epoch", 0),
            rows,
            torch.cuda.is_current_stream_capturing(),
            inputs.main_pool.data_ptr(),
            inputs.main_pool.shape[0],
            inputs.topk_indices.data_ptr(),
            None if mapping is None else mapping[0].data_ptr(),
        )

    @staticmethod
    def _shares_main_rows(raw: CSA2BackendForwardArgs, rows: int) -> bool:
        """Whether a generation tile's query rows can select the same MAIN rows.

        Only rows of one request share MAIN rows: a tile of one query row per request has
        no duplicates to skip. A standalone tile without forward state is assumed to share.
        """
        state = raw.state
        if state is None:
            return True
        requests = state.metadata.num_generations
        return requests <= 0 or rows > requests

    def _inputs_final(self, raw: CSA2BackendForwardArgs, query_rope) -> bool:
        """Whether this tile's kernel may run ahead of the launch preceding it on the stream.

        As for the direct decode's ``inputs_ready``: a module tile that starts its forward's
        queries on this stream (no projection event to wait for) reads a Q, SWA slots and a
        page table final before that launch. Only the kernel that takes Q before its RoPE
        reads it so. Whether its MAIN part is final too is :meth:`_main_ready`.
        """
        if query_rope is None or raw.state is None or raw.query_start != 0:
            return False
        if raw.query_ready is not None:
            return False
        layout = self.attention.sparse_params.layout
        return layout is not None and 0 <= self.attention.layer_idx < len(layout.compress_ratios)

    def _main_ready(self) -> bool:
        """Whether a runahead kernel may read its MAIN selections and rows before the wait.

        As for the direct decode's ``main_ready``: a Reuse layer's are its sources', final
        before the preceding launch, and a SWA-only layer has none; a Full or Reindex layer's
        indexer has just selected them, so its kernel reads them after the wait.
        """
        layout = self.attention.sparse_params.layout
        if layout is None or not 0 <= self.attention.layer_idx < len(layout.compress_ratios):
            return True
        return layout.layer(self.attention.layer_idx).mode in (CSA2Mode.REUSE, CSA2Mode.SWA)

    def _uses_selection_bank(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        swa_cache_format: SwaCacheFormat,
        quantized_output: bool = False,
        early_selection: bool = False,
    ) -> bool:
        """Whether a tile without a MAIN bank decodes its MAIN selections into a selection bank.

        With E4M3 SWA rows, the native NVFP4 MAIN kernel decodes each selected NVFP4
        MAIN record into its operand stage and needs no bank where it applies
        (``decodes_nvfp4_main``). Elsewhere in-kernel decoding converts each selected
        record inside the packed producer's warps, and a bank row per selection,
        decoded by its own kernel, lets the tile take the TMA gather schedule over
        unit-scale E4M3 rows instead, when its selections fit the bank's reserved rows.
        ``quantized_output`` is whether the tile stores the grouped MXFP8 output.

        A tile with ``early_selection`` (selections final before its projections)
        compacts and decodes its selection bank on the side stream beside them, so
        below ``_EARLY_SELECTION_BANK_ROWS`` rows it keeps the bank wherever the
        gather schedule applies: the decode overlaps the projections, and the gather
        kernel is faster than the NVFP4 MAIN kernel (GB300, 64 heads, SWA 128 + 512
        MAIN: 252 rows 24.0 against 29.0 us, 504 rows 45.9 against 55.9 us). AgentX
        DEP generation, NVFP4 MAIN kernel on every tile against the selection bank on
        every tile: 252 rows -2.2% and 504 rows -1.2% output throughput per GPU, but
        1252 rows +0.9%, where the decode beside the projections competes with them
        for the GPU (1252 rows: 91 us gather after an 87 us decode, against 119 us).
        """
        from ._prims_ts_packed import _select_gather_profile, _validate_geometry, decodes_nvfp4_main

        width = metadata.num_sparse_topk
        rows = metadata.num_tokens
        if swa_cache_format != "swa_fp8" or width <= _SWA_TILE or rows <= 0:
            return False
        heads = self.attention.num_heads
        if not (early_selection and rows < _EARLY_SELECTION_BANK_ROWS) and decodes_nvfp4_main(
            q.device, rows, heads, _SWA_TILE, width - _SWA_TILE, quantized_output=quantized_output
        ):
            return False
        if rows * (width - _SWA_TILE) > main_records_rows_limit():
            return False
        capacity = _validate_geometry(rows, heads, _SWA_TILE, width - _SWA_TILE)
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        return _select_gather_profile(rows, heads, capacity, sm_count) is not None

    def _mxfp8_output(
        self,
        q: torch.Tensor,
        metadata: CSA2TrtllmMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[torch.Tensor, ...] | None:
        """The module's MXFP8 O-A input buffers when this tile's kernel can write them.

        The module offers them for a context-only or generation-only forward, whose single
        tile is the whole batch; plans whose schedule finalizes in a split-KV reduction keep
        BF16.
        """
        raw = forward_args.sparse_backend_args
        state = getattr(raw, "state", None)
        mxfp8 = None if state is None else state.output_mxfp8
        if (
            mxfp8 is None
            or forward_args.attention_input_type
            not in (AttentionInputType.context_only, AttentionInputType.generation_only)
            or raw.query_start != 0
            or mxfp8[0].ndim != 3
            or mxfp8[0].shape[1] != metadata.num_tokens
        ):
            return None
        if not self.writes_mxfp8_output(q.device, metadata.num_tokens, metadata.num_sparse_topk):
            return None
        return mxfp8

    def writes_mxfp8_output(self, device: torch.device, rows: int, width: int) -> bool:
        """Whether a context-only or generation-only tile of ``rows`` rows and sparse ``width``
        writes the grouped MXFP8 O-A input it is offered, and then no BF16 output."""
        from ._prims_ts_packed import fuses_mxfp8_output

        return fuses_mxfp8_output(
            device, rows, self.attention.num_heads, _SWA_TILE, width - _SWA_TILE
        )
