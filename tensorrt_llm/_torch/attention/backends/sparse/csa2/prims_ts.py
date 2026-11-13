# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimTS sparse MLA reading persistent CSA2 FP8 SWA and NVFP4 MAIN records."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from ...interface import AttentionForwardArgs, AttentionInputType
from ._prims_ts_bank import main_bank_reserved_rows, main_records_rows_limit
from .params import CSA2BackendForwardArgs, PrimsTSMainFormat, SwaCacheFormat

if TYPE_CHECKING:
    from .backend import CSA2TrtllmAttention
    from .metadata import CSA2TrtllmMetadata

_PLAN_CACHE_KEY = "csa2_prims_ts"
_ARENA_CACHE_KEY = "csa2_prims_ts_arenas"
_HEAD_DIM = 512
_SWA_TILE = 128
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
    workspace. Plans own no additional defaults or cache tensors. Compact
    metadata is the only separate allocation, plus, for widths with MAIN rows,
    the reserved storage of the device's decoded context MAIN bank and its
    one-byte row marks (a phase that owns more MAIN rows grows them past the
    reservation), and the selection bank of generation tiles of up to
    ``generation_queries`` rows with its INT64 owner slot per row of a MAIN pool
    of up to ``main_pool_rows`` rows.
    """
    bank = 0
    if width > _SWA_TILE:
        bank = main_bank_reserved_rows() * (_HEAD_DIM + 1)
        if generation_queries > 0:
            selections = generation_queries * (width - _SWA_TILE)
            bank += _HEAD_DIM * min(
                max(selections, main_bank_reserved_rows()), main_records_rows_limit()
            )
            bank += 8 * (main_pool_rows + 1)
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


class CSA2PrimsTS:
    """Module compute helper; plans and serial scratch belong to query metadata."""

    # A selection bank converts each selected MAIN row once (dedup_main_records).
    dedup_selection_bank = True

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
    ) -> dict:
        from ._prims_ts_packed import PackedSparseMLA, precompile_packed_sparse_mla

        rows = metadata.num_tokens
        heads = self.attention.num_heads
        width = metadata.num_sparse_topk
        key = (
            q.device,
            rows,
            heads,
            width,
            swa_cache_format,
            main_cache_format,
            output_groups,
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
        if (main_cache_format, output_groups) not in arena["main_formats"]:
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
                valid_prefix=True,
                output_groups=output_groups,
            )
            arena["main_formats"].add((main_cache_format, output_groups))
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
            # its live prefix, so plain E4M3 rows may be gathered by TMA.
            valid_prefix=True,
            output_groups=output_groups,
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

        if (
            q.shape != (metadata.num_tokens, self.attention.num_heads * _HEAD_DIM)
            or q.dtype != torch.bfloat16
            or not q.is_cuda
            or not q.is_contiguous()
        ):
            raise ValueError("CSA2 PrimTS requires contiguous CUDA BF16[T,H*512] queries")
        if metadata.kv_input_mode != "packed":
            raise ValueError("CSA2 PrimTS requires packed KV input metadata without staging")
        self.attention._validate_helper_output(q, forward_args)
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
        raw = forward_args.sparse_backend_args
        if not isinstance(raw, CSA2BackendForwardArgs):
            raise TypeError("CSA2 requires CSA2BackendForwardArgs")
        swa_cache_format = raw.swa_cache_format or self.attention.swa_cache_format
        # An eager context phase reads its MAIN rows from a bank decoded once per
        # row, not per selection: all of its requests' rows, or only the selected
        # ones when its queries select fewer rows than the requests own.
        bank = metadata.prims_ts_main_bank
        mxfp8 = self._mxfp8_output(q, metadata, forward_args)
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
        with torch.cuda.device(q.device):
            # Reject cold capture before cache writes or indexer side effects.
            record = self._get_plan(q, metadata, swa_cache_format, main_cache_format, output_groups)
            inputs = self.attention.prepare_sparse_inputs(q, metadata, forward_args)
            if (inputs.swa_cache_format or "swa") != swa_cache_format:
                raise ValueError("CSA2 PrimTS prepared SWA format must match its planned format")
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
            if not early:
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
            query = self._context_query(q, raw, forward_args)
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
        from ._prims_ts_packed import fuses_mxfp8_output

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
        width = metadata.num_sparse_topk
        if not fuses_mxfp8_output(
            q.device, metadata.num_tokens, self.attention.num_heads, _SWA_TILE, width - _SWA_TILE
        ):
            return None
        return mxfp8
