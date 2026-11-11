# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-step batch state every attention metadata carries.

``AttentionCoreMetadata`` is the extension a metadata attaches first: it holds
the batch layout (query and KV lengths, request and token counts, request IDs,
prompt lengths, packed-token offsets) and the per-step inputs the engine
describes the batch with. Each metadata family subclasses it with its own
state (device buffers, block offsets, planner state) and derives that state
from the batch in ``derive()``. The metadata forwards every declared field, so
``metadata.seq_lens`` reads the core's.
"""

import contextlib
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

from ...memory_buffer_utils import get_memory_buffers
from .extensions import MetadataExtension

if TYPE_CHECKING:
    from ...pyexecutor.kv_cache.mixed_precision_staging import MixedPrecisionKVStaging
    from ...pyexecutor.llm_request import LlmRequest
    from .interface import BatchInputs


@dataclass
class AttentionRuntimeFeatures:
    chunked_prefill: bool = False
    cache_reuse: bool = False
    has_speculative_draft_tokens: bool = False
    # This is the chunk size for MLA chunked prefill, which splits KV cache into chunks.
    chunk_size: int = 0
    # The real chunk size for MLA chunked prefill is this value * chunk_size.
    chunked_prefill_buffer_batch_size: int = 4


class AttentionCoreMetadata(MetadataExtension):
    """Batch layout and per-step inputs of an attention metadata."""

    is_core = True
    fields = frozenset(
        {
            "max_num_requests",
            "max_num_tokens",
            "max_num_sequences",
            "mapping",
            "kv_cache_manager",
            "kv_layout",
            "runtime_features",
            "is_cuda_graph",
            "enable_context_mla_with_cached_kv",
            "enable_flash_mla",
            "enable_helix",
            "fp4_mla_state",
            "cache_indirection",
            "cuda_graph_buffers",
            "full_block_table_width",
            "mixed_precision_staging_name",
            "mixed_precision_staging",
            "seq_lens",
            "seq_lens_cuda",
            "seq_lens_kv",
            "seq_lens_kv_cuda",
            # The length buffers and counts behind the properties above, which
            # callers outside the core still read and write directly.
            "_seq_lens",
            "_seq_lens_cuda",
            "_seq_lens_kv",
            "_seq_lens_kv_cuda",
            "_num_contexts",
            "_num_generations",
            "_num_ctx_tokens",
            "_num_tokens",
            "num_contexts",
            "num_generations",
            "num_ctx_tokens",
            "num_tokens",
            "num_seqs",
            "context_lens",
            "is_cross",
            "position_ids",
            "kv_cache_params",
            "request_ids",
            "prompt_lens",
            "cu_q_seqlens",
            "cu_kv_seqlens",
            "all_rank_num_tokens",
            "padded_num_tokens",
            "num_chunked_ctx_requests",
            "beam_width",
            "multi_item_part_lens",
            "context_requests",
            "prefill_cuda_graphs",
            "kv_lens_offsets",
            "position_id_offsets",
            "lengths_changed",
            "steady_decode_kv_lens",
        }
    )
    methods = frozenset(
        {
            "clear_lengths_changed",
            "lengths_for_kernel_update",
            "resync_length_derived_state",
            "on_update",
            "bind_lengths",
            "mark_cuda_graph",
            "set_all_rank_num_tokens",
            "set_padded_num_tokens",
            "set_context_requests",
            "override_cached_token_lens",
            "set_cached_token_counts",
            "set_seq_lens",
            "adjust_seq_lens",
            "sync_seq_lens_to_device",
            "copy_lengths_from",
            "adjust_kv_lens",
            "set_kv_lens",
            "as_generation_rows",
            "prepare_mixed_precision_kv",
            "get_mixed_precision_kv_view",
            "mixed_precision_kv_staging",
        }
    )
    init_fields = {
        # Capacities: requests, tokens and sequences (requests times beams) in
        # one batch. The family fills in max_num_sequences when unset.
        "max_num_requests": 0,
        "max_num_tokens": 0,
        "max_num_sequences": None,
        "mapping": None,
        # The KV cache manager; None for attention without a KV cache.
        "kv_cache_manager": None,
        # Paged KV-cache block layout:
        # NHD: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
        # HND: [max_num_pages, 2, num_kv_heads, page_size, head_dim]
        "kv_layout": "HND",
        # Features the runtime may use; kernels may behave differently with them.
        "runtime_features": AttentionRuntimeFeatures(),
        # Whether this metadata is a CUDA graph's: buffers are bound once and
        # updated in place.
        "is_cuda_graph": False,
        "enable_context_mla_with_cached_kv": False,
        # The beam-search cache indirection buffer TRTLLM attention reads.
        "cache_indirection": None,
        # The buffer pool persistent buffers come from; None allocates directly.
        "cuda_graph_buffers": None,
        # Whether every step stages the whole block table, as for a metadata
        # whose device KV lengths run past the host snapshot between prepares
        # (a one-model draft over its own KV cache).
        "full_block_table_width": False,
        # Buffer-name prefix of the mixed-precision staging copy; metadata over
        # different KV cache managers in one buffer pool need different ones.
        "mixed_precision_staging_name": "mixed",
        "seq_lens": None,
        "num_contexts": 0,
        # Position of each token in each sequence; None when positional
        # embedding is applied outside the backend.
        "position_ids": None,
        "kv_cache_params": None,
        "seq_lens_kv": None,
        # Request ID and prompt length of each sequence. A context sequence's
        # prompt length is its token count; a generation sequence's is the
        # token count of its context phase.
        "request_ids": None,
        "prompt_lens": None,
        # Explicit query/KV sequence boundaries [num_seqs + 1] for kernels on
        # packed varlen inputs; sequence or segment boundaries, not
        # necessarily request boundaries.
        "cu_q_seqlens": None,
        "cu_kv_seqlens": None,
        # Token count of each attention-DP rank, and this rank's padded count.
        "all_rank_num_tokens": None,
        "padded_num_tokens": None,
        "beam_width": 1,
        # Per-request multi-item scoring layout; see ``TokensPrompt``.
        "multi_item_part_lens": None,
    }

    # Storage behind the length and count properties, read by their setters
    # while the base constructor sets the init fields.
    _seq_lens: Optional[torch.Tensor] = None
    _seq_lens_kv: Optional[torch.Tensor] = None
    _seq_lens_cuda: Optional[torch.Tensor] = None
    _seq_lens_kv_cuda: Optional[torch.Tensor] = None
    _num_contexts = 0
    _num_generations = 0
    _num_ctx_tokens = 0
    _num_tokens = 0

    #: Whether the family's kernels address the paged KV cache through
    #: attention-op block offsets, which the draft KV cache then also needs.
    pages_by_block_offsets: ClassVar[bool] = False
    #: Constructor arguments every metadata must be given.
    required_init_fields: ClassVar[Tuple[str, ...]] = ("max_num_requests", "max_num_tokens")
    # Copies share the pinned staging rings and the mixed-precision staging
    # cache (built from the shared buffer pool on the first prepare).
    shared = frozenset({"_host_staging", "mixed_precision_staging"})
    # A draft binds the target's batch description, counts and mixed-precision
    # staging copy, writes its query lengths in place, and keeps its KV cache
    # managers, buffer pool, pinned staging rings and saved length tensors.
    draft_bound = frozenset(
        {
            "max_num_requests",
            "max_num_tokens",
            "max_num_sequences",
            "mapping",
            "kv_layout",
            "runtime_features",
            "enable_context_mla_with_cached_kv",
            "cache_indirection",
            "_seq_lens_kv",
            "_seq_lens_kv_cuda",
            "_num_contexts",
            "_num_generations",
            "_num_ctx_tokens",
            "_num_tokens",
            "position_ids",
            "kv_cache_params",
            "request_ids",
            "prompt_lens",
            "cu_q_seqlens",
            "cu_kv_seqlens",
            "all_rank_num_tokens",
            "padded_num_tokens",
            "num_chunked_ctx_requests",
            "beam_width",
            "multi_item_part_lens",
            "context_requests",
            "prefill_cuda_graphs",
            "kv_lens_offsets",
            "position_id_offsets",
            # Over the target's KV cache the draft runs on the target's
            # staging copy; a separate draft manager's is built by the draft.
            "mixed_precision_staging",
        }
    )
    draft_written = frozenset({"_seq_lens", "_seq_lens_cuda"})
    draft_private = frozenset(
        {
            "kv_cache_manager",
            "cuda_graph_buffers",
            "is_cuda_graph",
            "_host_staging",
            "lengths_changed",
            "steady_decode_kv_lens",
            "full_block_table_width",
            "mixed_precision_staging_name",
        }
    )

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        # Pinned staging rings for stage_to_device, shared with metadata copies.
        self._host_staging: Dict[Any, Any] = self.shared_or("_host_staging", dict)
        # Mixed-precision KV: the staging copy attention reads, created by the
        # first prepare() on a manager with mixed-precision layers.
        self.mixed_precision_staging: Optional["MixedPrecisionKVStaging"] = self.shared_or(
            "mixed_precision_staging", lambda: None
        )
        # Host per-sequence query / KV lengths and their device copies (the KV
        # lengths default to the query lengths unless set, as for cross
        # attention), and the counts derived on the host whenever the lengths
        # or the context count change, so compiled code never computes them
        # from tensors. The length setters above may have set them already.
        for name in (
            "_seq_lens",
            "_seq_lens_kv",
            "_seq_lens_cuda",
            "_seq_lens_kv_cuda",
            "_num_contexts",
            "_num_generations",
            "_num_ctx_tokens",
            "_num_tokens",
        ):
            self.__dict__.setdefault(name, getattr(type(self), name))
        # Trailing context requests that are speculative-decoding extend
        # requests; their device KV lengths are corrected with the generation
        # rows.
        self.num_chunked_ctx_requests = 0
        # This step's scheduled context requests, or None when not given, and
        # whether context forwards may run in CUDA graphs.
        self.context_requests: Optional[List["LlmRequest"]] = None
        self.prefill_cuda_graphs = False
        # This step's device corrections to the KV lengths
        # (``BatchInputs.kv_lens_offsets``), or None.
        self.kv_lens_offsets: Optional[torch.Tensor] = None
        # This step's device corrections to the generation tokens' positions
        # (``BatchInputs.position_id_offsets``), or None.
        self.position_id_offsets: Optional[torch.Tensor] = None
        # Whether a length primitive changed the lengths in place since the
        # last prepare or refresh: state derived from them is stale, and the
        # attention forward refuses to run.
        self.lengths_changed = False
        # The scheduled KV lengths of a steady decode step
        # (``prepare_steady_decode_step``); None after a full prepare.
        self.steady_decode_kv_lens: Optional[List[int]] = None

    def derive(self) -> None:
        """Derive the family's per-step state from the batch; runs in ``prepare``."""

    def prepare_encoder_only(self) -> None:
        """Encoder-only (no-KV-cache) setup; defaults to the full ``prepare()``."""
        self.metadata.prepare()

    def on_update(self) -> None:
        """Recompute the token and request counts from the host lengths."""
        if (
            self._seq_lens is not None
            and self._seq_lens.shape[0] >= self._num_contexts
            and self._num_contexts >= 0
        ):
            self._num_ctx_tokens = self._seq_lens[: self._num_contexts].sum().item()
            self._num_generations = self._seq_lens.shape[0] - self._num_contexts
        if self._seq_lens_kv is not None:
            self._num_tokens = self._seq_lens_kv.sum().item()
        elif self._seq_lens is not None:
            self._num_tokens = self._seq_lens.sum().item()

    @property
    def seq_lens(self) -> Optional[torch.Tensor]:
        return self._seq_lens

    @seq_lens.setter
    def seq_lens(self, value: Optional[torch.Tensor]) -> None:
        self._seq_lens = value
        self.on_update()

        # The model executor sets seq_lens to None initially.
        if self._seq_lens is not None:
            self._seq_lens = maybe_pin_memory(self._seq_lens)

            if self.is_cuda_graph and self._seq_lens_cuda is not None:
                # Very important: do not reallocate if we are using CUDA graphs.
                # This copy is safe because the batch size is guaranteed to not
                # change in the CUDA graph case. The seqlens can change if we
                # are doing spec decode.
                self._seq_lens_cuda.copy_(self._seq_lens, non_blocking=True)
            else:
                self._seq_lens_cuda = self._seq_lens.cuda(non_blocking=True)

    @property
    def num_contexts(self) -> int:
        return self._num_contexts

    @num_contexts.setter
    def num_contexts(self, value: int) -> None:
        self._num_contexts = value
        self.on_update()

    @property
    def num_generations(self) -> int:
        return self._num_generations

    @num_generations.setter
    def num_generations(self, value: int) -> None:
        self._num_generations = value
        self.on_update()

    @property
    def seq_lens_cuda(self) -> Optional[torch.Tensor]:
        return self._seq_lens_cuda

    @property
    def seq_lens_kv(self) -> Optional[torch.Tensor]:
        return self._seq_lens_kv if self._seq_lens_kv is not None else self._seq_lens

    @seq_lens_kv.setter
    def seq_lens_kv(self, value: Optional[torch.Tensor]) -> None:
        self._seq_lens_kv = value
        self.on_update()
        # The model executor sets seqlens to None initially.
        if self._seq_lens_kv is not None:
            self._seq_lens_kv = maybe_pin_memory(self._seq_lens_kv)
            if self.is_cuda_graph and self._seq_lens_kv_cuda is not None:
                self._seq_lens_kv_cuda.copy_(self._seq_lens_kv, non_blocking=True)
            else:
                self._seq_lens_kv_cuda = self._seq_lens_kv.cuda(non_blocking=True)

    @property
    def seq_lens_kv_cuda(self) -> Optional[torch.Tensor]:
        if self._seq_lens_kv_cuda is not None:
            return self._seq_lens_kv_cuda
        return self._seq_lens_cuda

    @property
    def context_lens(self) -> torch.Tensor:
        """The query length of each context sequence, [num_contexts]."""
        return self.seq_lens[: self.num_contexts]

    @property
    def num_seqs(self) -> int:
        """The number of sequences in the batch."""
        return self.seq_lens.shape[0]

    @property
    def enable_flash_mla(self) -> bool:
        """Whether FlashMLA serves the generation phase (its extension is attached)."""
        from .features import FlashMlaMetadata

        return self.metadata.find_ext(FlashMlaMetadata) is not None

    @property
    def enable_helix(self) -> bool:
        """Whether Helix context parallelism is on (its extension is attached)."""
        from .features import HelixMetadata

        return self.metadata.find_ext(HelixMetadata) is not None

    @property
    def fp4_mla_state(self):
        """The FP4 MLA extension when it is attached, else None."""
        from .fp4_mla.state import Fp4MlaState

        return self.metadata.find_ext(Fp4MlaState)

    @property
    def is_cross(self) -> bool:
        """Whether this metadata is for cross attention."""
        return self.seq_lens is not self.seq_lens_kv

    @property
    def num_ctx_tokens(self) -> int:
        return self._num_ctx_tokens

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    # Length changes. The lengths and counts belong to the core; everything
    # else changes them through these methods, which keep the device buffers
    # in place and set ``lengths_changed``. Extensions correct them only from
    # their own ``prepare`` (``prepare`` clears the flag after every extension
    # prepared); after prepare only draft loops change them, on their draft
    # metadata, and then refresh the draft before its next forward.

    def clear_lengths_changed(self) -> None:
        """Mark the state derived from the lengths as current."""
        self.lengths_changed = False

    def resync_length_derived_state(self) -> None:
        """Re-derive the family state that follows the lengths eagerly.

        For a draft refresh after in-place length changes; state derived
        lazily is invalidated by the length primitives themselves.
        """

    def lengths_for_kernel_update(self) -> Dict[str, torch.Tensor]:
        """The device length buffers, for a fused kernel that updates them.

        The only way a kernel may write the lengths: the buffers are marked
        changed, so the next forward sees them refreshed. Keys are
        ``seq_lens_cuda`` and, for a family with device KV lengths,
        ``kv_lens_cuda``.
        """
        self.lengths_changed = True
        buffers = {"seq_lens_cuda": self._seq_lens_cuda}
        kv_lens = getattr(self, "kv_lens_cuda", None)
        if kv_lens is not None:
            buffers["kv_lens_cuda"] = kv_lens
        return buffers

    def apply_batch_inputs(self, inputs: "BatchInputs") -> None:
        """Take one step's batch description; ``prepare`` derives from it."""
        if inputs.seq_lens_cuda is not None:
            self.bind_lengths(inputs.seq_lens, inputs.seq_lens_cuda)
        elif inputs.seq_lens is not None:
            self.seq_lens = inputs.seq_lens
        if inputs.seq_lens_kv is not None:
            self.seq_lens_kv = inputs.seq_lens_kv
        self.beam_width = inputs.beam_width
        self.request_ids = inputs.request_ids
        self.prompt_lens = inputs.prompt_lens
        self.num_contexts = inputs.num_contexts
        self.num_chunked_ctx_requests = inputs.num_chunked_ctx_requests
        self.kv_cache_params = inputs.kv_cache_params
        self.cu_q_seqlens = inputs.q_offsets
        self.cu_kv_seqlens = inputs.kv_offsets
        self.kv_lens_offsets = inputs.kv_lens_offsets
        self.position_id_offsets = inputs.position_id_offsets
        self.multi_item_part_lens = inputs.multi_item_part_lens

    def bind_lengths(
        self,
        host: torch.Tensor,
        device: Optional[torch.Tensor] = None,
        *,
        num_contexts: Optional[int] = None,
        num_ctx_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        num_tokens: Optional[int] = None,
    ) -> None:
        """Bind the query lengths as given, without copying.

        ``device`` is bound as the device copy (shared with its owner, such as
        a parent metadata or a fixed-geometry tile); None stages ``host`` into
        the bound device buffer of a CUDA graph, or a new one. Counts given are
        taken as they are,
        so a caller that already knows them skips the host reductions; the
        others are recomputed from ``host``.
        """
        self._seq_lens = host
        if device is not None:
            self._seq_lens_cuda = device
        elif self.is_cuda_graph and self._seq_lens_cuda is not None:
            # Keep the graph's buffer; the batch size is fixed in a graph.
            self._seq_lens_cuda.copy_(host, non_blocking=True)
        else:
            self._seq_lens_cuda = host.cuda(non_blocking=True)
        if num_contexts is not None:
            self._num_contexts = num_contexts
        counts = (num_ctx_tokens, num_generations, num_tokens)
        if all(count is not None for count in counts):
            self._num_ctx_tokens, self._num_generations, self._num_tokens = counts
        else:
            self.on_update()

    def set_context_requests(
        self, requests: Optional[List["LlmRequest"]], prefill_cuda_graphs: bool
    ) -> None:
        """Take this step's scheduled context requests (None when not given)."""
        self.context_requests = requests
        self.prefill_cuda_graphs = prefill_cuda_graphs

    def override_cached_token_lens(self, num_cached_tokens_per_seq: List[int]) -> None:
        """Present other cached-token counts than the step's KV-cache parameters.

        For an extension that attends to fewer cached tokens than the batch
        holds (RocketKV's prompt budget); called from its ``prepare`` after
        the core derived. The step's ``kv_cache_params`` object is replaced,
        never changed in place; a family that derives KV lengths from the
        counts re-derives them.
        """
        self.kv_cache_params = dataclasses.replace(
            self.kv_cache_params, num_cached_tokens_per_seq=list(num_cached_tokens_per_seq)
        )

    def set_cached_token_counts(self, num_cached_tokens_per_seq: List[int]) -> None:
        """Present other host cached-token counts to this metadata's forwards.

        For a draft loop whose draft forwards keep fewer cached tokens than
        the step's KV-cache parameters count. The ``kv_cache_params`` object,
        which a draft metadata shares with its target, is replaced, never
        changed in place; nothing is re-derived, so the device KV lengths
        stay as the caller set them. It counts as a length change.
        """
        self.kv_cache_params = dataclasses.replace(
            self.kv_cache_params, num_cached_tokens_per_seq=list(num_cached_tokens_per_seq)
        )
        self.lengths_changed = True

    def set_all_rank_num_tokens(self, counts: Optional[List[int]]) -> None:
        """Set the token count of each attention-DP rank."""
        self.all_rank_num_tokens = counts

    def set_padded_num_tokens(self, padded_num_tokens: Optional[int]) -> None:
        """Set the token count the model runs on when the batch is padded (None: unpadded).

        The padding is chosen after ``prepare``, from the prepared token
        counts of every attention-DP rank.
        """
        self.padded_num_tokens = padded_num_tokens

    def mark_cuda_graph(self, capturing: bool) -> None:
        """Mark whether this metadata's buffers are bound for graph capture."""
        self.is_cuda_graph = capturing

    def set_seq_lens(
        self,
        rows: slice,
        *,
        host: Union[int, torch.Tensor, None] = None,
        device: Union[int, torch.Tensor, None] = None,
        update_counts: bool = False,
    ) -> None:
        """Overwrite the query lengths of ``rows`` in place, host and/or device.

        The buffers keep their addresses, so captured graphs stay valid. With
        ``update_counts`` the token and request counts are recomputed from the
        host lengths; otherwise the step's counts stay.
        """
        if host is not None:
            self._seq_lens[rows] = host
        if device is not None:
            self._seq_lens_cuda[rows] = device
        if update_counts:
            self.on_update()
        self.lengths_changed = True

    def adjust_seq_lens(self, rows: slice, delta: int) -> None:
        """Add ``delta`` to the host and device query lengths of ``rows``.

        The token and request counts are recomputed from the new lengths.
        """
        self._seq_lens[rows] += delta
        self._seq_lens_cuda[rows] += delta
        self.on_update()
        self.lengths_changed = True

    def copy_lengths_from(self, source: "AttentionCoreMetadata") -> None:
        """Start this metadata's lengths and counts from ``source``'s, in place.

        A draft loop starts a draft metadata from its target's lengths this
        way, inside the captured graph, so repeated warmup and capture runs
        begin from the same lengths. The token padding the engine set on the
        target after its ``prepare`` is taken as well.
        """
        if (
            self._seq_lens is not None
            and self._seq_lens is not source._seq_lens
            and self._seq_lens.shape == source._seq_lens.shape
        ):
            self._seq_lens.copy_(source._seq_lens)
        else:
            self._seq_lens = source._seq_lens.clone()
        self._num_contexts = source._num_contexts
        self._num_generations = source._num_generations
        self._num_ctx_tokens = source._num_ctx_tokens
        self._num_tokens = source._num_tokens
        num_seqs = self.num_seqs
        self._seq_lens_cuda[:num_seqs].copy_(source.seq_lens_cuda[:num_seqs])
        self.padded_num_tokens = source.padded_num_tokens
        self.all_rank_num_tokens = source.all_rank_num_tokens
        self.lengths_changed = True

    def sync_seq_lens_to_device(self) -> None:
        """Copy the host query lengths into the bound device buffer in place.

        The device copy takes the host lengths the state was derived from (an
        encoder CUDA graph's prepare sets them on the host), so this is no
        length change.
        """
        self._seq_lens_cuda.copy_(self._seq_lens, non_blocking=True)

    def adjust_kv_lens(
        self,
        rows: slice,
        delta: Union[int, torch.Tensor],
        *,
        min_value: Optional[int] = None,
    ) -> None:
        """Add ``delta`` to the device KV lengths of ``rows`` in place.

        ``kv_lens_cuda`` ([max_num_sequences], int32, without the extra KV
        tokens) of a family that keeps device KV lengths. With ``min_value``
        the new lengths are clamped from below. Nothing derived from the
        lengths is refreshed here; the draft is refreshed before its next
        forward.
        """
        kv_lens = self.kv_lens_cuda[rows]
        kv_lens += delta
        if min_value is not None:
            kv_lens.clamp_(min=min_value)
        self.lengths_changed = True

    def set_kv_lens(self, rows: slice, values: torch.Tensor) -> None:
        """Overwrite the device KV lengths of ``rows`` in place; see ``adjust_kv_lens``."""
        self.kv_lens_cuda[rows].copy_(values)
        self.lengths_changed = True

    def as_generation_rows(self) -> None:
        """Continue every context row of a draft metadata as a generation row.

        For the first step of a one-model draft loop: the context requests
        drafted their first token from the prompt and, from the next draft
        forward on, run like the generation requests. The context count becomes
        zero and the token and request counts are recomputed from the host
        lengths, which the draft loop sets first (``set_seq_lens``). A family
        core also drops the state it keeps only for context rows; the next
        draft refresh sees the new rows.
        """
        self._num_contexts = 0
        self.on_update()
        self.lengths_changed = True

    def cuda_graph_copy_changes(self, max_batch_size: int, max_draft_tokens: int) -> Dict[str, Any]:
        """Creation arguments a CUDA graph copy changes besides the graph flag and pool."""
        return {}

    def creation_value(self, name: str) -> Any:
        # The stored lengths, not the getters' fallbacks: KV lengths default
        # to the query lengths only while unset.
        if name in ("seq_lens", "seq_lens_kv"):
            return self.__dict__[f"_{name}"]
        return super().creation_value(name)

    def reset_cuda_graph_lengths(
        self,
        max_batch_size: int,
        *,
        max_draft_tokens: int,
        sub_cross_metadata: bool,
        encode_only: bool,
    ) -> None:
        """Bind graph-sized length buffers on a CUDA graph copy.

        The device query lengths are reallocated at the graph's batch size;
        a cross-attention sub-metadata keeps its parent's. Encoder graphs vary
        the query lengths per replay; decoder graphs are generation-only, so
        their context count is zero (encoder graphs set it per replay).
        """
        if not sub_cross_metadata:
            self._seq_lens_cuda = None
            if encode_only:
                self.seq_lens = torch.ones((max_batch_size,), dtype=torch.int)
            else:
                self.seq_lens = torch.ones((max_batch_size,), dtype=torch.int) * (
                    1 + max_draft_tokens
                )
        if self.is_cross:
            self.seq_lens_kv = torch.zeros((max_batch_size,), dtype=torch.int)
        if not encode_only:
            self.num_contexts = 0

    def _has_mixed_precision_staging(self) -> bool:
        """Whether attention runs on a staging copy of the mixed-precision layers.

        An extension whose kernels read the mixed-precision pools another way
        declares the field ``mixed_precision_staging_enabled`` as False (DSA
        reads its gather's FP8 scratch).
        """
        if not self.metadata.find("mixed_precision_staging_enabled", True):
            return False
        # isinstance guards against mocks and against V1 managers, which have
        # no mixed-precision layers.
        high_layers = getattr(self.kv_cache_manager, "mixed_precision_high_layers", None)
        return isinstance(high_layers, dict) and bool(high_layers)

    def prepare_mixed_precision_kv(self, layout: str, tokens_per_block: int) -> None:
        """Create the staging copy on first use, then refresh it per batch.

        The staging copy is of this metadata's KV cache manager; a one-model
        draft over a separate draft manager (``md.draft``) builds its own,
        under buffer names of its own.
        """
        from ...pyexecutor.kv_cache.mixed_precision_staging import MixedPrecisionKVStaging

        name = self._mixed_precision_staging_name()
        view = self.mixed_precision_staging
        if view is None or view.manager is not self.kv_cache_manager:
            # Built once per metadata object (eager or CUDA graph copy); the
            # reserved buffer-pool block is shared, so the staging copy is
            # allocated and zeroed once per rank and the KV budget profiling
            # run already accounts for it.
            buffers = (
                self.cuda_graph_buffers
                if self.cuda_graph_buffers is not None
                else get_memory_buffers()
            )
            view = MixedPrecisionKVStaging(
                self.kv_cache_manager,
                buffers,
                rows=self.max_num_sequences or self.max_num_requests,
                layout=layout,
                pool_layout=self.kv_layout,
                tokens_per_block=tokens_per_block,
                name=name,
            )
            logger.info_once(
                f"Mixed-precision KV staging ({name}) on this rank: "
                f"{view.staging_bytes / (1 << 20):.1f} MiB = {view.rows} "
                f"requests x {view.max_blocks_per_seq} pages x "
                f"{view.max_page_bytes} B ({layout}, {tokens_per_block} "
                "tokens per page)",
                key=f"{name}_precision_kv_staging",
            )
            self.mixed_precision_staging = view
        view.update(
            self.request_ids,
            self.kv_cache_params.num_cached_tokens_per_seq,
            self.seq_lens.tolist(),
            capture_graph=self.is_cuda_graph,
        )
        self._on_mixed_precision_kv_prepared(view)

    def _mixed_precision_staging_name(self) -> str:
        """Buffer-name prefix of this metadata's staging copy (creation setting)."""
        return self.mixed_precision_staging_name

    def _on_mixed_precision_kv_prepared(self, view: "MixedPrecisionKVStaging") -> None:
        """Backend hook: refresh backend-specific views of the staging cache."""

    def get_mixed_precision_kv_view(self, layer_idx: int) -> Optional["MixedPrecisionKVStaging"]:
        """Return the staging cache of a mixed-precision layer, else None."""
        staging = self.mixed_precision_staging
        if (
            staging is None
            or layer_idx not in staging.pools
            or staging.manager is not self.kv_cache_manager
        ):
            # No staging for a layer without a mixed-precision high pool, or of a manager not in use.
            return None
        return staging

    @contextlib.contextmanager
    def mixed_precision_kv_staging(self, layer_idx: int):
        """Run attention on the staging copy, then write the new tokens back."""
        view = self.get_mixed_precision_kv_view(layer_idx)
        if view is None:
            yield
            return
        view.convert(layer_idx)
        saved = self._swap_mixed_precision_kv(view)
        try:
            yield
        finally:
            self._restore_mixed_precision_kv(saved)
        view.convert(layer_idx, writeback=True)

    def _swap_mixed_precision_kv(self, view: "MixedPrecisionKVStaging") -> tuple:
        """Point this metadata at the staging cache; return what to restore."""
        saved = (self.kv_cache_manager, self.kv_layout)
        self.kv_cache_manager = view
        self.kv_layout = view.layout
        return saved

    def _restore_mixed_precision_kv(self, saved: tuple) -> None:
        """Undo _swap_mixed_precision_kv."""
        self.kv_cache_manager, self.kv_layout = saved
