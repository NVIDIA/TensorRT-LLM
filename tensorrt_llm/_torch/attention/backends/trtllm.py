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

import contextlib
import copy
import dataclasses
import math
import os
import weakref
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, FrozenSet, Iterator,
                    List, Optional, Tuple, Type)

import numpy as np
import torch

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

    from ...model_config import ModelConfig
    from .fmha.gqa_decode import GqaDecodeFmha, GqaDecodeSetup
    from .fmha.interface import Fmha
    from .sparse.dsa.metadata import TokenMajorGenView

from tensorrt_llm._utils import (get_sm_version, is_sm_100f, is_sm_120f,
                                 maybe_pin_memory, prefer_pinned)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import (AttentionMaskType, PositionEmbeddingType,
                                     RotaryScalingType)
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantMode

from ...distributed import cp_allgather
from ...kimi_k3_cache_policy import get_kimi_k3_fp8_kv_layer_ids, is_kimi_k3
from ...pyexecutor import step_inputs
from ...pyexecutor.config_utils import is_mla
from ...pyexecutor.h2d_cache import copy_h2d_pool_shared
from ...utils import (compute_swizzled_sf_shape, get_global_attrs,
                      get_model_extra_attrs, helix_local_len_tensor)
from ..kernels.sage_quant import SageQuantized, sage_quantize
from .core import AttentionCoreMetadata
from .extensions import MetadataExtension
from .features import DraftMetadata, FlashMlaMetadata, HelixMetadata
from .features.spec_decoding import SpecDecodingMetadata
from .fmha.fp4_mla import Fp4MlaFmha
from .fmha.manager import FmhaManager
from .fp4_mla import (can_fuse_fp4_mla_q_quant, load_fp4_mla_chunked_kv_cache,
                      scatter_fp4_mla_kv_cache)
from .fp4_mla.state import Fp4MlaState
from .interface import (AttentionBackend, AttentionForwardArgs,
                        AttentionInputType, AttentionMask, AttentionMetadata,
                        BatchInputs, KVCacheParams, MLAParams,
                        PositionalEmbeddingParams, PredefinedAttentionMask,
                        RopeParams, merge_attention_forward_args)
from .sparse.hooks import prepare_sparse_runtime_params
from .sparse.params import BlockSparseForwardInputs, SparseParams
from .sparse.skip_softmax import SkipSoftmaxParams
from .utils import log_attention_failure_context

_SKIP_CORRECTION_SUPPORTED_SMS = frozenset((100, 103))


def _resolve_skip_correction_threshold(threshold: float,
                                       sm_version: int,
                                       *,
                                       is_mla: bool = True) -> float:
    if not is_mla:
        return 0.0
    threshold = float(threshold)
    if threshold <= 0.0:
        return 0.0
    if sm_version in _SKIP_CORRECTION_SUPPORTED_SMS:
        return threshold
    logger.warning_once(
        "Skip-correction is supported only on SM100 and SM103; "
        f"disabling it on SM{sm_version}.",
        key="skip_correction_unsupported_sm",
    )
    return 0.0


def _resolve_uses_spcompress(sparse_params: Optional[SparseParams],
                             sm_version: int) -> bool:
    uses_spcompress = bool(
        isinstance(sparse_params, SkipSoftmaxParams)
        and sparse_params.uses_spcompress)
    if not uses_spcompress:
        return False
    if sm_version == 107:
        return True
    logger.warning_once(
        "spcompress is supported only on SM107; "
        f"disabling it on SM{sm_version}.",
        key="uses_spcompress_unsupported_sm",
    )
    return False


class _ContextPlan:
    """Host state of a context-only trtllm-gen call, shared by the layers of one forward.

    TrtllmAttentionMetadata drops it whenever the lengths or the batch can change (prepare(), KV-length and
    spec-dec updates, draft swaps); the bound tensors are also compared by identity. ``sig`` is the
    TrtllmAttention._context_plan_sig of the layer that built it.
    """

    __slots__ = ("kv_lens", "prompt_lens", "kv_lens_cuda", "prompt_lens_cuda",
                 "num_contexts", "num_ctx_tokens", "block_offsets", "workspace",
                 "kv_cache_manager", "pool_pointers", "sig", "fmha_cls",
                 "counter_rows", "total_num_blocks", "max_q_len", "max_kv_len")


@dataclasses.dataclass(frozen=True, kw_only=True)
class MlaLatentWriteTarget:
    """Where ``mla_rope_generation`` stores each token's new latent row.

    By default the row goes to the token's position in the paged cache. A
    target instead stores it in one row of a flat row array, such as a sparse
    cache arena: pool 0 of ``pool_pointers`` starts at the array, the layer's
    ``pool_mapping`` row selects pool 0, and each request owns a one-token page
    whose block offset in ``block_offsets`` [1, requests, 2, 1] is its row.
    An NVFP4 cache's pointers are [1, 2, 2]: the block scales go to the same row
    of a second array. ``write_slots`` [tokens] is 0 for a token to store and
    -1 for a token to skip, such as CUDA graph padding.
    """

    block_offsets: torch.Tensor
    pool_pointers: torch.Tensor
    pool_mapping: torch.Tensor
    write_slots: torch.Tensor


@dataclasses.dataclass(frozen=True)
class HelixMlaChunkPlan:
    """CP-local read ranges for one global MLA history iteration.

    Tensor fields are CUDA int64 tensors. ``cu_seq_lens`` has shape
    [CP, contexts + 1], ``local_offsets`` has shape [CP, contexts], and
    ``rank_offsets`` has shape [CP + 1]. Host ``rank_sizes`` drives the
    variable-size collective; ``max_local_seq_len`` sizes this rank's read.
    """

    cu_seq_lens: torch.Tensor
    local_offsets: torch.Tensor
    rank_offsets: torch.Tensor
    rank_sizes: list[int]
    max_local_seq_len: int


class TrtllmCoreMetadata(AttentionCoreMetadata):
    """TRTLLM attention state: device lengths, block offsets, workspaces and MLA buffers.

    ``allocate`` binds the graph-stable buffers on creation and on every
    metadata copy; ``derive`` publishes the step's lengths, block offsets and
    the runtime views the attention op reads.
    """

    pages_by_block_offsets = True
    # Every CUDA graph copy runs on the same workspaces.
    shared = frozenset({"workspace", "cuda_graph_workspace"})
    adopts_target_state = True
    # A draft binds the target's derived lengths, block offsets and context
    # buffers. Its loop advances the device KV lengths, turns its context
    # rows into generation rows and stages the FlashMLA block table in place;
    # its runtime views slice its own buffers, and the MLA prefixes and split
    # plan, with the flags saying whether they are current, are its own,
    # built lazily from its own lengths.
    draft_bound = frozenset({
        "_max_seq_len_storage",
        "_page_size_override",
        "_prepared_full_block_offsets",
        "chunked_global_offset",
        "chunked_loop_num",
        "chunked_seq_len",
        "ctx_cached_token_indptr",
        "ctx_kv_indptr",
        "ctx_uncached_token_indptr",
        "cu_chunked_seq_len",
        "host_chunked_global_offset",
        "host_chunked_seq_len",
        "host_ctx_cached_token_indptr",
        "host_ctx_kv_indptr",
        "host_ctx_uncached_token_indptr",
        "host_cu_chunked_seq_len",
        "host_merge_op_tensor",
        "host_total_kv_lens",
        "helix_context_local_slots",
        "helix_mla_chunk_plans",
        "kv_block_ids_per_seq",
        "kv_cache_block_offsets",
        "kv_lens",
        "kv_lens_runtime",
        "max_chunk_len_per_loop",
        "max_context_q_len_override",
        "max_ctx_cached_token_len",
        "max_ctx_kv_len",
        "max_ctx_seq_len",
        "merge_op_tensor",
        "num_ctx_cached_tokens",
        "num_sparse_topk",
        "prompt_lens_cpu",
        "prompt_lens_cpu_runtime",
        "prompt_lens_cuda",
        "prompt_lens_cuda_runtime",
        "trtllm_gen_jit_warmup",
        "use_paged_context_fmha",
        "workspace_reclaimable",
    })
    draft_written = frozenset(
        {"kv_lens_cuda", "host_request_types", "block_ids_per_seq"})
    draft_private = frozenset({
        "_mla_scheduler_buffers_valid",
        "_mla_ctx_cu_seqlens_valid",
        "_cute_dsl_mla_split_plan_key",
        "host_kv_cache_block_offsets",
        "kv_lens_cuda_runtime",
        "host_request_types_runtime",
        "workspace",
        "cuda_graph_workspace",
        "fmha_plan_caches",
        "mla_cu_q_rows",
        "mla_cu_kv_seqlens",
        "mla_ctx_cu_q_seqlens",
        "_cute_dsl_mla_split_plan",
        "context_plan",
        "_steady_block_table_key",
        "_aligned_kv_lens_staging",
        "_context_scratch_helix",
    })

    fields = frozenset({
        "block_ids_per_seq",
        "chunked_global_offset",
        "chunked_loop_num",
        "chunked_seq_len",
        "ctx_cached_token_indptr",
        "ctx_kv_indptr",
        "context_plan",
        "ctx_uncached_token_indptr",
        "cu_chunked_seq_len",
        "cuda_graph_workspace",
        "effective_beam_width",
        "effective_workspace",
        "fmha_plan_caches",
        "host_chunked_global_offset",
        "host_chunked_seq_len",
        "host_ctx_cached_token_indptr",
        "host_ctx_kv_indptr",
        "host_ctx_uncached_token_indptr",
        "host_cu_chunked_seq_len",
        "host_kv_cache_block_offsets",
        "host_kv_cache_pool_mapping",
        "host_kv_cache_pool_pointers",
        "host_merge_op_tensor",
        "host_request_types",
        "host_request_types_runtime",
        "host_total_kv_lens",
        "helix_context_local_slots",
        "helix_mla_chunk_plans",
        "kv_block_ids_per_seq",
        "kv_cache_block_offsets",
        "kv_lens",
        "kv_lens_cuda",
        "kv_lens_cuda_runtime",
        "kv_lens_runtime",
        "max_chunk_len_per_loop",
        "max_context_length",
        "max_context_q_len_override",
        "max_ctx_cached_token_len",
        "max_ctx_kv_len",
        "max_ctx_seq_len",
        "max_seq_len",
        "merge_op_tensor",
        "mla_ctx_cu_q_seqlens",
        "mla_cu_kv_seqlens",
        "mla_cu_q_rows",
        "num_ctx_cached_tokens",
        "num_sparse_topk",
        "page_size",
        "prompt_lens_cpu",
        "prompt_lens_cpu_runtime",
        "prompt_lens_cuda",
        "prompt_lens_cuda_runtime",
        "tokens_per_block",
        "trtllm_gen_jit_warmup",
        "use_paged_context_fmha",
        "workspace",
        "workspace_reclaimable",
    })
    methods = frozenset({
        "allocate_context_kv_indptrs",
        "bind_context_tile",
        "bind_fixed_kv_window",
        "bind_generation_tile",
        "bind_runtime_views",
        "aligned_kv_lens_staging",
        "bind_encoder_cuda_graph_seq_lens",
        "cache_context_plan",
        "clear_context_cached_kv",
        "cute_dsl_mla_split_plan",
        "disable_workspace_reclamation",
        "mark_trtllm_gen_jit_warmup",
        "override_page_size",
        "override_prompt_lens",
        "republish_kv_lens",
        "context_scratch_view",
        "set_num_sparse_topk",
        "set_use_paged_context_fmha",
        "stage_context_cached_kv",
        "substitute_runtime_views",
        "use_block_id_tables",
        "use_block_offsets",
        "use_workspace",
        "reallocate_workspace",
        "mla_prepare_ctx_cu_seqlens",
        "mla_prepare_scheduler_buffers",
        "pre_process_for_chunked_prefill",
        "prepare_context_mla_with_cached_kv",
        "prepare_encoder_cuda_graph_replay",
        "prepare_encoder_decoder_from_precomputed_lengths",
        "key_prepared_block_offsets",
        "prepare_steady_decode_step",
        "supports_steady_decode_step",
        "presented_token_major",
        "token_major_view",
    })
    init_fields = {
        "workspace": None,
        "cuda_graph_workspace": None,
        # The sequence bound of no-cache attention; a cache's comes from its
        # manager.
        "max_seq_len": None,
        # Encoder CUDA graph compatibility: overrides host-side max_context_q_len
        # so FMHA kernel launch params are stable across graph capture/replay even
        # when actual per-batch sequence lengths vary. Only set by
        # EncoderCUDAGraphRunner; None elsewhere.
        "max_context_q_len_override": None,
        # TRTLLM-Gen FMHA JIT warmup controls.
        "trtllm_gen_jit_warmup": False,
        # Block offsets of the KV cache, device and host.
        "kv_cache_block_offsets": None,
        "host_kv_cache_block_offsets": None,
        # Per-sequence block IDs: generation rows (compacted) and every row. None
        # unless an extension that needs them (FlashMLA) provides them.
        "block_ids_per_seq": None,
        "kv_block_ids_per_seq": None,
        # FMHA prologue buffers for the MLA generation path, hoisted out of the per-layer
        # RoPE kernel. `mla_cu_q_rows` counts Q ROWS (cumsum(q_lens) * num_heads); the two
        # cu_kv/ctx buffers count KV lengths and context TOKENS respectively.
        "mla_cu_q_rows": None,
        "mla_cu_kv_seqlens": None,
        "mla_ctx_cu_q_seqlens": None,
        # The DSA extension sets this; the dense path keeps 0.
        "num_sparse_topk": 0,
    }

    def __init__(self, metadata, *, source=None, **config) -> None:
        super().__init__(metadata, source=source, **config)
        self.workspace_reclaimable = True
        # Plan caches of the FMHA libraries, keyed by library. A planned wrapper owns
        # workspaces that CUDA graphs capture, so it lives exactly as long as this
        # metadata; every layer that runs with this metadata reuses it.
        self.fmha_plan_caches = {}
        # Context-only plan shared by the layers of one forward
        # (TrtllmAttention.cache_context_plan); dropped whenever the lengths or
        # the batch can change. The plan also compares its bound tensors by
        # identity.
        self.context_plan: Optional[_ContextPlan] = None
        # batch_block_table_key of the KV cache manager at the last block-offset
        # copy of prepare_steady_decode_step or of a prepare keyed by
        # key_prepared_block_offsets; derive clears it.
        self._steady_block_table_key = None
        # Whether the last derive copied the full block-offset table width.
        self._prepared_full_block_offsets = False
        # Helix prefill: int32 slots in each request's rank-local KV cache,
        # one per context token (-1 skips persistence; temporary Q/K still
        # cover every token), and the CP-local read ranges of each chunked
        # MLA history iteration. None without Helix cached-KV context MLA.
        self.helix_context_local_slots: Optional[torch.Tensor] = None
        self.helix_mla_chunk_plans: Optional[List[HelixMlaChunkPlan]] = None
        # Kernel-aligned staging copy of misaligned device KV lengths
        # (aligned_kv_lens_staging), built on first use.
        self._aligned_kv_lens_staging: Optional[torch.Tensor] = None
        # The Helix state context_scratch_view presents positions through when
        # no Helix is attached; built on first use.
        self._context_scratch_helix: Optional[HelixMetadata] = None
        # Storage behind max_seq_len, which the base constructor may have set
        # already (no-cache attention).
        self.__dict__.setdefault("_max_seq_len_storage", None)
        self._page_size_override = None
        self.use_paged_context_fmha = False
        self._mla_scheduler_buffers_valid = False
        self._mla_ctx_cu_seqlens_valid = False
        # CuTe DSL MLA decode plans its per-request split-KV counts once per step
        # (cute_dsl_mla.py); the plan buffer and the key that marks it current.
        self._cute_dsl_mla_split_plan = None
        self._cute_dsl_mla_split_plan_key = None
        # Every CUDA graph copy runs on its source's workspaces.
        self.cuda_graph_workspace = self.shared_or(
            "cuda_graph_workspace", lambda: self.cuda_graph_workspace)
        self.workspace = self.shared_or("workspace", lambda: self.workspace)
        buffers = metadata.cuda_graph_buffers
        self.use_paged_context_fmha = (
            self.runtime_features.chunked_prefill
            or self.runtime_features.cache_reuse
            or self.runtime_features.has_speculative_draft_tokens
        ) if self.runtime_features is not None else False
        # CUDA-graph metadata is a shallow copy that re-runs this method; give it
        # its own plan caches so each captured batch size plans its own wrappers.
        self.fmha_plan_caches = {}
        self._allocate_buffers(buffers)

    def _allocate_buffers(self, buffers) -> None:
        md = self.metadata
        if self.max_num_sequences is None:
            self.max_num_sequences = self.max_num_requests

        capture_graph = self.is_cuda_graph

        self.prompt_lens_cuda = md.get_empty(
            buffers,
            (self.max_num_sequences, ),
            cache_name="prompt_lens_cuda",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.prompt_lens_cpu = torch.empty_like(
            self.prompt_lens_cuda,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        self.kv_lens_cuda = md.get_empty_like(
            buffers,
            self.prompt_lens_cuda,
            cache_name="kv_lens_cuda",
            capture_graph=capture_graph,
        )
        self.kv_lens = torch.empty_like(self.kv_lens_cuda,
                                        device='cpu',
                                        pin_memory=prefer_pinned())
        self.host_total_kv_lens = torch.empty(2, device='cpu', dtype=torch.int)
        self.host_request_types = torch.empty_like(self.prompt_lens_cpu)

        # `+ 1` because these are exclusive-prefix arrays over num_seqs.
        self.mla_cu_q_rows = md.get_empty(
            buffers,
            (self.max_num_sequences + 1, ),
            cache_name="mla_cu_q_rows",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        self.mla_cu_kv_seqlens = md.get_empty(
            buffers,
            (self.max_num_sequences + 1, ),
            cache_name="mla_cu_kv_seqlens",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        # Own copy because `ctx_uncached_token_indptr` holds the same values but only
        # exists under `enable_context_mla_with_cached_kv`.
        self.mla_ctx_cu_q_seqlens = md.get_empty(
            buffers,
            (self.max_num_sequences + 1, ),
            cache_name="mla_ctx_cu_q_seqlens",
            dtype=torch.int,
            capture_graph=capture_graph,
        )
        # Entry 0 of an exclusive prefix sum is always 0 and the rebuilds below only
        # write from index 1, so zero it once here instead of per rebuild.
        self.mla_cu_q_rows[:1].zero_()
        self.mla_cu_kv_seqlens[:1].zero_()
        self.mla_ctx_cu_q_seqlens[:1].zero_()
        self._invalidate_mla_scheduler_buffers()

        if self.workspace is None:
            self.workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )

        if self.cuda_graph_workspace is None:
            self.cuda_graph_workspace = torch.empty(
                (0, ),
                device='cuda',
                dtype=torch.int8,
            )

        if self.kv_cache_manager is not None:
            num_attention_op_pools = getattr(self.kv_cache_manager,
                                             "num_attention_op_pools",
                                             self.kv_cache_manager.num_pools)
            self.kv_cache_block_offsets = md.get_empty(
                buffers,
                [
                    num_attention_op_pools, self.max_num_sequences, 2,
                    self.kv_cache_manager.max_blocks_per_seq
                ],
                cache_name="kv_cache_block_offsets",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_kv_cache_block_offsets = self.kv_cache_manager.host_kv_cache_block_offsets
            self.block_ids_per_seq = None
            self.kv_block_ids_per_seq = None

            if self.enable_context_mla_with_cached_kv:
                # for kv cache reuse/chunked context in MLA
                self.allocate_context_kv_indptrs(buffers, capture_graph)
                self.ctx_uncached_token_indptr = md.get_empty(
                    buffers,
                    (self.max_num_requests + 1, ),
                    cache_name="ctx_uncached_token_indptr",
                    dtype=torch.int64,
                    capture_graph=capture_graph,
                )
                self.host_ctx_uncached_token_indptr = torch.zeros_like(
                    self.ctx_uncached_token_indptr,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )

    def allocate_context_kv_indptrs(self, buffers, capture_graph: bool) -> None:
        """Allocate the context cached-token and KV-length prefix buffers.

        ``ctx_cached_token_indptr`` / ``ctx_kv_indptr`` [max_num_requests + 1]
        (int64) and their pinned host copies; filled per step by
        ``stage_context_cached_kv`` or the context-MLA preparation.
        """
        md = self.metadata
        self.ctx_cached_token_indptr = md.get_empty(
            buffers,
            (self.max_num_requests + 1, ),
            cache_name="ctx_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_ctx_cached_token_indptr = torch.zeros_like(
            self.ctx_cached_token_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )
        # context full seqlens include cached tokens and uncached tokens
        self.ctx_kv_indptr = md.get_empty(
            buffers,
            (self.max_num_requests + 1, ),
            cache_name="ctx_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_ctx_kv_indptr = torch.zeros_like(
            self.ctx_kv_indptr,
            device='cpu',
            pin_memory=prefer_pinned(),
        )

    def adopt_target_state(self, source: "TrtllmCoreMetadata") -> None:
        """Take the target's prepared state at a draft-loop start.

        The runtime KV lengths are the draft's own device lengths; its MLA
        prefixes and split plan are its own, built lazily from its own lengths.
        """
        super().adopt_target_state(source)
        self._invalidate_mla_scheduler_buffers()
        num_seqs = self.num_seqs
        self.kv_lens_cuda_runtime = self.kv_lens_cuda[:num_seqs]
        if getattr(source, "host_request_types_runtime", None) is not None:
            self.host_request_types_runtime = self.host_request_types[:num_seqs]

    def derive(self) -> None:
        md = self.metadata
        self.context_plan = None
        # Recomputed on first use this iteration; see mla_prepare_scheduler_buffers.
        self._invalidate_mla_scheduler_buffers()
        self._steady_block_table_key = None
        self.steady_decode_kv_lens = None
        self._prepared_full_block_offsets = False
        extra_attrs = get_model_extra_attrs()
        # If model extra attrs is set, attention_metadata is setup in executor.
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(md)
        if self.kv_cache_manager is None:
            # Convert the attention metadata to a TRT-LLM no cache attention metadata.
            assert self.kv_cache_manager is None, "no cache attention should not have KV cache manager"
            assert self._max_seq_len_storage is not None, "max_seq_len should be set for no cache attention"

            # setting kv cache params
            self.kv_cache_params = KVCacheParams(use_cache=False, )

            # trtllm attn metadata prepare() requires this
            self.prompt_lens = self.context_lens

            self.kv_cache_block_offsets = None
            self.block_ids_per_seq = None

        prompt_lens = torch.as_tensor(
            self.prompt_lens,
            dtype=torch.int,
            device='cpu',
        )
        self.prompt_lens_cpu[:self.num_seqs].copy_(prompt_lens)
        copy_h2d_pool_shared(self.prompt_lens_cuda[:self.num_seqs],
                             self.prompt_lens_cpu[:self.num_seqs],
                             pack=True)

        # number of tokens in the kv cache for each sequence in the batch
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device='cpu',
        ) if self.kv_cache_params.use_cache else None

        self._prepare_helix_context_slots(cached_token_lens)

        # number of tokens needed in the kv cache for each sequence after the next pass.
        kv_lens = cached_token_lens + \
            self.seq_lens_kv if cached_token_lens is not None else self.seq_lens_kv
        if kv_lens is self.seq_lens or kv_lens is self.seq_lens_kv:
            kv_lens = kv_lens.clone()

        # self.kv_lens is the valid kv cache length, while the self.kv_lens_cuda is
        # the sequence length including the cached tokens and the input tokens.
        self.kv_lens[:self.num_seqs].copy_(
            kv_lens + self.kv_cache_params.num_extra_kv_tokens)
        if self.kv_lens_offsets is not None:
            self.kv_lens_cuda[:self.num_seqs].copy_(maybe_pin_memory(
                kv_lens[:self.num_seqs]),
                                                    non_blocking=True)
            # Corrections known only on device; the host copies keep the
            # scheduled lengths.
            self.kv_lens_cuda[:self.num_seqs] += self.kv_lens_offsets[:self.
                                                                      num_seqs]
        else:
            # Nothing on the device touches the lengths before the forward, so
            # the copy may join the step-input pack.
            step_inputs.copy_(self.kv_lens_cuda[:self.num_seqs],
                              maybe_pin_memory(kv_lens[:self.num_seqs]))
        # total kv lens for context requests and generation requests, without extra tokens
        self.host_total_kv_lens[0] = kv_lens[:self.num_contexts].sum().item()
        self.host_total_kv_lens[1] = kv_lens[self.num_contexts:self.
                                             num_seqs].sum().item()
        self.host_request_types[:self.num_contexts].fill_(0)
        self.host_request_types[self.num_contexts:self.num_seqs].fill_(1)

        # prepare for kv cache reuse/chunked context in MLA
        if self.enable_context_mla_with_cached_kv:
            self.prepare_context_mla_with_cached_kv(cached_token_lens, kv_lens)

        # kv block offsets
        assert self.request_ids is not None
        if self.kv_cache_manager is not None:
            max_kv_len = int(self.kv_lens[:self.num_seqs].max())
            assert max_kv_len <= self.kv_cache_manager.max_seq_len, (
                f"The max KV cache length of input sequences ({max_kv_len}) "
                f"exceeds the KV cache manager's maximum supported length "
                f"({self.kv_cache_manager.max_seq_len}).")

            # On the non-speculative path the host kv_lens snapshot bounds
            # every block-table access, so the staged/H2D width can be capped
            # at the batch's maximum instead of max_seq_len's worth of
            # columns. Speculative decoding must stage the full width:
            # draft/tree sub-steps and the overlap scheduler advance
            # kv_lens_cuda on device past the host snapshot, and their
            # kernels dereference block columns a host-derived cap would
            # leave unstaged (uninitialized in this buffer).
            spec_active = (md.is_spec_decoding_enabled
                           or bool(self.kv_cache_params.num_extra_kv_tokens) or
                           (self.runtime_features is not None and
                            self.runtime_features.has_speculative_draft_tokens)
                           or self.full_block_table_width)
            max_blocks = None
            if not spec_active and self.kv_cache_manager.tokens_per_block:
                max_blocks = ceil_div(max_kv_len,
                                      self.kv_cache_manager.tokens_per_block)
            self.kv_cache_manager.prepare_batch_tables(self.request_ids,
                                                       self.num_contexts)
            self.kv_cache_manager.copy_batch_block_offsets(
                self.kv_cache_block_offsets,
                self.request_ids,
                self.beam_width,
                self.num_contexts,
                self.num_seqs,
                max_blocks=max_blocks)
            self._prepared_full_block_offsets = max_blocks is None

        # Don't pass self.kv_lens as kv_lens here because it includes extra
        # tokens. Use the actual KV length (without extra tokens) for
        # kv_lens_runtime, which becomes host_past_key_value_lengths and
        # eventually mMaxSeqLenKv.
        self.bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:self.num_seqs],
            kv_lens=kv_lens[:self.num_seqs],
            prompt_lens_cuda=self.prompt_lens_cuda[:self.num_seqs],
            prompt_lens_cpu=self.prompt_lens_cpu[:self.num_seqs],
            host_request_types=self.host_request_types[:self.num_seqs],
        )

        if self._has_mixed_precision_staging():
            # Native BF16 paged-context FMHA reads 32-token pages.
            self.prepare_mixed_precision_kv(
                "HND", min(self.kv_cache_manager.tokens_per_block, 32))
            if self.kv_lens_offsets is not None:
                self._sync_mixed_precision_staging_lengths()

    @property
    def effective_beam_width(self) -> int:
        # Only use this for the fallback kernel's beam_width argument.
        # Cross-attention reads request-scoped encoder K/V that is written once
        # and reused unchanged by every decoder beam. Metadata preparation still
        # uses beam_width to expand cross block-offset rows to decoder-sequence
        # scope, but the fallback kernel should treat the cross K/V cache as
        # non-beam-packed.
        return 1 if self.is_cross else self.beam_width

    @property
    def effective_workspace(self) -> Optional[torch.Tensor]:
        """Attention-kernel workspace, switching to the CUDA-graph copy under capture."""
        return self.cuda_graph_workspace if self.is_cuda_graph else self.workspace

    @property
    def max_context_length(self) -> int:
        """
        Upper bound for a single context window.
        Required max_seq_len for context-only attention cases like visual gen
        """
        return min(self.max_seq_len, self.max_num_tokens)

    @property
    def max_seq_len(self) -> int:
        """
        Returns the max sequence length.
        If the attention uses KV cache, it will return max_seq_len from the KV cache manager.
        If the attention is no cache, max_seq_len should be set manually by user.
        """
        if self.kv_cache_manager is not None:
            return self.kv_cache_manager.max_seq_len
        else:
            assert self._max_seq_len_storage is not None, "max_seq_len should be set for no kv cache attention"
            return self._max_seq_len_storage

    # Block tables given to a metadata without a KV cache manager; with one,
    # the core allocates or derives them from the manager.
    _manager_block_tables = frozenset({
        "kv_cache_block_offsets",
        "host_kv_cache_block_offsets",
        "block_ids_per_seq",
        "kv_block_ids_per_seq",
    })

    def creation_value(self, name: str) -> Any:
        # The value set for no-cache attention; a cache's comes from its manager.
        if name == "max_seq_len":
            return self._max_seq_len_storage
        if name in self._manager_block_tables and self.kv_cache_manager is not None:
            return None
        return super().creation_value(name)

    @max_seq_len.setter
    def max_seq_len(self, value: int) -> None:
        """
        Set the max sequence length for no cache attention.
        """
        self._max_seq_len_storage = value

    @property
    def tokens_per_block(self) -> Optional[int]:
        """
        Returns the number of tokens per block from the KV cache manager.
        """
        pools = self._extension_kv_cache_pools()
        if pools is not None:
            return pools[2]
        return self.kv_cache_manager.tokens_per_block if self.kv_cache_manager is not None else None

    @property
    def page_size(self) -> int:
        """
        Number of tokens per cache page.
        """
        page_size_override = getattr(self, '_page_size_override', None)
        if page_size_override is not None:
            return page_size_override
        assert self.kv_cache_manager is not None, "page_size requires a KV cache manager"
        return self.kv_cache_manager.tokens_per_block

    @page_size.setter
    def page_size(self, value: int) -> None:
        self._page_size_override = value

    @property
    def host_kv_cache_pool_pointers(self) -> Optional[torch.Tensor]:
        """
        Returns the host KV cache pool pointers from the KV cache manager if KV cache manager is not None.
        """
        pools = self._extension_kv_cache_pools()
        if pools is not None:
            return pools[0]
        return self.kv_cache_manager.kv_cache_pool_pointers if self.kv_cache_manager is not None else None

    @property
    def host_kv_cache_pool_mapping(self) -> Optional[torch.Tensor]:
        """
        Returns the host KV cache pool mapping from the KV cache manager if KV cache manager is not None.
        """
        pools = self._extension_kv_cache_pools()
        if pools is not None:
            return pools[1]
        return self.kv_cache_manager.kv_cache_pool_mapping if self.kv_cache_manager is not None else None

    def _extension_kv_cache_pools(
            self) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        # An extension's pools (CSA2 compute tiles) replace the manager's.
        return self.metadata.find("kv_cache_pools", None)

    def copy_lengths_from(self, source: "AttentionCoreMetadata") -> None:
        super().copy_lengths_from(source)
        num_seqs = self.num_seqs
        self.kv_lens_cuda[:num_seqs].copy_(source.kv_lens_cuda[:num_seqs])
        if self.host_request_types is not None:
            self.host_request_types[:num_seqs].copy_(
                source.host_request_types[:num_seqs])

    def resync_length_derived_state(self) -> None:
        # A draft step changed the lengths the layer-invariant prefixes derive
        # from; the next MLA layer rebuilds them, and the next context-only
        # call plans again.
        self.context_plan = None
        self._invalidate_mla_scheduler_buffers()
        self._sync_mixed_precision_staging_lengths()

    def _sync_mixed_precision_staging_lengths(self) -> None:
        """Keep the mixed-precision staging lengths in step with kv_lens_cuda.

        The staging copy loads the host lengths in ``prepare_mixed_precision_kv``;
        the step's device corrections (``kv_lens_offsets``) and a draft loop's
        length changes reach only kv_lens_cuda, so the staging lengths are
        taken from it on the device.
        """
        view = self.mixed_precision_staging
        if (view is None or view.lengths is None
                or view.manager is not self.kv_cache_manager):
            return
        lengths = view.lengths
        n = lengths.shape[1]
        lengths[1].copy_(self.kv_lens_cuda[:n])
        torch.sub(lengths[1], self.seq_lens_kv_cuda[:n], out=lengths[0])

    def _invalidate_mla_scheduler_buffers(self) -> None:
        # Spec-dec rewrites q_lens and kv_lens between sub-steps, so the cumulative
        # buffers below must be rebuilt before the next MLA layer reads them.
        self._mla_scheduler_buffers_valid = False
        self._mla_ctx_cu_seqlens_valid = False
        self._cute_dsl_mla_split_plan_key = None

    def _prepare_helix_context_slots(
            self, cached_token_lens: Optional[torch.Tensor]) -> None:
        """Map global context positions to the cache manager's Helix pages."""
        self.helix_context_local_slots = None
        cache_mapping = getattr(self.kv_cache_manager, "mapping", None)
        if (not self.enable_context_mla_with_cached_kv or self.num_contexts == 0
                or cached_token_lens is None or cache_mapping is None
                or not cache_mapping.has_cp_helix()):
            return
        # Context batches run eagerly. Build once per batch on the host,
        # then share the device view across all MLA layers.
        lengths = self.seq_lens[:self.num_contexts].to(torch.int64)
        starts = torch.cumsum(lengths, dim=0) - lengths
        positions = torch.arange(self.num_ctx_tokens,
                                 dtype=torch.int64,
                                 device="cpu")
        positions += torch.repeat_interleave(
            cached_token_lens[:self.num_contexts] - starts, lengths)
        page = self.kv_cache_manager.tokens_per_block
        owned = (positions //
                 page) % cache_mapping.cp_size == cache_mapping.cp_rank
        slots = helix_local_len_tensor(positions, page, cache_mapping.cp_size,
                                       cache_mapping.cp_rank)
        slots.masked_fill_(~owned, -1)
        self.helix_context_local_slots = maybe_pin_memory(slots.to(
            torch.int32)).to(self.seq_lens_cuda.device, non_blocking=True)

    def bind_runtime_views(
        self,
        *,
        kv_lens_cuda: torch.Tensor,
        kv_lens: torch.Tensor,
        prompt_lens_cuda: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
        host_request_types: torch.Tensor,
    ) -> None:
        """Bind the per-forward ``*_runtime`` views the FMHA kernels read.

        Shared by ``prepare``, ``prepare_encoder_only`` and the encoder CUDA
        graph binding so the set of runtime views stays in sync across paths.
        """
        self.context_plan = None
        self.kv_lens_cuda_runtime = kv_lens_cuda
        self.kv_lens_runtime: torch.Tensor = kv_lens
        self.prompt_lens_cuda_runtime: torch.Tensor = prompt_lens_cuda
        self.prompt_lens_cpu_runtime: torch.Tensor = prompt_lens_cpu
        self.host_request_types_runtime = host_request_types

    def substitute_runtime_views(
        self,
        *,
        kv_lens_cuda: Optional[torch.Tensor] = None,
        kv_lens: Optional[torch.Tensor] = None,
        prompt_lens_cuda: Optional[torch.Tensor] = None,
        prompt_lens_cpu: Optional[torch.Tensor] = None,
        context_total_kv_len: Optional[int] = None,
    ) -> None:
        """Present other tensors through the given ``*_runtime`` views.

        For callers whose kernels read different lengths through the same
        views (FP4 MLA reads the per-forward append lengths; chunked MLA
        context attention reads one chunk's). ``context_total_kv_len`` is the
        context KV total the op reads with them. Views not given stay bound;
        nothing is announced, so it may run inside an event.
        """
        if kv_lens_cuda is not None:
            self.kv_lens_cuda_runtime = kv_lens_cuda
        if kv_lens is not None:
            self.kv_lens_runtime = kv_lens
        if context_total_kv_len is not None:
            self.host_total_kv_lens[0] = context_total_kv_len
        if prompt_lens_cuda is not None:
            self.prompt_lens_cuda_runtime = prompt_lens_cuda
        if prompt_lens_cpu is not None:
            self.prompt_lens_cpu_runtime = prompt_lens_cpu

    def republish_kv_lens(self, kv_lens: torch.Tensor) -> None:
        """Publish corrected per-sequence KV lengths after ``derive``.

        ``kv_lens`` [num_seqs] (host, without the extra KV tokens) replaces the
        derived lengths: the host and device copies, the context/generation
        totals and the runtime views. For an extension whose kernels attend to
        other lengths than the batch's (Helix's rank-local generation rows),
        called from its ``prepare``.
        """
        num_seqs = self.num_seqs
        kv_lens = kv_lens[:num_seqs]
        self.kv_lens[:num_seqs].copy_(kv_lens +
                                      self.kv_cache_params.num_extra_kv_tokens)
        self.kv_lens_cuda[:num_seqs].copy_(maybe_pin_memory(kv_lens),
                                           non_blocking=True)
        self.host_total_kv_lens[0] = kv_lens[:self.num_contexts].sum().item()
        self.host_total_kv_lens[1] = kv_lens[self.num_contexts:].sum().item()
        self.bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:num_seqs],
            kv_lens=kv_lens,
            prompt_lens_cuda=self.prompt_lens_cuda_runtime,
            prompt_lens_cpu=self.prompt_lens_cpu_runtime,
            host_request_types=self.host_request_types_runtime,
        )

    def override_cached_token_lens(
            self, num_cached_tokens_per_seq: List[int]) -> None:
        super().override_cached_token_lens(num_cached_tokens_per_seq)
        if not self.kv_cache_params.use_cache:
            return
        cached = torch.tensor(num_cached_tokens_per_seq[:self.num_seqs],
                              dtype=torch.int,
                              device='cpu')
        self.republish_kv_lens(cached + self.seq_lens_kv[:self.num_seqs])

    def override_prompt_lens(self, prompt_lens: List[int]) -> None:
        """Present ``prompt_lens`` to the kernels in place of the batch's."""
        num_seqs = self.num_seqs
        self.prompt_lens_cpu[:num_seqs].copy_(
            torch.tensor(prompt_lens, dtype=torch.int, device="cpu"))
        self.prompt_lens_cuda[:num_seqs].copy_(self.prompt_lens_cpu[:num_seqs],
                                               non_blocking=True)
        self.substitute_runtime_views(
            prompt_lens_cuda=self.prompt_lens_cuda[:num_seqs],
            prompt_lens_cpu=self.prompt_lens_cpu[:num_seqs])

    def context_scratch_view(
        self,
        *,
        kv_cache_manager: Any,
        kv_cache_block_offsets: torch.Tensor,
        block_ids_per_seq: Optional[torch.Tensor],
        host_total_kv_lens: torch.Tensor,
        kv_lens_cuda: torch.Tensor,
        kv_lens: torch.Tensor,
        position_offsets: torch.Tensor,
        without: Tuple[Type[MetadataExtension], ...] = (),
    ) -> "TrtllmAttentionMetadata":
        """A view of this metadata over a disposable cache of the context requests.

        For kernels that write and read the context K/V in a scratch cache
        (FP4 MLA's FP8 context path): the view addresses ``kv_cache_manager``
        through ``kv_cache_block_offsets``, presents only the context rows
        with ``kv_lens_cuda`` / ``kv_lens`` as their KV lengths and gives the
        tokens' absolute positions as Helix position offsets. The extensions
        in ``without`` are left out. The view shares every buffer with this
        metadata and is valid for the current step.
        """
        view = copy.copy(self.metadata)
        for extension_type in without:
            view.remove_ext(extension_type)
        core = view.core()
        num_contexts = core.num_contexts
        core.kv_cache_manager = kv_cache_manager
        core.kv_cache_block_offsets = kv_cache_block_offsets
        core.block_ids_per_seq = block_ids_per_seq
        core.prompt_lens_cuda_runtime = core.prompt_lens_cuda_runtime[:
                                                                      num_contexts]
        core.prompt_lens_cpu_runtime = core.prompt_lens_cpu_runtime[:
                                                                    num_contexts]
        core.host_request_types_runtime = core.host_request_types_runtime[:
                                                                          num_contexts]
        core.kv_lens_cuda_runtime = kv_lens_cuda
        core.kv_lens_runtime = kv_lens
        core.host_total_kv_lens = host_total_kv_lens
        helix = view.find_ext(HelixMetadata)
        if helix is None:
            # Without Helix the view still presents the positions through
            # Helix state; it is built once and its buffers stay unused.
            if self._context_scratch_helix is None:
                self._context_scratch_helix = HelixMetadata(view)
            helix = self._context_scratch_helix
        helix = copy.copy(helix)
        helix.metadata = view
        helix.helix_position_offsets = position_offsets
        view.replace_ext(helix)
        return view

    def override_page_size(self, page_size: int) -> None:
        """Present ``page_size`` to the kernels in place of the manager's."""
        self.page_size = page_size

    def set_num_sparse_topk(self, num_sparse_topk: int) -> None:
        """Set the KV count each sparse attention query attends to."""
        self.num_sparse_topk = num_sparse_topk

    def set_use_paged_context_fmha(self, enabled: bool) -> None:
        """Choose whether context attention reads K/V from the paged cache."""
        self.use_paged_context_fmha = enabled

    def disable_workspace_reclamation(self) -> None:
        """Keep the workspace across forwards (graphs or staged FMHA state)."""
        self.workspace_reclaimable = False

    def use_block_id_tables(self, block_ids_per_seq: torch.Tensor,
                            kv_block_ids_per_seq: torch.Tensor) -> None:
        """Bind the per-sequence block-ID tables FlashMLA kernels read."""
        self.block_ids_per_seq = block_ids_per_seq
        self.kv_block_ids_per_seq = kv_block_ids_per_seq

    def as_generation_rows(self) -> None:
        """Continue every context row as a generation row; see the base method.

        The attention op reads the rows' request types from
        ``host_request_types``, and generation rows have no cached context KV,
        so both are reset with the context count.
        """
        # Allocated with the core's buffers; a core built without them has none.
        host_request_types = getattr(self, "host_request_types", None)
        if host_request_types is not None:
            host_request_types[:self.num_contexts].fill_(1)
        super().as_generation_rows()
        self.clear_context_cached_kv()

    def clear_context_cached_kv(self) -> None:
        """Mark the batch as having no cached context KV (generation sub-steps)."""
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0

    def stage_context_cached_kv(self, cached_token_lens: torch.Tensor,
                                kv_lens: torch.Tensor) -> None:
        """Publish the context requests' cached-token and KV-length prefixes.

        Fills ``ctx_cached_token_indptr`` / ``ctx_kv_indptr`` (see
        ``allocate_context_kv_indptrs``) and the context maxima from the
        per-sequence host lengths.
        """
        num_contexts = self.num_contexts
        if num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:num_contexts].sum(
            ).item()
            self.max_ctx_kv_len = kv_lens[:num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:num_contexts].max().item()
            torch.cumsum(
                cached_token_lens[:num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1:num_contexts + 1],
            )
            self.ctx_cached_token_indptr[:num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[:num_contexts + 1],
                non_blocking=True)
            torch.cumsum(
                kv_lens[:num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_kv_indptr[1:num_contexts + 1],
            )
            self.ctx_kv_indptr[:num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[:num_contexts + 1], non_blocking=True)
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

    def bind_fixed_kv_window(self, kv_len: int,
                             kv_cache_block_offsets: torch.Tensor) -> None:
        """Address a fixed staging window instead of the paged cache.

        Every sequence is a generation request attending to ``kv_len`` keys
        through ``kv_cache_block_offsets`` (CSA2 query tiles); the workspace
        is the one graph captures use.
        """
        self.num_sparse_topk = kv_len
        self.max_seq_len = kv_len
        self.kv_cache_params = KVCacheParams(use_cache=True)
        self.kv_cache_block_offsets = kv_cache_block_offsets
        self.kv_lens.fill_(kv_len)
        self.kv_lens_cuda.fill_(kv_len)
        self.prompt_lens_cpu.fill_(1)
        self.prompt_lens_cuda.fill_(1)
        self.host_request_types.fill_(1)
        self.host_total_kv_lens.zero_()
        # Each query count has independent native workspace storage.
        # Eager initialization and capture must use that same tensor.
        self.cuda_graph_workspace = self.workspace

    def use_block_offsets(self, kv_cache_block_offsets: torch.Tensor) -> None:
        """Address the paged KV cache through ``kv_cache_block_offsets`` after ``derive``.

        For a KV cache whose page lists the attention op reads from the
        cache's own tensor (VisualGen's block-causal cache) instead of the
        block offsets ``derive`` copied.
        """
        self.kv_cache_block_offsets = kv_cache_block_offsets

    def use_workspace(self, workspace: torch.Tensor) -> None:
        """Run on ``workspace`` both eagerly and in graph capture."""
        self.workspace = self.cuda_graph_workspace = workspace

    def reallocate_workspace(self, num_bytes: int) -> None:
        """Replace the eager workspace with a new one of ``num_bytes`` on its device.

        The old storage is released before the new one is allocated:
        shrinking it in place would retain it.
        """
        device = self.workspace.device
        self.workspace = None
        self.workspace = torch.empty(num_bytes, dtype=torch.int8, device=device)

    def mark_trtllm_gen_jit_warmup(self, enabled: bool) -> None:
        """Mark whether the next forwards are the TRTLLM-Gen FMHA JIT warmup."""
        self.trtllm_gen_jit_warmup = enabled

    def bind_generation_tile(self, query_lens_host: torch.Tensor,
                             query_lens_device: torch.Tensor,
                             q_offsets: torch.Tensor,
                             kv_offsets: torch.Tensor) -> None:
        """Bind a fixed-window generation batch over the caller's buffers.

        One single-token request per query row (see ``bind_fixed_kv_window``).
        """
        count = query_lens_host.numel()
        self.bind_lengths(query_lens_host,
                          query_lens_device,
                          num_contexts=0,
                          num_ctx_tokens=0,
                          num_generations=count,
                          num_tokens=count)
        self.bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda,
            kv_lens=self.kv_lens,
            prompt_lens_cuda=self.prompt_lens_cuda,
            prompt_lens_cpu=self.prompt_lens_cpu,
            host_request_types=self.host_request_types,
        )
        self.host_total_kv_lens[1] = count * self.num_sparse_topk
        self.cu_q_seqlens = q_offsets
        self.cu_kv_seqlens = kv_offsets

    def bind_context_tile(
        self,
        query_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        *,
        query_lens_host: torch.Tensor,
        query_lens_device: torch.Tensor,
        q_offsets: torch.Tensor,
        kv_offsets: torch.Tensor,
        copy_to_device: Callable[[str, torch.Tensor, torch.Tensor], None],
    ) -> None:
        """Bind a fixed-window context batch over the caller's buffers.

        Request ``i`` has ``query_lens[i]`` query tokens attending to
        ``kv_lens[i]`` keys (host int32 tensors); ``query_lens_host`` /
        ``query_lens_device`` already hold the query lengths.
        ``copy_to_device(key, destination, source)`` stages host data.
        """
        requests = query_lens.numel()
        num_tokens = int(query_lens.sum())
        self.bind_lengths(query_lens_host,
                          query_lens_device,
                          num_contexts=requests,
                          num_ctx_tokens=num_tokens,
                          num_generations=0,
                          num_tokens=num_tokens)
        self.kv_lens[:requests].copy_(kv_lens)
        copy_to_device("context_kv_lens", self.kv_lens_cuda[:requests], kv_lens)
        # THOP interprets context_lengths as Q lengths, independently of
        # past/current KV lengths. Keep them consistent with unfolded cuQ.
        self.prompt_lens_cpu[:requests].copy_(query_lens)
        copy_to_device("context_prompt_lens", self.prompt_lens_cuda[:requests],
                       query_lens)
        self.host_request_types[:requests].zero_()
        self.host_total_kv_lens.zero_()
        self.host_total_kv_lens[0] = int(kv_lens.sum())
        self.max_seq_len = max(self.num_sparse_topk, int(kv_lens.max()))
        self.cu_q_seqlens = q_offsets
        self.cu_kv_seqlens = kv_offsets
        self.bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:requests],
            kv_lens=self.kv_lens[:requests],
            prompt_lens_cuda=self.prompt_lens_cuda[:requests],
            prompt_lens_cpu=self.prompt_lens_cpu[:requests],
            host_request_types=self.host_request_types[:requests],
        )

    def cute_dsl_mla_split_plan(
        self,
        batch_size: int,
        device: torch.device,
        key: tuple,
        launch: Callable[[torch.Tensor], None],
    ) -> Optional[torch.Tensor]:
        """This step's split-KV plan of the CuTe DSL MLA decode, ``[batch_size]``.

        ``launch(plan)`` fills the plan buffer; it runs once per step, again
        only when ``key`` or the buffer changes, or after the lengths changed.
        None when the buffer would have to be allocated during graph capture.
        """
        plan = self._cute_dsl_mla_split_plan
        if plan is None or plan.numel() < batch_size or plan.device != device:
            if torch.cuda.is_current_stream_capturing():
                return None
            capacity = max(int(self.max_num_requests or 0), batch_size)
            plan = torch.empty(capacity, dtype=torch.int32, device=device)
            self._cute_dsl_mla_split_plan = plan
        full_key = (*key, plan.data_ptr())
        if self._cute_dsl_mla_split_plan_key != full_key:
            launch(plan)
            self._cute_dsl_mla_split_plan_key = full_key
        return plan[:batch_size]

    def mla_prepare_scheduler_buffers(
            self, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative Q rows / KV lengths for the MLA generation FMHA kernel.

        These are layer-invariant, so they are computed once per iteration instead of
        once per layer inside `applyMLARopeAndAssignQKVKernelGeneration`. The values
        match what that kernel produced:
          seqQOffset[b+1]  = num_heads * cumsum(q_lens)[b]   (Q ROWS, not tokens)
          seqKVOffsets[b+1] = cumsum(kv_lens)[b]

        Derived on device from `seq_lens_cuda` / `kv_lens_cuda` -- the same tensors the
        C++ op reads as `cache_seq_lens` -- so spec-dec sub-steps, which rewrite those
        in place after `prepare()`, are picked up once the buffers are invalidated.
        Reading the host mirrors instead would silently miss those edits, since MTP only
        touches the device copy. Results land in fixed-address buffers so a captured
        CUDA graph keeps reading the same pointers, and no D2H sync is introduced.
        """
        md = self.metadata
        extension_buffers = md.find("mla_scheduler_buffers", None)
        buffers = extension_buffers(num_heads) if extension_buffers else None
        if buffers is not None:
            return buffers
        num_ctx = self.num_contexts
        n_gen = self.num_seqs - num_ctx
        if not self._mla_scheduler_buffers_valid:
            if n_gen > 0:
                torch.cumsum(self.seq_lens_cuda[num_ctx:self.num_seqs],
                             0,
                             dtype=torch.int32,
                             out=self.mla_cu_q_rows[1:n_gen + 1])
                self.mla_cu_q_rows[1:n_gen + 1] *= num_heads
                # `kv_lens_cuda` is the pre-`num_extra_kv_tokens` length, which is what
                # the kernel's `cache_seq_lens` scan used. `self.kv_lens` is not.
                torch.cumsum(self.kv_lens_cuda[num_ctx:self.num_seqs],
                             0,
                             dtype=torch.int32,
                             out=self.mla_cu_kv_seqlens[1:n_gen + 1])
            self._mla_scheduler_buffers_valid = True
        return (self.mla_cu_q_rows[:n_gen + 1],
                self.mla_cu_kv_seqlens[:n_gen + 1])

    def mla_prepare_ctx_cu_seqlens(self) -> Optional[torch.Tensor]:
        """Token-wise cumulative Q seqlens over the context sequences.

        Layer-invariant, so it is built once per iteration into a fixed-address
        buffer, from `seq_lens_cuda` for the same reason as the generation buffers.
        Returns None when the batch has no context sequences.
        """
        num_ctx = self.num_contexts
        if num_ctx <= 0:
            return None
        if not self._mla_ctx_cu_seqlens_valid:
            torch.cumsum(self.seq_lens_cuda[:num_ctx],
                         0,
                         dtype=torch.int32,
                         out=self.mla_ctx_cu_q_seqlens[1:num_ctx + 1])
            self._mla_ctx_cu_seqlens_valid = True
        return self.mla_ctx_cu_q_seqlens[:num_ctx + 1]

    def token_major_view(self) -> Optional["TokenMajorGenView"]:
        """The one-row-per-query-token generation view an extension provides, or None."""
        view = self.metadata.find("token_major_gen_view", None)
        return view() if view is not None else None

    @contextlib.contextmanager
    def presented_token_major(
            self, view: Optional["TokenMajorGenView"]) -> Iterator[None]:
        """Temporarily present the generation half as one row per query token.

        The MLA rope and sparse-MLA generation ops learn their sequence count
        from ``host_context_lengths.size(0)`` and then require the query tokens
        to divide evenly across it. A ragged batch does not divide. Swapping in
        per-token views for the duration of those two calls makes the division
        trivial (seq_len == 1) and satisfies their asserts rather than weakening
        them.

        Scoped deliberately: ``num_seqs`` is read as a *request* count almost
        everywhere else -- spec workers sizing acceptance rectangles, the
        KV-length correction under the overlap scheduler, the indexer's own
        metadata -- so the token-major rows must never outlive these two calls.
        """
        if view is None:
            yield
            return
        saved = (self.kv_lens_cuda_runtime, self.kv_lens_runtime,
                 self.prompt_lens_cpu_runtime, self.kv_cache_block_offsets,
                 self.max_num_requests, self.prompt_lens_cuda_runtime,
                 self.host_request_types_runtime, self._num_generations,
                 self.max_num_sequences)
        try:
            self.kv_lens_cuda_runtime = view.sequence_length
            self.kv_lens_runtime = view.host_past_key_value_lengths
            self.prompt_lens_cpu_runtime = view.host_context_lengths
            self.kv_cache_block_offsets = view.kv_cache_block_offsets
            # The multi-CTA-KV counter and generation workspace must cover
            # token rows, which can outnumber requests. Use the static row
            # ceiling so the allocation remains stable across steps.
            self.max_num_requests = view.max_num_rows
            self.max_num_sequences = view.max_num_rows
            # Phased FMHA reads this count directly. Its public setter calls
            # on_update(), which would recompute the request count from seq_lens.
            self._num_generations = view.num_rows - self.num_contexts
            # Every runtime view has to agree on batch_size with
            # kv_lens_cuda_runtime, and the op reads request_types over all of
            # num_seqs -- both are the row count here, not the request count.
            self.prompt_lens_cuda_runtime = view.prompt_lens_cuda
            self.host_request_types_runtime = view.host_request_types
            yield
        finally:
            (self.kv_lens_cuda_runtime, self.kv_lens_runtime,
             self.prompt_lens_cpu_runtime, self.kv_cache_block_offsets,
             self.max_num_requests, self.prompt_lens_cuda_runtime,
             self.host_request_types_runtime, self._num_generations,
             self.max_num_sequences) = saved

    def _on_mixed_precision_kv_prepared(self, view) -> None:
        """Lay out the staging pages as TRTLLM block offsets for this batch."""
        batch = len(view.page_counts)
        # Filled with numpy on a pinned host table: per-step torch CPU ops
        # each open an OpenMP parallel region, which stalls on a busy host.
        host = torch.full((1, batch, 2, view.max_blocks_per_seq),
                          -1,
                          dtype=torch.int32,
                          pin_memory=prefer_pinned())
        table = host.numpy()
        for row, (start,
                  count) in enumerate(zip(view.page_starts, view.page_counts)):
            pages = np.arange(start, start + count,
                              dtype=np.int32) * view.kv_factor
            table[0, row, 0, :count] = pages
            table[0, row, 1, :count] = pages + view.kv_factor - 1
        block_offsets = view.buffers.get_buffer(
            (1, view.rows, 2, view.max_blocks_per_seq),
            torch.int32,
            f"{view.name}_block_offsets",
            self.is_cuda_graph,
        )[:, :batch]
        block_offsets.copy_(host, non_blocking=True)
        # Swapped in for kv_cache_block_offsets while a mixed-precision layer runs.
        view.block_offsets = block_offsets

    def _swap_mixed_precision_kv(self, view) -> tuple:
        """Also point the block offsets at the staging pages."""
        saved = (*super()._swap_mixed_precision_kv(view),
                 self.kv_cache_block_offsets)
        self.kv_cache_block_offsets = view.block_offsets
        return saved

    def _restore_mixed_precision_kv(self, saved: tuple) -> None:
        """Undo _swap_mixed_precision_kv, block offsets included."""
        super()._restore_mixed_precision_kv(saved[:2])
        self.kv_cache_block_offsets = saved[2]

    def prepare_encoder_decoder_from_precomputed_lengths(
            self, inputs: BatchInputs, *, num_ctx_tokens: int,
            num_generations: int, num_tokens: int, kv_lens: torch.Tensor,
            context_kv_tokens: int, generation_kv_tokens: int,
            max_kv_len: int) -> None:
        """Prepare encoder-decoder attention from precomputed lengths.

        ``inputs.seq_lens`` are bound as given with the precomputed token and
        request counts instead of reductions over them; ``inputs.prompt_lens``
        is the host prompt-length tensor.
        """
        md = self.metadata
        self.apply_batch_inputs(dataclasses.replace(inputs, seq_lens=None))
        self.bind_lengths(inputs.seq_lens,
                          num_contexts=inputs.num_contexts,
                          num_ctx_tokens=num_ctx_tokens,
                          num_generations=num_generations,
                          num_tokens=num_tokens)
        prompt_lens = inputs.prompt_lens
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self.metadata)

        assert self.kv_cache_manager is not None
        assert md.find("draft_or_none", None) is None
        assert not md.is_spec_decoding_enabled
        assert self.kv_cache_params.num_extra_kv_tokens == 0
        assert md.find_ext(FlashMlaMetadata) is None
        assert md.find_ext(HelixMetadata) is None
        assert not self.enable_context_mla_with_cached_kv
        assert self.request_ids is not None
        assert max_kv_len <= self.kv_cache_manager.max_seq_len, (
            f"The max KV cache length of input sequences ({max_kv_len}) "
            "exceeds the KV cache manager's maximum supported length "
            f"({self.kv_cache_manager.max_seq_len}).")

        num_seqs = self.num_seqs
        copy_h2d_pool_shared(self.prompt_lens_cuda[:num_seqs], prompt_lens)
        self.kv_lens_cuda[:num_seqs].copy_(kv_lens, non_blocking=True)
        self.host_total_kv_lens[0] = context_kv_tokens
        self.host_total_kv_lens[1] = generation_kv_tokens
        self.host_request_types[:self.num_contexts].fill_(0)
        self.host_request_types[self.num_contexts:num_seqs].fill_(1)

        max_blocks = None
        if self.kv_cache_manager.tokens_per_block:
            max_blocks = ceil_div(max_kv_len,
                                  self.kv_cache_manager.tokens_per_block)
        self.kv_cache_manager.copy_batch_block_offsets(
            self.kv_cache_block_offsets,
            self.request_ids,
            self.beam_width,
            self.num_contexts,
            num_seqs,
            max_blocks=max_blocks)
        self.bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:num_seqs],
            kv_lens=kv_lens,
            prompt_lens_cuda=self.prompt_lens_cuda[:num_seqs],
            prompt_lens_cpu=prompt_lens,
            host_request_types=self.host_request_types[:num_seqs],
        )

    @property
    def supports_steady_decode_step(self) -> bool:
        """Whether prepare_steady_decode_step may stand in for this metadata's prepare.

        Only when prepare derives nothing beyond the KV lengths and block
        offsets: no extension but spec-dec mode and the one-model draft (no
        FlashMLA, Helix, FP4 MLA, sparse, recurrent or cross-attention state),
        no cached-KV context MLA, mixed-precision KV or full generation page
        table, and a draft metadata that supports it too (its own steady step
        follows the target's).
        """
        md = self.metadata
        for extension in md._attached_extensions():
            if extension is not self and not isinstance(
                    extension, (SpecDecodingMetadata, DraftMetadata)):
                return False
        draft = md.find("draft_or_none", None)
        if draft is not None and not draft.supports_steady_decode_step:
            return False
        # A draft metadata takes its KV-cache parameters from the target when
        # it first refreshes; until then the target's decide.
        kv_cache_params = self.kv_cache_params
        return (self.kv_cache_manager is not None
                and not self.enable_context_mla_with_cached_kv
                and not self._has_mixed_precision_staging()
                and (kv_cache_params is None
                     or not kv_cache_params.use_full_generation_page_table))

    def key_prepared_block_offsets(self) -> None:
        """Key the block offsets the last prepare copied, so that the next
        prepare_steady_decode_step skips its copy while they are unchanged.

        A no-op unless that prepare copied the full table width.
        """
        if not self._prepared_full_block_offsets:
            return
        self._steady_block_table_key = self.kv_cache_manager.batch_block_table_key(
            self.request_ids)

    def prepare_steady_decode_step(self, inputs: BatchInputs,
                                   kv_lens: List[int]) -> None:
        """The prepare of a generation-only step that runs the batch of the previous prepare one step later.

        Requires supports_steady_decode_step and the same requests, prompt
        lengths, request types and beam width as that prepare. ``inputs`` is
        the step's batch description and ``kv_lens`` its scheduled KV lengths
        (without the extra speculative tokens). kv_lens_cuda[:num_seqs]
        already holds the scheduled lengths on the device, written by the
        engine's staging kernel; ``inputs.kv_lens_offsets`` (the overlap
        scheduler's corrections) is added to them here, as prepare does.
        Sets the host state that follows from ``kv_lens``, copies the block
        offsets of a KV cache manager only when its batch_block_table_key
        changed (the full table width, as speculative decoding's prepare
        does), then prepares the other extensions.
        """
        md = self.metadata
        self.apply_batch_inputs(inputs)
        self.steady_decode_kv_lens = list(kv_lens)
        self.context_plan = None
        self._invalidate_mla_scheduler_buffers()
        if get_model_extra_attrs() is None:
            get_global_attrs().attention_metadata = weakref.ref(md)
        num_seqs = self.num_seqs
        assert len(kv_lens) == num_seqs and self.num_contexts == 0
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        host_kv_lens = torch.tensor(kv_lens, dtype=torch.int)
        self.kv_lens[:num_seqs].copy_(host_kv_lens + num_extra_kv_tokens)
        if self.kv_lens_offsets is not None:
            # Corrections known only on device; the host copies keep the
            # scheduled lengths.
            self.kv_lens_cuda[:num_seqs] += self.kv_lens_offsets[:num_seqs]
        self.host_total_kv_lens[0] = 0
        self.host_total_kv_lens[1] = sum(kv_lens)
        max_kv_len = max(kv_lens) + num_extra_kv_tokens
        assert max_kv_len <= self.kv_cache_manager.max_seq_len, (
            f"The max KV cache length of input sequences ({max_kv_len}) "
            f"exceeds the KV cache manager's maximum supported length "
            f"({self.kv_cache_manager.max_seq_len}).")
        key = self.kv_cache_manager.batch_block_table_key(self.request_ids)
        if key is None or key != self._steady_block_table_key:
            self.kv_cache_manager.copy_batch_block_offsets(
                self.kv_cache_block_offsets,
                self.request_ids,
                self.beam_width,
                self.num_contexts,
                num_seqs,
                max_blocks=None)
        # A full prepare cleared the key, so a draft's first steady step after
        # it copies the full table and records its own manager's key here.
        self._steady_block_table_key = key
        self.bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:num_seqs],
            kv_lens=host_kv_lens,
            prompt_lens_cuda=self.prompt_lens_cuda[:num_seqs],
            prompt_lens_cpu=self.prompt_lens_cpu[:num_seqs],
            host_request_types=self.host_request_types[:num_seqs],
        )
        for extension in md._attached_extensions():
            if extension is not self:
                extension.prepare()
        self.clear_lengths_changed()

    def aligned_kv_lens_staging(self, num_lengths: int,
                                device: torch.device) -> Optional[torch.Tensor]:
        """An int32 buffer of at least ``num_lengths`` entries kept for this metadata.

        Kernels that need aligned length pointers copy a misaligned view (the
        generation rows of a mixed batch) into it, so every layer of a step
        passes the same pointer. Built on first use, at the request capacity;
        None while a CUDA graph is capturing and no such buffer exists yet.
        """
        staging = self._aligned_kv_lens_staging
        if (staging is None or staging.numel() < num_lengths
                or staging.device != torch.device(device)):
            if torch.cuda.is_current_stream_capturing():
                return None
            capacity = max(int(self.max_num_requests or 0), num_lengths)
            staging = torch.empty(capacity, dtype=torch.int32, device=device)
            self._aligned_kv_lens_staging = staging
        return staging

    def cache_context_plan(self, plan: Optional[_ContextPlan]) -> None:
        """Keep ``plan`` as the context-only plan the layers of this forward share.

        The plan is dropped whenever the lengths or the batch can change
        (prepare, the runtime views, a draft refresh).
        """
        self.context_plan = plan

    def prepare_encoder_only(self) -> None:
        """Fast path for encoder-only forward (eager + CUDA graph capture)."""
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self.metadata)

        # Encoder batches run without a KV cache. The block-offset / block-id
        # attributes default to None (declared as dataclass fields).
        self.kv_cache_params = KVCacheParams(use_cache=False)
        self._invalidate_mla_scheduler_buffers()

        # For encoder batches every request is a context request, so total
        # kv-tokens equals total q-tokens.
        self.host_total_kv_lens[0] = self._num_tokens

        # `host_request_types` is allocated with `torch.empty_like` (not
        # zeroed), and the full `prepare()` is what normally fills it. This
        # fast path skips `prepare()`, so explicitly mark every request as a
        # context request (type 0) here -- otherwise the FMHA reads whatever
        # garbage the buffer happened to hold and may treat a context segment
        # as a generation step.
        n = self.num_seqs
        self.host_request_types[:n].fill_(0)

        # Graph metadata binds these views once per key; eager refreshes them
        # because batch shape can vary between calls.
        if not self.is_cuda_graph:
            self.bind_runtime_views(
                kv_lens_cuda=self._seq_lens_cuda[:n],
                kv_lens=self._seq_lens[:n],
                prompt_lens_cuda=self._seq_lens_cuda[:n],
                prompt_lens_cpu=self._seq_lens[:n],
                host_request_types=self.host_request_types[:n],
            )

    def bind_encoder_cuda_graph_seq_lens(
            self,
            seq_lens_host: torch.Tensor,
            padded_batch_size: int,
            *,
            max_q_len: int,
            offsets: Optional[torch.Tensor] = None) -> None:
        """Bind stable seq_lens storage for one encoder CUDA graph key.

        ``max_q_len`` fixes the FMHA launch parameters for every replay of the
        key. ``offsets`` (device, int32, ``[padded_batch_size + 1]``) are the
        fixed query and KV token boundaries of the graph's sequence slots;
        None keeps ``cumsum(seq_lens)``.
        """
        self.max_context_q_len_override = max_q_len
        self._seq_lens = seq_lens_host[:padded_batch_size]
        self._num_contexts = padded_batch_size
        self._num_generations = 0
        self.request_ids = list(range(padded_batch_size))
        if offsets is not None:
            self.cu_q_seqlens = offsets
            self.cu_kv_seqlens = offsets

        self.host_request_types[:padded_batch_size].fill_(0)
        self.bind_runtime_views(
            kv_lens_cuda=self._seq_lens_cuda[:padded_batch_size],
            kv_lens=self._seq_lens,
            prompt_lens_cuda=self._seq_lens_cuda[:padded_batch_size],
            prompt_lens_cpu=self._seq_lens,
            host_request_types=self.host_request_types[:padded_batch_size],
        )
        self.host_total_kv_lens[1] = 0
        self.clear_lengths_changed()

    def prepare_encoder_cuda_graph_replay(self, seq_lens: List[int],
                                          padded_num_tokens: int) -> None:
        """Update per-replay encoder CUDA graph metadata in-place."""
        self._seq_lens.copy_(torch.tensor(seq_lens, dtype=torch.int))
        self._num_tokens = padded_num_tokens
        self._num_ctx_tokens = padded_num_tokens
        self.host_total_kv_lens[0] = padded_num_tokens

    def pre_process_for_chunked_prefill(
        self,
        chunked_seq_len: torch.Tensor,
        chunked_global_offset: torch.
        Tensor,  # [chunked_loop_num + 1, num_contexts]
        cu_chunked_seq_len: torch.Tensor,
        merge_op_tensor: torch.Tensor,
        max_chunk_len_per_loop: list[int],
        chunked_loop_num: int,
    ) -> None:
        """
        Pre-process the MLA layer for chunked prefill.
        This method is called before the forward pass to prepare the MLA layer for chunked prefill.
        """
        num_contexts = self.num_contexts
        chunk_size = self.runtime_features.chunk_size
        chunk_batch_size = self.runtime_features.chunked_prefill_buffer_batch_size
        total_chunk_size = chunk_size * chunk_batch_size
        remain_buffer_len = total_chunk_size
        current_batch_idx = 0
        max_chunk_len_per_loop.clear()
        max_chunk_len = 0
        # cal chunked_seq_len
        for batch_idx in range(num_contexts):
            cached_kv_len = self.kv_cache_params.num_cached_tokens_per_seq[
                batch_idx]
            while cached_kv_len > 0:
                used_buffer_len = min(remain_buffer_len, cached_kv_len)
                chunked_seq_len[current_batch_idx, batch_idx] = used_buffer_len
                max_chunk_len = max(max_chunk_len, used_buffer_len)
                remain_buffer_len -= used_buffer_len
                cached_kv_len -= used_buffer_len
                chunked_global_offset[
                    current_batch_idx + 1, batch_idx] = chunked_global_offset[
                        current_batch_idx,
                        batch_idx] + chunked_seq_len[current_batch_idx,
                                                     batch_idx]
                if remain_buffer_len == 0:
                    current_batch_idx += 1
                    remain_buffer_len = total_chunk_size
                    max_chunk_len_per_loop.append(max_chunk_len)
                    max_chunk_len = 0
        if len(max_chunk_len_per_loop) < chunked_loop_num:
            max_chunk_len_per_loop.append(max_chunk_len)
        assert len(
            max_chunk_len_per_loop
        ) == chunked_loop_num, f"max_chunk_len_per_loop size {len(max_chunk_len_per_loop)} != chunked_loop_num {chunked_loop_num}"
        for loop_idx in range(chunked_loop_num):
            cu_chunked_seq_len[loop_idx, 0] = 0
            torch.cumsum(chunked_seq_len[loop_idx, :num_contexts],
                         dim=0,
                         dtype=torch.int64,
                         out=cu_chunked_seq_len[loop_idx, 1:num_contexts + 1])
            for s in range(num_contexts):
                if chunked_seq_len[loop_idx, s] > 0 and (
                        loop_idx == 0 or chunked_seq_len[loop_idx - 1, s] == 0):
                    merge_op_tensor[loop_idx, s] = 2  # copy only
                elif chunked_seq_len[loop_idx, s] > 0:
                    merge_op_tensor[loop_idx, s] = 1  # merge
                else:
                    merge_op_tensor[loop_idx, s] = 0  # skip

        # set merge op for last attn
        for s in range(num_contexts):
            if self.kv_cache_params.num_cached_tokens_per_seq[s] == 0:
                merge_op_tensor[chunked_loop_num, s] = 2  # copy only
            else:
                merge_op_tensor[chunked_loop_num, s] = 1  # merge

    def _prepare_helix_mla_chunks(self) -> None:
        """Map global history chunks to compact reads on every CP rank."""
        self.helix_mla_chunk_plans = None
        if self.helix_context_local_slots is None:
            return
        mapping = self.kv_cache_manager.mapping
        page = self.kv_cache_manager.tokens_per_block
        offsets = self.host_chunked_global_offset[:-1]
        lengths = self.host_chunked_seq_len[:, :self.num_contexts].to(
            torch.int64)
        local_starts = torch.stack([
            helix_local_len_tensor(offsets, page, mapping.cp_size, rank)
            for rank in range(mapping.cp_size)
        ],
                                   dim=1)
        local_lengths = torch.stack([
            helix_local_len_tensor(offsets + lengths, page, mapping.cp_size,
                                   rank) for rank in range(mapping.cp_size)
        ],
                                    dim=1) - local_starts
        cu_local = torch.nn.functional.pad(local_lengths.cumsum(-1), (1, 0))
        sizes = cu_local[:, :, -1]
        rank_offsets = torch.nn.functional.pad(sizes.cumsum(-1), (1, 0))
        device = self.seq_lens_cuda.device
        device_cu = maybe_pin_memory(cu_local).to(device, non_blocking=True)
        device_starts = maybe_pin_memory(local_starts).to(device,
                                                          non_blocking=True)
        device_rank_offsets = maybe_pin_memory(rank_offsets).to(
            device, non_blocking=True)
        self.helix_mla_chunk_plans = [
            HelixMlaChunkPlan(
                cu_seq_lens=device_cu[loop],
                local_offsets=device_starts[loop],
                rank_offsets=device_rank_offsets[loop],
                rank_sizes=sizes[loop].tolist(),
                max_local_seq_len=local_lengths[loop,
                                                mapping.cp_rank].max().item(),
            ) for loop in range(self.chunked_loop_num)
        ]

    def prepare_context_mla_with_cached_kv(self,
                                           cached_token_lens: torch.Tensor,
                                           kv_lens: torch.Tensor) -> None:
        self.helix_mla_chunk_plans = None
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[:self.
                                                           num_contexts].sum(
                                                           ).item()
            self.max_ctx_cached_token_len = cached_token_lens[:self.
                                                              num_contexts].max(
                                                              ).item()
            self.max_ctx_kv_len = kv_lens[:self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[:self.num_contexts].max().item(
            )
            # determine the number of loop
            # currently we assume that the chunk size is the same as the max_num_tokens
            if self.runtime_features.chunked_prefill:
                chunk_size = self.runtime_features.chunk_size
                chunk_batch_size = self.runtime_features.chunked_prefill_buffer_batch_size
                total_chunk_size = chunk_size * chunk_batch_size
                self.chunked_loop_num = math.ceil(self.num_ctx_cached_tokens /
                                                  total_chunk_size)
                self.chunked_seq_len = torch.zeros(
                    (self.chunked_loop_num, self.num_seqs),
                    dtype=torch.int,
                    device='cuda',
                )
                self.host_chunked_seq_len = torch.zeros_like(
                    self.chunked_seq_len,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.cu_chunked_seq_len = torch.zeros(
                    (self.chunked_loop_num, self.num_contexts + 1),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_cu_chunked_seq_len = torch.zeros_like(
                    self.cu_chunked_seq_len,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.chunked_global_offset = torch.zeros(
                    (self.chunked_loop_num + 1, self.num_contexts),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_chunked_global_offset = torch.zeros_like(
                    self.chunked_global_offset,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )
                self.max_chunk_len_per_loop = []
                # For last chunk we use the uncached kv
                self.merge_op_tensor = torch.empty(
                    (self.chunked_loop_num + 1, self.num_contexts),
                    dtype=torch.int64,
                    device='cuda',
                )
                self.host_merge_op_tensor = torch.empty_like(
                    self.merge_op_tensor,
                    device='cpu',
                    pin_memory=prefer_pinned(),
                )

                self.pre_process_for_chunked_prefill(
                    chunked_seq_len=self.host_chunked_seq_len,
                    chunked_global_offset=self.host_chunked_global_offset,
                    cu_chunked_seq_len=self.host_cu_chunked_seq_len,
                    merge_op_tensor=self.host_merge_op_tensor,
                    max_chunk_len_per_loop=self.max_chunk_len_per_loop,
                    chunked_loop_num=self.chunked_loop_num)
                self.chunked_seq_len.copy_(self.host_chunked_seq_len,
                                           non_blocking=True)
                self.cu_chunked_seq_len.copy_(self.host_cu_chunked_seq_len,
                                              non_blocking=True)
                self.merge_op_tensor.copy_(self.host_merge_op_tensor,
                                           non_blocking=True)
                self.chunked_global_offset.copy_(
                    self.host_chunked_global_offset, non_blocking=True)
                self._prepare_helix_mla_chunks()
        else:
            self.num_ctx_cached_tokens = 0
            self.max_ctx_cached_token_len = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0
            # The indptrs below are read only by the context MLA kernels, and
            # every batch with context requests rewrites them.
            return
        torch.cumsum(cached_token_lens[:self.num_contexts],
                     dim=0,
                     dtype=torch.int64,
                     out=self.host_ctx_cached_token_indptr[1:self.num_contexts +
                                                           1])
        self.ctx_cached_token_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_cached_token_indptr[:self.num_contexts + 1],
            non_blocking=True)
        torch.cumsum(
            self.seq_lens[:self.num_contexts],
            dim=0,
            dtype=torch.int64,
            out=self.host_ctx_uncached_token_indptr[1:self.num_contexts + 1])
        self.ctx_uncached_token_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_uncached_token_indptr[:self.num_contexts + 1],
            non_blocking=True)
        torch.cumsum(kv_lens[:self.num_contexts],
                     dim=0,
                     dtype=torch.int64,
                     out=self.host_ctx_kv_indptr[1:self.num_contexts + 1])
        self.ctx_kv_indptr[:self.num_contexts + 1].copy_(
            self.host_ctx_kv_indptr[:self.num_contexts + 1], non_blocking=True)


class TrtllmAttentionMetadataNativeFields:
    """The metadata members the native attention op reads, with their types.

    The native op reads ``meta.<name>`` of the metadata by attribute name.
    scripts/generate_fmha_params.py mirrors TrtllmAttentionMetadata into the
    native AttentionMetadataParams from these annotations, because the
    metadata's members live on its core and extensions. Every name resolves on
    every TrtllmAttentionMetadata: the core's and the attached extensions'
    fields are forwarded, and TrtllmAttentionMetadata answers the Helix and
    FlashMLA members with None while those extensions are not attached.
    """

    max_num_requests: int
    max_num_sequences: Optional[int]
    cache_indirection: Optional[torch.Tensor]
    is_cross: bool
    effective_beam_width: int
    max_seq_len: int
    max_context_q_len_override: Optional[int]
    is_spec_decoding_enabled: bool
    is_spec_dec_tree: bool
    force_prepare_spec_dec_tree_mask: bool
    max_total_draft_tokens: Optional[int]
    trtllm_gen_jit_warmup: bool
    helix_position_offsets: Optional[torch.Tensor]
    helix_is_inactive_rank: Optional[torch.Tensor]
    kv_cache_block_offsets: Optional[torch.Tensor]
    block_ids_per_seq: Optional[torch.Tensor]
    flash_mla_tile_scheduler_metadata: Optional[torch.Tensor]
    flash_mla_num_splits: Optional[torch.Tensor]
    use_paged_context_fmha: bool
    num_sparse_topk: int
    max_context_length: int
    host_kv_cache_pool_pointers: Optional[torch.Tensor]
    host_kv_cache_pool_mapping: Optional[torch.Tensor]
    kv_lens_runtime: torch.Tensor
    prompt_lens_cpu_runtime: torch.Tensor


# Extension fields the native op reads by name, and their value while the
# declaring extension is not attached.
_NATIVE_FIELDS_OF_ABSENT_EXTENSIONS = {
    "helix_position_offsets": None,
    "helix_is_inactive_rank": None,
    "flash_mla_tile_scheduler_metadata": None,
    "flash_mla_num_splits": None,
    "is_spec_decoding_enabled": False,
    "is_spec_dec_tree": False,
    "max_total_draft_tokens": None,
}


class TrtllmAttentionMetadata(AttentionMetadata):
    core_metadata_type: ClassVar[
        Type[AttentionCoreMetadata]] = TrtllmCoreMetadata
    # The attention op reads cu_q_seqlens / cu_kv_seqlens as given.
    supports_explicit_offsets: ClassVar[bool] = True
    # The device KV lengths take the step's corrections in prepare.
    supports_kv_lens_offsets: ClassVar[bool] = True

    # Tensors produced and consumed inside the captured graph; the strict
    # buffer check skips them.
    graph_temporary_attrs: ClassVar[FrozenSet[str]] = frozenset()

    @classmethod
    def default_extension_types(cls, settings: dict[str, Any]) -> list:
        types = [
            *super().default_extension_types(settings), SpecDecodingMetadata
        ]
        mapping = settings.get("mapping")
        if mapping is not None and mapping.has_cp_helix():
            types.append(HelixMetadata)
        if callable(
                getattr(settings.get("kv_cache_manager"),
                        "get_fp4_mla_page_table_spec", None)):
            types.append(Fp4MlaState)
        return types

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup fails, so only while the extension
        # declaring a native-read field is not attached.
        if name in _NATIVE_FIELDS_OF_ABSENT_EXTENSIONS:
            return _NATIVE_FIELDS_OF_ABSENT_EXTENSIONS[name]
        return super().__getattr__(name)


class TrtllmAttention(AttentionBackend[TrtllmAttentionMetadata]):

    Metadata = TrtllmAttentionMetadata
    # Run context-only calls from the forward's shared TrtllmAttentionMetadata.context_plan.
    cache_context_plan: bool = False
    # FMHA library classes this instance tries before the registry's libraries.
    prepended_fmha_lib_classes: tuple = ()

    # Layer state the native attention op reads through FmhaParams.attn. The
    # generator (scripts/generate_fmha_params.py) collects these annotations;
    # attributes assigned from an annotated __init__ parameter need none.
    num_kv_heads: int
    is_mla_enable: bool
    v_head_dim: int
    rope_params: RopeParams
    q_scaling: float
    predicted_tokens_per_seq: int
    skip_correction_threshold: float
    uses_spcompress: bool
    q_lora_rank: Optional[int]
    kv_lora_rank: Optional[int]
    qk_nope_head_dim: Optional[int]
    qk_rope_head_dim: Optional[int]
    rope_append: Optional[bool]
    rotary_inv_freq: Optional[torch.Tensor]
    rotary_cos_sin: Optional[torch.Tensor]
    position_embedding_type: PositionEmbeddingType
    skip_softmax_stat: torch.Tensor
    quant_mode: QuantMode

    @staticmethod
    def is_sm_version_trtllm_gen_kernel(sm):
        return is_sm_100f(sm)

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_params: Optional[SparseParams] = None,
        kv_cache_dtype: str = "auto",
        skip_correction_threshold: float = 0.0,
        gqa_decode: Optional["GqaDecodeSetup"] = None,
        cache_context_plan: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the backend.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
            q_scaling (float): Scaling factor for QK. Defaults to 1.0 if None.
            pos_embd_params (PositionalEmbeddingParams): Optional parameters defining how positional embedding should be applied.
                                                         If None, positional embedding should be applied by the model before calling the backend.
                                                         Otherwise, the backend is in-charge of applying positional embedding and may cache K without embedding it first.
            mla_params (MLAParams): Optional parameters for MLA. If None, MLA is not enabled.
            sparse_params (SparseParams): Optional sparse-attention backend parameters
                (e.g. skip-softmax). Algorithm-specific fields are documented on the
                corresponding ``SparseParams`` subclass.
            kv_cache_dtype (str): KV-cache dtype selected by ``KvCacheConfig``. Accepted
                values are ``auto``, ``fp8``, ``fp8_ds_mla``, ``nvfp4``, and supported
                torch dtype strings. ``fp8_ds_mla`` selects the packed sparse-MLA cache
                used by DeepSeek-V4 and DSA on SM120/SM121.
            skip_correction_threshold (float): Runtime MLA threshold. Zero disables
                skip-correction.
            gqa_decode (GqaDecodeSetup): Enables the GQA decode FMHA library
                (fmha/gqa_decode.py) for this layer.
            cache_context_plan (bool): Plan a context-only forward once for the
                layers that share its metadata and run the trtllm-gen context
                kernel from that plan (dense, non-sparse layers only).
        """
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, **kwargs)
        self.sparse_params = sparse_params
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"
        self.is_mla_enable = mla_params is not None
        sparse_algorithm = getattr(self.sparse_params, "algorithm", None)
        if (self.is_mla_enable and sparse_algorithm in ("deepseek_v4", "dsa")
                and is_sm_120f() and self.kv_cache_dtype != "fp8_ds_mla"):
            raise ValueError(
                "DeepSeek-V4/DSA sparse MLA on SM120/SM121 requires "
                "kv_cache_config.dtype='fp8_ds_mla'.")
        self.mla_params = mla_params or MLAParams()
        self.v_head_dim = self.mla_params.v_head_dim if self.is_mla_enable else head_dim

        rope_params = None
        if pos_embd_params is not None:
            rope_params = pos_embd_params.rope
        self.rope_params = rope_params or RopeParams()

        self.q_scaling = q_scaling or 1.0
        self.predicted_tokens_per_seq = self.mla_params.predicted_tokens_per_seq
        self.attention_chunk_size = attention_chunk_size
        self.skip_correction_threshold = _resolve_skip_correction_threshold(
            skip_correction_threshold,
            get_sm_version(),
            is_mla=self.is_mla_enable)
        self.uses_spcompress = _resolve_uses_spcompress(sparse_params,
                                                        get_sm_version())

        if self.is_mla_enable:
            self.q_lora_rank = self.mla_params.q_lora_rank
            self.kv_lora_rank = self.mla_params.kv_lora_rank
            self.qk_nope_head_dim = self.mla_params.qk_nope_head_dim
            self.qk_rope_head_dim = self.mla_params.qk_rope_head_dim
            self.rope_append = self.mla_params.rope_append
        else:
            self.q_lora_rank = None
            self.kv_lora_rank = None
            self.qk_nope_head_dim = None
            self.qk_rope_head_dim = None
            self.rope_append = None

        self.rotary_inv_freq, self.rotary_cos_sin = self.rope_params.create_rope_const_params(
        )
        self.position_embedding_type = (pos_embd_params.type
                                        if pos_embd_params is not None else
                                        PositionEmbeddingType.learned_absolute)
        self.skip_softmax_stat = torch.zeros(2,
                                             dtype=torch.uint32,
                                             device='cuda')
        self.print_skip_softmax_stat = os.environ.get(
            "TRTLLM_PRINT_SKIP_SOFTMAX_STAT", "0") == "1"

        # Layer-level fp8 KV-cache scales. Stay at 1.0 (no-op) for the
        # PyTorch backend, which never overrides them. They guarantee the
        # kernel always receives a valid pointer, since several non-MLA
        # XQA kernels (cpp/kernels/xqa/mha.cu, mha_sm90.cu) deref
        # `kvCacheScale[0]` whenever `isKVCacheQuantized` is true and
        # do not check for nullptr. `attention/attention.py` only assigns
        # `forward_args.kv_scale_*` for fp4 KV cache, so without this
        # fallback the kernel takes nullptr on fp8-KV models → illegal
        # memory access.
        self.kv_cache_scaling_factor = torch.ones(1,
                                                  dtype=torch.float32,
                                                  device='cuda')
        self.kv_scale_quant_orig = self.kv_cache_scaling_factor
        self.kv_scale_orig_quant = 1.0 / self.kv_cache_scaling_factor

        self.local_layer_idx: Optional[int] = None
        self._fmha_manager: Optional[FmhaManager] = None
        self.gqa_decode = gqa_decode
        # A GQA decode library of this layer, which plans the module's KV append and deferred prologue (set with
        # the FMHA manager).
        self.gqa_decode_fmha: Optional["GqaDecodeFmha"] = None
        self.cache_context_plan = (cache_context_plan and not self.is_mla_enable
                                   and self.sparse_params is None
                                   and not self.print_skip_softmax_stat)
        # Layer part of the context plan key (set with the FMHA manager).
        self._context_plan_layer_key: tuple = ()
        # This layer's instance of the plan's FMHA class, the manager owning it,
        # and the multi-CTA counter rows the instance was prepared for.
        self._context_plan_fmha: Optional["Fmha"] = None
        self._context_plan_fmha_owner: Optional[FmhaManager] = None
        self._context_plan_counter_rows = 0
        if not skip_create_weights_in_init:
            self.update_quant_config(self.quant_config)

    @property
    def uses_fp4_mla_attention(self) -> bool:
        """Whether this layer executes dense FP4 MLA, not sparse NVFP4 storage."""
        return (self.is_mla_enable and self.has_fp4_kv_cache
                and self.sparse_params is None)

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        self.quant_config = new_quant_config or QuantConfig()
        self.quant_mode = self.quant_config.layer_quant_mode
        self.has_fp8_qdq = self.has_fp8_kv_cache = self.has_nvfp4 = False
        if self.quant_config is not None:
            self.has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache(
            )
            self.has_fp4_kv_cache = self.quant_config.layer_quant_mode.has_fp4_kv_cache(
            )

            self.has_fp8_qdq = self.quant_config.layer_quant_mode.has_fp8_qdq()
            self.has_fp8_block_wise = self.quant_config.layer_quant_mode.has_fp8_block_scales(
            )
            self.has_fp8_rowwise = self.quant_config.layer_quant_mode.has_fp8_rowwise(
            )
            self.has_nvfp4 = self.quant_config.layer_quant_mode.has_nvfp4()
            self.has_w4a8_nvfp4_fp8 = self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8(
            )
        self._fmha_manager = FmhaManager(self)
        # Every layer input of the FMHA selection and of the context workspace size.
        self._context_plan_layer_key = ()
        if self.cache_context_plan:
            self._context_plan_layer_key = (tuple(
                type(fmha)
                for fmha in self._fmha_manager.fmha_libs), self.num_heads,
                                            self.num_kv_heads, self.head_dim,
                                            self.rope_dim, self.quant_mode,
                                            self.position_embedding_type,
                                            self.attention_chunk_size,
                                            self.q_scaling, self.kv_cache_dtype)
        from .fmha.gqa_decode import GqaDecodeFmha
        self.gqa_decode_fmha = next((fmha
                                     for fmha in self._fmha_manager.fmha_libs
                                     if isinstance(fmha, GqaDecodeFmha)), None)

    @classmethod
    def runtime_workspace_bytes_per_token(cls, model_config: "ModelConfig",
                                          mapping: "Mapping") -> int:
        """Context-MLA workspace sized by summed attended KV length.

        Dense fp8 context-MLA stages expanded K/V. NVFP4 sparse MLA stages
        selected latent rows in fp8 plus selection/scan workspace. Cached
        tokens can grow both beyond the fresh-prefill profiling floor.
        """
        config = model_config.pretrained_config
        if not is_mla(config):
            return 0
        quant_config = model_config.quant_config
        quant_mode = getattr(quant_config, "quant_mode", None)
        sparse_algorithm = getattr(model_config.sparse_attention_config,
                                   "algorithm", None)
        nvfp4_dsa_context = (quant_mode is not None and getattr(
            quant_mode, "has_fp4_kv_cache", lambda: False)()
                             and sparse_algorithm == "dsa"
                             and get_sm_version() >= 100)
        if nvfp4_dsa_context:
            # The output row has one fp8 value per latent element. The gather
            # also allocates three int32 arrays, CUB scan storage, and aligned
            # padding; 64 B/token is a conservative bound for that auxiliary
            # workspace.
            nvfp4_gather_aux_bytes_per_token = 64
            return int(config.kv_lora_rank + config.qk_rope_head_dim +
                       nvfp4_gather_aux_bytes_per_token)

        nvfp4_dsv4_context = (quant_mode is not None and getattr(
            quant_mode, "has_fp4_kv_cache", lambda: False)()
                              and sparse_algorithm == "deepseek_v4"
                              and get_sm_version() >= 100)
        if nvfp4_dsv4_context:
            compress_ratios = [
                ratio for ratio in getattr(model_config.sparse_attention_config,
                                           "compress_ratios", []) if ratio > 1
            ]
            if not compress_ratios:
                return 0
            # Context compaction allocates one fp8 latent row and its
            # selection/scan workspace per active compressed token. Express
            # that in raw-KV-token units for the estimator and use the least
            # compressed layer as the whole-model upper bound.
            min_compress_ratio = min(compress_ratios)
            latent_dim = config.kv_lora_rank + config.qk_rope_head_dim
            nvfp4_gather_aux_bytes_per_compressed_token = 64
            return math.ceil(
                (latent_dim + nvfp4_gather_aux_bytes_per_compressed_token) /
                min_compress_ratio)

        if sparse_algorithm == "csa2":
            from .sparse.csa2.backend import CSA2TrtllmAttention
            return CSA2TrtllmAttention.runtime_workspace_bytes_per_token(
                model_config, mapping)

        has_fp4_kv_cache = (quant_mode is not None and getattr(
            quant_mode, "has_fp4_kv_cache", lambda: False)())
        # Dense FP4 MLA attends the current chunk and every cached-prefix
        # partition through the FP8 context kernel, even though the
        # model-wide cache quantization is NVFP4.
        nvfp4_uses_fp8_context = (has_fp4_kv_cache and
                                  model_config.sparse_attention_config is None)
        if has_fp4_kv_cache and is_kimi_k3(config):
            # Kimi K3 with an NVFP4 KV cache keeps FP8 fallback MLA layers,
            # which stage the same context buffers as an all-FP8 model when
            # this pipeline stage holds one of them.
            fp8_fallback_layers = get_kimi_k3_fp8_kv_layer_ids(
                config, nvfp4_enabled=True, mapping=mapping)
            if fp8_fallback_layers and fp8_fallback_layers.intersection(
                    mapping.pp_layers(config.num_hidden_layers)):
                nvfp4_uses_fp8_context = True
        sm_version = get_sm_version()
        fp8_context_mla = (quant_mode is not None
                           and (quant_mode.has_fp8_kv_cache()
                                or nvfp4_uses_fp8_context)
                           and (sm_version == 90 or is_sm_100f(sm_version)
                                or is_sm_120f(sm_version)))
        if not fp8_context_mla:
            return 0
        # Attention-DP runs the full head set per rank; otherwise heads shard across TP (mirror
        # mNumAttnHeads).
        attn_tp = 1 if mapping.enable_attention_dp else mapping.tp_size
        num_attn_heads = config.num_attention_heads // attn_tp
        # The buffer is skipped only where AttentionOp::useSparseMLA() holds, which needs all three of:
        #   * DSA / DeepSeek-V4 -- only these lower to the absorption path that reads K/V from the paged
        #     cache. Skip-softmax passes no sparse indices to C++, and its ignore-list can exclude a layer,
        #     so those layers still run dense MLA. The workspace is shared, so one dense layer forces the
        #     reserve.
        #   * SM 100 family -- where mUseTllmGen is true.
        #   * short-seq MHA fallback off -- it routes short contexts back through the dense path.
        # Match the runtime predicate, not just "a sparse config exists": over-reserving costs KV pool,
        # under-reserving OOMs mid-forward.
        sparse_mla = (sparse_algorithm in ("dsa", "deepseek_v4")
                      and is_sm_100f())
        short_seq_mha_enabled = int(
            os.environ.get("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", "0")) > 0
        stages_no_buffer = sparse_mla and not short_seq_mha_enabled
        return int(
            thop.get_context_mla_workspace_bytes_per_token(
                num_attn_heads=num_attn_heads,
                qk_rope_head_dim=config.qk_rope_head_dim,
                qk_nope_head_dim=config.qk_nope_head_dim,
                v_head_dim=config.v_head_dim,
                fp8_context_mla=fp8_context_mla,
                # Paged context MLA always uses separate Q/KV input; the term is otherwise gated to 0.
                separate_q_and_kv_input=True,
                sparse_mla=stages_no_buffer,
            ))

    @classmethod
    def runtime_workspace_fixed_bytes(cls, model_config: "ModelConfig",
                                      mapping: "Mapping", *,
                                      max_batch_size: int, max_num_tokens: int,
                                      max_seq_len: int,
                                      enable_cuda_graph: bool) -> int:
        if getattr(model_config.sparse_attention_config, "algorithm",
                   None) != "csa2":
            return 0
        from .sparse.csa2.backend import CSA2TrtllmAttention
        return CSA2TrtllmAttention.runtime_workspace_fixed_bytes(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            enable_cuda_graph=enable_cuda_graph)

    @classmethod
    def runtime_workspace_is_chunked_prefill_bounded(
            cls, model_config: "ModelConfig") -> bool:
        """NVFP4 sparse MLA gathers from the complete attended prefix."""
        config = model_config.pretrained_config
        if not is_mla(config):
            return True
        quant_config = model_config.quant_config
        quant_mode = getattr(quant_config, "quant_mode", None)
        sparse_algorithm = getattr(model_config.sparse_attention_config,
                                   "algorithm", None)
        if sparse_algorithm == "csa2":
            return False
        return not (quant_mode is not None
                    and getattr(quant_mode, "has_fp4_kv_cache", lambda: False)()
                    and sparse_algorithm in ("dsa", "deepseek_v4")
                    and get_sm_version() >= 100)

    def get_local_layer_idx(self, metadata: TrtllmAttentionMetadata) -> int:
        if self.local_layer_idx is not None:
            return self.local_layer_idx
        if metadata.kv_cache_manager is None:
            # Uncached: recomputed each call until a cache manager appears.
            return self.layer_idx
        self.local_layer_idx = metadata.kv_cache_manager.layer_offsets[
            self.layer_idx]
        return self.local_layer_idx

    def get_fp4_mla_local_layer_idx(self,
                                    metadata: TrtllmAttentionMetadata) -> int:
        """Return the compact index used by FP4 MLA-only side pools."""
        local_layer_idx = self.get_local_layer_idx(metadata)
        if metadata.kv_cache_manager is None:
            return local_layer_idx
        to_compact = getattr(metadata.kv_cache_manager,
                             "_fp4_mla_compact_layer_idx", None)
        if not callable(to_compact):
            raise RuntimeError(
                "FP4 MLA requires a cache manager with compact layer mapping.")
        return to_compact(local_layer_idx)

    def use_nvfp4_output(
        self,
        metadata: TrtllmAttentionMetadata,
        attention_mask: AttentionMask,
    ) -> bool:
        # Not running NVFP4
        if not self.has_nvfp4:
            return False

        # Default enabled, but allow manual disabling through `TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT=0`
        if not os.environ.get("TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT",
                              "1") == "1":
            return False

        use_paged_context_fmha = (
            metadata.runtime_features.chunked_prefill
            or metadata.runtime_features.cache_reuse
            or metadata.runtime_features.has_speculative_draft_tokens
        ) if metadata.runtime_features else False

        return self._is_nvfp4_output_kernel_available(
            tokens_per_block=metadata.tokens_per_block,
            attention_mask=attention_mask,
            use_paged_context_fmha=use_paged_context_fmha,
            is_mla_enable=self.is_mla_enable,
        )

    def get_quantize_output_dtype(
            self, use_nvfp4_output: bool) -> Optional[torch.dtype]:
        if use_nvfp4_output:
            # Use UINT8 as the container dtype for NVFP4.
            return torch.uint8
        elif (self.has_fp8_qdq or self.has_nvfp4 or self.has_fp8_block_wise
              or self.has_fp8_rowwise
              or self.has_w4a8_nvfp4_fp8) and (self.has_fp8_kv_cache
                                               or self.has_fp4_kv_cache):
            return torch.float8_e4m3fn
        return None

    def _quantize_sage_inputs(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Optional[SageQuantized]:
        """Quantize the Sage inputs of block-sparse and V-smoothing calls.

        The attention op quantizes dense SageAttention inside its kernels and
        cannot add a V mean back, so only block-sparse calls and calls with V
        smoothing are quantized here, given separate Q, K, and V without a KV
        cache. Each FMHA library decides whether it takes the quantized inputs;
        the attention op ignores them.
        """
        qk_quant_dtype = forward_args.sage_attn_qk_dtype
        if (qk_quant_dtype is None or k is None or v is None
                or metadata.kv_cache_manager is not None):
            return None
        inputs = forward_args.sparse_runtime_params.block_sparse_inputs
        if inputs is None and not forward_args.sage_attn_smooth_v:
            return None
        num_seqs = metadata.num_contexts
        # Device lengths add no host-to-device copy during CUDA Graph capture.
        cu_seqlens_q = torch.zeros(num_seqs + 1,
                                   dtype=torch.int32,
                                   device=q.device)
        cu_seqlens_kv = torch.zeros_like(cu_seqlens_q)
        cu_seqlens_q[1:] = metadata.seq_lens_cuda[:num_seqs].cumsum(
            0, dtype=torch.int32)
        cu_seqlens_kv[1:] = metadata.seq_lens_kv_cuda[:num_seqs].cumsum(
            0, dtype=torch.int32)
        quantized = sage_quantize(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            cu_seqlens_q,
            cu_seqlens_kv,
            q_block_size=forward_args.sage_attn_num_elts_per_blk_q,
            k_block_size=forward_args.sage_attn_num_elts_per_blk_k,
            qk_quant_dtype=qk_quant_dtype,
            smooth_k=forward_args.sage_attn_smooth_k,
            smooth_v=forward_args.sage_attn_smooth_v,
        )
        if inputs is None or not inputs.use_proxy_routes:
            return quantized
        # The K/V block means of proxy routes dequantize like the keys and
        # values they stand for: every mean is a key of its sequence with a
        # scale of its own, and reuses the V scales and means of this call.
        batch, blocks = inputs.k_summary.shape[:2]
        cu_seqlens_blocks = torch.arange(0, (batch + 1) * blocks,
                                         blocks,
                                         dtype=torch.int32,
                                         device=q.device)
        summary = sage_quantize(None,
                                inputs.k_summary.flatten(0, 1),
                                inputs.v_summary.flatten(0, 1),
                                None,
                                cu_seqlens_blocks,
                                k_block_size=1,
                                qk_quant_dtype=qk_quant_dtype,
                                stats=quantized)
        return quantized._replace(
            k_summary=summary.k.view(inputs.k_summary.shape),
            v_summary=summary.v.view(inputs.v_summary.shape),
            k_summary_scale=summary.k_scale)

    def _ensure_rope_table_size(self, required_max_positions: int) -> None:
        floats_per_position = self.rope_params.dim * (
            2 if self.rope_params.duplicate_data else 1)
        table_max_positions = (self.rotary_cos_sin.numel() //
                               floats_per_position
                               if self.rotary_cos_sin is not None else 0)
        self.rope_params.max_positions = max(self.rope_params.max_positions,
                                             table_max_positions,
                                             required_max_positions)
        if required_max_positions > table_max_positions:
            self.rotary_inv_freq, self.rotary_cos_sin = (
                self.rope_params.create_rope_const_params())

    def _is_nvfp4_output_kernel_available(
        self,
        *,
        tokens_per_block: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        use_paged_context_fmha: bool = False,
        is_mla_enable: bool = False,
    ) -> bool:
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            mask_type = AttentionMaskType.causal
        elif attention_mask == PredefinedAttentionMask.FULL:
            mask_type = AttentionMaskType.padding
        else:
            raise ValueError("Unexpected attention mask type")

        return torch.ops.trtllm.attention_supports_nvfp4_output(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            tokens_per_block,
            int(mask_type),
            self.quant_mode,
            use_paged_context_fmha,
            is_mla_enable,
        )

    def out_head_size(self, is_gen_only: bool) -> int:
        """Per-head width of the attention output for this phase.

        Single source of truth for the output buffer `create_output` allocates and
        the view a phased FMHA library takes over it; the two must not drift. MLA
        generation writes the latent (plus the rope part when it is not appended in
        place), while MLA context writes the already-projected V head.
        """
        if not self.is_mla_enable:
            return self.head_dim
        if not is_gen_only:
            return self.v_head_dim
        return (self.kv_lora_rank if self.rope_append else self.kv_lora_rank +
                self.qk_rope_head_dim)

    def _is_variable_window_kernel_available(
        self,
        *,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return torch.ops.trtllm.attention_supports_variable_window(
            dtype,
            output_dtype,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            metadata.tokens_per_block,
            self.quant_mode,
            metadata.max_num_requests,
            metadata.max_context_length,
            metadata.max_seq_len,
            self.attention_chunk_size,
            metadata.kv_cache_manager is not None,
            metadata.use_paged_context_fmha,
        )

    def create_output(self, q, *, is_quantize_output: bool,
                      metadata: TrtllmAttentionMetadata,
                      attention_mask: AttentionMask, is_gen_only: bool,
                      **kwargs) -> List[torch.Tensor]:
        use_nvfp4_output = False
        out_dtype = None
        if is_quantize_output:
            use_nvfp4_output = self.use_nvfp4_output(metadata, attention_mask)
            out_dtype = self.get_quantize_output_dtype(use_nvfp4_output)

        num_tokens = q.size(0)
        if out_dtype is None:
            out_dtype = q.dtype
        v_head_size = self.out_head_size(is_gen_only)
        if use_nvfp4_output:
            num_nvfp4_elements_per_container = 2
            scaling_vector_size = 16
            size_per_token = self.num_heads * v_head_size
            output = q.new_empty(
                (num_tokens,
                 size_per_token // num_nvfp4_elements_per_container),
                dtype=torch.uint8)
            padded_row, padded_col = compute_swizzled_sf_shape(
                num_tokens, size_per_token // scaling_vector_size)
            output_sf = q.new_empty(padded_row * padded_col, dtype=torch.uint8)
            return [output, output_sf]
        return [
            q.new_empty((num_tokens, self.num_heads * v_head_size),
                        dtype=out_dtype)
        ]

    @property
    def rope_dim(self) -> int:
        return self.rope_params.dim

    @property
    def rope_base(self) -> float:
        return self.rope_params.theta

    @property
    def rope_scale_type(self) -> RotaryScalingType:
        return self.rope_params.scale_type

    @property
    def rope_scale(self) -> float:
        return self.rope_params.scale

    @property
    def rope_short_m_scale(self) -> float:
        return self.rope_params.short_m_scale

    @property
    def rope_long_m_scale(self) -> float:
        return self.rope_params.long_m_scale

    @property
    def rope_max_positions(self) -> int:
        return self.rope_params.max_positions

    @property
    def rope_original_max_positions(self) -> int:
        return self.rope_params.original_max_positions

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute the TRTLLM attention backend."""
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if isinstance(metadata, AttentionMetadata):
            # No forward on lengths changed in place without a refresh.
            metadata.check_lengths_current()
        has_variable_window_starts = (forward_args.variable_window_token_starts
                                      is not None)
        has_variable_window_ends = forward_args.variable_window_token_ends is not None
        if has_variable_window_starts != has_variable_window_ends:
            raise ValueError(
                "variable_window_token_starts and variable_window_token_ends "
                "must be provided together")
        if (has_variable_window_starts and forward_args.attention_input_type
                == AttentionInputType.generation_only):
            raise ValueError(
                "variable-window bounds are supported only for context attention"
            )
        if self.cache_context_plan and not has_variable_window_starts and self._run_context_plan(
                q, k, v, metadata, forward_args):
            return forward_args.output
        assert isinstance(
            metadata,
            TrtllmAttentionMetadata,
        )
        token_major_view = (metadata.token_major_view()
                            if forward_args.attention_input_type
                            == AttentionInputType.generation_only else None)
        if (metadata.find_ext(FlashMlaMetadata) is not None
                and token_major_view is not None):
            raise ValueError(
                "FlashMLA does not support ragged token-major generation; "
                "disable FlashMLA for this layout.")
        # Cross-attention uses the THOP path; the trtllm-gen backend API does
        # not carry encoder K/V tensors yet.

        # Sparse mqa/gqa attention uses generation kernel which reads Q from qPtr (separate buffer).
        # Force paged context FMHA so QKV preprocessing writes Q to q_buf_2_.
        if (self.sparse_params is not None and getattr(
                self.sparse_params, 'algorithm', None) == 'mqa_gqa'):
            metadata.use_paged_context_fmha = True

        if self.is_mla_enable:
            # Context MLA uses separate qkv instead of paged_context_fmha
            metadata.use_paged_context_fmha = False

        if (forward_args.enable_dsv4_epilogue_fusion and
            (forward_args.output is None or forward_args.output_sf is None)):
            raise ValueError(
                "DSv4 epilogue fusion requires caller-provided output and "
                "output_sf buffers.")

        if forward_args.output is None:
            is_gen_only = (forward_args.attention_input_type ==
                           AttentionInputType.generation_only)
            outputs = self.create_output(
                q,
                is_quantize_output=forward_args.out_scale is not None,
                metadata=metadata,
                attention_mask=forward_args.attention_mask,
                use_paged_context_fmha=metadata.use_paged_context_fmha,
                is_mla_enable=self.is_mla_enable,
                is_gen_only=is_gen_only,
            )
            forward_args.output = outputs[0]
            forward_args.output_sf = outputs[1] if len(outputs) == 2 else None

        has_q_only = False
        if self.is_mla_enable:
            forward_args.is_fused_qkv = False
            forward_args.update_kv_cache = True
            has_q_only = k is None and v is None
        elif not metadata.is_cross and k is None and v is None:
            q_hidden_size = self.num_heads * self.head_dim
            qkv_hidden_size = q_hidden_size + 2 * self.num_kv_heads * self.head_dim
            has_q_only = q.size(-1) == q_hidden_size
            forward_args.is_fused_qkv = q.size(-1) == qkv_hidden_size
            forward_args.update_kv_cache = not has_q_only
        else:
            forward_args.is_fused_qkv = not metadata.is_cross and k is None
            forward_args.update_kv_cache = not metadata.is_cross or k is not None
        has_fused_qkv = forward_args.is_fused_qkv and k is None and v is None
        has_unfused_kv = (not forward_args.is_fused_qkv and k is not None
                          and v is not None)
        uses_cached_cross_kv = (metadata.is_cross
                                and not forward_args.update_kv_cache
                                and k is None and v is None)
        assert (has_fused_qkv or has_unfused_kv or has_q_only
                or uses_cached_cross_kv)
        # `quant_scale_qkv` only makes sense paired with `quant_q_buffer`: the
        # C++ op interprets the buffer as the destination of a pre-quantized
        # FP8 Q for the DSv4 fused norm+RoPE path. `quant_q_buffer` alone is
        # still valid for the regular FP8-KV-cache path.
        assert (forward_args.quant_scale_qkv is None
                or forward_args.quant_q_buffer is not None), (
                    "quant_scale_qkv requires quant_q_buffer to be set")
        if forward_args.cu_q_seqlens is None:
            forward_args.cu_q_seqlens = metadata.cu_q_seqlens
        if forward_args.cu_kv_seqlens is None:
            forward_args.cu_kv_seqlens = metadata.cu_kv_seqlens

        # Testing only: ``mla_rope_generation`` normally rotates q_pe, appends the
        # new latent to the paged cache, and fills the trtllm-gen scheduler
        # buffers (cumulative q/kv seqlens + the FMHA scheduler counter). When the
        # harness sets ``skip_mla_rope_generation`` it feeds a pre-RoPE'd fused_q,
        # so we skip only the RoPE and do the append + scheduler init here: the
        # generation FMHA only reads the cache, and the fallback path needs the
        # scheduler buffers (the flashinfer trtllm-gen decode kernel ignores them).
        if (self.is_mla_enable and forward_args.skip_mla_rope_generation
                and not getattr(self, "use_fp8_ds_mla", False)
                and forward_args.attention_input_type
                == AttentionInputType.generation_only):
            num_ctx = metadata.num_contexts
            n_gen = metadata.num_generations
            # Use the GPU-resident length tensors (no host->device copy) so this
            # stays CUDA-graph-capturable.
            gen_q_lens = metadata.seq_lens_cuda[num_ctx:num_ctx + n_gen].to(
                torch.int32)
            gen_kv_lens = metadata.kv_lens_cuda_runtime[num_ctx:num_ctx +
                                                        n_gen].to(torch.int32)
            cu_q = torch.zeros(n_gen + 1, dtype=torch.int32, device=q.device)
            cu_kv = torch.zeros(n_gen + 1, dtype=torch.int32, device=q.device)
            cu_q[1:] = torch.cumsum(gen_q_lens, dim=0).to(
                torch.int32) * self.num_heads
            cu_kv[1:] = torch.cumsum(gen_kv_lens, dim=0).to(torch.int32)
            forward_args.cu_q_seqlens = cu_q
            forward_args.cu_kv_seqlens = cu_kv
            if forward_args.fmha_scheduler_counter is None:
                forward_args.fmha_scheduler_counter = torch.zeros(
                    1, dtype=torch.uint32, device=q.device)
            else:
                forward_args.fmha_scheduler_counter.zero_()
            assert forward_args.latent_cache is not None
            from ...pyexecutor.kv_cache.mamba_cache_manager import \
                BaseMambaCacheManager

            # Hybrid (mamba/masked-layer) KV managers take the graph-safe
            # append; the same predicate interface.py uses to detect hybrid
            # managers for mamba metadata. Dense MLA models keep the
            # host-side path unchanged.
            if isinstance(metadata.kv_cache_manager, BaseMambaCacheManager):
                from .utils import \
                    append_mla_latent_cache_generation_cuda_graph_safe

                # The write positions must come from device tensors: the host
                # lists (request ids, seq lens, cached-token counts) are
                # frozen into the graph at capture time and corrupt the cache
                # on replay. The helper falls back to the host-side loop for
                # eager forwards only; under CUDA graphs it scatters
                # device-side for any uniform q_len (1 for plain decode,
                # 1 + draft_len for padded spec-dec verification batches).
                append_mla_latent_cache_generation_cuda_graph_safe(
                    metadata,
                    # NOTE: get_buffers / layer_offsets take the GLOBAL layer
                    # index (they map through layer_offsets internally).
                    # Passing the local offset double-maps and breaks hybrid
                    # models whose KV manager covers a masked layer subset
                    # (e.g. Kimi K3).
                    self.layer_idx,
                    forward_args.latent_cache,
                )
            else:
                # TODO(TRTLLM-15193): this host-side path freezes write
                # positions at CUDA graph capture time and may corrupt the
                # cache on replay IF a non-hybrid MLA model reaches this path
                # under graph capture. Investigate whether that is reachable;
                # if so, the graph-safe branch above is a correctness fix to
                # generalize, not an optimization.
                from .utils import append_mla_latent_cache
                append_mla_latent_cache(
                    metadata.kv_cache_manager,
                    self.get_local_layer_idx(metadata),
                    metadata.request_ids,
                    metadata.seq_lens.tolist(),
                    metadata.kv_cache_params.num_cached_tokens_per_seq,
                    forward_args.latent_cache,
                    kv_layout=metadata.kv_layout,
                    seq_start=num_ctx,
                )

        forward_args.sparse_runtime_params = prepare_sparse_runtime_params(
            self, q, k, v, metadata, forward_args)

        # Compute FlashMLA tile-scheduler metadata once per forward pass.
        # The flag is invalidated whenever FlashMLA inputs change. The metadata
        # must always match the compacted generation sub-batch, which is also
        # the layout used by block_ids_per_seq.
        flash_mla = metadata.find_ext(FlashMlaMetadata)
        if (flash_mla is not None and forward_args.attention_input_type
                != AttentionInputType.context_only
                and metadata.num_generations > 0):
            flash_mla.compute_scheduler_metadata(self.num_heads,
                                                 self.kv_lora_rank)

        # Blackwell first_sparse: refresh at layer 0 before kernel launch.
        if self.get_local_layer_idx(metadata) == 0 and (
                metadata.spec_bl_tree_first_sparse_mask_offset_kv is not None
                and metadata._seq_lens_cuda is not None):
            metadata.update_blackwell_first_sparse_mask_offset()

        if metadata.is_cross:
            if k is not None and v is not None:
                k_flat = k.contiguous().view(k.shape[0], -1)
                v_flat = v.contiguous().view(v.shape[0], -1)
                forward_args.cross_kv = torch.cat([k_flat, v_flat],
                                                  dim=1).contiguous()

            q_hidden_size = self.num_heads * self.head_dim
            kv_hidden_size = self.num_kv_heads * self.head_dim
            qkv_hidden_size = q_hidden_size + 2 * kv_hidden_size
            if q.shape[1] == q_hidden_size:
                fused_q = q.new_zeros((q.shape[0], qkv_hidden_size))
                fused_q[:, :q_hidden_size].copy_(q)
                q = fused_q
            else:
                assert q.shape[1] == qkv_hidden_size
            k = None
            v = None
            forward_args.is_fused_qkv = True

        attention_input_type = forward_args.attention_input_type
        if not self.is_mla_enable:
            if forward_args.is_fused_qkv:
                qkv_hidden_size = (self.num_heads +
                                   2 * self.num_kv_heads) * self.head_dim
                assert q.shape[1] == qkv_hidden_size
            else:
                q_hidden_size = self.num_heads * self.head_dim
                assert q.shape[1] == q_hidden_size
                if forward_args.update_kv_cache:
                    kv_hidden_size = self.num_kv_heads * self.head_dim
                    assert k.shape[1] == kv_hidden_size
                    assert v.shape[1] == kv_hidden_size
            num_tokens = q.shape[0]
            if k is not None and not metadata.is_cross:
                assert k.shape[0] == num_tokens
                assert v.shape[0] == num_tokens
        else:
            assert not forward_args.is_fused_qkv
            sparse_attn_indices = forward_args.sparse_runtime_params.sparse_attn_indices
            is_sparse_attn = sparse_attn_indices is not None and sparse_attn_indices.numel(
            ) > 0
            if attention_input_type == AttentionInputType.context_only and is_sparse_attn:
                assert k is None and v is None
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            elif attention_input_type == AttentionInputType.context_only:
                assert k is not None and v is not None
                qkv_hidden_size = self.num_heads * (self.qk_nope_head_dim +
                                                    self.qk_rope_head_dim)
            elif attention_input_type == AttentionInputType.generation_only:
                assert k is None and v is None
                qkv_hidden_size = self.num_heads * (self.kv_lora_rank +
                                                    self.qk_rope_head_dim)
            else:
                raise ValueError(
                    "In MLA, TrtllmAttention can only support context_only or generation_only, not mixed."
                )
            assert q.shape[
                1] == qkv_hidden_size, f"q.shape[1] must be equal to qkv_hidden_size, got {q.shape[1]=}, {qkv_hidden_size=}"

        batch_size = metadata.kv_lens_cuda_runtime.shape[0]
        assert metadata.kv_lens_runtime.shape[0] == batch_size
        assert metadata.prompt_lens_cuda_runtime.shape[0] == batch_size
        assert metadata.prompt_lens_cpu_runtime.shape[0] == batch_size
        assert metadata.host_request_types_runtime.shape[0] == batch_size

        self._ensure_rope_table_size(metadata.max_seq_len)

        # Prime `self.local_layer_idx` so FMHA implementations read a
        # populated int rather than the `None` placeholder.
        # The call is a fast cache hit after the first forward.
        self.local_layer_idx = self.get_local_layer_idx(metadata)
        if metadata.spec_decoding_bl_tree_mask is not None and self.local_layer_idx == 0:
            metadata.spec_decoding_bl_tree_mask.zero_()

        if self.print_skip_softmax_stat:
            self.skip_softmax_stat.zero_()

        # Propagate the KV cache manager's SWA window to the FMHA kernel when
        # the model does not specify one, so paged-context attention does not
        # read stale page-table entries (BAD_PAGE_INDEX).
        if forward_args.attention_window_size is None and metadata.kv_cache_manager is not None:
            window = metadata.kv_cache_manager.max_attention_window_vec[
                self.local_layer_idx]
            if window is not None:
                forward_args.attention_window_size = window
        if forward_args.attention_window_size is None:
            forward_args.attention_window_size = metadata.max_seq_len

        # Promote `out_scale_sf` -> `out_scale` for the NVFP4-output path
        # (kernel reads a single `out_scale` and interprets it as the SF
        # quant scale when `output_sf` is allocated). `output_sf` is
        # populated by `create_output` in `forward` above, so the
        # decision is correct only here, not at the attention/attention.py
        # call site where `output_sf` is always `None`.
        if (not forward_args.enable_dsv4_epilogue_fusion
                and forward_args.output_sf is not None
                and forward_args.out_scale_sf is not None):
            forward_args.out_scale = forward_args.out_scale_sf

        # Default `forward_args.kv_scale_*` to the layer-level mirrors when
        # the caller didn't populate them. `attention/attention.py` only sets
        # these for fp4 KV cache; fp8-KV models leave them `None`. Several
        # XQA kernels (mha.cu, mha_sm90.cu) deref `kvCacheScale[0]` when
        # `isKVCacheQuantized` is true and don't check for nullptr, so
        # passing `None` crashes with illegal memory access.
        if forward_args.kv_scale_orig_quant is None:
            forward_args.kv_scale_orig_quant = self.kv_scale_orig_quant
        if forward_args.kv_scale_quant_orig is None:
            forward_args.kv_scale_quant_orig = self.kv_scale_quant_orig

        # max_context_q_len_override is only set when encoder CUDA graphs are enabled.
        if metadata.max_context_q_len_override is not None:
            assert metadata.is_cuda_graph
            assert metadata.num_generations == 0
            assert metadata.kv_cache_manager is None
            assert metadata.num_contexts == metadata.num_seqs

        assert self._fmha_manager is not None
        # Sparse prediction has read the unquantized inputs; FMHA support checks
        # see the quantized ones.
        forward_args.sage_quantized = self._quantize_sage_inputs(
            q, k, v, metadata, forward_args)

        # On a ragged generation step the attention op is handed
        # one row per query token so its `num_tokens % num_seqs == 0` check
        # passes with seq_len == 1. Keep the presentation scoped around both
        # library selection and dispatch: support checks read the
        # same runtime metadata as the selected FMHA implementation.
        with metadata.presented_token_major(token_major_view):
            fmha = self._fmha_manager.select(self, q, k, v, metadata,
                                             forward_args)

            if fmha is None:
                if (getattr(forward_args, "qk_fp8_per_tensor", False)
                        or getattr(forward_args, "v_fp8_per_tensor", False)):
                    raise RuntimeError(
                        "The per-tensor FP8 attention recipe needs the PrimTS "
                        "FMHA library (TLLM_FMHA_LIBS=+prims_ts), which is not "
                        "enabled or does not support this request.")
                raise RuntimeError(
                    "No TRT-LLM attention FMHA library supports this request.")
            if metadata.is_cuda_graph or not fmha.supports_workspace_reclamation:
                # Conservatively disable reclamation for metadata used by graphs
                # or backends that can retain staged workspace state.
                metadata.workspace_reclaimable = False
            try:
                fmha.forward(q, k, v, metadata, forward_args)
            except RuntimeError as exc:
                log_attention_failure_context(
                    type(self).__name__, self.layer_idx, metadata,
                    forward_args.attention_window_size, exc)
                raise

        if self.print_skip_softmax_stat:
            total_blocks, skipped_blocks = self.skip_softmax_stat
            if total_blocks != 0:
                print(
                    f"SKIP_SOFTMAX_STAT: layer{self.layer_idx}: {skipped_blocks} / {total_blocks}"
                    f" = {skipped_blocks / total_blocks * 100: .2f}%")

        if forward_args.output_sf is None:
            return forward_args.output
        else:
            return forward_args.output, forward_args.output_sf

    def _context_plan_sig(self, q: torch.Tensor, fwd: AttentionForwardArgs,
                          metadata: TrtllmAttentionMetadata) -> tuple:
        """Key of a context plan besides the forward state that the metadata invalidates.

        With the layer's KV-pool bound (checked in _run_context_plan) it holds
        every input of the FMHA selection and of the context workspace size
        that _is_plain_context_call leaves variable.
        """
        return (self._context_plan_layer_key, q.dtype, fwd.output.dtype,
                fwd.attention_input_type, fwd.attention_window_size,
                fwd.attention_sinks is None, fwd.out_scale
                is None, metadata.use_paged_context_fmha)

    def _is_plain_context_call(self, q: torch.Tensor, k: Optional[torch.Tensor],
                               v: Optional[torch.Tensor],
                               metadata: TrtllmAttentionMetadata,
                               fwd: AttentionForwardArgs) -> bool:
        """Whether a call is a fused-QKV context-only call without an optional input that a plan does not cover."""
        return (k is None and v is None and fwd.output is not None
                and fwd.output_sf is None and fwd.out_scale_sf is None and
                fwd.attention_input_type != AttentionInputType.generation_only
                and fwd.attention_mask == PredefinedAttentionMask.CAUSAL
                and fwd.attention_mask_data is None
                and fwd.relative_attention_bias is None and fwd.cross_kv is None
                and fwd.latent_cache is None and fwd.q_pe is None
                and fwd.mrope_rotary_cos_sin is None
                and fwd.mrope_position_deltas is None
                and fwd.softmax_stats_tensor is None
                and fwd.quant_q_buffer is None and fwd.quant_scale_qkv is None
                and not fwd.enable_dsv4_epilogue_fusion
                and fwd.kv_norm_weight is None
                and fwd.sage_attn_qk_dtype is None and not fwd.qk_fp8_per_tensor
                and not fwd.v_fp8_per_tensor and fwd.kv_append is None
                and fwd.sparse_backend_args is None
                and not fwd.skip_mla_rope_generation
                and metadata.num_generations == 0 and metadata.num_contexts > 0
                and not metadata.is_cross
                and q.shape[0] == metadata.num_ctx_tokens and q.size(-1)
                == (self.num_heads + 2 * self.num_kv_heads) * self.head_dim)

    def _context_plan_matches(self, plan: Optional[_ContextPlan], sig: tuple,
                              metadata: TrtllmAttentionMetadata) -> bool:
        return (plan is not None and plan.sig == sig
                and plan.kv_lens is metadata.kv_lens_runtime
                and plan.prompt_lens is metadata.prompt_lens_cpu_runtime
                and plan.kv_lens_cuda is metadata.kv_lens_cuda_runtime
                and plan.prompt_lens_cuda is metadata.prompt_lens_cuda_runtime
                and plan.num_contexts == metadata.num_contexts
                and plan.num_ctx_tokens == metadata.num_ctx_tokens
                and plan.block_offsets is metadata.kv_cache_block_offsets
                and plan.workspace is metadata.effective_workspace
                and plan.kv_cache_manager is metadata.kv_cache_manager
                and plan.pool_pointers is metadata.host_kv_cache_pool_pointers)

    def _build_context_plan(self, q: torch.Tensor,
                            metadata: TrtllmAttentionMetadata,
                            fwd: AttentionForwardArgs,
                            sig: tuple) -> _ContextPlan:
        """Build the context plan the way forward() and PhasedFmha.forward() would, selecting for this layer.

        plan.fmha_cls stays None when the plan does not cover the call.
        """
        plan = _ContextPlan()
        plan.kv_lens = metadata.kv_lens_runtime
        plan.prompt_lens = metadata.prompt_lens_cpu_runtime
        plan.kv_lens_cuda = metadata.kv_lens_cuda_runtime
        plan.prompt_lens_cuda = metadata.prompt_lens_cuda_runtime
        plan.num_contexts = metadata.num_contexts
        plan.num_ctx_tokens = metadata.num_ctx_tokens
        plan.block_offsets = metadata.kv_cache_block_offsets
        plan.workspace = metadata.effective_workspace
        plan.kv_cache_manager = metadata.kv_cache_manager
        plan.pool_pointers = metadata.host_kv_cache_pool_pointers
        plan.sig = sig
        plan.fmha_cls = None
        if (metadata.kv_cache_manager is None or plan.block_offsets is None
                or plan.workspace is None or metadata.beam_width != 1
                or metadata.max_context_q_len_override is not None
                or metadata.spec_decoding_bl_tree_mask is not None
                or metadata.spec_bl_tree_first_sparse_mask_offset_kv is not None
                or metadata.helix_position_offsets is not None
                or metadata.find_ext(FlashMlaMetadata) is not None):
            return plan
        from .fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
        fmha = self._fmha_manager.select(self, q, None, None, metadata, fwd)
        if not isinstance(fmha, FlashInferTrtllmGenFmha):
            return plan
        # PhasedFmha.forward, context part (seq_offset 0, every sequence a context request)
        n = plan.num_contexts
        fmha.prepare_workspace(q, None, None, metadata, fwd, plan.workspace)
        plan.counter_rows = (metadata.max_num_sequences
                             or metadata.max_num_requests)
        plan.total_num_blocks = fmha._get_total_num_blocks(metadata)
        plan.max_q_len = int(plan.prompt_lens[0:n].max())
        plan.max_kv_len = int(plan.kv_lens[0:n].max())
        plan.fmha_cls = type(fmha)
        self._context_plan_fmha = fmha
        self._context_plan_fmha_owner = self._fmha_manager
        self._context_plan_counter_rows = plan.counter_rows
        return plan

    def _context_plan_fmha_for(self, plan: _ContextPlan) -> Optional["Fmha"]:
        """This layer's library instance of plan.fmha_cls, prepared for the plan's counter rows."""
        fmha = self._context_plan_fmha
        if (type(fmha) is not plan.fmha_cls
                or self._context_plan_fmha_owner is not self._fmha_manager):
            fmha = next((f for f in self._fmha_manager.fmha_libs
                         if type(f) is plan.fmha_cls), None)
            if fmha is None:
                return None
            self._context_plan_fmha = fmha
            self._context_plan_fmha_owner = self._fmha_manager
            self._context_plan_counter_rows = 0
        return fmha

    def _run_context_plan(self, q: torch.Tensor, k: Optional[torch.Tensor],
                          v: Optional[torch.Tensor],
                          metadata: TrtllmAttentionMetadata,
                          fwd: AttentionForwardArgs) -> bool:
        """Run a context-only call on the forward's plan; False: take the regular path."""
        if (not self._is_plain_context_call(q, k, v, metadata, fwd)
                or metadata.kv_lens_runtime is None
                or metadata.prompt_lens_cpu_runtime is None):
            return False
        # forward(): fused QKV input that also updates the KV cache, rope table,
        # layer index, window and KV scale defaults
        fwd.is_fused_qkv = True
        fwd.update_kv_cache = True
        if fwd.cu_q_seqlens is None:
            fwd.cu_q_seqlens = metadata.cu_q_seqlens
        if fwd.cu_kv_seqlens is None:
            fwd.cu_kv_seqlens = metadata.cu_kv_seqlens
        self._ensure_rope_table_size(metadata.max_seq_len)
        self.local_layer_idx = self.get_local_layer_idx(metadata)
        if (fwd.attention_window_size is None
                and metadata.kv_cache_manager is not None):
            fwd.attention_window_size = (
                metadata.kv_cache_manager.max_attention_window_vec[
                    self.local_layer_idx])
        if fwd.attention_window_size is None:
            fwd.attention_window_size = metadata.max_seq_len
        if fwd.kv_scale_orig_quant is None:
            fwd.kv_scale_orig_quant = self.kv_scale_orig_quant
        if fwd.kv_scale_quant_orig is None:
            fwd.kv_scale_quant_orig = self.kv_scale_quant_orig

        sig = self._context_plan_sig(q, fwd, metadata)
        plan = metadata.context_plan
        if not self._context_plan_matches(plan, sig, metadata):
            plan = self._build_context_plan(q, metadata, fwd, sig)
            metadata.context_plan = plan
        if plan.fmha_cls is None:
            return False
        fmha = self._context_plan_fmha_for(plan)
        if fmha is None:
            return False
        total_num_blocks = fmha._get_total_num_blocks(metadata)
        if total_num_blocks != plan.total_num_blocks:
            # The selection also depends on the KV-pool bound of the layer.
            plan = self._build_context_plan(q, metadata, fwd, sig)
            metadata.context_plan = plan
            if plan.fmha_cls is None:
                return False
            fmha = self._context_plan_fmha_for(plan)
            if fmha is None:
                return False
            total_num_blocks = plan.total_num_blocks
        if self._context_plan_counter_rows != plan.counter_rows:
            fmha.prepare_workspace(q, None, None, metadata, fwd, plan.workspace)
            self._context_plan_counter_rows = plan.counter_rows
        if metadata.is_cuda_graph or not fmha.supports_workspace_reclamation:
            metadata.workspace_reclaimable = False
        params = fmha.context_params(q, None, None, metadata, fwd,
                                     plan.workspace, total_num_blocks,
                                     plan.max_q_len, plan.max_kv_len)
        try:
            fmha.run_context(params)
        except RuntimeError as exc:
            log_attention_failure_context(
                type(self).__name__, self.layer_idx, metadata,
                fwd.attention_window_size, exc)
            raise
        return True

    @classmethod
    def support_fused_rope(cls) -> bool:
        return True

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True

    @classmethod
    def support_mla(cls) -> bool:
        return True

    @classmethod
    def support_fp4_kv_cache(cls) -> bool:
        return True

    def has_cached_kv_for_mla_context(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.num_ctx_cached_tokens > 0)

    def is_chunked_prefill_for_mla_context(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv
                and metadata.num_ctx_cached_tokens > 0
                and metadata.runtime_features.chunked_prefill)

    def has_cached_kv_for_mla_context_warmup(
        self,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        """KV cache reuse / chunked prefill MLA context check for warmup, do not check num_ctx_cached_tokens."""
        return (self.is_mla_enable and metadata.kv_cache_manager is not None
                and metadata.enable_context_mla_with_cached_kv)

    def load_paged_kv_cache_for_mla(
        self,
        metadata: TrtllmAttentionMetadata,
        out_dtype: torch.dtype,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out_dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None
        assert metadata.max_ctx_kv_len > 0
        assert metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens == metadata.host_ctx_kv_indptr[
            metadata.num_contexts]

        beam_width = 1

        compressed_kv, k_pe = torch.ops.trtllm.load_paged_kv_cache_for_mla(
            out_dtype,
            metadata.num_contexts,
            metadata.num_ctx_cached_tokens + metadata.num_ctx_tokens,
            metadata.max_ctx_kv_len,
            metadata.ctx_kv_indptr,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_quant_orig
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )

        return compressed_kv, k_pe

    def load_chunked_kv_cache_for_mla(
        self,
        metadata: TrtllmAttentionMetadata,
        num_ctx_cached_tokens: int,
        cu_chunked_seq_len: torch.Tensor,
        chunked_global_offset: torch.Tensor,
        chunked_max_seq_len: int,
        out_dtype: torch.dtype,
        helix_plan: HelixMlaChunkPlan | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert out_dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        if self.uses_fp4_mla_attention:
            return load_fp4_mla_chunked_kv_cache(
                metadata,
                self.layer_idx,
                num_ctx_cached_tokens=num_ctx_cached_tokens,
                cu_chunked_seq_len=cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=out_dtype,
                kv_lora_rank=self.mla_params.kv_lora_rank,
                qk_rope_head_dim=self.mla_params.qk_rope_head_dim,
            )
        if helix_plan is not None:
            mapping = metadata.kv_cache_manager.mapping
            rank = mapping.cp_rank
            local_tokens = helix_plan.rank_sizes[rank]
            if local_tokens:
                local_kv, local_pe = self.load_chunked_kv_cache_for_mla(
                    metadata, local_tokens, helix_plan.cu_seq_lens[rank],
                    helix_plan.local_offsets[rank],
                    helix_plan.max_local_seq_len, out_dtype)
                local = torch.cat((local_kv, local_pe), dim=-1)
            else:
                local = torch.empty((0, self.mla_params.kv_lora_rank +
                                     self.mla_params.qk_rope_head_dim),
                                    dtype=out_dtype,
                                    device=cu_chunked_seq_len.device)
            # Every rank enters the collective using the same global plan,
            # including ranks with no tokens in this history iteration.
            gathered = cp_allgather(local,
                                    mapping,
                                    dim=0,
                                    sizes=helix_plan.rank_sizes)
            assert isinstance(gathered, torch.Tensor)
            return torch.ops.trtllm.reorder_helix_mla_kv(
                gathered, cu_chunked_seq_len, chunked_global_offset,
                helix_plan.cu_seq_lens, helix_plan.local_offsets,
                helix_plan.rank_offsets,
                metadata.kv_cache_manager.tokens_per_block,
                self.mla_params.kv_lora_rank, self.mla_params.qk_rope_head_dim)

        if metadata.max_ctx_cached_token_len == 0:
            empty_kv = torch.empty((0, self.mla_params.kv_lora_rank),
                                   dtype=out_dtype,
                                   device=cu_chunked_seq_len.device)
            empty_k_pe = torch.empty((0, self.mla_params.qk_rope_head_dim),
                                     dtype=out_dtype,
                                     device=cu_chunked_seq_len.device)
            return empty_kv, empty_k_pe

        beam_width = 1

        output_kv, output_k_pe = torch.ops.trtllm.load_chunked_kv_cache_for_mla(
            out_dtype,
            metadata.num_contexts,
            num_ctx_cached_tokens,
            cu_chunked_seq_len,
            chunked_global_offset,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_quant_orig
            self.get_local_layer_idx(metadata),
            self.mla_params.kv_lora_rank,
            self.mla_params.qk_rope_head_dim,
            metadata.kv_cache_manager.tokens_per_block,
            chunked_max_seq_len,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )
        return output_kv, output_k_pe

    def append_fp4_mla_context_cache(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
    ) -> None:
        """Apply Q/K RoPE and append one context chunk to FP4 MLA V2 storage."""
        if not self.uses_fp4_mla_attention or self.mla_params is None:
            raise RuntimeError(
                "FP4 MLA context append requires dense FP4 MLA attention.")
        fp4_mla_state = metadata.find_ext(Fp4MlaState)
        if fp4_mla_state is None or fp4_mla_state.positions is None:
            raise RuntimeError(
                "FP4 MLA context append requires token positions.")
        self._ensure_rope_table_size(metadata.max_seq_len)
        hp_pool_updated = scatter_fp4_mla_kv_cache(
            metadata,
            latent_cache[:metadata.num_ctx_tokens],
            self.layer_idx,
            token_offset=0,
            phase="context",
            local_layer=self.get_fp4_mla_local_layer_idx(metadata),
            v_head_dim=self.mla_params.kv_lora_rank,
            rotary_cos_sin=self.rotary_cos_sin,
            q_context=q,
            q_nope_head_dim=self.mla_params.qk_nope_head_dim,
        )
        if not hp_pool_updated:
            raise RuntimeError(
                "Fused FP4 MLA context scatter did not update the HP pool.")

    def forward_fp4_mla_context_partition(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
        *,
        kv_lens_cuda: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Run one explicit-KV chunked-prefill partition of dense FP4 MLA.

        The partition attends through the FP8 MLA context kernel. Returns the
        attention output of the partition.
        """
        for fmha in self._fmha_manager.fmha_libs:
            if isinstance(fmha, Fp4MlaFmha):
                output = fmha.forward_context_partition(
                    q,
                    k,
                    v,
                    metadata,
                    forward_args,
                    kv_lens_cuda=kv_lens_cuda,
                    kv_lens_cpu=kv_lens_cpu,
                )
                if isinstance(output, tuple):
                    output = output[0]
                return output
        raise RuntimeError(
            "Dense FP4 MLA FMHA is unavailable for chunked prefill.")

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        **kwargs,
    ) -> None:
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        # Ensure RoPE cos/sin table covers the sequence length before the
        # kernel reads it.
        self._ensure_rope_table_size(metadata.kv_cache_manager.max_seq_len)

        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            metadata.num_contexts,
            metadata.ctx_cached_token_indptr,
            metadata.ctx_kv_indptr,
            metadata.max_ctx_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            metadata.kv_cache_block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            None,  # kv_scale_orig_quant
            0,  # residual_dim
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
            metadata.helix_context_local_slots,
        )

    def merge_attention_for_mla(
        self,
        merged_attn: torch.Tensor,
        temp_attn: torch.Tensor,
        softmax_stats: torch.Tensor,
        temp_softmax_stats: torch.Tensor,
        merge_op: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
    ) -> None:
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        torch.ops.trtllm.merge_chunked_attention_for_mla(
            merged_attn,
            temp_attn,
            softmax_stats,
            temp_softmax_stats,
            metadata.num_contexts,
            metadata.ctx_uncached_token_indptr,  # cu_q_seq_len
            metadata.max_ctx_seq_len,  # max_q_seq_len
            merge_op,
            self.num_heads,
            self.mla_params.v_head_dim,
        )

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Predict sparse KV indices when required by an algorithm."""
        return None, None

    def block_sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Optional[BlockSparseForwardInputs]:
        """Predict the block-sparse routing payload for one attention call.

        The default hands through routes that the attention module predicted
        before the core forward via ``sparse_backend_args``. Algorithms that
        predict inside the backend override this method and return ``None``
        for dense phases.
        """
        backend_args = forward_args.sparse_backend_args
        if backend_args is None:
            return None
        return backend_args.block_sparse_inputs

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Predict sparse attention indices when required by an algorithm."""
        return None, None

    def mla_rope_generation(
        self,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        quant_q_buffer: torch.Tensor,
        out_scale: Optional[torch.Tensor] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
        fuse_fp4_q_quant: bool = False,
        q_rope_applied: bool = False,
        latent_write_target: Optional[MlaLatentWriteTarget] = None,
    ) -> None:
        """
            fused_q (torch.Tensor): The tensor to store the fused q, with shape (num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim) on GPU.
            q_pe (torch.Tensor): The tensor to store the q_pe, with shape (num_tokens, num_heads, qk_rope_head_dim) on GPU.
            latent_cache (torch.Tensor): The tensor to store the latent cache, with shape (num_tokens, kv_lora_rank + qk_rope_head_dim) on GPU.
            cu_q_seqlens (torch.Tensor): The tensor to store the cu_q_seqlens, with shape (num_seqs + 1) on GPU.
            cu_kv_seqlens (torch.Tensor): The tensor to store the cu_kv_seqlens, with shape (num_seqs + 1) on GPU.
            fmha_scheduler_counter (torch.Tensor): The tensor to store the fmha_scheduler_counter, with shape (1) on GPU.
            mla_bmm1_scale (torch.Tensor): The tensor to store the mla_bmm1_scale, with shape (2) on GPU.
            mla_bmm2_scale (torch.Tensor): The tensor to store the mla_bmm2_scale, with shape (1) on GPU.
            quant_q_buffer (torch.Tensor): The tensor to store the quant_q_buffer, with shape (tokens, num_heads, kv_lora_rank + qk_rope_head_dim) on GPU.
            helix_position_offsets (torch.Tensor): The tensor to store the helix position offsets, with shape (num_tokens) on GPU.
            out_scale (torch.Tensor): The tensor to store the out_scale, with shape (1) on GPU.
            kv_norm_weight (torch.Tensor): kv_a_layernorm weight, spanning kv_lora_rank + qk_rope_head_dim. Non-None folds the RMSNorm into the KV kernel, which then reads `latent_cache` RAW, so the caller must NOT have normalized it.
            kv_norm_eps (float): Epsilon for that folded RMSNorm. Ignored when kv_norm_weight is None.
            precomputed_cu_seqlens (bool): cu_q_seqlens/cu_kv_seqlens already hold this step's values, so the kernel skips filling them. Required whenever the Q half is skipped, because the kernel that fills them is the one being dropped.
            precomputed_fmha_scheduler (bool): fmha_scheduler_counter and the two bmm scales are emitted elsewhere (the sparse index kernel), so the kernel skips them. Only valid with an FP8 KV cache and both scale tensors non-None.
            kv_only (bool): Run the KV half only; the Q half already ran (q_b_layernorm folded the Q RoPE). Mutually exclusive with kv_done_elsewhere.
            kv_done_elsewhere (bool): Run the Q half only; the KV half already ran, hoisted onto an aux stream. Mutually exclusive with kv_only. With both halves done, do not call this at all.
            quant_scale_qkv (torch.Tensor): Non-None means q_nope in quant_q_buffer is already FP8, so the kernel drops the q_nope quantize rows from its grid.
            fuse_fp4_q_quant (bool): Quantize Q in the fused FP4 RoPE/cache update.
            q_rope_applied (bool): quant_q_buffer already holds the complete rotated FP8 Q (the DSv4 q_b GEMM fusion wrote it), so the Q half must not rewrite the rope segment.
            latent_write_target (MlaLatentWriteTarget): Store the new latent rows there instead of in the paged cache.
        """

        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        # Ensure RoPE cos/sin table covers the sequence length before the
        # kernel reads it.
        self._ensure_rope_table_size(metadata.max_seq_len)

        if self.uses_fp4_mla_attention:
            if latent_write_target is not None:
                raise NotImplementedError(
                    "FP4 MLA generation cannot store its latent outside the paged cache"
                )
            self._fp4_mla_rope_generation(
                fused_q,
                q_pe,
                latent_cache,
                metadata,
                fuse_q_quant=fuse_fp4_q_quant,
            )
            return

        helix_tensor_params = [
            metadata.helix_position_offsets, metadata.helix_is_inactive_rank
        ]
        # Declared by the Helix extension, which is attached only under Helix.
        helix_spec_tokens_valid = getattr(metadata, "_helix_spec_tokens_valid",
                                          False)
        if latent_write_target is not None:
            if helix_spec_tokens_valid or kv_norm_weight is not None:
                raise NotImplementedError(
                    "A latent write target needs the per-token write slots, which Helix "
                    "speculative verify groups and the fused KV norm do not leave free"
                )
            # Each request's one-token page is the target row; the per-token slot
            # 0 addresses it, and -1 skips the token.
            helix_tensor_params.append(latent_write_target.write_slots)
        elif helix_spec_tokens_valid:
            # Speculative verify groups: per-token KV write slots (-1 = this
            # rank does not own the token's position). The append kernel then
            # gates and addresses per token instead of per sequence.
            helix_tensor_params.append(metadata.helix_local_slots)

        # Same presentation as the attention dispatch above: the
        # rope kernel derives batch_idx = tok / seq_len and the KV write offset
        # from a single scalar seq_len, which only describes a batch where every
        # request contributes the same number of tokens.
        with self.presented_token_major_for(metadata):
            self._mla_rope_generation_impl(
                metadata,
                fused_q,
                q_pe,
                latent_cache,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                mla_bmm1_scale,
                mla_bmm2_scale,
                quant_q_buffer,
                out_scale,
                helix_tensor_params,
                kv_norm_weight,
                kv_norm_eps,
                precomputed_cu_seqlens,
                precomputed_fmha_scheduler,
                kv_only,
                kv_done_elsewhere,
                quant_scale_qkv,
                q_rope_applied,
                latent_write_target=latent_write_target)

    @contextlib.contextmanager
    def presented_token_major_for(
            self, metadata: TrtllmAttentionMetadata) -> Iterator[None]:
        with metadata.presented_token_major(metadata.token_major_view()):
            yield

    def _mla_rope_generation_impl(
        self,
        metadata: TrtllmAttentionMetadata,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: torch.Tensor,
        out_scale: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        kv_norm_weight: Optional[torch.Tensor],
        kv_norm_eps: float,
        precomputed_cu_seqlens: bool,
        precomputed_fmha_scheduler: bool,
        kv_only: bool,
        kv_done_elsewhere: bool,
        quant_scale_qkv: Optional[torch.Tensor],
        q_rope_applied: bool,
        latent_write_target: Optional[MlaLatentWriteTarget] = None,
    ) -> None:
        if latent_write_target is None:
            # Read under the caller's presentation: a ragged verify batch
            # presents a token-major block table, one row per token.
            block_offsets = metadata.kv_cache_block_offsets
            pool_pointers = metadata.kv_cache_manager.kv_cache_pool_pointers
            pool_mapping = metadata.kv_cache_manager.kv_cache_pool_mapping
            layer_idx = self.get_local_layer_idx(metadata)
            tokens_per_block = metadata.kv_cache_manager.tokens_per_block
        else:
            block_offsets = latent_write_target.block_offsets
            pool_pointers = latent_write_target.pool_pointers
            pool_mapping = latent_write_target.pool_mapping
            layer_idx = 0
            tokens_per_block = 1
        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            self.rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            metadata.kv_lens_cuda_runtime,  # sequence_length
            metadata.kv_lens_runtime,  # host_past_key_value_lengths
            metadata.prompt_lens_cpu_runtime,  # host_context_lengths,
            metadata.num_contexts,
            block_offsets,
            pool_pointers,
            pool_mapping,
            None,  # kv_scale_orig_quant
            None,  # kv_scale_quant_orig
            getattr(self, "_nvfp4_mla_kv_scale_orig_quant", None),
            out_scale,
            metadata.block_ids_per_seq,
            helix_tensor_params,
            self.predicted_tokens_per_seq,
            layer_idx,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            getattr(metadata.kv_cache_manager, "mla_kv_cache_residual_dim", 0),
            tokens_per_block,
            metadata.max_seq_len,  # attention_window_size
            metadata.beam_width,
            self.quant_mode,
            self.q_scaling,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.rope_append,
            kv_norm_weight,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
            kv_only,
            kv_done_elsewhere,
            quant_scale_qkv,
            q_rope_applied,
        )

    def _fp4_mla_rope_generation(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        *,
        fuse_q_quant: bool = False,
    ) -> None:
        """Apply fused generation RoPE, Q quantization, and cache update."""
        assert self.kv_lora_rank is not None
        assert self.qk_rope_head_dim is not None

        if q_pe.shape[-1] != self.qk_rope_head_dim:
            raise RuntimeError(
                f"FP4 MLA q_pe last dimension must be {self.qk_rope_head_dim}, "
                f"got {q_pe.shape[-1]}.")
        if latent_cache.shape[-1] != self.kv_lora_rank + self.qk_rope_head_dim:
            raise RuntimeError(
                "FP4 MLA latent_cache last dimension must be "
                f"{self.kv_lora_rank + self.qk_rope_head_dim}, got "
                f"{latent_cache.shape[-1]}.")
        if not fuse_q_quant:
            raise RuntimeError(
                "FP4 MLA generation requires fused Q quantization.")
        if not self.rope_params.duplicate_data:
            raise RuntimeError(
                "FP4 MLA requires RopeParams.duplicate_data=True.")
        if not self.can_fuse_fp4_mla_q_quant(fused_q, q_pe, latent_cache,
                                             metadata):
            raise RuntimeError(
                "FP4 MLA fused Q quantization eligibility changed before "
                "launch.")
        if (self.rotary_cos_sin is None or q_pe.dtype != torch.bfloat16
                or fused_q.dtype != torch.bfloat16
                or latent_cache.dtype != torch.bfloat16
                or self.rotary_cos_sin.dtype != torch.float32):
            raise RuntimeError(
                "FP4 MLA generation requires fused BF16 RoPE/cache update "
                "with an FP32 rotary table.")

        hp_pool_updated = scatter_fp4_mla_kv_cache(
            metadata,
            latent_cache,
            self.layer_idx,
            token_offset=getattr(metadata, "num_ctx_tokens", 0),
            phase="generation",
            local_layer=self.get_fp4_mla_local_layer_idx(metadata),
            v_head_dim=self.kv_lora_rank,
            rotary_cos_sin=self.rotary_cos_sin,
            q_pe=q_pe,
            q_rope_out=fused_q[..., self.kv_lora_rank:],
            q_quant_input=fused_q,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
        )
        if not hp_pool_updated:
            raise RuntimeError(
                "Fused FP4 MLA RoPE/cache scatter did not update the HP pool.")

    def can_fuse_fp4_mla_q_quant(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
    ) -> bool:
        return bool(
            self.uses_fp4_mla_attention
            and can_fuse_fp4_mla_q_quant(metadata, fused_q, q_pe, latent_cache))
