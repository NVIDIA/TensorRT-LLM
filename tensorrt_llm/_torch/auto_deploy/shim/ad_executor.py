# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from collections import abc, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from strenum import StrEnum
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor._util import get_decoding_mode
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    AttentionTypeCpp,
    create_kv_cache_transceiver,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import BaseMambaCacheManager
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine, PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.py_executor_creator import get_guided_decoding_config
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler, TRTLLMSampler
from tensorrt_llm._torch.pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    RequestList,
    ScheduledRequests,
    SimpleScheduler,
)
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSampler
from tensorrt_llm._utils import get_free_port, mpi_rank, mpi_world_size, nvtx_range
from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData, check_mm_embed_cumsum_if_needed
from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy, MultimodalConfig, SamplerType
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from tensorrt_llm.mapping import Mapping

from ..custom_ops.attention_interface import AttentionType
from ..distributed.common import initialize_or_skip
from ..llm_args import LlmArgs
from ..transform.optimizer import InferenceOptimizer
from ..utils.cuda_graph import BypassCapturedGraphs
from ..utils.dist_config import DistConfig
from ..utils.logger import ad_logger
from .interface import CachedSequenceInterface, GetInferenceModel

_ATTENTION_TYPE_TO_CPP = {
    AttentionType.mha: AttentionTypeCpp.DEFAULT,
    AttentionType.mla: AttentionTypeCpp.MLA,
}

# Non-model multimodal metadata consumed before the exported graph or ignored by AD.
# These keys must NOT leak into the generic extra_args dict — entries there
# are expected to be tensors, and these may be scalars, lists, or nested dicts.
_RESERVED_MM_DATA_KEYS = frozenset(
    {
        "layout_metadata",
        "mm_bidirectional_blocks",
        "multimodal_embedding",
        "multimodal_embedding_lengths",
        "special_token_offsets",
        "multimodal_embed_mask_cumsum",
    }
)


@dataclass
class ReportingInfo:
    print_log: bool = False
    enable_iter_perf_stats: bool = False
    enable_iter_req_stats: bool = False


def _round_up_to_closest(batch_sizes: List[int], bs: int) -> Optional[int]:
    """Return closest batch size larger or equal to bs."""
    if bs > max(batch_sizes, default=0):
        return None
    return min(batch_sizes, key=lambda x: (x < bs, abs(x - bs)), default=None)


def _generate_dummy_request(
    resource_manager: ResourceManager, request_id: int, **request_kwargs
) -> Optional[LlmRequest]:
    # get resource managers we want
    kv_cache_manager: KVCacheManager = resource_manager.get_resource_manager(
        ResourceManagerType.KV_CACHE_MANAGER
    )
    slot_manager: SeqSlotManager = resource_manager.get_resource_manager(
        ResourceManagerType.SEQ_SLOT_MANAGER
    )

    # check if it's a hybrid kv-cache manager
    is_hybrid_cache = isinstance(kv_cache_manager, BaseMambaCacheManager)

    # check if we have a free page and free state available
    if not kv_cache_manager.get_num_free_blocks():
        return None
    if is_hybrid_cache and not kv_cache_manager.mamba_cache_free_blocks:
        return None

    # generate a dummy request
    dummy_request = kv_cache_manager.add_dummy_requests([request_id], **request_kwargs)[0]
    dummy_request.is_cuda_graph_dummy = True

    # NOTE: hack to avoid blocking a slot for the dummy request
    dummy_request.py_seq_slot = slot_manager.get_max_resource_count()
    dummy_request.seq_slot = dummy_request.py_seq_slot

    return dummy_request


def maybe_pad_for_cuda_graph(func):
    def wrapper(
        self: "ADEngine",
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        *args,
        **kwargs,
    ):
        def _call_func():
            return func(self, scheduled_requests, resource_manager, *args, **kwargs)

        def _call_func_eager():
            # Force captured-graph wrappers to short-circuit to eager (see
            # BypassCapturedGraphs): a rank whose shapes match a captured graph would
            # otherwise replay with stale capture-time per-rank scalar args.
            with BypassCapturedGraphs():
                return _call_func()

        # Pick the cudagraph-fallback path: only attention-DP mixed mode needs the
        # eager bypass (stale per-rank scalar args). Without attention-DP all ranks
        # share shapes, so the plain eager call keeps piecewise cudagraph for prefill
        # — bypassing would force every prefill eager (a needless low-concurrency hit).
        _fallback = (
            _call_func_eager
            if (self.enable_attention_dp and self.dist_config.tp_size > 1)
            else _call_func
        )

        # check conditions for current rank
        can_run_cuda_graph = self.cuda_graph_used and scheduled_requests.can_run_cuda_graph
        batch_size = scheduled_requests.batch_size

        # generate a persistent dummy request right away to ensure we can reserve the necessary
        # resources (kv page and state) the first time we can actually run cuda graph according to
        # this rank
        if can_run_cuda_graph and self.padding_dummy_request is None:
            self.padding_dummy_request = _generate_dummy_request(
                resource_manager,
                request_id=CUDA_GRAPH_DUMMY_REQUEST_ID,
                is_gen=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
                use_mrope=False,
                max_beam_width=self.max_beam_width,
            )

        # check if we can pad the batch based on the availability of the dummy request
        can_pad = self.padding_dummy_request is not None

        # in attention DP mode, we check all ranks
        if self.enable_attention_dp and self.dist_config.tp_size > 1:
            assert self.dist is not None, "Distributed object is required for attention DP mode"
            all_rank_info = self.dist.tp_allgather([can_run_cuda_graph, can_pad, batch_size])
        else:
            all_rank_info = [[can_run_cuda_graph, can_pad, batch_size]]

        # now let's check if we can in principle run cuda graph across all ranks
        can_run_cuda_graph_all = all(r_info[0] for r_info in all_rank_info)

        if not can_run_cuda_graph_all:
            return _fallback()

        # get closest cudagraph batch size based on max_batch_size across ALL ranks
        # NOTE: we assume uniform cudagraph batch sizes across all ranks ensuring all ranks get the
        # same closest cudagraph batch size here based on the max batch size across all ranks
        max_batch_size = max(r_info[2] for r_info in all_rank_info)
        cg_batch_size = _round_up_to_closest(self.cuda_graph_batch_sizes, max_batch_size)

        if cg_batch_size is None:
            return _fallback()

        # let's check if all ranks can pad the batch if they need to
        can_pad_all = all(r_info[1] or (r_info[2] == cg_batch_size) for r_info in all_rank_info)

        # fall back if we cannot run cudagraph due to padding issues
        if not can_pad_all:
            return _fallback()

        # check actual amount of padding needed
        num_padding = cg_batch_size - batch_size

        # we should only hit this point for either of these conditions
        assert num_padding == 0 or (num_padding > 0 and self.padding_dummy_request is not None), (
            "Padding should not be needed or available at this point"
        )

        # no padding needed on current rank
        if num_padding == 0:
            return _call_func()

        # pad the scheduled requests with the dummy request
        scheduled_requests.generation_requests.extend([self.padding_dummy_request] * num_padding)

        ret = _call_func()

        # truncate requests to remove the dummy requests we added
        scheduled_requests.generation_requests = scheduled_requests.generation_requests[
            :-num_padding
        ]

        return ret

    return wrapper


def _compute_window_local_view(
    all_indices: Sequence[int],
    front_removed: int,
    end_compute_i: int,
    group_window: int,
    tokens_per_block: int,
) -> Tuple[List[int], int, int, int]:
    """Compute the window-coherent metadata view for one (request, window) pair.

    The C++ KVCacheManager allocates blocks linearly during prefill and only
    front-evicts during generation (kvCacheManager.cpp::adjustBlocksIfNeeded
    is invoked from addToken, not addSequenceBatch).  Two regimes follow:

      * Pre-eviction (typically a single long prefill that has not yet been
        decoded past the window): ``front_removed == 0`` and the live page
        list covers the full sequence.  The kernel needs every live page and
        the unclamped local cache length; the sliding-window mask is applied
        inside the kernel.

      * Post-eviction (decode/extend has advanced past the window): the C++
        side has bumped ``mNumFrontBlocksRemovedPerWindow`` without popping
        ``mCacheBlockIds``, so the head of the page list is stale.  After
        slicing those entries off, the live cache length in window-local
        coords is ``end_compute_i - front_removed * tokens_per_block`` —
        bounded above by ``group_window + tokens_per_block - 1`` by the
        ``while (live >= window + page_size) detachFrontBlock`` loop.

    Both regimes collapse to a single rule: trust the C++ accounting and
    derive the local cache length from ``end_compute_i`` and ``front_removed``
    rather than artificially clamping to ``group_window``.  Disagreement
    between the Python and C++ sides (e.g. a caller passing a stale
    ``end_compute_i`` against a fresher ``front_removed``) is asserted on
    rather than silently masked.

    Args:
        all_indices: Full historical page list from ``get_cache_indices``,
            including stale front entries when eviction has fired.
        front_removed: Count of stale front entries, from
            ``get_num_front_blocks_removed``.
        end_compute_i: Global token position after this step's tokens are
            processed (``input_pos + seq_len``).
        group_window: Sliding-window size for this pool.  Currently only used
            for the defensive sanity assertion below; the window mask itself
            is applied inside the kernel via the ``sliding_window`` constant.
        tokens_per_block: Cache page size.

    Returns:
        active_indices: the live slice of all_indices to hand the kernel.
        extra_page: the next page after the live slice (used by the overlap
            scheduler's deferred-page insertion), or -1 when there is no
            slot past the live region.
        seq_len_with_cache: window-local cache length the kernel should see.
        last_page_len: derived from seq_len_with_cache and tokens_per_block.
    """
    live_len = len(all_indices) - front_removed
    # Window-local cache length: how many tokens the live pages actually hold.
    # Pre-eviction: end_compute_i (the full prefill so far).
    # Post-eviction: end_compute_i - front_removed * page_size (capped above by
    # window + page_size - 1 thanks to the C++ eviction loop).
    local_cache_len = end_compute_i - front_removed * tokens_per_block
    # Live cache length must fit inside the live pages.  The C++ KVCacheManager
    # keeps the two in sync (it bumps front_removed exactly as it frees blocks,
    # and allocates new ones for incoming tokens), so a violation means the
    # Python side and C++ side disagree on either end_compute_i or
    # front_removed -- fail loudly rather than silently clamp.
    live_capacity = live_len * tokens_per_block
    assert local_cache_len <= live_capacity, (
        f"window-local cache length {local_cache_len} exceeds live page capacity "
        f"{live_capacity} (live_len={live_len}, end_compute_i={end_compute_i}, "
        f"front_removed={front_removed}, tokens_per_block={tokens_per_block}). "
        f"C++ KVCacheManager accounting is expected to keep these in sync."
    )
    active_token_count = local_cache_len
    # Sanity: live cache should never exceed window + one spillover page once
    # front-eviction has fired (the C++ ``while (live >= window + page_size)
    # detachFrontBlock`` loop guarantees this).  Pre-eviction (front_removed=0,
    # e.g. a single long prefill) may legitimately exceed the window because
    # blocks are allocated linearly for the full prompt and only detached
    # during generation -- so we skip the check there.
    assert active_token_count <= group_window + tokens_per_block or front_removed == 0, (
        f"window-local cache length {active_token_count} exceeds "
        f"window {group_window} + page_size {tokens_per_block} "
        f"(end_compute_i={end_compute_i}, front_removed={front_removed})"
    )
    # Pages needed to address active_token_count tokens; equals live_len in
    # the common case (post-eviction live = window + spillover, pre-eviction
    # live = ceil(end_compute_i / page_size)).
    pages_for_active = (active_token_count + tokens_per_block - 1) // tokens_per_block
    num_active = min(live_len, pages_for_active)
    active_indices = list(all_indices[front_removed : front_removed + num_active])
    extra_slot_idx = front_removed + num_active
    extra_page = all_indices[extra_slot_idx] if len(all_indices) > extra_slot_idx else -1
    if active_token_count > 0:
        last_page_len = (active_token_count - 1) % tokens_per_block + 1
    else:
        last_page_len = 0
    return active_indices, extra_page, active_token_count, last_page_len


def _compute_cyclic_full_view(
    all_indices: Sequence[int],
    end_compute_i: int,
    tokens_per_block: int,
) -> Tuple[List[int], int, int, int]:
    """Compute the metadata view for a cyclic-SWA kernel (trtllm).

    Unlike ``_compute_window_local_view`` (which slices the block table down to
    the live sliding window for kernels that cannot cyclic-index), the trtllm
    ``thop.attention`` kernel applies the sliding-window mask itself by wrapping
    KV reads modulo the attention window. It therefore needs:

      * the FULL per-window block table (``all_indices`` verbatim, including any
        stale front-evicted entries -- the kernel's modulo indexing skips them),
        and
      * the GLOBAL (un-window-capped) KV length ``end_compute_i``.

    This mirrors the PyTorch backend, which copies the manager's full block list
    from index 0 and passes ``host_past_key_value_lengths == total KV length``.

    Returns the same 4-tuple shape as ``_compute_window_local_view``:
    ``(active_indices, extra_page, seq_len_with_cache, last_page_len)``.
    ``extra_page`` is always -1: the full table already contains the next page,
    so the overlap scheduler needs no deferred-page insertion.
    """
    active_indices = list(all_indices)
    seq_len_with_cache = end_compute_i
    if seq_len_with_cache > 0:
        last_page_len = (seq_len_with_cache - 1) % tokens_per_block + 1
    else:
        last_page_len = 0
    return active_indices, -1, seq_len_with_cache, last_page_len


class ADEngine(ModelEngine):
    """The AutoDeploy Engine (ADEngine) is the main engine interface to execute AutoDeploy models.

    It follows the `ModelEngine` abstractions and is responsible for building the ad-optimized
    model, converting TRT-LLM scheduled requests into ad-native (pytorch-native) inputs, running
    the model, and returning correctly formatted logits.
    """

    @property
    def _device(self) -> DeviceLikeType:
        return self.cache_seq_interface.device

    @classmethod
    def build_from_config(
        cls,
        ad_config: LlmArgs,
        dist_config: Optional[DistConfig] = None,
        # deprecation: Mapping will soon be replaced entirely by DistConfig
        mapping: Optional[Mapping] = None,
        dist: Optional[Distributed] = None,
    ):
        """Build the ADEngine using the LlmArgs that gets passed through from the LLM."""

        # update device to contain the current default device if it's in cuda
        device = torch.device(ad_config.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        device = str(device)

        factory = ad_config.create_factory()

        # Initialize CachedSequenceInterface - it will create SequenceInfo internally
        # using tokens_per_block from kv_cache_config
        cache_seq_interface = CachedSequenceInterface(
            max_seq_len=ad_config.max_seq_len,
            max_batch_size=ad_config.max_batch_size,
            device=device,
            kv_cache_config=ad_config.kv_cache_config,
            max_num_tokens=ad_config.max_num_tokens,
            vocab_size_padded=factory.vocab_size_padded,
            spec_config=ad_config.speculative_config,
            requires_uniform_kv_caches=ad_config.requires_uniform_kv_caches,
            reject_unmanaged_persistent_caches=ad_config.reject_unmanaged_persistent_caches,
        )

        reporting_info = ReportingInfo(
            print_log=False,
            enable_iter_perf_stats=ad_config.enable_iter_perf_stats,
            enable_iter_req_stats=ad_config.enable_iter_req_stats,
        )

        build_and_optimize = InferenceOptimizer(
            factory=factory, config=ad_config.transforms, dist_config=dist_config
        )

        # construct engine
        return cls(
            build_and_optimize,
            cache_seq_interface,
            ad_config=ad_config,
            dist_config=dist_config,
            mapping=mapping,
            dist=dist,
            reporting_info=reporting_info,
        )

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        cache_seq_interface: CachedSequenceInterface,
        ad_config: Optional[LlmArgs] = None,
        dist_config: Optional[DistConfig] = None,
        mapping: Optional[Mapping] = None,
        dist: Optional[Distributed] = None,
        reporting_info: ReportingInfo = ReportingInfo(),
    ) -> None:
        """Initialize the engine with model and CachedSequenceInterface.

        Args:
            get_inference_model: Callable that builds the inference model.
            cache_seq_interface: The CachedSequenceInterface containing sequence and cache config.
            ad_config: Optional LLM configuration.
            dist_config: DistConfig (single source of truth for distributed config within AD).
            mapping: Mapping for external TRT-LLM APIs (KV cache, sampler, etc.).
            reporting_info: Reporting configuration for logging.
        """
        # NOTE (lucaslie): create a fake Namespace to satisfy PyExecutor requirements...
        # This is not correctly declared in the base ModelEngine class though...
        self.llm_args = SimpleNamespace()
        self.llm_args.print_iter_log = reporting_info.print_log
        self.llm_args.enable_iter_perf_stats = reporting_info.enable_iter_perf_stats
        self.llm_args.enable_iter_req_stats = reporting_info.enable_iter_req_stats
        self.llm_args.max_num_tokens = cache_seq_interface.info.max_num_tokens
        self.llm_args.max_seq_len = cache_seq_interface.info.max_seq_len
        # AutoDeploy does not support the sleep/wakeup feature.
        self.llm_args.sleep_config = None
        self.llm_args.multimodal_config = (
            ad_config.multimodal_config if ad_config else MultimodalConfig()
        )
        self.iter_counter = 0
        self.iter_states = {}

        # NOTE (lucaslie): not a declared base member in the base class; required by PyExecutor...
        self.enable_attention_dp = dist_config.enable_attention_dp if dist_config else False

        if ad_config is not None:
            self.llm_args.stream_interval = ad_config.stream_interval
            self.llm_args.attention_dp_config = ad_config.attention_dp_config
            self.llm_args.batch_wait_timeout_ms = ad_config.batch_wait_timeout_ms
            self.llm_args.batch_wait_timeout_iters = ad_config.batch_wait_timeout_iters
            self.llm_args.batch_wait_max_tokens_ratio = ad_config.batch_wait_max_tokens_ratio
            self.max_beam_width = ad_config.max_beam_width
            self.spec_config = ad_config.speculative_config
            self._disable_overlap_scheduler = ad_config.disable_overlap_scheduler
            cache_transceiver_config = ad_config.cache_transceiver_config
            self._cache_transceiver_enabled = (
                cache_transceiver_config is not None
                and cache_transceiver_config.backend is not None
            )
            self.llm_args.max_stats_len = ad_config.max_stats_len
            self._enable_chunked_prefill = getattr(ad_config, "enable_chunked_prefill", False)
        else:
            self.llm_args.stream_interval = 1
            self.llm_args.attention_dp_config = None
            self.llm_args.batch_wait_timeout_ms = 0
            self.llm_args.batch_wait_timeout_iters = 0
            self.llm_args.batch_wait_max_tokens_ratio = 0.0
            self.max_beam_width = 1
            self.spec_config = None
            self._disable_overlap_scheduler = False
            self._cache_transceiver_enabled = False
            self.llm_args.max_stats_len = 1000
            self._enable_chunked_prefill = False

        # check for max total draft tokens
        if self.spec_config is not None:
            self.max_total_draft_tokens = self.spec_config.tokens_per_gen_step - 1
        else:
            self.max_total_draft_tokens = 0

        # ADEngine skips PyTorchModelEngine.__init__, so set the spec-decode
        # flags that shared PyExecutor code expects on a ModelEngine.
        self.is_spec_decode = self.spec_config is not None
        self.enable_spec_decode = self.is_spec_decode

        # For compatibility with PyTorchModelEngine utilities
        self.batch_size = cache_seq_interface.info.max_batch_size

        # Store the cache sequence interface
        self.cache_seq_interface = cache_seq_interface

        # build model
        self.model = get_inference_model(self.cache_seq_interface)
        # start fresh with fixed seed
        torch.manual_seed(42)

        # check cuda graph padding...
        # TODO: better mechanism to retrieve this information when we refactor LlmArgs
        if ad_config is None:
            self.cuda_graph_used = False
            self.cuda_graph_batch_sizes = []
        else:
            self.cuda_graph_used = ad_config.is_cuda_graph_enabled()
            cg_config = ad_config.cuda_graph_config
            self.cuda_graph_batch_sizes = (
                cg_config.batch_sizes if cg_config is not None and cg_config.batch_sizes else []
            )

        # keep a reference for one dummy request around
        self.padding_dummy_request: Optional[LlmRequest] = None

        # Reuse _execute_logit_post_processors from PyTorchModelEngine
        self.dist_config = dist_config
        self.mapping = mapping
        self.dist = dist
        self._execute_logit_post_processors = types.MethodType(
            PyTorchModelEngine._execute_logit_post_processors, self
        )

    def _release_cuda_graphs(self) -> None:
        def _reset_cuda_graph(graph: object) -> None:
            if isinstance(graph, torch.cuda.CUDAGraph):
                graph.reset()

        model = getattr(self, "model", None)
        if model is not None:
            for module in model.modules():
                if module.__class__.__name__ == "CapturedGraph":
                    cudagraphs = getattr(module, "cudagraphs", None)
                    if isinstance(cudagraphs, dict):
                        for graph in list(cudagraphs.values()):
                            _reset_cuda_graph(graph)
                        cudagraphs.clear()
                    module._cuda_graph_mem_pool = None
                    module._input_buffers = []
                    module._out_buffer_flat = None

                if module.__class__.__name__ == "PiecewiseCapturedGraph":
                    static_input_buffers = getattr(module, "_static_input_buffers", None)
                    if isinstance(static_input_buffers, dict):
                        static_input_buffers.clear()

                if module.__class__.__name__ == "ADPiecewiseRunner":
                    entries = getattr(module, "entries", None)
                    if isinstance(entries, dict):
                        for entry in entries.values():
                            _reset_cuda_graph(getattr(entry, "cuda_graph", None))
                        entries.clear()
                    module._graph_pool = None

        torch.cuda.empty_cache()

    def _store_prefill_multimodal_metadata(
        self,
        ordered_requests: RequestList,
        input_pos: List[int],
        cu_seqlen: List[int],
        num_prefill_seqs: int,
        extra_args: Dict[str, List[torch.Tensor]],
    ) -> None:
        """Stage per-request multimodal layout metadata needed by the VLM wrapper.

        The standard attention metadata captures request geometry (batch_info, cu_seqlen,
        seq_len_with_cache, input_pos), but not the original per-request multimodal span layout
        or multimodal special-token offsets needed to rebuild chunk-aware mRoPE positions.

        This tensorization happens in the executor because it still has the scheduled request
        list, per-request chunk boundaries, input_pos, cu_seqlen, and access to the raw
        request-side multimodal fields (multimodal_positions, multimodal_lengths, and
        py_multimodal_data["layout_metadata"]). By the time control reaches the exported
        wrapper/modeling path, Python request objects are gone and the model call can only
        consume concrete tensor kwargs.
        """
        if num_prefill_seqs == 0:
            return

        prefill_requests = ordered_requests[:num_prefill_seqs]
        if not any(getattr(req, "multimodal_positions", None) for req in prefill_requests):
            return

        mm_item_cu_seqlen: List[int] = [0]
        mm_item_types_flat: List[int] = []
        mm_token_positions_flat: List[int] = []
        mm_token_lengths_flat: List[int] = []
        mm_special_offsets_cu_seqlen: List[int] = [0]
        mm_special_offsets_flat: List[int] = []
        flat_start_list: List[int] = []
        count_list: List[int] = []
        cumsum_total_mm = 0

        for i, req in enumerate(prefill_requests):
            begin_compute = input_pos[i]
            end_compute = begin_compute + (cu_seqlen[i + 1] - cu_seqlen[i])
            mm_pos = getattr(req, "multimodal_positions", None)
            mm_len = getattr(req, "multimodal_lengths", None)
            layout_metadata = {}
            if req.py_multimodal_data:
                layout_metadata = req.py_multimodal_data.get(
                    "layout_metadata", req.py_multimodal_data
                )
            mm_item_types = layout_metadata.get("item_types", []) if layout_metadata else []
            mm_pos_list = list(mm_pos) if mm_pos is not None else []
            mm_len_list = list(mm_len) if mm_len is not None else []
            mm_item_types_list = list(mm_item_types)
            if len(mm_pos_list) != len(mm_len_list):
                raise ValueError(
                    "Mismatch between multimodal_positions and multimodal_lengths in "
                    f"request {i}: positions={len(mm_pos_list)}, lengths={len(mm_len_list)}"
                )
            if mm_item_types_list and len(mm_item_types_list) != len(mm_pos_list):
                raise ValueError(
                    "Mismatch between multimodal item_types and multimodal span arrays in "
                    f"request {i}: item_types={len(mm_item_types_list)}, "
                    f"positions={len(mm_pos_list)}, lengths={len(mm_len_list)}"
                )
            mm_item_cu_seqlen.append(mm_item_cu_seqlen[-1] + len(mm_pos_list))
            mm_item_types_flat.extend(mm_item_types_list)
            mm_token_positions_flat.extend(mm_pos_list)
            mm_token_lengths_flat.extend(mm_len_list)

            mm_data = req.py_multimodal_data or {}
            flat_cumsum = mm_data.get("multimodal_embed_mask_cumsum")
            # special_token_offsets indices into the dense MM-token-list (which includes both embeds and specials).
            # It does not index into the prompt-position-indexed cumsum.
            special_offsets = list(
                mm_data.get("special_token_offsets")
                or (layout_metadata or {}).get("special_token_offsets", [])
            )
            mm_special_offsets_cu_seqlen.append(
                mm_special_offsets_cu_seqlen[-1] + len(special_offsets)
            )
            mm_special_offsets_flat.extend(special_offsets)

            if not mm_pos or not mm_len:
                flat_start_list.append(0)
                count_list.append(0)
                continue

            all_prompt_tokens = req.get_tokens(0)
            check_mm_embed_cumsum_if_needed(
                req.py_multimodal_data,
                begin_compute=begin_compute,
                end_compute=end_compute,
                prompt_len=len(all_prompt_tokens),
            )
            if flat_cumsum is None:
                # Leave flat_start / count at 0 so concatenated cursors don't advance.
                flat_start_list.append(0)
                count_list.append(0)
                continue

            assert flat_cumsum.numel() == len(all_prompt_tokens), (
                f"embed_mask_cumsum length {flat_cumsum.numel()} != prompt length "
                f"{len(all_prompt_tokens)} for request {i}"
            )
            runtime = MultimodalRuntimeData(
                past_seen_token_num=begin_compute,
                chunk_end_pos=end_compute,
                embed_mask_cumsum=flat_cumsum,
            )
            flat_start_list.append(cumsum_total_mm + runtime.num_cached_mm_tokens)
            count_list.append(runtime.num_mm_tokens_in_chunk)
            cumsum_total_mm += runtime.total_embeds_in_request

        extra_args["mm_item_cu_seqlen"] = [
            torch.tensor(mm_item_cu_seqlen, dtype=torch.int32, device="cpu")
        ]
        extra_args["mm_item_types"] = [
            torch.tensor(mm_item_types_flat, dtype=torch.int32, device="cpu")
        ]
        extra_args["mm_token_positions"] = [
            torch.tensor(mm_token_positions_flat, dtype=torch.int32, device="cpu")
        ]
        extra_args["mm_token_lengths"] = [
            torch.tensor(mm_token_lengths_flat, dtype=torch.int32, device="cpu")
        ]
        extra_args["mm_special_offsets_cu_seqlen"] = [
            torch.tensor(mm_special_offsets_cu_seqlen, dtype=torch.int32, device="cpu")
        ]
        extra_args["mm_special_offsets"] = [
            torch.tensor(mm_special_offsets_flat, dtype=torch.int32, device="cpu")
        ]
        # Export multimodal slice bounds whenever the current prefill step only needs a
        # subset of the request's multimodal embeddings. This is required for regular
        # chunked prefill, but also for KV-cache reuse where begin_compute > 0 even when
        # chunked prefill is disabled in the config.
        needs_mm_chunk_bounds = self._enable_chunked_prefill or any(
            int(input_pos[i]) > 0 and getattr(req, "multimodal_positions", None)
            for i, req in enumerate(prefill_requests)
        )
        if needs_mm_chunk_bounds:
            extra_args["mm_chunk_flat_start"] = [
                torch.tensor(flat_start_list, dtype=torch.int64, device="cpu")
            ]
            extra_args["mm_chunk_count"] = [
                torch.tensor(count_list, dtype=torch.int64, device="cpu")
            ]

    @nvtx_range("ad_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tokens: Optional[torch.Tensor] = None,
        new_tokens_lens: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
    ) -> None:
        """Prepare inputs for AD Model from scheduled requests."""
        context_requests = scheduled_requests.context_requests
        if (
            context_requests
            and self._cache_transceiver_enabled
            and not self._disable_overlap_scheduler
        ):
            raise RuntimeError(
                "AutoDeploy disaggregated context workers do not support overlap scheduling. "
                "Set disable_overlap_scheduler=True, or use "
                "examples/auto_deploy/model_registry/configs/disagg_ctx.yaml when starting "
                "a context worker with cache_transceiver_config."
            )

        # cache manager
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        extend_requests = [
            r for r in scheduled_requests.generation_requests if get_draft_token_length(r) > 0
        ]
        generation_requests = [
            r for r in scheduled_requests.generation_requests if get_draft_token_length(r) == 0
        ]

        # This sanity checks our one-model spec dec flow, where extend requests are only present when doing spec dec
        # and take the place of generation requests.
        # This also makes sure that dummy requests, which are at the end of schedulred_requests.generation_requests,
        # end up at the end of gen_requests.
        assert len(extend_requests) == 0 or len(generation_requests) == 0

        gen_requests = extend_requests + generation_requests
        # Requests in order of context, extend, generation.
        ordered_requests = context_requests + gen_requests

        # sequence information
        input_ids: List[int] = []
        cu_seqlen: List[int] = [0]
        input_pos: List[int] = []

        # check new_tokens and setup overlap scheduler metadata
        # new_tokens.shape == [1+max_draft_len, max_batch_size, 1]
        has_new_tokens = new_tokens is not None
        if has_new_tokens:
            assert new_tokens.shape[2] == 1, f"{new_tokens.shape=} not supported in AD."
            nt_batch_size = new_tokens.shape[1]
        new_tokens_flat = new_tokens.flatten() if has_new_tokens else None

        # gather indices are used to gather tokens in new_tokens into input_ids
        slot_gather_indices = [] if has_new_tokens else None
        flat_gather_indices: List[int] = [] if has_new_tokens else None
        mask_scatter_indices: List[int] = [] if has_new_tokens else None
        extra_args: Dict[str, List[torch.Tensor]] = defaultdict(list)
        prompt_lens: List[int] = []
        dummy_token = -1
        has_context_multimodal_data = False

        # look at context requests first
        for request in context_requests:
            # store input ids and pos of first token in sequence
            # NOTE: begin_compute > 0 indicates block reuse
            # NOTE: end_compute is used for chunked prefill
            all_prompt_tokens = request.get_tokens(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]

            input_ids.extend(prompt_tokens)
            cu_seqlen.append(len(input_ids))
            input_pos.append(begin_compute)

            # For prefill requests, prompt_lens stores the context chunk size.
            # This is expected by the TRTLLM Attention backend where the prompt length
            # is used.
            prompt_lens.append(request.context_chunk_size)

            # store extra arguments
            if request.py_multimodal_data is not None:
                has_context_multimodal_data = True
                for k, v in request.py_multimodal_data.items():
                    if k in _RESERVED_MM_DATA_KEYS:
                        continue
                    extra_args[k].append(v)

        # store num_prefill and num_prefill_tokens
        num_prefill = len(context_requests)
        num_prefill_tokens = len(input_ids)

        for request in gen_requests:
            # Use overlap only for non-dummy requests with a previous batch slot.
            # Dummy requests do not need sampled tokens from the previous iteration.
            # First-step disagg decode requests have not appeared in a previous batch yet,
            # so their py_batch_idx is None.
            is_overlap = (
                not self._disable_overlap_scheduler
                and not request.is_dummy
                and request.py_batch_idx is not None
            )

            # check draft length
            draft_len = get_draft_token_length(request)

            # there are cases:
            # 1. No overlap: we are preparing for the current iteration --> use previous token count
            # 2. Overlap: we are preparing for the next iteration -->
            #    use max_beam_num_tokens; the overlap scheduler offset (new_tokens_lens - 1)
            #    applied in offset_with_new_lens_ accounts for not-yet-committed tokens.
            num_tokens_seen = request.max_beam_num_tokens
            if not is_overlap:
                num_tokens_seen -= 1

            # build input ids
            if is_overlap:
                input_ids.extend([dummy_token] * (1 + draft_len))
                start = request.py_batch_idx  # NOTE: this is seq_slot from the PREVIOUS iteration
                stride = nt_batch_size  # based on new_tokens_flat
                slot_gather_indices.append(start)
                flat_gather_indices.extend(range(start, start + (1 + draft_len) * stride, stride))
            else:
                input_ids.append(request.get_last_tokens(0))
                input_ids.extend([] if draft_len == 0 else request.py_draft_tokens)

            cu_seqlen.append(len(input_ids))
            input_pos.append(num_tokens_seen)
            prompt_lens.append(request.py_prompt_len)

            if is_overlap:
                mask_scatter_indices.extend(list(range(cu_seqlen[-2], cu_seqlen[-1])))

        # Store cache information for all requests.
        # All KV pools are hosted by a single C++ KVCacheManager; pool index is
        # the position of the window in self.cache_seq_interface.kv_group_windows
        # (the same source the kvcache transform used to register window groups
        # on SequenceInfo).  Per-window queries on the manager route to the
        # correct C++ pool via mLayerToWindowSize.
        kv_group_windows = self.cache_seq_interface.kv_group_windows
        # When the attention kernel applies the sliding-window mask itself via
        # cyclic KV indexing (trtllm), the executor must hand it the full
        # per-window block table and a global (un-window-capped) KV length --
        # the same contract as the PyTorch backend. Otherwise (triton /
        # flashinfer) host-slice the block table to the live window below.
        cyclic_swa = self.cache_seq_interface.kernel_handles_cyclic_swa
        # Cache hot lookups so the per-request loop avoids repeated C++
        # dispatch / hasattr calls.
        _tokens_per_block = kv_cache_manager.tokens_per_block
        _use_mamba = hasattr(kv_cache_manager, "mamba_cache_index")
        state_slot_idx: List[int] = []

        # Uniform per-pool transport: a single-pool deployment is just the
        # degenerate case of VSWA with one pool, so all pools (including 0)
        # take the same code path below.
        num_pools = len(kv_group_windows)
        cache_loc_per_pool: List[List[int]] = [[] for _ in range(num_pools)]
        cu_num_pages_per_pool: List[List[int]] = [[0] for _ in range(num_pools)]
        extra_page_per_seq_per_pool: List[List[int]] = [[] for _ in range(num_pools)]
        seq_len_with_cache_per_pool: List[List[int]] = [[] for _ in range(num_pools)]
        last_page_len_per_pool: List[List[int]] = [[] for _ in range(num_pools)]

        # Batched per-pool lookup preserves the #13560 host-overhead
        # optimization: one C++ call per pool instead of one per (request, pool).
        request_ids = [r.py_request_id for r in ordered_requests]
        batch_cache_indices_per_pool: List[List[List[int]]] = [
            kv_cache_manager.get_batch_cache_indices(request_ids, window_size=group_window)
            for group_window in kv_group_windows
        ]

        for i, request in enumerate(ordered_requests):
            # store seq slot idx (use mamba_cache_index if available)
            request.py_batch_idx = request.py_seq_slot
            if _use_mamba:
                state_slot_idx_i = kv_cache_manager.mamba_cache_index[request.py_request_id]
            else:
                state_slot_idx_i = request.py_seq_slot
            state_slot_idx.append(state_slot_idx_i)

            # get some info on the current request
            seq_len_i = cu_seqlen[i + 1] - cu_seqlen[i]
            end_compute_i = input_pos[i] + seq_len_i

            for pool_idx, group_window in enumerate(kv_group_windows):
                all_indices = batch_cache_indices_per_pool[pool_idx][i]
                if cyclic_swa:
                    # Cyclic-SWA kernels (trtllm) want the FULL per-window block
                    # table and the GLOBAL KV length; the kernel masks the window
                    # internally. No front-eviction slicing, so the
                    # get_num_front_blocks_removed C++ dispatch is skipped here.
                    (
                        active_indices,
                        extra_page,
                        active_token_count,
                        lpl_i,
                    ) = _compute_cyclic_full_view(
                        all_indices,
                        end_compute_i=end_compute_i,
                        tokens_per_block=_tokens_per_block,
                    )
                    num_active = len(active_indices)
                else:
                    # SWA front-eviction: get_batch_cache_indices returns the FULL
                    # historical page list including front-evicted entries (the
                    # C++ side bumps a counter rather than popping mCacheBlockIds).
                    # _compute_window_local_view slices it down to the live window
                    # in window-local coords.
                    front_removed = kv_cache_manager.get_num_front_blocks_removed(
                        request.py_request_id, window_size=group_window
                    )
                    (
                        active_indices,
                        extra_page,
                        active_token_count,
                        lpl_i,
                    ) = _compute_window_local_view(
                        all_indices,
                        front_removed=front_removed,
                        end_compute_i=end_compute_i,
                        group_window=group_window,
                        tokens_per_block=_tokens_per_block,
                    )
                    num_active = len(active_indices)

                cache_loc_per_pool[pool_idx].extend(active_indices)
                cu_num_pages_per_pool[pool_idx].append(
                    cu_num_pages_per_pool[pool_idx][i] + num_active
                )
                extra_page_per_seq_per_pool[pool_idx].append(extra_page)
                # seq_len_with_cache / last_page_len per pool (including 0).
                # Cyclic-SWA (trtllm): the global KV length for every pool.
                # Host-sliced (triton/flashinfer): the unclamped global value for
                # full-attention pools (window == max_seq_len, no clamping), and
                # the window-local coords for SWA pools under front-eviction --
                # identical to the legacy single-pool path for non-SWA models.
                seq_len_with_cache_per_pool[pool_idx].append(active_token_count)
                last_page_len_per_pool[pool_idx].append(lpl_i)

        # Store batch information based on prefill, decode, and extend requests.
        num_decode = len(generation_requests)
        num_decode_tokens = num_decode
        num_extend = len(extend_requests)
        num_extend_tokens = len(input_ids) - num_prefill_tokens - num_decode_tokens
        batch_info = [num_prefill, num_prefill_tokens]
        batch_info.extend([num_extend, num_extend_tokens])
        batch_info.extend([num_decode, num_decode_tokens])

        self._store_prefill_multimodal_metadata(
            ordered_requests=ordered_requests,
            input_pos=input_pos,
            cu_seqlen=cu_seqlen,
            num_prefill_seqs=num_prefill,
            extra_args=extra_args,
        )

        # update the sequence info object now (also triggers rescatter + host_prepare internally)
        self.cache_seq_interface.info.nest_sequences(
            input_ids,
            cu_seqlen=cu_seqlen,
            input_pos=input_pos,
            batch_info=batch_info,
            cache_loc_per_pool=cache_loc_per_pool if num_pools > 0 else None,
            cu_num_pages_per_pool=cu_num_pages_per_pool if num_pools > 0 else None,
            extra_page_per_seq_per_pool=extra_page_per_seq_per_pool if num_pools > 0 else None,
            seq_len_with_cache_per_pool=seq_len_with_cache_per_pool if num_pools > 0 else None,
            last_page_len_per_pool=last_page_len_per_pool if num_pools > 0 else None,
            slot_idx=state_slot_idx,
            prompt_lens=prompt_lens,
            gather_context_logits=gather_context_logits,
            _gather_idx=flat_gather_indices,
            _mask_scatter_indices=mask_scatter_indices,
            _ungathered_input_ids=new_tokens_flat,
            _gather_slot_idx=slot_gather_indices,
            _ungathered_new_lens=new_tokens_lens,
            **extra_args,
        )

        self.iter_states["num_ctx_requests"] = num_prefill
        self.iter_states["num_ctx_tokens"] = num_prefill_tokens
        self.iter_states["has_context_multimodal_data"] = has_context_multimodal_data
        # TODO: handle extend requests and draft requests for specdec
        self.iter_states["num_generation_tokens"] = num_decode_tokens + num_extend_tokens
        self.iter_states["ordered_requests"] = ordered_requests

    @nvtx_range("ad_run_forward")
    def _run_forward(self) -> Dict[str, Optional[torch.Tensor]]:
        """Run model forward and return outputs."""
        csi = self.cache_seq_interface

        if self.spec_config is not None:
            model_output = self.model(**csi.named_args, cache_seq_interface=csi)
        else:
            model_output = self.model(**csi.named_args)
        if self.iter_states.get("has_context_multimodal_data", False):
            # Multimodal prefill can leave image/context work in flight on the execution
            # stream after model.forward returns. Synchronize before the overlap scheduler
            # advances to the next step so later batch-state reuse does not race that work
            # and surface as a delayed CUDA illegal memory access.
            torch.cuda.current_stream().synchronize()

        # construct output dictionary
        if isinstance(model_output, abc.Mapping):
            output = dict(model_output)
        else:
            output = {"logits": model_output[0]}

        # squeeze logits and cast to float32
        logits = output["logits"]
        output["logits"] = self.cache_seq_interface.info.maybe_gather_and_squeeze(logits).float()

        return output

    def get_max_num_sequences(self) -> int:
        """Maximum number of sequences supported by the engine."""
        return self.cache_seq_interface.info.max_batch_size

    @torch.inference_mode()
    @maybe_pad_for_cuda_graph
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
    ):
        """Run forward from scheduled requests; main entrypoint that gets called by the executor."""
        # we don't support gather_context_logits in spec dec
        if self.spec_config is not None and self.spec_config.spec_dec_mode.without_logits():
            assert not gather_context_logits, "gather_context_logits not supported in spec dec"

        # convert requests and store in sequence info object
        new_tokens = getattr(new_tensors_device, "new_tokens", None)
        new_tokens_lens = getattr(new_tensors_device, "new_tokens_lens", None)
        self._prepare_inputs(
            scheduled_requests, resource_manager, new_tokens, new_tokens_lens, gather_context_logits
        )
        self.iter_counter += 1

        # compute outputs
        outputs = self._run_forward()

        if self.dist_config is not None:
            self._execute_logit_post_processors(scheduled_requests, outputs)

        return outputs


class TRTLLMSamplerModelConfig:
    def __init__(self, vocab_size_padded: int):
        self.config = SimpleNamespace()
        self.config.vocab_size = vocab_size_padded

        # Initialized to dummy values as they are not used in the C++ code underlying TRTLLMSampler.
        self.config.num_hidden_layers = 42
        self.config.hidden_size = 42
        self.config.num_attention_heads = 42


def instantiate_sampler(
    ad_config: LlmArgs,
    max_num_sequences: int,
    dist_mapping: Mapping,
    engine: ADEngine,
):
    spec_config = ad_config.speculative_config
    max_draft_len = 0 if spec_config is None else spec_config.max_draft_len
    max_total_draft_tokens = 0 if spec_config is None else spec_config.tokens_per_gen_step - 1

    # One-model spec dec: model performs sampling internally, returns pre-computed tokens
    if spec_config is not None and (
        spec_config.spec_dec_mode.is_eagle3_one_model()
        or spec_config.spec_dec_mode.is_mtp_eagle_one_model()
    ):
        sampler_args = TorchSampler.Args(
            max_seq_len=ad_config.max_seq_len,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_num_sequences=max_num_sequences,
            max_beam_width=ad_config.max_beam_width,
            disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
        )
        return Eagle3OneModelSampler(sampler_args)

    sampler_type = ad_config.sampler_type
    if sampler_type == SamplerType.auto:
        sampler_type = SamplerType.TorchSampler

    if sampler_type == SamplerType.TorchSampler:
        # Regular TorchSampler for non-speculative decoding.
        sampler_args = TorchSampler.Args(
            max_seq_len=ad_config.max_seq_len,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_num_sequences=max_num_sequences,
            max_beam_width=ad_config.max_beam_width,
            disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
        )
        sampler = TorchSampler(sampler_args)

    elif sampler_type == SamplerType.TRTLLMSampler:
        vocab_size_padded: int = engine.cache_seq_interface.info.vocab_size_padded
        sampler_model_config = TRTLLMSamplerModelConfig(vocab_size_padded)
        decoding_mode = get_decoding_mode(ad_config.decoding_config, ad_config.max_beam_width)
        sampler = TRTLLMSampler(
            model=sampler_model_config,
            model_dtype=torch.bfloat16,  # hardcoded as bfloat16; does not seem necessary in C++ code.
            mapping=dist_mapping,
            decoding_mode=decoding_mode,
            disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
            max_seq_len=ad_config.max_seq_len,
            max_batch_size=ad_config.max_batch_size,
            max_beam_width=ad_config.max_beam_width,
            decoding_config=ad_config.decoding_config,
            kv_cache_config=ad_config.kv_cache_config,
        )
    else:
        raise ValueError(f"Sampler type {sampler_type} is not supported.")

    return sampler


def create_autodeploy_executor(
    ad_config: LlmArgs,
    tokenizer: Optional[TokenizerBase] = None,
    resource_governor_queue=None,
):
    """Create an AutoDeploy executor from the given configuration and tokenizer.
    The tokenizer is required for guided decoding.

    This is the entrypoint API to the _autodeploy backend.
    """
    # initialize process groups
    world_size = mpi_world_size()
    rank = mpi_rank()

    # DistConfig is the single source of truth within AutoDeploy.
    # Mapping is derived only for external TRT-LLM APIs that still require it.
    dc = ad_config.init_dist_config(rank, world_size)
    dist_mapping = dc.to_mapping()

    dist = Distributed.get(dist_mapping)
    ad_logger.set_rank(rank)
    torch.cuda.set_device(rank)
    port = dist.broadcast(get_free_port())  # use MPI broadcast to pick a free port
    initialize_or_skip(rank, world_size, port)

    ad_logger.info(f"dist_config={dc}, {dist=}, {port=}")

    # Setup AutoTuner with distributed state for allreduce autotuning
    AutoTuner.get().setup_distributed_state(dist_mapping)

    # some config
    assert ad_config.max_beam_width <= 1, "_autodeploy + beam_search is not supported"

    max_num_sequences = ad_config.max_batch_size * dc.pp_size

    # initialize model engine
    engine = ADEngine.build_from_config(
        ad_config=ad_config, dist_config=dc, mapping=dist_mapping, dist=dist
    )

    spec_config = ad_config.speculative_config

    if spec_config is not None and ad_config.guided_decoding_backend is not None:
        raise ValueError(
            "Guided decoding is not currently supported for speculative decoding in AutoDeploy."
        )

    # resource managers
    # KVCacheManager is now created and managed by CachedSequenceInterface during the
    # initialize_cache/resize_kv_cache transform pipeline. Get it from the interface.
    kv_cache_manager = engine.cache_seq_interface.kv_cache_manager
    kv_cache_config_tuned = engine.cache_seq_interface.kv_cache_config_tuned
    seq_slot_manager = SeqSlotManager(max_num_sequences=max_num_sequences)

    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
            ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager,
        }
    )
    resource_manager.resource_managers.move_to_end(ResourceManagerType.KV_CACHE_MANAGER, last=True)

    # TODO: consider passing through scheduler_config arguments here. Not doing this for now since
    # it requires correctly setting up the C++ pybind scheduler config from the LLMArgs and then
    # processing the arguments here...

    # Chunked prefill
    if ad_config.enable_chunked_prefill:
        chunk_unit_size = kv_cache_config_tuned.tokens_per_block
        chunking_policy = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
        ctx_chunk_config: Tuple[StrEnum, int] = (chunking_policy, chunk_unit_size)
    else:
        ctx_chunk_config = None

    # scheduling
    capacitor_scheduler = BindCapacityScheduler(
        max_num_requests=ad_config.max_batch_size,
        kv_cache_manager=kv_cache_manager.impl,
        peft_cache_manager=None,
    )
    mb_scheduler = BindMicroBatchScheduler(
        max_batch_size=ad_config.max_batch_size,
        max_num_tokens=engine.cache_seq_interface.info.max_num_tokens,
        ctx_chunk_config=ctx_chunk_config,
    )
    scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)

    sampler = instantiate_sampler(
        ad_config=ad_config,
        max_num_sequences=max_num_sequences,
        dist_mapping=dist_mapping,
        engine=engine,
    )

    cache_transceiver_config = ad_config.cache_transceiver_config
    kv_cache_transceiver = None
    if cache_transceiver_config is not None and cache_transceiver_config.backend is not None:
        if isinstance(kv_cache_manager, BaseMambaCacheManager):
            # See https://github.com/NVIDIA/TensorRT-LLM/issues/14320.
            raise RuntimeError(
                "AutoDeploy disaggregated serving does not currently support Mamba/hybrid cache "
                "managers. A prerequisite for disaggregated serving of hybrid models is to use "
                "the C++ MambaCacheManager, which is currently not supported in AutoDeploy."
            )
        if cache_transceiver_config.max_tokens_in_buffer is None:
            # The buffer must hold the prompt's KV state (full prefill length).
            # We use max_seq_len as a safe upper bound on max ISL.
            cache_transceiver_config.max_tokens_in_buffer = (
                engine.cache_seq_interface.info.max_seq_len
            )

        cache_attention_type = engine.cache_seq_interface.attention_type
        if cache_attention_type is None:
            raise RuntimeError(
                "Cache transceiver is enabled, but AutoDeploy did not find a managed paged KV "
                "resource to provide attention_type."
            )
        if not isinstance(cache_attention_type, AttentionType):
            raise TypeError(f"attention_type must be AttentionType, got {cache_attention_type!r}")
        attention_type_cpp = _ATTENTION_TYPE_TO_CPP[cache_attention_type]

        kv_cache_transceiver = create_kv_cache_transceiver(
            dist_mapping,
            dist,
            kv_cache_manager,
            attention_type_cpp,
            cache_transceiver_config,
            mamba_cache_manager=None,
        )

    # Guided (structured) decoding.
    guided_decoder = None
    if (
        (guided_decoding_backend := ad_config.guided_decoding_backend) is not None
    ) and dc.pp_rank == dc.pp_size - 1:
        vocab_size_padded = engine.cache_seq_interface.info.vocab_size_padded
        if vocab_size_padded is None:
            raise RuntimeError(
                "Could not determine the vocabulary size. Required for guided decoding."
            )

        # The tokenizer may be None if stripped from MPI kwargs to avoid pickle
        # failures with trust_remote_code models. Reload it from the model path
        # when guided decoding needs it.
        if tokenizer is None and ad_config.model is not None:
            ad_logger.info("Tokenizer not provided; loading from model path for guided decoding")
            from tensorrt_llm.tokenizer import TransformersTokenizer

            tokenizer = TransformersTokenizer.from_pretrained(
                ad_config.model, trust_remote_code=ad_config.trust_remote_code
            )

        guided_decoding_config = get_guided_decoding_config(
            guided_decoding_backend=guided_decoding_backend, tokenizer=tokenizer
        )
        guided_decoder = GuidedDecoder(
            guided_decoding_config=guided_decoding_config,
            max_num_sequences=ad_config.max_batch_size,
            vocab_size_padded=vocab_size_padded,
        )

    # Speculative-decoding draft sizes must be forwarded to PyExecutor so that the
    # attention-DP dummy created in `_pad_attention_dp_dummy_request` is materialized
    # with `py_draft_tokens = [1] * max_total_draft_tokens` instead of `[]`. An empty
    # `py_draft_tokens` makes the dummy classify as decode in `_prepare_inputs` and
    # trips the eagle wrapper's `assert num_decode == 0` under MTP + attention_dp.
    max_draft_len = 0 if spec_config is None else spec_config.max_draft_len
    max_total_draft_tokens = 0 if spec_config is None else spec_config.tokens_per_gen_step - 1

    # creating the executor object
    py_executor = PyExecutor(
        resource_manager,
        scheduler,
        model_engine=engine,
        sampler=sampler,
        dist=dist,
        max_num_sequences=max_num_sequences,
        disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
        max_input_len=ad_config.max_input_len,
        max_batch_size=ad_config.max_batch_size,
        max_beam_width=ad_config.max_beam_width,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        guided_decoder=guided_decoder,
        kv_cache_transceiver=kv_cache_transceiver,
        resource_governor_queue=resource_governor_queue,
        garbage_collection_gen0_threshold=ad_config.garbage_collection_gen0_threshold,
    )
    return py_executor
