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

"""TriAttention KV-cache compression: periodic physical KV eviction during generation.

Every ``beta`` confirmed tokens, cached tokens are scored with a trigonometric
importance score from offline calibration and tokens outside the top-``budget``
keep set are physically deleted; decode runs the model's standard attention over
the compacted cache. Kept keys keep their original RoPE rotation (no re-RoPE).
KV pools must be read with ``kv_layout="HND"``. Calibration comes from the
official tool (github.com/WeianMao/triattention) and is converted at load.
"""

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import triton
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger

from ...distributed import allgather
from ...pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, _check_page_table_is_gpu_addressable
from ...pyexecutor.llm_request import LlmRequestState
from ...pyexecutor.resource_manager import KVCacheCompressionManager
from ...utils import next_positive_power_of_2
from ..compaction import build_compaction_params, compact
from .triattention_cute_score_fused import PADDED_HEAD_COLUMNS, build_score_pipeline
from .triattention_kernels import (
    fold_union_ranks,
    gather_mean_phase,
    reduce_per_head_scores,
    settle_ties,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig

    from ...pyexecutor.llm_request import LlmRequest
    from ...pyexecutor.scheduler import ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})

_MEAN_PHASE_OFFSETS = tuple(float(1 << exponent) for exponent in range(17))

# Physical TopK rows follow the 256-token reduce/tie kernel tiles.
_SELECTION_WIDTH_ALIGNMENT = 256


class _EvictionRequest(NamedTuple):
    """One due request and the cache state needed by its eviction round."""

    request: "LlmRequest"
    target_cache: object
    draft_cache: Optional[object]
    source_length: int
    target_tail_length: int


_BLOCK_OFFSET_ALIGNMENT = 4


def _allocate_block_offset_snapshot(
    manager: KVCacheManagerV2,
    anchor_pool: torch.Tensor,
    *,
    request_capacity: int,
    token_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the bounded V2 page-table snapshot used by an eviction round."""
    required_blocks = triton.cdiv(token_capacity, int(manager.tokens_per_block))
    staged_blocks = min(
        triton.cdiv(required_blocks, _BLOCK_OFFSET_ALIGNMENT) * _BLOCK_OFFSET_ALIGNMENT,
        int(manager.max_blocks_per_seq),
    )
    snapshot_shape = (int(manager.num_pools), request_capacity, 2, staged_blocks)
    block_offsets_host = torch.empty(
        snapshot_shape, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
    )
    if not manager.uses_device_page_table:
        _check_page_table_is_gpu_addressable(host_rows=block_offsets_host)
    block_offsets_device = torch.empty(snapshot_shape, dtype=torch.int32, device=anchor_pool.device)
    return block_offsets_host, block_offsets_device


_MEAN_PHASE_MAX_ROWS = 1 << 24


class _MeanPhaseTable:
    """Admission-sized mean-phase lookup used by every eviction round."""

    def __init__(self, omega: torch.Tensor, device: torch.device) -> None:
        self._omega = omega.to(device=device, dtype=torch.float32).contiguous()
        self.cos: Optional[torch.Tensor] = None
        self.sin: Optional[torch.Tensor] = None
        self.rows = 0
        self.num_freqs = int(self._omega.numel())

    def reserve(self, rows: int) -> None:
        """Cover positions ``[0, rows)`` with a power-of-two table."""
        rows = int(rows)
        if rows <= self.rows:
            return
        if rows > _MEAN_PHASE_MAX_ROWS:
            raise ValueError(f"a {rows}-row mean-phase table exceeds the exact-FP32 position range")
        target = next_positive_power_of_2(rows)
        target = min(max(target, 2 * self.rows), _MEAN_PHASE_MAX_ROWS)
        positions = torch.arange(target, device=self._omega.device, dtype=torch.float32)
        cos_table = torch.zeros(
            (target, self.num_freqs),
            dtype=torch.float32,
            device=self._omega.device,
        )
        sin_table = torch.zeros_like(cos_table)
        # Fixed summation order keeps table rebuilds bit-stable.
        for offset in _MEAN_PHASE_OFFSETS:
            angle = torch.outer(positions + offset, self._omega)
            cos_table += torch.cos(angle)
            sin_table += torch.sin(angle)
        scale = 1.0 / len(_MEAN_PHASE_OFFSETS)
        self.cos = cos_table.mul_(scale)
        self.sin = sin_table.mul_(scale)
        self.rows = target


class TriAttentionCompressionManager(KVCacheCompressionManager):
    """KV-cache compression manager for periodic TriAttention eviction."""

    # ---- construction ----

    def __init__(
        self,
        config: "TriAttentionKvCacheCompressionConfig",
        kv_cache_manager: KVCacheManagerV2,
        draft_kv_cache_manager: Optional[KVCacheManagerV2] = None,
        *,
        pretrained_config: "PretrainedConfig",
    ) -> None:
        super().__init__(config, kv_cache_manager, draft_kv_cache_manager)
        self.budget = config.budget
        self.beta = config.beta
        self.eviction_mode = config.eviction_mode
        if self.eviction_mode == "union" and not config.normalize_scores:
            logger.warning("TriAttention union mode enables score normalization")
        self.normalize_scores = self.eviction_mode == "union" or config.normalize_scores
        # Prompt always pinned; budget counts decode tokens only.
        self.pretrained_config = pretrained_config
        self.calibration_path = config.calibration_path
        self._load_calibration()

        self._prepared_generation_batch: Optional["ScheduledRequests"] = None
        # Manager-lifetime constants.
        self._num_extra_kv_tokens = int(kv_cache_manager.num_extra_kv_tokens)
        self._protected_tail_capacity = (
            self._num_extra_kv_tokens + int(kv_cache_manager._kv_reserve_draft_tokens) + 1
        )
        self._draft_protected_tail_capacity = 0
        if draft_kv_cache_manager is not None:
            self._draft_protected_tail_capacity = (
                int(draft_kv_cache_manager.num_extra_kv_tokens)
                + int(draft_kv_cache_manager._kv_reserve_draft_tokens)
                + 1
            )
        # The next-step reservation size is fixed; overlap only changes which
        # requests have it.
        self._overlap_tail_length = 1 + int(kv_cache_manager._kv_reserve_draft_tokens)
        # Fixed buffer geometry. These are TriAttention scratch dimensions,
        # not KV capacities owned by KVCacheManagerV2.
        self._request_capacity = int(kv_cache_manager.max_batch_size)
        max_draft_tokens = int(kv_cache_manager.max_total_draft_tokens)
        # Crossing a cadence can overshoot by D accepted draft tokens; one
        # suspended due round may resume with another 1 + D confirmed tokens.
        required_selection_width = self.budget + self.beta + 2 * max_draft_tokens + 1
        self._selection_width_capacity = (
            triton.cdiv(required_selection_width, _SELECTION_WIDTH_ALIGNMENT)
            * _SELECTION_WIDTH_ALIGNMENT
        )
        max_tail_capacity = max(
            self._protected_tail_capacity,
            self._draft_protected_tail_capacity,
        )
        if self._request_capacity * (self.budget + max_tail_capacity) >= 2**31:
            raise ValueError("TriAttention compaction offsets exceed the int32 range")
        # Manager-lifetime layer facts, resolved once: V2 fixes pp_layers at
        # construction and the model config is immutable on disk.
        self._global_layers = [int(layer) for layer in kv_cache_manager.pp_layers]
        (
            self._dense_layers,
            self._swa_layers,
            self._swa_window,
        ) = self._resolve_attention_layers()
        self._initialize_eviction_state()

    def _resolve_attention_layers(self) -> Tuple[List[int], List[int], Optional[int]]:
        """SWA layers here are stored at full length; the window applies only in the kernel."""
        global_layers = self._global_layers
        num_layers = len(global_layers)

        config_values = self.pretrained_config.get_text_config().to_dict()
        layer_types = config_values.get("layer_types")
        if not layer_types:
            use_sliding_window = config_values.get("use_sliding_window")
            has_swa_signal = (
                use_sliding_window
                if isinstance(use_sliding_window, bool)
                else any(
                    config_values.get(field)
                    for field in (
                        "sliding_window",
                        "sliding_window_size",
                        "sliding_window_pattern",
                        "max_window_layers",
                    )
                )
            )
            if has_swa_signal:
                raise ValueError(
                    "Model config exposes sliding-window metadata but no layer_types; "
                    "TriAttention cannot classify kernel-masked SWA layers safely"
                )
            return (list(range(num_layers)), [], None)
        if global_layers and max(global_layers) >= len(layer_types):
            raise ValueError(
                f"Model config has {len(layer_types)} layer_types entries, "
                f"but this PP rank references global layer {max(global_layers)}"
            )

        swa_layers = [
            local_layer
            for local_layer, global_layer in enumerate(global_layers)
            if "sliding" in str(layer_types[global_layer]).lower()
        ]
        swa_set = set(swa_layers)
        dense_layers = [layer for layer in range(num_layers) if layer not in swa_set]
        # GPT-OSS SWA keeps full-length V2 pools and masks in the kernel; native
        # sliding-eviction layouts such as Gemma 4 remain unsupported.
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        window_size = None
        if swa_layers:
            raw_window = config_values.get("sliding_window")
            if not isinstance(raw_window, int) or raw_window <= 0:
                raise ValueError(
                    "TriAttention requires a positive integer model sliding_window "
                    "when layer_types contains sliding attention"
                )
            if self.budget < raw_window:
                raise ValueError(
                    f"TriAttention budget={self.budget} must be at least "
                    f"the kernel-masked SWA window size {raw_window}"
                )
            window_size = raw_window
        return (dense_layers, swa_layers, window_size)

    def _load_calibration(self) -> None:
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            e_q = raw["E_q"]
            e_q_norm = raw["E_q_norm"]
            omega = raw["omega"]
            freq_scale_sq = raw["freq_scale_sq"]
        elif isinstance(raw, dict) and {"metadata", "stats"} <= set(raw):
            stats = raw["stats"]
            metadata = raw["metadata"]
            if "sampled_heads" in metadata:
                heads = [(int(layer), int(head)) for layer, head in metadata["sampled_heads"]]
            else:
                heads = [
                    (
                        int(key[len("layer") : key.index("_head")]),
                        int(key[key.index("_head") + len("_head") :]),
                    )
                    for key in stats
                ]
            num_layers = max(layer for layer, _ in heads) + 1
            num_heads = max(head for _, head in heads) + 1
            freq_count = int(next(iter(stats.values()))["q_mean_real"].numel())
            e_q = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64)
            e_q_norm = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.float32)
            for layer, head in heads:
                head_stats = stats[f"layer{layer:02d}_head{head:02d}"]
                e_q[layer, head] = torch.complex(
                    head_stats["q_mean_real"].float(),
                    head_stats["q_mean_imag"].float(),
                )
                e_q_norm[layer, head] = head_stats["q_abs_mean"].float()

            config = self.pretrained_config.get_text_config()
            # transformers >= 5.5 folds rope_theta/rope_type into
            # rope_parameters, but the executor's config loader clears it for
            # models with default (unscaled) RoPE and keeps the canonical
            # value on config.rope_theta.
            rope_params = config.to_dict()["rope_parameters"]
            if rope_params is None:
                rope_params = {"rope_type": "default", "rope_theta": config.rope_theta}
            if all(isinstance(value, dict) for value in rope_params.values()):
                raise ValueError(
                    "TriAttention does not support per-layer-type rope parameters "
                    f"(model_type={config.model_type})"
                )
            rope_type = rope_params["rope_type"]
            if rope_type == "default":
                # "default" has no ROPE_INIT_FUNCTIONS entry.
                head_dim = freq_count * 2
                base = float(rope_params["rope_theta"])
                positions = torch.arange(0, head_dim, 2, dtype=torch.float32)
                omega = (1.0 / (base ** (positions / head_dim)))[:freq_count].clone()
                attention_scale_sq = 1.0
            else:
                inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](config, device="cpu")
                omega = inv_freq.to(torch.float32)[:freq_count].clone()
                attention_scale_sq = float(attention_factor) ** 2
            freq_scale_sq = torch.full((freq_count,), attention_scale_sq, dtype=torch.float32)
            logger.info(
                f"TriAttention: converted official calibration {self.calibration_path}"
                f" -> E_q[L={num_layers}, H={num_heads}, F={freq_count}]"
            )
        else:
            got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise ValueError(
                f"Unrecognized calibration at {self.calibration_path}: expected the "
                f"official {{metadata, stats}} layout or "
                f"{sorted(_REQUIRED_CALIBRATION_KEYS)}; got {got}."
            )

        self._freq_scale_sq = freq_scale_sq.to(dtype=torch.float32)
        self._omega = omega
        # Pre-split query stats + MLR coefficient, shapes [L, H, F].
        self._calibration_q_real = e_q.real.to(torch.float32).contiguous()
        self._calibration_q_imag = e_q.imag.to(torch.float32).contiguous()
        self._calibration_mlr_coef = (
            e_q_norm.to(torch.float32) - e_q.abs().to(torch.float32)
        ).contiguous()

    # ---- framework hooks (call order) ----

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Grow scorer state only when this request raises its capacity high-water mark."""
        manager = self.kv_cache_manager
        prompt_length = int(request.py_prompt_len)
        max_decode_tokens = min(
            int(request.py_max_new_tokens),
            max(int(manager.max_seq_len) - prompt_length, 0),
        )
        first_evict_step = (self.budget // self.beta + 1) * self.beta
        if max_decode_tokens < first_evict_step:
            return
        self._phase.reserve(prompt_length + max_decode_tokens + 1)
        max_source_tokens = prompt_length + min(max_decode_tokens, self._selection_width_capacity)
        if max_source_tokens <= self._score_token_capacity:
            return

        # CuTe launches capture score buffers; retire the old capacity before replacing it.
        if self._launch_score is not None:
            self._compaction_done_event.synchronize()

        new_score_capacity = next_positive_power_of_2(max(max_source_tokens, 1024))
        new_score_capacity = min(new_score_capacity, int(manager.max_seq_len))
        score_tile_size = max(64, int(manager.tokens_per_block))
        new_score_capacity = triton.cdiv(new_score_capacity, score_tile_size) * score_tile_size
        self._build_score_runtime(score_token_capacity=new_score_capacity)

    def on_generation_step_begin(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Remember the next batch: overlap prepares it before updating the previous batch."""
        self._prepared_generation_batch = scheduled_batch

    def on_generation_step_end(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Compact after native KV-cache updates finalize the iteration.

        KVCacheManagerV2 must run first so capacity includes the written token and any rewind.
        """
        self._evict_due_requests(scheduled_batch)

    # ---- eviction round ----

    def _evict_due_requests(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Collect due requests, execute one eviction round, publish, and resize."""
        manager = self.kv_cache_manager
        eviction_requests: List[_EvictionRequest] = []
        # With overlap, the next batch reserves KV before the previous batch is
        # compacted. Those reserved slots are a byte-preserved tail, not score input.
        prepared_batch = self._prepared_generation_batch
        overlap_request_ids = (
            {request.py_request_id for request in prepared_batch.generation_requests}
            if prepared_batch is not None and prepared_batch is not scheduled_batch
            else set()
        )
        for request in scheduled_batch.generation_requests:
            if request.is_dummy or request.state in (
                LlmRequestState.GENERATION_COMPLETE,
                LlmRequestState.CONTEXT_INIT,
            ):
                continue
            request_id = request.py_request_id
            target_cache = manager.kv_cache_map.get(request_id)
            if target_cache is None or not target_cache.is_active:
                # Overlap scheduling may suspend a cache mid-flight; defer
                # this request (pre-launch) instead of failing the batch.
                continue
            draft_cache = None
            if self.draft_kv_cache_manager is not None:
                # A missing draft cache is a wiring bug: keep the precise KeyError.
                draft_cache = self.draft_kv_cache_manager.kv_cache_map[request_id]
                if not draft_cache.is_active:
                    continue
            target_tail_length = self._num_extra_kv_tokens + (
                self._overlap_tail_length if request_id in overlap_request_ids else 0
            )
            source_length = int(target_cache.capacity) - target_tail_length
            if source_length < target_cache.history_length:
                raise RuntimeError(
                    f"Request {request_id} KV length {source_length} is below "
                    f"finalized history {target_cache.history_length}"
                )
            prompt_length = int(request.py_prompt_len)
            # Restore the logical length from the physical cache and the
            # eviction count already published to the model runtime.
            compressed_tokens = int(request.py_num_compressed_tokens)
            logical_source_length = source_length + compressed_tokens
            confirmed_tokens = logical_source_length - prompt_length
            # The last compact ended at budget + compressed_tokens. This
            # watermark catches a beta boundary deferred by cache suspension.
            if (self.budget + compressed_tokens) // self.beta >= (confirmed_tokens // self.beta):
                continue
            if source_length <= prompt_length + self.budget:
                # Selection would be an identity: nothing to evict yet.
                continue
            decode_width = source_length - prompt_length
            if decode_width > self._selection_width_capacity:
                raise RuntimeError(
                    f"Request {request_id} TriAttention selection width "
                    f"{decode_width} exceeds compiled capacity "
                    f"{self._selection_width_capacity}"
                )
            eviction_requests.append(
                _EvictionRequest(
                    request=request,
                    target_cache=target_cache,
                    draft_cache=draft_cache,
                    source_length=source_length,
                    target_tail_length=target_tail_length,
                )
            )
        if not eviction_requests:
            return

        self._execute_eviction_round(eviction_requests)
        for item in eviction_requests:
            evicted = item.source_length - int(item.request.py_prompt_len) - self.budget
            # The manager's only channel to the runtime (feeds num_cached_tokens_per_seq).
            item.request.py_num_compressed_tokens += evicted
        self._resize_compacted_caches(eviction_requests)

    def _execute_eviction_round(
        self,
        eviction_requests: Sequence[_EvictionRequest],
    ) -> None:
        """Score, select, and compact one due request group."""
        manager = self.kv_cache_manager
        draft_manager = self.draft_kv_cache_manager
        stream = torch.cuda.current_stream(self._block_offsets_device.device)
        # PyExecutor already joins its execution stream before the final
        # compression resource update, so the round can use caller current.
        try:
            request_ids = [item.request.py_request_id for item in eviction_requests]
            logical_source_lengths = [
                item.source_length + int(item.request.py_num_compressed_tokens)
                for item in eviction_requests
            ]
            prompt_lengths = [int(item.request.py_prompt_len) for item in eviction_requests]
            source_lengths = [item.source_length for item in eviction_requests]
            dense_move_offsets, swa_move_offsets, draft_move_offsets = (
                self._compute_compaction_move_offsets(eviction_requests)
            )
            metadata_rows = (
                logical_source_lengths,
                source_lengths,
                prompt_lengths,
                dense_move_offsets,
                swa_move_offsets,
                draft_move_offsets,
            )
            # CPU may rewrite pinned staging only after its prior H2D completes.
            self._staging_reuse_event.synchronize()
            host_table = self._request_metadata_host_np
            for row, values in enumerate(metadata_rows):
                if values is not None:
                    host_table[row, : len(values)] = values
            # Native compaction keeps fixed-capacity metadata views; make
            # their unused request rows explicit no-ops.
            host_table[:3, len(eviction_requests) :] = 0
            try:
                self._stage_block_offset_snapshot(
                    manager,
                    request_ids,
                    self._block_offsets_host,
                    self._block_offsets_device,
                )
                if draft_manager is not None:
                    self._stage_block_offset_snapshot(
                        draft_manager,
                        request_ids,
                        self._draft_block_offsets_host,
                        self._draft_block_offsets_device,
                    )
                self._request_metadata_device.copy_(self._request_metadata_host, non_blocking=True)
            finally:
                self._staging_reuse_event.record(stream)

            request_count = len(eviction_requests)
            union = self.eviction_mode == "union"
            # In-place refresh: the compiled score launches captured these pointers.
            gather_mean_phase(
                self._logical_source_lengths_device,
                self._phase.cos,
                self._phase.sin,
                self._source_lengths_device,
                self._prompt_lengths_device,
                self._mean_cos,
                self._mean_sin,
                self._decode_lengths_device,
                self._swa_destination_bases,
                request_count=request_count,
                swa_rebase_delta=self._swa_rebase_delta,
            )
            self._launch_score(request_count)
            if union and self._union_tp_mapping is not None:
                # Max is order-free, so every TP rank keeps the same ordinals.
                gathered = allgather(
                    self._selection_scores_rows[:request_count],
                    self._union_tp_mapping,
                    dim=0,
                )
                fold_union_ranks(
                    gathered,
                    self._selection_scores_rows,
                    request_count=request_count,
                )
            if not union:
                reduce_per_head_scores(
                    self._score_scratch,
                    self._decode_lengths_device,
                    self._prompt_lengths_device,
                    self._row_mean,
                    self._row_inv_std,
                    self._selection_scores_rows,
                    self._selection_row_lengths,
                    request_count=request_count,
                    padded_head_columns=PADDED_HEAD_COLUMNS,
                    score_token_capacity=self._score_token_capacity,
                    per_layer=self.eviction_mode == "per_layer_perhead",
                    normalize_scores=self.normalize_scores,
                )
            self._select_kept_ordinals(request_count)
            compact(self._compaction_params, request_count)
        finally:
            # Target and draft V2 managers share this execution stream.
            self._compaction_done_event.record(stream)
            if manager._stream != stream:
                manager._stream.wait_event(self._compaction_done_event)

    def _compute_compaction_move_offsets(
        self,
        eviction_requests: Sequence[_EvictionRequest],
    ) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        """Build padded cumulative dense, SWA, and draft move offsets."""

        def cumulative_offsets(move_counts: List[int]) -> List[int]:
            offsets = [0]
            for count in move_counts:
                offsets.append(offsets[-1] + count)
            offsets.extend(offsets[-1:] * (self._request_capacity - len(move_counts)))
            return offsets

        tails = [int(item.target_tail_length) for item in eviction_requests]
        dense_offsets = cumulative_offsets([self.budget + tail for tail in tails])
        swa_offsets = None
        if self._swa_window is not None:
            swa_offsets = cumulative_offsets([self._swa_window + tail for tail in tails])
        draft_offsets = None
        if self.draft_kv_cache_manager is not None:
            draft_offsets = cumulative_offsets(
                [self.budget + self._draft_protected_tail_capacity] * len(eviction_requests)
            )
        return dense_offsets, swa_offsets, draft_offsets

    def _stage_block_offset_snapshot(
        self,
        manager: KVCacheManagerV2,
        request_ids: List[int],
        host_block_offsets: torch.Tensor,
        device_block_offsets: torch.Tensor,
    ) -> None:
        """Snapshot host block offsets before their asynchronous device copy."""
        if manager.uses_device_page_table:
            manager.materialize_block_offsets_snapshot(
                device_block_offsets,
                request_ids,
                host_staging=host_block_offsets,
                stream=torch.cuda.current_stream(device_block_offsets.device),
            )
        else:
            manager.index_mapper.gather_k_block_offsets(
                manager.host_kv_cache_block_offsets,
                host_block_offsets,
                request_ids,
                host_block_offsets.shape[-1],
            )
            copy_batch_block_offsets_to_device(
                host_block_offsets,
                device_block_offsets,
                self._identity_copy_indices_host[: len(request_ids)],
                manager.index_scales,
                manager.kv_offset,
                torch.cuda.current_stream(device_block_offsets.device).cuda_stream,
            )

    def _select_kept_ordinals(self, request_count: int) -> None:
        """Select top-k tokens and settle score ties into kept-ordinal rows."""
        rows = request_count * self._selection_rows_per_request
        # The trailing 1 is next_n: decode scores one query token per request.
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            self._selection_scores_rows[:rows],
            self._selection_row_lengths[:rows],
            self._provisional_rows[:rows],
            self.budget,
            1,
        )
        settle_ties(
            self._selection_scores_rows,
            self._selection_row_lengths,
            self._prompt_lengths_device,
            self._provisional_rows,
            self._kept_ordinal_rows,
            request_count=request_count,
            selection_rows_per_request=self._selection_rows_per_request,
        )

    def _resize_compacted_caches(self, eviction_requests: Sequence[_EvictionRequest]) -> None:
        for item in eviction_requests:
            target_capacity = (
                int(item.request.py_prompt_len) + self.budget + item.target_tail_length
            )
            if not item.target_cache.resize(target_capacity, None):
                raise RuntimeError(
                    "Failed to resize compacted target KV cache for "
                    f"request {item.request.py_request_id} to "
                    f"{target_capacity} tokens"
                )
        if self.draft_kv_cache_manager is None:
            return
        for item in eviction_requests:
            draft_capacity = (
                int(item.request.py_prompt_len) + self.budget + self._draft_protected_tail_capacity
            )
            if not item.draft_cache.resize(draft_capacity, None):
                raise RuntimeError(
                    "Failed to resize compacted draft KV cache for "
                    f"request {item.request.py_request_id} to "
                    f"{draft_capacity} tokens"
                )

    # ---- persistent state + score runtime ----

    def _initialize_eviction_state(self) -> None:
        """Create manager-lifetime state once."""
        target_layout = self._create_kv_layout()
        draft_layout = (
            self._create_kv_layout(draft=True) if self.draft_kv_cache_manager is not None else None
        )
        self._target_layout = target_layout
        self._draft_layout = draft_layout

        layer_pools = target_layout["layer_pools"]
        dense_layers = target_layout["dense_layers"]
        anchor_pool = layer_pools[dense_layers[0]]
        device = anchor_pool.device
        _, _, num_kv_heads, tokens_per_block, _ = anchor_pool.shape

        self._block_offsets_host = None
        self._block_offsets_device = None
        self._draft_block_offsets_host = None
        self._draft_block_offsets_device = None

        global_layers = self._global_layers
        if global_layers and max(global_layers) >= self._calibration_q_real.shape[0]:
            raise ValueError(
                f"TriAttention calibration has {self._calibration_q_real.shape[0]} layers, "
                f"but this PP rank references global layer {max(global_layers)}"
            )
        layer_ids = torch.as_tensor(
            global_layers,
            device=self._calibration_q_real.device,
            dtype=torch.long,
        )
        q_real = self._calibration_q_real.index_select(0, layer_ids)
        q_imag = self._calibration_q_imag.index_select(0, layer_ids)
        mlr_coef = self._calibration_mlr_coef.index_select(0, layer_ids)
        mapping = self.kv_cache_manager.mapping
        tp_size = 1 if mapping.enable_attention_dp else int(mapping.tp_size)
        if tp_size > 1:
            local_q_heads = int(q_real.shape[1]) // tp_size
            heads = slice(mapping.tp_rank * local_q_heads, (mapping.tp_rank + 1) * local_q_heads)
            q_real, q_imag, mlr_coef = q_real[:, heads], q_imag[:, heads], mlr_coef[:, heads]
        q_real, q_imag, mlr_coef, self._freq_scale_sq = (
            tensor.to(device=device, dtype=torch.float32).contiguous()
            for tensor in (q_real, q_imag, mlr_coef, self._freq_scale_sq)
        )
        self._union_tp_mapping = (
            mapping if (self.eviction_mode == "union" and tp_size > 1) else None
        )
        num_q_heads = int(q_real.shape[1])
        num_freqs = int(q_real.shape[2])
        self._score_q_real = q_real
        self._score_q_imag = q_imag
        self._score_mlr_coef = mlr_coef

        self._phase = _MeanPhaseTable(self._omega, device)
        self._num_layers = len(dense_layers)
        self._num_q_heads = num_q_heads
        self._num_kv_heads = int(num_kv_heads)
        self._allocate_metadata_buffers(
            device,
            num_freqs=num_freqs,
        )
        self._allocate_selection_buffers(device, tp_size=tp_size)

        self._compaction_params = ()
        self._score_scratch = None
        self._score_token_capacity = 0
        self._launch_score = None

        self._staging_reuse_event = torch.cuda.Event()
        self._staging_reuse_event.record(torch.cuda.current_stream(device))
        self._compaction_done_event = torch.cuda.Event()
        self._compaction_done_event.record(torch.cuda.current_stream(device))

        logger.info(
            f"TriAttention CuTe score configured: {self._num_q_heads}q/"
            f"{self._num_kv_heads}kv heads, {num_freqs} freqs, "
            f"{int(tokens_per_block)}-token pages"
        )

    def _allocate_metadata_buffers(
        self,
        device: torch.device,
        *,
        num_freqs: int,
    ) -> None:
        """Allocate fixed manager-lifetime host staging and device metadata."""
        row_count = 6
        request_capacity = self._request_capacity
        self._request_metadata_host = torch.empty(
            (row_count, request_capacity + 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._request_metadata_host_np = self._request_metadata_host.numpy()
        self._identity_copy_indices_host = torch.arange(
            request_capacity,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._request_metadata_device = torch.zeros(
            (row_count, request_capacity + 1), dtype=torch.int32, device=device
        )
        self._logical_source_lengths_device = self._request_metadata_device[0, :request_capacity]
        self._source_lengths_device = self._request_metadata_device[1, :request_capacity]
        self._prompt_lengths_device = self._request_metadata_device[2, :request_capacity]
        self._dense_move_offsets_device = self._request_metadata_device[3]
        self._swa_move_offsets_device = self._request_metadata_device[4]
        self._draft_move_offsets_device = self._request_metadata_device[5]

        self._swa_destination_bases = (
            torch.empty_like(self._prompt_lengths_device) if self._swa_window is not None else None
        )
        self._swa_rebase_delta = (
            self.budget - self._swa_window if self._swa_window is not None else 0
        )
        self._mean_cos = torch.empty(
            (request_capacity, num_freqs), dtype=torch.float32, device=device
        )
        self._mean_sin = torch.empty_like(self._mean_cos)

    def _allocate_selection_buffers(self, device: torch.device, *, tp_size: int) -> None:
        """Allocate fixed manager-lifetime TopK inputs and outputs."""
        request_capacity = self._request_capacity
        selection_width = self._selection_width_capacity
        union = self.eviction_mode == "union"
        self._selection_rows_per_request = (
            1
            if union
            else self._num_kv_heads
            * (self._num_layers if self.eviction_mode == "per_layer_perhead" else 1)
        )
        selection_rows = request_capacity * self._selection_rows_per_request
        selection_rect = selection_rows * selection_width
        if union:
            selection_rect = max(
                selection_rect,
                tp_size * request_capacity * selection_width,
            )
        if selection_rect >= 2**31:
            raise ValueError(f"selection rectangle overflows 32-bit indexing: {selection_rect}")

        self._decode_lengths_device = torch.full(
            (request_capacity,), selection_width, dtype=torch.int32, device=device
        )
        if union:
            self._selection_scores_rows = torch.empty(
                (request_capacity, selection_width),
                dtype=torch.float32,
                device=device,
            )
            self._selection_row_lengths = self._decode_lengths_device
        else:
            score_shape = (
                request_capacity,
                self._num_layers,
                self._num_q_heads,
                1,
            )
            self._row_mean = torch.empty(score_shape, dtype=torch.float32, device=device)
            self._row_inv_std = torch.empty_like(self._row_mean)
            self._selection_scores_rows = torch.empty(
                (selection_rows, selection_width),
                dtype=torch.float32,
                device=device,
            )
            self._selection_row_lengths = torch.full(
                (selection_rows,),
                selection_width,
                dtype=torch.int32,
                device=device,
            )

        self._provisional_rows = torch.zeros(
            (selection_rows, self.budget), dtype=torch.int32, device=device
        )
        self._kept_ordinal_rows = torch.empty_like(self._provisional_rows)

    def _build_score_runtime(
        self,
        *,
        score_token_capacity: int,
    ) -> None:
        """Build one scorer span and refresh its page-table bindings."""
        dense_layer = self._target_layout["dense_layers"][0]
        anchor_pool = self._target_layout["layer_pools"][dense_layer]
        request_capacity = self._request_capacity
        block_offsets_host, block_offsets_device = _allocate_block_offset_snapshot(
            self.kv_cache_manager,
            anchor_pool,
            request_capacity=request_capacity,
            token_capacity=score_token_capacity + self._protected_tail_capacity,
        )
        draft_block_offsets_host = None
        draft_block_offsets_device = None
        if self._draft_layout is not None:
            draft_anchor_pool = self._draft_layout["layer_pools"][0]
            draft_block_offsets_host, draft_block_offsets_device = _allocate_block_offset_snapshot(
                self.draft_kv_cache_manager,
                draft_anchor_pool,
                request_capacity=request_capacity,
                token_capacity=(score_token_capacity + self._draft_protected_tail_capacity),
            )

        score_scratch, launch_score = build_score_pipeline(
            self._target_layout,
            block_offsets=block_offsets_device,
            source_lengths=self._source_lengths_device,
            prompt_lengths=self._prompt_lengths_device,
            mean_cos=self._mean_cos,
            mean_sin=self._mean_sin,
            q_real=self._score_q_real,
            q_imag=self._score_q_imag,
            mlr_coef=self._score_mlr_coef,
            freq_scale_sq=self._freq_scale_sq,
            score_token_capacity=score_token_capacity,
            union_scores=(self._selection_scores_rows if self.eviction_mode == "union" else None),
        )

        compaction_params = [
            build_compaction_params(
                self._target_layout,
                block_offsets=block_offsets_device,
                kept_ordinals=self._kept_ordinal_rows,
                source_lengths=self._source_lengths_device,
                dense_destination_bases=self._prompt_lengths_device,
                dense_move_offsets=self._dense_move_offsets_device,
                protected_tail_capacity=self._protected_tail_capacity,
                swa_move_offsets=self._swa_move_offsets_device,
                swa_destination_bases=self._swa_destination_bases,
            )
        ]
        if self._draft_layout is not None:
            compaction_params.append(
                build_compaction_params(
                    self._draft_layout,
                    block_offsets=draft_block_offsets_device,
                    kept_ordinals=self._kept_ordinal_rows,
                    source_lengths=self._source_lengths_device,
                    dense_destination_bases=self._prompt_lengths_device,
                    dense_move_offsets=self._draft_move_offsets_device,
                    protected_tail_capacity=self._draft_protected_tail_capacity,
                )
            )

        # Publish new score state only after every allocation and compile succeeds.
        self._block_offsets_host = block_offsets_host
        self._block_offsets_device = block_offsets_device
        self._draft_block_offsets_host = draft_block_offsets_host
        self._draft_block_offsets_device = draft_block_offsets_device
        self._score_scratch = score_scratch
        self._score_token_capacity = score_token_capacity
        self._launch_score = launch_score
        self._compaction_params = tuple(compaction_params)

    def _create_kv_layout(self, *, draft: bool = False) -> Dict[str, object]:
        """Resolve one manager-lifetime V2 pool layout."""
        manager = self.draft_kv_cache_manager if draft else self.kv_cache_manager

        if draft:
            global_layers = [int(layer) for layer in manager.pp_layers]
            # The draft is never scored: all draft layers compact as dense.
            dense_layers: List[int] = list(range(len(global_layers)))
            swa_layers: List[int] = []
            swa_window: Optional[int] = None
        else:
            global_layers = self._global_layers
            dense_layers = self._dense_layers
            swa_layers = self._swa_layers
            swa_window = self._swa_window
        layer_pools = [manager.get_buffers(layer, kv_layout="HND") for layer in global_layers]
        # Canonical pool IDs come from V2; its lookup errors are the precise ones.
        layer_offsets = manager.layer_offsets
        layer_to_pool = manager.layer_to_pool_mapping_dict
        layer_pool_ids = tuple(
            int(layer_to_pool[layer_offsets[global_layer]]) for global_layer in global_layers
        )
        return dict(
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            layer_pool_ids=layer_pool_ids,
        )
