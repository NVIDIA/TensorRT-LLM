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

import cuda.bindings.driver as cuda_driver
import torch
import triton

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheCompressionManager
from tensorrt_llm._torch.utils import next_positive_power_of_2
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger

from ..compaction import build_compaction_params, compact
from .triattention_kernels import (
    _fold_union_ranks_kernel,
    _gather_mean_phase_kernel,
    _settle_ties_kernel,
    prepare_per_head_scores,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests
    from tensorrt_llm.llmapi.llm_args import TriAttentionKvCacheCompressionConfig


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})

# Generation requests skipped by every eviction step.
_SKIP_REQUEST_STATES = (
    LlmRequestState.GENERATION_COMPLETE,
    LlmRequestState.CONTEXT_INIT,
)

# Upper bound of the geometric integration offset ladder [1, 2, 4, ...].
_MAX_INTEGRATION_OFFSET = 65536


class _EvictionInput(NamedTuple):
    """One due request's eviction operands for a single round."""

    request: "LlmRequest"
    target_cache: object
    draft_cache: Optional[object]
    state: Dict[str, object]
    source_length: int
    logical_source_length: int
    prompt_length: int
    target_tail_length: int


def _allocate_block_offset_staging(
    anchor_pool: torch.Tensor,
    *,
    num_pools: int,
    request_capacity: int,
    token_capacity: int,
    max_source_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One pinned host snapshot + persistent device table pair in the native
    V2 ``[pool, request, K/V, block]`` layout (block width 4-aligned for the
    ``PackedInt`` copy ABI); the device follows the anchor KV pool. The staged
    width is clamped to the manager's live source-table width: static bucket
    slack never holds valid tokens and the native gather copies the full width."""
    tokens_per_block = int(anchor_pool.shape[3])
    page_count = (token_capacity + tokens_per_block - 1) // tokens_per_block
    staged_blocks_per_seq = min((page_count + 3) // 4 * 4, int(max_source_blocks))
    shape = (num_pools, request_capacity, 2, staged_blocks_per_seq)
    host = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned())
    device_table = torch.empty(shape, dtype=torch.int32, device=anchor_pool.device)
    return host, device_table


_MEAN_PHASE_MAX_ROWS = 1 << 24


def grow_mean_phase_table(phase: Dict[str, object], rows: int) -> None:
    """Cover positions ``[0, rows)``, rebuilding the table if it must grow."""
    rows = int(rows)
    if rows <= phase["rows"]:
        return
    if rows > _MEAN_PHASE_MAX_ROWS:
        raise ValueError(f"a {rows}-row mean-phase table exceeds the exact-FP32 position range")
    target = 1
    while target < rows:
        target *= 2
    target = min(max(target, 2 * phase["rows"]), _MEAN_PHASE_MAX_ROWS)
    omega = phase["omega"]
    positions = torch.arange(target, device=omega.device, dtype=torch.float32)
    cos_table = torch.zeros((target, omega.numel()), dtype=torch.float32, device=omega.device)
    sin_table = torch.zeros_like(cos_table)
    # Fixed summation order keeps the table bit-stable across rebuilds.
    for offset in phase["offset_values"]:
        angle = torch.outer(positions + offset, omega)
        cos_table += torch.cos(angle)
        sin_table += torch.sin(angle)
    scale = 1.0 / len(phase["offset_values"])
    phase["cos"] = cos_table.mul_(scale)
    phase["sin"] = sin_table.mul_(scale)
    phase["rows"] = target


class TriAttention(KVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring."""

    adjusts_generation_kv_length = True

    # ---- construction ----

    def __init__(
        self,
        config: "TriAttentionKvCacheCompressionConfig",
        kv_cache_manager: KVCacheManagerV2,
        draft_kv_cache_manager: Optional[KVCacheManagerV2] = None,
    ):
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        self.budget = config.budget
        self.beta = config.beta
        self.eviction_mode = config.eviction_mode
        self.normalize_scores = bool(config.normalize_scores)
        if self.eviction_mode == "union" and not self.normalize_scores:
            logger.warning(
                "TriAttention union eviction always z-normalizes scores; "
                "forcing normalize_scores=True"
            )
            self.normalize_scores = True
        # Prompt always pinned; budget counts decode tokens only.
        self.model_path = config.model_path
        self.calibration_path = config.calibration_path
        self._load_calibration()

        # Mean-phase table dict; buffer builds bind its device tables in place.
        self._phase: Optional[Dict[str, object]] = None

        # Per-request eviction progress.
        self._request_states: Dict[int, Dict[str, object]] = {}
        # In-flight overlap batch reference; membership resolves lazily.
        self._inflight_scheduled_batch: Optional[object] = None
        self._inflight_generation_request_ids: Optional[set] = None
        # Manager-lifetime constants.
        self._num_extra_kv_tokens = int(kv_cache_manager.num_extra_kv_tokens)
        self._protected_tail_capacity = (
            int(kv_cache_manager.num_extra_kv_tokens)
            + int(kv_cache_manager._kv_reserve_draft_tokens)
            + 1
        )
        self._draft_protected_tail_capacity: Optional[int] = None
        if draft_kv_cache_manager is not None:
            self._draft_protected_tail_capacity = (
                int(draft_kv_cache_manager.num_extra_kv_tokens)
                + int(draft_kv_cache_manager._kv_reserve_draft_tokens)
                + 1
            )
        self._generation_growth = 1 + int(kv_cache_manager._kv_reserve_draft_tokens)
        # Lazy resident eviction runtime: built at the first eviction, reused
        # across rounds, and replaced as a whole when a capacity axis grows.
        self._buffers_built = False
        # Round-ordering events: device-lifetime, created at the first build
        # and reused across capacity rebuilds.
        self._staging_reuse_event: Optional[torch.cuda.Event] = None
        self._block_offsets_ready_event: Optional[torch.cuda.Event] = None
        self._compaction_done_event: Optional[torch.cuda.Event] = None
        # Manager-lifetime layer facts, resolved once: V2 fixes pp_layers at
        # construction and the model config is immutable on disk.
        self._global_layers = [int(layer) for layer in kv_cache_manager.pp_layers]
        self._layer_partition = self._attention_layer_partition()
        # Target/draft runtime KV layouts, cached by the one resolver.
        self._kv_layout_caches: Dict[bool, Optional[Dict[str, object]]] = {
            False: None,
            True: None,
        }

    def _attention_layer_partition(self) -> Tuple[List[int], List[int], Optional[int]]:
        """SWA layers here are stored at full length; the window applies only in the kernel."""
        model_path = self.model_path
        global_layers = self._global_layers
        num_layers = len(global_layers)

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        config_values = config.get_text_config().to_dict()
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
        calibration = self._resolve_calibration()
        self._freq_scale_sq = calibration["freq_scale_sq"].to(dtype=torch.float32)
        self._omega = calibration["omega"]
        # Pre-split query stats + MLR coefficient, shapes [L, H, F].
        e_q = calibration["E_q"]
        self._calibration_q_real = e_q.real.to(torch.float32).contiguous()
        self._calibration_q_imag = e_q.imag.to(torch.float32).contiguous()
        self._calibration_mlr_coef = (
            calibration["E_q_norm"].to(torch.float32) - e_q.abs().to(torch.float32)
        ).contiguous()

    def _resolve_calibration(self) -> Dict[str, torch.Tensor]:
        """Load the calibration file, converting the official layout if needed."""
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            return raw
        if isinstance(raw, dict) and {"metadata", "stats"} <= set(raw):
            return self._convert_official_calibration(raw)
        got = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(
            f"Unrecognized calibration at {self.calibration_path}: expected the "
            f"official {{metadata, stats}} layout or "
            f"{sorted(_REQUIRED_CALIBRATION_KEYS)}; got {got}."
        )

    def _convert_official_calibration(self, raw) -> Dict[str, torch.Tensor]:
        """Convert the official calibration format to the runtime schema."""
        stats = raw["stats"]
        meta = raw["metadata"]
        if "sampled_heads" in meta:
            heads = [(int(a), int(b)) for a, b in meta["sampled_heads"]]
        else:
            heads = [
                (int(k[len("layer") : k.index("_head")]), int(k[k.index("_head") + len("_head") :]))
                for k in stats
            ]
        num_layers = max(layer for layer, _ in heads) + 1
        num_heads = max(h for _, h in heads) + 1
        freq_count = int(next(iter(stats.values()))["q_mean_real"].numel())
        E_q = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64)
        E_q_norm = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.float32)
        for layer, h in heads:
            s = stats[f"layer{layer:02d}_head{h:02d}"]
            E_q[layer, h] = torch.complex(s["q_mean_real"].float(), s["q_mean_imag"].float())
            E_q_norm[layer, h] = s["q_abs_mean"].float()
        omega, freq_scale_sq = self._rope_tables(freq_count)
        calib = {
            "E_q": E_q,
            "E_q_norm": E_q_norm,
            "omega": omega,
            "freq_scale_sq": freq_scale_sq,
        }
        logger.info(
            f"TriAttention: converted official calibration {self.calibration_path}"
            f" -> E_q[L={num_layers}, H={num_heads}, F={freq_count}]"
        )
        return calib

    def _rope_tables(self, freq_count: int):
        """Derive the RoPE frequency tables from the model config."""
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        ).get_text_config()
        # transformers >= 5.5 folds rope_theta/rope_type into rope_parameters.
        rope_params = config.to_dict()["rope_parameters"]
        if all(isinstance(value, dict) for value in rope_params.values()):
            raise ValueError(
                f"TriAttention does not support per-layer-type rope parameters ({self.model_path})"
            )
        rope_type = rope_params["rope_type"]
        if rope_type == "default":
            # "default" has no ROPE_INIT_FUNCTIONS entry; the analytic formula is its definition.
            head_dim = freq_count * 2
            base = float(rope_params["rope_theta"])
            positions = torch.arange(0, head_dim, 2, dtype=torch.float32)
            omega = (1.0 / (base ** (positions / head_dim)))[:freq_count].clone()
            scale_sq = 1.0
        else:
            inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](config, device="cpu")
            omega = inv_freq.to(torch.float32)[:freq_count].clone()
            scale_sq = float(attention_factor) ** 2
        return omega, torch.full((freq_count,), scale_sq, dtype=torch.float32)

    # ---- framework hooks (call order) ----

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Register the request for eviction tracking."""
        self._validate_request_capacity(request)
        self._request_states[request.py_request_id] = {
            "generation_steps": 0,
            "evicted_tokens": 0,
        }

    def on_generation_step_begin(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Snapshot the in-flight batch; mutation remains in final update."""
        self._inflight_scheduled_batch = scheduled_batch
        self._inflight_generation_request_ids = None

    def on_generation_step_end(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Compact after native KV-cache updates have finalized this iteration
        (must run after KVCacheManagerV2 so capacity reflects the written token and any rewind)."""
        with nvtx_range_debug("triattention.generation_step_end", color="blue"):
            self._evict_due_requests(scheduled_batch)

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's eviction state; the buffers stay resident."""
        self._request_states.pop(request.py_request_id, None)

    # ---- request capacity ----

    def _validate_request_capacity(self, request: "LlmRequest") -> None:
        """Reject a request whose pre-first-eviction peak cannot fit (only
        TriAttention can compute it: the framework's dense guards are off)."""
        speculative_overshoot = int(self.kv_cache_manager.max_draft_len)
        first_eviction_decode_length = (
            self.budget // self.beta + 1
        ) * self.beta + speculative_overshoot
        decode_capacity = min(int(request.py_max_new_tokens), first_eviction_decode_length)
        confirmed_capacity = int(request.py_prompt_len) + decode_capacity
        checked = [(self.kv_cache_manager, self._protected_tail_capacity, "target")]
        if self.draft_kv_cache_manager is not None:
            checked.append(
                (self.draft_kv_cache_manager, self._draft_protected_tail_capacity, "draft")
            )
        for manager, protected_tail, label in checked:
            required_capacity = confirmed_capacity + protected_tail
            pool_capacity = manager.get_num_available_tokens(
                token_num_upper_bound=confirmed_capacity,
                max_num_draft_tokens=int(manager._kv_reserve_draft_tokens) + 1,
            )
            table_capacity = manager.max_blocks_per_seq * manager.tokens_per_block
            if confirmed_capacity > pool_capacity or required_capacity > table_capacity:
                raise ValueError(
                    f"TriAttention {label} KV capacity is too small to reach the first "
                    f"eviction: request requires {required_capacity} tokens "
                    f"(prompt={request.py_prompt_len}, budget={self.budget}, "
                    f"beta={self.beta}, protected tail={protected_tail}), but the "
                    f"V2 pool covers {pool_capacity + protected_tail} tokens and "
                    f"its page table covers {table_capacity} tokens"
                )

    # ---- eviction round ----

    def _evict_due_requests(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Owner of the full eviction transaction: admission, cadence, launch,
        publication, and cache resize."""
        manager = self.kv_cache_manager
        eviction_inputs: List[_EvictionInput] = []
        with nvtx_range("triattention.metadata", color="cyan"):
            for request in scheduled_batch.generation_requests:
                if request.is_dummy or request.state in _SKIP_REQUEST_STATES:
                    continue
                request_id = request.py_request_id
                target_cache = manager.kv_cache_map.get(request_id)
                if target_cache is None or not target_cache.is_active:
                    # Overlap scheduling may suspend a cache mid-flight; defer
                    # this request (pre-launch) instead of failing the batch.
                    continue
                # Cadence gate first; capacity math and consistency raises run in the due branch.
                state = self._request_states[request_id]
                previous_step = state["generation_steps"]
                step = previous_step + 1 + int(request.py_num_accepted_draft_tokens)
                state["generation_steps"] = step
                if previous_step // self.beta >= step // self.beta:
                    continue
                # Speculative reserve + in-flight overlap growth: contiguous tail moved byte-for-byte.
                target_tail_length = self._num_extra_kv_tokens + (
                    self._inflight_generation_growth(scheduled_batch, request_id)
                )
                source_length = int(target_cache.capacity) - target_tail_length
                if source_length < target_cache.history_length:
                    raise RuntimeError(
                        f"Request {request_id} KV length {source_length} is below "
                        f"finalized history {target_cache.history_length}"
                    )
                prompt_length = int(request.py_prompt_len)
                if source_length <= prompt_length + self.budget:
                    # Selection would be an identity: nothing to evict yet.
                    continue
                draft_cache = None
                if self.draft_kv_cache_manager is not None:
                    # A missing draft cache is a wiring bug: the dict's KeyError is the report.
                    draft_cache = self.draft_kv_cache_manager.kv_cache_map[request_id]
                    if not draft_cache.is_active:
                        # Target and draft defer together (pre-launch).
                        continue
                eviction_inputs.append(
                    _EvictionInput(
                        request=request,
                        target_cache=target_cache,
                        draft_cache=draft_cache,
                        state=state,
                        source_length=source_length,
                        # Uncompressed logical position.
                        logical_source_length=source_length + state["evicted_tokens"],
                        prompt_length=prompt_length,
                        target_tail_length=target_tail_length,
                    )
                )
        if not eviction_inputs:
            return

        with nvtx_range_debug("triattention.resolve_layout", color="blue"):
            target_layout = self._runtime_kv_layout()
            draft_layout = (
                self._runtime_kv_layout(draft=True)
                if self.draft_kv_cache_manager is not None
                else None
            )
        with nvtx_range_debug("triattention.staging_lookup", color="blue"):
            # Retained spans always cover the model window (construction rejects budget < window).
            self._ensure_eviction_runtime(target_layout, draft_layout, eviction_inputs)
        # Ungated NVTX: the due count in the message shows each round's size.
        with nvtx_range(
            f"triattention.evict_request_group reqs={len(eviction_inputs)}",
            color="purple",
        ):
            self._execute_eviction_round(eviction_inputs)
        for item in eviction_inputs:
            evicted = item.source_length - item.prompt_length - self.budget
            item.state["evicted_tokens"] += evicted
            # The manager's only channel to the runtime (feeds num_cached_tokens_per_seq).
            item.request.py_num_compressed_tokens = item.state["evicted_tokens"]
        self._resize_compacted_caches(eviction_inputs)

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        inflight = self._inflight_scheduled_batch
        if inflight is None or scheduled_batch is inflight:
            return 0
        member_ids = self._inflight_generation_request_ids
        if member_ids is None:
            member_ids = {request.py_request_id for request in inflight.generation_requests}
            self._inflight_generation_request_ids = member_ids
        if request_id not in member_ids:
            return 0
        return self._generation_growth

    def _execute_eviction_round(
        self,
        eviction_inputs: Sequence[_EvictionInput],
    ) -> None:
        """Run one eviction round over the due cohort (every launch covers
        the full request capacity; padded rows carry zero lengths and stay inert)."""
        manager = self.kv_cache_manager
        draft_manager = self.draft_kv_cache_manager
        with nvtx_range_debug("triattention.page_table_stage", color="orange"):
            request_ids = [item.request.py_request_id for item in eviction_inputs]
            logical_source_lengths = [item.logical_source_length for item in eviction_inputs]
            prompt_lengths = [item.prompt_length for item in eviction_inputs]
            source_lengths = [item.source_length for item in eviction_inputs]
            dense_move_offsets, swa_move_offsets, draft_move_offsets = (
                self._compute_compaction_move_offsets(eviction_inputs)
            )
            stream = torch.cuda.current_stream(self._block_offsets_device.device)
            # int32 gate before any buffer or device work: the in-place numpy writes below wrap silently.
            max_logical_source_length = max(logical_source_lengths)
            rows = (
                (0, logical_source_lengths),
                (1, source_lengths),
                (2, prompt_lengths),
                (3, dense_move_offsets),
                (4, swa_move_offsets),
                (5, draft_move_offsets),
            )
            for row, values in rows:
                if (
                    values is not None
                    and not -0x80000000 <= min(values) <= max(values) <= 0x7FFFFFFF
                ):
                    raise ValueError(f"staged metadata row {row} exceeds the int32 range")
            # Host-staging reuse fence: prior cohort's async copies must finish before the pinned rows are rewritten.
            self._staging_reuse_event.synchronize()
            host_table = self._request_metadata_host_np
            for row, values in rows:
                if values is not None:
                    host_table[row, : len(values)] = values
            # Zero lengths keep the score kernel and selection inert for padded rows.
            host_table[:3, len(eviction_inputs) :] = 0
            grow_mean_phase_table(self._phase, int(max_logical_source_length) + 1)
            self._stage_block_offsets(
                manager,
                request_ids,
                self._block_offsets_host,
                self._block_offsets_device,
            )
            if draft_manager is not None:
                self._stage_block_offsets(
                    draft_manager,
                    request_ids,
                    self._draft_block_offsets_host,
                    self._draft_block_offsets_device,
                )
            try:
                self._request_metadata_device.copy_(self._request_metadata_host, non_blocking=True)
            finally:
                # Guards the pinned staging until the asynchronous copies complete.
                self._staging_reuse_event.record(stream)
        request_count = len(eviction_inputs)
        union = self.eviction_mode == "union"
        try:
            with nvtx_range("triattention.score", color="blue"):
                # In-place refresh: the compiled score launches captured these pointers.
                _gather_mean_phase_kernel[(request_count,)](
                    self._logical_source_lengths_device,
                    self._phase["cos"],
                    self._phase["sin"],
                    self._source_lengths_device,
                    self._prompt_lengths_device,
                    self._mean_cos,
                    self._mean_sin,
                    self._decode_lengths_device,
                    self._swa_destination_bases,
                    self._swa_rebase_delta,
                    NUM_FREQS=self._phase_num_freqs,
                    F_BLOCK=self._phase_f_block,
                    HAS_SWA=self._swa_destination_bases is not None,
                    num_warps=1,
                )
                cu_stream = cuda_driver.CUstream(stream.cuda_stream)
                self._compiled_score_by_request_count[request_count](
                    *self._cute_score_prefix,
                    self._cute_mean_cos,
                    self._cute_mean_sin,
                    *self._cute_score_tail,
                    request_count,
                    cu_stream,
                )
                if union:
                    # Normalized union reduction, written straight into the selection rows.
                    self._compiled_normalize_union_by_request_count[request_count](
                        self._cute_partial_stats,
                        *self._cute_selection_prefix,
                        self._cute_selection_scores_rows,
                        request_count,
                        cu_stream,
                    )
                    if self._union_tp_mapping is not None:
                        # Max-fold the rank-local unions into the global union (exact:
                        # max is order-free), so every rank keeps the same ordinals.
                        gathered = allgather(
                            self._selection_scores_rows[:request_count],
                            self._union_tp_mapping,
                            dim=0,
                        )
                        _fold_union_ranks_kernel[(request_count, self._fold_width_blocks)](
                            gathered,
                            self._selection_scores_rows,
                            request_count,
                            TP_SIZE=self._union_tp_size,
                            WIDTH=self._selection_width_capacity,
                        )
            with nvtx_range("triattention.select", color="yellow"):
                if not union:
                    # Per-head reduces read each decode window straight out of the scratch.
                    prepare_per_head_scores(
                        self._score_scratch,
                        self._decode_lengths_device,
                        self._prompt_lengths_device,
                        self._row_mean,
                        self._row_inv_std,
                        self._selection_scores_rows,
                        self._selection_row_lengths,
                        request_count=request_count,
                        num_layers=self._num_layers,
                        num_q_heads=self._num_q_heads,
                        num_kv_heads=self._num_kv_heads,
                        padded_head_columns=self._padded_head_columns,
                        score_token_capacity=self._score_token_capacity,
                        selection_width=self._selection_width_capacity,
                        per_layer=self.eviction_mode == "per_layer_perhead",
                        normalize_scores=self.normalize_scores,
                    )
                self._settle_top_tokens(request_count)
            with nvtx_range("triattention.compact", color="purple"):
                compact(self._compaction_params, request_count)
        finally:
            # Order V2 page-table reuse and resize after this cohort's compact.
            self._compaction_done_event.record(stream)
            manager._stream.wait_event(self._compaction_done_event)
            if draft_manager is not None:
                draft_manager._stream.wait_event(self._compaction_done_event)

    def _compute_compaction_move_offsets(
        self,
        eviction_inputs: Sequence[_EvictionInput],
    ) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        """Cumulative dense/SWA/draft move offsets for one due cohort (keep
        set plus protected tail per request; rows past the cohort repeat the final
        offset and contribute no moves)."""

        def padded_offsets(moves_per_request: List[int]) -> List[int]:
            offsets = [0]
            for moves in moves_per_request:
                offsets.append(offsets[-1] + moves)
            offsets.extend(offsets[-1:] * (self._request_capacity - len(moves_per_request)))
            return offsets

        tails = [int(item.target_tail_length) for item in eviction_inputs]
        dense = padded_offsets([self._keep_count + tail for tail in tails])
        swa = None
        if self._swa_window is not None:
            swa = padded_offsets([self._swa_window + tail for tail in tails])
        draft = None
        if self._draft_protected_tail_capacity is not None:
            draft = padded_offsets(
                [self._keep_count + self._draft_protected_tail_capacity] * len(eviction_inputs)
            )
        return dense, swa, draft

    def _stage_block_offsets(
        self,
        manager: KVCacheManagerV2,
        request_ids: List[int],
        host_block_offsets: torch.Tensor,
        device_block_offsets: torch.Tensor,
    ) -> None:
        """Gather the pinned snapshot before the async device copy: resize mutates
        the live host table. The round owner has already fenced host-staging reuse."""
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
            manager._stream.cuda_stream,
        )
        self._block_offsets_ready_event.record(manager._stream)
        torch.cuda.current_stream(device_block_offsets.device).wait_event(
            self._block_offsets_ready_event
        )

    def _settle_top_tokens(self, request_count: int) -> None:
        """Pick the top-k and settle ties into the kept-ordinal decision rows
        (the compaction contract packs them into move sources)."""
        rows = request_count * self._selection_rows_per_request
        # The trailing 1 is next_n: decode scores one query token per request.
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            self._selection_scores_rows[:rows],
            self._selection_row_lengths[:rows],
            self._provisional_rows[:rows],
            self._keep_count,
            1,
        )
        _settle_ties_kernel[(request_count, self._selection_rows_per_request)](
            self._selection_scores_rows,
            self._selection_row_lengths,
            self._prompt_lengths_device,
            self._provisional_rows,
            self._kept_ordinal_rows,
            WIDTH=self._selection_width_capacity,
            KEEP_COUNT=self._keep_count,
            SELECTION_ROWS=self._selection_rows_per_request,
        )

    def _resize_compacted_caches(self, eviction_inputs) -> None:
        with nvtx_range("triattention.resize", color="red"):
            with nvtx_range_debug("triattention.v2_resize", color="red"):
                families = [("target", "target_cache", None)]
                if self.draft_kv_cache_manager is not None:
                    # Same kept set: the draft shrinks to the same retained
                    # length plus its own fixed tail.
                    families.append(("draft", "draft_cache", self._draft_protected_tail_capacity))
                for label, cache_key, fixed_tail in families:
                    for item in eviction_inputs:
                        cache = getattr(item, cache_key)
                        request_id = item.request.py_request_id
                        tail = item.target_tail_length if fixed_tail is None else fixed_tail
                        resized_capacity = item.prompt_length + self.budget + tail
                        if not cache.resize(resized_capacity, None):
                            raise RuntimeError(
                                f"Failed to resize compacted {label} KV cache for "
                                f"request {request_id} to {resized_capacity} tokens"
                            )

    # ---- buffers + layout ----

    def _ensure_eviction_runtime(
        self,
        target_layout: Dict[str, object],
        draft_layout: Optional[Dict[str, object]],
        eviction_inputs: Sequence[_EvictionInput],
    ) -> None:
        """Per-round reuse gate over the three capacity axes; first round or
        growth replaces the resident runtime as a whole."""
        # Empty cohorts never reach here: _evict_due_requests no-ops pre-launch.
        needed_width = max(item.source_length - item.prompt_length for item in eviction_inputs)
        needed_score_tokens = max(item.source_length for item in eviction_inputs)
        needed_requests = len(eviction_inputs)
        if self._buffers_built:
            if (
                needed_width <= self._selection_width_capacity
                and needed_score_tokens <= self._score_token_capacity
                and needed_requests <= self._request_capacity
            ):
                return
            # This round outgrew the runtime: wait out the prior round (its
            # completion event orders after every use of the old epoch), then rebuild.
            if self._compaction_done_event is not None:
                self._compaction_done_event.synchronize()
            self._buffers_built = False

        needed_page_tokens = max(
            item.source_length + item.target_tail_length for item in eviction_inputs
        )
        manager = self.kv_cache_manager
        request_capacity = max(needed_requests, int(manager.max_batch_size))
        selection_width_capacity = max(
            needed_width,
            self.budget + 2 * self.beta + int(manager.max_total_draft_tokens or 0),
        )
        # Bucket sized by the presented cohorts, NOT max_seq_len (a floor there breaks 32-bit indexing).
        score_token_capacity = next_positive_power_of_2(max(int(needed_page_tokens), 1024))
        score_token_capacity = min(
            score_token_capacity, max(int(manager.max_seq_len), int(needed_page_tokens))
        )
        # The bucket capacity must be tile-aligned (mis-tiling stripes the
        # score scratch silently); the ceiling division constructs that fact.
        score_tile_tokens = max(64, int(manager.tokens_per_block))
        score_token_capacity = -(-score_token_capacity // score_tile_tokens) * score_tile_tokens

        first_pool = target_layout["layer_pools"][target_layout["dense_layers"][0]]
        if self._phase is None:
            # Host-only width offsets for the table builder (no device copy).
            self._phase = {
                "omega": self._omega.to(device=first_pool.device, dtype=torch.float32).contiguous(),
                "offset_values": [
                    float(1 << i) for i in range(_MAX_INTEGRATION_OFFSET.bit_length())
                ],
                "cos": None,
                "sin": None,
                "rows": 0,
            }
            grow_mean_phase_table(self._phase, int(score_token_capacity))
        self._rebuild_eviction_runtime(
            target_layout,
            draft_layout,
            request_capacity=request_capacity,
            score_token_capacity=score_token_capacity,
            selection_width_capacity=selection_width_capacity,
        )
        self._buffers_built = True

    def _rebuild_eviction_runtime(
        self,
        target_layout: Dict[str, object],
        draft_layout: Optional[Dict[str, object]],
        *,
        request_capacity: int,
        score_token_capacity: int,
        selection_width_capacity: int,
    ) -> None:
        """Build the resident eviction runtime for one capacity epoch as attributes
        (compiled kernels capture raw pool addresses: pools must stay alive and stay put)."""
        import cutlass
        import cutlass.cute as cute

        from .triattention_cute_score_fused import (
            _COMPILE_LOCK,
            _COMPILED_KERNELS,
            PADDED_HEAD_COLUMNS,
            SMALL_WORKLOAD_PAGE_SHARDS,
            STATS_FIELDS,
            _encode_tma_descriptors,
            _tensor_spec,
            _to_cute,
            _TriAttentionScoreKernel,
        )

        layer_pools = target_layout["layer_pools"]
        dense_layers = list(target_layout["dense_layers"])
        swa_layers = list(target_layout["swa_layers"])
        swa_window = target_layout["swa_window"]
        # Canonical layer -> V2 pool id tuple; it IS the staged plane slot map.
        layer_pool_ids = tuple(target_layout["layer_pool_ids"])
        num_page_table_slots = int(self.kv_cache_manager.num_pools)

        # The first dense layer anchors device and staging geometry.
        anchor_pool = layer_pools[dense_layers[0]]
        device = anchor_pool.device
        request_capacity = int(request_capacity)
        score_token_capacity = int(score_token_capacity)
        selection_width_capacity = int(selection_width_capacity)
        keep_count = int(self.budget)
        protected_tail_capacity = int(self._protected_tail_capacity)
        page_table_token_capacity = score_token_capacity + protected_tail_capacity

        q_real, q_imag, mlr_coef = self._local_score_calibration(target_layout["global_layers"])
        # TP splits attention heads contiguously per rank; the calibration is global.
        mapping = self.kv_cache_manager.mapping
        tp_size = 1 if mapping.enable_attention_dp else int(mapping.tp_size)
        if tp_size > 1:
            local_q_heads = int(q_real.shape[1]) // tp_size
            heads = slice(mapping.tp_rank * local_q_heads, (mapping.tp_rank + 1) * local_q_heads)
            q_real, q_imag, mlr_coef = q_real[:, heads], q_imag[:, heads], mlr_coef[:, heads]
        # Union reduces over ALL heads: rank-local rows are max-folded across the
        # TP group each round so the kept set matches the single-rank algorithm.
        self._union_tp_mapping = (
            mapping if (self.eviction_mode == "union" and tp_size > 1) else None
        )
        self._union_tp_size = tp_size
        q_real, q_imag, mlr_coef, freq_scale_sq = (
            tensor.to(device=device, dtype=torch.float32).contiguous()
            for tensor in (q_real, q_imag, mlr_coef, self._freq_scale_sq)
        )
        num_q_heads = int(q_real.shape[1])
        num_freqs = int(q_real.shape[2])

        self._request_capacity = request_capacity
        self._score_token_capacity = score_token_capacity
        self._selection_width_capacity = selection_width_capacity
        self._fold_width_blocks = triton.cdiv(selection_width_capacity, 1024)
        self._keep_count = keep_count

        # ---- block-offset staging (target, plus the co-compressed draft) -------
        self._block_offsets_host, self._block_offsets_device = _allocate_block_offset_staging(
            anchor_pool,
            num_pools=num_page_table_slots,
            request_capacity=request_capacity,
            token_capacity=page_table_token_capacity,
            max_source_blocks=int(self.kv_cache_manager.host_kv_cache_block_offsets.shape[-1]),
        )
        # The draft is never scored: these offsets feed only the draft compacts.
        self._draft_block_offsets_device = None
        self._draft_block_offsets_host = None
        if draft_layout is not None:
            draft_representatives = list(draft_layout["pool_representatives"])
            draft_anchor_pool = draft_layout["layer_pools"][draft_representatives[0]]
            # Construction-boundary invariant: the round shares one stream/event
            # contract, so the draft pools must live on the target device.
            if draft_anchor_pool.device != device:
                raise RuntimeError(
                    "TriAttention draft KV pools must share the target KV pool device"
                )
            self._draft_block_offsets_host, self._draft_block_offsets_device = (
                _allocate_block_offset_staging(
                    draft_anchor_pool,
                    num_pools=int(self.draft_kv_cache_manager.num_pools),
                    request_capacity=request_capacity,
                    token_capacity=score_token_capacity + int(self._draft_protected_tail_capacity),
                    max_source_blocks=int(
                        self.draft_kv_cache_manager.host_kv_cache_block_offsets.shape[-1]
                    ),
                )
            )

        # ---- per-round metadata table: one H2D copy; move-offsets rows need the +1 column ----
        self._request_metadata_host = torch.empty(
            (6, request_capacity + 1), dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        # numpy view over the pinned rows: per-round staging writes lists in place.
        self._request_metadata_host_np = self._request_metadata_host.numpy()
        self._identity_copy_indices_host = torch.arange(
            request_capacity, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        # Zero-filled: an unstaged cohort must gather the phase table's row 0.
        self._request_metadata_device = torch.zeros(
            (6, request_capacity + 1), dtype=torch.int32, device=device
        )
        self._logical_source_lengths_device = self._request_metadata_device[0, :request_capacity]
        self._source_lengths_device = self._request_metadata_device[1, :request_capacity]
        # Pinned per-request decode-window starts.
        self._prompt_lengths_device = self._request_metadata_device[2, :request_capacity]
        dense_move_offsets_row = self._request_metadata_device[3]
        swa_move_offsets_row = self._request_metadata_device[4]
        draft_move_offsets_row = self._request_metadata_device[5]
        # SWA staging geometry, bound once (the compaction plans stay opaque):
        # the phase gather rebases each request's SWA destination base in place.
        self._swa_window = int(swa_window) if swa_layers else None
        self._swa_destination_bases = (
            torch.empty_like(self._prompt_lengths_device) if swa_layers else None
        )
        self._swa_rebase_delta = keep_count - self._swa_window if swa_layers else 0
        self._mean_cos = torch.empty(
            (request_capacity, num_freqs), dtype=torch.float32, device=device
        )
        self._mean_sin = torch.empty_like(self._mean_cos)
        self._phase_num_freqs = int(self._phase["omega"].numel())
        self._phase_f_block = triton.next_power_of_2(self._phase_num_freqs)

        # ---- score state: one fused group across all dense layers --------------
        _, _, num_kv_heads, tokens_per_block, _ = anchor_pool.shape
        self._num_layers = len(dense_layers)
        self._num_q_heads = int(num_q_heads)
        self._num_kv_heads = int(num_kv_heads)
        dense_layer_slots = [layer_pool_ids[layer] for layer in dense_layers]
        seg_req_id = torch.arange(
            request_capacity, dtype=torch.int32, device=device
        ).repeat_interleave(self._num_layers)
        seg_layer_id = torch.tensor(list(dense_layers), dtype=torch.int32, device=device).repeat(
            request_capacity
        )
        block_offsets = self._block_offsets_device
        slots_t = torch.tensor(dense_layer_slots, dtype=torch.int64, device=device)
        req_idx = seg_req_id.to(torch.int64)
        slot_idx = slots_t.repeat(request_capacity)
        seg_page_off = slot_idx * block_offsets.stride(0) + req_idx * block_offsets.stride(1)

        max_segments = request_capacity * self._num_layers
        # The score plane must stay 32-bit indexable (wraparound = silent wild read).
        if (PADDED_HEAD_COLUMNS - 1) * max_segments * score_token_capacity >= 2**31:
            raise ValueError(
                "score bucket overflows the 32-bit score plane: "
                f"{(PADDED_HEAD_COLUMNS - 1) * max_segments * score_token_capacity}"
            )
        # Persistent buffers: the compiled kernels capture their device pointers.
        self._padded_head_columns = PADDED_HEAD_COLUMNS
        self._score_scratch = torch.empty(
            self._num_kv_heads * PADDED_HEAD_COLUMNS * max_segments * score_token_capacity,
            dtype=torch.float32,
            device=device,
        )
        # int32 is safe here: covered by the 2^31 score-plane audit above.
        seg_out_offset = (
            torch.arange(max_segments, dtype=torch.int64, device=device) * score_token_capacity
        ).to(torch.int32)
        union = self.eviction_mode == "union"

        # ---- score path: compiled per request-count/page-shard ----
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        # Per-shard partial score statistics.
        partial_stats_elements = (
            request_capacity
            * self._num_layers
            * num_q_heads
            * SMALL_WORKLOAD_PAGE_SHARDS
            * STATS_FIELDS
            if union
            else 1
        )
        self._partial_stats = torch.empty(
            partial_stats_elements,
            dtype=torch.float32,
            device=device,
        )
        self._tma_descriptors = _encode_tma_descriptors(
            list(layer_pools),
            [int(layer) for layer in dense_layers],
            int(num_freqs),
            int(tokens_per_block),
        )
        # Alignment sits with the operand it describes; source_lengths and
        # prompt_lengths are only 4-byte-aligned row views, read as per-CTA scalars.
        prefix_operands = (
            (block_offsets.view(-1), 16),
            (seg_page_off, 16),
            (seg_req_id, 16),
            (seg_layer_id, 16),
            (self._source_lengths_device, 4),
            (seg_out_offset, 16),
            (self._prompt_lengths_device, 4),
            (q_real.view(-1), 16),
            (q_imag.view(-1), 16),
            (mlr_coef.view(-1), 16),
        )
        torch_prefix = tuple(tensor for tensor, _ in prefix_operands)
        torch_tail = (
            freq_scale_sq,
            self._score_scratch,
            self._partial_stats,
            anchor_pool,
            self._tma_descriptors,
        )
        # No keep-alive twin: each cute handle below owns its operand's DLPack
        # capsule, which retains the underlying torch storage.
        self._cute_score_prefix = tuple(
            _to_cute(tensor, assumed_align=align) for tensor, align in prefix_operands
        )
        self._cute_score_tail = (
            _to_cute(freq_scale_sq),
            _to_cute(self._score_scratch),
            _to_cute(self._partial_stats),
            _to_cute(anchor_pool),
            _to_cute(self._tma_descriptors, assumed_align=128),
        )
        # Build-bound persistent launch operands: refreshed in place each round.
        self._cute_mean_cos = _to_cute(self._mean_cos.view(-1))
        self._cute_mean_sin = _to_cute(self._mean_sin.view(-1))
        self._compiled_score_by_request_count: Dict[int, object] = {}
        self._compiled_normalize_union_by_request_count: Dict[int, object] = {}
        page_shards_by_request_count: Dict[int, int] = {}
        self._cute_selection_prefix = (
            _to_cute(self._score_scratch),
            _to_cute(self._source_lengths_device, assumed_align=4),
            _to_cute(seg_out_offset),
            _to_cute(self._prompt_lengths_device, assumed_align=4),
        )
        self._cute_partial_stats = _to_cute(self._partial_stats)
        static_geometry = (
            request_capacity,
            self._num_layers,
            score_token_capacity,
            num_q_heads,
            self._num_kv_heads,
            num_freqs,
            int(tokens_per_block),
            tuple(int(value) for value in anchor_pool.shape),
            tuple(int(value) for value in anchor_pool.stride()),
        )
        tensor_specs = tuple(
            _tensor_spec(tensor)
            for tensor in (
                *torch_prefix,
                self._mean_cos.view(-1),
                self._mean_sin.view(-1),
                *torch_tail,
            )
        )
        variants = [(1, SMALL_WORKLOAD_PAGE_SHARDS)]
        if request_capacity > 1:
            variants.append((request_capacity, 2))
        # Per-head modes compile the score-only entry; union the fused pipeline.
        kernel_kwargs = dict(
            num_layers=self._num_layers,
            score_token_capacity=score_token_capacity,
            num_q_heads=num_q_heads,
            num_freqs=num_freqs,
            pool_shape=tuple(int(value) for value in anchor_pool.shape),
            pool_strides=tuple(int(value) for value in anchor_pool.stride()),
        )
        stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

        def _compiled_kernel(cache_key, build):
            with _COMPILE_LOCK:
                compiled = _COMPILED_KERNELS.get(cache_key)
                if compiled is None:
                    compiled = build()
                    _COMPILED_KERNELS[cache_key] = compiled
            return compiled

        variant_key = "triattention_cute_score_stats" if union else "triattention_cute_score"
        compiled_entries = self._compiled_score_by_request_count
        for request_count, page_shards in variants:
            cache_key = (
                variant_key,
                static_geometry,
                tensor_specs,
                request_count,
                page_shards,
            )
            compiled_entries[request_count] = _compiled_kernel(
                cache_key,
                lambda page_shards=page_shards: cute.compile(
                    _TriAttentionScoreKernel(
                        **kernel_kwargs,
                        page_shards=page_shards,
                        write_partial_stats=union,
                    ),
                    *self._cute_score_prefix,
                    self._cute_mean_cos,
                    self._cute_mean_sin,
                    *self._cute_score_tail,
                    cutlass.Int32(1),
                    stream,
                ),
            )
            page_shards_by_request_count[request_count] = page_shards

        if request_capacity > 1:
            small = compiled_entries[1]
            large = compiled_entries[request_capacity]
            for request_count in range(1, request_capacity + 1):
                # Give small cohorts the extra shard while the 2-shard grid stays under two waves.
                two_shard_ctas = request_count * self._num_layers * self._num_kv_heads * 2
                use_extra_score_shard = two_shard_ctas < 2 * sm_count
                compiled_entries[request_count] = small if use_extra_score_shard else large
                page_shards_by_request_count[request_count] = (
                    SMALL_WORKLOAD_PAGE_SHARDS if use_extra_score_shard else 2
                )

        logger.info(
            f"TriAttention CuTe score enabled: {self._num_q_heads}q/{self._num_kv_heads}kv heads, "
            f"{num_freqs} freqs, {int(tokens_per_block)}-token pages"
        )

        # ---- selection buffers (canonical row-major, one name per storage) -----
        self._decode_lengths_device = torch.full(
            (request_capacity,), selection_width_capacity, dtype=torch.int32, device=device
        )
        if union:
            self._selection_rows_per_request = 1
            self._selection_scores_rows = torch.empty(
                (request_capacity, selection_width_capacity), dtype=torch.float32, device=device
            )
            # One selection row per request: its length IS the staged valid width.
            self._selection_row_lengths = self._decode_lengths_device
            # Padded rows still need in-range ordinals for the finalizer's gather.
            self._provisional_rows = torch.zeros(
                (request_capacity, keep_count), dtype=torch.int32, device=device
            )
            # Kept decode ordinals.
            self._kept_ordinal_rows = torch.empty(
                (request_capacity, keep_count), dtype=torch.int32, device=device
            )
            self._cute_selection_scores_rows = _to_cute(self._selection_scores_rows.view(-1))
        else:
            selection_rows = (
                self._num_kv_heads
                if self.eviction_mode == "per_head"
                else self._num_layers * self._num_kv_heads
            )
            # The selection rectangle must stay 32-bit indexable (wraparound = wild reads).
            selection_rect = (
                request_capacity * selection_rows * max(selection_width_capacity, keep_count)
            )
            if selection_rect >= 2**31:
                raise ValueError(
                    f"per-head selection rectangle overflows 32-bit indexing: {selection_rect}"
                )
            self._selection_rows_per_request = selection_rows
            score_shape = (request_capacity, self._num_layers, self._num_q_heads, 1)
            self._row_mean = torch.empty(score_shape, dtype=torch.float32, device=device)
            self._row_inv_std = torch.empty_like(self._row_mean)
            self._selection_scores_rows = torch.empty(
                (request_capacity * selection_rows, selection_width_capacity),
                dtype=torch.float32,
                device=device,
            )
            self._selection_row_lengths = torch.full(
                (request_capacity * selection_rows,),
                selection_width_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._provisional_rows = torch.zeros(
                (request_capacity * selection_rows, keep_count), dtype=torch.int32, device=device
            )
            self._kept_ordinal_rows = torch.empty(
                (request_capacity * selection_rows, keep_count), dtype=torch.int32, device=device
            )

        if union:
            from .triattention_cute_selection import (
                _select_normalize_union_config,
                _TriAttentionNormalizeUnionKernel,
            )

            compiled_configs: Dict[Tuple[int, int, int, int], object] = {}
            for request_count in range(1, request_capacity + 1):
                page_shards = page_shards_by_request_count[request_count]
                config = _select_normalize_union_config(
                    request_count,
                    score_token_capacity,
                    sm_count,
                )
                config_key = (page_shards, *config)
                compiled_selection = compiled_configs.get(config_key)
                if compiled_selection is None:
                    cache_key = (
                        "triattention_cute_normalize_union",
                        static_geometry,
                        tensor_specs,
                        config_key,
                        _tensor_spec(self._selection_scores_rows),
                        _tensor_spec(self._partial_stats),
                    )
                    tokens_per_lane, token_subtiles, row_cluster_ctas = config
                    compiled_selection = _compiled_kernel(
                        cache_key,
                        lambda page_shards=page_shards,
                        tokens_per_lane=tokens_per_lane,
                        token_subtiles=token_subtiles,
                        row_cluster_ctas=row_cluster_ctas: cute.compile(
                            _TriAttentionNormalizeUnionKernel(
                                num_layers=self._num_layers,
                                score_token_capacity=score_token_capacity,
                                num_q_heads=num_q_heads,
                                # The finalizer maps real head rows onto N=8-padded planes.
                                num_kv_heads=self._num_kv_heads,
                                page_shards=page_shards,
                                tokens_per_lane=tokens_per_lane,
                                token_subtiles=token_subtiles,
                                row_cluster_ctas=row_cluster_ctas,
                                output_row_stride=selection_width_capacity,
                            ),
                            self._cute_partial_stats,
                            *self._cute_selection_prefix,
                            self._cute_selection_scores_rows,
                            cutlass.Int32(1),
                            stream,
                        ),
                    )
                    compiled_configs[config_key] = compiled_selection
                self._compiled_normalize_union_by_request_count[request_count] = compiled_selection

        # ---- compaction plans (opaque: only compact() interprets them) ---------
        compaction_params = [
            build_compaction_params(
                target_layout,
                block_offsets=self._block_offsets_device,
                kept_ordinals=self._kept_ordinal_rows,
                source_lengths=self._source_lengths_device,
                dense_destination_bases=self._prompt_lengths_device,
                # Per-round tails: the move offsets ride the staged metadata rows.
                dense_move_offsets=dense_move_offsets_row,
                protected_tail_capacity=protected_tail_capacity,
                swa_move_offsets=swa_move_offsets_row,
                swa_destination_bases=self._swa_destination_bases,
            )
        ]
        if draft_layout is not None:
            compaction_params.append(
                build_compaction_params(
                    draft_layout,
                    block_offsets=self._draft_block_offsets_device,
                    kept_ordinals=self._kept_ordinal_rows,
                    source_lengths=self._source_lengths_device,
                    dense_destination_bases=self._prompt_lengths_device,
                    dense_move_offsets=draft_move_offsets_row,
                    protected_tail_capacity=int(self._draft_protected_tail_capacity),
                )
            )
        self._compaction_params = tuple(compaction_params)

        # ---- round-ordering events ----------------------------------------------
        # Device-lifetime: created once and reused across capacity rebuilds
        # (they carry no pointer state; replacing them orphans in-flight ordering).
        if self._staging_reuse_event is None:
            # Host staging (pinned metadata + snapshots) reuse fence.
            self._staging_reuse_event = torch.cuda.Event()
            self._staging_reuse_event.record(torch.cuda.current_stream(device))
            # Manager-stream H2D of the block-offset tables has completed.
            self._block_offsets_ready_event = torch.cuda.Event()
            # This cohort's compact is done: manager may resize/reuse pages.
            self._compaction_done_event = torch.cuda.Event()

    def _local_score_calibration(
        self,
        global_layers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_layers = len(global_layers)
        if global_layers and max(global_layers) >= self._calibration_q_real.shape[0]:
            raise ValueError(
                f"TriAttention calibration has {self._calibration_q_real.shape[0]} layers, "
                f"but this PP rank references global layer {max(global_layers)}"
            )
        if global_layers == list(range(global_layers[0], global_layers[0] + num_layers)):
            layer_slice = slice(global_layers[0], global_layers[0] + num_layers)
            return (
                self._calibration_q_real[layer_slice],
                self._calibration_q_imag[layer_slice],
                self._calibration_mlr_coef[layer_slice],
            )
        layer_ids = torch.as_tensor(
            global_layers,
            device=self._calibration_q_real.device,
            dtype=torch.long,
        )
        return (
            self._calibration_q_real.index_select(0, layer_ids),
            self._calibration_q_imag.index_select(0, layer_ids),
            self._calibration_mlr_coef.index_select(0, layer_ids),
        )

    def _runtime_kv_layout(self, *, draft: bool = False) -> Dict[str, object]:
        # V2 pools are allocated once at manager init; the layout is a manager-lifetime contract.
        manager = self.draft_kv_cache_manager if draft else self.kv_cache_manager
        cached = self._kv_layout_caches[draft]
        if cached is not None:
            return cached

        if draft:
            global_layers = [int(layer) for layer in manager.pp_layers]
            if not global_layers:
                raise RuntimeError("TriAttention draft KV cache manager exposes no layers")
            # The draft is never scored: all draft layers compact as dense.
            dense_layers: List[int] = list(range(len(global_layers)))
            swa_layers: List[int] = []
            swa_window: Optional[int] = None
        else:
            global_layers = self._global_layers
            dense_layers, swa_layers, swa_window = self._layer_partition
            if not dense_layers:
                raise ValueError("TriAttention requires at least one full-attention layer")
        layout = self._build_runtime_kv_layout(
            manager,
            global_layers,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            label="draft " if draft else "",
        )
        self._kv_layout_caches[draft] = layout
        return layout

    def _build_runtime_kv_layout(
        self,
        manager: KVCacheManagerV2,
        global_layers: List[int],
        *,
        dense_layers: List[int],
        swa_layers: List[int],
        swa_window: Optional[int],
        label: str,
    ) -> Dict[str, object]:
        layer_pools = [manager.get_buffers(layer, kv_layout="HND") for layer in global_layers]
        # Canonical pool IDs, resolved once; every grouping derives from them
        # (V2 owns the mapping; its own lookup errors are the precise ones).
        layer_offsets = manager.layer_offsets
        layer_to_pool = manager.layer_to_pool_mapping_dict
        layer_pool_ids = tuple(
            int(layer_to_pool[layer_offsets[global_layer]]) for global_layer in global_layers
        )
        all_storage_groups: Dict[int, List[int]] = {}
        for layer, pool_id in enumerate(layer_pool_ids):
            all_storage_groups.setdefault(pool_id, []).append(layer)
        pool_representatives = tuple(layers[0] for layers in all_storage_groups.values())
        return dict(
            global_layers=global_layers,
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            layer_pool_ids=layer_pool_ids,
            pool_representatives=pool_representatives,
        )
