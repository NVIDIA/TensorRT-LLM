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
#
# DSpark speculative decoding: capture target-layer hidden states during the
# target forward, verify the previous block, and draft a new block in a single
# ``DSparkDraftModel.forward``.

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from .dspark_schedule import NEUTRAL_CONFIDENCE_LOGIT
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DSparkDecodingConfig

# Unscored-slot fill: sigmoid saturates to 1.0, so an unwritten row schedules
# as "verify the whole block" (fail-safe).
_NEUTRAL_CONFIDENCE_LOGIT = NEUTRAL_CONFIDENCE_LOGIT


@dataclass
class DSparkSpecMetadata(SpecMetadata):
    """Metadata for DSpark speculative decoding.

    Captures target-layer hidden states (mean over the mHC residual streams,
    per-layer width = model hidden size) into a
    ``[max_num_tokens, hidden_size * num_capture_layers]`` buffer that feeds
    ``DSparkDraftModel.forward``.
    """

    batch_indices_cuda: Optional[torch.Tensor] = None

    # Hidden state capture fields
    layers_to_capture: Optional[List[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    captured_hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            self._capture_layer_set = frozenset(self.layers_to_capture)
            self._layer_to_idx = {lid: i for i, lid in enumerate(self.layers_to_capture)}
            self.captured_hidden_states = torch.empty(
                (self.max_num_tokens, self.hidden_size * self.num_capture_layers),
                dtype=self.dtype,
                device="cuda",
            )
            logger.info(
                f"DSpark: capturing hidden states from layers {self.layers_to_capture}, "
                f"buffer shape {self.captured_hidden_states.shape}"
            )
        else:
            self.num_capture_layers = 0
            self._capture_layer_set = frozenset()
            self._layer_to_idx = {}

    def prepare(self):
        assert self.request_ids is not None
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # Maintain the request->slot map on the host (outside any captured
        # region) and mirror it into ``_batch_to_slot`` so the captured gen
        # forward indexes the rolling windows by tensor.
        worker = getattr(self, "_dspark_worker", None)
        if worker is not None and worker._win_inited:
            current = set(self.request_ids)
            for rid in list(worker._req_to_slot.keys()):
                if rid not in current:
                    slot = worker._req_to_slot.pop(rid)
                    worker._ctx_len[slot] = 0
                    worker._kv_windows[slot].zero_()
                    worker._free_slots.append(slot)
            # Assign a slot to every real gen request that never ran a
            # context/seed forward on this worker (disaggregated serving
            # prefills on the context server), so concurrent gen requests do
            # not share the scratch row. Context requests are seeded by
            # ``_seed_context_windows``; dummies stay on the scratch row.
            num_contexts = max(0, len(self.request_ids) - self.num_generations)
            for rid in self.request_ids[num_contexts:]:
                if (
                    rid != ATTENTION_DP_DUMMY_REQUEST_ID
                    and rid < worker._graph_dummy_id_floor
                    and rid not in worker._req_to_slot
                ):
                    worker._assign_slot(rid, reset=False)
            # Unknown request IDs (warmup / CUDA-graph padding, ADP idle, disagg
            # seeds without a real id) map to the throwaway scratch row so they
            # cannot overwrite a live request's rolling window.
            scratch = worker._scratch_slot
            mapping = torch.tensor(
                [worker._req_to_slot.get(rid, scratch) for rid in self.request_ids],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            worker._batch_to_slot[:num_seqs].copy_(mapping, non_blocking=True)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self._capture_layer_set

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        """Capture a target layer's hidden states into the buffer.

        An mHC-flattened ``[num_tokens, hc_mult * hidden]`` input is reduced by
        mean over the hc streams; a ``[num_tokens, hidden]`` input is stored
        as-is.
        """
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            if to_save.shape[-1] != self.hidden_size:
                hc_mult = to_save.shape[-1] // self.hidden_size
                to_save = to_save.reshape(num_tokens, hc_mult, self.hidden_size).mean(dim=1)
            self.captured_hidden_states[
                :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
            ].copy_(to_save, non_blocking=True)

    def get_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Get captured hidden states (all layers concatenated)."""
        if self.captured_hidden_states is None:
            return None
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class DSparkWorker(SpecWorkerBase):
    """Worker for DSpark speculative decoding.

    Drafts a whole block of ``block_size`` tokens in one
    ``DSparkDraftModel.forward`` over a worker-owned rolling window of
    projected captured context (no paged KV cache). Acceptance goes through
    :meth:`SpecWorkerBase.sample_and_accept_draft_tokens`, so the window state
    affects acceptance rate only, never correctness. Confidence scores feed
    only the verification scheduler's budget, never the acceptance decision.

    Reference: DeepSeek DeepSpec (https://github.com/deepseek-ai/DeepSpec).
    """

    def __init__(
        self,
        spec_config: "DSparkDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping

        # Per-slot rolling captured-context KV windows, built lazily on the
        # first forward (fixed-size for slot-indexed reads/writes).
        self._win_inited = False
        self._kv_windows: Optional[torch.Tensor] = None  # [max_batch, num_stages, win, hd]
        self._ctx_len: Optional[torch.Tensor] = None  # [max_batch] abs decode position
        self._win = 0

        # Slot management: ``_req_to_slot`` + ``_free_slots`` (host) are the
        # source of truth; ``_batch_to_slot`` is the CUDA mirror
        # (request-order -> slot) the captured gen forward indexes through.
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._batch_to_slot: Optional[torch.Tensor] = None  # [max_batch] long, cuda
        # Throwaway window row for padded / unknown request IDs (set to
        # ``max_batch`` in ``_lazy_init``); never handed out via ``_free_slots``.
        self._scratch_slot = 0

        # ``return_confidence`` is read once here (never per step) so the
        # captured draft graph cannot diverge. ``_confidence_logits`` is the
        # slot-indexed handoff to the verification scheduler.
        self.return_confidence = bool(spec_config.enable_confidence_scheduling)
        import os as _os

        from .dspark_sts import STS_COLLECT_ENV
        if _os.environ.get(STS_COLLECT_ENV) and not self.return_confidence:
            raise ValueError(
                f"{STS_COLLECT_ENV} is set but enable_confidence_scheduling is "
                f"off; the recorder is only built on the confidence path, so "
                f"this run would complete cleanly and collect NOTHING. Enable "
                f"confidence scheduling (a missing cost table keeps the "
                f"planner on the full block) or unset the variable.")
        self._confidence_logits: Optional[torch.Tensor] = None  # [max_batch+2, block]
        # Draft-pass stamps for the buffer above; allocated in ``_lazy_init``.
        self._confidence_stamp: Optional[torch.Tensor] = None
        self._draft_seq_host = 0
        self._draft_seq_cuda: Optional[torch.Tensor] = None
        # Row index of the permanently-neutral confidence row; see ``_lazy_init``.
        self._neutral_conf_row = 0
        # Host-side planner turning the confidence snapshot into this
        # iteration's draft length; built in ``_lazy_init``.
        self.verify_planner = None

        # DSpark is a one-engine drafter: the worker forward runs inside the
        # target's CUDA graph, so the gen draft path
        # (``_draft_gen_block_batched``) must stay host-sync-free and
        # capture-safe whenever ``cuda_graph_config`` is set.

        logger.info(
            f"DSparkWorker initialized with "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    def staged_confidence_buffer(self) -> Optional[torch.Tensor]:
        """The whole slot-indexed confidence buffer, or None when disabled.

        The whole buffer, not the current batch's rows: the planner reads it
        back by slot across the one-iteration lag.
        """
        return self._confidence_logits

    def confidence_stamp_buffer(self) -> Optional[torch.Tensor]:
        """The slot-indexed draft-pass stamps, or None before ``_lazy_init``."""
        return self._confidence_stamp

    def bump_draft_seq(self) -> int:
        """Advance the draft-pass sequence and return the PREVIOUS value.

        Must be called once per step from the executor's host path before the
        step's forward is enqueued; the returned previous value is the
        sequence of the last completed draft.
        """
        prev = self._draft_seq_host
        self._draft_seq_host += 1
        if self._draft_seq_cuda is not None:
            self._draft_seq_cuda.fill_(self._draft_seq_host)
        return prev

    def confidence_row_for(self, req_id: int) -> int:
        """Buffer row holding ``req_id``'s confidence, host-side.

        Falls back to the permanently-neutral row ("verify the full block" --
        fail-safe). Keyed by slot, not batch position, so the one-iteration-
        lagged snapshot survives batch reshuffles.
        """
        if self._confidence_logits is None:
            return self._neutral_conf_row
        return self._req_to_slot.get(req_id, self._neutral_conf_row)

    def sts_row_for(self, req_id: int) -> Optional[int]:
        """``confidence_row_for`` for the STS recorder: None, never neutral.

        Pairing a label against the never-written neutral row would fabricate
        a sample; an unresolvable request is dropped (counted as ``no_row``).
        """
        if self._confidence_logits is None:
            return None
        return self._req_to_slot.get(req_id)

    def verified_draft_seq(self) -> Optional[int]:
        """The draft pass that produced the block the CURRENT step verifies.

        Read at sampling time, after this step's ``bump_draft_seq``, so the
        verified pass is the previous value. Passes start at 1: a sequence of
        0 would false-match every never-written stamp row.
        """
        return self._draft_seq_host - 1 if self._draft_seq_host > 1 else None

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _lazy_init(self, draft_model, spec_metadata) -> None:
        block_size = int(draft_model.block_size)
        if block_size != self.max_draft_len:
            raise ValueError(
                "DSpark draft model block_size must equal worker max_draft_len; "
                f"got block_size={block_size} and max_draft_len={self.max_draft_len}"
            )

        if self._win_inited:
            return
        max_batch = spec_metadata.max_num_requests
        num_stages = draft_model.num_stages
        self._win = int(draft_model._attn_params["window_size"])
        head_dim = int(draft_model._attn_params["head_dim"])

        # Real requests occupy slots ``[0, max_batch)``; the extra scratch row
        # at index ``max_batch`` absorbs padded / unknown request IDs so they
        # can never overwrite a live request's rolling window. It is never
        # handed out through ``_free_slots``.
        self._scratch_slot = max_batch
        num_rows = max_batch + 1

        # CUDA-graph padding ids sit in
        # ``[CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, CUDA_GRAPH_DUMMY_REQUEST_ID]``;
        # real ids start at ``max_batch_size``, so a floor separates them.
        # Imported lazily to break the dspark -> cuda_graph_runner ->
        # speculative.utils -> dspark import cycle.
        from ..pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

        self._kv_windows = torch.zeros(
            (num_rows, num_stages, self._win, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._ctx_len = torch.zeros(num_rows, dtype=torch.long, device="cuda")
        self._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        if self.return_confidence:
            # Neutral fill = large positive logit (sigmoid ~ 1.0): unscored
            # slots schedule as "verify the full block" -- the fail-safe
            # direction. The neutral row sits one past the scratch row; it is
            # never in ``_batch_to_slot``, so no draft ever writes it.
            self._neutral_conf_row = num_rows
            self._confidence_logits = torch.full(
                (num_rows + 1, block_size),
                _NEUTRAL_CONFIDENCE_LOGIT,
                dtype=torch.float32,
                device="cuda",
            )
            # Per-row freshness stamps: the draft-pass sequence of the last
            # write, distinguishing fresh data from a stale replay or the
            # neutral fill. Written in-graph through the same ``slots`` scatter
            # as the confidence; the scalar is bumped outside the graph.
            self._confidence_stamp = torch.zeros(
                (num_rows + 1,), dtype=torch.int32, device="cuda")
            self._draft_seq_host = 0
            # Bumped by ``fill_`` from the executor's host path, outside
            # inference mode -- an inference tensor refuses that update, so
            # allocate this scalar with inference mode off. The stamp buffer
            # above intentionally stays an inference tensor (written in-graph
            # only).
            with torch.inference_mode(False):
                self._draft_seq_cuda = torch.zeros((), dtype=torch.int32,
                                                   device="cuda")
            self._build_verify_planner(draft_model, block_size, max_batch)
        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._win_inited = True
        logger.info(
            f"DSpark: allocated rolling KV windows "
            f"[{num_rows}, {num_stages}, {self._win}, {head_dim}] "
            f"({max_batch} request slots + 1 scratch row)"
        )

    def _build_verify_planner(self, draft_model, block_size: int, max_batch: int) -> None:
        """Construct the host-side verify planner (once, at lazy init).

        ``max_batch`` is only used to derive the verify-length tier ladder when
        the user did not configure one; see :func:`derive_verify_len_tiers` for
        why the ladder is a function of batch size.
        """
        import json

        from .dspark_planner import SpsCostTable
        from .dspark_schedule import DSparkScheduleConfig
        from .dspark_verify import DSparkVerifyPlanner

        cfg = DSparkScheduleConfig(block_size=block_size, min_verify_len=1)

        cost_table = None
        table_path = self.spec_config.confidence_sps_table_path
        if table_path:
            with open(table_path, encoding="utf-8") as f:
                raw = json.load(f)
            # The overhead fields are optional but not cosmetic: omitting them
            # asserts ``step_time_ms`` already measures whole steps at the
            # deployment's batch size.
            from .dspark_planner import check_table_fingerprint
            if (raw.get("_meta") or {}).get("lookup") != "interp":
                logger.warning(
                    f"DSpark cost table {table_path} carries no "
                    f"lookup='interp' marker: it may predate the "
                    f"interpolating consumer. Tables written for the old "
                    f"floor lookup dropped shelf-closing breakpoints on "
                    f"purpose, and interpolating across those gaps re-bills "
                    f"every mid-shelf total upward. Re-emit the table with a "
                    f"current profiler if trimming behaves oddly.")
            check_table_fingerprint(payload=raw, live={
                "tp": int(self.mapping.tp_size),
                "ep": int(self.mapping.moe_ep_size),
                "attention_dp": bool(self.mapping.enable_attention_dp),
                "block": int(block_size),
                "max_batch_size": int(max_batch),
                # moe_backend deliberately absent: a wrong live value would
                # false-reject a correct table; the table's value surfaces on
                # the "check manually" INFO line instead.
            })
            cost_table = SpsCostTable(
                token_counts=[int(v) for v in raw["token_counts"]],
                step_time_ms=[float(v) for v in raw["step_time_ms"]],
                fixed_overhead_ms=float(raw.get("fixed_overhead_ms", 0.0)),
                batch_sizes=[int(v) for v in raw.get("batch_sizes", [])],
                batch_overhead_ms=[float(v) for v in raw.get("batch_overhead_ms", [])],
            )
        else:
            # Flat = "unprofiled": the planner keeps verifying the full block
            # rather than trimming on a cost model where every token is free.
            cost_table = SpsCostTable.flat()
            logger.warning(
                "DSpark confidence scheduling is enabled but no "
                "confidence_sps_table_path was provided. Without a profiled step-cost "
                "curve the planner cannot tell a cheap verify token from an expensive "
                "one, so it will keep verifying the full block (no scheduling gain)."
            )

        # Calibration lives on the draft's confidence head; fall back to a plain
        # sigmoid when the head is absent (no confidence to calibrate anyway).
        from .dspark_sts import resolve_confidence_head
        head = resolve_confidence_head(draft_model)
        apply_calibration = head.apply_sts if head is not None else None
        sts_path = self.spec_config.confidence_sts_path
        if sts_path and head is None:
            # Refuse rather than degrade: a configured calibration that cannot
            # be attached is a wiring bug, never a preference.
            raise ValueError(
                f"confidence_sts_path is set but no confidence head was found "
                f"on {type(draft_model).__name__}; calibration cannot be "
                f"applied, so the planner would schedule on raw sigmoid "
                f"survivals. This is a model-wiring bug, not a config choice.")
        if sts_path and head is not None:
            # Accepts either key spelling ("sts_temperatures" here,
            # "temperatures" in SGLang); the vectors are interchangeable.
            from .dspark_sts import load_sts_temperatures_from_path
            temps = load_sts_temperatures_from_path(sts_path)
            head.load_sts_temperatures(torch.tensor(temps, dtype=torch.float32))
            logger.info(f"DSpark: loaded STS calibration from {sts_path}")

        tiers = self.spec_config.verify_len_tiers
        if not tiers:
            # Derive the ladder from the cost curve. Each tier is a captured
            # graph whose memory comes out of KV cache, so the derivation is
            # capped; tiers are only zero-loss at the batch size they were
            # derived for. A flat table yields [min, block_size]
            # (no scheduling).
            from .dspark_planner import derive_verify_len_tiers

            tiers = derive_verify_len_tiers(
                cost_table=cost_table,
                num_requests=max(int(max_batch), 1),
                block_size=block_size,
                min_verify_len=cfg.min_verify_len,
            )
            logger.info(f"DSpark: derived verify-length tiers {tiers}")
        self.verify_planner = DSparkVerifyPlanner(
            cfg=cfg,
            cost_table=cost_table,
            tiers=tiers,
            apply_calibration=apply_calibration,
        )

        # The env var overrides the config knob so the mode can be swapped
        # without editing a serving config.
        from .dspark_observability import (DSparkRaggedStats, RaggedVerifyMode,
                                           read_ragged_verify_mode)

        configured = (RaggedVerifyMode.COMPACT
                      if self.spec_config.enable_ragged_verify else
                      RaggedVerifyMode.STATIC)
        self.ragged_verify_mode = read_ragged_verify_mode(default=configured)
        if (self.verify_planner is not None
                and self.verify_planner.forced_budget_frac is not None
                and not self.ragged_verify_mode.trims_submitted_tokens):
            raise ValueError(
                "TLLM_DSPARK_FORCE_BUDGET_FRAC is set but ragged verify mode "
                f"'{self.ragged_verify_mode.value}' does not trim submitted "
                "tokens; the fraction would be logged as FORCED and silently "
                "never applied. Unset it or run compact mode.")
        # STS collection is env-gated; it costs one device->host copy per
        # decode step.
        from .dspark_sts import make_recorder_from_env
        self.sts_recorder = make_recorder_from_env(
            block_size=int(block_size),
            rank=int(self.mapping.rank),
            has_cost_table=bool(
                self.spec_config.confidence_sps_table_path),
            ragged_mode=self.ragged_verify_mode.value)

        self.ragged_stats = DSparkRaggedStats(mode=self.ragged_verify_mode,
                                              max_draft_len=block_size)
        # Shared (not copied) so the summary stays live.
        self.ragged_stats.planner_stats = self.verify_planner.stats
        logger.info(
            f"DSpark verify planner: tiers={self.verify_planner.tiers}, "
            f"profiled_cost_table={not cost_table.is_flat}, "
            f"ragged_verify_mode={self.ragged_verify_mode.value}"
        )
        if (self.ragged_verify_mode.trims_submitted_tokens
                and cost_table.is_flat):
            logger.warning(
                "DSpark ragged verify is set to 'compact' but the cost table is "
                "flat, so the planner's budget degenerates to verify-all and "
                "every request will receive the same full window. Profile a "
                "cost table and pass confidence_sps_table_path, or the ragged "
                "path will not be exercised."
            )

    def _assign_slot(self, req_id: int, reset: bool) -> int:
        """Get (or refresh) the slot for a request; reset clears its window."""
        if reset and req_id in self._req_to_slot:
            old = self._req_to_slot.pop(req_id)
            self._ctx_len[old] = 0
            self._kv_windows[old].zero_()
            self._free_slots.append(old)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                raise RuntimeError(
                    "DSpark has no free rolling-window slots for request "
                    f"{req_id}; increase max_num_requests"
                )
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            self._ctx_len[slot] = 0
            self._kv_windows[slot].zero_()
            if self._confidence_logits is not None:
                # A recycled slot still holds the previous occupant's scores;
                # reset to neutral so the first scheduling decision verifies
                # the full block.
                self._confidence_logits[slot].fill_(_NEUTRAL_CONFIDENCE_LOGIT)
        return self._req_to_slot[req_id]

    def _seed_context_windows(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ) -> None:
        """Seed context chunks using their absolute positions.

        A request can arrive in multiple prefill chunks. Only its first chunk
        starts at position zero and resets the persistent rolling window;
        continuation chunks append to the same request slot.
        """
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        flat_position_ids = position_ids.reshape(-1)
        context_offset = 0
        for i in range(attn_metadata.num_contexts):
            chunk_len = int(attn_metadata._seq_lens[i])
            chunk_positions = flat_position_ids[context_offset : context_offset + chunk_len].long()
            if chunk_len == 0:
                context_offset += chunk_len
                continue

            req_id = spec_metadata.request_ids[i]
            first_position = int(chunk_positions[0].item())
            slot = self._assign_slot(req_id, reset=first_position == 0)
            self._ctx_len[slot] = chunk_positions[-1] + 1

            if captured is not None:
                keep = min(self._win, chunk_len)
                hidden = captured[context_offset + chunk_len - keep : context_offset + chunk_len]
                # A prompt token at absolute position p is stored in frame p+1,
                # matching the generation path's start_pos convention.
                window_positions = chunk_positions[-keep:] + 1
                draft_model.write_context_windows(hidden, window_positions, self._kv_windows[slot])
            context_offset += chunk_len

    def _draft_gen_block_batched(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        total_target_tokens: int,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """CUDA-graph-safe batched gen draft (all gen requests in one forward).

        Must stay free of host syncs and data-dependent shapes. Returns block
        logits ``[num_gens, K, vocab]`` for ``sample_draft_tokens``, or
        ``None`` when there is nothing to draft. The full block is always
        proposed.
        """
        num_gens = batch_size - num_contexts
        # K is the draft block width (fixed by the checkpoint); Kp1 is the
        # target's per-gen-request token stride in ``captured``
        # (runtime_draft_len + 1, which shrinks when verification is trimmed).
        # Striding by max_draft_len instead reads into the next request's
        # hidden states.
        K = self.max_draft_len
        # ``or K``: runtime_draft_len defaults to 0 (not None), so an
        # ``is None`` check would leave a stride of 1 on unpopulated paths.
        runtime_draft_len = spec_metadata.runtime_draft_len or K
        Kp1 = int(runtime_draft_len) + 1
        device = accepted_tokens.device

        if num_gens == 0:
            return None
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        if captured is None:
            return None

        # gen-only graph batches have num_ctx_tokens == 0; mixed eager batches put
        # the gen tokens after the context tokens.
        gen_start = attn_metadata.num_ctx_tokens
        slots = self._batch_to_slot[num_contexts:batch_size]  # [G]
        nacc = num_accepted_tokens[num_contexts:batch_size].long()  # [G]
        gidx = nacc - 1  # [G] index of the bonus within each verified prefix

        # Bonus token = last accepted token of the verified prefix.
        bonus = (
            accepted_tokens[num_contexts:batch_size].gather(1, gidx.unsqueeze(1)).squeeze(1).long()
        )  # [G]

        # Start offset of each request's processed tokens inside ``captured``.
        # Uniform batches stride by Kp1; under ragged verification the offsets
        # must come from the same qo_indptr the input layout was packed with,
        # or request r silently reads request r-1's tail.
        qo_indptr = spec_metadata.qo_indptr
        if spec_metadata.verify_lens is not None and qo_indptr is not None:
            base = gen_start + qo_indptr[:num_gens].to(device=device, dtype=torch.long)
        else:
            arange_g = torch.arange(num_gens, device=device)
            base = gen_start + arange_g * Kp1  # [G]
        # Clamped like interim_base below: gidx comes from a device-side accept
        # count, so a bad one must not turn into an out-of-bounds read.
        main_hidden = captured[(base + gidx).clamp(min=0, max=captured.shape[0] - 1)]

        # Fixed-size [G, K] masked back-fill of the intermediate accepted
        # tokens (all but the bonus) into frames old+1 .. old+nacc-1. The
        # width stays at static K (capture-safe); ``nacc <= runtime_draft_len
        # + 1`` bounds the valid j to this request's own block, so the extra
        # columns are always masked out.
        old = self._ctx_len[slots]  # [G] pre-increment decode position
        j = torch.arange(K, device=device)  # [K]
        interim_valid = j.unsqueeze(0) < (nacc.unsqueeze(1) - 1)  # [G, K]
        interim_pos = old.unsqueeze(1) + 1 + j.unsqueeze(0)  # [G, K]
        interim_base = (base.unsqueeze(1) + j.unsqueeze(0)).clamp(
            min=0, max=captured.shape[0] - 1
        )  # [G, K] (clamped; invalid entries are masked out anyway)
        interim_hidden = captured[interim_base]  # [G, K, ncap*hidden]
        draft_model.write_context_windows_batched(
            interim_hidden, interim_pos, slots, interim_valid, self._kv_windows
        )

        # Advance the decode position by the accepted count; start_pos is the
        # post-increment ctx_len.
        start_pos = old + nacc  # [G]
        self._ctx_len[slots] = start_pos

        _toks, confidence, block_logits = draft_model.forward_batched(
            main_hidden,
            bonus,
            start_pos,
            kv_windows=self._kv_windows,
            slots=slots,
            temperature=0.0,
            return_confidence=self.return_confidence,
            return_logits=True,
            all_rank_num_tokens=all_rank_num_tokens,
        )
        # Stash the [G, K] confidence for the verification scheduler.
        # Slot-indexed (survives batch reshuffles), written in place (a
        # captured graph keeps the same storage), and scattered through
        # ``slots`` so a replay lands on the current batch's slots.
        if confidence is not None:
            self._confidence_logits[slots] = confidence.detach()
            # Same slots, same capture region: content-only update, graph-safe.
            self._confidence_stamp[slots] = self._draft_seq_cuda
        return block_logits

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        raw_logits = logits
        K = self.max_draft_len

        self._lazy_init(draft_model, spec_metadata)
        # Backref so DSparkSpecMetadata.prepare() can maintain the host slot map
        # and mirror it into _batch_to_slot for the CUDA-graph-safe gen path.
        spec_metadata._dspark_worker = self
        self._execute_guided_decoder_if_present(logits)

        # Acceptance: strict target-verify, or rejection sampling for a
        # non-greedy batch with valid draft_probs.
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        total_target_tokens = input_ids.shape[0]

        # CUDA-graph warmup guard: warmup forwards (is_cuda_graph set, stream
        # NOT yet capturing) run synthetic gen batches; snapshot/restore the
        # persistent window state so they are side-effect-free. During the
        # capture pass the stream IS capturing, so skip the save/restore and
        # let the ops be recorded; prefill resets wipe capture-time mutation.
        is_warmup = (
            spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
        )
        if is_warmup:
            saved_ctx_len = self._ctx_len.clone()
            saved_windows = self._kv_windows.clone()
            # The confidence buffer is persistent state too; warmup scores
            # must not reach the verification planner.
            saved_confidence = (
                None if self._confidence_logits is None else self._confidence_logits.clone()
            )

        # Seed prefill requests' rolling windows from their prompt's captured
        # context (affects acceptance rate only).
        if num_contexts > 0:
            self._seed_context_windows(
                draft_model,
                spec_metadata,
                attn_metadata,
                position_ids,
                total_target_tokens,
            )

        # FUSED_COMM MoE backends sync EP ranks with a phase-flip barrier:
        # every rank must invoke the draft MoE the same number of times with
        # the same globally-gathered per-rank token list (num_gens * block
        # each) or the barrier desyncs. ``all_rank_num_gens`` is gathered at
        # metadata-prep time, outside any capture region; None for non-ADP /
        # single-rank runs (local fallback in ``_forward_stage``).
        block = int(draft_model.block_size)
        all_rank_num_gens = spec_metadata.all_rank_num_gens
        # A rank with zero local gen requests must still cross the draft MoE's
        # barrier, but the router / shared-expert GEMMs reject 0-row input, so
        # it runs a 1-row dummy -- encoded as ``1`` in the shared per-rank list
        # so every rank agrees on the chunk count.
        all_rank_draft_tokens = (
            [max(1, int(g) * block) for g in all_rank_num_gens]
            if all_rank_num_gens is not None
            else None
        )
        global_has_gen = (
            max(all_rank_num_gens) > 0 if all_rank_num_gens is not None else num_gens > 0
        )

        if num_gens > 0:
            gen_logits = self._draft_gen_block_batched(
                draft_model,
                spec_metadata,
                attn_metadata,
                accepted_tokens,
                num_accepted_tokens,
                num_contexts,
                batch_size,
                total_target_tokens,
                all_rank_num_tokens=all_rank_draft_tokens,
            )
            if gen_logits is not None:
                gen_draft_tokens = self.sample_draft_tokens(
                    gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                )
                # The context one-hot must match the width published to
                # draft_probs, NOT gen_logits.shape[-1]: under TP the draft
                # logits are vocab-sharded before sample_draft_tokens gathers
                # them to full vocab.
                gen_vocab = spec_metadata.draft_probs_last_dim
            else:
                gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
                gen_vocab = None
        else:
            # No local gens: if any peer EP rank has some, still cross the
            # draft MoE barrier so the FUSED_COMM phase-flip stays lockstep.
            if global_has_gen:
                draft_model.run_moe_lockstep_noop(all_rank_draft_tokens, accepted_tokens.device)
            gen_draft_tokens = torch.empty((0, K), dtype=torch.int32, device="cuda")
            gen_vocab = None

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a legal one-hot so they are
        # a valid distribution when they become gen requests next iteration.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, gen_vocab)

        if num_contexts > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        else:
            next_draft_tokens = gen_draft_tokens

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        if is_warmup:
            self._ctx_len.copy_(saved_ctx_len)
            self._kv_windows.copy_(saved_windows)
            if saved_confidence is not None:
                self._confidence_logits.copy_(saved_confidence)

        outputs = {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }
        # cap-accept only: per-request trim written by ``apply_accept_caps``.
        # Rides the sampler's existing copy, so it costs no extra sync.
        cap_trim = spec_metadata.accept_cap_trim
        if cap_trim is not None and spec_metadata.accept_caps is not None:
            outputs["cap_trim_lens"] = cap_trim[:batch_size]
        return outputs
