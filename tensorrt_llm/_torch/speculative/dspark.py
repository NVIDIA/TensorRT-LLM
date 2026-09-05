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
# DSpark worker / metadata mirror the DFlash plumbing (capture target-layer
# hidden states, accept the previous block with standard verification, draft a
# new block in one backbone forward), adapted to DSpark's draft model which
# produces the whole block and fixed-shape confidence scores inside a single
# ``DSv4DSparkDraftModel.forward`` rather than via mask-token cross-attention.

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from .dflash import DFlashWorker, dflash_draft_slot_ids
from .dspark_schedule import (
    HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT,
    NATIVE_UNIFORM_VERIFY_OUTPUT,
    NEUTRAL_CONFIDENCE_LOGIT,
)
from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DSparkDecodingConfig

# Unscored-slot fill: sigmoid saturates to 1.0, so an unwritten row schedules
# as "verify the whole block" (fail-safe).
_NEUTRAL_CONFIDENCE_LOGIT = NEUTRAL_CONFIDENCE_LOGIT


def _publish_policy_window_output(
    outputs: dict, verify_lens: Optional[torch.Tensor], batch_size: int
) -> None:
    """Publish the authoritative verify-window source for this exact step."""
    if verify_lens is None:
        outputs[NATIVE_UNIFORM_VERIFY_OUTPUT] = True
        return
    if verify_lens.shape[0] >= batch_size:
        outputs["verify_lens"] = verify_lens[:batch_size]
    else:
        outputs[HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT] = True


@dataclass
class DSparkSpecMetadata(SpecMetadata):
    """Metadata for DSpark speculative decoding.

    Captures hidden states from the target model's ``layers_to_capture`` during
    the target forward pass. DSpark captures the *mean over the multi-head
    (mHC) residual streams* at each captured layer (handled by the target-side
    capture hook), concatenated across layers, and feeds them to the draft
    model's ``main_proj`` + ``main_norm`` (inside ``DSv4DSparkDraftModel.forward``)
    as the captured-context attention input (``main_x``).

    Mirrors :class:`DFlashSpecMetadata`; the only DSpark-specific detail is that
    the per-layer captured width is the model hidden size (post hc-mean), so the
    buffer is ``[max_num_tokens, hidden_size * num_capture_layers]``.
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

        # Set up hidden state capture buffer
        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            # O(1) lookups for is_layer_capture() and maybe_capture_hidden_states()
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

        # CUDA-graph-safe path: maintain the request->slot mapping on the host
        # (outside the captured region) and mirror it into ``_batch_to_slot`` so the
        # captured gen forward can index the rolling windows by tensor. Mirrors
        # ``DFlashSpecMetadata.prepare`` (dflash.py:96-113).
        worker = getattr(self, "_dspark_worker", None)
        if worker is not None and worker._win_inited:
            current = set(self.request_ids)
            for rid in list(worker._req_to_slot.keys()):
                if rid not in current:
                    slot = worker._req_to_slot.pop(rid)
                    worker._ctx_len[slot] = 0
                    worker._valid_len[slot] = 0
                    worker._position_initialized[slot] = False
                    worker._kv_windows[slot].zero_()
                    worker._free_slots.append(slot)
            # Assign a persistent rolling-window slot to every real generation
            # request that never ran a context/seed forward on this worker. In
            # disaggregated serving the prompt is prefilled (and the window
            # seeded) on the *context* server, so ``_seed_context_windows`` never
            # runs on the generation server and ``_req_to_slot`` stays empty;
            # without this, all concurrent gen requests fall through to the shared
            # scratch row below and corrupt each other's draft window at batch
            # size > 1 (GitHub #16767). Context-prefix entries are left to
            # ``_seed_context_windows``; the ADP-idle (id 0) and CUDA-graph
            # padding dummies are kept on the scratch row.
            num_contexts = max(0, len(self.request_ids) - self.num_generations)
            for rid in self.request_ids[num_contexts:]:
                if (
                    rid != ATTENTION_DP_DUMMY_REQUEST_ID
                    and rid < worker._graph_dummy_id_floor
                    and rid not in worker._req_to_slot
                ):
                    worker._assign_slot(rid, reset=False)
            # Unknown request IDs (synthetic warmup / CUDA-graph padding, ADP idle
            # requests, or disagg seed forwards without a real id) map to the
            # dedicated throwaway scratch row so they cannot overwrite a live
            # request's rolling window (they previously aliased to slot 0).
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
        """Capture hidden states from a target model layer into the buffer.

        DeepSeek-V4 keeps the multi-head (mHC) residual stream flattened as
        ``[num_tokens, hc_mult * hidden]``; DSpark captures the *mean over the hc
        streams* (reference ``h.mean(dim=2)`` with ``h`` shaped
        ``[*, hc_mult, hidden]``). We reduce here so the V4 decoder layer's
        existing capture call is unchanged. A ``[num_tokens, hidden]`` input
        (already reduced / non-mHC) is stored as-is.
        """
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            # mHC residual -> mean over the hc_mult streams.
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


class DSv4DSparkWorker(SpecWorkerBase):
    """Worker for DSpark speculative decoding.

    DSpark drafts a whole block of ``block_size`` tokens in one backbone forward
    (``DSv4DSparkDraftModel.forward``): it projects the captured target-layer hidden
    states (``main_proj`` + ``main_norm``) into the draft's captured-context
    attention, runs the ``num_stages`` DSpark blocks over a rolling captured
    window, refines the per-position logits with the Markov head, and predicts a
    per-position acceptance confidence used by the verification scheduler.

    Unlike DFlash, the draft does NOT use the paged KV cache or mask-token
    cross-attention: its attention K/V come from the worker-owned rolling window
    of projected captured context (one ``main_kv`` per decode step, per stage).
    Acceptance of the previous block goes through the unified
    :meth:`SpecWorkerBase.sample_and_accept_draft_tokens` (strict target-verify,
    or rejection sampling for a non-greedy batch), so greedy parity with no-spec
    is preserved regardless of draft quality.

    The rolling window is kept consistent across the whole decode: it is seeded
    from the prompt's captured context at prefill and back-filled with the
    intermediate accepted tokens of a multi-accept step (both via
    ``DSv4DSparkDraftModel.write_context_windows``), in addition to the per-step bonus
    write done by the generation path. These affect draft acceptance rate only,
    not correctness, which the standard target verify guarantees.

    Naming: workers are classified by *deployment form*, not by draft
    backbone (see :class:`DSparkWorker`). This one is form-specific
    because it owns a rolling captured-context window and drives the draft
    through attributes only an embedded DeepSeek-V4-Pro draft has --
    ``num_stages``, ``_attn_params``, ``write_context_windows``,
    ``write_context_windows_batched`` and ``forward_batched``. A standalone
    drafter has none of them and is served by :class:`DSparkWorker`.

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
        self._attention_warmup_attempted = False
        self._kv_windows: Optional[torch.Tensor] = None  # [max_batch, num_stages, win, hd]
        self._ctx_len: Optional[torch.Tensor] = None  # [max_batch] abs decode position
        self._valid_len: Optional[torch.Tensor] = None  # [max_batch] written window entries
        self._position_initialized: Optional[torch.Tensor] = None  # [max_batch] bool
        self._win = 0

        # Slot management. ``_req_to_slot`` (python dict) + ``_free_slots`` are the
        # source of truth, updated in prepare()/forward(); ``_batch_to_slot`` is the
        # CUDA mirror (request-order -> slot) read by the CUDA-graph-safe batched
        # gen path (set on the host in prepare(), so the captured forward indexes
        # the rolling windows through a tensor instead of a python dict lookup).
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._batch_to_slot: Optional[torch.Tensor] = None  # [max_batch] long, cuda
        # Index of the throwaway "scratch" window row that absorbs padded /
        # unknown request IDs (set in ``_lazy_init`` to ``max_batch``); it is
        # never handed out through ``_free_slots``.
        self._scratch_slot = 0

        # ``return_confidence`` is read once here (never per step) so the
        # captured draft graph cannot diverge. ``_confidence_logits`` is the
        # slot-indexed handoff to the verification scheduler.
        self.return_confidence = bool(getattr(spec_config, "enable_confidence_scheduling", False))
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

        # The generation draft path is the batched, host-sync-free
        # ``_draft_gen_block_batched`` + ``DSv4DSparkDraftModel.forward_batched`` +
        # ``dspark_attention_forward_batched``: it is correct in eager mode AND safe
        # to capture into the target's CUDA graph (DSpark is a one-engine drafter —
        # its worker forward runs inside that graph, so the draft path MUST be
        # capture-safe whenever ``cuda_graph_config`` is set).

        logger.info(
            f"DSv4DSparkWorker initialized with "
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

    def verified_draft_seq_cuda(self) -> Optional[torch.Tensor]:
        """The verified draft sequence as a device scalar for the prologue.

        ``_draft_seq_cuda`` already holds the CURRENT step's sequence by
        launch time (``bump_draft_seq`` fills it before the forward is
        enqueued), so the pass that stamped the block being verified is one
        less. Passing the buffer itself would make every fresh row read as
        stale and silently degrade the whole batch to neutral.
        """
        if self._draft_seq_cuda is None:
            return None
        return self._draft_seq_cuda - 1

    def batch_slot_view(self, num_rows: int) -> Optional[torch.Tensor]:
        """The batch-position -> confidence-row map staged this step.

        ``[num_rows]`` int64 on device; rows past the real batch point at the
        scratch row (their survival is zeroed by the prologue's ``num_real``
        mask, so the content never matters).
        """
        if self._batch_to_slot is None:
            return None
        return self._batch_to_slot[:num_rows]

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

        if not self._win_inited:
            max_batch = spec_metadata.max_num_requests
            num_stages = draft_model.num_stages
            self._win = int(draft_model._attn_params["window_size"])
            head_dim = int(draft_model._attn_params["head_dim"])

            # Real requests occupy slots ``[0, max_batch)``; one extra "scratch" row
            # at index ``max_batch`` absorbs padded / unknown request IDs (CUDA-graph
            # padding, ADP idle requests, or disagg seed forwards that arrive without
            # a real request id) so they can never overwrite a live request's rolling
            # window. Previously such IDs aliased to slot 0 and corrupted whichever
            # real request occupied it. The scratch row is never handed out through
            # ``_free_slots`` and its contents are throwaway.
            self._scratch_slot = max_batch
            num_rows = max_batch + 1

            # CUDA-graph padding requests carry ids in
            # ``[CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, CUDA_GRAPH_DUMMY_REQUEST_ID]``,
            # while real request ids start at ``max_batch_size`` and grow, so a simple
            # floor cleanly separates them. Together with ``ATTENTION_DP_DUMMY_REQUEST_ID``
            # (0) these dummies must route to the scratch row (see ``prepare()``) and
            # never consume a real slot. Imported lazily to break the
            # dspark -> cuda_graph_runner -> speculative.utils -> dspark import cycle.
            from ..pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

            self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

            self._kv_windows = torch.zeros(
                (num_rows, num_stages, self._win, head_dim),
                dtype=torch.bfloat16,
                device="cuda",
            )
            self._ctx_len = torch.zeros(num_rows, dtype=torch.long, device="cuda")
            self._valid_len = torch.zeros(num_rows, dtype=torch.long, device="cuda")
            self._position_initialized = torch.zeros(num_rows, dtype=torch.bool, device="cuda")
            self._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")
            if self.return_confidence:
                # Neutral fill = large positive logit (sigmoid ~ 1.0):
                # unscored slots fail safe by verifying the full block. The
                # neutral row sits one past the scratch row and is never used
                # as a draft-output destination.
                self._neutral_conf_row = num_rows
                self._confidence_logits = torch.full(
                    (num_rows + 1, block_size),
                    _NEUTRAL_CONFIDENCE_LOGIT,
                    dtype=torch.float32,
                    device="cuda",
                )
                self._confidence_stamp = torch.zeros(
                    (num_rows + 1,), dtype=torch.int32, device="cuda"
                )
                self._draft_seq_host = 0
                # The executor updates this scalar outside inference mode;
                # inference tensors reject that in-place host-side update.
                with torch.inference_mode(False):
                    self._draft_seq_cuda = torch.zeros((), dtype=torch.int32, device="cuda")
                self._build_verify_planner(draft_model, block_size)
            self._free_slots = deque(range(max_batch))
            self._req_to_slot = {}
            logger.info(
                f"DSpark: allocated rolling KV windows "
                f"[{num_rows}, {num_stages}, {self._win}, {head_dim}] "
                f"({max_batch} request slots + 1 scratch row)"
            )
            # Buffer state is complete independently of CuTe DSL prewarming.
            # A failed prewarm must not cause the next forward to recreate the
            # windows or reset the live slot maps.
            self._win_inited = True

        if self._attention_warmup_attempted:
            return

        # Prewarm the same self-JIT ops used by production before CUDA graph
        # capture. This is best-effort inside the named warmup entry; the ops
        # remain able to compile themselves on a later eager first use.
        from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if IS_CUTLASS_DSL_AVAILABLE:
            from ..custom_ops.dspark_attention_custom_op import (
                is_dsv4_dspark_attention_config_supported,
                warmup_fused_dsv4_dspark_attention,
            )

            if is_dsv4_dspark_attention_config_supported(
                block_size,
                int(draft_model._attn_params["n_heads"]),
                int(draft_model._attn_params["head_dim"]),
                int(draft_model._attn_params["window_size"]),
            ):
                warmup_fused_dsv4_dspark_attention(
                    block_size,
                    eps=float(draft_model._attn_params["eps"]),
                )
        self._attention_warmup_attempted = True

    def _build_verify_planner(self, draft_model, block_size: int) -> None:
        """Construct the host-side verify planner (once, at lazy init).

        ModelEngine owns and authenticates the deployment cost table. The
        worker creates only the policy state and calibration, then receives
        that exact runtime object before its first decision.
        """
        from .dspark_schedule import DSparkScheduleConfig
        from .dspark_verify import DSparkVerifyPlanner

        cfg = DSparkScheduleConfig(block_size=block_size, min_verify_len=1)

        table_path = self.spec_config.confidence_sps_table_path
        # ModelEngine owns the actual CUDA graph ladder, so it is the only
        # component that may read, authenticate, and construct the deployment
        # cost object. Reopening this path here creates a TOCTOU window in
        # which graph capture and scheduling can observe different files.
        logger.info(
            f"DSpark: deferring exact SPS table {table_path} to the validated "
            "ModelEngine runtime object"
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
                f"survivals. This is a model-wiring bug, not a config choice."
            )
        if sts_path and head is not None:
            # Accepts either key spelling ("sts_temperatures" here,
            # "temperatures" in SGLang); the vectors are interchangeable.
            from .dspark_sts import load_sts_temperatures_from_path

            temps = load_sts_temperatures_from_path(sts_path)
            head.load_sts_temperatures(torch.tensor(temps, dtype=torch.float32))
            logger.info(f"DSpark: loaded STS calibration from {sts_path}")

        self.verify_planner = DSparkVerifyPlanner(
            cfg=cfg,
            apply_calibration=apply_calibration,
            device_windows=bool(self.spec_config.enable_fused_confidence_scheduler),
        )
        logger.info(
            f"DSpark verify planner: max_verify_len={self.verify_planner.max_verify_len}, "
            f"device_windows={self.verify_planner.device_windows}"
        )

    def _assign_slot(self, req_id: int, reset: bool) -> int:
        """Get (or refresh) the slot for a request; reset clears its window."""
        if reset and req_id in self._req_to_slot:
            old = self._req_to_slot.pop(req_id)
            self._ctx_len[old] = 0
            self._valid_len[old] = 0
            self._position_initialized[old] = False
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
            self._valid_len[slot] = 0
            self._position_initialized[slot] = False
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
            self._position_initialized[slot] = True

            if captured is not None:
                self._valid_len[slot] = torch.clamp(
                    self._valid_len[slot] + chunk_len, max=self._win
                )
                keep = min(self._win, chunk_len)
                hidden = captured[context_offset + chunk_len - keep : context_offset + chunk_len]
                # A prompt token at absolute position p is stored in frame p+1,
                # matching the generation path's start_pos convention.
                window_positions = chunk_positions[-keep:] + 1
                draft_model.write_context_windows(hidden, window_positions, self._kv_windows[slot])
            context_offset += chunk_len

    def _advance_generation_state(
        self,
        slots: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        input_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bootstrap and advance per-slot decode state without host synchronization."""
        old = torch.where(self._position_initialized[slots], self._ctx_len[slots], input_positions)
        start_pos = old + num_accepted_tokens
        self._ctx_len[slots] = start_pos
        self._valid_len[slots] = torch.clamp(
            self._valid_len[slots] + num_accepted_tokens, max=self._win
        )
        self._position_initialized[slots] = torch.ones_like(slots, dtype=torch.bool)
        return old, start_pos

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
        position_ids: torch.Tensor,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """CUDA-graph-safe batched gen draft (all gen requests in one forward).

        Free of host syncs and data-dependent shapes: per-request quantities
        (``nacc``, the bonus, ``main_hidden``, ``start_pos``, the multi-accept
        back-fill) are gathered as tensors, slots come from the host-built
        ``_batch_to_slot`` mirror, and the backbone runs once via
        ``DSv4DSparkDraftModel.forward_batched``. Returns the per-position corrected
        block logits ``[num_gens, K, vocab]`` (or ``None`` when there is nothing to
        draft); the worker feeds them to ``SpecWorkerBase.sample_draft_tokens``.
        Confidence truncation stays disabled — the full block is proposed. When
        enabled, fixed-shape confidence scores are stored separately for
        verification scheduling; they never truncate the proposal.
        """
        num_gens = batch_size - num_contexts
        # K is the draft block width (fixed by the checkpoint); Kp1 is the
        # target's per-gen-request token stride in ``captured``
        # (runtime_draft_len + 1, which shrinks when verification is trimmed).
        # Striding by max_draft_len instead reads into the next request's
        # hidden states.
        K = self.max_draft_len
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
        # Ragged verification uses the same qo_indptr as the packed input;
        # uniform and bootstrap iterations use the accepted-row width.
        target_width = accepted_tokens.shape[1]
        qo_indptr = spec_metadata.qo_indptr
        if spec_metadata.verify_lens is not None and qo_indptr is not None:
            base = gen_start + qo_indptr[:num_gens].to(device=device, dtype=torch.long)
        else:
            arange_g = torch.arange(num_gens, device=device)
            base = gen_start + arange_g * target_width  # [G]
        main_hidden = captured[(base + gidx).clamp(min=0, max=captured.shape[0] - 1)]

        # Fixed-size ([G, K]) masked back-fill of the intermediate accepted tokens
        # (everything but the bonus) into the rolling window — same frames as the
        # eager path (old+1 .. old+nacc-1), with j >= nacc-1 masked out.
        # A disaggregated generation worker never sees prompt prefill, so a new
        # slot has no absolute decode position. Bootstrap it once from the first
        # target input position; locally-prefilled and existing slots keep their
        # monotonically advanced position.
        input_positions = position_ids.reshape(-1)[base].long()
        old, start_pos = self._advance_generation_state(slots, nacc, input_positions)
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

        # Surface the per-position corrected block logits ([num_gens, K, vocab])
        # and let SpecWorkerBase.sample_draft_tokens do the (greedy or rejection)
        # sampling + TP gather + draft_probs scatter, rather than argmaxing here.
        _toks, confidence, block_logits = draft_model.forward_batched(
            main_hidden,
            bonus,
            start_pos,
            kv_windows=self._kv_windows,
            slots=slots,
            valid_len=self._valid_len[slots],
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

    def _sample_draft_tokens_guided(
        self,
        gen_logits: torch.Tensor,
        spec_metadata: "DSparkSpecMetadata",
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        K: int,
    ):
        """
        Grammar-constrained draft sampling for the guided-decoding path.
        """
        vocab = gen_logits.shape[-1]
        # Lay the block out step-major ([K, batch, vocab]) so each step's slice is
        # a contiguous [batch, vocab] tensor.
        if num_contexts > 0:
            full_logits = gen_logits.new_zeros((K, batch_size, vocab))
            full_logits[:, num_contexts:, :] = gen_logits.transpose(0, 1)
        else:
            full_logits = gen_logits.transpose(0, 1).contiguous()

        gidx = (num_accepted_tokens - 1).clamp(min=0).unsqueeze(1).long()
        new_tokens = accepted_tokens.gather(1, gidx).squeeze(1).to(torch.int32)

        gen_draft_tokens = []
        for k in range(K):
            self.guided_decoder.add_draft_batch(new_tokens, num_accepted_tokens, draft_step=k)
            step_logits = full_logits[k]
            self.guided_decoder.execute_draft_batch(step_logits, draft_step=k)
            step_tokens = self.sample_draft_tokens(
                step_logits, spec_metadata, batch_size, draft_step=k
            )
            gen_draft_tokens.append(step_tokens[num_contexts:])
            new_tokens = step_tokens
        gen_draft_tokens = torch.stack(gen_draft_tokens, dim=1)
        return gen_draft_tokens

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

        # Target-verify acceptance via the unified SpecWorkerBase entry: it
        # reshapes the stored draft tokens (default (num_gens, runtime_draft_len)
        # hook), then routes to strict or rejection sampling. Greedy parity with
        # the previous hand-rolled path is preserved (rejection only engages for a
        # non-greedy batch with valid draft_probs).
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        total_target_tokens = input_ids.shape[0]

        # CUDA-graph warmup guard: the warmup forwards (is_cuda_graph set, stream
        # NOT yet capturing) run synthetic gen batches that would otherwise advance
        # the persistent rolling-window state. Snapshot and restore it so warmup is
        # side-effect-free. (During the capture pass itself the stream IS capturing,
        # so we skip the save/restore and let the ops be recorded. Capture-time
        # mutation stays on dummy/scratch state: a locally-prefilled request
        # resets its slot at position zero, while a disaggregated generation-only
        # request receives a freshly zeroed slot.)
        is_warmup = (
            getattr(spec_metadata, "is_cuda_graph", False)
            and not torch.cuda.is_current_stream_capturing()
        )
        if is_warmup:
            saved_ctx_len = self._ctx_len.clone()
            saved_valid_len = self._valid_len.clone()
            saved_position_initialized = self._position_initialized.clone()
            saved_windows = self._kv_windows.clone()
            # The confidence buffer is persistent state too; warmup scores
            # must not reach the verification planner.
            saved_confidence = (
                None if self._confidence_logits is None else self._confidence_logits.clone()
            )

        # Assign / reset window slots for context (prefill) requests and seed each
        # request's rolling KV window from its prompt's captured context, so the
        # first generation step drafts against real context instead of an all-zero
        # window (acceptance-rate only; verified decoding keeps output correct).
        if num_contexts > 0:
            self._seed_context_windows(
                draft_model,
                spec_metadata,
                attn_metadata,
                position_ids,
                total_target_tokens,
            )

        # FUSED_COMM MoE backends (DeepGEMM MegaMoE) synchronize EP ranks with an
        # in-kernel phase-flip NVLink barrier that flips on every kernel call, so
        # every rank must invoke the draft MoE the same number of times and with
        # the same globally-gathered per-rank token list, or the barrier desyncs
        # (hang / "unspecified launch failure"). The draft runs over generation
        # requests only, each expanded to ``block`` positions, so the per-rank
        # draft-MoE token count is ``num_gens * block``. ``all_rank_num_gens`` is
        # gathered at metadata-prep time (model_engine, outside any CUDA-graph
        # capture region); it is None for non-ADP / single-rank runs, where the
        # local ``[num_tokens]`` fallback in ``_forward_stage`` is correct.
        block = int(draft_model.block_size)
        all_rank_num_gens = getattr(spec_metadata, "all_rank_num_gens", None)
        # A rank with zero local gen requests still has to cross the draft MoE's
        # cross-rank barrier, but DeepseekV4MoE's router / shared-expert dense
        # GEMMs reject a 0-row input (cuBLAS CUBLAS_STATUS_INVALID_VALUE), so such
        # a rank runs a single 1-row dummy through the MoE (like ADP padding).
        # Encode that as ``1`` in the globally-shared per-rank token list so every
        # rank agrees on the FUSED_COMM chunk count and per-rank slice.
        all_rank_draft_tokens = (
            [max(1, int(g) * block) for g in all_rank_num_gens]
            if all_rank_num_gens is not None
            else None
        )
        global_has_gen = (
            max(all_rank_num_gens) > 0 if all_rank_num_gens is not None else num_gens > 0
        )

        if num_gens > 0:
            # The batched gen-block draft returns the per-position corrected block
            # logits [num_gens, K, vocab] and is CUDA-graph-safe.
            gen_logits = self._draft_gen_block_batched(
                draft_model,
                spec_metadata,
                attn_metadata,
                accepted_tokens,
                num_accepted_tokens,
                num_contexts,
                batch_size,
                total_target_tokens,
                position_ids,
                all_rank_num_tokens=all_rank_draft_tokens,
            )
            if gen_logits is not None:
                if self.guided_decoder is not None:
                    gen_draft_tokens = self._sample_draft_tokens_guided(
                        gen_logits,
                        spec_metadata,
                        accepted_tokens,
                        num_accepted_tokens,
                        num_contexts,
                        batch_size,
                        K,
                    )
                else:
                    # SpecWorkerBase samples the draft tokens.
                    gen_draft_tokens = self.sample_draft_tokens(
                        gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                    )
                # The context one-hot must match the width the gen scatter just
                # published to draft_probs, NOT gen_logits.shape[-1]: under TP the
                # draft logits are vocab-sharded and sample_draft_tokens gathers
                # them to full vocab before scattering, so the pre-gather shard
                # width would leave stale columns and corrupt rejection.
                gen_vocab = spec_metadata.draft_probs_last_dim
            else:
                gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
                gen_vocab = None
        else:
            # No local generation requests: if any peer EP rank has some, we must
            # still cross the draft MoE's cross-rank barrier the same number of
            # times (zero-token) so a FUSED_COMM phase-flip barrier stays lockstep.
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
            self._valid_len.copy_(saved_valid_len)
            self._position_initialized.copy_(saved_position_initialized)
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
        # Ragged steps: the TOKEN windows the step actually verified, straight
        # off the persistent layout buffer. Rides the sampler's existing D2H
        # batch; under device-window selection this is the ONLY place the host
        # can learn the true windows (py_verify_len then holds only the host
        # shape split), and the rewind arithmetic must use it.
        #
        # Gen-only steps only: the buffer is generation-indexed while every
        # other output row is batch-indexed, so on a mixed step (eager ragged
        # with finished contexts) the rows would misalign AND come up short.
        # Mixed steps keep exact windows on py_verify_len, so the sampler's
        # snapshot fallback stays correct there.
        _publish_policy_window_output(outputs, spec_metadata.verify_lens, batch_size)
        return outputs


class DSparkWorker(DFlashWorker):
    """Worker for a *standalone* DSpark drafter (DFlash lineage).

    DSpark is DFlash plus two extra heads, so the drafting plumbing is
    inherited wholesale from :class:`DFlashWorker` -- paged context K/V,
    slot management, the mask-token block forward -- and only the two
    head-driven policies are overridden here: the block-output slot
    convention (``shift_label``) and the Markov intra-block logit bias.

    Mirrors the model side, where ``GQADSparkForCausalLM`` extends
    ``DFlashForCausalLM`` with the same two heads.

    Naming: this is the unqualified DSpark worker because a separately
    shipped drafter is the ordinary case; :class:`DSv4DSparkWorker` carries
    the qualifier because a draft embedded in the target checkpoint is the
    special one. Workers are classified by *deployment form*, never by draft
    backbone -- so there is no ``Qwen3DSparkWorker``. Note the name meant the
    embedded worker before this split; both the rebind and the rename to
    ``DSv4DSparkWorker`` land in one commit so the swap reads as a unit.

    A worker is agnostic to the draft backbone: everything backbone-shaped is
    supplied by the draft model, which reports its own shapes
    (``_num_attn_layers``, ``_num_heads``, ``_num_kv_heads``, ``_head_dim``)
    and owns the operators (``_build_fused_kv_buffers``,
    ``precompute_context_kv``, ``dflash_forward``,
    ``apply_markov_chain_logits``, ``project_target_hidden``). The worker only
    allocates against the reported shapes and sequences the calls. An MLA
    drafter therefore reuses this class unchanged; its differences (fused-QKV
    assumptions, a 576-latent K/V layout) land in its own draft-model
    subclass. Naming workers by backbone would produce N classes with
    identical bodies.

    Deployment form is the axis the runtime state actually splits on: paged
    draft K/V here, a worker-owned rolling window in
    :class:`DSv4DSparkWorker`.
    """

    def set_draft_model(self, draft_model) -> None:
        """Reject an unsupported vocab mapping here rather than mid-decode.

        ``d2t`` is model-static, so a config mistake should surface at load and
        not as a ``NotImplementedError`` raised per decode step, possibly during
        CUDA-graph capture.
        """
        super().set_draft_model(draft_model)
        if self._d2t is not None and getattr(draft_model, "has_markov_head", False):
            raise NotImplementedError(
                "DSpark Markov head requires a shared draft/target vocab "
                "(d2t vocab mapping is not supported); drafter "
                f"{type(draft_model).__name__} declares one."
            )

    def _draft_block_width(self, draft_model) -> int:
        """Block width under the dspark ``shift_label`` convention.

        shift_label reads slots 0..K-1, so K draft tokens fit in K slots and
        the base class' K+1 over-demands by one -- enough to reject a block-7
        checkpoint at max_draft_len=7, which is how both published DSpark
        drafters are meant to run.
        """
        if getattr(draft_model, "_dspark_shift_label", False):
            return self.max_draft_len
        return super()._draft_block_width(draft_model)

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """Block-output slots under the dspark ``shift_label`` convention.

        The drafter checkpoint declares the convention, so it is read off the
        draft model rather than assumed: a DSpark drafter trained with the
        legacy DFlash slot layout keeps the base class' slots 1..K.
        """
        shift_label = getattr(draft_model, "_dspark_shift_label", False)
        return dflash_draft_slot_ids(
            num_gens, block_size, num_draft_tokens, shift_label, device="cuda"
        )

    def _refine_block_logits(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        inputs: dict,
        spec_metadata,
    ) -> torch.Tensor:
        """Add the greedy-chained Markov intra-block bias to the block logits.

        A DSpark drafter checkpoint may omit the Markov head (``markov_rank``
        0), which loads as a drafter without one; that case falls through to
        the unmodified backbone logits.
        """
        if not getattr(draft_model, "has_markov_head", False):
            return gen_logits
        return self._apply_dspark_markov_bias(
            draft_model, gen_logits, inputs["first_prev_tokens"], spec_metadata
        )

    def _apply_dspark_markov_bias(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        spec_metadata,
    ) -> torch.Tensor:
        """Apply the dspark vanilla-Markov intra-block bias to block logits.

        Reference (DeepSpec VanillaMarkov.sample_block_tokens, temperature 0):
        step i adds bias = markov_w2 @ markov_w1[prev_i] to the shared-lm_head
        logits, where prev_0 is the anchor (last accepted) token and prev_{i>0}
        is the greedy token from step i-1's biased logits. Greedy per-position
        argmax of the returned logits therefore reproduces the reference
        sampled chain; the rejection-sampling path samples from the same
        biased distributions (proposal conditioned on the greedy chain).

        Handles a TP vocab-sharded draft lm_head by slicing markov_w2's rows
        to this rank's contiguous shard and chaining through the TP-aware
        global argmax.
        """
        # The d2t guard lives in set_draft_model: it is model-static, so raising
        # it here would surface a load-time config error per decode step.
        # Unlike the d2t guard this one cannot move to set_draft_model: it
        # keys on the runtime logits width, and reproducing that at init would
        # duplicate the draft head's sharding rules. A standalone drafter
        # borrows the target lm_head, whose gather_output defaults to True, so
        # the logits normally arrive full-vocab and this branch is skipped.
        full_vocab = draft_model.markov_w2.shape[0]
        shard = gen_logits.shape[-1]
        vocab_slice = None
        if shard != full_vocab:
            mapping = self.mapping
            if (
                mapping is None
                or getattr(mapping, "enable_attention_dp", False)
                or shard * mapping.tp_size != full_vocab
            ):
                raise NotImplementedError(
                    f"DSpark Markov head: draft logits width {shard} does not "
                    f"match the drafter vocab {full_vocab} and is not a plain "
                    "TP column shard of it."
                )
            vocab_slice = slice(mapping.tp_rank * shard, (mapping.tp_rank + 1) * shard)

        def argmax_fn(step_logits):
            # Full-vocab token ids (TP-aware when sharded); tokens stay in
            # draft-vocab space, which is what markov_w1 indexes.
            return self.greedy_sample_draft_with_tp_gather(step_logits, spec_metadata).long()

        return draft_model.apply_markov_chain_logits(
            gen_logits,
            first_prev_tokens,
            argmax_fn=argmax_fn,
            vocab_slice=vocab_slice,
        )
