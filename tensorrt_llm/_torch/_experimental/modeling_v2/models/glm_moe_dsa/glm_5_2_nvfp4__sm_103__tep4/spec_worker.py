# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative worker of the glm_moe_dsa sm_103 tep4 target: the one-model MTP-Eagle
worker (`Glm52MTPEagleWorker`) with its greedy picks and the MTP draft loop's bookkeeping on
`kernels/spec/draft_loop.py`.

`ModelingV2Glm52Nvfp4Sm103Tep4` (`modeling.py`) builds the worker in place of the built-in
`MTPEagleWorker`.

The draft loop's bookkeeping is ported from the glm_moe_dsa draft loop (`models/glm_moe_dsa/shared/kernels/draft_loop/`,
TensorRT-LLM PR #20524 at d8735393c2). Around each one-model MTP decode step the built-in
`MTPEagleWorker` runs about a hundred small torch ops on the target's stream: the strict
acceptance (`cumprod`, compares, the forced-acceptance draw), the step-0 gather rows, the
metadata refresh before the loop and one metadata update after every draft step (the 2-D KV
lengths, three DeepGEMM schedules each built then copied, the fused DSA decode metadata, the
position advance), and the final draft-token stack and `next_new_tokens`. `DraftLoop` does
each group in one launch, with the same results:

* `accept`: the acceptance, the draft-0 input ids (the accepted tokens, flattened) and the
  step-0 gather rows;
* `step`: one metadata update (`MODE_REFRESH` before step 0, `MODE_STEP0`, `MODE_ADVANCE`),
  its host side calling the built-in host code; the device side is `draft_step`;
* `finalize`: `next_draft_tokens` and `next_new_tokens`.

The domain is checked on the host before any device work (`ineligibility`,
`step_ineligibility`); outside it the built-in code runs. `TRTLLM_GLM52_DRAFT_LOOP_VERIFY=1`
runs the built-in code first on the live state, restores the device state, runs the fused op
and counts every field that differs, on the device (`DraftLoopVerify`).
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import os
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import MTPIndexShareMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.speculative import MTPEagleWorker
from tensorrt_llm._torch.speculative.eagle3 import _reset_mtp_index_share
from tensorrt_llm.logger import logger

from .kernels.spec import draft_loop as _k
from .ops.sampling import cute_dsl_argmax as _argmax_op
from .ops.sampling.cute_dsl_argmax import cute_dsl_argmax

if TYPE_CHECKING:
    # Annotation only: `modeling.py` imports this module.
    from .modeling import GlmMoeDsaTep4ModelingV2

MODE_REFRESH = _k.MODE_REFRESH
MODE_STEP0 = _k.MODE_STEP0
MODE_ADVANCE = _k.MODE_ADVANCE


@functools.lru_cache(maxsize=1)
def enabled() -> bool:
    """`TRTLLM_GLM52_DRAFT_LOOP` (default 1; 0 keeps the built-in bookkeeping). Read once."""
    return os.environ.get("TRTLLM_GLM52_DRAFT_LOOP", "1") == "1"


_DECLINES_LOGGED: set[str] = set()


def log_decline(what: str, reason: str) -> None:
    """Log, once per process and reason, why `what` kept the built-in code."""
    if reason not in _DECLINES_LOGGED:
        _DECLINES_LOGGED.add(reason)
        logger.info(f"[glm52] draft loop: {what} keeps the built-in code: {reason}")


@functools.lru_cache(maxsize=1)
def verify_enabled() -> bool:
    """`TRTLLM_GLM52_DRAFT_LOOP_VERIFY=1`: the in-pipeline oracle (diagnostics). Read once."""
    return os.environ.get("TRTLLM_GLM52_DRAFT_LOOP_VERIFY", "0") == "1"


def ineligibility(worker, attn_metadata, spec_metadata) -> str | None:
    """Why this forward keeps the built-in bookkeeping, or None. Host only."""
    if not worker.is_mtp_eagle:
        return "not MTP Eagle one-model"
    if worker._uses_external_shared_target_kv:
        return "external shared-target KV draft loop"
    if worker.guided_decoder is not None or worker.sa_enhancer is not None:
        return "guided decoding or SA enhancer"
    if worker._auxiliary_state_handlers:
        return "auxiliary speculative state handlers"
    if getattr(worker.spec_config, "use_relaxed_acceptance_for_thinking", False):
        return "relaxed acceptance"
    if worker.use_dynamic_tree:
        return "dynamic tree"
    if getattr(spec_metadata, "enable_penalty", False):
        return "occurrence penalties"
    if worker._can_use_rejection_sampling(spec_metadata):
        return "rejection sampling"
    if spec_metadata.wants_advanced_draft_sampling:
        return "advanced draft sampling"
    if spec_metadata.is_ragged_verify:
        return "ragged verification"
    if attn_metadata.num_contexts != 0:
        return "context rows present"
    batch_size = attn_metadata.num_seqs
    draft_len = spec_metadata.runtime_draft_len
    if not 0 < draft_len <= _k.MAX_DRAFT_LEN or not 0 < batch_size <= _k.MAX_BATCH:
        return "batch or draft length outside the fused domain"
    draft_tokens = spec_metadata.draft_tokens
    if (
        draft_tokens is None
        or draft_tokens.dtype != torch.int32
        or draft_tokens.numel() != batch_size * draft_len
        or not draft_tokens.is_contiguous()
    ):
        return "draft-token buffer layout"
    if spec_metadata.batch_indices_cuda.dtype != torch.int32:
        return "batch-index dtype"
    model_config = worker.model_config
    if model_config is not None and model_config.mapping.enable_lm_head_tp_in_adp:
        return "LM-head TP in attention DP"
    return None


def step_ineligibility(md, batch_size: int) -> str | None:
    """Why the loop's metadata update keeps the built-in chain, or None. Host only.

    The domain is the GEN server's pure-decode DSA metadata: no context rows, the native
    fused decode metadata, uncompressed indexer lengths, no expanded MTP buffers, no ragged
    verification and no FP8-DS / FP4 MLA state. The dense top-k fallback and the GVR row
    reorder are in it: the built-in update builds both last, from the lengths it just wrote,
    and `fused_step` calls the same code at the same point.
    """
    from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
        _fused_dsa_meta_enabled,
    )

    if not isinstance(md, DSAtrtllmAttentionMetadata):
        return "metadata is not DSAtrtllmAttentionMetadata"
    cls = type(md)
    for name in ("update_for_spec_dec", "on_update_kv_lens", "on_update"):
        if getattr(cls, name) is not getattr(DSAtrtllmAttentionMetadata, name):
            return f"metadata overrides {name}"
    if md.kv_cache_manager is None or not is_dsa_cache_manager(md.kv_cache_manager):
        return "KV cache manager is not a DSA cache manager"
    if md.num_contexts != 0 or batch_size != md.num_seqs or md.num_tokens <= 0:
        return "not a pure decode batch"
    if not 0 < batch_size <= _k.MAX_BATCH:
        return f"batch size {batch_size} outside the fused domain"
    if not _fused_dsa_meta_enabled():
        return "native fused DSA metadata disabled"
    if md.use_fp8_ds_mla or getattr(md, "fp4_mla_state", None) is not None:
        return "FP8-DS/FP4 MLA metadata state"
    if md.enable_ragged_verification or md.is_ragged_verify:
        return "ragged verification"
    if md._indexer_compress_ratio > 1:
        return "compressed indexer lengths"
    if md.use_expanded_buffers_for_mtp:
        return "expanded MTP buffers"
    if md.max_draft_tokens <= 0:
        return "no draft tokens"
    if (
        md.kv_lens_cuda.dtype != torch.int32
        or md.seq_lens_cuda.dtype != torch.int32
        or md.indexer_k_cache_block_offsets.dtype != torch.int32
    ):
        return "unexpected length or block-offset dtype"
    if _k.schedule_split_kv(md.num_sms, md.kv_lens_cuda.device) is None:
        return "DeepGEMM schedule not reproduced"
    return None


def _host_on_update_kv_lens(md) -> None:
    """The rest of the built-in `on_update_kv_lens` after its decode-metadata writes (the
    host bookkeeping, then the GVR row reorder and the dense top-k fallback, each a no-op
    unless the step needs it)."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    TrtllmAttentionMetadata.on_update_kv_lens(md)
    md._invalidate_pool_view_cache()
    md.gen_indexer_kv_lens_cuda_runtime = md.get_indexer_kv_lens(
        md.kv_lens_cuda[md.num_contexts : md.num_seqs]
    )
    md._fused_dsa_meta_armed = True
    md._compute_kv_lens_row_reorder()
    md.prepare_dense_topk_indices(md.kv_lens_cuda, device=True)


def native_step(md, mode, batch_size, draft_len, num_accepted, position_ids, gather_ids):
    """The built-in loop's code for one metadata update (the oracle, and the fallback's
    reference): `on_update_kv_lens` before step 0, the step-0 switch and the per-step
    advance with their `update_for_spec_dec`. Returns the next step's positions."""
    from tensorrt_llm._torch.speculative.mtp import _select_mtp_position_ids

    if mode == MODE_REFRESH:
        md.on_update_kv_lens()
        return None
    new_position_ids = None
    if position_ids is not None:
        rows = gather_ids if mode == MODE_STEP0 else slice(0, batch_size)
        new_position_ids = _select_mtp_position_ids(position_ids, rows) + 1
    if mode == MODE_STEP0:
        md._seq_lens[:batch_size].fill_(1)
        md._seq_lens_cuda[:batch_size].fill_(1)
        md.on_update()
        md.num_contexts = 0
        md.kv_lens_cuda[:batch_size] -= draft_len - num_accepted
        md.update_for_spec_dec()
        md.use_spec_decoding = False
    else:
        md.kv_lens_cuda[:batch_size] += 1
        md.update_for_spec_dec()
    return new_position_ids


def fused_step(md, mode, batch_size, draft_len, num_accepted, position_ids, gather_ids):
    """`native_step` on `draft_step`: one launch plus the built-in host code."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    if mode == MODE_STEP0:
        # The host side of the step-0 switch to one query token per request.
        md._seq_lens[:batch_size].fill_(1)
        md.on_update()
        md.num_contexts = 0
    position_ids = _k.draft_step(
        md,
        mode=mode,
        batch_size=batch_size,
        draft_len=draft_len,
        split_kv=_k.schedule_split_kv(md.num_sms, md.kv_lens_cuda.device),
        num_accepted=num_accepted,
        position_ids=position_ids,
        gather_ids=gather_ids,
    )
    if mode != MODE_REFRESH:
        # `update_for_spec_dec`'s host side; its `on_update_kv_lens` follows.
        TrtllmAttentionMetadata.update_for_spec_dec(md)
        md.max_ctx_kv_len = 0
        md.num_ctx_cached_tokens = 0
        md.max_gen_seq_len = 1
    _host_on_update_kv_lens(md)
    if mode == MODE_STEP0:
        md.use_spec_decoding = False
    return position_ids


class DraftLoop:
    """The fused bookkeeping of one one-model MTP decode forward (see the module docstring).

    Built by `Glm52MTPEagleWorker.sample_and_accept_draft_tokens` when `ineligibility` is
    None; read by the worker's later overrides of the same forward."""

    def __init__(self, worker, attn_metadata, spec_metadata) -> None:
        self._worker = worker
        self._spec_metadata = spec_metadata
        self.batch_size = attn_metadata.num_seqs
        self.draft_len = spec_metadata.runtime_draft_len
        self.accepted: torch.Tensor | None = None
        self.num_accepted: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self.gather_ids: torch.Tensor | None = None
        self.next_new_tokens: torch.Tensor | None = None
        # The metadata state is per forward: once a step declines, the rest stay built-in.
        self.steps_engaged = True

    def accept(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The target's tokens (the worker's own pick), then the fused acceptance.

        Returns `(accepted [B, K+1] int32, num_accepted [B] int32)` as the built-in strict
        acceptance with the forced-acceptance override returns them."""
        from tensorrt_llm._torch.speculative import interface as _iface

        worker, spec_metadata = self._worker, self._spec_metadata
        batch_size, draft_len = self.batch_size, self.draft_len
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        verify = VERIFY if verify_enabled() else None
        if verify is not None:
            verify.before_accept(worker, logits, spec_metadata, batch_size, draft_len)
        target_tokens = worker._sample_tokens_for_batch(logits, spec_metadata, 0, batch_size)

        force_mode, base_total, frac = 0, 0, 0.0
        counter = pool = spec_metadata.batch_indices_cuda
        force = worker.force_num_accepted_tokens
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if force != 0.0 and not is_warmup:
            int_part = int(force)
            frac = force - int_part
            base_total = min(int_part + 1, draft_len + 1)
            if frac > 0.0 and base_total < draft_len + 1:
                worker._ensure_force_accept_rng_state(logits.device)
                force_mode = 2
                counter = worker._force_accept_rng_counter
                pool = worker._force_accept_rng_pool
            else:
                force_mode = 1
        self.accepted, self.num_accepted, self.input_ids, self.gather_ids = _k.accept(
            target_tokens,
            spec_metadata.draft_tokens,
            spec_metadata.batch_indices_cuda,
            counter,
            pool,
            batch_size=batch_size,
            draft_len=draft_len,
            force_mode=force_mode,
            base_total=base_total,
            frac=frac,
            counter_stride=_iface._FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            slot_stride=_iface._FORCE_ACCEPT_RNG_SLOT_STRIDE,
            pool_size=_iface._FORCE_ACCEPT_RNG_POOL_SIZE,
        )
        if verify is not None:
            verify.after_accept(self)
        return self.accepted, self.num_accepted

    def step(self, md, mode: int, position_ids, gather_ids) -> tuple[bool, torch.Tensor | None]:
        """One metadata update of the loop: `(fused, next positions)`; `fused` False when the
        caller must run the built-in update (no state has been modified then)."""
        if not self.steps_engaged:
            return False, None
        reason = step_ineligibility(md, self.batch_size)
        self._worker._glm_count("draft_step", reason is None)
        if reason is not None:
            log_decline("the metadata update", reason)
            self.steps_engaged = False
            return False, None
        args = (md, mode, self.batch_size, self.draft_len, self.num_accepted, position_ids,
                gather_ids)  # fmt: skip
        if verify_enabled():
            return True, VERIFY.step(*args)
        return True, fused_step(*args)

    def finalize(self, draft_tokens: list[torch.Tensor]) -> torch.Tensor | None:
        """`next_draft_tokens [B, K]`, with `next_new_tokens [B, K+1]` kept for the worker's
        `_prepare_next_new_tokens`; None when the built-in stack must run."""
        hit = len(draft_tokens) == self.draft_len and all(
            t.dtype == torch.int32 and t.shape == (self.batch_size,) and t.is_contiguous()
            for t in draft_tokens
        )
        self._worker._glm_count("draft_finalize", hit)
        if not hit:
            return None
        batch_indices = self._spec_metadata.batch_indices_cuda
        next_draft, self.next_new_tokens = _k.finalize(
            draft_tokens, self.accepted, self.num_accepted, batch_indices
        )
        if verify_enabled():
            VERIFY.finalize(self, draft_tokens, next_draft)
        return next_draft


# ---- the in-pipeline oracle (TRTLLM_GLM52_DRAFT_LOOP_VERIFY=1) --------------------------

_MAX_SNAPSHOT_NUMEL = 1 << 22


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        return t.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            t.element_size()])  # fmt: skip
    return t


class DraftLoopVerify:
    """Device-side mismatch counters of the fused ops against the built-in code.

    Each verified op first runs the built-in code on the live state, records its outputs,
    restores the device state it touched and runs the fused op; every differing field adds 1
    to its counter. The comparisons are ordinary stream work, so they are captured into the
    CUDA graphs and evaluated on every replay. The counters print at each eager accept and at
    exit. The fused results are the ones kept."""

    _POOL = 1024

    def __init__(self) -> None:
        self.names: list[str] = []
        self.pool: torch.Tensor | None = None
        # Per counter: the last mismatch's (first differing flat index, native value, fused
        # value, differing elements), and per batch size the metadata shapes at capture.
        self.detail: torch.Tensor | None = None
        self.shapes: dict[str, str] = {}
        self.last: dict | None = None
        self._accept_native = None

    def _counter(self, name: str, device) -> torch.Tensor:
        if self.pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("draft-loop verify counters must be created before capture")
            self.pool = torch.zeros((self._POOL,), dtype=torch.int64, device=device)
            self.detail = torch.zeros((self._POOL, 4), dtype=torch.int64, device=device)
        if name not in self.names:
            self.names.append(name)
        return self.pool[self.names.index(name)]

    def count(self, name: str, native: torch.Tensor, fused: torch.Tensor) -> None:
        counter = self._counter(name, native.device)
        if native.shape != fused.shape or native.dtype != fused.dtype:
            counter.add_(1)
        else:
            diff = (_bits(native) != _bits(fused)).reshape(-1)
            counter.add_(diff.any().to(torch.int64))
            if diff.numel():
                n_diff = diff.sum()
                at = torch.argmax(diff.to(torch.int32))
                pick = at.reshape(1).to(torch.int64)
                row = torch.cat([pick, _bits(native).reshape(-1).index_select(0, pick).to(torch.int64),
                                 _bits(fused).reshape(-1).index_select(0, pick).to(torch.int64),
                                 n_diff.reshape(1).to(torch.int64)])  # fmt: skip
                slot = self.detail[self.names.index(name)]
                slot.copy_(torch.where(n_diff > 0, row, slot))

    def call(self, name: str, device) -> None:
        self._counter(name, device).add_(1)

    def report(self, force: bool = False) -> None:
        if self.pool is None or torch.cuda.is_current_stream_capturing():
            return
        values = dict(zip(self.names, self.pool[: len(self.names)].tolist()))
        if force or values != self.last:
            self.last = values
            calls = {k: v for k, v in values.items() if k.endswith(":calls")}
            bad = {k: v for k, v in values.items() if v and not k.endswith(":calls")}
            print(
                f"[glm52] draft_loop verify: calls {calls} mismatching fields {bad or 'none'}",
                flush=True,
            )
            if bad:
                rows = self.detail[: len(self.names)].tolist()
                at = {k: rows[self.names.index(k)] for k in bad}
                print(f"[glm52] draft_loop verify: last mismatch [index, native, fused, n] {at}",
                      flush=True)  # fmt: skip
                print(f"[glm52] draft_loop verify: shapes {self.shapes}", flush=True)

    def before_accept(self, worker, logits, spec_metadata, batch_size, draft_len) -> None:
        """The built-in acceptance on the live state; the forced-acceptance counter it bumps
        is put back so the fused acceptance draws the same values. The draw state is built
        first: built inside the native call, it would be put back as absent."""
        if worker.force_num_accepted_tokens != 0.0:
            worker._ensure_force_accept_rng_state(logits.device)
        counter = worker._force_accept_rng_counter
        saved = counter.clone() if counter is not None else None
        draft = spec_metadata.draft_tokens.reshape(batch_size, draft_len)
        accepted, num_accepted = worker._sample_and_accept_draft_tokens_base(
            logits, draft, 0, batch_size, spec_metadata
        )
        if saved is not None:
            worker._force_accept_rng_counter.copy_(saved)
        rows = (spec_metadata.batch_indices_cuda[:batch_size] * (draft_len + 1)).long()
        self._accept_native = (accepted, num_accepted, rows + num_accepted - 1)

    def after_accept(self, loop: DraftLoop) -> None:
        accepted, num_accepted, rows = self._accept_native
        self._accept_native = None
        self.call("accept:calls", accepted.device)
        self.count("accept:accepted", accepted, loop.accepted)
        self.count("accept:num_accepted", num_accepted, loop.num_accepted)
        self.count("accept:input_ids", accepted.reshape(-1), loop.input_ids)
        self.count("accept:gather_ids", rows, loop.gather_ids)

    def step(self, md, mode, batch_size, draft_len, num_accepted, position_ids, gather_ids):
        names = sorted(
            n for n, v in vars(md).items()
            if isinstance(v, torch.Tensor) and v.is_cuda and 0 < v.numel() <= _MAX_SNAPSHOT_NUMEL
        )  # fmt: skip
        before = {n: getattr(md, n).clone() for n in names}
        native_pos = native_step(
            md, mode, batch_size, draft_len, num_accepted, position_ids, gather_ids
        )
        after = {n: getattr(md, n).clone() for n in names}
        for n in names:
            current = getattr(md, n)
            if current.shape == before[n].shape:
                current.copy_(before[n])
        fused_pos = fused_step(
            md, mode, batch_size, draft_len, num_accepted, position_ids, gather_ids
        )
        tag = f"{('refresh', 'step0', 'advance')[mode]}:B{batch_size}"
        expand = md.dsl_expand_factor if md.expand_for_dsl else 1
        self.shapes.setdefault(tag, (
            f"tokens={md.num_tokens} stride={md.gen_token_stride} "
            f"kv2d={tuple(md.kv_lens_cuda_2d.shape)} expand={expand}"))  # fmt: skip
        self.call(f"{tag}:calls", md.kv_lens_cuda.device)
        for n in names:
            self.count(f"{tag}:{n}", after[n], getattr(md, n))
        if native_pos is not None:
            self.count(f"{tag}:position_ids", native_pos, fused_pos)
        return fused_pos

    def finalize(self, loop: DraftLoop, draft_tokens, next_draft) -> None:
        stacked = torch.stack(draft_tokens, dim=1)
        expected_new = loop._worker._prepare_next_new_tokens_builtin(
            loop.accepted, stacked, loop._spec_metadata.batch_indices_cuda, loop.batch_size,
            loop.num_accepted,
        )  # fmt: skip
        self.call("finalize:calls", next_draft.device)
        self.count("finalize:next_draft", stacked, next_draft)
        self.count("finalize:next_new", expected_new, loop.next_new_tokens)


VERIFY = DraftLoopVerify()


@atexit.register
def _report_at_exit() -> None:
    try:
        VERIFY.report(force=True)
    except Exception:  # noqa: BLE001 - best effort at interpreter exit
        pass


# ---- the worker -------------------------------------------------------------------------


class Glm52MTPEagleWorker(MTPEagleWorker):
    """The one-model MTP-Eagle worker with its two all-greedy picks on the CuTe DSL argmax.

    * `greedy_sample_draft_with_tp_gather`: the draft token. This target's draft logits
      are the gathered full vocabulary on every rank (`MTPLayer.shared_head`), so the
      built-in worker takes its plain `argmax` branch; this is that argmax. A call with
      `mapping_lm_head_tp` (LM-head TP under attention DP) keeps the built-in code.
    * `_sample_tokens_for_batch`, all-greedy batch: the target's token. `torch.argmax`
      there, int64; int32 here, which every caller stores into an int32 tensor.

    `TRTLLM_GLM52_GREEDY_ARGMAX=0` keeps both picks on the built-in code.

    Logits the kernel does not take keep the built-in code, counted as misses. Until the
    core's argmax launcher is built (`post_load_weights`), or if it failed to build, both
    methods are the built-in ones and nothing is counted.

    On a pure decode step it also runs the draft loop's bookkeeping on one launch per group
    (`DraftLoop` above, ported from the glm_moe_dsa draft loop of PR #20524): the acceptance and
    draft-0 inputs, the metadata refresh before draft step 0 and the update after each draft
    step, and the final draft-token stack with `next_new_tokens`. Routes `draft_accept`,
    `draft_step`, `draft_finalize`; `TRTLLM_GLM52_DRAFT_LOOP=0` keeps the built-in code. Every
    other step, and everything else, is the built-in worker's.
    """

    # The draft loop's fused bookkeeping of the running forward (`_forward_impl`), else None.
    _glm_draft_loop: DraftLoop | None = None

    def __init__(
        self,
        spec_config,
        model_config: ModelConfig,
        use_separate_draft_kv_cache: bool,
        *,
        mapping,
        core: GlmMoeDsaTep4ModelingV2,
    ) -> None:
        super().__init__(spec_config, model_config, use_separate_draft_kv_cache, mapping=mapping)
        # The core's route counter and its argmax launcher (rebuilt at post_load_weights),
        # reached through a closure: the core belongs to the shell and is no submodule here.
        self._glm_count = core._count
        self._glm_argmax_rows = lambda: core._rows.argmax
        # TRTLLM_GLM52_GREEDY_ARGMAX (default 1): 0 keeps both greedy picks on the built-in
        # torch.argmax.
        self._glm_argmax_on = os.environ.get("TRTLLM_GLM52_GREEDY_ARGMAX", "1") == "1"

    def _glm_pick(
        self, logits: torch.Tensor, route: str, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """int32 argmax tokens of `logits` on the CuTe DSL kernel, or None for the built-in.
        `out` (int32 `[rows]`, contiguous), when given, receives them and is returned."""
        rows = self._glm_argmax_rows() if self._glm_argmax_on else None
        if rows is None:
            return None
        hit = _argmax_op.takes(logits)
        self._glm_count(route, hit)
        if not hit:
            return None
        if out is None:
            out = torch.empty([logits.shape[0]], dtype=torch.int32, device=logits.device)
        return cute_dsl_argmax(logits, out, rows)

    def greedy_sample_draft_with_tp_gather(
        self,
        logits: torch.Tensor,
        spec_metadata=None,
        mapping_lm_head_tp=None,
        tokens_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The base contract: with `tokens_out` the tokens are written there and it is returned."""
        if mapping_lm_head_tp is None:
            tokens = self._glm_pick(logits, "greedy_draft", out=tokens_out)
            if tokens is not None:
                return tokens
        return super().greedy_sample_draft_with_tp_gather(
            logits, spec_metadata, mapping_lm_head_tp, tokens_out=tokens_out
        )

    def _sample_tokens_for_batch(
        self, logits: torch.Tensor, spec_metadata, num_contexts: int, batch_size: int
    ) -> torch.Tensor:
        if spec_metadata.is_all_greedy_sample:
            tokens = self._glm_pick(logits, "greedy_target")
            if tokens is not None:
                return tokens
        return super()._sample_tokens_for_batch(logits, spec_metadata, num_contexts, batch_size)

    # ---- the draft loop's fused bookkeeping ----

    def _forward_impl(self, *args, **kwargs):
        """The built-in forward. `sample_and_accept_draft_tokens` decides the fused
        bookkeeping, the later overrides read it, and it never outlives the forward."""
        self._glm_draft_loop = None
        try:
            return super()._forward_impl(*args, **kwargs)
        finally:
            self._glm_draft_loop = None

    def sample_and_accept_draft_tokens(self, input_ids, logits, attn_metadata, spec_metadata):
        if verify_enabled():
            # The oracle's counters, at every eager forward (context steps included).
            VERIFY.report()
        if enabled():
            reason = ineligibility(self, attn_metadata, spec_metadata)
            self._glm_count("draft_accept", reason is None)
            if reason is not None and attn_metadata.num_contexts == 0:
                log_decline("the acceptance", reason)
            if reason is None:
                self._glm_draft_loop = DraftLoop(self, attn_metadata, spec_metadata)
                return self._glm_draft_loop.accept(logits)
        return super().sample_and_accept_draft_tokens(
            input_ids, logits, attn_metadata, spec_metadata
        )

    def prepare_1st_drafter_inputs(
        self, input_ids, position_ids, hidden_states, accepted_tokens, attn_metadata,
        spec_metadata, draft_model,
    ):  # fmt: skip
        loop = self._glm_draft_loop
        if loop is None:
            return super().prepare_1st_drafter_inputs(
                input_ids, position_ids, hidden_states, accepted_tokens, attn_metadata,
                spec_metadata, draft_model,
            )  # fmt: skip
        # No context rows: the draft-0 input ids are the accepted tokens, flattened.
        return {
            "input_ids": loop.input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def _forward_linear_draft_loop(
        self, inputs, attn_metadata, spec_metadata, draft_model, draft_kv_cache_manager,
        num_contexts, batch_size, num_accepted_tokens, original_all_rank_num_tokens,
        accepted_tokens=None, pre_draft_seq_lens=None,
    ):  # fmt: skip
        """The built-in MTP-Eagle draft loop restricted to the fused domain (no context rows,
        no guided decoding, SA enhancer or LM-head TP: `ineligibility`), with the
        metadata updates on `DraftLoop.step` and the stack on `DraftLoop.finalize`.

        Returns `(next_draft_tokens, next_new_tokens)` like the built-in loop; the fused path
        returns None for the second, and `_prepare_next_new_tokens` hands out the loop's."""
        from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager
        from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
            DSAtrtllmAttentionMetadata,
        )

        loop = self._glm_draft_loop
        if loop is None or not isinstance(attn_metadata, DSAtrtllmAttentionMetadata):
            return super()._forward_linear_draft_loop(
                inputs, attn_metadata, spec_metadata, draft_model, draft_kv_cache_manager,
                num_contexts, batch_size, num_accepted_tokens, original_all_rank_num_tokens,
                accepted_tokens, pre_draft_seq_lens,
            )  # fmt: skip
        draft_len = spec_metadata.runtime_draft_len
        uses_index_share = isinstance(attn_metadata, MTPIndexShareMetadata)
        tokens = []
        with contextlib.ExitStack() as scope:
            if uses_index_share:
                attn_metadata.set_in_mtp_draft_loop(True)
                attn_metadata.set_mtp_num_accepted(num_accepted_tokens)
                scope.callback(_reset_mtp_index_share, attn_metadata)
            md = scope.enter_context(
                self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager)
            )
            inputs["attn_metadata"] = md
            if is_dsa_cache_manager(draft_kv_cache_manager):
                # The refresh after the draft-cache swap (the built-in loop's comment).
                self._glm_draft_step(loop, md, MODE_REFRESH, None, num_accepted_tokens)
            for i in range(draft_len):
                if uses_index_share:
                    md.set_skip_topk(i > 0)
                hidden, _ = self._run_draft_forward(draft_model, inputs, spec_metadata, i)
                # One row per request: step 0's last accepted rows, then every row.
                rows = hidden[loop.gather_ids] if i == 0 else hidden[:batch_size]
                logits = draft_model.mtp_layers[0].shared_head(rows, draft_model.lm_head, md, True)
                token = self.sample_draft_tokens(logits, spec_metadata, batch_size, draft_step=i)
                tokens.append(token)
                last = i + 1 == draft_len
                position_ids = self._glm_draft_step(
                    loop, md,
                    MODE_STEP0 if i == 0 else MODE_ADVANCE,
                    None if last else inputs["position_ids"], num_accepted_tokens,
                )  # fmt: skip
                if not last:
                    inputs = {
                        "input_ids": token,
                        "position_ids": position_ids,
                        "hidden_states": rows,
                        "attn_metadata": md,
                        "spec_metadata": spec_metadata,
                    }
        stacked = loop.finalize(tokens)
        return (stacked if stacked is not None else torch.stack(tokens, dim=1)), None

    def _glm_draft_step(self, loop, md, mode, position_ids, num_accepted_tokens):
        """One metadata update of the draft loop, fused or built-in; the next positions."""
        gather_ids = loop.gather_ids if mode == MODE_STEP0 else None
        fused, next_positions = loop.step(md, mode, position_ids, gather_ids)
        if fused:
            return next_positions
        return native_step(
            md, mode, loop.batch_size, loop.draft_len, num_accepted_tokens, position_ids,
            gather_ids,
        )  # fmt: skip

    def _prepare_next_new_tokens(
        self, accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size,
        num_accepted_tokens,
    ):  # fmt: skip
        loop = self._glm_draft_loop
        if loop is not None and loop.next_new_tokens is not None:
            return loop.next_new_tokens
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size,
            num_accepted_tokens,
        )  # fmt: skip

    def _prepare_next_new_tokens_builtin(self, *args):
        """The built-in `_prepare_next_new_tokens` (the draft-loop oracle's reference)."""
        return super()._prepare_next_new_tokens(*args)
