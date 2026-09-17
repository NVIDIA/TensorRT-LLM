# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Router Replay (R3) capture — CUDA-graph-capable (device buffer + async D2H).

Captures, per request, the PRE-EPLB *logical* top-k expert ids that the MoE
router actually selected, so a trainer can replay exactly those routes and
eliminate the train/infer routing mismatch.

Method: Ma et al., "Stabilizing MoE Reinforcement Learning by Aligning Training
and Inference Routers" (arXiv:2510.11370) — Rollout Routing Replay (R3).

Design:

  1. RouteCapture — capture + per-request store + output.
     * ``capture(layer_id, topk_ids)`` is a one-line **device-buffer write**
       hooked in ``moe_scheduler.py`` right after ``routing_method.apply()``
       (pre-EPLB). It runs INSIDE the CUDA graph, so graphed decode steps are
       captured (a Python-side ``.cpu()`` hook would neither be capturable nor
       fire during graph replay). No sync on the forward path.
     * ``prepare(scheduled_batch)`` builds the row -> (req_id, abs_pos) layout
       from the scheduler's clean request/position layout each forward.
     * ``attach_routes(request)`` assembles [S-1, L, K] on finish.

  2. RouteCopier — one **non-blocking D2H per step** on a side stream into a
     pinned host ring, event-gated commit into the store. Because the in-graph
     write target address is fixed at capture time, there is a single device
     buffer; the next step's forward must not overwrite it until the prior D2H
     has read it -> a cross-stream ``wait_event`` (GPU-side, no host sync).

  3. SharedRouteCache — cross-request prefix reuse: on a prefix-cache hit the
     hit tokens are not recomputed (no MoE, no capture), so their routes are read
     back from a position-keyed cache; invalidated with the KV on
     ``reset_prefix_cache``.

Enable: ``LlmArgs.enable_return_routed_experts`` (engine) +
``SamplingParams.return_routed_experts`` (per request). Correct with CUDA graphs,
prefix caching, and the overlap scheduler all ON. CUTLASS / DeepGemm separated
routing only; fused backends fail closed (``assert_capturable``).
"""

from typing import Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

# Consumer contract (trainer-side replay): -1 => genuinely missing route
# (fail-closed). Final/padding positions are filled by the consumer's own
# pad-and-align step; this module never emits a dummy expert id for them.
_MISSING = -1

# FNV-1a cumulative rolling hash for prefix-cache keys (position-granular): the
# key for position p is a hash of tokens[:p+1], computed incrementally in O(len).
# tuple/int hashing is not PYTHONHASHSEED-salted, and this is fully deterministic.
_FNV_OFF = 1469598103934665603
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1

# Key under which the owning model engine registers its capturer in the
# per-forward model extra attrs (the same channel that carries ``moe_layers``),
# so the MoE routing hook reaches the capturer of the engine whose forward is
# running -- no process-wide state, so several engines/models can coexist.
ROUTE_CAPTURE_ATTR = "route_capture"


def get_active_route_capture() -> Optional["RouteCapture"]:
    """The capturer of the model engine whose forward is currently running.

    Reads the per-forward model extra attrs set by ``PyTorchModelEngine.forward``;
    ``None`` outside a forward or when Router Replay is disabled for that engine.
    """
    attrs = get_model_extra_attrs()
    if not attrs:
        return None
    return attrs.get(ROUTE_CAPTURE_ATTR)


class RouteCopier:
    """One non-blocking D2H per step (side stream) + event-gated commit.

    Single device buffer (its address is fixed at graph-capture time), pinned
    host ring of ``ring`` slots. ``stage`` copies the step's valid rows to a
    host slot without host sync; ``drain`` commits slots whose copy has landed;
    ``wait_before_overwrite`` makes the next forward wait on the last D2H so it
    cannot clobber the single device buffer mid-copy.
    """

    def __init__(self, device: torch.device, shape: Tuple[int, int, int], ring: int = 3) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._ring = [
            torch.empty(shape, dtype=torch.int32, pin_memory=prefer_pinned()) for _ in range(ring)
        ]
        # Per-slot events, re-recorded each time the slot is used (no per-step
        # Event allocation on the hot path).
        self._fwd_done: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(ring)]
        self._d2h: List[torch.cuda.Event] = [torch.cuda.Event() for _ in range(ring)]
        # (slot, layout, n): staged copies whose rows are not yet committed.
        self._pending: List[Tuple[int, list, int]] = []
        self._slot = 0
        self._last_d2h: Optional[torch.cuda.Event] = None

    def _slot_pending(self, slot: int) -> bool:
        return any(p[0] == slot for p in self._pending)

    def stage(self, buf: torch.Tensor, n: int, layout: list, commit_cb) -> None:
        """Copy this step's ``n`` valid rows of ``buf`` into the next host slot.

        A slot is never overwritten while a staged copy of it is still pending:
        landed copies are committed first (event-gated, no host sync), and the
        rare still-in-flight case is synchronized and committed before reuse.
        Otherwise a later ``drain`` would commit the new rows under the old
        layout.
        """
        if n <= 0:
            return
        self.drain(commit_cb)
        slot = self._slot
        if self._slot_pending(slot):  # ring wrapped onto an unfinished copy -- rare
            self._d2h[slot].synchronize()
            self.drain(commit_cb)
        fwd_done = self._fwd_done[slot]
        fwd_done.record(torch.cuda.current_stream(self._device))  # forward finished writing buf
        self._stream.wait_event(fwd_done)  # side stream waits for the forward
        d2h = self._d2h[slot]
        with torch.cuda.stream(self._stream):
            self._ring[slot][:n].copy_(buf[:n], non_blocking=True)
            d2h.record(self._stream)
        self._last_d2h = d2h
        self._pending.append((slot, list(layout), n))
        self._slot = (slot + 1) % len(self._ring)

    def wait_before_overwrite(self) -> None:
        # The single device buffer is about to be overwritten by the next
        # forward -- serialize AFTER the last D2H read it (GPU-side, no host sync).
        if self._last_d2h is not None:
            torch.cuda.current_stream(self._device).wait_event(self._last_d2h)

    def drain(self, commit_cb, force: bool = False) -> None:
        still = []
        for slot, layout, n in self._pending:
            ev = self._d2h[slot]
            if force or ev.query():
                if force and not ev.query():
                    ev.synchronize()
                commit_cb(self._ring[slot][:n], layout)
            else:
                still.append((slot, layout, n))
        self._pending = still


class RouteCapture:
    """Per-request store of pre-EPLB logical top-k, keyed by absolute position.

    Write-once per (request, position): a re-prefill / recompute must never
    overwrite an already-committed position -- KV is a recomputable cache, the
    store is immutable history. Each committed position holds the full [L, K]
    (all MoE layers for that token, filled in one forward).

    One instance per ``PyTorchModelEngine`` (``model_engine.route_capture``),
    built by ``RouteCapture.create``; the engine exposes it to the MoE routing
    hook through its model extra attrs (see ``get_active_route_capture``).
    """

    @classmethod
    def create(
        cls,
        *,
        rank: int,
        model_engine,
        enabled: bool,
        pp_size: int,
        is_spec_decode: bool,
        is_draft_model: bool,
    ) -> Optional["RouteCapture"]:
        """Build the engine's capturer, or ``None`` when Router Replay is off.

        Draft engines never capture: their MoE layers are not target routing.
        Fails closed on paths this capture cannot attribute correctly: pipeline
        parallelism (routes live on the last PP stage only) and speculative
        decoding / MTP (accepted-token remap not handled). All flags are passed
        explicitly so this does not depend on engine attribute initialization
        order.
        """
        if not enabled or is_draft_model:
            return None
        if pp_size > 1 or is_spec_decode:
            raise RuntimeError(
                "[R3] enable_return_routed_experts is not supported with "
                f"pipeline parallelism (pp_size={pp_size}) or speculative decoding "
                f"(is_spec_decode={is_spec_decode}); disable one of them or the feature."
            )
        return cls(rank=rank, model_engine=model_engine)

    # ------------------------------------------------------------------ #
    #  Per-forward hooks -- the MoE forward and py_executor call these
    # ------------------------------------------------------------------ #
    def set_iter(self, iter_id: int) -> None:
        self._iter_id = iter_id

    def capture(
        self, layer_id: int, token_selected_experts: torch.Tensor, row_offset: int = 0
    ) -> None:
        """Hook from moe_scheduler after ``routing_method.apply()`` (separated,
        pre-EPLB). ``token_selected_experts`` is [num_rows, top_k] int32 in
        scheduler token order == the current-forward layout order; ``row_offset``
        is the first forward row of this MoE chunk (a multi-chunk MoE forward
        calls the hook once per chunk with chunk-local rows). Runs inside the
        CUDA graph -- a pure device write, no sync, no ``.cpu()``. ``layer_id``
        (``moe.layer_idx``) and ``row_offset`` are Python ints, constant at
        capture time, so the slice write is baked into the captured graph.

        NOT gated on ``_layout``: the buffer write must be *recorded into the
        CUDA graph during capture* (graph capture runs in model_engine warmup,
        where py_executor's ``prepare`` has not set ``_layout``). Gating the
        write on ``_layout`` would skip it at capture time -> the replayed graph
        would never write the buffer -> graphed decode routes lost. The write is
        layout-independent (it just dumps router output in forward row order);
        ``_layout`` only gates the later per-row commit."""
        self._capture(int(layer_id), token_selected_experts, int(row_offset))

    def prepare(self, scheduled_batch, tokens_per_block: int = 0) -> None:
        if tokens_per_block and tokens_per_block > 0:
            self._tpb = int(tokens_per_block)
        if not getattr(self, "_tpb_warned", False):
            self._tpb_warned = True
            logger.debug(f"[R3][pfx] prepare tpb={self._tpb}")
        # _prepare first (records per-request prompt tokens/hashes), then
        # readback (uses them to fill hit prefixes from the cache).
        self._prepare(scheduled_batch)
        self._readback_prefix(scheduled_batch)

    def clear_shared(self) -> None:
        """Hook from py_executor.reset_prefix_cache: route validity == KV
        validity, so when the KV reuse state is reset the cached routes must be
        dropped too."""
        self._shared.clear()
        self._readback_done.clear()
        self._prefix_populated.clear()
        self._pop_cursor.clear()

    def finish_forward(self) -> None:
        self._finish_forward()

    def abort_forward(self) -> None:
        """Error-path counterpart of ``finish_forward`` (py_executor calls it when
        the forward raises): nothing is staged, the armed layout is dropped so the
        next forward starts clean. Per-request bookkeeping is deliberately left
        as is -- this step's positions simply stay missing, and ``assemble`` fails
        closed on them instead of shifting later rows onto the wrong positions."""
        self._layout = None

    def attach_routes(self, request) -> None:
        """Called from py_executor._handle_responses when a request finishes:
        assemble its routes and append them so they surface on
        ``CompletionOutput.routed_experts`` (backed by
        ``additional_generation_outputs["routed_experts"]``).

        Force-drain the copier first so the finishing request's last decode
        step (whose D2H was just issued this step) is committed before assemble.

        attach_routes can fire MULTIPLE times for the same finished request
        (_handle_responses revisits it). Attach + free EXACTLY ONCE, and only
        once assemble succeeds: on an incomplete store (assemble None) do NOT
        free -- retry on a later call. Freeing eagerly corrupts the store.
        After ``free`` the store is empty, so a repeated call is a no-op
        (``assemble`` returns None); ``_attached`` only guards the window
        between attaching and freeing.

        Errors are NOT swallowed here: an internal gap in the store (assemble
        raises ValueError) or a failed copy drain means the routes would be
        wrong, so the exception propagates to the executor instead of the
        request silently completing without ``routed_experts``."""
        if request.py_request_id in self._attached:
            return
        if self._copier is not None:
            self._copier.drain(self.commit, force=True)
        pyr = getattr(request, "py_result", None)
        if pyr is None:
            return
        rid = request.py_request_id
        routes = self.assemble(rid)
        if routes is None:
            return  # incomplete store: keep it and retry on a later call
        if pyr._additional_generation_outputs is None:
            pyr._additional_generation_outputs = {}
        pyr._additional_generation_outputs.setdefault("routed_experts", [])
        pyr.append_additional_generation_outputs("routed_experts", routes)
        self._attached.add(rid)
        self._populate_prefix(request, rid)  # store this req's blocks for reuse
        self.free(rid)  # bound memory once safely attached

    # ------------------------------------------------------------------ #
    def __init__(self, *, rank: int, model_engine=None) -> None:
        self.rank = rank
        self._me = model_engine
        self._iter_id: Optional[int] = None
        # store[req_id][abs_pos] -> int16 [L, K]  (write-once per (req, pos))
        self._store: Dict[int, Dict[int, torch.Tensor]] = {}
        self._gen_count: Dict[int, int] = {}  # decode-position fallback counter
        self._layout: Optional[List[Tuple[Optional[int], int]]] = None
        self._attached: set = set()

        # PR2 prefix-cache: SharedRouteCache — routes keyed by a self-computed
        # hash of the prefix token ids at block boundaries, so a prefix-cache hit
        # (whose tokens are NOT recomputed -> no capture) can copy the routes back
        # into its store. Lives on this capturer (which persists across
        # requests); cleared on reset_prefix_cache. key = hash(tuple(tokens[:end]))
        # -> int16 [tokens_per_block, L, K].
        self._shared: Dict[int, torch.Tensor] = {}
        self._tpb: int = 0  # tokens_per_block (from V2 kv mgr)
        self._readback_done: set = set()  # rids fully read back (once each)
        self._pfx_hits: int = 0  # diagnostics: blocks filled from cache
        # incremental populate: a request's prompt-region blocks are pushed to the
        # SharedRouteCache as soon as they land in the store (finish_forward,
        # after drain) — NOT at request finish, which is too late for concurrent
        # siblings that hit the same prompt before the owner finishes.
        self._req_toks: Dict[int, list] = {}  # rid -> prompt token list
        self._req_plen: Dict[int, int] = {}  # rid -> prompt_len
        self._req_hashes: Dict[int, list] = {}  # rid -> cumulative prefix hashes
        self._pop_cursor: Dict[int, int] = {}  # rid -> next prompt pos to store
        self._prefix_populated: set = set()  # rids whose prompt positions are all stored
        self._had_context: bool = False  # this forward had prefill -> force-drain

        # device buffer + copier: allocated lazily on the first eager capture
        # (warmup runs an eager forward before graph capture — model is loaded
        # and we are NOT stream-capturing there, so allocation is legal).
        self._buf: Optional[torch.Tensor] = None
        self._copier: Optional[RouteCopier] = None
        self._layer_pos: Dict[int, int] = {}  # global layer_idx -> [0, L)
        self._L: int = 0
        self._K: int = 0

    # ---- lazy device-buffer allocation (see class docstring) ----
    def _ensure_buffer(self, tse: torch.Tensor) -> None:
        if self._buf is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            # Cannot cudaMalloc during graph capture. Reaching here means the very
            # first MoE capture happened INSIDE a graph capture with no preceding
            # eager warmup forward: the write for this graph key would not be
            # recorded and its routes silently lost. TRT-LLM always runs an eager
            # warmup before graph capture, so fail loudly rather than degrade.
            raise RuntimeError(
                "[R3] first MoE capture happened during CUDA graph capture, before "
                "the route buffer was allocated by an eager warmup forward; routes "
                "for this graph would be missing."
            )
        # Read the MoE-layer registry via the CANONICAL context-local accessor
        # ``get_model_extra_attrs()`` — the exact source the MoE forward itself
        # uses (interface.py: extract_extra_attrs). It is set during forward, so
        # it is valid here (capture runs inside the MoE forward). This is far more
        # robust than reaching through ``me.model.model_config.extra_attrs``,
        # which is a different object and was empty at runtime (routes never
        # captured -> routed_experts missing).
        moe = None
        try:
            from tensorrt_llm._torch.utils import get_model_extra_attrs

            attrs = get_model_extra_attrs()
            if attrs is not None:
                moe = attrs.get("moe_layers")
        except Exception:
            moe = None
        if not moe:
            return  # not in a forward context yet / registry empty — retry next capture
        me = self._me
        max_tok = int(getattr(me, "max_num_tokens", 0)) if me is not None else 0
        if max_tok <= 0:
            max_tok = 8192  # safe fallback (recipe max_num_tokens)
        keys = sorted(int(k) for k in moe.keys())
        self._layer_pos = {lid: i for i, lid in enumerate(keys)}
        self._L = len(keys)
        self._K = int(tse.shape[1])
        # Allocate the persistent buffers as NORMAL tensors, not inference
        # tensors. _ensure_buffer runs inside the generation forward (under
        # torch.inference_mode), so a plain torch.empty here would yield inference
        # tensors; the later host copy_ in stage() runs OUTSIDE inference mode and
        # an inplace update to an inference tensor is disallowed. inference_mode
        # (False) makes them normal so the D2H copy is legal.
        with torch.inference_mode(False):
            self._buf = torch.empty(max_tok, self._L, self._K, dtype=torch.int32, device=tse.device)
            self._copier = RouteCopier(tse.device, (max_tok, self._L, self._K))
        logger.debug(
            f"[R3] route buffer allocated: L={self._L} K={self._K} max_num_tokens={max_tok}"
        )

    # ---- driven by py_executor, once per forward (via the hooks above) ----
    def _prepare(self, scheduled_batch) -> None:
        # Next forward is about to overwrite the single device buffer — make it
        # wait on the last D2H (GPU-side, no host sync).
        if self._copier is not None:
            self._copier.wait_before_overwrite()
        layout: List[Tuple[Optional[int], int]] = []
        self._had_context = False
        for req in scheduled_batch.context_requests:
            # A context request contributes ONLY this forward's chunk of prompt
            # tokens, mirroring model_engine._prepare_inputs:
            #   begin = req.context_current_position
            #   toks  = req.get_tokens(0)[begin : begin + req.context_chunk_size]
            # Using py_prompt_len (the full prompt) misattributes every row under
            # chunked prefill (the R3 KL blowup). With chunking off this reduces
            # to the full prompt (begin==0, chunk==prompt_len).
            toks = self._safe_tokens(req)
            try:
                begin = int(req.context_current_position)
                end = begin + int(req.context_chunk_size)
            except Exception:
                begin, end = 0, req.py_prompt_len
            n = len(toks[begin:end]) if toks is not None else max(end - begin, 0)
            if req.is_dummy:
                layout.extend((None, -1) for _ in range(n))
            else:
                rid = req.py_request_id
                layout.extend((rid, begin + off) for off in range(n))
                self._gen_count.setdefault(rid, 0)
                self._had_context = True
                # Record prompt tokens + prompt_len once, for incremental populate
                # (finish_forward can then push prompt blocks without the request).
                if rid not in self._req_toks and toks is not None:
                    try:
                        plen = int(req.py_prompt_len)
                    except Exception:
                        plen = len(toks)
                    self._req_toks[rid] = list(toks[:plen])
                    self._req_plen[rid] = plen
                    self._hashes_for(rid, self._req_toks[rid], plen)  # cache prefix hashes
        for req in scheduled_batch.generation_requests:
            if req.is_dummy:
                layout.append((None, -1))
            else:
                rid = req.py_request_id
                # Decode position from OUR OWN per-request counter, NOT the host
                # request state. Under the OVERLAP scheduler the host state lags one
                # decode step at this hook (its ``_update_requests`` runs AFTER the
                # forward, py_executor overlap loop), so ``max_beam_num_tokens - 1``
                # is off by one and EVERY decode route mis-attributes -> KL blows up
                # (0.0007 -> ~1.0). ``gen_count`` increments exactly once per
                # processed decode token for this request (dummies excluded), so
                # ``py_prompt_len + gen_count`` is the true position and is
                # overlap-safe (and identical to max_beam_num_tokens-1 without
                # overlap, where 0.0007 was validated).
                abs_pos = int(getattr(req, "py_prompt_len", 0)) + self._gen_count.get(rid, 0)
                layout.append((rid, abs_pos))
                self._gen_count[rid] = self._gen_count.get(rid, 0) + 1
        self._layout = layout

    @staticmethod
    def _safe_tokens(req) -> Optional[list]:
        try:
            return list(req.get_tokens(0))
        except Exception:
            return None

    def _finish_forward(self) -> None:
        # After the forward: stage one D2H of this step's valid rows, then commit
        # whatever prior D2Hs have already landed. Never syncs the forward path.
        if self._copier is not None and self._buf is not None and self._layout is not None:
            try:
                self._copier.stage(self._buf, len(self._layout), self._layout, self.commit)
                # On a prefill step, force-drain so THIS step's prompt routes land
                # in the store now (prefill is not the graphed decode hot loop, so
                # the sync is cheap) -- then incremental populate can push them to
                # the SharedRouteCache this same step, ahead of a sibling's hit.
                self._copier.drain(self.commit, force=self._had_context)
                self._incremental_populate()
            finally:
                # Disarm between forwards (excludes stray/warmup calls) even if
                # staging raised; a failure here must surface, not be logged away.
                self._layout = None
        else:
            self._layout = None

    def _capture(self, layer_id: int, tse: torch.Tensor, row_offset: int = 0) -> None:
        # Device-buffer write, inside the CUDA graph. Allocate lazily on the
        # first eager capture (warmup), then reuse the fixed-address buffer.
        if self._buf is None:
            self._ensure_buffer(tse)
            if self._buf is None:
                return
        pos = self._layer_pos.get(layer_id)
        if pos is None:
            return
        # tse is [n, K] int32 (moe_scheduler cast it) holding forward rows
        # [row_offset, row_offset + n) of this MoE call; buf is [max_tok, L, K].
        begin = max(int(row_offset), 0)
        end = min(begin + tse.shape[0], self._buf.shape[0])
        if end <= begin:
            return
        self._buf[begin:end, pos, :] = tse[: end - begin]

    # ---- commit staged host rows into the per-request store (write-once) ----
    def commit(self, host_rows: torch.Tensor, layout: list) -> None:
        """host_rows: pinned int32 [n, L, K] for this step's valid rows."""
        rows = host_rows.to(torch.int16)
        for r in range(min(rows.shape[0], len(layout))):
            rid, pos = layout[r]
            if rid is None:  # dummy / padding row
                continue
            per_req = self._store.setdefault(rid, {})
            if pos not in per_req:  # write-once per (req, pos)
                per_req[pos] = rows[r].clone()

    # ---- assemble on completion (production output path calls this) ----
    def assemble(self, req_id: int) -> Optional[torch.Tensor]:
        """This request's routes as [seq_len - 1, L, K] int16: every captured
        position 0..max(pos_map) inclusive, since each stored position is a
        complete routing decision (there is no forward for the very last
        generated token, so its own position is never captured -- that is the
        only "missing" row, and it is never present in ``pos_map`` to begin
        with). None if nothing meaningful captured. Fail-closed on an internal
        gap (a stored position below the max is missing)."""
        pos_map = self._store.get(req_id)
        if not pos_map:
            return None
        last = max(pos_map)
        if last <= 0:
            return None
        keep = last + 1  # inclusive: positions [0, last] -> `keep` rows
        sample = next(iter(pos_map.values()))
        L, K = int(sample.shape[0]), int(sample.shape[1])
        out = torch.full((keep, L, K), _MISSING, dtype=torch.int16)
        for pos in range(keep):
            row = pos_map.get(pos)
            if row is None:
                raise ValueError(
                    f"[R3] request {req_id} position {pos} missing (internal gap) -> fail-closed"
                )
            out[pos] = row
        return out

    def free(self, req_id: int) -> None:
        self._store.pop(req_id, None)
        self._gen_count.pop(req_id, None)
        self._req_toks.pop(req_id, None)
        self._req_plen.pop(req_id, None)
        self._req_hashes.pop(req_id, None)
        self._pop_cursor.pop(req_id, None)
        self._prefix_populated.discard(req_id)
        self._readback_done.discard(req_id)
        # Release the attach marker too: request ids can be reused by a later
        # request, and once the store is gone a repeated attach_routes call for
        # this request finds nothing to attach anyway.
        self._attached.discard(req_id)

    # ---- PR2 prefix-cache: SharedRouteCache read-back / populate (position-granular) ----
    def _hashes_for(self, rid: int, toks: Optional[list], plen: int) -> Optional[list]:
        """Cumulative prefix hashes for rid's prompt, cached. out[p] = FNV-1a of
        tokens[:p+1] — two requests sharing a prefix get the same list over the
        shared region, so any hit length reads back exactly (no block alignment)."""
        h = self._req_hashes.get(rid)
        if h is not None:
            return h
        if toks is None:
            return None
        acc = _FNV_OFF
        out: List[int] = []
        n = min(plen, len(toks)) if plen else len(toks)
        for i in range(n):
            acc = ((acc ^ ((int(toks[i]) + 1) & _MASK64)) * _FNV_PRIME) & _MASK64
            out.append(acc)
        self._req_hashes[rid] = out
        return out

    def _readback_prefix(self, scheduled_batch) -> None:
        """A prefix-cache hit does NOT recompute the first
        ``prepopulated_prompt_len`` tokens (no MoE -> no capture). Copy their
        routes back from the SharedRouteCache, POSITION by position (any hit
        length, aligned or not), so assemble has no gap. Retries each step until
        fully filled (the owner may populate slightly later)."""
        try:
            ctx = scheduled_batch.context_requests
        except Exception:
            return
        for req in ctx:
            try:
                if getattr(req, "is_dummy", False):
                    continue
                rid = req.py_request_id
                if rid in self._readback_done:
                    continue
                # prepopulated_prompt_len == reused (cached) token count.
                C = int(getattr(req, "prepopulated_prompt_len", 0) or 0)
                if C > 0 and not getattr(self, "_hit_warned", False):
                    self._hit_warned = True
                    logger.debug(
                        f"[R3][pfx] prefix HIT seen: prepopulated={C} "
                        f"cache_size={len(self._shared)}"
                    )
                if C <= 0:
                    self._readback_done.add(rid)
                    continue
                hashes = self._hashes_for(
                    rid,
                    self._req_toks.get(rid) or self._safe_tokens(req),
                    self._req_plen.get(rid) or int(getattr(req, "py_prompt_len", 0) or 0),
                )
                if not hashes:
                    continue
                per_req = self._store.setdefault(rid, {})
                missing = 0
                for p in range(min(C, len(hashes))):
                    if p in per_req:
                        continue
                    row = self._shared.get(hashes[p])
                    if row is None:
                        missing += 1  # not populated yet — retry
                        continue
                    per_req[p] = row  # reference, no copy
                    self._pfx_hits += 1
                    if not getattr(self, "_pfx_warned", False):
                        self._pfx_warned = True
                        logger.debug(
                            "[R3] prefix-cache readback active: filled position "
                            "from SharedRouteCache"
                        )
                if missing == 0:
                    self._readback_done.add(rid)
            except Exception:
                continue

    def _store_positions(self, rid: int, hashes: list) -> None:
        """Store rid's captured PROMPT-region positions into the SharedRouteCache
        (write-once, keyed by cumulative prefix hash). Advances a per-rid cursor
        over newly-landed contiguous positions, so each is visited once; stores a
        reference to the store's [L,K] row (no copy). Only the prompt region is
        stored — reuse only ever reads back [0, prepopulated_prompt_len)."""
        pos_map = self._store.get(rid)
        if not pos_map:
            return
        cur = self._pop_cursor.get(rid, 0)
        n = len(hashes)
        while cur < n:
            row = pos_map.get(cur)
            if row is None:
                break  # not captured yet — resume next step
            key = hashes[cur]
            if key not in self._shared:  # write-once
                self._shared[key] = row
                if not getattr(self, "_pop_warned", False):
                    self._pop_warned = True
                    logger.debug(
                        f"[R3][pfx] populate: stored first position, cache_size={len(self._shared)}"
                    )
            cur += 1
        self._pop_cursor[rid] = cur
        if cur >= n:
            self._prefix_populated.add(rid)

    def _incremental_populate(self) -> None:
        """Each step (after drain), push any request's now-landed prompt-region
        positions to the SharedRouteCache — as early as the routes land, so a
        concurrent sibling that hits the same prompt finds them."""
        if not self._req_toks:
            return
        for rid in list(self._store.keys()):
            if rid in self._prefix_populated:
                continue
            toks = self._req_toks.get(rid)
            plen = self._req_plen.get(rid)
            if toks is None or not plen:
                continue
            try:
                hashes = self._hashes_for(rid, toks, plen)
                if hashes:
                    self._store_positions(rid, hashes)
            except Exception:
                continue

    def _populate_prefix(self, request, rid: int) -> None:
        """Backstop at request finish: ensure this request's prompt positions are
        stored (in case incremental populate missed the tail)."""
        toks = self._req_toks.get(rid) or self._safe_tokens(request)
        plen = self._req_plen.get(rid) or int(getattr(request, "py_prompt_len", 0) or 0)
        if toks is None or not plen:
            return
        try:
            hashes = self._hashes_for(rid, toks, plen)
            if hashes:
                self._store_positions(rid, hashes)
        except Exception:
            return


def assert_capturable(moe_backend) -> None:
    """Fail closed unless the backend uses separated routing (top-k in Python).

    Fused backends (e.g. TRTLLMGen when not separated) compute top-k inside the
    kernel and never expose it at ``routing_method.apply()`` -> R3 cannot capture.
    The default CUTLASS backend returns True unconditionally.
    """
    if not moe_backend._supports_load_balancer():
        raise RuntimeError(
            f"[R3] router_replay requires separated routing, but MoE backend "
            f"{type(moe_backend).__name__} uses fused routing (top-k inside the "
            f"kernel). Use a separated-routing backend (CUTLASS is the default) or "
            f"disable router_replay."
        )
