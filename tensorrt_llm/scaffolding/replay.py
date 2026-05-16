import asyncio
import hashlib
import random
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .execution_trace import ExecutionTrace, TraceEvent
from .task import GenerationTask, TaskStatus
from .worker import Worker

# ---------------------------------------------------------------------------
# Deterministic per-branch RNG
# ---------------------------------------------------------------------------
#
# Trace replay generates synthetic token ids for every user / tool / assistant
# segment via :func:`_generate_random_token_ids`.  Two callers care about
# determinism:
#
#   * cross-pass radix-tree hits — when the same session is re-played a
#     second time, the prefix tokens it sends must hash bit-for-bit to the
#     blocks the first pass committed, otherwise no block past the system
#     prefix can hit (every per-call random redraw branches the radix tree).
#   * proactive KV-cache drops — the prefix the engine asks the server to
#     truncate must equal the longest input prompt that engine actually
#     sent for the corresponding conv id, otherwise the truncate walk
#     diverges from the cached chain at the first mismatched block.
#
# Both fall out of giving every :class:`QueueExecutor` an isolated
# :class:`random.Random` instance whose seed is a function of the session
# (or namespace) and the executor's branch path.  Sibling branches under
# the same ``parallel_start`` therefore each get an independent stream that
# does not depend on asyncio's interleaving order, and re-playing the same
# session two times reproduces every draw exactly.
#
# The factory pattern ``Callable[[Tuple[int, ...]], random.Random]`` keeps
# the seeding policy outside this module: callers (e.g. the trace replay
# client) decide how a session index, pass namespace, etc. compose into a
# seed.  The default factory returns a freshly-seeded ``Random`` per
# branch_path, which preserves the historical "non-deterministic across
# sessions" behaviour bit-for-bit when callers do not opt in.

RngFactory = Callable[[Tuple[int, ...]], random.Random]


def _default_rng_factory(_branch_path: Tuple[int, ...]) -> random.Random:
    """Return a freshly-seeded ``Random`` per branch — non-deterministic.

    Used when the caller does not provide an explicit factory; matches the
    pre-determinism behaviour where every replay drew from the module-level
    ``random`` state.
    """
    return random.Random()


def make_seeded_rng_factory(*seed_parts: Any) -> RngFactory:
    r"""Build a :data:`RngFactory` seeded by ``(seed_parts, branch_path)``.

    Per-branch seed is a SHA-256 hash of ``(seed_parts, branch_path)``.

    The seed parts are stringified and joined with ``\x00`` so two callers
    with different conventions (e.g. ``("replay", trace_idx, sess_idx)``
    vs. ``("replay", "v2", trace_idx, sess_idx)``) cannot collide unless
    they explicitly compose to the same byte string.  branch_path is
    appended last so sibling branches under the same parallel get
    independent streams.

    SHA-256 is used (rather than ``hash()`` or Python's tuple seed) so the
    seed is stable across Python versions, processes, and PYTHONHASHSEED
    settings — important because pass-1 and pass-2 of the same experiment
    typically run in different processes.
    """

    def factory(branch_path: Tuple[int, ...]) -> random.Random:
        key = "\x00".join(repr(p) for p in seed_parts)
        key += "\x00" + repr(tuple(branch_path))
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], "big")
        return random.Random(seed_int)

    return factory


class ReplayGenerationStats:
    """Per-assistant-generation token metrics collected during trace replay.

    For each assistant :class:`TraceEvent`, records the trace-file
    ``completion_tokens`` budget, the lengths actually produced by the
    worker during replay (full decode sequence and content after stripping
    leading reasoning tokens), and per-request server-side timings
    (``ttft_s`` / ``latency_s`` / ``usage_completion_tokens``) that the
    streaming OpenAI worker attaches to the :class:`GenerationTask`. The
    timings are what the trace-replay Pareto pipeline feeds to TPOT /
    ``intvty`` percentiles, matching SemiAnalysis's InferenceMAX
    ``benchmark_serving.py`` definition of per-request tokens/s/user.

    ``session_index`` / ``trace_index`` identify which concurrent session
    owns this stats object and, when traces are mixed round-robin, which
    trace in the mix the session is replaying. They are stamped on every
    recorded entry so per-LLM-call analyses can break the scatter into
    per-trace clusters without guessing from durations.
    """

    __slots__ = ("_entries", "session_index", "trace_index")

    def __init__(
        self,
        *,
        session_index: Optional[int] = None,
        trace_index: Optional[int] = None,
    ) -> None:
        self._entries: List[Dict[str, Optional[float]]] = []
        self.session_index = session_index
        self.trace_index = trace_index

    def record_assistant(
        self,
        *,
        trace_completion_tokens: int,
        replay_output_token_len: int,
        replay_content_token_len: int,
        reasoning_tokens: int,
        ttft_s: Optional[float] = None,
        latency_s: Optional[float] = None,
        usage_completion_tokens: Optional[int] = None,
        usage_prompt_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
        branch_path: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self._entries.append(
            {
                "trace_completion_tokens": trace_completion_tokens,
                "replay_output_token_len": replay_output_token_len,
                "replay_content_token_len": replay_content_token_len,
                "reasoning_tokens": reasoning_tokens,
                "ttft_s": ttft_s,
                "latency_s": latency_s,
                "usage_completion_tokens": usage_completion_tokens,
                "usage_prompt_tokens": usage_prompt_tokens,
                "session_index": self.session_index,
                "trace_index": self.trace_index,
                "request_id": request_id,
                "branch_path": list(branch_path) if branch_path is not None else [],
            }
        )

    @property
    def entries(self) -> List[Dict[str, Optional[float]]]:
        return list(self._entries)

    def sum_trace_completion_tokens(self) -> int:
        return sum(e["trace_completion_tokens"] for e in self._entries)

    def sum_replay_output_tokens(self) -> int:
        return sum(e["replay_output_token_len"] for e in self._entries)


class RetentionProbeStats:
    r"""Deferred retention-probe ledger for end-of-run KV-cache retention.

    Collect prefixes during the run, fire probes post-run to measure
    long-term KV-cache retention.

    During trace replay each :class:`QueueExecutor` calls
    :meth:`record_pending` at sentinel time (end-of-branch) with the
    conv id's last assistant-call input prefix.  No server request is
    issued at that point — the prefix and metadata are stashed in
    :attr:`_pending`.

    After **all** sessions complete, the caller iterates
    :attr:`pending_probes` and fires one ``max_tokens=1, ignore_eos=True``
    :class:`GenerationTask` per entry through the worker, then calls
    :meth:`record` with the returned ``request_id``.  This gives every
    probe an identical measurement context: the full post-workload radix
    tree, with all sessions' generation requests committed and (for
    ``with_drop``) all sub-agent drops applied.

    The retention rate computed downstream is:

    .. math::

        \\text{retention\\_rate} =
        \\frac{\\text{num\\_reused\\_blocks}}{\\lceil \\text{prefix\\_len}
        / \\text{tokens\\_per\\_block} \\rceil}

    100% = the entire chain is still cached at end-of-run; 0% = fully
    evicted or proactively freed.
    """

    __slots__ = ("_records", "_pending")

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._pending: List[Dict[str, Any]] = []

    def record_pending(
        self,
        *,
        branch_path: Tuple[int, ...],
        conv_id: int,
        prefix: List[int],
    ) -> None:
        """Stash a probe definition to be fired after all sessions complete."""
        self._pending.append(
            {
                "branch_path": tuple(branch_path),
                "conv_id": conv_id,
                "prefix": list(prefix),
            }
        )

    @property
    def pending_probes(self) -> List[Dict[str, Any]]:
        return list(self._pending)

    def record(
        self,
        *,
        branch_path: Tuple[int, ...],
        conv_id: int,
        request_id: Optional[str],
        prefix_len: int,
    ) -> None:
        self._records.append(
            {
                "branch_path": list(branch_path),
                "conv_id": conv_id,
                "request_id": request_id,
                "prefix_len": prefix_len,
            }
        )

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)


class DropPathStats:
    """Per-drop sentinel-probe ledger for the verification harness (P0.6).

    When a :class:`ReplayEngine` is constructed with a non-``None``
    ``drop_path_stats`` argument, every ``drop_kv_cache`` event the engine
    handles emits two extra ``max_tokens=1, ignore_eos=True``
    :class:`GenerationTask` "probes" — one immediately before the truncate
    POSTs to the server, one immediately after — for every conv id this
    branch owns.  Each probe's ``request_id`` (server-assigned, captured
    by the worker from the OpenAI streaming chunk) is recorded here and
    later joined with trtllm-serve's ``/perf_metrics`` drain to recover
    ``num_reused_blocks`` (the per-request KV-cache hit count) and
    ``free_num_blocks`` (the post-call snapshot of the free-block pool).

    The two derived assertions live downstream in the verification driver:

      * **Sentinel probe** (P0.6.a):
        ``before.num_reused_blocks - after.num_reused_blocks ≥
        ⌈(prefix_len - num_tokens_to_keep) / tokens_per_block⌉ − epsilon``.
        Failure means the truncate did not actually free those blocks.
      * **free_num_blocks delta** (P0.6.c):
        ``after.free_num_blocks - before.free_num_blocks ≥ 0`` and within
        the same lower bound.

    The class is stateful but cheap (one dict per probe).  Probes are
    issued serially per branch, so :attr:`_records` is built without
    locking.
    """

    __slots__ = ("_records", "_next_drop_index")

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._next_drop_index = 0

    def next_drop_index(self) -> int:
        """Return a fresh monotonic index for the next drop event."""
        idx = self._next_drop_index
        self._next_drop_index += 1
        return idx

    def record(
        self,
        *,
        drop_event_index: int,
        branch_path: Tuple[int, ...],
        conv_id: int,
        phase: str,
        request_id: Optional[str],
        prefix_len: int,
        num_tokens_to_keep: int,
    ) -> None:
        """Record one before/after probe.  ``phase`` ∈ {"before", "after"}."""
        if phase not in ("before", "after"):
            raise ValueError(f"phase must be 'before' or 'after' (got {phase!r})")
        self._records.append(
            {
                "drop_event_index": drop_event_index,
                "branch_path": list(branch_path),
                "conv_id": conv_id,
                "phase": phase,
                "request_id": request_id,
                "prefix_len": prefix_len,
                "num_tokens_to_keep": num_tokens_to_keep,
            }
        )

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)


def _generate_random_token_ids(length: int, rng: random.Random) -> List[int]:
    """Return *length* random token ids in the range [100, 30000].

    Draws are pulled from *rng* — a per-:class:`QueueExecutor`
    :class:`random.Random` instance keyed off the executor's branch path
    (see :data:`RngFactory`) — so re-playing the same session a second
    time produces an identical token-id stream.
    """
    if length <= 0:
        return []
    return [rng.randint(100, 30000) for _ in range(length)]


class QueueExecutor:
    """Consumes events from a single queue.

    One ``QueueExecutor`` is created per queue by ``QueueManager``.  It runs
    as an ``asyncio.Task`` that dequeues events until a sentinel (``None``)
    is received, at which point it sets ``done_event`` and exits.

    Event processing is a placeholder for future work — dequeued events are
    currently discarded.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        trace_id: str,
        worker: Worker,
        system_token_cache: Dict[str, List[int]],
        rng: random.Random,
        branch_path: Tuple[int, ...],
        generation_stats: Optional[ReplayGenerationStats] = None,
        drop_path_stats: Optional[DropPathStats] = None,
        retention_probe_stats: Optional[RetentionProbeStats] = None,
    ):
        self.queue = queue
        self.trace_id = trace_id
        self.worker = worker
        # Shared across QueueExecutors. Keyed by ``event.system_prompt_id`` for
        # tagged templates so multiple conversations using the same template
        # share a token-id prefix (simulating a real prefix-cache hit). Falls
        # back to ``f"conv:{conv_id}"`` for legacy / untagged system prompts.
        self._system_token_cache = system_token_cache
        self._rng = rng
        self._branch_path = branch_path
        self._generation_stats = generation_stats
        self._drop_path_stats = drop_path_stats
        self._retention_probe_stats = retention_probe_stats
        self._conversation_token_ids: Dict[int, List[List[int]]] = defaultdict(list)
        # Per-conv: the input prompt of the most recent assistant call.
        # Captured BEFORE storing the assistant's output back into segments,
        # so it is exactly the longest prompt the engine actually sent to
        # the server for that conv id (= the longest chain the server
        # committed for that conv id).  Used as the prefix for the
        # end-of-branch retention probe.
        self._last_gen_input_tokens: Dict[int, List[int]] = {}
        # Per-conv: token-count of the system message most recently committed
        # for this conv. Tracked separately because ``_system_token_cache`` is
        # keyed by template id (``event.system_prompt_id`` or ``"conv:<id>"``),
        # so it cannot be looked up directly by conv_id at drop_kv_cache time.
        self._system_token_len: Dict[int, int] = {}
        self.done_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        # Always set done_event so :meth:`QueueManager.wait_all_done` cannot
        # deadlock if ``_handle_message`` / ``worker.run_task`` raises (otherwise
        # the sentinel path that sets the event is never reached).
        #
        # Every dequeued event — including the ``None`` sentinel — must call
        # ``task_done`` so that :meth:`asyncio.Queue.join` (used by
        # :meth:`QueueManager.drain_branch` at parallel_start) accurately
        # reflects "queue currently drained".
        try:
            while True:
                event = await self.queue.get()
                try:
                    if event is None:  # sentinel
                        # Fire end-of-branch retention probes BEFORE returning,
                        # so the parent ``wait_all_done`` covers them and the
                        # client's downstream /perf_metrics drain sees the
                        # probe records.  For a child branch this is at
                        # parallel_end (immediately after any drop_kv_cache);
                        # for the root branch it is at end-of-session.
                        await self._emit_retention_probes()
                        return
                    if event.event_type == "tool_call":
                        await self._handle_tool_call(event)
                    elif event.event_type == "message":
                        await self._handle_message(event)
                    elif event.event_type == "drop_kv_cache":
                        await self._handle_drop_kv_cache(event)
                    # other types: no-op
                finally:
                    self.queue.task_done()
        finally:
            self.done_event.set()

    async def _handle_tool_call(self, event: TraceEvent):
        duration = event.duration_ms or 0.0
        if duration > 0:
            await asyncio.sleep(duration / 1000)

    async def _emit_retention_probes(self):
        """Stash one probe definition per root-branch conv id for post-run firing.

        Only the root-branch executor (``branch_path == ()``) emits
        probe definitions — these cover conv 0 (brief) and conv 1
        (Supervisor planner), which are the headline measurement.
        Child-branch executors (Researcher conv ids) skip emission so
        the post-run probe phase does not balloon to N×10 requests; the
        Researcher panel of the recipe's plot is confirmation-only and
        not worth the cache pollution that extra probe writes cause.

        No-op when ``retention_probe_stats`` was not attached to the
        engine.
        """
        if self._retention_probe_stats is None:
            return
        if self._branch_path:
            return
        if not self._last_gen_input_tokens:
            return
        for conv_id, prefix in self._last_gen_input_tokens.items():
            if not prefix:
                continue
            self._retention_probe_stats.record_pending(
                branch_path=self._branch_path,
                conv_id=conv_id,
                prefix=prefix,
            )

    async def _handle_drop_kv_cache(self, event: TraceEvent):
        """Free radix-tree blocks for every conv id this executor owns.

        Preserves system-prompt blocks.

        At the moment a ``drop_kv_cache`` event arrives, the executor's
        :attr:`_conversation_token_ids` holds, for each conv id seen on
        this branch, the exact concatenated token sequence that was sent
        as the prompt of that conv's most recent assistant call (plus
        the appended placeholder for that call's output).  That sequence
        is the longest path the engine ever pushed into the server's
        radix tree on this branch — so it is the right prefix to walk
        when issuing the truncate.

        ``num_tokens_to_keep`` is set to the system-prompt token count
        for conv ids that have one, and 0 for conv ids without.  This
        keeps the system-prompt blocks explicitly in the radix tree so
        future sessions that share the same system prefix enjoy
        guaranteed cache hits regardless of concurrent-session refcount.
        Conversation content past the system prompt is freed immediately.

        The chains the server committed from its **actual** output
        tokens (`asstK_actual` leaves siblinged off our placeholder
        chain at every assistant call) are not on the path we walk
        here, so they remain in the tree until LRU evicts them — see
        the "Matching invariant" section of the experiment plan for
        why this residual is expected.
        """
        if not self._conversation_token_ids:
            return

        prefixes: List[List[int]] = []
        keeps: List[int] = []
        conv_ids: List[int] = []
        for conv_id, segments in self._conversation_token_ids.items():
            prefix: List[int] = []
            for segment in segments:
                prefix.extend(segment)
            if not prefix:
                continue
            prefixes.append(prefix)
            system_len = self._system_token_len.get(conv_id, 0)
            keeps.append(system_len)
            conv_ids.append(conv_id)

        if not prefixes:
            return

        # If a drop-path verification ledger is attached, fire one
        # ``max_tokens=1, ignore_eos=True`` probe per (conv_id, phase) pair
        # immediately before and after the truncate.  Each probe's
        # request_id is recorded here; the per-request KV-cache hit
        # accounting is joined in later via /perf_metrics.  The probes
        # serialize the truncate against a real before/after measurement
        # so downstream verification can mechanically prove the truncate
        # actually freed the blocks the engine claimed it did (P0.6.a).
        #
        # A short barrier sleep BEFORE the before-probes lets the
        # executor's main loop finish committing the most-recent real
        # generation requests' KV blocks to the radix tree.  The OpenAI
        # streaming response returns the moment the last token is sent
        # (sequence-finished), but ``storeContextBlocks`` runs at the
        # NEXT executor iteration's boundary; without this barrier the
        # before-probe can race that commit and observe a partial radix
        # tree, which makes ``after.num_reused_blocks > before.num_reused_blocks``
        # (the commit catches up between the two probes) and breaks the
        # delta-positive sanity check.
        drop_index: Optional[int] = None
        if self._drop_path_stats is not None:
            await asyncio.sleep(0.5)

            drop_index = self._drop_path_stats.next_drop_index()
            for conv_id, prefix, keep in zip(conv_ids, prefixes, keeps):
                probe = GenerationTask(
                    input_tokens=prefix,
                    max_tokens=1,
                    ignore_eos=True,
                )
                status = await self.worker.run_task(probe)
                if status != TaskStatus.SUCCESS:
                    raise RuntimeError(f"drop-path before-probe failed: status={status}")
                self._drop_path_stats.record(
                    drop_event_index=drop_index,
                    branch_path=self._branch_path,
                    conv_id=conv_id,
                    phase="before",
                    request_id=probe.request_id,
                    prefix_len=len(prefix),
                    num_tokens_to_keep=keep,
                )

        await self.worker.send_kv_cache_truncate_tokens(prefixes, keeps)

        # Wait long enough for the executor's main loop to drain the
        # ``kv_cache_control_queue`` and apply the truncate.  The HTTP
        # POST returned the moment the request was enqueued — the
        # ``KVCacheManager.truncate_blocks`` call does not run until the
        # executor's per-iteration ``control_requests`` sweep picks it
        # up next.  Without this barrier, an after-probe issued
        # immediately after the POST can race the executor loop and
        # observe pre-truncate cache state, producing a false-negative
        # in the verification harness (and stale numbers in any other
        # caller that wants the truncate to be visible to its next
        # request).  Only triggered when probes will follow; the
        # headline measurement bursts are fire-and-forget and do not
        # need to wait.
        if self._drop_path_stats is not None:
            # The truncate POST returns the moment the request was
            # enqueued; the actual ``KVCacheManager.truncate_blocks``
            # call runs at the next executor iteration's
            # ``_sync_and_process_kv_cache_control_queue`` callsite
            # (see py_executor.py).  Empirically, the kvmon log shows
            # the post-truncate kv-block count change a few seconds
            # after the POST returns, so the barrier sleep below has
            # to be conservative.  Used only when verifying.
            await asyncio.sleep(3.0)

            assert drop_index is not None
            for conv_id, prefix, keep in zip(conv_ids, prefixes, keeps):
                probe = GenerationTask(
                    input_tokens=prefix,
                    max_tokens=1,
                    ignore_eos=True,
                )
                status = await self.worker.run_task(probe)
                if status != TaskStatus.SUCCESS:
                    raise RuntimeError(f"drop-path after-probe failed: status={status}")
                self._drop_path_stats.record(
                    drop_event_index=drop_index,
                    branch_path=self._branch_path,
                    conv_id=conv_id,
                    phase="after",
                    request_id=probe.request_id,
                    prefix_len=len(prefix),
                    num_tokens_to_keep=keep,
                )

    def _store_segment(self, conv_id: int, message_index, token_ids):
        """Store a token segment, overwriting if message_index is within bounds."""
        if message_index is not None and message_index < len(self._conversation_token_ids[conv_id]):
            self._conversation_token_ids[conv_id][message_index] = token_ids
        else:
            self._conversation_token_ids[conv_id].append(token_ids)

    async def _handle_message(self, event: TraceEvent):
        conv_id = event.conversation_id
        role = event.role
        message_index = event.message_index

        if role == "system":
            token_count = event.tokens or 0
            # Prefer the template-level identity so that different conversations
            # rendering the same system-prompt template share a token-id prefix.
            # Untagged events (legacy traces) fall back to the original
            # per-conversation key, preserving prior behavior.
            #
            # Generated token ids are drawn from this executor's per-branch RNG
            # (the same RNG used by user/tool/assistant segments), so re-playing
            # the same (session, branch_path) yields an identical token-id
            # stream. The cache is shared across executors of a single trace, so
            # whichever branch first sees a given ``cache_key`` writes the
            # canonical prefix and every later branch reads it back.
            cache_key = event.system_prompt_id or f"conv:{conv_id}"
            cached = self._system_token_cache.get(cache_key)
            if cached is None:
                cached = _generate_random_token_ids(token_count, self._rng)
                self._system_token_cache[cache_key] = cached
            elif len(cached) < token_count:
                # Same template, longer rendered system message: grow the
                # cached prefix in place. Subsequent shorter requests will
                # still slice a prefix that overlaps with this longer one.
                cached.extend(_generate_random_token_ids(token_count - len(cached), self._rng))
            # Always hand out a fresh slice. The cache list may be extended
            # later by a longer request; prior segments stored in
            # ``_conversation_token_ids`` must remain stable.
            token_ids = cached[:token_count]
            self._system_token_len[conv_id] = token_count
            self._store_segment(conv_id, message_index, token_ids)

        elif role in ("user", "tool"):
            token_ids = _generate_random_token_ids(event.tokens or 0, self._rng)
            self._store_segment(conv_id, message_index, token_ids)

        elif role == "assistant":
            # Build input from all accumulated segments
            input_tokens = []
            for segment in self._conversation_token_ids[conv_id]:
                input_tokens.extend(segment)

            # Capture the assistant call's input prompt as the seed for
            # this conv's end-of-branch retention probe.  Stored BEFORE
            # the gen task runs (so even if the gen call fails we still
            # have the most recent input to probe with), and overwritten
            # on every assistant call so the final value is the longest
            # prompt the engine actually sent for this conv.
            self._last_gen_input_tokens[conv_id] = list(input_tokens)

            completion_tokens = event.completion_tokens or 0
            reasoning_tokens = event.reasoning_tokens or 0
            if completion_tokens <= 0:
                raise ValueError(
                    "assistant message needs completion_tokens > 0 "
                    f"(got {event.completion_tokens!r}); cannot run generation"
                )

            gen_task = GenerationTask(
                input_tokens=input_tokens,
                max_tokens=completion_tokens,
                ignore_eos=True,
            )
            status = await self.worker.run_task(gen_task)
            if status != TaskStatus.SUCCESS:
                raise RuntimeError(f"GenerationTask failed with status {status}")

            # Strip leading reasoning tokens, keep only the content portion.
            # The ``/v1/completions`` endpoint does not stream ``token_ids``
            # with ``detokenize`` on, so fall back to synthetic ids of the
            # length the server actually produced (``usage.completion_tokens``
            # from the trailing usage chunk, which matches ``ignore_eos=True``
            # + ``max_tokens=completion_tokens`` in normal replay).
            output_len = gen_task.usage_completion_tokens
            if output_len is None:
                output_tokens = gen_task.output_tokens
                if output_tokens is not None:
                    output_len = len(output_tokens)
                else:
                    output_len = int(completion_tokens)
            output_tokens = gen_task.output_tokens
            if output_tokens is None:
                # ``/v1/completions`` does not stream token_ids when
                # detokenize is on (the trtllm-serve default), so the worker
                # cannot recover the server's actual generated ids.  Fall
                # back to a synthetic draw from the per-branch RNG; this
                # makes the placeholder sequence deterministic for the
                # (session, branch_path) pair, which is what makes pass-2
                # of the same session reproduce the prefix tokens pass-1
                # committed to the radix tree.
                output_tokens = _generate_random_token_ids(int(output_len), self._rng)
            content_tokens = output_tokens[reasoning_tokens:]
            if self._generation_stats is not None:
                self._generation_stats.record_assistant(
                    trace_completion_tokens=int(completion_tokens),
                    replay_output_token_len=len(output_tokens),
                    replay_content_token_len=len(content_tokens),
                    reasoning_tokens=int(reasoning_tokens),
                    ttft_s=gen_task.ttft_s,
                    latency_s=gen_task.latency_s,
                    usage_completion_tokens=gen_task.usage_completion_tokens,
                    usage_prompt_tokens=gen_task.usage_prompt_tokens,
                    request_id=gen_task.request_id,
                    branch_path=self._branch_path,
                )
            self._store_segment(conv_id, message_index, content_tokens)


class QueueManager:
    """Manages a pool of ``(asyncio.Queue, QueueExecutor)`` pairs.

    Tracks which ``trace_id`` each queue belongs to and maintains a mapping
    from ``branch_path`` (as a tuple) to ``queue_id`` for event routing.
    """

    def __init__(
        self,
        worker: Worker,
        system_token_cache: Dict[str, List[int]],
        rng_factory: RngFactory,
        generation_stats: Optional[ReplayGenerationStats] = None,
        drop_path_stats: Optional[DropPathStats] = None,
        retention_probe_stats: Optional[RetentionProbeStats] = None,
    ):
        self._worker = worker
        self._system_token_cache = system_token_cache
        self._rng_factory = rng_factory
        self._generation_stats = generation_stats
        self._drop_path_stats = drop_path_stats
        self._retention_probe_stats = retention_probe_stats
        self._queues: Dict[str, asyncio.Queue] = {}
        self._executors: Dict[str, QueueExecutor] = {}
        self._trace_ids: Dict[str, str] = {}
        self._branch_to_queue: Dict[Tuple[int, ...], str] = {}

    def allocate_queue(self, trace_id: str, branch_path: Tuple[int, ...]) -> str:
        """Create a queue + executor pair for *branch_path* and return its queue id.

        Registers *branch_path* against the resulting queue id.

        ``branch_path`` is taken as input here (rather than registered via
        a separate :meth:`register_branch` call) so the executor can be
        seeded with the per-branch RNG before its ``_run`` task starts —
        otherwise the first event the executor pulls from its queue could
        race the seeding and produce non-deterministic draws.
        """
        queue_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        executor = QueueExecutor(
            queue,
            trace_id,
            self._worker,
            self._system_token_cache,
            self._rng_factory(branch_path),
            tuple(branch_path),
            self._generation_stats,
            self._drop_path_stats,
            self._retention_probe_stats,
        )
        self._queues[queue_id] = queue
        self._executors[queue_id] = executor
        self._trace_ids[queue_id] = trace_id
        self._branch_to_queue[branch_path] = queue_id
        return queue_id

    def get_queue(self, branch_path: Tuple[int, ...]) -> asyncio.Queue:
        """Look up a queue by branch_path."""
        queue_id = self._branch_to_queue[branch_path]
        return self._queues[queue_id]

    def close_queue(self, queue_id: str):
        """Send sentinel ``None`` to the queue to signal completion."""
        self._queues[queue_id].put_nowait(None)

    async def drain_branch(self, branch_path: Tuple[int, ...]):
        """Block until the executor for *branch_path* has processed every event currently queued.

        Used as a synchronization barrier at ``parallel_start``: in a real
        agent run, the controller cannot dispatch child sub-tasks until the
        parent's pre-fork assistant generation (typically the one that
        produced the fork tool_call) has fully completed. Without this,
        replay would push child events into freshly-allocated child queues
        while the parent's last ``worker.run_task`` was still in flight,
        inflating server-side concurrency above the true agent topology.
        """
        queue_id = self._branch_to_queue[branch_path]
        await self._queues[queue_id].join()

    async def wait_all_done(self, queue_ids: List[str]):
        """Await ``done_event`` for each executor in *queue_ids*."""
        await asyncio.gather(*(self._executors[qid].done_event.wait() for qid in queue_ids))
        for qid in queue_ids:
            task = self._executors[qid]._task
            if not task.done():
                continue
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc

    def unregister_queue(self, queue_id: str):
        """Clean up all mappings for *queue_id*."""
        self._queues.pop(queue_id, None)
        self._executors.pop(queue_id, None)
        self._trace_ids.pop(queue_id, None)
        # Remove branch_path → queue_id entries
        to_remove = [bp for bp, qid in self._branch_to_queue.items() if qid == queue_id]
        for bp in to_remove:
            del self._branch_to_queue[bp]


class ReplayEngine:
    """Replays an ``ExecutionTrace`` by routing events to per-branch queues.

    The engine owns a ``QueueManager`` and iterates over trace events,
    creating child queues for ``parallel_start`` events and routing all
    other events to the queue registered for their ``branch_path``.

    ``system_token_cache`` memoizes the synthetic token ids generated for
    each conversation's ``role == "system"`` message. Passing a shared dict
    lets multiple :class:`ReplayEngine` instances (e.g. one per concurrent
    replay session of the same trace) all resolve the same ``conv_id`` to
    the same token-id list, so the server's prefix block reuse can hit on
    the system prefix across sessions; when left ``None`` the engine falls
    back to a fresh per-instance dict and the system cache is effectively
    session-local, matching the original behaviour.

    ``rng_factory`` controls the per-branch RNG used by
    :func:`_generate_random_token_ids` for every user / tool / assistant
    placeholder draw.  Passing a seeded factory built via
    :func:`make_seeded_rng_factory` makes the synthetic token-id stream
    deterministic for the session, which is what lets a second-pass replay
    of the same session reproduce the prefix tokens the first pass
    committed to the radix tree (and lets a proactive KV-cache drop walk
    the same chain).  When left ``None`` the engine uses a fresh
    :class:`random.Random` per branch — non-deterministic, matching the
    historical behaviour.
    """

    def __init__(
        self,
        worker: Worker,
        generation_stats: Optional[ReplayGenerationStats] = None,
        system_token_cache: Optional[Dict[str, List[int]]] = None,
        rng_factory: Optional[RngFactory] = None,
        drop_path_stats: Optional[DropPathStats] = None,
        retention_probe_stats: Optional[RetentionProbeStats] = None,
    ):
        self._system_token_cache: Dict[str, List[int]] = (
            system_token_cache if system_token_cache is not None else {}
        )
        self.queue_manager = QueueManager(
            worker=worker,
            system_token_cache=self._system_token_cache,
            rng_factory=rng_factory if rng_factory is not None else _default_rng_factory,
            generation_stats=generation_stats,
            drop_path_stats=drop_path_stats,
            retention_probe_stats=retention_probe_stats,
        )

    async def launch_trace(self, trace: ExecutionTrace):
        """Iterate *trace.events* and dispatch each event to its queue.

        Flow:
        1. Allocate a root queue for ``branch_path=()``.
        2. Walk ``trace.events`` in order:
           - ``parallel_start``: create child queues, push onto stack.
           - ``parallel_end``: close children, await completion, clean up.
           - All others: route to queue matching ``event.branch_path``.
        3. Close and await the root queue.
        """
        trace_id = trace.trace_id

        # Root queue for branch_path = ()
        root_queue_id = self.queue_manager.allocate_queue(trace_id, ())

        parallel_stack: List[List[str]] = []

        for event in trace.events:
            if event.event_type == "parallel_start":
                parent_path = tuple(event.branch_path or ())
                # Barrier: wait for the parent branch to finish processing
                # every event queued before this parallel_start. In a real
                # agent run, child sub-tasks can only be dispatched after
                # the parent's fork-producing generation has completed, so
                # children must not start hitting the server while the
                # parent's last ``worker.run_task`` is still in flight.
                await self.queue_manager.drain_branch(parent_path)

                num_branches = event.num_branches or 0
                child_queue_ids = []
                for i in range(num_branches):
                    child_path = parent_path + (i,)
                    child_qid = self.queue_manager.allocate_queue(trace_id, child_path)
                    child_queue_ids.append(child_qid)
                parallel_stack.append(child_queue_ids)

            elif event.event_type == "parallel_end":
                child_queue_ids = parallel_stack.pop()
                for qid in child_queue_ids:
                    self.queue_manager.close_queue(qid)
                await self.queue_manager.wait_all_done(child_queue_ids)
                for qid in child_queue_ids:
                    self.queue_manager.unregister_queue(qid)

            else:
                branch_path = tuple(event.branch_path or ())
                queue = self.queue_manager.get_queue(branch_path)
                await queue.put(event)

        # Close and await root queue
        self.queue_manager.close_queue(root_queue_id)
        await self.queue_manager.wait_all_done([root_queue_id])
        self.queue_manager.unregister_queue(root_queue_id)
