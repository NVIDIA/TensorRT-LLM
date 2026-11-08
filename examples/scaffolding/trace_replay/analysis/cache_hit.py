r"""Per-trace KV-cache hit upper-bound computation.

The single public entry point :func:`compute_cache_hit_upper_bound` takes a
parsed trace and returns a JSON-serializable record of per-request hit/miss
statistics, an aggregate summary, and per-(branch / depth / system-prompt)
rollups suitable for analyzing branched / parallel sub-agent workflows
(ToT, Open Deep Research).

Strict mode contract (this analyzer):

* Every ``role=system`` event MUST carry a non-empty ``system_prompt_id``
  UUID. Untagged system events are an error.
* ``drop_kv_cache`` events are an error — this analyzer assumes no eviction.
* All branches in the trace share a single global token prefix cache
  (the most permissive UB).

Algorithm:

1. **Pre-warm phase.** Walk the trace once to discover every distinct
   ``system_prompt_id``. Allocate a synthetic token sequence per UUID
   (longest length seen) and pre-insert its full blocks into the radix
   tree. Every block fully covered by a known system prompt is then a
   guaranteed cache hit on first request.

2. **Scoring phase.** Walk events in order, mirroring
   :class:`tensorrt_llm.scaffolding.trace_replay.replay.QueueExecutor`: per
   ``(branch_path, conversation_id)`` keep a list of segments indexed by
   ``message_index``. System segments draw tokens from the shared registry
   (cross-conversation reuse exact). User/tool/assistant-content segments
   use fresh allocator tokens. For each assistant request, assemble the
   prompt by concatenating segments, score it against the prefix tree,
   then optionally store the decode-time sequence into the cache when
   ``decode_kv_reuse=True``. With ``cot_pollutes_cache=True`` (default),
   the stored sequence is ``[prompt + reasoning + content]`` — reasoning
   token IDs are minted fresh and live only in the cache, never in the
   segment store, so future-turn prompts assembled from segments diverge
   from the cached prefix at the reasoning-insertion block (mirroring
   real TRT-LLM C++ KV manager behavior). With ``cot_pollutes_cache=False``
   the stored sequence is just ``[prompt + content]`` — the optimistic
   upper bound that treats reasoning as if it never occupied KV.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, MutableMapping, Optional, Sequence

from .blocks import (
    TokenPrefixCache,
    engine_aligned_block_count,
    engine_aligned_hit_blocks,
    validate_tokens_per_block,
)
from .branch_summary import compute_branch_rollups
from .streams import ConversationSegments, SystemPromptRegistry, TokenIdAllocator

SCHEMA = "scaffolding.cache_hit_trace.v6"


def compute_cache_hit_upper_bound(
    trace_data: Dict[str, Any],
    *,
    tokens_per_block: int = 32,
    decode_kv_reuse: bool = True,
    cot_pollutes_cache: bool = True,
    include_rollups: bool = True,
    trace_file: Any = None,
) -> Dict[str, Any]:
    """Compute the infinite-cache prefix-hit upper bound for one trace.

    The result is a strict upper bound on every per-request
    ``optimal_cache_block_hit_rate`` an in-line TRT-LLM engine can achieve
    when replaying the trace via
    ``tensorrt_llm.scaffolding.trace_replay.replay.QueueExecutor`` against
    ``/v1/completions``.

    Args:
        trace_data: Parsed contents of a ``*.trace.json`` file.
        tokens_per_block: KV-cache block size in tokens.
        decode_kv_reuse: When ``True`` (default), assistant decode tokens
            are inserted into the cache and may be reused as a prefix by
            later requests in the same conversation. When ``False``, decode
            tokens are still recorded in the conversation's segment list
            (so subsequent ``prompt_tokens`` reconciliation is correct) but
            are NOT inserted into the cache — modeling a deployment where
            decode-phase KV is dropped after the request.
        cot_pollutes_cache: Only meaningful when ``decode_kv_reuse=True``.
            When ``True`` (default), model real TRT-LLM C++ KV manager
            behavior: the decode-time KV stream stored in the radix tree
            is ``[prompt + reasoning + content]``, with reasoning occupying
            distinct positions between prompt and content. Subsequent
            turns whose prompts omit prior reasoning (the only thing
            scaffolding ever feeds back) then diverge from the cached
            prefix at the reasoning-insertion block, causing miss
            propagation through every block after it. When ``False``,
            fall back to the optimistic upper bound where reasoning is
            assumed absent from the cache: only ``[prompt + content]`` is
            inserted, so a later turn can hit through to the end of
            content. The gap between the two modes quantifies the KV
            cache waste caused by reasoning tokens.
        include_rollups: When ``True`` (default), compute per-(branch root /
            branch depth / system-prompt UUID) rollups.
        trace_file: Optional path-like value placed verbatim in the output.

    Returns:
        A JSON-serializable dictionary with ``schema``, ``algorithm``,
        ``summary``, ``requests`` and (optionally) ``rollups`` fields.
    """
    validate_tokens_per_block(tokens_per_block)

    events = trace_data.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Trace events must be a list")

    _reject_unsupported_events(events)

    allocator = TokenIdAllocator()
    system_registry = SystemPromptRegistry(allocator)
    segments = ConversationSegments(allocator, system_registry)
    prefix_cache = TokenPrefixCache()
    cached_block_count = [0]  # mutable cell so helpers can read/update

    def _commit_stream(token_ids):
        """Insert *token_ids* and update the running committed-block count.

        Uses engine-aligned ``ceil(L / block)`` block accounting. The token
        sequence itself is stored in the radix tree so future requests can
        match at single-token granularity (a partial trailing block hits
        when its content matches).
        """
        if not token_ids:
            return 0
        before = cached_block_count[0]
        cached_block_count[0] = max(
            before, engine_aligned_block_count(len(token_ids), tokens_per_block)
        )
        prefix_cache.insert(token_ids)
        return cached_block_count[0] - before

    # ---- Phase 1: pre-warm cache with every system prompt template. ----
    # Each distinct system_prompt_id contributes its full ceil(L/block)
    # blocks (matching the engine's commit count once the very first
    # request that uses it lands in the radix tree).
    system_prefix_lengths = _collect_system_prefix_lengths(events)
    for uuid_, length in system_prefix_lengths.items():
        token_ids = system_registry.tokens(uuid_, length)
        _commit_stream(token_ids)
    preloaded_system_blocks = cached_block_count[0]

    # ---- Phase 2: score each assistant request against the prefix tree. ----
    requests: List[Dict[str, Any]] = []
    totals = _Totals()

    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Event {event_index} is not an object: {event!r}")
        if event.get("event_type") != "message":
            continue

        role = event.get("role")
        branch_path = _branch_path(event)
        conversation_id = event.get("conversation_id")
        if not isinstance(conversation_id, int):
            raise ValueError(
                f"Event {event_index} has invalid conversation_id: {conversation_id!r}"
            )
        message_index = event.get("message_index")

        if role == "system":
            system_prompt_id = event.get("system_prompt_id")
            if not system_prompt_id:
                raise ValueError(
                    f"Event {event_index}: system message has no "
                    "system_prompt_id (strict mode requires UUID-tagged "
                    "system prompts)"
                )
            token_count = _as_nonnegative_int(event.get("tokens", 0), "tokens", event_index)
            segments.record_system(
                branch_path,
                conversation_id,
                message_index,
                system_prompt_id,
                token_count,
            )
            continue

        if role in ("user", "tool"):
            token_count = _as_nonnegative_int(event.get("tokens", 0), "tokens", event_index)
            segments.record_user_or_tool(branch_path, conversation_id, message_index, token_count)
            continue

        if role != "assistant" or "prompt_tokens" not in event:
            continue

        request = _score_assistant_event(
            event=event,
            event_index=event_index,
            branch_path=branch_path,
            conversation_id=conversation_id,
            message_index=message_index,
            segments=segments,
            prefix_cache=prefix_cache,
            commit_stream=_commit_stream,
            cached_block_count_cell=cached_block_count,
            tokens_per_block=tokens_per_block,
            decode_kv_reuse=decode_kv_reuse,
            cot_pollutes_cache=cot_pollutes_cache,
            request_index=len(requests),
        )
        requests.append(request)
        totals.add(request)

    rollups = compute_branch_rollups(requests) if include_rollups else None
    return _build_record(
        trace_data=trace_data,
        trace_file=trace_file,
        events=events,
        requests=requests,
        rollups=rollups,
        totals=totals,
        cached_blocks=cached_block_count[0],
        preloaded_system_blocks=preloaded_system_blocks,
        distinct_system_prompts=len(system_prefix_lengths),
        tokens_per_block=tokens_per_block,
        decode_kv_reuse=decode_kv_reuse,
        cot_pollutes_cache=cot_pollutes_cache,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _Totals:
    """Running sums across requests for a single trace."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.hit_tokens = 0
        self.hit_blocks = 0
        self.miss_blocks = 0

    def add(self, request: Dict[str, Any]) -> None:
        self.prompt_tokens += request["prompt_tokens"]
        self.hit_tokens += request["optimal_cache_hit_tokens"]
        self.hit_blocks += request["optimal_cache_hit_blocks"]
        self.miss_blocks += request["optimal_cache_miss_blocks"]


def _score_assistant_event(
    *,
    event: Dict[str, Any],
    event_index: int,
    branch_path: Sequence[int],
    conversation_id: int,
    message_index: Optional[int],
    segments: ConversationSegments,
    prefix_cache: TokenPrefixCache,
    commit_stream,
    cached_block_count_cell,
    tokens_per_block: int,
    decode_kv_reuse: bool,
    cot_pollutes_cache: bool,
    request_index: int,
) -> Dict[str, Any]:
    """Score a single assistant message event with engine-aligned accounting.

    Block counting follows the TRT-LLM engine exactly:
      * total blocks for the request = ``ceil(L_effective / block)``
      * reused blocks = ``min(ceil(matched_tokens / block), total)``
      * ``L_effective`` = length of segments concatenated in trace order
        (i.e., what ``QueueExecutor`` actually sends to the engine via
        ``/v1/completions`` as ``prompt=List[int]``). This may differ from
        the trace's recorded ``prompt_tokens`` field, which carries the
        original recording API's chat-template-formatted count.
    """
    trace_prompt_tokens = _as_nonnegative_int(
        event.get("prompt_tokens") or 0, "prompt_tokens", event_index
    )
    completion_tokens = _as_nonnegative_int(
        event.get("completion_tokens") or 0, "completion_tokens", event_index
    )
    reasoning_tokens = _as_nonnegative_int(
        event.get("reasoning_tokens") or 0, "reasoning_tokens", event_index
    )

    # Effective prompt = exactly what the replay engine sends.
    prompt_token_ids = segments.assemble_prompt(branch_path, conversation_id)
    effective_prompt_tokens = len(prompt_token_ids)
    n_total_blocks = engine_aligned_block_count(effective_prompt_tokens, tokens_per_block)

    matched_tokens = prefix_cache.match(prompt_token_ids)
    hit_blocks = engine_aligned_hit_blocks(matched_tokens, n_total_blocks, tokens_per_block)
    miss_blocks = n_total_blocks - hit_blocks

    hit_tokens = min(matched_tokens, effective_prompt_tokens)
    miss_tokens = effective_prompt_tokens - hit_tokens
    optimal_cache_block_hit_rate = hit_blocks / n_total_blocks if n_total_blocks else 0.0
    optimal_cache_hit_rate = (
        hit_tokens / effective_prompt_tokens if effective_prompt_tokens else 0.0
    )

    # The new prompt is committed first (so its own block count contributes
    # to ``cached_blocks_after_request``), then decode tokens extend it.
    new_cached_prompt_blocks = commit_stream(prompt_token_ids)

    # Decode-time storage: same semantics as before. Reasoning is hidden from
    # future-turn prompt assembly but is inserted into the radix tree under
    # ``cot_pollutes_cache=True`` to mirror the C++ KV manager.
    content_tokens = max(completion_tokens - reasoning_tokens, 0)
    if content_tokens > 0:
        segments.record_assistant_content(
            branch_path, conversation_id, message_index, content_tokens
        )

    cached_before_completion = cached_block_count_cell[0]
    if decode_kv_reuse and completion_tokens > 0:
        if cot_pollutes_cache and reasoning_tokens > 0:
            reasoning_token_ids = segments.allocate_tokens(reasoning_tokens)
            if content_tokens > 0:
                content_token_ids = segments.assemble_prompt(branch_path, conversation_id)[
                    effective_prompt_tokens:
                ]
            else:
                content_token_ids = []
            decode_stream = list(prompt_token_ids) + reasoning_token_ids + content_token_ids
            commit_stream(decode_stream)
        elif content_tokens > 0:
            tail_prompt = segments.assemble_prompt(branch_path, conversation_id)
            commit_stream(tail_prompt)
    new_cached_completion_blocks = cached_block_count_cell[0] - cached_before_completion
    new_cached_blocks = new_cached_prompt_blocks + new_cached_completion_blocks

    seed_uuid = segments.seed_uuid_for(branch_path, conversation_id)
    branch_path_list = list(branch_path)
    return {
        "request_index": request_index,
        "event_index": event_index,
        "conversation_id": conversation_id,
        "branch_path": branch_path_list,
        "branch_root": branch_path_list[0] if branch_path_list else None,
        "branch_depth": len(branch_path_list),
        "system_prompt_id_seed": seed_uuid,
        "message_index": message_index,
        # ``prompt_tokens`` mirrors the engine-visible length (effective).
        # The trace's original recorded value is exposed separately under
        # ``trace_prompt_tokens`` for cross-checks.
        "prompt_tokens": effective_prompt_tokens,
        "trace_prompt_tokens": trace_prompt_tokens,
        "tokens_per_block": tokens_per_block,
        "matched_prompt_tokens": min(matched_tokens, effective_prompt_tokens),
        "optimal_cache_hit_blocks": hit_blocks,
        "optimal_cache_miss_blocks": miss_blocks,
        "optimal_cache_hit_tokens": hit_tokens,
        "optimal_cache_miss_tokens": miss_tokens,
        "optimal_cache_hit_rate": optimal_cache_hit_rate,
        "optimal_cache_block_hit_rate": optimal_cache_block_hit_rate,
        "new_cached_prompt_blocks": new_cached_prompt_blocks,
        "new_cached_completion_blocks": new_cached_completion_blocks,
        "new_cached_blocks": new_cached_blocks,
        "new_cached_tokens": new_cached_blocks * tokens_per_block,
        "cached_blocks_after_request": cached_block_count_cell[0],
        "cached_tokens_after_request": cached_block_count_cell[0] * tokens_per_block,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reusable_completion_tokens": content_tokens,
        "finish_reason": event.get("finish_reason"),
        "tool_calls": event.get("tool_calls", []),
    }


def _reject_unsupported_events(events: List[Any]) -> None:
    """Raise on any event type the strict-mode analyzer cannot model."""
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if event_type == "drop_kv_cache":
            raise ValueError(
                f"Event {event_index}: drop_kv_cache events are not "
                "supported by this analyzer (it assumes an infinite, "
                "non-evicting cache)"
            )


def _collect_system_prefix_lengths(events: Sequence[Any]) -> Dict[str, int]:
    """Largest ``tokens`` seen per ``system_prompt_id`` across the trace."""
    lengths: Dict[str, int] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "message" or event.get("role") != "system":
            continue
        system_prompt_id = event.get("system_prompt_id")
        if not system_prompt_id:
            raise ValueError(
                f"Event {event_index}: system message has no system_prompt_id "
                "(strict mode requires UUID-tagged system prompts)"
            )
        token_count = event.get("tokens") or 0
        if not isinstance(token_count, int) or token_count <= 0:
            continue
        prev = lengths.get(system_prompt_id, 0)
        if token_count > prev:
            lengths[system_prompt_id] = token_count
    return lengths


def _branch_path(event: MutableMapping[str, Any]) -> List[int]:
    raw = event.get("branch_path", [])
    if not isinstance(raw, list) or not all(isinstance(x, int) for x in raw):
        raise ValueError(f"Invalid branch_path: {raw!r}")
    return raw


def _as_nonnegative_int(value: Any, field_name: str, event_index: int) -> int:
    if value is None:
        return 0
    if not isinstance(value, int):
        raise ValueError(f"Event {event_index} has non-integer {field_name}: {value!r}")
    if value < 0:
        raise ValueError(f"Event {event_index} has negative {field_name}: {value}")
    return value


def _build_record(
    *,
    trace_data: Dict[str, Any],
    trace_file: Any,
    events: List[Any],
    requests: List[Dict[str, Any]],
    rollups: Optional[Dict[str, List[Dict[str, Any]]]],
    totals: _Totals,
    cached_blocks: int,
    preloaded_system_blocks: int,
    distinct_system_prompts: int,
    tokens_per_block: int,
    decode_kv_reuse: bool,
    cot_pollutes_cache: bool,
) -> Dict[str, Any]:
    optimal_overall_cache_hit_rate = (
        totals.hit_tokens / totals.prompt_tokens if totals.prompt_tokens else 0.0
    )
    block_total = totals.hit_blocks + totals.miss_blocks
    optimal_overall_cache_block_hit_rate = totals.hit_blocks / block_total if block_total else 0.0
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trace_file": str(trace_file) if trace_file is not None else None,
        "trace_id": trace_data.get("trace_id"),
        "algorithm": {
            "cache_model": "infinite_token_radix_tree_engine_aligned",
            "preloaded_system_prompts": True,
            "tokens_per_block": tokens_per_block,
            "decode_kv_reuse": decode_kv_reuse,
            "cot_pollutes_cache": cot_pollutes_cache,
            "parallel_policy": "shared",
            "request_definition": (
                "assistant message events; prompt length = natural "
                "concatenation of stored segments (matches replay engine's "
                "/v1/completions input_tokens length)"
            ),
            "hit_definition": (
                "n_total_blocks = ceil(L_effective / block); "
                "reused_blocks = min(ceil(matched_tokens / block), n_total_blocks). "
                "Matches the TRT-LLM engine, whose KV cache hashes the partial "
                "trailing block by its actual content so the upper bound "
                "dominates every measured num_reused_blocks."
            ),
            "strict_mode": (
                "Every system message must carry a system_prompt_id UUID; "
                "drop_kv_cache events are rejected."
            ),
            "note": (
                "Phase 1 inserts every distinct system prompt's full token "
                "stream into a token-level radix tree (compressed-edge), so "
                "the first request hitting that template reuses ceil(L/block) "
                "blocks regardless of partial-block alignment. Phase 2 walks "
                "the trace event-by-event, accumulating per-(branch_path, "
                "conversation_id) message segments — system slices share a "
                "UUID-keyed token stream; user/tool/assistant-content use "
                "fresh synthetic IDs. Reasoning tokens are excluded from "
                "assistant content in the segment store because scaffolding "
                "does not feed hidden reasoning back. When cot_pollutes_cache "
                "is True (default), the decode-time stream inserted into the "
                "tree is [prompt + reasoning + content] with reasoning at "
                "distinct token IDs, mirroring the C++ KV manager; subsequent "
                "turns diverge from the cached prefix at the reasoning "
                "boundary. When False, only [prompt + content] is stored — "
                "the optimistic upper bound. decode_kv_reuse=False discards "
                "decode-phase KV (segments still grow so prompt assembly "
                "stays consistent). All branches share a single global cache."
            ),
        },
        "summary": {
            "event_count": len(events),
            "llm_request_count": len(requests),
            "tokens_per_block": tokens_per_block,
            "decode_kv_reuse": decode_kv_reuse,
            "cot_pollutes_cache": cot_pollutes_cache,
            "distinct_system_prompts": distinct_system_prompts,
            "preloaded_system_blocks": preloaded_system_blocks,
            "preloaded_system_tokens": preloaded_system_blocks * tokens_per_block,
            "total_prompt_tokens": totals.prompt_tokens,
            "optimal_total_cache_hit_blocks": totals.hit_blocks,
            "optimal_total_cache_miss_blocks": totals.miss_blocks,
            "optimal_total_cache_hit_tokens": totals.hit_tokens,
            "optimal_total_cache_miss_tokens": totals.prompt_tokens - totals.hit_tokens,
            "optimal_overall_cache_hit_rate": optimal_overall_cache_hit_rate,
            "optimal_overall_cache_block_hit_rate": optimal_overall_cache_block_hit_rate,
            "minimal_cache_blocks": cached_blocks,
            "minimal_cache_tokens": cached_blocks * tokens_per_block,
            "max_prompt_tokens": max((req["prompt_tokens"] for req in requests), default=0),
        },
        "requests": requests,
    }
    if rollups is not None:
        record["rollups"] = rollups
    return record
