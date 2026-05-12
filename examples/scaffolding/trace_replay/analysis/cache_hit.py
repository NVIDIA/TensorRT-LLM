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
* All branches in the trace share a single global ``BlockPrefixCache``
  (the most permissive UB).

Algorithm:

1. **Pre-warm phase.** Walk the trace once to discover every distinct
   ``system_prompt_id``. Allocate a synthetic token sequence per UUID
   (longest length seen) and pre-insert its full blocks into the radix
   tree. Every block fully covered by a known system prompt is then a
   guaranteed cache hit on first request.

2. **Scoring phase.** Walk events in order, mirroring
   :class:`tensorrt_llm.scaffolding.replay.QueueExecutor`: per
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

from .blocks import BlockPrefixCache, full_blocks, reusable_token_len, validate_tokens_per_block
from .branch_summary import compute_branch_rollups
from .streams import ConversationSegments, SystemPromptRegistry, TokenIdAllocator

SCHEMA = "scaffolding.cache_hit_trace.v3"


def compute_cache_hit_upper_bound(
    trace_data: Dict[str, Any],
    *,
    tokens_per_block: int = 32,
    exclude_last_token_from_blocks: bool = True,
    decode_kv_reuse: bool = True,
    cot_pollutes_cache: bool = True,
    include_rollups: bool = True,
    trace_file: Any = None,
) -> Dict[str, Any]:
    """Compute the infinite-cache prefix-hit record for one trace.

    Args:
        trace_data: Parsed contents of a ``*.trace.json`` file.
        tokens_per_block: KV-cache block size in tokens.
        exclude_last_token_from_blocks: Match TRT-LLM behavior where the
            request's last prompt token is not part of any reusable block.
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
    prefix_cache = BlockPrefixCache()

    # ---- Phase 1: pre-warm cache with every system prompt template. ----
    system_prefix_lengths = _collect_system_prefix_lengths(events)
    for uuid_, length in system_prefix_lengths.items():
        token_ids = system_registry.tokens(uuid_, length)
        usable_len = reusable_token_len(length, exclude_last_token_from_blocks)
        prefix_cache.insert_only(full_blocks(token_ids[:usable_len], tokens_per_block))
    preloaded_system_blocks = prefix_cache.cached_blocks

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
            tokens_per_block=tokens_per_block,
            exclude_last_token_from_blocks=exclude_last_token_from_blocks,
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
        prefix_cache=prefix_cache,
        preloaded_system_blocks=preloaded_system_blocks,
        distinct_system_prompts=len(system_prefix_lengths),
        tokens_per_block=tokens_per_block,
        exclude_last_token_from_blocks=exclude_last_token_from_blocks,
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
        self.cacheable_prompt_tokens = 0
        self.non_cacheable_tail_tokens = 0
        self.hit_tokens = 0
        self.hit_blocks = 0
        self.miss_blocks = 0

    def add(self, request: Dict[str, Any]) -> None:
        self.prompt_tokens += request["prompt_tokens"]
        self.cacheable_prompt_tokens += request["cacheable_prompt_tokens"]
        self.non_cacheable_tail_tokens += request["non_cacheable_tail_tokens"]
        self.hit_tokens += request["cache_hit_tokens"]
        self.hit_blocks += request["cache_hit_blocks"]
        self.miss_blocks += request["cache_miss_blocks"]


def _score_assistant_event(
    *,
    event: Dict[str, Any],
    event_index: int,
    branch_path: Sequence[int],
    conversation_id: int,
    message_index: Optional[int],
    segments: ConversationSegments,
    prefix_cache: BlockPrefixCache,
    tokens_per_block: int,
    exclude_last_token_from_blocks: bool,
    decode_kv_reuse: bool,
    cot_pollutes_cache: bool,
    request_index: int,
) -> Dict[str, Any]:
    """Score a single assistant message event and update bookkeeping."""
    prompt_tokens = _as_nonnegative_int(event["prompt_tokens"], "prompt_tokens", event_index)
    completion_tokens = _as_nonnegative_int(
        event.get("completion_tokens") or 0, "completion_tokens", event_index
    )
    reasoning_tokens = _as_nonnegative_int(
        event.get("reasoning_tokens") or 0, "reasoning_tokens", event_index
    )

    prompt_token_ids = segments.assemble_prompt(branch_path, conversation_id, prompt_tokens)
    block_input_len = reusable_token_len(prompt_tokens, exclude_last_token_from_blocks)
    request_blocks = full_blocks(prompt_token_ids[:block_input_len], tokens_per_block)
    cacheable_prompt_tokens = len(request_blocks) * tokens_per_block
    non_cacheable_tail_tokens = prompt_tokens - cacheable_prompt_tokens

    hit_blocks, new_cached_prompt_blocks = prefix_cache.insert_and_count_hit(request_blocks)
    miss_blocks = len(request_blocks) - hit_blocks
    hit_tokens = hit_blocks * tokens_per_block
    miss_tokens = prompt_tokens - hit_tokens
    cache_block_hit_rate = hit_blocks / len(request_blocks) if request_blocks else 0.0

    # Decode tokens are split into:
    #   * reasoning_tokens — hidden chain-of-thought; scaffolding never feeds
    #     these back as future-turn content, so they DO NOT enter the
    #     conversation segment store.
    #   * content_tokens (= completion - reasoning) — visible assistant
    #     content; recorded in segments so the next turn's prompt assembly
    #     includes it.
    # What gets inserted into the prefix cache depends on two flags:
    #   * decode_kv_reuse=False: nothing is inserted (model a deployment
    #     that drops decode-phase KV at request end).
    #   * decode_kv_reuse=True, cot_pollutes_cache=False: insert
    #     [prompt + content] only — the optimistic upper bound where
    #     reasoning is treated as if it never occupied any KV position.
    #   * decode_kv_reuse=True, cot_pollutes_cache=True (default): insert
    #     [prompt + reasoning + content] with reasoning at distinct token
    #     IDs. This mirrors the real TRT-LLM C++ KV manager, which stores
    #     the entire decode stream contiguously in the radix tree. Later
    #     turns whose prompts omit prior reasoning diverge from the cached
    #     prefix at the reasoning-insertion block — so the prompt prefix
    #     can still hit, but every block after the reasoning position
    #     becomes a miss.
    content_tokens = max(completion_tokens - reasoning_tokens, 0)
    cached_before_completion = prefix_cache.cached_blocks
    if content_tokens > 0:
        segments.record_assistant_content(
            branch_path, conversation_id, message_index, content_tokens
        )
    if decode_kv_reuse and completion_tokens > 0:
        if cot_pollutes_cache and reasoning_tokens > 0:
            reasoning_token_ids = segments.allocate_tokens(reasoning_tokens)
            if content_tokens > 0:
                prompt_plus_content = segments.assemble_prompt(
                    branch_path, conversation_id, prompt_tokens + content_tokens
                )
                content_token_ids = prompt_plus_content[prompt_tokens:]
            else:
                content_token_ids = []
            decode_stream = list(prompt_token_ids) + reasoning_token_ids + content_token_ids
            decode_block_len = reusable_token_len(
                prompt_tokens + reasoning_tokens + content_tokens,
                exclude_last_token_from_blocks,
            )
            prefix_cache.insert_and_count_hit(
                full_blocks(decode_stream[:decode_block_len], tokens_per_block)
            )
        elif content_tokens > 0:
            tail_prompt = segments.assemble_prompt(
                branch_path, conversation_id, prompt_tokens + content_tokens
            )
            tail_block_len = reusable_token_len(
                prompt_tokens + content_tokens, exclude_last_token_from_blocks
            )
            prefix_cache.insert_and_count_hit(
                full_blocks(tail_prompt[:tail_block_len], tokens_per_block)
            )
    new_cached_completion_blocks = prefix_cache.cached_blocks - cached_before_completion
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
        "prompt_tokens": prompt_tokens,
        "tokens_per_block": tokens_per_block,
        "cacheable_prompt_tokens": cacheable_prompt_tokens,
        "non_cacheable_tail_tokens": non_cacheable_tail_tokens,
        "cache_hit_blocks": hit_blocks,
        "cache_miss_blocks": miss_blocks,
        "cache_hit_tokens": hit_tokens,
        "cache_miss_tokens": miss_tokens,
        "cache_hit_rate": hit_tokens / prompt_tokens if prompt_tokens else 0.0,
        "cache_block_hit_rate": cache_block_hit_rate,
        "cacheable_token_hit_rate": (
            hit_tokens / cacheable_prompt_tokens if cacheable_prompt_tokens else 0.0
        ),
        "new_cached_prompt_blocks": new_cached_prompt_blocks,
        "new_cached_completion_blocks": new_cached_completion_blocks,
        "new_cached_blocks": new_cached_blocks,
        "new_cached_tokens": new_cached_blocks * tokens_per_block,
        "cached_blocks_after_request": prefix_cache.cached_blocks,
        "cached_tokens_after_request": prefix_cache.cached_blocks * tokens_per_block,
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
    prefix_cache: BlockPrefixCache,
    preloaded_system_blocks: int,
    distinct_system_prompts: int,
    tokens_per_block: int,
    exclude_last_token_from_blocks: bool,
    decode_kv_reuse: bool,
    cot_pollutes_cache: bool,
) -> Dict[str, Any]:
    overall_hit_rate = totals.hit_tokens / totals.prompt_tokens if totals.prompt_tokens else 0.0
    cacheable_hit_rate = (
        totals.hit_tokens / totals.cacheable_prompt_tokens
        if totals.cacheable_prompt_tokens
        else 0.0
    )
    block_total = totals.hit_blocks + totals.miss_blocks
    overall_block_hit_rate = totals.hit_blocks / block_total if block_total else 0.0
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trace_file": str(trace_file) if trace_file is not None else None,
        "trace_id": trace_data.get("trace_id"),
        "algorithm": {
            "cache_model": "infinite_radix_prefix_tree_of_kv_blocks",
            "preloaded_system_prompts": True,
            "tokens_per_block": tokens_per_block,
            "exclude_last_token_from_blocks": exclude_last_token_from_blocks,
            "decode_kv_reuse": decode_kv_reuse,
            "cot_pollutes_cache": cot_pollutes_cache,
            "parallel_policy": "shared",
            "request_definition": "assistant message events with prompt_tokens",
            "hit_definition": "Only complete matching blocks count as cache hits.",
            "strict_mode": (
                "Every system message must carry a system_prompt_id UUID; "
                "drop_kv_cache events are rejected."
            ),
            "note": (
                "Phase 1 inserts every distinct system prompt (keyed by "
                "system_prompt_id UUID) into the prefix tree before any "
                "request is scored. Phase 2 walks the trace event-by-event, "
                "accumulating per-(branch_path, conversation_id) message "
                "segments — system slices are drawn from a UUID-keyed shared "
                "token stream so different conversations using the same "
                "template hit those blocks; user/tool/assistant content uses "
                "fresh synthetic tokens. Reasoning tokens are excluded from "
                "assistant content in the segment store because scaffolding "
                "does not feed hidden reasoning back as content on the next "
                "request. When cot_pollutes_cache=True (default), the "
                "decode-time KV stream stored in the prefix tree is "
                "[prompt + reasoning + content] with reasoning at distinct "
                "token IDs, mirroring the real TRT-LLM C++ KV manager; "
                "subsequent turns then diverge from the cached prefix at the "
                "reasoning-insertion block. When False, only [prompt + "
                "content] is stored — the optimistic upper bound. "
                "decode_kv_reuse=False discards decode-phase KV from the "
                "cache (segments still grow so prompt assembly stays "
                "consistent). All branches share a single global cache."
            ),
        },
        "summary": {
            "event_count": len(events),
            "llm_request_count": len(requests),
            "tokens_per_block": tokens_per_block,
            "exclude_last_token_from_blocks": exclude_last_token_from_blocks,
            "decode_kv_reuse": decode_kv_reuse,
            "cot_pollutes_cache": cot_pollutes_cache,
            "distinct_system_prompts": distinct_system_prompts,
            "preloaded_system_blocks": preloaded_system_blocks,
            "preloaded_system_tokens": preloaded_system_blocks * tokens_per_block,
            "total_prompt_tokens": totals.prompt_tokens,
            "total_cacheable_prompt_tokens": totals.cacheable_prompt_tokens,
            "total_non_cacheable_tail_tokens": totals.non_cacheable_tail_tokens,
            "total_cache_hit_blocks": totals.hit_blocks,
            "total_cache_miss_blocks": totals.miss_blocks,
            "total_cache_hit_tokens": totals.hit_tokens,
            "total_cache_miss_tokens": totals.prompt_tokens - totals.hit_tokens,
            "overall_cache_hit_rate": overall_hit_rate,
            "overall_cache_block_hit_rate": overall_block_hit_rate,
            "cacheable_token_hit_rate": cacheable_hit_rate,
            "minimal_cache_blocks": prefix_cache.cached_blocks,
            "minimal_cache_tokens": prefix_cache.cached_blocks * tokens_per_block,
            "max_prompt_tokens": max((req["prompt_tokens"] for req in requests), default=0),
        },
        "requests": requests,
    }
    if rollups is not None:
        record["rollups"] = rollups
    return record
