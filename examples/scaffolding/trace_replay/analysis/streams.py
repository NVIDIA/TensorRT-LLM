r"""Synthetic token streams + system-prompt registry.

The scaffolding trace records token *counts*, not token IDs. To analyze
prefix-cache hit rate we synthesize deterministic token IDs:

* :class:`SystemPromptRegistry` shares one token-ID stream per
  ``system_prompt_id`` UUID — different conversations using the same template
  see the same IDs (so they hit each other's blocks). Untagged system events
  fall back to a per-conversation key.
* :class:`ConversationSegments` mirrors
  :class:`tensorrt_llm.scaffolding.replay.QueueExecutor`: it stores one token
  segment per ``message_index`` per ``(branch_path, conversation_id)``, and
  builds an assistant request's prompt by concatenating those segments.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple


class TokenIdAllocator:
    """Hands out fresh, monotonically increasing synthetic token IDs."""

    def __init__(self) -> None:
        self._next = 0

    def take(self, count: int) -> List[int]:
        if count <= 0:
            return []
        ids = list(range(self._next, self._next + count))
        self._next += count
        return ids


class SystemPromptRegistry:
    """Maps a ``system_prompt_id`` UUID to a shared synthetic token sequence.

    Every system event in the trace must carry a non-empty ``system_prompt_id``
    (validated upstream in :mod:`cache_hit`). Conversations sharing the same
    UUID see the same token IDs, modeling a real prefix-cache hit; different
    UUIDs get disjoint streams.

    The registry grows lazily: when a request asks for ``length`` tokens of a
    UUID and the cached list is shorter, more IDs are appended via the shared
    :class:`TokenIdAllocator`. Returned slices are stable — a longer
    subsequent request never invalidates earlier shorter slices.
    """

    def __init__(self, allocator: TokenIdAllocator) -> None:
        self._allocator = allocator
        self._streams: Dict[str, List[int]] = {}

    def tokens(self, system_prompt_id: str, length: int) -> List[int]:
        if not system_prompt_id:
            raise ValueError("SystemPromptRegistry requires a non-empty system_prompt_id")
        stream = self._streams.get(system_prompt_id)
        if stream is None:
            stream = []
            self._streams[system_prompt_id] = stream
        if len(stream) < length:
            stream.extend(self._allocator.take(length - len(stream)))
        return stream[:length]

    def all_streams(self) -> Mapping[str, List[int]]:
        return self._streams


class ConversationSegments:
    """Per-(branch_path, conversation_id) message-segment store.

    Mirrors :class:`tensorrt_llm.scaffolding.replay.QueueExecutor`: each
    conversation in each branch holds a list of token-ID segments indexed by
    ``message_index``. Assistant requests are scored by concatenating these
    segments to form the prompt. Branch inheritance: when a child branch
    first touches a conversation, it inherits a snapshot of the closest
    ancestor branch's segments, so a fork preserves prefix-cache hit
    potential against the parent.
    """

    def __init__(
        self,
        allocator: TokenIdAllocator,
        system_registry: SystemPromptRegistry,
    ) -> None:
        self._allocator = allocator
        self._system_registry = system_registry
        self._segments: Dict[Tuple[Tuple[int, ...], int], List[List[int]]] = {}
        # First system_prompt_id seen per (branch, conv); used downstream to
        # group requests by sub-agent / template identity.
        self._seed_uuids: Dict[Tuple[Tuple[int, ...], int], str] = {}

    def _segments_for(self, branch_path: Sequence[int], conversation_id: int) -> List[List[int]]:
        key = (tuple(branch_path), conversation_id)
        existing = self._segments.get(key)
        if existing is not None:
            return existing

        # Inherit a snapshot from the closest ancestor branch holding this
        # conversation. This keeps prefix-cache reuse correct across forks.
        for parent_len in range(len(branch_path) - 1, -1, -1):
            parent_key = (tuple(branch_path[:parent_len]), conversation_id)
            parent = self._segments.get(parent_key)
            if parent is not None:
                inherited = [list(seg) for seg in parent]
                self._segments[key] = inherited
                return inherited

        new_list: List[List[int]] = []
        self._segments[key] = new_list
        return new_list

    def _store(
        self,
        branch_path: Sequence[int],
        conversation_id: int,
        message_index: Optional[int],
        token_ids: List[int],
    ) -> None:
        segments = self._segments_for(branch_path, conversation_id)
        if message_index is not None and 0 <= message_index < len(segments):
            segments[message_index] = token_ids
        else:
            segments.append(token_ids)

    def record_system(
        self,
        branch_path: Sequence[int],
        conversation_id: int,
        message_index: Optional[int],
        system_prompt_id: str,
        token_count: int,
    ) -> None:
        if not system_prompt_id:
            raise ValueError(
                "record_system requires a non-empty system_prompt_id "
                "(strict mode: every system event must carry a UUID)"
            )
        token_ids = list(self._system_registry.tokens(system_prompt_id, token_count))
        self._store(branch_path, conversation_id, message_index, token_ids)
        seed_key = (tuple(branch_path), conversation_id)
        self._seed_uuids.setdefault(seed_key, system_prompt_id)

    def seed_uuid_for(self, branch_path: Sequence[int], conversation_id: int) -> Optional[str]:
        """Return the first ``system_prompt_id`` seen for this conversation.

        Falls back to the closest ancestor branch's seed (matching the
        segment-inheritance rule). Returns ``None`` when the conversation
        never recorded a system event.
        """
        path = tuple(branch_path)
        for parent_len in range(len(path), -1, -1):
            seed = self._seed_uuids.get((path[:parent_len], conversation_id))
            if seed is not None:
                return seed
        return None

    def record_user_or_tool(
        self,
        branch_path: Sequence[int],
        conversation_id: int,
        message_index: Optional[int],
        token_count: int,
    ) -> None:
        token_ids = self._allocator.take(token_count)
        self._store(branch_path, conversation_id, message_index, token_ids)

    def record_assistant_content(
        self,
        branch_path: Sequence[int],
        conversation_id: int,
        message_index: Optional[int],
        content_token_count: int,
    ) -> None:
        token_ids = self._allocator.take(content_token_count)
        self._store(branch_path, conversation_id, message_index, token_ids)

    def allocate_tokens(self, count: int) -> List[int]:
        """Mint fresh, distinct token IDs without storing them in any segment.

        Used to model decode-time KV positions that the real KV cache
        manager stores in the radix tree but the scaffolding controller
        does NOT feed back as future-turn content (canonically: reasoning
        / chain-of-thought tokens). Because the returned IDs are disjoint
        from every other allocated stream, a later request whose assembled
        prompt skips this region will diverge from the cached prefix at
        the block where these tokens were inserted — exactly mirroring
        what a TRT-LLM C++ KV manager would do.
        """
        return self._allocator.take(count)

    def assemble_prompt(
        self,
        branch_path: Sequence[int],
        conversation_id: int,
        prompt_len: int,
    ) -> List[int]:
        """Return synthetic prompt of length *prompt_len* for a request.

        Concatenates stored segments. If the recorded segments fall short of
        *prompt_len* (compact traces may omit ``tokens`` on some events),
        pads with fresh allocator tokens so length matches what the trace
        reports as the actual prompt length. If they overshoot, truncates.
        """
        segments = self._segments_for(branch_path, conversation_id)
        prompt: List[int] = []
        for seg in segments:
            prompt.extend(seg)
            if len(prompt) >= prompt_len:
                break
        if len(prompt) > prompt_len:
            return prompt[:prompt_len]
        if len(prompt) < prompt_len:
            prompt.extend(self._allocator.take(prompt_len - len(prompt)))
        return prompt
