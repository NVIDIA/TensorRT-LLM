# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional

import torch

from tensorrt_llm.sampling_params import (
    LogitsProcessor,
    SamplingParams,
    validate_thinking_token_budget,
)


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    """Force the reasoning end token sequence once a thinking budget is spent."""

    def __init__(
        self,
        thinking_token_budget: int,
        reasoning_start_token_ids: List[int],
        reasoning_end_token_ids: List[int],
    ) -> None:
        budget = validate_thinking_token_budget(thinking_token_budget)
        if budget is None:
            raise ValueError(
                "thinking_token_budget must be set when creating ThinkingBudgetLogitsProcessor"
            )
        if not reasoning_start_token_ids:
            raise ValueError("reasoning_start_token_ids must not be empty")
        if not reasoning_end_token_ids:
            raise ValueError("reasoning_end_token_ids must not be empty")
        self.thinking_token_budget = budget
        self.reasoning_start_token_ids = list(reasoning_start_token_ids)
        self.reasoning_end_token_ids = list(reasoning_end_token_ids)

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        del req_id, client_id
        if stream_ptr is None:
            self._apply(token_ids, logits)
            return
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            self._apply(token_ids, logits)

    def _apply(self, token_ids: List[List[int]], logits: torch.Tensor) -> None:
        for beam_idx, beam_token_ids in enumerate(token_ids):
            forced_token = self._forced_token(beam_token_ids)
            if forced_token is not None:
                self._force_token(logits, beam_idx, len(token_ids), forced_token)

    def _forced_token(self, token_ids: List[int]) -> Optional[int]:
        start_idx = _find_last_sequence_index(token_ids, self.reasoning_start_token_ids)
        if start_idx == -1:
            return None

        end_idx = _find_last_sequence_index(token_ids, self.reasoning_end_token_ids)
        if end_idx > start_idx:
            return None

        reasoning_start = start_idx + len(self.reasoning_start_token_ids)
        reasoning_token_count = len(token_ids) - reasoning_start
        partial_end_len = _longest_suffix_prefix_len(token_ids, self.reasoning_end_token_ids)
        if (
            partial_end_len > 0
            and reasoning_token_count - partial_end_len >= self.thinking_token_budget
        ):
            return self.reasoning_end_token_ids[partial_end_len]

        if reasoning_token_count >= self.thinking_token_budget:
            return self.reasoning_end_token_ids[0]
        return None

    @staticmethod
    def _force_token(logits: torch.Tensor, beam_idx: int, beam_count: int, token_id: int) -> None:
        if token_id < 0 or token_id >= logits.shape[-1]:
            raise ValueError(
                f"Forced reasoning end token id {token_id} is outside the "
                f"logits vocabulary dimension {logits.shape[-1]}"
            )

        target = logits
        if logits.dim() > 1 and logits.shape[0] == beam_count:
            target = logits[beam_idx]
        target[:] = float("-inf")
        target[..., token_id] = 0


class _ChainedLogitsProcessor(LogitsProcessor):
    def __init__(self, processors: List[Any]) -> None:
        self.processors = processors

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        token_ids: List[List[int]],
        stream_ptr: Optional[int],
        client_id: Optional[int],
    ) -> None:
        for processor in self.processors:
            processor(req_id, logits, token_ids, stream_ptr, client_id)


def add_thinking_budget_logits_processor(
    sampling_params: SamplingParams,
    *,
    reasoning_parser: Optional[str],
    tokenizer: Any,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Attach a thinking-budget logits processor to ``sampling_params``."""
    if sampling_params.thinking_token_budget is None:
        return
    existing = sampling_params.logits_processor
    if isinstance(existing, ThinkingBudgetLogitsProcessor):
        return
    if isinstance(existing, _ChainedLogitsProcessor) and any(
        isinstance(p, ThinkingBudgetLogitsProcessor) for p in existing.processors
    ):
        return
    if isinstance(existing, list) and any(
        isinstance(p, ThinkingBudgetLogitsProcessor) for p in existing
    ):
        return
    if reasoning_parser is None:
        raise ValueError("thinking_token_budget requires a configured reasoning_parser")
    if tokenizer is None:
        raise ValueError("thinking_token_budget requires a tokenizer")

    from .reasoning_parser import ReasoningParserFactory

    parser = ReasoningParserFactory.create_reasoning_parser(reasoning_parser, chat_template_kwargs)
    reasoning_start = _get_reasoning_boundary(parser, "start")
    reasoning_end = _get_reasoning_boundary(parser, "end")
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=sampling_params.thinking_token_budget,
        reasoning_start_token_ids=_encode_token_ids(tokenizer, reasoning_start),
        reasoning_end_token_ids=_encode_token_ids(tokenizer, reasoning_end),
    )

    if existing is None:
        sampling_params.logits_processor = processor
    elif isinstance(existing, list):
        existing.append(processor)
    else:
        sampling_params.logits_processor = _ChainedLogitsProcessor([existing, processor])


def _get_reasoning_boundary(parser: Any, boundary: str) -> str:
    if boundary == "start":
        candidates = ("reasoning_start", "CHANNEL_OPEN")
    else:
        candidates = ("reasoning_end", "CHANNEL_CLOSE")
    for attr in candidates:
        value = getattr(parser, attr, None)
        if value:
            return value
    raise ValueError(
        f"Reasoning parser {parser.__class__.__name__} does not define a "
        f"reasoning {boundary} string"
    )


def _encode_token_ids(tokenizer: Any, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _find_last_sequence_index(token_ids: List[int], sequence: List[int]) -> int:
    if not sequence or len(sequence) > len(token_ids):
        return -1
    for idx in range(len(token_ids) - len(sequence), -1, -1):
        if token_ids[idx : idx + len(sequence)] == sequence:
            return idx
    return -1


def _longest_suffix_prefix_len(token_ids: List[int], sequence: List[int]) -> int:
    max_len = min(len(token_ids), len(sequence) - 1)
    for length in range(max_len, 0, -1):
        if token_ids[-length:] == sequence[:length]:
            return length
    return 0
