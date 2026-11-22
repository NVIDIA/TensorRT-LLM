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

"""Host-side speculative widths used by scheduling and verification."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...llmapi.llm_args import DecodingBaseConfig
    from ..pyexecutor.llm_request import LlmRequest
    from ..pyexecutor.scheduler import ScheduledRequests


def uses_fixed_tree(config: DecodingBaseConfig) -> bool:
    return not config.is_linear_tree or (
        getattr(config, "use_dynamic_tree", False) and config.dynamic_tree_max_topK > 1
    )


def _token_room(request: LlmRequest, max_seq_len: int, cp_size: int) -> tuple[int, int]:
    current = request.get_num_tokens(0)
    generated = current - request.py_orig_prompt_len
    if cp_size > 1:
        # Disaggregated Helix ranks have different local prompt slices.
        current = request.total_input_len_cp // cp_size + generated
    return max_seq_len - current, request.py_max_new_tokens - generated


def remaining_output_tokens(request: LlmRequest, max_seq_len: int, cp_size: int = 1) -> int:
    return min(_token_room(request, max_seq_len, cp_size))


def request_draft_budget(
    request: LlmRequest, max_draft_len: int, max_seq_len: int, cp_size: int, *, is_pard: bool
) -> int:
    if request.is_dummy:
        return max_draft_len
    seq_room, gen_room = _token_room(request, max_seq_len, cp_size)
    # PARD submits K proposals and K-1 mask rows along with the target token.
    seq_limit = seq_room // 2 if is_pard else seq_room - 1
    return max(0, min(max_draft_len, seq_limit, gen_room - 1))


def batch_draft_budget(
    requests: Iterable[LlmRequest], max_draft_len: int, max_seq_len: int, cp_size: int, *, is_pard: bool
) -> int:
    """A uniform verify width that fits every real request in the batch."""
    return min(
        (request_draft_budget(r, max_draft_len, max_seq_len, cp_size, is_pard=is_pard) for r in requests),
        default=max_draft_len,
    )


def resize_draft_tokens(request: LlmRequest, draft_len: int, *, is_pard: bool, rejection_on: bool) -> None:
    """Keep the proposal prefix and pad unused input positions with token zero."""
    proposals = request.py_draft_tokens or []
    width = max(0, 2 * draft_len - 1) if is_pard else draft_len
    if rejection_on and not proposals:
        request.py_needs_onehot_draft_probs = True
    if len(proposals) == width:
        return
    if is_pard:
        proposals = proposals[:min((len(proposals) + 1) // 2, draft_len)]
    request.py_draft_tokens = proposals[:width] + [0] * max(0, width - len(proposals))


def get_draft_len_for_batch_size(
    draft_len_schedule: dict[int, int] | None, batch_size: int, max_draft_len: int
) -> int:
    """Look up the first inclusive batch-size threshold, with zero beyond the schedule."""
    if draft_len_schedule is None:
        return max_draft_len
    thresholds = list(draft_len_schedule)
    index = bisect_left(thresholds, batch_size)
    return draft_len_schedule[thresholds[index]] if index < len(thresholds) else 0


def resolve_draft_len(spec_config: DecodingBaseConfig | None,
                      scheduled_batch: ScheduledRequests,
                      *,
                      max_draft_len: int,
                      static_draft_len: int,
                      draft_len: int | None = None,
                      draft_budget: int | None = None,
                      speculation_permanently_disabled: bool = False) -> int:
    """Synchronize this batch's draft buffers and return its draft length.

    Dynamic and explicit lengths pad or truncate generation-request buffers to
    a uniform width, as required by CUDA graph replay and the attention kernel.
    Static decoding preserves the drafter's proposals instead.
    """
    if speculation_permanently_disabled:
        for request in scheduled_batch.generation_requests:
            request.py_draft_tokens = []
        return 0

    is_static = False
    if draft_len is None:
        if (spec_config is not None
                and spec_config.draft_len_schedule is not None
                and spec_config.spec_dec_mode.support_dynamic_draft_len()):
            draft_len = get_draft_len_for_batch_size(
                spec_config.draft_len_schedule, scheduled_batch.batch_size,
                max_draft_len)
        elif draft_budget is None:
            # Static decoding preserves the proposals produced by the drafter,
            # including requests intentionally entering with no draft tokens.
            return static_draft_len
        else:
            draft_len = static_draft_len
            is_static = True

    if draft_budget is not None:
        draft_len = min(draft_len, draft_budget)

    is_pard = spec_config is not None and spec_config.spec_dec_mode.is_pard()
    rejection_on = getattr(spec_config, "use_rejection_sampling", False)
    for request in scheduled_batch.generation_requests:
        if is_static and not request.py_draft_tokens:
            continue
        resize_draft_tokens(request, draft_len, is_pard=is_pard, rejection_on=rejection_on)

    return draft_len
