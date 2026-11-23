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
"""Prompt-tail transport for disaggregated generation-only requests.

A generation-only request reaches its generation worker together with the
context worker's KV cache. The worker reads the prompt *length* everywhere but
the prompt *ids* only near its end (Engram lookback, stop-sequence tails). With
``gen_prompt_tail_tokens=N`` a generation server's frontend sends the worker
only the last ``N`` ids of an eligible request and the number it left out; the
owning rank restores a full-length prompt whose left-out prefix holds zeros.
Requests that read full prompts keep them (``request_trim_blocker``); engine
features that do are refused by the config.
"""

from typing import Optional

from ..disaggregated_params import DisaggScheduleStyle
from ..logger import logger


def _is_set(value, neutral) -> bool:
    return value is not None and value != neutral


def request_trim_blocker(request, tail: int) -> Optional[str]:
    """Why ``request`` (a ``GenerationRequest``) must keep its full prompt ids, or None."""
    params = request.disaggregated_params
    if params is None or params.request_type != "generation_only":
        return "not a disaggregated generation-only request"
    if (
        params.schedule_style is not None
        and params.schedule_style != DisaggScheduleStyle.CONTEXT_FIRST
    ):
        return "a generation-first disaggregated request"
    if params.remote_tail_start is not None:
        return "a remote-tail request"
    if (
        params.multimodal_positions is not None
        or params.multimodal_embedding_handles is not None
        or params.multimodal_hashes is not None
    ):
        return "a multimodal request"
    if request.multimodal_params is not None and request.multimodal_params.has_content():
        return "a multimodal request"
    if request.encoder_input_token_ids is not None:
        return "an encoder-decoder request"
    sp = request.sampling_params
    if (
        _is_set(sp.repetition_penalty, 1.0)
        or _is_set(sp.presence_penalty, 0.0)
        or _is_set(sp.frequency_penalty, 0.0)
    ):
        return "repetition / presence / frequency penalties"
    if sp.prompt_logprobs is not None:
        return "prompt logprobs"
    if sp.logits_processor is not None or sp.apply_batched_logits_processor:
        return "logits processors"
    if sp.no_repeat_ngram_size:
        return "no_repeat_ngram_size"
    if sp.bad is not None or sp.bad_token_ids:
        return "bad words"
    if sp.return_context_logits:
        return "context logits"
    if not sp.exclude_input_from_output:
        return "exclude_input_from_output=False"
    if sp.use_beam_search or sp.n > 1 or (sp.best_of or 1) > 1:
        return "beam search / multiple sequences"
    if sp._decoder_output_token_prefix:
        return "a decoder output-token prefix"
    if sp.stop is not None and sp._stop_word_ids is None:
        return "untokenized stop strings"
    if any(len(word) > tail for word in sp._get_stop_words()):
        return "a stop sequence longer than the prompt tail"
    return None


class PromptTailPolicy:
    """A frontend's per-request trimming decision, reporting each outcome once in the log."""

    def __init__(self, tail: int):
        self.tail = tail
        self._reported = set()

    def __call__(self, request) -> int:
        """The number of trailing prompt ids to send for ``request``, or 0 to send all."""
        if request.prompt_len <= self.tail:
            return 0
        blocker = request_trim_blocker(request, self.tail)
        if blocker is None:
            self._report(
                "trimmed",
                f"prompt tail trimming: first trimmed request sends {self.tail} of "
                f"{request.prompt_len} prompt ids",
            )
            return self.tail
        self._report(
            blocker, f"prompt tail trimming: requests with {blocker} keep their full prompt"
        )
        return 0

    def _report(self, key: str, message: str) -> None:
        if key not in self._reported:
            self._reported.add(key)
            logger.info(message)
