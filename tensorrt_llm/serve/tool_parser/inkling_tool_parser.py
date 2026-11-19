# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/inkling_detector.py
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
import json
import re
from collections.abc import Mapping
from typing import List, Optional

from partial_json_parser.core.options import Allow

from tensorrt_llm.llmapi.inkling_tokens import (
    INKLING_CONTROL_TOKENS,
    INKLING_END_MESSAGE,
    INKLING_INVOKE_TOOL_JSON,
    INKLING_MESSAGE_MODEL,
)
from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, StructureInfo, ToolCallItem, _GetInfoFunc
from .utils import is_complete_json, partial_json_loads


class InklingToolParser(BaseToolParser):
    """Tool parser for Inkling's typed-content invocation blocks.

    The model emits a call as

        <|message_model|>name<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>

    so the payload is JSON but the framing is special tokens, and the tool name
    appears twice: once in the message header and once inside the payload. The
    header is advisory -- a mismatch means the two halves disagree about which
    tool is being called, which is not something to guess at, so the call is
    dropped rather than executed under either name.
    """

    # The delimiters are registered special tokens, so the serving layer has to
    # keep them in the decoded text or there is nothing left to parse.
    needs_raw_special_tokens = True

    def __init__(self):
        super().__init__()
        self.bot_token = INKLING_INVOKE_TOOL_JSON
        self.eot_token = INKLING_END_MESSAGE
        self.tool_call_regex = re.compile(
            re.escape(self.bot_token) + r"\s*(.*?)\s*" + re.escape(self.eot_token),
            re.DOTALL,
        )
        self._current_header_name: Optional[str] = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=self._clean_normal_text(text))

        try:
            calls: List[ToolCallItem] = []
            for match in self.tool_call_regex.finditer(text):
                # A malformed invocation is dropped on its own; the calls around it stand.
                try:
                    payload = json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    logger.warning(f"Inkling tool call payload is not JSON: {match.group(1)!r}")
                    continue
                if not isinstance(payload, Mapping):
                    logger.warning(f"Invalid Inkling tool call payload: {payload}")
                    continue
                _, header_name = self._split_trailing_tool_header(text[: match.start()])
                call = self._tool_call_item(payload, tools, len(calls), header_name=header_name)
                if call is not None:
                    calls.append(call)

            if not calls:
                # Every candidate was rejected (bad payload, or the header and
                # the payload named different tools). Follow the contract the
                # other parsers keep: normal_text is only what came BEFORE the
                # tool marker. The rejected region is dropped, never handed back
                # as visible content -- a half-parsed invocation is not an answer.
                prefix, _ = self._split_trailing_tool_header(text[: text.find(self.bot_token)])
                return StreamingParseResult(normal_text=self._clean_normal_text(prefix))

            normal_prefix, _ = self._split_trailing_tool_header(text[: text.find(self.bot_token)])
            return StreamingParseResult(
                normal_text=self._clean_normal_text(normal_prefix), calls=calls
            )
        except Exception as exc:
            logger.error(f"Error in Inkling detect_and_parse: {exc}")
            prefix = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=self._clean_normal_text(prefix))

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Report every complete invocation the buffer holds, and the text around them."""
        self._buffer += new_text
        normal_text = ""
        calls: List[ToolCallItem] = []
        while True:
            text, step_calls, consumed = self._parse_step(tools)
            normal_text += text
            calls.extend(step_calls)
            if not consumed:
                return StreamingParseResult(normal_text=normal_text, calls=calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Also release a held-back partial control token: the stream ended, so it is text.

        A trailing tool header or an unfinished invocation is not text and is dropped.
        """
        result = super().finish(tools)
        tail, self._buffer = self._buffer, ""
        if tail and self.bot_token not in tail and self._pending_tool_header_start(tail) is None:
            return StreamingParseResult(
                normal_text=result.normal_text + self._clean_normal_text(tail), calls=result.calls
            )
        return result

    def _parse_step(self, tools: List[Tool]) -> tuple:
        """Parse the buffer up to the end of its first invocation.

        Returns the normal text and the call deltas it yields, and whether a complete invocation
        was consumed (accepted or dropped), in which case the rest of the buffer may hold more.
        """
        current_text = self._buffer

        if self.bot_token not in current_text:
            header_start = self._pending_tool_header_start(current_text)
            if header_start is not None:
                # A `<|message_model|>name` header may still be a tool call in
                # the making; hold it back rather than emit the name as text.
                safe_text = current_text[:header_start]
                self._buffer = current_text[header_start:]
                return self._clean_normal_text(safe_text), [], False
            # Hold back a partial prefix of ANY token _clean_normal_text strips:
            # emitting a split control token leaks its first half as visible
            # text, where the completed token would have been removed.
            partial_len = max(
                self._ends_with_partial_token(current_text, token)
                for token in INKLING_CONTROL_TOKENS
            )
            if partial_len:
                safe_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
            else:
                safe_text = current_text
                self._buffer = ""
            return self._clean_normal_text(safe_text), [], False

        bot_pos = current_text.find(self.bot_token)
        pending_normal = ""
        if bot_pos > 0:
            normal_text, self._current_header_name = self._split_trailing_tool_header(
                current_text[:bot_pos]
            )
            self._buffer = current_text[bot_pos:]
            pending_normal = self._clean_normal_text(normal_text)
            current_text = self._buffer
            # The text before the call and the call itself are reported in the same pass (the
            # reference implementation returns the text first and leaves the call for the next
            # delta), so a delta that carries both is reported whole.

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        start_idx = len(self.bot_token)
        while start_idx < len(current_text) and current_text[start_idx].isspace():
            start_idx += 1

        try:
            # Partial strings are not needed: a call is reported only once its payload is whole.
            payload, end_idx = partial_json_loads(current_text[start_idx:], Allow.ALL & ~Allow.STR)
        except (ValueError, AssertionError, IndexError):
            # partial_json_parser reports malformed input as any of these (its own and the json
            # module's decode errors are ValueErrors; some shapes fail an internal assertion).
            # Malformed for good once its end token has arrived: drop just this invocation.
            end_pos = current_text.find(self.eot_token, start_idx)
            if end_pos < 0:
                return pending_normal, [], False
            payload_text = current_text[start_idx:end_pos]
            logger.warning(f"Inkling tool call payload is not JSON: {payload_text!r}")
            self._buffer = current_text[end_pos + len(self.eot_token) :]
            self._current_header_name = None
            return pending_normal, [], True
        if not is_complete_json(current_text[start_idx : start_idx + end_idx]):
            return pending_normal, [], False

        # The payload is whole: report the call, or drop this invocation only. Announcing the name
        # earlier would leave a call the client already saw when the rest of the payload fails.
        if isinstance(payload, Mapping):
            call = self._tool_call_item(
                payload, tools, self.current_tool_id, header_name=self._current_header_name
            )
        else:
            logger.warning(f"Invalid Inkling tool call payload: {payload}")
            call = None
        self._buffer = self._remaining_after_call(current_text, start_idx + end_idx)
        self._current_header_name = None
        if call is None:
            return pending_normal, [], True

        self._ensure_current_tool()
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": call.name,
            "arguments": json.loads(call.parameters),
        }
        self.streamed_args_for_tool[self.current_tool_id] = call.parameters
        calls = [
            ToolCallItem(tool_index=self.current_tool_id, name=call.name, parameters=""),
            ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=call.parameters),
        ]
        self.current_tool_id += 1
        return pending_normal, calls, True

    def structure_info(self) -> _GetInfoFunc:
        def info(name: str) -> StructureInfo:
            trigger = f"{INKLING_MESSAGE_MODEL}{name}{self.bot_token}"
            return StructureInfo(
                begin=f'{trigger}{{"name":"{name}","args":',
                end=f"}}{self.eot_token}",
                trigger=trigger,
            )

        return info

    def _tool_call_item(
        self,
        payload: Mapping,
        tools: List[Tool],
        call_index: int,
        *,
        header_name: Optional[str] = None,
    ) -> Optional[ToolCallItem]:
        name = payload.get("name")
        args = payload.get("args")
        if not isinstance(name, str) or not isinstance(args, Mapping):
            logger.warning(f"Invalid Inkling tool call payload: {payload}")
            return None
        if header_name is not None and header_name != name:
            logger.warning(
                f"Inkling tool header {header_name!r} does not match payload name {name!r}"
            )
            return None

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if name not in self._tool_indices:
            # Surface it anyway, which is what OpenAI does for a hallucinated
            # tool: the harness gets a structured call, returns a tool error and
            # the model can correct itself. Dropping it instead degrades the
            # invocation into terminal answer text.
            logger.warning(f"Surfacing Inkling call to undeclared tool: {name}")

        return ToolCallItem(
            tool_index=call_index, name=name, parameters=json.dumps(args, ensure_ascii=False)
        )

    def _ensure_current_tool(self) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

    def _split_trailing_tool_header(self, text: str) -> tuple:
        message_pos = self._pending_tool_header_start(text)
        if message_pos is None:
            return text, None
        header = text[message_pos + len(INKLING_MESSAGE_MODEL) :]
        return text[:message_pos], header.strip() or None

    def _pending_tool_header_start(self, text: str) -> Optional[int]:
        """Start of a trailing, still-forming tool-call header.

        That is a `<|message_model|>` whose header text carries no complete
        control token yet, so it may still turn into a tool call.
        """
        message_pos = text.rfind(INKLING_MESSAGE_MODEL)
        if message_pos < 0:
            return None
        header = text[message_pos + len(INKLING_MESSAGE_MODEL) :]
        if any(token in header for token in INKLING_CONTROL_TOKENS):
            return None
        return message_pos

    @staticmethod
    def _ends_with_partial_token(text: str, token: str) -> int:
        """Length of the longest proper prefix of `token` ending `text`."""
        for length in range(min(len(text), len(token) - 1), 0, -1):
            if text.endswith(token[:length]):
                return length
        return 0

    def _remaining_after_call(self, text: str, end_idx: int) -> str:
        remaining = text[end_idx:]
        if remaining.startswith(self.eot_token):
            return remaining[len(self.eot_token) :]
        if self.eot_token in remaining:
            return remaining.split(self.eot_token, 1)[1]
        return remaining

    def _clean_normal_text(self, text: str) -> str:
        for token in INKLING_CONTROL_TOKENS:
            text = text.replace(token, "")
        return text
