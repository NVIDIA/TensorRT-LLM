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
"""Tests for the MuseSpark header/message reasoning format."""

import importlib
import random
import sys

import pytest

from tensorrt_llm.llmapi.reasoning_parser import MuseSparkReasoningParser, ReasoningParserFactory

pytestmark = pytest.mark.cpu_only

MS_S, MS_M, MS_EOM, MS_EOT = "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"
MS_TOOL_BODY = (
    '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)
MS_TOOL_BODY2 = (
    '<atem:function_calls>\n<atem:invoke name="get_time">\n'
    '<atem:parameter name="tz">UTC</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)


def musespark_make_parser():
    return ReasoningParserFactory.create_reasoning_parser("musespark")


def musespark_stream_chunks(parser, text, chunks):
    """Feed ``text`` in the given chunk sizes; return (content, reasoning)."""
    content, reasoning = [], []
    pos = 0
    for n in chunks:
        r = parser.parse_delta(text[pos : pos + n])
        content.append(r.content)
        reasoning.append(r.reasoning_content)
        pos += n
    if pos < len(text):
        r = parser.parse_delta(text[pos:])
        content.append(r.content)
        reasoning.append(r.reasoning_content)
    tail = parser.finish()
    content.append(tail.content)
    reasoning.append(tail.reasoning_content)
    return "".join(content), "".join(reasoning)


# -------------------------------------------------------------- non-streaming
MS_CASES = [
    # reasoning only (truncated / max_tokens, no end marker)
    (f" to=self{MS_M}Let me think", "", "Let me think"),
    # reasoning closed, nothing else
    (f" to=self{MS_M}Let me think{MS_EOM}", "", "Let me think"),
    # reasoning + final answer (reference docstring shape 1)
    (f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", "R"),
    # bare <|message|> = public CoT (reference shape 2)
    (f"{MS_M}R{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", "R"),
    # final answer only (shape 3)
    (f" to=user{MS_M}C{MS_EOT}", "C", ""),
    # <|end_of_text|> stop
    (f" to=user{MS_M}C<|end_of_text|>", "C", ""),
    # Guided decoding pins the whole generation to one value, so the model
    # writes no routing header and the body runs straight into the end marker.
    # Both markers have to keep it: a forced tool_choice decodes exactly this
    # shape and the body is the call's entire arguments value.
    ('{"a": 2, "b": 3}<|end_of_text|>', '{"a": 2, "b": 3}', ""),
    ('{"a": 2, "b": 3}' + MS_EOT, '{"a": 2, "b": 3}', ""),
    # commentary + final answer: commentary is content (deviation from ref)
    (
        f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=commentary{MS_M}Working on it.{MS_EOM}"
        f"{MS_S}assistant to=user{MS_M}Done.{MS_EOT}",
        "Working on it.Done.",
        "R",
    ),
    # commentary + tool call: header replayed for the tool parser
    (
        f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=commentary{MS_M}Checking.{MS_EOM}"
        f"{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}{MS_EOT}",
        f"Checking.{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}",
        "R",
    ),
    # direct tool call as the first message (header has no <|start|>assistant)
    (f" to=get_weather{MS_M}{MS_TOOL_BODY}{MS_EOT}", f" to=get_weather{MS_M}{MS_TOOL_BODY}", ""),
    # dotted recipient
    (
        f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=media.create_image{MS_M}{MS_TOOL_BODY}{MS_EOT}",
        f"{MS_S}assistant to=media.create_image{MS_M}{MS_TOOL_BODY}",
        "R",
    ),
    # parallel calls across messages: end markers dropped, headers kept
    (
        f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}{MS_EOM}"
        f"{MS_S}assistant to=get_time{MS_M}{MS_TOOL_BODY2}{MS_EOT}",
        f"{MS_S}assistant to=get_weather{MS_M}{MS_TOOL_BODY}"
        f"{MS_S}assistant to=get_time{MS_M}{MS_TOOL_BODY2}",
        "R",
    ),
    # body text containing "<|" that is not a marker, and '<' / '|' chars
    (
        f" to=self{MS_M}a < b | c <|not_a_token|> d{MS_EOM}{MS_S}assistant to=user{MS_M}"
        f"x <| y{MS_EOT}",
        "x <| y",
        "a < b | c <|not_a_token|> d",
    ),
    # developer / assistant recipients are dropped
    (f" to=developer{MS_M}secret{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", ""),
    # <|eot|> straight after <|eom|>
    (f" to=self{MS_M}R{MS_EOM}{MS_EOT}", "", "R"),
    # text that is not a header at all streams as content (reference default)
    ("Hello there", "Hello there", ""),
    # model skipped <|eom|> before the next header
    (f" to=self{MS_M}R{MS_S}assistant to=user{MS_M}C{MS_EOT}", "C", "R"),
    # empty
    ("", "", ""),
]


@pytest.mark.parametrize(("text", "content", "reasoning"), MS_CASES)
def test_musespark_parse(text, content, reasoning):
    r = musespark_make_parser().parse(text)
    assert (r.content, r.reasoning_content) == (content, reasoning)


def test_musespark_parse_resets_state():
    p = musespark_make_parser()
    p.parse(f" to=self{MS_M}R")
    r = p.parse(f" to=user{MS_M}C{MS_EOT}")
    assert (r.content, r.reasoning_content) == ("C", "")


# ----------------------------------------------------- stripped-token fallback
MS_STRIPPED_CASES = [
    # ref _split_on_bare_recipients: self + user
    (" to=selfLet me thinkassistant to=userThe answer", "The answer", "Let me think"),
    # user only
    (" to=userThe answer", "The answer", ""),
    # self only: never returned as content
    (" to=selfprivate thought", "", "private thought"),
    # no header at all: ordinary content
    ("plain text", "plain text", ""),
    # commentary survives as content
    (" to=selfRassistant to=commentaryWorkingassistant to=userDone", "WorkingDone", "R"),
]


@pytest.mark.parametrize(("text", "content", "reasoning"), MS_STRIPPED_CASES)
def test_musespark_parse_stripped_special_tokens(text, content, reasoning):
    r = musespark_make_parser().parse(text)
    assert (r.content, r.reasoning_content) == (content, reasoning)


@pytest.mark.parametrize(("text", "content", "reasoning"), MS_STRIPPED_CASES)
def test_musespark_stream_stripped_special_tokens(text, content, reasoning):
    # No <|message|> ever arrives: header-shaped text is held until finish().
    p = musespark_make_parser()
    got = musespark_stream_chunks(p, text, [1] * len(text))
    assert got == (content, reasoning)


def test_musespark_stream_stripped_never_leaks_reasoning_before_finish():
    p = musespark_make_parser()
    text = " to=selfLet me think about this carefully"
    for ch in text:
        r = p.parse_delta(ch)
        assert r.content == "" and r.reasoning_content == ""
    tail = p.finish()
    assert tail.content == ""
    assert tail.reasoning_content == "Let me think about this carefully"


# ------------------------------------------------------------------ streaming
# Delta tables mirror the reference implementation's streaming case table.
MS_DELTA_CASES = [
    # header buffering: nothing emitted until <|message|> resolves recipient
    ([" to", "=se", "lf", MS_M, "R"], ["", "", "", "", ""], ["", "", "", "", "R"]),
    # case 3: reasoning starts in this delta
    ([f" to=self{MS_M}Rea", "soning"], ["", ""], ["Rea", "soning"]),
    # case 1: reasoning ends in delta, content starts in the same delta
    (
        [f" to=self{MS_M}R", f"est{MS_EOM}{MS_S}assistant to=user{MS_M}Con", "tent", MS_EOT],
        ["", "Con", "tent", ""],
        ["R", "est", "", ""],
    ),
    # case 2: between reasoning end and content start, nothing emitted
    (
        [f" to=self{MS_M}R{MS_EOM}", f"{MS_S}assistant", " to=", "user", MS_M, "C", MS_EOT],
        ["", "", "", "", "", "C", ""],
        ["R", "", "", "", "", "", ""],
    ),
    # case 5/6: content starts in delta, continues, ends (case 4)
    (
        [f" to=user{MS_M}Hel", "lo", f" world{MS_EOT}"],  # codespell:ignore
        [
            "Hel",  # codespell:ignore
            "lo",  # codespell:ignore
            " world",
        ],
        ["", "", ""],
    ),
    # partial end marker held back, then released as content
    ([f" to=user{MS_M}a<|e", "x", "b"], ["a", "<|ex", "b"], ["", "", ""]),
    # partial end marker that completes into <|eot|>
    ([f" to=user{MS_M}a<|e", "ot|>"], ["a", ""], ["", ""]),  # codespell:ignore
    # bare <|message|> public CoT
    ([f"{MS_M}think", f"{MS_EOM}{MS_S}assistant to=user{MS_M}C{MS_EOT}"], ["", "C"], ["think", ""]),
    # tool handoff: header replayed once the recipient is known, body streams
    (
        [
            " to=get_we",
            f"ather{MS_M}<atem:",  # codespell:ignore
            "function_calls>",
        ],  # codespell:ignore
        ["", f" to=get_weather{MS_M}<atem:", "function_calls>"],
        ["", "", ""],
    ),
    # default: text that cannot be a header streams as content immediately
    (["Hello", " world"], ["Hello", " world"], ["", ""]),
    # developer channel suppressed
    ([f" to=developer{MS_M}secret{MS_EOM}{MS_S}assistant to=user{MS_M}ok{MS_EOT}"], ["ok"], [""]),
]


@pytest.mark.parametrize(("deltas", "content", "reasoning"), MS_DELTA_CASES)
def test_musespark_parse_delta(deltas, content, reasoning):
    p = musespark_make_parser()
    got_c, got_r = [], []
    for d in deltas:
        r = p.parse_delta(d)
        got_c.append(r.content)
        got_r.append(r.reasoning_content)
    assert got_c == content
    assert got_r == reasoning


def test_musespark_finish_flushes_truncated_reasoning():
    p = musespark_make_parser()
    p.parse_delta(f" to=self{MS_M}abc<|e")
    tail = p.finish()
    assert tail.reasoning_content == "<|e" and tail.content == ""


def test_musespark_finish_flushes_truncated_content():
    p = musespark_make_parser()
    assert p.parse_delta(f" to=user{MS_M}abc<").content == "abc"
    assert p.finish().content == "<"


def test_musespark_headerless_body_survives_end_marker():
    """A forced tool call's arguments are the whole response.

    ``tool_choice={"type":"function",...}`` constrains generation to the
    arguments value, so nothing that looks like a routing header is ever
    emitted and the body runs straight into ``<|end_of_text|>``. The header
    state must not clear its buffer on that marker, or the arguments are lost:
    the caller would see HTTP 200, an empty message and no tool call.

    Streaming routes the first chunk before the marker is in the buffer, so
    this also pins the streamed and non-streamed paths to the same answer.
    """
    arguments = '{"a": 2, "b": 3}'
    for marker in ("<|end_of_text|>", MS_EOT):
        result = musespark_make_parser().parse(arguments + marker)
        assert result.content == arguments
        assert result.reasoning_content == ""

        streamed = musespark_make_parser()
        got = musespark_stream_chunks(streamed, arguments + marker, [1] * len(arguments + marker))
        assert got == (arguments, "")


def test_musespark_end_marker_still_ends_a_completed_turn():
    """Keeping a headerless body must not resurrect text the parser drops.

    After a closed message the header state sees ``<|start|>assistant`` (or a
    fragment of it) followed by the end marker; that is a header remnant, not a
    body, and it stays dropped.
    """
    for head in ("", MS_S + "assistant", " to=us"):
        result = musespark_make_parser().parse(f" to=user{MS_M}C{MS_EOM}{head}{MS_EOT}")
        assert result.content == "C"
        assert result.reasoning_content == ""


def test_musespark_finish_drops_partial_header_after_message():
    p = musespark_make_parser()
    p.parse_delta(f" to=self{MS_M}R{MS_EOM}{MS_S}assistant to=us")
    tail = p.finish()
    assert tail == type(tail)()


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize(("text", "_c", "_r"), MS_CASES + MS_STRIPPED_CASES)
def test_musespark_stream_matches_parse(seed, text, _c, _r):
    rng = random.Random(seed)
    chunks = []
    remaining = len(text)
    while remaining > 0:
        n = rng.randint(1, 7)
        chunks.append(min(n, remaining))
        remaining -= chunks[-1]
    expected = musespark_make_parser().parse(text)
    got = musespark_stream_chunks(musespark_make_parser(), text, chunks)
    assert got == (expected.content, expected.reasoning_content)


class _CountingMuseSparkParser(MuseSparkReasoningParser):
    """Turns "the parser stopped consuming" from a hang into a failure.

    The bug these tests pin wedged a live server: the header and body states
    handed the same buffer back and forth, `_feed`'s loop never ended, and
    because that loop runs in the server's event loop the whole frontend
    stopped answering, including ``/health``. A plain assertion on the parsed
    text would reproduce it as a test run that never finishes, which is a much
    worse signal than a failed assertion.
    """

    STEP_LIMIT = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0

    def _step(self, content, reasoning):
        self.steps += 1
        if self.steps > self.STEP_LIMIT:
            raise AssertionError(
                f"parser made {self.steps} steps without finishing: state "
                f"{self._state!r}, buffer {self._buffer[:64]!r}"
            )
        return super()._step(content, reasoning)


@pytest.mark.parametrize(
    ("text", "content", "reasoning"),
    [
        (f"{MS_S}hello world", "hello world", ""),
        (f"{MS_S}xy<|end_of_text|>", "xy", ""),
        (f" to=self{MS_M}think{MS_S}zz", "zz", "think"),
        (f" to=user{MS_M}A{MS_S}not a header", "Anot a header", ""),
    ],
)
def test_musespark_stray_start_marker_terminates(text, content, reasoning):
    """``<|start|>`` that is not followed by a header must still be consumed.

    The body state leaves ``<|start|>`` in the buffer so the header regexes see
    the canonical ``<|start|>assistant to=`` form. When the header state then
    ruled the text out as a header it handed the buffer back unchanged, the
    body state found the marker at index 0, emitted nothing and handed it back
    again. Under concurrent load with ``--ignore-eos`` the model emits exactly
    this, and the server stopped serving.
    """
    p = _CountingMuseSparkParser()
    result = p.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning


def test_musespark_stray_start_marker_terminates_streamed():
    p = _CountingMuseSparkParser()
    got_c, got_r = [], []
    for delta in (" to=self", MS_M, "think", MS_S, "zz", "!"):
        r = p.parse_delta(delta)
        got_c.append(r.content)
        got_r.append(r.reasoning_content)
    tail = p.finish()
    assert "".join(got_r) == "think"
    assert "".join(got_c) + tail.content == "zz!"


def test_musespark_feed_is_bounded_even_if_a_step_stops_consuming():
    """The backstop, independent of the specific cycle above.

    ``_feed`` runs in the server's event loop, so a step sequence that stops
    consuming must degrade to a bad parse, never to a spin.
    """

    class _NeverConsumes(MuseSparkReasoningParser):
        def __init__(self):
            super().__init__()
            self.steps = 0

        def _step(self, content, reasoning):
            self.steps += 1
            return True  # always "more to do", never consumes

    p = _NeverConsumes()
    result = p.parse_delta("abcdefgh")
    assert result.content == "" and result.reasoning_content == ""
    assert p.steps <= len("abcdefgh") + 8
    assert p._buffer == "abcdefgh"


MS_FUZZ_PIECES = [
    MS_S,
    MS_M,
    MS_EOM,
    MS_EOT,
    "<|end_of_text|>",
    " to=self",
    " to=user",
    " to=commentary",
    " to=developer",
    " to=get_weather",
    "assistant",
    "hello",
    " ",
    "",
    "<|start",
    "to=",
    "<atem:function_calls>",
    '{"a": 2}',
    "\n",
    "<|",
]  # codespell:ignore


@pytest.mark.parametrize("seed", range(300))
def test_musespark_always_consumes(seed):
    """No input makes the parser stop consuming, batched or streamed.

    The specific cycle above is pinned by name; this pins the property behind
    it. The parser runs inside the server's event loop, so an input it never
    finishes with is not a bad answer, it is a dead server, and marker soup is
    exactly what unconstrained decoding produces. The step counter turns a
    non-terminating parse into a failure instead of a test run that never
    returns.
    """
    rng = random.Random(seed)
    text = "".join(rng.choice(MS_FUZZ_PIECES) for _ in range(rng.randint(1, 14)))

    _CountingMuseSparkParser().parse(text)

    streamed = _CountingMuseSparkParser()
    pos = 0
    while pos < len(text):
        n = rng.randint(1, 6)
        streamed.parse_delta(text[pos : pos + n])
        pos += n
    streamed.finish()


def test_musespark_registry_and_flag():
    assert "musespark" in ReasoningParserFactory.keys()
    assert ReasoningParserFactory.needs_raw_special_tokens("musespark")
    assert MuseSparkReasoningParser.needs_raw_special_tokens is True
    assert isinstance(musespark_make_parser(), MuseSparkReasoningParser)


def test_importing_the_model_module_first_registers_the_parser(monkeypatch):
    """Importing the parser's own module first still registers it.

    The registry imports the model module and the model module imports the
    registry, so whichever is imported second sees the other half-initialized.
    """
    import tensorrt_llm.llmapi
    import tensorrt_llm.llmapi.reasoning

    for name in ("tensorrt_llm.llmapi.reasoning_parser", "tensorrt_llm.llmapi.reasoning.musespark"):
        monkeypatch.delitem(sys.modules, name)
    # The parent packages cache the submodules as attributes; restore those too.
    monkeypatch.setattr(
        tensorrt_llm.llmapi, "reasoning_parser", tensorrt_llm.llmapi.reasoning_parser
    )
    monkeypatch.setattr(
        tensorrt_llm.llmapi.reasoning, "musespark", tensorrt_llm.llmapi.reasoning.musespark
    )

    model_module = importlib.import_module("tensorrt_llm.llmapi.reasoning.musespark")
    registry = sys.modules["tensorrt_llm.llmapi.reasoning_parser"]
    parser_cls = model_module.MuseSparkReasoningParser
    assert registry.ReasoningParserFactory._parsers["musespark"][0] is parser_cls
    assert registry.MuseSparkReasoningParser is parser_cls
