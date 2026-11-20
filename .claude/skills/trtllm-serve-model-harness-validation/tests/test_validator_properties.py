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
"""Properties of validator rules, over generated inputs (CPU only, needs `hypothesis`).

- A re-ask (C-03) is a later request that appends blocks to the last user message and changes
  nothing else. Every such request is a re-ask, and no other change is.
- A declared error wording (C-16) matches a message exactly when its parts, split at "...",
  appear in it in order, in any case.
- A reply joins its raw record (C-13) by the first of its ids that names exactly one record.
  Failing that, an engine id that names several joins the one record not ruled out by evidence
  (times outside the reply's window, another output count), and only when that record's times
  are known; a disagg_request_id that names several joins nothing. The window holds every
  instant the recorded stamps can name.
- A degenerate run (M-02) is counted wherever it shows, in what was delivered or in the raw
  token ids; it fails R-04 only when it reached the client: delivered, or in the generation of a
  reply delivered as an answer (completed, or cut at its budget).
- A raw record of a generation the server stopped (partial, aborted, or with an end reason)
  holds only what the model wrote until then: what was delivered must be a prefix of it
  (RAW-05). A finished one is compared whole.
- Inventory is BLOCKED exactly when the replies' build_id was not checked: a declared revision
  that some reply does not carry, or replies that carry one when none was declared.
- A harness that reads over-length by text (C-10 `any_error`, pi) recovers from an error exactly
  when its text (status, code, type and message, in any case) holds one of its patterns and none
  of its exclusions; a reply that is not an error is never read as one.
- A resend is ours (H-02) exactly after an over-length answer or a client error other than
  capacity (429); after a capacity answer, a server error or a failed stream it is a retry.
- The request that sends a turn's calls back may bill less (USG-08) only when it declares strictly
  fewer of the turn's tools, and only by the UTF-8 bytes of the declarations it does not carry as
  they were, plus the tools preamble when none is left. A namespace's tools each carry its fields.
- GLM's raw view reads an argument value back exactly when it holds none of the format's argument
  markup. When it holds any, the view flags the model (M-07) or reads other arguments: markup
  that closes the value and writes whole arguments after it is, in the text, those arguments.

Run with pytest, or as a script, which is how the mutation gate (scripts/mutate.py; the method is
references/verify-logic.md) runs it. From the repository root:

    python3 .claude/skills/trtllm-serve-model-harness-validation/scripts/mutate.py \
        .claude/skills/trtllm-serve-model-harness-validation/tests/validator_properties.mutations \
        --python <validator venv python, with hypothesis> --runner <absolute path of this file>
"""

import copy
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SKILL, "scripts", "validate"))

import controls  # noqa: E402
import native  # noqa: E402
import pipeline  # noqa: E402
import profiles  # noqa: E402
import turnlib  # noqa: E402
import verdict  # noqa: E402

SETTINGS = settings(max_examples=300, deadline=None)
TEXT = st.text(alphabet="abcxyz .:'", max_size=12)
BLOCK = st.one_of(
    st.builds(lambda t: {"type": "text", "text": t}, TEXT),
    st.builds(lambda i, t: {"type": "tool_result", "tool_use_id": i, "content": t},
              st.sampled_from(["toolu_1", "toolu_2"]), TEXT),
)  # fmt: skip
OTHER = st.one_of(
    st.builds(
        lambda b: {"role": "assistant", "content": b}, st.lists(BLOCK, min_size=1, max_size=2)
    ),
    st.builds(lambda t: {"role": "system", "content": t}, TEXT),
)


@st.composite
def conversation(draw):
    """A messages body whose last user message carries a list of blocks."""
    before = draw(st.lists(st.one_of(OTHER, st.builds(lambda t: {"role": "user", "content": t}, TEXT)),
                           max_size=4))  # fmt: skip
    user = {"role": "user", "content": draw(st.lists(BLOCK, min_size=1, max_size=3))}
    after = draw(st.lists(OTHER, max_size=2))  # e.g. Claude Code's mid-conversation system message
    return {"model": "m", "max_tokens": 64, "stream": True, "messages": before + [user] + after}


def _last_user(body):
    return max(i for i, m in enumerate(body["messages"]) if m["role"] == "user")


def _reasked(first, later, later_first=False):
    """Whether `later`, answered, re-asks `first` (orders: 1 then 2, or swapped)."""
    a, b = pipeline._reask_key(first), pipeline._reask_key(later)
    order_a, order_b = (2, 1) if later_first else (1, 2)
    turn = {"thread": "th", "order": order_a, "reask": a}
    reasks = {("th", b[0]): [(order_b, tuple(b[1]))]}
    return pipeline.Pipeline._reasked(turn, reasks)


@SETTINGS
@given(conversation(), st.lists(BLOCK, min_size=1, max_size=2))
def test_appending_to_the_last_user_message_is_a_reask(body, appended):
    again = copy.deepcopy(body)
    again["messages"][_last_user(body)]["content"] += appended
    assert _reasked(body, again)


@SETTINGS
@given(conversation(), st.lists(BLOCK, min_size=1, max_size=2))
def test_an_earlier_request_does_not_reask_a_later_one(body, appended):
    again = copy.deepcopy(body)
    again["messages"][_last_user(body)]["content"] += appended
    assert not _reasked(body, again, later_first=True)


@SETTINGS
@given(conversation())
def test_the_same_request_is_not_a_reask(body):
    assert not _reasked(body, copy.deepcopy(body))


@SETTINGS
@given(conversation(), st.data())
def test_changing_any_other_message_is_not_a_reask(body, data):
    last = _last_user(body)
    others = [i for i in range(len(body["messages"])) if i != last]
    assume(others)
    i = data.draw(st.sampled_from(others))
    again = copy.deepcopy(body)
    again["messages"][i] = data.draw(OTHER.filter(lambda m: m != body["messages"][i]))
    again["messages"][last]["content"] += [{"type": "text", "text": "Continue."}]
    assert not _reasked(body, again)


@SETTINGS
@given(conversation(), st.sampled_from([("model", "other"), ("max_tokens", 32), ("stream", False),
                                        ("thinking", {"type": "adaptive"})]))  # fmt: skip
def test_changing_a_request_field_is_not_a_reask(body, change):
    again = copy.deepcopy(body)
    again[change[0]] = change[1]
    again["messages"][_last_user(body)]["content"] += [{"type": "text", "text": "Continue."}]
    assert not _reasked(body, again)


@SETTINGS
@given(conversation(), BLOCK, st.data())
def test_rewriting_the_last_user_message_is_not_a_reask(body, replacement, data):
    last = _last_user(body)
    blocks = body["messages"][last]["content"]
    j = data.draw(st.integers(0, len(blocks) - 1))
    assume(replacement != blocks[j])
    again = copy.deepcopy(body)
    again["messages"][last]["content"][j] = replacement
    again["messages"][last]["content"] += [{"type": "text", "text": "Continue."}]
    assert not _reasked(body, again)


PART = st.text(alphabet="abcdefgh'_:", min_size=1, max_size=6).filter(lambda p: p.strip() == p)


@SETTINGS
@given(
    st.lists(PART, min_size=1, max_size=3), st.lists(TEXT, min_size=4, max_size=4), st.booleans()
)
def test_a_message_holding_the_parts_in_order_matches(parts, fillers, upper):
    message = fillers[0] + "".join(p + f for p, f in zip(parts, fillers[1:] + [""] * 3))
    message = message.upper() if upper else message
    assert pipeline.wording_matches(" ... ".join(parts), message)


@SETTINGS
@given(st.lists(PART, min_size=1, max_size=3), st.data())
def test_a_message_missing_a_part_does_not_match(parts, data):
    k = data.draw(st.integers(0, len(parts) - 1))
    message = " ".join(p for i, p in enumerate(parts) if i != k)
    assume(parts[k].lower() not in message.lower())
    assert not pipeline.wording_matches(" ... ".join(parts), message)


@SETTINGS
@given(PART, PART)
def test_parts_out_of_order_do_not_match(first, second):
    assume(first.lower() not in second.lower() and second.lower() not in first.lower())
    assert not pipeline.wording_matches(f"{first} ... {second}", f"{second} {first}")


@SETTINGS
@given(TEXT)
def test_a_wording_without_parts_matches_nothing(message):
    assert not pipeline.wording_matches(" ... ", message)


# Raw join (C-13). Times are whole quarter seconds, so every bound is exact in binary.
SKEW = pipeline.RAW_CLOCK_SKEW_S
QUARTERS = st.integers(0, 480).map(lambda q: q / 4)
STAMP = st.one_of(st.none(), st.builds(lambda at, whole: (at, at + (1.0 if whole else 0.0)),
                                       QUARTERS, st.booleans()))  # fmt: skip
COUNT = st.one_of(st.none(), st.integers(0, 3))
READING = st.fixed_dictionaries({"started_at": STAMP, "finished_at": STAMP, "output_tokens": COUNT})
WINDOW = st.one_of(st.none(), st.builds(lambda at, length: (at, at + length), QUARTERS, QUARTERS))
MOMENT = st.datetimes(min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 31),
                      timezones=st.just(timezone.utc))  # fmt: skip


def _ruled_out(reading, window, reported):
    """Evidence that a raw record is not this reply's generation.

    A known stamp wholly outside the window widened by the skew, or a known output count other
    than the reply's.
    """
    low, high = window[0] - SKEW, window[1] + SKEW
    if any(s is not None and (s[1] < low or s[0] > high)
           for s in (reading["started_at"], reading["finished_at"])):  # fmt: skip
        return True
    count = reading["output_tokens"]
    return reported is not None and count is not None and count != reported


@SETTINGS
@given(st.lists(READING, min_size=2, max_size=4), WINDOW, COUNT)
def test_a_resolved_record_fits_the_reply(readings, window, reported):
    joined = pipeline.fitting_raw(readings, window, reported)
    if joined is not None:
        assert window is not None and any(joined is r for r in readings)
        assert joined["started_at"] is not None and joined["finished_at"] is not None
        assert not _ruled_out(joined, window, reported)


@SETTINGS
@given(st.lists(READING, min_size=2, max_size=4), WINDOW, COUNT)
# The boundary, every run: two records fit, the first with both times known; neither is joined.
@example([{"started_at": (10.0, 10.0), "finished_at": (11.0, 11.0), "output_tokens": None},
          {"started_at": (10.5, 10.5), "finished_at": (11.5, 11.5), "output_tokens": None}],
         (10.0, 12.0), None)  # fmt: skip
def test_a_resolved_record_is_the_only_one_not_ruled_out(readings, window, reported):
    joined = pipeline.fitting_raw(readings, window, reported)
    if joined is not None:
        assert all(_ruled_out(r, window, reported) for r in readings if r is not joined)


@st.composite
def one_left(draw):
    """Records under one engine id of which exactly one, with known times, is not ruled out."""
    window, reported = draw(WINDOW.filter(bool)), draw(COUNT)
    low, high = window[0] - SKEW, window[1] + SKEW

    def inside():  # a stamp that can name an instant in the widened window, edges included
        at = draw(st.integers(math.ceil(low * 4), math.floor(high * 4))) / 4
        return at, at + (1.0 if draw(st.booleans()) else 0.0)

    count = draw(COUNT if reported is None else st.sampled_from([None, reported]))
    fit = {"started_at": inside(), "finished_at": inside(), "output_tokens": count}
    others = draw(st.lists(READING, min_size=1, max_size=3))
    for other in others:
        if not _ruled_out(other, window, reported):  # give it evidence against it
            evidence = draw(
                st.sampled_from(["early", "late"] + (["count"] if reported is not None else []))
            )
            if evidence == "early":
                other["started_at"] = (low - 1.25, low - 0.25)
            elif evidence == "late":
                other["finished_at"] = (high + 0.25, high + 0.25)
            else:
                other["output_tokens"] = reported + 1
    at = draw(st.integers(0, len(others)))
    return others[:at] + [fit] + others[at:], fit, window, reported


@SETTINGS
@given(one_left())
def test_the_one_record_not_ruled_out_is_joined(case):
    readings, fit, window, reported = case
    assert pipeline.fitting_raw(readings, window, reported) is fit


@SETTINGS
@given(st.lists(READING, min_size=2, max_size=4), COUNT)
def test_without_a_window_no_record_is_resolved(readings, reported):
    assert pipeline.fitting_raw(readings, None, reported) is None


KEY = st.sampled_from(["r:5", "r:7", "d:5", "d:9"])
INDEX = st.dictionaries(KEY, st.lists(READING, min_size=1, max_size=3))


@st.composite
def keys_naming_one(draw):
    """An index and the keys a reply carries, one of which names exactly one record."""
    index = draw(INDEX)
    key = draw(KEY)
    index[key] = [draw(READING)]
    keys = draw(st.lists(KEY, max_size=3))
    keys.insert(draw(st.integers(0, len(keys))), key)
    return index, keys


@SETTINGS
@given(keys_naming_one(), WINDOW, COUNT)
def test_an_id_naming_one_record_joins_it_whatever_its_times(case, window, reported):
    index, keys = case
    unique = [k for k in keys if len(index.get(k, ())) == 1]
    joined, how = pipeline.join_raw(index, keys, window, reported)
    assert how == "unique" and joined is index[unique[0]][0]


@SETTINGS
@given(one_left(), st.sampled_from(["r:5", "r:7"]))
def test_an_engine_id_naming_several_records_joins_the_one_left(case, key):
    readings, fit, window, reported = case
    joined, how = pipeline.join_raw({key: readings}, [key], window, reported)
    assert how == "resolved" and joined is fit


@SETTINGS
@given(one_left(), st.sampled_from(["d:5", "d:9"]))
def test_a_disagg_id_naming_several_records_joins_nothing(case, key):
    readings, _, window, reported = case
    assert pipeline.join_raw({key: readings}, [key], window, reported) == (None, "ambiguous")


@SETTINGS
@given(INDEX, st.lists(KEY, max_size=4), WINDOW, COUNT)
def test_only_an_engine_id_naming_several_records_is_resolved(index, keys, window, reported):
    assume(not [k for k in keys if len(index.get(k, ())) == 1])
    joined, how = pipeline.join_raw(index, keys, window, reported)
    fits = [pipeline.fitting_raw(index[k], window, reported) for k in keys
            if k in index and k.startswith("r:")]  # fmt: skip
    fits = [f for f in fits if f is not None]
    if fits:
        assert how == "resolved" and joined is fits[0]
    else:
        assert joined is None
        assert how == ("ambiguous" if any(k in index for k in keys) else None)


def _recorder_stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


@SETTINGS
@given(MOMENT)
def test_a_stamp_names_the_instant_it_was_written_at(moment):
    at = moment.timestamp()
    for stamp in (_recorder_stamp(moment), moment.isoformat(),
                  moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")):  # fmt: skip
        earliest, latest = pipeline.instant(stamp)
        assert earliest <= at <= latest and latest - earliest <= 1.0


@SETTINGS
@given(MOMENT, QUARTERS, st.booleans())
def test_a_reply_window_holds_the_whole_exchange(arrived, seconds, whole):
    """Arrival to completion, as the recorder (whole seconds) or the request trace stamps them."""
    finished = arrived + timedelta(seconds=seconds)
    write = _recorder_stamp if whole else datetime.isoformat
    for response in ({"finished_at": write(finished)}, {"timing": {"total_s": seconds}}):
        window = pipeline.reply_window(write(arrived), response)
        assert window[0] <= arrived.timestamp() and finished.timestamp() <= window[1]


# Degenerate runs (M-02, R-04) and stopped raw records (C-13), through the whole pipeline: one
# Messages reply, its raw record joined by engine id 5 (msg_5_<hex>), traces in a scratch
# directory.
FEW = settings(max_examples=150, deadline=None)
ANTHROPIC = profiles.Profile(controls.CONTROL_PROFILES["anthropic"])
K3 = profiles.load_profile(os.path.join(SKILL, "profiles", "model", "kimi-k3.yaml"))
GLM = profiles.load_profile(os.path.join(SKILL, "profiles", "model", "glm-5.yaml"))
PAD_RUN = "[PAD]" * 20  # kimi-k3.yaml: [PAD], 20 in a row
ENDED = st.sampled_from([{}, {"aborted": False}, {"partial": False, "end_reason": None},
                         {"aborted": True}, {"partial": True}, {"end_reason": "error"},
                         {"aborted": True, "partial": True, "end_reason": "error"}])  # fmt: skip


def _stopped(ended):
    """Whether a record says the server stopped its generation: any of the hook's marks."""
    return bool(ended.get("aborted") or ended.get("partial") or ended.get("end_reason"))


def _messages_reply(blocks, ending):
    """A Messages reply delivering ``blocks``: finished (a stop reason), failed or cut."""
    if ending == "http500":
        return {"event": "response", "trace_id": "t1", "status": "completed", "http_status": 500,
                "response": {"kind": "json", "body": {"type": "error", "error": {
                    "type": "api_error", "message": "Generation failed"}}}}  # fmt: skip
    sse = _named(controls._anthropic_sse(blocks, ending if ending in ("end_turn", "max_tokens")
                                         else "end_turn"))  # fmt: skip
    head = sse.split("event: message_delta")[0]
    if ending == "error":  # the server failed the stream after it began
        error = {"type": "error", "error": {"type": "api_error", "message": "Generation failed"}}
        sse = head + "event: error\ndata: " + json.dumps(error) + "\n\n"
    elif ending == "cut":  # no terminal event
        sse = head
    return controls._sse_response("t1", sse)


def _named(sse):
    return sse.replace('"id": "m"', '"id": "msg_5_0123456789abcdef"', 1)


def _counts(model, blocks, ending, raw_record, thinking=True):
    request = controls._anth_request("t1", [{"role": "user", "content": "go"}],
                                     "2026-01-01T00:00:01Z")  # fmt: skip
    if thinking:
        request["body"]["thinking"] = {"type": "adaptive"}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        controls._write_trace(directory, [(request, _messages_reply(blocks, ending))])
        raw = []
        if raw_record is not None:
            with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
                fh.write(json.dumps(dict(raw_record, request_id="5")) + "\n")
            raw = [directory]
        return pipeline.run([directory], ANTHROPIC, model, raw_paths=raw)["counts"]


BLOCK_TEXT = st.sampled_from(["Done.", "", "[PAD]" * 19, PAD_RUN, "a" + "[PAD]" * 25 + "b"])
DELIVERED = st.lists(st.tuples(st.sampled_from(["text", "thinking"]), BLOCK_TEXT), max_size=2)
ENDING = st.sampled_from(["end_turn", "max_tokens", "error", "cut", "http500"])
RAW_IDS = st.one_of(st.none(), st.fixed_dictionaries({"run": st.sampled_from([0, 19, 20, 25]),
                                                      "ended": ENDED}))  # fmt: skip


@FEW
@given(DELIVERED, ENDING, RAW_IDS)
# The boundaries, every run: a stream cut before its terminal event is not an answer, one cut at
# its budget is, and a run written into a stopped generation counts where it was written.
@example([], "cut", {"run": 20, "ended": {}})
@example([], "max_tokens", {"run": 20, "ended": {}})
@example([], "error", {"run": 25, "ended": {"partial": True}})
def test_a_degenerate_run_fails_r04_only_when_it_reached_the_client(delivered, ending, raw):
    blocks = [{"type": kind, "text": text} for kind, text in delivered]
    record = None
    if raw is not None:
        ids = [163587, 1000, 163589] + [163839] * raw["run"] + [1001]
        record = dict({"text": "<|close|>think<|sep|><|open|>response<|sep|>Done."
                               "<|close|>response<|sep|>", "token_ids": ids}, **raw["ended"])  # fmt: skip
    counts = _counts(K3, blocks, ending, record)
    shown = ending != "http500" and any(PAD_RUN in text for _, text in delivered)
    written = raw is not None and ending != "http500" and raw["run"] >= 20
    answered = ending in ("end_turn", "max_tokens")
    assert counts.get("M-02", 0) == int(shown or written)
    assert counts.get("R-04", 0) == int(shown or (written and answered))
    assert counts.get("R-04:stopped", 0) == int(written and not shown and not answered)


PLAIN = st.text(alphabet="ab .", max_size=8)


@st.composite
def delivered_against(draw):
    """What the model wrote, and a delivered text: all of it, a prefix, more, or anything."""
    content = draw(PLAIN)
    how = draw(st.sampled_from(["same", "prefix", "more", "any"]))
    if how == "same":
        return content, content
    if how == "prefix":
        return content, content[: draw(st.integers(0, len(content)))]
    if how == "more":
        return content, content + draw(PLAIN.filter(bool))
    return content, draw(PLAIN)


@FEW
@given(delivered_against(), ENDED, st.sampled_from(["end_turn", "error"]))
def test_a_stopped_generation_is_compared_as_a_prefix(texts, ended, ending):
    content, delivered = texts
    blocks = [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": delivered}]
    record = dict({"text": "Plan.</think>" + content}, **ended)
    counts = _counts(GLM, blocks, ending, record, thinking=False)
    rules = {k.split(":")[1] for k in counts if k.startswith("C-13:RAW-")}
    assert counts.get("den:raw_compared") == 1
    if ending == "end_turn" and not _stopped(ended):
        assert rules == ({"RAW-03"} if delivered != content else set())
    else:
        assert rules == (set() if content.startswith(delivered) else {"RAW-05"})
    assert sum(v for k, v in counts.items() if k.startswith("C-13:stopped_raw:")) == int(
        _stopped(ended)
    )


@FEW
@given(st.lists(st.sampled_from([None, "b1", "b2"]), min_size=1, max_size=3),
       st.sampled_from([None, "", " ", "b1", " b1", "b2"]))  # fmt: skip
def test_inventory_is_blocked_exactly_when_a_build_went_unchecked(stamps, declared):
    user = {"role": "user", "content": "go"}
    done = controls._anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")
    pairs = []
    for i, stamp in enumerate(stamps):
        request = controls._anth_request(f"t{i}", [user], f"2026-01-01T00:00:0{i}Z")
        answer = controls._sse_response(f"t{i}", done)
        if stamp is not None:
            answer["build_id"] = stamp
        pairs.append((request, answer))
    revision = declared.strip() if declared else None
    revision = revision or None
    unchecked = [s for s in stamps if s != revision] if revision else [s for s in stamps if s]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        controls._write_trace(os.path.join(directory, "t"), pairs)
        harness = controls.write_profile(os.path.join(directory, "h.yaml"),
                                         controls.CONTROL_PROFILES["anthropic"])  # fmt: skip
        argv = ["--harness", harness, "--claims", "inventory", "--out-dir",
                os.path.join(directory, "v"), os.path.join(directory, "t")]  # fmt: skip
        if declared is not None:
            argv = ["--build-id", declared] + argv
        assert verdict.main(argv) in (0, 2)
        with open(os.path.join(directory, "v", "verdict.json")) as fh:
            result = json.load(fh)
    assert (result["claims"]["inventory"] == "BLOCKED") == bool(unchecked)
    assert all("--build-id" in b for b in result["claim_blockers"]["inventory"])
    record = result["revision"]
    assert record["declared_build_id"] == declared and record["checked"] == (revision is not None)
    assert record["replies_with_other_build_id"] == (
        len([s for s in stamps if s is not None and s != revision]) if revision else 0
    )
    assert record["build_ids_seen"] == {s: stamps.count(s) for s in set(stamps) if s}


# --------------------------------------------------------------------------- C-10 any_error, H-02

FILLER = st.text(alphabet="abcdefxyz _", max_size=8)  # can spell neither phrase below
ERROR_STATUS = st.one_of(st.none(), st.sampled_from([200, 399, 400, 413, 429, 500, 503]),
                         st.integers(100, 599))  # fmt: skip
TERMINAL_KIND = st.sampled_from(["completed", "error", "failed", "none"])
READS_OVERLENGTH = {"shape": "any_error", "matches": [r"prompt (?:is )?too long", r"^413(?:\s|:|$)"],
                    "unless": [r"rate limit"]}  # fmt: skip


def _is_error(status, kind):
    return (isinstance(status, int) and status >= 400) or kind in ("error", "failed")


@SETTINGS
@given(ERROR_STATUS, TERMINAL_KIND, FILLER, FILLER, st.booleans(), st.booleans(),
       st.sampled_from(["message", "code", "type"]), st.booleans())  # fmt: skip
# A 400 is an error whatever the stream says: the boundary, every run.
@example(400, "completed", "", "", True, False, "message", False)
def test_an_error_is_read_as_overlength_by_its_text(status, kind, before, after, matched, excluded,
                                                     where, upper):  # fmt: skip
    text = (
        before
        + ("prompt is too long" if matched else "")
        + (" rate limit" if excluded else "")
        + after
    )
    text = text.upper() if upper else text
    error = {"code": None, "type": None}
    terminal = {"kind": kind, "error": error, "message": None}
    if where == "message":
        terminal["message"] = text or None
    else:
        error[where] = text or None
    patterns = pipeline.overlength_patterns(READS_OVERLENGTH)
    # A status of 413 reads as over-length through the anchored pattern on its own.
    expect = _is_error(status, kind) and (matched or status == 413) and not excluded
    assert pipeline.overlength_read_by_text(patterns, status, terminal) == expect


@SETTINGS
@given(st.sampled_from(["(unclosed", "[", "a{2,1}"]), st.sampled_from(["matches", "unless"]))
def test_a_pattern_that_does_not_compile_is_a_usage_error(pattern, key):
    spec = dict(READS_OVERLENGTH, **{key: [pattern]})
    with pytest.raises(ValueError):
        pipeline.overlength_patterns(spec)


@SETTINGS
@given(st.sampled_from([None, {}, {"shape": "http_400", "matches": ["x"]}, {"shape": "in_stream"}]))
def test_only_any_error_compiles_patterns(spec):
    assert pipeline.overlength_patterns(spec) == {}


@SETTINGS
@given(ERROR_STATUS, st.booleans(), st.booleans())
def test_a_resend_is_ours_after_overlength_or_a_client_error_other_than_capacity(status, overlength,
                                                                                 failed):  # fmt: skip
    ours = overlength or (isinstance(status, int) and 400 <= status < 500 and status != 429)
    expect = "H-02" if ours else ("H-02:retry_after_failure" if failed else "H-02:other")
    assert pipeline.resend_cause(status, overlength, failed) == expect


# --------------------------------------------------------------------------- USG-08, fewer tools

TOOL_NAMES = ["exec_command", "view_image", "write_stdin", "get_goal"]
DESCRIPTION = st.text(alphabet="abcé ", max_size=8)  # é takes two bytes


def _bytes(declaration):
    return len(json.dumps(declaration, ensure_ascii=False).encode())


@st.composite
def tool_change(draw):
    """A turn's tools, the next request's, whether those are fewer, and what they dropped.

    The next request keeps each of the turn's tools as it was, changes its description, or drops
    it, and may declare a tool the turn did not. Dropped and changed declarations count with
    their bytes: the next prompt need not hold them.
    """
    names = draw(st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=4, unique=True))
    turn = [dict(controls._function(n, "p"), description=draw(DESCRIPTION)) for n in names]
    following, removed = [], 0
    for tool in turn:
        fate = draw(st.sampled_from(["keep", "change", "drop"]))
        if fate == "keep":
            following.append(tool)
            continue
        removed += _bytes(tool)
        if fate == "change":
            following.append(dict(tool, description=tool["description"] + "!"))
    others = [n for n in TOOL_NAMES if n not in names]
    if others and draw(st.booleans()):
        following.append(controls._function(draw(st.sampled_from(others)), "p"))
    following = draw(st.permutations(following))
    fewer = {t["name"] for t in following} < set(names)
    return turn, following, fewer, removed


@SETTINGS
@given(tool_change())
def test_a_prompt_may_shrink_only_by_the_tools_it_no_longer_declares(case):
    turn, following, fewer, removed = case
    bound = pipeline.removed_tools_bound(
        turnlib.tool_declarations(turn), turnlib.tool_declarations(following)
    )
    if not fewer:
        assert bound is None
    else:
        assert bound == removed + (0 if following else pipeline.TOOLS_PREAMBLE_TOKENS)


@SETTINGS
@given(st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=3, unique=True), DESCRIPTION)
def test_each_tool_of_a_namespace_carries_its_fields(names, description):
    tools = [controls._function(n, "p") for n in names]
    fields = {"type": "namespace", "name": "ns", "description": description}
    declared = turnlib.tool_declarations([dict(fields, tools=tools)])
    assert [d[0] for d in declared] == [f"ns.{n}" for n in names]
    assert [d[2] for d in declared] == [_bytes(fields) + _bytes(t) for t in tools]
    assert pipeline.removed_tools_bound(declared, ()) == (
        sum(d[2] for d in declared) + pipeline.TOOLS_PREAMBLE_TOKENS
    )


@FEW
@given(tool_change(), st.integers(-2, 2))
def test_usg08_excuses_fewer_tools_only_up_to_what_they_take(case, offset):
    turn, following, fewer, removed = case
    bound = removed + (0 if following else pipeline.TOOLS_PREAMBLE_TOKENS)
    shrank = max(1, (bound if fewer else 0) + offset)
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
    call = {"id": "fc_1", "type": "function_call", "name": turn[0]["name"], "call_id": "c1",
            "arguments": '{"p": "x"}', "status": "completed"}  # fmt: skip
    sent_back = {k: v for k, v in call.items() if k not in ("id", "status")}
    output = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    done = {"id": "msg_d", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}]}  # fmt: skip
    pairs = [
        (controls._codex_request("t1", [user], "2026-01-01T00:00:01Z", turn),
         controls._sse_response("t1", controls._responses_stream([call], 50_000))),
        (controls._codex_request("t2", [user, sent_back, output], "2026-01-01T00:00:02Z", following),
         controls._sse_response("t2", controls._responses_stream([done], 50_000 - shrank))),
    ]  # fmt: skip
    harness = profiles.Profile(controls.CONTROL_PROFILES["responses"])
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
        controls._write_trace(directory, pairs)
        counts = pipeline.run([directory], harness, GLM)["counts"]
    tripped = sum(v for k, v in counts.items() if k.startswith("C-14:USG-08:"))
    assert counts.get("den:next_input_pairs") == 1
    assert bool(tripped) == (not fewer or shrank > bound)
    assert counts.get("C-14:next_input_shrank_by_tools_change", 0) == (0 if tripped else 1)


# --------------------------------------------------------------------------- GLM argument values

ARGUMENT_MARKUP = ("<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")


@SETTINGS
@given(st.lists(st.sampled_from(["a", " ", "<", "/", ">"] + list(ARGUMENT_MARKUP)), max_size=6))
def test_glm_argument_markup_inside_a_value_is_the_models_slip(pieces):
    value = "".join(pieces)
    text = (
        f"Plan.</think><tool_call>f<arg_key>k</arg_key><arg_value>{value}</arg_value></tool_call>"
    )
    turn = native.read(profiles.native_card(GLM), text)
    if any(markup in value for markup in ARGUMENT_MARKUP):
        assert turn.anomalies or [c.args for c in turn.calls] != [[("k", value, None)]]
    else:
        assert turn.anomalies == [] and [c.args for c in turn.calls] == [[("k", value, None)]]


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q", "-p", "no:cacheprovider"]))
