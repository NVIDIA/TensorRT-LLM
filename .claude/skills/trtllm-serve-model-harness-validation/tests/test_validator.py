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
"""Tests of the validator in scripts/validate (CPU only, no TensorRT-LLM import).

The fast tests cover the parts every verdict depends on; the control matrix
(fake servers for all three protocols) runs with MH_RUN_CONTROLS=1.

    pytest .claude/skills/trtllm-serve-model-harness-validation/tests
    MH_RUN_CONTROLS=1 pytest .claude/skills/trtllm-serve-model-harness-validation/tests
"""

import hashlib
import json
import os
import sys

import pytest

VALIDATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                        "validate")  # fmt: skip
sys.path.insert(0, VALIDATE)

import controls  # noqa: E402
import native  # noqa: E402
import native_goldens  # noqa: E402
import pipeline  # noqa: E402
import profiles  # noqa: E402
import proto_anthropic  # noqa: E402
import proto_chat  # noqa: E402
import protocol_diff  # noqa: E402
import sdk_diff  # noqa: E402
import verdict  # noqa: E402
from tracelib import iter_lines, sse_frames  # noqa: E402
from turnlib import Item, RequestView, Tool, classify_call, payload_digest  # noqa: E402


def test_credential_scan_finds_secrets_in_environment_dumps():
    """`env` output in a tool result is JSON-escaped: the name follows a literal backslash-n."""
    import scan_credentials

    line = b'{"content": "HOME=/root\\nAPI_TOKEN=' + b"a1B2c3D4e5F6g7H8i9J0" + b'\\nPATH=/bin"}'
    assert [name for name, _ in scan_credentials.matches(line)] == ["secret_assignment"]
    short = b'{"content": "\\nANTHROPIC_AUTH_TOKEN=dummy\\n"}'
    assert list(scan_credentials.matches(short)) == []


def test_sse_frames_keep_every_frame():
    frames = sse_frames(
        'event: ping\ndata: {"type":"ping"}\n\n: keepalive\n\ndata: [DONE]\n\ndata: x\n\n'
    )
    assert [(f.name, f.kind) for f in frames] == [
        ("ping", "json"),
        (None, "comment"),
        (None, "done"),
        (None, "unparsable"),
    ]


def test_sse_events_read_crlf_framing():
    """`sse_events` (the reference checkers) frames a stream as `sse_frames` (the verdict) does."""
    from tracelib import sse_events

    body = 'event: response.created\ndata: {"type": "response.created"}\n\ndata: [DONE]\n\n'
    crlf = body.replace("\n", "\r\n")
    assert (
        list(sse_events(crlf))
        == list(sse_events(body))
        == [("response.created", {"type": "response.created"})]
    )
    assert [(f.name, f.kind) for f in sse_frames(crlf)] == [
        (f.name, f.kind) for f in sse_frames(body)
    ]


def test_check_stream_fails_an_unparsable_frame(tmp_path):
    """A frame an SDK client cannot read fails the reference checker, as it fails RSP-01."""
    import subprocess

    response = {"id": "resp_1", "model": "m", "output": []}
    frames = [("response.created", {"type": "response.created", "sequence_number": 0,
                                    "response": dict(response, status="in_progress")}),
              ("response.output_text.delta", "{not json"),
              ("response.completed", {"type": "response.completed", "sequence_number": 1,
                                      "response": dict(response, status="completed")})]  # fmt: skip
    body = "".join(
        f"event: {name}\ndata: {data if isinstance(data, str) else json.dumps(data)}\n\n"
        for name, data in frames
    )
    shard = tmp_path / "responses-1.jsonl"
    shard.write_text(json.dumps({"event": "response", "trace_id": "t1",
                                 "response": {"kind": "sse_text", "body": body}}) + "\n")  # fmt: skip
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "check_stream.py"), "--out",
                             str(out), str(shard)], capture_output=True)  # fmt: skip
    verdicts = json.loads(out.read_text())["verdicts"]
    assert result.returncode == 1
    assert {c for c, v in verdicts.items() if v != "PASS"} == {"unparsable_event"}


def test_check_stream_cannot_judge_a_cut_shard(tmp_path):
    """A cut shard is no FAIL: exit 2 with the shard's name, no traceback and no report."""
    import subprocess

    import zstandard

    done = {"type": "response.completed", "sequence_number": 0,
            "response": {"id": "resp_1", "model": "m", "output": [], "status": "completed"}}  # fmt: skip
    body = f"event: response.completed\ndata: {json.dumps(done)}\n\n"
    data = "".join(json.dumps({"event": "response", "trace_id": f"t{i}",
                               "response": {"kind": "sse_text", "body": body}}) + "\n"
                   for i in range(20))  # fmt: skip
    packed = zstandard.ZstdCompressor().compress(data.encode())
    shard = tmp_path / "responses-1.jsonl.zst"
    shard.write_bytes(packed[: len(packed) // 2])
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "check_stream.py"), "--out",
                             str(out), str(shard)], capture_output=True, text=True)  # fmt: skip
    assert (result.returncode, out.exists()) == (2, False), result.stderr
    assert str(shard) in result.stderr and "Traceback" not in result.stderr


def test_check_stream_judges_nothing_in_a_trace_without_streams(tmp_path):
    """A trace with no stream to read leaves every check BLOCKED: exit 2, not a pass."""
    import subprocess

    shard = tmp_path / "responses-1.jsonl"
    shard.write_text("")
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "check_stream.py"), "--out",
                             str(out), str(shard)], capture_output=True, text=True)  # fmt: skip
    verdicts = json.loads(out.read_text())["verdicts"]
    assert set(verdicts.values()) == {"BLOCKED"}
    assert result.returncode == 2, result.stderr


def _write_cut_gz(path, records):
    """A ``.gz`` shard of ``records`` cut in half: it cannot be read to its end."""
    import gzip

    packed = gzip.compress("".join(json.dumps(r) + "\n" for r in records).encode())
    path.write_bytes(packed[: len(packed) // 2])


@pytest.mark.parametrize("side,workers", [("requests", 1), ("responses", 2)])
def test_check_tool_calls_cannot_judge_a_cut_shard(tmp_path, side, workers):
    """A cut shard is no FAIL, read in-process or by a worker: exit 2 and no report."""
    import subprocess

    request = {"event": "request", "route": "/v1/responses", "body": {"model": "m", "input": "go"}}
    response = {"event": "response", "response": {"kind": "json", "body": {"output": []}}}
    records = {"requests": [dict(request, trace_id=f"t{i}") for i in range(20)],
               "responses": [dict(response, trace_id=f"t{i}") for i in range(20)]}  # fmt: skip
    for kind, rows in records.items():
        (tmp_path / f"{kind}-1.jsonl").write_text(json.dumps(rows[0]) + "\n")
    cut = tmp_path / f"{side}-2.jsonl.gz"
    _write_cut_gz(cut, records[side])
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "check_tool_calls.py"),
                             "--workers", str(workers), "--out", str(out), str(tmp_path)],
                            capture_output=True, text=True)  # fmt: skip
    assert (result.returncode, out.exists()) == (2, False), result.stdout + result.stderr
    assert str(cut) in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("name", [["exec"], {"name": "exec"}, 7], ids=["list", "dict", "number"])
def test_check_tool_calls_reads_a_name_that_is_not_a_string_as_malformed(name):
    """A call name that is not a string answers to no declaration: undeclared, not a crash."""
    import check_tool_calls

    tool = {"type": "function", "name": "exec", "parameters": {"type": "object"}}
    decl, _ = check_tool_calls.declarations({"tools": [tool]})
    call = {"type": "function_call", "name": name, "payload": "{}"}
    result = check_tool_calls.check_call(call, decl)
    assert (result["class"], result["detail"]) == ("undeclared", {"reason": "malformed_name"})
    assert check_tool_calls.check_call(dict(call, name="exec"), decl)["class"] == "valid"


@pytest.mark.parametrize("payload", ["", "  ", "{}", '{"path": "a"}', "{"])
@pytest.mark.parametrize("required", [[], ["path"]])
def test_check_tool_calls_reads_arguments_as_the_verdict_does(payload, required):
    """Blank function arguments are an empty object in the reference checker too."""
    import check_tool_calls
    from turnlib import payload_check

    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": required}
    reference = check_tool_calls.payload_check(payload, {"type": "function", "parameters": schema})
    assert reference[0] == payload_check(payload, Tool(name="read", schema=schema))[0]


def test_check_tool_calls_screens_a_closing_dsml_tag(tmp_path):
    """The reference screen reads the verdict's markers: a lone closing DSML tag is markup too."""
    import subprocess

    content = [{"type": "output_text", "text": "done</｜DSML｜function_calls>"}]
    final = {"id": "resp_1", "object": "response", "status": "completed",
             "output": [{"type": "message", "role": "assistant", "content": content}]}  # fmt: skip
    request = {"event": "request", "trace_id": "t1", "route": "/v1/responses",
               "body": {"model": "m", "input": "go"}}  # fmt: skip
    response = {"event": "response", "trace_id": "t1", "response": {"kind": "json", "body": final}}
    (tmp_path / "requests-1.jsonl").write_text(json.dumps(request) + "\n")
    (tmp_path / "responses-1.jsonl").write_text(json.dumps(response) + "\n")
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "check_tool_calls.py"),
                             "--out", str(out), str(tmp_path)], capture_output=True, text=True)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["counts"].get("screen:markup_in_message:bare") == 1


def test_harness_signals_refuses_a_file_it_cannot_read(tmp_path):
    """A shard's name says which side it holds: a file named otherwise is an error, not a crash."""
    import harness_signals

    shard, other = tmp_path / "requests-1.jsonl", tmp_path / "raw-1.jsonl"
    shard.write_text("")
    other.write_text("")
    assert harness_signals.expand([str(shard), str(tmp_path)]) == [str(shard)]
    with pytest.raises(SystemExit, match="raw-1.jsonl"):
        harness_signals.expand([str(other)])


def test_payload_digest_is_typed_and_order_free():
    assert payload_digest('{"a": 1, "b": [1, 2]}', "function") == payload_digest(
        '{"b":[1,2],"a":1}', "function"
    )
    assert payload_digest('{"a": true}', "function") != payload_digest('{"a": 1}', "function")
    assert payload_digest('{"a": [1, 2]}', "function") != payload_digest(
        '{"a": [2, 1]}', "function"
    )
    assert payload_digest("echo hi", "custom") != payload_digest("echo  hi", "custom")
    assert payload_digest('{"a": 1}', "function") == payload_digest('{"a": 1.0}', "function")


def test_payload_digest_keeps_integers_exact():
    """No float holds 2**53 + 1, and none holds a 400-digit integer a model may write."""
    assert payload_digest('{"n": 9007199254740993}', "function") != payload_digest(
        '{"n": 9007199254740992}', "function"
    )
    big = "1" + "0" * 400
    assert payload_digest('{"n": %s}' % big, "function") != payload_digest(
        '{"n": %s1}' % big, "function"
    )


def test_wilson_bounds_follow_the_rule_of_three():
    lower, upper = verdict.wilson(0, 20)
    assert lower == 0.0 and upper > 0.01  # as control X-10: 0 of 20 proves nothing at 1%
    _, upper = verdict.wilson(0, 3000)
    assert upper < 0.001


def test_classify_call_against_own_declarations():
    request = RequestView(protocol="responses", route="/v1/responses")
    request.tools = {"exec": Tool("exec", "functions", "custom"),
                     "wait": Tool("wait", "functions", "function",
                                  schema={"type": "object", "required": ["ms"]})}  # fmt: skip
    assert classify_call(Item("call", name="exec", call_kind="function", payload="{}"),
                         request)["class"] == "type_mismatch"  # fmt: skip
    assert classify_call(Item("call", name="functions.wait", call_kind="function", payload='{"ms": 1}'),
                         request)["class"] == "prefixed_name_of_declared"  # fmt: skip
    assert classify_call(Item("call", name="wait", call_kind="function", payload="{}"),
                         request)["class"] == "payload_invalid"  # fmt: skip
    assert classify_call(Item("call", name="nope", call_kind="function", payload="{}"),
                         request)["class"] == "undeclared"  # fmt: skip


def test_anthropic_stream_rules():
    good = controls._anthropic_sse([{"type": "text", "text": "hi"},
                                    {"type": "tool_use", "id": "t1", "name": "Read", "json": '{"a": 1}'}])  # fmt: skip
    reply = proto_anthropic.parse_reply({"response": {"kind": "sse_text", "body": good}}, None)
    assert reply.grammar == []
    assert [i.kind for i in reply.stream_items] == ["text", "call"]
    assert reply.terminal["kind"] == "completed"
    truncated = good.split("event: message_delta")[0]
    reply = proto_anthropic.parse_reply({"response": {"kind": "sse_text", "body": truncated}}, None)
    assert {"ANT-08", "ANT-10"} <= {v.rule for v in reply.grammar}


def test_chat_stream_rules():
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                           "function": {"name": "read", "arguments": '{"a"'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]  # fmt: skip
    body = "".join("data: " + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m",
                                              created=1)) + "\n\n" for c in chunks) + "data: [DONE]\n\n"  # fmt: skip
    reply = proto_chat.parse_reply({"response": {"kind": "sse_text", "body": body}}, None)
    assert reply.grammar == []
    assert reply.stream_items[-1].payload == '{"a": 1}'
    reply = proto_chat.parse_reply({"response": {"kind": "sse_text",
                                                 "body": body.replace("data: [DONE]\n\n", "")}}, None)  # fmt: skip
    assert "CHT-09" in {v.rule for v in reply.grammar}


@pytest.mark.skipif(sdk_diff.unavailable("anthropic-messages") is not None,
                    reason="anthropic SDK not installed (scripts/validate/requirements.txt)")  # fmt: skip
def test_sdk_reading_agrees_and_disagrees():
    good = controls._anthropic_sse([{"type": "text", "text": "hi"},
                                    {"type": "tool_use", "id": "t1", "name": "Read", "json": '{"a": 1}'}])  # fmt: skip
    record = {"http_status": 200, "response": {"kind": "sse_text", "body": good}}
    reply = proto_anthropic.parse_reply(record, None)
    assert sdk_diff.compare("anthropic-messages", record, reply) == []
    # Without `event:` lines the stream breaks ANT-15, and the SDK reads nothing (SDK-01).
    bare = "".join(line + "\n" for line in good.split("\n") if not line.startswith("event:"))
    record = {"http_status": 200, "response": {"kind": "sse_text", "body": bare}}
    reply = proto_anthropic.parse_reply(record, None)
    assert "ANT-15" in {v.rule for v in reply.grammar}
    assert [v.rule for v in sdk_diff.compare("anthropic-messages", record, reply)] == ["SDK-01"]
    # An error status is the error rules' business, not the SDK reading's.
    record = {"http_status": 500, "response": {"kind": "json", "body": {"type": "error"}}}
    assert (
        sdk_diff.compare("anthropic-messages", record, proto_anthropic.parse_reply(record, None))
        is None
    )


def test_native_goldens_read_back():
    """Every encoder render reads back exactly with the model profile's own native section."""
    skill = os.path.dirname(os.path.dirname(VALIDATE))
    with open(os.path.join(skill, "tests", "fixtures", "native_goldens.json")) as fh:
        goldens = json.load(fh)["goldens"]
    assert {g["model"] for g in goldens} == {"kimi-k3", "glm-5", "deepseek-v41"}
    for golden in goldens:
        model = profiles.load_profile(
            os.path.join(skill, "profiles", "model", golden["model"] + ".yaml")
        )
        turn = native.read(profiles.native_card(model), golden["text"])
        got = {"reasoning": turn.reasoning, "content": turn.content,
               "calls": [{"name": c.name, "args": [list(a) for a in c.args]} for c in turn.calls]}  # fmt: skip
        assert (got, turn.anomalies) == (golden["expected"], []), (golden["model"], golden["case"])
        if "vendor" in golden:  # DeepSeek's own parser reads the same render the same way
            calls = [{"name": c.name, "arguments": {k: v if t == "string" else json.loads(v)
                                                    for k, v, t in c.args}} for c in turn.calls]  # fmt: skip
            assert (
                calls == golden["vendor"]["calls"] and turn.content == golden["vendor"]["content"]
            )


def test_native_goldens_check_reads_the_served_checkpoints_renders_back(
    tmp_path, monkeypatch, capsys
):
    """--check passes renders the reader reads back, names one it cannot, and writes nothing."""
    skill = os.path.dirname(os.path.dirname(VALIDATE))
    with open(os.path.join(skill, "tests", "fixtures", "native_goldens.json")) as fh:
        renders = [g for g in json.load(fh)["goldens"] if g["model"] == "glm-5"]
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "chat_template.jinja").write_text("{{ messages }}")
    monkeypatch.setattr(native_goldens, "glm", lambda path: renders)

    assert native_goldens.main(["--glm", str(ckpt), "--check"]) == 0
    first = capsys.readouterr().out
    # A render whose call is never closed: the reader must not read it back as the message.
    case = next(i for i, g in enumerate(renders) if "</tool_call>" in g["text"])
    renders[case] = dict(renders[case], text=renders[case]["text"].replace("</tool_call>", "", 1))
    assert native_goldens.main(["--glm", str(ckpt), "--check"]) == 1
    second = capsys.readouterr().out

    sha = hashlib.sha256(b"{{ messages }}").hexdigest()[:16]
    assert f"glm: chat_template.jinja template_sha256 {sha}" in first
    assert "glm-5: 5 renders, 5 read back exactly, 5 equal to the shipped goldens" in first
    assert "glm-5: 5 renders, 4 read back exactly, 4 equal to the shipped goldens" in second
    assert f"FAIL {renders[case]['case']}:" in second
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt"]


def test_native_goldens_check_names_a_render_other_than_the_shipped_one(tmp_path, capsys):
    """A checkpoint's render that reads back exactly passes, and one unlike the shipped is named."""
    skill = os.path.dirname(os.path.dirname(VALIDATE))
    with open(os.path.join(skill, "tests", "fixtures", "native_goldens.json")) as fh:
        fixture = json.load(fh)
    renders = [g for g in fixture["goldens"] if g["model"] == "glm-5"]
    shipped = tmp_path / "shipped.json"
    first = dict(renders[0], text=renders[0]["text"] + " ")  # another checkpoint's render, say
    shipped.write_text(json.dumps(dict(fixture, goldens=[first] + renders[1:])))

    assert native_goldens.check(renders, shipped_path=str(shipped)) == 0
    out = capsys.readouterr().out
    assert "glm-5: 5 renders, 5 read back exactly, 4 equal to the shipped goldens" in out
    assert f"DIFFERS {renders[0]['case']}: reads back exactly" in out and "FAIL" not in out


def test_native_goldens_check_fails_when_it_checks_nothing(capsys):
    """--check without a checkpoint, or with no render, checks nothing: that is not a pass."""
    with pytest.raises(SystemExit) as no_checkpoint:
        native_goldens.main(["--check"])
    assert no_checkpoint.value.code == 2
    assert native_goldens.check([]) == 1
    assert "nothing was checked" in capsys.readouterr().out


def test_native_reader_flags_the_models_slips():
    card = {"kind": "glm_xml", "prompt_opens": "think", "stop_markers": ["<|observation|>"]}
    turn = native.read(card, "Plan.</think>Reading.<tool_call>Read<arg_key>p</arg_key><arg_value>a")
    assert "unclosed_call" in turn.anomalies and "unclosed_arg_value" in turn.anomalies
    turn = native.read(card, "Plan.</think>x</think>y")
    assert "think_marker_after_reasoning" in turn.anomalies
    # A call opened inside the reasoning, the close it owed written after the call: the model's
    # slip, unless the profile declares that serving repairs it; never a call read as reasoning.
    owed = "Plan.<tool_call>Read<arg_key>p</arg_key><arg_value>a</arg_value></tool_call></think>"
    turn = native.read(card, owed)
    assert (turn.reasoning, len(turn.calls), turn.anomalies) == ("Plan.", 1, ["call_in_reasoning"])
    turn = native.read(dict(card, repairs=["owed_think_close"]), owed)
    assert (turn.reasoning, len(turn.calls), turn.anomalies, turn.repairs) == (
        "Plan.", 1, [], ["owed_think_close"])  # fmt: skip
    turn = native.read(dict(card, repairs=["owed_think_close"]), owed.replace("</think>", ""))
    assert turn.anomalies == ["unclosed_think"] and turn.repairs == []
    k3 = {"kind": "kimi_xtml", "prompt_opens": "think"}
    turn = native.read(k3, '<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|>'
                           '<|open|>tools<|sep|><|open|>call tool="f" index="1"<|sep|>'
                           '<|open|>argument key="n" type="number"<|sep|>ten<|close|>argument<|sep|>'
                           '<|close|>call<|sep|>')  # fmt: skip
    assert turn.anomalies == ["argument_not_its_type"]  # stopping inside `tools` is the engine's


def test_stamps_of_each_writer_bound_the_reply_window():
    """When a reply's generation ran, from the stamps the recorder and the raw-output hook write.

    The recorder stamps whole seconds; the hook, trtllm-serve's request trace and a replay stamp
    microseconds. The raw records one engine id names on several generation workers are held
    against this window.
    """
    second = pipeline.instant("2026-10-28T03:04:05Z")
    assert second[1] - second[0] == 1.0
    exact = (second[0] + 0.25,) * 2
    assert pipeline.instant("2026-10-28T03:04:05.250000+00:00") == exact
    assert pipeline.instant("2026-10-28T03:04:05.250000Z") == exact
    assert pipeline.instant("") is None and pipeline.instant("03:04:05") is None
    arrived = "2026-10-28T03:04:05Z"
    assert pipeline.reply_window(arrived, {"finished_at": "2026-10-28T03:04:09Z"}) == (
        second[0], second[0] + 5.0)  # fmt: skip
    assert pipeline.reply_window(arrived, {"timing": {"total_s": 2.5}}) == (
        second[0], second[0] + 3.5)  # fmt: skip
    assert pipeline.reply_window(arrived, {}) is None
    assert pipeline.reply_window(None, {"finished_at": arrived}) is None


def test_sdk_versions_are_pinned():
    pins = sdk_diff.pinned()
    assert pins["anthropic"] and pins["openai"]


def test_todo_profile_fields_are_unknown():
    profile = profiles.Profile(
        {"outcomes": "TODO", "timeouts": {"idle_s": "TODO", "first_event_s": 30}}
    )
    assert profiles.outcome_rules(profile) is None
    assert profile.get_path("timeouts.idle_s") is None
    assert profile.get_path("timeouts.first_event_s") == 30


@pytest.mark.parametrize("harness, text, cls", [
    ("claude-code", "Permission to use Bash has been denied.", "aborted"),
    ("claude-code", "<tool_use_error>Error: No such tool available: Foo</tool_use_error>",
     "unsupported"),
    ("claude-code", "<tool_use_error>InputValidationError: Read failed</tool_use_error>",
     "arguments_rejected"),
    ("pi-messages", "Operation aborted", "aborted"),
    ("pi-messages", "Tool foo not found", "unsupported"),
    ("pi-messages", 'Validation failed for tool "read":', "arguments_rejected"),
])  # fmt: skip
def test_messages_error_wordings_count_only_on_an_error_result(harness, text, cls):
    """Over Messages an outcome wording in a result without is_error: true is the tool's output."""
    path = os.path.join(os.path.dirname(VALIDATE), "..", "profiles", "harness", f"{harness}.yaml")
    rules = profiles.outcome_rules(profiles.load_profile(path))
    assert rules.classify(text, True) == cls
    assert rules.classify(text, False) == rules.classify(text, None) == "result"


def test_session_rates_need_session_identity():
    report = {"counts": {"den:replies": 50, "den:calls": 40, "M-02": 0}, "sessions": 0,
              "sessions_with": {}, "outcome_classes": {}, "examples": {}, "by_rule": {}}  # fmt: skip
    thresholds = {"unit_for_rates": "session", "model_behavior": {"degenerate_rate": 0.5}}
    records = verdict.check_records(report, thresholds, profiles.Profile(), "window")
    m02 = next(r for r in records if r["id"] == "M-02")
    assert m02["status"] == "BLOCKED" and "session" in m02["reason"]


def _cell_report(tmp_path, profile_name, pairs):
    harness = controls.write_profile(
        str(tmp_path / "h.yaml"), controls.CONTROL_PROFILES[profile_name]
    )
    controls._write_trace(str(tmp_path / "t"), pairs)
    _, v = controls.run_verdict(str(tmp_path / "v"), [str(tmp_path / "t")], harness, "window",
                                "inventory,conformance")  # fmt: skip
    with open(tmp_path / "v" / "report.json") as fh:
        return v, json.load(fh)


def test_foreign_protocol_is_excluded_not_judged(tmp_path):
    ping = [{"role": "user", "content": "ping"}]
    chat_request = {"event": "request", "trace_id": "p1", "route": "/v1/chat/completions",
                    "body": {"model": "x", "stream": True, "messages": ping}}  # fmt: skip
    broken = "data: " + json.dumps({"object": "chat.completion.chunk", "choices": []}) + "\n\n"
    v, report = _cell_report(
        tmp_path, "anthropic", [(chat_request, controls._sse_response("p1", broken))]
    )
    assert report["counts"].get("inventory:foreign_protocol:chat-completions") == 1
    assert v["claims"]["conformance"] == "BLOCKED"  # nothing of this cell's protocol was judged
    assert not [t for t in v["tripped"] if t.startswith("CHT")]


def test_client_close_after_terminal_is_judged(tmp_path):
    user = {"role": "user", "content": "go"}
    done = controls._anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")
    pairs = [(controls._anth_request("t1", [user], "2026-01-01T00:00:01Z"),
              controls._sse_response("t1", done, status="client_disconnected"))]  # fmt: skip
    _, report = _cell_report(tmp_path, "anthropic", pairs)
    assert report["counts"].get("den:streams") == 1
    assert report["counts"].get("H-03:closed_after_terminal") == 1
    assert "H-03:client_disconnected" not in report["counts"]


def _diff_runs(tmp_path, **kwargs):
    chat, anth = controls.diff_runs(str(tmp_path), **kwargs)
    report = protocol_diff.compare({"chat": protocol_diff.load_run(chat),
                                    "messages": protocol_diff.load_run(anth)})  # fmt: skip
    return report["rows"][0]["messages"]["attribution"]


def test_protocol_diff_attributes_each_difference(tmp_path):
    assert _diff_runs(tmp_path / "a") == "equal"
    assert _diff_runs(tmp_path / "b", billed=9) == "server_render"
    assert _diff_runs(tmp_path / "c", text="no") == "divergent_generation"
    assert (
        _diff_runs(tmp_path / "e", blocks=[{"type": "thinking", "text": "ok"}]) == "server_output"
    )
    assert _diff_runs(tmp_path / "d", system="other") == "harness"


def test_protocol_diff_turns_leave_the_tools_out(tmp_path):
    """`request_equal.turns` compares the turns only, not the tools as each API wrote them."""
    chat, anth = controls.diff_runs(str(tmp_path))
    runs = {"chat": protocol_diff.load_run(chat), "messages": protocol_diff.load_run(anth)}
    conversation = runs["messages"][0]["conversation"]
    conversation["tools_literal"] = [(n, s + " ") for n, s in conversation["tools_literal"]]
    row = protocol_diff.compare(runs)["rows"][0]["messages"]
    assert row["attribution"] == "harness_literal"
    assert row["request_equal"]["turns"] is True


@pytest.mark.parametrize("damage", ["cut_gz", "cut_last_line"])
def test_protocol_diff_cannot_judge_a_cut_shard(tmp_path, capsys, damage):
    """A shard that cannot be read to its end is not conclusive (exit 2), not a difference."""
    chat, anth = controls.diff_runs(str(tmp_path), billed=9)
    runs = [f"chat={chat}", f"messages={anth}"]
    assert protocol_diff.main(["--out", str(tmp_path / "whole.json")] + runs) == 1
    if damage == "cut_gz":
        cut = tmp_path / "anth" / "requests-y.jsonl.gz"
        _write_cut_gz(cut, [{"event": "request", "trace_id": f"a{i}"} for i in range(20)])
    else:
        cut = tmp_path / "chat" / "responses-y.jsonl"
        cut.write_text('{"event": "response", "trace_id": "c9", "resp')
    out = tmp_path / "report.json"
    assert protocol_diff.main(["--out", str(out)] + runs) == 2
    assert not out.exists() and str(cut) in capsys.readouterr().err


def _diff_with_replies(tmp_path, side, edit):
    """Exit code and row of an equal pair whose ``side`` replies are replaced by ``edit(record)``."""
    chat, anth = controls.diff_runs(str(tmp_path / "d"))
    shard = os.path.join(chat if side == "chat" else anth, "responses-x.jsonl")
    with open(shard) as fh:
        records = [json.loads(line) for line in fh]
    with open(shard, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in map(edit, records) if r is not None)
    out = tmp_path / "report.json"
    code = protocol_diff.main(["--out", str(out), f"chat={chat}", f"messages={anth}"])
    return code, json.loads(out.read_text())["rows"][0]["messages"]


def _cut_by_client(record):
    first_frame = record["response"]["body"].split("\n\n")[0] + "\n\n"
    return dict(record, status="client_disconnected",
                response=dict(record["response"], body=first_frame))  # fmt: skip


LOST = {"kind": "text", "body": "ServerDisconnectedError"}


@pytest.mark.parametrize("side, edit, reason", [
    ("anth", lambda r: None, "no_reply"),
    ("chat", _cut_by_client, "client_cut"),
    ("anth", lambda r: dict(r, status="upstream_error", http_status=502, response=LOST), "recorder_lost"),
    ("anth", lambda r: dict(r, status="upstream_error", http_status=400), "recorder_lost"),
    ("anth", lambda r: dict(r, status="recorder_shutdown", http_status=None,
                            response={"kind": "none", "body": None}), "pending"),
], ids=["no_reply", "client_cut", "unreachable", "lost_after_a_400", "pending"])  # fmt: skip
def test_protocol_diff_does_not_judge_a_reply_the_server_did_not_end(tmp_path, side, edit, reason):
    """A missing reply, or one cut by its client or the recorder, is not the server's: exit 2."""
    code, row = _diff_with_replies(tmp_path, side, edit)
    assert (code, row["attribution"]) == (2, "not_judged")
    assert reason in row["not_judged"]


@pytest.mark.parametrize("edit, code, attribution", [
    (lambda r: dict(r, status="client_disconnected"), 0, "equal"),
    (lambda r: dict(r, http_status=400, response={"kind": "json", "body": {
        "type": "error", "error": {"type": "invalid_request_error", "message": "no"}}}),
     1, "server_rejected"),
], ids=["closed_after_the_end", "refused_by_the_server"])  # fmt: skip
def test_protocol_diff_judges_a_reply_the_server_ended(tmp_path, edit, code, attribution):
    """A client that closed after the terminal event cut nothing; a server's refusal is its own."""
    got_code, row = _diff_with_replies(tmp_path, "anth", edit)
    assert (got_code, row["attribution"], row["not_judged"]) == (code, attribution, [None, None])


def test_verdict_names_the_matrix_bytes_it_read(tmp_path):
    """V-02: a verdict names its thresholds by the SHA-256 of the matrix file as read."""
    import hashlib

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("thresholds:\n  unit_for_rates: reply\n")
    harness = controls.write_profile(
        str(tmp_path / "h.yaml"), controls.CONTROL_PROFILES["anthropic"]
    )
    done = controls._anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")
    user = {"role": "user", "content": "go"}
    controls._write_trace(str(tmp_path / "t"), [
        (controls._anth_request("t1", [user], "2026-01-01T00:00:01Z"),
         controls._sse_response("t1", done))])  # fmt: skip
    _, v = controls.run_verdict(str(tmp_path / "v"), [str(tmp_path / "t")], harness, "window",
                                "conformance", matrix=str(matrix))  # fmt: skip
    digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
    assert v["thresholds_from"] == f"{matrix}@{digest}"


@pytest.mark.parametrize("own, billed", [
    ({}, 50),
    ({"chat_template": "{{ messages }}"}, 53),
    ({"prompt_token_ids": [1, 2, 3]}, 53),
    ({"add_generation_prompt": False}, 53),
], ids=["opener_rendered", "own_template", "prompt_token_ids", "no_generation_prompt"])  # fmt: skip
def test_the_prompt_offset_applies_only_where_the_server_renders_the_opener(tmp_path, own, billed):
    """K3 bills its 3-token opener out of a Chat prompt it renders, not out of one it was given."""
    chat = controls.write_profile(str(tmp_path / "h.yaml"), controls.CONTROL_PROFILES["chat"])
    usage = {"prompt_tokens": billed, "completion_tokens": 1, "total_tokens": billed + 1}
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": usage}]  # fmt: skip
    reply = "".join("data: " + json.dumps(dict(c, id="chatcmpl-x", object="chat.completion.chunk",
                                               model="m", created=1)) + "\n\n"
                    for c in chunks) + "data: [DONE]\n\n"  # fmt: skip
    request = controls._chat_request(
        "c1", [{"role": "user", "content": "go"}], "2026-01-01T00:00:01Z"
    )
    request["body"].update(own, stream_options={"include_usage": True})
    record = dict(
        controls._sse_response("c1", reply), prompt_tokens={"count": 53, "source": "test"}
    )
    controls._write_trace(str(tmp_path / "t"), [(request, record)])
    _, v = controls.run_verdict(str(tmp_path / "v"), [str(tmp_path / "t")], chat, "window",
                                "conformance", model=controls.K3_PROFILE)  # fmt: skip
    c14 = next(r for r in v["checks"] if r["id"] == "C-14")
    assert c14["usage_vs_trace_prompt"] == 1 and "USG-07" not in v["tripped"]


def test_verdict_that_cannot_be_written_is_no_verdict(tmp_path, capsys):
    """An output directory that cannot be made gives exit 3 (no verdict), never 1 (a FAIL)."""
    harness = controls.write_profile(
        str(tmp_path / "h.yaml"), controls.CONTROL_PROFILES["anthropic"]
    )
    done = controls._anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")
    user = {"role": "user", "content": "go"}
    controls._write_trace(str(tmp_path / "t"), [
        (controls._anth_request("t1", [user], "2026-01-01T00:00:01Z"),
         controls._sse_response("t1", done))])  # fmt: skip
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    argv = ["--harness", harness, "--population", "controls", "--claims", "conformance",
            "--out-dir", str(blocker / "v"), str(tmp_path / "t")]  # fmt: skip
    assert verdict.main(argv) == 3
    assert "verdict.py: " in capsys.readouterr().err


def test_a_schema_ref_that_does_not_resolve_leaves_the_payload_unjudged():
    """A declared $ref nothing resolves makes the schema unusable, in the repair re-check too."""
    import check_tool_calls
    from turnlib import PI_EDIT_TOOL, payload_check

    missing = {"$ref": "#/$defs/missing"}
    schema = {"type": "object", "properties": {"a": missing}}
    assert payload_check('{"a": 1}', Tool(name="t", schema=schema)) == (
        True, {"reason": "declared_schema_invalid"})  # fmt: skip
    assert check_tool_calls.payload_check('{"a": 1}', {"type": "function", "parameters": schema}) == (
        True, {"reason": "declared_schema_invalid"})  # fmt: skip
    # The arguments as written miss `edits`; pi's repair adds it, and only then is the $ref reached.
    edit = {"type": "object", "required": ["path", "edits"], "additionalProperties": False,
            "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": missing}}}  # fmt: skip
    args = '{"path": "x", "oldText": "a", "newText": "b"}'
    assert payload_check(args, Tool(name=PI_EDIT_TOOL, schema=edit), ("pi_edit_normalize",)) == (
        True, {"reason": "declared_schema_invalid"})  # fmt: skip


def test_scan_credentials_reports_an_unreadable_shard_and_goes_on(tmp_path):
    """A .zst shard that is not zstd is an error in its own row, not the end of the scan."""
    import subprocess

    (tmp_path / "requests-1.jsonl.zst").write_bytes(b"not zstd\n" * 4)
    (tmp_path / "requests-2.jsonl").write_text('{"k": "Bearer ' + "a1B2c3D4" * 4 + '"}\n')
    out = tmp_path / "scan.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "scan_credentials.py"),
                             "--out", str(out), str(tmp_path)], capture_output=True)  # fmt: skip
    assert result.returncode == 0, result.stderr
    rows = {os.path.basename(r["file"]): r for r in json.loads(out.read_text())["files"]}
    assert rows["requests-1.jsonl.zst"]["error"] == "ShardError"
    assert "bearer" in rows["requests-2.jsonl"]["patterns"]


def test_scan_credentials_keeps_the_hits_read_before_a_cut(tmp_path):
    """A cut shard's lines before the damage are scanned: its row has the hits and the error."""
    import gzip
    import hashlib
    import stat
    import subprocess

    fake = "Bearer " + "fake" * 8  # obviously not a credential, but Bearer-shaped
    lines = [json.dumps({"k": fake})] + [
        json.dumps({"n": hashlib.sha256(str(i).encode()).hexdigest()}) for i in range(400)
    ]
    packed = gzip.compress(("\n".join(lines) + "\n").encode())
    shard = tmp_path / "requests-1.jsonl.gz"
    shard.write_bytes(packed[: len(packed) // 2])
    os.chmod(shard, 0o644)
    known = tmp_path / "known.txt"
    known.write_text("0" * 64 + "\n")
    out = tmp_path / "scan.json"
    result = subprocess.run([sys.executable, os.path.join(VALIDATE, "scan_credentials.py"),
                             "--restrict", "--known", str(known), "--out", str(out), str(shard)],
                            capture_output=True)  # fmt: skip
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    (row,) = report["files"]
    assert (row["error"], list(row["patterns"])) == ("ShardError", ["bearer"])
    assert report["matches_by_pattern"] == {"bearer": 1} and len(report["new_digests"]) == 1
    assert stat.S_IMODE(os.stat(shard).st_mode) == 0o400


def _read_to_end(path):
    """``(lines, error)``: what ``iter_lines`` yielded, and what it raised (None if nothing)."""
    lines = []
    try:
        for _, raw in iter_lines(str(path)):
            lines.append(raw)
    except Exception as error:  # noqa: BLE001 - which error, if any, is what the tests check
        return lines, error
    return lines, None


@pytest.mark.parametrize("suffix", [".zst", ".gz"])
def test_a_damaged_compressed_shard_is_an_error_not_a_short_read(tmp_path, suffix):
    """A cut, empty or corrupt stream raises, naming the file; an intact one reads every line."""
    import gzip

    import zstandard

    lines = [json.dumps({"trace_id": f"t{i}", "body": "x" * 40}).encode() for i in range(2000)]
    data = b"".join(line + b"\n" for line in lines)
    half = len(data) // 2 + 7  # two frames (gzip: members), split inside a line
    compress = zstandard.ZstdCompressor().compress if suffix == ".zst" else gzip.compress
    whole = compress(data[:half]) + compress(data[half:])
    shards = {"intact": whole, "cut": whole[: len(whole) * 3 // 4], "cut_last_byte": whole[:-1],
              "empty": b"", "trailing_garbage": whole + b"not compressed\n"}  # fmt: skip
    for name, content in shards.items():
        path = tmp_path / f"requests-{name}.jsonl{suffix}"
        path.write_bytes(content)
        read, error = _read_to_end(path)
        if name == "intact":
            assert (read, error) == (lines, None)
        else:
            named = isinstance(error, ValueError) and str(path) in str(error)
            assert named, (name, error, len(read))


def test_a_last_line_cut_short_is_an_error_not_a_short_read(tmp_path):
    """A last line without its newline reads when it is JSON; when it is not, it was cut short."""
    whole = tmp_path / "requests-1.jsonl"
    whole.write_bytes(b'{"a": 1}\nnot json\n{"b": 2}')
    assert _read_to_end(whole) == ([b'{"a": 1}', b"not json", b'{"b": 2}'], None)
    cut = tmp_path / "requests-2.jsonl"
    cut.write_bytes(b'{"a": 1}\n{"b": ')
    read, error = _read_to_end(cut)
    assert read == [b'{"a": 1}'] and isinstance(error, ValueError), (read, error)
    assert str(cut) in str(error) and "line 2" in str(error)


def test_a_cut_shard_stops_the_verdict_and_is_named(tmp_path, capsys):
    """Shards cut alike on both sides leave a population that still joins: no verdict (exit 3)."""
    import zstandard

    harness = controls.write_profile(
        str(tmp_path / "h.yaml"), controls.CONTROL_PROFILES["anthropic"]
    )
    done = controls._anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")
    user = {"role": "user", "content": "go"}
    trace = tmp_path / "t"
    trace.mkdir()
    for shard in (1, 2):  # two shards a side, so that several workers read them in parallel
        ids = [f"t{shard}-{i}" for i in range(20)]
        records = {"requests": [controls._anth_request(t, [user], "2026-01-01T00:00:01Z")
                                for t in ids],
                   "responses": [controls._sse_response(t, done) for t in ids]}  # fmt: skip
        for kind, recs in records.items():
            text = "".join(json.dumps(r) + "\n" for r in recs).encode()
            if shard == 1:
                (trace / f"{kind}-1.jsonl").write_bytes(text)
            else:
                packed = zstandard.ZstdCompressor().compress(text)
                (trace / f"{kind}-2.jsonl.zst").write_bytes(packed[: len(packed) // 2])
    for workers in ("1", "2"):
        code = verdict.main(["--harness", harness, "--population", "window", "--claims",
                             "inventory", "--workers", workers, "--out-dir",
                             str(tmp_path / f"v{workers}"), str(trace)])  # fmt: skip
        captured = capsys.readouterr()
        assert code == 3 and "-2.jsonl.zst" in captured.err, (code, captured.out)


def test_cross_cases(tmp_path):
    results = controls.cross_cases(str(tmp_path))
    assert results and all(r["ok"] for r in results), results


@pytest.mark.skipif(os.environ.get("MH_RUN_CONTROLS") != "1", reason="set MH_RUN_CONTROLS=1")
def test_control_matrix(tmp_path):
    assert controls.main(["--out-dir", str(tmp_path)]) == 0
