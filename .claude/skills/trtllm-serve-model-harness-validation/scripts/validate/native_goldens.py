#!/usr/bin/env python3
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
r"""Golden renders for the native readers (``native.py``), made by each checkpoint's own encoder.

Synthetic assistant messages are rendered by the model's encoder exactly as a
past turn is rendered into a prompt. The text after the generation prompt is
what the model is trained to emit; ``native.read`` must recover the message from
it. For DeepSeek V4.1 the vendor's own completion parser is run as well and its
reading is recorded beside the expected one.

    native_goldens.py --glm <ckpt dir> --check

reads every render of the checkpoint you serve back with its model profile's own reader, and
compares it with the shipped goldens (``tests/fixtures/native_goldens.json``). It writes nothing,
prints each encoder file's ``template_sha256`` (the first 16 hex digits of its SHA-256, as model
profiles record it), and exits 1 when a render does not read back exactly. A render that reads
back exactly but differs from the shipped one passes and is named (DIFFERS): the shipped goldens
pin the reader, not a checkpoint, and the reader has just read this checkpoint's own render. Run
it before trusting C-13 on a checkpoint whose ``template_sha256`` the profile does not list.

    native_goldens.py --kimi-k3 <ckpt dir> --glm <ckpt dir> --deepseek-v41 <ckpt dir> --out <file>

writes the goldens. Regenerate the shipped fixture with it only when a reader changes.

Only the synthetic cases below are rendered; nothing is read from traces.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

CASES = [
    {"id": "text_only", "reasoning": "Plan the answer.", "content": "The answer is 4.", "calls": []},
    {"id": "one_call", "reasoning": "Read the file first.", "content": "Reading it.",
     "calls": [{"name": "read", "arguments": {"path": "/tmp/a.txt"}}]},
    {"id": "typed_args", "reasoning": "Two calls.", "content": "",
     "calls": [{"name": "exec", "arguments": {"cmd": 'echo "a & b" <c>\nline 2 é', "timeout": 30,
                                              "dry": False, "env": {"K": "v"}, "argv": ["x", 1],
                                              "none": None}},
               {"name": "wait", "arguments": {"ms": 1.5}}]},
    {"id": "empty_reasoning", "reasoning": "", "content": "Done.", "calls": []},
    {"id": "no_args", "reasoning": "Just list.", "content": "", "calls": [{"name": "ls", "arguments": {}}]},
]  # fmt: skip


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


def _openai_message(case):
    return {"role": "assistant", "content": case["content"], "reasoning_content": case["reasoning"],
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                           for i, c in enumerate(case["calls"])] or None}  # fmt: skip


def _expected(case, value_text, type_of, content=None):
    return {
        "reasoning": case["reasoning"],
        "content": case["content"] if content is None else content,
        "calls": [{"name": c["name"], "args": [[k, value_text(v), type_of(v)]
                                               for k, v in c["arguments"].items()]}
                  for c in case["calls"]],
    }  # fmt: skip


def kimi_k3(ckpt):
    enc = _load(os.path.join(ckpt, "encoding_k3.py"), "encoding_k3")
    prefix = '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'
    out = []
    for case in CASES:
        messages = [{"role": "user", "content": "go"}, _openai_message(case)]
        messages[1]["tool_calls"] = [dict(tc, function=dict(tc["function"],
                                                            arguments=json.loads(tc["function"]["arguments"])))
                                     for tc in messages[1]["tool_calls"] or []] or None  # fmt: skip
        segments = enc.build_chat_segments(messages, add_generation_prompt=False, thinking=True)
        text = "".join(s.text for s in segments)
        generated = text[text.rindex(prefix) + len(prefix) :]
        out.append({"model": "kimi-k3", "case": case["id"], "text": generated,
                    "expected": _expected(case, enc._xtml_value, enc._xtml_type)})  # fmt: skip
    return out


def _hf_environment():
    """The jinja environment transformers renders chat templates with (apply_chat_template)."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)  # fmt: skip

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])  # fmt: skip
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def glm(ckpt):
    with open(os.path.join(ckpt, "chat_template.jinja")) as fh:
        template = _hf_environment().from_string(fh.read())
    out = []
    for case in CASES:
        message = _openai_message(case)
        for tc in message["tool_calls"] or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        text = template.render(messages=[{"role": "user", "content": "go"}, message],
                               add_generation_prompt=False)  # fmt: skip
        marker = "<|assistant|><think>"
        generated = text[text.rindex(marker) + len(marker) :]
        out.append({"model": "glm-5", "case": case["id"], "text": generated,
                    "expected": _expected(case, lambda v: v if isinstance(v, str)
                                          else json.dumps(v, ensure_ascii=False), lambda v: None,
                                          content=case["content"].strip())})  # fmt: skip
    return out


def deepseek_v41(ckpt):
    enc = _load(os.path.join(ckpt, "encoding", "encoding.py"), "encoding_v41")
    out = []
    for case in CASES:
        messages = [{"role": "user", "content": "go"}, _openai_message(case)]
        text = enc.encode_messages(messages, thinking_mode="thinking", drop_thinking=False)
        marker = enc.ASSISTANT_SP_TOKEN + enc.thinking_start_token
        generated = text[text.rindex(marker) + len(marker) :]
        vendor = enc.parse_message_from_completion_text(generated, thinking_mode="thinking")
        out.append({"model": "deepseek-v41", "case": case["id"], "text": generated,
                    "expected": _expected(case, lambda v: v if isinstance(v, str) else enc.to_json(v),
                                          lambda v: "string" if isinstance(v, str) else "json"),
                    "vendor": {"reasoning": vendor["reasoning_content"], "content": vendor["content"],
                               "calls": [{"name": tc["function"]["name"],
                                          "arguments": json.loads(tc["function"]["arguments"])}
                                         for tc in vendor["tool_calls"] or []]}})  # fmt: skip
    return out


SKILL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIPPED = os.path.join(SKILL, "tests", "fixtures", "native_goldens.json")
PROFILES = os.path.join(SKILL, "profiles", "model")
# The encoder each checkpoint renders with, as its model profile names it.
ENCODER_FILES = {"kimi_k3": "encoding_k3.py", "glm": "chat_template.jinja",
                 "deepseek_v41": os.path.join("encoding", "encoding.py")}  # fmt: skip


def template_sha256(path):
    """The first 16 hex digits of a file's SHA-256: a model profile's ``template_sha256``."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def read_back(golden, profile):
    """Why ``golden`` does not read back exactly with the profile's own reader, or None."""
    import native
    import profiles

    turn = native.read(profiles.native_card(profile), golden["text"])
    got = {"reasoning": turn.reasoning, "content": turn.content,
           "calls": [{"name": c.name, "args": [list(a) for a in c.args]} for c in turn.calls]}  # fmt: skip
    if turn.anomalies:
        return "anomalies: " + ", ".join(turn.anomalies)
    if got != golden["expected"]:
        return "reads back as another message"
    if "vendor" in golden:  # the vendor's own parser must read the render the same way
        calls = [{"name": c.name, "arguments": {k: v if t == "string" else json.loads(v)
                                                for k, v, t in c.args}} for c in turn.calls]  # fmt: skip
        if calls != golden["vendor"]["calls"] or turn.content != golden["vendor"]["content"]:
            return "the vendor's parser reads it differently"
    return None


def check(goldens, shipped_path=SHIPPED, profile_dir=PROFILES):
    """Print, per model, how many renders read back and equal the shipped ones.

    Returns 1 when a render does not read back, or when there is no render at all: a check that
    checked nothing has not passed.
    """
    import profiles

    with open(shipped_path) as fh:
        shipped = {(g["model"], g["case"]): g["text"] for g in json.load(fh)["goldens"]}
    if not goldens:
        print("no renders: nothing was checked")
        return 1
    failures = 0
    for model in sorted({g["model"] for g in goldens}):
        profile = profiles.load_profile(os.path.join(profile_dir, model + ".yaml"))
        renders = [g for g in goldens if g["model"] == model]
        problems = [(g["case"], read_back(g, profile)) for g in renders]
        problems = [(case, why) for case, why in problems if why]
        differ = [g["case"] for g in renders if shipped.get((model, g["case"])) != g["text"]]
        print(f"{model}: {len(renders)} renders, {len(renders) - len(problems)} read back exactly, "
              f"{len(renders) - len(differ)} equal to the shipped goldens")  # fmt: skip
        for case, why in problems:
            print(f"  FAIL {case}: {why}")
        for case in (c for c in differ if c not in dict(problems)):
            print(f"  DIFFERS {case}: reads back exactly, but is not the shipped render")
        failures += len(problems)
    return 1 if failures else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--kimi-k3")
    ap.add_argument("--glm")
    ap.add_argument("--deepseek-v41")
    ap.add_argument("--out", help="write the goldens to this file")
    ap.add_argument("--check", action="store_true",
                    help="write nothing: read every render back and compare it with the shipped goldens")  # fmt: skip
    args = ap.parse_args(argv)
    if bool(args.out) == args.check:
        ap.error("give exactly one of --out and --check")
    if not (args.kimi_k3 or args.glm or args.deepseek_v41):
        ap.error("name at least one checkpoint: --kimi-k3, --glm or --deepseek-v41")
    goldens = []
    sources = {}
    for flag, fn in (("kimi_k3", kimi_k3), ("glm", glm), ("deepseek_v41", deepseek_v41)):
        ckpt = getattr(args, flag)
        if ckpt:
            print(f"{flag}: {ENCODER_FILES[flag]} template_sha256 "
                  f"{template_sha256(os.path.join(ckpt, ENCODER_FILES[flag]))}")  # fmt: skip
            goldens += fn(ckpt)
            sources[flag] = os.path.basename(os.path.normpath(ckpt))
    if args.check:
        return check(goldens)
    with open(args.out, "w") as fh:
        json.dump({"schema": "trtllm-serve-model-harness-validation/native-goldens@1", "sources": sources,
                   "goldens": goldens}, fh, indent=1, ensure_ascii=False)  # fmt: skip
        fh.write("\n")
    print(f"{len(goldens)} goldens -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
