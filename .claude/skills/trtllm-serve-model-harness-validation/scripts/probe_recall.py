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
r"""A clean long-context recall probe against a served model, before any agent runs on it.

A unique passphrase sits inside natural filler at 2% and at 50% depth of a prompt of each target
size, and the model is asked for it at temperature 0. Recall that fails at one size while smaller
ones hold points at a size the model or the serving configuration fixes (a prefill chunk, a sparse
attention candidate pool, a position table), not at the model (references/smoke.md section 3c).
Probe on both sides of each such size, up to the served max_seq_len.

    probe_recall.py --base-url http://127.0.0.1:8000 --model <served name> \
        --sizes 1500,3000,7000,10000,60000,250000 [--route chat|messages] [--extra JSON]

- Sizes are targets in tokens, from a rough ratio; each line reports the server's own prompt count.
- ``--extra`` merges JSON into every request body. Turn thinking off with it where the model allows,
  or the answer may be spent on reasoning: ``'{"chat_template_kwargs": {"enable_thinking": false}}'``
  (Chat, for templates that read it) or ``'{"thinking": {"type": "disabled"}}'`` (Messages). For a
  model that always thinks, ask for its lowest effort and raise ``--max-tokens``.
- One line per probe; the model's answer is never printed, only its length.
- Exit 0 when every probe recalled its passphrase, 1 otherwise. A refused request (e.g. a size
  past ``max_seq_len``) is a probe that did not recall. Exit 2 when ``--sizes`` names no positive
  size: a probe that sent nothing recalled nothing.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

# Filler that never uses the passphrase's words, so it tests recall and not distractors.
FILLER = [
    "The river bends twice before it reaches the old mill.",
    "Farmers in the valley rotate barley and clover.",
    "A lighthouse keeper logs the weather every evening.",
    "The museum reopened its east wing after repairs.",
    "Train schedules change in winter to avoid icy tracks.",
    "Bakers start before dawn to prepare the bread.",
    "The orchestra rehearsed the second movement slowly.",
    "Engineers inspected the bridge cables in spring.",
    "Students gathered in the library to study geometry.",
    "Merchants traded wool and salt at the harbor market.",
]
TOKENS_PER_SENTENCE = 11.5  # about right for common tokenizers; the reply's usage has the truth
DEPTHS = (0.02, 0.5)


def build_prompt(target_tokens, depth, rng):
    """``(passphrase, prompt)``: the passphrase at ``depth`` of about ``target_tokens`` of filler."""
    passphrase = "TANGERINE-%04d" % rng.randint(0, 9999)
    sentences = [
        rng.choice(FILLER) for _ in range(max(1, int(target_tokens / TOKENS_PER_SENTENCE)))
    ]
    at = int(len(sentences) * depth)
    text = " ".join(
        sentences[:at] + [f"Note: the vault passphrase is {passphrase}."] + sentences[at:]
    )
    question = (
        "What is the vault passphrase mentioned in the text above? Reply with the passphrase only."
    )
    return passphrase, f"{text}\n\n{question}"


def ask(base_url, route, model, prompt, max_tokens, extra, timeout):
    """``(answer text, the server's prompt token count)`` for one request."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        **extra,
    }
    headers = {"content-type": "application/json"}
    if route == "messages":
        headers["anthropic-version"] = "2023-06-01"
    url = base_url.rstrip("/") + ("/v1/messages" if route == "messages" else "/v1/chat/completions")
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    usage = reply.get("usage") or {}
    if route == "messages":
        text = "".join(
            b.get("text", "") for b in reply.get("content") or [] if b.get("type") == "text"
        )
        return text, usage.get("input_tokens")
    message = ((reply.get("choices") or [{}])[0]).get("message") or {}
    return message.get("content") or "", usage.get("prompt_tokens")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", required=True, help="the server, without /v1")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--sizes", required=True, help="comma-separated target prompt sizes in tokens")
    ap.add_argument("--route", choices=("chat", "messages"), default="chat")
    ap.add_argument("--extra", type=json.loads, default={}, help="JSON merged into every body")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per request")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    try:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError:
        sizes = []
    if not sizes or min(sizes) <= 0:
        ap.error(f"--sizes {args.sizes!r}: give one or more positive token counts, e.g. 1500,3000")
    rng = random.Random(args.seed)
    failed = 0
    for size in sizes:
        for depth in DEPTHS:
            passphrase, prompt = build_prompt(size, depth, rng)
            start = time.monotonic()
            note = ""
            try:
                answer, prompt_tokens = ask(
                    args.base_url, args.route, args.model, prompt, args.max_tokens, args.extra,
                    args.timeout,
                )  # fmt: skip
            except urllib.error.HTTPError as exc:
                answer, prompt_tokens, note = "", None, f" refused: HTTP {exc.code}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                answer, prompt_tokens, note = "", None, f" failed: {type(exc).__name__}"
            recalled = passphrase in answer
            failed += not recalled
            print(
                f"target {size:>8} prompt_tokens {str(prompt_tokens):>8} depth {depth:.2f} "
                f"recalled {'yes' if recalled else 'no':<3} answer_chars {len(answer):>5} "
                f"{time.monotonic() - start:7.1f}s{note}",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
