<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Verify a logic fix by property

An example test pins down the inputs someone thought of. A property states what must hold for
every input, and a generator goes looking for the inputs nobody thought of. Use it for a fix to
pure Python logic: a tool or reasoning parser, a protocol or serialization path, request
validation, a validator rule. The output depends only on the input, with no GPU, model or
network.

**The rule that makes it worth anything:** before reporting a pass, show that the property
catches planted bugs. A property no bug can break is worse than none: its green run buys false
confidence.

## 1. Pick the shape

| Shape | What must hold | Examples here |
|---|---|---|
| Two paths agree | For every input, both paths give the same observable result. The oracle is free: you never say what the right answer is | Streaming vs one-shot parsing of the same text; the Chat and Responses renderings of one turn; the served parser vs the model's own encoder or SDK reading the same text |
| Round trip | `decode(encode(x)) == x`. If `encode` normalizes, use `encode(decode(encode(x))) == encode(x)` | Render a call in the model's format, then parse it |
| Invariant | A relation holds after every operation, whatever the sequence | Offsets stay in bounds; totals are conserved; order is preserved |
| Idempotence | Applying it twice changes nothing | `finish()` called twice; defaults applied twice |
| The old implementation as oracle | The new path agrees with the one it replaces | A refactor |

Skip, and say why:
- a float with a tolerance (that is an accuracy test);
- GPU numerics;
- a function with one plausible input (an example test is clearer);
- a property you can only state by restating the code. It mirrors the code and passes whatever
  the code does.

## 2. Derive it from the tests, then vary what they hold fixed

The checked-in tests already say what the format is. Take their inputs (texts, tools, schemas)
and generate over the dimension they never vary. That is almost always where the bug is:
- where the chunk boundaries fall;
- which order the fields arrive in;
- how many deltas carry the same content.

## 3. Bound the generator to what production can reach

An unbounded generator reports failures nobody can trigger. Constrain it to the real input
alphabet first, then widen it on purpose:
- special-token markers are single tokens, so a stream never splits them;
- a stream can stop early (a length limit, a cancellation), so a prefix is a reachable input;
- tools and schemas come from the request, so take them from the suite's fixtures.

Label every finding `reachable`, `reachable only at <condition>` or `generator-only`.

## 4. Gate it with planted bugs

Keep a `.mutations` file beside the test. Each entry is a bug a reviewer could plausibly
introduce:

```text
[mutation]
name = drop-trailing-args
file = tensorrt_llm/serve/tool_parser/base_tool_parser.py
find = argument_diff = cur_args_json[sent:]
replace = argument_diff = ""
```

`scripts/mutate.py` plants each bug in turn, re-runs the property and reports `DETECTED` or
`MISSED`. Run it from the repository root; `file` is relative to that root:

```bash
python3 .claude/skills/trtllm-serve-model-harness-validation/scripts/mutate.py <file.mutations> \
    --runner <property runner> [--runner-args "..."] [--python <interpreter with hypothesis>]
```

- **Start from a green baseline.** With a failing baseline, `DETECTED` proves nothing, and the
  runner says so.
- **`find` must match exactly once.**
- **A `MISSED` names behaviour the property does not constrain.** Strengthen the property, or
  record in the `.mutations` file why the gap is acceptable. Never delete the mutation, and
  never reword the property so that it passes.
- **Shared code is invisible to a two-path property by construction.** A bug planted in code
  both paths share changes both. Note it in the file.
- **The gate writes each bug into the real file, then restores it.** It restores on an error,
  Ctrl-C and SIGTERM, but nothing restores after SIGKILL. A gate over tens of mutations runs for
  tens of minutes, so give it no short time limit. After any run that did not print its
  `SUMMARY`, check that every `find` still matches once before you commit anything.

## 5. Read a counter-example back into the code

Hypothesis shrinks a failure to a minimal input. Before calling it a bug:
1. Map it to a code path and `file:line`.
2. Check whether production can reach it (section 3).
3. Group failures by root cause. Twenty red targets are usually two causes.
4. Reproduce it as a plain pytest test in the suite the code already has. That test is the
   regression test, and it must fail on the old code (`SKILL.md` section 5).

## 6. Keep it, and say what it is

- **Keep it beside the tests it generalizes.**
  - For serving code: a pytest file under `tests/unittest/` with `hypothesis` imported.
  - For the validator: this skill's `tests/test_validator_properties.py`, which is the worked
    example.

  The `.mutations` file stays next to it, as the record of what the property constrains.
- **Hypothesis settings.** Use `@settings(deadline=None)` for anything that parses, and build
  the parser inside the property, so no state leaks between examples.
- **Say what it is.** "No counter-example in N generated inputs within <bounds>", never
  "verified" or "proved": a generator samples the input space, it does not cover it.
- **Exit codes:**

  | Code | Meaning |
  |---|---|
  | 0 | Every check passed |
  | 1 | A check ran and failed |
  | 2 | Usage error |
  | 3 | UNAVAILABLE: a tool could not run, so nothing was checked. This is neither a pass nor a failure |
