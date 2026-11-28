# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a PDL kernel's `griddepcontrol.wait` sits in its compiled code (a test helper).

A kernel launched with programmatic dependent launch may start while its predecessor still writes
its inputs; it may read them only after `griddepcontrol.wait`. ptxas treats a non-coherent load
(`ld.global.nc`, SASS `LDG.E.CONSTANT`) as free to move across the wait (SASS `ACQBULK`) and across
volatile or strong stores once the wait is not the first statement of the entry block, while
coherent loads stay behind it. A source reading cannot show where the loads landed; the dumped SASS
can. `pre_wait_violations` lists what a kernel does before its wait that it must not:

* SASS: not exactly one `ACQBULK`; any global or generic memory instruction, branch or label before
  it (constant-bank `LDC` / `LDCU` parameter reads and special-register moves are fine);
* PTX: not exactly one `griddepcontrol.wait`; a global load or store, an atomic or a branch above
  it.

Compile with `CUTE_DSL_KEEP=ptx,cubin,sass` and `CUTE_DSL_DUMP_DIR=<dir>` (CuTe DSL dumps
`<name>.sass` and `<name>.<arch>.ptx` there) and pass the two texts.
"""

from __future__ import annotations

import re

#: SASS memory instructions that touch global or generic memory (constant-bank loads excluded).
_SASS_MEM = re.compile(
    r"^(LDG|LD|ST|STG|ATOM|ATOMG|RED|REDG|UTMALDG|UBLKCP|LDGSTS|CCTL|LDGDEPBAR|SUST|SULD|TLD|TEX)\b"
)
_SASS_LINE = re.compile(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_SASS_LABEL = re.compile(r"^\.L_x_\d+:")
_PTX_MEM = re.compile(
    r"\b(ld|st|atom|red|cp\.async|prefetch)\.(global|relaxed|acquire|volatile|nc)|ld\.global|\bbra\b"
)


def _opcode(text: str) -> str:
    """The instruction's opcode, past a predicate guard (`@P0`, `@!P1`)."""
    words = text.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return words[0] if words else ""


def sass_instructions(sass: str) -> list[tuple[str, str]]:
    """`(address, text)` of every instruction of a SASS listing, labels as `("label", name)`."""
    out = []
    for line in sass.splitlines():
        m = _SASS_LINE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
        elif _SASS_LABEL.match(line.strip()):
            out.append(("label", line.strip()))
    return out


def pre_wait_violations(sass: str, ptx: str) -> list[str]:
    """What the kernel does before its `griddepcontrol.wait` that a PDL launch must not (empty:
    the wait comes before every global access, branch and label)."""
    problems = []
    ins = sass_instructions(sass)
    waits = [i for i, (_, t) in enumerate(ins) if _opcode(t) == "ACQBULK"]
    if len(waits) != 1:
        problems.append(f"SASS has {len(waits)} ACQBULK, expected 1")
    pre = ins[: waits[0]] if waits else ins
    for addr, text in pre:
        if addr == "label":
            problems.append(f"SASS label before ACQBULK: {text}")
        elif _SASS_MEM.match(_opcode(text)) or "BRA" in text or "BSSY" in text:
            problems.append(f"SASS {addr} before ACQBULK: {text}")
    lines = ptx.splitlines()
    pwaits = [i for i, line in enumerate(lines) if "griddepcontrol.wait" in line]
    if len(pwaits) != 1:
        problems.append(f"PTX has {len(pwaits)} griddepcontrol.wait, expected 1")
    else:
        body = next((i for i, line in enumerate(lines) if line.strip() == "{"), 0)
        for line in lines[body : pwaits[0]]:
            s = line.strip()
            if _PTX_MEM.search(s) or re.match(r"^\$L__BB", s):
                problems.append(f"PTX before griddepcontrol.wait: {s}")
    return problems


def nc_loads(sass: str) -> int:
    """`LDG.E.CONSTANT` instructions (`ld.global.nc`) in a SASS listing."""
    return sum(1 for _, t in sass_instructions(sass) if "LDG" in t and ".CONSTANT" in t)
