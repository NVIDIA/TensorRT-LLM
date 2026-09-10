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
"""Nothing may cache an NCCL window buffer across the eager -> capture boundary.

The rule that keeps a captured graph from replaying into someone else's window
buffer is "do not hand one out while a capture is in progress". It is enforced
where the buffers are handed out, so it can only see a *request*. A buffer
obtained eagerly and then kept in a process-lifetime container defeats it
completely: the capture reuses the cached tensor, no request is made, nothing is
refused, and the eager-era address is baked into the graph.

That is not hypothetical. It is the shape the original campaign could never
explain -- v1 left 2 of 12 prompts collapsing -- and the reason a
claim-at-capture-open mechanism was built. Deleting that mechanism is only safe
while this property holds, so the property gets a test rather than a comment.

Pure source scan: no GPU, no build, no import of the extension.
"""

import re
from pathlib import Path

import pytest

CPP = Path(__file__).parents[4] / "cpp" / "tensorrt_llm"

# A container that outlives the call cannot hold a window buffer. Anything here
# needs a reason and a plan, not just an entry.
ALLOWED = {
    # The allocator IS the owner: its pool is the free list these buffers come
    # from, and it hands them out and takes them back. Excluding it is the point
    # of the scan, not a hole in it.
    "ncclUtils.cpp",
    "ncclUtils.h",
}

_STATIC_DECL = re.compile(
    r"\bstatic\b[^;{}\n]*\b(?:map|unordered_map|vector|set|unordered_set|optional|pair)\b[^;{}]*",
    re.MULTILINE)


def _offenders():
    found = []
    for path in sorted(CPP.rglob("*.cpp")) + sorted(CPP.rglob("*.h")):
        if path.name in ALLOWED:
            continue
        text = path.read_text(errors="ignore")
        if "NCCLWindowBuffer" not in text:
            continue
        for m in _STATIC_DECL.finditer(text):
            decl = m.group(0)
            if "NCCLWindowBuffer" in decl:
                line = text[:m.start()].count("\n") + 1
                found.append(f"  {path.name}:{line}: {' '.join(decl.split())[:110]}")
    return found


def test_the_scanner_can_see_the_type_at_all():
    """Guard against passing on an empty set.

    The type has to appear somewhere for the scan to mean anything; if it were
    renamed, every assertion below would pass while checking nothing.
    """
    hits = [p.name for p in CPP.rglob("*.h") if "NCCLWindowBuffer" in p.read_text(errors="ignore")]
    assert hits, "NCCLWindowBuffer not found in any header -- the scan is stale"


# Upstream has no violation today: the scan is here so that adding one fails
# rather than silently defeating the capture-time refusal. If a legitimate
# exception ever appears, record it here with a reason and a plan, and the
# staleness check below will force the entry to be removed once it is gone.
KNOWN: set[str] = set()


def test_no_new_process_lifetime_container_holds_a_window_buffer():
    offenders = _offenders()
    files = {line.strip().split(":")[0] for line in offenders}
    new = files - KNOWN
    assert not new, (
        "These cache an NCCL window buffer in a container that outlives the "
        "call, which defeats the capture-time refusal: the capture reuses the "
        "cached tensor, no request reaches the allocator, and the eager-era "
        "address is baked into the graph.\n" +
        "\n".join(l for l in offenders if l.strip().split(":")[0] in new))


def test_the_known_violation_is_still_there():
    """When it goes, delete it from KNOWN rather than leaving a stale entry.

    A KNOWN set that outlives what it describes silently widens the check.
    """
    files = {line.strip().split(":")[0] for line in _offenders()}
    stale = KNOWN - files
    assert not stale, f"KNOWN lists offenders that no longer exist: {stale}"
