# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nothing outside the musespark_open family reaches its target, except the router line.

With ``modeling_v2`` off the resolver returns before importing anything, so the
target can change no other model's behaviour -- the built-in MuseSparkOpen model's
included -- as long as no other module imports it. Read off disk as text, like
test_modeling_v2_claims.py: no module is imported beyond the package itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2

pytestmark = pytest.mark.cpu_only

_MODELING_V2 = Path(_modeling_v2.__file__).resolve().parent
_PACKAGE = _MODELING_V2.parents[2]
_FAMILY = _MODELING_V2 / "models" / "musespark_open"
_ROUTER_INDEX = _MODELING_V2 / "_router_index.py"

# A reference to the family's modules or to the target's classes.
_REFERENCE = re.compile(
    r"models[./]musespark_open\b|muse_spark_open_816b_a42b_nvfp4__sm_103__tep8"
    r"|ModelingV2MuseSparkOpen|MuseSparkOpenTep8ModelingV2"
)
_ROUTER_LINE = '"MuseSparkOpenOmniForCausalLM": "models.musespark_open.routing",'


def test_only_the_router_line_references_the_family():
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if _FAMILY in path.parents:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if not _REFERENCE.search(line):
                continue
            if path == _ROUTER_INDEX and line.strip() == _ROUTER_LINE:
                continue
            offenders.append(f"{path.relative_to(_PACKAGE.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "modules outside musespark_open reference it:\n" + "\n".join(offenders)


def test_the_router_line_is_present():
    lines = [line.strip() for line in _ROUTER_INDEX.read_text().splitlines()]
    assert lines.count(_ROUTER_LINE) == 1
