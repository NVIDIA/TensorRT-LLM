# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter-switch line on CPU.

At startup each helix8 generation rank logs, after the constants line and before the lever
variables, one line with every ``draft_*`` generation constant as the constants line states it
(a detail inside its parent's value), how many of the on / off ones are on, and ``fp``, the first
12 hex digits of the SHA-256 of the listed items: an A/B readout tells two drafter configurations
apart by that field. The README shows the line.

Read from the target's modeling.py (``ast``) and run over stand-ins; nothing is imported from it.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import pytest

__extra_import_path__ = ["."]
from test_rubin_helix8_draft_window import (  # noqa: E402
    _MODELING,
    _README,
    _constants,
    _exec,
    _tree,
)

pytestmark = pytest.mark.cpu_only

_LINE = re.compile(
    r"Kimi K3\.1 Rubin DFlash2 drafter switches: kimi_k31_pro_nvfp4__sm_107__helix8/generation "
    r"fp=(?P<fp>[0-9a-f]{12}) on=(?P<on>\d+)/(?P<flags>\d+): (?P<items>.*)"
)


def _ns() -> dict:
    tree = _tree(_MODELING)
    ns = _constants(tree)
    ns["hashlib"] = hashlib
    return _exec(ns, tree, "constants_line", "drafter_line", "log_constants")


def test_the_drafter_line_states_every_drafter_constant():
    ns = _ns()
    generation = ns["_CONSTANTS"]["generation"]
    m = _LINE.fullmatch(ns["drafter_line"]("generation"))
    assert m, ns["drafter_line"]("generation")
    items = m["items"].split(", ")
    keys = sorted(k for k in generation if k.startswith("draft_"))
    parents = [k for k in keys if ns["_DETAILS"].get(k) not in generation]
    assert [i.split("=", 1)[0] for i in items] == parents
    # Every item is the constants line's, details inside their parent's value.
    constants = ns["constants_line"]("generation").split(": ", 1)[1].split(", ")
    assert set(items) <= set(constants)
    for key in keys:
        assert f"{key}={generation[key]}" in m["items"], key
    flags = [generation[k] for k in keys if isinstance(generation[k], bool)]
    assert (int(m["on"]), int(m["flags"])) == (sum(flags), len(flags))
    assert m["fp"] == hashlib.sha256(m["items"].encode()).hexdigest()[:12]


def test_the_fingerprint_moves_with_any_drafter_switch():
    ns = _ns()
    base = _LINE.fullmatch(ns["drafter_line"]("generation"))["fp"]
    generation = dict(ns["_CONSTANTS"]["generation"])
    for key in (k for k in generation if k.startswith("draft_")):
        value = generation[key]
        flipped = (not value) if isinstance(value, bool) else f"{value}x"
        ns["_CONSTANTS"] = {"generation": {**generation, key: flipped}}
        assert _LINE.fullmatch(ns["drafter_line"]("generation"))["fp"] != base, key
    # A constant outside the drafter does not move it.
    ns["_CONSTANTS"] = {"generation": {**generation, "fc12_mma_n": 128}}
    assert _LINE.fullmatch(ns["drafter_line"]("generation"))["fp"] == base


def test_the_startup_lines_put_the_drafter_line_after_the_constants():
    ns = _ns()
    lines = []
    ns["logger"] = SimpleNamespace(info=lines.append)
    ns["log_constants"]("generation", lever_env={})
    assert lines[0].startswith("Kimi K3.1 Rubin constants: ")
    assert _LINE.fullmatch(lines[1]), lines[1]
    assert lines[2].startswith("Kimi K3.1 Rubin lever env: ")


#: The drafter switches as shipped. On: each cuts >= 1 % of the drafter's step time at some batch
#: of 4..11 and slows none of 2..13 by more than 0.5 % (row-gated where it would). B: the
#: drafter's bits; D: the same dtypes, another FP32 summation order (the acceptance length is
#: re-checked). Off: below that bar, or a census.
_SHIPPED = {
    "draft_agree_check": False,  # census
    "draft_agree_zero_padded": True,  # B
    "draft_block_rows": False,  # below the bar
    "draft_conv_norm": True,  # B
    "draft_conv_pdl": False,  # below the bar
    "draft_greedy_tokens": True,  # B
    "draft_history_aux": False,  # below the bar
    "draft_index_pairs": True,  # B
    "draft_lowm_gemm": True,  # B plan
    "draft_lowm_gemm_d_class": True,  # D plan
    "draft_noise_block": True,  # B
    "draft_norm_ctas": 7,  # D
    "draft_pairs_zero_padded": True,  # B
    "draft_q_a_kv_a_bands": "1-78",  # D at 6 / 12 rows, B at 18-78
    "draft_q_a_kv_a_fused": True,
    "draft_select_accept": True,  # B
    "draft_tail_rewind": True,  # B
    "draft_window_mla": True,  # D
    "draft_window_mla_bands": "1-30,70-120",
}


def test_the_stack_ships_the_kept_drafter_switches_on():
    ns = _ns()
    generation = ns["_CONSTANTS"]["generation"]
    for key, value in _SHIPPED.items():
        assert generation[key] == value and type(generation[key]) is type(value), key
    line = ns["drafter_line"]("generation")
    assert "draft_window_mla=True(draft_window_mla_bands=1-30,70-120)" in line
    assert "draft_q_a_kv_a_fused=True(draft_q_a_kv_a_bands=1-78)" in line
    assert "draft_lowm_gemm=True(draft_lowm_gemm_d_class=True)" in line
    assert "draft_conv_norm=True(draft_norm_ctas=7)" in line


def test_the_readme_shows_the_drafter_line():
    readme = _README.read_text()
    assert (
        "Kimi K3.1 Rubin DFlash2 drafter switches: kimi_k31_pro_nvfp4__sm_107__helix8/generation "
        "fp=<12 hex> on=<n>/<m>: draft_agree_check=<value>, ..."
    ) in readme
