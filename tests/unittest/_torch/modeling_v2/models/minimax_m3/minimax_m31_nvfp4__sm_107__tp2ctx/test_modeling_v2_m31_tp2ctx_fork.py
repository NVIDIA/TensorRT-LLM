# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp2ctx target's model is generated from the built-in MiniMax-M3 model it says it was (gen_tp2ctx_fork.py).

modeling.py records the git blob of the built-in modeling_minimaxm3.py it was generated from; the #20629 items are
merged into that source before generation, so the fork is no longer the built-in text verbatim (A1 on). The fork is
kept as generated from built-in blob 09913a5a, the source the served context worker ran; the built-in model on this
branch (blob dfe772f4) is newer, and regenerating the fork from it is a follow-up. This checks that the recorded blob
is that generating source and that the installed built-in is a known one (any other built-in change means: regenerate,
with the items merged again), that the fork registers no architecture and no legacy op name, and that its only
relative imports are its own modules. Reads source text only. CPU only.
"""

import hashlib
import re
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
import tensorrt_llm._torch.models as _models

pytestmark = pytest.mark.cpu_only

_FORK = (
    Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx" / "modeling.py"
)
_BUILTIN = Path(_models.__file__).resolve().parent / "modeling_minimaxm3.py"


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# The built-in modeling_minimaxm3.py the fork was generated from, and the built-in files known to sit next to it.
_GENERATED_FROM = "09913a5ae694fba590718476f12f0881bdf0a0ae"
_KNOWN_BUILTINS = {_GENERATED_FROM, "dfe772f4dc0461401790adf13138277c64dfe7fd"}


def test_the_fork_records_its_generating_builtin_model():
    recorded = re.search(r"^#   \S+ @ \S+ \(blob ([0-9a-f]{40})\)", _FORK.read_text(), flags=re.M)
    assert recorded is not None, (
        "modeling.py has no provenance header (generate it with gen_tp2ctx_fork.py)"
    )
    assert recorded.group(1) == _GENERATED_FROM
    assert _git_blob(_BUILTIN.read_bytes()) in _KNOWN_BUILTINS, (
        "the built-in modeling_minimaxm3.py changed again since the tp2ctx fork was generated: regenerate it "
        "(gen_tp2ctx_fork.py, with the items' #20629 hunks merged in again)"
    )


def test_the_fork_registers_no_architecture_and_no_legacy_op_name():
    fork = _FORK.read_text()
    assert "@register_auto_model(" not in fork
    assert "trtllm::minimax_m3_" not in fork
    relative = re.findall(r"^[ \t]*from \.\S*", fork, flags=re.M)
    # The fork imports its switches, the exact index scorer and the A2 tile select relatively
    # (from . import idx_scorer as _idx_scorer, from . import tsel as _tsel).
    assert relative == ["from .", "from .", "from .m31_switches"], relative
