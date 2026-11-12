# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family's SM107 code reads no environment variable.

What a target runs is in its constants (``K3Sm107Constants``) and the family's launch constants
(``ops/common/sm107_launch.py``), not in the environment. Engine-wide decisions the family follows
(forced MNNVL all-reduce, the MNNVL test override) come from the engine's own helpers. Exempt:

* ``routing.py``, which reads the deployment's environment as the workers see it to decline a
  deployment the targets do not serve (``TRTLLM_ENABLE_PDL`` off);
* the MXFP4 sm100 targets and their kernels (``kernels/k3_*``), which predate the rule; those
  kernels may read only the engine-wide switches (PDL, forced MNNVL all-reduce).
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_FAMILY_DIR = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl"
)
_EXEMPT_TARGETS = {
    "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1",
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4",
}
ENGINE_WIDE = {"TRTLLM_ENABLE_PDL", "TRTLLM_FORCE_MNNVL_AR"}
_ROUTING = Path("routing.py")

_READ = re.compile(r"\b(?:os\.environ|os\.getenv|getenv)\b")
_LITERAL = re.compile(
    r"""(?:os\.environ\.get|os\.environ\[|os\.getenv)\(?\s*["']([A-Z0-9_]+)["']"""
)


def _sm100_kernel(rel: Path) -> bool:
    """A kernel of the MXFP4 sm100 targets (``kernels/k3_*``; the SM107 ones are ``k3r_*``)."""
    return len(rel.parts) > 2 and rel.parts[0] == "kernels" and rel.parts[1].startswith("k3_")


def _reads():
    for path in sorted(_FAMILY_DIR.rglob("*.py")):
        rel = path.relative_to(_FAMILY_DIR)
        if rel.parts[0] in _EXEMPT_TARGETS or "__pycache__" in rel.parts or rel == _ROUTING:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _READ.search(line) or "get_env_enable_pdl" in line:
                yield rel, f"{rel}:{number}", line, _LITERAL.findall(line)


def test_sm107_code_reads_no_environment_variable():
    bad = [
        f"{where}: {line.strip()}"
        for rel, where, line, names in _reads()
        if not _sm100_kernel(rel) or not names or not set(names) <= ENGINE_WIDE
    ]
    assert not bad, "environment reads in the Kimi K3 family:\n" + "\n".join(bad)
