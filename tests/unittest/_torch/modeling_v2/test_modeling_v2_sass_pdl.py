# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PDL wait checker (`_sass_pdl`) on SASS / PTX shapes: the FIN reduce-scatter's (wait first,
then constant-bank reads and coherent loads), and the shapes ptxas produces when the wait is not
first (a non-coherent load hoisted above it), which it must flag."""

import pytest

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _sass_pdl import nc_loads, pre_wait_violations  # noqa: E402

pytestmark = pytest.mark.cpu_only

_PTX_OK = """
.visible .entry k(
)
{
\tld.param.u64 %rd1, [k_param_0];
\tmov.u32 %r1, %tid.x;
\tgriddepcontrol.wait;
\tld.relaxed.gpu.global.b32 %r2, [%rd1+4];
\tld.global.b32 %r3, [%rd2];
}
"""


def _sass(*lines: str) -> str:
    return "\n".join(
        f"        /*{16 * i:04x}*/                   {t} ;" for i, t in enumerate(lines)
    )


def test_wait_first_passes():
    sass = _sass(
        "LDC R1, c[0x0][0x37c]",
        "LDC.64 R2, c[0x0][0x490]",
        "LDCU.64 UR6, c[0x0][0x358]",
        "ACQBULK",
        "S2R R14, SR_CTAID.Y",
        "LDG.E.STRONG.GPU R2, desc[UR4][R2.64+0x4]",
        "LDG.E R31, desc[UR4][R20.64]",
    )
    assert pre_wait_violations(sass, _PTX_OK) == []
    assert nc_loads(sass) == 0


def test_hoisted_nc_load_is_flagged():
    sass = _sass(
        "LDC R1, c[0x0][0x37c]",
        "LDG.E.CONSTANT R31, desc[UR4][R20.64]",
        "STG.E.STRONG.SYS desc[UR4][R4.64], R6",
        "ACQBULK",
    )
    problems = pre_wait_violations(sass, _PTX_OK)
    assert any("LDG.E.CONSTANT" in p for p in problems)
    assert any("STG" in p for p in problems)
    assert nc_loads(sass) == 1


@pytest.mark.parametrize(
    "sass_lines, ptx, what",
    [
        (("LDC R1, c[0x0][0x37c]",), _PTX_OK, "0 ACQBULK"),
        (("ACQBULK", "ACQBULK"), _PTX_OK, "2 ACQBULK"),
        (("@P0 BRA 0x80", "ACQBULK"), _PTX_OK, "BRA"),
        (("ACQBULK",), _PTX_OK.replace("\tgriddepcontrol.wait;\n", ""), "0 griddepcontrol"),
        (
            ("ACQBULK",),
            _PTX_OK.replace("\tmov.u32", "\tld.global.nc.b32 %r9, [%rd3];\n\tmov.u32"),
            "PTX before",
        ),
    ],
)
def test_other_violations_are_flagged(sass_lines, ptx, what):
    problems = pre_wait_violations(_sass(*sass_lines), ptx)
    assert any(what in p for p in problems), problems
