# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests of the pre-wait SASS audit (``_sass_walk.py``) on hand-written listings.

The audit must follow control flow per warp role rather than listing order: a role
that never waits (a weight TMA warp that exits after its loads), a role whose wait
comes later in the listing than another role's, a role that leaves through a
``PLOP3``-guarded exit, and branches written to labels or to addresses.
"""

import pytest

__extra_import_path__ = ["."]
import _sass_walk as sass_walk  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _listing(*lines: str) -> str:
    body = "\n".join(
        f"        /*{address:04x}*/                   {text} ;" for address, text in lines
    )
    return "\n\tFunction : kernel\n" + body + "\n"


def _found(found, key):
    return [(ins.opcode, ins.address) for ins in found[key]]


TMA_ROLE = _listing(
    (0x00, "S2R R0, SR_TID.X"),
    (0x10, "SHF.R.U32.HI R1, RZ, 0x5, R0"),
    (0x20, "R2UR UR4, R1"),
    (0x30, "UISETP.NE.U32.AND UP0, UPT, UR4, 0x5, UPT"),
    (0x40, "BRA.U !UP0, 0x70"),
    (0x50, "ACQBULK"),
    (0x60, "BRA 0x90"),
    (0x70, "UTMALDG.3D [UR8], [UR10], desc[UR12]"),
    (0x80, "EXIT"),
    (0x90, "LDG.E R2, desc[UR6][R4.64]"),
    (0xA0, "EXIT"),
)


def test_a_role_that_never_waits_is_audited():
    groups = dict((tuple(warps), found) for warps, found in sass_walk.per_warp(TMA_ROLE, 256))
    assert set(groups) == {(0, 1, 2, 3, 4, 6, 7), (5,)}
    tma = groups[(5,)]
    assert _found(tma, "accesses") == [("UTMALDG.3D", 0x70)]
    assert _found(tma, "exits") == [("EXIT", 0x80)]
    others = groups[(0, 1, 2, 3, 4, 6, 7)]
    assert others["accesses"] == [] and others["exits"] == []
    # Any thread: the union.
    every = sass_walk.pre_wait(TMA_ROLE)
    assert _found(every, "accesses") == [("UTMALDG.3D", 0x70)]


LATER_WAIT = _listing(
    (0x00, "S2R R0, SR_TID.X"),
    (0x10, "ISETP.GE.U32.AND P0, PT, R0, 0x80, PT"),
    (0x20, "@P0 BRA `(.L_x_1)"),
    (0x30, "ACQBULK"),
    (0x40, "LDG.E R2, desc[UR6][R4.64]"),
    (0x50, "EXIT"),
    (0x60, "LDG.E.64.STRONG.GPU R4, desc[UR6][R6.64]"),
    (0x70, "ACQBULK"),
    (0x80, "STG.E desc[UR6][R8.64], R2"),
    (0x90, "EXIT"),
).replace("        /*0060*/", ".L_x_1:\n        /*0060*/")


def test_a_role_that_waits_later_in_the_listing_is_audited():
    groups = dict((tuple(warps), found) for warps, found in sass_walk.per_warp(LATER_WAIT, 256))
    assert _found(groups[(4, 5, 6, 7)], "accesses") == [("LDG.E.64.STRONG.GPU", 0x60)]
    assert groups[(0, 1, 2, 3)]["accesses"] == []
    assert all(found["exits"] == [] for found in groups.values())


PLOP3_EXIT = _listing(
    (0x00, "S2R R0, SR_TID.X"),
    (0x10, "SHF.R.U32.HI R1, RZ, 0x5, R0"),
    (0x20, "R2UR UR4, R1"),
    (0x30, "UISETP.GT.AND UP0, UPT, UR4, 0x3, UPT"),
    (0x40, "PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x8"),
    (0x50, "@P0 EXIT"),
    (0x60, "STG.E desc[UR4][R2.64], R5"),
    (0x70, "ACQBULK"),
    (0x80, "EXIT"),
)


def test_a_guarded_exit_is_taken_by_the_warps_it_guards():
    groups = dict((tuple(warps), found) for warps, found in sass_walk.per_warp(PLOP3_EXIT, 256))
    assert _found(groups[(4, 5, 6, 7)], "exits") == [("EXIT", 0x50)]
    assert groups[(4, 5, 6, 7)]["accesses"] == []
    assert _found(groups[(0, 1, 2, 3)], "accesses") == [("STG.E", 0x60)]
    assert groups[(0, 1, 2, 3)]["exits"] == []


def test_an_indirect_branch_is_reported():
    listing = _listing((0x00, "BRX R2 -0x10"), (0x10, "ACQBULK"), (0x20, "EXIT"))
    found = sass_walk.pre_wait(listing)
    assert _found(found, "indirect") == [("BRX", 0x00)]
