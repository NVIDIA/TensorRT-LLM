# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SASS wait-before-exit audit (``_sass_wait_audit``) on hand-written listings.

Host-only: the listings follow ``cuobjdump -sass``'s layout; the schedule test
runs the audit on the kernels of the captured decode graphs.
"""

from __future__ import annotations

import pytest

__extra_import_path__ = ["."]
from _sass_wait_audit import audit, parse, parse_ncu  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _listing(name: str, body: str) -> str:
    lines = ["", "\tcode for sm_103a", f"\t\tFunction : {name}", '\t.headerflags\t@"EF_CUDA_SM103"']
    address = 0
    for row in body.strip().splitlines():
        row = row.strip()
        if row.endswith(":"):
            lines.append(row)
            continue
        lines.append(
            f"        /*{address:04x}*/                   {row} ;   /* 0x000fe20000000800 */"
        )
        lines.append("                                                   /* 0x000fe20000000800 */")
        address += 0x10
    return "\n".join(lines) + "\n"


def _audit(body: str, threads: int = 32) -> dict:
    ((instructions, labels),) = parse(_listing("k", body)).values()
    return audit(instructions, labels, threads)


def test_a_wait_before_the_only_exit_passes():
    result = _audit(
        """
        LDC R1, c[0x0][0x37c]
        LDG.E.128.CONSTANT R4, desc[UR4][R2.64]
        ACQBULK
        LDG.E R8, desc[UR4][R6.64]
        STG.E desc[UR4][R6.64], R8
        PREEXIT
        EXIT
        .L_x_0:
        BRA `(.L_x_0)
        """
    )
    assert result["ok"], result
    assert result["acqbulk"] == 1 and result["preexit"] == 1
    # The weight load before the wait is reported; nothing after it is.
    assert result["global_loads_before_wait"] == {"LDG.E.128.CONSTANT": 1}
    assert result["global_stores_before_wait"] == {}


def test_an_exit_on_a_loaded_value_before_the_wait_fails():
    result = _audit(
        """
        LDG.E R0, desc[UR4][R2.64]
        ISETP.GE.AND P0, PT, R0, 0x10, PT
        @P0 EXIT
        ACQBULK
        EXIT
        """
    )
    assert not result["ok"]
    assert result["exit_without_wait"] == ["0x0020"]


def test_an_exit_of_the_ctas_past_the_work_passes_when_cta_0_waits():
    # CTAs 16.. exit at once; CTA 0 waits, so no launch completes before its predecessor.
    result = _audit(
        """
        S2R R0, SR_CTAID.X
        ISETP.GE.AND P0, PT, R0, 0x10, PT
        @P0 EXIT
        ACQBULK
        EXIT
        """
    )
    assert result["ok"] and result["waiting_thread"] == 0, result
    assert result["every_thread"]["exit_without_wait"] == ["0x0020"]


WARP_ROLES = """
    S2R R20, SR_TID.X
    SHF.R.U32.HI R0, RZ, 0x5, R20
    ISETP.NE.U32.AND P0, PT, R0, RZ, PT
    @P0 BRA 0x70
    UTMALDG.3D [UR8], [UR4]
    PREEXIT
    EXIT
    ISETP.NE.U32.AND P0, PT, R0, 0x1, PT
    @P0 EXIT
    ACQBULK
    EXIT
"""


def test_a_warp_specialized_kernel_passes_when_one_role_waits():
    # Warp 0 loads weights and exits; warp 1 waits (a split-K GEMM's producer / loader roles).
    alone = _audit(WARP_ROLES)
    assert not alone["ok"] and alone["threads_walked"] == [0], alone
    assert alone["exit_without_wait"] == ["0x0060"]
    result = _audit(WARP_ROLES, threads=128)
    assert result["ok"] and result["waiting_thread"] == 32, result
    assert result["threads_walked"] == [0, 32]
    assert result["global_loads_before_wait"] == {"UTMALDG.3D": 1}


def test_a_dry_launch_exit_is_reported_as_launch_uniform():
    result = _audit(
        """
        LDC R1, c[0x0][0x37c]
        LDCU UR4, c[0x0][0x414]
        ISETP.NE.U32.AND P0, PT, RZ, UR4, PT
        @P0 EXIT
        S2R R17, SR_TID.X
        ACQBULK
        EXIT
        """
    )
    assert result["ok"], result
    assert result["exit_without_wait"] == []
    assert result["launch_uniform_exits"] == [{"address": "0x0030", "reads": ["c[0x0][0x414]"]}]


def test_an_exit_on_a_thread_index_is_not_launch_uniform():
    result = _audit(
        """
        S2R R0, SR_TID.X
        LDCU UR4, c[0x0][0x414]
        ISETP.NE.U32.AND P0, PT, R0, UR4, PT
        @P0 EXIT
        ACQBULK
        EXIT
        """
    )
    # Thread 0 exits unless the parameter is 0: not decided by the launch alone.
    assert not result["ok"] and result["launch_uniform_exits"] == []
    assert result["exit_without_wait"] == ["0x0030"]


def test_address_targets_are_followed_and_unknown_ones_fail():
    result = _audit("@P0 BRA 0x30\nACQBULK\nEXIT\nACQBULK\nEXIT")
    assert result["ok"], result
    unresolved = _audit("@P0 BRA 0x38\nACQBULK\nEXIT")
    assert not unresolved["ok"] and unresolved["unresolved_branches"] == ["0x0000"]


def test_a_branch_around_the_wait_fails():
    result = _audit(
        """
        ISETP.NE.AND P0, PT, R0, RZ, PT
        @P0 BRA `(.L_x_1)
        ACQBULK
        .L_x_1:
        STG.E desc[UR4][R6.64], R8
        EXIT
        """
    )
    assert not result["ok"]
    assert result["global_stores_before_wait"] == {"STG.E": 1}


def test_branches_that_all_meet_a_wait_pass():
    result = _audit(
        """
        @P0 BRA `(.L_x_1)
        ACQBULK
        BRA `(.L_x_2)
        .L_x_1:
        BRA.U !UP0, `(.L_x_3)
        ACQBULK
        BRA `(.L_x_2)
        .L_x_3:
        ACQBULK
        .L_x_2:
        EXIT
        """
    )
    assert result["ok"], result
    assert result["acqbulk"] == 3


def test_a_uniform_branch_operand_makes_it_conditional():
    # BRA.U !UP0 may fall through: the EXIT after it is reachable without a wait.
    result = _audit(
        """
        BRA.U !UP0, `(.L_x_1)
        EXIT
        .L_x_1:
        ACQBULK
        EXIT
        """
    )
    assert not result["ok"]
    assert result["exit_without_wait"] == ["0x0010"]


def test_a_predicated_wait_is_not_a_wait_and_never_taken_branches_are_skipped():
    result = _audit(
        """
        @!PT BRA `(.L_x_1)
        @P1 ACQBULK
        EXIT
        .L_x_1:
        EXIT
        """
    )
    assert not result["ok"]
    assert result["exit_without_wait"] == ["0x0020"]


def test_a_loop_before_the_wait_terminates_and_passes():
    result = _audit(
        """
        .L_x_0:
        LDG.E.STRONG.SYS R4, desc[UR4][R2.64]
        ISETP.NE.AND P0, PT, R4, R5, PT
        @P0 BRA `(.L_x_0)
        ACQBULK
        EXIT
        """
    )
    assert result["ok"], result
    assert result["global_loads_before_wait"] == {"LDG.E.STRONG.SYS": 1}


def test_an_indirect_branch_or_a_kernel_without_a_wait_fails():
    assert not _audit("BRX R4 -0x10\nACQBULK\nEXIT")["ok"]
    assert _audit("BRX R4 -0x10\nACQBULK\nEXIT")["indirect_branches"] == ["0x0000"]
    assert not _audit("EXIT")["ok"]
    # A call: the callee and the instruction after it are both reachable.
    result = _audit(
        """
        CALL.REL.NOINC `($__internal_0_$__slowpath)
        ACQBULK
        EXIT
        $__internal_0_$__slowpath:
        RET.REL.NODEC R20 `(k)
        """
    )
    assert result["ok"], result


def test_parse_splits_functions_and_reads_guards():
    text = _listing("first", "ACQBULK\nEXIT") + _listing("second", "@!P2 EXIT\nACQBULK\nEXIT")
    functions = parse(text)
    assert sorted(functions) == ["first", "second"]
    instructions, labels = functions["second"]
    assert [(i.guard, i.opcode) for i in instructions] == [
        ("!P2", "EXIT"),
        ("", "ACQBULK"),
        ("", "EXIT"),
    ]
    assert labels == {}
    assert audit(*functions["first"])["ok"]
    assert not audit(*functions["second"])["ok"]


def test_nsight_compute_listings_are_read_with_absolute_targets():
    csv = "\n".join(
        [
            '"Kernel Name","nvjet_sm103_tst_example_TNT"',
            '"Address","Source"',
            '"0x328de3500","      S2R R2, SR_TID.X"',
            '"0x328de3510","      ISETP.LT.AND P0, PT, R2, 0x80, PT"',
            '"0x328de3520","      BRA !P0, 0x328de3550"',
            '"0x328de3530","      ACQBULK"',
            '"0x328de3540","      EXIT"',
            '"0x328de3550","      EXIT"',
            '"Kernel Name","nvjet_sm103_tst_example_TNT"',
            '"Address","Source"',
            '"0x1","      EXIT"',
        ]
    )
    functions = parse_ncu(csv)
    assert list(functions) == ["nvjet_sm103_tst_example_TNT"]
    instructions, labels = functions["nvjet_sm103_tst_example_TNT"]
    assert len(instructions) == 6 and instructions[2].targets == ("0x328de3550",)
    result = audit(instructions, labels)
    # Threads 128.. exit at once (another role); thread 0 waits.
    assert result["ok"] and result["waiting_thread"] == 0, result
    assert result["every_thread"]["exit_without_wait"] == ["0x328de3550"]
