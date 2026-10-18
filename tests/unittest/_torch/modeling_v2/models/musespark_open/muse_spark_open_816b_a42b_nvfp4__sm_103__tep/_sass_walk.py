# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What each warp role of a kernel can touch in global memory before ``griddepcontrol.wait``.

Pure Python, over one function of ``cuobjdump -sass`` text. ``pre_wait`` walks the
control-flow graph from the first instruction and stops a path at an unpredicated
``ACQBULK`` (``griddepcontrol.wait``). It returns the global-memory instructions
reachable on such a path (``accesses``), the ``EXIT`` instructions reachable without
a wait (``exits``) and the triggers (``PREEXIT``) reachable before a wait. The walk
follows control flow, not listing order, so a role branch that never waits (the
early expert kernel's weight TMA warp) or waits later is audited too.

With ``tid`` the walk is one thread's: it evaluates the values that depend only on
the thread index (``S2R SR_TID.X`` / ``SR_LANEID`` through moves, shifts, logic,
additions and compares, uniform registers included; ``ELECT`` picks the warp's
lane 0) and takes a branch, or skips a guarded instruction, the way that thread
does; anything else is unknown and taken both ways. ``per_warp`` runs it for every
warp's lane 0, i.e. once per warp role. Without ``tid`` every branch is taken both
ways (any thread). Either walk can only name more instructions than the thread
executes, never fewer.

Branch forms: ``BRA`` / ``JMP`` to an address (``0x1d0``) or a label
(`` `(.L_x_12) ``); a predicate operand (``BRA.U !UP0, 0x1d0``) or a guard
(``@!P0 BRA``) makes it conditional; ``CALL`` continues at the callee and after the
call; ``RET`` / ``BPT`` / an unguarded ``EXIT`` end a path; ``BRX`` / ``JMX``
(indirect) are reported in ``indirect`` and end the path.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Tuple

_INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;")
_LABEL = re.compile(r"^\s*([.$][^\s:]*)\s*:\s*$")
_GUARD = re.compile(r"^@(!?U?P[0-9T]+)\s+(.*)$")
_TARGET_LABEL = re.compile(r"`\(([^)\s]+)\)")
_HEX = re.compile(r"^-?0x[0-9a-fA-F]+$")
_PREDICATE = re.compile(r"^!?U?P[0-9T]+$")
_REGISTER = re.compile(r"^U?R[0-9]+$|^U?RZ$")
_MASK = 0xFFFFFFFF

#: Opcode roots of global-memory accesses (loads, stores, reductions, atomics, bulk copies).
GLOBAL = (
    "LDG",
    "STG",
    "RED",
    "REDG",
    "ATOMG",
    "LDGSTS",
    "UTMALDG",
    "UTMASTG",
    "UTMAREDG",
    "UTMAPF",
    "UBLKCP",
    "UBLKRED",
    "UBLKPF",
)


class Instruction(NamedTuple):
    address: int
    guard: str
    opcode: str
    operands: Tuple[str, ...]
    text: str


def parse(sass: str) -> Tuple[List[Instruction], Dict[str, int]]:
    """The first function of ``sass``: (instructions in order, {label: instruction index})."""
    instructions: List[Instruction] = []
    labels: Dict[str, int] = {}
    seen_function = False
    for line in sass.splitlines():
        if "Function :" in line:
            if seen_function:
                break
            seen_function = True
            continue
        label = _LABEL.match(line)
        if label:
            labels[label.group(1)] = len(instructions)
            continue
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        body = match.group(2).strip()
        guard = ""
        guarded = _GUARD.match(body)
        if guarded:
            guard, body = guarded.group(1), guarded.group(2)
        opcode, _, rest = body.partition(" ")
        operands = tuple(op.strip() for op in rest.split(",")) if rest.strip() else ()
        instructions.append(Instruction(int(match.group(1), 16), guard, opcode, operands, body))
    return instructions, labels


def _root(opcode: str) -> str:
    return opcode.split(".", 1)[0]


# ---- thread-index values -----------------------------------------------------


def _value(token: str, state: dict) -> Optional[int]:
    if token in ("RZ", "URZ"):
        return 0
    if _HEX.match(token):
        return int(token, 16) & _MASK
    if token.lstrip("-").isdigit():
        return int(token) & _MASK
    return state.get(token)


def _predicate(token: str, state: dict) -> Optional[bool]:
    negate = token.startswith("!")
    name = token[1:] if negate else token
    if name in ("PT", "UPT"):
        value = True
    else:
        value = state.get(name)
    if value is None:
        return None
    return (not value) if negate else value


def _compare(opcode: str, a: int, b: int) -> Optional[bool]:
    parts = opcode.split(".")
    if "U32" not in parts:
        a = a - (1 << 32) if a & 0x80000000 else a
        b = b - (1 << 32) if b & 0x80000000 else b
    for name, fn in (
        ("EQ", lambda: a == b),
        ("NE", lambda: a != b),
        ("LT", lambda: a < b),
        ("LE", lambda: a <= b),
        ("GT", lambda: a > b),
        ("GE", lambda: a >= b),
    ):
        if name in parts:
            return fn()
    return None


def _lop3(a: int, b: int, c: int, lut: int) -> int:
    out = 0
    for bit in range(32):
        index = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        out |= ((lut >> index) & 1) << bit
    return out


def _forget(ins: Instruction, state: dict) -> None:
    ops = ins.operands
    if _root(ins.opcode) == "R2P":
        for name in [k for k in state if k.startswith("P")]:
            del state[name]
    for k, op in enumerate(ops[:2]):
        name = op.lstrip("!")
        if k == 0 or _PREDICATE.match(op):
            state.pop(name, None)


def _evaluate(ins: Instruction, state: dict, tid: int) -> None:
    """Update ``state`` (register / predicate -> known value) for one executed instruction."""
    root, ops = _root(ins.opcode), ins.operands
    parts = ins.opcode.split(".")
    result = None
    if root in ("S2R", "S2UR", "CS2R") and len(ops) == 2:
        result = {"SR_TID.X": tid, "SR_LANEID": tid & 31}.get(ops[1])
    elif root in ("MOV", "UMOV", "R2UR") and len(ops) >= 2:
        result = _value(ops[1], state)
    elif root == "IMAD" and "MOV" in parts and len(ops) >= 4:
        result = _value(ops[3], state)
    elif root in ("SHF", "USHF") and len(ops) == 4:
        lo, shift, hi = (_value(op, state) for op in ops[1:])
        if None not in (lo, shift, hi) and "HI" in parts and "R" in parts and lo == 0:
            result = (hi >> (shift & 31)) & _MASK
        elif None not in (lo, shift) and "L" in parts and hi == 0:
            result = (lo << (shift & 31)) & _MASK
    elif root in ("LOP3", "ULOP3") and len(ops) == 6 and not _PREDICATE.match(ops[0]):
        a, b, c, lut = (_value(op, state) for op in ops[1:5])
        if None not in (a, b, c, lut):
            result = _lop3(a, b, c, lut)
    elif root in ("IADD3", "UIADD3") and len(ops) == 4:
        terms = [_value(op.lstrip("-"), state) for op in ops[1:]]
        if None not in terms:
            signs = [-1 if op.startswith("-") else 1 for op in ops[1:]]
            result = sum(s * t for s, t in zip(signs, terms)) & _MASK
    elif root in ("VIADD",) and len(ops) == 3:
        a, b = _value(ops[1], state), _value(ops[2], state)
        if None not in (a, b):
            result = (a + b) & _MASK
    elif root in ("ISETP", "UISETP") and len(ops) == 5 and ops[1] in ("PT", "UPT"):
        a, b = _value(ops[2], state), _value(ops[3], state)
        other = _predicate(ops[4], state)
        if None not in (a, b):
            value = _compare(ins.opcode, a, b)
            if value is not None and "AND" in parts and other is not None:
                state[ops[0]] = value and other
                return
            if value is not None and "OR" in parts and other is not None:
                state[ops[0]] = value or other
                return
        state.pop(ops[0], None)
        return
    elif root in ("PLOP3", "UPLOP3") and len(ops) >= 6:
        # PLOP3.LUT Pd, Pq, Pa, Pb, Pc, lut_d, lut_q
        a, b, c = (_predicate(op, state) for op in ops[2:5])
        for dest, lut in (
            (ops[0], _value(ops[5], state)),
            (ops[1], _value(ops[6], state) if len(ops) > 6 else None),
        ):
            if dest in ("PT", "UPT"):
                continue
            if None in (a, b, c, lut):
                state.pop(dest, None)
            else:
                state[dest] = bool((lut >> ((int(a) << 2) | (int(b) << 1) | int(c))) & 1)
        return
    elif root == "ELECT" and ops:
        state[ops[0]] = (tid & 31) == 0
        return
    _forget(ins, state)
    if result is not None and ops and _REGISTER.match(ops[0]) and ops[0] not in ("RZ", "URZ"):
        state[ops[0]] = result


# ---- the walk ---------------------------------------------------------------------


def _target(ins: Instruction, by_address: Dict[int, int], labels: Dict[str, int]):
    for op in reversed(ins.operands):
        label = _TARGET_LABEL.search(op)
        if label:
            return labels.get(label.group(1))
        if _HEX.match(op):
            return by_address.get(int(op, 16))
    return None


def _branch(ins: Instruction, state: dict, every_thread: bool) -> Optional[bool]:
    """True / False when the branch is known taken / not taken here; None when both."""
    if set(ins.opcode.split(".")[1:]) & {"DIV", "CONV", "ANY", "ALL"}:
        return None
    taken = True
    if ins.guard and ins.guard != "PT":
        guard = None if every_thread else _predicate(ins.guard, state)
        if guard is None:
            return None
        taken = guard
    if ins.operands and _PREDICATE.match(ins.operands[0]) and ins.operands[0] not in ("PT", "UPT"):
        pred = None if every_thread else _predicate(ins.operands[0], state)
        if pred is None:
            return None if taken else False
        taken = taken and pred
    return taken


def pre_wait(sass: str, tid: Optional[int] = None) -> dict:
    """Global accesses, exits and triggers reachable from the entry before a wait."""
    instructions, labels = parse(sass)
    by_address = {ins.address: i for i, ins in enumerate(instructions)}
    every_thread = tid is None
    accesses, exits, triggers, indirect, unresolved = {}, {}, {}, {}, {}
    visits: Dict[int, int] = {}
    seen_states = set()
    stack = [(0, {})] if instructions else []
    reached = set()
    while stack:
        i, state = stack.pop()
        while 0 <= i < len(instructions):
            key = (i, tuple(sorted(state.items())))
            if key in seen_states or visits.get(i, 0) >= 8:
                break
            seen_states.add(key)
            visits[i] = visits.get(i, 0) + 1
            reached.add(i)
            ins = instructions[i]
            root = _root(ins.opcode)
            guard = None
            if ins.guard and ins.guard != "PT":
                guard = None if every_thread else _predicate(ins.guard, state)
            else:
                guard = True
            if root in ("BRA", "JMP", "CALL"):
                target = _target(ins, by_address, labels)
                taken = True if root == "CALL" else _branch(ins, state, every_thread)
                if target is None:
                    unresolved[ins.address] = ins
                    if taken:
                        break
                elif taken is None:
                    stack.append((target, dict(state)))
                elif taken:
                    if root == "CALL":
                        stack.append((target, dict(state)))
                    else:
                        i = target
                        continue
                i += 1
                continue
            if guard is False:
                i += 1
                continue
            if root == "ACQBULK" and guard is True:
                break
            if root in GLOBAL:
                accesses[ins.address] = ins
            if root == "PREEXIT":
                triggers[ins.address] = ins
            if root == "EXIT":
                exits[ins.address] = ins
                if guard is True:
                    break
            if root in ("RET", "BPT") and guard is True:
                break
            if root in ("BRX", "JMX"):
                indirect[ins.address] = ins
                break
            if every_thread:
                _forget(ins, state)
            elif guard is True:
                _evaluate(ins, state, tid)
            else:
                _forget(ins, state)
            i += 1

    def order(items):
        return [items[a] for a in sorted(items)]

    return dict(
        accesses=order(accesses),
        exits=order(exits),
        triggers=order(triggers),
        indirect=order(indirect),
        unresolved=order(unresolved),
        reached=len(reached),
        total=len(instructions),
    )


def _fmt(items) -> str:
    return " ".join(f"{ins.opcode}@{ins.address:#x}" for ins in items) or "-"


def summary(found: dict) -> str:
    return (
        f"reached {found['reached']}/{found['total']}; global {len(found['accesses'])}: "
        f"{_fmt(found['accesses'])}; exits without a wait: {_fmt(found['exits'])}; "
        f"triggers: {_fmt(found['triggers'])}"
        + (f"; INDIRECT {_fmt(found['indirect'])}" if found["indirect"] else "")
        + (f"; UNRESOLVED {_fmt(found['unresolved'])}" if found["unresolved"] else "")
    )


def per_warp(sass: str, threads: int = 256) -> List[Tuple[List[int], dict]]:
    """``pre_wait`` for every warp's lane 0, warps with identical results grouped."""
    groups: List[Tuple[List[int], dict]] = []
    for warp in range(threads // 32):
        found = pre_wait(sass, 32 * warp)
        signature = summary(found).split("; ", 1)[1]
        for warps, other in groups:
            if summary(other).split("; ", 1)[1] == signature:
                warps.append(warp)
                break
        else:
            groups.append(([warp], found))
    return groups
