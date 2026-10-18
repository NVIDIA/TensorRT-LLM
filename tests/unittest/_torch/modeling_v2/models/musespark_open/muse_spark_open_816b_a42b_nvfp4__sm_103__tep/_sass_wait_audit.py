# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SASS audit: does a kernel execute ``griddepcontrol.wait`` before it can complete?

A kernel launched with programmatic dependent launch may start before its
predecessor completes. A grid completes only when all of its threads have
exited, and ``griddepcontrol.wait`` (SASS ``ACQBULK``) returns only once the
predecessor grid has completed and its memory is visible. So if some thread
of every launch executes the wait before it exits, the kernel completes only
after its predecessor, and a chain of such kernels carries "the first one
completed" to the last one's wait. The real-weight schedule test (``_census``)
finds, between two consecutive calls of the post-attention hop or of the fused finalize,
the kernels this property must hold for; this module checks it on their
machine code.

``audit`` checks it for the warp leaders of CTA 0 (cluster rank 0) -- thread 0,
32, 64, ... below the block size, threads every launch has -- and passes
(``ok``, ``waiting_thread``) when one of them waits on every path: it follows
that thread's paths from the first instruction -- branches whose predicates it
can evaluate from the special registers of that thread (thread / CTA /
cluster-rank ids, lane) through moves, integer arithmetic, logic ops and
compares, both ways otherwise -- and stops at an ``ACQBULK`` whose predicate is
true; an ``EXIT`` it reaches first is an exit without a wait. The same walk
with every branch taken both ways (``every_thread``) shows whether all threads
wait; warp-specialized kernels have roles that exit without waiting (a TMA
producer warp that only loads weights, say) while another role of the same
CTA waits, which is enough.
Branches: ``BRA`` / ``JMP`` to a label or an address, ``CALL`` (the callee and
the next instruction), ``RET`` / ``BPT`` (end of the path). Indirect branches
(``BRX`` / ``JMX``) cannot be followed and fail the audit. A predicated
``ACQBULK`` counts as a wait only where its predicate is known true. The
global memory instructions any thread can reach before a wait are reported:
loads (weights or other parameters may be read early by design) and stores.

One kind of early exit is reported apart (``launch_uniform_exits``): a
predicated ``EXIT`` in the kernel's straight-line prologue whose predicate is
computed only from constant bank 0 (kernel parameters and launch constants)
and immediates. Every CTA of a launch takes it or none does, so it decides per
launch whether the kernel runs at all (a "dry" launch that only loads the
module, for example); whether a launch in a graph can take it is a property of
the caller, which the audit reports the constants for and cannot decide.

``collect`` disassembles cubins (the CuTe DSL dump directories, a Triton
cache) and shared libraries with ``cuobjdump -sass`` and returns the functions
asked for by name. cuBLASLt's ``nvjet`` kernels are not in its library's fat
binaries; ``parse_ncu`` reads their SASS from Nsight Compute instead
(``ncu -k regex:nvjet --page source --print-source sass --csv <program>``).

Usage (standalone): ``python3 _sass_wait_audit.py [--out <json>] [--threads <block
size>] <cubin,
directory or library> [function name ...]`` or ``... [--out <json>] --ncu
<ncu csv> [function name ...]`` prints one JSON line per function (``--out``
also writes {function: [audit, ...]}).
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


class Instruction(NamedTuple):
    address: int
    guard: str  # "" (none), "PT", "!PT", "P0", "!UP1", ...
    opcode: str  # with modifiers, e.g. "BRA.U", "LDG.E.128"
    operands: str
    targets: Tuple[str, ...]  # branch / call targets: labels, or addresses as "0x..."


_FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
#: A label line: ``.L_x_12:`` or an internal subroutine's ``$__internal_0_$__...:``.
_LABEL = re.compile(r"^\s*([.$][^\s:]*)\s*:\s*$")
_INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;")
_GUARD = re.compile(r"^@(!?U?P[0-9T]+)\s+(.*)$")
_TARGET = re.compile(r"`\(([^)\s]+)\)")
_HEX = re.compile(r"^0x[0-9a-fA-F]+$")
#: Opcodes whose last operand may be a target address.
_JUMPS = ("BRA", "JMP", "CALL")


def _targets(opcode: str, operands: str) -> Tuple[str, ...]:
    labels = tuple(_TARGET.findall(operands))
    if labels or opcode.split(".", 1)[0] not in _JUMPS:
        return labels
    last = operands.split(",")[-1].strip()
    return (last.lower(),) if _HEX.match(last) else ()


#: Global-memory loads and stores (and bulk / TMA copies, reductions, atomics).
_LOADS = ("LDG", "LD", "UTMALDG", "UBLKCP", "LDGSTS", "UTMAPF", "UBLKPF")
_STORES = ("STG", "ST", "RED", "REDG", "ATOM", "ATOMG", "UTMASTG", "UTMAREDG", "UBLKRED")


def parse(sass: str) -> Dict[str, Tuple[List[Instruction], Dict[str, int]]]:
    """``cuobjdump -sass`` text -> {function: (instructions, {label: instruction index})}."""
    functions: Dict[str, Tuple[List[Instruction], Dict[str, int]]] = {}
    name: Optional[str] = None
    instructions: List[Instruction] = []
    labels: Dict[str, int] = {}
    for line in sass.splitlines():
        match = _FUNCTION.match(line)
        if match:
            if name is not None:
                functions[name] = (instructions, labels)
            name, instructions, labels = match.group(1), [], {}
            continue
        if name is None:
            continue
        match = _LABEL.match(line)
        if match:
            labels[match.group(1)] = len(instructions)
            continue
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        text = match.group(2).strip()
        guard = ""
        guarded = _GUARD.match(text)
        if guarded:
            guard, text = guarded.group(1), guarded.group(2)
        opcode, _, operands = text.partition(" ")
        operands = operands.strip()
        instructions.append(
            Instruction(
                int(match.group(1), 16), guard, opcode, operands, _targets(opcode, operands)
            )
        )
    if name is not None:
        functions[name] = (instructions, labels)
    return functions


def _base(opcode: str) -> str:
    return opcode.split(".", 1)[0]


def _is_global(opcode: str, kinds: Sequence[str]) -> bool:
    base = _base(opcode)
    if base not in kinds:
        return False
    # Generic LD / ST are global only with a 64-bit address (.E); LDS / LDL etc. are other bases.
    return base not in ("LD", "ST") or ".E" in opcode


def _conditional(instruction: Instruction) -> bool:
    """Whether a branch may fall through: a guard other than PT, or a deciding operand."""
    if instruction.guard and instruction.guard != "PT":
        return True
    operands = _TARGET.sub("", instruction.operands)
    if instruction.targets and _HEX.match(instruction.targets[0]):
        operands = operands.rsplit(",", 1)[0] if "," in operands else ""
    return bool(operands.strip().strip(",").strip())


#: Instructions that write two predicates first (``ISETP P0, PT, ...``).
_TWO_PREDICATE_DESTS = ("ISETP", "UISETP", "FSETP", "DSETP", "HSETP2", "PLOP3", "UPLOP3", "PSETP")
_CONSTANT_TOKEN = re.compile(
    r"^(-?\|?(RZ|URZ|PT|UPT|!PT|!UPT)\|?|-?\d+(\.\d+)?(e[+-]?\d+)?|-?0x[0-9a-fA-F]+|"
    r"c\[0x0\]\[0x[0-9a-fA-F]+\])$"
)
_REGISTER = re.compile(r"^[!\-|~]*(U?R\d+|U?P\d+)(\.[A-Za-z0-9_]+)*\|?$")


def _operands(instruction: Instruction) -> List[str]:
    return [t.strip() for t in instruction.operands.split(",") if t.strip()]


def _writes(instruction: Instruction) -> List[str]:
    """The registers an instruction writes (its leading destination operands)."""
    operands = _operands(instruction)
    if not operands or _base(instruction.opcode) in _JUMPS + ("EXIT", "RET", "BAR", "ACQBULK"):
        return []
    count = 2 if _base(instruction.opcode) in _TWO_PREDICATE_DESTS else 1
    wide = ".64" in instruction.opcode or ".WIDE" in instruction.opcode
    written = []
    for token in operands[:count]:
        match = _REGISTER.match(token)
        if match:
            written.append(match.group(1))
            pair = re.match(r"^(U?R)(\d+)$", match.group(1))
            if wide and pair:  # a 64-bit result also writes the next register
                written.append(f"{pair.group(1)}{int(pair.group(2)) + 1}")
    return written


def _launch_constants(instructions: List[Instruction], before: int, register: str, depth=0):
    """The constant-bank-0 operands ``register`` is computed from in ``instructions[:before]``
    (straight-line code), or None if it depends on anything else."""
    if depth > 16:
        return None
    for index in range(before - 1, -1, -1):
        ins = instructions[index]
        if register not in _writes(ins):
            continue
        if ins.guard not in ("", "PT") or _base(ins.opcode) in ("S2R", "S2UR", "CS2R"):
            return None
        count = 2 if _base(ins.opcode) in _TWO_PREDICATE_DESTS else 1
        constants = set()
        for token in _operands(ins)[count:]:
            token = token.split(" ")[0] if not token.startswith("c[") else token
            if _CONSTANT_TOKEN.match(token):
                if token.startswith("c["):
                    constants.add(token)
                continue
            match = _REGISTER.match(token)
            if not match:
                return None
            inner = _launch_constants(instructions, index, match.group(1), depth + 1)
            if inner is None:
                return None
            constants |= inner
        return constants
    return None


def _launch_uniform(instructions: List[Instruction], index: int, entered: set) -> Optional[list]:
    """For an ``EXIT`` at ``index``: the constants its guard is computed from, if the exit is
    in the straight-line prologue (nothing branches before it or into it) and the guard
    depends only on constant bank 0 and immediates; else None."""
    guard = instructions[index].guard
    if not guard or guard in ("PT", "!PT") or any(0 < i <= index for i in entered):
        return None
    if any(_base(ins.opcode) in _JUMPS + ("BRX", "JMX", "RET") for ins in instructions[:index]):
        return None
    constants = _launch_constants(instructions, index, guard.lstrip("!"))
    return None if constants is None else sorted(constants)


_MASK = 0xFFFFFFFF
#: Steps of one thread's walk before the audit gives up (inconclusive).
_MAX_STEPS = 400_000
#: Visits of one instruction before the registers that changed between visits are forgotten.
_WIDEN_AFTER = 8
#: The walked thread's x index, kept in the walk's register state under this key.
_TID = "%tid"


def _special(name: str, tid: int) -> Optional[int]:
    """A special register's value in thread ``tid`` (x; y = z = 0) of CTA 0, cluster rank 0."""
    if name == "SR_TID.X":
        return tid
    if name == "SR_LANEID":
        return tid % 32
    if name in ("SR_TID.Y", "SR_TID.Z", "SRZ"):
        return 0
    if name in ("SR_CTAID.X", "SR_CTAID.Y", "SR_CTAID.Z", "SR_CgaCtaId"):
        return 0
    return None


def _value(token: str, state: dict):
    """An operand's value in ``state`` (int for registers / immediates, bool for predicates)."""
    t = token.strip()
    negate = t.startswith("!")
    t = t.lstrip("!")
    if t in ("RZ", "URZ"):
        return 0
    if t in ("PT", "UPT"):
        return not negate
    if re.match(r"^-?0x[0-9a-fA-F]+$", t) or re.match(r"^-?\d+$", t):
        return int(t, 0) & _MASK
    match = _REGISTER.match(t)
    if not match or t.startswith(("-", "|", "~")):
        return None
    value = state.get(match.group(1))
    if value is None:
        return None
    return (not value) if negate else value


def _signed(value: int) -> int:
    return value - (1 << 32) if value & 0x80000000 else value


def _compare(opcode: str, a: int, b: int) -> Optional[bool]:
    parts = opcode.split(".")
    if "U32" not in parts:
        a, b = _signed(a), _signed(b)
    for cmp, result in (
        ("LT", a < b),
        ("LE", a <= b),
        ("GT", a > b),
        ("GE", a >= b),
        ("EQ", a == b),
        ("NE", a != b),
    ):
        if cmp in parts:
            return result
    return None


def _combine(opcode: str, value: bool, other) -> Optional[bool]:
    parts = opcode.split(".")
    if other is None:
        return None
    if "AND" in parts:
        return value and other
    if "OR" in parts:
        return value or other
    if "XOR" in parts:
        return value != other
    return None


def _lop3(a: int, b: int, c: int, lut: int) -> int:
    out = 0
    for minterm in range(8):
        if lut >> minterm & 1:
            out |= (
                (a if minterm & 4 else ~a) & (b if minterm & 2 else ~b) & (c if minterm & 1 else ~c)
            )
    return out & _MASK


def _evaluate(ins: Instruction, state: dict) -> dict:
    """Thread 0's registers after ``ins`` (which runs: its guard is true or absent)."""
    base = _base(ins.opcode)
    ops = _operands(ins)
    written = _writes(ins)
    new = {k: v for k, v in state.items() if k not in written}
    if not written:
        return new
    dest = written[0]
    value = None
    vals = [_value(t, state) for t in ops]
    try:
        if base in ("S2R", "S2UR", "CS2R") and len(ops) == 2:
            value = _special(ops[1], state.get(_TID, 0))
        elif base in ("MOV", "UMOV", "MOV32I", "R2UR") and len(ops) >= 2:
            value = vals[1]
        elif base in ("IADD3", "UIADD3") and len(ops) == 4 and None not in vals[1:]:
            value = (vals[1] + vals[2] + vals[3]) & _MASK
        elif (
            base in ("IMAD", "UIMAD")
            and len(ops) == 4
            and not any(m in ins.opcode for m in (".WIDE", ".HI", ".X"))
            and None not in vals[1:]
        ):
            value = (vals[1] * vals[2] + vals[3]) & _MASK
        elif base in ("LOP3", "ULOP3") and len(ops) == 6 and None not in vals[1:5]:
            if dest.lstrip("U").startswith("R"):
                value = _lop3(vals[1], vals[2], vals[3], vals[4])
        elif base in ("SHF", "USHF") and len(ops) == 4 and None not in vals[1:]:
            if ".R.U32.HI" in ins.opcode and vals[1] == 0:
                value = (vals[3] & _MASK) >> (vals[2] & 31)
            elif ".R.S32.HI" in ins.opcode and vals[1] == 0:
                value = (_signed(vals[3]) >> (vals[2] & 31)) & _MASK
            elif ".L.U32" in ins.opcode and vals[3] == 0 and ".HI" not in ins.opcode:
                value = (vals[1] << (vals[2] & 31)) & _MASK
        elif base in ("SEL", "USEL") and len(ops) == 4 and isinstance(vals[3], bool):
            value = vals[1] if vals[3] else vals[2]
        elif base in ("ISETP", "UISETP") and len(ops) >= 5 and ".EX" not in ins.opcode:
            if None not in vals[2:4]:
                cmp = _compare(ins.opcode, vals[2], vals[3])
                if cmp is not None:
                    value = _combine(ins.opcode, cmp, vals[4])
        elif base == "ELECT" and ops:
            value = state.get(_TID, 0) % 32 == 0  # the lowest active lane is elected
    except (TypeError, ValueError):
        value = None
    if value is not None:
        new[dest] = value
    return new


def _branch_taken(ins: Instruction, state: dict):
    """True / False / None (unknown) for a ``BRA`` / ``JMP`` in ``state`` (its guard aside)."""
    operands = _TARGET.sub("", ins.operands)
    if ins.targets and _HEX.match(ins.targets[0]):
        operands = operands.rsplit(",", 1)[0] if "," in operands else ""
    extra = [t.strip() for t in operands.split(",") if t.strip()]
    if not extra:
        return True
    if len(extra) == 1 and re.match(r"^!?U?P[0-9T]+$", extra[0]):
        return _value(extra[0], state)
    return None


def _guard(ins: Instruction, state: dict):
    return True if not ins.guard else _value(ins.guard, state)


def _thread_walk(instructions, targets, entered, tid: int) -> dict:
    """Thread ``tid``'s paths (see the module docstring): exits before a wait, walk statistics."""
    exits, uniform, indirect, unknown = set(), {}, set(), set()
    first: Dict[int, dict] = {}
    visits: Dict[int, int] = {}
    seen = set()
    todo = [(0, {_TID: tid})] if instructions else []
    steps = 0
    while todo:
        index, state = todo.pop()
        if index >= len(instructions):
            continue
        visits[index] = visits.get(index, 0) + 1
        if index not in first:
            first[index] = state
        elif visits[index] > _WIDEN_AFTER:
            state = {k: v for k, v in state.items() if first[index].get(k) == v}
        key = (index, frozenset(state.items()))
        if key in seen:
            continue
        seen.add(key)
        steps += 1
        if steps > _MAX_STEPS:
            return {"conclusive": False, "steps": steps}
        ins = instructions[index]
        base = _base(ins.opcode)
        guard = _guard(ins, state)
        if guard is False:
            todo.append((index + 1, state))
            continue
        if base == "ACQBULK":
            if guard is True:
                continue
            todo.append((index + 1, state))
            continue
        if base == "EXIT":
            constants = _launch_uniform(instructions, index, entered)
            if constants is not None:
                uniform[ins.address] = constants
            else:
                exits.add(ins.address)
            if guard is None:
                todo.append((index + 1, state))
            continue
        if base in ("BRX", "JMX"):
            indirect.add(ins.address)
            continue
        if base in ("RET", "BPT"):
            if guard is None:
                todo.append((index + 1, state))
            continue
        if base in _JUMPS:
            if not targets[index] or None in targets[index]:
                unknown.add(ins.address)
                continue
            if base == "CALL":
                for target in targets[index]:
                    todo.append((target, state))
                todo.append((index + 1, state))
                continue
            taken = _branch_taken(ins, state)
            if guard is None and taken is not False:
                taken = None
            if taken is not False:
                for target in targets[index]:
                    todo.append((target, state))
            if taken is not True:
                todo.append((index + 1, state))
            continue
        todo.append((index + 1, _evaluate(ins, state) if guard is True else _forget(ins, state)))
    return {
        "conclusive": True,
        "steps": steps,
        "exit_without_wait": sorted(exits),
        "launch_uniform": uniform,
        "indirect": sorted(indirect),
        "unresolved": sorted(unknown),
    }


def _forget(ins: Instruction, state: dict) -> dict:
    """``state`` after ``ins`` ran or not (its guard unknown): what it writes is unknown."""
    written = _writes(ins)
    return {k: v for k, v in state.items() if k not in written}


def audit(instructions: List[Instruction], labels: Dict[str, int], threads: int = 32) -> dict:
    """Whether some warp leader of CTA 0 among the first ``threads`` threads (the block size,
    when known) executes an ``ACQBULK`` on every path before it can exit (``ok``,
    ``waiting_thread``), and whether every thread does (``every_thread``)."""
    exits, indirect, unknown, uniform = [], [], [], []
    loads: Dict[str, int] = {}
    stores: Dict[str, int] = {}
    at = {ins.address: index for index, ins in enumerate(instructions)}
    targets = {}
    for index, ins in enumerate(instructions):
        resolved = []
        for target in ins.targets:
            if _HEX.match(target):
                resolved.append(at.get(int(target, 16)))
            else:
                resolved.append(labels.get(target))
        targets[index] = resolved
    entered = {t for resolved in targets.values() for t in resolved if t is not None}
    entered |= set(labels.values())
    seen, todo = set(), [0] if instructions else []
    while todo:
        index = todo.pop()
        if index in seen or index >= len(instructions):
            continue
        seen.add(index)
        ins = instructions[index]
        base = _base(ins.opcode)
        never = ins.guard == "!PT"
        if never:
            todo.append(index + 1)
            continue
        if base == "ACQBULK" and not ins.guard:
            continue
        if _is_global(ins.opcode, _LOADS):
            loads[ins.opcode] = loads.get(ins.opcode, 0) + 1
        if _is_global(ins.opcode, _STORES):
            stores[ins.opcode] = stores.get(ins.opcode, 0) + 1
        if base == "EXIT":
            constants = _launch_uniform(instructions, index, entered)
            if constants is None:
                exits.append(ins.address)
            else:
                uniform.append(
                    {"address": f"{ins.address:#06x}", "guard": ins.guard, "reads": constants}
                )
            if ins.guard and ins.guard != "PT":
                todo.append(index + 1)
            continue
        if base in ("BRX", "JMX"):
            indirect.append(ins.address)
            continue
        if base in ("RET", "BPT"):
            if ins.guard and ins.guard != "PT":
                todo.append(index + 1)
            continue
        if base in _JUMPS:
            if not targets[index] or None in targets[index]:
                unknown.append(ins.address)
                continue
            todo.extend(targets[index])
            if base == "CALL" or _conditional(ins):
                todo.append(index + 1)
            continue
        todo.append(index + 1)
    waits = sum(1 for ins in instructions if _base(ins.opcode) == "ACQBULK")
    triggers = sum(1 for ins in instructions if _base(ins.opcode) == "PREEXIT")
    # Warp leaders of CTA 0, thread 0 first: the first one that waits on every path decides.
    walks, waiting = {}, None
    for tid in range(0, min(max(int(threads), 32), 1024), 32):
        walk = walks[tid] = _thread_walk(instructions, targets, entered, tid)
        if (
            walk["conclusive"]
            and not walk["exit_without_wait"]
            and not walk["indirect"]
            and not walk["unresolved"]
        ):
            waiting = tid
            break
    shown = walks[0 if waiting is None else waiting]
    return {
        "ok": waits > 0 and waiting is not None,
        "acqbulk": waits,
        "preexit": triggers,
        "instructions": len(instructions),
        "waiting_thread": waiting,
        "threads_walked": sorted(walks),
        "walk_conclusive": shown["conclusive"],
        "walk_steps": shown["steps"],
        "exit_without_wait": [f"{a:#06x}" for a in shown.get("exit_without_wait", [])],
        "launch_uniform_exits": [
            {"address": f"{a:#06x}", "reads": reads}
            for a, reads in sorted(shown.get("launch_uniform", {}).items())
        ],
        "indirect_branches": [f"{a:#06x}" for a in shown.get("indirect", [])],
        "unresolved_branches": [f"{a:#06x}" for a in shown.get("unresolved", [])],
        "every_thread": {
            "ok": waits > 0 and not exits and not indirect and not unknown,
            "exit_without_wait": [f"{a:#06x}" for a in sorted(exits)],
            "launch_uniform_exits": uniform,
        },
        "global_loads_before_wait": loads,
        "global_stores_before_wait": stores,
    }


def disassemble(path: str, functions: Iterable[str] = (), timeout: float = 1800) -> str:
    """``cuobjdump -sass`` of a cubin or a library (only ``functions`` when given)."""
    command = ["cuobjdump", "-sass"]
    names = sorted(set(functions))
    if names:
        command += ["-fun", ",".join(names)]
    run = subprocess.run(command + [path], capture_output=True, text=True, timeout=timeout)
    return run.stdout


def _cubins(roots: Iterable[str]) -> List[str]:
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(root)
            continue
        for directory, _, files in os.walk(root):
            found += [os.path.join(directory, f) for f in files if f.endswith(".cubin")]
    return sorted(found)


def collect(
    names: Iterable[str],
    cubin_roots: Iterable[str] = (),
    libraries: Iterable[str] = (),
    threads: Optional[Dict[str, int]] = None,
) -> Dict[str, List[Tuple[str, dict]]]:
    """{name: [(source, audit), ...]} for every function asked for that was found.

    Every cubin under ``cubin_roots`` is disassembled whole; each library only for
    the names still missing. A name found in several binaries (one per compiled
    variant) is audited in each. ``threads``: block size per name (default 32).
    """
    threads = threads or {}
    wanted = set(names)
    symbols = [name.encode() for name in wanted]
    out: Dict[str, List[Tuple[str, dict]]] = {}
    for cubin in _cubins(cubin_roots):
        with open(cubin, "rb") as f:
            data = f.read()
        if not any(symbol in data for symbol in symbols):
            continue  # its symbol table names none of them
        for name, (instructions, labels) in parse(disassemble(cubin)).items():
            if name in wanted:
                out.setdefault(name, []).append(
                    (cubin, audit(instructions, labels, threads.get(name, 32)))
                )
    for library in libraries:
        missing = wanted - set(out)
        if not missing:
            break
        for name, (instructions, labels) in parse(disassemble(library, missing)).items():
            if name in missing:
                out.setdefault(name, []).append(
                    (library, audit(instructions, labels, threads.get(name, 32)))
                )
    return out


def parse_ncu(text: str) -> Dict[str, Tuple[List[Instruction], Dict[str, int]]]:
    """Nsight Compute's ``--page source --print-source sass --csv`` output -> as ``parse``.

    For kernels whose machine code ``cuobjdump`` cannot read (cuBLASLt keeps its
    ``nvjet`` kernels out of the library's fat binaries): one block per profiled
    launch, a ``"Kernel Name"`` row then ``"Address","Source"`` rows, branch
    targets as absolute addresses. A kernel profiled more than once keeps its
    first listing.
    """
    functions: Dict[str, Tuple[List[Instruction], Dict[str, int]]] = {}
    name: Optional[str] = None
    instructions: List[Instruction] = []
    for row in csv.reader(text.splitlines()):
        if len(row) >= 2 and row[0] == "Kernel Name":
            if name is not None and name not in functions:
                functions[name] = (instructions, {})
            name, instructions = row[1], []
            continue
        if name is None or len(row) < 2 or not _HEX.match(row[0].strip()):
            continue
        source = row[1].strip().rstrip(";").strip()
        guard = ""
        guarded = _GUARD.match(source)
        if guarded:
            guard, source = guarded.group(1), guarded.group(2)
        opcode, _, operands = source.partition(" ")
        operands = operands.strip()
        instructions.append(
            Instruction(int(row[0], 16), guard, opcode, operands, _targets(opcode, operands))
        )
    if name is not None and name not in functions:
        functions[name] = (instructions, {})
    return functions


def loaded_library(stem: str) -> Optional[str]:
    """The path of a shared library this process has mapped whose file name starts with ``stem``."""
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                path = line.split()[-1] if line.strip() else ""
                if os.path.basename(path).startswith(stem) and ".so" in path:
                    return path
    except OSError:
        return None
    return None


def main(argv: List[str]) -> int:
    out_path, threads = None, 32
    while len(argv) >= 2 and argv[0] in ("--out", "--threads"):
        if argv[0] == "--out":
            out_path = argv[1]
        else:
            threads = int(argv[1])
        argv = argv[2:]
    if not argv:
        print(__doc__)
        return 2
    path, names = argv[0], argv[1:]
    if path == "--ncu":
        path, names = names[0], names[1:]
        with open(path) as f:
            parsed_all = parse_ncu(f.read())
        functions = {
            name: [(path, audit(*parsed, threads))]
            for name, parsed in parsed_all.items()
            if not names or name in names
        }
    elif os.path.isdir(path) or path.endswith(".cubin"):
        functions = {}
        for cubin in _cubins([path]):
            for name, parsed in parse(disassemble(cubin)).items():
                if not names or name in names:
                    functions.setdefault(name, []).append((cubin, audit(*parsed, threads)))
    else:
        functions = {
            name: [(path, audit(*parsed, threads))]
            for name, parsed in parse(disassemble(path, names)).items()
        }
    ok = True
    for name, results in sorted(functions.items()):
        for source, result in results:
            ok &= result["ok"]
            print(json.dumps({"function": name, "source": source, **result}), flush=True)
    missing = sorted(set(names) - set(functions))
    if missing:
        print(json.dumps({"missing": missing}), flush=True)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(
                {
                    name: [{"source": s, **r} for s, r in results]
                    for name, results in functions.items()
                },
                f,
                indent=1,
            )
    return 0 if ok and not missing else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
