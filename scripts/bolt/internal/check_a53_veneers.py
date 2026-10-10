#!/usr/bin/env python3
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
"""Refuse to BOLT an AArch64 binary that still carries erratum-843419 veneers.

GNU ld's --fix-cortex-a53-843419 workaround (on by default in Ubuntu's aarch64
GCC) rewrites an offending load as a branch to a veneer that performs the load
and branches straight back:

    S:      b    T              <- was a load
    T:      <the displaced load>
    T+4:    b    S+4

BOLT reads the branch at S as a tail call, because T is a separate symbol. Tail
calls may clobber x16, so when BOLT relocates the function out of direct branch
range it patches the return site S+4 with an `adrp x16 / add x16 / br x16`
trampoline. S+4 is in the middle of a function, and if the original code was
holding a value in x16 there, that value is silently replaced with the address
of the trampoline. The result is a wild pointer and a SIGSEGV with no diagnostic
from any stage of the build.

cpp/CMakeLists.txt passes -mno-fix-cortex-a53-843419 under ENABLE_BOLT_COMPATIBLE
so these veneers are never emitted. This checks that the binary in hand actually
came from such a build, because the failure mode when it did not is silent
corruption rather than an error.

Detection is structural rather than by symbol name (`e843419*`), since the
symbols disappear from a stripped link while the hazard does not.

Exit codes:
  0  no veneers (safe to BOLT), or not an AArch64 ELF
  1  veneers present
  2  usage error
"""

import argparse
import struct
import sys

EM_AARCH64 = 183
SHF_EXECINSTR = 0x4
B_MASK, B_OP = 0xFC000000, 0x14000000


def _sx(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _branch_target(word: int, addr: int) -> int | None:
    """Target of an unconditional `b`, or None if this is not one."""
    if (word & B_MASK) != B_OP:
        return None
    return addr + (_sx(word & 0x03FFFFFF, 26) << 2)


def exec_sections(f) -> list[tuple[int, bytes]]:
    """(vaddr, contents) for each executable section of an AArch64 ELF64.

    Returns [] for anything else, so callers can treat "not applicable" and
    "nothing found" the same way.
    """

    def field(offset: int, fmt: str):
        """Unpack a fixed-width header field, or None if the file is short."""
        f.seek(offset)
        raw = f.read(struct.calcsize(fmt))
        return struct.unpack(fmt, raw) if len(raw) == struct.calcsize(fmt) else None

    # A four-byte \x7fELF prefix is enough for apply_bolt.is_elf to send a file
    # here, so every read below has to tolerate a truncated one.
    ident = field(0, "<16s")
    if ident is None:
        return []
    ident = ident[0]
    if ident[:4] != b"\x7fELF" or ident[4] != 2 or ident[5] != 1:
        return []
    machine = field(18, "<H")
    if machine is None or machine[0] != EM_AARCH64:
        return []

    shoff = field(0x28, "<Q")
    sh = field(0x3A, "<HH")
    if shoff is None or sh is None:
        return []
    shoff = shoff[0]
    shentsize, shnum = sh
    if not shoff or not shnum:
        return []

    out = []
    for i in range(shnum):
        f.seek(shoff + i * shentsize)
        hdr = f.read(64)
        if len(hdr) < 64:
            break
        _, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from("<IIQQQQ", hdr, 0)
        if sh_type == 8 or not sh_size:  # SHT_NOBITS / empty
            continue
        if not sh_flags & SHF_EXECINSTR:
            continue
        f.seek(sh_offset)
        out.append((sh_addr, f.read(sh_size)))
    return out


def find_veneers(path: str) -> list[tuple[int, int]]:
    """(branch_site, veneer) for every erratum-843419 veneer round trip."""
    with open(path, "rb") as f:
        secs = exec_sections(f)
    if not secs:
        return []

    def word_at(va: int) -> int | None:
        for base, data in secs:
            if base <= va < base + len(data) - 3:
                return struct.unpack_from("<I", data, va - base)[0]
        return None

    found = []
    for base, data in secs:
        words = struct.unpack_from(f"<{len(data) // 4}I", data, 0)
        for i, word in enumerate(words):
            src = base + 4 * i
            tgt = _branch_target(word, src)
            if tgt is None or tgt == src:
                continue
            back = word_at(tgt + 4)
            if back is None:
                continue
            if _branch_target(back, tgt + 4) == src + 4:
                found.append((src, tgt))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("elf", nargs="+")
    ap.add_argument("--list", action="store_true", help="print every veneer, not just the count")
    args = ap.parse_args()

    total = 0
    for path in args.elf:
        try:
            veneers = find_veneers(path)
        except OSError as exc:
            print(f"[a53-check] cannot read {path}: {exc}", file=sys.stderr)
            return 2
        total += len(veneers)
        if not veneers:
            continue
        print(f"[a53-check] {path}: {len(veneers)} cortex-a53-843419 veneers", file=sys.stderr)
        if args.list:
            for src, tgt in veneers:
                print(f"[a53-check]   b 0x{src:x} -> veneer 0x{tgt:x}", file=sys.stderr)

    if total:
        print(
            "[a53-check] BOLT would miscompile this binary. Build with "
            "ENABLE_BOLT_COMPATIBLE=ON so -mno-fix-cortex-a53-843419 is "
            "applied, then relink.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
