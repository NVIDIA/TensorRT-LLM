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
"""host_page_retention host policy: no immediate CPU-allocator page purges in a serving decode rank.

On a GPU that translates through ATS (an sm_103 GPU attached to Grace), every page-table change
of a process holding a CUDA context on that GPU (``munmap``,
``madvise(MADV_DONTNEED)``, ``mprotect``) also invalidates the GPU's address
translations, so TLB-miss-heavy kernels running at that moment (the sparse KV
gathers of MLA decode and of the indexer logits) slow down. PyTorch allocates
CPU tensors through a mimalloc that is linked statically into ``libc10.so``;
with ``MIMALLOC_PURGE_DELAY=0`` every free that empties an allocator page
returns it to the OS at once, so the decode rank's ordinary host work (input
preparation, sampling bookkeeping) issues such invalidations while the GPU
runs the step.

``install_host_policy`` sets mimalloc's ``purge_delay`` option of this process
to ``PURGE_DELAY_MS``: a freed allocator page goes back to the OS only after
it has stayed unused that long. The pages that each decode step's host work
frees are allocated again well within that time, so they are reused instead
of purged during the step, while memory that stays idle is still returned. It
changes no device operation, no kernel and no allocation size.
mimalloc exports none of its symbols from ``libc10.so``, so the option
functions are located through the library's ELF symbol table and the option
index is resolved by name from mimalloc's option table; every step is checked
and any mismatch leaves the allocator untouched (native fallback).
"""

import ctypes
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# mimalloc's purge_delay: -1 never purges, 0 purges at once, N > 0 after N ms.
PURGE_DELAY_MS = 1000
PURGE_DELAY_OPTION = "purge_delay"

_ELF_MAGIC = b"\x7fELF"
_SHT_SYMTAB = 2
_SHT_DYNSYM = 11
_PT_LOAD = 1
_STT_OBJECT = 1
_STT_FUNC = 2
_SHN_UNDEF = 0
# mi_option_desc_t: long value; mi_init_t init; mi_option_t option;
# const char* name; const char* legacy_name.
_DESC = struct.Struct("<qiiQQ")
_INIT_STATES = (0, 1, 2)  # UNINIT, DEFAULTED, INITIALIZED
_NEEDED = ("mi_option_get", "mi_option_set", "options")


class HostPolicyUnavailable(RuntimeError):
    """The mimalloc option table could not be located or failed a check."""


@dataclass(frozen=True)
class _Symbol:
    value: int
    size: int
    kind: int


def _read_symbols(path: str) -> Tuple[Dict[str, List[_Symbol]], Dict[str, _Symbol], int]:
    """Return the symbols this module needs and the link address of file offset 0.

    (``.symtab`` symbols named in ``_NEEDED``, one defined ``.dynsym`` function,
    ``p_vaddr - p_offset`` of the first ``PT_LOAD``). The ``.dynsym`` function
    anchors the load address of the library.
    """
    with open(path, "rb") as f:
        ident = f.read(16)
        if len(ident) < 16 or ident[:4] != _ELF_MAGIC or ident[4] != 2 or ident[5] != 1:
            raise HostPolicyUnavailable(f"{path}: not a 64-bit little-endian ELF file")
        f.seek(0)
        hdr = struct.unpack("<16sHHIQQQIHHHHHH", f.read(64))
        e_phoff, e_shoff, e_phentsize, e_phnum = hdr[5], hdr[6], hdr[9], hdr[10]
        e_shentsize, e_shnum = hdr[11], hdr[12]
        if e_phoff == 0 or e_phentsize != 56 or e_phnum == 0:
            raise HostPolicyUnavailable(f"{path}: no program headers")
        f.seek(e_phoff)
        loads = [
            ph
            for ph in (struct.unpack("<IIQQQQQQ", f.read(56)) for _ in range(e_phnum))
            if ph[0] == _PT_LOAD
        ]
        if not loads:
            raise HostPolicyUnavailable(f"{path}: no PT_LOAD segment")
        first_load = min(loads, key=lambda ph: ph[2])
        link_offset0 = first_load[3] - first_load[2]
        if e_shoff == 0 or e_shentsize != 64 or e_shnum == 0:
            raise HostPolicyUnavailable(f"{path}: no section headers")
        f.seek(e_shoff)
        sections = [struct.unpack("<IIQQQQIIQQ", f.read(64)) for _ in range(e_shnum)]

        def read_section(index):
            sec = sections[index]
            f.seek(sec[4])
            return f.read(sec[5])

        wanted: Dict[str, List[_Symbol]] = {}
        anchor: Dict[str, _Symbol] = {}
        for index, sec in enumerate(sections):
            sh_type, sh_link, sh_entsize = sec[1], sec[6], sec[9]
            if sh_type not in (_SHT_SYMTAB, _SHT_DYNSYM) or sh_entsize != 24:
                continue
            data = read_section(index)
            strtab = read_section(sh_link)
            for off in range(0, len(data) - 23, 24):
                st_name, st_info, _, st_shndx, st_value, st_size = struct.unpack_from(
                    "<IBBHQQ", data, off
                )
                if st_shndx == _SHN_UNDEF or st_name == 0:
                    continue
                end = strtab.find(b"\0", st_name)
                name = strtab[st_name:end].decode("ascii", "replace")
                kind = st_info & 0xF
                if sh_type == _SHT_SYMTAB and name in _NEEDED:
                    wanted.setdefault(name, []).append(_Symbol(st_value, st_size, kind))
                elif (
                    sh_type == _SHT_DYNSYM
                    and not anchor
                    and kind == _STT_FUNC
                    and st_value
                    and st_size
                ):
                    anchor[name] = _Symbol(st_value, st_size, kind)
    return wanted, anchor, link_offset0


def _unique(symbols: Dict[str, List[_Symbol]], name: str, kind: int) -> _Symbol:
    found = {s for s in symbols.get(name, ()) if s.kind == kind and s.value}
    if len(found) != 1:
        raise HostPolicyUnavailable(f"symbol {name!r}: expected one definition, found {len(found)}")
    return found.pop()


def _mapped_base(path: str) -> Optional[int]:
    """Start of the first mapping of ``path`` at file offset 0 in this process."""
    real = os.path.realpath(path)
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if (
                    len(parts) >= 6
                    and parts[2] == "00000000"
                    and os.path.realpath(parts[5]) == real
                ):
                    return int(parts[0].split("-")[0], 16)
    except OSError:
        return None
    return None


class MimallocOptions:
    """mimalloc's runtime options in an already-loaded library, by option name."""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise HostPolicyUnavailable(f"{path}: not found")
        symbols, anchor, link_offset0 = _read_symbols(path)
        missing = [n for n in _NEEDED if n not in symbols]
        if missing:
            raise HostPolicyUnavailable(f"{path}: no mimalloc symbols {missing}")
        if not anchor:
            raise HostPolicyUnavailable(f"{path}: no exported function to anchor the load address")
        get_sym = _unique(symbols, "mi_option_get", _STT_FUNC)
        set_sym = _unique(symbols, "mi_option_set", _STT_FUNC)
        table = _unique(symbols, "options", _STT_OBJECT)
        if table.size == 0 or table.size % _DESC.size:
            raise HostPolicyUnavailable(
                f"option table of {table.size} bytes is not a whole number of "
                f"{_DESC.size}-byte descriptors"
            )
        try:
            lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError as exc:
            raise HostPolicyUnavailable(f"{path}: not loaded in this process ({exc})") from exc
        ((anchor_name, anchor_sym),) = anchor.items()
        bias = ctypes.cast(lib[anchor_name], ctypes.c_void_p).value - anchor_sym.value
        mapped = _mapped_base(path)
        if mapped is not None and mapped != bias + link_offset0:
            raise HostPolicyUnavailable(
                f"load address mismatch: anchor bias {bias:#x} + {link_offset0:#x}, "
                f"/proc/self/maps {mapped:#x}"
            )
        self.path = path
        self._lib = lib  # keeps the handle (and the library) alive
        self._table = bias + table.value
        self._count = table.size // _DESC.size
        self._get = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int)(bias + get_sym.value)
        self._set = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_long)(bias + set_sym.value)
        self.names = self._check_table()

    def _descriptor(self, index: int) -> Tuple[int, int, int, int, int]:
        raw = ctypes.string_at(self._table + index * _DESC.size, _DESC.size)
        return _DESC.unpack(raw)

    def _check_table(self) -> List[str]:
        names = []
        for index in range(self._count):
            _, init, option, name_ptr, _ = self._descriptor(index)
            if option != index or init not in _INIT_STATES or not name_ptr:
                raise HostPolicyUnavailable(
                    f"option descriptor {index}: option={option} init={init}; not a "
                    f"mimalloc option table"
                )
            name = ctypes.c_char_p(name_ptr).value or b""
            if not 0 < len(name) <= 64 or not all(
                c == 0x5F or 0x61 <= c <= 0x7A or 0x30 <= c <= 0x39 for c in name
            ):
                raise HostPolicyUnavailable(f"option descriptor {index}: bad name {name!r}")
            names.append(name.decode())
        return names

    def index(self, name: str) -> int:
        if self.names.count(name) != 1:
            raise HostPolicyUnavailable(f"option {name!r} not found exactly once")
        return self.names.index(name)

    def get(self, name: str) -> int:
        index = self.index(name)
        value = self._get(index)
        stored = self._descriptor(index)[0]
        if value != stored:
            raise HostPolicyUnavailable(
                f"mi_option_get({name}) = {value} but the table holds {stored}"
            )
        return value

    def set(self, name: str, value: int) -> None:
        self._set(self.index(name), value)
        if self.get(name) != value:
            raise HostPolicyUnavailable(f"mi_option_set({name}, {value}) did not take effect")


def torch_libc10_path() -> str:
    import torch

    return os.path.join(os.path.dirname(torch.__file__), "lib", "libc10.so")


def apply_purge_policy(
    path: Optional[str] = None, purge_delay_ms: int = PURGE_DELAY_MS
) -> Tuple[int, int]:
    """Set mimalloc's purge delay in this process; return (previous, new) in ms.

    Raises:
        HostPolicyUnavailable: before changing anything, if any check fails.
    """
    options = MimallocOptions(path or torch_libc10_path())
    previous = options.get(PURGE_DELAY_OPTION)
    if previous != purge_delay_ms:
        options.set(PURGE_DELAY_OPTION, purge_delay_ms)
    return previous, purge_delay_ms


def install_host_policy(where: str = "after_warmup") -> bool:
    """Entry point of the ``host_page_retention`` package (``host_policy``); returns True when engaged.

    Called once per rank by the executor after warmup. Logs the decision once
    (``[glm_moe_dsa] host_page_retention/host_policy``); on any failed check the allocator is left
    as it was.
    """
    from ...dispatch import log_capability

    try:
        previous, new = apply_purge_policy()
    except HostPolicyUnavailable as exc:
        return log_capability("host_page_retention", "host_policy", False, f"{where}: {exc}")
    return log_capability(
        "host_page_retention",
        "host_policy",
        True,
        f"{where}: mimalloc {PURGE_DELAY_OPTION} {previous} -> {new} ms in libc10.so",
    )
