# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One shared, pinned host copy per node of DeepSeek-V4.1's Engram embedding tables.

The Engram tables (``layers.{1,14}.engram.embed.weight`` ``[~384M, 256]``
float8_e4m3fn plus ``.scale`` ``[~384M, 8]`` float8_e8m0fnu, 202.76 GB in all)
are larger than the rest of the checkpoint. Under ``host_offload`` they stay in
host memory and each GPU gathers the rows it needs through a mapped pointer
(the target's ``ops/embedding/engram_gather``). This module owns the host side, with two
backends selected by ``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND``
(``host_table_backend``):

* ``vmm`` (the default): CUDA VMM host-NUMA allocations (``cuMemCreate`` with
  ``CU_MEM_LOCATION_TYPE_HOST_NUMA``, 1 GiB slabs alternating between the CPU
  sockets the creating process may allocate on: both when it sees GPUs of both
  sockets, else its GPUs' socket), shared between processes as POSIX file descriptors over a
  node-local Unix socket. The GPU maps them in its own page tables at the
  allocation granularity, so a random row gather is not bound by the CPU page
  tables' 64 KiB pages. See ``VmmEngramHostTable``.
* ``shm``: the ``/dev/shm`` file described below, registered with
  ``cudaHostRegister``.

The rest of this docstring describes the ``shm`` backend.

* **One copy per node.** Local rank 0 fills a file on a tmpfs (``/dev/shm`` by
  default) and every local rank maps that file ``MAP_SHARED``, so the node holds
  the bytes once however many ranks map them. A named file,
  unlike a ``memfd`` handed between ranks, needs no common PID namespace and
  outlives the engine, which is what lets the next boot skip the fill.
* **Pinned per rank.** Each rank ``cudaHostRegister``s the table regions of its
  own mapping and checks that the tensor views report ``is_pinned()``;
  ``cudaHostGetDevicePointer`` alone also succeeds on pageable memory.
* **A fast, checked fill.** Parallel ``pread`` of the safetensors byte ranges
  straight into the shared mapping, with no intermediate tensors, and a CRC32
  per 64 MiB chunk recorded in a manifest at the head of the file.
* **Reuse.** If the file exists, is complete, and its manifest names the same
  checkpoint bytes (shard path, size, mtime, tensor offsets and shapes), the
  fill is skipped and a sample of the chunk CRCs is re-checked.

File layout: a 1 MiB header (magic, version, state, JSON manifest) followed by
the tensors, each starting at a multiple of 512 MiB, in a file whose size is a
multiple of 512 MiB. That is a PMD huge page on a 64 KiB-page arm64 kernel and a
multiple of 2 MiB elsewhere, so a tmpfs that allows huge pages (``huge=`` mount
option, or ``shmem_enabled`` not ``never``) or a hugetlbfs file
(``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_PATH``) can back the tensors with them. The
gaps are never written and occupy no memory on a tmpfs.

Page size decides the gather rate. The GPU translates every row address through
the process's page tables. With 64 KiB pages and a table past roughly 10 GB,
random row gathers become translation-bound. Measured on GB300 at 4096 tokens
x 24 lookups: 3.1 ms per call over the full 101 GB layer table, against 0.14 ms
for the same table on 512 MiB pages or on 2 MiB-granular CUDA host allocations.
``fill_shared_engram_table`` asks for huge pages (``MADV_HUGEPAGE``) and records in
the manifest whether it got them.

Placement: ``numa="interleave"`` (the default) spreads the pages over both
sockets. Every GPU then reads at its local rate, whereas memory on the remote
socket alone was 1.7x slower.

Lifetime: every attached process holds a shared ``flock`` on ``<path>.lock``
for as long as it is attached, and the kernel drops that lock when the process
exits or dies. The file is kept for reuse by default.
``remove_shared_engram_table`` unlinks it only when no process holds that lock,
unless ``force`` is given. A process that dies leaves nothing pinned, and the
survivors' mappings are unaffected. After an unlink, tmpfs frees the pages when
the last mapping goes away. A fill with ``owned=True`` (what an engine does)
also writes ``<path>.owner`` naming the new file; ``close(remove_if_owned=True)``
then removes that file once its last holder detaches. A file filled from the
command line has no owner and stays until ``remove``, so consecutive engines
reuse it.

Residency switch: ``TRTLLM_DSV41_ENGRAM_HOST_OFFLOAD`` (``host_offload_enabled``)
decides whether the DeepSeek-V4.1 target keeps its Engram tables here (unset or
a true value) or row-sharded on the devices (a false value).

The module imports only the standard library and torch at load time, so it also
runs as a script without importing ``tensorrt_llm``:
``python engram_host_table.py {fill,status,verify,remove} ...``.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import logging
import math
import mmap
import os
import platform
import re
import socket
import struct
import threading
import time
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "EngramTableSource",
    "FillReport",
    "SharedEngramHostTable",
    "VmmEngramHostTable",
    "attach_shared_engram_table",
    "default_table_path",
    "describe_shared_engram_table",
    "fill_shared_engram_table",
    "find_engram_table_sources",
    "host_offload_enabled",
    "host_table_backend",
    "open_shared_engram_table",
    "open_vmm_engram_table",
    "remove_shared_engram_table",
    "verify_shared_engram_table",
]

_logger = logging.getLogger(__name__)

PATH_ENV = "TRTLLM_DSV41_ENGRAM_HOST_TABLE_PATH"
"""str: environment variable overriding the shared file's path."""

HOST_OFFLOAD_ENV = "TRTLLM_DSV41_ENGRAM_HOST_OFFLOAD"
"""str: environment variable selecting the Engram tables' residency (see ``host_offload_enabled``)."""


def host_offload_enabled() -> bool:
    """Whether the Engram tables live in this module's shared pinned host copy.

    ``$TRTLLM_DSV41_ENGRAM_HOST_OFFLOAD`` unset: yes. ``1``/``true``/``yes``/``on``:
    yes. ``0``/``false``/``no``/``off``: no, the tables are row-sharded on the
    devices. Anything else, the empty string included, raises: a typo must not
    silently select a residency.
    """
    value = os.environ.get(HOST_OFFLOAD_ENV)
    if value is None:
        return True
    token = value.strip().lower()
    if token in ("1", "true", "yes", "on"):
        return True
    if token in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{HOST_OFFLOAD_ENV} must be a boolean value, got {value!r}")


BACKEND_ENV = "TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND"
"""str: environment variable selecting the host table's backend (see ``host_table_backend``)."""


def host_table_backend() -> str:
    """``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND``: ``vmm`` (unset: the default) or ``shm``.

    Anything else raises, so a typo cannot silently select a backend.
    """
    value = os.environ.get(BACKEND_ENV)
    if value is None:
        return "vmm"
    token = value.strip().lower()
    if token in ("vmm", "shm"):
        return token
    raise ValueError(f"{BACKEND_ENV} must be 'vmm' or 'shm', got {value!r}")


_MAGIC = b"DSV41ENG"
_VERSION = 1
_STATE_FILLING = 0
_STATE_READY = 1
_HEADER = struct.Struct("<8sIIQQ")  # magic, version, state, manifest bytes, file bytes
_MANIFEST_AT = 64
_HEADER_BYTES = 1 << 20
_ALIGN = 1 << 29
_CHUNK_BYTES = 1 << 26
_DIRECT_ALIGN = 1 << 20
_SAMPLE_CHUNKS = 8

_ENGRAM_KEY = re.compile(r"layers\.(\d+)\.engram\.embed\.(weight|scale)")
_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "U8": torch.uint8,
}
_MPOL_BIND = 2
_MPOL_INTERLEAVE = 3
_SYS_MBIND = {"aarch64": 235, "x86_64": 237}


def default_table_path() -> str:
    """Path of the node's shared table file: ``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_PATH``, else
    ``/dev/shm/trtllm-dsv41-engram-<uid>.bin``.

    A fixed name per user means a node can hold at most one table per user: a
    fill for different checkpoint bytes replaces the file rather than adding a
    second one.
    """
    return os.environ.get(PATH_ENV) or f"/dev/shm/trtllm-dsv41-engram-{os.getuid()}.bin"  # nosec B108


@dataclass(frozen=True)
class EngramTableSource:
    """One Engram table tensor as stored in a safetensors shard.

    Attributes:
        key: checkpoint name, e.g. ``layers.1.engram.embed.weight``.
        layer: the decoder layer the table belongs to.
        shard: absolute path of the shard that holds it.
        offset: absolute byte offset of the tensor's first byte in ``shard``.
        nbytes: tensor size in bytes (every element is one byte).
        shape: ``(rows, 256)`` for a weight, ``(rows, 8)`` for a scale.
        dtype: safetensors dtype string, ``F8_E4M3`` or ``F8_E8M0``.
    """

    key: str
    layer: int
    shard: str
    offset: int
    nbytes: int
    shape: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class FillReport:
    """What ``fill_shared_engram_table`` did.

    Attributes:
        path: the shared file.
        reused: True if a matching complete file was found and the fill was skipped.
        nbytes: table bytes (the sum of the tensors, not counting the header or gaps).
        seconds: wall time of the call.
        threads: reader threads used (or that would have been used).
        method: ``buffered`` or ``direct``.
        numa: the page placement policy applied to a new file.
        verified_chunks: chunk CRCs re-checked on reuse.
        numa_bytes: resident table bytes per NUMA node after a fill (empty on reuse).
    """

    path: str
    reused: bool
    nbytes: int
    seconds: float
    threads: int
    method: str
    numa: str
    verified_chunks: int
    numa_bytes: Dict[int, int]

    @property
    def gbps(self) -> float:
        """Table bytes per second of wall time, in GB/s."""
        return self.nbytes / self.seconds / 1e9 if self.seconds > 0 else math.inf


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _read_safetensors_header(path: str) -> Tuple[int, dict]:
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return 8 + length, json.loads(f.read(length))


def find_engram_table_sources(checkpoint_dir: str) -> List[EngramTableSource]:
    """Locate every ``layers.<L>.engram.embed.{weight,scale}`` tensor of a checkpoint.

    Reads ``model.safetensors.index.json`` and the header of each shard that
    holds a table. Returns the tensors ordered by layer, weight before scale.
    """
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    headers: Dict[str, Tuple[int, dict]] = {}
    found = []
    for key, shard_name in weight_map.items():
        match = _ENGRAM_KEY.fullmatch(key)
        if match is None:
            continue
        shard = os.path.realpath(os.path.join(checkpoint_dir, shard_name))
        if shard not in headers:
            headers[shard] = _read_safetensors_header(shard)
        data_start, header = headers[shard]
        entry = header[key]
        begin, end = entry["data_offsets"]
        shape = tuple(int(d) for d in entry["shape"])
        if entry["dtype"] not in _DTYPES:
            raise ValueError(f"{key}: unsupported dtype {entry['dtype']}")
        if end - begin != math.prod(shape):
            raise ValueError(
                f"{key}: {end - begin} bytes for shape {shape}; expected one byte per element"
            )
        found.append(
            EngramTableSource(
                key,
                int(match.group(1)),
                shard,
                data_start + begin,
                end - begin,
                shape,
                entry["dtype"],
            )
        )
    if not found:
        raise ValueError(f"no layers.<L>.engram.embed.{{weight,scale}} tensors in {index_path}")
    found.sort(key=lambda s: (s.layer, s.key.endswith(".scale")))
    by_layer: Dict[int, List[EngramTableSource]] = {}
    for source in found:
        by_layer.setdefault(source.layer, []).append(source)
    for layer, pair in by_layer.items():
        if len(pair) != 2 or pair[0].shape[0] != pair[1].shape[0]:
            raise ValueError(
                f"layer {layer}: expected a weight and a scale with equal row counts, got {pair}"
            )
    return found


def _layout(sources: Sequence[EngramTableSource]) -> dict:
    """Where each tensor goes in the shared file, and the checkpoint bytes it comes from."""
    offset = _round_up(_HEADER_BYTES, _ALIGN)
    tensors = []
    for source in sources:
        st = os.stat(source.shard)
        tensors.append(
            {
                "key": source.key,
                "layer": source.layer,
                "shape": list(source.shape),
                "dtype": source.dtype,
                "offset": offset,
                "nbytes": source.nbytes,
                "source": {
                    "shard": source.shard,
                    "shard_bytes": st.st_size,
                    "shard_mtime_ns": st.st_mtime_ns,
                    "offset": source.offset,
                },
            }
        )
        offset = _round_up(offset + source.nbytes, _ALIGN)
    last = tensors[-1]
    return {
        "format": _VERSION,
        "align": _ALIGN,
        "chunk_bytes": _CHUNK_BYTES,
        "total_bytes": _round_up(last["offset"] + last["nbytes"], _ALIGN),
        "tensors": tensors,
    }


def _identity(manifest: dict) -> dict:
    return {
        k: manifest.get(k) for k in ("format", "align", "chunk_bytes", "total_bytes", "tensors")
    }


def _read_header(path: str) -> Tuple[Optional[int], Optional[dict], int]:
    """(state, manifest, file bytes); state and manifest are None if the file has no valid header."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        raw = os.pread(fd, _HEADER_BYTES, 0)
    finally:
        os.close(fd)
    if len(raw) < _MANIFEST_AT:
        return None, None, size
    magic, version, state, manifest_bytes, _ = _HEADER.unpack_from(raw)
    if magic != _MAGIC or version != _VERSION or manifest_bytes > _HEADER_BYTES - _MANIFEST_AT:
        return None, None, size
    try:
        manifest = json.loads(raw[_MANIFEST_AT : _MANIFEST_AT + manifest_bytes])
    except ValueError:
        return None, None, size
    return state, manifest, size


def _write_header(mm: mmap.mmap, state: int, manifest: dict) -> None:
    body = json.dumps(manifest, separators=(",", ":")).encode()
    if len(body) > _HEADER_BYTES - _MANIFEST_AT:
        raise ValueError(
            f"manifest of {len(body)} bytes does not fit the {_HEADER_BYTES}-byte header"
        )
    mm[_MANIFEST_AT : _MANIFEST_AT + len(body)] = body
    mm[: _HEADER.size] = _HEADER.pack(_MAGIC, _VERSION, state, len(body), manifest["total_bytes"])


def _check(path: str, layout: Optional[dict]) -> Tuple[Optional[dict], str]:
    """The manifest of a complete file matching ``layout`` (any layout if None), else (None, why not)."""
    try:
        state, manifest, size = _read_header(path)
    except FileNotFoundError:
        return None, "absent"
    if manifest is None:
        return None, "no valid header"
    if state != _STATE_READY:
        return None, "incomplete (a fill started and did not finish)"
    if size != manifest["total_bytes"]:
        return None, f"{size} bytes, manifest says {manifest['total_bytes']}"
    if layout is not None and _identity(manifest) != _identity(layout):
        return None, "made from different checkpoint bytes"
    return manifest, "ready"


def _lock_path(path: str) -> str:
    return path + ".lock"


def _open_lock(path: str) -> int:
    return os.open(_lock_path(path), os.O_RDWR | os.O_CREAT, 0o600)


def _owner_path(path: str) -> str:
    return path + ".owner"


def _write_owner(path: str, owned: bool) -> None:
    """Record (or clear) that the file now at ``path`` was created by an engine."""
    marker = _owner_path(path)
    if owned:
        with open(marker, "w") as f:
            f.write(str(os.stat(path).st_ino))
        return
    try:
        os.unlink(marker)
    except FileNotFoundError:
        pass


def _owned_inode(path: str) -> Optional[int]:
    """The inode an engine recorded as its own at ``path``, or None."""
    try:
        with open(_owner_path(path)) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _flock_within(fd: int, operation: int, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(what) from None
            time.sleep(0.2)


def _lock_holders(path: str) -> List[Tuple[int, str]]:
    """(pid, READ|WRITE) of every flock on ``<path>.lock``, from /proc/locks."""
    try:
        inode = os.stat(_lock_path(path)).st_ino
        with open("/proc/locks") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    holders = []
    for line in lines:
        fields = line.split()
        if "->" in fields:
            fields.remove("->")
        if len(fields) >= 6 and fields[1] == "FLOCK" and int(fields[5].split(":")[-1]) == inode:
            holders.append((int(fields[4]), fields[3]))
    return holders


def _meminfo() -> Dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            name, value = line.split(":", 1)
            parts = value.split()
            out[name] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


def _cgroup_available() -> Optional[int]:
    """Bytes this cgroup (v2) can still charge before its memory.max, counting clean page cache
    as reclaimable; None without a limit."""
    try:
        with open("/proc/self/cgroup") as f:
            rel = next(line.split("::", 1)[1].strip() for line in f if line.startswith("0::"))
        root = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
        with open(os.path.join(root, "memory.max")) as f:
            limit = f.read().strip()
        if limit == "max":
            return None
        with open(os.path.join(root, "memory.current")) as f:
            current = int(f.read())
        stat = {}
        with open(os.path.join(root, "memory.stat")) as f:
            for line in f:
                name, value = line.split()
                stat[name] = int(value)
    except (OSError, StopIteration, ValueError):
        return None
    return int(limit) - current + max(0, stat.get("file", 0) - stat.get("shmem", 0))


def _check_capacity(path: str, nbytes: int, headroom_bytes: int) -> None:
    st = os.statvfs(os.path.dirname(os.path.abspath(path)))
    fs_free = st.f_bavail * st.f_frsize
    if fs_free < nbytes:
        raise RuntimeError(
            f"{os.path.dirname(path)} has {fs_free / 1e9:.1f} GB free; the table needs {nbytes / 1e9:.1f} GB"
        )
    available = _meminfo().get("MemAvailable", 0)
    cgroup = _cgroup_available()
    limit = available if cgroup is None else min(available, cgroup)
    if limit < nbytes + headroom_bytes:
        raise RuntimeError(
            f"the table needs {nbytes / 1e9:.1f} GB plus {headroom_bytes / 1e9:.1f} GB headroom; "
            f"{available / 1e9:.1f} GB MemAvailable, "
            f"cgroup {'unlimited' if cgroup is None else f'{cgroup / 1e9:.1f} GB'}"
        )


def _node_list(text: str) -> List[int]:
    nodes: List[int] = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            nodes.extend(range(int(lo), int(hi or lo) + 1))
    return nodes


def _device_socket_cpus(device: torch.device) -> Optional[set]:
    """CPUs of the NUMA node the device's PCI function reports, or None if it reports none."""
    from cuda.bindings import runtime as cudart

    err, bus = cudart.cudaDeviceGetPCIBusId(64, device.index)
    if err != cudart.cudaError_t.cudaSuccess:
        return None
    domain, bus_no, rest = bytes(bus).split(b"\x00", 1)[0].decode().lower().split(":")
    function = f"{int(domain, 16):04x}:{int(bus_no, 16):02x}:{rest}"
    try:
        with open(f"/sys/bus/pci/devices/{function}/numa_node") as f:
            node = int(f.read())
        if node < 0:
            return None
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            return set(_node_list(f.read()))
    except OSError:
        return None


def _cpu_memory_nodes() -> List[int]:
    """NUMA nodes with both CPUs and memory (on GB300: the two Grace sockets, not the GPU nodes)."""
    with open("/sys/devices/system/node/has_cpu") as f:
        cpu = set(_node_list(f.read()))
    with open("/sys/devices/system/node/has_memory") as f:
        mem = set(_node_list(f.read()))
    return sorted(cpu & mem)


def _mbind(addr: int, length: int, mode: int, nodes: Sequence[int]) -> None:
    number = _SYS_MBIND.get(platform.machine())
    if number is None:
        raise RuntimeError(f"no mbind syscall number for {platform.machine()}")
    words = max(nodes) // 64 + 1
    mask = (ctypes.c_ulong * words)()
    for node in nodes:
        mask[node // 64] |= 1 << (node % 64)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(
        ctypes.c_long(number),
        ctypes.c_void_p(addr),
        ctypes.c_ulong(length),
        ctypes.c_int(mode),
        mask,
        ctypes.c_ulong(words * 64 + 1),
        ctypes.c_uint(0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mbind: {os.strerror(err)}")


def _parse_numa_policy(numa: str) -> Tuple[Optional[int], List[int]]:
    """``interleave`` -> all CPU+memory nodes; ``local`` -> first touch; ``<n>`` -> bind to node n."""
    if numa == "local":
        return None, []
    if numa == "interleave":
        return _MPOL_INTERLEAVE, _cpu_memory_nodes()
    return _MPOL_BIND, [int(numa)]


def _mapping_numa_bytes(start: int, end: int) -> Dict[int, int]:
    """Resident bytes per NUMA node of this process's mappings inside [start, end)."""
    out: Dict[int, int] = {}
    with open("/proc/self/numa_maps") as f:
        for line in f:
            fields = line.split()
            if not start <= int(fields[0], 16) < end:
                continue
            page = next(
                (int(x.split("=")[1]) * 1024 for x in fields if x.startswith("kernelpagesize_kB=")),
                mmap.PAGESIZE,
            )
            for x in fields:
                if x[0] == "N" and "=" in x:
                    node, pages = x[1:].split("=")
                    out[int(node)] = out.get(int(node), 0) + int(pages) * page
    return out


def _mapping_smaps(start: int, end: int) -> Dict[str, int]:
    """Sum of this process's smaps fields (in bytes) over the mappings inside [start, end)."""
    wanted = (
        "Rss",
        "Pss",
        "Shared_Clean",
        "Shared_Dirty",
        "Private_Clean",
        "Private_Dirty",
        "AnonHugePages",
        "ShmemPmdMapped",
        "FilePmdMapped",
        "Locked",
    )
    out = dict.fromkeys(wanted, 0)
    inside = False
    with open("/proc/self/smaps") as f:
        for line in f:
            head = line.split(None, 1)[0]
            if "-" in head and not head.endswith(":"):
                inside = start <= int(head.split("-")[0], 16) < end
            elif inside and head[:-1] in out:
                out[head[:-1]] += int(line.split()[1]) * 1024
    return out


def _mmap_address(mm: mmap.mmap) -> int:
    anchor = ctypes.c_char.from_buffer(mm)
    try:
        return ctypes.addressof(anchor)
    finally:
        del anchor


def _read_buffered(fd: int, dst: memoryview, src_offset: int) -> None:
    done = 0
    while done < len(dst):
        n = os.preadv(fd, [dst[done:]], src_offset + done)
        if n <= 0:
            raise OSError(f"short read at byte {src_offset + done} ({done} of {len(dst)})")
        done += n


class _BounceBuffer:
    """A page-aligned buffer that one reader thread uses for its O_DIRECT reads."""

    def __init__(self, nbytes: int) -> None:
        self.buffer = mmap.mmap(-1, nbytes)
        self.view = memoryview(self.buffer)
        self._anchor = ctypes.c_char.from_buffer(self.buffer)
        self.address = ctypes.addressof(self._anchor)

    def release(self) -> None:
        self.view.release()
        del self._anchor
        self.buffer.close()


class _PerThreadBounce(threading.local):
    """Gives each thread its own ``_BounceBuffer`` and records it for release by the caller."""

    def __init__(
        self, nbytes: int, registry: List[_BounceBuffer], registry_lock: threading.Lock
    ) -> None:
        super().__init__()
        self.buffer = _BounceBuffer(nbytes)
        with registry_lock:
            registry.append(self.buffer)


def _parallel_fill(mm: mmap.mmap, layout: dict, threads: int, method: str) -> Dict[str, List[int]]:
    """Copy every tensor's checkpoint bytes into ``mm``; returns the CRC32 of each chunk."""
    if method not in ("buffered", "direct"):
        raise ValueError(f"method must be 'buffered' or 'direct', got {method!r}")
    chunk = layout["chunk_bytes"]
    jobs = []
    crcs: Dict[str, List[int]] = {}
    for t in layout["tensors"]:
        count = -(-t["nbytes"] // chunk)
        crcs[t["key"]] = [0] * count
        for i in range(count):
            n = min(chunk, t["nbytes"] - i * chunk)
            jobs.append(
                (
                    i,
                    t["key"],
                    t["source"]["shard"],
                    t["source"]["offset"] + i * chunk,
                    t["offset"] + i * chunk,
                    n,
                )
            )
    jobs.sort()  # chunk-major, so every shard is read concurrently
    flags = os.O_RDONLY | (os.O_DIRECT if method == "direct" else 0)
    fds = {shard: os.open(shard, flags) for shard in {j[2] for j in jobs}}
    view = memoryview(mm)
    bounces: List[_BounceBuffer] = []
    base = _mmap_address(mm) if method == "direct" else 0
    local = (
        _PerThreadBounce(chunk + 2 * _DIRECT_ALIGN, bounces, threading.Lock())
        if method == "direct"
        else None
    )

    def run(job: tuple) -> None:
        idx, key, shard, src, dst, n = job
        fd = fds[shard]
        if method == "direct":
            bounce = local.buffer
            lo = src - src % _DIRECT_ALIGN
            need = src + n - lo
            span = _round_up(need, _DIRECT_ALIGN)
            got = 0
            while got < need:
                r = os.preadv(fd, [bounce.view[got:span]], lo + got)
                if r <= 0:
                    raise OSError(f"short O_DIRECT read at byte {lo + got} of {shard}")
                got += r
            ctypes.memmove(base + dst, bounce.address + (src - lo), n)
        else:
            _read_buffered(fd, view[dst : dst + n], src)
            os.posix_fadvise(fd, src, n, os.POSIX_FADV_DONTNEED)
        crcs[key][idx] = zlib.crc32(view[dst : dst + n])

    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(run, jobs):
                pass
    finally:
        for fd in fds.values():
            os.close(fd)
        for bounce in bounces:
            bounce.release()
        view.release()
    return crcs


def _chunks_to_verify(manifest: dict, mode: str) -> List[Tuple[dict, int]]:
    if mode not in ("none", "sample", "full"):
        raise ValueError(f"verify must be 'none', 'sample' or 'full', got {mode!r}")
    picks = []
    for t in manifest["tensors"]:
        count = len(manifest["crc32"][t["key"]])
        if mode == "full":
            idxs = range(count)
        elif mode == "sample":
            idxs = sorted(
                {
                    round(i * (count - 1) / max(1, _SAMPLE_CHUNKS - 1))
                    for i in range(min(count, _SAMPLE_CHUNKS))
                }
            )
        else:
            idxs = []
        picks.extend((t, i) for i in idxs)
    return picks


def _verify(path: str, manifest: dict, mode: str, threads: int) -> Tuple[int, List[str]]:
    """(chunks checked, chunks whose CRC differs from the manifest)."""
    picks = _chunks_to_verify(manifest, mode)
    if not picks:
        return 0, []
    chunk = manifest["chunk_bytes"]
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, manifest["total_bytes"], flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    view = memoryview(mm)

    def crc_differs(pick: Tuple[dict, int]) -> Optional[str]:
        t, i = pick
        lo = t["offset"] + i * chunk
        hi = t["offset"] + min((i + 1) * chunk, t["nbytes"])
        return (
            None if zlib.crc32(view[lo:hi]) == manifest["crc32"][t["key"]][i] else f"{t['key']}#{i}"
        )

    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            bad = [b for b in pool.map(crc_differs, picks) if b is not None]
    finally:
        view.release()
        mm.close()
    return len(picks), bad


def _create_and_fill(
    path: str, layout: dict, threads: int, method: str, numa: str
) -> Dict[int, int]:
    total = layout["total_bytes"]
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(fd, total)
        mm = mmap.mmap(fd, total, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    except BaseException:
        os.close(fd)
        os.unlink(path)
        raise
    os.close(fd)
    try:
        base = _mmap_address(mm)
        mode, nodes = _parse_numa_policy(numa)
        for t in layout["tensors"]:
            length = _round_up(t["nbytes"], mmap.PAGESIZE)
            try:
                mm.madvise(mmap.MADV_HUGEPAGE, t["offset"], length)
            except OSError:
                pass  # a kernel without THP; the table is served from base pages
            if mode is not None:
                _mbind(base + t["offset"], length, mode, nodes)
        _write_header(mm, _STATE_FILLING, layout)
        start = time.monotonic()
        crcs = _parallel_fill(mm, layout, threads, method)
        seconds = time.monotonic() - start
        numa_bytes = _mapping_numa_bytes(base, base + total)
        manifest = dict(layout)
        manifest["crc32"] = crcs
        manifest["fill"] = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "unix_time": time.time(),
            "seconds": seconds,
            "threads": threads,
            "method": method,
            "numa": numa,
            "numa_bytes": {str(k): v for k, v in numa_bytes.items()},
            "huge_page_bytes": _mapping_smaps(base, base + total)["ShmemPmdMapped"],
        }
        _write_header(mm, _STATE_READY, manifest)
    except BaseException:
        mm.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    mm.close()
    return numa_bytes


def fill_shared_engram_table(
    checkpoint_dir: str,
    path: Optional[str] = None,
    *,
    threads: Optional[int] = None,
    method: str = "buffered",
    numa: str = "interleave",
    verify: str = "sample",
    headroom_bytes: int = 64 << 30,
    lock_timeout_s: float = 1800.0,
    owned: bool = False,
) -> FillReport:
    """Make ``path`` hold the checkpoint's Engram tables, filling it only if it does not already.

    Call on one process per node (local rank 0). A complete file whose
    manifest names the same checkpoint bytes, and whose sampled chunk CRCs
    still match, is reused. Otherwise the file is removed and refilled under
    an exclusive lock, so the node never holds two copies. That waits for
    processes still attached to the old file to detach, up to
    ``lock_timeout_s``. A fill that fails removes its partial file. One that
    is killed leaves the file marked incomplete, and the next call refills it.

    Args:
        checkpoint_dir: the HF checkpoint directory (``model.safetensors.index.json``).
        path: the shared file; ``default_table_path()`` if None.
        threads: reader threads; ``min(64, cpu_count)`` if None.
        method: ``buffered`` (``preadv`` into the mapping, then drop the shard's
            page cache) or ``direct`` (``O_DIRECT`` into a per-thread buffer, then copy).
        numa: page placement for a new file: ``interleave`` over the NUMA nodes
            that have CPUs and memory, ``local`` (where the reader threads run),
            or a node number.
        verify: on reuse, re-check ``none``, a ``sample`` of 8 chunks per tensor,
            or the ``full`` set of chunk CRCs.
        headroom_bytes: host memory that must remain available after a new fill;
            the fill refuses to start otherwise.
        lock_timeout_s: how long to wait for the exclusive lock.
        owned: when this call creates the file, mark it as the caller's (``<path>.owner``)
            so that ``close(remove_if_owned=True)`` removes it after its last holder
            detaches; without it a new file carries no owner. A reused file keeps
            whatever owner it had.
    """
    path = path or default_table_path()
    layout = _layout(find_engram_table_sources(checkpoint_dir))
    table_bytes = sum(t["nbytes"] for t in layout["tensors"])
    threads = threads or min(64, os.cpu_count() or 8)
    start = time.monotonic()
    lock_fd = _open_lock(path)
    try:
        _flock_within(
            lock_fd, fcntl.LOCK_SH, lock_timeout_s, f"{path} is being filled by another process"
        )
        for exclusive in (False, True):
            manifest, reason = _check(path, layout)
            if manifest is not None:
                checked, bad = _verify(path, manifest, verify, threads)
                if not bad:
                    return FillReport(
                        path,
                        True,
                        table_bytes,
                        time.monotonic() - start,
                        threads,
                        method,
                        numa,
                        checked,
                        {},
                    )
                reason = f"CRC mismatch in chunks {bad[:4]}"
            if exclusive:
                break
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            _flock_within(
                lock_fd,
                fcntl.LOCK_EX,
                lock_timeout_s,
                f"{path} needs a refill ({reason}) but is still attached by {_lock_holders(path)}",
            )
        if os.path.exists(path):
            _logger.warning("replacing %s: %s", path, reason)
            os.unlink(path)
        _check_capacity(path, layout["total_bytes"], headroom_bytes)
        numa_bytes = _create_and_fill(path, layout, threads, method, numa)
        _write_owner(path, owned)
        return FillReport(
            path, False, table_bytes, time.monotonic() - start, threads, method, numa, 0, numa_bytes
        )
    finally:
        os.close(lock_fd)


def verify_shared_engram_table(
    path: Optional[str] = None, *, full: bool = True, threads: Optional[int] = None
) -> Tuple[int, List[str]]:
    """Recompute chunk CRCs of a complete table file: (chunks checked, chunks that differ)."""
    path = path or default_table_path()
    manifest, reason = _check(path, None)
    if manifest is None:
        raise RuntimeError(f"{path}: {reason}")
    return _verify(
        path, manifest, "full" if full else "sample", threads or min(64, os.cpu_count() or 8)
    )


class SharedEngramHostTable:
    """A process's mapping of the node's shared Engram table file.

    Obtain one with ``attach_shared_engram_table`` or ``open_shared_engram_table``.
    ``weight(layer)`` / ``scale(layer)`` are CPU tensor views of the mapping,
    ``[rows, 256]`` float8_e4m3fn and ``[rows, 8]`` float8_e8m0fnu, that
    ``engram_gather`` reads directly with ``row_start=0, row_end=rows``. Never
    write through them: every rank on the node shares these bytes. The mapping
    is read-only (a write faults) unless registering it needs write access, which
    is the case where the device lacks ``cudaDevAttrHostRegisterReadOnlySupported``
    (GB300 with driver 580 does). The views stay valid until ``close()``. A CUDA
    graph that captured a gather on them must be released before then.

    Register on the device's socket (``attach_shared_engram_table`` binds the
    registering thread there by default). The GPU walks this process's page
    tables, and with the process unbound the socket-0 GPUs of a GB300 node
    gathered 1.45x slower than with it bound (measured).
    """

    def __init__(
        self,
        path: str,
        manifest: dict,
        mm: mmap.mmap,
        lock_fd: int,
        read_only: bool,
        inode: int = 0,
    ) -> None:
        self.path = path
        """str: the shared file."""
        self.manifest = manifest
        """dict: the file's manifest (layout, source identity, chunk CRCs, fill record)."""
        self.read_only = read_only
        """bool: whether this process mapped the file read-only."""
        self.inode = inode
        """int: inode of the file this process mapped."""
        self.register_seconds = 0.0
        """float: wall time of this process's ``cudaHostRegister`` calls."""
        self.register_cpus: Optional[set] = None
        """set: CPUs the registering thread was bound to (its device's socket), or None if left unbound."""
        self.fill_report: Optional[FillReport] = None
        """FillReport: what ``open_shared_engram_table`` did on local rank 0; None on the other ranks."""
        self._mm = mm
        self._lock_fd = lock_fd
        self._registered: List[int] = []
        self._closed = False
        # One storage per tensor, starting at the tensor: Tensor.is_pinned() asks CUDA about the
        # storage's first byte, and only the tensor regions are registered, not the header.
        self._views: Dict[str, torch.Tensor] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # torch warns when the buffer is read-only
            for t in manifest["tensors"]:
                region = torch.frombuffer(
                    mm, dtype=torch.uint8, count=t["nbytes"], offset=t["offset"]
                )
                self._views[t["key"]] = region.view(_DTYPES[t["dtype"]]).view(*t["shape"])
        first = manifest["tensors"][0]
        self._base = self._views[first["key"]].data_ptr() - first["offset"]

    @property
    def layers(self) -> List[int]:
        """Decoder layers that have a table, ascending."""
        return sorted({t["layer"] for t in self.manifest["tensors"]})

    @property
    def nbytes(self) -> int:
        """Table bytes, all layers, weights and scales."""
        return sum(t["nbytes"] for t in self.manifest["tensors"])

    def weight(self, layer: int) -> torch.Tensor:
        """``[rows, 256]`` float8_e4m3fn host view of ``layers.<layer>.engram.embed.weight``."""
        return self._views[f"layers.{layer}.engram.embed.weight"]

    def scale(self, layer: int) -> torch.Tensor:
        """``[rows, 8]`` float8_e8m0fnu host view of ``layers.<layer>.engram.embed.scale``."""
        return self._views[f"layers.{layer}.engram.embed.scale"]

    def register(self, device: Optional[torch.device] = None) -> None:
        """``cudaHostRegister`` every table region of this mapping and check ``is_pinned()``.

        ``Portable | Mapped``, plus ``ReadOnly`` for a read-only mapping. The
        CUDA context of ``device`` (the current device if None) performs the
        registration; ``Portable`` makes it valid in every context of the process.
        Registration pins (and so populates) every page of the table in this process.
        """
        if self._registered:
            return
        from cuda.bindings import runtime as cudart

        flags = cudart.cudaHostRegisterPortable | cudart.cudaHostRegisterMapped
        if self.read_only:
            flags |= cudart.cudaHostRegisterReadOnly
        device = _cuda_device(device)
        start = time.monotonic()
        with torch.cuda.device(device):
            for key, view in self._views.items():
                (err,) = cudart.cudaHostRegister(view.data_ptr(), view.numel(), flags)
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        f"cudaHostRegister({key}, {view.numel()} bytes, read_only={self.read_only}) failed: {err}"
                    )
                self._registered.append(view.data_ptr())
        self.register_seconds = time.monotonic() - start
        unpinned = [k for k, v in self._views.items() if not v.is_pinned()]
        if unpinned:
            raise RuntimeError(f"registered but not reported pinned: {unpinned}")

    def numa_bytes(self) -> Dict[int, int]:
        """Resident bytes of this mapping per NUMA node (from /proc/self/numa_maps)."""
        return _mapping_numa_bytes(self._base, self._base + self.manifest["total_bytes"])

    def smaps(self) -> Dict[str, int]:
        """This process's smaps totals for the mapping, in bytes (Rss, Pss, ShmemPmdMapped, ...)."""
        return _mapping_smaps(self._base, self._base + self.manifest["total_bytes"])

    def check_gather_views(self, device: torch.device) -> None:
        """Raise unless every table view is page-locked and has a mapped device pointer on ``device``."""
        from cuda.bindings import runtime as cudart

        for key, view in self._views.items():
            if not view.is_pinned():
                raise RuntimeError(f"{key}: the shared table view is not page-locked")
            with torch.cuda.device(device):
                err, _ = cudart.cudaHostGetDevicePointer(view.data_ptr(), 0)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"{key}: cudaHostGetDevicePointer {err}")

    def describe(self) -> str:
        """One log line: backend, file, fill or reuse, registration."""
        report, cpus = self.fill_report, self.register_cpus
        return (
            f"backend=shm {self.path} {self.nbytes / 1e9:.2f} GB"
            + (
                f", {'reused' if report.reused else 'filled'} in {report.seconds:.2f} s"
                if report is not None
                else ""
            )
            + f", registered in {self.register_seconds:.2f} s on CPUs "
            + (f"{min(cpus)}-{max(cpus)}" if cpus else "unbound")
            + f", read_only={self.read_only}"
        )

    def close(self, remove_if_unused: bool = False, remove_if_owned: bool = False) -> None:
        """Unregister, unmap and release this process's lock.

        Args:
            remove_if_unused: also unlink the file if no other process is attached.
            remove_if_owned: the same, but only when an engine's fill created the file
                (``fill_shared_engram_table(owned=True)``); a file filled from the command
                line is left for the next attach.
        """
        if self._closed:
            return
        self._closed = True
        if self._registered:
            from cuda.bindings import runtime as cudart

            for ptr in self._registered:
                cudart.cudaHostUnregister(ptr)
            self._registered.clear()
        self._views.clear()
        try:
            self._mm.close()
        except BufferError:
            _logger.warning(
                "%s: a caller still holds a table view; the mapping ends with it", self.path
            )
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        if remove_if_unused or (remove_if_owned and _owned_inode(self.path) == self.inode):
            remove_shared_engram_table(self.path)

    def __enter__(self) -> "SharedEngramHostTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _cuda_device(device: Optional[torch.device]) -> torch.device:
    device = torch.device("cuda") if device is None else torch.device(device)
    return device if device.index is not None else torch.device("cuda", torch.cuda.current_device())


def _read_only_registration_supported(device: torch.device) -> bool:
    from cuda.bindings import runtime as cudart

    err, value = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrHostRegisterReadOnlySupported, device.index
    )
    return err == cudart.cudaError_t.cudaSuccess and bool(value)


def attach_shared_engram_table(
    path: Optional[str] = None,
    *,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    register: bool = True,
    writable: Optional[bool] = None,
    wait_s: float = 0.0,
    bind_to_device_socket: bool = True,
) -> SharedEngramHostTable:
    """Map a complete table file in this process and (by default) register it.

    Args:
        path: the shared file; ``default_table_path()`` if None.
        checkpoint_dir: if given, the file must have been filled from exactly
            this checkpoint's bytes.
        device: the device whose context registers the mapping (current if None).
        register: ``cudaHostRegister`` the table regions (see ``SharedEngramHostTable.register``).
        writable: map read-write. None: only when ``register`` is set and the
            device cannot register read-only memory.
        wait_s: keep retrying this long while the file is absent, being filled or incomplete.
        bind_to_device_socket: run the registration on the CPUs of ``device``'s NUMA
            node that the calling thread may already use (that thread only; its previous
            affinity is restored). The registration populates this process's page tables
            for the whole table, and the GPU translates every gathered row through them.
    """
    path = path or default_table_path()
    if register:
        device = _cuda_device(device)
    if writable is None:
        writable = register and not _read_only_registration_supported(device)
    layout = _layout(find_engram_table_sources(checkpoint_dir)) if checkpoint_dir else None
    lock_fd = _open_lock(path)
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            reason = "being filled"
        else:
            manifest, reason = _check(path, layout)
            if manifest is not None:
                break
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        if time.monotonic() >= deadline:
            os.close(lock_fd)
            raise RuntimeError(f"shared Engram table {path} is not usable: {reason}")
        time.sleep(0.2)
    try:
        fd = os.open(path, os.O_RDWR if writable else os.O_RDONLY)
        try:
            inode = os.fstat(fd).st_ino
            prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
            mm = mmap.mmap(fd, manifest["total_bytes"], flags=mmap.MAP_SHARED, prot=prot)
        finally:
            os.close(fd)
    except BaseException:
        os.close(lock_fd)
        raise
    table = SharedEngramHostTable(path, manifest, mm, lock_fd, read_only=not writable, inode=inode)
    if register:
        previous = os.sched_getaffinity(0)
        # Narrow, never widen: a thread its launcher confined elsewhere stays where it is.
        cpus = (_device_socket_cpus(device) or set()) & previous if bind_to_device_socket else None
        try:
            if cpus:
                os.sched_setaffinity(0, cpus)
                table.register_cpus = cpus
            table.register(device)
        except BaseException:
            table.close()
            raise
        finally:
            if cpus:
                os.sched_setaffinity(0, previous)
    return table


def open_shared_engram_table(
    checkpoint_dir: str,
    *,
    local_rank: int,
    barrier: Optional[Callable[[], None]] = None,
    path: Optional[str] = None,
    device: Optional[torch.device] = None,
    wait_s: float = 1800.0,
    **fill_kwargs,
) -> SharedEngramHostTable:
    """The weight-loading entry point: local rank 0 fills (or reuses), every local rank attaches.

    Dispatches on ``host_table_backend()``: ``vmm`` goes to ``open_vmm_engram_table``
    (``path`` and the fill keywords then serve only its shm fallback, which a VMM
    create failure takes unless ``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND`` names
    ``vmm`` explicitly); ``shm`` fills and attaches the ``/dev/shm`` file as described here.

    Args:
        checkpoint_dir: the HF checkpoint directory.
        local_rank: this process's rank on the node; rank 0 fills.
        barrier: a node-local (or global) barrier, e.g. ``tensorrt_llm._utils.local_mpi_barrier``.
            With it, the other ranks attach after rank 0 returns. Without it,
            they poll the file for up to ``wait_s``.
        path: the shared file; ``default_table_path()`` if None.
        device: this rank's device (current if None).
        wait_s: how long a non-filling rank waits without a barrier.
        **fill_kwargs: forwarded to ``fill_shared_engram_table`` on rank 0.
    """
    if host_table_backend() == "vmm":
        shm_kwargs = {k: v for k, v in fill_kwargs.items() if k != "threads"}
        return open_vmm_engram_table(
            checkpoint_dir,
            local_rank=local_rank,
            barrier=barrier,
            device=device,
            wait_s=wait_s,
            threads=fill_kwargs.get("threads"),
            headroom_bytes=fill_kwargs.get("headroom_bytes", 64 << 30),
            shm_fallback_path=None if _vmm_forced() else (path or default_table_path()),
            shm_fill_kwargs=shm_kwargs,
        )
    path = path or default_table_path()
    report = None
    if local_rank == 0:
        report = fill_shared_engram_table(checkpoint_dir, path, **fill_kwargs)
        _logger.info(
            "Engram host table %s: %s %.1f GB in %.1f s",
            path,
            "reused" if report.reused else "filled",
            report.nbytes / 1e9,
            report.seconds,
        )
    if barrier is not None:
        barrier()
    table = attach_shared_engram_table(
        path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        wait_s=0.0 if barrier is not None else wait_s,
    )
    table.fill_report = report
    return table


# ---------------------------------------------------------------------------
# The vmm backend: CUDA VMM host-NUMA allocations shared by file descriptor.
#
# Why not the shm file: a gather through a registered tmpfs mapping is translated
# through the process's CPU page tables, and on a 64 KiB-page kernel whose shmem
# gets no huge pages (shmem_enabled=never, where MADV_COLLAPSE on shmem fails with
# EINVAL) a random-row gather over the 101 GB layer table is translation-bound:
# 3.1 ms at 4096 tokens x 24 lookups, against 0.14 ms for host-NUMA VMM memory,
# which the GPU maps in its own page tables at the allocation granularity.
#
# Sharing: the process that creates the table exports each slab's handle as a
# POSIX fd, and every process holding the table serves the fds over one abstract
# Unix socket per node, user and checkpoint layout (the first holder to bind the
# name serves; the others retry the bind every second, so the table stays
# reachable while any holder lives). A process imports the fds, maps them into its
# own address range and grants its device access. The physical memory lives until
# the last process that maps it exits.

_VMM_SLAB_BYTES = 1 << 30
_VMM_FD_BATCH = 64
_VMM_PROTOCOL = 1
_VMM_MIN_GRANULARITY = 1 << 21
_VMM_RETRY_MAX_SLEEP_S = 10.0

CREATE_RETRY_ENV = "TRTLLM_DSV41_ENGRAM_VMM_CREATE_RETRY_S"
"""str: environment variable bounding how long a VMM table create retries on host out-of-memory (seconds)."""


class VmmCreateError(RuntimeError):
    """The VMM table's host allocations could not be made (after the out-of-memory retries)."""


def vmm_create_retry_seconds() -> float:
    """``$TRTLLM_DSV41_ENGRAM_VMM_CREATE_RETRY_S`` (default 120): how long ``_vmm_create`` keeps retrying
    while the host is short of memory, for example while the previous engine's table is still being
    released. 0 tries once."""
    value = os.environ.get(CREATE_RETRY_ENV)
    if value is None:
        return 120.0
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            f"{CREATE_RETRY_ENV} must be a non-negative number of seconds, got {value!r}"
        )
    return seconds


def _vmm_forced() -> bool:
    """Whether ``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND`` names ``vmm`` explicitly (no shm fallback then)."""
    return os.environ.get(BACKEND_ENV, "").strip().lower() == "vmm"


def _numa_memory(nodes: Sequence[int]) -> Dict[int, Dict[str, int]]:
    """``{node: {"MemFree": bytes, "FilePages": bytes}}`` from ``/sys/devices/system/node/node*/meminfo``."""
    out: Dict[int, Dict[str, int]] = {}
    for node in nodes:
        fields: Dict[str, int] = {}
        try:
            with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
                for line in f:
                    parts = line.split()  # "Node 0 MemFree: 123456 kB"
                    if len(parts) >= 4 and parts[2].rstrip(":") in ("MemFree", "FilePages"):
                        fields[parts[2].rstrip(":")] = int(parts[3]) * 1024
        except OSError:
            pass
        out[node] = fields
    return out


def _cu_mem_create(size: int, node: int):
    """``cuMemCreate`` of ``size`` host bytes on NUMA ``node``: the raw (error, handle) result."""
    from cuda.bindings import driver as cu

    return cu.cuMemCreate(size, _vmm_prop(node), 0)


def default_vmm_socket_name(layout: dict) -> str:
    """Abstract Unix socket name serving the node's VMM table for ``layout``.

    ``$TRTLLM_DSV41_ENGRAM_HOST_TABLE_PATH`` (else ``trtllm-dsv41-engram-<uid>``)
    plus a hash of the layout identity, so two checkpoints never share a name.
    """
    base = os.environ.get(PATH_ENV) or f"trtllm-dsv41-engram-{os.getuid()}"
    ident = json.dumps(_identity(layout), sort_keys=True).encode()
    return f"{base}.vmm.{zlib.crc32(ident):08x}"


def _cu_call(result, what: str):
    """Unwrap a cuda-python driver result tuple, raising on an error code."""
    from cuda.bindings import driver as cu

    err = result[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")
    if len(result) == 1:
        return None
    return result[1] if len(result) == 2 else result[1:]


def _vmm_prop(node: int):
    from cuda.bindings import driver as cu

    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = node
    prop.requestedHandleTypes = (
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    return prop


def _visible_device_host_nodes() -> Dict[int, int]:
    """``{device ordinal: host NUMA node}`` of this process's visible devices (``CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID``;
    -1 where the device reports none)."""
    from cuda.bindings import driver as cu

    attr = getattr(cu.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID", None)
    out: Dict[int, int] = {}
    for i in range(int(_cu_call(cu.cuDeviceGetCount(), "cuDeviceGetCount"))):
        dev = _cu_call(cu.cuDeviceGet(i), "cuDeviceGet")
        out[i] = (
            -1
            if attr is None
            else int(_cu_call(cu.cuDeviceGetAttribute(attr, dev), "cuDeviceGetAttribute"))
        )
    return out


def _vmm_usable_nodes(nodes: Sequence[int]) -> Tuple[List[int], Dict[int, str]]:
    """The subset of ``nodes`` this process can place host-NUMA VMM memory on, and why each other node was refused.

    The driver refuses ``cuMemCreate(HOST_NUMA, node)`` with ``CUDA_ERROR_INVALID_VALUE`` for a
    CPU socket that none of the process's visible devices is attached to (for example socket 1
    when only GPUs 0 and 1 are visible). Each node is probed with a one-granule allocation, which
    is released at once. Needs a current CUDA context.
    """
    from cuda.bindings import driver as cu

    flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    usable: List[int] = []
    refused: Dict[int, str] = {}
    for node in nodes:
        prop = _vmm_prop(node)
        err, gran = cu.cuMemGetAllocationGranularity(prop, flag)
        if err != cu.CUresult.CUDA_SUCCESS:
            refused[node] = f"cuMemGetAllocationGranularity: {getattr(err, 'name', err)}"
            continue
        err, handle = cu.cuMemCreate(max(int(gran), _VMM_MIN_GRANULARITY), prop, 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            refused[node] = f"cuMemCreate: {getattr(err, 'name', err)}"
            continue
        _cu_call(cu.cuMemRelease(handle), "cuMemRelease")
        usable.append(node)
    return usable, refused


def _vmm_granularity(nodes: Sequence[int]) -> int:
    from cuda.bindings import driver as cu

    flag = cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    gran = max(
        int(
            _cu_call(
                cu.cuMemGetAllocationGranularity(_vmm_prop(n), flag),
                "cuMemGetAllocationGranularity",
            )
        )
        for n in nodes
    )
    gran = max(gran, _VMM_MIN_GRANULARITY)
    if _ALIGN % gran or _VMM_SLAB_BYTES % gran:
        raise RuntimeError(f"VMM granularity {gran} does not divide the table alignment {_ALIGN}")
    return gran


@dataclass(frozen=True)
class _VmmSlab:
    """One ``cuMemCreate`` allocation: ``size`` bytes at ``offset`` of the table's address range, on NUMA ``node``."""

    offset: int
    size: int
    node: int


def _vmm_plan(layout: dict, granularity: int, nodes: Sequence[int]) -> List[_VmmSlab]:
    """Cover every tensor of ``layout`` with slabs of at most 1 GiB, alternating over ``nodes``.

    With both sockets usable, alternating 1 GiB slabs spread each table evenly over them,
    which is what page interleaving does for the shm file: every GPU then reads half its
    rows from its own socket. With one usable socket every slab lands on it.
    """
    slabs: List[_VmmSlab] = []
    for t in layout["tensors"]:
        start = t["offset"]
        end = start + _round_up(t["nbytes"], granularity)
        while start < end:
            size = min(_VMM_SLAB_BYTES, end - start)
            slabs.append(_VmmSlab(start, size, nodes[len(slabs) % len(nodes)]))
            start += size
    return slabs


def _access_desc(location_type: str, location_id: int, flags: str):
    from cuda.bindings import driver as cu

    desc = cu.CUmemAccessDesc()
    desc.location.type = getattr(cu.CUmemLocationType, f"CU_MEM_LOCATION_TYPE_{location_type}")
    desc.location.id = location_id
    desc.flags = getattr(cu.CUmemAccess_flags, f"CU_MEM_ACCESS_FLAGS_PROT_{flags}")
    return desc


def _device_bytes(ptr: int, nbytes: int, device: torch.device) -> torch.Tensor:
    """A ``[nbytes]`` uint8 CUDA tensor on ``device`` over memory this module owns (no allocation, no copy).

    Built from the pointer and the device as given: ``torch.as_tensor`` on a
    ``__cuda_array_interface__`` would ask ``cudaPointerGetAttributes`` for the
    device, which names no device for host-NUMA memory.
    """
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, nbytes)
    view = torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (nbytes,), (1,))
    if view.data_ptr() != ptr or view.device != device:
        raise RuntimeError(
            f"view of {ptr:#x} landed at {view.data_ptr():#x} on {view.device}, expected {device}"
        )
    return view


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = sock.recv(n)
    if len(data) != n:
        raise RuntimeError(f"Engram VMM table server sent {len(data)} bytes, expected {n}")
    return data


def _vmm_fetch(name: str, deadline: float) -> Optional[Tuple[dict, List[int]]]:
    """(message, fds) from the node's server for ``name``, or None if nobody serves it before ``deadline``."""
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            sock.connect("\0" + name)
        except (ConnectionRefusedError, FileNotFoundError):
            sock.close()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)
            continue
        fds: List[int] = []
        try:
            sock.settimeout(120)
            (length,) = struct.unpack("<Q", _recv_exact(sock, 8))
            message = json.loads(_recv_exact(sock, length))
            want = len(message["slabs"])
            while len(fds) < want:
                data, got, _, _ = socket.recv_fds(sock, 16, _VMM_FD_BATCH)
                if not data:
                    raise RuntimeError(
                        f"Engram VMM table server closed after {len(fds)} of {want} fds"
                    )
                fds.extend(got)
            return message, fds
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise
        finally:
            sock.close()


class _VmmServer(threading.Thread):
    """Serves one table's fds on an abstract Unix socket whenever this process can bind the name."""

    def __init__(self, table: "VmmEngramHostTable") -> None:
        super().__init__(name="dsv41-engram-vmm-server", daemon=True)
        self._table = table
        self._stop_event = threading.Event()
        self.attempted = threading.Event()
        self.serving = False

    def run(self) -> None:
        while not self._stop_event.is_set():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                sock.bind("\0" + self._table.path)
            except OSError:
                sock.close()
                self.attempted.set()
                self._stop_event.wait(1.0)
                continue
            try:
                sock.listen(64)
                sock.settimeout(0.5)
                self.serving = True
                self.attempted.set()
                while not self._stop_event.is_set():
                    try:
                        conn, _ = sock.accept()
                    except socket.timeout:
                        continue
                    with conn:
                        try:
                            self._table._serve(conn)
                        except OSError as error:
                            _logger.warning("Engram VMM table: serving a client failed: %s", error)
            finally:
                self.serving = False
                sock.close()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)


class VmmEngramHostTable:
    """This process's mapping of the node's Engram tables in CUDA VMM host-NUMA memory.

    Obtain one with ``open_vmm_engram_table``. ``weight(layer)`` / ``scale(layer)``
    are CUDA tensor views of host memory (``[rows, 256]`` float8_e4m3fn and ``[rows,
    8]`` float8_e8m0fnu) that ``engram_gather`` reads with ``row_start=0,
    row_end=rows``; no device memory backs them. Never write through them: every
    process on the node shares these bytes. The views stay valid until ``close()``,
    and a CUDA graph that captured a gather on them must be released before then.
    """

    def __init__(
        self,
        name: str,
        manifest: dict,
        slabs: List[_VmmSlab],
        granularity: int,
        handles: list,
        fds: List[int],
        va: int,
        device: torch.device,
        created: bool,
    ) -> None:
        self.path = name
        """str: the abstract Unix socket name the table is served on."""
        self.manifest = manifest
        """dict: layout, source identity, chunk CRCs and the creator's fill record."""
        self.read_only = not created
        """bool: whether this process mapped the table without host write access (every importer)."""
        self.created = created
        """bool: whether this process created and filled the table."""
        self.register_seconds = 0.0
        """float: wall time of this process's import, map and access calls."""
        self.register_cpus: Optional[set] = None
        """None: the vmm backend binds no CPUs (kept for interface parity with the shm backend)."""
        self.fill_report: Optional[FillReport] = None
        """FillReport: what ``open_vmm_engram_table`` did on local rank 0; None on the other ranks."""
        self.device = device
        self._slabs = slabs
        self._granularity = granularity
        self._handles = handles
        self._fds = fds
        self._va = va
        self._closed = False
        self._server: Optional[_VmmServer] = None
        self._views: Dict[str, torch.Tensor] = {}
        with torch.cuda.device(device):
            for t in manifest["tensors"]:
                raw = _device_bytes(va + t["offset"], t["nbytes"], device)
                self._views[t["key"]] = raw.view(_DTYPES[t["dtype"]]).view(*t["shape"])

    @property
    def layers(self) -> List[int]:
        """Decoder layers that have a table, ascending."""
        return sorted({t["layer"] for t in self.manifest["tensors"]})

    @property
    def nbytes(self) -> int:
        """Table bytes, all layers, weights and scales."""
        return sum(t["nbytes"] for t in self.manifest["tensors"])

    @property
    def allocated_bytes(self) -> int:
        """Host bytes of the slabs (the tensors rounded up to the allocation granularity)."""
        return sum(s.size for s in self._slabs)

    def weight(self, layer: int) -> torch.Tensor:
        """``[rows, 256]`` float8_e4m3fn CUDA view of ``layers.<layer>.engram.embed.weight``."""
        return self._views[f"layers.{layer}.engram.embed.weight"]

    def scale(self, layer: int) -> torch.Tensor:
        """``[rows, 8]`` float8_e8m0fnu CUDA view of ``layers.<layer>.engram.embed.scale``."""
        return self._views[f"layers.{layer}.engram.embed.scale"]

    def host_buffer(self, key: str) -> memoryview:
        """Read-only CPU bytes of one tensor (this process has host read access to every slab)."""
        t = next(t for t in self.manifest["tensors"] if t["key"] == key)
        return (
            memoryview((ctypes.c_char * t["nbytes"]).from_address(self._va + t["offset"]))
            .cast("B")
            .toreadonly()
        )

    def numa_bytes(self) -> Dict[int, int]:
        """Allocated host bytes per NUMA node."""
        out: Dict[int, int] = {}
        for s in self._slabs:
            out[s.node] = out.get(s.node, 0) + s.size
        return out

    def verify(self, full: bool = True, threads: Optional[int] = None) -> Tuple[int, List[str]]:
        """Recompute chunk CRCs from this process's host view: (chunks checked, chunks that differ)."""
        picks = _chunks_to_verify(self.manifest, "full" if full else "sample")
        chunk = self.manifest["chunk_bytes"]
        base = memoryview(
            (ctypes.c_char * self.manifest["total_bytes"]).from_address(self._va)
        ).cast("B")

        def crc_differs(pick: Tuple[dict, int]) -> Optional[str]:
            t, i = pick
            lo = t["offset"] + i * chunk
            hi = t["offset"] + min((i + 1) * chunk, t["nbytes"])
            return (
                None
                if zlib.crc32(base[lo:hi]) == self.manifest["crc32"][t["key"]][i]
                else f"{t['key']}#{i}"
            )

        try:
            with ThreadPoolExecutor(max_workers=threads or min(64, os.cpu_count() or 8)) as pool:
                bad = [b for b in pool.map(crc_differs, picks) if b is not None]
        finally:
            base.release()
        return len(picks), bad

    def check_gather_views(self, device: torch.device) -> None:
        """Raise unless every table view is a CUDA view on ``device`` (what ``engram_gather`` must be handed:
        a CPU view of VMM memory passes ``cudaHostGetDevicePointer`` and then gathers wrong rows)."""
        for key, view in self._views.items():
            if not view.is_cuda or view.device != torch.device(device):
                raise RuntimeError(f"{key}: expected a CUDA view on {device}, got {view.device}")

    def describe(self) -> str:
        """One log line: backend, socket name, create or import, NUMA split, serving state."""
        report = self.fill_report
        numa = ",".join(f"node{n}={b / 1e9:.1f}GB" for n, b in sorted(self.numa_bytes().items()))
        return (
            f"backend=vmm {self.path} {self.nbytes / 1e9:.2f} GB in {len(self._slabs)} slabs ({numa})"
            + (
                f", {'imported' if report.reused else 'created and filled'} in {report.seconds:.2f} s"
                if report is not None
                else ""
            )
            + f", mapped in {self.register_seconds:.2f} s, {'creator' if self.created else 'importer'}"
            + f", serving={bool(self._server and self._server.serving)}"
        )

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(60)
        _, uid, _ = struct.unpack(
            "3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        )
        if uid != os.getuid():
            return
        body = json.dumps(
            {
                "protocol": _VMM_PROTOCOL,
                "manifest": self.manifest,
                "granularity": self._granularity,
                "slabs": [[s.offset, s.size, s.node] for s in self._slabs],
            },
            separators=(",", ":"),
        ).encode()
        conn.send(struct.pack("<Q", len(body)))
        conn.send(body)
        for i in range(0, len(self._fds), _VMM_FD_BATCH):
            batch = self._fds[i : i + _VMM_FD_BATCH]
            socket.send_fds(conn, [struct.pack("<I", len(batch))], batch)

    def start_serving(self, wait_s: float = 5.0) -> None:
        """Start (or keep) this process's server thread and wait for its first bind attempt."""
        if self._server is None:
            self._server = _VmmServer(self)
            self._server.start()
        self._server.attempted.wait(wait_s)

    def close(self, remove_if_unused: bool = False, remove_if_owned: bool = False) -> None:
        """Stop serving, unmap, release this process's handles and fds.

        The physical memory is freed when the last process that maps it closes or
        exits. ``remove_if_unused`` / ``remove_if_owned`` are accepted for interface
        parity with the shm backend and have nothing to remove.
        """
        del remove_if_unused, remove_if_owned
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.stop()
        self._views.clear()
        from cuda.bindings import driver as cu

        for s in self._slabs:
            cu.cuMemUnmap(self._va + s.offset, s.size)
        cu.cuMemAddressFree(self._va, self.manifest["total_bytes"])
        for h in self._handles:
            cu.cuMemRelease(h)
        self._handles = []
        for fd in self._fds:
            os.close(fd)
        self._fds = []

    def __enter__(self) -> "VmmEngramHostTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _vmm_map(
    slabs: List[_VmmSlab], handles: list, total_bytes: int, device: torch.device, host_write: bool
) -> int:
    """Reserve one address range, map every slab at its offset and grant ``device`` and the host access."""
    from cuda.bindings import driver as cu

    va = int(_cu_call(cu.cuMemAddressReserve(total_bytes, _ALIGN, 0, 0), "cuMemAddressReserve"))
    mapped = []
    try:
        for s, h in zip(slabs, handles):
            _cu_call(cu.cuMemMap(va + s.offset, s.size, 0, h, 0), "cuMemMap")
            mapped.append(s)
        # The creator writes through the host mapping; an importer asks for read-only access and
        # takes the first combination the driver accepts.
        if host_write:
            choices = [("READWRITE", "READWRITE")]
        else:
            choices = [("READ", "READ"), ("READWRITE", "READ"), ("READWRITE", "READWRITE")]
        descs = [
            [_access_desc("DEVICE", device.index, d), _access_desc("HOST", 0, h)]
            for d, h in choices
        ]
        for s in slabs:
            for i, pair in enumerate(descs):
                result = cu.cuMemSetAccess(va + s.offset, s.size, pair, len(pair))
                if result[0] == cu.CUresult.CUDA_SUCCESS:
                    break
                if i == len(descs) - 1:
                    _cu_call(result, f"cuMemSetAccess(device {device.index}, host)")
    except BaseException:
        for s in mapped:
            cu.cuMemUnmap(va + s.offset, s.size)
        cu.cuMemAddressFree(va, total_bytes)
        raise
    return va


def _vmm_allocate(
    slabs: List[_VmmSlab], ctx, threads: int, headroom_bytes: int
) -> Tuple[list, int]:
    """``cuMemCreate`` every slab; (handles, attempts).

    Host memory can be short for a while after another process released a table
    (its 203 GB are returned asynchronously, and VMM host memory is not charged to
    the cgroup), so a shortfall -- ``MemAvailable`` below the table plus
    ``headroom_bytes``, or ``CUDA_ERROR_OUT_OF_MEMORY`` from any slab -- releases
    whatever was allocated and retries with backoff for ``vmm_create_retry_seconds()``.
    Free memory per NUMA node is logged before the first attempt and after every
    shortfall; it is not a gate, since clean page cache counts against ``MemFree``
    and ``cuMemCreate`` reclaims it. Raises ``VmmCreateError`` when the time is up
    or on any other ``cuMemCreate`` error.
    """
    from cuda.bindings import driver as cu

    allocated = sum(s.size for s in slabs)
    need = {n: sum(s.size for s in slabs if s.node == n) for n in sorted({s.node for s in slabs})}
    retry_s = vmm_create_retry_seconds()
    deadline = time.monotonic() + retry_s

    def memory_line() -> str:
        numa = _numa_memory(list(need))
        cgroup = _cgroup_available()
        return (
            "need "
            + ", ".join(f"node{n} {b / 1e9:.1f} GB" for n, b in need.items())
            + "; MemFree "
            + ", ".join(f"node{n} {m.get('MemFree', 0) / 1e9:.1f} GB" for n, m in numa.items())
            + " (FilePages "
            + ", ".join(f"node{n} {m.get('FilePages', 0) / 1e9:.1f} GB" for n, m in numa.items())
            + f"); MemAvailable {_meminfo().get('MemAvailable', 0) / 1e9:.1f} GB"
            + f"; cgroup {'unlimited' if cgroup is None else f'{cgroup / 1e9:.1f} GB'}"
        )

    _logger.info("Engram VMM table create: %s", memory_line())
    attempts = 0
    while True:
        attempts += 1
        available = _meminfo().get("MemAvailable", 0)
        cgroup = _cgroup_available()
        limit = available if cgroup is None else min(available, cgroup)
        handles: list = [None] * len(slabs)
        failure: List[str] = []
        stop = threading.Event()

        def create(i: int) -> None:
            if stop.is_set():
                return
            _cu_call(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
            err, handle = _cu_mem_create(slabs[i].size, slabs[i].node)
            if err == cu.CUresult.CUDA_SUCCESS:
                handles[i] = handle
                return
            stop.set()
            failure.append(
                f"cuMemCreate({slabs[i].size}, node {slabs[i].node}): {getattr(err, 'name', err)}"
            )
            if err != cu.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
                raise VmmCreateError(failure[-1])

        short = None
        if limit < allocated + headroom_bytes:
            short = (
                f"{allocated / 1e9:.1f} GB plus {headroom_bytes / 1e9:.1f} GB headroom needed, "
                f"{limit / 1e9:.1f} GB available"
            )
        else:
            try:
                with ThreadPoolExecutor(max_workers=min(16, threads)) as pool:
                    for _ in pool.map(create, range(len(slabs))):
                        pass
            except BaseException:
                for h in handles:
                    if h is not None:
                        cu.cuMemRelease(h)
                raise
            if not failure:
                return handles, attempts
            for h in handles:
                if h is not None:
                    cu.cuMemRelease(h)
            short = failure[0]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VmmCreateError(
                f"the Engram VMM table could not be allocated after {attempts} attempt(s) over {retry_s:.0f} s "
                f"(${CREATE_RETRY_ENV}): {short}; {memory_line()}"
            )
        sleep_s = min(_VMM_RETRY_MAX_SLEEP_S, float(2 ** (attempts - 1)), remaining)
        _logger.warning(
            "Engram VMM table create attempt %d: host memory short (%s); %s; retrying in %.0f s (%.0f s left)",
            attempts,
            short,
            memory_line(),
            sleep_s,
            remaining,
        )
        time.sleep(sleep_s)


def _vmm_create(
    name: str, layout: dict, device: torch.device, threads: int, headroom_bytes: int
) -> Tuple[VmmEngramHostTable, float]:
    """Allocate every slab, map it read-write for this process and the host, fill it from the checkpoint.

    Slabs alternate over the CPU sockets this process may allocate on (``_vmm_usable_nodes``):
    both when it sees GPUs of both sockets, else only the socket of its GPUs. An importer on
    the other socket maps the same slabs and reads them over the socket link.
    """
    from cuda.bindings import driver as cu

    start = time.monotonic()
    with torch.cuda.device(device):
        torch.cuda.synchronize()  # makes the device's primary context current on this thread
        ctx = _cu_call(cu.cuCtxGetCurrent(), "cuCtxGetCurrent")
    if not int(ctx):
        ctx = _cu_call(cu.cuDevicePrimaryCtxRetain(device.index), "cuDevicePrimaryCtxRetain")
    candidates = _cpu_memory_nodes()
    _cu_call(cu.cuCtxPushCurrent(ctx), "cuCtxPushCurrent")
    try:
        nodes, refused = _vmm_usable_nodes(candidates)
    finally:
        _cu_call(cu.cuCtxPopCurrent(), "cuCtxPopCurrent")
    device_nodes = _visible_device_host_nodes()
    if not nodes:
        raise VmmCreateError(
            f"no CPU NUMA node of {candidates} accepts a host-NUMA VMM allocation from this process "
            f"(visible devices' host nodes {device_nodes}; {refused}); set {BACKEND_ENV}=shm"
        )
    if refused:
        _logger.info(
            "Engram VMM table: NUMA node(s) %s refuse host-NUMA allocations from this process (visible devices' "
            "host nodes %s: %s); placing every slab on %s",
            sorted(refused),
            device_nodes,
            refused,
            nodes,
        )
    granularity = _vmm_granularity(nodes)
    slabs = _vmm_plan(layout, granularity, nodes)
    handles, attempts = _vmm_allocate(slabs, ctx, threads, headroom_bytes)
    try:
        va = _vmm_map(slabs, handles, layout["total_bytes"], device, host_write=True)
    except BaseException:
        for h in handles:
            cu.cuMemRelease(h)
        raise
    alloc_seconds = time.monotonic() - start
    fds: List[int] = []
    try:
        buf = memoryview((ctypes.c_char * layout["total_bytes"]).from_address(va)).cast("B")
        fill_start = time.monotonic()
        crcs = _parallel_fill(buf, layout, threads, "buffered")
        fill_seconds = time.monotonic() - fill_start
        buf.release()
        posix_fd = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        for h in handles:
            fds.append(
                int(
                    _cu_call(
                        cu.cuMemExportToShareableHandle(h, posix_fd, 0),
                        "cuMemExportToShareableHandle",
                    )
                )
            )
    except BaseException:
        for fd in fds:
            os.close(fd)
        for s in slabs:
            cu.cuMemUnmap(va + s.offset, s.size)
        cu.cuMemAddressFree(va, layout["total_bytes"])
        for h in handles:
            cu.cuMemRelease(h)
        raise
    manifest = dict(layout)
    manifest["crc32"] = crcs
    manifest["fill"] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "unix_time": time.time(),
        "backend": "vmm",
        "alloc_seconds": alloc_seconds,
        "seconds": fill_seconds,
        "threads": threads,
        "granularity": granularity,
        "slab_bytes": _VMM_SLAB_BYTES,
        "numa_bytes": {str(n): sum(s.size for s in slabs if s.node == n) for n in nodes},
        "numa_nodes_refused": {str(n): why for n, why in refused.items()},
        "device_host_nodes": {str(d): n for d, n in device_nodes.items()},
        "create_attempts": attempts,
    }
    table = VmmEngramHostTable(
        name, manifest, slabs, granularity, handles, fds, va, device, created=True
    )
    table.register_seconds = alloc_seconds
    return table, alloc_seconds + fill_seconds


def _vmm_import(
    name: str, message: dict, fds: List[int], device: torch.device
) -> VmmEngramHostTable:
    """Import the served fds and map them for this process's device (read) and the host (read)."""
    from cuda.bindings import driver as cu

    if message.get("protocol") != _VMM_PROTOCOL:
        raise RuntimeError(
            f"Engram VMM table server speaks protocol {message.get('protocol')}, expected {_VMM_PROTOCOL}"
        )
    start = time.monotonic()
    manifest = message["manifest"]
    slabs = [_VmmSlab(*s) for s in message["slabs"]]
    posix_fd = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    handles = []
    try:
        for fd in fds:
            handles.append(
                _cu_call(
                    cu.cuMemImportFromShareableHandle(fd, posix_fd),
                    "cuMemImportFromShareableHandle",
                )
            )
        va = _vmm_map(slabs, handles, manifest["total_bytes"], device, host_write=False)
    except BaseException:
        for h in handles:
            cu.cuMemRelease(h)
        raise
    table = VmmEngramHostTable(
        name, manifest, slabs, message["granularity"], handles, fds, va, device, created=False
    )
    table.register_seconds = time.monotonic() - start
    return table


def open_vmm_engram_table(
    checkpoint_dir: str,
    *,
    local_rank: int,
    barrier: Optional[Callable[[], None]] = None,
    device: Optional[torch.device] = None,
    wait_s: float = 1800.0,
    threads: Optional[int] = None,
    headroom_bytes: int = 64 << 30,
    name: Optional[str] = None,
    shm_fallback_path: Optional[str] = None,
    shm_fill_kwargs: Optional[dict] = None,
) -> "VmmEngramHostTable | SharedEngramHostTable":
    """The vmm backend's weight-loading entry point; the same collective contract as ``open_shared_engram_table``.

    Local rank 0, under an exclusive lock per node, imports the table from a
    process on this node that already serves it (another engine, for one), or
    else creates and fills it; either way it serves the table from then on.
    After ``barrier()`` every other local rank imports it from a server and
    stands by to serve it too. Every rank maps the whole table.

    With ``shm_fallback_path``, a create that fails (``VmmCreateError``: host
    memory still short after the retries, or no usable NUMA node) makes local
    rank 0 warn and fill the ``shm`` table at that path instead, and every rank
    then returns that ``SharedEngramHostTable``. If rank 0 fails outright it still
    joins the barrier before raising, so the other ranks fail instead of hanging.

    Args:
        checkpoint_dir: the HF checkpoint directory.
        local_rank: this process's rank on the node.
        barrier: a node-local (or global) barrier. Without it, the other ranks wait
            up to ``wait_s`` for a server.
        device: this rank's device (current if None).
        wait_s: how long to wait for the per-node lock, and (without a barrier) for a server.
        threads: fill threads; ``min(64, cpu_count)`` if None.
        headroom_bytes: host memory that must remain available after a new table is allocated.
        name: the abstract socket name; ``default_vmm_socket_name(layout)`` if None.
        shm_fallback_path: the ``shm`` table file to fall back to; None: no fallback.
        shm_fill_kwargs: keywords for ``fill_shared_engram_table`` on a fallback.
    """
    device = _cuda_device(device)
    layout = _layout(find_engram_table_sources(checkpoint_dir))
    name = name or default_vmm_socket_name(layout)
    threads = threads or min(64, os.cpu_count() or 8)
    table: Optional[VmmEngramHostTable] = None
    shm_report: Optional[FillReport] = None
    failure: Optional[Exception] = None
    start = time.monotonic()
    if local_rank == 0:
        try:
            lock_fd = _open_lock(default_table_path() + ".vmm")
            try:
                _flock_within(
                    lock_fd,
                    fcntl.LOCK_EX,
                    wait_s,
                    f"the Engram VMM table lock of {name} is held elsewhere",
                )
                got = _vmm_fetch(name, time.monotonic() + 3.0)
                if got is not None and _identity(got[0]["manifest"]) == _identity(layout):
                    table = _vmm_import(name, got[0], got[1], device)
                    reused = True
                else:
                    if got is not None:
                        for fd in got[1]:
                            os.close(fd)
                        raise RuntimeError(f"{name} serves a table of different checkpoint bytes")
                    try:
                        table, _ = _vmm_create(name, layout, device, threads, headroom_bytes)
                    except VmmCreateError as error:
                        if shm_fallback_path is None:
                            raise
                        message = (
                            f"Engram VMM table create failed ({error}); falling back to the shm backend at "
                            f"{shm_fallback_path}. Set {BACKEND_ENV}=vmm to fail instead."
                        )
                        _logger.warning(message)
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                    reused = False
                if table is not None:
                    table.start_serving()
            finally:
                os.close(lock_fd)
            if table is None:
                shm_report = fill_shared_engram_table(
                    checkpoint_dir, shm_fallback_path, threads=threads, **(shm_fill_kwargs or {})
                )
            else:
                table.fill_report = FillReport(
                    name,
                    reused,
                    table.nbytes,
                    time.monotonic() - start,
                    threads,
                    "vmm",
                    "slab-interleave",
                    0,
                    table.numa_bytes(),
                )
                _logger.info("Engram host table %s", table.describe())
        except Exception as error:  # joined with the barrier below, then re-raised
            failure = error
    if barrier is not None:
        barrier()
    if failure is not None:
        raise failure
    if local_rank == 0 and table is None:
        shm = attach_shared_engram_table(
            shm_fallback_path, checkpoint_dir=checkpoint_dir, device=device, wait_s=0.0
        )
        shm.fill_report = shm_report
        return shm
    if table is None:
        got = None
        if barrier is not None and shm_fallback_path is not None:
            # Rank 0 serves before the barrier whenever it holds a VMM table, so one
            # refused connect plus a complete shm table for these bytes means it fell back.
            got = _vmm_fetch(name, time.monotonic())
            if got is None and _check(shm_fallback_path, layout)[0] is not None:
                _logger.warning(
                    "Engram VMM table %s is not served; attaching the shm table %s",
                    name,
                    shm_fallback_path,
                )
                return attach_shared_engram_table(
                    shm_fallback_path, checkpoint_dir=checkpoint_dir, device=device, wait_s=0.0
                )
        if got is None:
            got = _vmm_fetch(name, time.monotonic() + (30.0 if barrier is not None else wait_s))
        if got is None:
            raise RuntimeError(f"no process on this node serves the Engram VMM table {name}")
        if _identity(got[0]["manifest"]) != _identity(layout):
            for fd in got[1]:
                os.close(fd)
            raise RuntimeError(f"{name} serves a table of different checkpoint bytes")
        table = _vmm_import(name, got[0], got[1], device)
        table.start_serving(wait_s=0.0)
    return table


def remove_shared_engram_table(path: Optional[str] = None, *, force: bool = False) -> bool:
    """Unlink the table file if no process is attached (or regardless, with ``force``).

    Processes still attached keep a valid mapping after a forced unlink; the
    memory is freed when the last of them detaches. Returns True if the file was unlinked.
    """
    path = path or default_table_path()
    lock_fd = _open_lock(path)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not force:
                return False
        try:
            os.unlink(path)
        except FileNotFoundError:
            return False
        try:
            os.unlink(_owner_path(path))
        except FileNotFoundError:
            pass
        return True
    finally:
        os.close(lock_fd)


def describe_shared_engram_table(path: Optional[str] = None) -> dict:
    """State of the table file: header state, sizes, resident bytes, fill record, attached lock holders."""
    path = path or default_table_path()
    try:
        state, manifest, size = _read_header(path)
        st = os.stat(path)
    except FileNotFoundError:
        return {"path": path, "state": "absent"}
    out = {
        "path": path,
        "state": {
            None: "invalid",
            _STATE_FILLING: "filling-or-abandoned",
            _STATE_READY: "ready",
        }.get(state, "unknown"),
        "file_bytes": size,
        "resident_bytes": st.st_blocks * 512,
        "attached": _lock_holders(path),
        "engine_owned": _owned_inode(path) == st.st_ino,
    }
    if manifest is not None:
        out["tensors"] = [
            {k: t[k] for k in ("key", "shape", "dtype", "offset", "nbytes")}
            for t in manifest["tensors"]
        ]
        out["fill"] = manifest.get("fill")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``fill``, ``status``, ``verify``, ``remove``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("fill", help="fill (or reuse) the shared table")
    fill.add_argument("--checkpoint", required=True)
    fill.add_argument("--threads", type=int)
    fill.add_argument("--method", default="buffered", choices=("buffered", "direct"))
    fill.add_argument("--numa", default="interleave")
    fill.add_argument("--verify", default="sample", choices=("none", "sample", "full"))
    for name in ("status", "verify", "remove"):
        sub.add_parser(name)
    sub.choices["verify"].add_argument("--sample", action="store_true")
    sub.choices["remove"].add_argument("--force", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--path")
    args = parser.parse_args(argv)
    if args.command == "fill":
        report = fill_shared_engram_table(
            args.checkpoint,
            args.path,
            threads=args.threads,
            method=args.method,
            numa=args.numa,
            verify=args.verify,
        )
        print(json.dumps(dict(asdict(report), gbps=report.gbps)))
    elif args.command == "status":
        print(json.dumps(describe_shared_engram_table(args.path), indent=1))
    elif args.command == "verify":
        start = time.monotonic()
        checked, bad = verify_shared_engram_table(args.path, full=not args.sample)
        print(json.dumps({"checked": checked, "bad": bad, "seconds": time.monotonic() - start}))
        return 1 if bad else 0
    else:
        print(json.dumps({"removed": remove_shared_engram_table(args.path, force=args.force)}))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
