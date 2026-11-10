# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One host copy of a read-only table per host, shared by every rank there.

With `PLE_SHARED_HOST_ENV` set, the ranks of one host map one allocation of the PLE n-gram table
instead of one copy each, across processes and servers (a disaggregated node's context and
generation servers share no communicator). The memory comes from a `Backing`: the engine's
`weights._VmmHostBacking` is a CUDA VMM host allocation exported as a POSIX fd, because on GB300
registered shmem pages gather several times slower; `MemfdBacking` serves the CPU tests.

Processes meet through `<name>.lock` (an `flock`) and `<name>.json` (the published descriptor:
owner pid and start time, fd number, size) in `PLE_SHARED_HOST_DIR_ENV` (default `/dev/shm`):

* The first process to take the lock while no live descriptor exists is the **owner**: it creates
  and fills the allocation, then publishes the descriptor and releases the lock.
* Every other process **attaches** by duplicating the owner's fd (`pidfd_getfd`, else
  `/proc/<owner>/fd/<n>`) and mapping it, or keeps a private, unpublished copy if neither works.
* A process arriving while the owner fills is **pending**: it loads its other weights and waits
  on the lock only at the table. An owner's death releases the lock; the waiter becomes owner.

The allocation lives while any process maps it, so a crash leaks nothing. The bytes and the
gather match the per-rank copy, so output is bitwise identical.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import time
from typing import Callable, Optional, Protocol

#: `node` (one copy per host); unset, `0`, `off`, `false` or `no` keep one copy per rank.
PLE_SHARED_HOST_ENV = "TRTLLM_MODELING_V2_PLE_SHARED_HOST"
#: Where the rendezvous files live; must be a directory every sharing process sees (tmpfs).
PLE_SHARED_HOST_DIR_ENV = "TRTLLM_MODELING_V2_PLE_SHARED_HOST_DIR"
_DEFAULT_DIR = "/dev/shm"
_OFF = frozenset({"", "0", "off", "false", "no", "none"})
_SCOPE = "node"

OWNER = "owner"
ATTACHED = "attached"
PENDING = "pending"

# mmap(2) / mbind(2) constants (Linux, identical on aarch64 and x86_64).
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_SHARED = 0x01
_MPOL_PREFERRED = 1
_SYS_MBIND = {"aarch64": 235, "x86_64": 237}
# pidfd_open(2) / pidfd_getfd(2): the generic syscall table, shared by aarch64 and x86_64.
_SYS_PIDFD_OPEN = 434
_SYS_PIDFD_GETFD = 438

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.syscall.restype = ctypes.c_long
_MAP_FAILED = ctypes.c_void_p(-1).value


def shared_host_scope() -> Optional[str]:
    """`node`, or None (one copy per rank) from `PLE_SHARED_HOST_ENV`."""
    value = os.environ.get(PLE_SHARED_HOST_ENV, "").strip().lower()
    if value in _OFF:
        return None
    if value != _SCOPE:
        raise ValueError(f"{PLE_SHARED_HOST_ENV}={value!r}: expected {_SCOPE!r} or off")
    return _SCOPE


def _parse_cpulist(text: str) -> set:
    cpus = set()
    for part in filter(None, text.strip().split(",")):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def numa_node_of_cpus(cpus: set, sysfs: str = "/sys/devices/system/node") -> Optional[int]:
    """The NUMA node whose CPU list holds the lowest of `cpus`, or None when sysfs says nothing."""
    if not cpus:
        return None
    first = min(cpus)
    try:
        names = os.listdir(sysfs)
    except OSError:
        return None
    for name in sorted(names):
        if not (name.startswith("node") and name[4:].isdigit()):
            continue
        try:
            with open(os.path.join(sysfs, name, "cpulist")) as fh:
                if first in _parse_cpulist(fh.read()):
                    return int(name[4:])
        except OSError:
            continue
    return None


def _proc_start_time(pid: int) -> Optional[int]:
    """`pid`'s start time in clock ticks, None if gone; with the pid, it survives pid reuse."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
    except OSError:
        return None
    # The command name (field 2) may hold spaces and parentheses: split after its last ')'.
    return int(stat[stat.rindex(")") + 2 :].split()[19])


def _dup_from(pid: int, fd: int) -> int:
    """Duplicate `pid`'s fd `fd`: `pidfd_getfd` (needs ptrace-attach rights over `pid`), else
    reopen `/proc/<pid>/fd/<fd>` (enough for a memfd, not for a CUDA export)."""
    pidfd = _libc.syscall(ctypes.c_long(_SYS_PIDFD_OPEN), ctypes.c_int(pid), ctypes.c_uint(0))
    if pidfd >= 0:
        try:
            local = _libc.syscall(
                ctypes.c_long(_SYS_PIDFD_GETFD),
                ctypes.c_int(pidfd),
                ctypes.c_int(fd),
                ctypes.c_uint(0),
            )
            if local >= 0:
                return int(local)
            err = ctypes.get_errno()
        finally:
            os.close(pidfd)
    else:
        err = ctypes.get_errno()
    try:
        return os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR | os.O_CLOEXEC)
    except OSError as error:
        raise OSError(
            error.errno,
            f"cannot duplicate fd {fd} of pid {pid}: pidfd_getfd errno {err} "
            f"({os.strerror(err)}), /proc reopen {error.strerror}",
        ) from None


class Backing(Protocol):
    """Where a segment's bytes live: `create` returns the device-read address and an fd another
    process duplicates and passes to `attach`; `release` undoes either."""

    def create(self, nbytes: int) -> tuple[int, int]: ...

    def attach(self, fd: int, nbytes: int) -> int: ...

    def release(self, address: int, nbytes: int) -> None: ...

    def describe(self) -> str: ...


#: `register(host_pointer, nbytes, read_only) -> unregister()`: `MemfdBacking`'s device side.
Register = Callable[[int, int, bool], Callable[[], None]]


def _prefer_node(address: int, nbytes: int, node: int) -> bool:
    """Best effort `mbind(MPOL_PREFERRED, node)` over a mapping."""
    number = _SYS_MBIND.get(platform.machine())
    if number is None or node < 0:
        return False
    words = node // 64 + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // 64] = 1 << (node % 64)
    rc = _libc.syscall(
        ctypes.c_long(number),
        ctypes.c_void_p(address),
        ctypes.c_ulong(nbytes),
        ctypes.c_int(_MPOL_PREFERRED),
        mask,
        ctypes.c_ulong(words * 64 + 1),
        ctypes.c_uint(0),
    )
    return rc == 0


def _mmap(fd: int, nbytes: int) -> int:
    address = _libc.mmap(None, nbytes, _PROT_READ | _PROT_WRITE, _MAP_SHARED, fd, 0)
    if address in (None, _MAP_FAILED):
        err = ctypes.get_errno()
        raise OSError(err, f"mmap of {nbytes} bytes failed: {os.strerror(err)}")
    return int(address)


class MemfdBacking:
    """Shared memory (a sealed memfd) mapped in every process and handed to `register`."""

    def __init__(self, register: Register, node: Optional[int] = None) -> None:
        self._register = register
        self.node = node
        self._unregister: dict[int, Callable[[], None]] = {}
        self._munmap = _libc.munmap

    def create(self, nbytes: int) -> tuple[int, int]:
        fd = os.memfd_create("trtllm-ple", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(fd, nbytes)
            # Nobody may resize what everyone maps (a shrink would SIGBUS every reader).
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW)
            address = _mmap(fd, nbytes)
        except BaseException:
            os.close(fd)
            raise
        if self.node is not None:
            _prefer_node(address, nbytes, self.node)
        self._map(address, nbytes)
        return address, fd

    def attach(self, fd: int, nbytes: int) -> int:
        if os.fstat(fd).st_size != nbytes:
            raise ValueError(f"the shared memfd is {os.fstat(fd).st_size} bytes, not {nbytes}")
        address = _mmap(fd, nbytes)
        self._map(address, nbytes)
        return address

    def _map(self, address: int, nbytes: int) -> None:
        try:
            self._unregister[address] = self._register(address, nbytes, False)
        except BaseException:
            self._munmap(address, nbytes)
            raise

    def release(self, address: int, nbytes: int) -> None:
        unregister = self._unregister.pop(address, None)
        if unregister is not None:
            unregister()
        self._munmap(address, nbytes)

    def describe(self) -> str:
        return "memfd"


def segment_name(key: str, scope_label: str) -> str:
    """The rendezvous file stem: per user, per table content, per scope."""
    return f"trtllm-ple-{os.getuid()}-{scope_label}-{key[:24]}"


def table_key(*parts) -> str:
    """A content key over `parts` (bytes or anything `str`-able)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, (bytes, bytearray)) else str(part).encode())
        digest.update(b"\0")
    return digest.hexdigest()


class SharedHostSegment:
    """One process's handle on the scope's shared allocation.

    `role` is `OWNER` (fills, then `publish`es), `ATTACHED` (bytes ready) or `PENDING` (`wait`
    first). `pointer` is the device address (0 while pending); `private` marks an owner that could
    not attach to a live segment and keeps its copy to itself."""

    def __init__(
        self,
        key: str,
        nbytes: int,
        scope_label: str,
        backing: Backing,
        directory: Optional[str] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        assert nbytes > 0, nbytes
        self.key = key
        self.nbytes = nbytes
        self.scope_label = scope_label
        self.backing = backing
        self._log = log
        directory = directory or os.environ.get(PLE_SHARED_HOST_DIR_ENV, _DEFAULT_DIR)
        stem = os.path.join(directory, segment_name(key, scope_label))
        self.lock_path = stem + ".lock"
        self.descriptor_path = stem + ".json"
        self.role = PENDING
        self.private = False
        self.pointer = 0
        self.owner_pid = 0
        self.wait_seconds = 0.0
        self._fd = -1
        # Bound now: at interpreter shutdown the modules may already be torn down.
        self._close = os.close
        self._lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise
            return  # PENDING: someone holds the lock (an owner filling, or a quick reader)
        self._resolve_locked()

    def _resolve_locked(self) -> None:
        """With the lock held: attach to a live published segment, or become its owner."""
        if self._try_attach():
            self._unlock()
            return
        self._create()

    def _unlock(self) -> None:
        if self._lock_fd >= 0:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1

    def _read_descriptor(self) -> Optional[dict]:
        try:
            with open(self.descriptor_path) as fh:
                desc = json.load(fh)
        except (OSError, ValueError):
            return None
        if desc.get("key") != self.key or desc.get("nbytes") != self.nbytes:
            return None
        if _proc_start_time(int(desc["pid"])) != desc.get("start"):
            return None  # the owner is gone (or its pid was reused)
        return desc

    def _try_attach(self) -> bool:
        desc = self._read_descriptor()
        if desc is None:
            return False
        try:
            fd = _dup_from(int(desc["pid"]), int(desc["fd"]))
        except OSError as error:
            self._log(f"ple shared host: {error}; this process keeps a private copy")
            self.private = True
            return False
        try:
            self.pointer = self.backing.attach(fd, self.nbytes)
        except Exception as error:
            self._log(
                f"ple shared host: attaching pid {desc['pid']}'s segment failed ({error}); "
                f"this process keeps a private copy"
            )
            self.private = True
            return False
        finally:
            # The mapping holds the allocation; the duplicate is no longer needed.
            os.close(fd)
        self.role = ATTACHED
        self.owner_pid = int(desc["pid"])
        return True

    def _create(self) -> None:
        """Become the owner of a new, unfilled allocation; the lock stays held until `publish`."""
        self.pointer, self._fd = self.backing.create(self.nbytes)
        self.role = OWNER
        self.owner_pid = os.getpid()

    def wait(self, poll_seconds: float = 0.2, report_seconds: float = 30.0) -> None:
        """Pending -> attached (the owner published) or owner (it never will: it died, or the
        lock holder was only reading). Returns at once if not pending."""
        if self.role != PENDING:
            return
        t0 = time.monotonic()
        next_report = t0 + report_seconds
        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
            now = time.monotonic()
            if now >= next_report:
                self._log(
                    f"ple shared host: waiting {now - t0:.0f} s for the {self.scope_label} "
                    f"segment's owner to fill it ({self.lock_path})"
                )
                next_report = now + report_seconds
            time.sleep(poll_seconds)
        self.wait_seconds = time.monotonic() - t0
        self._resolve_locked()

    def publish(self) -> None:
        """Owner, after the bytes are complete (device writes synchronized): make the segment
        attachable and release the waiters. A private owner only releases the lock."""
        assert self.role == OWNER and self._fd >= 0, self.role
        if not self.private:
            desc = dict(
                key=self.key,
                nbytes=self.nbytes,
                pid=os.getpid(),
                start=_proc_start_time(os.getpid()),
                fd=self._fd,
                scope=self.scope_label,
                backing=self.backing.describe(),
            )
            tmp = f"{self.descriptor_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(desc, fh)
            os.replace(tmp, self.descriptor_path)
        self._unlock()

    def describe(self) -> str:
        extra = ""
        if self.role == OWNER:
            extra = ", private copy" if self.private else ""
        elif self.role == ATTACHED:
            extra = f" to pid {self.owner_pid}'s segment" + (
                f", waited {self.wait_seconds:.1f} s" if self.wait_seconds else ""
            )
        return f"shared per {self.scope_label} ({self.backing.describe()}, {self.role}{extra})"

    def _withdraw(self) -> None:
        """Owner: unlink the descriptor if it still names this pid and fd, before the fd closes."""
        try:
            with open(self.descriptor_path) as fh:
                desc = json.load(fh)
            if desc.get("pid") == os.getpid() and desc.get("fd") == self._fd:
                os.unlink(self.descriptor_path)
        except (OSError, ValueError, TypeError):
            pass

    def close(self) -> None:
        if self.role == OWNER and self._fd >= 0 and not self.private:
            self._withdraw()
        pointer, self.pointer = self.pointer, 0
        if pointer:
            self.backing.release(pointer, self.nbytes)
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            self._close(fd)
        if self._lock_fd >= 0:
            fd, self._lock_fd = self._lock_fd, -1
            self._close(fd)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
