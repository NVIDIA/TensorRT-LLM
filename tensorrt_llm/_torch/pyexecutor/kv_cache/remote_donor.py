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
"""Donate host memory to the remote cache levels of KV cache manager v2.

A donor publishes host memory in a rendezvous directory on a shared file system; a remote cache level
(``KvCacheConfig.remote_cache_config``; C++ ``storage/remoteMem.h``, which also describes the directory layout) claims
whole chunks of it with O_EXCL claim files and moves cold pages itself, so the donor does no work after publishing.

``RemoteMemDonor`` declares the donation (size, chunks, the files, the failure marker, start and close) and does not
pick how it is reached: its ``DonorTransport``, chosen apart (``make_donor``, ``RemoteCacheConfig.transport.kind``),
allocates and registers the memory:

* ``AgentDonorTransport``: anonymous memory registered with the donor's own transfer agent through the
  ``BaseTransferAgent`` interface (a NIXL agent); ``donor.<name>.desc`` holds the agent's ``AgentDesc``.
* ``FabricDonorTransport``: Grace DRAM allocated with cuMemCreate (``CU_MEM_LOCATION_TYPE_HOST_NUMA``), one
  allocation per chunk because an imported fabric allocation maps only whole; ``donor.<name>.fabric`` holds one
  64-byte handle per chunk. Chunks fill the GPU's host NUMA node first, then the other nodes; a donation that only
  partly fits is published with ``short_bytes``. Placement, and the optional reclaim-and-compact passes, are
  ``host_memory``'s.

A donor that cannot start, or whose owner fails, leaves ``donor.<name>.failed``, which clients ignore and capacity
checks count. In a serving process ``start_donation`` starts the donation (``remote_cache_config.donor``)
before the KV cache estimation, or, where no estimation runs (``TRTLLM_SKIP_KV_CACHE_ESTIMATION``, helix CP on KV
cache manager v2, encoder-decoder models), when the target ``KVCacheManagerV2`` is built; that manager owns it.

Trust model: the directory is private to the job's user (``ensure_rendezvous_directory``) and every file in it is
0600, since donor files hold NIXL memory registration keys and fabric handles; use one directory per job.
"""

import contextlib
import ctypes
import mmap
import os
import re
import socket
import stat
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

from tensorrt_llm.llmapi.llm_args import REMOTE_CACHE_CHUNK_BYTES, REMOTE_CACHE_DONOR_RESERVE_BYTES
from tensorrt_llm.logger import logger

from .host_memory import (
    MADV_HUGEPAGE,
    bind_to_numa_node,
    host_numa_node_of,
    host_numa_nodes,
    node_compaction_lock,
    place_chunks,
    populate,
)

if TYPE_CHECKING:
    from cuda.bindings.driver import CUmemAllocationProp, CUmemGenericAllocationHandle

    from tensorrt_llm.llmapi.llm_args import RemoteCacheConfig
    from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
        BaseTransferAgent,
        NixlTransferAgent,
    )


def _nvml_gpu_index(device: int) -> int:
    """NVML's index (nvidia-smi's numbering) of CUDA device `device`, found by its PCI bus id."""
    import pynvml
    from cuda.bindings import driver as drv

    _cuda_check(drv.cuInit(0))
    bus_id = _cuda_check(drv.cuDeviceGetPCIBusId(32, _cuda_check(drv.cuDeviceGet(device))))
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode()
    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetIndex(
            pynvml.nvmlDeviceGetHandleByPciBusId(bus_id.rstrip("\x00"))
        )
    finally:
        pynvml.nvmlShutdown()


def _physical_gpu_index(device: Optional[int]) -> int:
    """The physical index of CUDA device `device` (default: the current one): a plain index in CUDA_VISIBLE_DEVICES
    directly, any other entry (a UUID or MIG id) through the device's PCI bus id."""
    if device is None:
        import torch

        device = torch.cuda.current_device()
    visible = [
        v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()
    ]
    if not visible:
        return device
    if device >= len(visible):
        raise ValueError(
            f"cannot map CUDA device {device} to a GPU index through "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    if visible[device].isdigit():
        return int(visible[device])
    return _nvml_gpu_index(device)


def resolve_ucx_devices(devices: Optional[str], device: Optional[int] = None) -> Optional[str]:
    """Replace ``{GPU}`` in `devices` with this rank's physical GPU index (CUDA device `device`, default the current
    one, through CUDA_VISIBLE_DEVICES), so that "rdma_vf_rail{GPU}:1" gives each rank the rail next to its GPU."""
    if not devices or "{" not in devices:
        return devices
    resolved = devices.replace("{GPU}", str(_physical_gpu_index(device)))
    if "{" in resolved or "}" in resolved:
        raise ValueError(f"UCX devices {devices!r}: unknown token (known: {{GPU}})")
    return resolved


def nixl_ucx_backend_params(devices: Optional[str]) -> dict:
    """The NIXL UCX backend parameters that restrict an agent to `devices` (UCX_NET_DEVICES form, e.g.
    "mlx5_0:1,mlx5_1:1", ``{GPU}`` already resolved); empty for None.

    NIXL's UCX backend reads only ``device_list``: device names split on ", ", each given port 1 (``ucx_devices``, the
    key its plugin lists, is never read). A device on another port cannot be expressed and raises."""
    if not devices:
        return {}
    names = []
    for entry in devices.split(","):
        name, _, port = entry.strip().partition(":")
        if not name or port not in ("", "1"):
            raise ValueError(
                f"UCX devices {devices!r}: {entry.strip()!r} is not '<device>' or '<device>:1' (NIXL's UCX "
                "backend uses port 1 of each device)"
            )
        names.append(name)
    return {"device_list": ", ".join(names)}


def _log(message: str) -> None:
    logger.info(f"KVCM2 remote donor: {message}")


_ENV_REFERENCE = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def resolve_rendezvous_directory(configured: str) -> str:
    """``remote_cache_config.directory`` with ``~`` and environment variables expanded and trailing separators
    stripped (a path ending in one follows a symlink). A variable left unset raises: the path would otherwise be one
    literal directory shared by every run without it, whose leftover donor files would count as live donors."""
    directory = os.path.expandvars(os.path.expanduser(configured))
    unset = sorted({m.group(0) for m in _ENV_REFERENCE.finditer(directory)})
    if unset:
        raise ValueError(
            f"remote_cache_config.directory {configured!r} names unset environment variable(s) {', '.join(unset)}"
        )
    return directory.rstrip(os.sep) or os.sep


def ensure_rendezvous_directory(directory: str) -> None:
    """Create `directory` and its missing parents as 0700, or accept an existing one only if it is a directory (not a
    symlink to one) that this user owns and nobody else can write. Clients pin the directory they checked (C++
    ``RemoteMemClient``)."""
    directory = directory.rstrip(os.sep) or os.sep
    missing = []
    parent = directory
    while parent and not os.path.lexists(parent):
        missing.append(parent)
        parent = os.path.dirname(parent)
    for path in reversed(missing):
        with contextlib.suppress(FileExistsError):  # another rank made it first
            os.mkdir(path, 0o700)
    st = os.lstat(directory)
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(
            f"KVCM2 remote rendezvous directory {directory} must be a directory, not a symlink or another file"
        )
    if st.st_uid != os.geteuid() or st.st_mode & 0o022:
        raise PermissionError(
            f"KVCM2 remote rendezvous directory {directory} must be owned by uid {os.geteuid()} and not writable by "
            f"group or others (owner uid {st.st_uid}, mode {st.st_mode & 0o777:o})"
        )


def _write_atomically(path: str, data: bytes) -> None:
    # A fresh name, created exclusively and never through a symlink: a planted file or link is not written through.
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class _DonorFiles:
    """The rendezvous files a donor published, shared with its finalizer. Once withdrawn, nothing more is published:
    at exit the finalizer may run while a background start is still publishing (daemon threads keep running)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paths: list[str] = []
        self.withdrawn = False

    def withdraw(self, failed: Optional[tuple[str, bytes]] = None) -> None:
        """Unlink the published files and refuse later ones; with `failed` (path, text), leave that marker too."""
        with self.lock:
            self.withdrawn = True
            for path in reversed(self.paths):
                with contextlib.suppress(OSError):
                    os.unlink(path)
            self.paths.clear()
            if failed is not None:
                with contextlib.suppress(OSError):
                    _write_atomically(*failed)


def _withdraw_unowned(
    files: _DonorFiles, failed_path: str, marker: bytes, transport: "DonorTransport"
) -> None:
    """Finalizer of a donor nobody closed: withdraw its files, leave the failure marker and free its memory.

    It holds the transport, not the donor: e.g. a fabric donation's pinned chunks and retained primary context
    otherwise stay until the process exits, though the donor is gone (a failed startup in a process that lives on).
    """
    files.withdraw((failed_path, marker))
    transport.release_unowned()


# Linux values (asm-generic, so x86_64 and aarch64 alike); Python's mmap module lacks some of them before 3.13.
_MAP_NORESERVE = 0x4000


def _cuda_check(result: object) -> Any:
    """Unwrap a cuda.bindings driver call's (CUresult, value...) tuple, raising on an error."""
    from cuda.bindings import driver as drv

    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        name = drv.cuGetErrorName(err)[1]
        raise RuntimeError(name.decode() if isinstance(name, bytes) else str(name))
    if isinstance(result, tuple):
        return result[1] if len(result) == 2 else result[1:]
    return None


_GIB = 1 << 30


def _meta_fields(path: str) -> dict:
    fields = {"segments": []}
    with open(path) as f:
        for line in f:
            key, _, value = line.strip().partition(" ")
            if key == "segment":
                fields["segments"].append([int(x) for x in value.split()])
            elif key:
                fields[key] = value
    return fields


def scan_donors(
    directory: str, malformed: Optional[list[str]] = None
) -> tuple[list[dict], list[dict]]:
    """The donors of a rendezvous directory that have reported: (published, failed). A published donor offers the
    whole chunks of its segments; a failed one left donor.<name>.failed (RemoteMemDonor.publish_failure). Files that
    name no transport count as neither; their names are appended to `malformed`."""
    published, failed = [], []
    try:
        files = sorted(os.listdir(directory))
    except FileNotFoundError:
        return published, failed
    for file in files:
        if not file.startswith("donor."):
            continue
        name = file[len("donor.") :].rsplit(".", 1)[0]
        try:
            if file.endswith(".ready"):
                fields = _meta_fields(os.path.join(directory, f"donor.{name}.meta"))
                if "transport" not in fields:
                    # Malformed: every donor names its transport, as clients require.
                    if malformed is not None:
                        malformed.append(f"donor.{name}.meta")
                    continue
                chunk = int(fields.get("chunk_bytes", 0))
                offered = sum(b - b % chunk for _, b in fields["segments"]) if chunk else 0
                published.append(
                    {
                        "name": name,
                        "host": fields.get("host", "?"),
                        "transport": fields["transport"],
                        "offered_bytes": offered,
                        "short_bytes": int(fields.get("short_bytes", 0)),
                        "numa_chunks": fields.get("numa_chunks", ""),
                    }
                )
            elif file.endswith(".failed"):
                fields = _meta_fields(os.path.join(directory, file))
                if "transport" not in fields:
                    if malformed is not None:
                        malformed.append(file)
                    continue
                failed.append(
                    {
                        "name": name,
                        "host": fields.get("host", "?"),
                        "transport": fields["transport"],
                        "planned_bytes": int(fields.get("planned_bytes", 0)),
                        "error": fields.get("error", "?"),
                    }
                )
        except (OSError, ValueError):
            continue  # a file being replaced; the next scan sees it
    return published, failed


def check_remote_capacity(
    directory: str,
    num_donors: int,
    need_bytes: int,
    *,
    timeout_s: float,
    transport: Optional[str] = None,
    poll_s: float = 2.0,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Wait until `num_donors` donors have published or failed (at most `timeout_s`), or until they offer
    `need_bytes`, then raise if the offer falls short, naming the failed and short donors. A job whose donors cannot
    cover its clients thus stops when the last donor reports instead of after the clients' claim wait.

    With `transport`, only donors of that transport count, as the clients skip the others."""
    log = log or _log
    need = need_bytes
    expected = num_donors
    deadline = time.monotonic() + timeout_s
    warned: set[str] = set()
    while True:
        malformed: list[str] = []
        published, failed = scan_donors(directory, malformed)
        for file in malformed:
            if file not in warned:
                warned.add(file)
                logger.warning(
                    f"KVCM2 remote donor file {file} in {directory} is malformed; it is not counted"
                )
        if transport is not None:
            published = [p for p in published if p["transport"] == transport]
            failed = [f for f in failed if f["transport"] == transport]
        offered = sum(p["offered_bytes"] for p in published)
        reported = len(published) + len(failed)
        if offered >= need or reported >= expected or time.monotonic() >= deadline:
            break
        time.sleep(poll_s)
    summary = (
        f"{len(published)} of {expected} planned donors published {offered / _GIB:.0f} GiB, {len(failed)} failed, "
        f"{max(0, expected - reported)} not reported; the plan's clients claim {need / _GIB:.0f} GiB"
        + (f"; malformed (not counted): {', '.join(malformed)}" if malformed else "")
    )
    if offered < need:
        notes = [f"{f['name']} ({f['host']}) failed: {f['error']}" for f in failed]
        notes += [
            f"{p['name']} ({p['host']}) is {p['short_bytes'] / _GIB:.0f} GiB short"
            + (f" (chunks per NUMA node {p['numa_chunks']})" if p["numa_chunks"] else "")
            for p in published
            if p["short_bytes"]
        ]
        raise RuntimeError(
            f"KVCM2 remote level capacity check failed in {directory}: {summary}"
            + (f". {'; '.join(notes)}" if notes else "")
            + ". Raise remote_cache_config.donor.size or the number of donating ranks (level.expected.num_donors), "
            "lower level.size (level.expected.claim_size), or for the fabric transport lower "
            "transport.donor_reserve_size."
        )
    log(f"KVCM2 remote level capacity check passed in {directory}: {summary}")
    return {"published": published, "failed": failed, "offered_bytes": offered, "need_bytes": need}


class DonorTransport:
    """How a donation is made reachable: allocates the memory its transfers need, registers it and hands the donor
    what clients read (``RemoteMemDonor._publish``); frees it again. Chosen apart from the donation itself
    (``make_donor``)."""

    # The transport's name, as written to the donor's files.
    kind: str = ""

    @classmethod
    def options_from(cls, remote_config: "RemoteCacheConfig") -> dict:
        """The constructor options of this transport that ``remote_config`` sets."""
        return {}

    def start(self, donor: "RemoteMemDonor") -> None:
        """Allocate and register `donor`'s memory, then publish it (``donor._publish``)."""
        raise NotImplementedError

    def release_unowned(self) -> None:
        """Free what ``start`` made, without the donor (also the finalizer of a donor nobody closed); teardown keeps
        going past errors."""


@contextlib.contextmanager
def _donor_thread_cuda(device: Optional[int]) -> Iterator[None]:
    """CUDA state of the thread that starts or releases a donation, for the duration of that step.

    Relaxed stream-capture mode: the start may run in the background while the serving thread captures a CUDA graph
    in global mode, and a capture-unsafe CUDA call here (e.g. host-memory registration) would invalidate that capture.
    `device`'s primary context current: CUDA calls the transport makes land on the donor's GPU, not on device 0.
    """
    from cuda.bindings import driver as drv

    _cuda_check(drv.cuInit(0))
    capture_mode = _cuda_check(
        drv.cuThreadExchangeStreamCaptureMode(
            drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        )
    )
    cu_device = context = None
    try:
        if device is not None:
            cu_device = _cuda_check(drv.cuDeviceGet(device))
            context = _cuda_check(drv.cuDevicePrimaryCtxRetain(cu_device))
            _cuda_check(drv.cuCtxPushCurrent(context))
        yield
    finally:
        steps = [lambda: drv.cuThreadExchangeStreamCaptureMode(capture_mode)]
        if context is not None:
            steps = [drv.cuCtxPopCurrent, lambda: drv.cuDevicePrimaryCtxRelease(cu_device), *steps]
        for step in steps:
            try:
                _cuda_check(step())
            except RuntimeError as error:
                _log(f"leaving the donor thread's CUDA state failed: {error}")


@contextlib.contextmanager
def _donor_release_cuda(device: Optional[int], name: str) -> Iterator[None]:
    """``_donor_thread_cuda`` for a release, which keeps going without it: e.g. a failed start's cleanup on the donor
    thread, which may run while the serving thread captures a CUDA graph in global mode, or a finalizer at exit."""
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(_donor_thread_cuda(device))
        except (ImportError, RuntimeError) as error:
            _log(f"{name}: releasing without the donor thread's CUDA state: {error}")
        yield


class RemoteMemDonor:
    """One donation: the host memory a rank gives remote cache levels, published by ``start()`` in the rendezvous
    directory and withdrawn by ``close()``.

    The donation declares the memory (size, chunks, placement, rendezvous files, failure marker, lifecycle); its
    ``transport`` (``AgentDonorTransport`` or ``FabricDonorTransport``) allocates and registers it. A donor collected
    or left at exit without ``close()`` withdraws its files and leaves the failure marker."""

    transport: Optional[DonorTransport] = None
    directory: str = ""
    nbytes: int = 0
    chunk_bytes: int = REMOTE_CACHE_CHUNK_BYTES
    numa_node: Optional[int] = None
    device: Optional[int] = None
    name: str = ""
    host: str = ""
    _files: Optional[_DonorFiles] = None  # files written, withdrawn by close() or the finalizer
    # start_in_background()'s thread, joined by close().
    _start_thread: Optional[threading.Thread] = None
    # Serializes close(): a failed background start closes on its own thread.
    _close_lock: Optional[threading.Lock] = None
    # Withdraws the files of a donor nobody closed; close() detaches it.
    _finalizer: Optional[weakref.finalize] = None

    def __init__(
        self,
        directory: str,
        nbytes: int,
        transport: DonorTransport,
        *,
        chunk_bytes: int = REMOTE_CACHE_CHUNK_BYTES,
        name: Optional[str] = None,
        numa_node: Optional[int] = None,
        device: Optional[int] = None,
    ) -> None:
        if nbytes < chunk_bytes:
            raise ValueError(f"donation size {nbytes} is below one chunk ({chunk_bytes})")
        self.transport = transport
        self.directory = directory
        # Whole chunks only: a client never sees a partial chunk.
        self.nbytes = nbytes - nbytes % chunk_bytes
        self.chunk_bytes = chunk_bytes
        self.numa_node = numa_node
        self.device = device
        host = socket.gethostname().split(".")[0]
        # Unique per donor, also within one process: clients never take claims from a withdrawn name again.
        self.name = name or f"kvcm2-donor-{host}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.host = host
        self._files = _DonorFiles()
        self._start_thread = None
        self._close_lock = threading.Lock()
        self._finalizer = None

    def _path(self, suffix: str) -> str:
        return os.path.join(self.directory, f"donor.{self.name}.{suffix}")

    def _publish(
        self, extra_files: list[tuple[str, bytes]], meta_lines: list[str], timing: str
    ) -> None:
        meta = "\n".join(
            [
                "version 1",
                f"name {self.name}",
                f"host {self.host}",
                f"pid {os.getpid()}",
                f"chunk_bytes {self.chunk_bytes}",
                f"transport {self.transport.kind}",
                *meta_lines,
                "",
            ]
        ).encode()
        with self._files.lock:
            if self._files.withdrawn:
                raise RuntimeError(f"{self.name} was withdrawn before it published")
            for suffix, data in [*extra_files, ("meta", meta), ("ready", b"")]:
                _write_atomically(self._path(suffix), data)
                self._files.paths.append(self._path(suffix))
        _log(
            f"{self.name} donates {self.nbytes / (1 << 30):.2f} GiB in {self.nbytes // self.chunk_bytes} chunk(s) "
            f"of {self.chunk_bytes / (1 << 30):.2f} GiB via {self.directory} ({timing})"
        )

    def _failure_text(self, error: BaseException) -> bytes:
        return "\n".join(
            [
                "version 1",
                f"name {self.name}",
                f"host {self.host}",
                f"pid {os.getpid()}",
                f"transport {self.transport.kind}",
                f"planned_bytes {self.nbytes}",
                f"error {error!r}".replace("\n", " "),
                "",
            ]
        ).encode()

    def publish_failure(self, error: BaseException) -> None:
        """Leave donor.<name>.failed (planned bytes and the error) so capacity checks count this donor as reported
        instead of waiting for it. Clients ignore the file: they list only .ready donors."""
        try:
            ensure_rendezvous_directory(self.directory)
            _write_atomically(self._path("failed"), self._failure_text(error))
        except OSError as write_error:
            _log(f"{self.name}: cannot write its failure marker: {write_error}")

    def start(self) -> None:
        if self._finalizer is None:
            self._finalizer = weakref.finalize(
                self,
                _withdraw_unowned,
                self._files,
                self._path("failed"),
                self._failure_text(RuntimeError("the donor's owner exited without closing it")),
                self.transport,
            )
        self.transport.start(self)

    def start_or_report(self) -> None:
        """``start()``; on failure, release what was made and leave the failure marker (``publish_failure``)."""
        try:
            self.start()
        except Exception as error:  # noqa: BLE001 - a failed donor only removes capacity
            logger.warning(f"KVCM2 remote donor: {self.name} failed to start: {error!r}")
            self.close(error)

    def start_in_background(self) -> None:
        """``start_or_report()`` on a thread, so the caller does not wait for the memory to be faulted in and
        registered; clients wait for the ``.ready`` file."""
        self._start_thread = threading.Thread(
            target=self.start_or_report, name="kvcm2-remote-donor", daemon=True
        )
        self._start_thread.start()

    def wait_started(self) -> None:
        """Wait for a background start to publish or fail."""
        thread = self._start_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def close(self, error: Optional[BaseException] = None) -> None:
        """Wait for a background start, then withdraw the files and the transport's memory; with `error`, leave the
        failure marker too. Clients still holding chunks fail their next transfer instead of reading stale bytes.

        A failed background start closes on its own thread; ``_start_thread`` stays set until that close ends, so
        ``wait_started()`` returns only once the failed donation is released and marked."""
        thread = self._start_thread
        on_start_thread = thread is threading.current_thread()
        if thread is not None and not on_start_thread:
            thread.join()
        with self._close_lock:
            if self._finalizer is not None:
                self._finalizer.detach()
            self._withdraw()
        if error is not None:
            self.publish_failure(error)
        if not on_start_thread:
            self._start_thread = None

    def _withdraw(self) -> None:
        self._files.withdraw()
        self.transport.release_unowned()


def _make_nixl_agent(name: str, backend_params: dict) -> "NixlTransferAgent":
    """A NIXL transfer agent (the C++ binding's ``NixlTransferAgent``, a ``BaseTransferAgent``) named `name`."""
    from tensorrt_llm.tensorrt_llm_transfer_agent_binding import BaseAgentConfig, NixlTransferAgent

    config = BaseAgentConfig()
    config.name = name
    config.use_prog_thread = True
    config.multi_thread = False
    config.use_listen_thread = False
    # This agent's own UCX devices; the process-wide UCX_NET_DEVICES stays as the serving process needs it.
    if backend_params:
        config.backend_params = backend_params
    return NixlTransferAgent(config)


class AgentDonorTransport(DonorTransport):
    """Anonymous memory registered with the donor's own transfer agent, through the ``BaseTransferAgent`` interface
    only (``make_agent``, default a NIXL agent); ``donor.<name>.desc`` holds the agent's ``AgentDesc``.
    ``ucx_devices`` (``{GPU}`` resolved at start) selects the agent's UCX devices."""

    kind = "nixl"
    ucx_devices: Optional[str] = None
    _map: Optional[mmap.mmap] = None
    _address: int = 0
    _nbytes: int = 0  # the registered donation
    _name: str = ""  # the donor's, for logs
    _agent = None
    _registered: bool = False
    # The donor's CUDA device, current while the agent is made or released.
    _device: Optional[int] = None

    def __init__(
        self,
        *,
        ucx_devices: Optional[str] = None,
        make_agent: Callable[[str, dict], "BaseTransferAgent"] = _make_nixl_agent,
    ) -> None:
        self.ucx_devices = ucx_devices
        self._make_agent = make_agent
        self._map = None
        self._address = 0
        self._nbytes = 0
        self._name = ""
        self._agent = None
        self._registered = False
        self._device = None

    @classmethod
    def options_from(cls, remote_config: "RemoteCacheConfig") -> dict:
        return {"ucx_devices": remote_config.transport.ucx_devices}

    def start(self, donor: "RemoteMemDonor") -> None:
        from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
            MemoryDesc,
            MemoryDescs,
            MemoryType,
        )

        ensure_rendezvous_directory(donor.directory)
        backend_params = nixl_ucx_backend_params(
            resolve_ucx_devices(self.ucx_devices, donor.device)
        )
        self._name = donor.name
        self._nbytes = donor.nbytes
        self._device = donor.device
        t0 = time.monotonic()
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_NORESERVE
        self._map = mmap.mmap(-1, donor.nbytes, flags=flags)
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        try:
            self._map.madvise(MADV_HUGEPAGE)
        except OSError:
            pass
        if donor.numa_node is not None and not bind_to_numa_node(
            self._address, donor.nbytes, donor.numa_node
        ):
            _log(f"libnuma unavailable; {donor.name} relies on first touch for NUMA placement")
        # Fault every page in now: registration does not pin memory where the NIC pages on demand, and a resident
        # donation fails here rather than mid-run and never makes a remote write wait on a page fault.
        populate(self._map, self._address, donor.nbytes)
        t1 = time.monotonic()

        # NIXL/UCX may make CUDA calls while creating the agent and registering host memory.
        with _donor_thread_cuda(donor.device):
            self._agent = self._make_agent(donor.name, backend_params)
            self._agent.register_memory(
                MemoryDescs(MemoryType.DRAM, [MemoryDesc(self._address, donor.nbytes, 0)])
            )
            self._registered = True
        t2 = time.monotonic()
        desc = self._agent.get_local_agent_desc().serialize()
        donor._publish(
            [("desc", desc)],
            [f"segment {self._address} {donor.nbytes}"],
            f"map + populate {t1 - t0:.1f} s, register {t2 - t1:.1f} s",
        )

    def release_unowned(self) -> None:
        # Clients still holding chunks fail their next transfer at the NIC. Deregistering and dropping the agent run in
        # the start's CUDA state (_donor_thread_cuda), like registering: NIXL/UCX may make CUDA calls there too.
        if self._agent is not None:
            with _donor_release_cuda(self._device, self._name):
                if self._registered:
                    from tensorrt_llm.tensorrt_llm_transfer_agent_binding import (
                        MemoryDesc,
                        MemoryDescs,
                        MemoryType,
                    )

                    try:
                        self._agent.deregister_memory(
                            MemoryDescs(
                                MemoryType.DRAM, [MemoryDesc(self._address, self._nbytes, 0)]
                            )
                        )
                    except Exception as error:  # noqa: BLE001 - teardown keeps going
                        _log(f"{self._name}: deregistering failed: {error}")
                    self._registered = False
                self._agent = None
        if self._map is not None:
            try:
                self._map.close()
            except BufferError:
                pass
            self._map = None


class FabricDonorTransport(DonorTransport):
    """Grace DRAM allocated with cuMemCreate (``CU_MEM_LOCATION_TYPE_HOST_NUMA``), one allocation per chunk because an
    imported fabric allocation maps only whole, published as one 64-byte CUDA fabric handle per chunk in
    ``donor.<name>.fabric``. Chunks fill the donor's NUMA node (default: the node of its CUDA device) first, then the
    other nodes, leaving ``reserve_bytes`` free per node; a donation that only partly fits is published with
    ``short_bytes``. With ``compact_host_memory`` a refused chunk gets reclaim-and-compact passes on its node first
    (``host_memory.place_chunks``). Clients map the chunks and copy on their own streams, so no transfer agent is
    involved."""

    kind = "fabric"
    name: str = ""  # the donor's, for logs
    chunk_bytes: int = REMOTE_CACHE_CHUNK_BYTES
    reserve_bytes: int = REMOTE_CACHE_DONOR_RESERVE_BYTES
    compact_host_memory: bool = False
    _fabric: Optional[list] = None  # the cuMemCreate handles, one per chunk
    # The CUDA device ordinal whose primary context the donation retains while it lives (None: not started).
    _device: Optional[int] = None

    def __init__(
        self,
        *,
        reserve_bytes: int = REMOTE_CACHE_DONOR_RESERVE_BYTES,
        compact_host_memory: bool = False,
    ) -> None:
        self.reserve_bytes = reserve_bytes
        self.compact_host_memory = compact_host_memory
        self._fabric = []
        self._device = None

    @classmethod
    def options_from(cls, remote_config: "RemoteCacheConfig") -> dict:
        return {
            "reserve_bytes": remote_config.transport.donor_reserve_size,
            "compact_host_memory": remote_config.transport.compact_host_memory,
        }

    def _release_fabric(self) -> None:
        from cuda.bindings import driver as drv

        for handle in self._fabric:
            try:
                _cuda_check(drv.cuMemRelease(handle))
            except Exception as error:  # noqa: BLE001 - teardown keeps going
                _log(f"{self.name}: releasing a fabric chunk failed: {error}")
        self._fabric = []

    @staticmethod
    def _fabric_prop(drv: ModuleType, node: int) -> "CUmemAllocationProp":
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        prop.location.id = node
        prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        return prop

    def _try_create(
        self, drv: ModuleType, prop: "CUmemAllocationProp"
    ) -> Optional["CUmemGenericAllocationHandle"]:
        """One chunk, or None when the node is out of memory (any other error raises)."""
        try:
            return _cuda_check(drv.cuMemCreate(self.chunk_bytes, prop, 0))
        except RuntimeError as error:
            if "OUT_OF_MEMORY" in str(error):
                return None
            raise

    def start(self, donor: "RemoteMemDonor") -> None:
        """Allocate the donation one chunk at a time, placed by place_chunks, and publish their fabric handles in chunk
        order. What fits is published with short_bytes; a donor that fits nothing raises."""
        from cuda.bindings import driver as drv

        self.name = donor.name
        self.chunk_bytes = donor.chunk_bytes
        ensure_rendezvous_directory(donor.directory)
        handles = []
        planned = donor.nbytes
        chunk_gib = self.chunk_bytes / _GIB
        reserve_gib = self.reserve_bytes / _GIB
        ordinal = 0 if donor.device is None else donor.device
        # cuMemCreate and the export may run while the serving thread captures a CUDA graph in global mode.
        with _donor_thread_cuda(ordinal):
            device = _cuda_check(drv.cuDeviceGet(ordinal))
            # Held for the donation's life; release_unowned releases it after the chunks.
            _cuda_check(drv.cuDevicePrimaryCtxRetain(device))
            self._device = ordinal
            try:
                numa = (
                    donor.numa_node
                    if donor.numa_node is not None
                    else host_numa_node_of(drv, device)
                )
                nodes = [numa] + [n for n in host_numa_nodes() if n != numa]
                props = {n: self._fabric_prop(drv, n) for n in nodes}
                gran = _cuda_check(
                    drv.cuMemGetAllocationGranularity(
                        props[numa],
                        drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                    )
                )
                if self.chunk_bytes % gran:
                    raise ValueError(
                        f"chunk size {self.chunk_bytes} is not a multiple of the fabric allocation granularity {gran}"
                    )

                def create(node: int) -> Optional["CUmemGenericAllocationHandle"]:
                    handle = self._try_create(drv, props[node])
                    if handle is not None:
                        # Released by the except below if anything fails later.
                        self._fabric.append(handle)
                    return handle

                t0 = time.monotonic()
                placed, per_node = place_chunks(
                    planned // self.chunk_bytes,
                    nodes,
                    create,
                    chunk_gib=chunk_gib,
                    reserve_gib=reserve_gib,
                    node_lock=lambda node: node_compaction_lock(donor.directory, donor.host, node),
                    compact_passes=self.compact_host_memory,
                )
                for handle in placed:
                    exported = _cuda_check(
                        drv.cuMemExportToShareableHandle(
                            handle, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0
                        )
                    )
                    handles.append(bytes(exported.data))
                    # The C++ client reads 64-byte CUmemFabricHandles back to back.
                    if len(handles[-1]) != 64:
                        raise RuntimeError(f"fabric handle of {len(handles[-1])} B, expected 64")
                t1 = time.monotonic()
                if not handles:
                    raise RuntimeError(
                        f"no NUMA node can hold a {chunk_gib:.2f} GiB fabric chunk and keep its "
                        f"{reserve_gib:.0f} GiB reserve"
                    )
            except BaseException:
                # Nothing is published: release what was made.
                self._release_fabric()
                raise
        donor.nbytes = len(handles) * self.chunk_bytes
        short = planned - donor.nbytes
        placement = ",".join(f"{n}:{c}" for n, c in per_node.items() if c)
        meta = [
            f"numa {numa}",
            f"numa_chunks {placement}",
            f"segment 0 {donor.nbytes}",
        ]
        if short:
            meta.append(f"short_bytes {short}")
            _log(
                f"{self.name}: only {donor.nbytes / _GIB:.0f} of {planned / _GIB:.0f} GiB fit as fabric memory "
                f"(reserve {reserve_gib:.0f} GiB per node)"
            )
        donor._publish(
            [("fabric", b"".join(handles))],
            meta,
            f"fabric memory, chunks per NUMA node {placement}, {len(handles)} allocation(s) created in "
            f"{t1 - t0:.1f} s",
        )

    def release_unowned(self) -> None:
        if self._device is not None:
            from cuda.bindings import driver as drv

            # In the start's CUDA state: a failed start releases on the donor thread, possibly during a capture.
            with _donor_release_cuda(self._device, self.name):
                self._release_fabric()
                try:
                    _cuda_check(
                        drv.cuDevicePrimaryCtxRelease(_cuda_check(drv.cuDeviceGet(self._device)))
                    )
                except Exception as error:  # noqa: BLE001 - teardown keeps going
                    _log(f"{self.name}: releasing the primary context failed: {error}")
            self._device = None
        self._fabric = []


# The donor transport of each RemoteCacheConfig.transport.kind.
DONOR_TRANSPORTS = {"nixl": AgentDonorTransport, "fabric": FabricDonorTransport}
TRANSPORTS = tuple(DONOR_TRANSPORTS)


def make_donor(
    directory: str,
    nbytes: int,
    transport: str = "nixl",
    *,
    chunk_bytes: int = REMOTE_CACHE_CHUNK_BYTES,
    name: Optional[str] = None,
    numa_node: Optional[int] = None,
    device: Optional[int] = None,
    **transport_options: object,
) -> RemoteMemDonor:
    """A donation of `nbytes` in `directory`, made reachable by `transport` (one of ``TRANSPORTS``) built with
    `transport_options` (``ucx_devices`` for NIXL, ``reserve_bytes`` and ``compact_host_memory`` for fabric)."""
    if transport not in DONOR_TRANSPORTS:
        raise ValueError(f"transport {transport!r} is not one of {TRANSPORTS}")
    return RemoteMemDonor(
        directory,
        nbytes,
        DONOR_TRANSPORTS[transport](**transport_options),
        chunk_bytes=chunk_bytes,
        name=name,
        numa_node=numa_node,
        device=device,
    )


def _donor_for(remote_config: "RemoteCacheConfig", device: Optional[int]) -> RemoteMemDonor:
    return make_donor(
        resolve_rendezvous_directory(remote_config.directory),
        remote_config.donor.size,
        remote_config.transport.kind,
        device=device,
        **DONOR_TRANSPORTS[remote_config.transport.kind].options_from(remote_config),
    )


def start_donation(remote_config: "RemoteCacheConfig", device: int) -> RemoteMemDonor:
    """Start donating ``remote_config.donor.size`` bytes of host memory near CUDA device `device`, in the background.
    Every start error, ``{GPU}`` resolution included, surfaces as the donor's failure marker."""
    donor = _donor_for(remote_config, device)
    directory = donor.directory
    donor.start_in_background()
    size = remote_config.donor.size
    donated = size - size % REMOTE_CACHE_CHUNK_BYTES  # as RemoteMemDonor rounds it
    if donated != size:
        logger.warning(
            f"remote_cache_config.donor.size {size} is not a whole number of "
            f"{REMOTE_CACHE_CHUNK_BYTES}-byte chunks; {donated} bytes are donated"
        )
    logger.info(
        f"KV cache manager v2 donates {donated / _GIB:.2f}GiB of host memory "
        f"to remote cache levels via {directory} ({remote_config.transport.kind})"
    )
    return donor


def publish_donation_failure(remote_config: "RemoteCacheConfig", error: BaseException) -> None:
    """Leave the failure marker of a donation this rank planned but never started, so peers stop waiting for it."""
    _donor_for(remote_config, None).publish_failure(error)
