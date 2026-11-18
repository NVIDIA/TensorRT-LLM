# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin µGPU dies ("locality domains" in CUDA 13.4): which die runs an SM, which die
homes a byte of device memory, and die-localized allocations.

Rubin is two dies. Each has its own L2 and HBM stacks, so a load from an SM to memory
homed on the other die crosses the die-to-die link: cold misses take ~455 ns instead of
~290 ns, and a full-chip read of default memory reaches ~12.6-13.6 TB/s against ~16.5
when every SM reads only memory on its own die.

- **Memory.** ``cuMemCreate`` with ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`` homes
  physical memory on one die (2 MiB granularity); ``localized_empty`` maps such chunks
  (of either die, in any pattern) into one VA range and returns it as a tensor.
  Default allocations interleave the dies every 4 KiB: the home die of the 4 KiB
  page ``k`` (0-511) of a 2 MiB physical page is ``parity(k & 0x1DB) ^ c``, with one
  bit ``c`` per 2 MiB page (a hash of higher physical address bits). ``page_domains``
  measures ``c`` for each 2 MiB page of a tensor; the map holds for the allocation's
  lifetime.
- **SMs.** A kernel reads ``%smid`` and looks up ``sm_domains(device)``. Green contexts
  split by locality domain give the official SM sets (the driver attributes 100 SMs
  to each die of a 212-SM part); a latency probe (cold loads to each die's localized
  memory) places every SM and must agree with those sets, else the map is rejected.
  Clusters never span dies.

Single-die devices report one domain: every helper then degrades to "everything is
domain 0" and callers keep their die-oblivious schedules.
"""

import functools
import statistics
import weakref

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from .host import to_cute
from .primitives import asm, launch

PAGE, CHUNK = 4096, 2 << 20
# Cold-load medians are ~700 (near die) vs ~1100 (far die) SM cycles.
MIN_GAP_CYCLES = 150
# Home die of the 4 KiB page k of a 2 MiB physical page, up to the per-chunk bit c.
PAGE_MASK = 0x1DB
PAGE_PARITY = tuple(bin(k & PAGE_MASK).count("1") & 1 for k in range(CHUNK // PAGE))
_attr = cuda.CUdevice_attribute


def _checked(result, operation):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


@functools.cache
def domain_count(device: int) -> int:
    """Locality domains (dies) of ``device``; 1 when the driver reports none."""
    status, count = cuda.cuDeviceGetAttribute(
        _attr.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device
    )
    return count if status == cuda.CUresult.CUDA_SUCCESS and count > 1 else 1


class LocalizedMemory:
    """One VA range whose 2 MiB chunks are physical memory homed on chosen dies.
    Exposes ``__cuda_array_interface__`` (uint8) so ``torch.as_tensor`` views it; the
    tensor keeps this object (and so the mapping) alive."""

    def __init__(self, nbytes, domains, device):
        self.device = device
        chunks = -(-nbytes // CHUNK)
        if isinstance(domains, int):
            domains = [domains]
        self.domains = tuple(domains[i % len(domains)] for i in range(chunks))
        self.nbytes = chunks * CHUNK
        self.handles: list = []
        self.base = int(
            _checked(cuda.cuMemAddressReserve(self.nbytes, CHUNK, 0, 0), "reserve")
        )
        self._finalizer = weakref.finalize(
            self, LocalizedMemory._release, self.base, self.nbytes, self.handles
        )
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        start = 0
        for i in range(1, chunks + 1):  # one handle per run of chunks on one die
            if i < chunks and self.domains[i] == self.domains[start]:
                continue
            size = (i - start) * CHUNK
            prop = _prop(device, self.domains[start])
            handle = _checked(cuda.cuMemCreate(size, prop, 0), "create")
            self.handles.append(handle)
            _checked(
                cuda.cuMemMap(self.base + start * CHUNK, size, 0, handle, 0), "map"
            )
            start = i
        _checked(cuda.cuMemSetAccess(self.base, self.nbytes, [access], 1), "access")

    @staticmethod
    def _release(base, nbytes, handles):
        if handles:
            cuda.cuMemUnmap(base, nbytes)
        for handle in handles:
            cuda.cuMemRelease(handle)
        cuda.cuMemAddressFree(base, nbytes)

    @property
    def __cuda_array_interface__(self):
        return dict(
            shape=(self.nbytes,), typestr="|u1", data=(self.base, False), version=3
        )

    def domain_of(self, address: int) -> int:
        return self.domains[(address - self.base) // CHUNK]


def _prop(device, domain):
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    location = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    prop.location.type = location
    prop.location.localized.deviceId = device
    prop.location.localized.localityDomainId = domain
    return prop


def localized_empty(nbytes, domains, device):
    """An uninitialized uint8 tensor of ``nbytes`` (rounded up to 2 MiB) on ``device``
    whose 2 MiB chunk ``i`` is homed on die ``domains[i % len(domains)]`` (an int: every
    chunk). The tensor owns the mapping."""
    device = torch.device("cuda", torch.device(device).index or 0)
    with torch.accelerator.device_index(device.index):
        torch.cuda.current_stream(device)  # the primary context is current
        memory = LocalizedMemory(nbytes, domains, device.index)
        return torch.as_tensor(memory, device=device)[:nbytes]


def localized_like(tensor, domains):
    """A contiguous copy of ``tensor`` in localized memory (2 MiB chunk ``i`` on die
    ``domains[i % len(domains)]``): static data such as weights placed per die."""
    nbytes = tensor.numel() * tensor.element_size()
    raw = localized_empty(nbytes, domains, tensor.device)
    out = raw.view(tensor.dtype).view(tensor.shape)
    out.copy_(tensor)
    return out


@functools.cache
def domain_stream(device: int, domain: int, coscheduled: int = 2):
    """A torch stream of a green context holding the SMs the driver attributes to
    ``domain``, in groups of ``coscheduled`` SMs (8 for cluster launches); kept for
    the process. Benchmarks only: a live green context slows the process's later
    kernels (~0.25 us per MoE layer in a decode chain)."""
    stream, green, handle = _green_stream(device, domain, coscheduled)
    stream._k3_green = (green, handle)  # keep the context alive with the stream
    return stream


def _green_stream(device: int, domain: int, coscheduled: int = 2):
    """(torch stream, green context, stream handle) for ``domain``; the caller
    destroys them (``_destroy_green``)."""
    dev = _checked(cuda.cuDeviceGet(device), "device")
    sm = cuda.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    whole = _checked(cuda.cuDeviceGetDevResource(dev, sm), "device resource")
    params = cuda.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
    params.smCount, params.coscheduledSmCount = 0, coscheduled
    params.flags = (
        cuda.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
    )
    params.localityDomainId = domain
    result, _ = _checked(cuda.cuDevSmResourceSplit(1, whole, 0, [params]), "split")
    desc = _checked(cuda.cuDevResourceGenerateDesc([result[0]], 1), "descriptor")
    flags = cuda.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
    green = _checked(cuda.cuGreenCtxCreate(desc, dev, flags), "green context")
    nonblocking = cuda.CUstream_flags.CU_STREAM_NON_BLOCKING
    handle = _checked(cuda.cuGreenCtxStreamCreate(green, nonblocking, 0), "stream")
    stream = torch.cuda.ExternalStream(int(handle), device=torch.device("cuda", device))
    return stream, green, handle


def _destroy_green(green, handle):
    _checked(cuda.cuStreamDestroy(handle), "stream destroy")
    _checked(cuda.cuGreenCtxDestroy(green), "green context destroy")


@asm((Uint64, Uint32))
def timed_load(address):
    """Cycles of one L2-allocating load whose address depends on the first clock read
    and whose second clock read waits for the data."""
    code = "{ .reg .u64 t0, t1, z, a; .reg .pred p; mov.u64 t0, %clock64; "
    code += "shr.u64 z, t0, 63; add.u64 a, $2, z; ld.global.cg.u32 $1, [a]; "
    code += "setp.ne.u32 p, $1, 0x7fffffff; mov.u64 t1, 0; @p mov.u64 t1, %clock64; "
    return code + "sub.u64 $0, t1, t0; }", "=l,=r,l,~{memory}", (address,)


@asm(Uint32, pure=True)
def smid():
    return "mov.u32 $0, %smid;", "=r", ()


class SmProbe:
    """One CTA per SM (shared memory forces it): thread 0 records ``%smid`` at
    ``smids[cta]`` and times the loads at ``addresses[smid, i]`` (0: none) into
    ``cycles``."""

    @cute.jit
    def __call__(self, smids, addresses, cycles, stream):
        args = (smids, addresses, cycles, stream)
        launch(self.kernel, args, smids.shape[0], 32, pdl=False, smem=150 * 1024)

    @cute.kernel
    def kernel(self, args):
        smids, addresses, cycles = args
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if thread == 0:
            sm = Int32(smid())
            smids[cta] = sm
            if sm < addresses.shape[0]:
                for i in cutlass.range(addresses.shape[1]):
                    address = Int64(addresses[sm, i])
                    if address != 0:  # 0: no load
                        elapsed, _ = timed_load(address)
                        cycles[sm, i] = Int64(elapsed)


@functools.cache
def _sm_probe(device, ctas, slots, loads):
    smids = make_fake_tensor(Int32, (ctas,), (1,), assumed_align=4)
    addresses = make_fake_tensor(Int64, (slots, loads), (loads, 1), assumed_align=8)
    cycles = make_fake_tensor(Int64, (slots, loads), (loads, 1), assumed_align=8)
    with torch.accelerator.device_index(device):
        return cute.compile(SmProbe(), smids, addresses, cycles, make_fake_stream())


def _launch_probe(device, smids, addresses, cycles, stream):
    compiled = _sm_probe(device, smids.shape[0], *addresses.shape)
    compiled(*(to_cute(t, 4) for t in (smids, addresses, cycles)), stream)


def _probe(device, addresses):
    """Cycles [slots, n] of cold loads at ``addresses`` [slots, n] (row = smid), and
    the smids of the SMs that ran (one CTA each)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    dev = torch.device("cuda", device)
    smids = torch.full((sms,), -1, dtype=torch.int32, device=dev)
    cycles = torch.zeros(addresses.shape, dtype=torch.int64, device=dev)
    # Every probed line cold, and no dirty lines left to write back during the probe:
    # write 512 MiB (4x the L2), then read it (clean lines evict the dirty ones).
    flush = torch.ones(128 << 20, dtype=torch.int32, device=dev)
    flush.sum()
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _launch_probe(device, smids, addresses, cycles, stream)
    torch.accelerator.synchronize(dev)
    del flush
    return cycles.cpu(), smids.cpu()


def _probe_memory(device):
    """Two 64 MiB buffers homed on die 0 and die 1 (probe targets)."""
    return tuple(localized_empty(64 << 20, d, device) for d in range(2))


@functools.cache
def green_sms(device: int) -> tuple[frozenset, ...]:
    """The smids each green context of one domain runs on (the driver's sets)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    dev = torch.device("cuda", device)
    sets = []
    for domain in range(domain_count(device)):
        stream, green, handle = _green_stream(device, domain)
        seen = set()
        try:
            with torch.cuda.stream(stream):
                for _ in range(4):  # a few launches: CTAs land on every SM of the set
                    smids = torch.full((sms,), -1, dtype=torch.int32, device=dev)
                    cycles = torch.zeros((1, 1), dtype=torch.int64, device=dev)
                    addresses = torch.zeros_like(cycles)
                    cu_stream = cuda.CUstream(stream.cuda_stream)
                    _launch_probe(device, smids, addresses, cycles, cu_stream)
                    stream.synchronize()
                    seen |= {int(s) for s in smids.cpu() if s >= 0}
        finally:
            torch.accelerator.synchronize(dev)
            _destroy_green(green, handle)
        sets.append(frozenset(seen))
    return tuple(sets)


@functools.cache
def sm_domains(device: int, check_green: bool = False) -> tuple[int, ...]:
    """Die of every smid (index) of ``device``; all zeros on single-die devices. Raises
    unless every SM sees a clear near / far latency gap and each die gets at least 40%
    of the SMs. ``check_green`` (tests): also require agreement with the driver's
    green-context SM sets; not by default, as creating a green context slows the
    process's later kernels (~+0.3 us per MoE layer in a decode chain) even after it
    is destroyed."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if domain_count(device) == 1:
        return (0,) * sms
    memory = _probe_memory(device)
    loads = 16
    slots = 2 * sms  # smids are < 2 x SMs on every part seen; larger ones raise below
    lines = torch.arange(slots * loads, dtype=torch.int64).view(slots, loads)
    offsets = (lines * 61 % ((64 << 20) // 128)) * 128  # distinct lines per smid
    rows = []
    for buffer in memory:
        addresses = (offsets + buffer.data_ptr()).to(f"cuda:{device}")
        cycles, smids = _probe(device, addresses)
        rows.append(cycles)
    seen = sorted(int(s) for s in smids)
    if seen[0] < 0 or seen[-1] >= slots or len(set(seen)) != sms:
        raise RuntimeError(f"die probe: smids {seen[:4]}..{seen[-4:]} not one per SM")
    medians = [(rows[0][s].median().item(), rows[1][s].median().item()) for s in seen]
    unclear = [s for s, (a, b) in zip(seen, medians) if abs(a - b) < MIN_GAP_CYCLES]
    if unclear:
        raise RuntimeError(f"die probe: no clear near / far gap on smids {unclear}")
    table = [-1] * (seen[-1] + 1)
    for s, (a, b) in zip(seen, medians):
        table[s] = 0 if a < b else 1
    if min(table.count(0), table.count(1)) < 0.4 * sms:
        raise RuntimeError(
            f"die probe: {table.count(0)} / {table.count(1)} SMs per die"
        )
    if check_green:
        for domain, members in enumerate(green_sms(device)):
            wrong = [s for s in members if table[s] != domain]
            if wrong:
                raise RuntimeError(
                    f"die probe disagrees with the driver on smids {wrong}"
                )
    return tuple(table)


@asm((Uint32,) * 4)
def stream_load(address):
    """16 bytes at ``address`` through L2 (not L1)."""
    code = "ld.global.cg.v4.u32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (address,)


@asm(Int64)
def global_time():
    return "mov.u64 $0, %globaltimer;", "=l", ()


class StreamProbe:
    """One CTA per SM (shared memory forces it): its 256 threads stream ``loads``
    16-byte words each from ``sources[smid]`` (contiguous) and thread 0 records the
    CTA's start and end ``%globaltimer`` at ``times[smid]``."""

    def __init__(self, ctas, loads):
        self.ctas, self.loads = ctas, loads

    @cute.jit
    def __call__(self, sources, times, stream):
        args = (sources, times, stream)
        launch(self.kernel, args, self.ctas, 256, pdl=False, smem=150 * 1024)

    @cute.kernel
    def kernel(self, args):
        sources, times = args
        thread, _, _ = cute.arch.thread_idx()
        sm = Int32(smid())
        cute.arch.sync_threads()
        start = global_time()
        base = Int64(sources[sm]) + Int64(thread) * 16
        acc = Uint32(0)
        for i in cutlass.range_constexpr(self.loads):
            words = stream_load(base + i * 256 * 16)
            acc = acc ^ words[0] ^ words[1] ^ words[2] ^ words[3]
        if acc == Uint32(0x9E3779B9):  # data dependence on every load; never true
            times[sm, 2] = Int64(acc)
        cute.arch.sync_threads()
        if thread == 0:
            times[sm, 0] = start
            times[sm, 1] = global_time()


@functools.cache
def _stream_probe(device, ctas, slots, loads):
    sources = make_fake_tensor(Int64, (slots,), (1,), assumed_align=8)
    times = make_fake_tensor(Int64, (slots, 3), (3, 1), assumed_align=8)
    kernel = StreamProbe(ctas, loads)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, sources, times, make_fake_stream())


# Per SM, streaming from its own die's memory must take at most this fraction of the
# time streaming from the other die takes, when every SM streams at once (~0.8 on
# healthy parts).
LOCAL_RATIO = 0.95
STREAM_LOADS = 128  # 512 KiB per SM


@functools.cache
def stream_check(device: int) -> tuple[int, ...]:
    """The smids whose streaming disagrees with ``sm_domains``: with every SM reading
    512 KiB from memory homed on its mapped die, then on the other die (cold, three
    rounds, medians), an SM whose local read is not clearly faster (``LOCAL_RATIO``)
    is mapped to the wrong die, or its die's localized memory is not local. Empty when
    the map holds; also on single-die devices."""
    if domain_count(device) == 1:
        return ()
    seen, local, remote = stream_times(device)
    return tuple(
        s for s, a, b in zip(seen, local, remote) if a > LOCAL_RATIO * b or a <= 0
    )


def stream_times(device: int, table=None):
    """(smids, local ns, remote ns) of ``stream_check``'s measurement of ``table``
    (default ``sm_domains``)."""
    table = sm_domains(device) if table is None else table
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    slots = 2 * sms  # as ``sm_domains``
    seen = [s for s, d in enumerate(table) if d >= 0]
    dense = {s: i for i, s in enumerate(seen)}
    slice_bytes = 256 * 16 * STREAM_LOADS
    memory = [localized_empty(len(seen) * slice_bytes, d, device) for d in range(2)]
    for buffer in memory:
        buffer.fill_(1)
    dev = torch.device("cuda", device)
    compiled = _stream_probe(device, sms, slots, STREAM_LOADS)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    elapsed: dict[str, list] = {"local": [], "remote": []}
    for _ in range(3):
        for kind in ("local", "remote"):
            sources = torch.zeros(slots, dtype=torch.int64)
            for s in seen:
                die = table[s] if kind == "local" else 1 - table[s]
                sources[s] = memory[die].data_ptr() + dense[s] * slice_bytes
            sources = sources.to(dev)
            times = torch.zeros((slots, 3), dtype=torch.int64, device=dev)
            flush = torch.ones(128 << 20, dtype=torch.int32, device=dev)
            flush.sum()  # every probed line cold (as ``_probe``)
            compiled(to_cute(sources, 8), to_cute(times, 8), stream)
            torch.accelerator.synchronize(dev)
            del flush
            times = times.cpu()
            elapsed[kind].append([int(times[s, 1] - times[s, 0]) for s in seen])
    local, remote = (
        [statistics.median(r[i] for r in elapsed[k]) for i in range(len(seen))]
        for k in ("local", "remote")
    )
    return seen, local, remote


def chunk_homes(tensor: torch.Tensor, chunks, sms: int = 4) -> list[int]:
    """The measured home die (-1: no clear answer) of each listed 2 MiB chunk of a
    ``localized_empty`` / ``localized_like`` tensor (which starts on a chunk): cold
    loads to distinct lines of the chunk, each after an untimed load to the chunk's
    last line (warms the TLB), timed from ``sms`` SMs of each die (``sm_domains``);
    the die whose SMs read it faster by ``MIN_GAP_CYCLES`` homes it."""
    device = tensor.device.index
    if domain_count(device) == 1:
        return [0] * len(chunks)
    table = sm_domains(device)
    members = [[s for s, d in enumerate(table) if d == domain] for domain in range(2)]
    base = tensor.data_ptr()
    assert base % CHUNK == 0, "a localized tensor starts on a 2 MiB chunk"
    jobs = []  # (smid, chunk position, die of the timing SM, line)
    for k, chunk in enumerate(chunks):
        for die in range(2):
            for j in range(sms):
                sm = members[die][(k * sms + j) % len(members[die])]
                line = (chunk * 977 + (die * sms + j) * 131) % (CHUNK // 128 - 1)
                jobs.append((sm, k, die, base + chunk * CHUNK + line * 128))
    loads = [0] * len(table)
    for sm, *_ in jobs:
        loads[sm] += 2
    addresses = torch.zeros((len(table), max(loads)), dtype=torch.int64)
    filled = [0] * len(table)
    slots = []
    for sm, k, die, address in jobs:
        i = filled[sm]
        addresses[sm, i] = base + chunks[k] * CHUNK + CHUNK - 128  # TLB warm-up
        addresses[sm, i + 1] = address
        filled[sm] += 2
        slots.append((sm, i + 1, k, die))
    cycles, _ = _probe(device, addresses.to(f"cuda:{device}"))
    times: list[tuple[list, list]] = [([], []) for _ in chunks]
    for sm, i, k, die in slots:
        times[k][die].append(float(cycles[sm, i]))
    homes = []
    for near0, near1 in times:
        a, b = statistics.median(near0), statistics.median(near1)
        homes.append(
            0 if b - a > MIN_GAP_CYCLES else 1 if a - b > MIN_GAP_CYCLES else -1
        )
    return homes


def sm_domain_table(device) -> torch.Tensor:
    """``sm_domains`` as an int32 device tensor indexed by ``%smid``."""
    device = torch.device(device)
    return torch.tensor(sm_domains(device.index), dtype=torch.int32, device=device)


def page_domains(tensor: torch.Tensor, probes: int = 16) -> torch.Tensor:
    """Home die (int8, CPU; -1 unknown) of every 4 KiB page of the 2 MiB chunks that
    ``tensor``'s own bytes touch, from the start of the first one. Default
    (interleaved) allocations only. The probes read only ``tensor``'s bytes (a view's
    span, never the rest of its storage): ``probes`` lines per chunk, inside the span,
    are timed from SMs of both dies, each after an untimed load to the span's last line
    in that chunk (warms the TLB); chunks without a clear answer are probed again with
    4x the lines, then left unknown (as is a chunk holding fewer than 2 whole lines of
    the span)."""
    device = tensor.device.index
    first = tensor.data_ptr()
    extent = sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) + 1
    end = first + extent * tensor.element_size() if tensor.numel() else first
    start = first // CHUNK * CHUNK
    chunks = -(-(end - start) // CHUNK)
    if domain_count(device) == 1 or chunks == 0:
        return torch.zeros(chunks * (CHUNK // PAGE), dtype=torch.int8)
    # Each chunk's bytes of the span, relative to the chunk.
    spans = [
        (max(first - c, 0), min(end - c, CHUNK))
        for c in (start + k * CHUNK for k in range(chunks))
    ]
    flips = _chunk_flips(device, start, range(chunks), probes, spans)
    retry = [c for c, flip in enumerate(flips) if flip < 0]
    if retry:
        again = _chunk_flips(
            device, start, retry, 4 * probes, [spans[c] for c in retry]
        )
        for c, flip in zip(retry, again):
            flips[c] = flip
    parity = torch.tensor(PAGE_PARITY, dtype=torch.int8)
    unknown = torch.full_like(parity, -1)
    return torch.cat([parity ^ flip if flip >= 0 else unknown for flip in flips])


def _chunk_flips(device, start, chunks, probes, spans):
    """The bit c (-1: unclear) of each listed chunk at ``start + chunk * 2 MiB``; the
    probes and the warm-up load stay inside the chunk's ``spans`` entry (lo, hi bytes
    relative to the chunk)."""
    table = sm_domains(device)
    members = [[s for s, d in enumerate(table) if d == domain] for domain in range(2)]
    lines = CHUNK // 128
    jobs = []  # (smid, chunk index in the list, page, die of the timing SM, line)
    for k, chunk in enumerate(chunks):
        lo, hi = (-(-spans[k][0] // 128), spans[k][1] // 128)  # whole lines
        if hi - lo < 2:
            continue  # no probe line besides the warm-up line: unclear
        for p in range(probes):
            page = (chunk * 97 + p * 67) % (CHUNK // PAGE)
            line = page * (PAGE // 128) + (page * 13 + chunk) % (PAGE // 128)
            line = lo + line % lines % (hi - 1 - lo)  # never the warm line (hi - 1)
            page = line * 128 // PAGE
            # Half the probes are local under c = 0 and half remote, whatever c is.
            die = PAGE_PARITY[page] ^ (p & 1)
            group = members[die]
            sm = group[(k * probes + p) // 2 % len(group)]
            jobs.append((sm, k, chunk, page, die, line, hi - 1))
    if not jobs:
        return [-1] * len(chunks)
    loads = [0] * len(table)
    for sm, *_ in jobs:
        loads[sm] += 2
    addresses = torch.zeros((len(table), max(loads)), dtype=torch.int64)
    groups: list[tuple[list, list]] = [([], []) for _ in chunks]  # (remote, local)
    filled = [0] * len(table)
    for sm, k, chunk, page, die, line, warm in jobs:
        i, base = (filled[sm], start + chunk * CHUNK)
        addresses[sm, i] = base + warm * 128
        addresses[sm, i + 1] = base + line * 128
        groups[k][die == PAGE_PARITY[page]].append((sm, i + 1))
        filled[sm] += 2
    cycles, _ = _probe(device, addresses.to(f"cuda:{device}"))
    # c = 0 iff the probes local under c = 0 are the faster half. Medians tolerate
    # slow single loads (a few percent of probes take 500+ ns).
    flips = []
    for remote, local in groups:
        if not remote or not local:
            flips.append(-1)
            continue
        fast, slow = (
            statistics.median([float(cycles[s]) for s in v]) for v in (local, remote)
        )
        gap = (slow - fast) / 2.42  # ns; local and remote misses differ by ~165
        flips.append(0 if gap > 80 else 1 if gap < -80 else -1)
    return flips
