# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locality domains ("uGPU") of a multi-die GPU (Rubin VR200: 2 domains of 100 SMs + 12 remainder SMs) for the
streaming decode kernels of this target: the SM -> domain map, per-domain weight placement, and the CTA -> SM
assignment a persistent kernel uses to read only its own die's memory.

A kernel launched with one CTA per SM reads ``%smid`` once and takes its logical block index from a host-built
permutation (:meth:`LocalityMap.logical_map`), so every per-CTA structure (work plan row, epochs, roles) follows its
die. Weights that are split by output rows into per-domain halves (:meth:`LocalityMap.split_rows`) live in one virtual
range whose halves are physically in the two domains. Placement only: the arithmetic is unchanged, so results are
bitwise those of the unplaced weights.

The SM -> domain map is attested by the driver: the SMs are split by locality domain
(``cuDevSmResourceSplit`` with ``CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID``), and a ``%smid`` probe runs on a green
context of each split. Allocations use driver VMM with ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`` and are checked
with ``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL``; any failure raises (fail closed)."""

from __future__ import annotations

import threading
from typing import Dict, List, Sequence, Tuple

import torch

_LOCK = threading.Lock()
_MAPS: Dict[int, "LocalityMap"] = {}
_PROBE_SRC = r"""
extern "C" __global__ void smid_probe(int* out) {
    if (threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        out[blockIdx.x] = (int)s;
    }
}
"""


def _drv():
    from cuda.bindings import driver

    return driver


def _ck(res, what: str):
    """cuda-python returns (CUresult, *values): raise on an error, return the values."""
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"locality: {what} failed: {err}")
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


class _Mapping:
    """One virtual range backed by per-domain physical allocations, exposed to torch through
    ``__cuda_array_interface__``. Lives as long as a tensor that views it (the tensor holds this object)."""

    def __init__(self, device: int, pieces: Sequence[Tuple[int, int]]):
        """``pieces``: (bytes, domain) in address order; each a multiple of the allocation granularity."""
        d = _drv()
        self.device = device
        self.size = sum(n for n, _ in pieces)
        self.va = _ck(d.cuMemAddressReserve(self.size, 0, 0, 0), "cuMemAddressReserve")
        self.handles = []
        off = 0
        try:
            for nbytes, dom in pieces:
                prop = _alloc_prop(device, dom)
                h = _ck(d.cuMemCreate(nbytes, prop, 0), f"cuMemCreate(domain {dom})")
                self.handles.append((h, nbytes))
                _ck(d.cuMemMap(int(self.va) + off, nbytes, 0, h, 0), "cuMemMap")
                off += nbytes
            acc = d.CUmemAccessDesc()
            acc.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            acc.location.id = device
            acc.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _ck(d.cuMemSetAccess(self.va, self.size, [acc], 1), "cuMemSetAccess")
        except Exception:
            self.close()
            raise
        self.__cuda_array_interface__ = {
            "shape": (self.size,),
            "typestr": "|u1",
            "data": (int(self.va), False),
            "version": 3,
            "strides": None,
            "stream": None,
        }

    def close(self):
        d = _drv()
        off = 0
        for h, n in self.handles:
            d.cuMemUnmap(int(self.va) + off, n)
            d.cuMemRelease(h)
            off += n
        self.handles = []
        if self.va is not None:
            d.cuMemAddressFree(self.va, self.size)
            self.va = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_LIBCUDA = None


def _libcuda():
    global _LIBCUDA
    if _LIBCUDA is None:
        import ctypes

        _LIBCUDA = ctypes.CDLL("libcuda.so.1")
    return _LIBCUDA


def _alloc_prop(device: int, domain: int):
    d = _drv()
    prop = d.CUmemAllocationProp()
    prop.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    prop.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    prop.location.localized.deviceId = device
    prop.location.localized.localityDomainId = domain
    return prop


class LocalityMap:
    """The locality domains of one device. Build with :meth:`get` before CUDA-graph capture."""

    def __init__(self, device: int):
        d = _drv()
        _ck(d.cuInit(0), "cuInit")
        dev = _ck(d.cuDeviceGet(device), "cuDeviceGet")
        attr = d.CUdevice_attribute
        self.device = device
        self.n_domains = int(
            _ck(
                d.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, dev),
                "domain count",
            )
        )
        self.sms_per_domain = int(
            _ck(
                d.cuDeviceGetAttribute(
                    attr.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, dev
                ),
                "SMs per domain",
            )
        )
        self.num_sms = int(
            _ck(d.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "SMs")
        )
        if self.n_domains != 2:
            raise RuntimeError(
                f"locality: {self.n_domains} locality domains (this layer handles 2)"
            )
        self._domain_sms = self._probe(d, dev)
        dom = [-1] * self.num_sms
        for k, sms in enumerate(self._domain_sms):
            for s in sms:
                if dom[s] != -1:
                    raise RuntimeError(f"locality: SM {s} probed in two domains")
                dom[s] = k
        self.sm_domain: List[int] = dom
        self.remainder: List[int] = [s for s in range(self.num_sms) if dom[s] < 0]
        rt = _alloc_prop(device, 0)
        self.granularity = int(
            _ck(
                d.cuMemGetAllocationGranularity(
                    rt, d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
                ),
                "granularity",
            )
        )

    @classmethod
    def get(cls, device) -> "LocalityMap":
        idx = torch.device(device).index if not isinstance(device, int) else device
        idx = torch.cuda.current_device() if idx is None else idx
        with _LOCK:
            m = _MAPS.get(idx)
            if m is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "LocalityMap probes the device: build it before CUDA-graph capture"
                    )
                m = _MAPS[idx] = cls(idx)
        return m

    def sms(self, domain: int) -> List[int]:
        return list(self._domain_sms[domain])

    @property
    def sm_table(self) -> torch.Tensor:
        """int32 [num_sms] on the device: (domain << 16) | rank within the domain's sorted SMs; -1 = remainder SM.
        For kernels that pick work by %smid themselves."""
        t = getattr(self, "_sm_table", None)
        if t is None:
            v = [-1] * self.num_sms
            for k, sms in enumerate(self._domain_sms):
                for i, s in enumerate(sms):
                    v[s] = (k << 16) | i
            t = self._sm_table = torch.tensor(v, dtype=torch.int32, device=f"cuda:{self.device}")
        return t

    def _probe(self, d, dev) -> List[List[int]]:
        """Split the SMs by locality domain (strict, no backfill) and record %smid of a probe kernel launched on a
        green context of each part: the driver's own SM -> domain assignment."""
        from cuda.bindings import nvrtc

        full = _ck(
            d.cuDeviceGetDevResource(dev, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
            "dev resource",
        )
        params = []
        for k in range(self.n_domains):
            p = d.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            p.smCount = self.sms_per_domain
            p.flags = d.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
            p.localityDomainId = k
            params.append(p)
        res, _rem = _ck(
            d.cuDevSmResourceSplit(self.n_domains, full, 0, params), "cuDevSmResourceSplit"
        )
        # the probe module, compiled for this device
        major = _ck(
            d.cuDeviceGetAttribute(
                d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
            ),
            "cc",
        )
        minor = _ck(
            d.cuDeviceGetAttribute(
                d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
            ),
            "cc",
        )
        prog = _ck(
            nvrtc.nvrtcCreateProgram(_PROBE_SRC.encode(), b"smid_probe.cu", 0, [], []),
            "nvrtcCreateProgram",
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if int(err) != 0:
            raise RuntimeError(f"locality: probe compile failed: {err}")
        n = _ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        cubin = b" " * n
        _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
        out: List[List[int]] = []
        nb = 4 * self.sms_per_domain  # more blocks than SMs: every SM of the part shows up
        torch.cuda.set_device(self.device)
        buf = torch.full((nb,), -1, dtype=torch.int32, device=f"cuda:{self.device}")
        for k in range(self.n_domains):
            desc = _ck(d.cuDevResourceGenerateDesc([res[k]], 1), "resource desc")
            g = _ck(
                d.cuGreenCtxCreate(desc, dev, d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM),
                "green ctx",
            )
            st = _ck(
                d.cuGreenCtxStreamCreate(g, d.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
                "green stream",
            )
            ctx = _ck(d.cuCtxFromGreenCtx(g), "ctx from green ctx")
            _ck(d.cuCtxPushCurrent(ctx), "push ctx")
            try:
                mod = _ck(d.cuModuleLoadData(cubin), "probe module")
                fn = _ck(d.cuModuleGetFunction(mod, b"smid_probe"), "probe fn")
                buf.fill_(-1)
                torch.cuda.synchronize(self.device)
                import ctypes

                arg = ctypes.c_void_p(buf.data_ptr())
                _ck(
                    d.cuLaunchKernel(
                        fn, nb, 1, 1, 32, 1, 1, 0, st, ((arg,), (ctypes.c_void_p,)), 0
                    ),
                    "probe launch",
                )
                _ck(d.cuStreamSynchronize(st), "probe sync")
                sms = sorted(set(int(x) for x in buf.tolist() if x >= 0))
                d.cuModuleUnload(mod)
            finally:
                d.cuCtxPopCurrent()
                d.cuStreamDestroy(st)
                d.cuGreenCtxDestroy(g)
            if len(sms) != self.sms_per_domain:
                raise RuntimeError(
                    f"locality: domain {k} probed {len(sms)} SMs, expected {self.sms_per_domain}"
                )
            out.append(sms)
        return out

    def logical_map(self, per_domain: int, extra: int) -> Tuple[torch.Tensor, List[int]]:
        """The SM -> logical block index permutation for a kernel of ``2 * per_domain + extra`` CTAs, one per SM:
        logical blocks [d * per_domain, (d + 1) * per_domain) run on domain d's first ``per_domain`` SMs; logical
        blocks 2 * per_domain .. run on the rest (the leftover domain SMs, alternating domains, then the remainder
        SMs). Returns (int32 [num_sms] on the device, -1 = no block; the domain of each extra block, -1 remainder)."""
        if 2 * per_domain + extra != self.num_sms:
            raise ValueError(
                f"locality: {2 * per_domain} + {extra} CTAs must cover the {self.num_sms} SMs"
            )
        if per_domain > self.sms_per_domain:
            raise ValueError(f"locality: {per_domain} CTAs per domain > {self.sms_per_domain} SMs")
        lmap = [-1] * self.num_sms
        for k in range(2):
            for i, s in enumerate(self._domain_sms[k][:per_domain]):
                lmap[s] = k * per_domain + i
        left = [self._domain_sms[k][per_domain:] for k in range(2)]
        rest = [x for pair in zip(*left) for x in pair] + self.remainder
        ext_dom = []
        for i, s in enumerate(rest):
            lmap[s] = 2 * per_domain + i
            ext_dom.append(self.sm_domain[s])
        assert sorted(lmap) == list(range(self.num_sms))
        return torch.tensor(lmap, dtype=torch.int32, device=f"cuda:{self.device}"), ext_dom

    def domain_streams(self) -> List[torch.cuda.ExternalStream]:
        """One non-blocking stream per domain on a green context holding exactly that domain's SMs (kept alive with
        this map): work launched on stream d runs only on domain d's SMs. Fork / join with events from the caller's
        stream; graph capture follows the forks. Built once, before capture."""
        st = getattr(self, "_streams", None)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("locality.domain_streams: build before CUDA-graph capture")
            d = _drv()
            dev = _ck(d.cuDeviceGet(self.device), "cuDeviceGet")
            full = _ck(
                d.cuDeviceGetDevResource(dev, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
                "dev resource",
            )
            params = []
            for k in range(self.n_domains):
                p = d.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
                p.smCount = self.sms_per_domain
                p.flags = d.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
                p.localityDomainId = k
                params.append(p)
            res, _rem = _ck(
                d.cuDevSmResourceSplit(self.n_domains, full, 0, params), "cuDevSmResourceSplit"
            )
            self._green = []
            st = []
            for k in range(self.n_domains):
                desc = _ck(d.cuDevResourceGenerateDesc([res[k]], 1), "resource desc")
                g = _ck(
                    d.cuGreenCtxCreate(
                        desc, dev, d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
                    ),
                    "green",
                )
                s_ = _ck(
                    d.cuGreenCtxStreamCreate(g, d.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
                    "green stream",
                )
                self._green.append((g, s_))
                st.append(torch.cuda.ExternalStream(int(s_), device=f"cuda:{self.device}"))
            self._streams = st
        return st

    def domain_of(self, ptr: int) -> int:
        """The locality domain ordinal of a device address (-1: not placed in one domain). Through libcuda
        directly: cuda-python's attribute helper does not know this attribute."""
        import ctypes

        d = _drv()
        lib = _libcuda()
        out = ctypes.c_int(-9)
        attr = int(d.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL)
        err = lib.cuPointerGetAttribute(ctypes.byref(out), ctypes.c_int(attr), ctypes.c_uint64(ptr))
        if err != 0:
            raise RuntimeError(
                f"locality: cuPointerGetAttribute(LOCALITY_DOMAIN_ORDINAL) failed: {err}"
            )
        return int(out.value)

    def place_split(self, t: torch.Tensor, split_row: int) -> torch.Tensor:
        """Split-mapped placement: a copy of ``t`` (contiguous, rows along dim 0) with the SAME shape and layout, in one
        virtual range whose rows [0, split_row) are physically in domain 0 and rows [split_row, N) in domain 1. The
        split must fall on an allocation granule (rows [0, split_row) a multiple of 2 MiB on VR200); the tail is padded
        to a granule in domain 1. Stock consumers read it as a plain tensor; no extra memory once ``t`` is dropped."""
        if not t.is_contiguous():
            raise ValueError("locality.place_split: needs a contiguous tensor")
        row_bytes = t[0].numel() * t.element_size() if t.dim() > 1 else t.element_size()
        head = split_row * row_bytes
        total = t.numel() * t.element_size()
        if not 0 < head < total or head % self.granularity:
            raise ValueError(
                f"locality.place_split: rows [0, {split_row}) = {head} B must be a granule multiple "
                f"({self.granularity} B) inside the {total} B tensor"
            )
        tail = -(-(total - head) // self.granularity) * self.granularity
        m = _Mapping(self.device, [(head, 0), (tail, 1)])
        flat = torch.as_tensor(m, device=f"cuda:{self.device}")
        out = flat[:total].view(t.dtype).view(t.shape)
        out.copy_(t)
        out._locality_mapping = m
        for q, k in (
            (out.data_ptr(), 0),
            (out.data_ptr() + head - 1, 0),
            (out.data_ptr() + head, 1),
            (out.data_ptr() + total - 1, 1),
        ):
            if self.domain_of(q) != k:
                raise RuntimeError(f"locality.place_split: byte {q:#x} is not in domain {k}")
        return out

    def replace_param_(self, param: torch.nn.Parameter, split_row: int) -> torch.nn.Parameter:
        """Re-place a parameter in place with :meth:`place_split` (swaps ``param.data``; the old block returns to the
        caching allocator, so call ``torch.cuda.empty_cache()`` between large re-placements)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("locality.replace_param_: not during CUDA-graph capture")
        param.data = self.place_split(param.data, split_row)
        return param

    def split_rows(self, t: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Place ``t`` (contiguous, 1-byte elements, rows along ``dim`` split into two equal halves) as
        [2, *t.shape[:dim], t.shape[dim] // 2, *t.shape[dim + 1:]]: half k (rows [k h, (k + 1) h) of every leading
        index) in domain k's memory, one virtual range. Each half's byte size must be a multiple of the allocation
        granularity (2 MiB on VR200). The copy is exact; the caller drops ``t`` afterwards."""
        if t.element_size() != 1 or not t.is_contiguous():
            raise ValueError("locality.split_rows: needs a contiguous 1-byte tensor")
        n = t.shape[dim]
        if n % 2:
            raise ValueError(f"locality.split_rows: {n} rows don't split in halves")
        half_bytes = t.numel() // 2
        if half_bytes % self.granularity:
            raise ValueError(
                f"locality.split_rows: a half is {half_bytes} B, not a multiple of the {self.granularity} B granularity"
            )
        m = _Mapping(self.device, [(half_bytes, 0), (half_bytes, 1)])
        flat = torch.as_tensor(m, device=f"cuda:{self.device}")
        shape = (2,) + tuple(t.shape[:dim]) + (n // 2,) + tuple(t.shape[dim + 1 :])
        out = flat.view(torch.uint8).view(shape)
        src = t.view(torch.uint8)
        for k in range(2):
            out[k].copy_(src.narrow(dim, k * (n // 2), n // 2))
        out._locality_mapping = m  # the tensor keeps its mapping alive
        for k in range(2):
            p = out[k].data_ptr()
            for q in (p, p + half_bytes - 1):
                if self.domain_of(q) != k:
                    raise RuntimeError(f"locality.split_rows: byte {q:#x} is not in domain {k}")
        return out
