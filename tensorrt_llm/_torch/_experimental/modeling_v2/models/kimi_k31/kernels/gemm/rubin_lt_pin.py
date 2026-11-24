# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 GEMMs through cuBLASLt at a pinned algorithm (K3.1 Rubin helix8 targets).

torch's ``F.linear``, the engine's ``cublas_mm`` and its router GEMM run cuBLASLt with the
algorithm its heuristic ranks first for the problem. Another algorithm of the same heuristic list
can run the problem faster with the same bits: an algorithm fixes how each output element's K sum
is accumulated (the kernel's K chain, and with split-K the partition and the reduce), so two
algorithms that accumulate alike write the same words. ``lt_gemm`` runs ``out = a @ b`` with the
algorithm a pin (:class:`LtPin`) names: it asks cuBLASLt's heuristic for the problem as the
measurement did (64 results under five preferences: 64 MiB of workspace and no reduction mask, no
workspace, 64 MiB with each reduction scheme alone) and takes the first algorithm whose
configuration matches every field the pin sets.

The binding is ctypes over the libcublasLt torch loaded (the soname); constants and struct sizes are
cuBLAS 13's (``cublasLt.h``), so a library whose ``cublasLtGetVersion`` is not 13.x is refused
(``LtUnavailable``, and a caller keeps its own GEMM). A plan (descriptors, the algorithm, its
workspace) is built per problem layout and pin outside CUDA graph capture and kept for the process;
a captured call replays it.
Calls of a plan that holds a workspace must be ordered on one stream.
"""

from __future__ import annotations

import ctypes
from typing import NamedTuple, Optional

import torch

# cublas_api.h / library_types.h / cublasLt.h (cuBLAS 13)
_SUCCESS = 0
_OP_N, _OP_T = 0, 1
_COMPUTE_32F = 68
_R_32F, _R_16BF = 0, 14
_DESC_TRANSA, _DESC_TRANSB = 3, 4
_LAYOUT_BATCH_COUNT, _LAYOUT_BATCH_STRIDE = 5, 6
_PREF_MAX_WORKSPACE, _PREF_REDUCTION_MASK = 1, 3
#: cublasLtMatmulAlgoConfigAttributes_t, in LtPin's field order: (attribute, ctypes type).
_CONFIG = (
    (0, ctypes.c_int32),  # CUBLASLT_ALGO_CONFIG_ID
    (1, ctypes.c_uint32),  # TILE_ID
    (6, ctypes.c_uint32),  # STAGES_ID
    (8, ctypes.c_uint16),  # CLUSTER_SHAPE_ID
    (2, ctypes.c_int32),  # SPLITK_NUM
    (3, ctypes.c_uint32),  # REDUCTION_SCHEME
    (4, ctypes.c_uint32),  # CTA_SWIZZLING
    (5, ctypes.c_uint32),  # CUSTOM_OPTION
    (7, ctypes.c_uint16),  # INNER_SHAPE_ID
)
#: The heuristic queries, in order: (workspace bytes, reduction scheme mask or None).
_PREFERENCES = ((64 << 20, None), (0, None), (64 << 20, 1), (64 << 20, 2), (64 << 20, 4))
_RESULTS = 64
_LIB_NAMES = ("libcublasLt.so.13", "libcublasLt.so")
#: cublasLtGetVersion() range whose enums and struct sizes are the ones above (cuBLAS 13.x).
_VERSIONS = range(130000, 140000)


class LtPin(NamedTuple):
    """A cuBLASLt algorithm by its configuration; a None field matches any value."""

    algo: Optional[int]
    tile: Optional[int]
    stages: Optional[int]
    cluster: Optional[int]
    splitk: Optional[int]
    reduction: Optional[int]
    swizzle: Optional[int]
    custom: Optional[int]
    inner: Optional[int]

    def name(self) -> str:
        """``algo66 tile15 st35 cl11 sk1 r0 sw0 co1 is0`` (``*`` for an unset field)."""
        tags = ("algo", "tile", "st", "cl", "sk", "r", "sw", "co", "is")
        return " ".join(f"{t}{'*' if v is None else v}" for t, v in zip(tags, self))


class LtUnavailable(RuntimeError):
    """The pin cannot run here: no plan under capture, or the heuristic does not offer it."""


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


_VP = ctypes.c_void_p
_SIGNATURES = {
    "cublasLtCreate": [ctypes.POINTER(_VP)],
    "cublasLtMatmulDescCreate": [ctypes.POINTER(_VP), ctypes.c_int, ctypes.c_int],
    "cublasLtMatmulDescSetAttribute": [_VP, ctypes.c_int, _VP, ctypes.c_size_t],
    "cublasLtMatrixLayoutCreate": [
        ctypes.POINTER(_VP),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
    ],
    "cublasLtMatrixLayoutSetAttribute": [_VP, ctypes.c_int, _VP, ctypes.c_size_t],
    "cublasLtMatmulPreferenceCreate": [ctypes.POINTER(_VP)],
    "cublasLtMatmulPreferenceDestroy": [_VP],
    "cublasLtMatmulPreferenceSetAttribute": [_VP, ctypes.c_int, _VP, ctypes.c_size_t],
    "cublasLtMatmulAlgoGetHeuristic": [_VP] * 7
    + [ctypes.c_int, ctypes.POINTER(_Heuristic), ctypes.POINTER(ctypes.c_int)],
    "cublasLtMatmulAlgoConfigGetAttribute": [
        ctypes.POINTER(_Algo),
        ctypes.c_int,
        _VP,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ],
    "cublasLtMatmul": [_VP] * 12 + [ctypes.POINTER(_Algo), _VP, ctypes.c_size_t, _VP],
}

_LIB: list = []
_HANDLES: dict = {}
#: (pin, operand layouts, dtypes, device) -> _Plan, per process.
_PLANS: dict = {}


def _lib() -> ctypes.CDLL:
    if not _LIB:
        lib = None
        for name in _LIB_NAMES:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            raise LtUnavailable("libcublasLt is not loadable")
        lib.cublasLtGetVersion.restype = ctypes.c_size_t
        lib.cublasLtGetVersion.argtypes = []
        version = int(lib.cublasLtGetVersion())
        if version not in _VERSIONS:
            raise LtUnavailable(
                f"cuBLASLt {version} is not cuBLAS 13: the binding's enums are 13's"
            )
        for fn, args in _SIGNATURES.items():
            getattr(lib, fn).restype = ctypes.c_int
            getattr(lib, fn).argtypes = args
        _LIB.append(lib)
    return _LIB[0]


def _ok(status: int, what: str) -> None:
    if status != _SUCCESS:
        raise LtUnavailable(f"{what}: cuBLAS status {status}")


def _set(setter, obj, attr: int, ctype, value) -> None:
    v = ctype(value)
    _ok(setter(obj, attr, ctypes.byref(v), ctypes.sizeof(v)), f"attribute {attr}")


def lt_problem(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> dict:
    """The column-major cuBLASLt description of ``out = a @ b`` (per batch).

    cuBLASLt computes ``D [N, M] = op(A) [N, K] @ op(B) [K, M]`` column-major, with ``A := b^T``,
    ``B := a^T`` and ``D := out^T``: a row-major operand is a column-major one transposed for free,
    a transposed view (a stored weight's ``.t()``) is taken with ``op = T``. 2-D ``[M, K] @ [K, N]``
    or batched 3-D ``[B, M, K] @ [B, K, N]``; ``out`` needs a unit column stride. Raises ValueError
    for another layout."""
    if a.dim() != b.dim() or a.dim() != out.dim() or a.dim() not in (2, 3):
        raise ValueError(f"operands of rank {a.dim()}, {b.dim()}, {out.dim()}")
    if a.dim() == 2:
        (m, k), (k2, n), (m2, n2) = a.shape, b.shape, out.shape
        batch, sba, sbb, sbo = 1, 0, 0, 0
        (sra, sca), (srb, scb), (sro, sco) = a.stride(), b.stride(), out.stride()
    else:
        (batch, m, k), (bb, k2, n), (bo, m2, n2) = a.shape, b.shape, out.shape
        if not batch == bb == bo:
            raise ValueError(f"batches {batch}, {bb}, {bo}")
        (sba, sra, sca), (sbb, srb, scb), (sbo, sro, sco) = a.stride(), b.stride(), out.stride()
    if (k, m, n) != (k2, m2, n2):
        raise ValueError(f"shapes {tuple(a.shape)} @ {tuple(b.shape)} -> {tuple(out.shape)}")
    if sco != 1 and n > 1:
        raise ValueError("out has no unit column stride")
    if scb == 1 or n == 1:  # b row-major [K, N]: column-major [N, K]
        transa, a_rows, a_cols, lda = _OP_N, n, k, srb
    elif srb == 1:  # b = W^T of a row-major W [N, K]
        transa, a_rows, a_cols, lda = _OP_T, k, n, scb
    else:
        raise ValueError("b has no unit stride")
    if sca == 1 or k == 1:  # a row-major [M, K]: column-major [K, M]
        transb, b_rows, b_cols, ldb = _OP_N, k, m, sra
    elif sra == 1:
        transb, b_rows, b_cols, ldb = _OP_T, m, k, sca
    else:
        raise ValueError("a has no unit stride")
    for rows, ld in ((a_rows, lda), (b_rows, ldb), (n, sro)):
        if ld < rows:
            raise ValueError(f"leading dimension {ld} < rows {rows}")
    return dict(
        m=n,
        n=m,
        k=k,
        batch=batch,
        transa=transa,
        transb=transb,
        A=dict(rows=a_rows, cols=a_cols, ld=lda, stride=sbb),
        B=dict(rows=b_rows, cols=b_cols, ld=ldb, stride=sba),
        D=dict(rows=n, cols=m, ld=sro, stride=sbo),
    )


def _dtype(t: torch.Tensor) -> int:
    return {torch.bfloat16: _R_16BF, torch.float32: _R_32F}[t.dtype]


def _handle(device: torch.device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _HANDLES:
        h = _VP()
        with torch.cuda.device(index):
            _ok(_lib().cublasLtCreate(ctypes.byref(h)), "cublasLtCreate")
        _HANDLES[index] = h
    return _HANDLES[index]


class _Plan:
    """One problem's descriptors, the pinned algorithm and its workspace."""

    def __init__(self, prob: dict, d_type: int, pin: LtPin, device: torch.device):
        lib = _lib()
        self.handle = _handle(device)
        self.desc = _VP()
        _ok(lib.cublasLtMatmulDescCreate(ctypes.byref(self.desc), _COMPUTE_32F, _R_32F), "desc")
        set_desc = lib.cublasLtMatmulDescSetAttribute
        _set(set_desc, self.desc, _DESC_TRANSA, ctypes.c_int32, prob["transa"])
        _set(set_desc, self.desc, _DESC_TRANSB, ctypes.c_int32, prob["transb"])
        self.layouts = {}
        for name, typ in (("A", _R_16BF), ("B", _R_16BF), ("D", d_type)):
            spec, lay = prob[name], _VP()
            _ok(
                lib.cublasLtMatrixLayoutCreate(
                    ctypes.byref(lay), typ, spec["rows"], spec["cols"], spec["ld"]
                ),
                f"layout {name}",
            )
            if prob["batch"] > 1:
                set_lay = lib.cublasLtMatrixLayoutSetAttribute
                _set(set_lay, lay, _LAYOUT_BATCH_COUNT, ctypes.c_int32, prob["batch"])
                _set(set_lay, lay, _LAYOUT_BATCH_STRIDE, ctypes.c_int64, spec["stride"])
            self.layouts[name] = lay
        self.alpha, self.beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        found = self._find(pin)
        if found is None:
            raise LtUnavailable(f"cuBLASLt's heuristic does not offer {pin.name()} for the problem")
        self.algo, ws = found
        self.workspace = torch.empty(max(ws, 1), dtype=torch.uint8, device=device)
        self.ws_bytes = ws

    def _config(self, algo: _Algo) -> tuple:
        values = []
        for attr, ctype in _CONFIG:
            v, written = ctype(0), ctypes.c_size_t(0)
            st = _lib().cublasLtMatmulAlgoConfigGetAttribute(
                ctypes.byref(algo), attr, ctypes.byref(v), ctypes.sizeof(v), ctypes.byref(written)
            )
            values.append(int(v.value) if st == _SUCCESS else None)
        return tuple(values)

    def _find(self, pin: LtPin):
        """(algorithm, workspace bytes) of the first heuristic result the pin matches, or None."""
        lib, lay = _lib(), self.layouts
        seen = set()
        for ws, mask in _PREFERENCES:
            pref = _VP()
            _ok(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "preference")
            try:
                set_pref = lib.cublasLtMatmulPreferenceSetAttribute
                _set(set_pref, pref, _PREF_MAX_WORKSPACE, ctypes.c_uint64, ws)
                if mask is not None:
                    _set(set_pref, pref, _PREF_REDUCTION_MASK, ctypes.c_uint32, mask)
                res, got = (_Heuristic * _RESULTS)(), ctypes.c_int(0)
                st = lib.cublasLtMatmulAlgoGetHeuristic(
                    self.handle,
                    self.desc,
                    lay["A"],
                    lay["B"],
                    lay["D"],
                    lay["D"],
                    pref,
                    _RESULTS,
                    res,
                    ctypes.byref(got),
                )
            finally:
                lib.cublasLtMatmulPreferenceDestroy(pref)
            if st != _SUCCESS:
                continue
            for i in range(got.value):
                if res[i].state != _SUCCESS:
                    continue
                key = bytes(res[i].algo.data)
                if key in seen or res[i].workspaceSize > _PREFERENCES[0][0]:
                    continue
                seen.add(key)
                config = self._config(res[i].algo)
                if all(p is None or p == c for p, c in zip(pin, config)):
                    algo = _Algo()
                    ctypes.memmove(
                        ctypes.byref(algo), ctypes.byref(res[i].algo), ctypes.sizeof(_Algo)
                    )
                    return algo, int(res[i].workspaceSize)
        return None

    def run(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        lay = self.layouts
        _ok(
            _lib().cublasLtMatmul(
                self.handle,
                self.desc,
                ctypes.byref(self.alpha),
                _VP(b.data_ptr()),
                lay["A"],
                _VP(a.data_ptr()),
                lay["B"],
                ctypes.byref(self.beta),
                _VP(out.data_ptr()),
                lay["D"],
                _VP(out.data_ptr()),
                lay["D"],
                ctypes.byref(self.algo),
                _VP(self.workspace.data_ptr()),
                self.ws_bytes,
                _VP(torch.cuda.current_stream(out.device).cuda_stream),
            ),
            "cublasLtMatmul",
        )


def lt_plan_key(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, pin: LtPin) -> tuple:
    return (
        tuple(pin),
        tuple(a.shape),
        a.stride(),
        tuple(b.shape),
        b.stride(),
        tuple(out.shape),
        out.stride(),
        out.dtype,
        out.device,
    )


def lt_gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, pin: LtPin) -> None:
    """``out = a @ b`` (FP32 accumulation, one rounding to ``out``'s dtype) on the current stream,
    with the algorithm ``pin`` names. Builds the layout's plan on its first call, outside capture."""
    key = lt_plan_key(a, b, out, pin)
    plan = _PLANS.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise LtUnavailable("no plan for this layout outside capture")
        plan = _Plan(lt_problem(a, b, out), _dtype(out), pin, out.device)
        _PLANS[key] = plan
    plan.run(a, b, out)
