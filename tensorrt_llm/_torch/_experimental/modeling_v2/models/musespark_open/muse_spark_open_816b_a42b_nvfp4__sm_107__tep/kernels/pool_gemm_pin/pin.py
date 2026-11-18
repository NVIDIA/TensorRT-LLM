# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``pool_gemm_pin``: the MoE pool GEMM of pinned row counts on one pinned cuBLASLt algorithm.

Where the call is: every MoE layer's input prep projects its BF16 ``[rows, 8192]`` input
through the replicated pool ``Linear(8192, 4096)`` (``MuseSparkOpenMoE.prepare_moe_inputs`` ->
``_replicated_projection``; above 8 rows that is ``module(hidden_states)``, the built-in
``Linear`` -> ``UnquantizedLinearMethod`` -> ``F.linear``). ``F.linear`` takes cuBLAS's
heuristic pick for the row count. On sm107 at 744..768 rows that pick is
``nvjet_sm107_tst_192x128_64x11_2x2_2cta_h`` (grid 132), while the library's own heuristic list
holds a 128x128 split-K-1 kernel (``nvjet_sm107_tst_128x128_64x13_1x2_2cta_v``, grid 192) that
runs the same product faster and writes the same bits.

``install`` re-classes every layer's pool ``Linear`` with ``PinnedPoolLinear`` in front of the
class it has (``dense_gemms``' ``MuseSparkOpenTepDenseLinear``, whose low-M dispatch serves only
small calls, or the built-in ``Linear``). Its ``forward`` runs the pinned algorithm
(``PinnedPoolGemm``) for a call it ``takes``: a 2-D contiguous BF16 input of a pinned row count
on the module's own device, a BF16 ``[4096, 8192]`` contiguous weight, no bias, no LoRA, the
default GEMM backend, not compiling.
Every other call is the class behind it, unchanged. The choice reads shapes, dtypes and the
row count only, so every rank takes the same path at every call. A call that
``context_fp8_dense``'s MXFP8 copy of the pool takes (a large context call on an aggregated
server) is declined, so that lever keeps its calls.

Values. Bitwise the built-in path's: split-K 1 with the same 64-wide K step, so every output
element accumulates the same products in the same order. Measured on sm107 with cuBLASLt
13.8.0.4 at every served row count 584..672 and 744..768 (step 8) on the checkpoint's 62 pool
weights and real-gain inputs (embedding rows, heavy-tailed rows with massive channels, zero
padding rows, large magnitudes), 64 input / weight pairs per row count, and run to run. The
pin is refused (``install`` raises ``ValueError``) on another device capability, another
cuBLASLt version, when the library's heuristic list does not hold the pinned configuration,
or when it fails ``cublasLtMatmulAlgoCheck`` at a pinned row count.
"""

from __future__ import annotations

import ctypes
import functools
from typing import Dict, FrozenSet, Iterable, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.logger import logger

#: The pool projection: hidden 8192 -> expert hidden 4096.
HIDDEN = 8192
POOLED = 4096
#: The row counts a pin may cover: decode steps of 65..96 requests at K = 7 (8 rows each).
MIN_ROWS = 520
MAX_ROWS = 768
ROW_STEP = 8
#: The device capabilities and cuBLASLt versions (``cublasLtGetVersion``) the pin is validated on.
VALIDATED_CAPABILITIES = frozenset({(10, 7)})
VALIDATED_LT_VERSIONS = frozenset({130800})
#: The pinned algorithm's configuration (``cublasLtMatmulAlgoConfigGetAttribute``): algo id 66,
#: tile 20 (128x128), stages 35, custom option 0, cluster shape 5 (1x2), split-K 1, no
#: reduction, no swizzle: ``nvjet_sm107_tst_128x128_64x13_1x2_2cta_v``.
PIN_CONFIG: Tuple[Tuple[str, int], ...] = (
    ("id", 66),
    ("tile", 20),
    ("splitk", 1),
    ("red", 0),
    ("swz", 0),
    ("custom", 0),
    ("stages", 35),
    ("inner", 0),
    ("cluster", 5),
)
#: The row count whose heuristic list is searched first for the pinned configuration (where it
#: was found), then the pinned row counts themselves.
SOURCE_ROWS = 680

# cuBLASLt enums (cublasLt.h).
_COMPUTE_32F = 68
_R_32F = 0
_R_16BF = 14
_DESC_TRANSA = 3
_DESC_TRANSB = 4
_PREF_MAX_WORKSPACE = 1
_OP_N = 0
_OP_T = 1
_HEURISTIC_COUNT = 128
#: The search preference's workspace: the pinned algorithm itself runs without one.
_SEARCH_WORKSPACE = 64 << 20
#: ``(name, attribute, size)`` of the configuration attributes ``PIN_CONFIG`` names.
_CONFIG_ATTRS = (
    ("id", 0, 4),
    ("tile", 1, 4),
    ("splitk", 2, 4),
    ("red", 3, 4),
    ("swz", 4, 4),
    ("custom", 5, 4),
    ("stages", 6, 4),
    ("inner", 7, 2),
    ("cluster", 8, 2),
)


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


def pinned_rows(ranges: Iterable[Tuple[int, int]]) -> FrozenSet[int]:
    """The row counts the pin covers: every multiple of 8 in the inclusive ``ranges``.

    Raises:
        ValueError: for a range outside ``MIN_ROWS..MAX_ROWS`` or one with no multiple of 8.
    """
    rows = set()
    for low, high in ranges:
        if not MIN_ROWS <= low <= high <= MAX_ROWS:
            raise ValueError(
                f"[musespark_open] pool_gemm_pin rows {low}..{high} outside {MIN_ROWS}..{MAX_ROWS}"
            )
        covered = [r for r in range(low, high + 1) if r % ROW_STEP == 0]
        if not covered:
            raise ValueError(f"[musespark_open] pool_gemm_pin rows {low}..{high}: no multiple of 8")
        rows.update(covered)
    return frozenset(rows)


def _library() -> ctypes.CDLL:
    """The cuBLASLt library this process's torch loaded (the one ``F.linear`` runs)."""
    if torch.cuda.is_available():
        # cuBLAS(Lt) loads lazily: one small GEMM makes sure it is mapped.
        x = torch.zeros(8, 8, device="cuda", dtype=torch.bfloat16)
        torch.matmul(x, x)
        torch.cuda.synchronize()
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libcublasLt.so" in line:
                return ctypes.CDLL(line.split()[-1])
    raise ValueError("[musespark_open] pool_gemm_pin: no cuBLASLt library is loaded")


def _check(status: int, what: str) -> None:
    if int(status) != 0:
        raise ValueError(f"[musespark_open] pool_gemm_pin: cuBLASLt {what} status {int(status)}")


class PinnedPoolGemm:
    """``F.linear(x, weight)`` for the pool's BF16 ``[rows, 8192] x [4096, 8192]`` on the pinned
    cuBLASLt algorithm, at the row counts ``rows`` (built for the current device)."""

    def __init__(self, rows: Iterable[int]) -> None:
        self.rows = frozenset(rows)
        self._reported = set()
        capability = tuple(torch.cuda.get_device_capability())
        if capability not in VALIDATED_CAPABILITIES:
            raise ValueError(
                f"[musespark_open] pool_gemm_pin is validated on sm_107 only, not sm_"
                f"{capability[0]}{capability[1]}"
            )
        lib = _library()
        self.version = int(lib.cublasLtGetVersion())
        if self.version not in VALIDATED_LT_VERSIONS:
            raise ValueError(
                f"[musespark_open] pool_gemm_pin is validated on cuBLASLt "
                f"{sorted(VALIDATED_LT_VERSIONS)}, not {self.version}"
            )
        self.device = torch.cuda.current_device()
        self._lib = lib
        self._handle = ctypes.c_void_p()
        _check(lib.cublasLtCreate(ctypes.byref(self._handle)), "create")
        self._desc = ctypes.c_void_p()
        _check(
            lib.cublasLtMatmulDescCreate(
                ctypes.byref(self._desc), ctypes.c_int(_COMPUTE_32F), ctypes.c_int(_R_32F)
            ),
            "matmul descriptor",
        )
        # D[4096, rows] (column-major) = op_T(W as K x N column-major) x X (K x rows).
        for attr, op in ((_DESC_TRANSA, _OP_T), (_DESC_TRANSB, _OP_N)):
            value = ctypes.c_int32(op)
            _check(
                lib.cublasLtMatmulDescSetAttribute(
                    self._desc, ctypes.c_int(attr), ctypes.byref(value), ctypes.c_size_t(4)
                ),
                "transpose",
            )
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)
        self._weight_layout = self._layout(HIDDEN, POOLED, HIDDEN)
        self._layouts: Dict[int, Tuple[ctypes.c_void_p, ctypes.c_void_p]] = {
            rows: (self._layout(HIDDEN, rows, HIDDEN), self._layout(POOLED, rows, POOLED))
            for rows in sorted(self.rows)
        }
        self._algo = self._find()
        for rows in sorted(self.rows):
            workspace = self._algo_check(rows)
            if workspace != 0:
                raise ValueError(
                    f"[musespark_open] pool_gemm_pin: the pinned algorithm asks for "
                    f"{workspace} bytes of workspace at {rows} rows"
                )

    def _layout(self, rows: int, cols: int, ld: int) -> ctypes.c_void_p:
        layout = ctypes.c_void_p()
        _check(
            self._lib.cublasLtMatrixLayoutCreate(
                ctypes.byref(layout),
                ctypes.c_int(_R_16BF),
                ctypes.c_uint64(rows),
                ctypes.c_uint64(cols),
                ctypes.c_int64(ld),
            ),
            "matrix layout",
        )
        return layout

    def _config(self, algo: _Algo) -> Tuple[Tuple[str, Optional[int]], ...]:
        config = []
        for name, attr, size in _CONFIG_ATTRS:
            value = ctypes.c_uint64(0)
            written = ctypes.c_size_t(0)
            status = self._lib.cublasLtMatmulAlgoConfigGetAttribute(
                ctypes.byref(algo),
                ctypes.c_int(attr),
                ctypes.byref(value),
                ctypes.c_size_t(size),
                ctypes.byref(written),
            )
            config.append((name, int(value.value) if int(status) == 0 else None))
        return tuple(config)

    def _find(self) -> _Algo:
        """The pinned configuration as the library's own heuristic result (never hand-built)."""
        preference = ctypes.c_void_p()
        _check(self._lib.cublasLtMatmulPreferenceCreate(ctypes.byref(preference)), "preference")
        try:
            limit = ctypes.c_uint64(_SEARCH_WORKSPACE)
            _check(
                self._lib.cublasLtMatmulPreferenceSetAttribute(
                    preference,
                    ctypes.c_int(_PREF_MAX_WORKSPACE),
                    ctypes.byref(limit),
                    ctypes.c_size_t(8),
                ),
                "preference workspace",
            )
            for rows in (SOURCE_ROWS,) + tuple(sorted(self.rows)):
                x_layout, d_layout = (
                    self._layouts[rows]
                    if rows in self._layouts
                    else (self._layout(HIDDEN, rows, HIDDEN), self._layout(POOLED, rows, POOLED))
                )
                results = (_Heuristic * _HEURISTIC_COUNT)()
                count = ctypes.c_int(0)
                status = self._lib.cublasLtMatmulAlgoGetHeuristic(
                    self._handle,
                    self._desc,
                    self._weight_layout,
                    x_layout,
                    d_layout,
                    d_layout,
                    preference,
                    ctypes.c_int(_HEURISTIC_COUNT),
                    results,
                    ctypes.byref(count),
                )
                if int(status) != 0:
                    continue
                for i in range(count.value):
                    if self._config(results[i].algo) == PIN_CONFIG:
                        return _Algo((ctypes.c_uint64 * 8)(*list(results[i].algo.data)))
        finally:
            self._lib.cublasLtMatmulPreferenceDestroy(preference)
        raise ValueError(
            f"[musespark_open] pool_gemm_pin: cuBLASLt {self.version} lists no algorithm "
            f"{dict(PIN_CONFIG)}"
        )

    def _algo_check(self, rows: int) -> int:
        x_layout, d_layout = self._layouts[rows]
        result = _Heuristic()
        _check(
            self._lib.cublasLtMatmulAlgoCheck(
                self._handle,
                self._desc,
                self._weight_layout,
                x_layout,
                d_layout,
                d_layout,
                ctypes.byref(self._algo),
                ctypes.byref(result),
            ),
            f"algorithm check at {rows} rows",
        )
        return int(result.workspaceSize)

    def __call__(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The pinned product of a call ``takes`` accepted, on the current stream."""
        rows = hidden_states.shape[0]
        x_layout, d_layout = self._layouts[rows]
        output = torch.empty((rows, POOLED), device=hidden_states.device, dtype=torch.bfloat16)
        status = self._lib.cublasLtMatmul(
            self._handle,
            self._desc,
            ctypes.byref(self._alpha),
            ctypes.c_void_p(weight.data_ptr()),
            self._weight_layout,
            ctypes.c_void_p(hidden_states.data_ptr()),
            x_layout,
            ctypes.byref(self._beta),
            ctypes.c_void_p(output.data_ptr()),
            d_layout,
            ctypes.c_void_p(output.data_ptr()),
            d_layout,
            ctypes.byref(self._algo),
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
            ctypes.c_void_p(torch.cuda.current_stream(hidden_states.device).cuda_stream),
        )
        if int(status) != 0:
            raise RuntimeError(
                f"[musespark_open] pool_gemm_pin: cublasLtMatmul status {int(status)}"
            )
        return output

    def report(self, rows: int) -> None:
        """Log the first pinned call at each row count, so a served log shows what ran."""
        if rows not in self._reported:
            self._reported.add(rows)
            logger.info(
                f"[musespark_open] pool_gemm_pin: a {rows}-row pool GEMM runs the pinned algorithm"
            )

    def takes(self, module: Linear, hidden_states: torch.Tensor, lora_params) -> bool:
        """Whether ``module(hidden_states)`` runs the pinned algorithm."""
        weight = getattr(module, "weight", None)
        fp8 = module.__dict__.get("_context_fp8")
        return (
            not torch.compiler.is_compiling()
            and not lora_params
            and weight is not None
            and getattr(module, "bias", None) is None
            and getattr(module, "lora", None) is None
            and not getattr(module, "use_cute_dsl_bf16_gemm", False)
            and not getattr(module, "use_custom_cublas_mm", False)
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.is_cuda
            and hidden_states.device.index == self.device
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] in self.rows
            and hidden_states.shape[1] == HIDDEN
            and hidden_states.is_contiguous()
            and weight.device == hidden_states.device
            and weight.dtype == torch.bfloat16
            and tuple(weight.shape) == (POOLED, HIDDEN)
            and weight.is_contiguous()
            and not (fp8 is not None and fp8.takes(hidden_states, None))
        )


class PinnedPoolLinear:
    """Mixin re-classed in front of a pool ``Linear``'s class: pinned row counts on the pin."""

    _pool_gemm_pin: Optional[PinnedPoolGemm] = None

    def forward(self, input, *, all_reduce_params=None, lora_params=None, layer_idx=None):
        pin = self._pool_gemm_pin
        if pin is not None and all_reduce_params is None and pin.takes(self, input, lora_params):
            pin.report(input.shape[0])
            return pin(input, self.weight)
        return super().forward(
            input, all_reduce_params=all_reduce_params, lora_params=lora_params, layer_idx=layer_idx
        )


@functools.lru_cache(maxsize=None)
def _pinned_class(base: type) -> type:
    """``base`` with ``PinnedPoolLinear`` in front, one class per base."""
    return type(f"{base.__name__}PoolGemmPin", (PinnedPoolLinear, base), {})


def install(model: torch.nn.Module, ranges: Sequence[Tuple[int, int]]) -> PinnedPoolGemm:
    """Re-class every MoE layer's pool ``Linear`` with ``PinnedPoolLinear`` in front; returns the
    pin, built for the current device at the multiples of 8 in ``ranges``.

    Every layer is checked and the pin built before any module is re-classed.

    Raises:
        ValueError: if a layer has no ``MuseSparkOpenMoE`` with a ``Linear`` pool, or the pin is
            refused (see the module docstring).
    """
    rows = pinned_rows(ranges)
    layers = list(model.layers)
    wrong = [
        layer.layer_idx
        for layer in layers
        if not isinstance(layer.feed_forward, MuseSparkOpenMoE)
        or not isinstance(getattr(layer.feed_forward, "pooler", None), Linear)
    ]
    if wrong:
        raise ValueError(f"[musespark_open] pool_gemm_pin: layers {wrong} have no MoE pool Linear")
    pin = PinnedPoolGemm(rows)
    for layer in layers:
        pooler = layer.feed_forward.pooler
        if not isinstance(pooler, PinnedPoolLinear):
            pooler.__class__ = _pinned_class(type(pooler))
        pooler._pool_gemm_pin = pin
    logger.info(
        f"[musespark_open] pool_gemm_pin: the pool GEMM of {len(rows)} row counts "
        f"({min(rows)}..{max(rows)}) on cuBLASLt {pin.version} algorithm {dict(PIN_CONFIG)} in "
        f"{len(layers)} MoE layers; other calls keep the built-in F.linear"
    )
    return pin
