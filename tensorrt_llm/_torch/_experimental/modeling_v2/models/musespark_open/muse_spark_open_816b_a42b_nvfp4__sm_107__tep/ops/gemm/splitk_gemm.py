# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Blackwell low-M BF16/FP16 GEMM with an in-kernel cluster split-K reduction.

Each cluster rank accumulates an exact K slice in FP32. Peers publish partials
to rank 0 through DSMEM; rank 0 reduces, casts, and stores once. The public
``A[M, K] @ B[K, N]`` problem is swapped internally, so tile dimensions below
use kernel coordinates: kernel-M carries public N and kernel-N carries public M.

This target's copy of flashinfer's ``flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py``
with four optional changes merged in. Each is a setting of ``configure``; the first, second and
fourth are this target's build and on by default, ``wide64`` is off by default; with all of them
off this is flashinfer's kernel:

* ``mailbox_v4``: the conflict-free mailbox layout, same bits.
* ``rs_split_k``: the reduce-scatter epilogue at the listed split-K sizes, same bits; with it on,
  split_k 8 is served too.
* ``wide64``: 64-wide kernel-N tiles and up to 64 public rows.
* ``oversized_smem``: the device's oversized shared memory, where the driver reports it, for
  deeper pipelines.

CuTe DSL names a kernel after its class's module path and name, so this module's kernel is
``...__sm_107__tepopsgemmsplitk_gemmSplitKDenseGemmKernel...`` where flashinfer's was
``flashinfergemmkernelsdense_bf16_gemm_sm100_splitkSplitKDenseGemmKernel``. ``serve_flashinfer_path`` makes
flashinfer's module path resolve here for the built-in low-M dispatcher.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import sys
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

#: Per-CTA SMEM capacity CuTeDSL reports for the sm_100 family this kernel is written for.
_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")

#: ``oversized_smem`` (``configure``) lets a tactic use the device's oversized shared memory.
#: The capacity is the device's CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK, read at
#: run time, served once the function's CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE is
#: ALLOW_OVERSIZED_SHARED_MEMORY; it is never below the per-block opt-in above. Deeper pipelines
#: stage more of the weight before the dependency wait; the stage count changes no arithmetic, so
#: a tactic's bits depend only on its split_k as before. A tactic that fits the opt-in launches
#: exactly as with the setting off (the opt-in request and its carveout); only a deeper one
#: requests the oversized capacity, through
#: ``_install_oversized_mode``'s CuTe DSL hook. Without the device attribute or the driver's
#: shared-memory mode the capacity stays at the opt-in. ``_resolve_smem_capacity`` sets
#: ``_SMEM_CAPACITY_BYTES`` and the stage cap ``_MAX_AB_STAGES`` from it; off they are the opt-in
#: and 12.
_OPTIN_SMEM_BYTES = _SMEM_CAPACITY_BYTES

#: The settings ``configure`` sets; all off is flashinfer's kernel. Each is described where it
#: is read: ``_mailbox_v4_enabled``, ``_rs_for``, ``_wide64_enabled``, and the oversized capacity
#: above. The defaults are this target's build: the conflict-free mailbox, the reduce-scatter
#: epilogue at split-K 8 and the oversized shared memory (each the same bits as the build it
#: replaces); ``wide64`` is a lever's.
_MAILBOX_V4 = True
_OVERSIZED_SMEM = True
_RS_SPLIT_K: Tuple[int, ...] = (8,)
_WIDE64 = False
#: Whether ``_SMEM_CAPACITY_BYTES`` and ``_MAX_AB_STAGES`` follow ``_OVERSIZED_SMEM`` yet. The
#: device is queried at the first validation, autotune list, producer key or path switch (or at
#: ``configure``), never at import.
_SMEM_RESOLVED = False


def _oversized_capacity() -> int:
    try:
        _cuda.cuInit(0)
        err, dev = _cuda.cuDeviceGet(0)
        if int(err) != 0:
            return _OPTIN_SMEM_BYTES
        attr = _cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK
        _ = _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE
        _ = _cuda.CUsharedMemoryMode.CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY
        err, value = _cuda.cuDeviceGetAttribute(attr, dev)
        return max(_OPTIN_SMEM_BYTES, int(value)) if int(err) == 0 else _OPTIN_SMEM_BYTES
    except Exception:  # noqa: BLE001 - an older driver or bindings keep the opt-in
        return _OPTIN_SMEM_BYTES


def _install_oversized_mode() -> None:
    """CuTe DSL sets a kernel's CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES from its launch request through
    ``base_dsl.runtime.cuda.set_kernel_attribute``; that attribute refuses values above the opt-in even in the
    oversized mode, so a request above the opt-in sets CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE = ALLOW_OVERSIZED instead.
    Requests within the opt-in pass through unchanged (any kernel, any module)."""
    from cutlass.base_dsl.runtime import cuda as _dsl_cuda

    if getattr(_dsl_cuda.set_kernel_attribute, "_musespark_splitk_oversized", False):
        return
    original = _dsl_cuda.set_kernel_attribute
    mode = int(_cuda.CUsharedMemoryMode.CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY)
    mode_attr = _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE

    def set_kernel_attribute(kernel, attribute, value, device=None):
        if (
            attribute == _cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            and int(value) > _OPTIN_SMEM_BYTES
        ):
            if device is None:
                result = _cuda.cuFuncSetAttribute(kernel, mode_attr, mode)
            else:
                result = _cuda.cuKernelSetAttribute(mode_attr, mode, kernel, device)
            err = result[0] if isinstance(result, tuple) else result
            if int(err) != 0:
                raise RuntimeError(
                    f"split-K oversized shared memory: CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE failed ({err})"
                )
            return None
        return original(kernel, attribute, value, device)

    set_kernel_attribute._musespark_splitk_oversized = True
    _dsl_cuda.set_kernel_attribute = set_kernel_attribute


def configure(
    *,
    mailbox_v4: bool = True,
    oversized_smem: bool = True,
    rs_split_k: Tuple[int, ...] = (8,),
    wide64: bool = False,
) -> None:
    """Set the kernel's optional changes; with all of them off this is flashinfer's kernel.

    The defaults are the module's (this target's build). ``lever_settings.apply`` sets ``wide64``
    from the model's plan before any install, so before any split-K tactic is validated, compiled
    or launched.

    Args:
        mailbox_v4: the conflict-free split-K mailbox layout (bitwise the same output).
        oversized_smem: tactics may use the device's oversized shared memory per CTA (deeper
            pipelines, the same bits); a tactic above the opt-in launches in the oversized mode.
        rs_split_k: the split-K sizes whose calls take the reduce-scatter epilogue (bitwise the
            same output); any size also lets split_k 8 validate. Empty: the epilogue is off.
        wide64: tactics with 64-wide kernel-N tiles and calls of up to 64 public rows.
    """
    global _MAILBOX_V4, _OVERSIZED_SMEM, _RS_SPLIT_K, _WIDE64, _SMEM_RESOLVED
    _MAILBOX_V4 = bool(mailbox_v4)
    _RS_SPLIT_K = tuple(int(split_k) for split_k in rs_split_k)
    _WIDE64 = bool(wide64)
    _OVERSIZED_SMEM = bool(oversized_smem)
    _SMEM_RESOLVED = False
    _resolve_smem_capacity()
    _RS_ANNOUNCED.clear()


def _resolve_smem_capacity() -> None:
    """Set ``_SMEM_CAPACITY_BYTES`` and ``_MAX_AB_STAGES`` from ``_OVERSIZED_SMEM``, once per setting.

    Called before any tactic is validated, listed, keyed or compiled, so the kernels see the
    resolved capacity; the device is queried here, never at import.
    """
    global _SMEM_CAPACITY_BYTES, _MAX_AB_STAGES, _SMEM_RESOLVED
    if _SMEM_RESOLVED:
        return
    _SMEM_RESOLVED = True
    _SMEM_CAPACITY_BYTES = _OPTIN_SMEM_BYTES
    if _OVERSIZED_SMEM:
        _SMEM_CAPACITY_BYTES = _oversized_capacity()
        if _SMEM_CAPACITY_BYTES > _OPTIN_SMEM_BYTES:
            _install_oversized_mode()
        print(
            "[musespark_open] splitk_gemm: oversized_smem: split-K tactics may use "
            f"{_SMEM_CAPACITY_BYTES} B of shared memory per CTA (opt-in {_OPTIN_SMEM_BYTES} B); "
            "deeper tactics launch in the oversized shared-memory mode",
            flush=True,
        )
    _MAX_AB_STAGES = 20 if _SMEM_CAPACITY_BYTES > _OPTIN_SMEM_BYTES else 12


#: K extent of one CTA tile.
_CTA_K = 128

#: Kernel-M tiles; 64 increases CTA count for low-M decode shapes.
_SUPPORTED_MMA_M = (64, 128)

#: Kernel-N carries public M, which is limited to 32.
_SUPPORTED_MMA_N = (8, 16, 32)

#: Physical cluster-K sizes; split 1 compiles out the DSMEM path.
_SUPPORTED_SPLIT_K = (1, 2, 3, 4)

#: Cluster-K sizes served only while the reduce-scatter epilogue is on (``rs_split_k`` not
#: empty), whose mailbox is (split_k - 1) / split_k of a tile instead of split_k - 1 tiles. Not in
#: the autotuner's space.
_RS_ONLY_SPLIT_K = (8,)


def _rs_env() -> bool:
    """Whether the reduce-scatter epilogue is on at any split-K size (validation; no log line)."""
    return bool(_RS_SPLIT_K)


def _rs_only_allows(split_k: int) -> bool:
    """Whether ``split_k`` is one of the sizes that take the reduce-scatter epilogue."""
    return split_k in _RS_SPLIT_K


#: Largest public M this low-M policy serves.
_MAX_M = 32

#: Bytes per FP32 partial exchanged through DSMEM.
_FP32_BYTES = 4

#: DSMEM mailbox base alignment, in bytes.
_MAILBOX_ALIGN_BYTES = 128

#: Size and alignment of one mbarrier, in bytes.
_MBARRIER_BYTES = 8

#: Size and alignment of the TMEM base pointer slot.
_TMEM_POINTER_BYTES = 4

#: Bytes per BF16/FP16 element.
_AB_ELEMENT_BYTES = 2

#: Alignment of the A/B shared-memory buffers.
_AB_BUFFER_ALIGN_BYTES = 1024

#: A/B pipeline stage bounds. One stage is useful only for a single K tile.
_MIN_AB_STAGES = 1
_DEFAULT_MIN_AB_STAGES = 2
_MAX_AB_STAGES = 20 if _SMEM_CAPACITY_BYTES > _OPTIN_SMEM_BYTES else 12


@dataclasses.dataclass(frozen=True)
class SplitKTactic:
    """One specialization; mma_m carries public N and mma_n carries public M."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    tactic: SplitKTactic,
    ab_stages: int,
) -> int:
    """Mirror the device allocator's shared-memory layout."""
    cursor = (
        _align_up(
            tactic.mma_m * _CTA_K * _AB_ELEMENT_BYTES * ab_stages,
            _AB_BUFFER_ALIGN_BYTES,
        )
        + tactic.mma_n * _CTA_K * _AB_ELEMENT_BYTES * ab_stages
    )

    cursor = _align_up(cursor, _MBARRIER_BYTES)
    cursor += 2 * ab_stages * _MBARRIER_BYTES
    cursor += 3 * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES)
    cursor += _TMEM_POINTER_BYTES

    if tactic.split_k == 1:
        return cursor

    mailbox_bytes = (tactic.split_k - 1) * tactic.mma_m * tactic.mma_n * _FP32_BYTES
    if _rs_env() and _rs_only_allows(tactic.split_k):
        # Reduce-scatter: a rank receives one thread group's values from each of the other ranks
        # (only at the ``rs_split_k`` sizes)
        mailbox_bytes //= tactic.split_k
    return (
        _align_up(
            _align_up(cursor, _MAILBOX_ALIGN_BYTES) + mailbox_bytes,
            _MBARRIER_BYTES,
        )
        + _MBARRIER_BYTES
    )


def _max_ab_stages_for(
    tactic: SplitKTactic,
    smem_capacity: int,
) -> int:
    _resolve_smem_capacity()
    return next(
        (
            stages
            for stages in range(_MAX_AB_STAGES, -1, -1)
            if _smem_bytes(tactic, stages) <= smem_capacity
        ),
        0,
    )


def validate_tactic(
    tactic: SplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int | None = None,
) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)`` (``smem_capacity`` None: the configured)."""
    _resolve_smem_capacity()
    if smem_capacity is None:
        smem_capacity = _SMEM_CAPACITY_BYTES
    if tactic.mma_m not in _SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    if tactic.mma_n not in _SUPPORTED_MMA_N and not (tactic.mma_n == 64 and _wide64_enabled()):
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    if tactic.split_k not in _SUPPORTED_SPLIT_K and not (
        tactic.split_k in _RS_ONLY_SPLIT_K and _rs_env()
    ):
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], got {tactic.ab_stages}"
        )
    max_m = _WIDE64_MAX_M if _wide64_enabled() else _MAX_M
    if not 1 <= m <= max_m:
        raise ValueError(f"this low-M policy requires 1 <= M <= {max_m}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % _CTA_K or (k // _CTA_K) % tactic.split_k:
        raise ValueError(
            f"K={k} with CTA_K={_CTA_K} does not divide evenly across split_k={tactic.split_k}"
        )
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages)
    if smem_bytes > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {smem_bytes} B of shared memory but only "
            f"{smem_capacity} B are available; max ab_stages is "
            f"{_max_ab_stages_for(tactic, smem_capacity)}"
        )


def autotune_tactics(
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int | None = None,
) -> list[SplitKTactic]:
    """Return valid tactics in the shape-specific stage window (``smem_capacity`` None: the
    configured)."""
    _resolve_smem_capacity()
    if smem_capacity is None:
        smem_capacity = _SMEM_CAPACITY_BYTES
    tactics: list[SplitKTactic] = []
    for mma_m in _SUPPORTED_MMA_M:
        for mma_n in _SUPPORTED_MMA_N:
            for split_k in _SUPPORTED_SPLIT_K:
                min_stages = _MIN_AB_STAGES if k == _CTA_K else _DEFAULT_MIN_AB_STAGES
                base = SplitKTactic(mma_m, mma_n, split_k, min_stages)
                try:
                    validate_tactic(base, m, n, k, smem_capacity=smem_capacity)
                except ValueError:
                    continue
                max_stages = _max_ab_stages_for(base, smem_capacity)
                # Short K favors shallow pipelines; long K stays near the cap.
                tactics.extend(
                    dataclasses.replace(base, ab_stages=ab_stages)
                    for ab_stages in (
                        range(min_stages, min(max_stages, 6) + 1)
                        if k <= 4 * _CTA_K
                        else range(
                            min(max(5, max_stages - 2), max_stages),
                            max_stages + 1,
                        )
                    )
                )
    return tactics


def default_tactic(m: int, n: int, k: int) -> SplitKTactic:
    """Choose the default occupancy-oriented tactic."""
    _resolve_smem_capacity()
    if n <= 512:
        mma_m = 64
        mma_n = 16 if n == 512 and m > 24 else 8
        requested_split = 4
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        if n <= 3072:
            mma_m = 128 if m <= 16 else 64
            requested_split = 4 if m <= 16 else 2
        elif n < 8192:
            mma_m = 64
            requested_split = 2
        else:
            mma_m = 128 if k <= 1024 and m <= 24 else 64
            requested_split = 1

    if k <= 4 * _CTA_K:
        requested_split = 1
    split_k = next(
        split_k
        for split_k in reversed(_SUPPORTED_SPLIT_K)
        if split_k <= requested_split and (k // _CTA_K) % split_k == 0
    )
    tactic = SplitKTactic(mma_m, mma_n, split_k, _MIN_AB_STAGES)
    max_stages = _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES)
    tactic = dataclasses.replace(
        tactic,
        ab_stages=(
            _MIN_AB_STAGES
            if k == _CTA_K
            else _DEFAULT_MIN_AB_STAGES
            if k <= 2 * _CTA_K and m > 8
            else min(max_stages, 6)
            if k <= 4 * _CTA_K
            else max_stages
        ),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


__all__ = [
    "SplitKTactic",
    "autotune_tactics",
    "configure",
    "default_tactic",
    "producer_key",
    "run_splitk_dense",
    "serve_flashinfer_path",
]


# ---- device code begin: the kernel and its compilation. ``producer_key`` fingerprints the code lines between this
# ---- marker and the end marker (comment and blank lines excluded); an edit there needs a new producer census.
@dsl_user_op
def _map_shared_rank(
    smem_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> "Int32":
    """Map an SMEM pointer into a peer CTA's address space."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0,
    value1,
    value2,
    value3,
    smem_ptr: "cute.Pointer",
    mbar_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> None:
    """Publish four FP32 partials into a peer's SMEM, crediting 16 bytes."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


#: Rank that gathers partials and stores the output.
OWNER_RANK = 0


class SplitKDenseGemmKernel:
    """Standalone BF16/FP16 GEMM with a cluster-local split-K reduction."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        mailbox_v4: bool = False,
        rs: bool = False,
    ) -> None:
        # ``mailbox_v4`` lays the split-K mailbox out as
        # [peer][value group of 4][epilogue thread][4] instead of [peer][epilogue thread][value].
        self.mailbox_v4 = mailbox_v4
        # ``rs``: the reduce-scatter epilogue (``configure``'s ``rs_split_k``).
        self.rs = rs and tactic.split_k > 1
        self.acc_dtype = cutlass.Float32
        self.cta_m = tactic.mma_m
        self.cta_n = tactic.mma_n
        self.cta_k = _CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.use_pdl = use_pdl
        self.has_bias = has_bias

        self.threads_per_cta = 256
        self.epilog_threads = 128
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)
        # The opt-in request (and its carveout) unless this tactic needs the oversized capacity.
        self.launch_smem = (
            _SMEM_CAPACITY_BYTES
            if _smem_bytes(tactic, tactic.ab_stages) > _OPTIN_SMEM_BYTES
            else _OPTIN_SMEM_BYTES
        )

        values_per_thread = (tactic.mma_m * tactic.mma_n) // self.epilog_threads
        if values_per_thread % 4:
            raise ValueError(
                f"CTA tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                f"{values_per_thread} "
                "values per epilogue thread; remote stores require a multiple of 4"
            )
        self.mailbox_elements = (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES
        if self.rs:
            # Reduce-scatter: epilogue thread group g (epilog_threads / split_k threads) of every CTA belongs
            # to cluster rank g, which receives that group's values from the other split_k - 1 ranks.
            if self.epilog_threads % tactic.split_k:
                raise ValueError(
                    f"split_k {tactic.split_k} does not divide {self.epilog_threads} epilogue threads"
                )
            self.rs_group = self.epilog_threads // tactic.split_k
            self.rs_mailbox_elements = (tactic.split_k - 1) * self.rs_group * values_per_thread
            self.rs_expected_bytes = self.rs_mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.launch_smem),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.rs_mailbox_elements if self.rs else self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(is_a and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not is_a and self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)
        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(
        self,
        bar_empty,
        empty_phase: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
    ):
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            if self.rs or split_rank == OWNER_RANK:
                cute.arch.mbarrier_wait(bar_tma_epilog, 0)
                cute_ext.partition_and_copy(
                    cute.make_tiled_copy_D(
                        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                        tiled_copy_t2r,
                    ).get_slice(epi_tid),
                    cute.flat_divide(gBias_tile, epi_tile)[None, None, 0, 0],
                    rBias,
                )
                rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.rs):
            # Reduce-scatter: thread group g = epi_tid // rs_group of every CTA sends its FP32
            # partial to cluster rank g (unless it is rank g); rank g's group g sums the split_k
            # partials of its values in cluster-rank order (rank 0's, then rank 1's, ...: the owner
            # path's additions in the owner path's order, so the same bits), adds the bias, rounds
            # once and stores them. Every rank receives (split_k - 1) x rs_group x
            # values_per_thread FP32 instead of the owner receiving split_k - 1 tiles.
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            group_values = cutlass.const_expr(self.rs_group * values_per_thread)
            dest = epi_tid // self.rs_group
            lt = epi_tid % self.rs_group
            if dest != split_rank:
                # this sender's slot in rank dest's mailbox: ranks below dest keep their index, ranks above shift down
                j = split_rank - cutlass.select_(split_rank > dest, Int32(1), Int32(0))
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator + j * group_values + slot,
                        bar_reduce,
                        dest,
                    )
            else:
                if lt == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.rs_expected_bytes)
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for value_idx in cutlass.range_constexpr(values_per_thread):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4 + value_idx % 4
                    own = rAcc[value_idx]
                    acc = own
                    for r in cutlass.range_constexpr(self.split_k):
                        # rank r's partial: this CTA's own, or its slot in the mailbox (clamped when unused)
                        jr = cutlass.min(
                            Int32(r) - cutlass.select_(Int32(r) > split_rank, Int32(1), Int32(0)),
                            Int32(self.split_k - 2),
                        )
                        term = cutlass.select_(
                            Int32(r) == split_rank, own, mailbox[jr * group_values + slot]
                        )
                        if cutlass.const_expr(r == 0):
                            acc = term
                        else:
                            acc = acc + term
                    rAcc[value_idx] = acc
                if cutlass.const_expr(self.has_bias):
                    rAcc.store(rAcc.load() + rBiasAcc.load())
                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1 and not self.rs):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    if cutlass.const_expr(self.mailbox_v4):
                        # A warp's 16-byte stores of one value group are contiguous (512 bytes).
                        slot = (value_idx // 4) * (self.epilog_threads * 4) + epi_tid * 4
                    else:
                        slot = epi_tid * values_per_thread + value_idx
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator + (split_rank - Int32(1)) * values_per_peer + slot,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        if cutlass.const_expr(self.mailbox_v4):
                            slot = (
                                (value_idx // 4) * (self.epilog_threads * 4)
                                + epi_tid * 4
                                + value_idx % 4
                            )
                        else:
                            slot = epi_tid * values_per_thread + value_idx
                        # Same additions in the same order (own, then peers in rank order).
                        rAcc[value_idx] = rAcc[value_idx] + mailbox[peer * values_per_peer + slot]

        if cutlass.const_expr(not self.rs):
            if split_rank == OWNER_RANK:
                if cutlass.const_expr(self.has_bias):
                    rAcc.store(rAcc.load() + rBiasAcc.load())

                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


import torch as _torch

_SUPPORTED_TORCH_DTYPES = (_torch.bfloat16, _torch.float16)


@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        stream,
    )


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    bias: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0])),
        stream,
    )


def _from_dlpack_dynamic(tensor, leading_dim: int, assumed_align: int = 32):
    return from_dlpack(tensor, assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _detect_leading_dim(tensor: _torch.Tensor) -> int:
    # Ignore synthetic batch stride, including 1x1.
    for dim, stride in enumerate(tensor.stride()[1:], start=1):
        if stride == 1:
            return dim
    raise ValueError("tensor has no stride-1 dimension")


def _make_layout_tensor(
    shape: tuple[int, ...], dtype: _torch.dtype, leading_dim: int
) -> _torch.Tensor:
    permutation = [dim for dim in range(len(shape)) if dim != leading_dim] + [leading_dim]
    return _torch.empty(
        tuple(shape[dim] for dim in permutation), dtype=dtype, device="cuda"
    ).permute([permutation.index(dim) for dim in range(len(shape))])


def _make_compile_repr_tensors(
    dtype: _torch.dtype,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    m, n, k, batch = 64, 8, _CTA_K, 1
    tensors = tuple(
        _from_dlpack_dynamic(_make_layout_tensor(shape, dtype, leading_dim), leading_dim)
        for shape, leading_dim in (
            ((batch, n, k), a_leading),
            ((batch, k, m), b_leading),
            ((batch, n, m), c_leading),
        )
    )
    if not has_bias:
        return (*tensors, None)
    return (
        *tensors,
        _from_dlpack_dynamic(
            _torch.empty((n,), dtype=dtype, device="cuda").as_strided(
                size=(batch, n, m), stride=(0, 1, 0)
            ),
            1,
            2,
        ),
    )


def _to_cute_swap(a, b, out, bias):
    a_swap = b.unsqueeze(0).transpose(-2, -1)
    b_swap = a.unsqueeze(0).transpose(-2, -1)
    c_swap = out.unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(_detect_leading_dim(tensor) for tensor in (a_swap, b_swap, c_swap))
    cute_tensors = tuple(
        _from_dlpack_dynamic(tensor, leading_dim)
        for tensor, leading_dim in zip((a_swap, b_swap, c_swap), leading_dims, strict=True)
    )
    if bias is None:
        return (*cute_tensors, None, leading_dims)
    return (
        *cute_tensors,
        _from_dlpack_dynamic(
            bias.as_strided(size=(1, c_swap.shape[1], c_swap.shape[2]), stride=(0, 1, 0)),
            1,
            2,
        ),
        leading_dims,
    )


@functools.cache
def _get_compiled_splitk_kernel(
    dtype,
    tactic: SplitKTactic,
    use_pdl: bool,
    has_bias: bool,
    leading_dims: tuple[int, int, int],
    mailbox_v4: bool = False,
    rs: bool = False,
):
    if dtype not in _SUPPORTED_TORCH_DTYPES:
        raise ValueError(f"split-K dense GEMM supports {_SUPPORTED_TORCH_DTYPES}; got {dtype}")

    kernel = SplitKDenseGemmKernel(
        tactic=tactic,
        use_pdl=use_pdl,
        has_bias=has_bias,
        mailbox_v4=mailbox_v4,
        rs=rs,
    )
    compile_tensors = _make_compile_repr_tensors(dtype, has_bias, *leading_dims)
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    if has_bias:
        compiled = cute_ext.compile(_bmm_bias, kernel, *compile_tensors, stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, kernel, *compile_tensors[:3], stream)
    return compiled


# ---- device code end


def _validate_runtime_tensors(a, b, bias, out) -> tuple[int, int, int]:
    tensors = (a, b, out) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, out, and bias must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("split-K dense GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype not in _SUPPORTED_TORCH_DTYPES or any(tensor.dtype != a.dtype for tensor in tensors):
        raise ValueError("a, b, out, and bias must share BF16 or FP16 dtype")
    if any(not (tensor.is_contiguous() or tensor.t().is_contiguous()) for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be dense row-major or column-major matrices")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != n or not bias.is_contiguous()):
        raise ValueError(
            f"bias must be contiguous with shape {(n,)}, "
            f"got shape {tuple(bias.shape)} and stride {bias.stride()}"
        )

    return m, n, k


#: ``mailbox_v4`` (``configure``) compiles the split-K kernels with the conflict-free mailbox
#: layout. The original layout strides the 128 epilogue threads by ``values_per_thread`` FP32
#: words, a ``values_per_thread``-way shared-memory bank conflict for the peers' ``st.async`` and
#: the owner's reads (16-way for a 64x32 tile: ~1.06 us of a 5-6 us launch); grouped by 4 values
#: the stores are conflict-free and the reads 4-way. Every output element is still own + peer 1 +
#: ... in cluster-rank order, rounded once: bitwise the same output. Read per call (the
#: compiled-kernel cache keys on it), so a process can hold both builds; off is the stock kernel,
#: call path and cache key unchanged.
def _mailbox_v4_enabled() -> bool:
    return _MAILBOX_V4


#: ``rs_split_k`` (``configure``) compiles the split-K calls of the listed split-K sizes (> 1)
#: with a reduce-scatter epilogue: each cluster rank reduces and stores one group of epilogue
#: threads' values instead of rank 0 reducing the whole tile, so a CTA takes (split_k - 1) /
#: split_k of a tile in partials instead of split_k - 1 tiles. Every output element is still rank
#: 0's partial + rank 1's + ... in cluster-rank order, plus the bias, rounded once: bitwise the
#: same output. ``(8,)`` takes the pool GEMMs at split-K 8, which validate only with it on. Every
#: other split-K call takes the path it takes with the epilogue off (the conflict-free mailbox
#: under ``mailbox_v4``, else the stock one): bitwise the same output either way, so only the
#: epilogue's cost moves. Read per call (the compiled-kernel cache keys on it); empty is the stock
#: epilogue.
_RS_ANNOUNCED = []


def _rs_enabled() -> bool:
    on = bool(_RS_SPLIT_K)
    if on and not _RS_ANNOUNCED:
        _RS_ANNOUNCED.append(True)
        import sys as _sys

        sizes = sorted(set(_RS_SPLIT_K))
        print(
            f"[musespark_open] dense split-K: reduce-scatter epilogue at split-K {sizes}",
            file=_sys.stderr,
            flush=True,
        )
    return on


def _rs_for(split_k: int) -> bool:
    """Whether a split-K call of ``split_k`` takes the reduce-scatter epilogue."""
    if split_k <= 1 or not _rs_enabled():
        return False
    return split_k in _RS_SPLIT_K


#: ``wide64`` (``configure``) lets a caller name tactics with 64-wide kernel-N tiles (``mma_n``
#: 64) and serve up to 64 public rows (33..64-row TP8 decode steps, 5..8 requests at K = 7), as
#: the dense dispatcher's ``tep8_wide64`` does for QKV and o_proj. Only validation widens:
#: ``autotune_tactics`` and ``default_tactic`` keep the stock lists, so the built-in low-M
#: autotuner races what it raced before. Calls above 32 rows always take the conflict-free mailbox
#: (same bits, as ``mailbox_v4``): with 32 values per epilogue thread the original layout is a
#: 32-way bank conflict. A row's bits depend only on ``split_k`` (the K slices and their order),
#: not on the tile widths or the row count.
_WIDE64_MAX_M = 64


def _wide64_enabled() -> bool:
    return _WIDE64


def _launch_options(rows: int, split_k: int) -> dict:
    """The epilogue options a call of ``rows`` public rows at ``split_k`` compiles with: the
    reduce-scatter epilogue (at the ``rs_split_k`` sizes), else the conflict-free mailbox
    (``mailbox_v4``, or ``wide64``'s calls above 32 rows), else flashinfer's kernel.
    ``run_splitk_dense`` launches with them and ``producer_key`` keys on them."""
    if _rs_for(split_k):
        return {"mailbox_v4": _mailbox_v4_enabled(), "rs": True}
    if _mailbox_v4_enabled() or (_wide64_enabled() and rows > _MAX_M):
        return {"mailbox_v4": True}
    return {}


def run_splitk_dense(
    a,
    b,
    bias,
    out,
    pdl: bool,
    tactic: SplitKTactic,
):
    """Run ``A[M,K] @ B[K,N]`` with the ``mm_bf16`` layouts."""
    validate_tactic(tactic, *_validate_runtime_tensors(a, b, bias, out))
    has_bias = bias is not None
    cute_tensors = _to_cute_swap(a, b, out, bias)
    compiled = _get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=pdl,
        has_bias=has_bias,
        leading_dims=cute_tensors[4],
        **_launch_options(a.shape[0], tactic.split_k),
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    if has_bias:
        compiled(*cute_tensors[:4], stream)
    else:
        compiled(*cute_tensors[:3], stream)
    return out


@functools.lru_cache(maxsize=None)
def _device_code_md5() -> str | None:
    """md5 of the code lines between the device-code markers of this file (comment and blank lines excluded)."""
    try:
        with open(__file__, encoding="utf-8") as f:
            lines = f.read().splitlines()
        begin = next(
            i for i, line in enumerate(lines) if line.startswith("# ---- device code begin")
        )
        end = next(i for i, line in enumerate(lines) if line.startswith("# ---- device code end"))
    except (OSError, StopIteration):
        return None
    code = [line.rstrip() for line in lines[begin + 1 : end]]
    code = [line for line in code if line and not line.lstrip().startswith("#")]
    # A fingerprint of the kernel source, not a security hash.
    return hashlib.md5("\n".join(code).encode(), usedforsecurity=False).hexdigest()


def producer_key(rows: int, tactic: SplitKTactic, use_pdl: bool = True) -> tuple:
    """What a bias-free call of ``rows`` public rows at ``tactic`` compiles and launches, as a census key.

    ``(device-code md5, tactic, epilogue, mailbox layout, shared-memory request, PDL)``: the device
    code is the region between the markers above (``_device_code_md5``), the rest is what
    ``run_splitk_dense`` compiles this call with under this module's settings (``configure``,
    ``_launch_options``; the request from the kernel's own ``launch_smem``). A producer census
    (``projection_publish.CERTIFIED_SPLITK_PRODUCERS_TP8``) certified on this key holds through
    host-side edits elsewhere in this file (settings accessors, tactic tables, comments); a change
    to the kernel's code or to what a call compiles with is a new key.
    """
    _resolve_smem_capacity()
    kernel = SplitKDenseGemmKernel(
        tactic=tactic, use_pdl=use_pdl, has_bias=False, **_launch_options(rows, tactic.split_k)
    )
    return (
        _device_code_md5(),
        dataclasses.astuple(tactic),
        "reduce-scatter" if kernel.rs else "owner",
        "v4" if kernel.mailbox_v4 else "flat",
        "opt-in" if kernel.launch_smem == _OPTIN_SMEM_BYTES else int(kernel.launch_smem),
        bool(kernel.use_pdl),
    )


#: flashinfer-python's module path of this kernel (0.6.18 ships the stock copy there).
FLASHINFER_MODULE = "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk"


def serve_flashinfer_path() -> bool:
    """Make flashinfer's split-K module path resolve to this module for the rest of the process,
    when a setting is on.

    The built-in low-M dispatcher (``tensorrt_llm/_torch/modules/low_m_gemm.py``,
    ``TRTLLM_LOW_M_GEMM_BACKEND=auto``) races flashinfer's split-K kernel (``autotune_tactics``,
    ``default_tactic``, ``SplitKTactic``, ``run_splitk_dense``, imported by path at each call)
    against cuBLAS and the direct kernel on every BF16 ``Linear`` call of at most 32 rows that no
    table of this target takes: the LM head among them. From here on that race runs this file's
    kernels and tactic lists under its settings, with no flashinfer patch. Code that bound
    flashinfer's functions before this call keeps them. With every setting off this module is
    flashinfer's kernel and flashinfer's module is left alone. Returns whether the path resolves
    here.
    """
    this = sys.modules[__name__]
    if sys.modules.get(FLASHINFER_MODULE) is this:
        return True
    _resolve_smem_capacity()
    gated = (
        _mailbox_v4_enabled()
        or _rs_env()
        or _wide64_enabled()
        or _SMEM_CAPACITY_BYTES > _OPTIN_SMEM_BYTES
    )
    if not gated:
        return False
    try:
        import flashinfer.gemm.kernels as _flashinfer_kernels
    except Exception:  # noqa: BLE001 - no flashinfer: nothing imports its path
        return False
    sys.modules[FLASHINFER_MODULE] = this
    _flashinfer_kernels.dense_bf16_gemm_sm100_splitk = this
    print(
        f"[musespark_open] splitk_gemm: {FLASHINFER_MODULE} resolves to the target's split-K "
        f"kernel ({__name__}) in this process: the built-in low-M dispatcher races its configured "
        "kernels",
        flush=True,
    )
    return True
