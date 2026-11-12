# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 ``y = x @ weight.T`` for at most 32 rows on the low-M split-K kernel, at a fixed tactic.

The built-in low-M dispatcher (``_torch/modules/low_m_gemm.py``) races the
flashinfer CuTe-DSL split-K and direct kernels against cuBLAS per shape on
each rank and caches the winner per rank; ranks can and do settle on
different kernels for one shape, and plain ``nn.Linear`` / ``F.linear`` call
sites never reach it. This entry is the split-K kernel the dispatcher would
launch, with the tactic chosen by the caller: one tactic per shape, the same
on every rank, measured once per target GPU and TP size (``SPLITK_TACTICS_BY_SM_TP``).
The kernel is this target's copy (``ops/gemm/splitk_gemm.py``) of flashinfer's
``dense_bf16_gemm_sm100_splitk``.

The kernel accumulates in fp32 and rounds once; its K-split differs from
cuBLAS's, so results differ from ``F.linear`` in the last bf16 place on a few
elements -- the same distance from the fp64 product as cuBLAS.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    OpWrapper,
    true_fp32_matmul,
)

#: The split-K tactic ``(mma_m, mma_n, split_k, ab_stages)`` per ``(M, K, N)`` of a
#: tep8 drafter (two trays) on sm103, chosen on GB300 with cold weights and PDL on;
#: each 8-row one beats cuBLAS at its shape. Shapes not listed are not taken by this op's
#: callers. The blocks of 2..4 requests (16, 24 and 32 rows) are listed for the three shapes
#: whose cuBLAS pick there is a split-K GEMM plus a reduce kernel (the convolution coefficient,
#: qkv / context K/V and gate_up projections); o_proj and down_proj stay on cuBLAS's single-pass
#: GEMM, so ``drafter_pdl_chain``'s entry triggers, which need all four successors listed at a
#: width (``entry_rows``), keep acting on the one-request block only. The convolution
#: coefficient rows are the tep4 table's, measured on GB300 at the same replicated shape; the
#: qkv rows keep the 8-row ``split_k`` 4 (80 CTAs; ``mma_n`` 16 holds two requests' rows, 32
#: three or four, at the 8 stages ``split_k`` 4 leaves it); the gate_up rows keep the 8-row
#: grid (``mma_m`` 128, ``split_k`` 2, 112 CTAs) at the deepest pipeline the wider ``mma_n``
#: leaves. Chosen from the shapes, not from a per-call sweep at tep8, so the table is in
#: ``INSTALL_CHECKED``: a row flashinfer refuses stops the drafter's install instead of raising
#: in the engine's warmup.
SPLITK_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # DFlash 2 convolution coefficient projection (replicated).
    (8, 8192, 2048): (64, 8, 4, 10),
    # drafter qkv_proj per rank, and the fused context K/V projection.
    (8, 8192, 1280): (64, 16, 4, 9),
    # drafter o_proj per rank.
    (8, 1024, 8192): (128, 8, 1, 6),
    # drafter down_proj per rank.
    (8, 3584, 8192): (64, 8, 1, 12),
    # drafter gate_up_proj per rank.
    (8, 8192, 7168): (128, 8, 2, 6),
    # DFlash fc, one TP8 column shard (``drafter_fc_shard``).
    (8, 40960, 1024): (64, 8, 4, 12),
    # Blocks of 2..4 requests (see above).
    (16, 8192, 2048): (64, 16, 4, 8),
    (24, 8192, 2048): (64, 16, 2, 11),
    (32, 8192, 2048): (64, 16, 2, 11),
    (16, 8192, 1280): (64, 16, 4, 9),
    (24, 8192, 1280): (64, 32, 4, 8),
    (32, 8192, 1280): (64, 32, 4, 8),
    (16, 8192, 7168): (128, 16, 2, 6),
    (24, 8192, 7168): (128, 32, 2, 5),
    (32, 8192, 7168): (128, 32, 2, 5),
}

#: The same for a tep4 drafter (one tray) on sm103: every TP-sharded projection is
#: twice the tep8 one along its sharded dimension; the replicated convolution
#: coefficient projection is the tep8 one. The fastest split-K tactic of each shape in
#: chains of cold weights with PDL on; it beats cuBLAS on the qkv, gate_up and
#: convolution shapes, ties it on down_proj and trails it slightly on o_proj in
#: isolation, where ``drafter_pdl_chain``'s early weight load is what it is for. Besides the
#: one-request block (8 rows) the table lists the blocks of 2..4 requests (16, 24 and 32
#: rows), each the fastest split-K tactic at that row count on GB300's 152 SMs over cold
#: weights with PDL on; every one is a single wave of at most 128 CTAs.
SPLITK_TACTICS_TP4: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # DFlash 2 convolution coefficient projection (replicated).
    (8, 8192, 2048): (64, 8, 4, 10),
    # drafter qkv_proj per rank, and the fused context K/V projection.
    (8, 8192, 2560): (64, 8, 2, 11),
    # drafter o_proj per rank.
    (8, 2048, 8192): (64, 8, 1, 11),
    # drafter down_proj per rank.
    (8, 7168, 8192): (64, 8, 1, 11),
    # drafter gate_up_proj per rank.
    (8, 8192, 14336): (128, 8, 1, 6),
    (16, 8192, 2048): (64, 16, 4, 8),
    (16, 8192, 2560): (64, 16, 2, 10),
    (16, 2048, 8192): (64, 16, 1, 10),
    (16, 7168, 8192): (64, 16, 1, 10),
    (16, 8192, 14336): (128, 16, 1, 6),
    (24, 8192, 2048): (64, 16, 2, 11),
    (24, 8192, 2560): (64, 32, 2, 9),
    (24, 2048, 8192): (64, 32, 1, 9),
    (24, 7168, 8192): (64, 32, 1, 9),
    (24, 8192, 14336): (128, 32, 1, 5),
    (32, 8192, 2048): (64, 16, 2, 11),
    (32, 8192, 2560): (64, 32, 2, 9),
    (32, 2048, 8192): (64, 32, 1, 9),
    (32, 7168, 8192): (64, 32, 1, 9),
    (32, 8192, 14336): (128, 32, 1, 5),
}


#: The measured table of each ``(compute capability, drafter TP size)``.
class OnBatchSplitK(tuple):
    """A tactic ``(mma_m, mma_n, split_k, ab_stages)`` run on ``batch_splitk_gemm``'s copy of the
    split-K kernel instead of flashinfer's.

    The copy has the same tiles and the same summation (each output element the FP32 sum of the
    ``split_k`` K-slice partials in cluster-rank order, rounded once), so its bits are flashinfer's
    at the same ``split_k``; its epilogue reduce-scatters the partials over the cluster instead of
    sending every peer's whole tile to one owner, which leaves room for a deeper ring. A plain
    4-tuple everywhere else (``len``, unpacking, equality).
    """

    __slots__ = ()

    def __new__(cls, *fields: int) -> "OnBatchSplitK":
        return super().__new__(cls, fields)

    def __repr__(self) -> str:
        return f"bsplitk{tuple(self)!r}"


#: ``drafter_multi_request_cells`` (sm103 TP8 only): the blocks of 2..4 requests (16, 24 and 32
#: rows) run these cells in place of the TP8 table's rows chosen from the shapes. Each qkv /
#: gate_up / convolution coefficient cell keeps its row's ``split_k`` (bitwise the table's row).
#: The o_proj and down_proj cells, which the low-M dispatcher's per-rank race served (cuBLAS on
#: the measured ranks), are single K-slice GEMMs (``split_k`` 1). With every successor listed at a
#: width, ``drafter_pdl_chain``'s entry triggers act on these widths too.
MULTI_REQUEST_CELLS_TP8: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # qkv_proj / context K/V: 16 rows one stage deeper; 24 / 32 rows on the reduce-scatter copy,
    # one more stage than flashinfer fits.
    (16, 8192, 1280): (64, 16, 4, 10),
    (24, 8192, 1280): OnBatchSplitK(64, 32, 4, 9),
    (32, 8192, 1280): OnBatchSplitK(64, 32, 4, 9),
    # Convolution coefficient projection: 16 rows one stage deeper (``ConvEpilogue`` fits its
    # stages beside its staging tile). The 24 / 32-row rows stay: ``mma_n`` 32 there was a large
    # loss with the convolution epilogue.
    (16, 8192, 2048): (64, 16, 4, 10),
    # gate_up_proj: the same tiles on the reduce-scatter copy.
    (16, 8192, 7168): OnBatchSplitK(128, 16, 2, 6),
    (24, 8192, 7168): OnBatchSplitK(128, 32, 2, 5),
    (32, 8192, 7168): OnBatchSplitK(128, 32, 2, 5),
    # o_proj per rank: one K slice, the fastest tile per row count in the chain.
    (16, 1024, 8192): OnBatchSplitK(128, 16, 1, 6),
    (24, 1024, 8192): OnBatchSplitK(128, 32, 1, 5),
    (32, 1024, 8192): (64, 32, 1, 8),
    # down_proj per rank: one K slice, the fastest tile per row count in the chain.
    (16, 3584, 8192): (64, 16, 1, 11),
    (24, 3584, 8192): (64, 32, 1, 9),
    (32, 3584, 8192): OnBatchSplitK(64, 32, 1, 9),
}

#: Whether ``splitk_tactics`` merges ``MULTI_REQUEST_CELLS_TP8`` into the sm103 TP8 table:
#: ``drafter_multi_request_cells``, pushed here by ``lever_settings.apply`` (default off).
_MULTI_REQUEST_CELLS = False


def configure(*, multi_request_cells: bool) -> None:
    """Set this module's lever-backed settings (``lever_settings.apply``), once per process."""
    global _MULTI_REQUEST_CELLS
    _MULTI_REQUEST_CELLS = bool(multi_request_cells)


@functools.lru_cache(maxsize=None)
def _with_multi_request_cells(key: tuple) -> dict:
    """The table of ``key`` with ``MULTI_REQUEST_CELLS_TP8`` merged in; built (and logged) once per
    process."""
    from tensorrt_llm.logger import logger

    logger.info(
        "[musespark_open] drafter_multi_request_cells: drafter blocks of 16, 24 and 32 rows run "
        f"{len(MULTI_REQUEST_CELLS_TP8)} measured TP8 cells (qkv / gate_up / coefficient rows at "
        "their split_k, o_proj / down_proj on one K slice; "
        f"{sum(isinstance(v, OnBatchSplitK) for v in MULTI_REQUEST_CELLS_TP8.values())} on the "
        "reduce-scatter split-K copy)"
    )
    return {**SPLITK_TACTICS_BY_SM_TP[key], **MULTI_REQUEST_CELLS_TP8}


SPLITK_TACTICS_BY_SM_TP: dict[
    tuple[tuple[int, int], int], dict[tuple[int, int, int], tuple[int, int, int, int]]
] = {
    ((10, 3), 8): SPLITK_TACTICS,
    ((10, 3), 4): SPLITK_TACTICS_TP4,
}

#: The ``(compute capability, drafter TP size)`` tables whose every row ``spec_worker.install``
#: checks with flashinfer's own validation before the drafter serves (``refused_rows``), as
#: ``dense_gemms.install`` checks its cells: sm103 TP8, whose 16..32-row rows were chosen from
#: the shapes, not from a sweep. Every row of the
#: other tables ran in the sweep that chose it; their installs are unchanged.
INSTALL_CHECKED: frozenset[tuple[tuple[int, int], int]] = frozenset({((10, 3), 8)})

#: The kernel's row limit (the dispatcher's ``_MAX_M``).
#: Shapes no table lists still run the built-in GEMM.
MAX_M = 32


_DEVICE_CAPABILITY: tuple[int, int] | None = None


def device_capability() -> tuple[int, int]:
    """This process's CUDA device compute capability, read once.

    On a host without CUDA, where no kernel launches, the target's own sm103.
    """
    global _DEVICE_CAPABILITY
    if _DEVICE_CAPABILITY is None:
        _DEVICE_CAPABILITY = (
            tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (10, 3)
        )
    return _DEVICE_CAPABILITY


def splitk_tactics(
    tp_size: int = 8, capability: tuple[int, int] | None = None
) -> dict[tuple[int, int, int], tuple[int, int, int, int]]:
    """The tactic table of a drafter on a TP group of ``tp_size`` on ``capability``.

    ``capability`` defaults to this process's device (``device_capability``).

    Raises:
        ValueError: if ``SPLITK_TACTICS_BY_SM_TP`` has no table for the pair.
    """
    key = (tuple(device_capability() if capability is None else capability), tp_size)
    table = SPLITK_TACTICS_BY_SM_TP.get(key)
    if table is None:
        raise ValueError(
            f"fixed_low_m_gemm: no split-K tactics for SM {key[0]} at TP {key[1]} "
            f"(supported: {sorted(SPLITK_TACTICS_BY_SM_TP)})"
        )
    if key == ((10, 3), 8) and _MULTI_REQUEST_CELLS:
        return _with_multi_request_cells(key)
    return table


def tactic_for(m: int, k: int, n: int, tactics: dict | None = None):
    """The fixed tactic for this shape in ``tactics`` (a ``splitk_tactics`` table; the
    device's TP8 one when None), or None when the shape is not covered."""
    return (splitk_tactics() if tactics is None else tactics).get((m, k, n))


@functools.lru_cache(maxsize=None)
def tactic_error(m: int, k: int, n: int, tactic: tuple[int, int, int, int]) -> str | None:
    """Why the split-K kernel cannot run ``tactic`` on ``x [m, k] @ weight [n, k].T``, or None.

    flashinfer's own validation (``validate_tactic``: shared memory, K divisibility, tile
    limits), the check ``dense_gemm.kernel_error`` makes for the target's cells; no launch.
    """
    try:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            validate_tactic,
        )
    except ImportError as error:
        return f"flashinfer's split-K kernel is not importable ({error})"
    try:
        validate_tactic(SplitKTactic(*tactic), m, n, k)
    except Exception as error:  # noqa: BLE001 - ValueError by contract; any error refuses the row
        return f"{type(error).__name__}: {error}"
    return None


@functools.lru_cache(maxsize=None)
def batch_splitk_tactic_error(
    m: int, k: int, n: int, tactic: tuple[int, int, int, int]
) -> str | None:
    """Why ``batch_splitk_gemm`` cannot run ``tactic`` on ``x [m, k] @ weight [n, k].T``, or None.

    The ``OnBatchSplitK`` counterpart of ``tactic_error``: that one asks flashinfer, whose shared
    memory budget is the smaller of the two, so it refuses the deeper rings this kernel is chosen
    for. No launch.
    """
    try:
        from . import batch_splitk_gemm as _bsk
    except ImportError as error:  # pragma: no cover - import guard
        return f"the batch split-K kernel is not importable ({error})"
    try:
        _bsk.validate(_bsk.BatchSplitKTactic(*tactic), m, n, k)
    except ValueError as error:
        return f"{type(error).__name__}: {error}"
    return None


def refused_rows(tp_size: int = 8, capability: tuple[int, int] | None = None) -> list[str]:
    """The rows of the ``splitk_tactics(tp_size, capability)`` table the kernel refuses
    (``tactic_error``), one ``"(M, K, N) tactic: reason"`` each; empty when every row
    validates, and for a table outside ``INSTALL_CHECKED``, whose rows are not checked.

    Raises:
        ValueError: as ``splitk_tactics``, for a pair without a table.
    """
    key = (tuple(device_capability() if capability is None else capability), tp_size)
    table = splitk_tactics(tp_size, key[0])
    if key not in INSTALL_CHECKED:
        return []
    refused = []
    for (m, k, n), tactic in sorted(table.items()):
        # An OnBatchSplitK row runs on batch_splitk_gemm's copy of the kernel, not flashinfer's,
        # and its reduce-scatter epilogue leaves room for a deeper ring -- so it carries stage
        # counts flashinfer refuses at the same tile, by design. Checking it with
        # ``tactic_error`` asks the wrong kernel and rejects a row that runs.
        error = (
            batch_splitk_tactic_error(m, k, n, tactic)
            if isinstance(tactic, OnBatchSplitK)
            else tactic_error(m, k, n, tuple(tactic))
        )
        if error is not None:
            refused.append(f"({m}, {k}, {n}) {tuple(tactic)}: {error}")
    return refused


def takes(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the kernel takes these operands (else the caller runs the built-in GEMM)."""
    return (
        x.dim() == 2
        and weight.dim() == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and 1 <= x.shape[0] <= MAX_M
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % 128 == 0
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
        and not torch.is_grad_enabled()
    )


class FixedLowMGemm(OpWrapper):
    """``x [M, K] @ weight [N, K].T -> [M, N]`` bf16, on the CuTe-DSL split-K kernel at ``tactic``."""

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: none. Calls
    ``ops/gemm/splitk_gemm.run_splitk_dense`` (flashinfer's ``dense_bf16_gemm_sm100_splitk``, moved here)
    (CuTe-DSL, JIT-compiled per tactic and shape on first use; the first call
    must not be inside a CUDA-graph capture) with the weight as the ``[K, N]``
    column-major view; both operands are passed detached (no autograd).
    Allocates the output per call.

    Call sites (``drafter_fixed_gemms`` / ``drafter_fc_shard``, ``drafter_worker.py``): the
    drafter's convolution coefficient projections, its context K/V projection,
    its TP-sharded ``qkv`` / ``o_proj`` / ``gate_up`` / ``down`` projections, and the
    DFlash ``fc`` shard -- each only at the shapes in the table of the drafter's
    TP size (``splitk_tactics``).

    Numerics: fp32 accumulation, one rounding to bf16; the K-split order is
    the tactic's. Within one bf16 ulp of the fp32 ``reference`` per element
    (the test's band); not bitwise ``F.linear``.

    Hard limits
      M <= 32, K % 128 == 0, bf16, contiguous, 32-byte aligned operands
      (``takes``), no bias.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tactic: tuple[int, int, int, int],
        enable_pdl: bool,
    ) -> torch.Tensor:
        from .splitk_gemm import SplitKTactic, run_splitk_dense

        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        if isinstance(tactic, OnBatchSplitK):
            from . import batch_splitk_gemm as _bsk

            return _bsk.run(
                x.detach(), weight.detach(), out, _bsk.BatchSplitKTactic(*tactic), enable_pdl
            )
        # DLPack refuses tensors that require grad; a trainable nn.Linear
        # weight still does under inference mode.
        return run_splitk_dense(
            x.detach(), weight.detach().t(), None, out, enable_pdl, SplitKTactic(*tactic)
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tactic: tuple[int, int, int, int],
        enable_pdl: bool,
    ) -> torch.Tensor:
        """fp32 product, one rounding to bf16. ``tactic`` and ``enable_pdl`` are ignored."""
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tactic: tuple[int, int, int, int],
        enable_pdl: bool,
    ) -> None:
        assert takes(x, weight), (
            "operands must be bf16, contiguous, 32-byte aligned, x [M <= 32, K] with K % 128 == 0 "
            f"matching weight [N, K]; got x {tuple(x.shape)} {x.dtype} weight "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
        assert len(tuple(tactic)) == 4, (
            f"tactic must be (mma_m, mma_n, split_k, ab_stages), got {tactic}"
        )


fixed_low_m_gemm = FixedLowMGemm()
