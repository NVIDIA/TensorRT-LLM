# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``batch_dense_gemms``: the attention block's dense projections on decode steps of 33..512 rows.

Above 32 rows the built-in ``Linear`` never asks the low-M dispatcher
(``_should_apply_low_m_gemm``: at most 32 rows), so a decode step of five or
more requests runs every QKV and attention output projection through cuBLAS
(``F.linear``) and the ``dense_gemms`` dispatchers never see the call. On a
sliding-window layer the output projection then streams its 33.5 MB weight
(TEP4 rank, ``[8192, 2048]`` BF16) from HBM after three latency-bound kernels
that move a few MB between them -- the decode prep, the sliding-window FMHA
over a 2048-token window and the attention norm-gate -- and the norm-gate
before it is a plain launch, so the projection also pays its launch latency
in series.

While ``DecodeTarget`` arms this path for one decode forward of
``MIN_ROWS..MAX_ROWS`` rows (``BatchDenseGemms.armed``):

* L2 prefetch of the output projection's weight on the sliding-window layers,
  issued by the layer's multi-request decode prep (``prep_prefetch``, the
  ``prefetch`` operand of ``ops/attention/multi_request_prep_append``) on steps of
  ``PREP_PREFETCH_MIN_ROWS..PREP_PREFETCH_MAX_ROWS`` rows: every prep program
  issues one bulk L2 prefetch of an equal slice of the weight right after its
  wait, so the requests leave the TMA units of all SMs before the sliding-window
  FMHA, which loads K/V through those units, starts. The o_proj GEMM (the
  built-in cuBLAS call) then finds most of its weight in L2. Issuing the requests
  lengthens the prep, which is on the step's critical path, by about the time
  33.5 MB takes to reach L2; up to 64 rows the o_proj and the PDL
  norm-gate below gain more than that, on larger steps they do not, so larger
  steps do not issue it. The prefetch runs inside a kernel the step already
  waits for, on every rank alike: a prefetch kernel on a side stream next to the
  FMHA would share the FMHA's TMA units and run for a time that varies per rank
  and layer, which the cross-rank post-attention hop after the o_proj then waits
  for. Not on the full-attention
  layers: their decode FMHA streams the whole KV cache (HBM-bound), so a prefetch
  there only trades bandwidth.
* The attention norm-gate as a PDL kernel: on a call of at most
  ``TRITON_NORM_GATE_PDL_MAX_ROWS`` rows, where the built-in ``apply_output_gate``
  launches its Triton kernel, the ``boundary_early_release`` PDL copy of that
  kernel (``ops/attention/norm_gate``) on a grid of at most
  ``NORM_GATE_MAX_PROGRAMS`` programs: the built-in grid is one 128-thread program
  per (row, head), and above about 1,300 of them the PDL copy runs longer than the
  plain launch; there each program runs the same statements for 2 consecutive
  (row, head) pairs (``norm_gate_heads_per_program``); at 65..128 rows the
  built-in launch, since there the earlier start of the projection behind a PDL
  copy does not end it earlier on the full-attention layers;
  where it would launch the CuTe ``rmsnorm_gate`` (above 128 rows with
  ``MUSESPARK_OPEN_LARGE_NORM_GATE=1``), the PDL copy of that kernel
  (``ops/attention/rmsnorm_gate_pdl``). Both wait before their first global read
  and let the projection launch right after that wait, so the projection's
  launch overlaps the norm-gate.
* The QKV projection at the row counts of ``QKV_TGV_ROWS``: flashinfer's TGV CuTe
  GEMM (the tactic of ``QKV_TGV_CONFIG``) instead of cuBLAS's pick, which at those
  row counts is a kernel that streams the 75.5 MB weight well below the HBM
  bandwidth. Only where the built-in call is ``F.linear`` on a bias-free BF16
  weight (no custom cuBLAS GEMM, no CuTe DSL BF16 GEMM, no LoRA).

Numerics: every value is computed by the same kernels on the same operands in the
same order as the built-in, or, for the QKV rows of ``QKV_TGV_ROWS``, by a kernel
that reduces every output over K in the same order: at those row counts cuBLAS's
pick does not split K, both kernels accumulate the 8192-long dot products tile by
tile along K with the same FP32 tensor-core MMA and round once to BF16, so they
give the same bits (a split-K pick would not, which is why rows where cuBLAS splits
K are not in the table). The prefetch is an L2 hint on an immutable weight that
writes nothing, and the PDL launches change only when a kernel may start.
Bit-identical.

Memory order and progress (the PDL norm-gate, the TGV QKV and the prep prefetch). A
PDL dependent's ``griddepcontrol.wait`` returns only once its whole predecessor grid
has completed and its stores are visible; both norm-gate copies read nothing before
their wait. The norm-gate was a plain launch, a full barrier in every layer. The
transports whose buffers are reused across layers at these rows -- the wide
post-attention hop (``ops/allreduce/post_attn_hop_wide``), the fused finalize
all-reduce and the wide MoE all-reduce -- rest on "every kernel between two calls
waits on its predecessor before it completes", which both copies keep, and the
built-in routing kernel (a plain launch) still sits between any two calls of each.
The TGV QKV GEMM, launched with PDL, may issue its weight loads before its
``griddepcontrol.wait`` (immutable bytes) and reads the activation only after it;
every CTA's activation-load warp executes that wait, so it too completes only after
its predecessor, and its successor (the decode prep) waits before its first read.
The prep issues the prefetch after its own wait and still releases its dependent
(the FMHA) only at its end; the prefetch reads only the immutable weight, writes
nothing (its keep-alive store is never taken) and waits on nothing.

LL safety: armed only on decode forwards of 33..512 rows at TEP4 (sm103 and sm107); the
8-row paths and the 9..32-row band never see it.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, FrozenSet, Iterator, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod
from tensorrt_llm.logger import logger

from ...dispatch import LAUNCH_PDL
from ...ops.attention import norm_gate as _norm_gate
from ...ops.attention import rmsnorm_gate_pdl as _rmsnorm_gate_pdl
from ..multi_request_prep_append.attention import MultiRequestPrepAppendAttention

try:
    from flashinfer.gemm.kernels.tgv_gemm_cute_ext import run_tgv_cute_ext
except ImportError:  # pragma: no cover - the served image has flashinfer's TGV GEMM
    run_tgv_cute_ext = None

#: The decode forwards this path arms: five to 64 requests of 8 rows (K = 7). Below are
#: the 8-row LL paths and the 9..32-row band; above, the target's other wide paths stop.
MIN_ROWS = 33
MAX_ROWS = 512
#: The built-in ``apply_output_gate``'s Triton bound: up to this many rows it launches
#: ``_attention_norm_gate_kernel``, above it the CuTe ``rmsnorm_gate`` (or the unfused norm).
TRITON_NORM_GATE_MAX_ROWS = 128
#: Up to this many rows the Triton kernel's PDL copy replaces it.
TRITON_NORM_GATE_PDL_MAX_ROWS = 64
#: The PDL copy's grid is kept at most this many programs: above it, each program runs the
#: built-in statements for several consecutive (row, head) pairs (``norm_gate_heads_per_program``).
NORM_GATE_MAX_PROGRAMS = 1280
#: Decode steps whose multi-request prep issues the o_proj weight's L2 prefetch
#: (``prep_prefetch``) on the sliding-window layers.
PREP_PREFETCH_MIN_ROWS = 33
PREP_PREFETCH_MAX_ROWS = 64
#: Decode-step row counts at which the QKV projection runs flashinfer's TGV GEMM: those of
#: the captured decode graphs where cuBLAS's pick does not split K (same bits) and is slower
#: than the TGV tactic below, measured per device against that device's cuBLAS picks. On
#: GB300 (sm_103) at 176 and 208 rows cuBLAS picks a faster kernel and keeps the call; at
#: 40, 48 and 80 rows it splits K.
QKV_TGV_ROWS = frozenset({144, 160, 192, 224, 240, 256})
#: The same table on R200 (sm_107): empty. There cuBLAS's QKV pick does not split K from 136
#: rows (every TGV tactic gives its bits), but it is faster than every TGV tactic at every
#: captured row count of 40..512; below 136 rows
#: no TGV tactic gives its bits. So at sm_107 the path keeps the built-in QKV GEMM and serves
#: only the PDL norm-gates.
QKV_TGV_ROWS_SM107 = frozenset()
#: The table per device compute capability; any other device keeps the built-in QKV GEMM.
QKV_TGV_ROWS_BY_CAPABILITY = {(10, 3): QKV_TGV_ROWS, (10, 7): QKV_TGV_ROWS_SM107}
#: The TGV tactic of the table, as flashinfer's ``_TGV_CUTE_EXT_TACTIC_CONFIGS`` lists it:
#: (CTA M, CTA N, A/B stages, 2-CTA MMA) = a 64 x 128 tile, 7 stages, 2-CTA.
QKV_TGV_CONFIG = (64, 128, 7, True)
#: The alignment in bytes the TGV GEMM assumes for every operand (``assumed_align=32``).
TGV_ALIGN = 32


def call_rows(input_tensor: torch.Tensor) -> int:
    """Rows of a projection call (every leading dimension flattened)."""
    k = int(input_tensor.shape[-1]) if input_tensor.dim() >= 1 else 0
    return input_tensor.numel() // k if k > 0 else 0


def tgv_tactic(configs: Optional[Sequence[Sequence]] = None) -> Optional[int]:
    """The index of ``QKV_TGV_CONFIG`` in a TGV tactic table -- ``configs``, by default
    flashinfer's ``_TGV_CUTE_EXT_TACTIC_CONFIGS`` -- or ``None`` when the table is not
    importable or does not hold it."""
    if configs is None:
        try:
            from flashinfer.gemm.kernels import tgv_gemm_cute_ext
        except ImportError:
            return None
        configs = getattr(tgv_gemm_cute_ext, "_TGV_CUTE_EXT_TACTIC_CONFIGS", ())
    table = [tuple(config) for config in configs]
    return table.index(QKV_TGV_CONFIG) if QKV_TGV_CONFIG in table else None


class BatchDenseGemms:
    """What ``DecodeTarget`` holds under ``batch_dense_gemms``: the arming switch the re-classed
    projections, the multi-request preps and the norm-gates read, and a census.

    Host state only: under a CUDA graph each arming is a per-capture constant.
    """

    def __init__(
        self,
        plan,
        qkv_tgv_rows: FrozenSet[int] = QKV_TGV_ROWS,
        qkv_tgv_tactic: Optional[int] = None,
    ) -> None:
        self.plan = plan
        #: The QKV rows of the TGV table; empty without a tactic.
        self.qkv_tgv_rows = frozenset(qkv_tgv_rows) if qkv_tgv_tactic is not None else frozenset()
        self.qkv_tgv_tactic = qkv_tgv_tactic
        self.active = False
        #: Calls per kind in armed forwards: TGV QKVs, prep prefetches, norm-gates by copy,
        #: and calls left to the built-in (with why, once each).
        self.counts: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}

    @staticmethod
    def takes(rows: int) -> bool:
        """Whether a decode forward of ``rows`` rows is armed (every rank sees the same rows)."""
        return MIN_ROWS <= rows <= MAX_ROWS

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Arm the projections, preps and norm-gates for one decode forward; disarm on any exit."""
        self.plan.batch = self
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.plan.batch = None

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def _decline(self, what: str) -> None:
        self.declined[what] = self.declined.get(what, 0) + 1
        if self.declined[what] == 1:
            logger.warning(f"[musespark_open] batch_dense_gemms: {what}; it runs the built-in path")

    def projection(
        self,
        site,
        input_tensor: torch.Tensor,
        builtin: Callable[[], torch.Tensor],
        module: Optional[torch.nn.Module] = None,
    ):
        """A dense projection call of an armed forward at ``site`` (``dense_gemms``'s ``_Site``).

        ``builtin()`` runs the built-in GEMM (``Linear.apply_linear``); ``module`` is the
        re-classed ``Linear`` when the call carries no bias and no LoRA, else ``None``. The
        QKV site runs the TGV GEMM at the rows of the table; every other call runs the
        built-in GEMM.
        """
        if site.role != "qkv" or module is None or not self.qkv_tgv_rows:
            return builtin()
        if call_rows(input_tensor) not in self.qkv_tgv_rows or not tgv_takes(module, input_tensor):
            return builtin()
        self._count("qkv_tgv")
        return qkv_tgv(module.weight, input_tensor, self.qkv_tgv_tactic)

    def prep_prefetch(self, attention, rows: int) -> Optional[torch.Tensor]:
        """The weight the multi-request decode prep of ``attention`` asks the L2 to fetch on a
        step of ``rows`` rows -- its o_proj weight on a sliding-window layer in the band of
        ``PREP_PREFETCH_*_ROWS`` of an armed forward -- or ``None``."""
        if not self.active or not PREP_PREFETCH_MIN_ROWS <= rows <= PREP_PREFETCH_MAX_ROWS:
            return None
        if not getattr(attention, "is_sliding", False):
            return None
        weight = attention.o_proj.weight
        if not (weight.is_cuda and weight.is_contiguous() and weight.data_ptr() % 16 == 0):
            self._decline("an o_proj weight the bulk L2 prefetch does not take")
            return None
        self._count("prep_prefetch")
        return weight

    def norm_gate(self, attention, attention_output, gate, trigger: str) -> Optional[torch.Tensor]:
        """The PDL copy of the kernel the built-in ``apply_output_gate`` would launch for this
        call, or ``None`` (the built-in runs) outside an armed forward, outside the row band or
        where the built-in would take another branch."""
        if not self.active or not isinstance(attention_output, torch.Tensor):
            return None
        rows = int(attention_output.shape[0]) if attention_output.dim() >= 1 else 0
        if not self.takes(rows) or not _builtin_gate_fits(attention, attention_output, gate):
            return None
        eps = attention.o_head_norm.variance_epsilon
        if rows <= TRITON_NORM_GATE_MAX_ROWS:
            if rows > TRITON_NORM_GATE_PDL_MAX_ROWS:
                return None
            self._count("norm_gate_triton")
            heads = int(attention_output.shape[1]) // _norm_gate.HEAD_DIM
            return _norm_gate.attention_norm_gate(
                attention_output,
                gate,
                eps,
                trigger,
                heads_per_program=norm_gate_heads_per_program(rows, heads),
            )
        if not getattr(attention, "_large_norm_gate_enabled", False):
            return None
        if torch.compiler.is_compiling() or not _rmsnorm_gate_pdl.takes(attention_output, gate):
            self._decline("an output gate the CuTe rmsnorm_gate copy does not take")
            return None
        self._count("norm_gate_cute")
        return _rmsnorm_gate_pdl.attention_rmsnorm_gate_pdl(attention_output, gate, eps, trigger)

    def census(self) -> str:
        """One line of what the armed forwards did (logged by the tests)."""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))


def norm_gate_heads_per_program(rows: int, heads: int) -> int:
    """(row, head) pairs per program of the PDL Triton norm-gate: the smallest power of two that
    divides ``heads`` and keeps the grid at most ``NORM_GATE_MAX_PROGRAMS`` programs (else the
    largest such power of two)."""
    per_program = 1
    while rows * heads // per_program > NORM_GATE_MAX_PROGRAMS and heads % (2 * per_program) == 0:
        per_program *= 2
    return per_program


def tgv_takes(module: torch.nn.Module, input_tensor: torch.Tensor) -> bool:
    """Whether the built-in call of ``module`` on ``input_tensor`` (no bias, no LoRA) is
    cuBLAS's ``F.linear`` on operands the TGV GEMM takes: a contiguous BF16 weight and
    activation, ``TGV_ALIGN``-byte aligned, and none of the built-in's other GEMM paths."""
    weight = getattr(module, "weight", None)
    return (
        run_tgv_cute_ext is not None
        and isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)
        and not getattr(module, "use_custom_cublas_mm", False)
        and not getattr(module, "use_cute_dsl_bf16_gemm", False)
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and weight.is_cuda
        and weight.is_contiguous()
        and weight.data_ptr() % TGV_ALIGN == 0
        and input_tensor.dtype == torch.bfloat16
        and input_tensor.device == weight.device
        and input_tensor.dim() >= 2
        and input_tensor.is_contiguous()
        and input_tensor.data_ptr() % TGV_ALIGN == 0
        and input_tensor.shape[-1] == weight.shape[1]
    )


def qkv_tgv(weight: torch.Tensor, input_tensor: torch.Tensor, tactic: int) -> torch.Tensor:
    """``F.linear(input_tensor, weight)`` by flashinfer's TGV GEMM, tactic ``tactic``, launched
    with PDL: a fresh contiguous ``[*, N]`` BF16 tensor, as the built-in returns."""
    x = input_tensor.reshape(-1, input_tensor.shape[-1])
    out = torch.empty(x.shape[0], weight.shape[0], dtype=torch.bfloat16, device=x.device)
    run_tgv_cute_ext(x, weight.t(), None, out, LAUNCH_PDL, tactic)
    return out.view(*input_tensor.shape[:-1], weight.shape[0])


def _builtin_gate_fits(attention, attention_output: torch.Tensor, gate) -> bool:
    """The operand conditions the built-in ``apply_output_gate`` checks before either of its
    kernels (row bounds and the large-gate alignment are checked by the caller)."""
    return (
        gate is not None
        and isinstance(gate, torch.Tensor)
        and attention_output.is_cuda
        and attention_output.dtype == torch.bfloat16
        and attention_output.ndim == 2
        and attention_output.stride(-1) == 1
        and attention.head_dim == _norm_gate.HEAD_DIM
        and attention_output.shape[1] % attention.head_dim == 0
        and gate.is_cuda
        and gate.dtype == torch.bfloat16
        and gate.ndim in (2, 3)
        and gate.stride(-1) == 1
        and gate.shape[0] == attention_output.shape[0]
        and gate.numel() == attention_output.numel()
        and (gate.ndim == 2 or gate.shape[-1] == attention.head_dim)
    )


def check(model: torch.nn.Module, plan) -> None:
    """Whether ``install`` may run; changes nothing.

    Raises:
        ValueError: without the ``dense_gemms`` plan, with a QKV projection ``dense_gemms``
            did not re-class, an attention whose output gate is not the
            ``boundary_early_release`` PDL mixin, or a head width the norm-gate copies do
            not take.
    """
    from ..dense_gemms.dispatcher import MuseSparkOpenTepDenseLinear

    if plan is None:
        raise ValueError("[musespark_open] batch_dense_gemms needs the dense_gemms plan")
    for index, layer in enumerate(model.layers):
        attention = layer.self_attn
        if not isinstance(attention.qkv_proj, MuseSparkOpenTepDenseLinear):
            raise ValueError(
                f"[musespark_open] batch_dense_gemms: layer {index} qkv is not re-classed by "
                "dense_gemms"
            )
        if not hasattr(attention, "_norm_gate_trigger"):
            raise ValueError(
                f"[musespark_open] batch_dense_gemms: layer {index}'s attention does not carry the "
                "boundary_early_release PDL norm-gate"
            )
        if attention.head_dim != _norm_gate.HEAD_DIM:
            raise ValueError(
                f"[musespark_open] batch_dense_gemms: layer {index} head_dim {attention.head_dim}, "
                f"the norm-gate copies take {_norm_gate.HEAD_DIM}"
            )


def install(
    model: torch.nn.Module, plan, capability: Optional[Tuple[int, int]] = None
) -> BatchDenseGemms:
    """Hand every attention the switch; return it.

    Every layer is checked before anything changes (``check``). The QKV table is the one of
    the device's ``capability`` (``QKV_TGV_ROWS_BY_CAPABILITY``; default: the current
    device's); it stays empty on another device or when flashinfer's TGV tactic table does not
    hold ``QKV_TGV_CONFIG``.
    """
    check(model, plan)
    if capability is None and torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
    rows = QKV_TGV_ROWS_BY_CAPABILITY.get(capability, frozenset())
    tactic = tgv_tactic()
    if tactic is None and run_tgv_cute_ext is not None:
        logger.warning(
            f"[musespark_open] batch_dense_gemms: flashinfer's TGV tactic table holds no "
            f"{QKV_TGV_CONFIG}; the QKV projections keep the built-in GEMM"
        )
    switch = BatchDenseGemms(plan, rows, tactic)
    for layer in model.layers:
        layer.self_attn._batch_dense_gemms = switch
    prefetching = sum(
        1
        for layer in model.layers
        if layer.self_attn.is_sliding
        and isinstance(layer.self_attn, MultiRequestPrepAppendAttention)
    )
    logger.info(
        f"[musespark_open] batch_dense_gemms: decode steps of {MIN_ROWS}..{MAX_ROWS} rows; at "
        f"{PREP_PREFETCH_MIN_ROWS}..{PREP_PREFETCH_MAX_ROWS} rows the multi-request prep "
        f"prefetches the o_proj weight into L2 on {prefetching} sliding-window layers of "
        f"{len(model.layers)}; "
        f"the attention norm-gate runs as a PDL kernel up to {TRITON_NORM_GATE_PDL_MAX_ROWS} rows "
        f"(Triton copy, at most {NORM_GATE_MAX_PROGRAMS} programs) and above "
        f"{TRITON_NORM_GATE_MAX_ROWS} (CuTe rmsnorm_gate copy); the QKV projection runs the TGV "
        f"GEMM (tactic {tactic}) at {sorted(switch.qkv_tgv_rows)} rows"
    )
    return switch
