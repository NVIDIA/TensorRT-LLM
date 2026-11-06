# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's dense projections: each on its checkpoint-specialised CuTe DSL kernel at the
decode row counts, and on the shared prefill GEMM at context row counts."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.gemm import lm_head_n38720_k6144_bf16 as _lm_head
from ...kernels.gemm import mlp_dense_down_proj_n6144_k3072_bf16 as _dense_down
from ...kernels.gemm import mlp_dense_gate_up_proj_n6144_k6144_bf16 as _dense_gate_up
from ...kernels.gemm import mlp_gate_m64_n256_k6144_bf16_fp32 as _router
from ...kernels.gemm import mlp_shared_experts_down_proj_m64_n6144_k512_bf16 as _shared_down
from ...kernels.gemm import mlp_shared_experts_gate_up_proj_m64_n1024_k6144_bf16 as _shared_gate_up
from ...kernels.gemm import moe_prefill_shared as _shared_prefill
from ...kernels.gemm import self_attn_indexer_pre_indexer_proj_m64_n160_k6144_tf32 as _ix_fused
from ...kernels.gemm import self_attn_indexer_wq_b_m64_n4096_k2048_bf16 as _ix_qb
from ...kernels.gemm import self_attn_kv_a_proj_with_mqa_m64_n2624_k6144_bf16_l2prefetch as _qkva
from ...kernels.gemm import self_attn_o_proj_m64_n6144_k4096_bf16 as _o_proj
from ...kernels.gemm import self_attn_prefill_projections_bf16 as _prefill
from ...kernels.gemm import self_attn_q_b_proj_m64_n4096_k2048_bf16 as _q_b

#: The kernel modules, by the name the target calls each projection.
_MODULES = {
    "qkva": _qkva,
    "q_b": _q_b,
    "o_proj": _o_proj,
    "ix_qb": _ix_qb,
    "ix_fused": _ix_fused,
    "router": _router,
    "dense_gate_up": _dense_gate_up,
    "dense_down": _dense_down,
    "shared_gate_up": _shared_gate_up,
    "shared_down": _shared_down,
    "lm_head": _lm_head,
}

#: The fp32 indexer projection compiles per row count: the bf16 projections'
#: decode row counts plus the prefill chunk sizes this engine produces. Its
#: kernel itself takes any row count.
IX_FUSED_ROWS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 1099, 3282, 8192)


@dataclass(frozen=True)
class ProjectionShape:
    """What one kernel takes: `x [rows, k] in_dtype`, `weight [n, k] in_dtype`,
    `out [rows, n] out_dtype`, and the byte alignment of each of the three."""

    n: int
    k: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    align: tuple[int, int, int]


def shape_of(kernel: str) -> ProjectionShape:
    """The operand shapes, dtypes and alignments of `kernel`."""
    if kernel not in _MODULES:
        # A projection only the prefill GEMM serves (mtp_eh, the kv_b expansion halves,
        # the fp8 attention's e4m3 entries): its entry's contract.
        abi = _prefill.ABIS[PREFILL_ENTRIES[kernel]]
        return ProjectionShape(
            abi.out_features, abi.in_features, torch.bfloat16, abi.out_dtype, _PREFILL_ALIGN
        )
    module = _MODULES[kernel]
    if kernel == "ix_fused":
        return ProjectionShape(module.N, module.K, torch.float32, torch.float32, (16, 16, 16))
    abi = module.ABI
    return ProjectionShape(
        abi.output_features, abi.input_features, torch.bfloat16, abi.output_dtype, (128, 128, 32)
    )


#: The decode row counts above 64 each kernel has a schedule for (its module's
#: `MTP_DECODE_M_VALUES`): an MTP3 / MTP5 verify or draft-step-0 batch of 32 sequences
#: at 4 or 6 rows each. Built only on request (`build_all(..., rows=MTP_ROWS[kernel])`):
#: the target builds them for an engine drafting more than one token, and sends them
#: decode rows only.
MTP_ROWS: dict[str, tuple[int, ...]] = {
    name: tuple(getattr(module, "MTP_DECODE_M_VALUES", ())) for name, module in _MODULES.items()
}

#: Every other row count each kernel has a schedule for. `build_all` compiles these.
ROWS: dict[str, tuple[int, ...]] = {
    name: IX_FUSED_ROWS
    if name == "ix_fused"
    else tuple(m for m in module.SUPPORTED_M_VALUES if m not in MTP_ROWS[name])
    for name, module in _MODULES.items()
}


@dataclass(frozen=True)
class ProjectionRunner:
    """One compiled projection kernel at one row count.

    `launch(x, weight, out)` is the kernel's own runner, already compiled.
    `rows`, `n`, `k` and the dtypes are what it was compiled for.
    """

    kernel: str
    rows: int
    n: int
    k: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


def build(kernel: str, rows: int, weight: torch.Tensor) -> ProjectionRunner:
    """Compile `kernel` for `rows` rows against `weight`. Call at load.

    `weight` is the row-major `[N, K]` weight of any one layer; the runner
    takes the weight per call, so one runner serves every layer. Compiling
    against the real weight makes the kernel check that weight's layout now:
    after their first call the bf16 kernels check only the row count.

    Costs one CuTe DSL compile (the kernel module caches it per device and
    row count, so a second `build` of the same pair is free) and one launch
    on a zero input. Raises if `rows` is in neither `ROWS[kernel]` nor
    `MTP_ROWS[kernel]`, or the compile fails.
    """
    shape = shape_of(kernel)
    rows = int(rows)
    if rows not in ROWS[kernel] + MTP_ROWS[kernel]:
        raise ValueError(
            f"{kernel}: no schedule for {rows} rows; compiled rows are "
            f"{ROWS[kernel] + MTP_ROWS[kernel]}"
        )
    x = torch.zeros([rows, shape.k], dtype=shape.in_dtype, device=weight.device)
    out = torch.empty([rows, shape.n], dtype=shape.out_dtype, device=weight.device)
    with torch.cuda.device(weight.device):
        launch = _MODULES[kernel].build()
        launch(x, weight, out)
    return ProjectionRunner(kernel, rows, shape.n, shape.k, shape.in_dtype, shape.out_dtype, launch)


def at_rows(runner: ProjectionRunner, rows: int) -> ProjectionRunner | None:
    """`runner`'s compiled kernel launched on `rows` rows, at most the count it was built
    for: a runner pinned to `rows` that runs the cell's own instance (same schedule, same
    tiles, same K order) on the step's exact rows, so each output row is bitwise the cell's
    for that row -- no padding, no copy. For an MTP3 / MTP5 decode row count without a cell
    (`MTP_ROWS` holds 128 and 192; a 24-request batch is 96 or 144 rows): the smallest cell
    above it serves it.

    None where the cell's backend is compiled for one M (the tcgen05 cells, M <= 64) or
    `rows` is outside `1..runner.rows`. Builds nothing: the instance compiled and launched
    when `runner` was built, so this is safe inside a CUDA-graph capture.
    """
    launch = getattr(runner.launch, "at_rows", None)
    takes = getattr(runner.launch, "takes_rows", None)
    rows = int(rows)
    if launch is None or takes is None or not takes() or not 0 < rows <= runner.rows:
        return None
    return ProjectionRunner(
        runner.kernel, rows, runner.n, runner.k, runner.in_dtype, runner.out_dtype, launch
    )


def build_all(
    kernel: str, weight: torch.Tensor, rows: tuple[int, ...] | None = None
) -> dict[int, ProjectionRunner]:
    """`build` at every row count in `rows` (default `ROWS[kernel]`, which leaves out
    `MTP_ROWS[kernel]`).

    Returns `{rows: runner}` for the row counts that compiled. A row count
    whose compile raises is left out rather than raised: the large prefill
    row counts are shapes a given engine may never produce, and the target
    sends a row count without a runner to cuBLAS. Compare the keys with
    `rows` to see what was left out.
    """
    runners = {}
    for m in ROWS[kernel] if rows is None else rows:
        try:
            runners[int(m)] = build(kernel, m, weight)
        except Exception:  # noqa: BLE001 -- that row count goes to cuBLAS at the call site
            continue
    return runners


class _EagerTwin:
    """`launch(x, weight, out)` of a decode persistent-GEMM cell whose eager calls take the same
    kernel compiled as a TVM-FFI function (CTX-HOST-DIET, TRTLLM_GLM52_CTX_HOST_DIET=1).

    The cell's own launcher builds three DLPack cute tensors and packs the JIT executor's
    arguments on every call: about 100 us of Python per call for a mixed step's q_b at 16
    generation rows (cProfile, B300 serving run). The twin is the prefill GEMM's
    `compile_persistent_gemm_tvm_ffi` of the cell's schedule and kernel name -- the same kernel
    class, `bmm` entry, `can_implement` gate and dynamic layouts as the cell's
    `compile_persistent_gemm`, so the same device code -- and a call checks what the cell's
    launcher checks (shapes) plus device and 16-byte alignment, then hands the three `bmm`
    torch views to one TVM-FFI call on torch's current stream. Calls inside a CUDA-graph
    capture take the cell's own launcher, so captured decode graphs are unchanged.
    """

    def __init__(self, launch, fn, rows: int, n: int, k: int, orientation: str, device):
        self.launch = launch
        self.fn = fn
        self.orientation = orientation
        self.device = device
        self.x_shape = (rows, k)
        self.w_shape = (n, k)
        self.out_shape = (rows, n)

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> object:
        if torch.cuda.is_current_stream_capturing():
            return self.launch(x, weight, out)
        if (
            x.shape != self.x_shape
            or weight.shape != self.w_shape
            or out.shape != self.out_shape
            or x.device != self.device
        ):
            raise ValueError(
                f"projection twin compiled for x {self.x_shape}, weight {self.w_shape}, out "
                f"{self.out_shape} on {self.device}; got {tuple(x.shape)}, {tuple(weight.shape)}, "
                f"{tuple(out.shape)} on {x.device}"
            )
        if (x.data_ptr() | weight.data_ptr() | out.data_ptr()) & 15:
            raise ValueError("projection twin: x, weight and out must be 16-byte aligned")
        if weight.requires_grad:
            weight = weight.detach()
        return self.fn(*_prefill._bmm_operand_views(x, weight, out, self.orientation))


def with_eager_twin(runner: ProjectionRunner, weight: torch.Tensor) -> ProjectionRunner:
    """`runner` with its eager calls on a TVM-FFI twin of its kernel (`_EagerTwin`) where it is
    a plain persistent-GEMM cell, else `runner` itself. Call at load: one CuTe DSL compile and
    one launch on zeros per cell."""
    spec = getattr(runner.launch, "persistent", None)
    if spec is None or spec[1] is None:
        return runner
    schedule, prefix = spec
    dev = weight.device
    x = torch.zeros([runner.rows, runner.k], dtype=runner.in_dtype, device=dev)
    out = torch.empty([runner.rows, runner.n], dtype=runner.out_dtype, device=dev)
    with torch.cuda.device(dev):
        fn = _prefill.compile_persistent_gemm_tvm_ffi(
            x, weight, out, schedule, kernel_name_prefix=prefix
        )
        twin = _EagerTwin(
            runner.launch, fn, runner.rows, runner.n, runner.k, schedule.orientation, dev
        )
        twin(x, weight, out)
    return dataclasses.replace(runner, launch=twin)


#: The prefill GEMM's entry for each projection it can serve, by the target's name.
#: "mtp_eh" is the MTP front's replicated `[M, 12288] x [6144, 12288]` eh_proj, which
#: has no decode kernel here (its decode form is `cute_dsl_mtp_eh_proj`'s K-split).
PREFILL_ENTRIES = {
    "qkva": "kv_a_proj_with_mqa",
    "q_b": "q_b_proj",
    "o_proj": "o_proj",
    "router": "router",
    "dense_gate_up": "dense_gate_up",
    "dense_down": "dense_down",
    "shared_gate_up": "shared_gate_up",
    "shared_down": "shared_down",
    "mtp_eh": "mtp_eh_proj",
    # The non-absorbed context attention's K / V expansion: the `[M, 576]` latent rows,
    # read in place from the q_a | kv_a projection row (row stride 2624), times one half
    # of the augmented kv_b weight `[4096, 576]`. Prefill only, like mtp_eh.
    "kv_b_k": "kv_b_expand_k",
    "kv_b_v": "kv_b_expand_v",
    # The fp8 context attention's operands, e4m3 out (fp32 accumulator converted nearest,
    # saturating): K from the K half, V from the V half without its zero rope block (`x`
    # the latent row's c_kv part `[M, 512]`, W_UV `[4096, 512]`), the context rows' Q from
    # q_b. The caller folds the power-of-two dequantization scales into the weights.
    "kv_b_k_e4m3": "kv_b_expand_k_e4m3",
    "kv_b_v512_e4m3": "kv_b_expand_v512_e4m3",
    "q_b_e4m3": "q_b_proj_e4m3",
}
#: The projections only the prefill GEMM serves (no decode kernel), with their output dtype.
_PREFILL_ONLY: dict[str, torch.dtype] = {
    "mtp_eh": torch.bfloat16,
    "kv_b_k": torch.bfloat16,
    "kv_b_v": torch.bfloat16,
    "kv_b_k_e4m3": torch.float8_e4m3fn,
    "kv_b_v512_e4m3": torch.float8_e4m3fn,
    "q_b_e4m3": torch.float8_e4m3fn,
}
#: The prefill GEMM serves every row count 1..PREFILL_MAX_ROWS.
PREFILL_MAX_ROWS = _prefill.MAX_ROWS
#: Its operands are taken as 16-byte aligned and never checked.
_PREFILL_ALIGN = (16, 16, 16)


@dataclass(frozen=True)
class PrefillProjectionRunner:
    """The prefill GEMM for one projection: one compiled instance per row range of
    its schedule table, any row count 1..`max_rows`.

    `launch(x, weight, out)` is the kernel's own launcher. It caches a view of each
    weight it meets, keyed by the weight's address, and checks the weight's shape
    and layout only the first time.
    """

    kernel: str
    n: int
    k: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    max_rows: int
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]
    #: The entry takes `x` with a row stride wider than K (unit inner stride, a multiple
    #: of 8 elements): the kv_b expansion reads the latent rows in place.
    row_strided_input: bool = False


def build_prefill(kernel: str, weight: torch.Tensor) -> PrefillProjectionRunner:
    """Build the prefill GEMM for `kernel` against `weight`. Call at load.

    `weight` is the row-major `[N, K]` bf16 weight of any one layer; the runner
    takes the weight per call, so one runner serves every layer. Compiles every
    row range of the entry's schedule table (two for the attention projections),
    and for kv_a / o_proj every tile-width candidate the kernel picks per row count
    above 2048 rows, shared process-wide per entry, then launches each compiled
    instance once on zeros, so no context step meets a first compile or a first
    module load.

    Raises if the vendored kernel has no entry for `kernel`, or if the entry's
    output dtype is not the projection's (the router's logits are fp32): the
    target then keeps cuBLAS for that projection's context rows.
    """
    entry = PREFILL_ENTRIES[kernel]
    abi = _prefill.ABIS.get(entry)
    if abi is None:
        raise ValueError(f"{kernel}: the vendored prefill GEMM has no {entry!r} entry")
    expected = _PREFILL_ONLY.get(kernel) or shape_of(kernel).out_dtype
    if abi.out_dtype != expected:
        raise ValueError(f"{kernel}: the prefill GEMM writes {abi.out_dtype}, not {expected}")
    dev = weight.device
    with torch.cuda.device(dev):
        launch = _prefill.build(entry, weight)
        # One row count per compiled instance: each row range, and each tile candidate.
        for rows in launch.warm_rows(floor=1):
            x = torch.zeros([rows, abi.in_features], dtype=torch.bfloat16, device=dev)
            out = torch.empty([rows, abi.out_features], dtype=abi.out_dtype, device=dev)
            launch(x, weight, out)
    return PrefillProjectionRunner(
        kernel,
        abi.out_features,
        abi.in_features,
        torch.bfloat16,
        abi.out_dtype,
        PREFILL_MAX_ROWS,
        launch,
        bool(abi.row_strided_input),
    )


def build_prefill_all(weights: dict[str, torch.Tensor]) -> dict[str, PrefillProjectionRunner]:
    """`build_prefill` for every `{kernel: weight}`; `{kernel: runner}` for those that
    built. One that raises is left out, and the target keeps cuBLAS for it."""
    runners = {}
    for kernel, weight in weights.items():
        try:
            runners[kernel] = build_prefill(kernel, weight)
        except Exception:  # noqa: BLE001 -- that projection's context rows take cuBLAS
            continue
    return runners


#: The shared expert's prefill down kernel: `[rows, 512] x [6144, 512]`, rows 65..8192,
#: every operand 128-byte aligned (checked by the kernel on every call).
SHARED_DOWN_PREFILL_MIN_ROWS = 65
SHARED_DOWN_PREFILL_MAX_ROWS = 8192


@dataclass(frozen=True)
class SharedDownPrefillRunner:
    """The shared expert's prefill down projection, one kernel on a symbolic row count:
    any row count in `min_rows..max_rows`. `launch(x, weight, out)` is the kernel's
    compiled call, which checks shapes, dtypes, alignment and overlap per call."""

    kernel: str
    n: int
    k: int
    in_dtype: torch.dtype
    out_dtype: torch.dtype
    min_rows: int
    max_rows: int
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


def build_shared_down_prefill(weight: torch.Tensor) -> SharedDownPrefillRunner:
    """Compile the shared expert's prefill down projection (two-CTA clusters over every
    SM, N tile 256, 6 stages, PDL as the engine sets it) and launch it once on 65 zero
    rows against `weight`, the row-major bf16 `[6144, 512]` down weight of any layer.
    Call at load. Raises if the compile fails: the target then keeps the two-launch path
    for the shared expert's context rows.
    """
    dev = weight.device
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    shape = shape_of("shared_down")
    rows = SHARED_DOWN_PREFILL_MIN_ROWS
    with torch.cuda.device(dev):
        launch = _shared_prefill.compile_down(
            grid_ctas=num_sms, tile_n=256, stages=6, use_pdl=_USE_PDL
        )
        x = torch.zeros([rows, shape.k], dtype=torch.bfloat16, device=dev)
        out = torch.empty([rows, shape.n], dtype=torch.bfloat16, device=dev)
        launch(x, weight, out)
    return SharedDownPrefillRunner(
        "shared_down",
        shape.n,
        shape.k,
        torch.bfloat16,
        torch.bfloat16,
        SHARED_DOWN_PREFILL_MIN_ROWS,
        SHARED_DOWN_PREFILL_MAX_ROWS,
        launch,
    )


Runner = ProjectionRunner | PrefillProjectionRunner | SharedDownPrefillRunner


class CuteDslProjection(OpWrapper):
    """`out = x @ weight.T`, on the kernel `runner` was compiled for. Returns `out`.

    `x` is `[rows, K]`, `weight` the row-major `[N, K]` nn.Linear weight
    (not the column-major `[K, N]` view `cublas_mm` takes: pass that view's
    `.t()`, which is the same memory), `out` a caller-owned `[rows, N]`
    buffer. All three contiguous. Operands are bf16 and accumulate in fp32,
    except `ix_fused`, which takes fp32 operands and multiplies in TF32. The
    output is bf16, except `router` and `ix_fused`, which write fp32.

    The eleven decode kernels, by the name the target uses (per-rank shapes, TP4):

        qkva            [M, 6144] x [2624, 6144]  q_a | kv_a | k_pe
        q_b             [M, 2048] x [4096, 2048]
        o_proj          [M, 4096] x [6144, 4096]
        ix_qb           [M, 2048] x [4096, 2048]  indexer q
        ix_fused        [M, 6144] x [160, 6144]   indexer k | head weights, fp32
        router          [M, 6144] x [256, 6144]   fp32 logits
        dense_gate_up   [M, 6144] x [6144, 6144]  layers 0-2
        dense_down      [M, 3072] x [6144, 3072]
        shared_gate_up  [M, 6144] x [1024, 6144]  shared expert
        shared_down     [M, 512]  x [6144, 512]
        lm_head         [M, 6144] x [38720, 6144] this rank's vocabulary shard

    A `PrefillProjectionRunner` runs the same product on the prefill GEMM at any row
    count up to 8192: bf16 operands, fp32 accumulation, the entry's output dtype.
    Its entries (`PREFILL_ENTRIES`) are the bf16 projections above except ix_qb, plus
    the MTP front's replicated eh_proj and the non-absorbed context attention's K / V
    expansion (`x` the latent rows read in place, row stride 2624):

        mtp_eh          [M, 12288] x [6144, 12288]
        kv_b_k, kv_b_v  [M, 576]   x [4096, 576]   one half of the augmented kv_b
        kv_b_k_e4m3     [M, 576]   x [4096, 576]   e4m3 out (fp8 attention's K)
        kv_b_v512_e4m3  [M, 512]   x [4096, 512]   e4m3 out (fp8 attention's V, c_kv only)
        q_b_e4m3        [M, 2048]  x [4096, 2048]  e4m3 out (fp8 attention's Q)

    A `SharedDownPrefillRunner` runs shared_down on the shared expert's own prefill
    kernel at 65..8192 rows, every operand 128-byte aligned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # One cell per backend each kernel's schedule table picks at the rows
    # chosen (TGV, tcgen05, persistent), plus a prefill row count for the
    # kernels whose prefill schedule differs in kind. Decode is 1..64 rows
    # (32 sequences x MTP 2); prefill chunks reach max_num_tokens 8192.
    CELLS: tuple[Cell, ...] = (
        Cell(why="qkva, one decode row: TGV split-K", spec=dict(kernel="qkva", rows=1)),
        Cell(
            why="qkva, full decode batch: tcgen05 with the whole weight slice prefetched to L2",
            spec=dict(kernel="qkva", rows=64),
        ),
        Cell(
            why="qkva at max_num_tokens: persistent 2-CTA prefill schedule",
            spec=dict(kernel="qkva", rows=8192),
        ),
        Cell(why="q_b, one decode row: persistent", spec=dict(kernel="q_b", rows=1)),
        Cell(
            why="q_b, full decode batch: tcgen05 with the weight TMA ahead of the PDL wait",
            spec=dict(kernel="q_b", rows=64),
        ),
        Cell(
            why="o_proj at MTP width 2: persistent with the PDL wait/signal role",
            spec=dict(kernel="o_proj", rows=2),
        ),
        Cell(why="o_proj's one TGV row count", spec=dict(kernel="o_proj", rows=8)),
        Cell(
            why="o_proj, full decode batch: persistent rastered along N",
            spec=dict(kernel="o_proj", rows=64),
        ),
        Cell(why="ix_qb, one decode row: persistent", spec=dict(kernel="ix_qb", rows=1)),
        Cell(
            why="ix_qb, full decode batch: tcgen05 split-K 2",
            spec=dict(kernel="ix_qb", rows=64),
        ),
        Cell(
            why="ix_fused, one row: fp32 operands, 8-CTA split-K cluster",
            spec=dict(kernel="ix_fused", rows=1),
        ),
        Cell(why="ix_fused, one full 64-row tile", spec=dict(kernel="ix_fused", rows=64)),
        Cell(
            why="ix_fused at a prefill chunk that ends mid-tile",
            spec=dict(kernel="ix_fused", rows=1099),
        ),
        Cell(why="router, one row: TGV with an fp32 epilogue", spec=dict(kernel="router", rows=1)),
        Cell(
            why="router, full decode batch: tcgen05 with an fp32 epilogue",
            spec=dict(kernel="router", rows=64),
        ),
        Cell(
            why="dense_gate_up at 24 rows, a decode row count most projections do not compile",
            spec=dict(kernel="dense_gate_up", rows=24),
        ),
        Cell(
            why="dense_gate_up at a prefill tail: the swizzled persistent schedule",
            spec=dict(kernel="dense_gate_up", rows=7491),
        ),
        Cell(why="dense_down, one decode row", spec=dict(kernel="dense_down", rows=1)),
        Cell(why="dense_down, full decode batch", spec=dict(kernel="dense_down", rows=64)),
        Cell(
            why="shared_gate_up, one row: TGV split-K 4", spec=dict(kernel="shared_gate_up", rows=1)
        ),
        Cell(
            why="shared_gate_up, full decode batch: tcgen05",
            spec=dict(kernel="shared_gate_up", rows=64),
        ),
        Cell(
            why="shared_down at an odd row count: it compiles every M in 1..64",
            spec=dict(kernel="shared_down", rows=37),
        ),
        Cell(
            why="shared_down at the smallest prefill chunk: persistent",
            spec=dict(kernel="shared_down", rows=1099),
        ),
        Cell(
            why="lm_head, one row: the 2-CTA swapped schedule", spec=dict(kernel="lm_head", rows=1)
        ),
        Cell(
            why="lm_head at 37 rows, a draining batch on the 48-row bucket's schedule",
            spec=dict(kernel="lm_head", rows=37),
        ),
        Cell(why="lm_head, full decode batch", spec=dict(kernel="lm_head", rows=64)),
        Cell(
            why="lm_head at 1024 rows: the direct-orientation context schedule",
            spec=dict(kernel="lm_head", rows=1024),
        ),
        # The prefill GEMM: one compiled instance per row range (split at 2048 rows
        # for the attention projections), any row count inside it. The cells take
        # the smallest row count the target sends there, a mid one, both sides of
        # the split, and the ceiling.
        Cell(
            why="prefill qkva at 65 rows, the first row count past decode: one partial tile",
            spec=dict(kernel="qkva", rows=65, prefill=True),
        ),
        Cell(
            why="prefill q_b at 513 rows, the small-T schedule mid-range, not a tile multiple",
            spec=dict(kernel="q_b", rows=513, prefill=True),
        ),
        Cell(
            why="prefill o_proj at 2048 rows, the last row count of the small-T schedule",
            spec=dict(kernel="o_proj", rows=2048, prefill=True),
        ),
        Cell(
            why="prefill o_proj at 2049 rows, the first of the large-T schedule",
            spec=dict(kernel="o_proj", rows=2049, prefill=True),
        ),
        Cell(
            why="prefill qkva at max_num_tokens, 8192 rows, the large-T schedule's ceiling",
            spec=dict(kernel="qkva", rows=8192, prefill=True),
        ),
        # The dense MLP, the shared expert, the router and the MTP eh projection use
        # the default row-range schedules; one cell each, spread over the range.
        Cell(
            why="prefill dense_gate_up at 65 rows, the first row count past decode",
            spec=dict(kernel="dense_gate_up", rows=65, prefill=True),
        ),
        Cell(
            why="prefill dense_down at max_num_tokens, 8192 rows, K = 3072",
            spec=dict(kernel="dense_down", rows=8192, prefill=True),
        ),
        Cell(
            why="prefill shared_gate_up at 513 rows, N = 1024 (the MTP module's shared expert)",
            spec=dict(kernel="shared_gate_up", rows=513, prefill=True),
        ),
        Cell(
            why="prefill shared_down at 2049 rows, K = 512",
            spec=dict(kernel="shared_down", rows=2049, prefill=True),
        ),
        Cell(
            why="prefill router at 1099 rows: fp32 logits from the prefill GEMM",
            spec=dict(kernel="router", rows=1099, prefill=True),
        ),
        Cell(
            why="prefill kv_b_k at 987 rows on the latent tail of a 2624-wide row (K = 576, "
            "row-strided input)",
            spec=dict(kernel="kv_b_k", rows=987, prefill=True, x_row_stride=2624),
        ),
        Cell(
            why="prefill kv_b_v at 8192 rows, the large-T schedule, row-strided input",
            spec=dict(kernel="kv_b_v", rows=8192, prefill=True, x_row_stride=2624),
        ),
        Cell(
            why="prefill kv_b_k_e4m3 at 7255 rows, row-strided input, e4m3 out",
            spec=dict(kernel="kv_b_k_e4m3", rows=7255, prefill=True, x_row_stride=2624),
        ),
        Cell(
            why="prefill kv_b_v512_e4m3 at 1 row, row-strided input, e4m3 out",
            spec=dict(kernel="kv_b_v512_e4m3", rows=1, prefill=True, x_row_stride=2624),
        ),
        Cell(
            why="prefill q_b_e4m3 at 2049 rows, the second schedule group, e4m3 out",
            spec=dict(kernel="q_b_e4m3", rows=2049, prefill=True),
        ),
        Cell(
            why="prefill mtp_eh at 4100 rows, K = 12288 (the enorm/hnorm concat)",
            spec=dict(kernel="mtp_eh", rows=4100, prefill=True),
        ),
        # The shared expert's own prefill down kernel: one compiled instance on a
        # symbolic row count, 256-row two-CTA tiles. The smallest count it serves, one
        # that ends mid-tile, and the ceiling.
        Cell(
            why="shared-expert prefill down at 65 rows, its first count: one partial tile",
            spec=dict(kernel="shared_down", rows=65, shared_prefill=True),
        ),
        Cell(
            why="shared-expert prefill down at 4609 rows, not a tile multiple",
            spec=dict(kernel="shared_down", rows=4609, shared_prefill=True),
        ),
        Cell(
            why="shared-expert prefill down at 8192 rows, the token budget",
            spec=dict(kernel="shared_down", rows=8192, shared_prefill=True),
        ),
    )
    # MTP3 / MTP5: every decode cell above 64 rows (`MTP_ROWS`), one per kernel and row
    # count; the backend each takes is its kernel module's measured table.
    CELLS += tuple(
        Cell(
            why=f"{kernel} at {rows} rows: 32 sequences x MTP{rows // 32 - 1} width {rows // 32}",
            spec=dict(kernel=kernel, rows=rows),
        )
        for kernel, mtp_rows in MTP_ROWS.items()
        for rows in mtp_rows
    )

    note = """
    The target compiles, per kernel, the row counts in `ROWS` (the kernel's
    own schedule table: 1, 2, 4, 8, 16, 32, 48, 64, plus 24 for the dense MLP,
    every count in 1..64 for shared_down and lm_head (plus 128..8192 in powers
    of two for lm_head, whose rows are a step's logits rows), and the prefill
    chunks 1099, 3282,
    7491, 7527, 7554, 7578, 8192; `IX_FUSED_ROWS` for ix_fused) through
    `build_all`, which leaves out a row count that fails to compile. Every
    other row count, and any that was left out, goes to `cublas_mm` at the
    target's call site; the op only ever runs its own kernel.

    `MTP_ROWS` (the kernel modules' `MTP_DECODE_M_VALUES`) are decode cells above
    64 rows: 32 sequences at MTP3 (128) or MTP5 (192). `build_all` leaves them out
    unless asked; the target builds them only for an engine drafting more than one
    token and sends them a decode step's rows only, so a context step of the same
    row count keeps the paths below.

    Above 64 rows, a row count with no decode runner goes to the prefill GEMM
    when `build_prefill` built that projection's entry (`build_prefill_all` leaves
    out an entry the vendored kernel does not have, or whose output dtype is not
    the projection's), else to `cublas_mm`. ix_fused's context rows take the
    indexer's own split op (`cute_dsl_indexer_pre_proj_split`), and ix_qb's stay
    on cuBLAS. The cells drive every entry the target builds: qkva, q_b, o_proj,
    the dense MLP, the shared expert, the router (fp32 out) and mtp_eh.

    A trunk layer's shared expert at a context row count takes
    `build_shared_down_prefill`'s runner when it and the shared gate/up prefill
    runner both built, writing straight into the routed output the second grouped
    GEMM accumulates onto; the MTP layer's shared expert keeps `_project`.

    A decode runner is pinned to the row count it was built for and raises on
    any other; the prefill runners take any count in their range. `at_rows`
    re-pins an MTP cell's runner to fewer rows: the same compiled instance at a
    runtime row count, each row bitwise the cell's (the target's other MTP3 /
    MTP5 decode row counts, 65..191, on the next cell up). No runner binds the
    weight: one runner serves every layer.

    The bf16 kernels check their operands' layout on the first call only (the
    one `build` makes). A later call with a non-contiguous or misaligned
    operand is not checked by the kernel; `is_valid` states that contract.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: Runner,
    ) -> torch.Tensor:
        runner.launch(x, weight, out)
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: Runner,
    ) -> torch.Tensor:
        """`x @ weight.T`, rounded once to `out`'s dtype. `out` is not written;
        `runner` is ignored.

        bf16 outputs accumulate in true fp32 (TF32 off, see
        `true_fp32_matmul`); fp32 outputs accumulate in fp64, so the reference
        stays the more accurate side. For ix_fused this is the exact product of
        the fp32 operands, with no TF32 rounding: the test's band covers the
        kernel's operand rounding.
        """
        acc = torch.float64 if out.dtype == torch.float32 else torch.float32
        with true_fp32_matmul():
            res = x.to(acc) @ weight.to(acc).t()
        if out.dtype == torch.float8_e4m3fn:
            # The kernel converts nearest with saturation; torch's cast does not saturate.
            res = res.clamp(-448.0, 448.0)
        return res.to(out.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: Runner,
    ) -> None:
        # The bf16 kernels validate shapes, strides and alignment when they
        # compile and never again: after `build` nothing checks a strided or
        # misaligned operand. The square dense_gate_up is where that bites: the
        # column-major view `cublas_mm` takes has the right shape. The prefill
        # GEMM checks x and out on every call, but a weight only the first time
        # it meets that weight's address: a view of the same storage reuses the
        # cached operand. It takes every operand as 16-byte aligned.
        n, k = runner.n, runner.k
        if isinstance(runner, PrefillProjectionRunner):
            rows, aligns = int(x.shape[0]), _PREFILL_ALIGN
            assert 1 <= rows <= runner.max_rows, (
                f"the prefill GEMM serves 1..{runner.max_rows} rows, got {rows}"
            )
        elif isinstance(runner, SharedDownPrefillRunner):
            # The shared expert's prefill kernel checks rows, shapes, dtypes,
            # alignment and overlap on every call and raises: nothing to add.
            return
        else:
            rows, aligns = runner.rows, shape_of(runner.kernel).align
        assert tuple(x.shape) == (rows, k), (
            f"x must be [{rows}, {k}] for this runner, got {tuple(x.shape)}"
        )
        assert tuple(weight.shape) == (n, k), (
            f"weight must be the row-major [N, K] = [{n}, {k}] buffer, got {tuple(weight.shape)}"
        )
        assert tuple(out.shape) == (rows, n), f"out must be [{rows}, {n}], got {tuple(out.shape)}"
        strided_x = getattr(runner, "row_strided_input", False)
        for name, t in (("x", x), ("weight", weight), ("out", out)):
            if name == "x" and strided_x:
                assert x.stride(1) == 1 and x.stride(0) % 8 == 0, (
                    f"x must have unit inner stride and a row stride that is a multiple of 8, "
                    f"got {tuple(x.stride())}"
                )
                continue
            assert t.is_contiguous(), (
                f"{name} must be contiguous (a column-major weight view must be passed as .t())"
            )
        assert x.dtype == weight.dtype == runner.in_dtype, (
            f"x and weight must be {runner.in_dtype}, got {x.dtype} and {weight.dtype}"
        )
        # The router writes fp32 logits; after the first call not every
        # backend checks that `out` is fp32.
        assert out.dtype == runner.out_dtype, (
            f"{runner.kernel} writes {runner.out_dtype}; out is {out.dtype}"
        )
        for name, t, align in zip(("x", "weight", "out"), (x, weight, out), aligns):
            assert t.data_ptr() % align == 0, f"{name} must be {align}-byte aligned"


cute_dsl_projection = CuteDslProjection()
