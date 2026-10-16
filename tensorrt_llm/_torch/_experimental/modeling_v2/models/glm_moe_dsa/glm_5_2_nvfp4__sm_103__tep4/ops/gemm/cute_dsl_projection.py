# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's dense projections: each on its checkpoint-specialised CuTe DSL kernel at the
decode row counts, and on the shared prefill GEMM at context row counts."""

from __future__ import annotations

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
        # A projection only the prefill GEMM serves (mtp_eh): its entry's contract.
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


#: Every row count each kernel has a schedule for. `build_all` compiles these.
ROWS: dict[str, tuple[int, ...]] = {
    name: IX_FUSED_ROWS if name == "ix_fused" else tuple(module.SUPPORTED_M_VALUES)
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
    on a zero input. Raises if `rows` is not in `ROWS[kernel]` or the
    compile fails.
    """
    shape = shape_of(kernel)
    rows = int(rows)
    if rows not in ROWS[kernel]:
        raise ValueError(f"{kernel}: no schedule for {rows} rows; compiled rows are {ROWS[kernel]}")
    x = torch.zeros([rows, shape.k], dtype=shape.in_dtype, device=weight.device)
    out = torch.empty([rows, shape.n], dtype=shape.out_dtype, device=weight.device)
    with torch.cuda.device(weight.device):
        launch = _MODULES[kernel].build()
        launch(x, weight, out)
    return ProjectionRunner(kernel, rows, shape.n, shape.k, shape.in_dtype, shape.out_dtype, launch)


def build_all(
    kernel: str, weight: torch.Tensor, rows: tuple[int, ...] | None = None
) -> dict[int, ProjectionRunner]:
    """`build` at every row count in `rows` (default `ROWS[kernel]`).

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


def build_prefill(kernel: str, weight: torch.Tensor) -> PrefillProjectionRunner:
    """Build the prefill GEMM for `kernel` against `weight`. Call at load.

    `weight` is the row-major `[N, K]` bf16 weight of any one layer; the runner
    takes the weight per call, so one runner serves every layer. Compiles every
    row range of the entry's schedule table (two for the attention projections),
    shared process-wide per entry, then launches once per range on zeros, so no
    context step meets a first compile or a first module load.

    Raises if the vendored kernel has no entry for `kernel`, or if the entry's
    output dtype is not the projection's (the router's logits are fp32): the
    target then keeps cuBLAS for that projection's context rows.
    """
    entry = PREFILL_ENTRIES[kernel]
    abi = _prefill.ABIS.get(entry)
    if abi is None:
        raise ValueError(f"{kernel}: the vendored prefill GEMM has no {entry!r} entry")
    expected = torch.bfloat16 if kernel == "mtp_eh" else shape_of(kernel).out_dtype
    if abi.out_dtype != expected:
        raise ValueError(f"{kernel}: the prefill GEMM writes {abi.out_dtype}, not {expected}")
    dev = weight.device
    with torch.cuda.device(dev):
        launch = _prefill.build(entry, weight)
        low = 1
        for high, _ in abi.schedule_table():
            x = torch.zeros([low, abi.in_features], dtype=torch.bfloat16, device=dev)
            out = torch.empty([low, abi.out_features], dtype=abi.out_dtype, device=dev)
            launch(x, weight, out)
            low = high + 1
    return PrefillProjectionRunner(
        kernel,
        abi.out_features,
        abi.in_features,
        torch.bfloat16,
        abi.out_dtype,
        PREFILL_MAX_ROWS,
        launch,
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

    The ten decode kernels, by the name the target uses (per-rank shapes, TP4):

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

    A `PrefillProjectionRunner` runs the same product on the prefill GEMM at any row
    count up to 8192: bf16 operands, fp32 accumulation, the entry's output dtype.
    Its entries (`PREFILL_ENTRIES`) are the bf16 projections above except ix_qb, plus
    the MTP front's replicated eh_proj:

        mtp_eh          [M, 12288] x [6144, 12288]

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

    note = """
    The target compiles, per kernel, the row counts in `ROWS` (the kernel's
    own schedule table: 1, 2, 4, 8, 16, 32, 48, 64, plus 24 for the dense MLP,
    every count in 1..64 for shared_down, and the prefill chunks 1099, 3282,
    7491, 7527, 7554, 7578, 8192; `IX_FUSED_ROWS` for ix_fused) through
    `build_all`, which leaves out a row count that fails to compile. Every
    other row count, and any that was left out, goes to `cublas_mm` at the
    target's call site; the op only ever runs its own kernel.

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
    any other; the prefill runners take any count in their range. No runner
    binds the weight: one runner serves every layer.

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
            return (x.to(acc) @ weight.to(acc).t()).to(out.dtype)

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
        for name, t in (("x", x), ("weight", weight), ("out", out)):
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
