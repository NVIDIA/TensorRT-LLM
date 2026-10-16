# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's MoE router GEMM, `[rows, 6144] x [256, 6144]` to fp32 logits, on three CuTe DSL
kernels chosen by row count: SIMT (1..5), tensor-core split-K partials (6..64), and the
prefill router (65..8192)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cutlass.torch as cutlass_torch
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.moe import moe_frontend_router as _vec2
from ...kernels.moe import moe_frontend_router_tc as _tc
from ...kernels.moe import moe_prefill_frontend_schedule as _sched
from ...kernels.moe import moe_prefill_router_partials as _prefill

EXPERTS = 256
HIDDEN = 6144
#: The SIMT router serves 1..5 rows and writes `[rows, 256]`; the tensor-core router
#: 6..64 and writes `[rows, SPLIT_K, 256]` partials that the fused top-k sums in
#: registers.
SIMT_MAX_ROWS = 5
SPLIT_K = _tc._SPLIT_K
DECODE_MAX_ROWS = 64
#: The prefill router's row domain and its K split: one plane, `[rows, 1, 256]`.
PREFILL_MIN_ROWS = DECODE_MAX_ROWS + 1
PREFILL_MAX_ROWS = 8192
PREFILL_SPLIT_K = 1


def decode_splits(rows: int) -> int:
    """The K partials the decode router writes at `rows`: 1 means a 2-D `[rows, 256]`
    output, otherwise `[rows, splits, 256]`."""
    return 1 if rows <= SIMT_MAX_ROWS else SPLIT_K


def output_shape(rows: int) -> tuple[int, ...]:
    """The fp32 output a runner writes at `rows`."""
    if rows > DECODE_MAX_ROWS:
        return (rows, PREFILL_SPLIT_K, EXPERTS)
    if rows <= SIMT_MAX_ROWS:
        return (rows, EXPERTS)
    return (rows, SPLIT_K, EXPERTS)


@dataclass(frozen=True)
class RouterRunner:
    """A compiled router kernel.

    `kind` is "simt" or "tc_partials" (pinned to `rows`), or "prefill" (any row count in
    `PREFILL_MIN_ROWS..PREFILL_MAX_ROWS`, `rows` None). `launch(x, weight, out)` is the
    kernel's call, already compiled.
    """

    kind: str
    rows: int | None
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]


def _simt_launch(compiled) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    def launch(x, weight, out):
        # The kernel's ABI has a zero plane it leaves untouched without `zero_output`;
        # the logits stand in for it, as the kernel's own runner passes them.
        compiled(
            _vec2._as_tensor(x, 1),
            _vec2._as_tensor(weight, 1),
            _vec2._as_tensor(out, 1),
            _vec2._as_tensor(out, 1),
            cutlass_torch.current_stream(),
        )

    return launch


def _tc_launch(gemm) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    k_split = HIDDEN // SPLIT_K

    def launch(x, weight, out):
        gemm(
            int(x.shape[0]),
            EXPERTS,
            k_split,
            SPLIT_K,
            x.data_ptr(),
            weight.data_ptr(),
            out.permute(0, 2, 1),
            HIDDEN,
            k_split,
            HIDDEN,
            k_split,
        )

    return launch


def build(rows: int, device) -> RouterRunner:
    """Compile the decode router for exactly `rows` rows (1..64) and launch it once on
    zeros. Call at load.

    Rows 1..5 take the SIMT kernel (one expert per CTA), 6..64 the tensor-core split-K
    kernel; both with PDL as the engine sets it. Costs one CuTe DSL compile per row
    count (cached per process by the kernel modules). Raises if `rows` is outside 1..64.
    """
    rows = int(rows)
    if not 1 <= rows <= DECODE_MAX_ROWS:
        raise ValueError(f"the decode router serves 1..{DECODE_MAX_ROWS} rows, got {rows}")
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    w = torch.zeros(EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    out = torch.zeros(output_shape(rows), dtype=torch.float32, device=device)
    with torch.cuda.device(device):
        if rows <= SIMT_MAX_ROWS:
            runner = RouterRunner(
                "simt", rows, _simt_launch(_vec2.compile_router_vec2(x, w, out, use_pdl=_USE_PDL))
            )
        else:
            runner = RouterRunner(
                "tc_partials",
                rows,
                _tc_launch(_tc.compile_router_tc_partials(x, w, out, use_pdl=_USE_PDL)),
            )
        runner.launch(x, w, out)
    return runner


def build_all(device, rows=range(1, DECODE_MAX_ROWS + 1)) -> dict[int, RouterRunner]:
    """`build` at every row count in `rows` (default 1..64); `{rows: runner}`. Raises on
    the first failure: the decode front end has no partial form."""
    return {int(r): build(r, device) for r in rows}


def build_prefill(device) -> RouterRunner:
    """Compile the prefill router once per schedule geometry its row domain selects
    (`propose_schedule(rows)`: 64x64 tiles with 12 stages up to 4096 rows, 128x128 with 6
    above), K unsplit, at most one CTA per SM, PDL as the engine sets it; launch each once
    on zeros. Call at load. The runner picks the geometry per call by row count.

    Raises if a compile fails: the target then keeps cuBLAS for the context rows.
    """
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    handles: dict[tuple[int, int, int], Callable[..., None]] = {}
    first: dict[tuple[int, int, int], int] = {}
    for rows in range(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS + 1):
        s = _sched.propose_schedule(rows)
        first.setdefault((s.router_m, s.router_n, s.router_stages), rows)
    w = torch.zeros(EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)
    with torch.cuda.device(device):
        for (m, n, st), rows in first.items():
            h = _prefill.compile_router(
                split_k=PREFILL_SPLIT_K,
                physical_m=m,
                tile_n=n,
                stages=st,
                maximum_ctas=num_sms,
                use_pdl=_USE_PDL,
            )
            handles[(m, n, st)] = h
            x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device=device)
            out = torch.zeros(output_shape(rows), dtype=torch.float32, device=device)
            _prefill.validate_inputs(x, w, out, PREFILL_SPLIT_K)
            h(x, w, out, rows)

    def launch(x, weight, out):
        rows = int(x.shape[0])
        s = _sched.propose_schedule(rows)
        handles[(s.router_m, s.router_n, s.router_stages)](x, weight, out, rows)

    return RouterRunner("prefill", None, launch)


class CuteDslMoeRouter(OpWrapper):
    """The MoE router GEMM: `out = x @ weight.T` in fp32, as K partials. Returns `out`.

    `x` is bf16 `[rows, 6144]`, `weight` the row-major bf16 `[256, 6144]` gate weight,
    `out` a caller-owned fp32 buffer of `output_shape(rows)`:

        simt         1..5 rows     [rows, 256]          the logits
        tc_partials  6..64 rows    [rows, 4, 256]       four K-quarter partials
        prefill      65..8192 rows [rows, 1, 256]       the logits, one plane

    The partials are not summed here: the fused top-k adds them in registers, in
    increasing split order, before the sigmoid.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row on the SIMT router", spec=dict(rows=1)),
        Cell(why="5 rows, the SIMT router's ceiling", spec=dict(rows=5)),
        Cell(why="6 rows, the first tensor-core split-K row count", spec=dict(rows=6)),
        Cell(why="the full decode batch on split-K partials", spec=dict(rows=64)),
        Cell(why="65 rows, the first prefill row count: 64x64 tiles", spec=dict(rows=65)),
        Cell(why="4096 rows, the last of the 64x64 prefill geometry", spec=dict(rows=4096)),
        Cell(why="4097 rows, the first of the 128x128 prefill geometry", spec=dict(rows=4097)),
        Cell(why="8192 rows, the token budget", spec=dict(rows=8192)),
    )

    note = """
    The target builds the decode router at every row count 1..64 (`build_all`) and the
    prefill router once (`build_prefill`). The trunk's decode row counts always take the
    decode router; a context row count takes the prefill router when it built and
    otherwise the stock projection (`cute_dsl_projection` "router" prefill GEMM or
    cuBLAS) at the call site. The MTP layer runs this op only at its decode widths; its
    context steps take the stock projection.

    The decode kernels' compiled handles skip the shape checks the kernel modules' own
    runners repeat per call; `is_valid` states them. The router reads x under PDL: the
    target hands it the collective's output in place when that is 128-byte aligned and
    contiguous, and otherwise stages it, since a staging copy would cut the PDL edge.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: RouterRunner,
    ) -> torch.Tensor:
        runner.launch(x, weight, out)
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: RouterRunner,
    ) -> torch.Tensor:
        """Each K slice's product in fp64, rounded once to fp32, in `out`'s shape: slice
        `s` of `out.shape[1]` covers K columns `[s * 6144 / splits, (s + 1) * 6144 /
        splits)`. `out` is not written; `runner` is ignored."""
        rows = int(x.shape[0])
        splits = 1 if out.dim() == 2 else int(out.shape[1])
        xs = x.double().reshape(rows, splits, HIDDEN // splits)
        ws = weight.double().reshape(EXPERTS, splits, HIDDEN // splits)
        partials = torch.einsum("rsk,esk->rse", xs, ws).float()
        return partials.reshape(out.shape)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: RouterRunner,
    ) -> None:
        # The compiled handles read raw pointers and dynamic layouts: a strided view, a
        # wrong output plane count or a misaligned buffer is read or written as if it
        # matched.
        rows = int(x.shape[0])
        if runner.kind == "prefill":
            assert PREFILL_MIN_ROWS <= rows <= PREFILL_MAX_ROWS, (
                f"the prefill router serves {PREFILL_MIN_ROWS}..{PREFILL_MAX_ROWS} rows, got {rows}"
            )
        else:
            assert rows == runner.rows, f"this runner is compiled for {runner.rows} rows"
        assert tuple(x.shape) == (rows, HIDDEN) and x.dtype == torch.bfloat16
        assert tuple(weight.shape) == (EXPERTS, HIDDEN) and weight.dtype == torch.bfloat16, (
            "weight must be the row-major bf16 [256, 6144] gate weight"
        )
        assert tuple(out.shape) == output_shape(rows) and out.dtype == torch.float32, (
            f"out must be fp32 {output_shape(rows)}, got {out.dtype} {tuple(out.shape)}"
        )
        for name, t in (("x", x), ("weight", weight), ("out", out)):
            assert t.is_contiguous(), f"{name} must be contiguous"
        out_align = 32 if runner.kind == "prefill" else 16
        for name, t, align in (("x", x, 16), ("weight", weight, 16), ("out", out, out_align)):
            assert t.data_ptr() % align == 0, f"{name} must be {align}-byte aligned"


cute_dsl_moe_router = CuteDslMoeRouter()
