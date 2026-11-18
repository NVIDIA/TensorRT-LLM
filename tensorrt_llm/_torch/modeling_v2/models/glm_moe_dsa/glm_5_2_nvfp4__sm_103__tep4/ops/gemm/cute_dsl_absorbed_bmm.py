# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absorbed-MLA K-B and V-B batched GEMMs on their CuTe DSL kernels: the decode kernels
at their row-count groups, the prefill kernels at every other row count."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

from ...kernels.gemm import self_attn_k_b_bmm_h16_m64_n512_k192_bf16 as _k_b
from ...kernels.gemm import self_attn_prefill_kv_b_bmm_bf16 as _prefill
from ...kernels.gemm import self_attn_v_b_bmm_h16_m64_n256_k512_bf16 as _v_b

_MODULES = {"k_b": _k_b, "v_b": _v_b}

#: The two decode row-count groups; one compiled object serves each.
GROUPS = ("m_le_8", "m_ge_16")
#: The MTP3 / MTP5 decode row counts (128, 192: 32 sequences x 4 / 6), one compiled object.
#: The bf16 cells, the fp8 q-base K-B beside the rope kernel of widths 4 / 6
#: (`build_decode_fp8_q_base`), and the V-B MXF8 side output writing the wide decode
#: o_proj's 192-row operand (`build_v_b_mxf8_side`).
MTP_GROUPS = ("m_ge_128",)
#: The row counts that object also launches at (runtime M): every MTP3 / MTP5 decode row
#: count above 64, a batch of fewer than 32 sequences included; each row bitwise what the
#: 128 / 192 cells compute for it. Its runner's `rows` stays (128, 192), the cells.
MTP_RUNTIME_ROWS = _k_b.MTP_GROUP_RUNTIME_ROWS
#: The row counts the M >= 16 group's object (`GROUPS[1]`) also launches at (runtime M):
#: every decode row count in 9..64, each row bitwise what that group's cells compute for it.
#: For the MTP3 / MTP5 decode counts without a cell (12 / 18); its runner's `rows` stays the
#: cells (16, 32, 48, 64). The V-B MXF8 side output then writes those rows of o_proj's
#: operand (and the zero-filled tail of the 32-row tile as quantized zeros).
DECODE_RUNTIME_ROWS = _k_b.DECODE_GROUP_RUNTIME_ROWS
#: The prefill kernels serve every row count 1..PREFILL_MAX_ROWS.
PREFILL_MAX_ROWS = _prefill.MAX_ROWS


@dataclass(frozen=True)
class _Layout:
    """`a` is `[heads, M, k]` sliced from token-major `[M, heads, a_row]`
    storage, `out` is `[heads, M, n]` sliced from `[M, heads, out_row]`."""

    heads: int
    n: int
    k: int
    a_row: int
    out_row: int


#: K-B reads the 192-wide nope half of q `[M, 16, 256]` and writes the
#: 512-wide latent half of the fused query `[M, 16, 576]`. V-B reads the
#: attention output `[M, 16, 512]` and writes v `[M, 16, 256]`.
LAYOUTS = {
    "k_b": _Layout(heads=16, n=512, k=192, a_row=256, out_row=576),
    "v_b": _Layout(heads=16, n=256, k=512, a_row=512, out_row=256),
}


@dataclass(frozen=True)
class AbsorbedBmmRunner:
    """One compiled object of one kernel; serves every row count in `rows`.
    `group` is one of `GROUPS`, or "prefill"."""

    kernel: str
    group: str
    rows: Sequence[int]
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


def _views(kernel: str, m: int, dev: torch.device):
    """Zero `(a, weight, out)` for `m` rows, in the strided layouts the target passes."""
    lay, bf16 = LAYOUTS[kernel], torch.bfloat16
    a = torch.zeros([m, lay.heads, lay.a_row], dtype=bf16, device=dev)[..., : lay.k]
    weight = torch.zeros([lay.heads, lay.n, lay.k], dtype=bf16, device=dev)
    out = torch.empty([m, lay.heads, lay.out_row], dtype=bf16, device=dev)[..., : lay.n]
    return a.transpose(0, 1), weight, out.transpose(0, 1)


def build(kernel: str, group: str, device_id: int) -> AbsorbedBmmRunner:
    """Compile `kernel` ("k_b" or "v_b") for row-count group `group` (one of `GROUPS`
    or `MTP_GROUPS`) on `device_id`. Call at load.

    Compiles against the strided views the target passes (see `LAYOUTS`) and
    launches once on zeros, so nothing compiles in the forward. Costs one CuTe
    DSL compile per (kernel, group), cached by the kernel module. Raises on an
    unknown kernel or group.
    """
    module = _MODULES[kernel]
    if group not in GROUPS + MTP_GROUPS:
        raise ValueError(f"{kernel}: group must be one of {GROUPS + MTP_GROUPS}, got {group!r}")
    rows = tuple(module.rows_of_group(group))
    m = rows[-1]
    with torch.cuda.device(device_id):
        launch = module.build(active_m=m)
        launch(*_views(kernel, m, torch.device("cuda", device_id)))
    return AbsorbedBmmRunner(kernel, group, rows, launch)


def build_decode_fp8_q_base(group: str, device_id: int) -> AbsorbedBmmRunner:
    """The decode K-B for row-count group `group` with the e4m3 q-base epilogue, releasing
    its PDL dependent right after its activation wait. Call at load.

    The epilogue rounds each output FP32 -> BF16 -> E4M3 (nearest, saturating) into the
    uint8 `[16, M, 512]` transpose view of the sparse decode MLA's `[M, 16, 576]` fp8
    query: the bytes the rope generation kernel would write from the bf16 K-B output. The
    dependent it releases early is that rope kernel built with `q_base_from_kb`, which
    does not read this output and waits for it only at its end. Launched once on zeros.
    Costs one CuTe DSL compile per group. The MTP group's object (`MTP_GROUPS`, the rope
    kernel of widths 4 / 6 beside it) also launches at every row count of
    `MTP_RUNTIME_ROWS`, as the plain MTP cells do."""
    if group not in GROUPS + MTP_GROUPS:
        raise ValueError(f"k_b: group must be one of {GROUPS + MTP_GROUPS}, got {group!r}")
    rows = tuple(_k_b.rows_of_group(group))
    m, lay, dev = rows[-1], LAYOUTS["k_b"], torch.device("cuda", device_id)
    with torch.cuda.device(device_id):
        launch = _k_b.build(emit_fp8_q_base=True, active_m=m, early_signal=True)
        a, weight, _ = _views("k_b", m, dev)
        quant = torch.zeros([m, lay.heads, lay.out_row], dtype=torch.uint8, device=dev)
        launch(a, weight, quant[..., : lay.n].transpose(0, 1))
        torch.cuda.synchronize(device_id)
    return AbsorbedBmmRunner("k_b", f"fp8_{group}", rows, launch)


def build_v_b_mxf8_side(
    group: str, device_id: int, operand: tuple[torch.Tensor, torch.Tensor]
) -> AbsorbedBmmRunner:
    """The decode V-B for row-count group `group` whose epilogue also writes its bf16
    output `[M, 4096]` (token-major) as an MXF8 operand -- the activation quant's bytes --
    into `operand = (xq [R, 4096] uint8, sfa)`: the decode o_proj launcher's operand
    (`DecodeMxf8Runner.operand`; R = 64 for `GROUPS`, 192 for the MTP group, which launches
    at every row count of `MTP_RUNTIME_ROWS` and writes the zero-filled rows of its last
    64-row tile past M as quantized zeros). Its addresses are baked into the kernel, so the
    buffers must live as long as the runner and never move. The bf16 output is the plain
    cell's. Launched once on zeros at load. Costs one CuTe DSL compile per group."""
    if group not in GROUPS + MTP_GROUPS:
        raise ValueError(f"v_b: group must be one of {GROUPS + MTP_GROUPS}, got {group!r}")
    xq, sfa = operand
    lay = LAYOUTS["v_b"]
    width = lay.heads * lay.n
    held = 64 if group in GROUPS else max(MTP_RUNTIME_ROWS)
    if xq.dtype != torch.uint8 or tuple(xq.shape) != (held, width) or not xq.is_contiguous():
        raise ValueError(
            f"v_b: group {group}'s operand must be a contiguous uint8 [{held}, {width}]"
        )
    side = _v_b.Mxf8SideOutput(xq.data_ptr(), sfa.data_ptr(), width, lay.n, held)
    rows = tuple(_v_b.rows_of_group(group))
    with torch.cuda.device(device_id):
        launch = _v_b.build(active_m=rows[-1], mxf8_side=side)
        launch(*_views("v_b", rows[-1], torch.device("cuda", device_id)))
        torch.cuda.synchronize(device_id)
    return AbsorbedBmmRunner("v_b", f"x8_{group}", rows, launch)


def build_prefill(kernel: str, device_id: int) -> AbsorbedBmmRunner:
    """The prefill kernel for `kernel` ("k_b" or "v_b") on `device_id`. Call at load.

    One launcher serves every row count 1..PREFILL_MAX_ROWS through two schedules,
    split at `SCHEDULE_SPLIT_ROWS` (2048) rows. Each compiles on its first launch,
    so both are driven here once, at the largest row count each serves, on zero
    views in the target's layout. Costs two CuTe DSL compiles and a transient
    ~0.2 GB of zeros. Raises on an unknown kernel.
    """
    dev = torch.device("cuda", device_id)
    with torch.cuda.device(device_id):
        launch = _prefill.build(kernel)
        for m in (_prefill.SCHEDULE_SPLIT_ROWS, PREFILL_MAX_ROWS):
            launch(*_views(kernel, m, dev))
        torch.cuda.synchronize(device_id)
    return AbsorbedBmmRunner(kernel, "prefill", range(1, PREFILL_MAX_ROWS + 1), launch)


def build_prefill_fp8_q_base(device_id: int) -> AbsorbedBmmRunner:
    """The prefill K-B with the e4m3 q-base epilogue on `device_id`. Call at load.

    Same kernel, schedules and accumulation as `build_prefill("k_b", ...)`; the epilogue
    rounds each output FP32 -> BF16 -> E4M3 (nearest, saturating) and stores the bytes
    into `out`, the uint8 `[16, M, 512]` transpose view of the FMHA's `[M, 16, 576]` fp8
    query. That is the conversion the context rope/append kernel applies to the bf16
    K-B output, so the fp8 query is byte-identical, and the bf16 `fused_q[..., :512]` the
    rope kernel would otherwise re-read (129 MB at 8k rows) is never written. Both
    schedules are launched once here on zeros. Costs two CuTe DSL compiles.
    """
    dev = torch.device("cuda", device_id)
    lay = LAYOUTS["k_b"]
    with torch.cuda.device(device_id):
        launch = _prefill.build("k_b", emit_fp8_q_base=True)
        for m in (_prefill.SCHEDULE_SPLIT_ROWS, PREFILL_MAX_ROWS):
            a, weight, _ = _views("k_b", m, dev)
            quant = torch.zeros([m, lay.heads, lay.out_row], dtype=torch.uint8, device=dev)
            launch(a, weight, quant[..., : lay.n].transpose(0, 1))
        torch.cuda.synchronize(device_id)
    return AbsorbedBmmRunner("k_b", "prefill_fp8", range(1, PREFILL_MAX_ROWS + 1), launch)


class CuteDslAbsorbedBmm(OpWrapper):
    """`out[h] = a[h] @ weight[h].T` for every head, in place. Returns `out`.

    `a` is `[16, M, K]` and `out` `[16, M, N]`, both the `transpose(0, 1)`
    views of token-major storage that `bmm_out` takes (see `LAYOUTS`).
    `weight` is `[16, N, K]` contiguous: the transpose of the `[16, K, N]`
    operand `bmm_out` takes, which for K-B is a copy the target makes at load.
    Only the first N columns of each `out` row are written.

        k_b   [16, M, 192] x [16, 512, 192] -> [16, M, 512]
        v_b   [16, M, 512] x [16, 256, 512] -> [16, M, 256]
    """

    ARCHS = frozenset({Arch.SM_103})

    # Two compiled objects per kernel serve rows (2, 4, 8) and (16, 32, 48, 64).
    # Cells take each group's ends, plus 48: the one row count that is not a
    # multiple of the large group's 32-row tile. The prefill cells take the row
    # counts the decode groups leave out on both sides of 64 (a mixed step's
    # generation rows are 1..63), and both sides of the prefill split at 2048.
    CELLS: tuple[Cell, ...] = (
        Cell(why="K-B at MTP width 2, the small group's smallest", spec=dict(kernel="k_b", rows=2)),
        Cell(why="K-B at the small group's largest", spec=dict(kernel="k_b", rows=8)),
        Cell(why="K-B at the large group's smallest", spec=dict(kernel="k_b", rows=16)),
        Cell(why="K-B at 48 rows, a partial last row tile", spec=dict(kernel="k_b", rows=48)),
        Cell(why="K-B at the full decode batch, 32 x MTP 2", spec=dict(kernel="k_b", rows=64)),
        Cell(why="V-B at MTP width 2", spec=dict(kernel="v_b", rows=2)),
        Cell(why="V-B at the small group's largest", spec=dict(kernel="v_b", rows=8)),
        Cell(why="V-B at 48 rows, a partial last row tile", spec=dict(kernel="v_b", rows=48)),
        Cell(why="V-B at the full decode batch", spec=dict(kernel="v_b", rows=64)),
        Cell(why="K-B at MTP3's decode batch, 32 x 4 rows", spec=dict(kernel="k_b", rows=128)),
        Cell(why="K-B at MTP5's decode batch, 32 x 6 rows", spec=dict(kernel="k_b", rows=192)),
        Cell(why="V-B at MTP3's decode batch, 32 x 4 rows", spec=dict(kernel="v_b", rows=128)),
        Cell(why="V-B at MTP5's decode batch, 32 x 6 rows", spec=dict(kernel="v_b", rows=192)),
        Cell(
            why="prefill K-B at one row: a decode step's lone row and a mixed step's smallest",
            spec=dict(kernel="k_b", rows=1, prefill=True),
        ),
        Cell(
            why="prefill V-B at 63 rows: a mixed step's generation rows, one under a tile",
            spec=dict(kernel="v_b", rows=63, prefill=True),
        ),
        Cell(
            why="prefill K-B at 65 rows, the first context row count past decode",
            spec=dict(kernel="k_b", rows=65, prefill=True),
        ),
        Cell(
            why="prefill V-B at 2048 rows, the small-T schedule's last row count",
            spec=dict(kernel="v_b", rows=2048, prefill=True),
        ),
        Cell(
            why="prefill K-B at 2049 rows, the large-T schedule's first, a ragged tile",
            spec=dict(kernel="k_b", rows=2049, prefill=True),
        ),
        Cell(
            why="prefill V-B at max_num_tokens, 8192 rows",
            spec=dict(kernel="v_b", rows=8192, prefill=True),
        ),
        Cell(
            why="decode V-B writing o_proj's MXF8 operand at MTP width 2",
            spec=dict(kernel="v_b", rows=2, x8=True),
        ),
        Cell(
            why="decode V-B writing o_proj's MXF8 operand at 48 rows, a partial tile",
            spec=dict(kernel="v_b", rows=48, x8=True),
        ),
        Cell(
            why="decode V-B writing o_proj's MXF8 operand at the full decode batch",
            spec=dict(kernel="v_b", rows=64, x8=True),
        ),
        Cell(
            why="MTP3 decode V-B writing the wide o_proj's MXF8 operand, 32 x 4 rows",
            spec=dict(kernel="v_b", rows=128, x8=True),
        ),
        Cell(
            why="MTP5 decode V-B writing the wide o_proj's MXF8 operand, 32 x 6 rows",
            spec=dict(kernel="v_b", rows=192, x8=True),
        ),
        Cell(
            why="MTP5 decode V-B writing the wide o_proj's operand at 150 rows, 25 x 6",
            spec=dict(kernel="v_b", rows=150, x8=True),
        ),
        Cell(
            why="decode fp8 q-base K-B at MTP width 2, releasing its dependent early",
            spec=dict(kernel="k_b", rows=2, fp8=True, decode=True),
        ),
        Cell(
            why="decode fp8 q-base K-B at the small group's largest",
            spec=dict(kernel="k_b", rows=8, fp8=True, decode=True),
        ),
        Cell(
            why="decode fp8 q-base K-B at 48 rows, a partial last row tile",
            spec=dict(kernel="k_b", rows=48, fp8=True, decode=True),
        ),
        Cell(
            why="decode fp8 q-base K-B at the full decode batch",
            spec=dict(kernel="k_b", rows=64, fp8=True, decode=True),
        ),
        Cell(
            why="MTP-group fp8 q-base K-B at MTP3's decode batch, 32 x 4 rows",
            spec=dict(kernel="k_b", rows=128, fp8=True, decode=True),
        ),
        Cell(
            why="MTP-group fp8 q-base K-B at MTP5's decode batch, 32 x 6 rows",
            spec=dict(kernel="k_b", rows=192, fp8=True, decode=True),
        ),
        Cell(
            why="fp8 q-base K-B at 65 rows, the first context row count past decode",
            spec=dict(kernel="k_b", rows=65, prefill=True, fp8=True),
        ),
        Cell(
            why="fp8 q-base K-B at 2049 rows, the large-T schedule's first, a ragged tile",
            spec=dict(kernel="k_b", rows=2049, prefill=True, fp8=True),
        ),
        Cell(
            why="fp8 q-base K-B at max_num_tokens, 8192 rows",
            spec=dict(kernel="k_b", rows=8192, prefill=True, fp8=True),
        ),
    )

    note = """
    The target builds both decode groups of both kernels at load (row counts 2,
    4, 8, 16, 32, 48 and 64) and both prefill kernels (every row count
    1..8192). A row count above 8192, or every row count outside the decode
    groups if a prefill build failed, goes to `bmm_out` at the target's call
    site, with the `[16, K, N]` weight that op takes. An engine drafting more
    than one token also builds `MTP_GROUPS` (128 and 192 rows) and sends them a
    decode step's rows only; a context step of those row counts keeps the
    prefill kernel. The M >= 16 group's objects also launch at every count in
    9..64 (`DECODE_RUNTIME_ROWS`), which the target sends them only from the
    rows without a cell of an MTP3 / MTP5 decode step (12 / 18) or eager draft.

    On the context step the target takes the fp8 q-base K-B (`build_prefill_fp8_q_base`)
    instead when the rows go on the context rope/append kernel: K-B then writes the FMHA's
    e4m3 query bytes itself and the rope kernel skips its q-base groups. On a pure decode
    step the decode groups' fp8 q-base K-B (`build_decode_fp8_q_base`; on an MTP3 / MTP5
    decode step the MTP group's) runs beside the rope kernel, which then skips the q-base,
    and the decode V-B built with
    `build_v_b_mxf8_side` writes o_proj's MXF8 operand; elsewhere the decode groups keep the bf16
    kernels and the full rope. On an MTP3 / MTP5 decode step the MTP group's V-B built with
    `build_v_b_mxf8_side` writes the wide decode o_proj's 192-row MXF8 operand the same way.

    Both kernel kinds check every call: row count in the runner's range, and
    every shape and stride of `a` and `out` exactly as compiled. Anything else
    raises. The prefill kernel checks a weight only the first time it meets
    that weight's address, which is what `is_valid` covers.
    """

    def __call__(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: AbsorbedBmmRunner,
    ) -> torch.Tensor:
        runner.launch(a, weight, out)
        return out

    def reference(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: AbsorbedBmmRunner,
    ) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated (TF32 off),
        rounded once. For the fp8 q-base runner, those bf16 values converted to e4m3
        (nearest, saturating), as uint8 bytes. `out` is not written."""
        with true_fp32_matmul():
            prod = a.float() @ weight.float().transpose(1, 2)
        if runner.group in ("prefill_fp8", "fp8_m_le_8", "fp8_m_ge_16", "fp8_m_ge_128"):
            bf16 = prod.to(torch.bfloat16).float().clamp(-448.0, 448.0)
            return bf16.to(torch.float8_e4m3fn).view(torch.uint8)
        return prod.to(out.dtype)

    def is_valid(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        runner: AbsorbedBmmRunner,
    ) -> None:
        # The prefill kernel caches its weight operand by address and validates a
        # weight only on first sight, so a later tensor at the same address with
        # another shape or layout (the `[16, K, N]` view `bmm_out` takes) runs on
        # the cached `[16, N, K]` operand.
        lay = LAYOUTS[runner.kernel]
        assert (
            tuple(weight.shape) == (lay.heads, lay.n, lay.k)
            and weight.is_contiguous()
            and weight.dtype == torch.bfloat16
        ), (
            f"weight must be the contiguous bf16 [{lay.heads}, {lay.n}, {lay.k}] operand, "
            f"got {tuple(weight.shape)} with strides {weight.stride()}"
        )


cute_dsl_absorbed_bmm = CuteDslAbsorbedBmm()
