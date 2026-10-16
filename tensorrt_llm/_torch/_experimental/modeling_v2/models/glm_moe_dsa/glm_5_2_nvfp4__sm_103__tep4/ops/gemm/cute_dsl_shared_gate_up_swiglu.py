# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared expert's gate/up GEMM and SwiGLU in one CuTe DSL launch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL as _USE_PDL

from ...kernels.gemm import mlp_shared_experts_gate_up_swiglu_m64_n1024_k6144_bf16 as _kernel
from ...kernels.gemm import moe_prefill_shared as _prefill
from ...kernels.gemm import moe_prefill_shared_splitk as _splitk
from ...kernels.moe import moe_prefill_schedule as _geo

#: Decode row counts the target compiles this entry for. The kernel takes
#: any row count in 1..64, one compile each.
ROWS = (1, 2, 4, 8, 16, 32, 48, 64)

#: `[M, 6144] x [1024, 6144] -> [M, 1024]`, then SwiGLU halves it to 512.
_K, _GU, _OUT = _kernel.INPUT_FEATURES, _kernel.GATE_UP_FEATURES, _kernel.OUTPUT_FEATURES

Runner = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


def prepare_weight(weight: torch.Tensor) -> torch.Tensor:
    """One layer's kernel weight, from that layer's checkpoint weight.

    `weight` is the row-major `[1024, 6144]` gate/up weight, rows
    `[gate_0..gate_511, up_0..up_511]`. Returns a new `[1024, 6144]` tensor
    with the rows interleaved `[gate_0, up_0, gate_1, up_1, ...]`, the pairing
    the kernel's epilogue reads. 12 MB per layer.

    The result belongs to the layer it came from. The op takes it per call,
    so the target keeps one per layer and passes that layer's: a single
    prepared weight passed for every layer runs every layer against one
    layer's shared expert, and the output still looks like fluent text.
    """
    return _kernel.prepare_weight(weight)


def build(rows: int, prepared_weight: torch.Tensor) -> Runner:
    """Compile the kernel for exactly `rows` rows (1..64). Call at load.

    `prepared_weight` is any one layer's `prepare_weight` result: the kernel
    checks its layout on this first call and never again, and the runner does
    not keep it. Costs one CuTe DSL compile (cached by the kernel module per
    row count) and one launch on zeros. Raises if `rows` is outside 1..64 or
    the compile fails.
    """
    rows = int(rows)
    dev = prepared_weight.device
    x = torch.zeros([rows, _K], dtype=torch.bfloat16, device=dev)
    out = torch.empty([rows, _OUT], dtype=torch.bfloat16, device=dev)
    with torch.cuda.device(dev):
        runner = _kernel.build()
        runner(x, prepared_weight, out)
    return runner


def build_all(prepared_weight: torch.Tensor, rows: tuple[int, ...] = ROWS) -> dict[int, Runner]:
    """`build` at every row count in `rows`; `{rows: runner}` for those that
    compiled. A row count whose compile raises is left out, and the target
    takes the two-launch path (projection, then SwiGLU) for it."""
    runners = {}
    for m in rows:
        try:
            runners[int(m)] = build(m, prepared_weight)
        except Exception:  # noqa: BLE001 -- that row count takes the two-launch path
            continue
    return runners


#: The prefill kernels' row domain: above the decode kernel's 64, up to the token budget.
PREFILL_MIN_ROWS = 65
PREFILL_MAX_ROWS = 8192

#: The prefill up families by name (`select_shared_geometry(rows, num_sms).family`): a
#: cluster-local split-K kernel while its CTA demand fits the device, then the two-CTA
#: persistent kernel at N 128, then at N 256.
PREFILL_FAMILIES = ("splitk_up64_s12_k2", "cta2_up128_s8", "cta2_up256_s6")


def prefill_family(rows: int, num_sms: int) -> str:
    """The up family the prefill runner launches at `rows` on a `num_sms`-SM device."""
    return _geo.select_shared_geometry(int(rows), int(num_sms)).family


def _compile_family(family: str, num_sms: int):
    if family == "splitk_up64_s12_k2":
        return _splitk.compile_projection(
            up=True, tile_n=64, stages=12, split_k=2, use_pdl=_USE_PDL
        )
    if family == "cta2_up128_s8":
        return _prefill.compile_up(grid_ctas=num_sms, tile_n=128, stages=8, use_pdl=_USE_PDL)
    if family == "cta2_up256_s6":
        return _prefill.compile_up(grid_ctas=num_sms, tile_n=256, stages=6, use_pdl=_USE_PDL)
    raise ValueError(f"unknown shared family {family!r}")


@dataclass(frozen=True)
class PrefillGateUpRunner:
    """The prefill up kernels: one compiled family per CTA-demand range, any row count in
    `PREFILL_MIN_ROWS..PREFILL_MAX_ROWS`. Called as `runner(x, prepared_weight, out)`; it
    launches `prefill_family(rows, num_sms)`'s kernel. Each kernel checks its operands'
    shapes, dtypes, alignment and overlap on every call and raises on a mismatch."""

    num_sms: int
    families: Mapping[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]]

    def __call__(self, x: torch.Tensor, prepared_weight: torch.Tensor, out: torch.Tensor):
        return self.families[prefill_family(int(x.shape[0]), self.num_sms)](x, prepared_weight, out)


def build_prefill(prepared_weight: torch.Tensor) -> PrefillGateUpRunner:
    """Compile every up family the prefill row domain selects on this device, each on a
    symbolic row count (two-CTA families on every SM, PDL as the engine sets it), and
    launch each once at the first row count it serves, against `prepared_weight`. Call
    at load. Raises if a compile fails: the target then keeps the two-launch path
    (prefill projection, then SwiGLU) for context rows.
    """
    dev = prepared_weight.device
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    first: dict[str, int] = {}
    for rows in range(PREFILL_MIN_ROWS, PREFILL_MAX_ROWS + 1):
        first.setdefault(prefill_family(rows, num_sms), rows)
    families = {}
    with torch.cuda.device(dev):
        for family, rows in first.items():
            families[family] = _compile_family(family, num_sms)
            x = torch.zeros([rows, _K], dtype=torch.bfloat16, device=dev)
            out = torch.empty([rows, _OUT], dtype=torch.bfloat16, device=dev)
            families[family](x, prepared_weight, out)
    return PrefillGateUpRunner(num_sms, families)


class CuteDslSharedGateUpSwiglu(OpWrapper):
    """`out = silu(x @ W_gate.T) * (x @ W_up.T)`, one launch. Returns `out`.

    `x` is bf16 `[M, 6144]`, `prepared_weight` the `[1024, 6144]` result of
    `prepare_weight` for this layer, `out` a caller-owned bf16 `[M, 512]`
    buffer. All contiguous and 128-byte aligned. The GEMM accumulates in fp32
    (the decode kernel and the prefill split-K family over an internal split-K
    reduction); SwiGLU runs in the epilogue, so the
    `[M, 1024]` intermediate never reaches memory.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The shared expert runs in every MoE layer and the MTP layer, at the
    # step's full row count. The kernel picks its schedule by row count;
    # cells cover the three schedules the target's row counts reach.
    # The prefill runner's cells sit at both edges of each up family's row
    # range on the device under test (`edge` first / last), so they follow
    # the device's SM count rather than naming row counts.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row: the 96-CTA low schedule", spec=dict(rows=1)),
        Cell(why="MTP width 2, one sequence", spec=dict(rows=2)),
        Cell(why="48 rows: the 128-CTA high schedule", spec=dict(rows=48)),
        Cell(why="the full decode batch: the tail schedule", spec=dict(rows=64)),
        Cell(
            why="prefill at 65 rows: the split-K up family's first count",
            spec=dict(prefill="splitk_up64_s12_k2", edge="first"),
        ),
        Cell(
            why="prefill: the split-K up family's last count",
            spec=dict(prefill="splitk_up64_s12_k2", edge="last"),
        ),
        Cell(
            why="prefill: the two-CTA N128 family's first count",
            spec=dict(prefill="cta2_up128_s8", edge="first"),
        ),
        Cell(
            why="prefill: the two-CTA N128 family's last count",
            spec=dict(prefill="cta2_up128_s8", edge="last"),
        ),
        Cell(
            why="prefill: the two-CTA N256 family's first count",
            spec=dict(prefill="cta2_up256_s6", edge="first"),
        ),
        Cell(
            why="prefill at 8192 rows, the token budget: the two-CTA N256 family",
            spec=dict(prefill="cta2_up256_s6", edge="last"),
        ),
    )

    note = """
    The target compiles one runner per row count in `ROWS` (1, 2, 4, 8, 16, 32,
    48, 64). Any other decode row count takes the two-launch path
    (`cute_dsl_projection` "shared_gate_up" or cuBLAS, then SwiGLU) at the
    target's call site.

    A trunk layer's context rows (65..8192) take the `build_prefill` runner when
    it built, paired with `cute_dsl_projection`'s shared-down prefill runner;
    otherwise, and always for the MTP layer's shared expert, the two-launch path.
    The prefill kernels round the gate and up sums to bf16 before SwiGLU (the
    two-launch path's boundary) and accumulate split-K partials in fp32.

    A decode runner is pinned to its row count and raises on any other; the
    prefill runner takes any count in 65..8192. Neither binds the weight: one
    runner serves every layer, and the weight passed decides which layer's
    expert runs.
    """

    def __call__(
        self,
        x: torch.Tensor,
        prepared_weight: torch.Tensor,
        out: torch.Tensor,
        runner: Runner | PrefillGateUpRunner,
    ) -> torch.Tensor:
        runner(x, prepared_weight, out)
        return out

    def reference(
        self,
        x: torch.Tensor,
        prepared_weight: torch.Tensor,
        out: torch.Tensor,
        runner: Runner | PrefillGateUpRunner,
    ) -> torch.Tensor:
        """SwiGLU of the fp32 product (TF32 off), rounded once to `out`'s
        dtype. Reads gate rows at even and up rows at odd indices of
        `prepared_weight`, which is the pairing `prepare_weight` makes. `out`
        is not written; `runner` is ignored."""
        with true_fp32_matmul():
            gate = x.float() @ prepared_weight[0::2].float().t()
            up = x.float() @ prepared_weight[1::2].float().t()
        return (F.silu(gate) * up).to(out.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        prepared_weight: torch.Tensor,
        out: torch.Tensor,
        runner: Runner | PrefillGateUpRunner,
    ) -> None:
        # The decode kernel validates its operands on a runner's first call (the
        # one `build` makes) and never again: a later strided or misaligned
        # operand is read as if compact and aligned. The row count it does check.
        # The prefill kernels check rows, shapes, dtypes, alignment and overlap on
        # every call and raise, so there is nothing for this guard to add.
        if isinstance(runner, PrefillGateUpRunner):
            return
        rows = int(x.shape[0])
        for name, t, shape in (
            ("x", x, (rows, _K)),
            ("prepared_weight", prepared_weight, (_GU, _K)),
            ("out", out, (rows, _OUT)),
        ):
            assert tuple(t.shape) == shape, f"{name} must be {list(shape)}, got {tuple(t.shape)}"
            assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert t.data_ptr() % 128 == 0, f"{name} must be 128-byte aligned"


cute_dsl_shared_gate_up_swiglu = CuteDslSharedGateUpSwiglu()
