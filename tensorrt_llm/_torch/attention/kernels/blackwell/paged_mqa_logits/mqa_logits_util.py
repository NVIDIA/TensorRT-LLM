# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CuTe DSL helpers for Blackwell paged MQA logits."""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, testing
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.memory import SmemAllocator


@dataclass(frozen=True)
class KernelLaunch:
    """A prepared kernel launch reusable for dispatch and timing."""

    compiled_fn: Callable[..., object]
    arguments: testing.JitArguments
    logits: torch.Tensor

    def run(self) -> None:
        """Dispatch the prepared kernel once."""
        self.compiled_fn(*self.arguments.args, **self.arguments.kwargs)


# SMEM b32 load/store wrappers — `cute.arch.ld_shared / st_shared` are not
# exposed in the DSL surface, so we wrap raw PTX.


@dsl_user_op
def ld_shared_b32(smem_ptr, *, loc=None, ip=None) -> Int32:
    # `_Pointer.toint()` returns Int32 (SMEM addr fits in 32 bits).
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [smem_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_b32(smem_ptr, val: Int32, *, loc=None, ip=None) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# Packed f16x2 and bf16x2 arithmetic helpers shared by the epilogues.


def ceil_div(x: int, y: int) -> int:
    """Return ``ceil(x / y)`` for integers."""
    return (x + y - 1) // y


def align(x: int, alignment: int) -> int:
    """Round ``x`` up to a multiple of ``alignment``."""
    return ceil_div(x, alignment) * alignment


def _select(cond, a, b):
    """``a if cond else b`` for both plain Python values and CuTe DSL runtime
    values. The atom-split helpers below are shared by host code and the kernel
    trace; the DSL does not stage control flow inside plain callees, so a Python
    ``if`` on a traced comparison fails (PHASE_DYNAMIC_TO_STATIC_BOOL). A real
    ``bool`` means host Python; anything else goes through ``cutlass.select_``.
    """
    if isinstance(cond, bool):
        return a if cond else b
    # select_ accepts plain ints for the branches but returns a raw ir.Value,
    # which has no arithmetic operators; wrap it back into a Numeric so the
    # callers' `base + ...` / `qa * base + ...` keep working (same idiom as
    # ld_shared_b32 above wrapping inline_asm's result).
    return Int32(cutlass.select_(cond, a, b))


def atom_valid_tokens(atom_idx: int, num_atoms: int, base: int, rem: int) -> int:
    """Real (non-padded) token count for one atom under the even-remainder
    split of ``next_n`` into ``num_atoms`` atoms (``base = next_n // num_atoms``,
    ``rem = next_n % num_atoms``). The first ``rem`` atoms get ``base + 1``
    tokens, the rest get ``base``; sums to ``next_n`` with no padding.
    ``rem == 0``
    degenerates to the constant ``base`` (today's fixed-stride atom width).
    Safe on a traced ``atom_idx`` (see :func:`_select`).
    """
    return base + _select(atom_idx % num_atoms < rem, 1, 0)


def atom_token_offset(atom_idx: int, num_atoms: int, base: int, rem: int) -> int:
    """Start token (within one sequence's next_n) of one atom under the same
    even-remainder split as :func:`atom_valid_tokens`. ``rem == 0`` degenerates
    to ``(atom_idx % num_atoms) * base`` (today's fixed-stride offset).
    Safe on a traced ``atom_idx`` (see :func:`_select`).
    """
    qa = atom_idx % num_atoms
    return qa * base + _select(qa < rem, qa, rem)


def atom_tokens_after(atom_idx: int, num_atoms: int, base: int, rem: int, next_n: int) -> int:
    """Token count of the atoms AFTER atom ``atom_idx`` within its sequence:
    the causal-mask stagger (atom i, 0 = oldest, sees ctx shortened by this
    much so the split's staggered causal limits match the unsplit next_n).
    Shared by the kernel's ``_atom_ctx_len`` and the host schedule builder,
    which must agree exactly. ``rem == 0`` degenerates to today's
    ``(num_atoms - 1 - i) * atom``.
    """
    return next_n - (
        atom_token_offset(atom_idx, num_atoms, base, rem)
        + atom_valid_tokens(atom_idx, num_atoms, base, rem)
    )


# Supported range of next_n (speculative tokens per sequence) for the decode
# kernels. An atom never exceeds next_n, so this also bounds the atom: the
# register weight-cache tables are verified spill-free (SASS LDL/STL count) up
# to this many slots; a larger next_n is rejected rather than silently split.
MAX_NEXT_N = 16


def legal_atoms(num_heads: int, max_n: int, n_align: int, max_atom: int = MAX_NEXT_N):
    """Atoms (kernel next_n per CTA) whose MMA N tile ``atom * num_heads`` the
    kernel can build: ``N <= max_n`` (tcgen05 256 / FP4 TMEM budget), ``N %
    n_align == 0`` and ``atom <= max_atom`` (an atom never exceeds next_n).

    The N granularity is 8, not 16: PTX ISA tcgen05.mma "Matrix Shape" (Table
    48) lists N in steps of 8 for ``.kind::f8f6f4`` and ``.kind::mxf4``; steps
    of 16 apply only to an 8-bit MN-major B operand (Table 63; the DSL's
    ``tcgen05.MmaOp`` enforces the same rule), and both kernels feed B K-major.
    Empty when even the narrowest legal atom does not fit."""
    return tuple(
        a
        for a in range(1, max_atom + 1)
        if a * num_heads <= max_n and (a * num_heads) % n_align == 0
    )


def is_uneven_split(next_n: int, num_next_n_atoms: int, atom=None) -> bool:
    """True when the tile holds padded slots: the atoms differ in token count
    (``next_n % num_next_n_atoms != 0``) or the tile ``atom`` is wider than the
    even atom (``atom=None`` means the even atom)."""
    base, rem = divmod(next_n, num_next_n_atoms)
    return rem != 0 or (atom is not None and atom != base)


def validate_atom_split(
    next_n: int, num_heads: int, num_next_n_atoms: int, atom, kernel_atoms
) -> int:
    """Legality of an explicit atom split for a decode launch; returns the tile
    width. Called first thing by _prepare_{fp8,fp4}_kernel_launch so the public
    fp8/fp4_paged_mqa_logits() entry points get the same checks run() applies:
    next_n within the spill-verified range, a positive atom count, an explicit
    width for an uneven split, a width that covers the widest atom and matches
    the atom count, and a width that is a legal, tuned N tile for this head
    count (``kernel_atoms`` = legal_atoms(num_heads, N_max, 8)). Pure scalar
    arithmetic (microseconds); raises ValueError with the offending numbers.
    """
    if not (1 <= next_n <= MAX_NEXT_N):
        raise ValueError(f"next_n={next_n} outside the supported range 1..{MAX_NEXT_N}")
    if not (1 <= num_next_n_atoms <= next_n):
        raise ValueError(f"num_next_n_atoms={num_next_n_atoms} must be in 1..next_n ({next_n})")
    base, rem = divmod(next_n, num_next_n_atoms)
    if atom is None:
        if rem:
            raise ValueError(
                f"num_next_n_atoms={num_next_n_atoms} does not divide next_n={next_n}: "
                "atom (the fixed tile width) must be passed explicitly"
            )
        width = base
    elif isinstance(atom, bool) or not isinstance(atom, int):
        # The public entry points take an int (or None). "auto" / "direct" are
        # run()'s conveniences; they are resolved there, never here.
        raise ValueError(
            f"atom must be an int or None, got {atom!r} ({type(atom).__name__}); "
            "'auto' is only accepted by run(next_n_atom=...)"
        )
    else:
        width = atom
    widest = base + (1 if rem else 0)
    if width < widest:
        raise ValueError(
            f"atom={width} is narrower than the widest atom ({widest}) of "
            f"next_n={next_n} split into {num_next_n_atoms} atoms"
        )
    if ceil_div(next_n, width) != num_next_n_atoms and not (rem == 0 and width == base):
        raise ValueError(
            f"num_next_n_atoms={num_next_n_atoms} is inconsistent with atom={width} "
            f"for next_n={next_n} (ceil_div(next_n, atom) = {ceil_div(next_n, width)})"
        )
    if width not in kernel_atoms:
        raise ValueError(
            f"atom={width} with num_heads={num_heads} gives an MMA N tile of "
            f"{width * num_heads}, not a legal / tuned tile on this arch; legal atoms: "
            f"{tuple(kernel_atoms)}"
        )
    return width


def choose_atom_split(
    batch: int,
    ctx: int,
    next_n: int,
    num_sms: int,
    kernel_atoms,
    split_kv_tokens: int = 256,
    tie: str = "max_num_atoms",
    allow_uneven: bool = False,
) -> int:
    """Pick the atom size (kernel next_n) decomposing ``next_n`` so as to
    minimize the persistent-scheduler wave count for a (batch, ctx) shape.

    An "atom" runs ``atom`` next-token positions per launched q-row; ``next_n``
    is split into ``num_atoms`` pieces, expanding the effective batch to
    ``batch * num_atoms`` (each piece re-reads the same KV, so num_atoms x
    HBM). Wave count is estimated as ``ceil(tasks / num_sms)`` where a task is
    one (expanded-batch-row, KV-chunk) tile and a ``ctx``-token sequence spans
    ``ceil(ctx / split_kv_tokens)`` KV chunks. Fewer waves -> better SM occupancy.

    Ties on wave count -- the small-batch regime, where splitting fills otherwise
    idle SMs within the same wave -- are broken by ``tie``:
      - "max_num_atoms": largest num_atoms == smallest atom == most SMs busy
                    per wave, at the cost of num_atoms x KV re-reads (acceptable
                    when SMs, not HBM, are the limiter -- the small-batch regime).
      - "max_atom":  largest atom == fewest atoms == least HBM traffic.

    ``kernel_atoms`` is the arch/kernel-supported atom set (e.g. FP4 Rubin
    (1,2,3,4) / Blackwell (1,2,3); FP8 (1,2,3,4)).

    ``allow_uneven``: when False only atoms exactly dividing ``next_n`` are
    considered. When True, and only if ``next_n`` has no exact divisor > 1 in
    ``kernel_atoms`` (e.g. 5, 7, 11, 13), every atom is a candidate with
    ``num_atoms = ceil_div(next_n, atom)`` (even-remainder split, see
    atom_valid_tokens / atom_token_offset). With an exact divisor available the
    uneven candidates are skipped: the wave model ignores the padded slots'
    cost, and an uneven split measured slower than the exact one there.

    Returns the chosen ``atom`` (``atom == next_n`` -> run direct).

    For an uneven ``next_n`` the pick is a wave-count estimate only, not
    guaranteed fastest: candidates differ in per-tile epilogue cost (weight
    cache coverage, register pressure) the model does not see. next_n is a
    model constant, so enumerate the legal atoms on the deployment's batch /
    ctx range, measure, and pass the winner explicitly (``num_next_n_atoms`` /
    ``atom`` on fp8/fp4_paged_mqa_logits(), ``next_n_atom`` on run()); treat
    this result as the fallback.

    Analytical (no timing). Ported from the TRT-LLM unit-test heuristic
    ``_choose_atom_split``; validated against a measured direct-vs-split sweep
    (81% match with the empirical winner, 100% in the decisive large/small-batch
    regions; mismatches are a narrow near-crossover band where split wins <=7%).
    """
    # Uneven candidates only when the exact-divisor set is just {1}.
    has_divisor = any(a > 1 and next_n % a == 0 for a in kernel_atoms)
    use_uneven = allow_uneven and not has_divisor
    cands = []
    for atom in kernel_atoms:
        if next_n % atom == 0:
            num_atoms = next_n // atom
        elif use_uneven:
            num_atoms = ceil_div(next_n, atom)
        else:
            continue
        tasks = batch * num_atoms * ceil_div(ctx, split_kv_tokens)
        waves = ceil_div(tasks, num_sms)
        # An uneven candidate (padding slots: atom*num_atoms - next_n > 0) beats
        # the exact atom=1 fallback only by strictly fewer waves; at equal waves
        # atom=1 (no padding) is kept.
        uneven = 1 if atom * num_atoms != next_n else 0
        cands.append((waves, uneven, num_atoms, atom))
    if not cands:
        raise ValueError(f"no atom in {tuple(kernel_atoms)} divides next_n={next_n}")
    if tie == "max_num_atoms":
        cands.sort(key=lambda c: (c[0], c[1], -c[2]))
    elif tie == "max_atom":
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
    else:
        raise ValueError(f"unknown tie={tie!r}; expected 'max_num_atoms' or 'max_atom'")
    return cands[0][3]


def arbitrate_elem_tolerance(
    logits: torch.Tensor,
    ref: torch.Tensor,
    valid: torch.Tensor,
    atol: float,
    rtol: float,
    row_fp64,
    max_rows: int = 64,
) -> tuple[bool, str]:
    """Second-opinion check for the element-wise tolerance, FlashAttention style.

    ``|logits - ref| <= atol + rtol * |ref|`` with a fixed fp32 atol fails on a
    handful of elements at very large B x ctx: both the kernel and the fp32
    torch reference re-associate sums whose terms reach a few hundred, so each
    is a couple of ulp (~3e-5 at 500) from the true value, in different
    directions, while the final logit itself can be small (cancellation);
    kernel-vs-fp64 and reference-vs-fp64 errors are the same size.

    So instead of loosening the constants, the elements that fail the fast
    check are re-judged against a float64 recompute of their rows:
    ``|logits - ref64| <= 2 * |ref - ref64| + atol`` -- the kernel may not be
    worse than twice the same-precision reference's own rounding error. A real
    defect (wrong value, wrong position) is far outside that band. Only the
    offending rows are recomputed (``row_fp64(row) -> float64 [cols]``, NaN or
    -inf where the row is not in context), capped at ``max_rows``; more
    failing rows than that is treated as a failure, not rounding.

    Returns (ok, note) where note summarizes the arbitration for the log.
    """
    err = (logits.double() - ref.double()).abs()
    over = (err > atol + rtol * ref.double().abs()) & valid
    n_over = int(over.sum().item())
    if n_over == 0:
        return True, ""
    rows = over.any(dim=1).nonzero().flatten().tolist()
    if len(rows) > max_rows:
        return False, (
            f"fp64 arbitration refused: {n_over} elements over tolerance in "
            f"{len(rows)} rows (> {max_rows}); treating as a real mismatch"
        )
    worst_ratio = 0.0
    n_bad = 0
    for r in rows:
        f64 = row_fp64(r).to(logits.device, torch.float64)
        cols = over[r].nonzero().flatten()
        f64_c = f64[cols]
        fin = torch.isfinite(f64_c)
        if not bool(fin.all()):
            n_bad += int((~fin).sum().item())  # kernel wrote where fp64 says out of context
            cols, f64_c = cols[fin], f64_c[fin]
        e_k = (logits[r, cols].double() - f64_c).abs()
        e_r = (ref[r, cols].double() - f64_c).abs()
        bound = 2.0 * e_r + atol
        n_bad += int((e_k > bound).sum().item())
        if e_k.numel():
            worst_ratio = max(worst_ratio, float((e_k / bound.clamp_min(1e-30)).max().item()))
    note = (
        f"fp64 arbitration: {n_over} elements over tolerance in {len(rows)} rows, "
        f"{n_bad} beyond 2x the fp32 reference's own error (worst kernel/bound "
        f"ratio {worst_ratio:.2f})"
    )
    return n_bad == 0, note


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return the symmetric normalized difference between two tensors."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)


def compute_schedule_metadata(
    context_lens_cpu: torch.Tensor, block_kv: int, num_ctas: int
) -> torch.Tensor:
    """Build the persistent CTA schedule boundaries on the CPU.

    Each row contains ``(q_idx, kv_idx_half)``. Adjacent rows delimit one
    CTA's work range. Both paged MQA kernels use two math warp groups, so the
    second coordinate counts pairs of ``block_kv`` compute tiles.

    Args:
        context_lens_cpu: Per-sequence context lengths in a CPU tensor.
        block_kv: Number of KV tokens computed by one math warp group.
        num_ctas: Number of persistent CTAs sharing the work.

    Returns:
        An int32 CPU tensor with shape ``(num_ctas + 1, 2)``.
    """
    batch_size = context_lens_cpu.shape[0]
    splits_per_seq = []
    total_splits = 0
    for batch_idx in range(batch_size):
        context_len = int(context_lens_cpu[batch_idx].item())
        num_kv = ceil_div(context_len, block_kv)
        num_splits = ceil_div(num_kv, 2)
        splits_per_seq.append(num_splits)
        total_splits += num_splits

    splits_per_cta = total_splits // num_ctas
    remainder = total_splits % num_ctas
    schedule = torch.zeros((num_ctas + 1, 2), dtype=torch.int32)

    cumulative_splits = 0
    seq_idx = 0
    for cta_idx in range(num_ctas + 1):
        target = cta_idx * splits_per_cta + min(cta_idx, remainder)
        while seq_idx < batch_size and cumulative_splits + splits_per_seq[seq_idx] <= target:
            cumulative_splits += splits_per_seq[seq_idx]
            seq_idx += 1
            if seq_idx >= batch_size:
                break
        if seq_idx >= batch_size:
            schedule[cta_idx] = torch.tensor([batch_size, 0], dtype=torch.int32)
        else:
            local_split_idx = target - cumulative_splits
            schedule[cta_idx] = torch.tensor([seq_idx, local_split_idx], dtype=torch.int32)
    return schedule


# Single-CTA single-warp kernel: register-array num_segs → warp inclusive scan
# with carry → SMEM prefix_sum → per-SM linear scan emitting (q_idx, kv_split).
class PagedMQALogitsMetadataKernel:
    """Compile-time params: aligned_batch_size (multiple of 32), split_kv, num_sms.

    Runtime: context_lens [B] int32 (cuda), schedule_meta [num_sms+1, 2] int32 (cuda),
    batch_size (Int32).
    """

    def __init__(self, aligned_batch_size: int, split_kv: int, num_sms: int):
        assert aligned_batch_size > 0 and aligned_batch_size % 32 == 0
        self.aligned_batch_size = aligned_batch_size
        self.split_kv = split_kv
        self.num_sms = num_sms

    @cute.jit
    def __call__(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(context_lens, schedule_meta, batch_size).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
    ):
        kAligned = cutlass.const_expr(self.aligned_batch_size)
        SPLIT_KV = cutlass.const_expr(self.split_kv)
        kNumSMs = cutlass.const_expr(self.num_sms)
        kNumChunks = cutlass.const_expr(kAligned // 32)
        # Cover sm_idx ∈ [0, kNumSMs] (inclusive) in 32-lane strips.
        kMaxSmChunks = cutlass.const_expr((kNumSMs + 32) // 32)

        lane_idx = cute.arch.lane_idx()

        smem = SmemAllocator()
        prefix_sum = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((kAligned,), order=(0,)),
            byte_alignment=128,
        )

        # Phase 1: per-lane register array of ceil_div(ctx, SPLIT_KV).
        # Out-of-range lanes contribute 0 (matches CUDA's q_idx<batch_size guard).
        num_segs = [cutlass.Int32(0)] * kNumChunks
        for k in cutlass.range_constexpr(kNumChunks):
            q_idx = cutlass.Int32(k * 32) + lane_idx
            ctx_len = cutlass.Int32(0)
            if q_idx < batch_size:
                ctx_len = context_lens[q_idx]
            num_segs[k] = (ctx_len + (SPLIT_KV - 1)) // SPLIT_KV

        # Phase 2: warp-level inclusive scan, with carry across chunks, → SMEM.
        sum_carry = cutlass.Int32(0)
        for k in cutlass.range_constexpr(kNumChunks):
            x = num_segs[k]
            for i in cutlass.range_constexpr(5):  # log2(32) = 5
                offset = 1 << i
                y = cute.arch.shuffle_sync_up(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
                if lane_idx >= offset:
                    x = x + y
            x = x + sum_carry
            prefix_sum[k * 32 + lane_idx] = x
            # Broadcast lane-31's value to all lanes for the next chunk.
            sum_carry = cute.arch.shuffle_sync(x, 31)

        # Phase 3: distribute `total` segments across kNumSMs CTAs.
        # Each lane handles sm_idx = lane, lane+32, ... up to kNumSMs inclusive.
        total = sum_carry
        q_div = total // kNumSMs
        r_mod = total % kNumSMs

        for s in cutlass.range_constexpr(kMaxSmChunks):
            sm_idx_local = cutlass.Int32(s * 32) + lane_idx
            if sm_idx_local <= kNumSMs:
                # seg_starts = sm * q + min(sm, r). Pre-declare so it survives
                # the dynamic if/else (DSL rule: no first-assignment inside
                # dynamic control flow).
                seg_starts = sm_idx_local * q_div
                if sm_idx_local <= r_mod:
                    seg_starts = seg_starts + sm_idx_local
                else:
                    seg_starts = seg_starts + r_mod

                # Linear scan: q_idx_out = number of j ∈ [0, batch_size) with
                # prefix_sum[j] <= seg_starts. prefix_sum is non-decreasing, so
                # the predicate is monotone — we can replace the CUDA `while`
                # with a constexpr-bounded for-scan.
                q_idx_out = cutlass.Int32(0)
                for j in cutlass.range_constexpr(kAligned):
                    in_range = cutlass.Int32(j) < batch_size
                    advance = cutlass.Boolean(False)
                    if in_range:
                        if prefix_sum[j] <= seg_starts:
                            advance = cutlass.Boolean(True)
                    if advance:
                        q_idx_out = cutlass.Int32(j + 1)

                kv_split_idx = seg_starts
                if q_idx_out > 0:
                    kv_split_idx = seg_starts - prefix_sum[q_idx_out - 1]

                schedule_meta[sm_idx_local, 0] = q_idx_out
                schedule_meta[sm_idx_local, 1] = kv_split_idx


def frontend_cache_token() -> bool:
    """Cache-key component for the active DSL frontend.

    ``BaseDSL.enable_pyir()`` flips a process-global mode, so a compiled
    artifact is only valid for the frontend that produced it. Any
    ``functools.cache`` around a compile must key on this token, or a cubin
    compiled under one frontend is silently served to the other (e.g. the
    frontendnext round of a dual-frontend pytest reusing the v1 artifact).

    Scans every DSL singleton rather than ``BaseDSL._get_dsl()``: the latter
    returns the first-registered instance, which may be a preprocess=False
    DSL that ``enable_pyir()`` skips, while the mode push covers every
    preprocessor-carrying singleton — ``any`` mirrors that.
    """
    from cutlass.base_dsl.dsl import DSLSingletonMeta

    return any(
        bool(getattr(getattr(dsl, "envar", None), "enable_pyir", False))
        for dsl in DSLSingletonMeta._instances.values()
    )


@functools.cache
def compile_paged_mqa_logits_metadata_kernel(
    aligned_b: int, split_kv: int, num_ctas: int, frontendnext: bool
):
    """Compile and cache the metadata kernel using fake tensors and TVM FFI."""
    sym_B = cute.sym_int()
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_B,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas + 1, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kern = PagedMQALogitsMetadataKernel(aligned_b, split_kv, num_ctas)
    compiled = cute.compile(
        kern,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    print(f"[compile] schedule_meta aligned_b={aligned_b} split_kv={split_kv} num_sms={num_ctas}")
    return compiled


def get_paged_mqa_logits_metadata_cute_dsl(
    context_lens: torch.Tensor,
    compute_block_kv_per_wg: int,
    num_ctas: int,
    num_math_wg: int = 2,
    num_next_n_atoms: int = 1,
    next_n_atom: int = 1,
    *,
    next_n: int,  # the model's logical next-token count; never derived here
) -> torch.Tensor:
    """Compute schedule metadata on the GPU with the CuTe DSL kernel.

    This has the same contract as :func:`compute_schedule_metadata`, but
    expects a CUDA int32 vector and returns the schedule on the same device.
    ``SPLIT_KV`` is ``compute_block_kv_per_wg * num_math_wg``.

    In-kernel atom split (``num_next_n_atoms > 1``): the schedule must enumerate
    ``batch * num_next_n_atoms`` q_atom tasks. We expand the native ``[batch]``
    context_lens to ``[batch*num_atoms]`` here (atom i, 0=oldest, sees ctx -
    (num_atoms-1-i)*next_n_atom, matching the split's staggered causal limits)
    and feed the unchanged scheduler kernel. ``num_next_n_atoms == 1`` is a
    no-op. Mirrors DeepGEMM's ``num_next_n_atoms`` metadata argument.
    """
    # context_lens is caller input, so validate with real exceptions (asserts
    # would be stripped under `python -O`).
    if not (context_lens.is_cuda and context_lens.dtype == torch.int32 and context_lens.dim() == 1):
        raise ValueError(
            "context_lens must be a 1-D CUDA int32 tensor; got "
            f"cuda={context_lens.is_cuda}, dtype={context_lens.dtype}, "
            f"dim={context_lens.dim()}"
        )
    if num_next_n_atoms > 1:
        if next_n_atom < 1:
            raise ValueError(f"next_n_atom must be >= 1 for a split; got {next_n_atom}")
        # Causal stagger = tokens after each atom, the same atom_tokens_after
        # the kernel's _atom_ctx_len uses (degenerates to (na-1-i)*next_n_atom
        # for an even split). next_n is the model config and is required, not
        # derived: next_n_atom * num_next_n_atoms is only next_n for an even
        # split; for an uneven one (next_n=5 as {3,2}) or a tile wider than
        # the atom it would stagger the schedule differently from the kernel
        # and leave KV tiles unscheduled.
        base, rem = divmod(next_n, num_next_n_atoms)
        off = torch.tensor(
            [
                atom_tokens_after(i, num_next_n_atoms, base, rem, next_n)
                for i in range(num_next_n_atoms)
            ],
            device=context_lens.device,
            dtype=torch.int32,
        )
        # A sequence shorter than an atom's offset (ctx < next_n, e.g. a brand-new
        # short seq under MTP) yields a small negative per-atom ctx. No clamp is
        # needed: the offset is < next_n <= 4, so ceil_div(ctx, block_kv) floors
        # such an atom to 0 work in both the scheduler and the kernel, agreeing.
        context_lens = (context_lens.unsqueeze(1) - off.unsqueeze(0)).reshape(-1).contiguous()
    batch_size = int(context_lens.shape[0])
    aligned_b = max(((batch_size + 31) // 32) * 32, 32)
    # One CTA handles tile_m is block_kv.
    split_kv = compute_block_kv_per_wg * num_math_wg

    schedule = torch.empty((num_ctas + 1, 2), dtype=torch.int32, device=context_lens.device)
    compiled = compile_paged_mqa_logits_metadata_kernel(
        aligned_b, split_kv, num_ctas, frontend_cache_token()
    )
    compiled(context_lens, schedule, batch_size)
    return schedule


def verify_schedule_meta(
    kernel_ctx_lens: torch.Tensor,
    schedule_meta: torch.Tensor,
    num_sms: int,
) -> None:
    """Check the GPU schedule metadata against the CPU implementation."""
    ref = compute_schedule_metadata(kernel_ctx_lens.cpu(), 128, num_sms).to(kernel_ctx_lens.device)
    if not torch.equal(ref, schedule_meta):
        diff = (ref != schedule_meta).nonzero(as_tuple=False)
        raise AssertionError(
            f"[verify_meta] FAIL: {diff.shape[0]} mismatched entries "
            f"(B={kernel_ctx_lens.shape[0]} num_sms={num_sms}); "
            f"first diff at {diff[0].tolist()}: ref={ref[tuple(diff[0])]}, "
            f"dsl={schedule_meta[tuple(diff[0])]}"
        )
    print(f"[verify_meta] PASS (B={kernel_ctx_lens.shape[0]} num_sms={num_sms})")
