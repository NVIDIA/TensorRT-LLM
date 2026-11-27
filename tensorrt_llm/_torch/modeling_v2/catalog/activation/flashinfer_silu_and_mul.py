# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated SiLU activation via the flashinfer silu_and_mul kernel."""

from __future__ import annotations

import torch
from torch.nn import functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class FlashinferSiluAndMul(OpWrapper):
    """`silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2`, new tensor out.

    The gate and up projections arrive concatenated along the last dim, which
    is why the op takes one tensor and halves it rather than taking two.

    `torch.ops.trtllm.flashinfer_silu_and_mul` (`custom_ops/flashinfer_custom_ops.py`)
    calls `flashinfer.activation.silu_and_mul`, whose JIT module launches
    `act_and_mul_kernel` (flashinfer's `activation.cuh`): one CTA per row, fp32
    math, one rounding to the input dtype on the store.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # Exercise decode and prefill across narrow and wide intermediates.
    # Model-specific shapes live in the test driver.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, 2 x moe_intermediate (2048)",
            spec=dict(shape=(1, 4096), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 4096), dtype=torch.bfloat16),
        ),
        Cell(
            why="the dense layers' wider intermediate, 2 x 18432",
            spec=dict(shape=(16, 36864), dtype=torch.bfloat16),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).
    Metadata consumed: none.

    Preconditions, each an `is_valid` guard: `x` is contiguous, its base is
    16-byte aligned, and each half of its last dim is a whole number of
    16-byte vectors. The kernel addresses row r at `r * x.shape[-1]` elements
    from the base pointer and ignores strides: a column slice, or a view of
    every other row, is accepted and every row but the first is computed from
    the wrong elements (measured: 93.7% of the outputs wrong at widths 12288 and
    3072, 16 rows; at least 99.9% at 6144 and 1536). A
    contiguous view that starts 2 or 8 bytes off a 16-byte boundary faults with
    `torch.AcceleratorError: CUDA error: misaligned address`, which poisons the
    context. A single row sliced out of a wider buffer is contiguous, admitted,
    and computed right.

    Raised by the op itself, so not guarded (the test drives both): fp32 input,
    `RuntimeError: Check failed: (false) is false: silu_and_mul(tvm::ffi::TensorView,
    tvm::ffi::TensorView, bool)::<lambda()> failed to dispatch data type`; zero
    rows, `RuntimeError: Check failed: (err == cudaSuccess) is false: Failed to
    launch kernel: invalid argument` -- a caller with a possibly empty batch
    skips the call. fp16 is accepted (one probe at 16 x 3072, inside torch's
    fp16 band); no target uses it here, so it is not a cell.

    PDL. The trtllm op passes `enable_pdl` from `get_env_enable_pdl()`
    (`TRTLLM_ENABLE_PDL`, on unless set to something other than "1"), read on
    every call. The certifying run had it unset, so PDL on, which is how this
    target's engine runs. With it at "0" the output was measured bitwise equal
    at 9216 rows for widths 12288 and 3072, and at 16 and 32768 rows for widths
    49152, 6144 and 1536; that path is not a cell.

    Rows. The block size is set by the width alone (768 threads at 12288, 192 at
    3072, one 16-byte vector per thread per half) and the row count only sets
    the grid. Batch-invariant at both target widths: every call on the leading
    1..9215 rows is bitwise equal to those rows of the 9216-row call, and every
    call on the leading 1..32767 rows to those of the 32768-row call. CUDA
    graphs: captured at 1..16 rows and replayed on new data at widths 12288 and
    3072, the output is bitwise equal to an eager call on that data; one eager
    call on the capture stream precedes each capture. Widths 4096 and 36864 were
    not captured.

    Widths 6144 and 1536 (dense layer 0 and the shared expert at tep8 and
    tp8). Profiled on sm_103: grid = rows, 384 and 96 threads, 32 registers.
    Each thread makes one vector pass over each half, as at 12288 and 3072,
    and the scalar loop over the `d % (threads * 8)` remainder is empty.
    Batch-invariant at both: every call on the leading 1..32767 rows is
    bitwise equal to those rows of the 32768-row call. CUDA graphs captured at
    1..16 rows replay on new data bitwise equal to eager.

    Width 49152 (dep4's replicated dense layer 0). Profiled on sm_103: grid =
    rows, 1024 threads (the launcher's cap; a half is 3072 vectors), 32
    registers, so two CTAs are resident per SM and a wave is 304 rows on 152
    SMs. Read from flashinfer's `activation.cuh`: each thread then makes
    exactly three vector passes over each half, and the kernel's scalar loop
    over the `d % (threads * 8)` remainder is empty (at 36864 it would cover
    2048 columns). Batch-invariant here too: every call on the leading
    1..32767 rows is bitwise equal to those rows of the 32768-row call (the
    `large` cell; 20.9 s, 11.0 s of it the op calls), and pre-merge's sample
    (939 counts, 0.27 s) holds the same. CUDA graphs captured at 1..16 rows
    replay bitwise equal to eager.

    Numerics, on each eager cell's own randn draw. At 9216 rows the output is
    bitwise equal to `reference` at both target widths; at 32768 rows one
    output of each draw is one bf16 step off (1 of 201326592 at 12288, 1 of
    50331648 at 3072), at worst 0.11 of torch's bf16 band. At 16x that scale,
    one step on 1730 of 56623104 outputs (9216 x 12288), 405 of 14155776
    (9216 x 3072), 5886 of 201326592 (32768 x 12288) and 1424 of 50331648
    (32768 x 3072), plus 3, 0, 1 and 1 outputs of +-0 from the `__expf` tail
    below; none reads more than 0.073 of the band. At 32768 x 49152 the two
    eager cells' draws put 2 and 4 of 805306368 outputs one step off, 0.071 of
    the band; at 16x, 23458 and 24063 one step off plus 6 and 10 of +-0, 0.072
    of the band. At 32768 rows the 6144 cell's draw puts 1 of 100663296 outputs
    one step off (0.011 of the band) and the 1536 cell's none of 25165824; at
    16x, 3033 and 712 one step off plus 3 and 0 of +-0, 0.072 of the band.
    The kernel's `silu` uses `__expf`, which shows only in the tails, measured
    with up = 1: a gate in
    [-88.5, -87.5] returns +-0 where the fp32 reference is a normal float of
    magnitude at most 8.8e-37, and a subnormal gate or up reads as zero. Both
    are inside torch's bf16 band (atol 1e-5); an infinite up times such a gate
    gives nan where the reference gives +-inf. inf and nan otherwise propagate
    as in the reference (gate -inf gives nan in both).

    Uncertified: widths other than 4096, 12288, 3072, 36864, 49152, 6144 and
    1536; width 4096 at rows other than 1 and 2048, and width 36864 at rows
    other than 16; rows above 32768 at 12288, 3072, 49152, 6144 and 1536 (the
    probes below are not cells); fp16; PDL off. At 49152 pre-merge drives only
    the sampled row counts; the other counts in 1..32767 are the `large` cell's.

    Hard limits (sm_103), at the target widths. Offsets are int64 in
    `act_and_mul_kernel`. The serving envelopes reach none of these: 9216 rows
    on the FP8 checkpoint's target, 32768 on the NVFP4 ones (at most 1610612736
    input elements, 75% of 2^31, at dep4's 49152, and 805306368 output
    elements). The input's byte offset does pass 2^31 inside row 21845 at
    49152, so every 32768-row call there crosses it; offsets are typed-pointer
    arithmetic, and those calls are gated against `reference`. The op reads no
    KV pool, slot or page table, so no pool size bounds it.
      in offset   row * width, int64         2^31 elements, inside row 174762 (12288),
                                             row 699050 (3072) and row 43690 (49152);
                                             2^32 inside row 87381 (49152)
                                             2^31 elements inside row 349525 (6144)
                                             and row 1398101 (1536): computed, not
                                             probed (32768 rows reach 201326592
                                             elements at 6144, 9.4% of 2^31)
      out offset  row * width / 2, int64     2^31 elements, inside row 349525 (12288),
                                             row 1398101 (3072) and row 87381 (49152)
                                             2^31 elements inside row 699050 (6144)
                                             and row 2796202 (1536): computed, not
                                             probed
                  probed at 349526 rows (12288) and 1398102 rows (3072), crossing
                  both: passed -- all but 9 and 15 rows bitwise equal to the
                  reference, those scattered on both sides of the crossings and
                  1 bf16 ulp off, inside torch's bf16 band
                  probed at 87382 rows (49152), crossing all three: passed --
                  bitwise equal to the same rows called in 8192-row slices, the
                  rows straddling the crossings (43689..43692, 87379..87381)
                  bitwise equal to one-row calls, 9 outputs 1 bf16 step off the
                  reference, 0.11 of the band
      rows        one CTA per row, grid x    2^31 - 1 rows    unprobed (13.2 TB of
                                                              input at width 3072)
                  flashinfer's generated `silu_and_mul` launcher narrows the int64
                  row count to the grid's unsigned 32-bit x dimension (computed,
                  not probed)
    No argument is an index carried through fp32.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_silu_and_mul(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """fp32-accumulated; the kernel computes in fp32 and rounds once."""
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor) -> None:
        # The op's own check is on the *row*, `x.shape[-1] * itemsize % 16 == 0`,
        # which a final dimension of 24 passes. The kernel vectorizes over the
        # two halves, so what has to be 16-byte aligned is the *half*: at 24 the
        # second half starts mid-vector and the launch dies with `CUDA
        # misaligned address`, poisoning the context rather than raising.
        width = x.shape[-1]
        assert width % 2 == 0, f"x.shape[-1] must be even to split in half; got {width}"
        half_bytes = (width // 2) * x.element_size()
        assert half_bytes % 16 == 0, (
            f"x.shape[-1] // 2 must be a whole number of 16-byte vectors; "
            f"{width} halves to {half_bytes} bytes, which is not a multiple of 16 "
            f"-- the kernel would fault on the misaligned second half"
        )
        assert half_bytes >= 16, (
            f"x.shape[-1] // 2 must hold at least one 16-byte vector; {width} "
            f"gives {half_bytes} bytes, for which the computed block size is 0"
        )
        # The kernel reads row r at `r * width` from the base pointer and never
        # looks at a stride: a view whose rows are not packed is accepted and
        # every row past the first is computed from the wrong elements.
        assert x.is_contiguous(), (
            f"x must be contiguous; got strides {tuple(x.stride())} for shape "
            f"{tuple(x.shape)} -- the kernel ignores strides and would read the wrong rows"
        )
        # Packed rows and aligned halves still fault if the base is off a
        # 16-byte boundary, the same unrecoverable `misaligned address`.
        assert x.data_ptr() % 16 == 0, (
            f"x must start 16-byte aligned; its base is {x.data_ptr() % 16} bytes "
            f"off -- the kernel's vector loads would fault"
        )


flashinfer_silu_and_mul = FlashinferSiluAndMul()
