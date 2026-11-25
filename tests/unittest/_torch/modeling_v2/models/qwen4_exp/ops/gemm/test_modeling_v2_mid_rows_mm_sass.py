# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SASS check of the mid-rows GEMM: no non-coherent global load in any operand mode.

The GEMM is a programmatic dependent launch whose epilogue warps read the activation scales
(the previous launch's output, FP8 mode) after a `griddepcontrol.wait` that is not the
kernel's first statement. ptxas treats `ld.global.nc` data as read-only for the kernel's
lifetime and may hoist such a load above that wait (seen with this toolchain on another PDL
kernel of the target), where it would read the scales before their producer wrote them. The
kernel therefore loads every scalar with coherent `ld.global`; this test compiles each operand
mode for sm_103a (`CUTE_DSL_ARCH`, no GPU launch), dumps PTX and SASS, and fails on any
`ld.global.nc` / `LDG.*.CONSTANT`. It also checks that the dump holds the grid-dependency wait
(`ACQBULK`), so a variant that silently dropped PDL does not pass vacuously.

Needs the CuTe DSL and the CUDA toolchain (as every test of this directory), not a GPU.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_CHILD = textwrap.dedent(
    """
    import sys
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm import mid_rows_mm as op

    n, k, rows, fp8, w_mode, ks = (int(v) for v in sys.argv[1:7])
    plan = op.Plan(ks, 4, 4, 1)
    kern = op._MidMm(
        n=n, k=k, plan=plan, mpad=op.mma_rows(rows), fp8=bool(fp8), w_mode=w_mode, pdl=True
    )
    dt = cutlass.Float8E4M3FN if fp8 else cutlass.BFloat16
    gm, p = cute.AddressSpace.gmem, 0x7F0000000000
    out = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int32(), n), stride_order=(1, 0), assumed_align=16
    )
    cute.compile(
        kern,
        cutlass.Int32(rows),
        make_ptr(dt, p, gm, assumed_align=16),
        make_ptr(dt, p, gm, assumed_align=16),
        out,
        make_ptr(cutlass.Float32, p, gm, assumed_align=4),
        make_ptr(cutlass.Float32, p, gm, assumed_align=16),
        make_fake_stream(),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    """
)

#: `(why, N, K, rows, fp8, weight-scale mode, K split)`: one compile a mode.
_VARIANTS = (
    ("FP8, one weight scale, split-K (GDN out / o at tp1)", 2560, 6144, 48, 1, 1, 6),
    ("FP8, one scale a weight row, no split (vocab head)", 4096, 2560, 128, 1, 2, 1),
    ("bf16, split-K (attention-DP output projection)", 2560, 6144, 48, 0, 0, 6),
    ("bf16, no split (attention-DP input projection)", 16480, 2560, 64, 0, 0, 1),
)


@pytest.mark.parametrize("variant", _VARIANTS, ids=[v[0][:40] for v in _VARIANTS])
def test_no_non_coherent_global_load(variant, tmp_path: Path) -> None:
    env = dict(
        os.environ,
        CUTE_DSL_ARCH="sm_103a",
        CUTE_DSL_KEEP="ptx,sass",
        CUTE_DSL_DUMP_DIR=str(tmp_path),
        CUTE_DSL_CACHE_DIR=str(tmp_path / "cache"),
        CUTE_DSL_DISABLE_FILE_CACHING="1",
    )
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, *(str(v) for v in variant[1:])],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    ptx = [p.read_text() for p in tmp_path.glob("*.ptx")]
    sass = [p.read_text() for p in tmp_path.glob("*.sass")]
    assert ptx and sass, f"no PTX / SASS dumped in {sorted(os.listdir(tmp_path))}"
    for text in ptx:
        assert "griddepcontrol.wait" in text, "the variant lost its grid-dependency wait"
        assert "ld.global.nc" not in text, "a non-coherent global load in the PTX"
    for text in sass:
        assert "ACQBULK" in text, "no grid-dependency wait in the SASS"
        constant = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"\bLDG\b[\w.]*\.CONSTANT", line)
        ]
        assert not constant, f"non-coherent global loads in the SASS: {constant[:4]}"
