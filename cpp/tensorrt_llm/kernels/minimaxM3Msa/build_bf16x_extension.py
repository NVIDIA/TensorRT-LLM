# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the vendored MSA BF16X / E4M3X sparse-prefill forward (msa_kernels/prefill_bf16x) ahead of time.

The vendored package compiled it on first use (prefill_bf16x/jit.py, JitSpec) with torch.utils.cpp_extension.load; this
runs the same load at build time for the one spec the MiniMax-M3.1 targets serve: the E4M3 data path as the E4M3X
transcription, block-scale shift 0, the packed-FP16 dequant, sm_107a. The library lands where prefill_bf16x/jit.py's
AOT loader looks for it.
usage: build_bf16x_extension.py <prefill_bf16x dir> <CUTLASS root> <output dir> <build dir>
"""

import shutil
import sys
from pathlib import Path

import jinja2
from torch.utils.cpp_extension import load

ARCH = "107a"
SHIFT = 0
# JitSpec(target_arch="107a", block_scale_shift=0, dequant_mode="fp16", data_path="e4m3").module_name with E4M3X on.
MODULE = "_fmha_sm100_prefill_attention_bf16x_e4m3x_shift0_fp16_107a"
# JitSpec._template_params for the E4M3 data path.
TEMPLATE_PARAMS = dict(q_heads_per_kv=16, head_dim=128, page_size=128, q_stages=8, score_stages=2)


def flags(cuda_home: Path):
    """JitSpec._load_arguments for that spec (keep in sync with prefill_bf16x/jit.py)."""
    shift = f"-DFMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT={SHIFT}"
    e4m3x = "-DFMHA_SM100_PREFILL_BF16X_E4M3X=1"
    return dict(
        extra_cflags=["-O3", "-DNDEBUG", "-std=c++20", shift, e4m3x],
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-lineinfo",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-static-global-template-stub=false",
            "-Xptxas=-O3",
            f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}",
            shift,
            "-DFMHA_SM100_PREFILL_BF16X_HAS_QMUL4=0",
            "-DFMHA_SM100_PREFILL_BF16X_SM107=1",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100",
            e4m3x,
        ],
        extra_ldflags=["-lcuda", f"-Wl,-rpath,{cuda_home / 'lib64'}"],
    )


def main() -> None:
    package, cutlass, output, build = (Path(p) for p in sys.argv[1:5])
    csrc = package / "csrc"
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    template = jinja2.Template(
        (csrc / "templates/prefill_attention_inst.cu.jinja").read_text(encoding="utf-8")
    )
    (build / "prefill_attention_inst.cu").write_text(
        template.render(**TEMPLATE_PARAMS), encoding="utf-8"
    )
    import torch.utils.cpp_extension as ext

    load(
        name=MODULE,
        sources=[
            str(csrc / "api/prefill_attention_api.cpp"),
            str(csrc / "api/prefill_attention_binding.cpp"),
            str(csrc / "api/locality_domains.cu"),
            str(build / "prefill_attention_inst.cu"),
        ],
        extra_include_paths=[
            str(csrc / "api"),
            str(csrc / "include"),
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
        ],
        build_directory=str(build),
        verbose=False,
        with_cuda=True,
        is_python_module=False,
        **flags(Path(ext.CUDA_HOME)),
    )
    shutil.copy2(build / f"{MODULE}.so", output / f"{MODULE}.so")


if __name__ == "__main__":
    main()
