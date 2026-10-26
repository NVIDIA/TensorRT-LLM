# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the MSA fmha_sm100 variant sources that the build compiles ahead of time.

The vendored MSA package (tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/fmha_sm100,
see 3rdparty/vendor_sources.lock.yaml) compiled each FMHA variant at run time from two jinja
templates. TensorRT-LLM builds the variants the MiniMax-M3.1 targets call instead: this script
renders those templates for the variants in VARIANTS into generated/, with the parameters the
vendored jit.py would use, and the CMake file in this directory compiles them with jit.py's
flags. Run it after changing VARIANTS or the vendored templates; --check fails when the
checked-in sources are stale.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import jinja2

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_PACKAGE = _REPO / "tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/fmha_sm100"
_GENERATED = _HERE / "generated"

# The variant keys (jit.py's dispatch indices: dtype, qo tile, single warp group, sparse mode,
# page size, split-KV, pack factor) the MiniMax-M3.1 targets reach: the OnlyScore proxy of the
# MSA indexer over 128-token pages, BF16 (0) or FP8 (1) index keys. One KV head with one or two
# index heads per rank packs query rows by 1 or 2; packed rows up to 64 take the single warp
# group, up to 128 the 128-row tile, longer rows the 256-row tile (no split-KV there).
_SHAPES = (
    "0_0_2_1_0_0",
    "0_0_2_1_0_1",
    "0_1_2_1_0_0",
    "0_1_2_1_0_1",
    "1_1_2_1_0_0",
    "0_0_2_1_1_0",
    "0_0_2_1_1_1",
    "0_1_2_1_1_0",
)
VARIANTS = tuple(f"{dtype}_{shape}" for dtype in (0, 1) for shape in _SHAPES)


def _load_jit():
    spec = importlib.util.spec_from_file_location("_msa_fmha_jit", _PACKAGE / "jit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _params(jit, variant: str) -> dict:
    indices = [int(i) for i in variant.split("_")]
    dims = jit._FMHA_SM100_DISPATCH
    if len(indices) != len(dims):
        raise ValueError(f"variant {variant} needs {len(dims)} indices")
    params = {}
    for index, (_, values) in zip(indices, dims):
        params.update(values[index][1])
    if jit._FMHA_SM100_IMPOSSIBLE(params):
        raise ValueError(f"variant {variant} is not a valid combination: {params}")
    params["func_name"] = f"fmha_sm100_{variant}"
    params["variant_name"] = variant
    return params


def render() -> dict[str, str]:
    jit = _load_jit()
    csrc = _PACKAGE / "csrc"
    inst = jinja2.Template((csrc / "fmha_sm100_inst.jinja").read_text())
    run = jinja2.Template((csrc / "fmha_sm100_variant_run.cu.jinja").read_text())
    files = {}
    for variant in VARIANTS:
        params = _params(jit, variant)
        files[f"fmha_sm100_inst_{variant}.cu"] = inst.render(**params)
        files[f"fmha_sm100_run_{variant}.cu"] = run.render(**params)
    files["variants.cmake"] = (
        "set(MSA_FMHA_SM100_VARIANTS\n" + "".join(f"    {v}\n" for v in VARIANTS) + ")\n"
    )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated/ is stale")
    args = parser.parse_args()
    files = render()
    stale = sorted(
        name
        for name, text in files.items()
        if not (_GENERATED / name).is_file() or (_GENERATED / name).read_text() != text
    )
    extra = (
        sorted(p.name for p in _GENERATED.glob("*") if p.is_file() and p.name not in files)
        if _GENERATED.is_dir()
        else []
    )
    if args.check:
        if stale or extra:
            print(f"stale: {stale}; unexpected: {extra}", file=sys.stderr)
            return 1
        return 0
    _GENERATED.mkdir(exist_ok=True)
    for name in extra:
        (_GENERATED / name).unlink()
    for name, text in files.items():
        (_GENERATED / name).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
