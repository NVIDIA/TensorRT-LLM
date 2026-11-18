# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Render the MSA fmha_sm100 variant sources that the build compiles ahead of time.

The MSA package fmha_sm100 (fetched and patched by CMake, see the msa entry of 3rdparty/fetch_content.json)
compiles each FMHA variant at run time from two jinja
templates. The opt-in ahead-of-time build compiles the variants the MiniMax-M3.1 targets call: this script
renders those templates for the variants in VARIANTS into the output directory, with the parameters
the package's jit.py would use, and the CMake file in this directory runs it at configure time and
compiles the rendered sources with jit.py's flags. Files are rewritten only when their content
changes, so an unchanged render does not rebuild anything.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import jinja2

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


def _load_jit(package: Path):
    spec = importlib.util.spec_from_file_location("_msa_fmha_jit", package / "jit.py")
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


def render(package: Path) -> dict[str, str]:
    jit = _load_jit(package)
    csrc = package / "csrc"
    inst = jinja2.Template((csrc / "fmha_sm100_inst.jinja").read_text())
    run = jinja2.Template((csrc / "fmha_sm100_variant_run.cu.jinja").read_text())
    files = {}
    for variant in VARIANTS:
        params = _params(jit, variant)
        # One trailing newline, as the repository's end-of-file check keeps it.
        files[f"fmha_sm100_inst_{variant}.cu"] = inst.render(**params).rstrip("\n") + "\n"
        files[f"fmha_sm100_run_{variant}.cu"] = run.render(**params).rstrip("\n") + "\n"
    files["variants.cmake"] = (
        "set(MSA_FMHA_SM100_VARIANTS\n" + "".join(f"    {v}\n" for v in VARIANTS) + ")\n"
    )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="directory to render the sources into"
    )
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="the fmha_sm100 package directory to render from",
    )
    args = parser.parse_args()
    out = args.out
    files = render(args.package)
    out.mkdir(parents=True, exist_ok=True)
    for path in out.iterdir():
        if path.is_file() and path.name not in files:
            path.unlink()
    for name, text in files.items():
        target = out / name
        if not target.is_file() or target.read_text() != text:
            target.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
