# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Offline cache builder for Q8KV4 sparse prefill attention."""

from __future__ import annotations

import argparse
import os

from .jit import FMHA_SM100_PREFILL_BF16X_ARCH


def precompile(
    arch: str | None = None, block_scale_shifts=(0, 3), data_paths=("e4m3",)
) -> list[str]:
    """Build the cache entries used by the lazy runtime JIT: one per block-scale shift for the
    E4M3 data path, one (shift 0) for the BF16 data path."""

    previous_arch = os.environ.get(FMHA_SM100_PREFILL_BF16X_ARCH)
    if arch is not None:
        os.environ[FMHA_SM100_PREFILL_BF16X_ARCH] = arch
    try:
        from . import jit

        jit._target_arch.cache_clear()
        jit._load_extension_for_arch.cache_clear()
        uris = []
        for data_path in data_paths:
            shifts = block_scale_shifts if data_path == jit.E4M3_DATA_PATH else (0,)
            for shift in shifts:
                spec = jit.gen_jit_spec(block_scale_shift=shift, data_path=data_path)
                spec.build_and_load()
                uris.append(spec.uri)
        return uris
    finally:
        if previous_arch is None:
            os.environ.pop(FMHA_SM100_PREFILL_BF16X_ARCH, None)
        else:
            os.environ[FMHA_SM100_PREFILL_BF16X_ARCH] = previous_arch
        if "jit" in locals():
            jit._target_arch.cache_clear()


def build_aot(aot_dir: str, arch: str | None = None, block_scale_shifts=(0,), data_paths=("e4m3",)) -> list[str]:
    """Build the extensions offline and copy each library into ``aot_dir`` (the CPP-E4M3X prebuilt set that
    ``jit.load_extension`` takes before the JIT)."""
    import shutil
    from pathlib import Path

    from . import jit

    precompile(arch, block_scale_shifts, data_paths)
    previous_arch = os.environ.get(FMHA_SM100_PREFILL_BF16X_ARCH)
    if arch is not None:
        os.environ[FMHA_SM100_PREFILL_BF16X_ARCH] = arch
    try:
        jit._target_arch.cache_clear()
        out = Path(aot_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = []
        for data_path in data_paths:
            shifts = block_scale_shifts if data_path == jit.E4M3_DATA_PATH else (0,)
            for shift in shifts:
                spec = jit.gen_jit_spec(block_scale_shift=shift, data_path=data_path)
                library = spec.recipe().lookup()
                if library is None:
                    raise RuntimeError(f"no built library for {spec.uri}")
                shutil.copyfile(library, out / f"{spec.module_name}.so")
                names.append(f"{spec.module_name}.so")
        return names
    finally:
        if previous_arch is None:
            os.environ.pop(FMHA_SM100_PREFILL_BF16X_ARCH, None)
        else:
            os.environ[FMHA_SM100_PREFILL_BF16X_ARCH] = previous_arch
        jit._target_arch.cache_clear()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None, help="Target SM, for example 103a")
    parser.add_argument("--block-scale-shifts", default="0,3",
                        help="Comma-separated block-scale shifts to build (E4M3 data path)")
    parser.add_argument("--data-paths", default="e4m3",
                        help="Comma-separated data paths to build: e4m3, bf16")
    parser.add_argument("--aot-dir", default=None,
                        help="Also copy the built libraries here (the prebuilt set jit.load_extension takes first)")
    args = parser.parse_args()
    shifts = tuple(int(value) for value in args.block_scale_shifts.split(","))
    data_paths = tuple(value.strip() for value in args.data_paths.split(","))
    if args.aot_dir:
        print("\n".join(build_aot(args.aot_dir, args.arch, shifts, data_paths)))
    else:
        print("\n".join(precompile(args.arch, shifts, data_paths)))


if __name__ == "__main__":
    _main()


__all__ = ["build_aot", "precompile"]
