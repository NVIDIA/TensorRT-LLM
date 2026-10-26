# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack the MSA CuTe DSL objects a reference run compiled into the archive the wheel ships.

The vendored MSA trees (msa_kernels/fmha_sm100, msa_kernels/fmha_sm100_q8kv8) export each CuTe DSL kernel they compile
to their AOT cache (MM_SPARSE_ATTN_AOT_CACHE/<tree>/...). Run the serving cells once on the built wheel with an empty
cache, then pack it: build_wheel.py extracts cute_aot/msa_cute_aot.txz into tensorrt_llm/libs/msa_cute_aot, which the
caches read before compiling. Objects are keyed by the toolchain and the kernel sources, so a stale one is never loaded.
usage: pack_cute_aot.py <MM_SPARSE_ATTN_AOT_CACHE root> [<archive>]
"""

import sys
import tarfile
from pathlib import Path

TREES = ("fmha_sm100", "fmha_sm100_q8kv8")


def main() -> int:
    root = Path(sys.argv[1])
    archive = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(__file__).parent / "cute_aot" / "msa_cute_aot.txz"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "w:xz") as tar:
        for tree in TREES:
            for path in sorted((root / tree).rglob("*")):
                if path.is_file() and path.suffix in (".o", ".json"):
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with open(path, "rb") as f:
                        tar.addfile(info, f)
                    count += path.suffix == ".o"
    print(f"packed {count} objects into {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
