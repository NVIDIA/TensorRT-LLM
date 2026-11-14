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
"""Pack the MSA CuTe DSL objects a run compiled into an archive.

The vendored MSA trees (msa_kernels/fmha_sm100, msa_kernels/fmha_sm100_q8kv8) export each CuTe DSL kernel they compile
to their AOT cache (MM_SPARSE_ATTN_AOT_CACHE/<tree>/...). Pack a populated cache to carry it to other hosts of the
same toolchain, and extract the archive into MM_SPARSE_ATTN_AOT_CACHE there; the caches read it before compiling.
Objects are keyed by the toolchain and the kernel sources, so a stale one is never loaded. No archive ships with
TensorRT-LLM.
usage: pack_cute_aot.py <MM_SPARSE_ATTN_AOT_CACHE root> <archive>
"""

import sys
import tarfile
from pathlib import Path

TREES = ("fmha_sm100_m31", "fmha_sm100_q8kv8_m31")


def main() -> int:
    root = Path(sys.argv[1])
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    archive = Path(sys.argv[2])
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
