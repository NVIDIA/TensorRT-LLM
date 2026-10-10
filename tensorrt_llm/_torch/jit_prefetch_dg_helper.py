# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only DeepGEMM compile helper for ``jit_prefetch.py``. Run as a script.

    jit_prefetch_dg_helper.py <deep_gemm package dir> <arch_major> <arch_minor> <num_sms>

Reads one JSON request per stdin line, ``{"tag": int, "spec": str}``, where
``spec`` is a JSON object, either ``{"op": "fp8_fp4_gemm_nt", "m", "n", "k",
"a", "b", "d", "recipe"}`` or ``{"op": "m_grouped_fp8_fp4_gemm_nt_masked", "g",
"m", "n", "k", "expected_m", "a", "b", "recipe"}`` (dtype names as in
``torch``). Compiles the variant with
DeepGEMM's compile-only entry point into ``DG_JIT_CACHE_DIR`` and writes
``{"tag": int, "ok": bool, "s": float, "err": str, "built": bool}``.

The process has no visible GPU and never launches a kernel:
``DG_JIT_COMPILE_ONLY=1`` keeps DeepGEMM from creating its cuBLASLt handle and
CUDA workspace, and ``set_compile_target`` supplies the target arch and SM
count that the kernel heuristic and the NVCC arch flag would otherwise read
from the device. The DeepGEMM extension is imported from the installed
``tensorrt_llm`` package directory without running ``tensorrt_llm/__init__``.
"""

import importlib.util
import json
import os
import sys
import time


def _load_deep_gemm(pkg_dir: str):
    """Load DeepGEMM's extension module and initialize it, nothing else.

    Only the compiled module is needed: its ``init`` sets the library and CUDA
    paths the compiler uses, which is what ``deep_gemm/__init__.py`` does after
    importing its Python helpers (layout utilities, torch.distributed helpers,
    legacy Triton kernels) that a compile-only process has no use for.
    """
    ext = None
    for name in os.listdir(pkg_dir):
        if name.startswith("deep_gemm_cpp_tllm") and name.endswith(".so"):
            ext = os.path.join(pkg_dir, name)
            break
    if ext is None:
        raise RuntimeError(f"no deep_gemm_cpp_tllm extension under {pkg_dir}")
    spec = importlib.util.spec_from_file_location("deep_gemm_cpp_tllm", ext)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    # Same arguments as deep_gemm/__init__.py: library root, CUDA home.
    mod.init(os.path.join(pkg_dir, "deep_gemm"), cuda_home)
    return mod


def main():
    pkg_dir, arch_major, arch_minor, num_sms = sys.argv[1:5]
    os.environ["DG_JIT_COMPILE_ONLY"] = "1"
    import torch

    dg = _load_deep_gemm(pkg_dir)
    dg.set_compile_target(int(arch_major), int(arch_minor), int(num_sms))

    out = sys.stdout
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        tag = req["tag"]
        t0 = time.time()
        try:
            s = json.loads(req["spec"])
            if s["op"] == "fp8_fp4_gemm_nt":
                built = dg.compile_only_fp8_fp4_gemm_nt(
                    int(s["m"]),
                    int(s["n"]),
                    int(s["k"]),
                    getattr(torch, s["a"]),
                    getattr(torch, s["b"]),
                    getattr(torch, s["d"]),
                    tuple(s["recipe"]),
                )
            elif s["op"] == "m_grouped_fp8_fp4_gemm_nt_masked":
                built = dg.compile_only_m_grouped_fp8_fp4_gemm_nt_masked(
                    int(s["g"]),
                    int(s["m"]),
                    int(s["n"]),
                    int(s["k"]),
                    int(s["expected_m"]),
                    getattr(torch, s["a"]),
                    getattr(torch, s["b"]),
                    tuple(s["recipe"]),
                )
            elif s["op"] == "mega_moe":
                built = dg.compile_only_fp8_fp4_mega_moe(
                    int(s["num_ranks"]), int(s["num_experts"]), int(s["max_tokens"]),
                    int(s["topk"]), int(s["num_tokens"]), int(s["hidden"]),
                    int(s["inter"]), 0, s["activation"], s["clamp"], bool(s["fast_math"]),
                    s["situ_beta"], s["situ_linear_beta"],
                )  # fmt: skip
            elif s["op"] == "paged_mqa_logits_metadata":
                built = dg.compile_only_paged_mqa_logits_metadata(
                    int(s["next_n"]), bool(s["is_varlen"]), int(s["num_sms"])
                )
            elif s["op"] == "mqa_logits":
                built = dg.compile_only_mqa_logits(
                    int(s["num_heads"]), int(s["head_dim"]), bool(s["is_fp4"]),
                    bool(s["is_mx_sf"]), bool(s["compressed"]),
                    getattr(torch, s["logits"]), getattr(torch, s["weights"]),
                )  # fmt: skip
            elif s["op"] == "paged_mqa_logits":
                built = dg.compile_only_paged_mqa_logits(
                    int(s["next_n"]), int(s["num_heads"]), int(s["head_dim"]),
                    int(s["block_kv"]), bool(s["is_fp4"]), bool(s["is_mx_sf"]),
                    bool(s["is_varlen"]), getattr(torch, s["logits"]),
                    getattr(torch, s["weights"]),
                )  # fmt: skip
            else:
                raise ValueError(f"unsupported op {s['op']}")
            resp = {"tag": tag, "ok": True, "s": time.time() - t0, "err": "", "built": bool(built)}
        except Exception as e:  # noqa: BLE001 - report, never die
            resp = {
                "tag": tag,
                "ok": False,
                "s": time.time() - t0,
                "err": f"{type(e).__name__}: {e}"[:500],
                "built": False,
            }
        out.write(json.dumps(resp) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
