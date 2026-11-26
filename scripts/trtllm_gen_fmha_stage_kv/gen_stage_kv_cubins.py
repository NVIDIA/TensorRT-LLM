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
r"""Build the stage-KV trtllm-gen FMHA cubins (E4M3 Q/KV, BF16 out, paged causal context, fixed K/V stage).

Run on a GPU of the target arch (GB200 -> sm100a, GB300 -> sm103a) with TensorRT-LLM installed. The shipped cubins
(Qwen3.5 full attention, 32 / 2 heads under TP 1..8 and attention DP) come from:

  python3 scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py --work-dir <dir> \
      --heads 32:2,16:1,8:1,4:1,64:4 --head-dim 256 --tokens-per-page 64 --stage 256

For each kernel the runner selects over the context warmup grid (``TllmGenFmhaKernel::nvrtcConfig``), the generator in
libtensorrt_llm.so writes its source and cubin; the source gets the E4 edit below, is compiled with NVRTC and the
generator's options, and lands in ``cubin/`` as ``<Kernel>StageKv<N>E4.cubin.tar.zst`` with its entry in
``cubin/kernelMetaInfoStageKv.h``. The unedited source is compiled the same way and must reproduce the generator's
cubin byte for byte, which shows that the offline compile matches the runner's JIT.

E4 (no arithmetic change): in the softmax task's per-KV-tile loop, release the S tile (the "P ready" signal to the MMA
warp) before acquiring the next softmax-stats slot. That acquire waits on the correction task and guards nothing the MMA
needs, yet the generated order puts it between P(j) and the MMA of tile j + 1.
"""

import argparse
import ctypes
import hashlib
import importlib.util
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

REPO = Path(__file__).resolve().parents[2]
FMHA_DIR = REPO / "cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha"
CUBIN_DIR = FMHA_DIR / "cubin"
META_HEADER = CUBIN_DIR / "kernelMetaInfoStageKv.h"
EXPORT_DIR = FMHA_DIR / "trtllmGen_fmha_export"
# What the cubins are built against and the runner sets their parameters with: the kernel parameter block and its
# host-side setup (KernelParams.h takes FmhaOptions and KernelTraits), and the device headers the generated source
# compiles with (trtllm/). Their hash is recorded per block in META_HEADER and checked by
# tests/unittest/_torch/attention/test_fmha_paged_context_fp8.py, so a change to any of them forces a regeneration.
KERNEL_INPUT_HEADERS = (
    "KernelParams.h",
    "KernelParamsDecl.h",
    "KernelTraits.h",
    "KernelConfigBase.h",
    "FmhaOptions.h",
    "Dsv4Constants.h",
)
STAGE_KV_ARCHIVE_GLOB = "FmhaSm{sm}*StageKv*E4.cubin.tar.zst"

# Context-warmup grid of fmhaKernels.h (kContextWarmup* / kDefaultWarmupSeqLenKvCandidates), up to 128k tokens.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEQ_LENS_Q = [
    1,
    128,
    129,
    256,
    257,
    512,
    513,
    1024,
    1025,
    2048,
    2049,
    4096,
    4097,
    8192,
    8193,
    16384,
    16385,
    32768,
    32769,
    65536,
    65537,
    131072,
]
SEQ_LENS_KV = [
    1,
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1408,
    1536,
    1664,
    1792,
    1920,
    2048,
    2176,
    2304,
    2432,
    2560,
    2688,
    2816,
    2944,
    3072,
    3328,
    3584,
    3840,
    4096,
    4352,
    4608,
    4864,
    5120,
    5376,
    5632,
    5888,
    6144,
    6656,
    7168,
    7680,
    8192,
    8704,
    9216,
    9728,
    10240,
    10752,
    11264,
    11776,
    12288,
    13312,
    14336,
    16384,
    20480,
    24576,
    32768,
    40960,
    49152,
    65536,
    98304,
    131072,
]
# E4: the block that acquires the next softmax-stats slot, then releases the S tile; swapped. Matched on the source
# with comment-only lines ignored; each comment line moves with the code line after it, so the line count is unchanged.
ACQUIRE = """{
if ((loopOffsetN)>=(int32_t{0})) {
tmemSoftmaxLocal0ProdToken = tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
}
}
{
tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState, tmemSoftmaxLocal0ProdToken);
}""".split("\n")
RELEASE = """if (!(isLastLoopIter)) {
if ((loopOffsetN)>=(int32_t{0})) {
{
tmemS0SrcStack.mPipeline.consumer_release(tmemS0ConsReleaseState);
}
++tmemS0ConsReleaseState;
}
}""".split("\n")

HEADER_TEMPLATE = """/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Generated by scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py; do not edit.
#pragma once

#include "kernelMetaInfo.h"

#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// trtllm-gen FMHA kernels generated for a caller-fixed head dim per K/V shared-memory stage
// (TllmGenFmhaRunnerParams::mHeadDimPerStageKv). The kernel hash does not cover that option, so the runner keeps these
// apart from sTllmGenFmhaKernelMetaInfos and matches them only for callers that fix the stage.
struct TllmGenFmhaStageKvKernelMetaInfo
{
    int mHeadDimPerStageKv;
    TllmGenFmhaKernelMetaInfo mMeta;
};

// clang-format off
@DECLS@
inline std::vector<TllmGenFmhaStageKvKernelMetaInfo> const& getTllmGenFmhaStageKvKernelMetaInfos()
{
    static std::vector<TllmGenFmhaStageKvKernelMetaInfo> const sInfos = {
@ENTRIES@    };
    return sInfos;
}

// clang-format on

} // namespace kernels

TRTLLM_NAMESPACE_END
"""


def log(*args: object) -> None:
    print("[gen_stage_kv]", *args, flush=True)


def kernel_input_files(export_dir: Path = EXPORT_DIR) -> list[Path]:
    """The files kernel_inputs_sha256 covers, in hashing order."""
    files = [export_dir / name for name in KERNEL_INPUT_HEADERS]
    files += sorted(p for p in (export_dir / "trtllm").rglob("*") if p.is_file())
    return files


def kernel_inputs_sha256(export_dir: Path = EXPORT_DIR) -> str:
    """sha256 over the path (relative to export_dir) and bytes of every kernel_input_files entry."""
    h = hashlib.sha256()
    for path in kernel_input_files(export_dir):
        rel = path.relative_to(export_dir).as_posix().encode()
        data = path.read_bytes()
        h.update(len(rel).to_bytes(8, "little") + rel + len(data).to_bytes(8, "little") + data)
    return h.hexdigest()


def fail(msg: str) -> NoReturn:
    sys.exit(f"gen_stage_kv: {msg}")


@contextmanager
def redirect_fds(path: Path) -> Iterator[None]:
    """Send the process's stdout / stderr (C++ logging included) to a file."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with open(path, "w") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)


def trtllm_package_dir() -> Path:
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        fail("tensorrt_llm is not installed")
    return Path(list(spec.submodule_search_locations)[0])


def build_extension(build_dir: Path, fmha_build_dir: Path) -> ModuleType:
    import torch.utils.cpp_extension as ce

    pkg = trtllm_package_dir()
    cuda_home = Path(ce.CUDA_HOME or "/usr/local/cuda")
    json_include = REPO / "cpp/build/_deps/json-src/include"
    if not json_include.is_dir():
        # The fmha build dir of the installed package always exists; its CMake tree has the same _deps layout.
        json_include = (
            Path(os.path.realpath(fmha_build_dir / "cutlass")).parent / "json-src/include"
        )
    if not json_include.is_dir():
        fail(f"nlohmann json headers not found (looked in the repo and next to {fmha_build_dir})")
    includes = [
        REPO / "cpp",
        REPO / "cpp/include",
        FMHA_DIR,
        FMHA_DIR / "trtllmGen_fmha_export",
        json_include,
        cuda_home / "include",
        cuda_home / "include/cccl",
        fmha_build_dir / "cutlass/include",
        fmha_build_dir,
    ]
    flags = [
        "-O2",
        "-std=c++20",
        "-fvisibility=hidden",
        "-DTLLM_GEN_EXPORT_INTERFACE",
        "-DTLLM_FMHA_TRTLLM_COMPAT",
        "-DTLLM_RUBIN_FEATURES",
        "-DTLLM_ENABLE_CUDA",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DEXCLUDE_SM_100F",
        "-DTRTLLM_ABI_NAMESPACE=_v1",
        "-Wno-deprecated-declarations",
    ]
    libdir = pkg / "libs"
    os.makedirs(build_dir, exist_ok=True)
    return ce.load(
        name="trtllm_gen_fmha_stage_kv_export",
        sources=[str(Path(__file__).with_name("export_kernel.cpp"))],
        extra_include_paths=[str(p) for p in includes],
        extra_cflags=flags,
        extra_ldflags=[f"-L{libdir}", "-ltensorrt_llm", f"-Wl,-rpath,{libdir}", "-lcuda"],
        build_directory=str(build_dir),
        verbose=False,
    )


IDENTITY_KEYS = [
    "ctaDim",
    "tileSizeQ",
    "tileSizeKv",
    "stepQ",
    "stepKv",
    "headDimPerCtaV",
    "headDimQk",
    "headDimV",
    "headDimPerStageKv",
    "qkvLayout",
    "numTokensPerPage",
    "maskType",
    "kernelType",
    "tileScheduler",
    "multiCtasKvMode",
    "groupsHeadsQ",
    "groupsTokensHeadsQ",
    "reuseSmemKForV",
    "uses2CtaMma",
    "sparseAttn",
    "skipsSoftmax",
    "fp16Softmax",
    "usesSpcompress",
    "enablesBf16QFp8KvKOnlyTransform",
    "separateTransformedKv",
    "fusesDsv4InvRopeFp8Quant",
    "usesDsv4Ue8m0ScaleO",
    "clusterDimX",
    "json",
]


def enumerate_kernels(
    ext: ModuleType,
    sm: int,
    num_sms: int,
    heads: list[tuple[int, int]],
    head_dim: int,
    page: int,
    stage: int,
) -> dict[str, tuple[dict[str, object], list[tuple[int, ...]]]]:
    """Kernel name -> (identity, [shape args...]) over the grid; every shape of a name must give the same identity."""
    kernels = {}
    for (hq, hkv), b, q, kv in itertools.product(heads, BATCH_SIZES, SEQ_LENS_Q, SEQ_LENS_KV):
        if kv < q:
            continue
        args = (sm, num_sms, hq, hkv, head_dim, b, q, kv, page, stage)
        d = ext.select_kernel(*args)
        ident = {k: d[k] for k in IDENTITY_KEYS}
        if d["name"] not in kernels:
            kernels[d["name"]] = (ident, [args])
            continue
        if kernels[d["name"]][0] != ident:
            diff = {
                k: (kernels[d["name"]][0][k], ident[k])
                for k in IDENTITY_KEYS
                if kernels[d["name"]][0][k] != ident[k]
            }
            fail(f"{d['name']} has two configs over the grid: {diff}")
        kernels[d["name"]][1].append(args)
    return kernels


def make_exec_dir(exec_dir: Path, fmha_build_dir: Path) -> Path:
    """Create a stand-in for the runner's exec path.

    It links to the kernel headers of the installed build, so the generator reads the same headers but writes its
    dumps here instead of into the build tree.
    """
    exec_dir.mkdir(parents=True, exist_ok=True)
    for entry in ("cuda", "cuda_ptx", "cutlass", "trtllm", "KernelParams.h", "KernelParamsDecl.h"):
        if not (fmha_build_dir / entry).exists():
            fail(f"{fmha_build_dir / entry} is missing: the generator's kernel headers moved")
        link = exec_dir / entry
        if not link.is_symlink():
            link.symlink_to(os.path.realpath(fmha_build_dir / entry))
    return exec_dir


def generate(
    ext: ModuleType, args: tuple[int, ...], exec_dir: Path, out_dir: Path
) -> tuple[dict[str, object], Path, Path, str]:
    """Generator source, cubin and NVRTC options of the kernel selected for args, moved into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with redirect_fds(out_dir / "generator.log"):
        d = ext.generate_kernel(*args, f"{exec_dir}/")
    text = (out_dir / "generator.log").read_text(errors="replace")
    src = re.findall(r"Dumped NVRTC kernel source to: (\S+)", text)
    cub = re.findall(r"Dumped NVRTC CUBIN to: (\S+)", text)
    m = re.findall(r"NVRTC Compiler options: (.*)", text)
    if len(src) != 1 or len(cub) != 1 or not m:
        fail(
            f"generator output not found (sources {src}, cubins {cub}, options logged {len(m)}): the "
            f"emitsNvRtcSource / emitsCubin switches or the log format changed; see {out_dir / 'generator.log'}"
        )
    src, cub = [Path(shutil.move(p, out_dir / Path(p).name)) for p in (src[0], cub[0])]
    return d, src, cub, m[-1].strip()


def split_options(line: str) -> list[str]:
    """The generator logs its NVRTC options space-separated."""
    return [o for o in line.split(" ") if o]


# Local static of jitify2::detail::get_jitsafe_headers_map() and its guard variable.
JITIFY_MAP = "_ZZN7jitify26detailL23get_jitsafe_headers_mapEvE19jitsafe_headers_map"
JITIFY_MAP_GUARD = "_ZGVZN7jitify26detailL23get_jitsafe_headers_mapEvE19jitsafe_headers_map"


def jitify_headers(ext: ModuleType) -> dict[bytes, bytes]:
    """The built-in headers the generator's NVRTC front end (jitify2) supplies for <type_traits> etc.

    They are part of the generator's compile: the offline compile reproduces its cubin only with the same set. Read
    in-process from libtensorrt_llm.so after the first generation initialized the map.
    """
    lib = trtllm_package_dir() / "libs/libtensorrt_llm.so"
    syms = {}
    try:
        nm = subprocess.run(
            ["nm", "--defined-only", str(lib)], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as err:
        fail(f"cannot list the symbols of {lib}: {err}")
    for line in nm.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in (JITIFY_MAP, JITIFY_MAP_GUARD):
            syms.setdefault(parts[2], []).append(int(parts[0], 16))
    if any(len(syms.get(k, [])) != 1 for k in (JITIFY_MAP, JITIFY_MAP_GUARD)):
        fail(f"expected one jitify2 header map in {lib}, found {syms}: jitify2's symbols changed")
    real = os.path.realpath(lib)
    base = None
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and os.path.realpath(parts[5]) == real and int(parts[2], 16) == 0:
            base = int(parts[0].split("-")[0], 16)
            break
    if base is None:
        fail(f"{lib} is not mapped in this process")
    if ctypes.c_uint8.from_address(base + syms[JITIFY_MAP_GUARD][0]).value == 0:
        fail("the jitify2 header map is not initialized yet")
    headers = {k.encode(): v for k, v in ext.jitify_headers(base + syms[JITIFY_MAP][0]).items()}
    if not any(name.endswith(b"type_traits") for name in headers):
        fail(
            f"the jitify2 header map holds no <type_traits> ({len(headers)} entries): wrong symbol or layout"
        )
    return headers


# jitify2 adds these to the generator's options.
JITIFY_OPTIONS = ["-include=jitify_preinclude.h", "-default-device"]


def nvrtc_compile(
    source: str, name: str, options: list[str], headers: dict[bytes, bytes], log_path: Path
) -> bytes:
    """NVRTC compile as the generator does it: program name = kernel name, jitify2's headers in memory."""
    from cuda.bindings import nvrtc

    names, sources = list(headers), list(headers.values())
    err, prog = nvrtc.nvrtcCreateProgram(
        source.encode(), name.encode(), len(sources), sources, names
    )
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        fail(f"nvrtcCreateProgram failed for {name}: {err}")
    opts = [o.encode() for o in options + JITIFY_OPTIONS]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, n = nvrtc.nvrtcGetProgramLogSize(prog)
    buf = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, buf)
    Path(log_path).write_bytes(buf)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        fail(f"NVRTC failed for {name} ({err}); log {log_path}")
    _, n = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * n
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    nvrtc.nvrtcDestroyProgram(prog)
    return cubin


def apply_e4(source: str) -> str:
    lines = source.split("\n")
    code = [i for i, ln in enumerate(lines) if not ln.lstrip().startswith("//")]
    # The loop counter's generated name (loopOffset<id>) differs between kernels; it must be one name in the block.
    norm = [re.sub(r"\bloopOffset\d+\b", "loopOffsetN", lines[i]) for i in code]
    pattern = ACQUIRE + RELEASE
    hits = [k for k in range(len(code) - len(pattern) + 1) if norm[k : k + len(pattern)] == pattern]
    if len(hits) != 1:
        fail(
            f"expected one stats-acquire / S-release block, found {len(hits)}; the generator changed, re-derive "
            "the E4 edit"
        )
    k = hits[0]
    counters = {
        m for j in range(len(pattern)) for m in re.findall(r"\bloopOffset\d+\b", lines[code[k + j]])
    }
    if len(counters) != 1:
        fail(f"the E4 block uses several loop counters {counters}")
    # Each comment line belongs to the code line after it.
    first = code[k - 1] + 1 if k > 0 else 0
    split = code[k + len(ACQUIRE) - 1] + 1
    last = code[k + len(pattern) - 1] + 1
    edited = lines[:first] + lines[split:last] + lines[first:split] + lines[last:]
    assert len(edited) == len(lines)
    return "\n".join(edited)


def smem_bytes(cubin: bytes, func_name: str) -> int:
    """Dynamic shared memory of the kernel, from the <name>GetSmemSize helper the generator emits."""
    import torch
    from cuda.bindings import driver as cu

    def check(res: tuple[Any, ...]) -> Any:
        if res[0] != cu.CUresult.CUDA_SUCCESS:
            fail(f"CUDA driver call failed reading the shared memory of {func_name}: {res[0]}")
        return res[1] if len(res) > 1 else None

    torch.cuda.init()
    mod = check(cu.cuModuleLoadData(cubin))
    fn = check(cu.cuModuleGetFunction(mod, f"{func_name}GetSmemSize".encode()))
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    ptr = ctypes.c_void_p(out.data_ptr())
    params = (ctypes.c_void_p * 1)(ctypes.addressof(ptr))
    check(cu.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, ctypes.addressof(params), 0))
    torch.cuda.synchronize()
    check(cu.cuModuleUnload(mod))
    return int(out.item())


def write_archive(stem: str, cubin: bytes) -> None:
    with tempfile.TemporaryDirectory(dir=CUBIN_DIR.parent) as tmp:
        (Path(tmp) / f"{stem}.cubin").write_bytes(cubin)
        subprocess.run(
            [
                "cmake",
                "-E",
                "tar",
                "cf",
                str(CUBIN_DIR / f"{stem}.cubin.tar.zst"),
                "--zstd",
                "--mtime=1970-01-01UTC",
                "--",
                f"{stem}.cubin",
            ],
            cwd=tmp,
            check=True,
        )


def meta_entry(sm: int, stem: str, func: str, ident: dict[str, object], smem: int, sha: str) -> str:
    b = lambda v: "true" if v else "false"  # noqa: E731
    i = ident
    return (
        "{ %d, { DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_BF16, %d, %d, %d, %d, "
        '%d, %d, %d, kSM_%d, %s_cubin, %s_cubin_len, "%s", %d, %d, %d, %d, %d, %d, %d, %d, 0, 0, 0, 0, '
        '%s, %s, %s, %s, %d, %s, %s, %s, %s, %s, %s, %s, "%s", false } },\n'
    ) % (
        i["headDimPerStageKv"],
        i["tileSizeQ"],
        i["tileSizeKv"],
        i["stepQ"],
        i["stepKv"],
        i["headDimPerCtaV"],
        i["headDimQk"],
        i["headDimV"],
        sm,
        stem,
        stem,
        func,
        smem,
        i["ctaDim"],
        i["qkvLayout"],
        i["numTokensPerPage"],
        i["maskType"],
        i["kernelType"],
        i["tileScheduler"],
        i["multiCtasKvMode"],
        b(i["groupsHeadsQ"]),
        b(i["groupsTokensHeadsQ"]),
        b(i["reuseSmemKForV"]),
        b(i["uses2CtaMma"]),
        i["sparseAttn"],
        b(i["skipsSoftmax"]),
        b(i["fp16Softmax"]),
        b(i["usesSpcompress"]),
        b(i["enablesBf16QFp8KvKOnlyTransform"]),
        b(i["separateTransformedKv"]),
        b(i["fusesDsv4InvRopeFp8Quant"]),
        b(i["usesDsv4Ue8m0ScaleO"]),
        sha,
    )


_BLOCK_RE = r"// BEGIN %s sm(\d+) (\S+)\n#ifndef EXCLUDE_SM_\d+\n(.*?)#endif\n// END %s sm\1 \2\n"
_STEM_RE = r"extern unsigned char const (\w+)_cubin\[\];"


def config_key(head_dim: int, tokens_per_page: int, stage: int) -> str:
    """Header block key of one generator configuration (the kernels of other page sizes / stages live apart)."""
    return f"h{head_dim}p{tokens_per_page}s{stage}"


def write_header(sm: int, config: str, decls: str, entries: str) -> set[str]:
    """Replace this SM's blocks of this configuration in the header; keep the other SMs' and configurations'.

    Returns the cubin stems the header references for this SM afterwards.
    """
    blocks = {"decls": {}, "entries": {}}
    text = META_HEADER.read_text() if META_HEADER.exists() else ""
    for kind in blocks:
        for m in re.finditer(_BLOCK_RE % (kind, kind), text, re.S):
            blocks[kind][(int(m.group(1)), m.group(2))] = m.group(3)
    blocks["decls"][(sm, config)] = f"// Kernel inputs sha256: {kernel_inputs_sha256()}\n" + decls
    blocks["entries"][(sm, config)] = entries
    render = {
        kind: "".join(
            f"// BEGIN {kind} sm{s} {c}\n#ifndef EXCLUDE_SM_{s}\n{body}#endif\n// END {kind} sm{s} {c}\n"
            for (s, c), body in sorted(blocks[kind].items())
        )
        for kind in blocks
    }
    META_HEADER.write_text(
        HEADER_TEMPLATE.replace("@DECLS@", render["decls"]).replace("@ENTRIES@", render["entries"])
    )
    return {
        stem
        for (s, _), body in blocks["decls"].items()
        if s == sm
        for stem in re.findall(_STEM_RE, body)
    }


def remove_stale_archives(sm: int, keep: set[str]) -> list[Path]:
    """Delete this SM's stage-KV archives in cubin/ that no header entry references; return their paths.

    Those are kernels a regeneration no longer selects, which CMake would still link (it globs the directory).
    """
    stale = [
        path
        for path in sorted(CUBIN_DIR.glob(STAGE_KV_ARCHIVE_GLOB.format(sm=sm)))
        if path.name[: -len(".cubin.tar.zst")] not in keep
    ]
    for path in stale:
        path.unlink()
    return stale


def parse_heads(text: str) -> list[tuple[int, int]]:
    """'32:2,16:1' -> [(32, 2), (16, 1)]: (Q heads, KV heads) per rank of every layout the cubins serve."""
    heads = []
    for item in text.split(","):
        hq, sep, hkv = item.partition(":")
        if not sep or not hq.isdigit() or not hkv.isdigit() or int(hkv) == 0 or int(hq) % int(hkv):
            raise argparse.ArgumentTypeError(
                f"bad --heads entry {item!r}: want <q heads>:<kv heads>"
            )
        heads.append((int(hq), int(hkv)))
    return heads


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--work-dir", required=True, type=Path, help="generator output, extension build, logs"
    )
    ap.add_argument(
        "--heads",
        required=True,
        type=parse_heads,
        help="(Q heads):(KV heads) per rank of every layout, comma-separated, e.g. 32:2,16:1",
    )
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--tokens-per-page", type=int, required=True)
    ap.add_argument(
        "--stage", type=int, required=True, help="head dim per K/V stage the caller fixes"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="select and generate, but write nothing into the repo",
    )
    a = ap.parse_args()

    # The runner launches these kernels with PDL unless TRTLLM_ENABLE_PDL=0, so they must wait on their grid
    # dependency: generate them with PDL regardless of the caller's environment (read before the library loads).
    os.environ["TRTLLM_ENABLE_PDL"] = "1"
    import torch

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm not in (100, 103):
        fail(f"run on SM100 / SM103, got SM{sm}")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    work = a.work_dir.resolve() / f"sm{sm}"
    work.mkdir(parents=True, exist_ok=True)
    fmha_build_dir = (
        trtllm_package_dir().parent / "cpp/build/tensorrt_llm/kernels/trtllmGenKernels/fmha"
    )
    if not fmha_build_dir.is_dir():
        fail(f"{fmha_build_dir} not found: run from an editable install with its C++ build tree")
    ext = build_extension(work / "ext", fmha_build_dir)
    exec_dir = make_exec_dir(work / "exec", fmha_build_dir)

    kernels = enumerate_kernels(ext, sm, num_sms, a.heads, a.head_dim, a.tokens_per_page, a.stage)
    log(
        f"SM{sm} ({num_sms} SMs): {len(kernels)} kernel(s) over the grid:",
        ", ".join(f"{n} ({len(v[1])} shapes)" for n, v in sorted(kernels.items())),
    )

    decls, entries, headers = "", "", None
    for name, (ident, shapes) in sorted(kernels.items()):
        # Two shapes far apart in the grid must generate the same source.
        outs = [
            generate(ext, shapes[i], exec_dir, work / "gen" / name / f"shape{i}")
            for i in sorted({0, len(shapes) - 1})
        ]
        d, src_path, gen_cubin_path, opt_line = outs[0]
        source = src_path.read_text()
        for _, other_src, _, other_opts in outs[1:]:
            if other_src.read_text() != source or other_opts != opt_line:
                fail(f"{name}: shapes {shapes[0]} and {shapes[-1]} generate different kernels")
        options = split_options(opt_line)
        (work / "gen" / name / "nvrtc_options.txt").write_text("\n".join(options) + "\n")
        log(
            f"{name}: generator source {src_path} ({len(source)} bytes), options: {' '.join(options)}"
        )

        headers = headers or jitify_headers(ext)
        plain = nvrtc_compile(source, name, options, headers, work / "gen" / name / "plain.log")
        gen_cubin = gen_cubin_path.read_bytes()
        if plain != gen_cubin:
            fail(
                f"{name}: the offline compile does not reproduce the generator's cubin "
                f"({len(plain)} vs {len(gen_cubin)} bytes); fix the compile before trusting the E4 cubin"
            )
        log(
            f"{name}: offline compile reproduces the generator cubin ({len(plain)} bytes, "
            f"sha256 {hashlib.sha256(plain).hexdigest()[:16]})"
        )

        func = f"{name}StageKv{a.stage}E4"
        e4_source = apply_e4(source).replace(name, func)
        (work / "gen" / name / f"{func}.cu").write_text(e4_source)
        e4 = nvrtc_compile(e4_source, func, options, headers, work / "gen" / name / "e4.log")
        smem_plain, smem_e4 = smem_bytes(plain, name), smem_bytes(e4, func)
        if smem_plain != smem_e4:
            fail(f"{name}: smem differs after E4 ({smem_plain} vs {smem_e4})")
        sha = hashlib.sha256(e4).hexdigest()
        stem = func[0].upper() + func[1:]
        (work / "gen" / name / f"{stem}.cubin").write_bytes(e4)
        log(
            f"{func}: {len(e4)} bytes, smem {smem_e4}, {ident['ctaDim']} threads, sha256 {sha[:16]}"
        )
        decls += (
            f"extern unsigned char const {stem}_cubin[];\n"
            f"extern unsigned int const {stem}_cubin_len;\n"
        )
        entries += meta_entry(sm, stem, func, ident, smem_e4, sha)
        if not a.dry_run:
            write_archive(stem, e4)

    if not a.dry_run:
        config = config_key(a.head_dim, a.tokens_per_page, a.stage)
        keep = write_header(sm, config, decls, entries)
        log(
            f"wrote {len(kernels)} archive(s) to {CUBIN_DIR} and the SM{sm} {config} entries of {META_HEADER}"
        )
        for path in remove_stale_archives(sm, keep):
            log(f"removed {path.name}: no header entry references it")


if __name__ == "__main__":
    main()
