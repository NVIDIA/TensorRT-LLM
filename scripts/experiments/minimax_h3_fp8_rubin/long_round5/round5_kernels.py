# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and qualify the two fixed SM107 conversion kernels outside request timing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import cuda.bindings.driver as cuda
import torch
from round5_pdl_edges import prove as prove_pdl_edges
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
THREADS = 256
BLOCKS = 848
GUARD = 256
COEFFICIENTS = (1, 40503, 19373, 32769, 65535, 25173, 13849, 45377)
SHIFTS = (0, 1123, 4591, 32768, 65535, 173, 8191, 49157)
TAILS = (1, 2, 3, 7, 8, 9, 15, 16, 17, 255, 256, 257, 2047, 2048, 2049, 4097)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def guarded(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    count = 1
    for value in shape:
        count *= value
    width = torch.empty((), dtype=dtype).element_size()
    owner = torch.full((count * width + 2 * GUARD,), 165, dtype=torch.uint8, device="cuda")
    return owner, owner[GUARD:-GUARD].view(dtype).reshape(shape)


def guards_ok(owner: torch.Tensor) -> bool:
    return bool((owner[:GUARD] == 165).all()) and bool((owner[-GUARD:] == 165).all())


def build(name: str, cache: Path) -> ModuleType:
    """Compile a content-addressed extension with precise FP32 operations."""
    sources = [ROOT / (name + suffix) for suffix in (".cpp", ".cu")]
    flags = ["-O3", "-lineinfo", "-arch=sm_107a", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]
    digest = hashlib.sha256(("".join(sha(p) for p in sources) + repr(flags)).encode()).hexdigest()
    folder = cache / (name + "-" + digest[:16])
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    return load(
        name="h3_long5_" + name + "_" + digest[:16],
        sources=[str(p) for p in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=flags + ["--keep", "--keep-dir=" + str(folder)],
        build_directory=str(folder),
        verbose=True,
    )


def inspect_kernel(module: ModuleType, mode: int, *, pdl: bool, vector: bool = True) -> dict:
    """Inspect the actual linked CUfunction and its SASS before any launch."""
    values = list(module.kernel_info(mode, torch.cuda.current_device()))
    handle = cuda.CUfunction(values[0])
    status, name = cuda.cuFuncGetName(handle)
    assert int(status) == 0
    name = name.decode() if isinstance(name, bytes) else str(name)
    observed = {}
    for field, attribute in (
        ("regs", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS),
        ("local", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        ("max_threads", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        ("static_smem", cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
    ):
        status, value = cuda.cuFuncGetAttribute(attribute, handle)
        assert int(status) == 0
        observed[field] = int(value)
    assert list(observed.values()) == values[1:5]
    assert observed["local"] == observed["static_smem"] == 0
    assert 0 < observed["regs"] * THREADS <= 65536 and observed["max_threads"] >= THREADS
    tool = shutil.which("cuobjdump")
    if tool is None:
        raise RuntimeError("cuobjdump is required before experimental GPU execution")
    result = subprocess.run(
        [tool, "--dump-sass", module.__file__],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    parts = re.split(r"Function\s*:\s*(\S+)", result.stdout)
    sections = [parts[i + 1] for i in range(1, len(parts), 2) if parts[i] == name]
    assert len(sections) == 1
    sass = sections[0]
    if vector:
        assert re.search(r"\bLDG\S*128\b", sass)
        assert re.search(r"\bSTG\S*" + ("64" if pdl else "128") + r"\b", sass)
    if pdl:
        lines = sass.splitlines()
        waits = [i for i, line in enumerate(lines) if "ACQBULK" in line]
        reads = [i for i, line in enumerate(lines) if re.search(r"\bLDG(?:[.\s])", line)]
        assert waits and reads and waits[0] < min(reads) and "PREEXIT" in sass
    else:
        assert "ACQBULK" not in sass and "PREEXIT" not in sass
    return dict(name=name, driver=observed, extension_sha256=sha(Path(module.__file__)))


def capture_launch(
    module: ModuleType,
    raw: torch.Tensor,
    out: torch.Tensor,
    resource: dict,
    *,
    scale: torch.Tensor | None = None,
    factor: torch.Tensor | None = None,
    pdl: bool = True,
) -> dict:
    """Inspect one uninstantiated graph and require an untouched poisoned output."""
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(graph):
            if scale is None:
                module.out(raw, out, 1, BLOCKS, THREADS)
            else:
                module.out_config(raw, scale, factor, out, 0, pdl, BLOCKS, THREADS)
        assert not graph._has_graph_exec
        handle = cuda.CUgraph(graph.raw_cuda_graph())
        status, _, count = cuda.cuGraphGetNodes(handle, 0)
        assert int(status) == 0 and int(count) == 1
        status, nodes, count = cuda.cuGraphGetNodes(handle, count)
        assert int(status) == 0 and int(count) == 1
        status, params = cuda.cuGraphKernelNodeGetParams(nodes[0])
        assert int(status) == 0
        status, name = cuda.cuFuncGetName(params.func)
        assert int(status) == 0
        name = name.decode() if isinstance(name, bytes) else str(name)
        assert name == resource["name"]
        assert [params.gridDimX, params.gridDimY, params.gridDimZ] == [BLOCKS, 1, 1]
        assert [params.blockDimX, params.blockDimY, params.blockDimZ] == [THREADS, 1, 1]
        assert params.sharedMemBytes == 0 and bool((out.reshape(-1).view(torch.uint8) == 165).all())
        return dict(
            grid=[BLOCKS, 1, 1],
            block=[THREADS, 1, 1],
            capture_only=True,
            never_instantiated_or_replayed=True,
            poison_unchanged=True,
            pdl=int(pdl) if scale is not None else None,
        )
    finally:
        graph.reset()


def prepare(scales: torch.Tensor, cache: Path) -> tuple[ModuleType, ModuleType, torch.Tensor, dict]:
    """Qualify fresh binaries against installed GPU operators, including all BF16 bit patterns.

    ``scales`` is the frozen model's contiguous CUDA FP32 tensor of shape [50].
    This preparation runs before request timing and is repeated for each new process.
    """
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(ROOT / name) == expected, name
    properties = torch.cuda.get_device_properties(0)
    assert (
        torch.cuda.get_device_capability(0) == (10, 7) and properties.multi_processor_count == 212
    )
    assert scales.is_cuda and scales.dtype == torch.float32 and scales.shape == (50,)
    assert scales.is_contiguous() and bool(scales.isfinite().all()) and bool((scales > 0).all())
    pdl = os.environ.get("TRTLLM_ENABLE_PDL") != "0"
    quant, cast = build("staticQuant", cache), build("sharedCast", cache)
    resources = {
        "quant": inspect_kernel(quant, 0, pdl=True),
        "factor": inspect_kernel(quant, 3, pdl=True, vector=False),
        "expression_factor": inspect_kernel(quant, 4, pdl=True, vector=False),
        "cast": inspect_kernel(cast, 1, pdl=False),
    }
    factors = quant.factors(scales, True, pdl)
    expression_factors = quant.factors(scales, False, pdl)
    assert same(factors, expression_factors)
    small = torch.zeros((1, 9), device="cuda", dtype=torch.bfloat16)
    owner_q, out_q = guarded(tuple(small.shape), torch.float8_e4m3fn)
    owner_c, out_c = guarded(tuple(small.shape), torch.float32)
    captures = {
        "quant": capture_launch(
            quant, small, out_q, resources["quant"], scale=scales[:1], factor=factors[:1], pdl=pdl
        ),
        "cast": capture_launch(cast, small, out_c, resources["cast"], pdl=False),
    }
    assert guards_ok(owner_q) and guards_ok(owner_c)
    info = quant.kernel_info(0, torch.cuda.current_device())
    captures["PDL_edges"] = prove_pdl_edges(torch, cuda, quant, cuda.CUfunction(info[0]))
    indices = torch.arange(65536, dtype=torch.int32)
    bits = (
        torch.stack([((indices * c + s) & 65535) for c, s in zip(COEFFICIENTS, SHIFTS)], dim=1)
        .to(torch.int16)
        .contiguous()
    )
    assert all(torch.unique(bits[:, j]).numel() == 65536 for j in range(8))
    raw = bits.view(torch.bfloat16).cuda()
    owner, out = guarded(tuple(raw.shape), torch.float8_e4m3fn)
    for index in range(50):
        scale, factor = scales[index : index + 1], factors[index : index + 1]
        native = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(raw, scale)[0]
        quant.out_config(raw, scale, factor, out, 0, pdl, BLOCKS, THREADS)
        assert same(out, native) and guards_ok(owner)
    owner, out = guarded(tuple(raw.shape), torch.float32)
    cast.out(raw, out, 1, BLOCKS, THREADS)
    assert same(out, raw.float()) and guards_ok(owner)
    for length in TAILS:
        tail = (
            ((torch.arange(length, dtype=torch.int32) * 40503 + 32640) & 65535)
            .to(torch.int16)
            .view(torch.bfloat16)
            .reshape(1, length)
            .cuda()
        )
        oq, q = guarded(tuple(tail.shape), torch.float8_e4m3fn)
        oc, c = guarded(tuple(tail.shape), torch.float32)
        quant.out_config(tail, scales[:1], factors[:1], q, 0, pdl, BLOCKS, THREADS)
        cast.out(tail, c, 1, BLOCKS, THREADS)
        assert same(q, torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(tail, scales[:1])[0])
        assert same(c, tail.float()) and guards_ok(oq) and guards_ok(oc)
    torch.cuda.synchronize()
    proof = dict(
        source_manifest_sha256=sha(ROOT / "source_manifest.json"),
        unary_qualified=True,
        scales=50,
        BF16_patterns_each_position=65536,
        positions=8,
        tails=list(TAILS),
        captures=captures,
        resources=resources,
        factor_expression_vs_RN_bits_equal=True,
        pdl=int(pdl),
        scale_bits_u32=[int(x) & 0xFFFFFFFF for x in scales.cpu().view(torch.int32).tolist()],
        factor_bits_u32=[int(x) & 0xFFFFFFFF for x in factors.cpu().view(torch.int32).tolist()],
        source_files={p.name: sha(p) for p in ROOT.glob("*") if p.suffix in (".cpp", ".cu")},
    )
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "qualification.json").write_text(json.dumps(proof, indent=2) + "\n")
    return quant, cast, factors, proof
