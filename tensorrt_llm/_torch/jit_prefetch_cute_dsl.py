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
"""CuTe DSL support for ``jit_prefetch.py``: compile in a helper, load in the executor.

``cute.compile`` never consults CuTe DSL's on-disk cache, so a helper cannot
warm a cache the executor reads. Instead the helper compiles the variant from
fake tensors with the real call's layout and an explicit ``--gpu-arch``,
exports it (``export_to_c``) to ``<dir>/<key>.o``, and the executor, at the
point it would call ``cute.compile``, loads that file with
``cute.runtime.load_module``: a ~1 ms load replaces a compile of 0.2 s to
several seconds.

Covered kernels, each a builder in ``_BUILDERS``:

* FlashInfer's CuTe DSL RMSNorm and fused add + RMSNorm (TVM-FFI kernels).
  The compiled form depends on dtype, hidden size, weight bias, PDL, SM
  version and whether the rows are contiguous (``flashinfer.norm``'s own
  cache key); every combination an engine's norm layers can reach is
  enumerated once from the model (``FlashInferNormProvider``).
* the NVFP4 dense GEMM's autotune tactics (``Nvfp4GemmProvider``): the
  TVM-FFI kernel of ``CuteDSLNVFP4BlackwellRunner``, one of the backends the
  unified NVFP4 GEMM op's autotuner races. It is shape-generic, so the
  tactic and the cluster occupancy select it.

A kernel family is added with a spec function, a builder (the kernel object
and the fake arguments its op compiles it with) in ``_BUILDERS``, a provider
that lists the specs a model reaches, and a loader the op calls where it
would call ``cute.compile``.

A spec is canonical JSON naming the kernel and its compile-time arguments;
its SHA-1 is the file name, so helper and executor agree without sharing
state. The helper runs with no visible GPU: the device properties kernel
constructors and the CuTe DSL compiler read (SM version, SM count, shared
memory) are taken from the executor's device and
answered by a shim (``device_props`` / ``install_device_shim``).
"""

import hashlib
import json
import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

KIND = "cute_dsl"
_FI_NORM = "flashinfer_norm"


def spec_key(spec: str) -> str:
    return hashlib.sha1(spec.encode(), usedforsecurity=False).hexdigest()[:20]


def fi_norm_spec(
    op: str,
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
    contiguous: bool,
) -> str:
    """Spec for one FlashInfer CuTe DSL norm variant.

    ``op`` is ``"rmsnorm"`` or ``"fused_add_rmsnorm"``; the other fields are
    the arguments of FlashInfer's ``_get_compiled_<op>_kernel`` getter, which
    is FlashInfer's own cache key for the compiled kernel.
    """
    return json.dumps(
        {
            "kernel": _FI_NORM,
            "op": op,
            "dtype": dtype_str,
            "H": int(H),
            "weight_bias": float(weight_bias),
            "pdl": bool(enable_pdl),
            "sm": int(sm_version),
            "contiguous": bool(contiguous),
        },
        sort_keys=True,
    )


def _build_fi_norm(s: Dict[str, Any]):
    """Return (kernel, fake_args, compile_options) for one FlashInfer norm.

    The kernel object and fake tensors are built exactly as FlashInfer's
    ``_get_compiled_rmsnorm_kernel`` / ``_get_compiled_fused_add_rmsnorm_kernel``
    build them (row count symbolic, compact or symbolically strided rows), so
    the export has the same TVM-FFI signature as the kernel FlashInfer would
    compile.
    """
    import math

    import cutlass.cute as cute
    from cutlass import Float32, Int64
    from flashinfer.norm.utils import get_cutlass_dtype

    dtype = get_cutlass_dtype(s["dtype"])
    H, contiguous = s["H"], s["contiguous"]
    if s["op"] == "rmsnorm":
        from flashinfer.norm.kernels.rmsnorm import RMSNormKernel as Kernel
    elif s["op"] == "fused_add_rmsnorm":
        from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel as Kernel
    else:
        raise ValueError(f"unknown FlashInfer norm op {s['op']!r}")
    kernel_obj = Kernel(dtype, H, s["weight_bias"], sm_version=s["sm"])
    sym_m = cute.sym_int(64)
    if contiguous:
        align = math.gcd(128, H * (dtype.width // 8))

        def row_tensor():
            return cute.runtime.make_fake_compact_tensor(
                dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
            )
    else:

        def row_tensor():
            stride = cute.sym_int64(divisibility=kernel_obj.vec_size)
            return cute.runtime.make_fake_tensor(dtype, (sym_m, H), (stride, 1), assumed_align=16)

    a, b = row_tensor(), row_tensor()
    w = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if s["op"] == "rmsnorm":
        args = [a, w, b]  # (input, weight, out)
    else:
        args = [a, b, w]  # (input, residual, weight)
    args += [Int64(1), Float32(1e-6), s["pdl"], stream]
    return kernel_obj, args, "--enable-tvm-ffi"


# The TVM-FFI NVFP4 dense GEMM of ``CuteDSLNVFP4BlackwellRunner``, one of the
# backends the unified NVFP4 GEMM op's autotuner races; keyed like the
# runner's ``kernel_cache`` plus the cluster occupancy it compiles with.
_NVFP4_GEMM = "cute_dsl_nvfp4_dense_gemm"


def nvfp4_gemm_spec(
    sf_vec_size: int,
    mma_tiler_mn,
    cluster_shape_mn,
    swap_ab: bool,
    use_prefetch: bool,
    swizzle_size: int,
    raster_along_m: bool,
    max_active_clusters: int,
) -> str:
    """Spec for one TVM-FFI ``CuteDSLNVFP4BlackwellRunner`` kernel."""
    return json.dumps(
        {
            "kernel": _NVFP4_GEMM,
            "sf_vec_size": int(sf_vec_size),
            "mma_tiler": [int(x) for x in mma_tiler_mn],
            "cluster": [int(x) for x in cluster_shape_mn],
            "swap_ab": bool(swap_ab),
            "use_prefetch": bool(use_prefetch),
            "swizzle_size": int(swizzle_size),
            "raster_along_m": bool(raster_along_m),
            "max_active_clusters": int(max_active_clusters),
        },
        sort_keys=True,
    )


def _build_nvfp4_gemm(s: Dict[str, Any]):
    """Return (wrapper, fake args, options) for one NVFP4 dense GEMM tactic.

    The runner passes every extent as a runtime 64-bit scalar, the operands
    as pointers (FP4 A/B 32-byte aligned, the FP8 scale factors and the BF16
    output 16-byte aligned) and alpha as the one-element FP32 tensor
    ``from_dlpack`` makes of it; the batch count (1), the occupancy and
    ``swap_ab`` are compile-time. So any address and extent compile the
    kernel the runner would compile for any shape.
    """
    import cutlass
    import cutlass.cute as cute

    from .kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )
    from .kernels.blackwell.utils import make_ptr

    gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        s["sf_vec_size"],
        tuple(s["mma_tiler"]),
        tuple(s["cluster"]),
        s["use_prefetch"],
        s["swizzle_size"],
        s["raster_along_m"],
    )

    def ptr(dtype, align):
        return make_ptr(dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=align)

    i64 = cutlass.Int64
    alpha = cute.runtime.make_fake_tensor(cutlass.Float32, (1,), (1,), assumed_align=16)
    args = [i64(1), i64(1), i64(1), i64(1), i64(1), i64(1), 1]  # m n k sf_m sf_n sf_k batch
    args += [ptr(cutlass.Float4E2M1FN, 32), ptr(cutlass.Float4E2M1FN, 32)]
    args += [ptr(cutlass.Float8E4M3FN, 16), ptr(cutlass.Float8E4M3FN, 16)]
    args += [ptr(cutlass.BFloat16, 16), alpha, s["max_active_clusters"]]
    args += [cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), s["swap_ab"]]
    return gemm.wrapper, (args, {}), "--opt-level 2 --enable-tvm-ffi"


_BUILDERS = {
    _FI_NORM: _build_fi_norm,
    _NVFP4_GEMM: _build_nvfp4_gemm,
}
# Kernels compiled with TVM-FFI are loaded as TVM-FFI functions.
_TVM_FFI = {_FI_NORM, _NVFP4_GEMM}


def build(spec: str, gpu_arch: str, out_dir: str) -> bool:
    """Compile ``spec`` and export it to ``<out_dir>/<key>.o``.

    Returns True if it compiled, False if the file already existed. The file
    appears atomically (written under a temporary name, then renamed). The
    ranks of a node share ``out_dir``; a per-variant lock makes one helper
    compile while the others wait and then find the file.
    """
    import fcntl

    key = spec_key(spec)
    final = os.path.join(out_dir, f"{key}.o")
    if os.path.exists(final):
        return False
    with open(os.path.join(out_dir, f".{key}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(final):
            return False
        _compile_export(spec, gpu_arch, out_dir, key, final)
    return True


def _compile_export(spec: str, gpu_arch: str, out_dir: str, key: str, final: str) -> None:
    import cutlass.cute as cute

    s = json.loads(spec)
    fn, args, options = _BUILDERS[s["kernel"]](s)
    args, kwargs = args if isinstance(args, tuple) else (args, {})
    options = f"{options} --gpu-arch {gpu_arch}".strip()
    compiled = cute.compile(fn, *args, **kwargs, options=options)
    tmp_name = f".{key}.{os.getpid()}"
    if s["kernel"] in _TVM_FFI:
        compiled.export_to_c(
            os.path.join(out_dir, f"{tmp_name}.o"),
            function_name=f"p{key}",
            export_only_tvm_ffi_symbols=True,
        )
    else:
        compiled.export_to_c(out_dir, tmp_name, f"p{key}")
    os.replace(os.path.join(out_dir, f"{tmp_name}.o"), final)
    try:
        os.remove(os.path.join(out_dir, f"{tmp_name}.h"))
    except FileNotFoundError:
        pass


def load(spec: str, out_dir: str):
    """The compiled function for ``spec`` if a helper exported it, else None."""
    path = os.path.join(out_dir, f"{spec_key(spec)}.o")
    if not os.path.exists(path):
        return None
    import cutlass.cute as cute

    tvm_ffi = json.loads(spec)["kernel"] in _TVM_FFI
    return cute.runtime.load_module(path, enable_tvm_ffi=tvm_ffi)[f"p{spec_key(spec)}"]


def gpu_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


def device_props() -> Dict[str, Any]:
    """The properties of the executor's device that kernel constructors read
    on the host (passed to the helpers, which have no visible GPU)."""
    import torch

    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    props = {
        "major": p.major,
        "minor": p.minor,
        "multi_processor_count": p.multi_processor_count,
        "shared_memory_per_block_optin": p.shared_memory_per_block_optin,
        "shared_memory_per_multiprocessor": p.shared_memory_per_multiprocessor,
        "name": p.name,
    }
    return props


def _shim_cute_dsl_device_attributes(props: Dict[str, Any]) -> None:
    """Answer CuTe DSL's own driver queries made while compiling (the shared
    memory per SM, for kernels with a minimum blocks-per-SM hint) from
    ``props``; any other attribute raises as it would without a GPU."""
    try:
        from cutlass.base_dsl.runtime import cuda as cuda_helpers
    except ImportError:
        return
    attrs = cuda_helpers.cuda.CUdevice_attribute
    known = {
        attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: props["major"],
        attrs.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: props["minor"],
        attrs.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: props["multi_processor_count"],
    }
    if "shared_memory_per_multiprocessor" in props:
        known[attrs.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR] = props[
            "shared_memory_per_multiprocessor"
        ]
    orig = cuda_helpers.get_device_attribute

    def get_device_attribute(attribute, device_id: int = 0):
        if attribute in known:
            return known[attribute]
        return orig(attribute, device_id)

    cuda_helpers.get_device_attribute = get_device_attribute


def install_occupancy_kernel_cache() -> bool:
    """Have CuTe DSL's ``HardwareInfo`` compile its occupancy probe kernel
    once per device instead of at every query.

    ``HardwareInfo.get_max_active_clusters`` compiles an empty kernel (into a
    fresh temporary dump directory, so CuTe DSL's compile cache never hits)
    to ask the driver for cluster occupancy, 60-100 ms each time, and the
    runners call it whenever they compile or load a kernel: in warmup, and
    for a kernel first needed while serving. The kernel is the same for
    every query on a device; only the driver queries that follow depend on
    the cluster size. Returns False if CuTe DSL is unavailable."""
    import threading

    try:
        from cutlass.utils.hardware_info import HardwareInfo
    except ImportError:
        return False
    orig = HardwareInfo._get_device_function
    if getattr(orig, "_jitp_cached", False):
        return True
    functions: Dict[int, Any] = {}
    lock = threading.Lock()

    def _get_device_function(self):
        with lock:
            fn = functions.get(self.device_id)
            if fn is None:
                fn = functions[self.device_id] = orig(self)
        return fn

    _get_device_function._jitp_cached = True
    _get_device_function._jitp_inner = orig
    HardwareInfo._get_device_function = _get_device_function
    return True


def install_device_shim(props: Dict[str, Any]) -> None:
    """In a helper process: answer the device queries of kernel constructors
    (``torch.cuda.get_device_properties`` and friends) with ``props``."""
    import torch

    ns = types.SimpleNamespace(**props)
    torch.cuda.get_device_properties = lambda device=None: ns
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (props["major"], props["minor"])
    _shim_cute_dsl_device_attributes(props)


class Nvfp4GemmProvider:
    """Every TVM-FFI NVFP4 dense GEMM tactic the model's NVFP4 layers reach.

    An NVFP4 ``Linear`` whose allowed backends include CuTe DSL (the default)
    runs ``trtllm::nvfp4_gemm``, whose autotuner benchmarks, among the other
    backends' tactics, every tactic ``CuteDSLNVFP4BlackwellRunner`` lists for
    each M bucket it tunes (powers of two up to ``max_num_tokens``), and the
    first benchmark of each compiles it on the executor thread (~0.5 s each,
    about a hundred per model). The runner's kernel is shape-generic, so the
    set follows from the layers' ``(n, k)`` and the M buckets, which the
    runner's own ``get_valid_tactics`` is asked for on ``meta`` tensors, and
    the cluster occupancy of this device.
    """

    def __init__(self, model, max_num_tokens: int):
        import torch

        from .._utils import get_sm_version
        from .modules.linear import Linear

        self.shapes: List[Tuple[int, int]] = []
        self.specs: List[str] = []
        if get_sm_version() not in (100, 103) or not max_num_tokens:
            return
        shapes = set()
        for mod in model.modules():
            if not isinstance(mod, Linear):
                continue
            if "cutedsl" not in (getattr(mod, "nvfp4_allowed_backends", None) or ()):
                continue
            # The quant mode, not the quant method: right after construction
            # the layers have no quant method yet, but their FP4 weights
            # (two values per byte, rows padded as the GEMM sees them) exist.
            mode = getattr(getattr(mod, "quant_config", None), "layer_quant_mode", None)
            if mode is None or not mode.has_nvfp4():
                continue
            w = getattr(mod, "weight", None)
            if w is None or w.dim() != 2 or w.dtype != torch.uint8:
                continue
            shapes.add((int(w.shape[0]), int(w.shape[1])))
        self.shapes = sorted(shapes)
        if self.shapes:
            self.specs = self._enumerate(int(max_num_tokens))

    def __bool__(self):
        return bool(self.specs)

    def _enumerate(self, max_num_tokens: int) -> List[str]:
        import cutlass
        import torch

        from .custom_ops.cute_dsl_custom_ops import CuteDSLNVFP4BlackwellRunner
        from .utils import get_last_power_of_2_num_tokens_buckets

        runner = CuteDSLNVFP4BlackwellRunner(torch.bfloat16)
        meta = torch.device("meta")
        tactics = set()
        for m in get_last_power_of_2_num_tokens_buckets(max_num_tokens):
            for n, k_bytes in self.shapes:
                a = torch.empty(m, k_bytes, dtype=torch.uint8, device=meta)
                b = torch.empty(n, k_bytes, dtype=torch.uint8, device=meta)
                for t in runner.get_valid_tactics([a, b], None):
                    tiler, cluster, swap_ab, prefetch, swizzle, raster = runner._unpack_tactic(t)
                    tactics.add((tuple(tiler), tuple(cluster), bool(swap_ab), bool(prefetch),
                                 int(swizzle), bool(raster)))  # fmt: skip
        # The cluster occupancy, computed as the runner computes it.
        hw = cutlass.utils.HardwareInfo()
        occupancy: Dict[int, int] = {}
        specs = []
        for tiler, cluster, swap_ab, prefetch, swizzle, raster in sorted(tactics):
            size = cluster[0] * cluster[1]
            if size not in occupancy:
                occupancy[size] = hw.get_max_active_clusters(size)
            specs.append(
                nvfp4_gemm_spec(
                    16, tiler, cluster, swap_ab, prefetch, swizzle, raster, occupancy[size]
                )  # fmt: skip
            )
        return specs

    def enumerate_specs(self) -> List[str]:
        return list(self.specs)


def install_nvfp4_gemm_loader(loader: Callable[[str], Optional[Callable]]) -> bool:
    """Route the TVM-FFI compiles of ``CuteDSLNVFP4BlackwellRunner`` through
    ``loader(spec)``; returns False if CuTe DSL is unavailable."""
    try:
        from .custom_ops import cute_dsl_custom_ops as ops
    except ImportError:
        return False

    def _loader(*key):
        return loader(nvfp4_gemm_spec(*key))

    ops.nvfp4_gemm_loader = _loader
    return True


class FlashInferNormProvider:
    """Every FlashInfer CuTe DSL RMSNorm variant the model's norm layers reach.

    ``RMSNorm`` (``modules/rms_norm.py``) calls FlashInfer's rmsnorm, or its
    fused add + RMSNorm when given a residual, for bf16/fp16 activations.
    FlashInfer keys the compiled kernel on (dtype, hidden size, weight bias,
    PDL, SM version, contiguous rows); a norm's hidden size and dtype are
    fixed, and PDL and the SM version are fixed per process, so each layer
    shape reaches at most four variants (plain or fused add, contiguous or
    strided rows). All of them are enumerated up front; a batch adds nothing.
    """

    def __init__(self, model):
        import torch

        from .flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
        from .modules.rms_norm import RMSNorm

        self.specs: List[str] = []
        if not IS_FLASHINFER_AVAILABLE or not _flashinfer_norm_uses_cute_dsl():
            return
        from flashinfer.norm.utils import get_sm_version
        from flashinfer.utils import device_support_pdl

        device = torch.device("cuda", torch.cuda.current_device())
        pdl = bool(get_env_enable_pdl()) and device_support_pdl(device)
        sm = get_sm_version(device)
        shapes = set()
        for mod in model.modules():
            if not isinstance(mod, RMSNorm) or getattr(mod, "use_cuda_tile", False):
                continue
            if (
                getattr(mod, "is_nvfp4", False)
                and getattr(mod, "nvfp4_scale", None) is not None
                and not getattr(mod, "use_gemma", False)
            ):
                continue  # the fused RMSNorm + NVFP4 quantize kernel, not FlashInfer
            weight = getattr(mod, "weight", None)
            if weight is None or weight.dtype not in (torch.bfloat16, torch.float16):
                continue
            dtype_str = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[weight.dtype]
            bias = 1.0 if getattr(mod, "use_gemma", False) else 0.0
            shapes.add((dtype_str, int(weight.shape[-1]), bias))
        self.shapes = sorted(shapes)
        for dtype_str, H, bias in self.shapes:
            for op in ("rmsnorm", "fused_add_rmsnorm"):
                for contiguous in (True, False):
                    self.specs.append(fi_norm_spec(op, dtype_str, H, bias, pdl, sm, contiguous))

    def __bool__(self):
        return bool(self.specs)

    def enumerate_specs(self) -> List[str]:
        return list(self.specs)


def _flashinfer_norm_uses_cute_dsl() -> bool:
    """Whether FlashInfer's norm dispatch takes its CuTe DSL kernels here."""
    try:
        import torch
        from flashinfer import norm as fi_norm
    except ImportError:
        return False
    use_cuda = getattr(fi_norm, "_use_cuda_norm", None)
    if use_cuda is None:
        return False
    return not use_cuda(torch.device("cuda", torch.cuda.current_device()))


def install_flashinfer_norm_loader(
    loader: Callable[[str], Optional[Callable]],
) -> List[str]:
    """Route FlashInfer's compiled-norm getters through ``loader``.

    ``loader(spec)`` returns a helper-exported kernel, or None. On None the
    original getter runs, called the way FlashInfer calls it, so a variant
    warmup already compiled comes from FlashInfer's own cache and any other
    compiles in place as before. Each wrapped getter caches its result
    (``functools.cache``), so a variant is resolved once per process. Returns
    the names of the getters that were wrapped.
    """
    import functools

    wrapped = []
    targets: List[Tuple[Any, str, str]] = []
    try:
        from flashinfer.norm.kernels import fused_add_rmsnorm as fa_mod
        from flashinfer.norm.kernels import rmsnorm as rms_mod
    except ImportError:
        return wrapped
    targets.append((rms_mod, "_get_compiled_rmsnorm_kernel", "rmsnorm"))
    targets.append((fa_mod, "_get_compiled_fused_add_rmsnorm_kernel", "fused_add_rmsnorm"))
    for module, name, op in targets:
        orig = getattr(module, name, None)
        if orig is None or getattr(orig, "_jitp_wrapped", False):
            continue

        def make(orig=orig, op=op):
            @functools.cache
            def getter(dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous=True):
                spec = fi_norm_spec(
                    op, dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous
                )
                fn = loader(spec)
                if fn is not None:
                    return fn
                # FlashInfer passes ``contiguous`` by keyword; the same call
                # form finds the entries its cache already holds.
                return orig(
                    dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous=contiguous
                )

            getter._jitp_wrapped = True
            getter._jitp_original = orig
            return getter

        setattr(module, name, make())
        wrapped.append(name)
    return wrapped


# ---------------------------------------------------------------------------
# What an engine builds ahead, and when (called from jit_prefetch_engine.py)
# ---------------------------------------------------------------------------


def start_model_builds(model, rank: int, llm_args) -> bool:
    """Start the CuTe DSL helpers on the kernels ``model``'s layers compile
    in warmup's first forwards, once the model is constructed, so that the
    helpers' start-up and these builds overlap the weight load: the NVFP4
    dense GEMM tactics the NVFP4 GEMM's autotuner races
    (``Nvfp4GemmProvider``, M buckets up to the engine's ``max_num_tokens``)
    and FlashInfer's CuTe DSL RMSNorm variants (``FlashInferNormProvider``).
    Returns whether anything was started."""
    from tensorrt_llm.logger import logger

    started = False
    try:
        from .cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not IS_CUTLASS_DSL_AVAILABLE:
            return False
        from .jit_prefetch import JitPrefetcher

        # Every later cluster-occupancy query reuses one probe kernel.
        install_occupancy_kernel_cache()
        # They follow from the layers' geometry and are compiled in warmup's
        # first forwards: the NVFP4 GEMM tactics by the autotuner, the
        # FlashInfer norms by the first forward.
        nvfp4_provider = Nvfp4GemmProvider(model, getattr(llm_args, "max_num_tokens", None) or 0)
        norm_provider = FlashInferNormProvider(model)
        if not (nvfp4_provider or norm_provider):
            return False
        prefetcher = JitPrefetcher.init(rank)
        if prefetcher is None:
            return False
        # Queued in the order warmup needs them: the norms in the first
        # forward, ahead of the GEMM tactics.
        if norm_provider:
            started = prefetcher.enable_flashinfer_norm(norm_provider) or started
        if nvfp4_provider:
            started = prefetcher.enable_nvfp4_gemm(nvfp4_provider) or started
    except Exception as e:  # noqa: BLE001 - the kernels then compile in place
        logger.warning(f"[JIT prefetch] CuTe DSL builds not planned: {type(e).__name__}: {e}")
    return started


def register_providers(prefetcher, engine) -> None:
    """At the end of warmup, queue the FlashInfer norm variants of
    ``engine``'s loaded model not queued once the model was constructed."""
    from .._utils import get_sm_version
    from .cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103):
        return
    norm_provider = FlashInferNormProvider(engine.model)
    if norm_provider:
        prefetcher.enable_flashinfer_norm(norm_provider)
