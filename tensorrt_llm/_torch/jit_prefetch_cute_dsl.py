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

* the fused K1+K2+K3 kernel of Kimi K3's KDA prefill
  (``_launch_fused_k123_inv``). Its compiled form depends only on constexpr
  specializers (heads, varlen, bias, safe gate, beta sigmoid, chunk alignment
  ``varlen_pure``, cu/ci dtypes); shapes are runtime arguments, and which
  variant a batch needs is known on the host before the forward pass.
* FlashInfer's CuTe DSL RMSNorm and fused add + RMSNorm (TVM-FFI kernels).
  The compiled form depends on dtype, hidden size, weight bias, PDL, SM
  version and whether the rows are contiguous (``flashinfer.norm``'s own
  cache key); every combination an engine's norm layers can reach is
  enumerated once from the model.

A spec is canonical JSON naming the kernel and its compile-time arguments;
its SHA-1 is the file name, so helper and executor agree without sharing
state. The helper runs with no visible GPU: the few device properties kernel
constructors read (shared memory per block, SM version) are taken from the
executor's device and answered by a shim (``device_props`` /
``install_device_shim``).
"""

# No ``from __future__ import annotations``: CuTe DSL reads the scalar
# annotations of the ``@cute.jit`` wrapper below at trace time and resolves
# string annotations against module globals, where ``cutlass`` is not bound.
import hashlib
import inspect
import json
import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

KIND = "cute_dsl"
_K123 = "kda_fused_k123"
_FI_NORM = "flashinfer_norm"
_BT = 64
_K_DIM = 128


def spec_key(spec: str) -> str:
    return hashlib.sha1(spec.encode()).hexdigest()[:20]


def k123_spec(
    H: int,
    has_bias: bool,
    safe_gate: bool,
    varlen_pure: bool,
    cu_dtype: str,
    ci_dtype: str,
    use_beta_sigmoid: bool,
    beta_dtype: str,
) -> str:
    """Spec for one varlen K123 variant (the prefill path; varlen fixes B=1)."""
    return json.dumps(
        {
            "kernel": _K123,
            "H": int(H),
            "has_bias": bool(has_bias),
            "safe_gate": bool(safe_gate),
            "varlen_pure": bool(varlen_pure),
            "cu": cu_dtype,
            "ci": ci_dtype,
            "beta_sigmoid": bool(use_beta_sigmoid),
            "beta": beta_dtype,
        },
        sort_keys=True,
    )


def k123_spec_from_cache_key(cache_key: tuple) -> Optional[str]:
    """Spec for a ``_fused_k123_cache`` key, or None if not a varlen key."""
    H, is_varlen, _dev, has_bias, safe_gate, varlen_pure, cu_ci, _t, beta_sig, beta_dt = cache_key
    if not is_varlen or cu_ci is None:
        return None
    return k123_spec(
        H,
        has_bias,
        safe_gate,
        varlen_pure,
        str(cu_ci[0]).replace("torch.", ""),
        str(cu_ci[1]).replace("torch.", ""),
        beta_sig,
        str(beta_dt).replace("torch.", ""),
    )


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


def _cutlass_int(name: str):
    import cutlass

    return {"int64": cutlass.Int64, "int32": cutlass.Int32}[name]


def _build_k123(s: Dict[str, Any]):
    """Return (host_fn, fake_args, compile_options) for one varlen K123 variant.

    The fake tensors reproduce the layouts ``_launch_fused_k123_inv`` passes:
    q/k/g [1, T, H, K] bf16, beta [1, T, H], scratch buffers from
    ``_get_buffers`` (T + one chunk of slack), cu [N+1], ci [NT, 2]. ``from_dlpack``
    wrappers are static-layout, so T / N / NT here are placeholders: the
    kernel reads every extent from its runtime scalar arguments.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl_kernels.blackwell.kimi_k3_kda.fused_k123 import make_host_function

    H, K, BT = s["H"], _K_DIM, _BT
    T, NT, N = 337, 7, 3  # placeholders; see docstring
    Ta = T + BT
    inner = make_host_function(
        1,
        NT,
        H,
        is_varlen=True,
        T_padded=T,
        has_bias=s["has_bias"],
        use_safe_gate=s["safe_gate"],
        use_beta_sigmoid=s["beta_sigmoid"],
        varlen_pure=s["varlen_pure"],
    )

    @cute.jit
    def host_fn(
        mQ, mK, mG, mA_log, mBeta, mBetaActivated, scale: cutlass.Float32,
        mKscaled, mKg, mQscaled, mGkLast, mAqk, mAkk, mCuSeqlens, mChunkIndices,
        mDtBias, lower_bound_val: cutlass.Float32, rt_nt: cutlass.Int32,
        rt_b: cutlass.Int32, rt_t_total: cutlass.Int32, valid_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):  # fmt: skip
        # export_to_c types scalars from annotations; the kernel's host_fn
        # leaves scale / lower_bound unannotated.
        inner(mQ, mK, mG, mA_log, mBeta, mBetaActivated, scale, mKscaled, mKg, mQscaled,
              mGkLast, mAqk, mAkk, mCuSeqlens, mChunkIndices, mDtBias, lower_bound_val,
              rt_nt, rt_b, rt_t_total, valid_tokens, stream)  # fmt: skip

    def t(dtype, *shape):
        stride, acc = [], 1
        for d in reversed(shape):
            stride.append(acc)
            acc *= d
        return cute.runtime.make_fake_tensor(
            dtype, tuple(shape), tuple(reversed(stride)), assumed_align=16
        )

    bf, f32 = cutlass.BFloat16, cutlass.Float32
    beta_t = f32 if s["beta"] == "float32" else bf
    cu_t, ci_t = _cutlass_int(s["cu"]), _cutlass_int(s["ci"])
    args = [
        t(bf, 1, T, H, K),  # q
        t(bf, 1, T, H, K),  # k
        t(bf, 1, T, H, K),  # g
        t(f32, H),  # A_log
        t(beta_t, 1, T, H),  # beta
        t(bf, 1, Ta, H),  # beta_activated
        cutlass.Float32(1.0),  # scale
        t(bf, 1, Ta, H, K),  # k_scaled
        t(bf, 1, Ta, H, K),  # kg
        t(bf, 1, Ta, H, K),  # q_scaled
        t(f32, 1, NT, H, K),  # gk_last_exp
        t(bf, 1, Ta, H, BT),  # A_qk
        t(bf, 1, Ta, H, BT),  # A_kk
        t(cu_t, N + 1),  # cu_seqlens
        t(ci_t, NT, 2),  # chunk_indices
        t(f32, H, K) if s["has_bias"] else t(f32, 1, 1),  # dt_bias
        cutlass.Float32(0.0),  # lower_bound
        cutlass.Int32(NT),
        cutlass.Int32(1),
        cutlass.Int32(T),
        cutlass.Int32(T),
        cute.runtime.make_fake_stream(),
    ]
    return host_fn, args, ""


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


# Radix top-k runners in custom_ops.cute_dsl_custom_ops whose ``_compile``
# builds the kernel from fake tensors and its cache key alone (TVM-FFI).
_TOPK = "cute_dsl_topk"
TOPK_RUNNERS = (
    "CuteDSLTopKDecodeSingleCTARunner",
    "CuteDSLTopKDecodeRadixFilterSPMultiCTARunner",
    "CuteDSLTopKPrefillSingleCTARunner",
    "CuteDSLTopKDecodeSinglePassMultiCTARunner",
    "CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner",
)


def topk_spec(runner: str, key: tuple) -> str:
    """Spec for one radix top-k runner's ``_compile(*key)``. The key holds
    cutlass dtypes, ints, bools and strings."""
    return json.dumps({"kernel": _TOPK, "runner": runner, "key": _encode(key)}, sort_keys=True)


def _encode(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return {"tuple": [_encode(x) for x in v]}
    name = getattr(v, "__name__", None)
    import cutlass

    if name is not None and getattr(cutlass, name, None) is v:
        return {"cutlass": name}
    raise TypeError(f"cannot encode top-k key element {v!r}")


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if "tuple" in v:
            return tuple(_decode(x) for x in v["tuple"])
        import cutlass

        return getattr(cutlass, v["cutlass"])
    return v


def _build_topk(s: Dict[str, Any]):
    """Return the compiled top-k kernel (``_compile`` builds it from fake
    tensors; the helper exports it)."""
    from .custom_ops import cute_dsl_custom_ops as ops

    runner = getattr(ops, s["runner"])
    key = _decode(s["key"])
    runner._compile(*key)
    cached = runner.kernel_cache[key]
    return cached[0] if isinstance(cached, tuple) else cached


_BUILDERS = {_K123: _build_k123, _FI_NORM: _build_fi_norm}
# Builders that return an already compiled kernel instead of (fn, args, options).
_PRECOMPILED = {_TOPK: _build_topk}
# Kernels compiled with TVM-FFI are loaded as TVM-FFI functions.
_TVM_FFI = {_FI_NORM, _TOPK}


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
    if s["kernel"] in _PRECOMPILED:
        # The kernel's own compile path (its options include the TVM-FFI
        # flag); the target arch comes from CUTE_DSL_ARCH in the helper.
        compiled = _PRECOMPILED[s["kernel"]](s)
    else:
        fn, args, options = _BUILDERS[s["kernel"]](s)
        options = f"{options} --gpu-arch {gpu_arch}".strip()
        compiled = cute.compile(fn, *args, options=options)
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
    try:  # the radix top-k cluster kernels clamp their cluster size to this
        from .cute_dsl_kernels.blackwell.top_k import single_pass_multi_cta_radix_topk_cluster as c

        props["max_cluster_size"] = int(c._query_max_cluster_size())
    except Exception:  # noqa: BLE001 - CuTe DSL missing, or the query failed
        pass
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


def install_device_shim(props: Dict[str, Any]) -> None:
    """In a helper process: answer the device queries of kernel constructors
    (``torch.cuda.get_device_properties`` and friends, the radix top-k max
    cluster size, the top-k runners' SM count) with ``props``."""
    import torch

    ns = types.SimpleNamespace(**props)
    torch.cuda.get_device_properties = lambda device=None: ns
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (props["major"], props["minor"])
    _shim_cute_dsl_device_attributes(props)
    if "max_cluster_size" in props:
        try:
            from .custom_ops import cute_dsl_custom_ops as ops
            from .cute_dsl_kernels.blackwell.top_k import (
                single_pass_multi_cta_radix_topk_cluster as cluster_mod,
            )
        except ImportError:
            return
        size = int(props["max_cluster_size"])
        cluster_mod._query_max_cluster_size = lambda: size
        if hasattr(ops, "_query_max_cluster_size"):
            ops._query_max_cluster_size = lambda: size


class KdaPrefillProvider:
    """K123 variants a Kimi K3 KDA prefill batch needs.

    The heads / bias / gate / beta settings come from the model's KDA layers
    (all layers share them); a batch selects ``varlen_pure`` from its context
    chunk lengths. Every variant is also queued once in the background, so a
    batch shape seen for the first time finds it compiled.
    """

    def __init__(self, model):
        from .modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

        cfgs = set()
        for mod in model.modules():
            if not isinstance(mod, KimiKDALinearAttention):
                continue
            H = getattr(mod, "num_heads", None)
            if H is None:
                continue
            has_bias = getattr(mod, "dt_bias", None) is not None
            # The mixer passes safe_gate = (gate_lower_bound is not None).
            safe_gate = getattr(mod, "gate_lower_bound", None) is not None
            cfgs.add((int(H), bool(has_bias), bool(safe_gate)))
        self.cfgs = sorted(cfgs)

    def __bool__(self):
        return bool(self.cfgs)

    def specs(self, varlen_pure: bool) -> List[str]:
        # The production mixer: in-kernel beta sigmoid, fp32 beta (b_proj
        # output .float()), int64 cu_seqlens / chunk_indices.
        return [
            k123_spec(H, b, sg, varlen_pure, "int64", "int64", True, "float32")
            for H, b, sg in self.cfgs
        ]

    def plan(self, ctx_chunk_lens: List[int]) -> List[str]:
        if not ctx_chunk_lens:
            return []
        # A single unaligned sequence is sentinel-padded onto the aligned
        # variant (see _chunk_kda_fwd), so only a multi-sequence batch with an
        # unaligned length needs the ragged one.
        pure = len(ctx_chunk_lens) == 1 or all(n % _BT == 0 for n in ctx_chunk_lens)
        return self.specs(pure)

    def enumerate_specs(self) -> List[str]:
        return self.specs(True) + self.specs(False)


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


def _modules_with_attention_backends(model) -> List[Any]:
    """``model.modules()`` plus the modules owned by its attention backends.

    An attention backend is a plain object, not an ``nn.Module``, and some
    sparse backends build their indexer (and its ``TopK``) in the first
    forward, so after warmup these are reachable only through the backend's
    attributes (``attn``, ``mqa``, ``mha``, ``backend`` of the attention
    modules, then the backend's ``indexer``).
    """
    import torch

    if model is None:
        return []
    out, seen = [], set()

    def add(mod):
        for m in mod.modules():
            if id(m) not in seen:
                seen.add(id(m))
                out.append(m)

    add(model)
    for mod in list(out):
        for name in ("attn", "mqa", "mha", "backend", "attn_backend"):
            backend = mod.__dict__.get(name)
            # Only plain Python backends: a module is already walked, and an
            # extension or slotted object has no attributes to look through.
            attrs = getattr(backend, "__dict__", None)
            if attrs is None or isinstance(backend, torch.nn.Module):
                continue
            for value in list(attrs.values()):
                if isinstance(value, torch.nn.Module):
                    add(value)
    return out


# The loader the wrapped top-k ``_compile``s consult (None: compile in place).
_topk_loader: Optional[Callable[[str], Optional[Callable]]] = None


def _wrap_topk_runners() -> List[str]:
    """Wrap each radix top-k runner's ``_compile`` once: under ``_KeyCapture``
    it records its key and stops the launch; otherwise it consults
    ``_topk_loader`` before compiling. Returns the names of the runners whose
    ``_compile`` is wrapped."""
    from .custom_ops import cute_dsl_custom_ops as ops

    wrapped = []
    for name in TOPK_RUNNERS:
        runner = getattr(ops, name, None)
        if runner is None:
            continue
        # A runner that inherits _compile gets its own wrapper: the spec names
        # the runner whose kernel_cache (and kernel class) the key belongs to.
        orig = inspect.getattr_static(runner, "_compile", None)
        if not isinstance(orig, classmethod):
            continue
        if getattr(orig.__func__, "_jitp_wrapped", False) and "_compile" in runner.__dict__:
            wrapped.append(name)
            continue
        inner = getattr(orig.__func__, "_jitp_inner", orig.__func__)

        def make(inner=inner, name=name):
            def _compile(cls, *key):
                capture = _KeyCapture.active()
                if capture is not None:
                    capture.append((name, key))
                    raise _KeyCaptured()
                if key in cls.kernel_cache:
                    return
                loader = _topk_loader
                fn = None
                if loader is not None:
                    try:
                        fn = loader(topk_spec(name, key))
                    except TypeError:
                        fn = None
                if fn is not None:
                    cls.kernel_cache[key] = fn
                    return
                inner(cls, *key)

            _compile._jitp_wrapped = True
            _compile._jitp_inner = inner
            return classmethod(_compile)

        setattr(runner, "_compile", make())
        wrapped.append(name)
    return wrapped


def install_topk_loader(loader: Callable[[str], Optional[Callable]]) -> List[str]:
    """Route the radix top-k runners' ``_compile`` through ``loader``.

    ``loader(spec)`` returns a helper-exported kernel, which is stored in the
    runner's ``kernel_cache`` under the key ``_compile`` would have used, or
    None to compile as before. Returns the names of the wrapped runners.
    """
    global _topk_loader
    _topk_loader = loader
    return _wrap_topk_runners()


class _KeyCaptured(Exception):
    """Raised by a wrapped top-k ``_compile`` under ``_KeyCapture``."""


class _KeyCapture:
    """Within the context, the wrapped top-k ``_compile``s record their key
    and return without compiling (the forward that follows fails on the
    missing cache entry, which the planner expects)."""

    _local = __import__("threading").local()

    def __enter__(self) -> List[Tuple[str, tuple]]:
        self._local.keys = []
        return self._local.keys

    def __exit__(self, *exc):
        self._local.keys = None
        return False

    @classmethod
    def active(cls) -> Optional[List[Tuple[str, tuple]]]:
        return getattr(cls._local, "keys", None)


class TopKProvider:
    """Every radix top-k variant the model's ``TopK`` modules can reach.

    A ``TopK`` with a CuTe DSL radix implementation compiles one kernel per
    key of the runner the dispatch picks: the score width bucket
    (``next_pow2`` of the score columns), and in decode the row count's band
    (single CTA, radix select, or a multi-CTA cluster size), plus the
    engine-fixed top_k / next_n / dtype. The provider replays the real
    dispatch (``torch.ops.trtllm.cute_dsl_indexer_topk_decode`` /
    ``..._prefill_blackwell``) on ``meta`` scores for every width bucket up to
    the longest score row and every row band up to the token budget, with the
    runners' compiles capturing their key instead of compiling.
    """

    def __init__(self, model, max_num_tokens: int, max_seq_len: int):
        import torch

        from .modules.top_k import TopK, TopKImplementation

        self.cfgs = set()
        radix = TopKImplementation.CUTE_DSL_RADIX
        for mod in _modules_with_attention_backends(model):
            if not isinstance(mod, TopK):
                continue
            for phase, impl in (
                ("prefill", mod.prefill_implementation),
                ("decode", mod.decode_implementation),
            ):
                if impl == radix:
                    self.cfgs.add((phase, int(mod.top_k)))
        self.max_rows = max(1, int(max_num_tokens))
        self.max_cols = max(1, int(max_seq_len))
        self.dtype = torch.float32

    def __bool__(self):
        return bool(self.cfgs)

    def _widths(self) -> List[int]:
        out, w = [], 16
        while w < 2 * self.max_cols:
            out.append(w)
            w *= 2
        return out

    def _rows(self, next_n: int) -> List[int]:
        # One row count per band of the decode dispatch (cluster sizes change
        # at 4 / 8 / 32 / 64 / SM count rows) and around the SM count.
        cands = {1, 2, 4, 5, 8, 9, 16, 32, 33, 64, 65, 128, 148, 152, 153, 160, 256, 512, 4096}
        rows = sorted(r for r in cands if r <= self.max_rows)
        return [max(next_n, r - r % next_n) for r in rows]

    def enumerate_specs(self, next_ns=(1,)) -> List[str]:
        import torch

        from .custom_ops import cute_dsl_custom_ops as ops
        from .modules.top_k import _CUTE_DSL_PREFILL_COPY_BITS

        # The ops' Python bodies (CustomOpDef._init_fn): calling the ops on
        # meta tensors would dispatch to their fake implementations.
        _wrap_topk_runners()  # so their _compile records the key and stops
        decode = getattr(ops.cute_dsl_indexer_topk_decode, "_init_fn", None)
        prefill = getattr(ops.cute_dsl_indexer_topk_prefill_blackwell, "_init_fn", None)
        specs = []
        meta = torch.device("meta")
        for phase, top_k in sorted(self.cfgs):
            body = prefill if phase == "prefill" else decode
            if body is None:
                continue
            # Scores narrower than top_k (short prefixes; CSA2 also selects
            # over compressed rows) still run the kernel, keyed on their width.
            for cols in self._widths():
                if phase == "prefill":
                    batches = [(1, 1)]
                else:
                    batches = [(r, n) for n in next_ns for r in self._rows(n)]
                for rows, next_n in batches:
                    scores = torch.empty(rows, cols, dtype=self.dtype, device=meta)
                    out = torch.empty(rows, top_k, dtype=torch.int32, device=meta)
                    with _KeyCapture() as keys:
                        try:
                            if phase == "prefill":
                                bounds = torch.empty(rows, dtype=torch.int32, device=meta)
                                body(scores, bounds, bounds, out, top_k,
                                     _CUTE_DSL_PREFILL_COPY_BITS)  # fmt: skip
                            else:
                                lens = torch.empty(rows // next_n, dtype=torch.int32, device=meta)
                                body(scores, lens, out, top_k, next_n)
                        except _KeyCaptured:
                            pass
                    for runner, key in keys:
                        specs.append(topk_spec(runner, key))
        return sorted(set(specs))


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
