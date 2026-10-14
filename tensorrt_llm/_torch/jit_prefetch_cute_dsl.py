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


_BUILDERS = {_K123: _build_k123, _FI_NORM: _build_fi_norm}
# Kernels compiled with TVM-FFI are loaded as TVM-FFI functions.
_TVM_FFI = {_FI_NORM}


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
    return {
        "major": p.major,
        "minor": p.minor,
        "multi_processor_count": p.multi_processor_count,
        "shared_memory_per_block_optin": p.shared_memory_per_block_optin,
        "name": p.name,
    }


def install_device_shim(props: Dict[str, Any]) -> None:
    """In a helper process: answer the device queries of kernel constructors
    (``torch.cuda.get_device_properties`` and friends) with ``props``."""
    import torch

    ns = types.SimpleNamespace(**props)
    torch.cuda.get_device_properties = lambda device=None: ns
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (props["major"], props["minor"])


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

    ``loader(spec)`` returns a helper-exported kernel, or None to let
    FlashInfer compile it as usual. Each getter keeps FlashInfer's own cache
    (``functools.cache``), so a variant is resolved once per process either
    way. Returns the names of the getters that were wrapped.
    """
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
        inner = getattr(orig, "__wrapped__", orig)

        def make(inner=inner, op=op):
            import functools

            @functools.cache
            def getter(dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous=True):
                spec = fi_norm_spec(
                    op, dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous
                )
                fn = loader(spec)
                if fn is not None:
                    return fn
                return inner(dtype_str, H, weight_bias, enable_pdl, sm_version, contiguous)

            getter._jitp_wrapped = True
            return getter

        setattr(module, name, make())
        wrapped.append(name)
    return wrapped
