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
``cute.runtime.load_module``: a ~1 ms load replaces a multi-second compile.

The first kernel covered is the fused K1+K2+K3 kernel of Kimi K3's KDA
prefill (``_launch_fused_k123_inv``). Its compiled form depends only on
constexpr specializers (heads, varlen, bias, safe gate, beta sigmoid, chunk
alignment ``varlen_pure``, cu/ci dtypes); shapes are runtime arguments. A
process therefore needs at most a handful of variants, and which one a batch
needs is known on the host before the forward pass (whether every context
chunk length is a multiple of 64).

A spec is canonical JSON naming the kernel and its compile-time arguments;
its SHA-1 is the file name, so helper and executor agree without sharing
state.
"""

# No ``from __future__ import annotations``: CuTe DSL reads the scalar
# annotations of the ``@cute.jit`` wrapper below at trace time and resolves
# string annotations against module globals, where ``cutlass`` is not bound.
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

KIND = "cute_dsl"
_K123 = "kda_fused_k123"
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


def _cutlass_int(name: str):
    import cutlass

    return {"int64": cutlass.Int64, "int32": cutlass.Int32}[name]


def _build_k123(s: Dict[str, Any]):
    """Return (host_fn, fake_args) for one varlen K123 variant.

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
    return host_fn, args


_BUILDERS = {_K123: _build_k123}


def build(spec: str, gpu_arch: str, out_dir: str) -> bool:
    """Compile ``spec`` and export it to ``<out_dir>/<key>.o``.

    Returns True if it compiled, False if the file already existed. The file
    appears atomically (written under a temporary name, then renamed).
    """
    import cutlass.cute as cute

    key = spec_key(spec)
    final = os.path.join(out_dir, f"{key}.o")
    if os.path.exists(final):
        return False
    s = json.loads(spec)
    host_fn, args = _BUILDERS[s["kernel"]](s)
    compiled = cute.compile(host_fn, *args, options=f"--gpu-arch {gpu_arch}")
    tmp_name = f".{key}.{os.getpid()}"
    compiled.export_to_c(out_dir, tmp_name, f"p{key}")
    os.replace(os.path.join(out_dir, f"{tmp_name}.o"), final)
    try:
        os.remove(os.path.join(out_dir, f"{tmp_name}.h"))
    except FileNotFoundError:
        pass
    return True


def load(spec: str, out_dir: str):
    """The compiled function for ``spec`` if a helper exported it, else None."""
    path = os.path.join(out_dir, f"{spec_key(spec)}.o")
    if not os.path.exists(path):
        return None
    import cutlass.cute as cute

    return cute.runtime.load_module(path)[f"p{spec_key(spec)}"]


def gpu_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}a"


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
