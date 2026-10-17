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
"""Shared helpers of the fused_kv_indexer_projection tests and benchmarks.

* ``module(layer)``: the native ``DeepseekV32Attention`` of a GLM-5.2 layer
  at TP8 rank 0 with real checkpoint weights (absorbed_query's fixture; layer 78 is the
  MTP layer), with its real ``input_layernorm`` gain.
* ``native_hidden``: the tensor the native GEMMs consume: the native
  ``RMSNorm`` module (FlashInfer kernel) applied with the real gain to
  synthetic residual-stream rows; ``real_hidden``: rows of the real
  normalized MoE-input fixture (real-text prefill, GLM-5.2 NVFP4).
* ``native(attn, h)``: the native ``kv_a_proj_with_mqa`` call
  (``dsv3_fused_a_gemm_op``) and ``F.linear`` on the indexer's fused
  ``wk | weights_proj`` weight.
* error statistics against an FP64 oracle and the frozen budget (the deferred_norm
  budget for the same GEMM).
* ``proxy_producer``: a CuTe DSL stand-in for the preceding kernel. It is
  launched with programmatic dependent launch, releases its dependents at
  entry, spins for a given time and only then writes the activation; a
  dependent that read it before its dependency wait would see stale data.
"""

import glob
import os
import sys

import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "absorbed_query")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import absorbed_query_fixture  # noqa: E402

H = 6144
EPS = 1e-5
MTP_LAYER = 78
REAL_HIDDEN_DIR = os.environ.get("GLM_MOE_DSA_REAL_MOE_INPUTS_DIR", "")

_MODS = {}
_MODEL_CONFIG = {}


def model_config(tp_rank: int = 0):
    if tp_rank not in _MODEL_CONFIG:
        _MODEL_CONFIG[tp_rank] = absorbed_query_fixture.build_model_config(tp_rank)
    return _MODEL_CONFIG[tp_rank]


def module(layer: int):
    """``(attn, input_layernorm gain)`` of ``layer`` (cached)."""
    if layer not in _MODS:
        with torch.inference_mode():
            _MODS[layer] = (
                absorbed_query_fixture.build_native_attention(layer, 0, model_config(0)),
                absorbed_query_fixture.load_input_layernorm(layer).cuda(),
            )
    return _MODS[layer]


def make_residual(rows: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    """Residual-stream-like rows: Gaussian plus a few large channels."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, H, generator=g, device="cuda") * 2.0
    hot = torch.randint(0, H, (12,), generator=g, device="cuda")
    x[:, hot] *= 30.0
    return (x * scale).bfloat16()


_NORMS = {}


def native_rmsnorm(gamma: torch.Tensor):
    """The native ``RMSNorm`` module (as the decoder layers build it)."""
    key = gamma.data_ptr()
    if key not in _NORMS:
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        norm = RMSNorm(hidden_size=H, eps=EPS, dtype=torch.bfloat16).cuda()
        norm.weight.data.copy_(gamma)
        _NORMS[key] = norm
    return _NORMS[key]


def native_hidden(rows: int, seed: int, gamma: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Native RMSNorm output of synthetic residual rows, times ``scale``."""
    h = native_rmsnorm(gamma)(make_residual(rows, seed))
    if scale != 1.0:
        h = (h.float() * scale).bfloat16()
    return h.contiguous()


_REAL = []


def real_hidden(rows: int, offset: int = 0):
    """``rows`` rows of a real normalized hidden state, or ``None`` if the
    fixture is not mounted (first 2-D BF16 ``[N, 6144]`` tensor of the first
    file)."""
    if not _REAL:
        files = sorted(glob.glob(os.path.join(REAL_HIDDEN_DIR, "*.pt")))
        if not files:
            return None
        obj = torch.load(files[0], map_location="cpu", weights_only=False)
        cands = (
            obj.values()
            if isinstance(obj, dict)
            else (obj if isinstance(obj, (list, tuple)) else [obj])
        )
        found = None
        for t in cands:
            if (
                isinstance(t, torch.Tensor)
                and t.dim() == 2
                and t.shape[1] == H
                and t.dtype == torch.bfloat16
            ):
                found = t
                break
        if found is None:
            return None
        _REAL.append(found)
    t = _REAL[0]
    return t[offset : offset + rows].contiguous().cuda()


def native(attn, h: torch.Tensor):
    kva = attn.kv_a_proj_with_mqa(h)
    kw = None
    if attn.mqa.indexer is not None:
        kw = torch.nn.functional.linear(h, attn.mqa.indexer._fused_wk_wp_weight)
    return kva, kw


def fp64_oracle(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return h.double() @ weight.double().t()


def norm_rms_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.double()
    return float((y.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(t):
        v = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(v < 0, -(v + 32768), v)

    return (ordered(a) - ordered(b)).abs()


def compare(cand: torch.Tensor, nat: torch.Tensor, oracle: torch.Tensor) -> dict:
    ulp = bf16_ulp_distance(cand, nat)
    return {
        "bit_identical": bool(torch.equal(cand.view(torch.int16), nat.view(torch.int16))),
        "allclose_native": bool(torch.allclose(cand.float(), nat.float(), rtol=0.1, atol=1e-2)),
        "e_cand": norm_rms_err(cand, oracle),
        "e_native": norm_rms_err(nat, oracle),
        "max_ulp_vs_native": int(ulp.max()),
        "frac_diff_vs_native": float((ulp > 0).float().mean()),
        "max_abs_vs_native": float((cand.float() - nat.float()).abs().max()),
    }


def within_budget(stats: dict) -> bool:
    """Frozen budget of a non-bit-identical output."""
    return stats["allclose_native"] and stats["e_cand"] <= 1.25 * stats["e_native"] + 2.0**-12


# --------------------------------------------------------------------------
# Proxy producer (PDL): releases dependents at entry, writes after a delay.
# --------------------------------------------------------------------------


def _asm(res, args, text, cons, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _spin_ns(ns: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [ns.ir_value()],
        "{\n\t.reg .u64 t0, t;\n\t.reg .pred p;\n\tmov.u64 t0, %globaltimer;\n\t"
        "GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_SPIN:\n\tmov.u64 t, %globaltimer;\n\tsub.u64 t, t, t0;\n\t"
        "setp.lt.u64 p, t, $0;\n\t@p bra GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_SPIN;\n\t}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy16(dst: Int64, src: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [dst.ir_value(), src.ir_value()],
        "{\n\t.reg .b32 a, b, c, d;\n\tld.global.v4.b32 {a, b, c, d}, [$1];\n\t"
        "st.global.v4.b32 [$0], {a, b, c, d};\n\t}",
        "l,l",
        loc=loc,
        ip=ip,
    )


class _ProxyKernel:
    """Grid ``[8, rows]`` x 96 threads; CTA ``(c, r)`` copies 1536 B of row ``r``."""

    @cute.jit
    def __call__(
        self,
        src: Int64,
        dst: Int64,
        delay_ns: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        stream: CUstream,
    ):
        self.kernel(src, dst, delay_ns, anchor).launch(
            grid=[8, rows, 1], block=[96, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, src: Int64, dst: Int64, delay_ns: Int64, anchor: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        _spin_ns(delay_ns)
        off = Int64(row) * Int64(H * 2) + Int64((cta * Int32(96) + tid) * Int32(16))
        _copy16(dst + off, src + off)


_PROXY = {}


def proxy_producer(src: torch.Tensor, dst: torch.Tensor, delay_ns: int) -> None:
    """Launch the proxy producer on the current stream (PDL): ``dst <- src``
    (``[R, 6144]`` BF16) after spinning ``delay_ns``."""
    if "k" not in _PROXY:
        from cutlass.cute import runtime as cute_runtime

        z = Int64(0)
        anchor = cute_runtime.make_fake_compact_tensor(
            Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        _PROXY["k"] = cute.compile(
            _ProxyKernel(),
            z,
            z,
            z,
            anchor,
            Int32(1),
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    assert src.shape == dst.shape and src.shape[1] == H and src.is_contiguous()
    _PROXY["k"](
        src.data_ptr(), dst.data_ptr(), int(delay_ns), dst.view(torch.int32).view(-1), src.shape[0]
    )
