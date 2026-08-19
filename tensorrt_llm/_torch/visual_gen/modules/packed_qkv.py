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
"""Packed joint-QKV projection recipes (concat elimination for double-stream DiT).

The packed projection writes both per-stream merged-QKV projections straight
into row slices of one packed buffer ``[1, S_txt + S_img, q_dim + 2*kv_dim]``,
replacing the per-stream projections + seq-dim ``torch.cat``. Under the
production per-block ``torch.compile`` the cat materializes as an inductor
``addmm+cat`` kernel; and the naive fix — ``addmm(out=slice)`` inlined in the
compiled region — is functionalized (the mutation is rewritten to a temp +
write-back) and inductor does not re-inplace extern kernels into slice views,
so a cat-sized copy-back kernel is emitted instead (measured 2.3% of a
Qwen-Image denoise step). Housing the mutation inside a functional
``torch.library`` custom op (same pattern as ``trtllm::fused_dit_qk_norm_rope``)
makes it an opaque extern that OWNS its output buffer: the compiler never sees
the internal ``out=`` writes and no copy-back exists.

Organization — every GEMM recipe of the packed projection lives in this file
as a triple, so recipes stay next to each other and dispatch happens in one
place:

1. a FUNCTIONAL leaf op ``trtllm_vgoa::packed_qkv_proj_<recipe>``
   (``mutates_args=()``, ``register_fake``; allocates the packed buffer and
   projects into its row slices),
2. a per-Linear capability census (may this Linear be computed by the
   recipe's leaf op on its native parameters, bypassing ``Linear.forward``?),
3. a builder that pulls the merged Linears' native parameters and calls the
   leaf op,

registered together in ``_PACKED_QKV_RECIPES``. Models interact only with the
two dispatch entry points and stay recipe-agnostic:

- :func:`select_packed_qkv_recipe` — post-load census, returns a recipe key
  or ``None`` (caller falls back to merged forward + seq-dim cat, which is
  correct for every configuration),
- :func:`build_packed_qkv` — call-time dispatch to the selected recipe.

Extension contract (FP8/NVFP4): a quantized recipe is the same triple — a
sibling leaf op whose eager body calls a GEMM that accepts a caller-provided
output (FlashInfer's ``gemm_fp8_nt_groupwise`` / ``mm_fp4`` expose ``out=``;
TRT-LLM's own thop GEMMs allocate their outputs and cannot be used), a census
that strict-type-checks the matching ``LinearMethod`` and validates its scale
parameters, and a builder passing the merged Linear's native weight/scale
tensors (which ``WeightMode.FUSED_QKV_LINEAR`` loading already provides).
Register the triple in ``_PACKED_QKV_RECIPES`` and every model using this
seam picks it up unchanged. Leaf ops must not be routed through
``LinearMethod.apply``: a custom op only takes plain tensors, and any
``out=``-mutation inlined OUTSIDE an op body re-emits the copy-back above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch

from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod


@dataclass(frozen=True)
class PackedQKVRecipe:
    """One GEMM recipe of the packed joint-QKV projection."""

    # Per-Linear capability census: (linear, *, out_features, in_features) ->
    # may this Linear be computed by the recipe's leaf op on its native
    # parameters (bypassing ``Linear.forward`` / ``quant_method.apply``)?
    supports: Callable[..., bool]
    # Call-time builder: (encoder_hidden_states, hidden_states, txt_qkv_proj,
    # img_qkv_proj) -> packed joint QKV, via the recipe's functional leaf op.
    build: Callable[[torch.Tensor, torch.Tensor, Linear, Linear], torch.Tensor]


# ===========================================================================
# Recipe "bf16": unquantized bf16 — raw addmm with the cublasLt bias epilogue
# ===========================================================================


@torch.library.custom_op("trtllm_vgoa::packed_qkv_proj_bf16", mutates_args=())
def _packed_qkv_proj_bf16(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    txt_weight_packed: torch.Tensor,
    txt_bias_packed: torch.Tensor,
    img_weight_packed: torch.Tensor,
    img_bias_packed: torch.Tensor,
) -> torch.Tensor:
    """Concat-free packed joint QKV: project into ``out=`` row slices.

    Rows [0, S_txt) are the text stream, rows [S_txt, S_txt + S_img) the
    image stream, columns [q | k | v] — bit-compatible with the per-stream
    merged projection + seq-dim ``torch.cat`` layout. Both row-slices of the
    freshly allocated buffer are contiguous 2-D matrices (B == 1, enforced
    by the caller's runtime guard), so eager addmm takes the cublasLt
    bias-epilogue path with zero extra elementwise kernels.
    """
    s_txt = encoder_hidden_states.shape[1]
    s_img = hidden_states.shape[1]
    packed_dim = txt_weight_packed.shape[0]
    qkv = hidden_states.new_empty((1, s_txt + s_img, packed_dim))
    qkv_rows = qkv.view(s_txt + s_img, packed_dim)
    torch.addmm(
        txt_bias_packed,
        encoder_hidden_states[0],
        txt_weight_packed.t(),
        out=qkv_rows[:s_txt],
    )
    torch.addmm(
        img_bias_packed,
        hidden_states[0],
        img_weight_packed.t(),
        out=qkv_rows[s_txt:],
    )
    return qkv


@_packed_qkv_proj_bf16.register_fake
def _packed_qkv_proj_bf16_fake(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    txt_weight_packed: torch.Tensor,
    txt_bias_packed: torch.Tensor,
    img_weight_packed: torch.Tensor,
    img_bias_packed: torch.Tensor,
) -> torch.Tensor:
    """Shape/dtype-only fake impl for tracing (dynamo/fake-tensor)."""
    s_txt = encoder_hidden_states.shape[1]
    s_img = hidden_states.shape[1]
    packed_dim = txt_weight_packed.shape[0]
    return hidden_states.new_empty((1, s_txt + s_img, packed_dim))


def linear_supports_packed_addmm(linear: Linear, *, out_features: int, in_features: int) -> bool:
    """Whether ``linear`` may be computed by raw ``torch.addmm`` on its
    native ``weight``/``bias`` (the ``trtllm_vgoa::packed_qkv_proj_bf16``
    lane).

    The packed op bypasses ``Linear.forward`` / ``quant_method.apply``, so
    this census must reject every Linear feature the bypass would silently
    skip: quantized methods (strict type check — the FP8 linear methods
    subclass ``UnquantizedLinearMethod``), non-bf16 or non-CUDA weights,
    missing bias (the op signature requires one), TP collectives
    (``gather_output``), LoRA, and the non-default GEMM backends. Callers add
    their own model-level gating (TP size, fused-rope preconditions, batch
    shape) on top.
    """
    if type(linear.quant_method) is not UnquantizedLinearMethod:
        return False
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if weight is None or bias is None:
        return False
    if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        return False
    if not weight.is_cuda:
        return False
    if weight.shape[0] != out_features or weight.shape[1] != in_features:
        return False
    if bias.shape[0] != out_features:
        return False
    # The packed addmm bypasses Linear.forward: it must not skip an
    # allgather, a LoRA branch, or a non-default GEMM backend.
    if linear.gather_output or getattr(linear, "lora", None) is not None:
        return False
    if linear.use_custom_cublas_mm or linear.use_cute_dsl_bf16_gemm:
        return False
    return True


def _build_packed_qkv_bf16(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    txt_qkv_proj: Linear,
    img_qkv_proj: Linear,
) -> torch.Tensor:
    return torch.ops.trtllm_vgoa.packed_qkv_proj_bf16(
        encoder_hidden_states,
        hidden_states,
        txt_qkv_proj.weight,
        txt_qkv_proj.bias,
        img_qkv_proj.weight,
        img_qkv_proj.bias,
    )


# ===========================================================================
# Recipe "fp8_block" / "nvfp4" (future): see the extension contract in the
# module docstring — sibling leaf op + strict-typed census + builder reading
# the merged Linear's native quantized weight/scale parameters, one
# ``_PACKED_QKV_RECIPES`` entry below.
# ===========================================================================


# Ordered dispatch table: the first recipe whose census passes every merged
# Linear wins.
_PACKED_QKV_RECIPES: dict[str, PackedQKVRecipe] = {
    "bf16": PackedQKVRecipe(
        supports=linear_supports_packed_addmm,
        build=_build_packed_qkv_bf16,
    ),
}


def select_packed_qkv_recipe(
    linears: Iterable[Linear], *, out_features: int, in_features: int
) -> Optional[str]:
    """Post-load dispatch seam: the first registered recipe that every merged
    Linear supports, or ``None`` (caller keeps the merged-forward + seq-dim
    cat fallback, which is correct for every configuration)."""
    linears: Tuple[Linear, ...] = tuple(linears)
    for name, recipe in _PACKED_QKV_RECIPES.items():
        if all(
            recipe.supports(linear, out_features=out_features, in_features=in_features)
            for linear in linears
        ):
            return name
    return None


def build_packed_qkv(
    recipe: str,
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    txt_qkv_proj: Linear,
    img_qkv_proj: Linear,
) -> torch.Tensor:
    """Call-time dispatch: build the packed joint QKV buffer
    ``[1, S_txt + S_img, q_dim + 2*kv_dim]`` via ``recipe``'s functional
    leaf op, reading the merged Linears' native parameters."""
    return _PACKED_QKV_RECIPES[recipe].build(
        encoder_hidden_states, hidden_states, txt_qkv_proj, img_qkv_proj
    )
