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
"""Oracles for the attention_output_tail.

``native_attn_tail`` runs the exact native TensorRT-LLM operations of the GEN
tail: ``MLA._bmm_bf16_out`` for W_UV (the BF16 branch of
``MLA.forward_absorption_generation``) followed by the ROW-parallel ``o_proj``
GEMM (``UnquantizedLinearMethod.apply``). ``oracle_attn_tail`` is an
independent FP64 reference that keeps the native BF16 rounding of the
intermediate ``v``.
"""

from types import SimpleNamespace
from typing import Optional, Tuple

import torch


def native_attn_tail(
    latent: torch.Tensor,
    w_uv: torch.Tensor,
    w_o: torch.Tensor,
    num_rows: int,
    out_rows: Optional[int] = None,
    o_proj: Optional[torch.nn.Module] = None,
    return_v: bool = False,
):
    """Run the native W_UV BMM and o_proj GEMM.

    Args:
        latent: ``[>= num_rows, H * C]`` or ``[>= num_rows, H, C]`` BF16.
        w_uv: ``[H, V, C]`` BF16 ``v_b_proj``.
        w_o: ``[N, H * V]`` BF16 local o_proj weight (ignored when ``o_proj``
            is given).
        num_rows: live rows written by W_UV.
        out_rows: rows of the o_proj input buffer (native ``create_output``
            rows); rows past ``num_rows`` are zero-filled here so the oracle is
            defined, whereas the native buffer leaves them uninitialized.
        o_proj: optional native ``Linear`` (ROW parallel, no reduction) to run
            instead of ``F.linear`` on ``w_o``.
        return_v: also return the ``[out_rows, H * V]`` intermediate.
    """
    from tensorrt_llm._torch.attention.mla import MLA

    heads, v_dim, c_dim = w_uv.shape
    if out_rows is None:
        out_rows = num_rows
    attn_out = torch.zeros((out_rows, heads * v_dim), dtype=w_uv.dtype, device=latent.device)
    lat = latent[:num_rows].reshape(num_rows, heads, c_dim)
    v_view = attn_out[:num_rows].view(num_rows, heads, v_dim)
    native_cfg = SimpleNamespace(use_cute_dsl_bf16_bmm=False)
    MLA._bmm_bf16_out(
        native_cfg, lat.transpose(0, 1), w_uv, w_uv.transpose(1, 2), v_view.transpose(0, 1)
    )
    if o_proj is not None:
        partial = o_proj(attn_out)
    else:
        partial = torch.nn.functional.linear(attn_out, w_o)
    if return_v:
        return partial, attn_out
    return partial


def oracle_attn_tail(
    latent: torch.Tensor, w_uv: torch.Tensor, w_o: torch.Tensor, num_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP64 reference with the native BF16 rounding of the intermediate.

    Returns:
        ``(partial [num_rows, N] float64, v [num_rows, H * V] bfloat16)``.
    """
    heads, v_dim, c_dim = w_uv.shape
    lat = latent[:num_rows].reshape(num_rows, heads, c_dim).double()
    v = torch.einsum("rhc,hvc->rhv", lat, w_uv.double())
    v_bf16 = v.reshape(num_rows, heads * v_dim).to(torch.bfloat16)
    partial = v_bf16.double() @ w_o.double().t()
    return partial, v_bf16


def error_stats(x: torch.Tensor, ref: torch.Tensor) -> dict:
    """Normalized RMS / max errors of ``x`` against ``ref`` (both cast to FP64)."""
    diff = x.double() - ref.double()
    ref_rms = ref.double().pow(2).mean().sqrt().item()
    return {
        "rel_rms": diff.pow(2).mean().sqrt().item() / max(ref_rms, 1e-300),
        "max_abs": diff.abs().max().item(),
        "ref_rms": ref_rms,
        "ref_max": ref.double().abs().max().item(),
    }
