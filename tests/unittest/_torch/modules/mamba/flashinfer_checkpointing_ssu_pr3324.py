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

"""Benchmark-only wrapper for FlashInfer PR 3324 checkpointing SSU.

The vendored CUDA/C++ sources in ``flashinfer_checkpointing_ssu_pr3324/``
come from https://github.com/flashinfer-ai/flashinfer/pull/3324.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import jinja2
import torch
from flashinfer.compilation_context import CompilationContext
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import JitSpec, gen_jit_spec
from flashinfer.jit.utils import write_if_different

_ROOT = Path(__file__).resolve().parent / "flashinfer_checkpointing_ssu_pr3324"
_CSRC_DIR = _ROOT / "csrc"
_INCLUDE_DIR = _ROOT / "include"
_SUPPORTED_MAJORS = [8, 9, 10, 11, 12]

_DTYPE_MAP = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
    torch.float32: "float",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
}

_FILENAME_SAFE_DTYPE_MAP = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.float8_e4m3fn: "e4m3",
}


def _arch_flags() -> list[str]:
    context = CompilationContext()
    return context.get_nvcc_flags_list(supported_major_versions=_SUPPORTED_MAJORS)


def _uri(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrix_a_dtype: torch.dtype,
    state_index_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    philox_rounds: int,
    enable_pdl: bool,
) -> str:
    dtype = _FILENAME_SAFE_DTYPE_MAP
    uri = (
        "trtllm_pr3324_checkpointing_ssu_v4_"
        f"s_{dtype[state_dtype]}_i_{dtype[input_dtype]}_dt_{dtype[dt_dtype]}_"
        f"w_{dtype[weight_dtype]}_a_{dtype[matrix_a_dtype]}_"
        f"si_{dtype[state_index_dtype]}_d_{dim}_ds_{dstate}_"
        f"np_{npredicted}_mw_{max_window}_hpg_{heads_per_group}"
    )
    if state_scale_dtype is not None:
        uri += f"_sc_{dtype[state_scale_dtype]}"
    if philox_rounds > 0:
        uri += f"_pr_{philox_rounds}"
    if enable_pdl:
        uri += "_pdl"
    return uri


def _gen_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrix_a_dtype: torch.dtype,
    state_index_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    philox_rounds: int,
    enable_pdl: bool,
) -> JitSpec:
    uri = _uri(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrix_a_dtype,
        state_index_dtype,
        state_scale_dtype,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        philox_rounds,
        enable_pdl,
    )
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    with open(_CSRC_DIR / "checkpointing_ssu_customize_config.jinja") as file:
        config_template = jinja2.Template(file.read())

    state_scale_type = _DTYPE_MAP[state_scale_dtype] if state_scale_dtype is not None else "void"
    config = config_template.render(
        state_dtype=_DTYPE_MAP[state_dtype],
        input_dtype=_DTYPE_MAP[input_dtype],
        dt_dtype=_DTYPE_MAP[dt_dtype],
        weight_dtype=_DTYPE_MAP[weight_dtype],
        matrixA_dtype=_DTYPE_MAP[matrix_a_dtype],
        stateIndex_dtype=_DTYPE_MAP[state_index_dtype],
        state_scale_type=state_scale_type,
        dim=dim,
        dstate=dstate,
        npredicted=npredicted,
        max_window=max_window,
        heads_per_group=heads_per_group,
        philox_rounds=philox_rounds,
        enable_pdl="true" if enable_pdl else "false",
    )
    write_if_different(gen_directory / "checkpointing_ssu_config.inc", config)

    source_paths = []
    for filename in (
        "checkpointing_ssu.cu",
        "checkpointing_ssu_kernel_inst.cu",
        "checkpointing_ssu_jit_binding.cu",
    ):
        source_path = _CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(source_path) as file:
            write_if_different(dest_path, file.read())

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cuda_cflags=_arch_flags(),
        extra_include_paths=[_INCLUDE_DIR],
    )


@functools.cache
def _get_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrix_a_dtype: torch.dtype,
    state_index_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    philox_rounds: int,
    enable_pdl: bool,
):
    return _gen_module(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrix_a_dtype,
        state_index_dtype,
        state_scale_dtype,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        philox_rounds,
        enable_pdl,
    ).build_and_load()


def checkpointing_ssu(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    d_split: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    quantized_state_dtypes = (torch.int8, torch.float8_e4m3fn)
    if state.dtype in quantized_state_dtypes:
        if state_scale is None:
            raise ValueError(f"state dtype {state.dtype} requires state_scale")
    elif state_scale is not None:
        raise ValueError(f"state_scale must be None for non-quantized state dtype {state.dtype}")
    if cu_seqlens is not None:
        npredicted = max_seqlen if max_seqlen is not None else old_x.size(1)
    else:
        if max_seqlen is not None:
            raise ValueError("max_seqlen is only valid with cu_seqlens")
        npredicted = x.size(1)

    max_window = old_x.size(1)
    if max_window > 16:
        raise ValueError(f"PR3324 checkpointing SSU supports max_window <= 16, got {max_window}")
    if npredicted > max_window:
        raise ValueError(f"npredicted ({npredicted}) must be <= max_window ({max_window})")

    if d_split is None:
        d_split = 1
    if state.dtype in quantized_state_dtypes and d_split != 1:
        raise ValueError(f"8-bit state requires d_split=1, got {d_split}")

    state_index_dtype = (
        state_batch_indices.dtype if state_batch_indices is not None else torch.int32
    )
    nheads = state.size(1)
    ngroups = B.size(-2)
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    heads_per_group = nheads // ngroups

    if rand_seed is None:
        philox_rounds = 0
    elif philox_rounds <= 0:
        raise ValueError(f"philox_rounds must be > 0 with rand_seed, got {philox_rounds}")

    weight_dtype = (
        D.dtype if D is not None else (dt_bias.dtype if dt_bias is not None else dt.dtype)
    )

    module = _get_module(
        state.dtype,
        x.dtype,
        dt.dtype,
        weight_dtype,
        A.dtype,
        state_index_dtype,
        state_scale.dtype if state_scale is not None else None,
        state.size(2),
        state.size(3),
        npredicted,
        max_window,
        heads_per_group,
        philox_rounds,
        enable_pdl,
    )
    module.checkpointing_ssu(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        old_x,
        old_B,
        old_dt,
        old_cumAdt,
        cache_buf_idx,
        prev_num_accepted_tokens,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        state_scale,
        rand_seed,
        d_split,
        cu_seqlens,
    )
    return out
