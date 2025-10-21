# SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import subprocess
from collections import namedtuple
from enum import IntEnum
from itertools import product

sm2name = {
    70: 'volta',
    72: 'volta',
    75: 'turing',
    80: 'ampere',
    86: 'ampere',
    87: 'ampere',
    89: 'ada',
    90: 'hopper',
    120: 'blackwell',
}

dtype2traits = {
    'int8': 'imma_int8_int32_traits',
    'fp16': 'hmma_fp16_traits',
    'fp16_fp32': 'hmma_fp32_traits',
    'bf16': 'hmma_bf16_traits',
    'e4m3': 'qmma_e4m3_fp32_traits',
    'e4m3_fp32': 'qmma_e4m3_fp32_traits',
    'e4m3_fp16': 'qmma_e4m3_fp16_traits'
}

dtype2OutputType = {
    'int8': 'int8_t',
    'fp16': 'fp16_t',
    'fp16_fp32': 'fp16_t',
    'bf16': 'bf16_t',
    'e4m3': 'e4m3_t',
    'e4m3_fp32': 'e4m3_t',
    'e4m3_fp16': 'e4m3_t',
}

dtype2bytes = {
    'int8': 1,
    'fp16': 2,
    'fp16_fp32': 2,
    'bf16': 2,
    'e4m3': 1,
    'e4m3_fp32': 1,
    'e4m3_fp16': 1
}

# TODO merge with above?
hopper_dtype2traits = {
    'int8': 'igmma_int8_int32_traits',
    'fp16': 'hgmma_fp16_traits',
    'fp16_fp32': 'hgmma_fp32_traits',
    'bf16': 'hgmma_bf16_traits',
    'e4m3': 'qgmma_e4m3_fp32_traits',
    'e4m3_fp32': 'qgmma_e4m3_fp32_traits',
}

# The minimal instruction shapes per warp group.
# TODO should this not be known to the trait itself?
hopper_traits2shape = {
    'Hopper_igmma_int8_int32_traits': (64, 8, 32),
    'Hopper_hgmma_fp16_traits': (64, 8, 16),
    'Hopper_hgmma_fp32_traits': (64, 8, 16),
    'Hopper_hgmma_bf16_traits': (64, 8, 16),
    'Hopper_qgmma_e4m3_fp32_traits': (64, 8, 32),
}

dtype2typename = {
    'int8': 'DATA_TYPE_INT8',
    'fp16': 'DATA_TYPE_FP16',
    'fp16_fp32': 'DATA_TYPE_FP16',
    'bf16': 'DATA_TYPE_BF16',
    'e4m3': 'DATA_TYPE_E4M3',
    'e4m3_fp16': 'DATA_TYPE_E4M3',
    'e4m3_fp32': 'DATA_TYPE_E4M3',
}

pythonBoolean2cpp = {True: 'true', False: 'false'}


# same definition as fused_multihead_attention.h.
class AttentionMaskType(IntEnum):
    PADDING = 0
    CAUSAL = 1
    SLIDING_OR_CHUNKED_CAUSAL = 2
    CUSTOM_MASK = 3


class InputLayout(IntEnum):
    PACKED_QKV = 0
    CONTIGUOUS_Q_KV = 1
    Q_PAGED_KV = 2
    SEPARATE_Q_K_V = 3


spec_fields = (
    'sm',
    'dtype',
    'seq_len',
    'head_size',
    'warps_m',
    'warps_n',
    'version',
    'interleaved',
    'ldgsts_q',
    'ldgsts_k',
    'ldgsts_v',
    'share_smem_k_v',
    'loop_step',
    'has_noloop',
    'noloop_step',
    'unroll_threshold',
    'has_scale_max',
    'ctas_per_head',
    'sm_mma',
    'head_interleaved',
    # new added fields (only used by flash attention implementation)
    'flash_attention',
    'kv_loop_step',
    'flash_attention_bh_upper_threshold',  # to deprecate; not actively used
    'limit_qk_fragments',
    'limit_v_fragments',
    'tiled',
    # fields for warp specialized kernel
    'warp_specialization',
    'q_tile_buffers',
    'kv_tile_buffers',
    'scheduling_mode',
    # attention qkv input layout.
    'input_layout',
    # fused MHCA.
    'cross_mha',
    # other features
    'alibi',
    'enable_attn_logit_softcapping',
    'return_softmax_stats',
    'disabled_mask_types',
    'head_size_v',
    'sage_block_sizes',
    'output_dtype',
    'is_mtp')
kernel_spec = namedtuple('kernel_spec', spec_fields)
kernel_spec.__new__.__defaults__ = (
    1,  # ctas_per_head
    1,  # sm_mma
    True,  # head_interleaved
    False,  # flash_attention
    64,  # kv_loop_step
    -1,  # flash_attention_bh_upper_threshold
    False,  # limit_qk_fragments
    False,  # limit_v_fragments
    0,  # tiled
    False,  # warp_specialization
    1,  # q_tile_buffers
    1,  # kv_tile_buffers
    0,  # scheduling_mode
    InputLayout.PACKED_QKV,
    0,  # cross_mha
    True,  # alibi
    False,  # enable_attn_logit_softcapping
    False,  # return_softmax_stats
    None,  # disabled_mask_types
    0,  # head size of V
    None,  # sage_block_sizes
    None,  # output_dtype, same as dtype by default.
    False)  # use MTP or not

generate_cu_trtllm = os.environ.get('GENERATE_CU_TRTLLM',
                                    'False').lower() == 'true'

ns_open = r"""
namespace tensorrt_llm
{
namespace kernels
{
// clang-format off
""" if generate_cu_trtllm else ""

ns_close = r"""
// clang-format on
} // namespace kernels
} // namespace tensorrt_llm
""" if generate_cu_trtllm else ""

copyright = '''\
/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
''' if not generate_cu_trtllm else r"""/*
* SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
* AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
"""

makefile_template = '''\

# The combination of supported gencodes.
GENCODES  = $(GENCODE_SM70)
GENCODES += $(GENCODE_SM72)
GENCODES += $(GENCODE_SM75)
GENCODES += $(GENCODE_SM80)
GENCODES += $(GENCODE_SM86)
GENCODES += $(GENCODE_SM87)
GENCODES += $(GENCODE_SM89)
GENCODES += $(GENCODE_SM90)
GENCODES += $(GENCODE_SM100)
GENCODES += $(GENCODE_SM120)

OBJECTS_MHA  = obj/fused_multihead_attention.cpp.o
OBJECTS_MHCA = obj/fused_multihead_cross_attention.cpp.o

{objects}

{cubins}

SOFTMAX_SRC   = $(wildcard src/softmax*.cu)
SOFTMAX_OBJ   = $(patsubst src/softmax%.cu, obj/softmax%.cu.o, $(SOFTMAX_SRC))
OBJECTS_MHA  += $(SOFTMAX_OBJ)
OBJECTS_MHA  += obj/convert.cu.o
OBJECTS_MHCA += $(SOFTMAX_OBJ)
OBJECTS_MHCA += obj/convert.cu.o
'''


def get_makefile_code(specs_names):
    objects = '\n'.join([
        'OBJECTS_MHA += obj/{}.o'.format(fname)
        for kspec, fname, lname, kname in specs_names
    ])
    objects = objects + '\n' + '\n'.join([
        'OBJECTS_MHCA += obj/{}.o'.format(fname)
        for kspec, fname, lname, kname in specs_names
    ])

    cubins = '\n'.join([
        'CUBINS += cubin/{}.cubin'.format(fname)
        for kspec, fname, lname, kname in specs_names
    ])
    return makefile_template.format(objects=objects,
                                    cubins=cubins,
                                    copyright=copyright)


MAX_STGS_PER_LOOP = 4

kernel_template = '''\
{copyright}

//We can disable the FADD trick for archs with F2IP
#if {disable_fadd_trick} // disable_fadd_trick
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif // USE_I2F_EMULATION_TRICK

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif // USE_F2I_EMULATION_TRICK
#endif // disable_fadd_trick

#include <cuda.h>
#include <stdexcept>

#if CUDA_VERSION >= {min_cuda_version}


#if !{use_multi_cta} // !use_multi_cta
#include <fused_multihead_attention_kernel_{kernel_variant}.h>
#endif // !use_multi_cta

#if !{use_multi_cta} && {has_noloop} // !use_multi_cta && has_noloop
#include <fused_multihead_attention_kernel_1xN_noloop.h>
#endif // !use_multi_cta && has_noloop

#if {cross_mha} // cross_mha
#if {has_noloop} // has_noloop
#include <fused_multihead_cross_attention_kernel_1xN_noloop.h>
#endif  // has_noloop
#include <fused_multihead_cross_attention_kernel_1xN.h>
#endif // cross_mha

#if  {use_multi_cta} // use_multi_cta
#include <fused_multihead_attention_kernel_1xN_multi_cta.h>
#endif

using Attention_mask_type = fmha::Attention_mask_type;
using Launch_params = bert::Fused_multihead_attention_launch_params;

#if !{cross_mha} // !cross_mha
using Kernel_traits = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {seq_len},
    {head_size},
    {loop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags}>;

using Kernel_traits_causal = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {seq_len},
    {head_size},
    {loop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags},
    /*causal mask*/ 3>;
#endif // Not cross attention

#if !{use_multi_cta} && !{cross_mha} // !use_multi_cta && !cross_mha

extern "C"
__global__
void {kernel_name}({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}<Kernel_traits>(params);
}}

extern "C"
__global__
void {causal_kernel_name}({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}<Kernel_traits_causal>(params);
}}

void {launcher_name}(
    const {params_type} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{

  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name},
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem_size));
    }}
    dim3 grid(params.h, params.b);
    {causal_kernel_name}<<<grid, Kernel_traits_causal::THREADS, Kernel_traits_causal::BYTES_PER_SMEM, stream>>>(params);
  }} else {{
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name},
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem_size));
    }}
    dim3 grid(params.h, params.b);
    {kernel_name}<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(params);
  }}
}}

#endif // !use_multi_cta && !cross_mha

#if !{use_multi_cta} && {has_noloop} && !{cross_mha} // !use_multi_cta && has_noloop && !cross_mha

using Kernel_traits_nl = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {seq_len},
    {head_size},
    {noloop_step},
    1,
    {warps_m} * {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */ >;

static_assert(Kernel_traits_nl::CTAS_PER_HEAD == 1, "");

using Kernel_traits_nl_causal = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {seq_len},
    {head_size},
    {noloop_step},
    1,
    {warps_m} * {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*causal mask*/ 3>;

static_assert(Kernel_traits_nl_causal::CTAS_PER_HEAD == 1, "");
static_assert(Kernel_traits_nl_causal::MASK_VERSION == 3, "");

extern "C"
__global__
void {kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_1xN_nl<Kernel_traits_nl>(params);
}}

extern "C"
__global__
void {causal_kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_1xN_nl<Kernel_traits_nl_causal>(params);
}}

void {launcher_name}_nl(
    const {params_type} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{


  constexpr int loop_iters = ({seq_len} + {noloop_step}-1) / {noloop_step};
  static_assert(loop_iters * {noloop_step} >= {seq_len}, "");
  dim3 grid(params.h, params.b, loop_iters);
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
    constexpr int smem_size = Kernel_traits_nl_causal::BYTES_PER_SMEM;
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name}_nl,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem_size));
    }}
    {causal_kernel_name}_nl<<<grid, Kernel_traits_nl_causal::THREADS, Kernel_traits_nl_causal::BYTES_PER_SMEM, stream>>>(params);
  }} else {{
    constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name}_nl,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem_size));
    }}
    {kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
  }}
}}

#endif // !use_multi_cta && has_noloop && !cross_mha

#if {cross_mha} // cross_mha
#if !{use_multi_cta} && {has_noloop} // !use_multi_cta && has_noloop

using Kernel_traits_nl = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {seq_len},
    {head_size},
    {noloop_step},
    1,
    {warps_m} * {warps_n},
    {ctas_per_head},
    {kernel_flags}>;

static_assert(Kernel_traits_nl::CTAS_PER_HEAD == 1, "");

extern "C"
__global__
void {kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_mhca_1xN_nl<Kernel_traits_nl>(params);
}}

void {launcher_name}_nl(
    const {params_type} &params,
    // const Launch_params &launch_params, // TODO
    cudaStream_t stream){{

  constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
  if( smem_size >= 48*1024 ) {{
    FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name}_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
  }}
  const int loop_iters = (params.s_q + {noloop_step}-1) / {noloop_step};
  // if (loop_iters * {noloop_step} != params.s_q) {{
  //   throw std::runtime_error("Incorrect seq len -- loop_iters * noloop_step != params.s_q");
  // }}
  assert(loop_iters * {noloop_step} >= params.s_q);
  dim3 grid(params.h, params.b, loop_iters);
  {kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>(params);
}}

#endif // !use_multi_cta && has_noloop

#if !{use_multi_cta} // !use_multi_cta

using Kernel_traits = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {seq_len},
    {head_size},
    {loop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags}>;

extern "C"
__global__
void {kernel_name}({params_type} params){{
  fused_multihead_attention::device_mhca_1xN<Kernel_traits>(params);
}}

void {launcher_name}(
    const {params_type} &params,
    // const Launch_params &launch_params, // TODO
    cudaStream_t stream){{

  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( smem_size >= 48*1024 ) {{
    FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
  }}
  dim3 grid(params.h, params.b);
  {kernel_name}<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(params);
}}

#endif // !use_multi_cta

#endif // cross_mha

#if {use_multi_cta} // use_multi_cta

// If that assert gets triggered - increase the value of MAX_STGS_PER_LOOP in "setup.py".
static_assert(Kernel_traits::Gmem_tile_o::STGS_PER_LOOP <= {MAX_STGS_PER_LOOP}, "");

extern "C"
__global__
void {kernel_name}({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_multi_cta<Kernel_traits>(params);
}}

extern "C"
__global__
void {causal_kernel_name}({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_multi_cta<Kernel_traits_causal>(params);
}}

void {launcher_name}(
    const {params_type} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{

  assert(params.heads_per_wave != 0 && \"Heads per wave is not set, but multi cta is requested\");

  // Clear the barriers and locks.
  cudaMemsetAsync(params.counters, 0, 3*params.heads_per_wave*sizeof(int), stream);

  // We may use more than 48kB of shared memory.
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
    constexpr int smem_size = Kernel_traits_causal::BYTES_PER_SMEM;
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name},
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size));
    }}
    // Launch one wave.
    dim3 grid(Kernel_traits_causal::CTAS_PER_HEAD, params.heads_per_wave), block(Kernel_traits_causal::THREADS);
    void *params_ = (void*) &params;
    FMHA_CHECK_CUDA(cudaLaunchCooperativeKernel((void*) &{causal_kernel_name}, grid, block, (void**) &params_, smem_size, stream));
  }} else {{
    constexpr size_t smem_size = Kernel_traits::BYTES_PER_SMEM;
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name},
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_size));
    }}
    // Launch one wave.
    dim3 grid(Kernel_traits::CTAS_PER_HEAD, params.heads_per_wave), block(Kernel_traits::THREADS);
    void *params_ = (void*) &params;
    FMHA_CHECK_CUDA(cudaLaunchCooperativeKernel((void*) &{kernel_name}, grid, block, (void**) &params_, smem_size, stream));
  }}
}}

#endif // use_multi_cta

void {launcher_name}_get_max_heads_per_wave(int *heads_per_wave) {{
#if {use_multi_cta} // use_multi_cta
    // Determine the number of SMs and CTAs.
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp props;
    FMHA_CHECK_CUDA(cudaGetDeviceProperties(&props, dev));

    // The number of CTAs per SM.
    constexpr size_t smem_size = Kernel_traits::BYTES_PER_SMEM;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm,
                                                            &{kernel_name},
                                                            Kernel_traits::THREADS,
                                                            smem_size));

    // The number of heads per wave.
    *heads_per_wave = props.multiProcessorCount * ctas_per_sm / Kernel_traits::CTAS_PER_HEAD;
#else // use_multi_cta
    *heads_per_wave = 0;
#endif // use_multi_cta
}}

#else // CUDA_VERSION >= {min_cuda_version}

void {launcher_name}(
    const {params_type} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

#if {has_noloop} // has_noloop

void {launcher_name}_nl(
    const {params_type} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

#endif // has_noloop

#endif // CUDA_VERSION >= {min_cuda_version}
'''

flash_attention_kernel_template = '''\
{copyright}

//We can disable the FADD trick for archs with F2IP
#if {disable_fadd_trick} // disable_fadd_trick
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif // USE_I2F_EMULATION_TRICK

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif // USE_F2I_EMULATION_TRICK
#endif // disable_fadd_trick

#include <cuda.h>

#if CUDA_VERSION >= {min_cuda_version}

#include <fused_multihead_flash_attention_kernel_noloop.h>
#include <fused_multihead_flash_attention_kernel_noloop_tiled.h>
#include <fused_multihead_flash_attention_kernel.h>

{include_str}

{local_ns_open}
{bert_launch_params}
{attn_mask_type_str}

#if 0 // has_noloop (unconditionally disabled since not maintained & not actively used)
using Kernel_traits = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {loop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags}>;

extern "C"
__global__
void {kernel_name}({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}<Kernel_traits>(params);
}}

void {launcher_name}(
    const {params_type} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{

  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( smem_size >= 48*1024 ) {{
    FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
  }}
  dim3 grid(params.h, params.b);
  {kernel_name}<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>(params);
}}

#endif // has_noloop

#if {has_noloop} && !{tiled} // has_noloop && !tiled
using Kernel_traits_nl = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*dense mask*/ 2,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_},
    {sage_block_size_q},
    {sage_block_size_k},
    {sage_block_size_v}>;

using Kernel_traits_nl_causal = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*causal mask*/ 3,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_}>;

using Kernel_traits_nl_sliding_or_chunked_causal = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*sliding window causal mask*/ 4,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_}>;

using Kernel_traits_nl_custom_mask = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*custom mask*/ 5,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_}>;

#if {padding_mask} // padding_mask

extern "C"
__global__
void {kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_nl>(params);
}}

#endif // padding mask

#if {causal_mask} // causal_mask

extern "C"
__global__
void {causal_kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_nl_causal>(params);
}}

#endif // causal mask

#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask

extern "C"
__global__
void {sliding_or_chunked_causal_kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_nl_sliding_or_chunked_causal>(params);
}}

#endif // sliding_or_chunked_causal_mask

#if {custom_mask} // custom_mask

extern "C"
__global__
void {custom_mask_kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_nl_custom_mask>(params);
}}

#endif // custom_mask

void {launcher_name}_nl(
    {const_fused_multihead_attention_params_v2_str} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{

  // runtime q_loop_iters
  int loop_iters = ( params.s + {noloop_step} - 1 )  / {noloop_step};
  // dim3 grid(params.h, params.b, loop_iters);
  dim3 grid(loop_iters, params.h, params.b); // better locality
  constexpr int smem_size = Kernel_traits_nl::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
#if {causal_mask} // causal_mask
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name}_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }}
    {causal_kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // causal mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {{
#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {{
       FMHA_CHECK_CUDA(cudaFuncSetAttribute({sliding_or_chunked_causal_kernel_name}_nl,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }}
    {sliding_or_chunked_causal_kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // sliding_or_chunked_causal_mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {{
#if {padding_mask} // padding_mask
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name}_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }}
    {kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // padding_mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {{
#if {custom_mask} // custom_mask
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({custom_mask_kernel_name}_nl,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }}
    {custom_mask_kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // custom mask
  }}
}}

#endif // has_noloop && !tiled

#if {tiled} // tiled

using Kernel_traits_nl_tiled = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*dense mask*/ 2,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_},
    {sage_block_size_q},
    {sage_block_size_k},
    {sage_block_size_v}>;

using Kernel_traits_nl_tiled_causal = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*causal mask*/ 3,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_}>;

using Kernel_traits_nl_tiled_sliding_or_chunked_causal = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*sliding window causal mask*/ 4,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_}>;

using Kernel_traits_nl_tiled_custom_mask = fmha::{kernel_traits}<
    fmha::{instruction_traits},
    {kv_loop_step},
    {head_size},
    {head_size_v},
    {noloop_step},
    {warps_m},
    {warps_n},
    {ctas_per_head},
    {kernel_flags} | 0x200 /* no_loop flag */,
    /*custom mask*/ 5,
    /*bmm2_fp16_epilogue*/ true,
    {output_dtype_}>;

#if {padding_mask} // padding_mask

extern "C"
__global__
void {kernel_name}_nl_tiled({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl_tiled<Kernel_traits_nl_tiled>(params);
}}

#endif // padding_mask

#if {causal_mask} // causal_mask

extern "C"
__global__
void {causal_kernel_name}_nl_tiled({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl_tiled<Kernel_traits_nl_tiled_causal>(params);
}}

#endif // causal mask

#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask

extern "C"
__global__
void {sliding_or_chunked_causal_kernel_name}_nl_tiled({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl_tiled<Kernel_traits_nl_tiled_sliding_or_chunked_causal>(params);
}}

#endif // sliding_or_chunked_causal_mask

#if {custom_mask} // custom_mask

extern "C"
__global__
void {custom_mask_kernel_name}_nl_tiled({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl_tiled<Kernel_traits_nl_tiled_custom_mask>(params);
}}

#endif // custom mask

// Granular tiling
void {launcher_name}_nl_tiled(
    {const_fused_multihead_attention_params_v2_str} &params,
    const Launch_params &launch_params,
    cudaStream_t stream){{
  // runtime q_loop_iters
  using Cta_tile_o = typename Kernel_traits_nl_tiled::Cta_tile_o;
  int ctas_per_o_row = (params.d + Cta_tile_o::N - 1) / Cta_tile_o::N;
  int loop_iters = ( params.s + {noloop_step} - 1 )  / {noloop_step};
  dim3 grid(loop_iters * ctas_per_o_row, params.h, params.b);
  constexpr int smem_size = Kernel_traits_nl_tiled::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
#if {causal_mask} // causal_mask
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name}_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }}
    {causal_kernel_name}_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>({params_str});
#endif // causal mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {{
#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {{
       FMHA_CHECK_CUDA(cudaFuncSetAttribute({sliding_or_chunked_causal_kernel_name}_nl_tiled,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }}
    {sliding_or_chunked_causal_kernel_name}_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>({params_str});
#endif // sliding_or_chunked_causal_mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {{
#if {padding_mask} // padding_mask
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name}_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }}
    {kernel_name}_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>({params_str});
#endif // padding_mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {{
#if {custom_mask} // custom_mask
    if( smem_size >= 48*1024 ) {{
      FMHA_CHECK_CUDA(cudaFuncSetAttribute({custom_mask_kernel_name}_nl_tiled,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           smem_size));
    }}
    {custom_mask_kernel_name}_nl_tiled<<<grid, Kernel_traits_nl_tiled::THREADS, Kernel_traits_nl_tiled::BYTES_PER_SMEM, stream>>>({params_str});
#endif // custom mask
  }}
}}

#endif // tiled

#else // CUDA_VERSION >= {min_cuda_version}

void {launcher_name}(const {params_type} &params, cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

void {launcher_name}_nl(const {params_type} &params, cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

void {launcher_name}_nl_tiled(const {params_type} &params, cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

#endif // CUDA_VERSION >= {min_cuda_version}
{local_ns_close}
'''

kernel_hopper_template = '''\
{copyright}

//We can disable the FADD trick for archs with F2IP
#if {disable_fadd_trick}
#ifdef USE_I2F_EMULATION_TRICK
#undef USE_I2F_EMULATION_TRICK
#endif

#ifdef USE_F2I_EMULATION_TRICK
#undef USE_F2I_EMULATION_TRICK
#endif
#endif

#include <cuda.h>

#if CUDA_VERSION >= {min_cuda_version}

#include <fused_multihead_attention_kernel_{kernel_variant}.h>
#if {has_noloop}
#include <fused_multihead_attention_kernel_{kernel_variant}_noloop.h>
#endif

#if {use_tma}
// only included if tma is used.
#include <fmha/hopper/tma_descriptor.h>
#endif //use_tma

{include_str}
{local_ns_open}
{bert_launch_params}
{attn_mask_type_str}

using Traits_p = fmha::{instruction_traits_p};
using Traits_o = fmha::{instruction_traits_o};

using Kernel_traits = {kernel_traits}<
                       Traits_p,
                       Traits_o,
                       {seq_len},
                       {head_size},
                       {loop_step},
                       {warps_m},
                       {warps_n},
                       2,
                       {kernel_flags}>;

using Kernel_traits_causal = {kernel_traits}<
                              Traits_p,
                              Traits_o,
                              {seq_len},
                              {head_size},
                              {loop_step},
                              {warps_m},
                              {warps_n},
                              3,
                              {kernel_flags}>;

using Kernel_traits_sliding_or_chunked_causal = {kernel_traits}<
                                           Traits_p,
                                           Traits_o,
                                           {seq_len},
                                           {head_size},
                                           {loop_step},
                                           {warps_m},
                                           {warps_n},
                                           4,
                                           {kernel_flags}>;

#if {use_tma} // use_tma

#if {padding_mask} // padding_mask

extern "C"
__global__
void {kernel_name}(const __grid_constant__ {params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_tma<Kernel_traits>(params);
}}

#endif // padding_mask

#if {causal_mask} // causal_mask

extern "C"
__global__
void {causal_kernel_name}(const __grid_constant__ {params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_tma<Kernel_traits_causal>(params);
}}

#endif // causal mask

#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask

extern "C"
__global__
void {sliding_or_chunked_causal_kernel_name}(const __grid_constant__ {params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_tma<Kernel_traits_sliding_or_chunked_causal>(params);
}}

#endif // sliding_or_chunked_causal_mask

#else

#if {padding_mask}

extern "C"
__global__
void {kernel_name}(const __grid_constant__ {params_type} params){{
  fused_multihead_attention::device_{kernel_variant}<Kernel_traits>(params);
}}

#endif // padding_mask

#if {causal_mask} // causal_mask

extern "C"
__global__
void {causal_kernel_name}(const __grid_constant__ {params_type} params){{
  fused_multihead_attention::device_{kernel_variant}<Kernel_traits_causal>(params);
}}

#endif // causal mask

#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask

extern "C"
__global__
void {sliding_or_chunked_causal_kernel_name}(const __grid_constant__ {params_type} params){{
  fused_multihead_attention::device_{kernel_variant}<Kernel_traits_sliding_or_chunked_causal>(params);
}}
#endif

#endif // sliding_or_chunked_causal_mask

void {launcher_name}({fused_multihead_attention_params_v2_str} &params,
    const Launch_params &launch_params, cudaStream_t stream){{
  // setting TMA descriptors if needed.
  // use_tma = {use_tma}
#if {use_tma}
    // declare TMA desc for Q, K, V
    typename fmha::Multiple_tma_descriptor<3> tma_desc_QKV;

    // GMEM pointers, the offset between each batch is d*3*h*seqlen
    // qkv pointer
    char *qkv_ptr = reinterpret_cast<char*>(params.qkv_ptr);

    // tensor size
    uint32_t tensor_size_qkv[3];
    tensor_size_qkv[2] = 1;
    tensor_size_qkv[1] = params.is_s_padded ? params.s * params.b : launch_params.seqlens[params.b];
    tensor_size_qkv[0] = (params.h + 2 * params.h_kv) * params.d;

    // box size for Q
    uint32_t box_size_q[3];
    box_size_q[2] = 1;
    box_size_q[1] = {loop_step}; // STEP size
    box_size_q[0] = {head_size}; // head_size

    // box size for k and v
    uint32_t box_size_kv[3];
    box_size_kv[2] = 1;
    box_size_kv[1] = params.s; // S, should not be actual_s, OOB will be filled with zeros.
    box_size_kv[0] = {head_size}; // head_size

    // stride size
    uint64_t tensor_stride_qkv[2];
    tensor_stride_qkv[0] = tensor_size_qkv[0] * Traits_p::BITS_PER_ELEMENT_A / 8;
    tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];

    // traversal stride
    uint32_t traversal_stride_qkv[3] = {{1, 1, 1}};

    // OOB fill zeros
    uint32_t oob_fill = 0;

    // FP32 to TF32 conversion disabled
    uint32_t fp32_to_tf32 = 0;

    //setup the descriptors

    //setup the descriptor for Q
    tma_desc_QKV.set_tma_desctriptor(reinterpret_cast<void*>(qkv_ptr),
                                fmha::cudaTmaDescFormat::F16_RN, // tma format (data type). For now hardcode to fp16
                                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                                fmha::cudaTmaDescSwizzle::SWIZZLE_128B,
                                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                                tensor_size_qkv,
                                tensor_stride_qkv,
                                traversal_stride_qkv,
                                box_size_q,
                                oob_fill,
                                fp32_to_tf32,
                                &params.tma_desc_q);

    // setup the descriptor for K
    tma_desc_QKV.set_tma_desctriptor(reinterpret_cast<void*>(qkv_ptr),
                                fmha::cudaTmaDescFormat::F16_RN, // tma format (data type). For now hardcode to fp16
                                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                                fmha::cudaTmaDescSwizzle::SWIZZLE_128B,
                                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                                tensor_size_qkv,
                                tensor_stride_qkv,
                                traversal_stride_qkv,
                                box_size_kv,
                                oob_fill,
                                fp32_to_tf32,
                                &params.tma_desc_k);

    // setup the descriptor for V
    tma_desc_QKV.set_tma_desctriptor(reinterpret_cast<void*>(qkv_ptr),
                                fmha::cudaTmaDescFormat::F16_RN, // tma format (data type). For now hardcode to fp16
                                fmha::cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                                fmha::cudaTmaDescSwizzle::SWIZZLE_128B,
                                fmha::cudaTmaDescPromotion::PROMOTION_DISABLED,
                                tensor_size_qkv,
                                tensor_stride_qkv,
                                traversal_stride_qkv,
                                box_size_kv,
                                oob_fill,
                                fp32_to_tf32,
                                &params.tma_desc_v);


#endif // use_tma
  dim3 grid(params.h, params.b);
  // Use the same smem_size for all traits.
  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
#if {causal_mask} // causal_mask
    if( smem_size >= 48*1024 ) {{
       FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name},
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }}
    {causal_kernel_name}<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>({params_str});
#endif // causal mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {{
#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {{
       FMHA_CHECK_CUDA(cudaFuncSetAttribute({sliding_or_chunked_causal_kernel_name},
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }}
    {sliding_or_chunked_causal_kernel_name}<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>({params_str});
#endif // sliding_or_chunked_causal_mask
  }} else {{
#if {padding_mask} // padding_mask
    constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }}
    {kernel_name}<<<grid, Kernel_traits::THREADS, Kernel_traits::BYTES_PER_SMEM, stream>>>({params_str});
#endif // padding_mask
  }}
}}

#if {has_noloop}


using Kernel_traits_nl = {kernel_traits}<
                          Traits_p,
                          Traits_o,
                          {seq_len},
                          {head_size},
                          {noloop_step},
                          {warps_m},
                          {warps_n},
                          2,
                          {kernel_flags}>;

using Kernel_traits_causal_nl = {kernel_traits}<
                                 Traits_p,
                                 Traits_o,
                                 {seq_len},
                                 {head_size},
                                 {noloop_step},
                                 {warps_m},
                                 {warps_n},
                                 3,
                                 {kernel_flags}>;

using Kernel_traits_sliding_or_chunked_causal_nl = {kernel_traits}<
                                              Traits_p,
                                              Traits_o,
                                              {seq_len},
                                              {head_size},
                                              {noloop_step},
                                              {warps_m},
                                              {warps_n},
                                              4,
                                              {kernel_flags}>;

#if {padding_mask} // padding_mask

extern "C"
__global__
void {kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_nl>(params);
}}

#endif // padding_mask

#if {causal_mask} // causal_mask

extern "C"
__global__
void {causal_kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_causal_nl>(params);
}}

#endif // causal mask

#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask

extern "C"
__global__
void {sliding_or_chunked_causal_kernel_name}_nl({params_type} params){{
  fused_multihead_attention::device_{kernel_variant}_nl<Kernel_traits_sliding_or_chunked_causal_nl>(params);
}}

#endif // sliding_or_chunked_causal_mask

void {launcher_name}_nl({fused_multihead_attention_params_v2_str} &params,
    const Launch_params& launch_params, cudaStream_t stream){{
  constexpr int loop_iters = {seq_len} / {noloop_step};
  static_assert(loop_iters * {noloop_step} == {seq_len}, "");
  dim3 grid(params.h, params.b, loop_iters);

  // Use the same smem_size for all traits.
  constexpr int smem_size = Kernel_traits::BYTES_PER_SMEM;
  if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
#if {causal_mask} // causal_mask
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name}_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }}
    {causal_kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // causal mask
  }} else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {{
#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({sliding_or_chunked_causal_kernel_name}_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }}
    {sliding_or_chunked_causal_kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // sliding_or_chunked_causal_mask
  }} else {{
#if {padding_mask} // padding_mask
    if( smem_size >= 48*1024 ) {{
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name}_nl,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem_size));
    }}
    {kernel_name}_nl<<<grid, Kernel_traits_nl::THREADS, Kernel_traits_nl::BYTES_PER_SMEM, stream>>>({params_str});
#endif // padding_mask
  }}
}}

#endif

#else

void {launcher_name}(const {params_type} &params, cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

#if {has_noloop}

void {launcher_name}_nl(const {params_type} &params, cudaStream_t stream){{
    assert(false && "Unsupported CUDA version");
}}

#endif

#endif
{local_ns_close}
'''

kernel_hopper_warp_specialization_template = '''\
{copyright}

#include <fused_multihead_attention_utils.h>
#include <fmha/hopper/gmma_descriptor.h>
#include <fmha/hopper/smem_tile.h>
#include <fmha/utils.h>
#include <fmha/hopper/compute_tile.h>

#include <fmha/warpspec/kernel_traits.h>
#include <fmha/warpspec/dma.h>
#include <fmha/warpspec/compute.h>

{include_str}
////////////////////////////////////////////////////////////////////////////////////////////////////
{local_ns_open}
#if CUDA_VERSION >= {min_cuda_version}

static constexpr int DMA2COMPUTE_DEPTH = 1;
{num_compute_groups_str}
static constexpr bool USE_TMA_STORE = {use_tma_store_flag};

{bert_launch_params}
{attn_mask_type_str}

using Ktraits = {kernel_traits_header}
                {loop_step},
                {kv_loop_step},
                {head_size},
                {head_size_v},
                {q_tile_buffers},
                {kv_tile_buffers},
                NUM_COMPUTE_GROUPS,
                DMA2COMPUTE_DEPTH,
                0,
                {heads_interleaved_flag},
                false,
                {enable_mutex_flag},
                {scheduling_mode},
                {input_layout_flag},
                USE_TMA_STORE,
                {enable_attn_logit_softcapping_flag},
                {return_softmax_stats_flag},
                {output_dtype_},
                {sage_block_size_q},
                {sage_block_size_k},
                {sage_block_size_v}>;

using Ktraits_causal = {kernel_traits_header}
                       {loop_step},
                       {kv_loop_step},
                       {head_size},
                       {head_size_v},
                       {q_tile_buffers},
                       {kv_tile_buffers},
                       NUM_COMPUTE_GROUPS,
                       DMA2COMPUTE_DEPTH,
                       1,
                       {heads_interleaved_flag},
                       {has_alibi},
                       {enable_mutex_flag},
                       {scheduling_mode},
                       {input_layout_flag},
                       USE_TMA_STORE,
                       {enable_attn_logit_softcapping_flag},
                       {return_softmax_stats_flag},
                       {output_dtype_}>;

using Ktraits_sliding_or_chunked_causal = {kernel_traits_header}
                                      {loop_step},
                                      {kv_loop_step},
                                      {head_size},
                                      {head_size_v},
                                      {q_tile_buffers},
                                      {kv_tile_buffers},
                                      NUM_COMPUTE_GROUPS,
                                      DMA2COMPUTE_DEPTH,
                                      2,
                                      {heads_interleaved_flag},
                                      {has_alibi},
                                      {enable_mutex_flag},
                                      {scheduling_mode},
                                      {input_layout_flag},
                                      USE_TMA_STORE && false,
                                      {enable_attn_logit_softcapping_flag},
                                      {return_softmax_stats_flag},
                                      {output_dtype_}>;

using Ktraits_custom_mask = {kernel_traits_header}
                            {loop_step},
                            {kv_loop_step},
                            {head_size},
                            {head_size_v},
                            {q_tile_buffers},
                            {kv_tile_buffers},
                            NUM_COMPUTE_GROUPS,
                            DMA2COMPUTE_DEPTH,
                            3,
                            {heads_interleaved_flag},
                            {has_alibi},
                            {enable_mutex_flag},
                            {scheduling_mode},
                            {input_layout_flag},
                            USE_TMA_STORE && false,
                            {enable_attn_logit_softcapping_flag},
                            {return_softmax_stats_flag},
                            {output_dtype_}>;

////////////////////////////////////////////////////////////////////////////////////////////////////

#if {padding_mask} // padding_mask

using Shared = typename Ktraits::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits::THREADS, 1)
void {kernel_name}(
    const __grid_constant__ {params_type} params){{

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared *shared = reinterpret_cast<Shared *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {{  // dma + sched

        {setmaxnreg_dma_str}
        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits::DMA_GROUP_TRANSPOSE_V ) {{
            fmha::ws::DMA<Ktraits>::Device dma_device(elect_one);
            dma_device.{run_fct_name}(params, shared);
        }} else {{
            fmha::ws::DMA<Ktraits>::Device dma_device(elect_one);
            if( tidx < 32 ) {{
                dma_device.{run_fct_name}(params, shared);
            }}
        }}

    }} else {{  // math

        {setmaxnreg_compute_str}

        fmha::ws::Compute<fmha::{instruction_traits}, Ktraits> compute;
        compute.run(warp_group, tidx, shared, params);
    }}
}}

#endif // padding mask

////////////////////////////////////////////////////////////////////////////////////////////////////

#if {causal_mask} // causal_mask

using Shared_causal = typename Ktraits_causal::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_causal::THREADS, 1)
void {causal_kernel_name}(
    const __grid_constant__ {params_type} params){{

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_causal *shared = reinterpret_cast<Shared_causal *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {{  // dma + sched

        {setmaxnreg_dma_str}
        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits_causal::DMA_GROUP_TRANSPOSE_V ) {{
            fmha::ws::DMA<Ktraits_causal>::Device dma_device(elect_one);
            dma_device.{run_fct_name}(params, shared);
        }} else {{
            fmha::ws::DMA<Ktraits_causal>::Device dma_device(elect_one);
            if( tidx < 32 ) {{
                dma_device.{run_fct_name}(params, shared);
            }}
        }}

    }} else {{  // math

        {setmaxnreg_compute_str}

        fmha::ws::Compute<fmha::{instruction_traits}, Ktraits_causal> compute;
        compute.run(warp_group, tidx, shared, params);
    }}
}}

#endif // causal mask

////////////////////////////////////////////////////////////////////////////////////////////////////

#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask

using Shared_sliding_or_chunked_causal = typename Ktraits_sliding_or_chunked_causal::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_sliding_or_chunked_causal::THREADS, 1)
void {sliding_or_chunked_causal_kernel_name}(
    const __grid_constant__ {params_type} params){{

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_sliding_or_chunked_causal *shared =
        reinterpret_cast<Shared_sliding_or_chunked_causal *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {{  // dma + sched

        {setmaxnreg_dma_str}
        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits_sliding_or_chunked_causal::DMA_GROUP_TRANSPOSE_V ) {{
            fmha::ws::DMA<Ktraits_sliding_or_chunked_causal>::Device dma_device(elect_one);
            dma_device.{run_fct_name}(params, shared);
        }} else {{
            fmha::ws::DMA<Ktraits_sliding_or_chunked_causal>::Device dma_device(elect_one);
            if( tidx < 32 ) {{
                dma_device.{run_fct_name}(params, shared);
            }}
        }}

    }} else {{  // math

        {setmaxnreg_compute_str}

        fmha::ws::Compute<fmha::{instruction_traits}, Ktraits_sliding_or_chunked_causal> compute;
        compute.run(warp_group, tidx, shared, params);
    }}
}}

#endif // sliding_or_chunked_causal_mask

////////////////////////////////////////////////////////////////////////////////////////////////////

#if {custom_mask} // custom_mask

using Shared_custom_mask = typename Ktraits_custom_mask::Shared;

extern "C"
__global__ __launch_bounds__(Ktraits_custom_mask::THREADS, 1)
void {custom_mask_kernel_name}(
    const __grid_constant__ {params_type} params){{

    extern __shared__ char smem_[];
    char *smem_aligned = fmha::align_1024(smem_);

    Shared_custom_mask *shared =
        reinterpret_cast<Shared_custom_mask *>(&smem_aligned[0]);
    shared->init(threadIdx.x == 0);
    __syncthreads();

    // special trick to avoid wrap_sync (leads to illegal instruction)
    int warp_group = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    int tidx = threadIdx.x % 128;

    if( warp_group == NUM_COMPUTE_GROUPS ) {{  // dma + sched

        {setmaxnreg_dma_str}
        uint32_t elect_one = tidx == 0;

        // Need all threads involved when the dam group needs to transpose the v tile explicltly.
        if constexpr ( Ktraits_custom_mask::DMA_GROUP_TRANSPOSE_V ) {{
            fmha::ws::DMA<Ktraits_custom_mask>::Device dma_device(elect_one);
            dma_device.{run_fct_name}(params, shared);
        }} else {{
            fmha::ws::DMA<Ktraits_custom_mask>::Device dma_device(elect_one);
            if( tidx < 32 ) {{
                dma_device.{run_fct_name}(params, shared);
            }}
        }}

    }} else {{  // math

        {setmaxnreg_compute_str}

        fmha::ws::Compute<fmha::{instruction_traits}, Ktraits_custom_mask> compute;
        compute.run(warp_group, tidx, shared, params);
    }}
}}

#endif // custom_mask

////////////////////////////////////////////////////////////////////////////////////////////////////

void {launcher_name}(
    {fused_multihead_attention_params_v2_str} &params,
    const Launch_params &launch_params, cudaStream_t stream){{

    {TMA_config}
    if( Ktraits::SCHEDULING_MODE > 0 ) {{
        FMHA_CHECK_CUDA(cudaMemsetAsync(params.tile_id_counter_ptr, 0, sizeof(uint32_t), stream));
    }}

    dim3 block_size;

    if( Ktraits::SCHEDULING_MODE == 0 ) {{
        block_size.y = std::min(params.b * params.h, launch_params.multi_processor_count);
        // distribute m steps to multiple blocks (fully utilize SMs)
        // block.x = blocks that handle single head, block.y = blocks that handle different heads
        size_t sms_per_head = (launch_params.multi_processor_count) / block_size.y;
        // Take multiple compute groups into consideration.
        size_t m_steps = size_t((params.s + {loop_step} * NUM_COMPUTE_GROUPS - 1) / ({loop_step} * NUM_COMPUTE_GROUPS));

        // 2 * {bytes_per_elt} stands for kv cache and {bytes_per_elt} bytes per element.
        size_t size_in_bytes = block_size.y * params.s * params.d * 2 * {bytes_per_elt};
        if( size_in_bytes <= launch_params.device_l2_cache_size ) {{
            // strategy 1: limit to only 1 wave
            block_size.x = std::min(m_steps, sms_per_head);
        }} else {{
            // strategy 2: fully unroll the q loops (contiguous blocks handle all q loops)
            block_size.x = m_steps;
        }}
        params.num_tiles = params.b * params.h;
    }} else if( Ktraits::SCHEDULING_MODE == 1 ) {{
        // Get the max total M steps
        // Take multiple compute groups into consideration.
        size_t m_steps = size_t((params.s + {loop_step} * NUM_COMPUTE_GROUPS - 1) / ({loop_step} * NUM_COMPUTE_GROUPS));
        params.num_tiles_per_head = static_cast<uint32_t>(m_steps);
        params.num_tiles = static_cast<uint32_t>(m_steps * params.b * params.h);
        if (launch_params.attention_mask_type == Attention_mask_type::CAUSAL) {{
            // 2 * {bytes_per_elt} stands for kv cache and {bytes_per_elt} bytes per element.
            size_t size_in_bytes = params.b * params.h * params.s * params.d * 2 * {bytes_per_elt};
            params.use_balanced_scheduling = (size_in_bytes <= launch_params.device_l2_cache_size);
        }}

        block_size.x = 1;
        block_size.y = std::min(static_cast<int>(params.num_tiles), launch_params.multi_processor_count);
    }} else {{
        assert(false && "Invalid SCHEDULING_MODE");
    }}

    // Reuse the same bytes_per_smem for launching kernels.
    constexpr int SMEM_BYTES = Ktraits::BYTES_PER_SMEM;
    if( launch_params.attention_mask_type == Attention_mask_type::PADDING ) {{
#if {padding_mask} // padding_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        {kernel_name}
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>({params_str});
#endif // padding_mask
    }} else if( launch_params.attention_mask_type == Attention_mask_type::CAUSAL ) {{
#if {causal_mask} // causal_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({causal_kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        {causal_kernel_name}
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>({params_str});
#endif // causal mask
    }} else if( launch_params.attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL ) {{
#if {sliding_or_chunked_causal_mask} // sliding_or_chunked_causal_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({sliding_or_chunked_causal_kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        {sliding_or_chunked_causal_kernel_name}
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>({params_str});
#endif // sliding_or_chunked_causal_mask
    }} else if( launch_params.attention_mask_type == Attention_mask_type::CUSTOM_MASK ) {{
#if {custom_mask} // custom_mask
        FMHA_CHECK_CUDA(cudaFuncSetAttribute({custom_mask_kernel_name},
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM_BYTES));

        {custom_mask_kernel_name}
            <<<block_size, Ktraits::THREADS, SMEM_BYTES, stream>>>({params_str});
#endif // custom mask
    }}

}}

#endif
{local_ns_close}
'''


def encode_name(kernel_spec):
    effective_sm, sm_name = get_effective_sm_and_name(kernel_spec)
    # Is it a kernel for the interleaved NC/32HW32 INT8 layout?
    il_tag = '_il' if kernel_spec.interleaved else ''
    # Is it using the quantization scaling factor as an approximation of the max in softmax?
    scale_max_tag = '_scale_max' if kernel_spec.has_scale_max else ''
    # Deal with multi-CTA kernels for which the sequence length is seq_len per CTA * # of CTAs.
    seqlen = kernel_spec.seq_len * kernel_spec.ctas_per_head
    # The qkv layout.
    qkv_layout_tag = ''
    if kernel_spec.input_layout == InputLayout.PACKED_QKV:
        qkv_layout_tag = '_qkv'
    elif kernel_spec.input_layout == InputLayout.Q_PAGED_KV:
        qkv_layout_tag = '_q_paged_kv'
    elif kernel_spec.input_layout == InputLayout.SEPARATE_Q_K_V:
        qkv_layout_tag = '_q_k_v'
    else:
        qkv_layout_tag = '_q_kv'
    # for SM90 kernels, let's also differentiate ldgsts and tma kernels
    feature_tags = ''
    if (effective_sm == 90):
        # let's think about where to insert tma/ldgsts in the string before MR. [Timmy]
        if (kernel_spec.ldgsts_q == True):
            tma_or_ldgsts = '_ldgsts'
        else:
            tma_or_ldgsts = '_tma'
        if kernel_spec.warp_specialization:
            warp_specialization_tag = '_ws'
            # hopper warp-specialized kernels has specialized optimization for cases without alibi.
            if kernel_spec.alibi:
                feature_tags += '_alibi'
            if kernel_spec.return_softmax_stats:
                feature_tags += '_softmax'
        else:
            warp_specialization_tag = ''
    else:
        tma_or_ldgsts = ''
        warp_specialization_tag = ''

    if kernel_spec.enable_attn_logit_softcapping:
        feature_tags += '_softcapping'
    if kernel_spec.sage_block_sizes:
        feature_tags += f"_sage_{'_'.join(map(str, kernel_spec.sage_block_sizes))}"
    if kernel_spec.output_dtype:
        feature_tags += f"_output_{kernel_spec.output_dtype}"
    if kernel_spec.ctas_per_head > 1:
        fmt = 'fmha_v{version}{il_tag}_{dtype}_' + str(
            seqlen
        ) + '_{head_size}{attrib}{scale_max_tag}{tma_or_ldgsts}_sm{sm}'
    elif kernel_spec.flash_attention:
        fmt = 'fmha_v{version}{il_tag}_flash_attention_{dtype}_{loop_step}_{kv_loop_step}_S{qkv_layout_tag}_{head_size}{head_size_v_str}{attrib}{feature_tags}{scale_max_tag}{tma_or_ldgsts}{warp_specialization_tag}_sm{sm}'
    elif kernel_spec.cross_mha:
        fmt = 'fmha_mhca_{dtype}_{seq_len}_{head_size}{scale_max_tag}{tma_or_ldgsts}_sm{sm}'
    else:
        fmt = 'fmha_v{version}{il_tag}_{dtype}_{seq_len}_{head_size}{attrib}{scale_max_tag}{tma_or_ldgsts}_sm{sm}'
    head_size_v_str = "" if kernel_spec.head_size_v == 0 else f"x{kernel_spec.head_size_v}"
    # Assemble the name of the kernel.
    name_base = fmt.format(**kernel_spec._asdict(),
                           head_size_v_str=head_size_v_str,
                           il_tag=il_tag,
                           qkv_layout_tag=qkv_layout_tag,
                           scale_max_tag=scale_max_tag,
                           tma_or_ldgsts=tma_or_ldgsts,
                           warp_specialization_tag=warp_specialization_tag,
                           feature_tags=feature_tags,
                           attrib='__placeholder__')

    # Produce file, launch function and kernel names.
    fname = name_base.replace('__placeholder__', '')
    if seqlen >= 1024 and not kernel_spec.flash_attention:
        fname += '.no_i2f_f2i'
    fname += '.cu'
    lname = ('run_' + name_base).replace('__placeholder__', '')
    kname = name_base + '_kernel'

    # remove causal
    fname = fname.replace("causal_", "")
    return fname, lname, kname


def get_GMMA_shape(instruction_traits, m, n, k, warps_n):
    gmma_k = hopper_traits2shape[instruction_traits][-1]

    # gmma shape is 64xgmma_nx16, gmma_n should be as big as possible, but not bigger than n
    # gmma_n should also be smaller than 256
    gmma_m = 64
    gmma_n = 0
    # find the largest supported n
    n_supported = [(i + 1) * 8 for i in range(32)][::-1]
    n_target = n // warps_n
    assert n_target * warps_n == n
    assert n_supported[0] == 256 and n_supported[-1] == 8
    for cand_n in n_supported:
        if n_target % cand_n == 0:
            gmma_n = cand_n
            break
    assert gmma_n > 0, "No supported GMMA_N found!"

    return gmma_m, gmma_n, gmma_k


def enable_mutex(kspec):
    fp32_accu_dtype = kspec.dtype in ['fp16_fp32', 'bf16']
    enable_mutex = 'false' if (fp32_accu_dtype
                               or kspec.head_size <= 64) else 'true'
    return enable_mutex


def enable_tma_store(kspec):
    output_dtype = kspec.output_dtype if kspec.output_dtype is not None else kspec.dtype
    # TMA copies data in the 16B granularity.
    return 'true' if (output_dtype in ['e4m3', 'e4m3_fp32']
                      and kspec.head_size % 16 == 0) else 'false'


def get_reg_count(kspec):
    # if kspec.paged_kv_input and kspec.dtype in ['fp16', 'fp16_fp32', 'bf16']:
    #     dma_reg_count = 72
    #     compute_reg_count = 216
    if kspec.input_layout == InputLayout.Q_PAGED_KV:
        dma_reg_count = 56
        compute_reg_count = 224
    else:
        dma_reg_count = 40
        compute_reg_count = 232
    return dma_reg_count, compute_reg_count


def get_hopper_instruction_traits(instruction_traits, kernel_spec):
    gmma_shape_p = get_GMMA_shape(instruction_traits, kernel_spec.loop_step,
                                  kernel_spec.seq_len, kernel_spec.head_size,
                                  kernel_spec.warps_n)

    instruction_traits_p = f'{instruction_traits}<{", ".join([str(x) for x in gmma_shape_p])}, false, false>'

    gmma_shape_o = get_GMMA_shape(instruction_traits, kernel_spec.loop_step,
                                  kernel_spec.head_size, kernel_spec.seq_len, 1)
    instruction_traits_o = f'{instruction_traits}<{", ".join([str(x) for x in gmma_shape_o])}, true, false>'

    return instruction_traits_p, instruction_traits_o


def get_effective_sm_and_name(kspec):
    sm = kspec.sm
    # Override the mma instruction with an older one.
    if kspec.sm_mma in sm2name:
        assert kspec.sm_mma <= kspec.sm, "Instruction version should be at most target arch"
        sm = kspec.sm_mma
    sm_name = sm2name[sm]
    return sm, sm_name


def selected_mask_types(kspec):
    # by default, we generate all combinations.
    # '1' means true, '0' means false.
    padding_mask = '1'
    causal_mask = '1'
    sliding_or_chunked_causal_mask = '1'
    custom_mask = '1'
    # only generate certain needed combinations of input_layout and mask types for trt-llm.
    if "GENERATE_CUBIN" in os.environ:
        if kspec.sage_block_sizes:
            # SageAttention only needs padding mask now
            causal_mask = '0'
            sliding_or_chunked_causal_mask = '0'
            custom_mask = '0'
        elif (kspec.head_size, kspec.head_size_v) == (192, 128):
            # MLA context phase only needs causal mask and padding mask (for chunked prefill) now
            sliding_or_chunked_causal_mask = '0'
            custom_mask = '0'
        elif (kspec.head_size, kspec.head_size_v) == (576, 512):
            # MLA generation phase only needs padding mask (MtpMask) now
            causal_mask = '0'
            sliding_or_chunked_causal_mask = '0'
            custom_mask = '0'
        # encoder models (head_size = 32 / 64 / 128) need packed_qkv input layout + padding mask.
        elif kspec.input_layout == InputLayout.PACKED_QKV:
            # NOTE: 72/80 are added for vision transformer
            if kspec.head_size not in [32, 64, 72, 80, 128]:
                padding_mask = '0'
        # only cross attention (head_size = 32/64/128) needs contiguous_q_kv input layout + padding mask / custom_mask.
        elif kspec.input_layout == InputLayout.CONTIGUOUS_Q_KV:
            causal_mask = '0'
            sliding_or_chunked_causal_mask = '0'
            if kspec.head_size not in [32, 64, 72, 128]:
                padding_mask = '0'
                custom_mask = '0'
        # paged kv cache is always needed in gpt variants.
        # cross-attention also needs paged kv cache.
        elif kspec.input_layout == InputLayout.Q_PAGED_KV:
            if kspec.head_size not in [32, 64, 128]:
                padding_mask = '0'

        # alibi specialized kernels only need causal mask.
        if (kspec.alibi and kspec.warp_specialization):
            padding_mask = '0'
            sliding_or_chunked_causal_mask = '0'
            custom_mask = '0'

        # enable_attn_logit_softcapping kernels only need causal mask or sliding_or_chunked_causal_mask.
        if kspec.enable_attn_logit_softcapping:
            padding_mask = '0'
            custom_mask = '0'

    return padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask


def get_kernel_code(kspec, kname, lname):
    min_cuda_version = 0  # no restriction

    # The architecture that determines the instruction.
    effective_sm, sm_name = get_effective_sm_and_name(kspec)

    if effective_sm >= 80:
        min_cuda_version = 11000

    launcher_name = lname
    causal_kernel_name = kname.replace('__placeholder__', '_causal')
    custom_mask_kernel_name = kname.replace('__placeholder__', '_custom_mask')
    sliding_or_chunked_causal_kernel_name = kname.replace(
        '__placeholder__', '_sliding_or_chunked_causal')
    kernel_name = kname.replace('__placeholder__', '')

    # FIXME: use separate parameters when generating cubins for trtllm.
    if not kspec.cross_mha:
        params_type = 'bert::Fused_multihead_attention_params_v{}'.format(
            kspec.version)
    else:
        params_type = 'bert::Fused_multihead_attention_params_mhca'

    if (effective_sm < 90):
        instruction_traits = sm_name.capitalize() + '_' + dtype2traits[
            kspec.dtype]
    elif (effective_sm == 90):
        instruction_traits = sm_name.capitalize() + '_' + hopper_dtype2traits[
            kspec.dtype]
        # for hopper, we differentiate instruction_traits_o and instruction_traits_p
        instruction_traits_p, instruction_traits_o = get_hopper_instruction_traits(
            instruction_traits, kspec)
        #print(instruction_traits_p, instruction_traits_o)

    if (effective_sm < 90):
        if kspec.flash_attention:
            kernel_variant = 'flash_attention'
        else:
            kernel_variant = '1xN' if kspec.warps_m == 1 else '2x2'
    elif (effective_sm == 90):
        if kspec.warps_n > 1:
            # for hopper we slice the problem along the M dim.
            kernel_variant = '4xN' + '_hopper'
        else:
            kernel_variant = '4x1' + '_hopper'

    if (effective_sm < 90):
        kernel_traits = 'Kernel_traits_'
    elif (effective_sm == 90):
        kernel_traits = 'FMHA_kernel_traits_hopper_'

    if kspec.interleaved:
        kernel_traits += 'interleaved_v2'
    elif kspec.cross_mha:
        kernel_traits += 'fmhca'
    else:
        kernel_traits += 'v{}'.format(kspec.version)

    # decide whether to paged_kv kernel traits for ampere-style kernels.
    if effective_sm < 90:
        if kspec.input_layout == InputLayout.Q_PAGED_KV:
            kernel_traits += '_paged_kv_cache'
        elif kspec.input_layout == InputLayout.CONTIGUOUS_Q_KV:
            kernel_traits += '_contiguous_kv_cache'
        elif kspec.input_layout == InputLayout.SEPARATE_Q_K_V:
            kernel_traits += '_q_k_v'

    flags = 0
    if kspec.ldgsts_q:
        flags |= 1
    if kspec.ldgsts_k:
        flags |= 2
    if kspec.ldgsts_v:
        flags |= 4
    if kspec.share_smem_k_v and not kspec.limit_qk_fragments:
        flags |= 8
    if kspec.has_scale_max:
        flags |= 16
    if not kspec.head_interleaved:
        flags |= 32
    if kspec.limit_qk_fragments:
        flags |= 128
    if kspec.limit_v_fragments:
        flags |= 256
    if kspec.has_noloop:
        # NOTE do not use flags 512 = 0x200 as it is reserved; do not add to flags because it
        # will be selectively added to no-loop kernel trait upon generating .cu templates
        pass
    if kspec.enable_attn_logit_softcapping:
        flags |= 2048
    if kspec.tiled:
        flags |= 4096
    if kspec.is_mtp:
        flags |= 8192

    # only generate certain needed combinations of input_layout and mask types for trt-llm.
    padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask = \
        selected_mask_types(kspec)

    if any(selected_mask_flag == '1'
           for selected_mask_flag in selected_mask_types(kspec)):
        padding_mask, causal_mask, sliding_or_chunked_causal_mask, custom_mask = \
            selected_mask_types(kspec)
    else:
        return None

    kernel_flags = '0x{:02x}u'.format(flags)

    heads_interleaved_flag = pythonBoolean2cpp[kspec.head_interleaved]

    disable_fadd_trick = 1 if effective_sm >= 86 else 0  # this will force generating F2IP

    enable_mutex_flag = enable_mutex(kspec)

    has_alibi = pythonBoolean2cpp[kspec.alibi]

    input_layout_flag = str(int(kspec.input_layout))

    run_fct_name = 'run_packed_qkv' if kspec.input_layout == InputLayout.PACKED_QKV else \
        'run_separate_q_and_kv'

    dma_reg_count, compute_reg_count = get_reg_count(kspec)

    use_tma_store_flag = enable_tma_store(kspec)

    enable_attn_logit_softcapping_flag = pythonBoolean2cpp[
        kspec.enable_attn_logit_softcapping]

    return_softmax_stats_flag = pythonBoolean2cpp[kspec.return_softmax_stats]

    # needed by warpspec kernels.
    fp8_kernel = kspec.dtype in ["e4m3", "e4m3_fp32"]
    kernel_traits_header =  "fmha::ws::Kernel_traits_Hopper_qgmma_e4m3_fp32<" if fp8_kernel \
        else f"fmha::ws::Kernel_traits<fmha::{instruction_traits},"

    # output type.
    output_dtype_ = f"fmha::{dtype2OutputType[kspec.output_dtype if kspec.output_dtype is not None else kspec.dtype]}"

    # sage attention block sizes.
    sage_block_size_q = 0
    sage_block_size_k = 0
    sage_block_size_v = 0
    if fp8_kernel and kspec.sage_block_sizes:
        assert kspec.output_dtype is not None, "output_dtype must be specified for fp8 sage attention kernels"
        sage_block_size_q = kspec.sage_block_sizes[0]
        sage_block_size_k = kspec.sage_block_sizes[1]
        sage_block_size_v = kspec.sage_block_sizes[2]

    TMA_config = r'''
    // TMA configuration
    // Note that this may only need to init once during inference (for different layers)
    // Reuse the same traits for initializing tma descriptors.
    fmha::ws::DMA<Ktraits>::Host dma_host;
    dma_host.init_params(params, launch_params, stream);
    ''' if not generate_cu_trtllm else ''
    params_str = 'reinterpret_cast<bert::Fused_multihead_attention_params_v2 &>(params)' if generate_cu_trtllm else 'params'
    attn_mask_type_str = 'using Attention_mask_type = ContextAttentionMaskType;' if generate_cu_trtllm else 'using Attention_mask_type = fmha::Attention_mask_type;'
    bert_launch_params = '' if generate_cu_trtllm else 'using Launch_params = bert::Fused_multihead_attention_launch_params;'
    include_str = '#include "../fused_multihead_attention_common.h"' if generate_cu_trtllm else ''
    num_compute_groups_str = '' if generate_cu_trtllm else 'static constexpr int NUM_COMPUTE_GROUPS = 2;'
    fused_multihead_attention_params_v2_str = 'Fused_multihead_attention_params_v2' if generate_cu_trtllm else f'{params_type}'
    const_fused_multihead_attention_params_v2_str = 'Fused_multihead_attention_params_v2' if generate_cu_trtllm else f'const {params_type}'
    setmaxnreg_dma_str = r'''
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int DMA_REG_COUNT = {dma_reg_count};
        asm volatile("{{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}}" ::"n"(DMA_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif
'''.format(dma_reg_count=dma_reg_count) if generate_cu_trtllm else r'''
        const int DMA_REG_COUNT = {dma_reg_count};
        asm volatile("{{setmaxnreg.dec.sync.aligned.u32  %0; \n\t}}" ::"n"(DMA_REG_COUNT));'''.format(
        dma_reg_count=dma_reg_count)
    setmaxnreg_compute_str = r'''
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
        const int COMPUTE_REG_COUNT = {compute_reg_count};
        asm volatile("{{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}}" ::"n"(COMPUTE_REG_COUNT));
#else
        asm volatile("trap;\n");
#endif
'''.format(compute_reg_count=compute_reg_count) if generate_cu_trtllm else r'''
        const int COMPUTE_REG_COUNT = {compute_reg_count};
        asm volatile("{{setmaxnreg.inc.sync.aligned.u32 %0; \n\t}}" ::"n"(COMPUTE_REG_COUNT));'''.format(
        compute_reg_count=compute_reg_count)
    local_ns_open = ns_open if generate_cu_trtllm else ''
    local_ns_close = ns_close if generate_cu_trtllm else ''

    tmp = dict(locals(), **kspec._asdict())

    if (effective_sm < 90):
        if kspec.flash_attention:
            code = flash_attention_kernel_template.format(
                **tmp,
                copyright=copyright,
                use_multi_cta=False,
                MAX_STGS_PER_LOOP=MAX_STGS_PER_LOOP)
        else:
            use_multi_cta = 1 if kspec.ctas_per_head > 1 else 0
            code = kernel_template.format(**tmp,
                                          copyright=copyright,
                                          use_multi_cta=use_multi_cta,
                                          MAX_STGS_PER_LOOP=MAX_STGS_PER_LOOP)
    elif (effective_sm == 90):
        use_tma = 1
        if (kspec.ldgsts_q == True):
            use_tma = 0
        if kspec.warp_specialization:
            code = kernel_hopper_warp_specialization_template.format(
                **tmp,
                copyright=copyright,
                use_tma=use_tma,
                bytes_per_elt=dtype2bytes[kspec.dtype])
        else:
            code = kernel_hopper_template.format(**tmp,
                                                 copyright=copyright,
                                                 use_tma=use_tma)
    return code


def get_api_code(specs_names):

    def get_signature(lname, version, cross_mha, use_tma):
        # The architecture that determines the instruction.
        effective_sm, sm_name = get_effective_sm_and_name(kspec)
        if cross_mha:
            return 'void {}(const Params_mhca &params, cudaStream_t stream);'.format(
                lname)
        elif effective_sm >= 90:
            # need to set tma desc in params
            return 'void {}(Params_v{} &params, const Launch_params &launch_params, cudaStream_t stream);'.format(
                lname, version)
        else:
            return 'void {}(const Params_v{} &params, const Launch_params &launch_params, cudaStream_t stream);'.format(
                lname, version)

    signatures = []
    for kspec, fname, lname, kname in specs_names:
        effective_sm, _ = get_effective_sm_and_name(kspec)
        use_tma = effective_sm == 90 and not kspec.ldgsts_q
        signatures.append(
            get_signature(lname, kspec.version, kspec.cross_mha, use_tma))
        if kspec.has_noloop and not kspec.tiled:
            signatures.append(
                get_signature(lname + '_nl', kspec.version, kspec.cross_mha,
                              use_tma))
        elif kspec.tiled:
            signatures.append(
                get_signature(lname + '_nl_tiled', kspec.version,
                              kspec.cross_mha, use_tma))
        if not kspec.warp_specialization:
            signatures.append(
                'void {}_get_max_heads_per_wave(int*);'.format(lname))
    signatures = '\n'.join(signatures)

    #v1
    # - normal
    # - no loop
    #v2
    # - normal
    # - no loop
    # - normal interleaved
    # - no loop interleaved
    # - flash attention no loop
    # - flash attention no loop tiled
    # - flash attention warp_specialized (on Hopper)

    def gen_unroll_check(kspec):
        code = 'if (!{has_noloop} || (!force_unroll && (ignore_b1opt || b > {unroll_threshold})))'.format(
            **kspec._asdict())
        if kspec.flash_attention:
            code = 'if (!{has_noloop} || (!force_unroll && (ignore_b1opt || b * h > {unroll_threshold})))'.format(
                **kspec._asdict())
        return code

    def gen_call(kspec, lname):
        effective_sm, _ = get_effective_sm_and_name(kspec)
        data_type = dtype2typename[kspec.dtype]
        output_data_type = data_type
        if kspec.output_dtype:
            output_data_type = dtype2typename[kspec.output_dtype]
        il_check = ''
        if kspec.version == 2 and kspec.dtype in ["fp16", "bf16"]:
            il_check += "&& use_flash_attention " if kspec.flash_attention else '&& !use_flash_attention '
        if kspec.version == 2:
            # attention input layout.
            il_check += f'&& attention_input_layout == {kspec.input_layout.value} '
            # interleaved layout or not.
            il_check += '&& interleaved ' if kspec.interleaved else '&& !interleaved '
            if effective_sm == 90:
                il_check += "&& !use_tma " if kspec.ldgsts_q else '&& use_tma '
                il_check += "&& warp_specialization " if kspec.warp_specialization else '&& !warp_specialization '
            else:
                il_check += "&& !warp_specialization && !use_tma "
            # Different accumulation types.
            if '_fp32' in kspec.dtype or 'bf16' in kspec.dtype or kspec.dtype == 'e4m3':
                il_check += '&& force_fp32_acc '
            else:
                il_check += '&& !force_fp32_acc '
            # whether support alibi or not.
            if kspec.warp_specialization:
                il_check += '&& params.has_alibi ' if kspec.alibi else '&& !params.has_alibi '
                il_check += '&& params.softmax_stats_ptr != nullptr ' if kspec.return_softmax_stats else '&& params.softmax_stats_ptr == nullptr '
            # use enable_attn_logit_softcapping or not.
            il_check += '&& enable_attn_logit_softcapping ' if kspec.enable_attn_logit_softcapping else '&& !enable_attn_logit_softcapping '
            # check sage block sizes
            sage_block_size_q = 0
            sage_block_size_k = 0
            sage_block_size_v = 0
            if kspec.sage_block_sizes:
                # override the data_type to output type, otherwise it is always E4M3
                data_type = output_data_type
                sage_block_size_q = kspec.sage_block_sizes[0]
                sage_block_size_k = kspec.sage_block_sizes[1]
                sage_block_size_v = kspec.sage_block_sizes[2]
            il_check += f'&& sage_block_size_q == {sage_block_size_q} ' \
                f'&& sage_block_size_k == {sage_block_size_k} ' \
                f'&& sage_block_size_v == {sage_block_size_v} '

        il_check += '&& params.use_int8_scale_max ' if kspec.has_scale_max else '&& !params.use_int8_scale_max '

        slen = kspec.seq_len * kspec.ctas_per_head if not kspec.flash_attention else 0

        ## NOTE: need to tune here
        if kspec.has_noloop and not kspec.flash_attention:
            call_stmt = '''\
if( data_type == {data_type} && output_data_type == {output_data_type} && s == {slen} && d == {head_size} && sm == {sm}
    {il_check}) {{

    {unroll_check} {{
        {lname}(params, launch_params, stream);
    }} else {{
        {lname}_nl(params, launch_params, stream);
    }}

}} '''.format(**kspec._asdict(),
              data_type=data_type,
              output_data_type=output_data_type,
              slen=slen,
              lname=lname,
              il_check=il_check,
              unroll_check=gen_unroll_check(kspec))

        elif kspec.flash_attention:  #NOTE: flash attention uses no_loop as default
            # TypeError: got multiple values for keyword argument if using key 'head_size_v', so 'dv' instead
            dv = kspec.head_size_v or kspec.head_size
            if kspec.tiled:  # higher precedence; does not require bh_upper_thres
                call_stmt = '''\
if( data_type == {data_type} && output_data_type == {output_data_type} && d == {head_size} && dv == {dv} && sm == {sm}
    {il_check} && use_tiled) {{

    {lname}_nl_tiled(params, launch_params, stream);

}} '''.format(**kspec._asdict(),
                data_type=data_type,
                output_data_type=output_data_type,
                slen=slen,
                lname=lname,
                il_check=il_check,
                dv=dv)
            # warp specialization kernels need launch_params
            elif kspec.warp_specialization:
                call_stmt = '''\
if( data_type == {data_type} && output_data_type == {output_data_type} && d == {head_size} && dv == {dv} && sm == {sm}
    {il_check}) {{

    {lname}(params, launch_params, stream);

}} '''.format(**kspec._asdict(),
                data_type=data_type,
                output_data_type=output_data_type,
                slen=slen,
                lname=lname,
                il_check=il_check,
                dv=dv)
            else:
                call_stmt = '''\
if( data_type == {data_type} && output_data_type == {output_data_type} && d == {head_size} && dv == {dv} && sm == {sm}
    && !use_tiled {il_check}) {{

    {lname}_nl(params, launch_params, stream);

}} '''.format(**kspec._asdict(),
                data_type=data_type,
                output_data_type=output_data_type,
                slen=slen,
                lname=lname,
                il_check=il_check,
                dv=dv)
        else:
            call_stmt = '''\
if( data_type == {data_type} && output_data_type == {output_data_type} && s == {slen} && d == {head_size} && sm == {sm}
    {il_check}) {{

    {lname}(params, launch_params, stream);

}} '''.format(**kspec._asdict(),
              data_type=data_type,
              output_data_type=output_data_type,
              slen=slen,
              lname=lname,
              il_check=il_check)
        return call_stmt

    def gen_call_fmhca(kspec, lname):
        effective_sm, _ = get_effective_sm_and_name(kspec)
        data_type = dtype2typename[kspec.dtype]
        il_check = ''
        if kspec.version == 2:
            il_check = '&& interleaved ' if kspec.interleaved else '&& !interleaved '
        if effective_sm == 90:
            il_check += "&& !use_tma " if kspec.ldgsts_q else '&& use_tma '
        il_check += '&& params.use_int8_scale_max ' if kspec.has_scale_max else '&& !params.use_int8_scale_max '

        s_kv_len = kspec.seq_len
        if kspec.has_noloop:
            call_stmt = '''\
if( data_type == {data_type} && s_kv == {s_kv_len} && d == {head_size} && sm == {sm} {il_check}) {{

    {unroll_check} {{
        {lname}(params, stream);
    }} else {{
        {lname}_nl(params, stream);
    }}

}} '''.format(**kspec._asdict(),
              data_type=data_type,
              s_kv_len=s_kv_len,
              lname=lname,
              il_check=il_check,
              unroll_check=gen_unroll_check(kspec))

        else:
            call_stmt = '''\
if( data_type == {data_type} && s_kv == {s_kv_len} && d == {head_size} && sm == {sm} {il_check}) {{
        {lname}(params, stream);
    }} '''.format(**kspec._asdict(),
                  data_type=data_type,
                  s_kv_len=s_kv_len,
                  lname=lname,
                  il_check=il_check)
        return call_stmt

    calls_v2 = [
        gen_call(kspec, lname)
            for kspec, fname, lname, kname in specs_names \
            if kspec.version == 2 and kspec.cross_mha == 0
    ]

    calls_v2 = 'else '.join(calls_v2) if len(calls_v2) > 0 else 'if( false ) {}'

    calls_v1 = [
        gen_call(kspec, lname) for kspec, fname, lname, kname in specs_names
        if kspec.version == 1 and kspec.cross_mha == 0
    ]

    calls_v1 = 'else '.join(calls_v1) if len(calls_v1) > 0 else 'if( false ) {}'

    calls_mhca = [
        gen_call_fmhca(kspec, lname)
        for kspec, fname, lname, kname in specs_names if kspec.cross_mha == 1
    ]

    calls_mhca = 'else '.join(calls_mhca) if len(
        calls_mhca) > 0 else 'if( false ) {}'

    def gen_warp_spec(kspec):
        data_type = dtype2typename[kspec.dtype]
        if kspec.sage_block_sizes is not None:
            assert kspec.output_dtype is not None
            # override the data_type to output type, otherwise it is always E4M3
            data_type = dtype2typename[kspec.output_dtype]
        slen = kspec.seq_len * kspec.ctas_per_head
        effective_sm, _ = get_effective_sm_and_name(kspec)
        warp_spec_check = ''
        nl_warps_m = kspec.warps_m if effective_sm == 90 else 1
        nl_warps_n = kspec.warps_n if effective_sm == 90 else kspec.warps_m * kspec.warps_n
        if kspec.version == 2 and kspec.dtype in ["fp16", "bf16"]:
            warp_spec_check += "&& use_flash_attention " if kspec.flash_attention else '&& !use_flash_attention '
        if kspec.version == 2:
            if effective_sm == 90:
                warp_spec_check += "&& !use_tma " if kspec.ldgsts_q else '&& use_tma '
                warp_spec_check += "&& warp_specialization " if kspec.warp_specialization else '&& !warp_specialization '
            else:
                warp_spec_check += '&& !use_tma && !warp_specialization '

        if kspec.flash_attention:  # NOTE support any sequence
            return '''\
if( data_type == {data_type} && d == {head_size} && sm == {sm} {warp_spec_check}
    && version == {version} ) {{
    warps_m = {warps_m};
    warps_n = {warps_n};
}} '''.format(**locals(),
              **kspec._asdict(),
              unroll_check=gen_unroll_check(kspec))
        return '''\
if( data_type == {data_type} && s == {slen} && d == {head_size} && sm == {sm} {warp_spec_check}
    && version == {version} ) {{
    {unroll_check} {{
      warps_m = {warps_m};
      warps_n = {warps_n};
    }} else {{
      warps_m = {nl_warps_m};
      warps_n = {nl_warps_n};
    }}
}} '''.format(**locals(),
              **kspec._asdict(),
              unroll_check=gen_unroll_check(kspec))

    warp_specs = 'else '.join([gen_warp_spec(spec[0]) for spec in specs_names])
    if len(warp_specs) > 0:
        warp_specs += 'else {\n\tassert(false && "Unsupported config");\n}'

    # Generate the cta spec.
    def gen_cta_spec(spec):
        kspec, _, lname, _ = spec
        slen = kspec.seq_len * kspec.ctas_per_head
        return '''\
if( data_type == {data_type} && s == {slen} && d == {head_size} && use_multi_ctas
    && version == {version} ) {{

    ctas_per_head = {ctas_per_head};
    {lname}_get_max_heads_per_wave(&max_heads_per_wave);

}} '''.format(**locals(),
              **kspec._asdict(),
              data_type=dtype2typename[kspec.dtype])

    cta_specs = 'else '.join([
        gen_cta_spec(spec) for spec in specs_names if spec[0].ctas_per_head > 1
    ])

    api_code = '''\
{copyright}
#pragma once

#include <cuda.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_cross_attention.h>
#include <tuple>

using Params_v1         = bert::Fused_multihead_attention_params_v1;
using Params_v2         = bert::Fused_multihead_attention_params_v2;
using Params_mhca       = bert::Fused_multihead_attention_params_mhca;
using Launch_params     = bert::Fused_multihead_attention_launch_params;

{signatures}

inline void run_fmha_v1(Params_v1 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        Data_type output_data_type,
                        int sm,
                        cudaStream_t stream=0){{
const size_t s                 = params.s;
const size_t b                 = params.b;
const size_t d                 = params.d;
const bool force_unroll        = launch_params.force_unroll;
const bool ignore_b1opt        = launch_params.ignore_b1opt;

const bool use_flash_attention = false;

{calls_v1}
else {{
    assert(false && "Unsupported config.");
}}

}}

// Note: transitioning to moving kernel launch parameters into launch_params to reduce the
// occurrences the interface needs to be modified
inline void run_fmha_v2(Params_v2 &params,
                        const Launch_params &launch_params,
                        Data_type data_type,
                        Data_type output_data_type,
                        int sm,
                        cudaStream_t stream=0) {{

const size_t s = params.s;
const size_t b = params.b;
const size_t h = params.h;
const size_t d = params.d;
const size_t dv = params.dv;
const size_t sage_block_size_q = params.sage.q.block_size;
const size_t sage_block_size_k = params.sage.k.block_size;
const size_t sage_block_size_v = params.sage.v.block_size;

const bool interleaved                       = launch_params.interleaved;
const bool force_unroll                      = launch_params.force_unroll;
const bool ignore_b1opt                      = launch_params.ignore_b1opt;
const bool force_fp32_acc                    = launch_params.force_fp32_acc;
const bool warp_specialization               = launch_params.warp_specialization;
const bool use_tma                           = launch_params.use_tma;
const bool use_flash_attention               = launch_params.flash_attention;
const bool enable_attn_logit_softcapping     = launch_params.enable_attn_logit_softcapping;
const int  attention_input_layout            = static_cast<int>(launch_params.attention_input_layout);
// tiled variant uses ldgsts
const bool  use_tiled            = launch_params.use_granular_tiling;

{calls_v2}
else {{
    assert(false && "Unsupported config.");
}}

}}

#if __guard_fmhca_placeholder__ // fmhca api header

inline void run_fmhca(Params_mhca &params,
                      const Launch_params &launch_params,
                      Data_type data_type,
                      int sm,
                      cudaStream_t stream=0) {{

const size_t s_kv   = params.s;
const size_t b      = params.b;
const size_t d      = params.d_padded;

const bool interleaved  = launch_params.interleaved;
const bool force_unroll = launch_params.force_unroll;
const bool ignore_b1opt = launch_params.ignore_b1opt;

{calls_mhca}
else {{
    assert(false && "Unsupported config");
}}

}}

#endif // fmhca api header

inline std::tuple<size_t, size_t, size_t> get_warps(Launch_params& launch_params,
                                                    int sm,
                                                    Data_type data_type,
                                                    size_t s,
                                                    size_t b,
                                                    size_t d,
                                                    int version) {{
    size_t warps_m, warps_n, warps_k = 1;
    const bool interleaved           = launch_params.interleaved;
    const bool use_tma               = launch_params.use_tma;
    const bool force_unroll          = launch_params.force_unroll;
    const bool ignore_b1opt          = launch_params.ignore_b1opt;
    const bool use_flash_attention   = launch_params.flash_attention;
    // tiled variant uses ldgsts
    const bool use_tiled             = launch_params.use_granular_tiling;
    const bool warp_specialization   = launch_params.warp_specialization;

{warp_specs}

    return std::make_tuple(warps_m, warps_n, warps_k);
}}

// The constant is defined in "setup.py".
constexpr int MAX_STGS_PER_LOOP = {MAX_STGS_PER_LOOP};

// The number of CTAs and threads per CTA to launch the kernel.
inline void get_grid_size(int &heads_per_wave,
                          int &ctas_per_head,
                          int sm,
                          Data_type data_type,
                          size_t b,
                          size_t s,
                          size_t h,
                          size_t d,
                          bool use_multi_ctas,
                          int version) {{

    // Determine the number of CTAs per head (kernel constant).
    int max_heads_per_wave = 0;
    ctas_per_head = 1;
    heads_per_wave = b*h;
{cta_specs}

    // Adjust the number of heads per wave.
    if( heads_per_wave > max_heads_per_wave ) {{
        heads_per_wave = max_heads_per_wave;
    }}
}}

'''.format(**locals(), copyright=copyright, MAX_STGS_PER_LOOP=MAX_STGS_PER_LOOP)
    return api_code


ktraits_code_template = '''
#include "fused_multihead_attention_kernel.h"
#include "fmha/kernel_traits.h"
#include "fmha/hopper/kernel_traits.h"
#include <fmha/warpspec/kernel_traits.h>

using namespace fmha;

int main(){{
{print_kernel_specs}
}}
'''


def get_kernel_traits_code(specs_names):
    print_kernel_specs = []

    for kspec, fname, lname, kname in specs_names:
        effective_sm, sm_name = get_effective_sm_and_name(kspec)
        if (effective_sm < 90):
            instruction_traits = sm_name.capitalize() + '_' + dtype2traits[
                kspec.dtype]
        elif (effective_sm == 90):
            instruction_traits = sm_name.capitalize(
            ) + '_' + hopper_dtype2traits[kspec.dtype]
            instruction_traits_p, instruction_traits_o = get_hopper_instruction_traits(
                instruction_traits, kspec)

        if (effective_sm < 90):
            kernel_traits = 'Kernel_traits_'
        elif (effective_sm == 90):
            kernel_traits = 'FMHA_kernel_traits_hopper_'

        if kspec.interleaved:
            kernel_traits += 'interleaved_v2'
        elif kspec.cross_mha:
            kernel_traits += 'fmhca'
        else:
            kernel_traits += 'v{}'.format(kspec.version)

        # needed by warpspec kernels.
        fp8_kernel = kspec.dtype in ["e4m3", "e4m3_fp32"]
        kernel_traits_header =  "fmha::ws::Kernel_traits_Hopper_qgmma_e4m3_fp32<" if fp8_kernel \
            else f"fmha::ws::Kernel_traits<fmha::{instruction_traits},"

        flags = 0
        if kspec.ldgsts_q:
            flags |= 1
        if kspec.ldgsts_k:
            flags |= 2
        if kspec.ldgsts_v:
            flags |= 4
        if kspec.share_smem_k_v:
            flags |= 8
        if kspec.has_scale_max:
            flags |= 16
        if not kspec.head_interleaved:
            flags |= 32
        if kspec.limit_qk_fragments:
            flags |= 128
        if kspec.limit_qk_fragments:
            flags |= 256
        if kspec.has_noloop:
            # NOTE do not use flags 512 = 0x200 as it is reserved; do not add to flags because it
            # will be selectively added to no-loop kernel trait upon generating .cu templates
            pass
        if kspec.enable_attn_logit_softcapping:
            flags |= 2048
        if kspec.tiled:
            flags |= 4096
        if kspec.is_mtp:
            flags |= 8192

        kernel_flags = '0x{:02x}u'.format(flags)

        heads_interleaved_flag = pythonBoolean2cpp[kspec.head_interleaved]

        enable_mutex_flag = enable_mutex(kspec)

        has_alibi = pythonBoolean2cpp[kspec.alibi]

        return_softmax_stats_flag = pythonBoolean2cpp[
            kspec.return_softmax_stats]

        input_layout_flag = str(int(kspec.input_layout))

        enable_attn_logit_softcapping_flag = pythonBoolean2cpp[
            kspec.enable_attn_logit_softcapping]

        tmp = dict(locals(), **kspec._asdict())

        if effective_sm < 90:
            snippet = '''    {{
            using Kernel_traits = {kernel_traits}<
                fmha::{instruction_traits},
                {seq_len},
                {head_size},
                {head_size_v},
                {loop_step},
                {warps_m},
                {warps_n},
                {ctas_per_head},
                {kernel_flags}>;
            printf("%s %d %d %s %d %d\\n",
                \"{kernel_name}\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {loop_step},
                {unroll_threshold});
        }}'''.format(**tmp, kernel_name=kname.replace('__placeholder__', ''))
            snippet_nl = '''    {{
            using Kernel_traits = {kernel_traits}<
                fmha::{instruction_traits},
                {seq_len},
                {head_size},
                {head_size_v},
                {noloop_step},
                1,
                {warps_m} * {warps_n},
                {ctas_per_head},
                {kernel_flags} | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\\n",
                \"{kernel_name}_nl\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {noloop_step},
                {unroll_threshold});
        }}'''.format(**tmp, kernel_name=kname.replace('__placeholder__', ''))
            snippet_flash = '''    {{
            using Kernel_traits = {kernel_traits}<
                fmha::{instruction_traits},
                {kv_loop_step},
                {head_size},
                {head_size_v},
                {loop_step},
                {warps_m},
                {warps_n},
                {ctas_per_head},
                {kernel_flags}>;
            printf("%s %d %d %s %d %d\\n",
                \"{kernel_name}\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {loop_step},
                {unroll_threshold});
        }}'''.format(**tmp, kernel_name=kname.replace('__placeholder__', ''))
            snippet_flash_nl_template = '''    {{
            using Kernel_traits = {kernel_traits}<
                fmha::{instruction_traits},
                {kv_loop_step},
                {head_size},
                {head_size_v},
                {noloop_step},
                {warps_m},
                {warps_n},
                {ctas_per_head},
                {kernel_flags} | 0x200 /* no_loop flag */>;
            printf("%s %d %d %s %d %d\\n",
                \"{kname}_nl\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {noloop_step},
                {unroll_threshold});
        }}'''.format(**tmp)
            snippet_flash_nl = snippet_flash_nl_template.replace(
                '__placeholder__', '')
            snippet_flash_nl_tiled = snippet_flash_nl_template.replace(
                '__placeholder__', '').replace('_nl', '_nl_tiled')
            snippet_flash_nl_causal = snippet_flash_nl_template.replace(
                '__placeholder__', '_causal')
            snippet_flash_nl_tiled_causal = snippet_flash_nl_template.replace(
                '__placeholder__', '_causal').replace('_nl', '_nl_tiled')
            snippet_flash_nl_sliding_or_chunked_causal = snippet_flash_nl_template.replace(
                '__placeholder__', '_sliding_or_chunked_causal')
            snippet_flash_nl_tiled_sliding_or_chunked_causal = snippet_flash_nl_template.replace(
                '__placeholder__',
                '_sliding_or_chunked_causal').replace('_nl', '_nl_tiled')
            snippet_flash_nl_custom_mask = snippet_flash_nl_template.replace(
                '__placeholder__', '_custom_mask')
            snippet_flash_nl_tiled_custom_mask = snippet_flash_nl_template.replace(
                '__placeholder__', '_custom_mask').replace('_nl', '_nl_tiled')
        elif effective_sm >= 90 and kspec.warp_specialization:  #GMMA warpspec flash
            snippet_ws_template = ''' {{
            static constexpr int DMA2COMPUTE_DEPTH = 1;
            static constexpr int NUM_COMPUTE_GROUPS = 2;

            using Kernel_traits = {kernel_traits_header}
                                  {loop_step},
                                  {kv_loop_step},
                                  {head_size},
                                  {head_size_v},
                                  {q_tile_buffers},
                                  {kv_tile_buffers},
                                  NUM_COMPUTE_GROUPS,
                                  DMA2COMPUTE_DEPTH,
                                  mask_type,
                                  {heads_interleaved_flag},
                                  {has_alibi},
                                  {enable_mutex_flag},
                                  {scheduling_mode},
                                  {input_layout_flag},
                                  __use_tma_store__ /* USE_TMA_STORE */,
                                  {enable_attn_logit_softcapping_flag},
                                  {return_softmax_stats_flag}>;

            printf("%s %d %d %s %d %d\\n",
                \"{kname}\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {loop_step},
                {unroll_threshold});
        }}'''.format(**tmp)
            snippet_ws = snippet_ws_template.replace('__placeholder__', '').\
                                                   replace('mask_type', '0').\
                                                   replace('__use_tma_store__', 'true')
            snippet_ws_causal = snippet_ws_template.replace('__placeholder__', '_causal').\
                                                          replace('mask_type', '1').\
                                                          replace('__use_tma_store__', 'true')
            snippet_ws_sliding_or_chunked_causal = \
                snippet_ws_template.replace('__placeholder__', '_sliding_or_chunked_causal').\
                                       replace('mask_type', '2').\
                                       replace('__use_tma_store__', 'false')
            snippet_ws_custom_mask = \
                snippet_ws_template.replace('__placeholder__', '_custom_mask').\
                                       replace('mask_type', '2').\
                                       replace('__use_tma_store__', 'true')
        elif effective_sm >= 90:  #GMMA no flash yet
            snippet_template = '''    {{
            using Traits_p = fmha::{instruction_traits_p};
            using Traits_o = fmha::{instruction_traits_o};

            using Kernel_traits = {kernel_traits}<
                Traits_p,
                Traits_o,
                {seq_len},
                {head_size},
                {loop_step},
                {warps_m},
                {warps_n},
                2,
                {kernel_flags}>;
            printf("%s %d %d %s %d %d\\n",
                \"{kname}\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {loop_step},
                {unroll_threshold});
        }}'''.format(**tmp)
            snippet_nl_template = '''    {{
            using Traits_p = fmha::{instruction_traits_p};
            using Traits_o = fmha::{instruction_traits_o};

            using Kernel_traits = {kernel_traits}<
                Traits_p,
                Traits_o,
                {seq_len},
                {head_size},
                {noloop_step},
                {warps_m},
                {warps_n},
                2,
                {kernel_flags}>;
            printf("%s %d %d %s %d %d\\n",
                \"{kname}_nl\",
                Kernel_traits::BYTES_PER_SMEM,
                Kernel_traits::THREADS,
                \"{fname}\",
                {noloop_step},
                {unroll_threshold});
        }}'''.format(**tmp)
            snippet = snippet_template.replace('__placeholder__', '')
            snippet_causal = snippet_template.replace(
                '__placeholder__', '_sliding_or_chunked_causal')
            snippet_sliding_or_chunked_causal = snippet_template.replace(
                '__placeholder__', '_causal')
            snippet_nl = snippet_nl_template.replace('__placeholder__', '')
            snippet_nl_causal = snippet_nl_template.replace(
                '__placeholder__', '_causal')
            snippet_nl_sliding_or_chunked_causal = snippet_nl_template.replace(
                '__placeholder__', '_sliding_or_chunked_causal')

        # only generate certain needed combinations of input_layout and mask types for trt-llm.
        selected_types = selected_mask_types(kspec)

        padding_mask = int(selected_types[0])
        causal_mask = int(selected_types[1])
        sliding_or_chunked_causal_mask = int(selected_types[2])
        custom_mask = int(selected_types[3])

        if not padding_mask:
            snippet = None
            snippet_nl = None
            snippet_ws = None
            snippet_flash_nl = None
            snippet_flash_nl_tiled = None
        if not causal_mask:
            snippet_causal = None
            snippet_nl_causal = None
            snippet_ws_causal = None
            snippet_flash_nl_causal = None
            snippet_flash_nl_tiled_causal = None
        if not sliding_or_chunked_causal_mask:
            snippet_sliding_or_chunked_causal = None
            snippet_nl_sliding_or_chunked_causal = None
            snippet_ws_sliding_or_chunked_causal = None
            snippet_flash_nl_sliding_or_chunked_causal = None
            snippet_flash_nl_tiled_sliding_or_chunked_causal = None
        if not custom_mask:
            snippet_ws_custom_mask = None
            snippet_flash_nl_custom_mask = None
            snippet_flash_nl_tiled_custom_mask = None

        if kspec.flash_attention:
            pass
            # print_kernel_specs.append(snippet_flash) # disabled as looped flash performs worse
        else:
            print_kernel_specs.append(snippet)
            if 'snippet_causal' in locals():
                print_kernel_specs.append(snippet_causal)
            if 'snippet_sliding_or_chunked_causal' in locals():
                print_kernel_specs.append(snippet_sliding_or_chunked_causal)
        if kspec.has_noloop:
            if kspec.flash_attention and kspec.tiled == 1:
                print_kernel_specs.append(snippet_flash_nl_tiled)
                print_kernel_specs.append(snippet_flash_nl_tiled_causal)
                print_kernel_specs.append(
                    snippet_flash_nl_tiled_sliding_or_chunked_causal)
                print_kernel_specs.append(snippet_flash_nl_tiled_custom_mask)
            elif kspec.flash_attention and kspec.tiled == 0:
                print_kernel_specs.append(snippet_flash_nl)
                print_kernel_specs.append(snippet_flash_nl_causal)
                print_kernel_specs.append(
                    snippet_flash_nl_sliding_or_chunked_causal)
                print_kernel_specs.append(snippet_flash_nl_custom_mask)
            else:
                print_kernel_specs.append(snippet_nl)
                if 'snippet_nl_causal' in locals():
                    print_kernel_specs.append(snippet_nl_causal)
                if 'snippet_nl_sliding_or_chunked_causal' in locals():
                    print_kernel_specs.append(
                        snippet_nl_sliding_or_chunked_causal)

        if kspec.warp_specialization:
            print_kernel_specs.append(snippet_ws)
            print_kernel_specs.append(snippet_ws_causal)
            print_kernel_specs.append(snippet_ws_sliding_or_chunked_causal)
            print_kernel_specs.append(snippet_ws_custom_mask)
    # remove none.
    print_kernel_specs = [
        spec for spec in print_kernel_specs if spec is not None
    ]
    print_kernel_specs = '\n'.join(print_kernel_specs)

    code = ktraits_code_template.format(print_kernel_specs=print_kernel_specs)
    return code


# For now:
# 1. Hopper head_size 128 kernel uses cubins for performance regressions.
# 2. Hopper sm89 with e4m3/e4m3_fp32 dtype uses cubins for accuracy regressions (will be fixed).
# You should set the condition `use_cubin_header` to false if you have modified the source codes of those kernels that use cubins.
# This ensures that the kernels will be recompiled using the updated source code rather than relying on precompiled cubins.
def use_cubin_header(sm, head_size, dtype):
    return (sm == 90 and head_size == 128) or (sm == 89 and 'e4m3' in dtype)


def get_cubin_header(kernel_traits, specs_names):
    cubins = []
    cubin_lens = []
    cubins_dict = {}
    cubin_lens_dict = {}
    for kspec, fname, lname, kname in specs_names:
        if generate_cu_trtllm and not use_cubin_header(
                kspec.sm, kspec.head_size, kspec.dtype):
            continue
        name = fname.replace('.', '_')
        data = 'extern unsigned char cubin_{name}_cubin[];'.format(name=name)
        size = 'extern uint32_t cubin_{name}_cubin_len;'.format(name=name)
        if kspec.sm in cubins_dict:
            cubins_dict[kspec.sm].append(data)
            cubin_lens_dict[kspec.sm].append(size)
        else:
            cubins_dict[kspec.sm] = [data]
            cubin_lens_dict[kspec.sm] = [size]

    metadata_v1 = []
    # Only metadata_v2 is used by TRT-LLM.
    metadata_v2 = []
    metadata_v2_dict = {}
    unroll_config_v1 = []
    unroll_config_v2 = []
    for kname, smem, threads, fname, unroll_step, unroll_threshold in kernel_traits:
        name = fname.replace('.', '_')
        cubin_name = 'cubin_{name}_cubin'.format(name=name)
        kname_remove_causal = kname.replace("_causal", "")
        tname = (kname.replace('flash_attention_', '').replace(
            '_scale_max', '').replace('_nl', '').replace('_tiled', '').replace(
                'tma_',
                '').replace('ldgsts_', '').replace('causal_', '').replace(
                    'alibi_', '').replace('softmax_', '').replace(
                        'sliding_or_chunked_', '').replace(
                            'custom_mask_', '').replace('qkv_', '').replace(
                                'q_kv_', '').replace('q_paged_kv_', '').replace(
                                    'q_k_v_', '').replace('ws_', '').replace(
                                        'softcapping_',
                                        '').replace('sage_',
                                                    '').replace('output_', ''))
        flash_attention = 'flash_attention' in kname
        warp_specialization = 'tma_ws' in kname
        toks = tname.split('_')
        #   0  1                x         -7  -6  -5-4-3   -2   -1   x
        #fmha_v2(_flash_attention)_fp16(_fp32)_64_64_S_16_sm80_kernel(_nl)
        #   0  1  2            -5 -4 -3  -2  -1
        #fmha_v2_il_fp16(_fp32)_64_64_sm80_kernel
        # print(kname)
        version = toks[1][1]
        sm = toks[-2][2:]
        if '_output' in kname:
            output_prec = toks[-3].upper()
            toks.pop(-3)
        else:
            output_prec = None
        if '_sage_' in kname:
            # example:
            # kname: fmha_v2_flash_attention_e4m3_64_256_S_qkv_128_sage_64_64_128_bf16_tma_ws_sm90_kernel
            # tname: fmha_v2_e4m3_64_256_S_128_sage_64_64_128_bf16_sm90_kernel
            sage_block_sizes = toks[-5:-2]
            toks.pop(-5)
            toks.pop(-4)
            toks.pop(-3)
        else:
            sage_block_sizes = (0, 0, 0)
        head_size = toks[-3]
        if 'x' in head_size:
            (head_size, head_size_v) = head_size.split('x')
        else:
            head_size_v = head_size
        # flash attention kernel encodes variable seqlen as S, but only number 0 fits in the metadata struct
        seq_len = 0 if toks[-4] == 'S' else toks[-4]
        q_step = unroll_step
        kv_step = seq_len
        if flash_attention:
            kv_step = toks[-5]
            q_step = toks[-6]
        prec = toks[-5].upper()
        is_fp32_accu = 'false'
        if flash_attention:
            prec = toks[-7].upper()
            # fp16_fp32 --> HMMA with FP32 accumulation
            if toks[-8].upper() in ['E4M3', 'E5M2', 'FP16', 'BF16']:
                if prec == 'FP32':
                    is_fp32_accu = 'true'
                prec = toks[-8].upper()

        elif toks[-6].upper() in ['E4M3', 'E5M2', 'FP16', 'BF16', 'FP32']:
            # in this case, toks[-6] = data type, toks[-5] = acc data type
            prec = toks[-5].upper()
            # fp16_fp32 --> HMMA with FP32 accumulation
            if toks[-6].upper() in ['E4M3', 'E5M2', 'FP16', 'BF16']:
                if prec == 'FP32':
                    is_fp32_accu = 'true'
                prec = toks[-6].upper()

        # fp8 or bf16 always accumulates on fp32
        if prec in ['E4M3', 'E5M2', 'BF16']:
            is_fp32_accu = 'true'
        if output_prec is None:
            output_prec = prec

        is_il = pythonBoolean2cpp['_il' in kname]
        attention_mask_type = AttentionMaskType.PADDING
        is_tiled = pythonBoolean2cpp['_tiled' in kname]

        # Attention mask type:
        # padding (0), causal_mask (1), sliding_or_chunked_causal_mask (2), custom_mask (3).
        if '_custom_mask' in kname:
            attention_mask_type = AttentionMaskType.CUSTOM_MASK
        elif '_sliding_or_chunked_causal' in kname:
            attention_mask_type = AttentionMaskType.SLIDING_OR_CHUNKED_CAUSAL
        elif '_causal' in kname:
            attention_mask_type = AttentionMaskType.CAUSAL

        attention_mask_type_value = attention_mask_type.value

        # Attention input layout:
        # packed_qkv (0), contiguous_q_kv (1), q_paged_kv (2), separate_q_k_v (3).
        attention_input_layout = InputLayout.PACKED_QKV
        if '_q_kv' in kname:
            attention_input_layout = InputLayout.CONTIGUOUS_Q_KV
        elif '_q_paged_kv' in kname:
            attention_input_layout = InputLayout.Q_PAGED_KV
        elif '_q_k_v' in kname:
            attention_input_layout = InputLayout.SEPARATE_Q_K_V

        attention_input_layout_value = attention_input_layout.value

        # hopper warpspecialized kernels have specialized ones for cases without alibi.
        is_alibi_supported = pythonBoolean2cpp['_ws' not in kname
                                               or '_alibi' in kname]

        return_softmax_stats_flag = pythonBoolean2cpp[sm != '90' or (
            sm == '90' and '_softmax' in kname)]

        # meta_unroll_step
        meta_unroll_step = unroll_step if ('_nl' in kname
                                           or '_ws' in kname) else '0'

        is_flash_atten = pythonBoolean2cpp[flash_attention]

        is_warp_specialization = pythonBoolean2cpp[warp_specialization]

        has_softcapping_scale = 'true' if 'softcapping' in kname else 'false'

        unroll_spec = '''\
{{ kSM_{sm}, DATA_TYPE_{prec}, {seq_len}, {head_size}, {unroll_threshold} }}\
'''.format(**locals())

        if 'v1' in kname:
            code = '''\
{{ DATA_TYPE_{prec}, {seq_len}, {head_size}, kSM_{sm},  {cubin_name}, {cubin_name}_len, \"{kname}\", {smem}, {threads} }}\
'''.format(**locals())
            metadata_v1.append(code)
            if '_nl' in kname:
                unroll_config_v1.append(unroll_spec)
        elif 'v2' in kname:
            if generate_cu_trtllm:

                def get_lname_from_kname(kname: str) -> str:
                    if use_cubin_header(int(sm), int(head_size), prec.lower()):
                        return 'nullptr'
                    lname = kname.replace('_kernel', '')
                    mask_types = [
                        '_sliding_or_chunked_causal', '_custom_mask', '_causal'
                    ]
                    for mask_type in mask_types:
                        lname = lname.replace(mask_type, '')
                    lname = 'run_' + lname

                    return lname

                lname = get_lname_from_kname(kname)
                code = '''\
{{ DATA_TYPE_{prec}, DATA_TYPE_{output_prec}, {seq_len}, {q_step}, {kv_step}, {head_size}, {head_size_v}, \
{sage_block_sizes[0]}, {sage_block_sizes[1]}, {sage_block_sizes[2]}, kSM_{sm}, {cubin_name}, \
{cubin_name}_len, \"{kname}\", {smem}, {threads}, {meta_unroll_step}, {attention_mask_type_value}, \
{attention_input_layout_value}, {is_il}, {is_flash_atten}, {is_warp_specialization}, {is_fp32_accu}, \
{is_alibi_supported}, {is_tiled}, {has_softcapping_scale}, {return_softmax_stats_flag}, {lname}}}\
'''.format(**locals()) if use_cubin_header(int(sm), int(head_size),
                                           prec.lower()) else '''\
{{ DATA_TYPE_{prec}, DATA_TYPE_{output_prec}, {seq_len}, {q_step}, {kv_step}, {head_size}, {head_size_v}, \
{sage_block_sizes[0]}, {sage_block_sizes[1]}, {sage_block_sizes[2]}, kSM_{sm}, nullptr, \
0, \"{kname}\", {smem}, {threads}, {meta_unroll_step}, {attention_mask_type_value}, \
{attention_input_layout_value}, {is_il}, {is_flash_atten}, {is_warp_specialization}, {is_fp32_accu}, \
{is_alibi_supported}, {is_tiled}, {has_softcapping_scale}, {return_softmax_stats_flag}, {lname}}}\
'''.format(**locals())
            else:
                code = '''\
{{ DATA_TYPE_{prec}, DATA_TYPE_{output_prec}, {seq_len}, {q_step}, {kv_step}, {head_size}, {head_size_v}, \
{sage_block_sizes[0]}, {sage_block_sizes[1]}, {sage_block_sizes[2]}, kSM_{sm}, {cubin_name}, \
{cubin_name}_len, \"{kname}\", {smem}, {threads}, {meta_unroll_step}, {attention_mask_type_value}, \
{attention_input_layout_value}, {is_il}, {is_flash_atten}, {is_warp_specialization}, {is_fp32_accu}, \
{is_alibi_supported}, {is_tiled}, {has_softcapping_scale}, {return_softmax_stats_flag}}}\
'''.format(**locals())
            if sm in metadata_v2_dict:
                metadata_v2_dict[sm].append(code)
            else:
                metadata_v2_dict[sm] = [code]
            if '_nl' in kname:
                unroll_config_v2.append(unroll_spec)
            if generate_cu_trtllm and lname != 'nullptr':
                launcher = 'extern void {lname}(Fused_multihead_attention_params_v2& params, const Launch_params& launch_params, cudaStream_t stream);'.format(
                    lname=lname)
                if int(sm) in cubins_dict:
                    if launcher not in cubins_dict[int(sm)]:
                        cubins_dict[int(sm)].append(launcher)
                else:
                    cubins_dict[int(sm)] = [launcher]
        elif 'mhca' in kname:
            code = '''\
{{ DATA_TYPE_{prec}, {seq_len}, {q_step}, {kv_step}, {head_size}, kSM_{sm},  {cubin_name}, {cubin_name}_len, \"{kname}\", {smem}, {threads}, {meta_unroll_step}, {is_il} }}\
'''.format(**locals())
            metadata_v2.append(code)
        else:
            assert False, 'Something terrible happened'

    metadata_v1 = ',\n'.join(metadata_v1)
    # Add macros to only include needed cubins during compilation.
    if bool(metadata_v2_dict):
        metadata_v2 = ''
        for sm in metadata_v2_dict.keys():
            macro_begin = f"#ifndef EXCLUDE_SM_{sm}"
            macro_end = f"#endif\n\n"
            metadata_v2 += macro_begin + '\n' + (',\n'.join(
                metadata_v2_dict[sm]))
            last_key = list(metadata_v2_dict.keys())[-1]
            metadata_v2 += ('' if sm == last_key else ',') + '\n' + macro_end
    else:
        metadata_v2 = ',\n'.join(metadata_v2)
    # Add macros to only include needed cubins during compilation.
    for sm in cubins_dict.keys():
        macro_begin = f"#ifndef EXCLUDE_SM_{sm}"
        macro_end = f"#endif\n"
        cubins.extend([macro_begin] + cubins_dict[sm] + [macro_end])
        if sm in cubin_lens_dict:
            cubin_lens.extend([macro_begin] + cubin_lens_dict[sm] + [macro_end])

    unroll_config_v1 = ',\n'.join(unroll_config_v1)
    unroll_config_v2 = ',\n'.join(unroll_config_v2)
    cubins = '\n'.join(cubins)
    cubin_lens = '\n'.join(cubin_lens)
    local_ns_open = ns_open
    local_ns_close = ns_close if generate_cu_trtllm else '}'
    launcher_line = '''
    void (*launcher)(Fused_multihead_attention_params_v2& params, const Launch_params& launch_params, cudaStream_t stream);''' if generate_cu_trtllm else ''
    if "GENERATE_CUBIN" in os.environ:
        code = '''\
{copyright}
#pragma once

{local_ns_open}

{cubins}

{cubin_lens}

static const struct FusedMultiHeadAttentionKernelMetaInfoV2
{{
    Data_type mDataTypeIn;
    Data_type mDataTypeOut;
    unsigned int mS;
    unsigned int mStepQ;
    unsigned int mStepKV;
    unsigned int mD;
    unsigned int mDV;
    unsigned int mSageBlockSizeQ;
    unsigned int mSageBlockSizeK;
    unsigned int mSageBlockSizeV;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    int mAttentionMaskType;
    int mAttentionInputLayout;
    bool mInterleaved;
    bool mFlashAttention;
    bool mWarpSpecialization;
    bool mFP32Accumulation;
    bool mAlibiSupported;
    bool mTiled;
    bool mEnableAttnLogitSoftcapping;
    bool mReturnSoftmaxStats;{launcher_line}
}} sMhaKernelMetaInfosV2[] = {{
{metadata_v2}
}};
{local_ns_close}

'''.format(**locals(), copyright=copyright)

    else:
        code = '''\
{copyright}
#pragma once

{cubins}

{cubin_lens}

static const struct TestMetaV1
{{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
}} metaV1[] = {{
{metadata_v1}
}};

static const struct TestMetaV2
{{
    Data_type mDataTypeIn;
    Data_type mDataTypeOut;
    unsigned int mS;
    unsigned int mStepQ;
    unsigned int mStepKV;
    unsigned int mD;
    unsigned int mDV;
    unsigned int mSageBlockSizeQ;
    unsigned int mSageBlockSizeK;
    unsigned int mSageBlockSizeV;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    int mAttentionMaskType;
    int mAttentionInputLayout;
    bool mInterleaved;
    bool mFlashAttention;
    bool mWarpSpecialization;
    bool mFP32Accumulation;
    bool mAlibiSupported;
    bool mTiled;
    bool mEnableAttnLogitSoftcapping;
    bool mReturnSoftmaxStats;
}} metaV2[] = {{
{metadata_v2}
}};
}}

'''.format(**locals(), copyright=copyright)

    return code


# This is used to add some kernels running in cubins for passing CI cases.
def modify_cubin_header(cubin_header):
    result = cubin_header

    # for CI cases
    def add_kernel_line(result, target, addition):
        pos = result.find(target)
        if pos != -1:
            end_pos = result.find('\n', pos)
            if end_pos == -1:
                end_pos = len(result)
            result = result[:end_pos + 1] + addition + result[end_pos:]
        return result

    target = "#ifndef EXCLUDE_SM_80"
    addition = """extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_cu_cubin[];
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_cu_cubin_len;"""
    result = add_kernel_line(result, target, addition)

    def modify_kernel_line(result, target, new_line):
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if target in line:
                lines[i] = new_line
                break
        return '\n'.join(lines)

    target = "fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_causal_sm80_kernel_nl_tiled"
    new_line = '{ DATA_TYPE_FP16, DATA_TYPE_FP16, 0, 64, 128, 128, 128, 0, 0, 0, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_cu_cubin, cubin_fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_sm80_cu_cubin_len, "fmha_v2_flash_attention_fp16_64_128_S_q_paged_kv_128_causal_sm80_kernel_nl_tiled", 81920, 128, 64, 1, 2, false, true, false, false, true, true, false, true, nullptr},'
    result = modify_kernel_line(result, target, new_line)

    # make sure only one empty line at the end
    lines = result.split('\n')
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append('')

    return '\n'.join(lines)


def generate_files(specs_names):

    kfiles = []
    valid_specs_names = []

    for kspec, fname, lname, kname in specs_names:
        code = get_kernel_code(kspec, kname, lname)
        # some kernels are skipped when generating cubins for trt-llm.
        if code is None:
            continue
        # add valid specs names
        valid_specs_names.append((kspec, fname, lname, kname))
        path = os.path.join('./generated', fname)
        # HACK: do not overwrite kernel file in case of collision; kernel selection logic can still be flaky
        # TODO: allow profiling multiple kernel implementations satisfying the given problem size
        if path not in kfiles:
            with open(path, 'w') as f:
                f.write(code)
        kfiles.append(path)

    api_code = get_api_code(valid_specs_names).replace(
        '__guard_fmhca_placeholder__', 'false')
    with open('./generated/fused_multihead_attention_api.h', 'w') as f:
        f.write(api_code)

    api_code = get_api_code(valid_specs_names).replace(
        '__guard_fmhca_placeholder__', 'true')
    with open('./generated/fused_multihead_cross_attention_api.h', 'w') as f:
        f.write(api_code)

    mk_code = get_makefile_code(valid_specs_names)

    with open('./generated/makefile', 'w') as f:
        f.write(mk_code)

    print_kernel_traits_code = get_kernel_traits_code(valid_specs_names)
    with open('./generated/print_kernel_traits.cu', 'w') as f:
        f.write(print_kernel_traits_code)

    # Make sure we have a bin directory.
    if not os.path.exists('bin'):
        os.mkdir('bin')
    cmd = 'nvcc -I src -Xcompiler -Wno-enum-compare --std=c++17 -o bin/print_traits.exe generated/print_kernel_traits.cu'.split(
    )
    if 'CUDA_PATH' in os.environ:
        cmd[0] = os.environ['CUDA_PATH'] + '/bin/' + cmd[0]
    print('Running command "{}" to build "bin/print_traits.exe":'.format(
        ' '.join(cmd)))
    process = subprocess.Popen(cmd,
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    print('Running "bin/print_traits.exe":')
    process = subprocess.Popen('bin/print_traits.exe',
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('utf-8').strip()
    # this gives: kname, smem bytes, threads_per_cta, loop_step
    kernel_traits = [traits.split() for traits in output.splitlines()]
    cubin_header = get_cubin_header(kernel_traits, valid_specs_names)
    if generate_cu_trtllm:
        cubin_header = modify_cubin_header(cubin_header)

    with open('./generated/fmha_cubin.h', 'w') as f:
        f.write(cubin_header)


def enumerate_hgmma_tma_kernels(specs, sm=90):
    specs.append(
        kernel_spec(
            sm=sm,
            sm_mma=90,
            dtype='fp16',
            seq_len=[64, 128, 256],
            head_size=64,
            warps_m=4,  #4x1 warpgroups
            warps_n=1,
            version=2,
            interleaved=False,
            ldgsts_q=False,
            ldgsts_k=False,
            ldgsts_v=False,
            share_smem_k_v=False,
            loop_step=64,
            has_noloop=0,
            noloop_step=64,
            unroll_threshold=1,
            has_scale_max=False))


# Note this will be used in TRT-LLM.
def enumerate_hgmma_ldgsts_kernels(specs, sm=90, dtype='fp16'):

    for enable_attn_logit_softcapping in [False, True]:
        specs.append(
            kernel_spec(
                sm=sm,
                sm_mma=90,
                dtype=dtype,
                seq_len=[64, 128, 256],
                head_size=[32, 64],
                warps_m=4,  #4x1 warpgroups
                warps_n=1,
                version=2,
                interleaved=False,
                ldgsts_q=
                True,  # for Hopper kernels, ldgsts = False signals TMA usage.
                ldgsts_k=True,
                ldgsts_v=True,
                share_smem_k_v=False,
                loop_step=64,
                has_noloop=1,
                noloop_step=64,
                unroll_threshold=1,
                has_scale_max=False,
                enable_attn_logit_softcapping=enable_attn_logit_softcapping))

        specs.append(
            kernel_spec(
                sm=sm,
                sm_mma=90,
                dtype=dtype,
                seq_len=[384, 512],
                head_size=[32, 64],
                warps_m=4,  #4x1 warpgroups
                warps_n=2,
                version=2,
                interleaved=False,
                ldgsts_q=
                True,  # for Hopper kernels, ldgsts = False signals TMA usage.
                ldgsts_k=True,
                ldgsts_v=True,
                share_smem_k_v=False,
                loop_step=64,
                has_noloop=1,
                noloop_step=64,
                unroll_threshold=1,
                has_scale_max=False,
                enable_attn_logit_softcapping=enable_attn_logit_softcapping))


# Note this will be used in TRT-LLM.
def enumerate_hgmma_flash_warpspec_kernels(specs, sm=90, dtype='fp16'):

    scheduling_mode = int(os.getenv('SCHEDULING_MODE', '1'))

    # use specialized kernels for cases without alibi scales.
    # there is a numeric issues when applying the exp2f scale optimization and alibi scale at the same time.
    combinations = product([False, True], [False, True], \
                           [InputLayout.PACKED_QKV, InputLayout.CONTIGUOUS_Q_KV,
                            InputLayout.Q_PAGED_KV, InputLayout.SEPARATE_Q_K_V], [False, True])
    for (alibi, return_softmax, input_layout,
         enable_attn_logit_softcapping) in combinations:
        # alibi and enable_attn_logit_softcapping shouldn't be used together.
        if alibi and enable_attn_logit_softcapping:
            continue
        # for normal attention, we only need contiguous kv as input layout when returning softmax.
        skip_combination = return_softmax and input_layout != InputLayout.CONTIGUOUS_Q_KV
        # for context mla, we need separate qkv as input layout when returning softmax.
        skip_mla_combination = return_softmax and input_layout != InputLayout.SEPARATE_Q_K_V
        if not skip_combination:
            # only specify
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=[32, 40, 48, 64],
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=256,
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout))

            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=[72, 80, 96, 104, 128],
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=128,
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout))

            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=[160, 192, 256],
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=64,
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout))
        '''
        smem size = (q_step * d * q_buffers * NUM_COMPUTE_GROUPS
                    + (kv_step * d + kv_step * dv) * kv_buffers) * ele_size
        Originally, head size is padded to next_power_of_2<d> and next_power_of_2<dv>.
        For fp16/bf16 context MLA (d=192/dv=128), d is padded to 256, and dv remains 128,
            if kv_step=64, then smem_size = 160 KB, it is OK but wastes much smem.
            if kv_step=128, then smem_size = 256 KB, it is too big for Hopper (228KB smem per SM).
        But in fact, 'next multiply of 128 bytes' is needed only, due to TMA 128B swizzle mode.
        Then for fp16/bf16 context MLA, d remains 192 (192 * 2 = 128 * 3), and dv remains 128,
            if kv_step = 128, then smem_size = 208 KB, smem is fully utilized.
        '''
        if not skip_mla_combination:
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=192,
                    head_size_v=128,
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=128,
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout))


# Note this will be used in TRT-LLM.
def enumerate_qgmma_flash_warpspec_kernels(specs,
                                           sm=90,
                                           dtype='e4m3',
                                           sage_block_sizes=None,
                                           output_dtype=None):

    scheduling_mode = int(os.getenv('SCHEDULING_MODE', '1'))

    # use specialized kernels for cases without alibi scales.
    # there is a numeric issues when applying the exp2f scale optimization and alibi scale at the same time.
    combinations = product([False, True], \
        [InputLayout.PACKED_QKV, InputLayout.CONTIGUOUS_Q_KV,
         InputLayout.Q_PAGED_KV, InputLayout.SEPARATE_Q_K_V],
        [False, True], [False, True])
    for (alibi, input_layout, enable_attn_logit_softcapping,
         return_softmax) in combinations:
        # alibi and bmm1_tanh_scale shouldn't be used together.
        if alibi and enable_attn_logit_softcapping:
            continue
        # for normal attention, we do not need return softmax for ws fp8 kernels currently.
        # also fp8 input and bf16 output is only needed for MLA kernel.
        skip_combination = return_softmax or (output_dtype is not None)
        # for context mla, we need separate qkv as input layout when returning softmax.
        skip_mla_combination = return_softmax and input_layout != InputLayout.SEPARATE_Q_K_V
        if not skip_combination:
            # D <= 64: KV_STEP = 256
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=[32, 40, 48, 64],
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=256,
                    kv_tile_buffers=4,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout,
                    sage_block_sizes=sage_block_sizes,
                    output_dtype=output_dtype))

            # 64 < D <=128: KV_STEP = 128
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=[80, 96, 104, 128],
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=256,
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout,
                    sage_block_sizes=sage_block_sizes,
                    output_dtype=output_dtype))

            # 128 < D <=256: KV_STEP = 128
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=[160, 192, 256],
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=
                    128,  # use 128 kv step size to avoid register spilling
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout,
                    sage_block_sizes=sage_block_sizes,
                    output_dtype=output_dtype))

        if not skip_mla_combination:
            # context MLA (192x128)
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=90,
                    dtype=dtype,
                    seq_len=0,  # support any sequence length
                    head_size=192,
                    head_size_v=128,
                    warps_m=4,  #4x1 warpgroups
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=
                    False,  # for Hopper kernels, ldgsts = False signals TMA usage.
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    q_tile_buffers=1,  # only used by warp specialized kernels
                    has_noloop=0,
                    noloop_step=64,
                    kv_loop_step=128,
                    kv_tile_buffers=2,  # only used by warp specialized kernels
                    unroll_threshold=1,
                    has_scale_max=False,
                    flash_attention=True,
                    warp_specialization=True,
                    alibi=alibi,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    return_softmax_stats=return_softmax,
                    scheduling_mode=scheduling_mode,
                    input_layout=input_layout,
                    sage_block_sizes=sage_block_sizes,
                    output_dtype=output_dtype))


def enumerate_igmma_kernels(specs, sm=90):
    specs.append(
        kernel_spec(
            sm=sm,
            sm_mma=90,
            dtype='int8',
            seq_len=[64, 128, 256, 384],
            head_size=64,
            warps_m=4,  #4x1 warpgroups
            warps_n=1,
            version=2,
            interleaved=False,
            ldgsts_q=
            True,  # for Hopper kernels, ldgsts = False signals TMA usage.
            ldgsts_k=True,
            ldgsts_v=True,
            share_smem_k_v=False,
            loop_step=64,
            has_noloop=1,
            noloop_step=64,
            unroll_threshold=1,
            has_scale_max=False))

    specs.append(
        kernel_spec(
            sm=sm,
            sm_mma=90,
            dtype='int8',
            seq_len=[512],
            head_size=64,
            warps_m=4,  #4x2 warpgroups
            warps_n=2,
            version=2,
            interleaved=False,
            ldgsts_q=
            True,  # for Hopper kernels, ldgsts = False signals TMA usage.
            ldgsts_k=True,
            ldgsts_v=True,
            share_smem_k_v=False,
            loop_step=64,
            has_noloop=1,
            noloop_step=64,
            unroll_threshold=1,
            has_scale_max=False))


def enumerate_hmma_kernels(specs, sm=80, dtype='fp16'):
    # The following kernels are hmma-based kernels tuned for sm90
    if sm == 90:
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=80,
                        dtype=dtype,
                        seq_len=[64, 128, 256],
                        head_size=[64, 72],
                        warps_m=1,
                        warps_n=4,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=80,
                        dtype=dtype,
                        seq_len=[384, 512],
                        head_size=[64, 72],
                        warps_m=1,
                        warps_n=8,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=86,
                    dtype=dtype,
                    seq_len=384,
                    head_size=64,
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=64,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=86,
                    dtype='fp16',
                    seq_len=384,
                    head_size=64,
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=64,
                    unroll_threshold=1,
                    has_scale_max=False))

    #  S=1024 split over 4 CTAs.
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=256,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=0,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False,
                    ctas_per_head=4))

    #- S=512: STEP=32, STEP NL=-- FLAGS=0x9 (0x9 for SM86!)
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=512,
                    head_size=64,
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype=dtype,
                    seq_len=512,
                    head_size=[16, 32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=512,
                    head_size=[16, 32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #- S=384: STEP=48, STEP NL=-- FLAGS=0x9 (0x9 for SM86!)
    #  TODO warps_n=4 leads to 2 pred regs, which is not supported
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=384,
                    head_size=64,
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=48,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype=dtype,
                    seq_len=384,
                    head_size=64,
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=48,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=256: STEP=32, STEP NL=32 FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=256,
                    head_size=[16, 32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype=dtype,
                    seq_len=256,
                    head_size=[16, 32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    # #-  S=128: STEP=NA, STEP NL=32 FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=128,
                    head_size=[16, 32, 64],
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype=dtype,
                    seq_len=128,
                    head_size=[16, 32, 64],
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=96:  STEP=32, STEP NL=-- FLAGS=0x1 TODO noloop does not work - illegal memory access: we run LDSM.T x4 which is oob.
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=96,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=96,
                    has_noloop=0,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype=dtype,
                    seq_len=96,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=96,
                    has_noloop=0,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=64:  STEP=32, STEP NL=-- FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='fp16',
                    seq_len=64,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype=dtype,
                    seq_len=64,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    if sm == 75:
        #- FP16
        #- S=512: STEP=32, STEP NL=-- FLAGS=0x9 (0x9 for SM86!)
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=[384, 512],
                        head_size=[16, 32, 64],
                        warps_m=1,
                        warps_n=8,
                        version=1,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=[384, 512],
                        head_size=[16, 32, 64],
                        warps_m=1,
                        warps_n=8,
                        version=2,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=256,
                        head_size=[16, 32, 64],
                        warps_m=1,
                        warps_n=4,
                        version=1,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=256,
                        head_size=[16, 32, 64],
                        warps_m=1,
                        warps_n=4,
                        version=2,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=128,
                        head_size=[16, 32, 64],
                        warps_m=2,
                        warps_n=2,
                        version=1,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=128,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=128,
                        head_size=[16, 32, 64],
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=128,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=75,
                        dtype='fp16',
                        seq_len=64,
                        head_size=[16, 32, 64],
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=True,
                        loop_step=64,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

    #-  S=384: STEP=32, STEP NL=32 FLAGS=0x8
    #-  S=256: STEP=32, STEP NL=32 FLAGS=0x8
    #-  S=128: STEP=32, STEP NL=32 FLAGS=0x8
    #-  S=128: STEP=NA, STEP NL=32 FLAGS=0x8
    #-  S=96:  STEP=32, STEP NL=-- FLAGS=0x8
    #-  S=64:  STEP=32, STEP NL=-- FLAGS=0x8

    #SM 72
    #- Int8 (same for interleaved)
    #-  S=384: STEP=32, STEP NL=-- FLAGS=0x0
    #-  S=256: STEP=64, STEP NL=-- FLAGS=0x0
    #-  S=192: STEP=64, STEP NL=-- FLAGS=0x0
    #-  S=128: STEP=NA, STEP NL=-- FLAGS=0x8
    #-  S=96
    #-  S=64


def enumerate_hmma884_kernels(specs, sm=70):
    #- FP16
    #- S=512: STEP=32, STEP NL=-- FLAGS=0x9 (0x9 for SM86!)
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=[384, 512],
                    head_size=[64],
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=16,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=[384, 512],
                    head_size=[64],
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=16,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=[384, 512],
                    head_size=[32],
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=[384, 512],
                    head_size=[32],
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=256: STEP=32, STEP NL=32 FLAGS=0x8
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=[128, 256],
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=[128, 256],
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    # SEQLEN 96
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=96,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    # SEQLEN 64
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=64,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    # SEQLEN 32
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=70,
                    dtype='fp16',
                    seq_len=32,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))


def enumerate_hmma_paged_kv_flash_kernels(specs, sm=80, dtype='fp16'):
    for enable_attn_logit_softcapping in [False, True]:
        enumerate_hmma_flash_kernels_base(specs, sm, dtype,
                                          InputLayout.PACKED_QKV,
                                          enable_attn_logit_softcapping)


def enumerate_hmma_flash_kernels(specs, sm=80, dtype='fp16', head_size_v=0):
    input_layouts = [
        InputLayout.PACKED_QKV, InputLayout.CONTIGUOUS_Q_KV,
        InputLayout.Q_PAGED_KV
    ]
    # Deepseek MLA (context 192/128 separate-q-k-v)
    if head_size_v == 128:
        input_layouts.append(InputLayout.SEPARATE_Q_K_V)
    for (input_layout,
         enable_attn_logit_softcapping) in product(input_layouts,
                                                   [False, True]):
        enumerate_hmma_flash_kernels_base(specs, sm, dtype, input_layout,
                                          enable_attn_logit_softcapping,
                                          head_size_v)


# Note this will be used in TRT-LLM.
def enumerate_hmma_flash_kernels_base(specs,
                                      sm=80,
                                      dtype='fp16',
                                      input_layout=InputLayout.PACKED_QKV,
                                      enable_attn_logit_softcapping=False,
                                      head_size_v=0):
    #- FP16 Flash Attention (use nl as default)
    # Any Sequence Length H = 16/32/40/48/64/80/128/160/256/512 flash attention

    # Note: sm70, sm72 are based on hmma8x8x4, while sm75+ is based on hmma16x8x16
    # sm75 and sm80+ use the same underlying trait class; but for historical reasons we prefer not
    # to change the appearance of the trait class. So:
    #  - Volta uses Volta_hmma_fp16_traits
    #  - Turing uses Turing_hmma_fp16_traits
    #  - Ampere uses Ampere_hmma_fp16_traits but is effectively an alias of Turing_hmma_fp16_traits
    #  - Ada and Hopper use Ampere_hmma_fp16_traits
    sm_mma = 0
    if sm in [70, 72]:
        sm_mma = 70
    elif sm in [75]:
        sm_mma = 75
    elif sm in [80, 86, 87, 89, 90, 100, 120]:
        sm_mma = 80

    # _nl_tiled kernels; higher precedence than _nl kernels
    # params[head_size] = [q_step, kv_step]
    tiled_params_q_kv_step = {
        16: [128, 128],
        32: [128, 128],
        40: [128, 128],
        48: [128, 128],
        64: [128, 128],
        72: [64, 128],
        80: [64, 128],
        96: [64, 128],
        104: [64, 128],
        128: [64, 128],
        160: [64, 128],
        192: [64, 128],
        256: [64, 128],
        512: [64, 64],
        576: [64, 64]
    }
    for head_size, [q_loop_step,
                    kv_loop_step] in tiled_params_q_kv_step.items():
        if sm_mma == 80:
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=sm_mma,
                    dtype=dtype,
                    flash_attention=True,
                    tiled=1,
                    seq_len=0,  # means any sequence here
                    kv_loop_step=kv_loop_step,
                    limit_qk_fragments=False,
                    limit_v_fragments=False,
                    head_size=head_size,
                    head_size_v=head_size_v,
                    warps_m=4,
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=q_loop_step,
                    has_noloop=1,
                    noloop_step=q_loop_step,
                    unroll_threshold=1,
                    has_scale_max=False,
                    ctas_per_head=1,
                    input_layout=input_layout,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping,
                    is_mtp=(head_size == 576 and head_size_v == 512)))

    for head_size in [
            16, 32, 40, 48, 64, 72, 80, 96, 104, 128, 160, 192, 256, 512
    ]:
        if sm == 70 and (head_size > 256 or head_size == 16):
            continue
        # TODO: test head_size=512 on sm75
        if sm == 75 and head_size > 256:
            continue

        # tune ldgsts
        ldgsts_q = True
        ldgsts_k = True
        ldgsts_v = True
        if head_size >= 256:
            ldgsts_k = False
            ldgsts_v = False
        if head_size > 256:
            ldgsts_q = False
        if sm < 80:
            ldgsts_q = False
            ldgsts_k = False
            ldgsts_v = False

        # tune kv fragment double buffer
        limit_qk_fragments = False
        limit_v_fragments = False
        if head_size >= 256:
            limit_qk_fragments = True
            limit_v_fragments = True
        elif head_size >= 128 and sm == 70:
            limit_qk_fragments = True
            limit_v_fragments = True

        # tune kv_loop step
        q_loop_step = 64
        kv_loop_step = 64
        if head_size > 128:
            kv_loop_step = 16
        elif (head_size > 64 and sm == 70):
            kv_loop_step = 16
        elif head_size > 32:
            kv_loop_step = 32

        if sm < 80 or head_size > 128:
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=sm_mma,
                    dtype=dtype,
                    flash_attention=True,
                    seq_len=0,  # means any sequence here
                    kv_loop_step=kv_loop_step,
                    limit_qk_fragments=limit_qk_fragments,
                    limit_v_fragments=limit_v_fragments,
                    head_size=head_size,
                    head_size_v=head_size_v,
                    warps_m=4,
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=ldgsts_q,
                    ldgsts_k=ldgsts_k,
                    ldgsts_v=ldgsts_v,
                    share_smem_k_v=False,
                    loop_step=q_loop_step,
                    has_noloop=1,
                    noloop_step=q_loop_step,
                    unroll_threshold=1,
                    has_scale_max=False,
                    ctas_per_head=1,
                    input_layout=input_layout,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping)
            )
        elif head_size <= 128:
            # q_step = 64, kv_step = 32
            specs.append(
                kernel_spec(
                    sm=sm,
                    sm_mma=sm_mma,
                    dtype=dtype,
                    flash_attention=True,
                    seq_len=0,  # means any sequence here
                    kv_loop_step=kv_loop_step,
                    limit_qk_fragments=limit_qk_fragments,
                    limit_v_fragments=limit_v_fragments,
                    head_size=head_size,
                    head_size_v=head_size_v,
                    warps_m=4,
                    warps_n=1,
                    version=2,
                    interleaved=False,
                    ldgsts_q=ldgsts_q,
                    ldgsts_k=ldgsts_k,
                    ldgsts_v=ldgsts_v,
                    share_smem_k_v=False,
                    loop_step=q_loop_step,
                    has_noloop=1,
                    noloop_step=q_loop_step,
                    unroll_threshold=1,
                    has_scale_max=False,
                    ctas_per_head=1,
                    input_layout=input_layout,
                    enable_attn_logit_softcapping=enable_attn_logit_softcapping)
            )


def enumerate_qgmma_kernels(specs, sm=90):
    specs.append(
        kernel_spec(
            sm=sm,
            sm_mma=90,
            dtype='e4m3',
            seq_len=[64, 128, 192, 256, 384],
            head_size=64,
            warps_m=4,  #4x1 warpgroups
            warps_n=1,
            version=2,
            interleaved=False,
            ldgsts_q=
            True,  # for Hopper kernels, ldgsts = False signals TMA usage.
            ldgsts_k=True,
            ldgsts_v=True,
            share_smem_k_v=False,
            loop_step=64,
            has_noloop=1,
            noloop_step=64,
            unroll_threshold=1,
            has_scale_max=False))

    specs.append(
        kernel_spec(
            sm=sm,
            sm_mma=90,
            dtype='e4m3',
            seq_len=[512],
            head_size=64,
            warps_m=4,  #4x2 warpgroups
            warps_n=2,
            version=2,
            interleaved=False,
            ldgsts_q=
            True,  # for Hopper kernels, ldgsts = False signals TMA usage.
            ldgsts_k=True,
            ldgsts_v=True,
            share_smem_k_v=False,
            loop_step=64,
            has_noloop=1,
            noloop_step=64,
            unroll_threshold=1,
            has_scale_max=False))


def enumerate_qmma_kernels(specs, sm=89):
    # SM89 (Ada) fp8
    # Head Size 64

    # generate fp16 acc first
    # NOTE: generate only one acc type if it is used for cubin loading
    #       or modify the TestMetaV2 to have acc_type
    for dtype in ['e4m3_fp16', 'e4m3_fp32']:
        # SEQ 64
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=89,
                        dtype=dtype,
                        seq_len=64,
                        head_size=64,
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=64,
                        has_noloop=0,
                        noloop_step=16,
                        unroll_threshold=1,
                        has_scale_max=False))

        # SEQ 96
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=89,
                        dtype=dtype,
                        seq_len=96,
                        head_size=64,
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=96,
                        has_noloop=1,
                        noloop_step=16,
                        unroll_threshold=1,
                        has_scale_max=False))

        # SEQ 128
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=89,
                        dtype=dtype,
                        seq_len=128,
                        head_size=64,
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=128,
                        has_noloop=1,
                        noloop_step=16,
                        unroll_threshold=1,
                        has_scale_max=False))

        # SEQ 192/256/384
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=89,
                        dtype=dtype,
                        seq_len=[192, 256, 384],
                        head_size=64,
                        warps_m=1,
                        warps_n=4,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        # SEQ 512
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=89,
                        dtype=dtype,
                        seq_len=512,
                        head_size=64,
                        warps_m=1,
                        warps_n=8,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))


def enumerate_qmma_flash_kernels(specs,
                                 sm=89,
                                 dtype='e4m3_fp32',
                                 head_sizes=None,
                                 sage_block_sizes=None,
                                 output_dtype=None):
    # ((head_size, head_size_v), (q_loop_step, kv_loop_step), tiled).
    params_q_kv_step = [
        (32, (128, 128), 0),
        (40, (128, 128), 0),
        (48, (128, 128), 0),
        (64, (128, 128), 0),
        (72, (64, 32), 0),
        (80, (64, 32), 0),
        (96, (64, 32), 0),
        (104, (64, 32), 0),
        (128, (64, 32), 0),
        (160, (64, 32), 0),
        (192, (64, 32), 0),
        (256, (64, 32), 0),
        # MLA kernels.
        ((192, 128), (64, 64), 1),
        ((576, 512), (64, 64), 1),
    ]
    input_layouts = [
        InputLayout.PACKED_QKV, InputLayout.CONTIGUOUS_Q_KV,
        InputLayout.Q_PAGED_KV, InputLayout.SEPARATE_Q_K_V
    ]
    for (head_size_params, (q_loop_step, kv_loop_step), tiled), input_layout in \
            product(params_q_kv_step, input_layouts):
        # head_size_v = 0 means head_size_v is the same as head_size
        if isinstance(head_size_params, tuple):
            head_size = head_size_params[0]
            head_size_v = head_size_params[1]
        else:
            head_size = head_size_params
            head_size_v = 0
        # skip if head_size is not in head_sizes
        if head_sizes is not None and head_size not in head_sizes:
            continue
        # skip if head_size_v is not 128 for separate-q-k-v
        if input_layout == InputLayout.SEPARATE_Q_K_V and head_size_v != 128:
            continue
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=89,
                        dtype=dtype,
                        seq_len=0,
                        head_size=head_size,
                        head_size_v=head_size_v,
                        warps_m=4,
                        warps_n=1,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=True,
                        ldgsts_v=True,
                        share_smem_k_v=False,
                        loop_step=q_loop_step,
                        has_noloop=1,
                        noloop_step=q_loop_step,
                        kv_loop_step=kv_loop_step,
                        tiled=tiled,
                        unroll_threshold=1,
                        has_scale_max=False,
                        flash_attention=True,
                        limit_qk_fragments=False,
                        limit_v_fragments=False,
                        ctas_per_head=1,
                        input_layout=input_layout,
                        sage_block_sizes=sage_block_sizes,
                        output_dtype=output_dtype,
                        is_mtp=(head_size == 576 and head_size_v == 512)))


def enumerate_imma_kernels(specs, sm=80):
    if sm == 90:
        # The following kernels are imma-based kernels tuned for sm90
        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=80,
                        dtype='int8',
                        seq_len=[64, 128, 256],
                        head_size=64,
                        warps_m=1,
                        warps_n=4,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

        specs.append(
            kernel_spec(sm=sm,
                        sm_mma=80,
                        dtype='int8',
                        seq_len=[384, 512],
                        head_size=64,
                        warps_m=1,
                        warps_n=8,
                        version=2,
                        interleaved=False,
                        ldgsts_q=True,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=32,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False))

    # # SM 80 / 86
    # #- Int8 (same for interleaved)

    #-  S=1024 split over 4 CTAs.
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=256,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=0,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False,
                    ctas_per_head=4))

    #-  S=512: STEP=32, STEP NL=32 FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=512,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=512,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    # D=16: currently needs to run with Turing traits due to K=16 for BMM1.
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=512,
                    head_size=16,
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True if sm >= 80 else False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    # D=16: currently needs to run with Turing traits due to K=16 for BMM1.
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=512,
                    head_size=16,
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True if sm >= 80 else False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=384: STEP=32, STEP NL=32 FLAGS=0x1
    specs.append(
        kernel_spec(
            sm=sm,
            sm_mma=80,
            dtype='int8',
            seq_len=384,
            head_size=64,
            warps_m=1,
            warps_n=8,  # required by pred packing.
            version=1,
            interleaved=False,
            ldgsts_q=True,
            ldgsts_k=False,
            ldgsts_v=False,
            share_smem_k_v=True,
            loop_step=32,
            has_noloop=1,
            noloop_step=32,
            unroll_threshold=1,
            has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=[192, 256],
                    head_size=64,
                    warps_m=1,
                    warps_n=4,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=[192, 256, 384],
                    head_size=64,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=[192, 256, 384],
                    head_size=64,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=True,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=256: STEP=32, STEP NL=32 FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=256,
                    head_size=32,
                    warps_m=1,
                    warps_n=4,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=256,
                    head_size=32,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=256,
                    head_size=16,
                    warps_m=1,
                    warps_n=4,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True if sm >= 80 else False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=256,
                    head_size=16,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True if sm >= 80 else False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #  S=192: STEP=64, STEP NL=32 FLAGS=0x1
    #-  S=128: STEP=NA, STEP NL=16 FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=128,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=128,
                    head_size=16,
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True if sm >= 80 else False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=128,
                    head_size=[32, 64],
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=128,
                    head_size=16,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True if sm >= 80 else False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=128,
                    head_size=[32, 64],
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=True,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=96
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=96,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=96,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=96,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=96,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=96,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=True,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=96,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    # #-  S=64:
    # TODO noloop doesn't work - need to adjust packing into registers for
    # Mma_tile_p::MMAS_N == 1 => Mma_tile_o::MMAS_K == 1 (at least on SM8x)
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=64,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=0,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=64,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=0,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=80,
                    dtype='int8',
                    seq_len=64,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=True,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=0,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    # This config compiles IMMA 1x4 kernels for SM90
    #specs.append(kernel_spec(sm=90,
    #    sm_mma=80,
    #    dtype='int8',
    #    seq_len=[128,192,256, 384],
    #    head_size=64,
    #    warps_m=1,
    #    warps_n=4,
    #    version=2,
    #    interleaved=False,
    #    ldgsts_q=True,
    #    ldgsts_k=False,
    #    ldgsts_v=False,
    #    share_smem_k_v=False,
    #    loop_step=32,
    #    has_noloop=0,
    #    noloop_step=32,
    #    unroll_threshold=1,
    #    has_scale_max=False))

    #- Int8 (same for interleaved)
    #-  S=512: STEP=32, STEP NL=32 FLAGS=0x1
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=[384, 512],
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=1,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=16,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=256,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=1,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=512,
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=[192, 256, 384],
                    head_size=[32, 64],
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=16,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=384: STEP=32, STEP NL=32 FLAGS=0x0
    #-  S=256: STEP=32, STEP NL=32 FLAGS=0x0
    #-  S=128: STEP=32, STEP NL=32 FLAGS=0x0
    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=128,
                    head_size=[32, 64],
                    warps_m=2,
                    warps_n=2,
                    version=1,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    specs.append(
        kernel_spec(sm=sm,
                    sm_mma=75,
                    dtype='int8',
                    seq_len=128,
                    head_size=[32, 64],
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=128,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False))

    #-  S=192: STEP=64, STEP NL=64 FLAGS=0x0
    #-  S=128: STEP=NA, STEP NL=16 FLAGS=0x8
    #-  S=96
    #-  S=64


def enumerate_cross_mha_kernels(specs):
    # TODO: combine cross_mha and mha kernel enumeration
    #-  S_Q=4096, S_KV=128:  STEP=64, STEP NL=64
    # HEAD_SIZE: 64
    # SM 70
    if 'ENABLE_SM70' in os.environ:
        specs.append(
            kernel_spec(sm=70,
                        dtype='fp16',
                        seq_len=128,
                        head_size=64,
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=64,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False,
                        cross_mha=1))

    # SM 75
    specs.append(
        kernel_spec(sm=75,
                    dtype='fp16',
                    seq_len=128,
                    head_size=64,
                    warps_m=2,
                    warps_n=2,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 80
    specs.append(
        kernel_spec(sm=80,
                    dtype='fp16',
                    seq_len=128,
                    head_size=64,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=64,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 86
    specs.append(
        kernel_spec(sm=86,
                    dtype='fp16',
                    seq_len=128,
                    head_size=64,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=64,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 89
    specs.append(
        kernel_spec(sm=89,
                    dtype='fp16',
                    seq_len=128,
                    head_size=64,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=64,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    #-  S_Q=1024, S_KV=128:  STEP=64, STEP NL=32
    # HEAD_SIZE: 128
    # SM 70
    if 'ENABLE_SM70' in os.environ:
        specs.append(
            kernel_spec(sm=70,
                        dtype='fp16',
                        seq_len=128,
                        head_size=128,
                        warps_m=2,
                        warps_n=2,
                        version=2,
                        interleaved=False,
                        ldgsts_q=False,
                        ldgsts_k=False,
                        ldgsts_v=False,
                        share_smem_k_v=False,
                        loop_step=64,
                        has_noloop=1,
                        noloop_step=32,
                        unroll_threshold=1,
                        has_scale_max=False,
                        cross_mha=1))

    # SM 75
    specs.append(
        kernel_spec(sm=75,
                    dtype='fp16',
                    seq_len=128,
                    head_size=128,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=False,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 80
    specs.append(
        kernel_spec(sm=80,
                    dtype='fp16',
                    seq_len=128,
                    head_size=128,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 86
    specs.append(
        kernel_spec(sm=86,
                    dtype='fp16',
                    seq_len=128,
                    head_size=128,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=64,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 89
    specs.append(
        kernel_spec(sm=89,
                    dtype='fp16',
                    seq_len=128,
                    head_size=128,
                    warps_m=1,
                    warps_n=4,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=64,
                    has_noloop=1,
                    noloop_step=32,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    #-  S_KV=128:  STEP=32, STEP NL=32
    # HEAD_SIZE: 256
    # SM 70
    # specs.append(kernel_spec(sm=70,
    #     dtype='fp16',
    #     seq_len=128,
    #     head_size=256,
    #     warps_m=2,
    #     warps_n=2,
    #     version=2,
    #     interleaved=False,
    #     ldgsts_q=False,
    #     ldgsts_k=False,
    #     ldgsts_v=False,
    #     share_smem_k_v=True,
    #     loop_step= 32,
    #     has_noloop=1,
    #     noloop_step=16,
    #     unroll_threshold=1,
    #     has_scale_max=False,
    #     cross_mha=1))

    # # SM 75
    # specs.append(kernel_spec(sm=75,
    #     dtype='fp16',
    #     seq_len=128,
    #     head_size=256,
    #     warps_m=1,
    #     warps_n=8,
    #     version=2,
    #     interleaved=False,
    #     ldgsts_q=False,
    #     ldgsts_k=False,
    #     ldgsts_v=False,
    #     share_smem_k_v=True,
    #     loop_step= 32,
    #     has_noloop=1,
    #     noloop_step=16,
    #     unroll_threshold=1,
    #     has_scale_max=False,
    #     cross_mha=1))

    # SM 80
    specs.append(
        kernel_spec(sm=80,
                    dtype='fp16',
                    seq_len=128,
                    head_size=256,
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=True,
                    ldgsts_v=True,
                    share_smem_k_v=False,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 86
    specs.append(
        kernel_spec(sm=86,
                    dtype='fp16',
                    seq_len=128,
                    head_size=256,
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))

    # SM 89
    specs.append(
        kernel_spec(sm=89,
                    dtype='fp16',
                    seq_len=128,
                    head_size=256,
                    warps_m=1,
                    warps_n=8,
                    version=2,
                    interleaved=False,
                    ldgsts_q=True,
                    ldgsts_k=False,
                    ldgsts_v=False,
                    share_smem_k_v=True,
                    loop_step=32,
                    has_noloop=1,
                    noloop_step=16,
                    unroll_threshold=1,
                    has_scale_max=False,
                    cross_mha=1))


def enumerate_kernels():
    if not os.path.exists('./generated'):
        os.mkdir('./generated')

    specs = []

    # TODO we have to select the unroll_threshold over a grid of b and h for each arch

    # Current fp16 384 kernel does 1x8 (smem limit), STEP=48. FP16 does not currently have noloop.

    # SM 90
    enumerate_hgmma_tma_kernels(specs, sm=90)
    enumerate_hgmma_ldgsts_kernels(specs, sm=90, dtype='fp16')
    enumerate_hgmma_ldgsts_kernels(specs, sm=90, dtype='bf16')
    if 'ENABLE_HMMA_FP32' in os.environ:
        enumerate_hgmma_ldgsts_kernels(specs, sm=90, dtype='fp16_fp32')
    enumerate_igmma_kernels(specs, sm=90)
    enumerate_qgmma_kernels(specs, sm=90)
    # need to add bf16 kernels if needed
    enumerate_hgmma_flash_warpspec_kernels(specs, sm=90, dtype='fp16')
    enumerate_hgmma_flash_warpspec_kernels(specs, sm=90, dtype='bf16')
    enumerate_qgmma_flash_warpspec_kernels(specs, sm=90, dtype='e4m3')
    enumerate_qgmma_flash_warpspec_kernels(specs,
                                           sm=90,
                                           dtype='e4m3',
                                           output_dtype="bf16")

    # For now SageAttention only needs BF16
    # block_size_q should be divisible by 64
    # block_size_k should be divisible by 8
    # block_size_v should be divisible by 32
    for sage_block_sizes in [(64, 64, 64), (64, 64, 128), (64, 64, 256),
                             (64, 128, 64), (64, 128, 128), (64, 128, 256)]:
        enumerate_qgmma_flash_warpspec_kernels(
            specs,
            sm=90,
            dtype='e4m3',
            sage_block_sizes=sage_block_sizes,
            output_dtype="bf16")

    if 'ENABLE_HMMA_FP32' in os.environ:
        enumerate_hgmma_flash_warpspec_kernels(specs, sm=90, dtype='fp16_fp32')
    # Optionally generate HMMA kernels on SM90 for comparison.
    if 'SM90_USE_HMMA' in os.environ:
        print("WARNING: GENERATING HMMA KERNELS INSTEAD OF HGMMA FOR SM90")
        enumerate_hmma_kernels(specs, sm=90, dtype='fp16')
        enumerate_hmma_kernels(specs, sm=90, dtype='bf16')

    # SM90 IGMMA
    if 'SM90_USE_IMMA' in os.environ:
        print("WARNING: GENERATING IMMA KERNELS INSTEAD OF IGMMA FOR SM90")
        enumerate_imma_kernels(specs, sm=90)

    # SM 89
    if 'ENABLE_SM89_QMMA' in os.environ:
        enumerate_qmma_kernels(specs, sm=89)
        enumerate_qmma_flash_kernels(specs, sm=89, dtype='e4m3_fp32')
        # Add bf16 output MLA kernels.
        enumerate_qmma_flash_kernels(specs,
                                     sm=89,
                                     dtype='e4m3_fp32',
                                     head_sizes=[192, 576],
                                     output_dtype="bf16")
        # Sage Attention on Ada only supports block_size = (64, 32, 32)
        enumerate_qmma_flash_kernels(specs,
                                     sm=89,
                                     dtype='e4m3_fp32',
                                     sage_block_sizes=(64, 32, 32),
                                     output_dtype="bf16")
        enumerate_qmma_flash_kernels(specs,
                                     sm=89,
                                     dtype='e4m3_fp32',
                                     sage_block_sizes=(64, 32, 32),
                                     output_dtype="fp16")

    enumerate_imma_kernels(specs, sm=89)
    enumerate_hmma_kernels(specs, sm=89, dtype='fp16')
    enumerate_hmma_kernels(specs, sm=89, dtype='bf16')
    enumerate_hmma_flash_kernels(specs, sm=89, dtype='fp16')
    enumerate_hmma_flash_kernels(specs, sm=89, dtype='bf16')

    # SM 80 / 86
    enumerate_imma_kernels(specs, sm=80)
    enumerate_hmma_kernels(specs, sm=80, dtype='fp16')
    enumerate_hmma_kernels(specs, sm=80, dtype='bf16')
    enumerate_hmma_flash_kernels(specs, sm=80, dtype='fp16')
    enumerate_hmma_flash_kernels(specs, sm=80, dtype='bf16')

    enumerate_imma_kernels(specs, sm=86)
    enumerate_hmma_kernels(specs, sm=86, dtype='fp16')
    enumerate_hmma_kernels(specs, sm=86, dtype='bf16')
    enumerate_hmma_flash_kernels(specs, sm=86, dtype='fp16')
    enumerate_hmma_flash_kernels(specs, sm=86, dtype='bf16')

    # SM 90 (only generate paged_kv_fmha hmma kernels)
    enumerate_hmma_paged_kv_flash_kernels(specs, sm=90, dtype='fp16')
    enumerate_hmma_paged_kv_flash_kernels(specs, sm=90, dtype='bf16')

    if 'ENABLE_SM100' in os.environ:
        # SM 100
        enumerate_hmma_flash_kernels(specs, sm=100, dtype='fp16')
        enumerate_hmma_flash_kernels(specs, sm=100, dtype='bf16')
        enumerate_hmma_flash_kernels(specs,
                                     sm=100,
                                     dtype='bf16',
                                     head_size_v=128)
        enumerate_hmma_flash_kernels(specs,
                                     sm=100,
                                     dtype='bf16',
                                     head_size_v=512)

    if 'ENABLE_SM120' in os.environ:
        # SM 120
        enumerate_hmma_flash_kernels(specs, sm=120, dtype='fp16')
        enumerate_hmma_flash_kernels(specs, sm=120, dtype='bf16')
        enumerate_hmma_flash_kernels(specs,
                                     sm=120,
                                     dtype='bf16',
                                     head_size_v=128)
        enumerate_hmma_flash_kernels(specs,
                                     sm=120,
                                     dtype='bf16',
                                     head_size_v=512)
        enumerate_qmma_kernels(specs, sm=120)
        enumerate_qmma_flash_kernels(specs, sm=120, dtype='e4m3_fp32')
        # Add bf16 output MLA kernels.
        enumerate_qmma_flash_kernels(specs,
                                     sm=120,
                                     dtype='e4m3_fp32',
                                     head_sizes=[192, 576],
                                     output_dtype="bf16")

    if 'ENABLE_HMMA_FP32' in os.environ:
        enumerate_hmma_flash_kernels(specs, sm=80, dtype='fp16_fp32')
        enumerate_hmma_flash_kernels(specs, sm=86, dtype='fp16_fp32')
        enumerate_hmma_flash_kernels(specs, sm=89, dtype='fp16_fp32')
        # SM 90 (only generate paged_kv_fmha hmma kernels)
        enumerate_hmma_paged_kv_flash_kernels(specs, sm=90, dtype='fp16_fp32')
        if 'ENABLE_SM100' in os.environ:
            # SM 100
            enumerate_hmma_flash_kernels(specs, sm=100, dtype='fp16_fp32')
        if 'ENABLE_SM120' in os.environ:
            # SM 120
            enumerate_hmma_flash_kernels(specs, sm=120, dtype='fp16_fp32')

    for sm in [80, 86, 89, 90]:
        if not (sm == 90 and "GENERATE_CUBIN" in os.environ):
            # Hopper uses warp-specialized kernels instead (hasn't been merged yet).
            enumerate_hmma_flash_kernels(specs,
                                         sm=sm,
                                         dtype='bf16',
                                         head_size_v=128)
        enumerate_hmma_flash_kernels(specs,
                                     sm=sm,
                                     dtype='bf16',
                                     head_size_v=512)

    # SM 75
    enumerate_imma_kernels(specs, sm=75)
    enumerate_hmma_kernels(specs, sm=75)
    enumerate_hmma_flash_kernels(specs, sm=75)

    # SM 70
    if 'ENABLE_SM70' in os.environ:
        enumerate_hmma884_kernels(specs, sm=70)
        enumerate_hmma_flash_kernels(specs, sm=70)

    # TODO: refactor this; maybe adding a option to enumerate_*mma_kernels()
    enumerate_cross_mha_kernels(specs)

    # Expand the cartesian product of the list fields "seq_len" and "head_size".
    specs_expanded = []
    list_like = lambda x: isinstance(x, list) or isinstance(x, tuple)
    for kspec in specs:
        tmp_s = kspec.seq_len
        tmp_d = kspec.head_size
        tmp_dtype = kspec.dtype
        tmp_exp = [kspec._replace(seq_len=s)
                   for s in tmp_s] if list_like(tmp_s) else [kspec]
        tmp_exp = [
            tmp_ks._replace(head_size=d) for d in tmp_d for tmp_ks in tmp_exp
        ] if list_like(tmp_d) else tmp_exp
        tmp_exp = [
            tmp_ks._replace(dtype=dt) for dt in tmp_dtype for tmp_ks in tmp_exp
        ] if list_like(tmp_dtype) else tmp_exp
        specs_expanded.extend(tmp_exp)

    # Sanitize kernel specs
    specs_expanded = [
        kspec for kspec in specs_expanded if kspec.sm >= kspec.sm_mma
    ]

    # Expand the list for the cross-MHA kernels.
    # TRT-LLM uses the head_interleaved=False mode.
    if 'GENERATE_CUBIN' in os.environ:
        specs_expanded = [
            kspec._replace(head_interleaved=False) for kspec in specs_expanded
        ]
    # yapf: disable
    specs_names = [(kspec, *encode_name(kspec)) for kspec in specs_expanded
                  # Volta is deprecated in TRT-LLM.
                  if  (kspec.sm            in [80, 86, 89, 90, 120]
                  and kspec.dtype         in ['fp16', 'bf16', 'fp16_fp32', 'e4m3', 'e4m3_fp32']
                  and kspec.head_size     <= 256
                  and kspec.head_size_v   == 0
                  and kspec.sage_block_sizes is None
                  and kspec.version       == 2
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == True
                  and kspec.input_layout != InputLayout.SEPARATE_Q_K_V
                  or (kspec.sm == 90
                  and kspec.dtype         in ['fp16', 'bf16', 'fp16_fp32']
                  and kspec.head_size     <= 256
                  and kspec.ldgsts_q  == True
                  and kspec.version       == 2
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == False)
                  # Clip/SigLip support.
                  or  (kspec.sm           == 100
                  and kspec.dtype         in ['fp16', 'bf16', 'fp16_fp32', 'e4m3', 'e4m3_fp32']
                  and kspec.head_size     == 80
                  and kspec.head_size_v   == 0
                  and kspec.sage_block_sizes is None
                  and kspec.version       == 2
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == True
                  and kspec.input_layout != InputLayout.SEPARATE_Q_K_V)
                  # Deepseek MLA (generation 576/512 paged)
                  or (kspec.sm            in [90, 100, 120]
                  and kspec.dtype         in ['bf16', 'e4m3_fp32']
                  and kspec.head_size     == 576
                  and kspec.head_size_v   == 512
                  and kspec.input_layout == InputLayout.Q_PAGED_KV
                  and kspec.sage_block_sizes is None
                  and kspec.version       == 2
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == True
                  and kspec.warp_specialization == False
                  and kspec.tiled == True)
                  # Deepseek MLA (context 192/128 separate-q-k-v)
                  or (kspec.sm            in [90, 100, 120]
                  and kspec.dtype         in ['bf16', 'e4m3', 'e4m3_fp32']
                  and kspec.head_size     == 192
                  and kspec.head_size_v   == 128
                  and kspec.input_layout == InputLayout.SEPARATE_Q_K_V
                  and kspec.sage_block_sizes is None
                  and kspec.version       == 2
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == True
                  and ((kspec.warp_specialization == True and kspec.alibi == False)   # sm90
                    or (kspec.warp_specialization == False and kspec.tiled == True))  # non-sm90
                  and kspec.enable_attn_logit_softcapping == False)
                  # SageAttention (warp_spec, head_size in (80, 128), packed QKV, padding mask)
                  or (kspec.sm            == 90
                  and kspec.head_size     in [80, 128]
                  and kspec.version       == 2
                  and kspec.sage_block_sizes in [(64, 64, 256)]
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == True
                  and kspec.warp_specialization == True
                  and kspec.input_layout == InputLayout.PACKED_QKV
                  and kspec.alibi == False
                  and kspec.enable_attn_logit_softcapping == False)
                  # SageAttention on Ada (head_size in (80, 128), packed QKV, padding mask)
                  or (kspec.sm            == 89
                  and kspec.head_size     in [80, 128]
                  and kspec.sage_block_sizes in [(64, 32, 32)]
                  and kspec.output_dtype in ['fp16', 'bf16']
                  and kspec.version       == 2
                  and kspec.cross_mha     == False
                  and kspec.flash_attention == True
                  and kspec.warp_specialization == False
                  and kspec.input_layout == InputLayout.PACKED_QKV))
                  # only generate head_size = 128/256 for attn_logit_softcapping operation.
                  and (kspec.head_size == 128 or kspec.head_size == 256 or not kspec.enable_attn_logit_softcapping)]
    # yapf: enable

    generate_files(specs_names)


if __name__ == '__main__':
    enumerate_kernels()

# General restrictions
# FP16: no s=192
# FP16: no Volta
# Interleaved only for Int8

# v1:
# 384 should have 1x8 kernels not to exceed xmmas_n = 4
# No support for interleaved

# v2:
#

# TODO record all step and smem configs
