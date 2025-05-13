# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
from collections import namedtuple

sys.path.append("../")
from setup import get_effective_sm_and_name, get_kernel_code

dtype2traits = {
    'int8': 'imma_int8_int32_traits',
    'fp16': 'hmma_fp16_traits',
    'fp16_fp32': 'hmma_fp32_traits',
    'bf16': 'hmma_bf16_traits',
    'e4m3': 'qmma_e4m3_fp32_traits',
    'e4m3_fp32': 'qmma_e4m3_fp32_traits',
    'e4m3_fp16': 'qmma_e4m3_fp16_traits'
}

fmha_dgrad_v2_flash_attention_template = '''\
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

#include "fused_multihead_attention_fprop.h"
#include "fused_multihead_attention_dgrad_kernel_1xN_reload.h"
#include "fused_multihead_attention_dgrad_kernel_1xN_flash.h"

const int STEPQ = {q_loop_step};
const int STEPK = {kv_loop_step};

template<typename Kernel_traits>
__global__ void s_{head_size}_dot_do_o_compute_kernel(Fused_multihead_attention_fprop_params params) {{
    fused_multihead_attention::compute_dot_do_o<Kernel_traits>(params);
}}

template<typename Kernel_traits>
__global__ __launch_bounds__(Kernel_traits::Cta_tile_p::THREADS_PER_CTA)
void fmha_dgrad_v2_{dtype}_S_{head_size}_sm{sm}_kv_inner_loop_kernel(
    Fused_multihead_attention_fprop_params params) {{
    fused_multihead_attention::compute_dq_dk_dv_1xN_kv_inner_loop<Kernel_traits>(params);
}}

template<typename Kernel_traits>
__global__ void s_{head_size}_convert_dq_to_16bits_kernel(Fused_multihead_attention_fprop_params params) {{
    fused_multihead_attention::convert_dq_to_16bits<Kernel_traits>(params);
}}

template<typename Kernel_traits, typename Kernel_traits_causal>
void run_fmha_dgrad_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}_(
    const Fused_multihead_attention_fprop_params &params,
    cudaStream_t stream) {{

    size_t smem_size = 0;

    // The instruction traits.
    using Traits_p = typename Kernel_traits::Traits_p;
    // The CTA tile for P.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The smem tile for dO.
    using Smem_tile_do_a = fmha::Smem_tile_a<Traits_p, Cta_tile_p, fmha::Row>;
    // The CTA tile for O.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The input tile for dO.
    smem_size += Smem_tile_do_a::BYTES_PER_TILE * 2;  // dO

    // The input tiles for Q, K and V.
    smem_size += Kernel_traits::Smem_tile_q::BYTES_PER_TILE * 2;  // Q
    smem_size += Kernel_traits::Smem_tile_k::BYTES_PER_TILE;      // K
    smem_size += Kernel_traits::Smem_tile_v::BYTES_PER_TILE;      // V

    // The tile in shared memory to reorganize dQ.
    using Smem_tile_dq = Smem_tile_dq_red<Traits_p, Cta_tile_o>;
    smem_size += Smem_tile_dq::BYTES_PER_TILE;

    // The tile to store S/dP.
    using Smem_tile_st = Smem_tile_mma_transposed<Traits_p, Cta_tile_p>;
    smem_size += Smem_tile_st::BYTES_PER_TILE * 2;

    dim3 grid(params.h, params.b, 8);
    s_{head_size}_dot_do_o_compute_kernel<Kernel_traits>
        <<<grid, Kernel_traits::THREADS, 0, stream>>>(params);

    auto kernel = params.is_causal
                      ? &fmha_dgrad_v2_{dtype}_S_{head_size}_sm{sm}_kv_inner_loop_kernel<Kernel_traits_causal>
                      : &fmha_dgrad_v2_{dtype}_S_{head_size}_sm{sm}_kv_inner_loop_kernel<Kernel_traits>;

    if( smem_size >= 48 * 1024 ) {{
        FMHA_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }}
    grid = dim3((params.s + STEPK -1)/ STEPK, params.b, params.h);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, stream>>>(params);

    s_{head_size}_convert_dq_to_16bits_kernel<Kernel_traits><<<params.total_s, 128, 0, stream>>>(params);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}}

void run_fmha_dgrad_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}(
    const Fused_multihead_attention_fprop_params &params,
    cudaStream_t stream) {{

    // HEADS_INTERLEAVED = (FLAGS & 0x20u) == 0u;
    // SEQUENCES_INTERLEAVED = (FLAGS & 0x400) != 0u
    // for example, [s, b, h, 3, d] --> flag = 0x400

    using Kernel_traits = fmha::Kernel_traits_v2<{instruction_traits},
                                                 STEPK,
                                                 {head_size},  // Valid_D
                                                 STEPQ,
                                                 {warps_m},
                                                 {warps_n},
                                                 1,
                                                 {kernel_flags}>;
    using Kernel_traits_causal = fmha::Kernel_traits_v2_causal_mask<{instruction_traits},
                                                                    STEPK,
                                                                    {head_size},  // Valid_D
                                                                    STEPQ,
                                                                    {warps_m},
                                                                    {warps_n},
                                                                    1,
                                                                    {kernel_flags}>;
    static_assert(Kernel_traits::VERSION == 2);

    run_fmha_dgrad_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}_<Kernel_traits, Kernel_traits_causal>(
        params, stream);
}}
'''

fmha_fprop_v2_flash_attention_template = '''\
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

#include "fused_multihead_attention_fprop.h"
#include "fused_multihead_attention_flash_attention_fprop_kernel.h"

using Kernel_traits = fmha::Kernel_traits_v2<
    {instruction_traits},
    {kv_loop_step},
    {head_size},
    {q_loop_step},
    {warps_m},
    {warps_n},
    1,
    {kernel_flags}>;

using Kernel_traits_causal = fmha::Kernel_traits_v2_causal_mask<
    {instruction_traits},
    {kv_loop_step},
    {head_size},
    {q_loop_step},
    {warps_m},
    {warps_n},
    1,
    {kernel_flags}>;

template<bool IS_TRAINING, typename Kernel_traits>
__global__ void
fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel(
    Fused_multihead_attention_fprop_params params,
    const int total_heads) {{

    fused_multihead_attention::device_flash_attention<Kernel_traits, IS_TRAINING>(params,
                                                                                  total_heads);
}}

template<bool IS_TRAINING, typename Kernel_traits>
__global__ void
fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel_nl(Fused_multihead_attention_fprop_params params,
                                                  const int num_full_heads,
                                                  const int num_main_groups,
                                                  const int main_group_size,
                                                  const int main_steps,
                                                  const int rest_steps) {{

    fused_multihead_attention::device_flash_attention<Kernel_traits, IS_TRAINING>(
        params, num_full_heads, num_main_groups, main_group_size, main_steps, rest_steps);
}}

template<typename Kernel_traits, typename Kernel_traits_causal>
void run_fmha_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}_(
    Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
    const bool configure) {{

    auto kernel =
        launch_params.is_causal
            ? (launch_params.is_training
                   ? &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel<true, Kernel_traits>
                   : &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel<false, Kernel_traits>)
            : (launch_params.is_training
                   ? &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel<true, Kernel_traits>
                   : &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel<false, Kernel_traits>);

    constexpr int smem_size = fused_multihead_attention::get_dynamic_smem_size<Kernel_traits>();

    if( smem_size >= 48 * 1024 ) {{
        FMHA_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }}

    const int sm_count = launch_params.props->multiProcessorCount;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size));
    int total_ctas = sm_count * ctas_per_sm;

    const int heads_total = launch_params.params.b * launch_params.params.h;
    if( configure ) {{

        using Mma_tile_p =
            typename Kernel_traits::Traits_p::template Mma_tile<typename Kernel_traits::Cta_tile_p>;
        const size_t STEPS = (launch_params.params.s + Kernel_traits::Cta_tile_p::M - 1) /
                             Kernel_traits::Cta_tile_p::M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;

        size_t heads_per_cta = ((heads_total + total_ctas - 1) / total_ctas);
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8;
        launch_params.elts_per_thread = heads_per_cta * elts_per_head;
        return;
    }}

    dim3 grid(total_ctas);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(launch_params.params,
                                                                              heads_total);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}}

template<typename Kernel_traits, typename Kernel_traits_causal>
void run_fmha_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}_nl_(
    Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
    const bool configure) {{

    auto kernel =
        launch_params.is_causal
            ? (launch_params.is_training
                   ? &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel_nl<true, Kernel_traits_causal>
                   : &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel_nl<false, Kernel_traits_causal>)
            : (launch_params.is_training
                   ? &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel_nl<true, Kernel_traits>
                   : &fmha_flash_attention_fprop_v2_{dtype}_S_{head_size}_kernel_nl<false, Kernel_traits>);

    constexpr int smem_size = fused_multihead_attention::get_dynamic_smem_size<Kernel_traits>();

    if( smem_size >= 48 * 1024 ) {{
        FMHA_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }}

    const int sm_count = launch_params.props->multiProcessorCount;
    int ctas_per_sm;
    FMHA_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size));
    int total_ctas = sm_count * ctas_per_sm;

    // hack to distribute M steps to blocks (more waves)
    const int full_steps =
        (launch_params.params.s + Kernel_traits::Cta_tile_p::M - 1) / Kernel_traits::Cta_tile_p::M;
    const int heads_total = launch_params.params.b * launch_params.params.h;
    total_ctas = std::min(total_ctas * 8, heads_total * full_steps);

    if( configure ) {{
        const int heads_total = launch_params.params.b * launch_params.params.h;
        std::tie(launch_params.num_full_heads,
                 launch_params.num_main_groups,
                 launch_params.heads_last_wave,
                 launch_params.main_steps,
                 launch_params.rest_steps,
                 launch_params.elts_per_thread) =
            work_dist<Kernel_traits>(launch_params.params.s, total_ctas, heads_total);
        return;
    }}

    dim3 grid(total_ctas);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
        launch_params.params,
        launch_params.num_full_heads,
        launch_params.num_main_groups,
        launch_params.heads_last_wave,
        launch_params.main_steps,
        launch_params.rest_steps);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}}

void run_fmha_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}(
    Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
    const bool configure) {{
    if( launch_params.is_nl ) {{
        run_fmha_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}_nl_<Kernel_traits, Kernel_traits_causal>(
            launch_params, configure);
    }} else {{
        run_fmha_v2_flash_attention_{dtype}_S_{head_size}_sm{sm}_<Kernel_traits, Kernel_traits_causal>(
            launch_params, configure);
    }}
}}
'''


def get_kernel_code(kspec):
    min_cuda_version = 0  # no restriction

    # The architecture that determines the instruction.
    effective_sm, sm_name = get_effective_sm_and_name(kspec)

    if (effective_sm < 90):
        dtype = kspec.dtype
        if kspec.dtype == 'fp16' and kspec.ctype == 'fp32':
            dtype = 'fp16_fp32'
        if kspec.dtype == 'bf16' and kspec.ctype == 'fp32':
            dtype = 'bf16'
        instruction_traits = 'fmha::' + sm_name.capitalize(
        ) + '_' + dtype2traits[dtype]

    return instruction_traits


def get_fname(kspec):
    fmt = 'fmha_{train_op}_v{version}_flash_attention_{dtype}_S_{head_size}_kernel.sm{sm}.cu'
    # Assemble the name of the kernel.
    name_base = fmt.format(**kspec._asdict())
    return name_base


def generate_kernels(kspec):
    instruction_traits = get_kernel_code(kspec)
    flags = 0
    if kspec.ldgsts_q:
        flags |= 1
    if kspec.ldgsts_k:
        flags |= 2
    if kspec.ldgsts_v:
        flags |= 4
    if kspec.share_smem_k_v:
        flags |= 8
    if not kspec.head_interleaved:
        flags |= 32
    if not kspec.k_in_regs:
        flags |= 64
    if kspec.sequence_interleaved:
        flags |= 1024

    kernel_flags = '0x{:02x}u'.format(flags)

    tmp = dict(locals(), **kspec._asdict())

    if kspec.train_op == 'fprop':
        return fmha_fprop_v2_flash_attention_template.format(**tmp)
    else:
        return fmha_dgrad_v2_flash_attention_template.format(**tmp)


if __name__ == '__main__':
    if not os.path.exists('./flash_attention_kernels'):
        os.mkdir('./flash_attention_kernels')

    spec_fields = ('sm', 'sm_mma', 'dtype', 'ctype', 'seq_len', 'head_size',
                   'warps_m', 'warps_n', 'version', 'interleaved', 'ldgsts_q',
                   'ldgsts_k', 'ldgsts_v', 'share_smem_k_v', 'k_in_regs',
                   'q_loop_step', 'head_interleaved', 'sequence_interleaved',
                   'kv_loop_step', 'train_op')
    kernel_spec = namedtuple('kernel_spec', spec_fields)

    specs = []

    for head_size in [40, 64, 80, 96, 128]:
        specs.append(
            kernel_spec(
                sm=80,
                sm_mma=80,
                dtype='fp16',
                ctype='fp32',
                seq_len=0,  # any sequence
                head_size=head_size,
                warps_m=4,
                warps_n=1,
                version=2,
                interleaved=False,
                ldgsts_q=True,
                ldgsts_k=False,
                ldgsts_v=True,
                share_smem_k_v=False,
                k_in_regs=False,
                head_interleaved=False,
                sequence_interleaved=False,
                q_loop_step=64,
                kv_loop_step=32,
                train_op='fprop'))

    for head_size in [40, 64, 80, 96, 128]:
        specs.append(
            kernel_spec(
                sm=80,
                sm_mma=80,
                dtype='fp16',
                ctype='fp32',
                seq_len=0,  # any sequence
                head_size=head_size,
                warps_m=1,
                warps_n=8,
                version=2,
                interleaved=False,
                ldgsts_q=False,
                ldgsts_k=False,
                ldgsts_v=False,
                share_smem_k_v=False,
                k_in_regs=True,
                head_interleaved=False,
                sequence_interleaved=False,
                q_loop_step=16,
                kv_loop_step=128,
                train_op='dgrad'))

    for head_size in [40, 64, 80, 96, 128]:
        specs.append(
            kernel_spec(
                sm=80,
                sm_mma=80,
                dtype='bf16',
                ctype='fp32',
                seq_len=0,  # any sequence
                head_size=head_size,
                warps_m=4,
                warps_n=1,
                version=2,
                interleaved=False,
                ldgsts_q=True,
                ldgsts_k=False,
                ldgsts_v=True,
                share_smem_k_v=False,
                k_in_regs=False,
                head_interleaved=False,
                sequence_interleaved=False,
                q_loop_step=64,
                kv_loop_step=32,
                train_op='fprop'))

    for head_size in [40, 64, 80, 96, 128]:
        specs.append(
            kernel_spec(
                sm=80,
                sm_mma=80,
                dtype='bf16',
                ctype='fp32',
                seq_len=0,  # any sequence
                head_size=head_size,
                warps_m=1,
                warps_n=8,
                version=2,
                interleaved=False,
                ldgsts_q=False,
                ldgsts_k=False,
                ldgsts_v=False,
                share_smem_k_v=False,
                k_in_regs=True,
                head_interleaved=False,
                sequence_interleaved=False,
                q_loop_step=16,
                kv_loop_step=128,
                train_op='dgrad'))

    for kspec in specs:
        fname = get_fname(kspec)
        code = generate_kernels(kspec)
        path = os.path.join('./flash_attention_kernels', fname)
        with open(path, 'w') as f:
            f.write(code)
