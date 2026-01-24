// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze
//   and Meng, Chenlin and Zhu, Jun-Yan and Han, Song}, journal={arXiv preprint arXiv:2411.05007}, year={2024}
// }

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "Tensor.h"
#include "common.h"

namespace nunchaku::kernels
{

void gemm_w4a4(Tensor act,          // packed act [M, K / 2]
    Tensor wgt,                     // packed act [N, K / 2]
    Tensor out,                     // linear     [M, N]
    Tensor qout,                    // packed act [M, N / 2]
    Tensor ascales,                 // packed as  [K / 64, M]
    Tensor wscales,                 // packed ws  [K / 64, N]
    Tensor oscales,                 // packed as  [N / 64, M]
    Tensor poolout,                 // linear     [M / PoolSize, N]
    Tensor lora_act_in,             // packed lora_act [M, R]
    Tensor lora_up,                 // packed lora_wgt [N, R]
    Tensor lora_down,               // packed lora_wgt [N, R]
    Tensor lora_act_out,            // packed lora_act [M, R]
    Tensor norm_q,                  // linear     [HEAD_DIM]
    Tensor norm_k,                  // linear     [HEAD_DIM]
    Tensor rotary_emb,              // linear     [M, HEAD_DIM / 2, 2, 2]
    Tensor bias,                    // packed ws  [N]
    Tensor smooth_factor,           // packed ws  [N], for quantization of the next layer
    Tensor out_vk,                  // linear     [B, num_heads, head_dim + 1, head_dim]
    Tensor out_linearattn,          // linear     [B, (M), N / 3]
    bool act_unsigned,
    std::vector<float> lora_scales, // [R / 16]
    bool fuse_silu, bool fp4, float alpha, Tensor wcscales,
    Tensor out_q,                   // packed attention [B, H, M, D]
    Tensor out_k,                   // packed attention [B, H, M, D]
    Tensor out_v,                   // packed attention [B, H, M, D]
    int attn_tokens);

void quantize_w4a4_act_fuse_lora(Tensor input, Tensor output, Tensor oscales, Tensor lora_down, Tensor lora_act_out,
    Tensor smooth = {}, bool fuse_glu = false, bool fp4 = false);

}; // namespace nunchaku::kernels
