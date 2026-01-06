// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
//   journal={arXiv preprint arXiv:2411.05007},
//   year={2024}
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

#include "torch.h"
#include "zgemm.h"

namespace nunchaku::ops {

void gemm_w4a4(std::optional<torch::Tensor> act,            // packed act [M, K / 2]
               std::optional<torch::Tensor> wgt,            // packed act [N, K / 2]
               std::optional<torch::Tensor> out,            // linear     [M, N]
               std::optional<torch::Tensor> qout,           // packed act [M, N / 2]
               std::optional<torch::Tensor> ascales,        // packed as  [K / 64, M]
               std::optional<torch::Tensor> wscales,        // packed ws  [K / 64, N]
               std::optional<torch::Tensor> oscales,        // packed as  [N / 64, M]
               std::optional<torch::Tensor> poolout,        // linear     [M / PoolSize, N]
               std::optional<torch::Tensor> lora_act_in,    // packed lora_act [M, R]
               std::optional<torch::Tensor> lora_up,        // packed lora_wgt [N, R]
               std::optional<torch::Tensor> lora_down,      // packed lora_wgt [N, R]
               std::optional<torch::Tensor> lora_act_out,   // packed lora_act [M, R]
               std::optional<torch::Tensor> norm_q,         // linear     [HEAD_DIM]
               std::optional<torch::Tensor> norm_k,         // linear     [HEAD_DIM]
               std::optional<torch::Tensor> rotary_emb,     // linear     [M, HEAD_DIM / 2, 2, 2]
               std::optional<torch::Tensor> bias,           // packed ws  [N]
               std::optional<torch::Tensor> smooth_factor,  // packed ws  [N], for quantization of the next layer
               std::optional<torch::Tensor> out_vk,         // linear     [B, num_heads, head_dim + 1, head_dim]
               std::optional<torch::Tensor> out_linearattn, // linear     [B, (M), N / 3]
               bool act_unsigned,
               std::vector<float> lora_scales,
               bool fuse_silu,
               bool fp4,
               float alpha,
               std::optional<torch::Tensor> wcscales,
               std::optional<torch::Tensor> out_q, // packed attention [B, H, M, D]
               std::optional<torch::Tensor> out_k, // packed attention [B, H, M, D]
               std::optional<torch::Tensor> out_v, // packed attention [B, H, M, D]
               int attn_tokens) {

    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        return ret;
    };
    nunchaku::kernels::gemm_w4a4(getTensor(act),
                                 getTensor(wgt),
                                 getTensor(out),
                                 getTensor(qout),
                                 getTensor(ascales),
                                 getTensor(wscales),
                                 getTensor(oscales),
                                 getTensor(poolout),
                                 getTensor(lora_act_in),
                                 getTensor(lora_up),
                                 getTensor(lora_down),
                                 getTensor(lora_act_out),
                                 getTensor(norm_q),
                                 getTensor(norm_k),
                                 getTensor(rotary_emb),
                                 getTensor(bias),
                                 getTensor(smooth_factor),
                                 getTensor(out_vk),
                                 getTensor(out_linearattn),
                                 act_unsigned,
                                 lora_scales,
                                 fuse_silu,
                                 fp4,
                                 alpha,
                                 getTensor(wcscales),
                                 getTensor(out_q),
                                 getTensor(out_k),
                                 getTensor(out_v),
                                 attn_tokens);
    // Tensor::synchronizeDevice();
}

void quantize_w4a4_act_fuse_lora(std::optional<torch::Tensor> input,
                                std::optional<torch::Tensor> output,
                                std::optional<torch::Tensor> oscales,
                                std::optional<torch::Tensor> lora_down,
                                std::optional<torch::Tensor> lora_act_out,
                                std::optional<torch::Tensor> smooth,
                                bool fuse_glu,
                                bool fp4) {
    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        return ret;
    };
    nunchaku::kernels::quantize_w4a4_act_fuse_lora(getTensor(input),
                                                   getTensor(output),
                                                   getTensor(oscales),
                                                   getTensor(lora_down),
                                                   getTensor(lora_act_out),
                                                   getTensor(smooth),
                                                   fuse_glu,
                                                   fp4);
}

}; // namespace nunchaku::ops
