# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch


def torch_rope_reference(x, freqs_cis, input_positions):
    freqs_cis = freqs_cis[None, input_positions : input_positions + x.shape[1]]
    freqs_cis = freqs_cis[:, :, None]  #          [b, s, h_d//2, 2  ] --> [b, s, 1  , h_d//2, 2]
    # print("FCS", freqs_cis.shape)
    xshaped = x.float().unflatten(-1, (-1, 2))  # [b, s, n_h   , h_d] --> [b, s, n_h, h_d//2, 2]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )  # [b, s, n_h, h_d//2, 2]

    return x_out2.flatten(-2).type_as(x)  # [b, s, n_h, h_d//2, 2] --> [b, s, n_h, h_d]


def torch_reference_mha_stage2(values, logsumexp):
    max_logsumexp = torch.max(logsumexp, axis=-1, keepdim=True)[0]  # [b, n_heads, 1]
    sumexp = torch.exp(logsumexp - max_logsumexp)  # [b, n_heads, num_blocks]
    aggregate_sumexp = torch.sum(sumexp, axis=-1)  # [b, n_heads]
    output = values * sumexp[:, :, :, None]  # [b, n_heads, num_blocks, d_head]
    output = output / aggregate_sumexp[:, :, None, None]
    output = torch.sum(output, axis=2)
    return output
