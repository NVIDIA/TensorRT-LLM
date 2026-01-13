# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pynvml
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from visual_gen.layers.linear import ditLinear
from visual_gen.configs.op_manager import LinearOpManager


def get_compute_capability():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    pynvml.nvmlShutdown()
    return str(cc[0]) + str(cc[1])


if __name__ == "__main__":
    sm_version = get_compute_capability()
    if sm_version != "120":
        print(f"SVDQuant is only supported on SM 120, skipping test")
        exit(0)

    hf_hub_download(
        repo_id="mit-han-lab/nunchaku-flux.1-dev",
        filename="svdq-fp4_r32-flux.1-dev.safetensors",
        local_dir="./svd_fp4_checkpoint",
    )
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        local_dir="./bf16_checkpoint",
    )

    linear_impls = "svd-nvfp4"
    linear_name = "transformer_blocks.0.attn"

    bf16_weights = load_file("./bf16_checkpoint/transformer/diffusion_pytorch_model-00001-of-00003.safetensors")
    to_q_weight = bf16_weights[linear_name + ".to_q.weight"]
    to_k_weight = bf16_weights[linear_name + ".to_k.weight"]
    to_v_weight = bf16_weights[linear_name + ".to_v.weight"]
    to_q_bias = bf16_weights[linear_name + ".to_q.bias"]
    to_k_bias = bf16_weights[linear_name + ".to_k.bias"]
    to_v_bias = bf16_weights[linear_name + ".to_v.bias"]

    weight = torch.cat([to_q_weight, to_k_weight, to_v_weight], dim=0).to("cuda")
    bias = torch.cat([to_q_bias, to_k_bias, to_v_bias], dim=0).to("cuda")

    in_features = weight.shape[1]
    out_features = weight.shape[0]
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    input = torch.randn(3, 4096, in_features, dtype=torch.bfloat16, device="cuda")
    ref_output = torch.matmul(input, weight.transpose(0, 1)) + bias

    LinearOpManager.set_linear_type(linear_impls)
    visual_gen_linear = ditLinear(in_features, out_features, device="cuda", dtype=torch.bfloat16)

    visual_gen_linear.name = linear_name + ".to_qkv"
    weights_table = load_file("./svd_fp4_checkpoint/svdq-fp4_r32-flux.1-dev.safetensors")
    svd_weight_name_table = {
        "attn.to_qkv": "qkv_proj",
        "attn.to_added_qkv": "qkv_proj_context",
        "attn.to_out.0": "out_proj",
        "attn.to_add_out": "out_proj_context",
        "ff.net.0.proj": "mlp_fc1",
        "ff.net.2": "mlp_fc2",
        "ff_context.net.0.proj": "mlp_context_fc1",
        "ff_context.net.2": "mlp_context_fc2",
        "proj_out0": "out_proj",
        "proj_mlp": "mlp_fc1",
        "proj_out1": "mlp_fc2",
    }
    visual_gen_linear.load_fp4_weight(weights_table, svd_weight_name_table)
    output = visual_gen_linear(input)

    print("max output", torch.abs(output).max())
    print("max ref_output", torch.abs(ref_output).max())
    print("max diff", torch.abs(output - ref_output).max())
    assert torch.abs(output - ref_output).max() < 2, "max diff is too large"
