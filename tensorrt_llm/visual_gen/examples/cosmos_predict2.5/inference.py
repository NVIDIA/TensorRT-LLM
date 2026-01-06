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

"""Base model inference script."""

from pathlib import Path
from typing import Annotated

import pydantic
import torch
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)
from einops import rearrange

# ADDED BY US #
import visual_gen
from visual_gen.layers import ditAttnProcessor, apply_visual_gen_linear


class ditCosmosAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ditAttnProcessor().visual_gen_attn

    def set_context_parallel_group(self, process_group, ranks, stream):
        # cp will be automatically handled by visual_gen
        # this function is a placeholder to satisfy the interface
        pass

    def forward(self, query, key, value, *args, **kwargs):
        assert len(args) == 0
        assert len(kwargs) == 0

        # Follow Cosmos's implementation to use bfloat16
        dtype = torch.bfloat16
        query = query.to(dtype)
        key = key.to(dtype)
        value = value.to(dtype)

        results = self.attn(query, key, value, tensor_layout="NHD")
        return rearrange(results, "b ... h l -> b ... (h l)")


def enable_visual_gen(pipe, args, cp_method="ulysses"):
    cp_size = args.context_parallel_size
    # we supported 3 cp methods: cp(allgather kv), ring, ulysses
    if cp_method == "cp":
        cp_key = "dit_cp_size"
    elif cp_method == "ring":
        cp_key = "dit_ring_size"
    elif cp_method == "ulysses":
        cp_key = "dit_ulysses_size"
    else:
        raise ValueError(f"Invalid cp method: {cp_method}")

    # The perf may different on different SM versions and shapes. We recommend some examples here.
    sm_version = torch.cuda.get_device_capability()
    attn_type = "default"  # `default`: pytorch bf16 attn. Hopper: `sage-attn`, `trtllm-attn`, `te`, `te-fp8`; Blackwell: `te`, `te-fp8`, `flash-attn4`.
    if sm_version <= (9, 0):
        attn_type = "sage-attn"
    else:
        attn_type = "te-fp8"
    linear_type = "default"  # `default`: pytorch bf16 linear. Hopper: `trtllm-fp8-blockwise`, `trtllm-fp8-per-tensor`, `te-fp8-blockwise`, `te-fp8-per-tensor`; Blackwell: `trtllm-nvfp4`, `te-fp8-per-tensor`
    if sm_version <= (9, 0):
        linear_type = "trtllm-fp8-blockwise"
    else:
        linear_type = "te-fp8-per-tensor"

    visual_gen.setup_configs(
        parallel={
            cp_key: cp_size,
        },
        attn={
            "type": attn_type,
        },
        linear={
            "type": linear_type,  # `default`: pytorch bf16 linear. Hopper: `trtllm-fp8-blockwise`, `trtllm-fp8-per-tensor`, `te-fp8-blockwise`, `te-fp8-per-tensor`; Blackwell: `trtllm-nvfp4`, `te-fp8-per-tensor`
        },
    )
    # replace the attention with visual_gen_cosmos_attention
    for module in pipe.model.modules():
        if hasattr(module, "attn_op"):
            setattr(module, "attn_op", ditCosmosAttention())

    apply_visual_gen_linear(pipe.model, load_parameters=True, quantize_weights=True)
    # seems like FlashInfer's RMSNorm is slower than TE's on B200.
    # apply_visual_gen_norm(pipe.model, rmsnorm=["q_norm", "k_norm"], load_parameters=True)
    pipe.model = torch.compile(pipe.model)


# END OF US CODES #


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s).
    If multiple files are provided, the model will be loaded once and all the samples will be run sequentially.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""


def main(
    args: Args,
):
    inference_samples = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.inference import Inference

    inference = Inference(args.setup)
    # ADDED BY US #
    enable_visual_gen(inference.pipe, args.setup)
    # END OF US CODES #
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
