# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

from tensorrt_llm._torch.auto_deploy import LLM, AutoDeployConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="The HF model to use for onnx export.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4,
        help="The max sequence length to use for the model.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        # NOTE(yoco): Originally this is 2, however, don't know why, when set to 2,
        # the batch_size will collapse static int 2 even we explicitly it is dynamic axis.
        # And more weird, when set to 13, the batch_size will be dynamic.
        default=13,  # to enable dynamic batch_size, the match size must > 1
        help="The max batch size to use for the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use for the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory to save the exported ONNX model.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="The name of the exported ONNX model.",
    )
    args = parser.parse_args()

    print(f"Constructing model from {args.model}")

    # Prepare the AutoDeploy config, mode is export_driveos_llm_onnx
    ad_config = AutoDeployConfig(
        model=args.model,
        mode="export_driveos_llm_onnx",
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )
    ad_config.attn_backend = "torch"
    if args.output_dir is not None:
        ad_config.transforms["export_to_onnx"]["output_dir"] = args.output_dir
    if args.output_name is not None:
        ad_config.transforms["export_to_onnx"]["output_name"] = args.output_name
    _ = LLM(**ad_config.to_llm_kwargs())


if __name__ == "__main__":
    main()
