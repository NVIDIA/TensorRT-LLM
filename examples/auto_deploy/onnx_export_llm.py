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
"""ONNX export script for AutoDeploy models.

This script exports a HuggingFace model to ONNX format using the AutoDeploy
transform pipeline directly, without initializing the full LLM executor.
"""

import argparse

from tensorrt_llm._torch.auto_deploy.export import export_onnx
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace model to ONNX format using AutoDeploy."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The HF model to use for onnx export.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use when exporting the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The directory to save the exported ONNX model.",
    )
    args = parser.parse_args()

    print(f"Constructing model from {args.model}")

    # to enable dynamic batch_size, the batch size must > 1
    # NOTE(yoco): Originally this is 2, however, don't know why, when set to 2,
    # the batch_size will collapse static int 2 even we explicitly it is dynamic axis.
    # And more weird, when set to 13, the batch_size will be dynamic.
    # Probably some value between 2 and 13 will work,
    # We use 13 here for debugging purpose.
    max_batch_size = 13
    max_seq_len = 4

    # Prepare the AutoDeploy config, mode is export_edgellm_onnx
    ad_config = LlmArgs(
        model=args.model,
        mode="export_edgellm_onnx",
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        device=args.device,
    )
    ad_config.attn_backend = "torch"
    if args.output_dir is not None:
        ad_config.transforms["export_to_onnx"]["output_dir"] = args.output_dir

    # Use direct InferenceOptimizer instead of LLM to avoid executor initialization
    export_onnx(ad_config)


if __name__ == "__main__":
    main()
