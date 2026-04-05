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
    """CLI entry point for exporting HuggingFace models to ONNX format."""
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
        help="The directory to save the exported ONNX model (language / text model).",
    )
    parser.add_argument(
        "--visual_output_dir",
        type=str,
        default=None,
        help="Optional. For VLM, the directory to save the exported vision ONNX (vision_model.onnx). "
        "When set, the pipeline may write visual subgraph to this dir if supported.",
    )
    parser.add_argument(
        "--model_factory",
        type=str,
        default=None,
        help="Model factory name (e.g. AutoModelForCausalLM, AutoModelForImageTextToText). "
        "Default is AutoModelForCausalLM. For VLM, use AutoModelForImageTextToText or --vlm.",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Use AutoModelForImageTextToText factory for vision-language models. "
        "Equivalent to --model_factory AutoModelForImageTextToText.",
    )
    args = parser.parse_args()

    model_factory = (
        "AutoModelForImageTextToText"
        if args.vlm
        else (args.model_factory or "AutoModelForCausalLM")
    )

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
        model_factory=model_factory,
    )
    ad_config.attn_backend = "torch"
    if args.output_dir is not None:
        ad_config.transforms["export_to_onnx"]["output_dir"] = args.output_dir
        ad_config.transforms["rewrite_embedding_to_inputs_embeds"]["output_dir"] = args.output_dir
        if "extract_embedding_to_safetensors" in ad_config.transforms:
            ad_config.transforms["extract_embedding_to_safetensors"]["output_dir"] = args.output_dir
    if args.visual_output_dir is not None and "export_vision_to_onnx" in ad_config.transforms:
        ad_config.transforms["export_vision_to_onnx"]["visual_output_dir"] = args.visual_output_dir

    # Use direct InferenceOptimizer instead of LLM to avoid executor initialization
    export_onnx(ad_config)


if __name__ == "__main__":
    main()
