# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import os

# isort: off
import torch
import tensorrt as trt
# isort: on
from pipeline_tensorrt_llm_stable_diffusion import \
    TensorRTLLMStableDiffusionPipeline

import tensorrt_llm
from tensorrt_llm.utils import deserialize_engine


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompt',
        default='a photo of an astronaut riding a horse on mars',
        type=str)
    parser.add_argument('--model_dir',
                        default='stable-diffusion-v1-5',
                        type=str)
    parser.add_argument('--image', default='image.png', type=str)
    parser.add_argument(
        '--unet_engine',
        default=
        'stable_diffusion_outputs/stable_diffusion_float16_tp1_rank0.engine',
        type=str)
    parser.add_argument('--log_level', type=str, default='warning')

    return parser.parse_args()


def is_fp16_engine(engine_file):
    engine = deserialize_engine(engine_file)
    inspector = engine.create_engine_inspector()
    engine_info = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON)
    del inspector
    return bool(engine_info.find("FP16 format") != -1)


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    assert os.path.exists(
        args.unet_engine), "Engine file {} doesn't exists!".format(
            args.unet_engine)

    torch_dtype = torch.float16 if is_fp16_engine(
        args.unet_engine) else torch.float32

    pipe = TensorRTLLMStableDiffusionPipeline.from_pretrained(
        args.model_dir, torch_dtype=torch_dtype)
    pipe = pipe.to("cuda")
    pipe.unet_engine_path = args.unet_engine
    prompt = args.prompt
    image = pipe(prompt).images[0]

    image.save(args.image)
