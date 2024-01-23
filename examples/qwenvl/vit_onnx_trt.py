# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List

import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForCausalLM

from tensorrt_llm._utils import str_dtype_to_torch


class Preprocss:

    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def encode(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith(
                    "https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return images


class ONNX_TRT:

    def __init__(self, image_size):
        self.image_size = image_size

    def export_onnx(self, onnx_file_path, pretrained_model_path, image_url):
        print("Start converting ONNX model!")
        image_pre_obj = Preprocss(self.image_size)
        torch_dtype = str_dtype_to_torch("float16")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="cuda",
            torch_dtype=torch_dtype,
            fp16=True,
            trust_remote_code=True,
        ).eval()
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        image = image_pre_obj.encode(image_url).to(device)
        if not os.path.exists('image.pt'):
            torch.save(image, 'image.pt')

        model_visual = model.transformer.visual
        model_visual.eval()

        torch.onnx.export(model_visual,
                          image.to('cuda'),
                          onnx_file_path,
                          opset_version=17,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {
                              0: 'batch'
                          }})

    def generate_trt_engine(self,
                            onnxFile,
                            planFile,
                            minBS=1,
                            optBS=2,
                            maxBS=4):
        print("Start converting TRT engine!")
        from time import time

        import tensorrt as trt
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnxFile, 'rb') as model:
            if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                print("Failed parsing %s" % onnxFile)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnxFile)

        nBS = -1
        nMinBS = minBS
        nOptBS = optBS
        nMaxBS = maxBS
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, self.image_size, self.image_size]
        profile.set_shape(inputT.name,
                          [nMinBS, 3, self.image_size, self.image_size],
                          [nOptBS, 3, self.image_size, self.image_size],
                          [nMaxBS, 3, self.image_size, self.image_size])

        config.add_optimization_profile(profile)

        t0 = time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time()
        if engineString == None:
            print("Failed building %s" % planFile)
        else:
            print("Succeeded building %s in %d s" % (planFile, t1 - t0))
        print("plan file is", planFile)
        with open(planFile, 'wb') as f:
            f.write(engineString)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # onnx/visual_encoder
    parser.add_argument('--onnxFile',
                        type=str,
                        default='visual_encoder/visual_encoder.onnx',
                        help='')
    parser.add_argument('--pretrained_model_path',
                        type=str,
                        default='Qwen-VL-Chat',
                        help='')
    parser.add_argument('--planFile',
                        type=str,
                        default='plan/visual_encoder/visual_encoder_fp16.plan',
                        help='')
    parser.add_argument('--only_trt',
                        action='store_true',
                        help='Run only convert the onnx to TRT engine.')
    parser.add_argument('--minBS', type=int, default=1)
    parser.add_argument('--optBS', type=int, default=1)
    parser.add_argument('--maxBS', type=int, default=4)
    parser.add_argument('--image_url', type=list, default=['./pics/demo.jpeg'])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    onnx_file_dir = os.path.dirname(args.onnxFile)
    if not onnx_file_dir == '' and not os.path.exists(onnx_file_dir):
        os.makedirs(onnx_file_dir)
    plan_file_dir = os.path.dirname(args.planFile)
    if not os.path.exists(plan_file_dir):
        os.makedirs(plan_file_dir)

    onnx_trt_obj = ONNX_TRT(448)  # or ONNX_TRT(config.visual['image_size'])

    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile,
                                         args.minBS, args.optBS, args.maxBS)
    else:
        onnx_trt_obj.export_onnx(args.onnxFile, args.pretrained_model_path,
                                 args.image_url)
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile,
                                         args.minBS, args.optBS, args.maxBS)
