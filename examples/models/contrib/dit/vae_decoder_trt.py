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

import torch
from diffusers.models import AutoencoderKL


class TRT_Exporter(object):

    def __init__(self, pytorch_model, max_batch_size, latent_channel,
                 latent_shape):
        self.pytorch_model = pytorch_model
        self.max_batch_size = max_batch_size
        self.latent_channel = latent_channel
        self.latent_shape = latent_shape

    def export_onnx(self, onnxFile):
        print(f"Start exporting ONNX model to {onnxFile}!")
        latent = torch.randn(self.max_batch_size, self.latent_channel,
                             *self.latent_shape).cuda()
        self.pytorch_model.cuda().eval()
        with torch.inference_mode():
            torch.onnx.export(self.pytorch_model,
                              latent,
                              onnxFile,
                              opset_version=17,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={'input': {
                                  0: 'batch'
                              }})

    def generate_trt_engine(self, onnxFile, planFile):
        print(f"Start exporting TRT model to {planFile}!")
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
        nMinBS = 1
        nMaxBS = self.max_batch_size
        nOptBS = (nMaxBS + nMinBS) // 2
        inputT = network.get_input(0)
        inputT.shape = [nBS, self.latent_channel, *self.latent_shape]
        profile.set_shape(inputT.name,
                          [nMinBS, self.latent_channel, *self.latent_shape],
                          [nOptBS, self.latent_channel, *self.latent_shape],
                          [nMaxBS, self.latent_channel, *self.latent_shape])

        config.add_optimization_profile(profile)

        t0 = time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time()
        if engineString is None:
            print("Failed building %s" % planFile)
        else:
            print("Succeeded building %s in %d s" % (planFile, t1 - t0))
        print("plan file is", planFile)
        with open(planFile, 'wb') as f:
            f.write(engineString)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae",
                        type=str,
                        choices=["ema", "mse"],
                        default="mse")
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument("--image-size",
                        type=int,
                        choices=[256, 512],
                        default=512)
    parser.add_argument('--onnxFile',
                        type=str,
                        default='vae_decoder/onnx/visual_encoder.onnx',
                        help='')
    parser.add_argument('--planFile',
                        type=str,
                        default='vae_decoder/plan/visual_encoder_fp16.plan',
                        help='')
    parser.add_argument('--only_trt',
                        action='store_true',
                        help='Run only convert the onnx to TRT engine.')
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

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    vae.forward = vae.decode
    latant_shape = [args.image_size // 8] * 2
    onnx_trt_obj = TRT_Exporter(vae, args.max_batch_size, 4, latant_shape)
    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile)
    else:
        onnx_trt_obj.export_onnx(args.onnxFile)
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile)
