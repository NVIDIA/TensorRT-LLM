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
from collections import OrderedDict

from diffusers import UNet2DConditionModel

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.models.unet.unet_2d_condition import \
    UNet2DConditionModel as TRTUNet
from tensorrt_llm.models.unet.weights import load_from_hf_unet
from tensorrt_llm.network import net_guard
from tensorrt_llm.utils import str_dtype_to_trt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        default='stable-diffusion-v1-5/unet',
                        type=str)
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--timing_cache', type=str, default='model.cache')
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--output_dir',
                        type=str,
                        default='stable_diffusion_outputs')
    return parser.parse_args()


def prepare_inputs(args):
    dtype = str_dtype_to_trt(args.dtype)
    # Prepare inputs
    sample = tensorrt_llm.Tensor(name='sample',
                                 dtype=dtype,
                                 shape=[2, 4, 64, 64],
                                 dim_range=OrderedDict([
                                     ('batch_size', [2]),
                                     ('channel', [4]),
                                     ('height', [64]),
                                     ('width', [64]),
                                 ]))
    timestep = tensorrt_llm.Tensor(name='timestep',
                                   dtype=dtype,
                                   shape=[2],
                                   dim_range=OrderedDict([
                                       ('batch_size', [2]),
                                   ]))
    ehs = tensorrt_llm.Tensor(name='ehs',
                              dtype=dtype,
                              shape=[2, 77, 768],
                              dim_range=OrderedDict([
                                  ('batch_size', [2]),
                                  ('length', [77]),
                                  ('hidden_size', [768]),
                              ]))
    return (sample, timestep, ehs)


def build(args):
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    tensorrt_llm.set_default_dtype(args.dtype)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    builder = Builder()
    builder_config = builder.create_builder_config(
        name='stable_diffusion',
        precision=args.dtype,
        output_dir=args.output_dir,
        timing_cache=args.timing_cache,
        profiling_verbosity=args.profiling_verbosity)

    # Initialize Module
    tensorrt_llm_unet = TRTUNet(cross_attention_dim=768)
    if args.model_dir is not None:
        hf_unet = UNet2DConditionModel(layers_per_block=2,
                                       cross_attention_dim=768).from_pretrained(
                                           args.model_dir)
        load_from_hf_unet(hf_unet, tensorrt_llm_unet)

    # Module -> Network
    network = builder.create_network()
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_unet.named_parameters())

        # Forward
        inputs = prepare_inputs(args)
        sample_out = tensorrt_llm_unet(*inputs)

        # Mark outputs
        sample_out.mark_output('sample_out', str_dtype_to_trt(args.dtype))

    tensorrt_llm.graph_rewriting.optimize(network)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    with open(os.path.join(args.output_dir, 'stable_diffusion.engine'),
              'wb') as f:
        f.write(engine())


if __name__ == '__main__':
    args = parse_arguments()
    build(args)
