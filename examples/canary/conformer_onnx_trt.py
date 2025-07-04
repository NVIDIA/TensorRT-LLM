# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import time

import click
import tensorrt as trt
import torch


class ConformerTRT:

    def __init__(self,
                 checkpoint_dir,
                 max_feat_len,
                 min_feat_len=32,
                 opt_feat_len=None,
                 minBS=1,
                 optBS=None,
                 maxBS=4):
        self.checkpoint_dir = checkpoint_dir
        with open(os.path.join(checkpoint_dir, 'encoder/config.json'),
                  'r') as f:
            self.encoder_config = json.load(f)

        self.dtype = self.encoder_config['dtype']

        if opt_feat_len is None:
            opt_feat_len = int((max_feat_len + min_feat_len) / 2)

        if optBS is None:
            optBS = int((minBS + maxBS) / 2)

        if opt_feat_len > max_feat_len or opt_feat_len < min_feat_len:
            raise Exception(
                f"Invalid opt_feat_len should be min_feat_len < opt_feat_len < max_feat_len "
            )

        if optBS > maxBS or optBS < minBS:
            raise Exception(f"Invalid optBS should be minBS < optBS < maxBS")

        self.feat_dim = self.encoder_config['feat_in']

        self.min_feat_len = min_feat_len
        self.opt_feat_len = opt_feat_len

        self.max_feat_len = max_feat_len

        self.minBS = minBS
        self.optBS = optBS
        self.maxBS = maxBS

        self.encoder_config['min_feat_len'] = min_feat_len
        self.encoder_config['opt_feat_len'] = opt_feat_len
        self.encoder_config['max_feat_len'] = max_feat_len

        self.encoder_config['min_batch_size'] = minBS
        self.encoder_config['opt_batch_size'] = optBS
        self.encoder_config['max_batch_size'] = maxBS

    def generate_trt_engine(self, engine_dir):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if self.dtype == "bfloat16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif self.dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)

        #config.flags = config.flags
        parser = trt.OnnxParser(network, logger)
        onnx_file = os.path.join(self.checkpoint_dir, 'encoder/encoder.onnx')

        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
                print("Failed parsing %s" % onnx_file)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnx_file)

        nBS = -1
        nFeats = -1
        nMinBS = self.minBS

        nMaxBS = self.maxBS

        nOptBS = self.optBS

        input_feat = network.get_input(0)
        input_len = network.get_input(1)
        input_feat.shape = [nBS, self.feat_dim, nFeats]
        input_len.shape = [nBS]
        profile.set_shape(
            input_feat.name,
            [nMinBS, self.feat_dim, self.min_feat_len],
            [nOptBS, self.feat_dim, self.opt_feat_len],
            [nMaxBS, self.feat_dim, self.max_feat_len],
        )
        profile.set_shape(
            input_len.name,
            [nMinBS],
            [nOptBS],
            [nMaxBS],
        )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        plan_path = os.path.join(engine_dir, "encoder")
        os.makedirs(plan_path, exist_ok=True)

        plan_file = os.path.join(plan_path, 'encoder.plan')
        config_file = os.path.join(plan_path, 'config.json')


        if engineString == None:
            print("Failed building %s" % plan_file)
        else:
            print("Succeeded building %s in %d s" % (plan_file, t1 - t0))
            with open(plan_file, "wb") as f:
                f.write(engineString)
            with open(config_file, 'w') as jf:
                json.dump(self.encoder_config, jf)



@click.command()
@click.option("--min_BS", default=1, type=int, help="Minimum batch size")
@click.option("--opt_BS", default=None, type=int, help="Optimum batch size")
@click.option("--max_BS", default=4, type=int, help="Maximum batch size")
@click.option("--max_feat_len",
              default=3001,
              type=int,
              help="Maximum input length")
@click.option("--min_feat_len",
              default=32,
              type=int,
              help="Minimum input length")
@click.option("--opt_feat_len",
              default=None,
              type=int,
              help="Optimum input length")
@click.argument("checkpoint_dir",
                type=click.Path(exists=True, dir_okay=True, file_okay=False),
                required=True)
@click.argument("engine_dir", type=str, required=True)
def main(checkpoint_dir, engine_dir, max_feat_len, min_feat_len, opt_feat_len,
         min_bs, opt_bs, max_bs):

    conformer = ConformerTRT(checkpoint_dir, max_feat_len, min_feat_len,
                             opt_feat_len, min_bs, opt_bs, max_bs)

    conformer.generate_trt_engine(engine_dir)


if __name__ == "__main__":
    main()
