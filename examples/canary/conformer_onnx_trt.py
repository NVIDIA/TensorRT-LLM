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

import tensorrt as trt

import click


class ConformerTRT:

    def __init__(self, config_file, min_feat_len=32, max_feat_len=12001, opt_feat_len=3001):

        with open(config_file,'r') as f:
            self.encoder_config = json.load(f)
        print(f"{config_file}: {self.encoder_config}")
        self.dtype=self.encoder_config['dtype']
        self.feat_in=self.encoder_config['feat_in']
        self.min_feat_len=min_feat_len
        self.max_feat_len=max_feat_len
        if opt_feat_len is None:
            opt_feat_len=int((self.max_feat_len+self.min_feat_len)/2)
        if opt_feat_len > self.max_feat_len or opt_feat_len < self.min_feat_len:
            raise Exception(f"Invalid opt_feat_len should be min_feat_len<opt_feat_len<max_feat_len ")
        self.opt_feat_len=opt_feat_len



    def generate_trt_engine(self,
                            onnx_file,
                            engine_dir,
                            minBS=1,
                            optBS=None,
                            maxBS=4):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if self.dtype=="bfloat16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif self.dtype=="float16":
            config.set_flag(trt.BuilderFlag.FP16)

        #config.flags = config.flags
        parser = trt.OnnxParser(network, logger)

        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
                print("Failed parsing %s" % onnx_file)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnx_file)

        nBS = -1
        nFeats = -1
        nMinBS = minBS

        nMaxBS = maxBS
        if optBS is None:
            optBS=int((minBS+maxBS)/2)
        nOptBS = optBS
        input_feat = network.get_input(0)
        input_len=network.get_input(1)
        input_feat.shape = [nBS, self.feat_in, nFeats]
        input_len.shape = [nBS]
        profile.set_shape(
            input_feat.name,
            [nMinBS, self.feat_in, self.min_feat_len],
            [nOptBS, self.feat_in, self.opt_feat_len],
            [nMaxBS, self.feat_in, self.max_feat_len],
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
        plan_path=os.path.join(engine_dir, "encoder")
        os.makedirs(plan_path, exist_ok=True)

        plan_file=os.path.join(plan_path,'encoder.plan')
        config_file=os.path.join(plan_path,'config.json')
        if engineString == None:
            print("Failed building %s" % plan_file)
        else:
            print("Succeeded building %s in %d s" % (plan_file, t1 - t0))
            with open(plan_file, "wb") as f:
                f.write(engineString)
            with open(config_file, 'w') as jf:
                json.dump(self.encoder_config,jf)





@click.command()
@click.option("--config_file", default="None")
@click.option("--min_BS", default=1, type=int, help="Minimum batch size")
@click.option("--opt_BS", default=None, type=int, help="Optimum batch size")
@click.option("--max_BS", default=4, type=int, help="Maximum batch size")
@click.argument("onnx_file", type=click.Path(exists=True),required=True)
@click.argument("engine_dir", type=str, required=True)
def main(onnx_file, engine_dir, config_file, min_bs, opt_bs, max_bs):
    if config_file == 'None':
        config_file = os.path.join(os.path.dirname(onnx_file), 'config.json')



    conformer = ConformerTRT(config_file)
    conformer.generate_trt_engine(onnx_file, engine_dir, min_bs, opt_bs, max_bs)


if __name__ == "__main__":
    main()

