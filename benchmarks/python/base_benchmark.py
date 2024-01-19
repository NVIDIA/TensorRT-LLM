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
import json
import os
import subprocess
import time
from collections import OrderedDict

import torch

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode


def get_compute_cap():
    output = subprocess.check_output(
        ['nvidia-smi', "--query-gpu=compute_cap", "--format=csv"])
    _, csv_value, *_ = output.splitlines()
    return str(int(float(csv_value) * 10))


def get_csv_filename(model, dtype, tp_size, mode, **kwargs):
    sm = get_compute_cap()
    if len(kwargs) == 0:
        kw_pairs = ""
    else:
        kw_pairs = "_" + "_".join([str(k) + str(v) for k, v in kwargs.items()])
    return f'{model}_{dtype}_tp{tp_size}_{mode}{kw_pairs}_sm{sm}.csv'


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        # engine object is already complies with python buffer protocol, no need to
        # convert it to bytearray before write, converting to bytearray consumes lots of memory
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


class BaseBenchmark(object):

    def __init__(self,
                 engine_dir,
                 model_name,
                 dtype,
                 rank,
                 world_size,
                 serial_build: bool = False):
        self.engine_dir = engine_dir
        self.model_name = model_name
        self.dtype = dtype
        self.runtime_rank = rank
        self.world_size = world_size
        self.engine_model_name = model_name
        self.quant_mode = QuantMode(0)
        self.enable_fp8 = False
        if engine_dir is not None:
            # Read config from engine directory
            config_path = os.path.join(engine_dir, 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            # Sanity checks
            config_dtype = self.config['builder_config']['precision']
            assert dtype == config_dtype, f"Engine dtype ({config_dtype}) != Runtime dtype ({dtype})"
            world_size = self.config['builder_config']['tensor_parallel']
            assert world_size == self.world_size, \
                (f'Engine world size ({world_size}) != Runtime world size ({self.world_size})')
            # Load config into self
            for key, value in self.config['builder_config'].items():
                if key == "quant_mode":
                    self.quant_mode = QuantMode(value)
                elif key in "name":
                    self.engine_model_name = value
                else:
                    setattr(self, key, value)
            self.enable_fp8 = self.quant_mode.has_fp8_qdq()
            self.fp8_kv_cache = self.quant_mode.has_fp8_kv_cache()
            for key, value in self.config['plugin_config'].items():
                # Same effect as self.use_foo_plugin = config.json["foo_plugin"]
                if "plugin" in key:
                    key = "use_" + key
                setattr(self, key, value)

        self.engine_name = get_engine_name(self.engine_model_name, self.dtype,
                                           self.world_size, self.runtime_rank)
        self.runtime_mapping = tensorrt_llm.Mapping(world_size=self.world_size,
                                                    rank=self.runtime_rank,
                                                    tp_size=self.world_size)
        if not serial_build:
            torch.cuda.set_device(self.runtime_rank %
                                  self.runtime_mapping.gpus_per_node)

        self.csv_filename = ""  # lazy init

    def get_report_dict(self, benchmark_profiler=None):
        report_fields = [
            "model_name", "world_size", "num_heads", "num_kv_heads",
            "num_layers", "hidden_size", "vocab_size", "precision",
            "batch_size", "input_length", "output_length", "gpu_peak_mem(gb)",
            "build_time(s)", "tokens_per_sec", "percentile95(ms)",
            "percentile99(ms)", "latency(ms)", "compute_cap"
        ]
        report_dict = OrderedDict.fromkeys(report_fields)
        report_dict["model_name"] = self.model_name
        report_dict["world_size"] = self.world_size
        report_dict["precision"] = self.dtype
        report_dict["quantization"] = str(self.quant_mode)
        report_dict["compute_cap"] = "sm" + get_compute_cap()
        return report_dict

    def get_csv_filename(self):
        if len(self.csv_filename) == 0:
            self.csv_filename = get_csv_filename(self.model_name,
                                                 self.dtype,
                                                 self.world_size,
                                                 self.mode,
                                                 fp8linear=int(self.enable_fp8))
        return self.csv_filename

    def print_report_header(self, csv=False, benchmark_profiler=None):
        if csv and self.runtime_rank == 0:
            report_dict = self.get_report_dict(benchmark_profiler)
            line = ",".join(report_dict.keys())
            print(line)
            with open(self.get_csv_filename(), "a") as file:
                file.write(line + "\n")

    def get_config(self):
        raise NotImplementedError

    def prepare_inputs(self, config):
        raise NotImplementedError

    def run(self, inputs, config, benchmark_profiler=None):
        raise NotImplementedError

    def report(self, config, latency):
        raise NotImplementedError
