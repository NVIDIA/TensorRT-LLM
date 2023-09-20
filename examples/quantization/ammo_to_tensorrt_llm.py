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
"""An example to convert an AMMO exported model to tensorrt_llm."""

import argparse

from ammo.deploy.llm import load_model_configs, model_config_to_tensorrt_llm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument("--max_output_len", type=int, default=512)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--engine_dir", type=str, default="/tmp/ammo")
    parser.add_argument("--gpus", type=int, default=1)

    return parser.parse_args()


def main(args):
    model_configs = load_model_configs(args.model_config,
                                       inference_tensor_parallel=args.gpus)

    model_config_to_tensorrt_llm(
        model_configs,
        args.engine_dir,
        gpus=len(model_configs),
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
