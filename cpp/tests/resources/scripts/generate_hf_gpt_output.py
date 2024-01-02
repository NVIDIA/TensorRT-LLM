#!/usr/bin/env python3
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

from pathlib import Path

import run_hf


def generate_hf_output(data_type: str,
                       output_name: str,
                       max_output_len: int = 8):

    model = 'gpt2'
    resources_dir = Path(__file__).parent.resolve().parent
    models_dir = resources_dir / 'models'
    model_dir = models_dir / model

    data_dir = resources_dir / 'data'
    input_file = data_dir / 'input_tokens.npy'
    output_dir = data_dir / model / 'huggingface'

    run_hf.generate(model_dir=str(model_dir),
                    data_type=data_type,
                    input_file=str(input_file),
                    output_npy=str(output_dir / (output_name + '.npy')),
                    output_csv=str(output_dir / (output_name + '.csv')),
                    max_output_len=max_output_len)


def generate_hf_outputs():
    generate_hf_output(data_type='fp32',
                       output_name='output_tokens_fp32_huggingface')
    generate_hf_output(data_type='fp16',
                       output_name='output_tokens_fp16_huggingface')


if __name__ == '__main__':
    generate_hf_outputs()
