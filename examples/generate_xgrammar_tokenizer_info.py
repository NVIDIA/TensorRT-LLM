#!/usr/bin/env python3
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

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from tensorrt_llm.llmapi.tokenizer import _xgrammar_tokenizer_info


def generate_xgrammar_tokenizer_info(args):

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    tokenizer_info = _xgrammar_tokenizer_info(tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(str(args.output_dir / "xgrammar_tokenizer_info.json"), 'w') as f:
        json.dump(tokenizer_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=Path,
                        default=None,
                        required=True,
                        help="HF model directory")
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        required=True,
        help="File path to save xgrammar's info. in json format")
    args = parser.parse_args()
    generate_xgrammar_tokenizer_info(args)
