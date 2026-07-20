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
#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_lora")
    parser.add_argument("output")
    parser.add_argument("num_loras", type=int)

    args = parser.parse_args()

    lora_path = Path(args.input_lora)
    weights_path = lora_path / "model.lora_weights.npy"
    config_path = lora_path / "model.lora_config.npy"

    weights = np.load(weights_path)
    config = np.load(config_path)

    for i in range(args.num_loras):
        out_path = Path(args.output) / str(i)
        os.makedirs(out_path, exist_ok=True)
        w = np.random.normal(0, 2, weights.shape).astype(weights.dtype)
        np.save(out_path / "model.lora_weights.npy", w)
        np.save(out_path / "model.lora_config.npy", config)


if __name__ == "__main__":
    main()
