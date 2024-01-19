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
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.resolve() /
                    "../examples/gpt/utils"))  # more precise, avoid confusion
from convert import generate_int8


def dist(x, y):
    x = x.flatten().astype(float)
    y = y.flatten().astype(float)

    l2_x = np.linalg.norm(x)
    l2_y = np.linalg.norm(y)
    l2_diff = np.linalg.norm(x - y)

    cos = np.dot(x, y) / (l2_x * l2_y)
    angle = np.arccos(cos.clip(-1, 1)) * 180 / np.pi

    return l2_diff, angle


class TestINT8Export(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)
        w = 2 * (self.rng.random([128, 256]) - 0.5)  # weights in [-1, 1]
        x = 10 * (self.rng.random([5, 128]) - 0.5)  # x in [-5, 5]
        y = x @ w

        ranges = {
            "x": torch.from_numpy(np.abs(x).max(axis=0)),
            "y": torch.from_numpy(np.abs(y).max(axis=0)),
            "w": torch.from_numpy(np.abs(w).max(axis=0)),
        }
        values = generate_int8(w, ranges)

        self.x, self.y, self.w = x, y, w
        self.values = values

    def test_weight_quantization(self):
        w_angle = dist(
            self.values["weight.int8"] * self.values["scale_w_quant_orig"],
            self.w)[1]
        w_angle_col = dist(
            self.values["weight.int8.col"] *
            self.values["scale_w_quant_orig.col"], self.w)[1]

        self.assertTrue(np.abs(w_angle) < 0.5)
        self.assertTrue(np.abs(w_angle_col) < 0.5)
        self.assertTrue(np.abs(w_angle_col) < np.abs(w_angle))

    def test_e2e_gemm_quantization(self):
        # mimic what CUTLASS would do
        x_i8 = (self.x * self.values["scale_x_orig_quant"]).round().clip(
            -127, 127)
        y_i32 = x_i8 @ self.values["weight.int8"].astype(np.int32)
        y_quant = y_i32 * self.values["scale_y_accum_quant"] * self.values[
            "scale_y_quant_orig"]
        y_angle = dist(self.y, y_quant)[1]

        y_i32_col = x_i8 @ self.values["weight.int8.col"].astype(np.int32)
        y_quant_col = y_i32_col * self.values[
            "scale_y_accum_quant.col"] * self.values["scale_y_quant_orig"]
        y_angle_col = dist(self.y, y_quant_col)[1]

        self.assertTrue(np.abs(y_angle) < 0.5)
        self.assertTrue(np.abs(y_angle_col) < 0.5)
        self.assertTrue(np.abs(y_angle_col) < np.abs(y_angle))


if __name__ == '__main__':
    unittest.main()
