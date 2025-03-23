# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import pandas

sys.path.append(f"{os.path.dirname(__file__)}/..")
from accuracy_core import compute_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_total", type=int, default=14042)
    parser.add_argument("--sigma", type=float, default=50)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()

    # MMLU
    # num_total = 14042
    # sigma = 49

    # CNN Dailymail
    # num_total = 11490
    # sigma = 11.06

    data = []
    for alpha in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        num_samples = 32
        while num_samples <= args.num_total:
            threshold, theta = compute_threshold(num_samples,
                                                 0,
                                                 sigma=args.sigma,
                                                 alpha=alpha,
                                                 beta=args.beta)
            data.append([alpha, num_samples, threshold, theta])
            num_samples *= 2

        threshold, theta = compute_threshold(args.num_total, 0, args.sigma,
                                             alpha, args.beta)
        data.append([alpha, args.num_total, threshold, theta])

    df = pandas.DataFrame(
        data, columns=['alpha', 'num_samples', 'threshold', 'theta'])
    df = df.set_index(['alpha', 'num_samples']).unstack()

    print("===========================================================\n"
          "= THETA (MINIMUM DETECTABLE EFFECT)\n"
          "===========================================================")
    print(df['theta'])
    print("===========================================================\n"
          "= GAMMA - REFERENCE\n"
          "===========================================================")
    print(df['threshold'])
