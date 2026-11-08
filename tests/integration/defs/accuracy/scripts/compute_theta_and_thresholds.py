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

import pandas
import scipy


def compute_theta(num_samples: int,
                  sigma: float,
                  alpha: float = 0.05,
                  beta: float = 0.2):
    scale = (2 * sigma**2 / num_samples)**0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    z_beta = scipy.stats.norm.ppf(beta)
    theta = -(z_alpha + z_beta) * scale
    return theta


def compute_threshold(num_samples: int,
                      ref_accuracy: float,
                      sigma: float,
                      alpha: float = 0.05,
                      higher_is_better: bool = True):
    scale = (2 * sigma**2 / num_samples)**0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    if higher_is_better:
        return ref_accuracy + z_alpha * scale
    else:
        return ref_accuracy - z_alpha * scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples_total", type=int, default=8192)
    parser.add_argument("--sigma", type=float, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()

    data = []
    num_samples = 32
    while num_samples < args.num_samples_total:
        theta = compute_theta(num_samples,
                              args.sigma,
                              alpha=args.alpha,
                              beta=args.beta)
        threshold = compute_threshold(num_samples,
                                      0,
                                      args.sigma,
                                      alpha=args.alpha)
        data.append([num_samples, theta, threshold])
        num_samples *= 2

    num_samples = args.num_samples_total
    theta = compute_theta(num_samples,
                          args.sigma,
                          alpha=args.alpha,
                          beta=args.beta)
    threshold = compute_threshold(num_samples, 0, args.sigma, alpha=args.alpha)
    data.append([num_samples, theta, threshold])

    df = pandas.DataFrame(
        data, columns=['num_samples', 'theta', 'threshold-reference'])
    df = df.set_index('num_samples')
    print(df)
