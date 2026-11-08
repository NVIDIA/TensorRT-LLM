# MIT License

# Copyright (c) 2023 OpenBMB

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

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

# reference: https://github.com/OpenBMB/InfiniteBench/blob/main/src/args.py

from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help=
        "Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument('--data_dir',
                   type=str,
                   default='../data',
                   help="The directory of data.")
    p.add_argument("--output_dir",
                   type=str,
                   default="../results",
                   help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        help=
        "The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it will default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help=
        "The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data."
    )  # noqa
    p.add_argument(
        "--stop_idx",
        type=int,
        help=
        "The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset."
    )  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--preds_file",
        type=str,
        help="The path of prediction file.",
    )
    return p.parse_args()
