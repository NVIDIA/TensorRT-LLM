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

import argparse
import csv
import os

import numpy as np


def csv_to_npy(input_file, output_file, pad_id, verbose):
    data = []
    with open(input_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for line in csv_reader:
            data.append([int(e) for e in line])
    max_input_length = max([len(x) for x in data])
    data = [row + [pad_id] * (max_input_length - len(row)) for row in data]
    data = np.array(data, dtype='int32')
    if (verbose):
        print(data, data.dtype)
    np.save(output_file, data)


def npy_to_csv(input_file, output_file, verbose):
    data = np.load(input_file)
    if (verbose):
        print(data, data.dtype)
    np.savetxt(output_file, data, delimiter=",", fmt='%i')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_file',
        type=str,
        help='Read token ids from this file. Must be csv or npy.')
    parser.add_argument('output_file',
                        type=str,
                        help='Write token ids this file. Must be csv or npy.')
    parser.add_argument(
        '-p',
        '--pad_id',
        type=int,
        help=
        'Token id used for padding csv input with different sequence lengths.',
        default=-1)
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    _, input_ext = os.path.splitext(args.input_file)
    _, output_ext = os.path.splitext(args.output_file)

    if (input_ext == '.csv' and output_ext == '.npy'):
        print('Converting csv to npy')
        csv_to_npy(args.input_file, args.output_file, args.pad_id, args.verbose)
    elif (input_ext == '.npy' and output_ext == '.csv'):
        print('Converting npy to csv')
        npy_to_csv(args.input_file, args.output_file, args.verbose)
    else:
        print('unknown file extensions')
