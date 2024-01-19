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

import fileinput as _fi
import glob as _gl
import pathlib as _pl
import shutil as _sh
import sys


def copy_file(src_num: int, dst_num: int):
    src_files = _gl.glob(f"decoderMaskedMultiheadAttention{src_num}_*.cu")
    for src_file in src_files:
        dst_file = src_file.replace(
            f"decoderMaskedMultiheadAttention{src_num}",
            f"decoderMaskedMultiheadAttention{dst_num}")
        print(f"{src_file} -> {dst_file}")
        _sh.copyfile(src_file, dst_file)

        for line in _fi.input(dst_file, inplace=True):
            print(line.replace(f"kSizePerHead = {src_num}",
                               f"kSizePerHead = {dst_num}"),
                  end="")

    old_file = _pl.Path(f"decoderMaskedMultiheadAttention{dst_num}.cu")
    if old_file.exists():
        print(f"Removing {old_file}")
        old_file.unlink()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} <src_num> <dst_num>".format(sys.argv[0]))
        sys.exit(1)

    src_num = int(sys.argv[1])
    dst_num = int(sys.argv[2])
    if src_num == dst_num:
        print("src_num == dst_num")
        sys.exit(0)

    copy_file(src_num, dst_num)
