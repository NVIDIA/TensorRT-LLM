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

import subprocess
import sys
import time

if __name__ == '__main__':
    case = ''
    for arg in sys.argv[1:]:
        if '--gtest_filter=' in arg:
            case = arg.removeprefix('--gtest_filter=')

    gtest = subprocess.Popen(sys.argv[1:])

    if case:
        import multiprocessing.connection

        with multiprocessing.connection.Client("/tmp/profiling_scribe.unix",
                                               "AF_UNIX") as client:
            client.send({
                "type": "gtest_case",
                "timestamp": time.time(),
                "case": case,
                "pid": gtest.pid
            })

    gtest.wait()
    exit(gtest.returncode)
