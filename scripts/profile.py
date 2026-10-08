#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Repository-local entry point for the universal --profile launcher."""

import sys
from pathlib import Path

# cProfile imports the standard-library module named 'profile'. Do not let
# this launcher shadow it through Python's initial script-directory entry.
_script_directory = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _script_directory]
sys.path.insert(0, str(_script_directory.parent))

from trtllm_profile.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
