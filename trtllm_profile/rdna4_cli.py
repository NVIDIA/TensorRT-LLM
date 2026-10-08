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
"""Select ROCm before importing the product, including for CPU-side doctor diagnostics."""

import os


def main() -> None:
    if os.environ.get("TRTLLM_BACKEND", "auto").lower() not in ("auto", "rocm"):
        raise SystemExit("trtllm-rdna4 requires the ROCm backend; use TRTLLM_BACKEND=rocm or auto")
    os.environ["TRTLLM_BACKEND"] = "rocm"
    from tensorrt_llm.rocm.cli import main as run

    run()


__all__ = ["main"]
