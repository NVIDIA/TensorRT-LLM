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
"""Lazy CLI dispatch: ROCm commands never import NVIDIA command dependencies."""

from tensorrt_llm import _BACKEND


def serve() -> None:
    if _BACKEND == "rocm":
        from tensorrt_llm.rocm.cli import serve_main

        serve_main()
    else:
        from .serve import main

        main()


def bench() -> None:
    if _BACKEND == "rocm":
        from tensorrt_llm.rocm.cli import bench_main

        bench_main()
    else:
        from .bench import main

        main()


def evaluate() -> None:
    if _BACKEND == "rocm":
        raise SystemExit(
            "The upstream NVIDIA evaluation suite is not ported. Use trtllm-rdna4 validate for ROCm correctness tests."
        )
    from .eval import main

    main()
