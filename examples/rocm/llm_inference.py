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
"""Minimal portable inference; append --profile for source/component/resource reports."""

import argparse
import os

if os.environ.get("TRTLLM_BACKEND", "auto").lower() not in ("auto", "rocm"):
    raise SystemExit("This example requires TRTLLM_BACKEND=rocm or auto")
os.environ["TRTLLM_BACKEND"] = "rocm"

from tensorrt_llm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--kernels", choices=("torch", "hip"), default="torch")
    parser.add_argument("--attn-backend", choices=("sdpa", "eager", "hip"), default="sdpa")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    options = parser.parse_args()
    prompts = options.prompt or ["Explain wavefront execution briefly."]
    with LLM(
        model=options.model,
        device=options.device,
        dtype=options.dtype,
        kernels=options.kernels,
        attn_backend=options.attn_backend,
        max_batch_size=len(prompts),
        local_files_only=options.local_files_only,
    ) as engine:
        for result in engine.generate(
            prompts, SamplingParams(temperature=0, max_tokens=options.max_tokens)
        ):
            print(result.outputs[0].text)


if __name__ == "__main__":
    main()
