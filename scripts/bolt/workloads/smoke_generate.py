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
"""Minimal smoke workload for the BOLT flow.

Loads a model and runs a few generations to exercise the TRT-LLM host control
paths (executor, scheduler, KV cache manager) in libtensorrt_llm.so so the
instrumented libraries emit non-empty .fdata. This is a PLUMBING smoke test,
not a representative profiling workload.

Model is taken from $BOLT_SMOKE_MODEL (HF id or local path). Keep it small for
a single-node dry run.
"""

import os
import sys


def main() -> int:
    model = os.environ.get("BOLT_SMOKE_MODEL")
    if not model:
        print(
            "[ERROR] Set BOLT_SMOKE_MODEL to a (small) HF id or local model path, e.g.\n"
            "        export BOLT_SMOKE_MODEL=/models/Llama-3.1-8B-Instruct",
            file=sys.stderr,
        )
        return 2

    from tensorrt_llm import LLM, SamplingParams

    tp = int(os.environ.get("BOLT_SMOKE_TP", "1"))
    max_tokens = int(os.environ.get("BOLT_SMOKE_MAX_TOKENS", "64"))
    n_prompts = int(os.environ.get("BOLT_SMOKE_PROMPTS", "32"))

    print(f"[INFO] smoke: model={model} tp={tp} max_tokens={max_tokens} prompts={n_prompts}")
    llm = LLM(model=model, tensor_parallel_size=tp)
    try:
        prompts = ["Explain in one sentence why code locality matters."] * n_prompts
        outputs = llm.generate(prompts, SamplingParams(max_tokens=max_tokens))
        print(f"[INFO] smoke: generated {len(outputs)} completion(s)")
        if outputs:
            print(f"[INFO] sample: {outputs[0].outputs[0].text[:120]!r}")
    finally:
        # Ensure clean shutdown so instrumented libraries flush their .fdata.
        del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
