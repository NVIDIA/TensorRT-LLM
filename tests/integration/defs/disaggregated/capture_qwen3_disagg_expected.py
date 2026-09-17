# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""One-off capture script for migrating test_disaggregated_simple_llama.

Migrates test_disaggregated_simple_llama in test_disaggregated_single_gpu.py
from TinyLlama-1.1B-Chat-v1.0 to Qwen3-0.6B.

test_disaggregated_simple_llama asserts exact hardcoded output token IDs
and text produced by a single (prompt, greedy-sampling) request split
across a context_only step and a generation_only step. Disaggregation
only splits prefill (1 token) and decode (the rest) across two workers;
with greedy decoding (temperature=0) and matching KV precision, the
resulting tokens are identical to a single plain generate() call, so this
script reproduces the same values without depending on the
MPIPoolExecutor control channel that module currently needs (broken on
Open MPI 5, see https://nvbugs/6770878, hence that whole module being
skipped).

Not a pytest test -- run directly:
    python3 capture_qwen3_disagg_expected.py

Requires LLM_MODELS_ROOT set and a single free GPU. Paste the printed
values into test_disaggregated_simple_llama's verify_disaggregated() call,
then delete this script.
"""

import os

from tensorrt_llm import LLM, SamplingParams

MODEL_PATH = os.path.join(os.environ["LLM_MODELS_ROOT"], "Qwen3", "Qwen3-0.6B")
PROMPT = "What is the capital of Germany?"
MAX_TOKENS = 25


def main():
    llm = LLM(
        model=MODEL_PATH,
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
    )

    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True, temperature=0)
    outputs = llm.generate([PROMPT], sampling_params)
    output = outputs[0].outputs[0]

    print("=== Paste these into test_disaggregated_simple_llama ===")
    print(f"expected_output_ids[0] (context_only) = {output.token_ids[0]!r}")
    print(f"expected_output (generation_only text) = {output.text!r}")
    print(f"expected_output_ids (generation_only)  = {list(output.token_ids)!r}")


if __name__ == "__main__":
    main()
