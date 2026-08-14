<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Examples

Concrete usage examples for the host performance optimization skill.

---

## Example 1: Basic Line Profiler Benchmark

```bash
#!/bin/bash
set -e

SCRIPT_DIR=$(dirname $(realpath "$0"))
mkdir -p $SCRIPT_DIR/lp_results

export LLM_MODELS_ROOT=/path/to/models
export MODEL_PATH=$LLM_MODELS_ROOT/Llama-3.1-8B-Instruct
export ISL=1024
export OSL=1024
export TP_SIZE=4
export CONCURRENCY=128

export TLLM_LINE_PROFILER_ENABLED=True
export TLLM_LINE_PROFILER_PATH=$SCRIPT_DIR/lp_results/profile_$(date +%Y%m%d_%H%M%S).txt

# Optional: profile additional functions not in the default set
# export TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.model_engine.PyTorchModelEngine._prepare_tp_inputs"

python -m tensorrt_llm.llm_api benchmark \
    --model $MODEL_PATH \
    --tp_size $TP_SIZE \
    --concurrency $CONCURRENCY \
    --input_len $ISL \
    --output_len $OSL

echo "Profile results saved to: $TLLM_LINE_PROFILER_PATH"
```

---

## Example 2: Profile Specific Functions

When you suspect a specific function is the bottleneck:

```bash
export TLLM_LINE_PROFILER_ENABLED=True
export TLLM_LINE_PROFILER_PATH=./specific_profile.txt

# Profile a class method
export TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.sampler.TorchSampler._process_requests"

# Profile a standalone function (use :: delimiter)
export TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.sampler::_group_requests_by_strategy_key"

# Profile multiple functions
export TLLM_LINE_PROFILER_FUNCTIONS="module.Class.method1,module.Class.method2"

python your_workload.py
```

---

## Example 3: Full Multi-Round Optimization Session

This is a real example from optimizing `_prepare_tp_inputs` on Llama 3.1 8B, TP=4, B200 GPUs.

```
Round 0 (Baseline):
  -> Profile with default functions
  -> Identify _forward_step spends 98.7% in model_engine.forward()
  -> forward() is NOT in the default profiler output
  -> Drill down: add _prepare_tp_inputs to TLLM_LINE_PROFILER_FUNCTIONS
  -> Re-profile to get line-level data inside _prepare_tp_inputs

Round 1 (REDUNDANT_ITER):
  -> Hotspot: previous_request_ids list comprehension (9.4s, 16.4%)
     Iterates over 279K generation requests to build a list,
     but the same requests were already iterated in the categorization loop
  -> Options:
     | Option | Description              | Savings | Risk |
     |--------|--------------------------|---------|------|
     | A      | Collect during existing loop | ~9.4s  | Low  |
     | B      | Set comparison instead     | ~small  | Med  |
     | C      | Incremental maintenance    | ~9.4s  | High |
  -> Selected: A (highest savings, lowest risk)
  -> Apply, run UTs, re-profile
  -> Result: _prepare_tp_inputs 57.7s -> 47.7s (-17.3%)

Round 2 (DEAD_WORK):
  -> Hotspot: MultimodalParams construction for text-only model (7.8s, 16.4%)
     Creates MultimodalParams, calls strip_for_generation(), checks has_content()
     for every generation request -- always returns False for Llama (text-only)
  -> Options:
     | Option | Description                          | Savings | Risk |
     |--------|--------------------------------------|---------|------|
     | A      | Guard with py_multimodal_data check  | ~7.8s   | Low  |
     | B      | Model-level is_multimodal flag       | ~7.8s   | Med  |
     | C      | Cache MultimodalParams per request   | partial | High |
  -> Selected: A (simple truthiness guard)
  -> Apply, run UTs, re-profile
  -> Result: _prepare_tp_inputs 47.7s -> 41.1s (-13.7%)

Round 3 (FUNCALL):
  -> Hotspot: has_cp_helix() per-request (1.5s) + first_beam=0 assignment (1.3s)
  -> Options:
     | Option | Description                              | Savings | Risk |
     |--------|------------------------------------------|---------|------|
     | A      | Cache has_cp_helix only                  | ~1.5s   | Low  |
     | B      | Cache has_cp_helix + simplify first_beam | ~2.8s   | Low  |
     | C      | B + ternary for beam_width               | ~4s     | Low  |
  -> Selected: C initially, but ternary was SLOWER (CPython pitfall!)
  -> Rolled back to B
  -> Result: _prepare_tp_inputs 41.1s -> 40.1s (-2.4%)

Final Summary:
  _prepare_tp_inputs: 57.7s -> 40.1s (-30.5% cumulative)
  Mean TPOT:          37.4ms -> 28.9ms (-22.6%)
  Output throughput:  ~3200 -> 3799 tok/s (+18.7%)
```

---

## Example 4: Workspace Suffix Convention

```bash
# Tag each round with a descriptive suffix
EXTRA_SUFFIX=round0_baseline bash profile.sh
EXTRA_SUFFIX=round1_eliminate_redundant_iter bash profile.sh
EXTRA_SUFFIX=round2_skip_multimodal bash profile.sh
EXTRA_SUFFIX=round3_cache_invariants bash profile.sh

# Use UPPER_WORKSPACE_DIR to isolate agentic runs from manual runs
UPPER_WORKSPACE_DIR=/path/to/workspace_agentic EXTRA_SUFFIX=round0_baseline bash profile.sh
```
