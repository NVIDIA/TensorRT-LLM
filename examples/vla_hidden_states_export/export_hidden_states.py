# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Build a TRT-LLM engine with full_hidden_states output (v0.x TRT backend).

NOTE: This script applies to the legacy TRT backend (v0.7-v0.21) which uses
`trtllm-build`. On v1.x (main branch), the build command has changed to
`trtllm serve` / `trtllm bench`. For v1.x, refer to Solution B in the README
(modify model forward to return hidden_states dict).

Prerequisites for v0.x:
    1. Apply patches/modeling_utils_v0x.patch to tensorrt_llm/models/modeling_utils.py
    2. This adds `hidden_states.mark_output('full_hidden_states', ...)` before
       gather_last_token_logits, preserving the full 3D tensor.

Verify the engine has 'full_hidden_states' as an output after build:

    python -c "
    import tensorrt as trt
    runtime = trt.Runtime(trt.Logger())
    with open('rank0.engine', 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    outputs = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    print('Outputs:', outputs)
    assert 'full_hidden_states' in outputs, 'mark_output patch not applied!'
    "
"""
