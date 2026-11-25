# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine step-loop hooks every DeepSeek-V4.1 target declares (``runtime_hooks``).

They are the built-in model's: ``DeepseekV4ForCausalLM.runtime_hooks`` (``_DEEPSEEK_V4_RUNTIME_HOOKS`` /
``_DEEPSEEK_V4_SPEC_RUNTIME_HOOKS`` in ``models/modeling_deepseekv4.py``), which ``DeepseekV41ForCausalLM``
inherits unchanged. Without them the engine runs its all-off default, so a target would pay the host and
launch work the built-in model skips:

* ``stage_decode_inputs``: a decode step's attention metadata goes up through a private copy stream, and a graph
  replay's ids, positions and speculative inputs in one launch each;
* ``prioritize_decode_graphs``: decode CUDA graphs (at most 256 tokens per step) are captured on a high-priority
  stream;
* ``prompts_to_owners``: under multi-rank attention DP, rank 0 sends each admitted prompt only to its owner rank;

and with one-model speculative decoding (DSpark) also

* ``loop_on_execution_stream``: the overlap loop's own stream work and the forward are ordered on one stream;
* ``skip_adp_count_gather``: the attention-DP graph-shape exchange agrees on one captured generation-only graph,
  so the step skips the all-gather of the input counts;
* ``skip_verify_row_gather``: a generation-only step verifies its hidden states in place (every row, in order).
"""

from __future__ import annotations

import dataclasses

from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

STEP_HOOKS = RuntimeHooks(
    stage_decode_inputs=True,
    prioritize_decode_graphs=True,
    prompts_to_owners=True,
)
SPEC_STEP_HOOKS = dataclasses.replace(
    STEP_HOOKS,
    loop_on_execution_stream=True,
    skip_adp_count_gather=True,
    skip_verify_row_gather=True,
)


def step_hooks(spec_config) -> RuntimeHooks:
    """The built-in model's hooks for a model whose speculative config is ``spec_config`` (None: no speculation)."""
    return SPEC_STEP_HOOKS if spec_config is not None else STEP_HOOKS
