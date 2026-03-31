# Simulation Mode Phase 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run TinyLlama through the real TRT-LLM Python scheduler with mocked model execution — no real forward pass, requests complete with dummy tokens.

**Architecture:** Inject `SimModelEngine` (returns dummy logits) and `SimSampler` (advances request state with dummy tokens) into `PyExecutor` via the existing `create_py_executor()` construction path. `PyExecutor` and the scheduler are completely unmodified.

**Tech Stack:** Python, PyTorch (tensor creation only), TRT-LLM PyExecutor/Scheduler internals, C++ nanobind LlmRequest API.

**Spec:** `slop/specs/2026-03-30-simulation-mode-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `tensorrt_llm/llmapi/llm_args.py` | Modify | Add `simulation_mode: bool` field to `TorchLlmArgs` |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Create | `SimModelEngine(ModelEngine)` — dummy logits, no GPU forward |
| `tensorrt_llm/_torch/pyexecutor/sim_sampler.py` | Create | `SimSampler(Sampler)` — dummy tokens, state advancement |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Modify | Branch on `simulation_mode` in `create_py_executor()` |

---

### Task 1: Add `simulation_mode` flag to `TorchLlmArgs`

**Files:**
- Modify: `tensorrt_llm/llmapi/llm_args.py:3485` (insert before the `@property` block at line 3487)

- [ ] **Step 1: Add the field**

Open `tensorrt_llm/llmapi/llm_args.py` and add the following field to `TorchLlmArgs`, right after the `video_pruning_rate` field (around line 3485) and before the `@property def quant_config` block:

```python
    simulation_mode: bool = Field(
        default=False,
        description="Enable simulation mode. Skips model weight loading and "
        "replaces model forward with dummy outputs. "
        "The scheduler runs normally.",
        status="prototype",
    )
```

- [ ] **Step 2: Verify the field is accessible**

Run a quick smoke test:

```bash
cd /home/gvenkatarama/trtllm-wt/hisim-port/TensorRT-LLM
python -c "from tensorrt_llm.llmapi.llm_args import TorchLlmArgs; args = TorchLlmArgs(simulation_mode=True); print(f'simulation_mode={args.simulation_mode}')"
```

Expected output: `simulation_mode=True`

- [ ] **Step 3: Commit**

```bash
git add tensorrt_llm/llmapi/llm_args.py
git commit -s -m "[None][feat] Add simulation_mode flag to TorchLlmArgs"
```

---

### Task 2: Create `SimModelEngine`

**Files:**
- Create: `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py`

- [ ] **Step 1: Create the file**

Create `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` with the following content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import torch

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

from .model_engine import ModelEngine
from .scheduler import ScheduledRequests
from .resource_manager import ResourceManager


class SimModelEngine(ModelEngine):
    """Model engine that returns dummy logits without running GPU inference.

    Used in simulation mode to exercise the scheduler and request state
    machine without loading model weights or executing forward passes.
    """

    def __init__(self, llm_args: TorchLlmArgs, vocab_size: int,
                 max_num_sequences: int):
        self.llm_args = llm_args
        self.vocab_size = vocab_size
        self._max_num_sequences = max_num_sequences

        # Attributes read by PyExecutor.__init__ and _executor_loop
        self.spec_config = None
        self.enable_attention_dp = False
        self.iter_states = {}
        self.is_warmup = False

        logger.info("[SimModelEngine] Initialized (vocab_size=%d, "
                    "max_num_sequences=%d)", vocab_size, max_num_sequences)

    def get_max_num_sequences(self) -> int:
        return self._max_num_sequences

    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tensors_device=None,
                gather_context_logits: bool = False,
                cache_indirection_buffer=None,
                num_accepted_tokens_device=None):
        num_ctx_requests = scheduled_requests.num_context_requests
        num_ctx_tokens = sum(r.context_chunk_size
                            for r in scheduled_requests.context_requests)
        num_gen_tokens = len(scheduled_requests.generation_requests)

        self.iter_states = {
            'num_ctx_requests': num_ctx_requests,
            'num_ctx_tokens': num_ctx_tokens,
            'num_generation_tokens': num_gen_tokens,
        }

        total_tokens = num_ctx_tokens + num_gen_tokens
        logits = torch.zeros(total_tokens, self.vocab_size)
        return {'logits': logits}
```

- [ ] **Step 2: Verify import**

```bash
python -c "from tensorrt_llm._torch.pyexecutor.sim_model_engine import SimModelEngine; print('SimModelEngine imported OK')"
```

Expected: `SimModelEngine imported OK`

- [ ] **Step 3: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_model_engine.py
git commit -s -m "[None][feat] Add SimModelEngine for simulation mode"
```

---

### Task 3: Create `SimSampler`

**Files:**
- Create: `tensorrt_llm/_torch/pyexecutor/sim_sampler.py`

- [ ] **Step 1: Create the file**

Create `tensorrt_llm/_torch/pyexecutor/sim_sampler.py` with the following content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.logger import logger

from .llm_request import LlmRequestState
from .resource_manager import ResourceManager
from .sampler import SampleState, SampleStateTensors, Sampler
from .scheduler import ScheduledRequests


class SimSampler(Sampler):
    """Sampler that generates dummy tokens and advances request state.

    Used in simulation mode. Each call to update_requests adds one dummy
    token per request and checks whether max_new_tokens has been reached.
    """

    DUMMY_TOKEN_ID = 0

    def sample_async(self, scheduled_requests: ScheduledRequests,
                     model_outputs: dict, num_context_logits_prefix_sum: list,
                     resource_manager: Optional[ResourceManager] = None):
        all_requests = (scheduled_requests.context_requests +
                        scheduled_requests.generation_requests)
        return SampleState(requests=all_requests)

    def update_requests(self, state: SampleState,
                        resource_manager: Optional[ResourceManager] = None):
        for request in state.requests:
            if request.is_generation_complete_state:
                continue

            # Advance sequence length by one dummy token.
            # add_new_token is a C++ binding on LlmRequest that appends
            # the token and updates internal sequence length tracking.
            request.add_new_token(self.DUMMY_TOKEN_ID, 0)
            request.py_decoding_iter += 1

            num_generated = request.get_num_tokens(0) - request.orig_prompt_len
            if num_generated >= request.max_new_tokens:
                request.state = LlmRequestState.GENERATION_COMPLETE
                request.set_finished_reason(FinishReason.LENGTH, 0)

    def is_generation_model(self) -> bool:
        return True
```

- [ ] **Step 2: Verify import**

```bash
python -c "from tensorrt_llm._torch.pyexecutor.sim_sampler import SimSampler; print('SimSampler imported OK')"
```

Expected: `SimSampler imported OK`

- [ ] **Step 3: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/sim_sampler.py
git commit -s -m "[None][feat] Add SimSampler for simulation mode"
```

---

### Task 4: Wire sim path into `create_py_executor()`

**Files:**
- Modify: `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`

This is the most complex task. We add a `_create_sim_py_executor()` helper and branch to it early in `create_py_executor()`.

- [ ] **Step 1: Add the sim imports**

At the top of `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`, add after the existing imports (around line 40, after the `.guided_decoder` import):

```python
from .sim_model_engine import SimModelEngine
from .sim_sampler import SimSampler
```

- [ ] **Step 2: Add `_create_sim_py_executor()` helper function**

Add this function before the existing `create_py_executor()` definition (around line 225):

```python
def _create_sim_py_executor(
    llm_args: TorchLlmArgs,
    checkpoint_dir: str,
    checkpoint_loader,
) -> "PyExecutor":
    """Create a PyExecutor in simulation mode.

    Loads only the HF model config (no weights), creates SimModelEngine
    and SimSampler, but uses the real KV cache manager and scheduler.
    """
    from ._util import create_py_executor_instance
    from ..distributed import Distributed

    skip_est = os.environ.get("TRTLLM_SKIP_KV_CACHE_ESTIMATION", '0') == '1'

    mapping = _get_mapping(llm_args.parallel_config.to_mapping())
    dist = Distributed.get(mapping)

    # Load model config to get vocab_size and model-specific params
    config_kwargs = {
        'trust_remote_code': True,
        'mm_encoder_only': llm_args.mm_encoder_only,
    }
    if llm_args.parallel_config:
        config_kwargs['mapping'] = llm_args.parallel_config.to_mapping()
    model_config = checkpoint_loader.load_config(checkpoint_dir,
                                                  **config_kwargs)
    vocab_size = model_config.vocab_size

    (
        max_beam_width,
        max_num_tokens,
        max_seq_len,
        max_batch_size,
    ) = llm_args.get_runtime_sizes()

    max_num_sequences = max_batch_size * mapping.pp_size

    kv_cache_config = llm_args.kv_cache_config
    tokens_per_block = kv_cache_config.tokens_per_block

    # Sim engine and sampler
    model_engine = SimModelEngine(llm_args, vocab_size, max_num_sequences)
    sampler = SimSampler()

    # Real KV cache — scheduler needs it for capacity decisions
    resources = {}

    # We need a minimal model object for KvCacheCreator. Build a shim that
    # exposes model_config so the KV cache can determine layer count,
    # num_kv_heads, head_size, etc.
    class _SimModelShim:
        """Minimal shim so KvCacheCreator can read model_config."""

        def __init__(self, config):
            from tensorrt_llm._torch.models.modeling_utils import ModelConfig
            self.model_config = ModelConfig.from_pretrained_config(config)

        def named_modules(self):
            return iter([])

    model_engine.model = _SimModelShim(model_config)

    # Set attributes that KvCacheCreator and create_py_executor_instance read
    model_engine.max_seq_len = max_seq_len
    model_engine.max_num_tokens = max_num_tokens
    model_engine.batch_size = max_batch_size
    model_engine.max_beam_width = max_beam_width
    model_engine.mapping = mapping
    model_engine.attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=llm_args.enable_chunked_prefill,
        cache_reuse=kv_cache_config.enable_block_reuse,
    )

    kv_cache_creator = KvCacheCreator(
        model_engine=model_engine,
        draft_model_engine=None,
        mapping=mapping,
        net_max_seq_len=max_seq_len,
        kv_connector_manager=None,
        max_num_tokens=max_num_tokens,
        max_beam_width=max_beam_width,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        llm_args=llm_args,
        speculative_config=None,
        profiling_stage_data=None,
        sparse_attention_config=None,
        execution_stream=torch.cuda.Stream(),
        draft_config=None,
        skip_est=skip_est,
    )
    estimating_kv_cache = kv_cache_creator.try_prepare_estimation()
    kv_cache_creator.build_managers(resources, estimating_kv_cache)
    max_seq_len = kv_cache_creator._max_seq_len

    scheduler_config = llm_args.scheduler_config
    execution_stream = torch.cuda.Stream()

    ctx_chunk_config = None
    if llm_args.enable_chunked_prefill:
        from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy
        ctx_chunk_config = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED

    py_executor = create_py_executor_instance(
        dist=dist,
        resources=resources,
        mapping=mapping,
        llm_args=llm_args,
        ctx_chunk_config=ctx_chunk_config,
        model_engine=model_engine,
        start_worker=False,
        sampler=sampler,
        drafter=None,
        guided_decoder=None,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        max_num_tokens=max_num_tokens,
        scheduler_config=scheduler_config,
        execution_stream=execution_stream,
    )

    if estimating_kv_cache:
        kv_cache_creator.configure_kv_cache_capacity(py_executor)

    logger.info("[SimMode] PyExecutor created in simulation mode")
    return py_executor
```

- [ ] **Step 3: Add the early branch in `create_py_executor()`**

In the existing `create_py_executor()` function, add the branch right after the config loading (after line 258, which is `llm_args = ModelLoader.load_config_and_apply_defaults(...)`):

Insert after line 258:

```python
    if llm_args.simulation_mode:
        return _create_sim_py_executor(llm_args, checkpoint_dir,
                                       checkpoint_loader)
```

- [ ] **Step 4: Run a smoke test**

This requires a GPU (for KV cache). Run:

```bash
cd /home/gvenkatarama/trtllm-wt/hisim-port/TensorRT-LLM
python -c "
from tensorrt_llm.llmapi import LLM, TorchLlmArgs
llm = LLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
           torch_llm_args=TorchLlmArgs(simulation_mode=True))
output = llm.generate(['Hello world'])
print(f'Output: {output}')
print('SUCCESS: Simulation mode POC works!')
"
```

Expected: The script completes without error. The output will contain dummy tokens (token ID 0 repeated). No real model forward pass runs.

- [ ] **Step 5: Debug attribute errors (if any)**

If Step 4 fails with `AttributeError` on `model_engine`, add the missing attribute as a stub on `SimModelEngine`. Common ones to watch for:

- `model_engine.model` — we set this to `_SimModelShim` in `_create_sim_py_executor`
- `model_engine.kv_cache_dtype_byte_size` — add `self.kv_cache_dtype_byte_size = None`
- `model_engine.attn_metadata` — add `self.attn_metadata = None`
- `model_engine.enable_spec_decode` — add `self.enable_spec_decode = False`
- `model_engine.runtime_draft_len` — add `self.runtime_draft_len = 0`
- `model_engine.max_draft_len` — add `self.max_draft_len = 0`

Add any needed attributes to `SimModelEngine.__init__()` in `sim_model_engine.py` and re-run.

- [ ] **Step 6: Commit**

```bash
git add tensorrt_llm/_torch/pyexecutor/py_executor_creator.py
git add tensorrt_llm/_torch/pyexecutor/sim_model_engine.py  # if modified in step 5
git commit -s -m "[None][feat] Wire simulation mode into create_py_executor"
```

---

### Task 5: End-to-end validation

**Files:**
- No new files. Validation only.

- [ ] **Step 1: Run with multiple prompts**

```bash
python -c "
from tensorrt_llm.llmapi import LLM, TorchLlmArgs, SamplingParams
llm = LLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
           torch_llm_args=TorchLlmArgs(simulation_mode=True))
outputs = llm.generate(
    ['Hello world', 'What is AI?', 'Tell me a joke'],
    sampling_params=[SamplingParams(max_tokens=16)] * 3,
)
for i, out in enumerate(outputs):
    token_ids = out.outputs[0].token_ids
    print(f'Prompt {i}: generated {len(token_ids)} tokens, ids={token_ids[:5]}...')
print('SUCCESS: Multi-prompt simulation works!')
"
```

Expected: Each prompt generates exactly 16 tokens (all token ID 0). The scheduler batches them normally.

- [ ] **Step 2: Verify normal mode still works**

```bash
python -c "
from tensorrt_llm.llmapi import LLM, TorchLlmArgs, SamplingParams
llm = LLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
           torch_llm_args=TorchLlmArgs(simulation_mode=False))
output = llm.generate(['Hello'], sampling_params=SamplingParams(max_tokens=5))
token_ids = output[0].outputs[0].token_ids
print(f'Normal mode: {len(token_ids)} tokens, ids={token_ids}')
assert all(t != 0 for t in token_ids), 'Normal mode should produce real tokens'
print('SUCCESS: Normal mode unaffected!')
"
```

Expected: Normal mode produces real (non-zero) tokens — confirms we didn't break anything.

- [ ] **Step 3: Final commit (if any fixups were needed)**

```bash
git add -u
git commit -s -m "[None][fix] Fix simulation mode edge cases from e2e validation"
```

Only run this step if Step 1 or 2 required code changes. Skip if everything passed cleanly.
