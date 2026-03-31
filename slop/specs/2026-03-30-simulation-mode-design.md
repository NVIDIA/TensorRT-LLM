# TensorRT-LLM Simulation Mode — Design Spec

## Problem

Evaluating TRT-LLM serving configurations (batch sizes, scheduling policies, KV cache sizing, concurrency levels) currently requires running real GPU inference. This is slow, expensive, and limits the iteration speed for finding optimal deployment configurations.

HiSim (in `tair-kvcache`) solved this for SGLang by hooking the real scheduler, mocking GPU execution, and predicting batch times analytically via AIConfigurator. We want the same capability natively in TRT-LLM.

## Goal

Add a simulation mode to TRT-LLM that runs the real Python scheduler with mocked model execution, enabling fast GPU-free benchmarking of serving configurations.

## Roadmap

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| **0** | Minimal POC | `simulation_mode=True` -> TinyLlama completes with dummy tokens |
| **1** | Config + predictor interface | `SimConfig` Pydantic model, `InferTimePredictor` ABC, constant predictor |
| **2** | AIConfigurator integration | Real batch time predictions, calibration against measured data |
| **3** | Metrics output | `metrics.json`, `request.jsonl`, `iteration.jsonl` matching HiSim format |
| **4** | GPU-free mode | Mock KV cache manager (CPU-only capacity tracking) |
| **5** | CLI integration | `trtllm-bench --mode sim` and/or `trtllm-serve --sim` |
| **6** | Simulated clock + OFFLINE mode | `StateManager`, trace replay without wall-clock sleeping |
| **7** | Multi-GPU / distributed | TP/EP/PP support via config |

---

## Phase 0: Minimal POC (This Spec)

### Success Criteria

```python
from tensorrt_llm.llmapi import LLM, TorchLlmArgs

llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
           torch_llm_args=TorchLlmArgs(simulation_mode=True))
output = llm.generate(["Hello world"])
# Completes with dummy tokens. No real model forward pass executed.
```

### Approach: Mock ModelEngine + Mock Sampler (Approach B)

Keep `PyExecutor` completely unmodified. Inject:
- `SimModelEngine` — returns dummy logits, no real model loaded
- `SimSampler` — generates dummy tokens, advances request state

The real scheduler (`SimpleUnifiedScheduler` or `KVCacheV2Scheduler`) runs unmodified, making real scheduling decisions based on real KV cache capacity.

### Architecture

```
User: LLM(..., TorchLlmArgs(simulation_mode=True))
  |
  v
create_py_executor()              # py_executor_creator.py
  |-- if simulation_mode:
  |     |-- Load HF config only (no model weights)
  |     |-- SimModelEngine(vocab_size, max_num_sequences)
  |     |-- SimSampler()
  |     |-- Real KV cache manager (GPU memory for capacity tracking)
  |     |-- Real scheduler (unchanged)
  |     +-- PyExecutor(scheduler, sim_model_engine, sim_sampler, ...)
  |
  v
PyExecutor._executor_loop()       # UNCHANGED — runs as-is
  |-- _schedule()                  # Real scheduler makes real batch decisions
  |-- _forward_step()              # Calls SimModelEngine.forward() -> dummy logits
  |-- _sample_async()              # Calls SimSampler -> dummy token
  |-- _update_requests()           # SimSampler advances state, checks finish
  +-- loop until all requests GENERATION_COMPLETE
```

### Component 1: `simulation_mode` flag

**File:** `tensorrt_llm/llmapi/llm_args.py`

Add a single boolean field to `TorchLlmArgs`:

```python
simulation_mode: bool = Field(
    default=False,
    description="Enable simulation mode. Skips model weight loading and replaces "
                "model forward with dummy outputs. Scheduler runs normally.")
```

No nested config object for Phase 0. Just a boolean.

### Component 2: `SimModelEngine`

**New file:** `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py`

Implements `ModelEngine` ABC. Key responsibilities:
- `forward()` returns `{'logits': torch.zeros(total_tokens, vocab_size)}`
- Populates `iter_states` dict (executor reads `num_ctx_tokens` for stats)
- Sets attributes that `PyExecutor.__init__` reads: `llm_args`, `spec_config=None`, `enable_attention_dp=False`, `iter_states={}`, `is_warmup=False`

Does NOT:
- Load model weights
- Create CUDA graphs
- Run torch.compile
- Handle speculative decoding, multimodal, LoRA

Estimated size: ~40 lines.

### Component 3: `SimSampler`

**New file:** `tensorrt_llm/_torch/pyexecutor/sim_sampler.py`

Implements `Sampler` ABC. Key responsibilities:
- `sample_async()` bundles requests into a `SampleState` (no GPU kernel)
- `update_requests()` for each request:
  - Calls `request.add_new_token(DUMMY_TOKEN, beam=0)` (C++ binding that advances sequence length)
  - Increments `request.py_decoding_iter`
  - Checks `get_num_tokens(0) - orig_prompt_len >= max_new_tokens` -> marks `GENERATION_COMPLETE`
- `is_generation_model()` returns `True`

Does NOT:
- Run GPU sampling kernels
- Handle beam search, logprobs, stop words, speculative tokens
- Produce meaningful token IDs

Estimated size: ~40 lines.

### Component 4: Construction wiring

**File:** `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`

In `create_py_executor()`, add an early branch when `llm_args.simulation_mode is True`:

1. Call `ModelLoader.load_config_and_apply_defaults()` to apply model-specific defaults to `llm_args` (reads config only, no weights)
2. Call `checkpoint_loader.load_config()` to get the `PretrainedConfig` object — extract `vocab_size` from it
3. Create `SimModelEngine(llm_args, vocab_size, max_num_sequences)`
4. Create `SimSampler()`
5. Skip: real model loading, torch.compile, CUDA graphs, draft model, calibrator
6. Continue with real KV cache setup and real scheduler creation
7. Pass sim engine + sim sampler to `create_py_executor_instance()`

The existing `create_py_executor_instance()` is unchanged — it receives `model_engine` and `sampler` as parameters and doesn't care which implementation they are.

### What still requires GPU (Phase 0)

- KV cache manager allocates GPU memory for block tracking
- `torch.cuda.Stream()` is created for execution stream
- `Distributed` init may touch CUDA

This is acceptable for Phase 0. Phase 4 eliminates the GPU requirement.

### What is explicitly out of scope

- SimConfig / PredictorConfig Pydantic models (Phase 1)
- AIConfigurator integration (Phase 2)
- Metrics output files (Phase 3)
- CPU-only KV cache mock (Phase 4)
- CLI integration with trtllm-bench (Phase 5)
- Simulated clock / OFFLINE mode (Phase 6)
- Multi-GPU support (Phase 7)
- Unit tests (will add in Phase 1 alongside config)
- Any modification to `PyExecutor` itself

### Key Files Touched

| File | Change |
|------|--------|
| `tensorrt_llm/llmapi/llm_args.py` | Add `simulation_mode: bool` to `TorchLlmArgs` |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | **New.** `SimModelEngine(ModelEngine)` |
| `tensorrt_llm/_torch/pyexecutor/sim_sampler.py` | **New.** `SimSampler(Sampler)` |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Branch on `simulation_mode` in `create_py_executor()` |

### Risks

1. **`PyExecutor` reads many attributes off `model_engine`** — `llm_args`, `spec_config`, `iter_states`, `model`, `enable_attention_dp`, `kv_cache_dtype_byte_size`, `attn_metadata`. SimModelEngine must stub all accessed attributes or the loop will crash. Risk: missing an attribute access.
   - Mitigation: Run TinyLlama end-to-end, fix AttributeErrors as they surface.

2. **`SimSampler.update_requests` must advance state correctly** — If the request state machine isn't followed exactly, PyExecutor's loop may hang or crash.
   - Mitigation: Follow `EarlyStopSampler` pattern (already works) but add token-by-token generation.

3. **KV cache manager may not work without a real model config** — The KV cache creator reads model-specific config (num_layers, num_kv_heads, head_size).
   - Mitigation: HF config provides all of this. We load config but not weights.

### Reference Implementation

HiSim (`slop/tair-kvcache/hisim/`) does the same thing for SGLang:
- `sglang_hook.py` — hooks scheduler, model runner
- `sglang_mock_class.py` — mock KV pools
- `time_predictor/aiconfigurator.py` — batch time prediction
- `simulation/manager/state.py` — simulated clock
- `slop/hisim_constraints.md` — documented limitations
