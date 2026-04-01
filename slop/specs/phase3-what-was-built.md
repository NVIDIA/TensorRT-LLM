# Phase 3 + 3.5: What Was Built

**Date**: 2026-03-31 (Phase 3), 2026-04-01 (Phase 3.5)
**Branch**: `venky/hisim-port`
**Commits**: `09e2115be` (Phase 3), `2f70180b3` (vision doc), `99763a897` (Phase 3.5)

## Phase 3: Simulated Clock

### Problem
Phase 2's `SimModelEngine.forward()` called `time.sleep(predicted_time)` to simulate batch execution time. This polluted wall-clock measurements — AIC predicts ~13ms but real wall-clock was ~205ms due to framework overhead.

### What Changed

| File | Change |
|------|--------|
| `tensorrt_llm/_torch/pyexecutor/sim_clock.py` | **Created** — `SimClock` class accumulating predicted iteration times |
| `tensorrt_llm/_torch/pyexecutor/sim_model_engine.py` | Replaced `time.sleep(predicted_time)` with `self.clock.step(predicted_time)` |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Creates `SimClock`, passes to engine, stores on `sim_config._clock` |
| `tensorrt_llm/llmapi/sim_config.py` | Added `_clock: PrivateAttr` to `SimConfig` |
| `tests/unittest/sim/test_sim_clock.py` | **Created** — 5 tests for SimClock |

### Key Discovery: Prefill Produces First Token
The prefill iteration generates the first output token. So `max_tokens=8` produces:
- 1 prefill iteration (generates token 1) + 7 decode iterations (tokens 2-8)
- **Total: 8 iterations**, not 9 as originally expected
- With constant predictor (10ms prefill, 5ms decode): 10 + 7×5 = **45ms**

### Limitations Discovered (Fixed in Phase 3.5)
1. **Cross-process clock**: `sim_config._clock` was set on MPI worker's copy, invisible to caller
2. **Executor restart bug**: KV cache estimation shutdown left `is_shutdown=True`, preventing restart

---

## Phase 3.5: Single-Process Sim Mode

### Problem
TRT-LLM's `LLM` class spawns executor in a separate MPI worker process (even for TP=1). The `SimClock` was set on the worker's copy of `sim_config`, making it invisible to the caller. For TP>1, `TLLM_WORKER_USE_SINGLE_PROCESS=1` was ignored because multi-process spawn happens before the single-process check.

### Root Cause Analysis
Traced the full executor creation flow:
```
GenerationExecutor.create()
  ├── if spawn_workers (TP>1): → multi-process IPC (NEVER reaches single-process check)
  ├── if TLLM_WORKER_USE_SINGLE_PROCESS=1 (TP=1 only): → single-process worker
  └── else (TP=1 default): → multi-process IPC for streaming perf
```

HiSim avoids this entirely because SGLang runs single-process with `torch.distributed` (no MPI). TRT-LLM's MPI boundary is unique to its architecture.

### What Changed

| File | Change |
|------|--------|
| `tensorrt_llm/_torch/pyexecutor/sim_distributed.py` | **Created** — `SimDistributed` mock that no-ops all communication |
| `tensorrt_llm/executor/executor.py` | Added sim mode early-return: force `use_worker=True` before any MPI logic |
| `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py` | Use `SimDistributed` instead of `Distributed.get()`, auto-skip KV estimation |
| `tests/unittest/sim/test_sim_distributed.py` | **Created** — 13 tests for SimDistributed |
| `slop/test_sim.py` | Removed env var hacks, added TP=2 test, AIC TP comparison |

### SimDistributed Design

```python
class SimDistributed(Distributed):
    """No-op communication for sim mode."""
    def barrier(self): pass
    def broadcast(self, obj, root=0): return obj
    def allgather(self, obj, root=0): return [obj]
    def allreduce(self, obj, op=ReduceOp.SUM): return obj
    # ... all communication methods are identity/no-op
    # PP send/recv raise NotImplementedError (sim doesn't support PP)
```

Key design decisions:
- **Not a singleton** (unlike HiSim's StateManager) — instance per executor
- **Inherits base class properties** — `rank`, `tp_size`, `pp_size` come from mapping
- **PP not supported** — `send_object`/`recv_object` raise (sim mode is TP-only for now)

### Executor Routing

```python
# In GenerationExecutor.create(), BEFORE all other routing:
if llm_args.sim_config is not None:
    return _create_ipc_executor(..., use_worker=True)  # single-process
```

This intercepts before `spawn_workers` or `mpirun_launch` checks, so it works for any `model_world_size`.

### Mapping Handling

```python
# In _create_sim_py_executor:
mapping = copy.deepcopy(llm_args.parallel_config.to_mapping())
mapping.rank = 0  # Force rank 0, not mpi_rank()
dist = SimDistributed(mapping)
```

The mapping carries the real TP/PP config (used by scheduler and AIC predictor), but rank is always 0 since there's only one process.

### Verification Results

| Test | Result |
|------|--------|
| TP=1 constant predictor | Clock visible, 45.0ms, 8 iters |
| TP=2 constant predictor | Clock visible, 45.0ms, 8 iters |
| AIC TP=1 vs TP=2 | Different times (ratio 1.16x), proves TP flows through |
| No env var hacks | No `TRTLLM_SKIP_KV_CACHE_ESTIMATION`, no `TLLM_WORKER_USE_SINGLE_PROCESS` |
| Unit tests | 57 passing |

### Architecture After Phase 3.5

```
LLM(model, sim_config=SimConfig(...), tensor_parallel_size=2)
  │
  ├── GenerationExecutor.create()
  │   └── sim_config detected → force use_worker=True (single-process)
  │
  └── GenerationExecutorWorker (same process)
      └── create_py_executor()
          └── _create_sim_py_executor()
              ├── SimDistributed(mapping)  — no-op communication
              ├── SimModelEngine + SimClock — predicted time accumulation
              ├── SimSampler — dummy token generation
              ├── Real KV cache (skip estimation)
              ├── Real scheduler (capacity-aware)
              └── sim_config._clock = clock  — visible to caller!
```

### Injection Seams for Phase 4+
1. **`SimClock.step()`** — Phase 4 extends to record per-iteration breakdown
2. **`sim_config._clock`** — Phase 4 metrics writer reads accumulated time
3. **`SimDistributed`** — Deep-Sim SIMULATE mode reuses this for single-process op-level sim
4. **Single-process executor routing** — Deep-Sim SIMULATE uses same path
