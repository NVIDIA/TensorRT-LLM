# TensorRT-LLM Simulation Mode — Design Spec

## Problem

Evaluating TRT-LLM serving configurations (batch sizes, scheduling policies, KV cache sizing, concurrency levels) currently requires running real GPU inference. This is slow, expensive, and limits the iteration speed for finding optimal deployment configurations.

HiSim (in `tair-kvcache`) solved this for SGLang by hooking the real scheduler, mocking GPU execution, and predicting batch times analytically via AIConfigurator. We want the same capability natively in TRT-LLM.

## Goal

Add a simulation mode to TRT-LLM that runs the real Python scheduler with mocked model execution, enabling fast GPU-free benchmarking of serving configurations.

## Roadmap

```mermaid
gantt
    title Simulation Mode Phases
    dateFormat YYYY-MM-DD
    axisFormat %b %d

    section Done
    Phase 0 - Minimal POC              :done, p0, 2026-03-30, 1d
    Phase 1 - Config + Predictor        :done, p1, 2026-03-31, 1d
    Phase 2 - AIConfigurator            :done, p2, 2026-03-31, 1d

    section Planned
    Phase 3 - Simulated Clock           :active, p3, 2026-04-01, 2d
    Phase 4 - Metrics Output            :p4, after p3, 2d
    Phase 5 - CLI Integration           :p5, after p4, 2d
    Phase 6 - Multi-GPU                 :p6, after p5, 3d
```

| Phase | Goal | Key Deliverable | Status |
|-------|------|-----------------|--------|
| **0** | Minimal POC | `simulation_mode=True` -> TinyLlama completes with dummy tokens | **Done** |
| **1** | Config + predictor interface | `SimConfig` Pydantic model, `InferTimePredictor` ABC, constant predictor | **Done** |
| **2** | AIConfigurator integration | Real batch time predictions via AIC SDK | **Done** |
| **3** | Simulated clock | `SimClock`, no `time.sleep()`, accumulate predicted times | **Done** |
| **3.5** | Single-process mode | Force single-process executor for sim, fix clock visibility, TP>1 | **Done** |
| **4** | Metrics output | Per-request TTFT/TPOT/ITL, per-iteration breakdown, `metrics.json` | **Done** |
| **5** | CLI integration | `trtllm-bench throughput --sim [--sim-config]` with 3-tier verification | **Done** |
| **6** | Request arrival modeling | Staggered arrivals, `--request-rate`, online serving sim | Planned |
| **7** | Multi-GPU / distributed | PP support, disagg KV transfer modeling | Planned |

**Dropped**: Mock KV cache manager (GPU for KV cache is acceptable).

**Long-term vision**: Deep-Sim (`slop/specs/2026-04-01-deep-sim-vision.md`) —
replace AIC batch predictor with per-op timing hooks embedded in TRT-LLM ops.
v1 (AIC-Sim) builds the serving sim infrastructure; Deep-Sim replaces only the
prediction layer. Components designed for reusability: `SimClock`, `SimConfig`,
`SimSampler`, single-process mode, metrics output, CLI integration.

**Reordering rationale** (2026-04-02): Phase 5 (CLI) moved before arrival modeling
because `trtllm-bench` submits all requests at once (batch mode) — no arrival
modeling needed for parity. Phase 6 (arrival modeling) is for future online
serving simulation. TP>1 already works via SimDistributed (Phase 3.5).
Simulated clock (originally Phase 6) was pulled forward because `time.sleep()` pollutes
wall-clock measurements — metrics from Phase 4 need a virtual clock to be meaningful.
GPU-free mode (originally Phase 4) was dropped — requiring a GPU for KV cache capacity
tracking is acceptable for this project's scope.

## Architecture Overview

```mermaid
graph TB
    subgraph User API
        A["LLM(model, sim_config=SimConfig(...))"]
    end

    subgraph "Construction (py_executor_creator.py)"
        B["Load HF Config Only<br/><i>no model weights</i>"]
        C["Predictor Factory"]
        C1["ConstantPredictor"]:::sim
        C2["AIConfiguratorPredictor"]:::sim
        D["SimModelEngine"]:::sim
        E["SimSampler"]:::sim
        F["Real KV Cache Manager"]:::real
        G["Real Scheduler"]:::real
    end

    subgraph "Executor Loop (PyExecutor — UNCHANGED)"
        H["_schedule()"]:::real
        I["_forward_step()"]
        J["SimModelEngine.forward()"]:::sim
        K["predict() → time.sleep()"]:::sim
        L["_sample_async()"]
        M["SimSampler.update_requests()"]:::sim
        N{"All requests<br/>complete?"}
    end

    A --> B
    B --> C
    C -->|"name=constant"| C1
    C -->|"name=aiconfigurator"| C2
    C1 --> D
    C2 --> D
    D --> G
    E --> G
    F --> G

    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N -->|No| H
    N -->|Yes| O["Return Results"]

    classDef sim fill:#4CAF50,stroke:#2E7D32,color:#fff
    classDef real fill:#2196F3,stroke:#1565C0,color:#fff
```

**Legend:** <span style="color:#4CAF50">■ Green = Simulation components (new code)</span> · <span style="color:#2196F3">■ Blue = Real TRT-LLM components (unchanged)</span>

## Component Dependency Graph

```mermaid
graph LR
    subgraph "Config Layer (Pydantic)"
        SC["SimConfig"]:::config
        PC["PredictorConfig"]:::config
    end

    subgraph "Predictor Layer"
        ABC["InferTimePredictor<br/><i>ABC</i>"]:::abc
        CP["ConstantPredictor"]:::pred
        AIC["AIConfiguratorPredictor"]:::pred
        SB["SimBatch"]:::data
        SBR["SimBatchRequest"]:::data
    end

    subgraph "Engine Layer"
        SME["SimModelEngine"]:::engine
        SS["SimSampler"]:::engine
    end

    subgraph "Executor Layer (unchanged)"
        PE["PyExecutor"]:::real
        SCHED["Scheduler"]:::real
        KV["KV Cache Manager"]:::real
    end

    SC --> PC
    PC -->|"name dispatch"| CP
    PC -->|"name dispatch"| AIC
    CP --> ABC
    AIC --> ABC
    ABC --> SB
    SB --> SBR
    SME -->|"calls predict()"| ABC
    SME --> PE
    SS --> PE
    SCHED --> PE
    KV --> PE

    classDef config fill:#FF9800,stroke:#E65100,color:#fff
    classDef abc fill:#9C27B0,stroke:#6A1B9A,color:#fff
    classDef pred fill:#4CAF50,stroke:#2E7D32,color:#fff
    classDef data fill:#607D8B,stroke:#37474F,color:#fff
    classDef engine fill:#00BCD4,stroke:#00838F,color:#fff
    classDef real fill:#2196F3,stroke:#1565C0,color:#fff
```

**Legend:** <span style="color:#FF9800">■ Config</span> · <span style="color:#9C27B0">■ ABC</span> · <span style="color:#4CAF50">■ Predictors</span> · <span style="color:#607D8B">■ Data</span> · <span style="color:#00BCD4">■ Engines</span> · <span style="color:#2196F3">■ Real (unchanged)</span>

## File Map

```mermaid
graph TD
    subgraph "tensorrt_llm/llmapi/"
        A1["sim_config.py<br/>SimConfig + PredictorConfig"]:::config
        A2["llm_args.py<br/>sim_config: Optional[SimConfig]"]:::modified
    end

    subgraph "tensorrt_llm/_torch/pyexecutor/"
        B1["sim_predictor.py<br/>SimBatch, InferTimePredictor ABC,<br/>ConstantPredictor"]:::new
        B2["sim_predictor_aic.py<br/>AIConfiguratorPredictor"]:::new
        B3["sim_model_engine.py<br/>SimModelEngine"]:::new
        B4["sim_sampler.py<br/>SimSampler"]:::new
        B5["py_executor_creator.py<br/>_create_sim_py_executor()"]:::modified
    end

    subgraph "tests/unittest/sim/"
        C1["test_sim_config.py"]:::test
        C2["test_sim_predictor.py"]:::test
        C3["test_sim_predictor_aic.py"]:::test
    end

    A1 --> A2
    A1 --> B5
    B1 --> B3
    B2 --> B3
    B3 --> B5
    B4 --> B5

    classDef new fill:#4CAF50,stroke:#2E7D32,color:#fff
    classDef modified fill:#FF9800,stroke:#E65100,color:#fff
    classDef config fill:#FF9800,stroke:#E65100,color:#fff
    classDef test fill:#9E9E9E,stroke:#616161,color:#fff
```

**Legend:** <span style="color:#4CAF50">■ New files</span> · <span style="color:#FF9800">■ Modified files</span> · <span style="color:#9E9E9E">■ Test files</span>

---

## Phase 0: Minimal POC

### Success Criteria

```python
from tensorrt_llm.llmapi import LLM, TorchLlmArgs

llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
           torch_llm_args=TorchLlmArgs(simulation_mode=True))
output = llm.generate(["Hello world"])
# Completes with dummy tokens. No real model forward pass executed.
```

See `slop/specs/phase0-what-was-built.md` for full details.

## Phase 1: Config + Predictor Interface

See `slop/specs/phase1-what-was-built.md` for full details.

## Phase 2: AIConfigurator Integration

See `slop/specs/phase2-what-was-built.md` for full details.

## Phases 3-6: Planned

See individual phase specs when created.

---

## Reference Implementation

HiSim (`slop/tair-kvcache/hisim/`) does the same thing for SGLang:
- `sglang_hook.py` — hooks scheduler, model runner
- `sglang_mock_class.py` — mock KV pools
- `time_predictor/aiconfigurator.py` — batch time prediction
- `simulation/manager/state.py` — simulated clock
- `slop/hisim_constraints.md` — documented limitations
