# Tree-Based Inference (MCTS & TOT)

## Introduction

Tree-based methods explore multiple reasoning paths before producing a final answer. This directory provides two controllers:

- MCTS (Monte Carlo Tree Search): balances exploration/exploitation via UCB1 and backpropagates rewards.
- TOT (Tree of Thoughts): expands multiple candidate thoughts per step, evaluates, and prunes.

Both integrate with Scaffolding `Controller`/`Worker` and optionally use a reward model (PRM).

## Attribution and design notes

- The `TOTController` is inspired by Tree-of-Thoughts (Yao et al., 2023). It follows the core idea (multi-thought expansion, scoring via value/PRM or self-eval, and tree-based selection).
- Key implementation differences:
  - Uses a configurable `branch_factor` per level rather than a single global beam width.
  - Supports an optional reward controller (PRM) for scoring; can be disabled to use simpler heuristics.
  - Provides multiple selection strategies (`best`, `vote`, `random`) with a simple best-path finalization by default, instead of strictly enforcing the paper’s BFS/DFS + majority vote variants.
  - Other engineering choices reflect integration with this repository’s `Controller`/`Worker` scaffolding.

## Controllers

- `MCTSController`
  - Selection → Expansion → Simulation → Backpropagation
  - Parameters: `max_depth`, `max_iterations`, `exploration_constant`, `num_thoughts_per_step`
  - Final answer is generated from a compact “Problem + Steps” trajectory

- `TOTController`
  - Level-wise expansion of thoughts with evaluation and pruning
  - Parameters: `max_depth`, `max_iterations` (guard), `num_thoughts_per_step`, `selection_strategy` (best | vote | random), `branch_factor`
  - Final answer is generated from the best reasoning path

## Quickstart: minimal reproducible examples

- MCTS (PyTorch backend)
```python
from tensorrt_llm.scaffolding import (
    MCTSController, NativeGenerationController, PRMController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import TRTLLMWorker

# 1) Spin up workers
workers = {}
# Generation worker
gen_worker = TRTLLMWorker.init_with_new_llm(
    model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    backend="pytorch",
    max_batch_size=4,
    max_num_tokens=8192,
    kv_cache_free_gpu_memory_fraction=0.1,  # tunable via --gen_kv_cache_free_gpu_memory_fraction
)
workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker

# Reward worker (optional) + controller
reward_worker = TRTLLMWorker.init_with_new_llm(
    model_dir="Qwen/Qwen2.5-Math-PRM-7B",
    backend="pytorch",
    max_batch_size=4,
    max_num_tokens=8192,
    kv_cache_free_gpu_memory_fraction=0.2,  # tunable via --reward_kv_cache_free_gpu_memory_fraction
    disable_overlap_scheduler=True,         # pass --reward_overlap_scheduler to enable overlap
)
workers[PRMController.WorkerTag.REWARD] = reward_worker
reward_controller = PRMController(tokenizer=reward_worker.tokenizer)

# 2) Controllers
generation_controller = NativeGenerationController(sampling_params={
    "max_tokens": 4096,
    "temperature": 0.8,
})
controller = MCTSController(
    generation_controller=generation_controller,
    reward_controller=reward_controller,
    max_depth=4,
    max_iterations=20,
    exploration_constant=1.414,
    num_thoughts_per_step=3,
)

# 3) Run
llm = ScaffoldingLlm(controller, workers=workers)
prompts = [
    "Question 1",
    "Question 2"
]
results = llm.generate(prompts)
print(results[0].outputs[0].text)
llm.shutdown(shutdown_workers=True)
```

- TOT (PyTorch backend)
```python
from tensorrt_llm.scaffolding import (
    NativeGenerationController, PRMController, TOTController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import TRTLLMWorker

# 1) Spin up workers
workers = {}
# Generation worker
gen_worker = TRTLLMWorker.init_with_new_llm(
    model_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    backend="pytorch",
    max_batch_size=4,
    max_num_tokens=8192,
    kv_cache_free_gpu_memory_fraction=0.1,  # tunable via --gen_kv_cache_free_gpu_memory_fraction
)
workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker

# Reward worker (optional) + controller
reward_worker = TRTLLMWorker.init_with_new_llm(
    model_dir="Qwen/Qwen2.5-Math-PRM-7B",
    backend="pytorch",
    max_batch_size=4,
    max_num_tokens=8192,
    kv_cache_free_gpu_memory_fraction=0.2,  # tunable via --reward_kv_cache_free_gpu_memory_fraction
    disable_overlap_scheduler=True,         # pass --reward_overlap_scheduler to enable overlap
)
workers[PRMController.WorkerTag.REWARD] = reward_worker
reward_controller = PRMController(tokenizer=reward_worker.tokenizer)

# 2) Controllers
generation_controller = NativeGenerationController(sampling_params={
    "max_tokens": 8192,
    "temperature": 0.8,
})
controller = TOTController(
    generation_controller=generation_controller,
    reward_controller=reward_controller,
    max_depth=3,
    max_iterations=15,
    num_thoughts_per_step=3,
    selection_strategy="best",
    branch_factor=2,
)

# 3) Run
llm = ScaffoldingLlm(controller, workers=workers)
prompts = [
    "Question 1",
    "Question 2"
]
results = llm.generate(prompts)
print(results[0].outputs[0].text)
llm.shutdown(shutdown_workers=True)
```

## Examples

- `TensorRT-LLM/examples/scaffolding/contrib/TreeInference/run_mcts_example.py`
- `TensorRT-LLM/examples/scaffolding/contrib/TreeInference/run_tot_example.py`

Basic usage (PyTorch backend):
```bash
python TensorRT-LLM/examples/scaffolding/contrib/TreeInference/run_mcts_example.py \
  --model_dir <generation_model> \
  --reward_model_dir <reward_model> \
  [--gen_kv_cache_free_gpu_memory_fraction 0.1] \
  [--reward_kv_cache_free_gpu_memory_fraction 0.2] \
  [--reward_overlap_scheduler]

python TensorRT-LLM/examples/scaffolding/contrib/TreeInference/run_tot_example.py \
  --model_dir <generation_model> \
  [--gen_kv_cache_free_gpu_memory_fraction 0.1] \
  [--reward_kv_cache_free_gpu_memory_fraction 0.2] \
  [--reward_overlap_scheduler]
```

Notes:
- Reward model is optional. If provided, PRM-based scoring is used.
- Increase `num_thoughts_per_step`, `max_depth`, or `max_iterations` for broader/deeper search.
- For long outputs, raise generation `max_tokens` and consider adding a stop sequence (e.g., `</think>`).
- kv_cache_free_gpu_memory_fraction: when sharing a GPU between gen and reward, allocating explicit KV headroom reduces OOMs and allocator pressure. Values like 0.1 (gen) and 0.2 (reward) worked well in small-scale runs; adjust to your model size and memory.
- reward overlap: disabling overlap on reward is a pragmatic choice for spiky PRM scoring; it lowers contention with the gen worker. It’s safe to enable if reward is isolated on another GPU or you have sufficient headroom.


## References

- Yao et al., Tree of Thoughts (2023)
- Browne et al., Survey of MCTS (2012)
