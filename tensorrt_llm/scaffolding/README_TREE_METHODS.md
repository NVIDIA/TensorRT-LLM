# Tree-Based Inference (MCTS & TOT)

## Introduction

Tree-based methods explore multiple reasoning paths before producing a final answer. This directory provides two controllers:

- MCTS (Monte Carlo Tree Search): balances exploration/exploitation via UCB1 and backpropagates rewards.
- TOT (Tree of Thoughts): expands multiple candidate thoughts per step, evaluates, and prunes.

Both integrate with Scaffolding `Controller`/`Worker` and optionally use a reward model (PRM).

## Controllers

- `MCTSController`
  - Selection → Expansion → Simulation → Backpropagation
  - Parameters: `max_depth`, `max_iterations`, `exploration_constant`, `num_thoughts_per_step`
  - Final answer is generated from a compact “Problem + Steps” trajectory

- `TOTController`
  - Level-wise expansion of thoughts with evaluation and pruning
  - Parameters: `max_depth`, `max_iterations` (guard), `num_thoughts_per_step`, `selection_strategy` (best | vote | random), `branch_factor`
  - Final answer is generated from the best reasoning path

## Examples

- `examples/scaffolding/run_mcts_example.py`
- `examples/scaffolding/run_tot_example.py`

Basic usage (PyTorch backend):
```bash
python TensorRT-LLM/examples/scaffolding/run_mcts_example.py \
  --model_dir <path_to_generation_model> \
  --reward_model_dir <path_to_reward_model>

python TensorRT-LLM/examples/scaffolding/run_tot_example.py \
  --model_dir <path_to_generation_model>
```

Notes:
- Reward model is optional. If provided, PRM-based scoring is used.
- Increase `num_thoughts_per_step`, `max_depth`, or `max_iterations` for broader/deeper search.
- For long outputs, raise generation `max_tokens` and consider adding a stop sequence (e.g., `</think>`).


## References

- Yao et al., Tree of Thoughts (2023)
- Browne et al., Survey of MCTS (2012)
- Silver et al., AlphaGo (2016) 
