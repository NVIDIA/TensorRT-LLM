# Tree-Based Inference Methods (MCTS and TOT)

This document describes the tree-based inference time methods implemented in the TensorRT-LLM Scaffolding framework and how to run the provided examples. The two supported methods are Monte Carlo Tree Search (MCTS) and Tree of Thoughts (TOT).

## Overview

Tree-based inference enables LLMs to explore multiple reasoning paths before committing to a final answer. At a high level:
- **MCTS** balances exploration and exploitation with UCB1 and backpropagates rewards to guide search.
- **TOT** generates multiple candidate “thoughts” per step, evaluates them, and expands the most promising paths.

These approaches are particularly useful for multi-step reasoning, math word problems, planning, and tasks with verifiable answers.

## Core Classes

Defined in `tensorrt_llm/scaffolding/tree_controllers.py`:

- **`TreeNode`**: Base node with parent/children, visits, value, depth, terminal flag, and helpers.
- **`MCTSNode`**: Extends `TreeNode` with reward and UCB1-based child selection.
- **`TOTNode`**: Extends `TreeNode` with thought text, confidence, reasoning, and evaluation score.
- **`MCTSController`**: Orchestrates selection → expansion → simulation → backpropagation; produces a final answer after following the best path.
- **`TOTController`**: Breadth-first expansion of thoughts with self-evaluation and selection; produces a final answer based on the best thought trajectory.

Both controllers integrate with the scaffolding stack via `GenerationTask` and can optionally use a reward controller (e.g., a reward model) for scoring.

## Key Parameters

### MCTSController
- **`max_depth`**: Maximum depth of the search tree.
- **`max_iterations`**: Total MCTS iterations.
- **`exploration_constant`**: UCB1 exploration parameter (higher → more exploration).
- **`num_thoughts_per_step`**: Number of candidate next steps generated per expansion.

Internally, the controller:
- Builds a tree rooted at the original problem.
- Selects leaves by UCB1, expands via the generation controller, and simulates with either a reward controller or a heuristic.
- Backpropagates rewards and finally generates a succinct answer conditioned on the best path.

### TOTController
- **`max_depth`**: Reasoning depth (number of steps/levels).
- **`max_iterations`**: Upper bound on traversal work (used as a guard; breadth-first expansion is primary).
- **`num_thoughts_per_step`**: Number of approaches generated per node.
- **`selection_strategy`**: Selection policy for next-level thoughts: `"best"`, `"vote"`, or `"random"`.

Internally, the controller:
- Generates multiple approaches (“thoughts”), evaluates them, and selects the most promising to expand.
- At the end, chooses the best leaf by evaluation score and generates a concise answer from the reasoning path.

## Example Scripts

Two runnable examples live under `examples/scaffolding/` and use a JSONL dataset for prompts and (optional) reference answers:
- `run_mcts_example.py`
- `run_tot_example.py`

Both scripts accept a `--jsonl_file` (default: `./test.jsonl`) and read each line as a JSON object with the following schema:
- `problem` or `question`: The prompt to solve.
- `answer` (optional): Reference answer, printed for comparison.

An example dataset is provided at `examples/scaffolding/test.jsonl` with AIME-style questions.

## How to Run

Prerequisites:
- A compatible generation model path for `--model_dir` (PyTorch backend in the examples)
- Optional reward model path for `--reward_model_dir` (MCTS only in the examples)
- The JSONL dataset file (default provided)

### MCTS
```bash
python TensorRT-LLM/examples/scaffolding/run_mcts_example.py \
  --model_dir <path_to_generation_model> \
  --reward_model_dir <path_to_reward_model> \
  --jsonl_file TensorRT-LLM/examples/scaffolding/test.jsonl \
  --max_depth 4 \
  --max_iterations 20 \
  --exploration_constant 1.414 \
  --num_thoughts_per_step 3
```
Notes:
- `--reward_model_dir` is optional. If omitted, a heuristic simulation reward is used.
- The script prints the problem, the reference answer (if present), then the MCTS solution.

### TOT
```bash
python TensorRT-LLM/examples/scaffolding/run_tot_example.py \
  --model_dir <path_to_generation_model> \
  --jsonl_file TensorRT-LLM/examples/scaffolding/test.jsonl \
  --max_depth 3 \
  --max_iterations 15 \
  --num_thoughts_per_step 3 \
  --selection_strategy best
```
Notes:
- `--reward_model_dir` is supported but optional.
- The script prints the problem, the reference answer (if present), then the TOT solution.

## JSONL Format Details

Each line should be a valid JSON object. Supported keys:
```json
{"problem": "...", "answer": "..."}
{"question": "...", "answer": "..."}
```
If both `problem` and `question` are present, the scripts use the first non-empty one. `answer` is optional and is only printed for reference.

## Tuning Tips
- Increase `num_thoughts_per_step` for broader exploration at each step (higher cost).
- Increase `max_depth` for deeper reasoning (be mindful of latency).
- For MCTS, raise `max_iterations` for more exhaustive search; tune `exploration_constant` for your task.
- Provide a reward model for MCTS when verifiable scoring is available.

## Troubleshooting
- Reduce `max_depth`/`max_iterations`/`num_thoughts_per_step` if inference is slow or memory-heavy.
- Ensure model paths are valid and the backend (`backend="pytorch"` in the examples) is supported.
- Verify your JSONL file has one JSON object per line with `problem` or `question` keys.

## References
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
- Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
- Silver et al., "Mastering the Game of Go with Deep Neural Networks and Tree Search" (2016) 
