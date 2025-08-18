# Tree-Based Inference Time Compute Methods

This document describes the implementation of tree-based inference time compute methods for the TensorRT-LLM Scaffolding framework, including Monte Carlo Tree Search (MCTS) and Tree of Thoughts (TOT).

## Overview

Tree-based inference methods enable LLMs to explore multiple reasoning paths and make more deliberate decisions, similar to human System-2 thinking. These methods can significantly improve performance on complex reasoning tasks that require strategic planning and multi-step problem solving.

## Implemented Methods

### 1. Monte Carlo Tree Search (MCTS)

MCTS is a tree search algorithm that balances exploration and exploitation using the UCB1 (Upper Confidence Bound) formula. It's particularly effective for tasks where the solution space can be explored systematically.

**Key Features:**
- UCB1-based node selection for optimal exploration/exploitation balance
- Configurable exploration constant
- Support for reward models or heuristic evaluation
- Backpropagation of rewards through the tree
- Early termination for high-quality solutions

**Best Use Cases:**
- Mathematical problem solving
- Game playing (e.g., Game of 24)
- Strategic planning tasks
- Problems with verifiable solutions

### 2. Tree of Thoughts (TOT)

TOT enables deliberate problem solving by maintaining a tree of coherent language sequences (thoughts) and using search algorithms to find the best reasoning path.

**Key Features:**
- Multiple thought generation at each step
- Self-evaluation of thought quality
- Flexible selection strategies (best, vote, random)
- Breadth-first exploration
- Confidence scoring for thoughts

**Best Use Cases:**
- Creative writing
- Complex reasoning problems
- Multi-step planning
- Tasks requiring consideration of multiple approaches

## Architecture

### Core Classes

#### TreeNode
Base class for all tree nodes with common functionality:
- Parent-child relationships
- Path tracking
- Depth management
- Terminal state handling

#### MCTSNode
Specialized node for MCTS with:
- UCB1 score calculation
- Visit counts and values
- Reward tracking
- Child selection logic

#### TOTNode
Specialized node for TOT with:
- Thought representation
- Confidence scoring
- Evaluation metrics
- Reasoning storage

#### TreeSearchController
Abstract base class providing:
- Common tree search interface
- Task creation utilities
- Worker tag management
- Search orchestration

## Usage Examples

### Basic MCTS Usage

```python
from tensorrt_llm.scaffolding import (
    MCTSController, NativeGenerationController, 
    ScaffoldingLlm, TRTLLMWorker
)

# Create controllers
generation_controller = NativeGenerationController(sampling_params={
    "temperature": 0.7,
    "max_tokens": 200,
})

mcts_controller = MCTSController(
    generation_controller=generation_controller,
    reward_controller=None,  # Optional reward model
    max_depth=4,
    max_iterations=50,
    exploration_constant=1.414,
    num_thoughts_per_step=3
)

# Set up worker mapping
worker_mapping = {
    MCTSController.WorkerTag.GENERATION: llm_worker,
}

# Create scaffolding LLM
llm = ScaffoldingLlm(mcts_controller, worker_mapping)

# Solve a problem
results = llm.generate(
    ["Solve for x: 2x + 5 = 13"], 
    goal="Solve the mathematical problem step by step"
)
```

### Basic TOT Usage

```python
from tensorrt_llm.scaffolding import (
    TOTController, NativeGenerationController,
    ScaffoldingLlm, TRTLLMWorker
)

# Create controllers
generation_controller = NativeGenerationController(sampling_params={
    "temperature": 0.6,
    "max_tokens": 250,
})

tot_controller = TOTController(
    generation_controller=generation_controller,
    max_depth=4,
    max_iterations=40,
    num_thoughts_per_step=3,
    selection_strategy="best"
)

# Set up worker mapping
worker_mapping = {
    TOTController.WorkerTag.GENERATION: llm_worker,
}

# Create scaffolding LLM
llm = ScaffoldingLlm(tot_controller, worker_mapping)

# Solve a complex reasoning problem
results = llm.generate([
    "A farmer has chickens and rabbits. He counts 35 heads and 94 legs. "
    "How many of each animal does he have?"
], goal="Solve this step-by-step with clear reasoning")
```

## Configuration Parameters

### MCTS Parameters

- `max_depth`: Maximum tree depth (default: 5)
- `max_iterations`: Maximum search iterations (default: 100)
- `exploration_constant`: UCB1 exploration parameter (default: 1.414)
- `num_thoughts_per_step`: Thoughts generated per expansion (default: 3)

### TOT Parameters

- `max_depth`: Maximum reasoning steps (default: 4)
- `max_iterations`: Maximum search iterations (default: 50)
- `num_thoughts_per_step`: Thoughts per step (default: 3)
- `selection_strategy`: Strategy for thought selection ("best", "vote", "random")

## Performance Characteristics

### MCTS
- **Strengths**: Systematic exploration, proven convergence properties, good for games
- **Computational Cost**: O(iterations × depth × thoughts_per_step)
- **Memory Usage**: Grows with tree size, manageable with pruning

### TOT
- **Strengths**: Natural language reasoning, flexible evaluation, interpretable
- **Computational Cost**: O(depth × thoughts_per_step × evaluations)
- **Memory Usage**: Linear with depth and branching factor

## Integration with Scaffolding Framework

Both controllers integrate seamlessly with the existing scaffolding framework:

1. **Task Compatibility**: Work with `GenerationTask` and `RewardTask`
2. **Worker Integration**: Support all existing worker types
3. **Concurrent Execution**: Leverage scaffolding's concurrency features
4. **Modular Design**: Can be combined with other controllers

## Running Examples

### MCTS Example
```bash
python examples/scaffolding/run_mcts_example.py --model_dir /path/to/model
```

### TOT Example  
```bash
python examples/scaffolding/run_tot_example.py --model_dir /path/to/model
```

### With Reward Model
```bash
python examples/scaffolding/run_mcts_example.py \
    --model_dir /path/to/generation/model \
    --reward_model_dir /path/to/reward/model
```

## Benchmarking Results

Based on research literature, these methods show significant improvements:

### Game of 24
- **Standard prompting**: ~4% success rate
- **Chain-of-Thought**: ~4% success rate  
- **TOT**: ~74% success rate
- **MCTS**: Expected ~60-70% success rate

### Complex Reasoning
- **Standard**: Variable performance
- **Tree methods**: 20-50% improvement on multi-step problems

## Future Enhancements

1. **Hybrid Methods**: Combine MCTS and TOT for different problem phases
2. **Adaptive Parameters**: Dynamic adjustment based on problem complexity
3. **Parallel Search**: Distributed tree exploration
4. **Memory Optimization**: Tree pruning and state compression
5. **Domain-Specific Adaptations**: Specialized versions for different domains

## Contributing

When contributing to tree-based methods:

1. Maintain compatibility with the scaffolding framework
2. Add comprehensive tests for new features
3. Document performance characteristics
4. Provide usage examples
5. Consider computational efficiency

## References

1. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
2. "Monte Carlo Tree Search: A Review of Recent Modifications and Applications" (Browne et al., 2012)
3. "Mastering the Game of Go with Deep Neural Networks and Tree Search" (Silver et al., 2016)

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce `max_depth` or `num_thoughts_per_step`
2. **Slow Performance**: Lower `max_iterations` or use simpler evaluation
3. **Poor Quality**: Increase `num_thoughts_per_step` or improve prompts
4. **Worker Errors**: Ensure proper worker initialization and mapping

### Performance Tuning

1. **For Speed**: Reduce iterations and depth, use heuristic evaluation
2. **For Quality**: Increase thoughts per step, use reward models
3. **For Memory**: Implement tree pruning, reduce branching factor
4. **For Accuracy**: Improve prompt engineering, add verification steps 