import copy
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.scaffolding.controller import Controller
from tensorrt_llm.scaffolding.task import GenerationTask, RewardTask, Task
from tensorrt_llm.executor.result import CompletionOutput, GenerationResult


@dataclass
class TreeNode:
    """Base class for tree nodes in tree-based inference methods."""
    state: str = ""
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    is_terminal: bool = False
    depth: int = 0
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this node is the root (has no parent)."""
        return self.parent is None
    
    def add_child(self, child: 'TreeNode') -> 'TreeNode':
        """Add a child node and return it."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child
    
    def get_path_to_root(self) -> List['TreeNode']:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))


@dataclass
class MCTSNode(TreeNode):
    """Node for Monte Carlo Tree Search."""
    reward: float = 0.0
    untried_actions: List[str] = field(default_factory=list)
    
    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 score for this node."""
        if self.visits == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select the child with the highest UCB1 score."""
        if not self.children:
            return self
        return max(self.children, key=lambda child: child.ucb1_score(exploration_constant))


@dataclass
class TOTNode(TreeNode):
    """Node for Tree of Thoughts."""
    thought: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    evaluation_score: float = 0.0
    
    def __post_init__(self):
        if not self.thought and self.state:
            self.thought = self.state


class MCTSController(Controller):
    """Monte Carlo Tree Search Controller for scaffolding framework."""
    
    class WorkerTag(Enum):
        GENERATION = "generation"
        REWARD = "reward"
    
    def __init__(self,
                 generation_controller: Controller,
                 reward_controller: Optional[Controller] = None,
                 max_depth: int = 5,
                 max_iterations: int = 100,
                 exploration_constant: float = 1.414,
                 num_thoughts_per_step: int = 3):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.num_thoughts_per_step = num_thoughts_per_step
    
    def process(self, tasks: List[Task], **kwargs) -> Any:
        """Process tasks using MCTS with yield-based orchestration."""
        assert len(tasks) == 1, "MCTS Controller only supports single task processing"
        task = tasks[0]
        goal = kwargs.get('goal', 'Solve the problem step by step')
        initial_state = getattr(task, 'input_str', str(task)) or ""

        # Build tree
        root = MCTSNode(state=initial_state)

        for iteration in range(self.max_iterations):
            # Selection
            node = root
            while not node.is_leaf() and not node.is_terminal:
                node = node.select_best_child(self.exploration_constant)

            # Expansion
            if not node.is_terminal and node.depth < self.max_depth:
                prompt = self._create_expansion_prompt(node, goal)
                gen_task = GenerationTask.create_from_prompt(prompt)
                gen_task.max_tokens = 200
                gen_task.temperature = 0.7
                
                # Delegate to generation controller
                yield from self.generation_controller.process([gen_task])
                thoughts = self._parse_thoughts(gen_task.output_str or "")
                for thought in thoughts[: self.num_thoughts_per_step]:
                    child_state = f"{node.state}\n{thought}".strip()
                    node.add_child(MCTSNode(state=child_state, reward=0.0))
                if node.children:
                    node = random.choice(node.children)

            # Simulation (evaluate node)
            if self.reward_controller is not None:
                # Create a proper task for reward evaluation
                # input_str should be the original problem, output_str should be the response to evaluate
                reward_task = GenerationTask()
                reward_task.input_str = initial_state
                
                # Create a proper result with CompletionOutput before setting output_str
                completion_output = CompletionOutput(index=0, text=node.state)
                # Create a mock GenerationResult - we need sampling_params for the constructor
                from tensorrt_llm.sampling_params import SamplingParams
                mock_sampling_params = SamplingParams()
                reward_result = GenerationResult.__new__(GenerationResult)
                reward_result._outputs = [completion_output]
                reward_result.sampling_params = mock_sampling_params
                reward_task.result = reward_result
                yield from self.reward_controller.process([reward_task])
                # Get reward from the reward controller
                if hasattr(self.reward_controller, 'scores') and self.reward_controller.scores:
                    reward = float(self.reward_controller.scores[0])
                else:
                    reward = 0.5  # Default reward
            else:
                reward = min(1.0, len(node.state.split()) / 100.0)

            # Backpropagation
            self._backpropagate(node, reward)

        # Pick best leaf
        best_leaf = self._select_best_leaf(root)
        path = best_leaf.get_path_to_root()

        # Final answer generation based on the best path
        reasoning = "\n".join([n.state for n in path])
        final_prompt = (
            f"{goal}\n\nHere is a coherent reasoning trajectory.\n"
            f"{reasoning}\n\nNow provide the final answer succinctly."
        )
        final_task = GenerationTask.create_from_prompt(final_prompt)
        final_task.max_tokens = 256
        final_task.temperature = 0.2
        yield from self.generation_controller.process([final_task])

        # Assign the result to the original task
        tasks[0].result = final_task.result
    
    def _create_expansion_prompt(self, node: MCTSNode, goal: str) -> str:
        """Create a prompt for expanding a node."""
        return f"""Goal: {goal}

Current state:
{node.state}

Generate {self.num_thoughts_per_step} possible next steps or thoughts to progress toward the goal. 
Each thought should be a coherent reasoning step or action.
Format your response as:
1. [First thought]
2. [Second thought] 
3. [Third thought]"""
    
    def _parse_thoughts(self, text: str) -> List[str]:
        """Parse generated thoughts from text."""
        thoughts = []
        lines = (text or "").strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and '.' in line:
                thought = line.split('.', 1)[-1].strip()
                if thought:
                    thoughts.append(thought)
            elif line.startswith(('-', '*')):
                thoughts.append(line[1:].strip())
        return thoughts
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate the reward up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent
    
    def _select_best_leaf(self, root: MCTSNode) -> MCTSNode:
        """Select the best leaf node from the tree."""
        best_leaf = root
        best_score = -float('inf')
        def traverse(node: MCTSNode):
            nonlocal best_leaf, best_score
            if node.is_leaf() and node.visits > 0:
                avg_value = node.value / node.visits
                if avg_value > best_score:
                    best_score = avg_value
                    best_leaf = node
            for child in node.children:
                traverse(child) 
        traverse(root)
        return best_leaf


class TOTController(Controller):
    """Tree of Thoughts Controller for scaffolding framework."""
    
    class WorkerTag(Enum):
        GENERATION = "generation"
        REWARD = "reward"
    
    def __init__(self,
                 generation_controller: Controller,
                 reward_controller: Optional[Controller] = None,
                 max_depth: int = 4,
                 max_iterations: int = 50,
                 num_thoughts_per_step: int = 3,
                 selection_strategy: str = "best"):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.num_thoughts_per_step = num_thoughts_per_step
        self.selection_strategy = selection_strategy  # "best", "vote", "random"
    
    def process(self, tasks: List[Task], **kwargs) -> Any:
        """Process tasks using Tree of Thoughts with yield-based orchestration."""
        assert len(tasks) == 1, "TOT Controller only supports single task processing"
        task = tasks[0]
        goal = kwargs.get('goal', 'Solve the problem step by step')
        root_state = getattr(task, 'input_str', str(task)) or ""

        root = TOTNode(state=root_state, thought="Initial problem")
        current_level: List[TOTNode] = [root]

        for depth in range(self.max_depth):
            next_level: List[TOTNode] = []
            for node in current_level:
                if node.is_terminal:
                    continue
                # Generate thoughts for this node
                gen_prompt = self._generate_prompt(node, goal)
                gen_task = GenerationTask.create_from_prompt(gen_prompt)
                gen_task.max_tokens = 300
                gen_task.temperature = 0.8
                yield from self.generation_controller.process([gen_task])
                thoughts = self._parse_approaches(gen_task.output_str or "")

                # Evaluate each thought
                evaluated_thoughts: List[Dict[str, Any]] = []
                for thought in thoughts[: self.num_thoughts_per_step]:
                    eval_prompt = self._evaluation_prompt(thought, goal, node.state)
                    eval_task = GenerationTask.create_from_prompt(eval_prompt)
                    eval_task.max_tokens = 150
                    eval_task.temperature = 0.3
                    yield from self.generation_controller.process([eval_task])
                    evaluation = self._parse_evaluation(eval_task.output_str or "")
                    evaluated_thoughts.append({
                        'thought': thought,
                        'score': evaluation['score'],
                        'confidence': evaluation['confidence'],
                        'reasoning': evaluation['reasoning']
                    })

                # Select top thoughts and create children
                for thought_data in self._select_thoughts(evaluated_thoughts):
                    child_state = self._combine_state_and_thought(node.state, thought_data['thought'])
                    child = TOTNode(
                        state=child_state,
                        thought=thought_data['thought'],
                        confidence=thought_data['confidence'],
                        reasoning=thought_data['reasoning'],
                        evaluation_score=thought_data['score']
                    )
                    node.add_child(child)
                    next_level.append(child)

            if not next_level:
                break
            current_level = next_level

        # Choose best leaf solution
        best_node = self._select_best_solution(root)
        path = best_node.get_path_to_root()
        steps_desc = []
        for i, n in enumerate(path):
            if i == 0:
                steps_desc.append(f"Problem: {n.state}")
            else:
                steps_desc.append(f"Step {i}: {n.thought}")
        reasoning = "\n".join(steps_desc)

        # Generate final solution based on selected thoughts
        final_prompt = (
            f"{goal}\n\nYou have the following proposed steps. Use them to produce the final answer.\n"
            f"{reasoning}\n\nProvide the final answer succinctly."
        )
        final_task = GenerationTask.create_from_prompt(final_prompt)
        final_task.max_tokens = 256
        final_task.temperature = 0.2
        yield from self.generation_controller.process([final_task])

        tasks[0].result = final_task.result

    def _generate_prompt(self, node: TOTNode, goal: str) -> str:
        return f"""Goal: {goal}

Current progress:
{node.state}

Generate {self.num_thoughts_per_step} different approaches or next steps to progress toward the goal.
Each approach should be distinct and well-reasoned.

Format your response as:
Approach 1: [detailed approach]
Approach 2: [detailed approach]
Approach 3: [detailed approach]"""

    def _parse_approaches(self, text: str) -> List[str]:
        approaches: List[str] = []
        lines = (text or "").strip().split('\n')
        current = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if (line.lower().startswith('approach') and ':' in line):
                if current:
                    approaches.append(current.strip())
                current = line.split(':', 1)[1].strip()
            elif current:
                current += " " + line
        if current:
            approaches.append(current.strip())
        return approaches

    def _evaluation_prompt(self, thought: str, goal: str, current_state: str) -> str:
        return f"""Goal: {goal}

Current state:
{current_state}

Proposed next step:
{thought}

Evaluate this proposed step on a scale of 1-10 considering:
1. How well it progresses toward the goal
2. How feasible it is to execute
3. How likely it is to lead to a successful solution

Provide your evaluation in this format:
Score: [1-10]
Confidence: [High/Medium/Low]
Reasoning: [brief explanation]"""

    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        lines = (text or "").strip().split('\n')
        score = 5.0
        confidence = 'Medium'
        reasoning = 'No reasoning provided'
        for line in lines:
            line = line.strip()
            if line.startswith('Score:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith('Confidence:'):
                confidence = line.split(':', 1)[1].strip()
            elif line.startswith('Reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
        return {'score': score, 'confidence': confidence, 'reasoning': reasoning}

    def _select_thoughts(self, evaluated_thoughts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not evaluated_thoughts:
            return []
        if self.selection_strategy == "best":
            return sorted(evaluated_thoughts, key=lambda x: x['score'], reverse=True)[: min(2, len(evaluated_thoughts))]
        if self.selection_strategy == "vote":
            return evaluated_thoughts[:2]
        return random.sample(evaluated_thoughts, min(2, len(evaluated_thoughts)))
    
    def _combine_state_and_thought(self, current_state: str, thought: str) -> str:
        """Combine current state with new thought."""
        if not current_state.strip():
            return thought
        return f"{current_state}\n\nNext step: {thought}"
    
    def _select_best_solution(self, root: TOTNode) -> TOTNode:
        """Select the best solution from all leaf nodes."""
        best_node = root
        best_score = -float('inf')
        
        def traverse(node: TOTNode):
            nonlocal best_node, best_score
            if node.is_leaf():
                # For leaf nodes, use evaluation score if available, otherwise use depth as heuristic
                score = node.evaluation_score if hasattr(node, 'evaluation_score') else node.depth
                if score > best_score:
                    best_score = score
                    best_node = node
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return best_node 