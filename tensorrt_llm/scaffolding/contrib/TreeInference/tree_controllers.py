# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tensorrt_llm.executor.result import CompletionOutput, GenerationResult
from tensorrt_llm.scaffolding.controller import Controller, ParallelProcess
from tensorrt_llm.scaffolding.task import GenerationTask, Task


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
        """node has no children"""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """node has no parent?"""
        return self.parent is None

    def add_child(self, child: 'TreeNode') -> 'TreeNode':
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
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_best_child(self,
                          exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select the child with the highest UCB1 score."""
        if not self.children:
            return self
        return max(self.children,
                   key=lambda child: child.ucb1_score(exploration_constant))


@dataclass
class TOTNode(TreeNode):
    """Node for Tree of Thoughts."""
    thought: str = ""
    confidence: str = "Medium"
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
                 num_thoughts_per_step: int = 3,
                 expansion_parallel_samples: int = 1):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.num_thoughts_per_step = num_thoughts_per_step
        self.expansion_parallel_samples = max(1, expansion_parallel_samples)

    def process(self, tasks: List[Task], **kwargs) -> Any:
        """Process tasks using MCTS with yield-based orchestration."""
        assert len(
            tasks) == 1, "MCTS Controller only supports single task processing"
        task = tasks[0]
        goal = kwargs.get('goal', 'Solve the problem step by step')
        initial_state = getattr(task, 'input_str', str(task)) or ""
        root = MCTSNode(state=initial_state)

        for iteration in range(self.max_iterations):
            # Selection
            node = root
            while not node.is_leaf() and not node.is_terminal:
                node = node.select_best_child(self.exploration_constant)

            # Expansion
            if not node.is_terminal and node.depth < self.max_depth:
                prompt = self._create_expansion_prompt(node, goal)

                # Parallelize expansion sampling if requested
                gen_controllers: List[Controller] = []
                gen_tasks_wrapped: List[List[GenerationTask]] = []
                gen_kwargs_list: List[Dict[str, Any]] = []

                for _ in range(self.expansion_parallel_samples):
                    gen_task = GenerationTask.create_from_prompt(prompt)
                    gen_task.max_tokens = 200
                    gen_task.temperature = 0.7
                    gen_controllers.append(self.generation_controller.clone())
                    gen_tasks_wrapped.append([gen_task])
                    gen_kwargs_list.append({})

                if gen_controllers:
                    yield ParallelProcess(gen_controllers, gen_tasks_wrapped,
                                          gen_kwargs_list)

                # Collect and merge thoughts from all parallel samples
                merged_thoughts: List[str] = []
                seen = set()
                for [gen_task] in gen_tasks_wrapped:
                    for t in self._parse_thoughts(gen_task.output_str or ""):
                        if t not in seen:
                            merged_thoughts.append(t)
                            seen.add(t)

                for thought in merged_thoughts[:self.num_thoughts_per_step]:
                    child_state = f"{node.state}\n{thought}".strip()
                    node.add_child(MCTSNode(state=child_state, reward=0.0))
                if node.children:
                    node = random.choice(node.children)

            # Evaluate node
            if self.reward_controller is not None:
                reward_task = GenerationTask()
                reward_task.input_str = initial_state
                completion_output = CompletionOutput(index=0, text=node.state)

                from tensorrt_llm.sampling_params import SamplingParams
                mock_sampling_params = SamplingParams()
                reward_result = GenerationResult.__new__(GenerationResult)
                reward_result._outputs = [completion_output]
                reward_result.sampling_params = mock_sampling_params
                reward_task.result = reward_result

                yield from self.reward_controller.process([reward_task])
                # Get reward from the reward controller
                if hasattr(self.reward_controller,
                           'scores') and self.reward_controller.scores:
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
        steps_desc = []
        if path:
            steps_desc.append(f"Problem: {path[0].state}")
        for i in range(1, len(path)):
            # Each child state's last line is the appended thought
            last_line = (path[i].state.split('\n')[-1]).strip()
            steps_desc.append(f"Step {i}: {last_line}")
        reasoning = "\n".join(steps_desc)
        final_prompt = (
            f"{goal}\n\nHere is a coherent reasoning trajectory.\n"
            f"{reasoning}\n\nNow provide the final answer succinctly.")
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
3. [Third thought]
...
"""

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
                 selection_strategy: str = "best",
                 branch_factor: int = 2):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.num_thoughts_per_step = num_thoughts_per_step
        self.selection_strategy = selection_strategy  # "best", "vote", "random"
        self.branch_factor = max(1, branch_factor)

    def process(self, tasks: List[Task], **kwargs) -> Any:
        """Process tasks using Tree of Thoughts with yield-based orchestration."""
        assert len(
            tasks) == 1, "TOT Controller only supports single task processing"
        task = tasks[0]
        goal = kwargs.get('goal', 'Solve the problem step by step')
        root_state = getattr(task, 'input_str', str(task)) or ""

        root = TOTNode(state=root_state, thought="Initial problem")
        current_level: List[TOTNode] = [root]
        iterations = 0
        stop = False

        for depth in range(self.max_depth):
            if stop:
                break

            next_level: List[TOTNode] = []

            # 1) Parallel generation for all nodes in the current level
            gen_controllers: List[Controller] = []
            gen_tasks_wrapped: List[List[GenerationTask]] = []
            gen_kwargs_list: List[Dict[str, Any]] = []
            node_order: List[TOTNode] = []

            for node in current_level:
                if stop:
                    break

                if node.is_terminal:
                    continue
                gen_prompt = self._generate_prompt(node, goal)
                gen_task = GenerationTask.create_from_prompt(gen_prompt)
                gen_task.max_tokens = 512
                gen_task.temperature = 0.8

                gen_controllers.append(self.generation_controller.clone())
                gen_tasks_wrapped.append([gen_task])
                gen_kwargs_list.append({})
                node_order.append(node)
                iterations += 1
                if (iterations >= self.max_iterations):
                    stop = True
                    break

            if gen_controllers:
                yield ParallelProcess(gen_controllers, gen_tasks_wrapped,
                                      gen_kwargs_list)

            # 2) Parse thoughts per node, then (optionally) reward scoring per node
            evaluated_by_node: Dict[int, List[Dict[str, Any]]] = {}

            # Prepare a single batched reward request across all nodes (leverages worker concurrency)
            all_reward_tasks: List[GenerationTask] = []
            node_to_task_indices: Dict[int, List[int]] = {}

            for idx, (node, [gen_task
                             ]) in enumerate(zip(node_order,
                                                 gen_tasks_wrapped)):
                thoughts = self._parse_approaches(gen_task.output_str or "")

                evaluated_thoughts: List[Dict[str, Any]] = []
                if not thoughts:
                    evaluated_by_node[idx] = evaluated_thoughts
                    continue

                if self.reward_controller is not None:
                    # Build reward tasks for this node
                    reward_indices_for_node: List[int] = []
                    from tensorrt_llm.sampling_params import SamplingParams
                    for thought in thoughts[:self.num_thoughts_per_step]:
                        reward_task = GenerationTask()
                        reward_task.input_str = root_state

                        candidate_content = self._combine_state_and_thought(
                            node.state, thought)

                        completion_output = CompletionOutput(
                            index=0, text=candidate_content)
                        mock_sampling_params = SamplingParams()
                        reward_result = GenerationResult.__new__(
                            GenerationResult)
                        reward_result._outputs = [completion_output]
                        reward_result.sampling_params = mock_sampling_params
                        reward_task.result = reward_result

                        reward_indices_for_node.append(len(all_reward_tasks))
                        all_reward_tasks.append(reward_task)

                    node_to_task_indices[idx] = reward_indices_for_node

                    evaluated_by_node[idx] = [{
                        'thought': t,
                        'score': 0.0,
                        'confidence': 'Medium',
                        'reasoning': 'PRM score'
                    } for t in thoughts[:self.num_thoughts_per_step]]
                else:
                    # Fallback: sequential lightweight LLM self-eval for this node
                    for thought in thoughts[:self.num_thoughts_per_step]:
                        eval_prompt = self._evaluation_prompt(
                            thought, goal, node.state)
                        eval_task = GenerationTask.create_from_prompt(
                            eval_prompt)
                        eval_task.max_tokens = 256
                        eval_task.temperature = 0.3

                        yield from self.generation_controller.process(
                            [eval_task])
                        evaluation = self._parse_evaluation(eval_task.output_str
                                                            or "")
                        evaluated_thoughts.append({
                            'thought':
                            thought,
                            'score':
                            evaluation['score'],
                            'confidence':
                            evaluation['confidence'],
                            'reasoning':
                            evaluation['reasoning']
                        })
                    evaluated_by_node[idx] = evaluated_thoughts

            # Run all reward evaluations in a single batch
            if self.reward_controller is not None and all_reward_tasks:
                yield from self.reward_controller.process(all_reward_tasks)
                scores = getattr(self.reward_controller, 'scores', None) or []

                for node_idx, indices in node_to_task_indices.items():
                    thoughts_for_node = evaluated_by_node[node_idx]
                    for local_j, task_index in enumerate(indices):
                        if task_index < len(scores):
                            normalized_score = float(scores[task_index])
                            if 0.0 <= normalized_score <= 1.0:
                                normalized_score *= 10.0
                            thoughts_for_node[local_j][
                                'score'] = normalized_score
                            thoughts_for_node[local_j]['confidence'] = (
                                'High' if normalized_score >= 8.0 else
                                'Medium' if normalized_score >= 5.0 else 'Low')

            # 3) Selection and child creation
            for idx, node in enumerate(node_order):
                evaluated_thoughts = evaluated_by_node.get(idx, [])
                if not evaluated_thoughts:
                    continue
                selected_thoughts = self._select_thoughts(evaluated_thoughts)
                if not selected_thoughts:
                    continue
                for thought_data in selected_thoughts:
                    child_state = self._combine_state_and_thought(
                        node.state, thought_data['thought'])
                    child = TOTNode(state=child_state,
                                    thought=thought_data['thought'],
                                    confidence=thought_data['confidence'],
                                    reasoning=thought_data['reasoning'],
                                    evaluation_score=thought_data['score'])
                    node.add_child(child)
                    next_level.append(child)

            if stop or not next_level:
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
            f"{reasoning}\n\nProvide the final answer succinctly.")
        final_task = GenerationTask.create_from_prompt(final_prompt)
        final_task.max_tokens = 1024
        final_task.temperature = 0.2
        # If the model uses R1-style <think> blocks, stop at the closing tag to avoid extra content
        try:
            if isinstance(final_task.stop, list):
                if '</think>' not in final_task.stop:
                    final_task.stop.append('</think>')
            elif isinstance(final_task.stop, str) and final_task.stop:
                final_task.stop = [final_task.stop, '</think>']
            else:
                final_task.stop = ['</think>']
        except Exception:
            final_task.stop = ['</think>']
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
        current: List[str] = []

        import re

        def is_new_item(line: str) -> Optional[str]:
            """Return the content of a new item header if the line starts a new approach/step item.
            Supports:
            - 'Approach N: ...' or 'Step N: ...'
            - 'N. ...'
            - '- ...' or '* ...'
            """
            line_stripped = line.strip()
            if not line_stripped:
                return None

            # Approach/Step N: ...
            m = re.match(r'^(?:approach|step)\s*\d+\s*[:\-\.]\s*(.*)$',
                         line_stripped,
                         flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()

            # Numbered list: '1. ...'
            m = re.match(r'^\d+\.\s*(.*)$', line_stripped)
            if m:
                return m.group(1).strip()

            # Bulleted list: '- ...' or '* ...'
            m = re.match(r'^[\-\*]\s*(.*)$', line_stripped)
            if m:
                return m.group(1).strip()

            return None

        def flush_current():
            nonlocal current
            if current:
                content = ' '.join(s for s in current).strip()
                if content:
                    approaches.append(content)
                current = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            header_content = is_new_item(line)
            if header_content is not None:
                # Start of a new item
                flush_current()
                if header_content:
                    current = [header_content]
                else:
                    current = []
            else:
                # Continuation of the current item
                current.append(line)

        flush_current()

        # Fallbacks if nothing was parsed
        if approaches:
            return approaches
        if not approaches:
            # Try splitting by blank lines
            paragraphs = [
                p.strip() for p in re.split(r'\n\s*\n', text or '')
                if p.strip()
            ]
            if paragraphs:
                return paragraphs
            # Final fallback to the whole text
            if (text or '').strip():
                return [(text or '').strip()]
            return []

    def _evaluation_prompt(self, thought: str, goal: str,
                           current_state: str) -> str:
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
        reasoning_lines: List[str] = []

        import re

        reasoning_started = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if 'score' in lower:
                # Extract the first number (handles '8', '8.5', '8/10')
                nums = re.findall(r'(\d+(?:\.\d+)?)', line)
                if nums:
                    try:
                        parsed = float(nums[0])
                        # Normalize if it's like '0.8' (unlikely) or clamp
                        if parsed <= 1.0 and '/10' in lower:
                            parsed *= 10.0
                        score = max(0.0, min(10.0, parsed))
                    except Exception:
                        pass
            elif 'confidence' in lower:
                # Normalize to High/Medium/Low if possible
                val = line.split(':', 1)[1].strip() if ':' in line else line
                v = val.lower()
                if 'high' in v or 'strong' in v:
                    confidence = 'High'
                elif 'low' in v or 'weak' in v:
                    confidence = 'Low'
                else:
                    confidence = 'Medium'
            elif 'reason' in lower:
                reasoning_started = True
                val = line.split(':', 1)[1].strip() if ':' in line else line
                if val:
                    reasoning_lines.append(val)
            else:
                if reasoning_started:
                    reasoning_lines.append(line)

        reasoning = ' '.join(reasoning_lines).strip() or 'No reasoning provided'
        return {
            'score': score,
            'confidence': confidence,
            'reasoning': reasoning
        }

    def _select_thoughts(
            self, evaluated_thoughts: List[Dict[str,
                                                Any]]) -> List[Dict[str, Any]]:
        if not evaluated_thoughts:
            return []
        if self.selection_strategy == "best":
            return sorted(
                evaluated_thoughts, key=lambda x: x['score'],
                reverse=True)[:min(self.branch_factor, len(evaluated_thoughts))]
        if self.selection_strategy == "vote":
            # Confidence-aware selection: prioritize High > Medium > Low confidence, then score
            def confidence_weight(conf: str) -> int:
                c = (conf or '').lower()
                if 'high' in c:
                    return 2
                if 'low' in c:
                    return 0
                return 1

            return sorted(
                evaluated_thoughts,
                key=lambda x: (confidence_weight(x.get('confidence', 'Medium')),
                               x.get('score', 0.0)),
                reverse=True)[:min(self.branch_factor, len(evaluated_thoughts))]
        return random.sample(evaluated_thoughts,
                             min(self.branch_factor, len(evaluated_thoughts)))

    def _combine_state_and_thought(self, current_state: str,
                                   thought: str) -> str:
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
                score = node.evaluation_score if hasattr(
                    node, 'evaluation_score') else node.depth
                if score > best_score:
                    best_score = score
                    best_node = node
            for child in node.children:
                traverse(child)

        traverse(root)
        return best_node
