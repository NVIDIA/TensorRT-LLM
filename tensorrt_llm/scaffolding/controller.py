import copy
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping

import torch
from torch.nn import functional as F

from tensorrt_llm.scaffolding.math_utils import get_digit_majority_vote_result
from tensorrt_llm.scaffolding.task import (GenerationTask, ScaffoldingOutput,
                                           Task)


class ScaffoldingOutput:

    def __init__(self):
        self.output_str = None
        # reserved for customized controller
        self.customized_output = None


class Controller(ABC):

    def clone(self):
        return copy.deepcopy(self)

    def generate(self, prompt: str, **kwargs) -> ScaffoldingOutput:
        task = GenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()

    def process(self, tasks: List[Task], **kwargs):
        raise NotImplementedError


@dataclass(frozen=True)
class ParallelProcess:
    controllers: List[Controller]
    tasks_list: List[List[Task]]
    kwargs_list: List[Mapping[str, Any]]


# Controller runs multiple generation tasks.
class NativeGenerationController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, custom_sampling_params: dict = None):
        super().__init__()
        self.custom_sampling_params = copy.deepcopy(
            custom_sampling_params) if custom_sampling_params else None

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            task.worker_tag = self.WorkerTag.GENERATION
            if self.custom_sampling_params:
                for key, value in self.custom_sampling_params.items():
                    if hasattr(task, key) and getattr(task, key) is None:
                        setattr(task, key, value)

        yield tasks


class NativeRewardController(Controller):

    class WorkerTag(Enum):
        REWARD = "reward"

    def process(self, tasks: List[Task], **kwargs):
        task = GenerationTask()
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        yield tasks


class QwenRewardController(NativeRewardController):
    """
    Controller that integrate multi Generation output into one prompt and get
    reward values from reward model.
    """

    def __init__(self, tokenizer, separate_token="<extra_0>"):  # nosec B107
        super().__init__()
        self.tokenizer = tokenizer
        self.separate_token = separate_token

    def _make_step_rewards(self, logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(
            -1)  # bs, seq_len, num_labels=2

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(
                -1, 2)[:, 1]  # num_separate_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def process(self, tasks: List[Task], **kwargs):
        # Combine messages using chat template
        content = "".join(
            (task.output_str + self.separate_token) for task in tasks)
        messages = [
            {
                "role":
                "system",
                "content":
                "Please reason step by step, and put your final answer within \\boxed{}."
            },
            {
                "role": "user",
                "content": tasks[0].input_str
            },
            {
                "role": "assistant",
                "content": content
            },
        ]
        combined_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        # TODO: support input_ids as model input, avoid doing it again in worker
        merged_task = GenerationTask.create_from_prompt(combined_prompt)
        merged_task.worker_tag = self.WorkerTag.REWARD

        # TODO: pack this logic
        merged_task.max_tokens = 1
        merged_task.return_context_logits = True

        yield [merged_task]

        assert merged_task.context_logits is not None
        # TODO: consider running on cpu to not interrupt worker or move
        # tokenizer to a worker
        input_ids = self.tokenizer.encode(
            combined_prompt,
            return_tensors="pt",
        ).to(merged_task.context_logits.device)

        # TODO: align add_special_tokens with SamplingParams
        token_masks = (input_ids == self.tokenizer.encode(
            self.separate_token, add_special_tokens=True)[0])
        all_scores_res = self._make_step_rewards(merged_task.context_logits,
                                                 token_masks)

        return all_scores_res


# Controller runs a single generation task with majority vote.
class MajorityVoteController(Controller):

    def __init__(self,
                 generation_controller: Controller,
                 default_sample_num: int = 1):
        super().__init__()
        self.generation_controller = generation_controller
        self.default_sample_num = default_sample_num

    def clone(self):
        # As we don't know the behavior of the generation_controller's clone method,
        # we explicitly call clone method instead of simply using deepcopy.
        generation_controller = self.generation_controller.clone()
        return MajorityVoteController(generation_controller,
                                      self.default_sample_num)

    def process(self,
                tasks: List[Task],
                sample_num: int = 1,
                generation_kwargs: dict = {},
                majority_vote_kwargs: dict = {}):
        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [
            self.generation_controller.clone() for _ in range(sample_num)
        ]
        tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(generation_kwargs) for _ in range(sample_num)
        ]

        yield ParallelProcess(generation_controllers, tasks_list,
                              generation_kwargs_list)

        candidates = [tasks[0].output_str for tasks in tasks_list]
        result = self.majority_vote(candidates, **majority_vote_kwargs)

        assert isinstance(result, str), "majority_vote failed"
        # The task returned by majority vote does not have output_tokens and logits.
        tasks[0].output_str = result

    def majority_vote(self, candidates: List[str], **kwargs) -> str:
        return get_digit_majority_vote_result(candidates)


# Controller runs a single generation task with best of N.
class BestOfNController(Controller):

    def __init__(self,
                 generation_controller: Controller,
                 reward_controller: Controller,
                 default_sample_num: int = 4):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.default_sample_num = default_sample_num

    def clone(self):
        generation_controller = self.generation_controller.clone()
        reward_controller = self.reward_controller.clone()
        return BestOfNController(generation_controller, reward_controller,
                                 self.default_sample_num)

    def process(self,
                tasks: List[Task],
                sample_num: int = 4,
                generation_kwargs: dict = {},
                reward_kwargs: dict = {},
                select_best_kwargs: dict = {}):
        assert len(tasks) == 1, "BestOfNController only supports one task"
        task = tasks[0]

        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [
            self.generation_controller for _ in range(sample_num)
        ]
        generation_kwargs_list = [generation_kwargs for _ in range(sample_num)]
        generation_tasks_list = [copy.deepcopy(task) for _ in range(sample_num)]

        # yield from self.generation_controller.process(generation_tasks_list,
        #                                               **generation_kwargs)
        yield ParallelProcess(generation_controllers,
                              [[t] for t in generation_tasks_list],
                              generation_kwargs_list)

        reward_values = yield from self.reward_controller.process(
            generation_tasks_list, **reward_kwargs)

        best_task = self.select_best(generation_tasks_list, reward_values,
                                     **select_best_kwargs)
        task.output_str = best_task.output_str

    def select_best(self, tasks: List[Task], reward_values, **kwargs) -> Task:
        max_index = torch.argmax(torch.tensor(reward_values)).item()
        return tasks[max_index]
