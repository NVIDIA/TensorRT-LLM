import copy
from abc import ABC
from enum import Enum
from typing import Any, List, Mapping, Tuple

import torch
from torch.nn import functional as F

from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.logger import logger
from tensorrt_llm.scaffolding.math_utils import get_digit_majority_vote_result
from tensorrt_llm.scaffolding.task import GenerationTask, Task


class Controller(ABC):

    def __init__(self):
        self.task_collections = {}

    def clone(self):
        return copy.deepcopy(self)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        task = GenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()

    def process(self, tasks: List[Task], **kwargs):
        raise NotImplementedError


class ParallelProcess:

    def __init__(self, controllers: List[Controller],
                 tasks_list: List[List[Task]], kwargs_list: List[Mapping[str,
                                                                         Any]]):
        self.sub_gens = []
        for controller, tasks, kwargs in zip(controllers, tasks_list,
                                             kwargs_list):
            gen = controller.process(tasks, **kwargs)
            self.sub_gens.append(gen)


# Controller runs multiple generation tasks.
class NativeGenerationController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, sampling_params: dict = None, streaming: bool = False):
        super().__init__()
        if sampling_params is None:
            sampling_params = {}
        for key, value in list(sampling_params.items()):
            if key not in GenerationTask.__annotations__:
                logger.warning(
                    f"{key} is not a supported field for GenerationTask")
                sampling_params.pop(key)
        self.sampling_params = sampling_params
        self.streaming = streaming

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            task.worker_tag = self.WorkerTag.GENERATION
            for key, value in self.sampling_params.items():
                if getattr(task, key) is None:
                    setattr(task, key, value)
            task.streaming = self.streaming

        yield tasks


class NativeRewardController(Controller):

    def __init__(self):
        self.scores = None

    class WorkerTag(Enum):
        REWARD = "reward"

    def process(self, tasks: List[Task], **kwargs):
        task = GenerationTask()
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        yield tasks


class PRMController(NativeRewardController):
    """
    Use PRM(Process Reward Model) to get the score of output. Will split
    output into multi steps if `split_steps` is True. Otherwise will only
    extract last token score.

    Output:
        The scores of each task will be stored in `self.scores`.

    Example:
        Suppose the model output is split using a special token like <extra_0>:
        Input: "Step1,...<extra_0>Step2,...\\boxed{answer}.<extra_0>."
        The function will mask out logits and remain only scores at separate_token.
        Each represent the probability score for each step, eg: [0.98, 1.0].
        We can assume the output is good when product of all probabilities is high.
    """

    def __init__(
            self,
            tokenizer,
            split_steps=True,
            step_token="\n\n",
            separate_token="<extra_0>",  # nosec B107
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split_steps = split_steps
        self.step_token = step_token
        self.separate_token = separate_token

    def _calc_steps_score(self, logits, token_mask):
        probs = F.softmax(logits, dim=-1)  # seq_len, num_labels=2
        masked_probs = probs * token_mask.unsqueeze(-1)[0]

        # only keep the logits at the separate_token
        step_probs = masked_probs[masked_probs != 0].view(-1, 2)[:, 1]
        score = torch.prod(step_probs).item()
        return score

    def _calc_last_token_score(self, logits):
        # seq_len, num_labels=2
        probs = F.softmax(logits, dim=-1)
        score = probs[-1, 1].item()
        return score

    def process(self, tasks: List[Task], **kwargs):
        reward_tasks = []
        for task in tasks:
            if self.split_steps:
                steps = task.output_str.split(self.step_token)
                content = "".join(
                    (step + self.separate_token) for step in steps)
            else:
                content = self.separate_token + task.output_str + self.separate_token
            # Combine messages using chat template
            messages = [
                {
                    "role":
                    "system",
                    "content":
                    "Please reason step by step, and put your final answer within \\boxed{}."
                },
                {
                    "role": "user",
                    "content": task.input_str
                },
                {
                    "role": "assistant",
                    "content": content
                },
            ]
            processed_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

            # TODO: support input_ids as model input, avoid doing it again in worker
            reward_task = GenerationTask.create_from_prompt(processed_prompt)
            reward_task.worker_tag = self.WorkerTag.REWARD

            # TODO: pack this logic
            reward_task.max_tokens = 1
            reward_task.return_context_logits = True
            reward_tasks.append(reward_task)

        yield reward_tasks

        scores = []
        for reward_task in reward_tasks:
            assert reward_task.context_logits is not None
            # TODO: consider running on cpu to not interrupt worker or move
            # tokenizer to a worker
            input_ids = self.tokenizer.encode(
                reward_task.input_str,
                return_tensors="pt",
            ).to(reward_task.context_logits.device)

            if self.split_steps:
                # TODO: align add_special_tokens with SamplingParams
                token_mask = (input_ids == self.tokenizer.encode(
                    self.separate_token, add_special_tokens=True)[0])
                score = self._calc_steps_score(reward_task.context_logits,
                                               token_mask)
            else:
                score = self._calc_last_token_score(reward_task.context_logits)
            scores.append(score)

        self.scores = scores


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

        majority_index, majority_answer = self.majority_vote(
            tasks_list, **majority_vote_kwargs)

        assert isinstance(majority_answer, str), "majority_vote failed"
        # The task returned by majority vote does not have output_tokens and logits.
        tasks[0].result = tasks_list[majority_index][0].result

    def majority_vote(self, candidates_tasks: List[List[Task]],
                      **kwargs) -> Tuple[int, str]:
        candidates = [tasks[0].output_str for tasks in candidates_tasks]
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
        generation_tasks = [copy.deepcopy(task) for _ in range(sample_num)]

        yield ParallelProcess(generation_controllers,
                              [[t] for t in generation_tasks],
                              generation_kwargs_list)

        yield from self.reward_controller.process(generation_tasks,
                                                  **reward_kwargs)

        assert self.reward_controller.scores is not None
        reward_values = self.reward_controller.scores

        for i, gen_task, reward_value in zip(range(sample_num),
                                             generation_tasks, reward_values):
            logger.info(
                f"[output {i}, score {reward_value}]:\n{gen_task.output_str}")

        best_task, best_idx = self.select_best(generation_tasks, reward_values,
                                               **select_best_kwargs)
        task.result = best_task.result

    def select_best(self, tasks: List[Task], reward_values, **kwargs) -> Task:
        max_index = torch.argmax(torch.tensor(reward_values)).item()
        return tasks[max_index], max_index
