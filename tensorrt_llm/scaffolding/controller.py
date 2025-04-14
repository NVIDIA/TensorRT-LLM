import copy
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Tuple

from tensorrt_llm.scaffolding.math_utils import get_digit_majority_vote_result
from tensorrt_llm.scaffolding.task import (GenerationTask, RewardTask,
                                           ScaffoldingOutput, Task)


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


# Controller runs multiple reward tasks.
class NativeRewardController(Controller):

    class WorkerTag(Enum):
        REWARD = "reward"

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        yield tasks


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
                 default_sample_num: int = 1):
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
                sample_num: int = 1,
                generation_kwargs: dict = {},
                reward_kwargs: dict = {},
                select_best_kwargs: dict = {}):
        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [
            self.generation_controller.clone() for _ in range(sample_num)
        ]
        self.generation_tasks_list = [tasks for _ in range(sample_num)]
        generation_kwargs_list = [generation_kwargs for _ in range(sample_num)]

        yield ParallelProcess(generation_controllers,
                              self.generation_tasks_list,
                              generation_kwargs_list)

        # Some best of N algorithms create sample_num reward task lists while some just create one.
        # We maintain generic here as much as possible.
        self.reward_tasks_list = self.create_reward_tasks(
            self.generation_tasks_list)
        reward_paraller_num = len(self.reward_tasks_list)
        reward_controllers = [
            self.reward_controller.clone() for _ in range(reward_paraller_num)
        ]
        reward_kwargs_list = [reward_kwargs for _ in range(reward_paraller_num)]

        yield ParallelProcess(reward_controllers, self.reward_tasks_list,
                              reward_kwargs_list)

        # may used for upper layer controllers
        self.best_generation_task, self.best_reward_task = self.select_best(
            self.generation_tasks_list, self.reward_tasks_list,
            **select_best_kwargs)
        tasks = self.best_generation_task

    def select_best(self, generation_tasks: List[List[Task]],
                    reward_tasks: List[List[Task]],
                    **kwargs) -> Tuple[List[Task], List[Task]]:
        assert len(generation_tasks[0]) == 1 and isinstance(generation_tasks[0][0], GenerationTask), \
            "Should not use default select_best implementation for BestOfNController"
        assert len(reward_tasks[0]) == 1 and isinstance(reward_tasks[0][0], RewardTask), \
            "Should not use default select_best implementation for BestOfNController"
        # select the best generation task and reward task
        max_reward_value_index = reward_tasks.index(
            max(reward_tasks, key=lambda x: x[0].reward_value))
        return generation_tasks[max_reward_value_index], reward_tasks[
            max_reward_value_index]
