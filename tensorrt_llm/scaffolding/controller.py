import copy
from abc import ABC
from enum import Enum
from typing import List, Tuple

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


# Controller runs multiple generation tasks.
class NativeGenerationController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, custom_sampling_params: dict = None):
        super().__init__()
        self.custom_sampling_params = copy.deepcopy(custom_sampling_params)

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            if not isinstance(task, GenerationTask):
                raise ValueError(
                    "NativeGenerationController requires exactly one GenerationTask"
                )

        for task in tasks:
            task.worker_tag = self.WorkerTag.GENERATION
            if kwargs.get("custom_sampling_params"):
                task.custom_sampling_params = kwargs.get(
                    "custom_sampling_params")
            elif self.custom_sampling_params:
                task.custom_sampling_params = self.custom_sampling_params

        yield tasks


# Controller runs multiple reward tasks.
class NativeRewardController(Controller):

    class WorkerTag(Enum):
        REWARD = "reward"

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            if not isinstance(task, RewardTask):
                raise ValueError(
                    "NativeRewardController requires exactly one RewardTask")

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

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1 and isinstance(tasks[0], GenerationTask), \
            "MajorityVoteController requires exactly one GenerationTask"

        sample_num = kwargs.get("sample_num", self.default_sample_num)
        generation_tasks = [copy.deepcopy(tasks[0]) for _ in range(sample_num)]
        yield from self.generation_controller.process(
            generation_tasks, **kwargs.get("generation_kwargs", {}))

        candidates = [task.output_str for task in generation_tasks]
        result = self.majority_vote(candidates,
                                    **kwargs.get("majority_vote_kwargs", {}))

        assert (isinstance(result, str))
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

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1 and isinstance(tasks[0], GenerationTask), \
            "BestOfNController requires exactly one GenerationTask"

        sample_num = kwargs.get("sample_num", self.default_sample_num)
        generation_tasks = [tasks[0].deepcopy() for _ in range(sample_num)]
        yield from self.generation_controller.process(
            generation_tasks, **kwargs.get("generation_kwargs"))

        reward_tasks = [
            RewardTask.create_from_generation_task(generation_task)
            for generation_task in generation_tasks
        ]
        yield from self.reward_controller.process(reward_tasks,
                                                  **kwargs.get("reward_kwargs"))

        # may used for upper layer controllers
        self.best_generation_task, self.best_reward_task = (self.select_best(
            generation_tasks, reward_tasks, **kwargs.get("select_best_kwargs")))
        tasks[0] = self.best_generation_task

    def select_best(self, generation_tasks: List[GenerationTask],
                    reward_tasks: List[RewardTask],
                    **kwargs) -> Tuple[GenerationTask, RewardTask]:
        max_reward_value_index = reward_tasks.index(
            max(reward_tasks, key=lambda x: x.reward_value))
        return generation_tasks[max_reward_value_index], reward_tasks[
            max_reward_value_index]
