import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import List
from collections import deque

from tensorrt_llm.scaffolding import Controller, NativeGenerationController, ParallelProcess, Task

@dataclass
class ConfidenceInfo:
    conf_grouped: float
    conf_list: list[float]
    conf_group_list: deque[float]
    conf_group_size: int
    conf_threshold: float

def update_confidence_info(confidence_info: ConfidenceInfo, logprobs: List[float]):
    if len(logprobs) > 1:
        new_conf = -sum(logprobs[1:]) / len(logprobs[1:])
    else:
        new_conf = 0.0

    confidence_info.conf_list.append(new_conf)

    if len(confidence_info.conf_group_list) < confidence_info.conf_group_size:
        confidence_info.conf_group_list.append(new_conf)
        confidence_info.conf_grouped += new_conf
    else:
        confidence_info.conf_grouped -= confidence_info.conf_group_list.popleft()
        confidence_info.conf_group_list.append(new_conf)
        confidence_info.conf_grouped += new_conf


class DeepThinkOfflineController(NativeGenerationController):

    def __init__(self, conf_group_size: int, conf_threshold: float, sampling_params: dict = None, streaming: bool = False):
        super().__init__(sampling_params, streaming)
        self.confidence_info = ConfidenceInfo(conf_grouped=0.0, conf_list=[], conf_group_list=deque([]), conf_group_size=conf_group_size, conf_threshold=conf_threshold)

    def process(self, tasks: List[Task], **kwargs):
        yield from super().process(tasks, **kwargs)
        assert (len(tasks) == 1, "DeepThinkOfflineController only supports one task")
        for logprobs in tasks[0].result.outputs[0].logprobs:
            update_confidence_info(self.confidence_info, logprobs)


class DeepThinkMajorityVoteController(Controller):
    def __init__(self, generation_controller: DeepThinkOfflineController,
                 sample_num: int = 1):
        self.generation_controller = generation_controller
        self.sample_num = sample_num

    def clone(self):
        return DeepThinkMajorityVoteController(self.generation_controller.clone(), self.default_sample_num)

    def process(self, tasks: List[Task], generation_kwargs: dict = {}):
        generation_controllers = [
            self.generation_controller.clone() for _ in range(self.sample_num)
        ]
        tasks_list = [copy.deepcopy(tasks) for _ in range(self.sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(generation_kwargs) for _ in range(self.sample_num)
        ]

        yield ParallelProcess(generation_controllers, tasks_list, generation_kwargs_list)

        candidates = [tasks[0].output_str for tasks in tasks_list]
        return majority_answer


class DeepThinkOnlineController(Controller):
    def __init__(self, generation_controller: DeepThinkOfflineController,
                 sample_num: int = 1):
        self.generation_controller = generation_controller
        self.sample_num = sample_num

    def clone(self):
        return DeepThinkOnlineController(self.generation_controller.clone(), self.sample_num)
        