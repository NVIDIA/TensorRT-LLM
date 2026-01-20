import copy
from typing import List, Optional

import numpy as np

from tensorrt_llm.scaffolding import (Controller, ParallelProcess,
                                      StreamGenerationTask, Task)

from .deep_conf_utils import ConfidenceInfo, majority_vote


class DeepConfOfflineController(Controller):

    def __init__(self, generation_controller: Controller, conf_group_size: int,
                 conf_threshold: float, **kwargs):
        super().__init__()
        self.generation_controller = generation_controller
        self.conf_group_size = conf_group_size
        self.conf_threshold = conf_threshold
        self.confidence_info = ConfidenceInfo(conf_group_size=conf_group_size,
                                              conf_threshold=conf_threshold)

    def clone(self):
        return DeepConfOfflineController(
            generation_controller=self.generation_controller.clone(),
            conf_group_size=self.conf_group_size,
            conf_threshold=self.conf_threshold,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks) == 1, "DeepConfOfflineController only supports one task"

        yield from self.generation_controller.process(tasks, **kwargs)

        for logprobs_dict, token_id in zip(tasks[0].logprobs,
                                           tasks[0].output_tokens):
            self.confidence_info.update_confidence_info(logprobs_dict, token_id)
        tasks[0].customized_result_fields[
            'confidence_info'] = self.confidence_info


class DeepConfOfflineMajorityVoteController(Controller):

    def __init__(self,
                 generation_controller: DeepConfOfflineController,
                 sample_num: int,
                 vote_policy: str = 'majority',
                 **kwargs):
        super().__init__()
        self.generation_controller = generation_controller
        self.sample_num = sample_num
        self.vote_policy = vote_policy

    def clone(self):
        return DeepConfOfflineMajorityVoteController(
            generation_controller=self.generation_controller.clone(),
            sample_num=self.sample_num,
            vote_policy=self.vote_policy)

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks) == 1, "DeepConfMajorityVoteController only supports one task"

        generation_controllers = [
            self.generation_controller.clone() for _ in range(self.sample_num)
        ]
        tasks_list = [copy.deepcopy(tasks) for _ in range(self.sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(kwargs) for _ in range(self.sample_num)
        ]
        yield ParallelProcess(generation_controllers, tasks_list,
                              generation_kwargs_list)

        tasks[0].result = majority_vote(tasks_list,
                                        vote_policy=self.vote_policy).result


class DeepConfOnlineController(Controller):

    def __init__(self, generation_controller: Controller, conf_group_size: int,
                 conf_threshold: float, **kwargs):
        super().__init__()
        self.generation_controller = generation_controller
        self.conf_group_size = conf_group_size
        self.conf_threshold = conf_threshold
        self.confidence_info = ConfidenceInfo(conf_group_size=conf_group_size,
                                              conf_threshold=conf_threshold)

    def clone(self):
        return DeepConfOnlineController(
            generation_controller=self.generation_controller.clone(),
            conf_group_size=self.conf_group_size,
            conf_threshold=self.conf_threshold,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks) == 1, "DeepThinkOnlineController only supports one task"
        online_task = StreamGenerationTask.create_from_generation_task(
            tasks[0], streaming_step=self.confidence_info.conf_group_size)

        last_step_index = 0
        while online_task.end_flag == False:
            generated_token_num = len(
                online_task.output_tokens) if online_task.output_tokens else 0
            for i in range(last_step_index, generated_token_num):
                logprobs_dict, token_id = online_task.logprobs[
                    i], online_task.output_tokens[i]
                self.confidence_info.update_confidence_info(
                    logprobs_dict, token_id)
                if self.confidence_info.should_stop():
                    online_task.cancel_flag = True
                    break

            last_step_index = generated_token_num
            yield from self.generation_controller.process([online_task],
                                                          **kwargs)

        tasks[0].result = online_task.result
        tasks[0].customized_result_fields[
            'confidence_info'] = self.confidence_info


class DeepConfOnlineMajorityVoteController(Controller):

    def __init__(self,
                 warmup_generation_controller: DeepConfOfflineController,
                 final_generation_controller: DeepConfOnlineController,
                 sample_num: int,
                 vote_policy: str = 'majority',
                 warmup_sample_num: int = 5,
                 confidence_percentile: int = 90,
                 **kwargs):
        super().__init__()
        self.warmup_generation_controller = warmup_generation_controller
        self.final_generation_controller = final_generation_controller
        self.sample_num = sample_num
        self.warmup_sample_num = warmup_sample_num
        assert sample_num >= warmup_sample_num, f"{sample_num=} must be greater than {warmup_sample_num=}"
        self.final_sample_num = sample_num - warmup_sample_num
        self.confidence_percentile = confidence_percentile
        self.vote_policy = vote_policy

    def clone(self):
        return DeepConfOnlineMajorityVoteController(
            warmup_generation_controller=self.warmup_generation_controller.
            clone(),
            final_generation_controller=self.final_generation_controller.clone(
            ),
            sample_num=self.sample_num,
            vote_policy=self.vote_policy,
            warmup_sample_num=self.warmup_sample_num,
            confidence_percentile=self.confidence_percentile,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(
            tasks
        ) == 1, "DeepConfOnlineMajorityVoteController only supports one task"
        # warm up to get conf_threshold
        warmup_tasks_list = yield from self.parallel_generate(
            tasks=tasks,
            sample_num=self.warmup_sample_num,
            generation_controller=self.warmup_generation_controller,
            **kwargs)

        conf_bar = self.get_conf_bar(warmup_tasks_list)
        final_tasks_list = yield from self.parallel_generate(
            tasks=tasks,
            sample_num=self.final_sample_num,
            generation_controller=self.final_generation_controller,
            conf_bar=conf_bar,
            **kwargs)

        tasks[0].result = majority_vote(warmup_tasks_list + final_tasks_list,
                                        vote_policy=self.vote_policy).result

    def parallel_generate(self,
                          tasks: List[Task],
                          sample_num,
                          generation_controller,
                          conf_bar: Optional[float] = None,
                          **kwargs):
        if sample_num == 0:
            return []

        generation_controllers = [
            generation_controller.clone() for _ in range(sample_num)
        ]
        if conf_bar is not None:
            for generation_controller in generation_controllers:
                generation_controller.confidence_info.conf_threshold = conf_bar
        tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(kwargs) for _ in range(sample_num)
        ]
        yield ParallelProcess(generation_controllers, tasks_list,
                              generation_kwargs_list)

        return tasks_list

    def get_conf_bar(self, tasks_list: List[List[Task]]) -> float:
        min_confs = [
            tasks[0].customized_result_fields['confidence_info'].
            get_min_conf_grouped() for tasks in tasks_list
        ]
        if len(min_confs) > 0:
            return float(
                np.percentile(min_confs, 100 - self.confidence_percentile))
        return self.final_generation_controller.conf_threshold
