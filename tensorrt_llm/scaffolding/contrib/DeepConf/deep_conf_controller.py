import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Mapping, Tuple
from collections import deque

from tensorrt_llm.scaffolding import Controller, NativeGenerationController, StreamGenerationTask, MajorityVoteController,ParallelProcess, Task

@dataclass
class ConfidenceInfo:
    conf_grouped: float
    conf_list: list[float]
    conf_group_list: deque[float]
    conf_group_size: int
    conf_threshold: float

# This function is used to fake expand the logprobs_dict to the topk size
# TODO: remove this function after the topk logprobs is supported with torch backend
def fake_expand_logprobs_dict(logprobs_dict: Mapping[int, 'Logprob'], topk: int):
    for token_id, logprob in logprobs_dict.items():
        real_token_id = token_id
        real_logprob = logprob

    for i in range(topk - 1):
        new_logprob = copy.deepcopy(logprob)
        new_logprob.logprob = new_logprob.logprob / (i + 1)
        logprobs_dict[real_token_id + i + 1] = new_logprob

def update_confidence_info(confidence_info: ConfidenceInfo, token_dict: Mapping[int, 'Logprob'], token_id: int):
    #print("fredw test update_confidence_info len(token_dict): " + str(len(token_dict)))
    if len(token_dict) > 1:
        sum = 0.0
        for logprob_token_id, logprob in token_dict.items():
            if logprob_token_id != token_id:
                sum += logprob.logprob
        new_conf = -sum / (len(token_dict) - 1)
    else:
        new_conf = 0.0

    confidence_info.conf_list.append(new_conf)

    if len(confidence_info.conf_group_list) < confidence_info.conf_group_size:
        confidence_info.conf_grouped += new_conf
    else:
        confidence_info.conf_grouped -= confidence_info.conf_group_list.popleft()
        confidence_info.conf_grouped += new_conf


class DeepConfOfflineController(NativeGenerationController):

    def __init__(self, 
                 conf_group_size: int, 
                 conf_threshold: float, 
                 logprobs_topk:int = 20,
                 sampling_params: dict = None, 
                 streaming: bool = False):
        super().__init__(sampling_params, streaming)
        self.logprobs_topk = logprobs_topk
        self.confidence_info = ConfidenceInfo(conf_grouped=0.0, conf_list=[], conf_group_list=deque([]), conf_group_size=conf_group_size, conf_threshold=conf_threshold)

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1, "DeepConfOfflineController only supports one task"
        tasks[0].num_logprobs = self.logprobs_topk
        yield from super().process(tasks, **kwargs)
        for logprobs_dict, token_id in zip(tasks[0].logprobs, tasks[0].output_tokens):
            fake_expand_logprobs_dict(logprobs_dict, self.logprobs_topk)
            update_confidence_info(self.confidence_info, logprobs_dict, token_id)
        tasks[0].costimized_result_fields['confidence_info'] = self.confidence_info


class DeepConfMajorityVoteController(MajorityVoteController):
    
    # TODO: implement majority vote
    # ConfidenceInfo is on costimized_result_fields['confidence_info'] of each task
    def majority_vote(self, candidates_tasks: List[Task], **kwargs) -> Tuple[int, str]:
        pass


class DeepConfOnlineController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, 
                 conf_group_size: int, 
                 conf_threshold: float,
                 logprobs_topk:int = 20,
                 sampling_params: dict = None):
        super().__init__()
        self.sampling_params = sampling_params
        self.logprobs_topk = logprobs_topk
        self.confidence_info = ConfidenceInfo(conf_grouped=0.0, conf_list=[], conf_group_list=deque([]), conf_group_size=conf_group_size, conf_threshold=conf_threshold)
    
    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1, "DeepThinkOnlineController only supports one task"
        online_task = StreamGenerationTask.create_from_generation_task(tasks[0])
        online_task.streaming_step = self.confidence_info.conf_group_size
        online_task.num_logprobs = self.logprobs_topk
        online_task.worker_tag = self.WorkerTag.GENERATION

        for key, value in self.sampling_params.items():
            if getattr(online_task, key) is None:
                setattr(online_task, key, value)

        last_step_index = 0
        while online_task.end_flag == False:
            yield [online_task]
            for i in range(last_step_index, len(online_task.output_tokens)):
                logprobs_dict, token_id = online_task.logprobs[i], online_task.output_tokens[i]
                fake_expand_logprobs_dict(logprobs_dict, self.logprobs_topk)
                update_confidence_info(self.confidence_info, logprobs_dict, token_id)
            if self.should_stop(self.confidence_info):
                online_task.cancel_flag = True
                yield [online_task]
                break
            last_step_index = len(online_task.output_tokens)
        tasks[0].result = online_task.result
        tasks[0].costimized_result_fields['confidence_info'] = self.confidence_info

    # TODO: implement should_stop
    def should_stop(self, confidence_info: ConfidenceInfo) -> bool:
        return False


class AdaptiveMajorityVoteController(Controller):
    def __init__(self, generation_controller: Controller, sample_num_per_round: int, max_sample_num: float):
        super().__init__()
        self.generation_controller = generation_controller
        self.sample_num_per_round = sample_num_per_round
        self.max_sample_num = max_sample_num

    def clone(self):
        return AdaptiveMajorityVoteController(self.generation_controller.clone(), self.sample_num, self.max_sample_num)
    
    def process(self, tasks: List[Task], **kwargs):
        candidates = []
        should_continue = True
        sample_next_round = self.sample_num_per_round
        while len(candidates) < self.max_sample_num and should_continue:
            sample_num = sample_next_round
            generation_controllers = [
                self.generation_controller.clone() for _ in range()
            ]
            tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
            generation_kwargs_list = [
                copy.deepcopy(kwargs) for _ in range(sample_num)
            ]
            yield ParallelProcess(generation_controllers, tasks_list, generation_kwargs_list)
            
            should_continue, sample_next_round = self.next_round_judge(candidates, tasks_list)
            candidates.extend(tasks_list)

        return self.majority_vote(candidates, **kwargs)

    def next_round_judge(self, candidates: List[Task], tasks_list: List[Task]) -> Tuple[bool, int]:
        pass

    def majority_vote(self, candidates: List[Task], **kwargs) -> Tuple[int, str]:
        pass