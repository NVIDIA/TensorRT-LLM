import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import List, Mapping

import numpy as np

from tensorrt_llm.scaffolding import Task, extract_answer_from_boxed


@dataclass
class ConfidenceInfo:
    conf_grouped: float = 0.0
    conf_group_size: int = 128
    conf_threshold: float = 0.0
    conf_list: list[float] = field(default_factory=list)
    conf_group_list: deque[float] = field(default_factory=deque)
    avg_conf_group_list: list[float] = field(default_factory=list)

    def get_min_conf_grouped(self):
        if len(self.avg_conf_group_list) == 0:
            print(
                f"Warning: no valid conf group yet, maybe you should decrease conf_group_size"
            )
            return self.conf_threshold
        return min(self.avg_conf_group_list)

    def should_stop(self):
        return self.avg_conf_group_list[-1] < self.conf_threshold if len(
            self.conf_list) >= self.conf_group_size else False

    def get_statistics(self,
                       tail_tokens: int = 2048,
                       bottom_percent: float = 0.1):
        # global mean confidence
        self.mean_conf = np.mean(self.conf_list)

        # tail mean confidence
        tail_conf_list = self.conf_list[-tail_tokens:] if len(
            self.conf_list) > tail_tokens else self.conf_list
        self.tail_mean_conf = np.mean(tail_conf_list)

        # bottom window mean confidence and min window confidence
        if len(self.avg_conf_group_list) == 0:
            self.bottom_window_mean_conf = np.mean(self.conf_list)
            self.min_window_conf = self.bottom_window_mean_conf
        else:
            num_bottom = max(
                1, int(len(self.avg_conf_group_list) * bottom_percent))
            if num_bottom == 1:
                self.bottom_window_mean_conf = np.min(self.avg_conf_group_list)
            else:
                self.bottom_window_mean_conf = np.mean(
                    np.partition(self.avg_conf_group_list,
                                 num_bottom - 1)[:num_bottom])
            self.min_window_conf = np.min(self.avg_conf_group_list)

    def update_confidence_info(self, token_dict: Mapping[int, 'Logprob'],
                               token_id: int):
        mean_logprob = np.mean(
            [logprob_obj.logprob for logprob_obj in token_dict.values()])
        new_conf = round(-mean_logprob, 3)
        self.conf_list.append(new_conf)
        self.conf_grouped += new_conf
        self.conf_group_list.append(new_conf)
        if len(self.conf_group_list) > self.conf_group_size:
            self.conf_grouped -= self.conf_group_list.popleft()
        if len(self.conf_group_list) == self.conf_group_size:
            self.avg_conf_group_list.append(self.conf_grouped /
                                            self.conf_group_size)


def basic_majority_vote(tasks: List[Task], answers, **kwargs) -> Task:
    majority_answer = Counter(answers).most_common(1)[0][0]
    return tasks[answers.index(majority_answer)]


def weighted_majority_vote(tasks: List[Task],
                           answers,
                           confidences,
                           filter_top_percent=1.0,
                           type="",
                           **kwargs) -> Task:
    if filter_top_percent < 1.0:
        sorted_indices = np.argsort(confidences)[::-1]
        num_keep = max(1, int(len(confidences) * filter_top_percent))
        save_indices = sorted_indices[:num_keep].tolist()

        tasks = [tasks[i] for i in save_indices]
        answers = [answers[i] for i in save_indices]
        confidences = [confidences[i] for i in save_indices]

    answer_to_weights = {}
    for answer, confidence in zip(answers, confidences):
        answer_to_weights[answer] = answer_to_weights.get(answer,
                                                          0.0) + confidence
    majority_answer = max(answer_to_weights, key=answer_to_weights.get)
    return tasks[answers.index(majority_answer)]


def majority_vote(tasks_list: List[List[Task]], vote_policy: str = 'majority'):
    tasks = [tasks[0] for tasks in tasks_list]
    for task in tasks:
        task.customized_result_fields[
            'extracted_answer'] = extract_answer_from_boxed(task.output_str)
        task.customized_result_fields['confidence_info'].get_statistics()
    valid_tasks = [
        task for task in tasks
        if task.customized_result_fields['extracted_answer']
    ]

    if len(valid_tasks) == 0:
        print(
            "Warning: No valid tasks, maybe you should increase max_output_len, a random task will be returned"
        )
        return random.choice(tasks)

    answers = [
        task.customized_result_fields['extracted_answer']
        for task in valid_tasks
    ]
    confidences = [
        task.customized_result_fields['confidence_info'] for task in valid_tasks
    ]

    match vote_policy:
        case 'majority':
            return basic_majority_vote(valid_tasks, answers=answers)
        case 'mean_confidence_weighted':
            mean_confidences = [conf.mean_conf for conf in confidences]
            return weighted_majority_vote(valid_tasks,
                                          answers=answers,
                                          confidences=mean_confidences,
                                          type=vote_policy)
        case 'tail_confidence_weighted':
            tail_confidences = [conf.tail_mean_conf for conf in confidences]
            return weighted_majority_vote(valid_tasks,
                                          answers=answers,
                                          confidences=tail_confidences,
                                          type=vote_policy)
        case 'bottom_window_weighted':
            bottom_window_confidences = [
                conf.bottom_window_mean_conf for conf in confidences
            ]
            return weighted_majority_vote(valid_tasks,
                                          answers=answers,
                                          confidences=bottom_window_confidences,
                                          type=vote_policy)
        case 'min_window_weighted':
            min_window_confidences = [
                conf.min_window_conf for conf in confidences
            ]
            return weighted_majority_vote(valid_tasks,
                                          answers=answers,
                                          confidences=min_window_confidences,
                                          type=vote_policy)
        case 'top10_tail_filtered':
            tail_confidences = [conf.tail_mean_conf for conf in confidences]
            return weighted_majority_vote(valid_tasks,
                                          answers=answers,
                                          confidences=tail_confidences,
                                          filter_top_percent=0.1,
                                          type=vote_policy)
        case 'top10_bottom_window_filtered':
            bottom_window_confidences = [
                conf.bottom_window_mean_conf for conf in confidences
            ]
            return weighted_majority_vote(valid_tasks,
                                          answers=answers,
                                          confidences=bottom_window_confidences,
                                          filter_top_percent=0.1,
                                          type=vote_policy)
        case _:
            raise NotImplementedError(
                f"Vote policy '{vote_policy}' is not implemented")
