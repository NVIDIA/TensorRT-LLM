import os
import pickle

import torch

from tensorrt_llm.logger import logger


class ExpertStatistic:
    expert_statistic_obj = None

    @staticmethod
    def get():
        return ExpertStatistic.expert_statistic_obj

    @staticmethod
    def create(rank_id: int):
        # Enabled if EXPERT_STATISTIC_ITER_RANGE is set.
        span = os.environ.get('EXPERT_STATISTIC_ITER_RANGE', None)
        if span is None:
            return
        try:
            start, stop = span.strip().split('-')
            start, stop = int(start), int(stop)
        except ValueError as e:
            raise ValueError(str(e))
        ExpertStatistic.expert_statistic_obj = ExpertStatistic(
            rank_id, start, stop)

    @staticmethod
    def set_iter(iter_id: int):
        if ExpertStatistic.expert_statistic_obj is not None:
            return ExpertStatistic.expert_statistic_obj._set_iter(iter_id)

    @staticmethod
    def set_layer(layer_id: int):
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._set_layer(layer_id)

    @staticmethod
    def maybe_add_info(expert_count: int, token_selected_experts: torch.Tensor):
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._maybe_add_info(
                expert_count, token_selected_experts)

    def __init__(self, rank_id: int, start: int, stop: int):
        self.current_iter_id = None
        self.current_layer = None
        self.rank_id = rank_id
        self.start = start
        self.stop = stop
        self.records = {}

    @property
    def should_record(self) -> bool:
        return self.current_iter_id is not None and self.start <= self.current_iter_id < self.stop

    def _set_iter(self, iter_id: int) -> bool:
        self.current_iter_id = iter_id
        if iter_id == self.stop:
            logger.info(
                f'[ExpertStatistic] Rank={self.rank_id}, saving iter={iter_id}, start={self.start}, stop={self.stop}'
            )
            path = os.environ.get('EXPERT_STATISTIC_PATH', 'expert_statistic')
            filename = f'rank_{self.rank_id}.pkl'
            full_filename = os.path.join(path, filename)
            with open(full_filename, 'wb') as f:
                pickle.dump(self.records, f)
        return self.should_record

    def _set_layer(self, layer: int):
        self.current_layer = layer

    def _maybe_add_info(self, expert_count: int,
                        token_selected_experts: torch.Tensor):
        if not self.should_record:
            return

        key = (self.current_iter_id, self.current_layer)
        counts = torch.bincount(token_selected_experts.flatten(),
                                minlength=expert_count)
        self.records[key] = counts.cpu()
