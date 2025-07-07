import json
import os

import safetensors
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
    def set_iter(iter_id: int) -> bool:
        if ExpertStatistic.expert_statistic_obj is not None:
            return ExpertStatistic.expert_statistic_obj._set_iter(iter_id)
        else:
            return False

    @staticmethod
    def set_layer(layer_id: int) -> None:
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._set_layer(layer_id)

    @staticmethod
    def maybe_add_info(expert_count: int,
                       token_selected_experts: torch.Tensor) -> None:
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._maybe_add_info(
                expert_count, token_selected_experts)

    def __init__(self, rank_id: int, start: int, stop: int) -> None:
        self.current_iter_id = None
        self.current_layer = None
        self.rank_id = rank_id
        self.start = start
        self.stop = stop
        self._meta_info = None
        self._records = {}
        self._cooccurrence = {}
        self._token_selected_experts = {}

    @property
    def should_record(self) -> bool:
        return self.current_iter_id is not None and self.start <= self.current_iter_id < self.stop

    def _set_iter(self, iter_id: int) -> bool:
        self.current_iter_id = iter_id
        if iter_id == self.start or iter_id == self.stop:
            logger.info(
                f'[ExpertStatistic] Rank={self.rank_id}, iter={iter_id}, start={self.start}, stop={self.stop}'
            )
        if iter_id == self.stop:
            path = os.environ.get('EXPERT_STATISTIC_PATH', 'expert_statistic')
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            if self.rank_id == 0:
                with open(f"{path}/meta_info.json", "w") as f:
                    json.dump(self._meta_info, f)
            safetensors.torch.save_file(
                self._records, f"{path}/rank{self.rank_id}.safetensors")
            safetensors.torch.save_file(
                self._cooccurrence,
                f"{path}/cooccurrence_rank{self.rank_id}.safetensors")
            safetensors.torch.save_file(
                self._token_selected_experts,
                f"{path}/token_selected_experts_rank{self.rank_id}.safetensors")
        return self.should_record

    def _set_layer(self, layer: int) -> None:
        self.current_layer = layer

    def _maybe_add_info(self, expert_count: int,
                        token_selected_experts: torch.Tensor) -> None:
        if not self.should_record:
            return

        if self._meta_info is None:
            self._meta_info = {
                "num_experts": expert_count,
                "num_experts_per_token": token_selected_experts.size(-1)
            }

        key = f"{self.current_iter_id}_{self.current_layer}"
        counts = token_selected_experts.flatten().bincount(
            minlength=expert_count).to(torch.int32)
        if key not in self._records:
            self._records[key] = counts.cpu()
        else:
            self._records[key] += counts.cpu()

        cooccurrence = torch.zeros(expert_count,
                                   expert_count,
                                   dtype=torch.int32,
                                   device=token_selected_experts.device)
        for i in range(token_selected_experts.size(0)):
            rows, cols = torch.meshgrid(token_selected_experts[i],
                                        token_selected_experts[i],
                                        indexing="ij")
            cooccurrence[rows, cols] += 1
        if key not in self._cooccurrence:
            self._cooccurrence[key] = cooccurrence.cpu()
        else:
            self._cooccurrence[key] += cooccurrence.cpu()

        if key not in self._token_selected_experts:
            self._token_selected_experts[key] = token_selected_experts.cpu()
        else:
            self._token_selected_experts[key] = torch.cat([
                self._token_selected_experts[key],
                token_selected_experts.cpu()
            ],
                                                          dim=0)
