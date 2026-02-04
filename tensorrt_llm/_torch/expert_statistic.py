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
    def should_record() -> bool:
        if ExpertStatistic.expert_statistic_obj is not None:
            return ExpertStatistic.expert_statistic_obj._should_record
        return False

    @staticmethod
    def set_iter(iter_id: int) -> None:
        if ExpertStatistic.expert_statistic_obj is not None:
            ExpertStatistic.expert_statistic_obj._set_iter(iter_id)

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
        self._records = {}        # Raw token_selected_experts tensors
        self._records_bincount = {}  # Aggregated bincount results (legacy format)

    @property
    def _should_record(self) -> bool:
        # Changed from `< self.stop` to `<= self.stop` to match nsys profiling behavior
        # Now "10-20" means iter 10, 11, ..., 20 (inclusive) for both nsys and statistics
        return self.current_iter_id is not None and self.start <= self.current_iter_id <= self.stop

    def _save(self) -> None:
        """Save recorded statistics to files in both formats."""
        logger.info(
            f'[ExpertStatistic] Rank={self.rank_id}, saving after stop={self.stop}, start={self.start}'
        )
        path = os.environ.get('EXPERT_STATISTIC_PATH', 'expert_statistic')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        # Save metadata
        if self.rank_id == 0:
            with open(f"{path}/meta_info.json", "w") as f:
                json.dump(self._meta_info, f)

        # Save bincount results (legacy format, same as before)
        safetensors.torch.save_file(
            self._records_bincount, f"{path}/rank{self.rank_id}.safetensors")

        # Save raw token_selected_experts tensors (new format)
        safetensors.torch.save_file(
            self._records, f"{path}/rank{self.rank_id}_raw.safetensors")

    def _set_iter(self, iter_id: int) -> None:
        # Save after the stop iteration has been recorded (when moving to stop+1)
        # This ensures iter=stop's data is captured before saving
        if self.current_iter_id == self.stop:
            self._save()
        self.current_iter_id = iter_id

    def _set_layer(self, layer: int) -> None:
        self.current_layer = layer

    def _maybe_add_info(self, expert_count: int,
                        token_selected_experts: torch.Tensor) -> None:
        if not self._should_record:
            return

        if self._meta_info is None:
            self._meta_info = {
                "num_experts": expert_count,
                "num_experts_per_token": token_selected_experts.size(-1)
            }

        key = f"{self.current_iter_id}_{self.current_layer}"

        # Save raw token_selected_experts tensor (new format)
        self._records[key] = token_selected_experts.cpu().contiguous()

        # Also compute and save bincount (legacy format)
        counts = torch.bincount(token_selected_experts.flatten(),
                                minlength=expert_count)
        if key not in self._records_bincount:
            self._records_bincount[key] = counts.cpu()
        else:
            self._records_bincount[key] += counts.cpu()
