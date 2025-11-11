from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional

import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager


class BalanceMethod(IntEnum):
    NotModified = 1
    Balanced = 2
    ImbalancedRanks = 3
    ImbalancedExperts = 4


class RunnerBase(ABC):
    @abstractmethod
    def create_run_pack(
        self,
        run_type: str,
        batch_size: int,
        seq_len_q: int,
        seq_len_kv_cache: int,
        kv_cache_manager: KVCacheManager,
        attn_workspace: Optional[torch.Tensor] = None,
    ):
        pass

    @abstractmethod
    def replace_routing_method(self, balance_method: BalanceMethod, balance_ratio: float):
        pass

    @staticmethod
    @abstractmethod
    def create_kv_cache_manager(
        pretrained_model_name_or_path,
        mapping,
        tokens_per_block,
        max_batch_size,
        max_seq_len,
        layer_indices,
    ):
        pass

    @staticmethod
    @abstractmethod
    def create_mapping(enable_attention_dp: bool):
        pass
