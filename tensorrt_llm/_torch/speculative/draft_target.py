from dataclasses import dataclass

import torch

from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class DraftTargetConfig(SpecConfig):
    spec_dec_name: str = "DRAFT_TARGET"

    def __post_init__(self):
        if self.draft_model_path is None:
            raise ValueError("Path to Draft weights must be specified.")

        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)
        self.num_extra_kv_tokens = 0

    def update_from_model_config(self, model_config):
        pass

    def get_draft_model_prompt(self,
                               input_tokens: torch.Tensor) -> torch.Tensor:
        return input_tokens


@dataclass
class DraftTargetSpecMetadata(SpecMetadata):

    def __post_init__(self):
        pass

    def prepare(self):
        pass
