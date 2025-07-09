from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.speculative.drafter import Drafter

from .interface import SpecConfig, SpeculativeDecodingMode


@dataclass
class UserProvidedConfig(SpecConfig):
    """
    Configuration for user provided speculative decoding.
    """
    # The name of speculative decoding.
    spec_dec_name = "USER_PROVIDED"

    num_extra_kv_tokens: int = 0
    max_draft_len: int = 0
    drafter: Optional[Drafter] = None

    def __post_init__(self) -> None:
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)

    def update_from_model_config(self, model_config):
        pass
