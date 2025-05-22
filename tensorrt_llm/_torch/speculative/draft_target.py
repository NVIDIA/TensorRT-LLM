from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch

from ..pyexecutor.sampler import SampleState, SampleStateTensors, TorchSampler
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class DraftTargetConfig(SpecConfig):
    spec_dec_name: str = "DRAFT_TARGET"
    pytorch_weights_path: Optional[str] = None

    def __post_init__(self):
        if self.pytorch_weights_path is None:
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



class DraftTargetSampler(TorchSampler):

    def _batch_sample(self, scheduled_requests, model_outputs) -> SampleState:
        logits = model_outputs["logits"]
        new_tokens_device = torch.argmax(logits, dim=-1)
        device = SampleStateTensors(new_tokens=new_tokens_device)
        host = SampleStateTensors(
            new_tokens=new_tokens_device.to('cpu', non_blocking=True))
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        return SampleState(scheduled_requests=scheduled_requests,
                           logits=logits,
                           device=device,
                           host=host,
                           sampler_event=sampler_event)
