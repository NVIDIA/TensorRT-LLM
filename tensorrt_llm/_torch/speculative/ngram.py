from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch

from ..pyexecutor.decoder import DecoderState, TorchDecoder
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class NGRAMConfig(SpecConfig):
    spec_dec_name: str = "NGRAM"

    def __post_init__(self):
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)

    def update_from_model_config(self, model_config):
        pass


@dataclass
class NGRAMSpecMetadata(SpecMetadata):
    max_draft_tokens: int = 1024
    prompt_lookup_num_tokens: int = 10
    end_id: int = -1
    is_keep_all: bool = True
    is_use_oldest: bool = True

    def __post_init__(self):
        pass

    def prepare(self):
        pass

    def maybe_capture_hidden_states(self, layer_id: int,
                                    hidden_states: torch.Tensor,
                                    residual: torch.Tensor) -> None:
        pass


class NGRAMDecoder(TorchDecoder):

    def _batch_decode(self, scheduled_requests, model_outputs):
        # Is it needed?
        logits = model_outputs["logits"]
        new_tokens_device = torch.argmax(logits, dim=-1)
        if "d2t" in model_outputs:
            d2t = model_outputs["d2t"]
            new_tokens_device = d2t[new_tokens_device] + new_tokens_device
        new_tokens_host = new_tokens_device.to('cpu', non_blocking=True)
        new_tensors_device = {"new_tokens_device": new_tokens_device}
        new_tensors_host = {"new_tokens_host": new_tokens_host}
        decoder_event = torch.cuda.Event()
        decoder_event.record()
        return DecoderState(scheduled_requests=scheduled_requests,
                            logits=logits,
                            new_tensors_device=new_tensors_device,
                            new_tensors_host=new_tensors_host,
                            decoder_event=decoder_event)
