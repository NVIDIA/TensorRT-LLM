from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch

from ..pyexecutor.decoder import TorchDecoder
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class Eagle3Config(SpecConfig):
    spec_dec_name: str = "EAGLE3"
    eagle_weights_path: Optional[str] = None
    num_layers: int = 0

    def __post_init__(self):
        if self.eagle_weights_path is None:
            raise ValueError("Path to EAGLE3 weights must be specified.")

        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)
        self.num_extra_kv_tokens = 0

    def update_from_model_config(self, model_config):
        self.num_layers = model_config.num_hidden_layers

    def get_draft_model_prompt(self,
                               input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Eagle3 always throws away the first token when processing draft inputs
        """
        return input_tokens[1:]


@dataclass
class Eagle3SpecMetadata(SpecMetadata):
    hidden_states: List[torch.Tensor] = field(default_factory=list)
    num_layers: int = 0
    layers_to_capture: Tuple[int, ...] = field(init=False)
    target_model_embed_tokens: Optional[torch.nn.Module] = None

    def __post_init__(self):
        if self.num_layers == 1:
            # For the draft model, we have to capture hiddens states
            # manually outside of the decoder layer.
            self.layers_to_capture = ()
        else:
            if self.num_layers <= 5:
                raise ValueError("Not enough hidden layers for EAGLE")

            self.layers_to_capture = (1, self.num_layers // 2 - 1,
                                      self.num_layers - 3)

    def prepare(self):
        self.hidden_states = []

    def maybe_capture_hidden_states(self, layer_id: int,
                                    hidden_states: torch.Tensor,
                                    residual: torch.Tensor) -> None:
        if layer_id in self.layers_to_capture:
            # TODO: write directly into a pre-allocated buffer for
            # CUDA graph support.
            self.hidden_states.append(hidden_states + residual)

    def get_hidden_states(
            self,
            scheduled_requests,
            num_rejected_tokens: Optional[Dict] = None) -> List[torch.Tensor]:
        req_id_to_gather_ids = {}
        seq_start = 0
        for req_id, seqlen in zip(self.request_ids, self.seq_lens):
            if num_rejected_tokens is not None:
                if req_id in num_rejected_tokens:
                    req_id_to_gather_ids[req_id] = list(
                        range(seq_start,
                              seq_start + seqlen - num_rejected_tokens[req_id]))
            else:
                req_id_to_gather_ids[req_id] = [seq_start + seqlen - 1]

            seq_start += seqlen

        hidden_states_gather_ids = []
        for req in chain(scheduled_requests.context_requests,
                         scheduled_requests.generation_requests):
            hidden_states_gather_ids.extend(
                req_id_to_gather_ids[req.py_request_id])

        return [h[hidden_states_gather_ids] for h in self.hidden_states]


class Eagle3Decoder(TorchDecoder):

    def _batch_decode(self, scheduled_requests, model_outputs):
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
        return new_tensors_device, new_tensors_host, decoder_event
