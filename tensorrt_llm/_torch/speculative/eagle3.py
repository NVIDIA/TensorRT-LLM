from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch

from ..pyexecutor.decoder import DecoderState, TorchDecoder
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class Eagle3Config(SpecConfig):
    spec_dec_name: str = "EAGLE3"
    eagle_weights_path: Optional[str] = None
    num_layers: int = 0
    hidden_size: int = 0

    def __post_init__(self):
        if self.eagle_weights_path is None:
            raise ValueError("Path to EAGLE3 weights must be specified.")

        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)
        self.num_extra_kv_tokens = 0

    def update_from_model_config(self, model_config):
        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size

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
    hidden_size: int = 0

    def __post_init__(self):
        if self.num_layers == 1:
            self.layers_to_capture = (1, )
        else:
            if self.num_layers <= 5:
                raise ValueError("Not enough hidden layers for EAGLE")

            self.layers_to_capture = (1, self.num_layers // 2 - 1,
                                      self.num_layers - 4)

        self.hidden_states = []
        if self.is_cuda_graph:
            # CUDA graphs need to use the same buffers between runs.
            max_seqlen = self.max_num_requests * (self.max_draft_tokens + 1)
            hidden_state_shape = (max_seqlen, self.hidden_size)
            for layer in self.layers_to_capture:
                self.hidden_states.append(
                    torch.empty(hidden_state_shape, device='cuda'))

    def prepare(self):
        if not self.is_cuda_graph:
            self.hidden_states = []

    def maybe_capture_hidden_states(
            self,
            layer_id: int,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor] = None) -> None:
        if not self.is_cuda_graph:
            if layer_id in self.layers_to_capture:
                to_save = hidden_states + residual if residual is not None else hidden_states
                self.hidden_states.append(to_save)
        else:
            assert len(self.hidden_states) == len(self.layers_to_capture)
            for i, captured_layer_id in enumerate(self.layers_to_capture):
                if captured_layer_id == layer_id:
                    to_save = hidden_states + residual if residual is not None else hidden_states
                    self.hidden_states[i].copy_(to_save)
                    break

    def get_hidden_states(
            self,
            scheduled_requests,
            num_rejected_tokens: Optional[Dict] = None) -> torch.Tensor:
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

        if len(self.hidden_states) == 1:
            return self.hidden_states[0][hidden_states_gather_ids]
        else:
            return torch.cat(
                [h[hidden_states_gather_ids] for h in self.hidden_states],
                dim=-1)


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
        return DecoderState(scheduled_requests=scheduled_requests,
                            logits=logits,
                            new_tensors_device=new_tensors_device,
                            new_tensors_host=new_tensors_host,
                            decoder_event=decoder_event)
