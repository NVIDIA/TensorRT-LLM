import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set

import torch

from tensorrt_llm._utils import local_mpi_rank

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecMetadata

if TYPE_CHECKING:
    from ...llmapi.llm_args import SaveHiddenStatesDecodingConfig


class SaveHiddenStatesResourceManager(BaseResourceManager):
    """
    Resource manager for SaveHiddenStates mode.
    Manages the hidden states buffer and handles saving to disk.
    """

    def __init__(self, config: "SaveHiddenStatesDecodingConfig",
                 dtype: torch.dtype, hidden_size: int, max_num_requests: int,
                 max_num_tokens: int):
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.max_num_requests = max_num_requests

        # Config for saving
        self._output_directory = config.output_directory
        self._file_prefix = config.file_prefix
        self._write_interval = config.write_interval
        self._last_hidden_in_save = config._last_hidden_in_save

        # State tracking
        self._iter = 1
        self._saved_state: List[dict] = []

        # Allocate hidden states buffer
        # Shape: [max_num_tokens, hidden_size * num_capture_layers]
        self.hidden_states = torch.empty(
            (max_num_tokens, hidden_size * config.num_capture_layers),
            dtype=dtype,
            device='cuda')

        os.makedirs(self._output_directory, exist_ok=True)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for req in scheduled_batch.all_requests():
            req.py_max_new_tokens = 1

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0

    def process_and_save(self, scheduled_requests: ScheduledRequests,
                         spec_metadata: "SaveHiddenStatesSpecMetadata") -> None:
        """
        Process context requests and save hidden states to disk.
        Called after the target model forward.

        Args:
            scheduled_requests: The scheduled requests for this iteration
            spec_metadata: The spec metadata containing layers_to_capture info
        """
        for request in sorted(
                scheduled_requests.context_requests,
                key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            self._process_request(request, spec_metadata)
            if self._iter % self._write_interval == 0:
                self._write_to_file()
            self._iter += 1

    def _process_request(self, request: LlmRequest,
                         spec_metadata: "SaveHiddenStatesSpecMetadata") -> None:
        if local_mpi_rank() != 0:
            return

        input_ids = torch.tensor(list(request.get_tokens(0)),
                                 dtype=torch.long,
                                 device='cpu')
        num_tokens = input_ids.shape[0]
        # This is always the last post norm state.
        hidden_states = self.hidden_states[:num_tokens,
                                           -self.hidden_size:].cpu().clone()

        out_dict = {
            "id": self._iter,
            "input_ids": input_ids,
            "hidden_state": hidden_states,
        }

        layers_to_capture = spec_metadata.layers_to_capture
        if len(layers_to_capture) > 1:
            if self._last_hidden_in_save:
                # Duplicate the final post norm state in aux_hidden_states only
                # if the user explicitly asked to capture layer -1
                out_dict[
                    "aux_hidden_states"] = self.hidden_states[:
                                                              num_tokens, :].cpu(
                                                              ).clone()
            else:
                out_dict[
                    "aux_hidden_states"] = self.hidden_states[:num_tokens, :
                                                              -self.
                                                              hidden_size].cpu(
                                                              ).clone()

        self._saved_state.append(out_dict)

    def _write_to_file(self) -> None:
        if local_mpi_rank() == 0:
            output_path = os.path.join(self._output_directory,
                                       f"{self._file_prefix}_{self._iter}.pt")
            torch.save(self._saved_state, output_path)
        self._saved_state = []


@dataclass
class SaveHiddenStatesSpecMetadata(SpecMetadata):
    """
    Metadata for SaveHiddenStates mode.
    Captures hidden states from specified layers during forward pass.
    """
    # The layers to capture hidden states from
    layers_to_capture: Optional[Set[int]] = None
    # The hidden size
    hidden_size: int = 0
    # Max number of tokens
    max_num_tokens: int = 0
    # Data type for hidden states
    dtype: torch.dtype = torch.bfloat16
    # Reference to the resource manager
    resource_manager: Optional[SaveHiddenStatesResourceManager] = None
    # Number of layers in the model (used for layer indexing)
    num_model_layers: int = 0

    def __post_init__(self):
        if self.layers_to_capture is None:
            if self.num_model_layers <= 5:
                raise ValueError(
                    "Not enough hidden layers for default SaveHiddenStates capture"
                )
            from .eagle3 import _get_eagle3_default_capture_layers

            self.layers_to_capture = _get_eagle3_default_capture_layers(
                self.num_model_layers) + (-1, )
        else:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            # Handle -1 (last layer) - move to end of list
            if self.layers_to_capture and self.layers_to_capture[0] == -1:
                self.layers_to_capture = self.layers_to_capture[1:] + [
                    self.layers_to_capture.pop(0)
                ]
        self.num_capture_layers = len(self.layers_to_capture)

    def is_layer_capture(self, layer_id: int):
        return layer_id in self.layers_to_capture

    def maybe_capture_hidden_states(
            self,
            layer_id: int,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor] = None) -> None:
        """Capture hidden states from the target model during forward pass."""
        if self.resource_manager is None:
            return

        resource_hidden_states = self.resource_manager.hidden_states
        for i, captured_layer_id in enumerate(self.layers_to_capture):
            if captured_layer_id == layer_id:
                to_save = hidden_states + residual if residual is not None else hidden_states
                to_save = to_save.to(dtype=resource_hidden_states.dtype)
                num_tokens = to_save.shape[0]
                resource_hidden_states[:num_tokens,
                                       i * self.hidden_size:(i + 1) *
                                       self.hidden_size].copy_(
                                           to_save, non_blocking=True)
                break
