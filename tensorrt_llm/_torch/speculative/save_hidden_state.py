import os
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._utils import local_mpi_rank

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .drafter import Drafter
from .eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata


@dataclass
class SaveHiddenStatesSpecMetadata(Eagle3SpecMetadata):
    save_last_layer_post_norm: bool = False

    def is_final_output_capture(self):
        return self.save_last_layer_post_norm

    def maybe_capture_final_hidden_states(self,
                                          hidden_states: torch.Tensor) -> None:
        if self.save_last_layer_post_norm:
            # Assume no chunking, BS=1
            eagle3_hidden_states = self.eagle3_resource_manager.last_hidden_states
            eagle3_hidden_states[:hidden_states.shape[0], :].copy_(
                hidden_states)


class SaveHiddenStatesResourceManager(Eagle3ResourceManager):

    def __init__(self, config: "SaveHiddenStatesDecodingConfig",
                 dtype: torch.dtype, hidden_size: int, max_num_requests: int,
                 max_seq_len: int, max_num_tokens: int):
        super().__init__(config, dtype, hidden_size, max_num_requests,
                         max_seq_len, max_num_tokens)
        self.last_hidden_states = None
        if config.save_last_layer_post_norm:
            self.last_hidden_states = torch.empty(
                (max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device='cuda')


class SaveHiddenStatesDrafter(Drafter):

    def __init__(
        self,
        spec_config: "SaveHiddenStatesDecodingConfig",
        spec_resource_manager: SaveHiddenStatesResourceManager,
    ):
        super().__init__(spec_config.max_concurrency)
        self.spec_config = spec_config
        self.max_draft_len = spec_config.max_draft_len
        self._iter = 1
        self._output_directory = spec_config.output_directory
        self._file_prefix = spec_config.file_prefix
        self._write_interval = spec_config.write_interval
        self._saved_state = []
        self.spec_resource_manager = spec_resource_manager
        os.makedirs(self._output_directory, exist_ok=True)

    def _process_request(
            self, request: LlmRequest,
            resource_manager: SaveHiddenStatesResourceManager) -> None:
        out_dict = {}
        if local_mpi_rank() == 0:
            input_ids = torch.tensor(list(request.get_tokens(0)),
                                     dtype=torch.long,
                                     device='cpu')
            hidden_size = resource_manager.hidden_size
            num_tokens = input_ids.shape[0]
            if self.spec_config.save_last_layer_post_norm:
                hidden_states = resource_manager.last_hidden_states[:
                                                                    num_tokens, :].cpu(
                                                                    ).clone()
            else:
                hidden_states = resource_manager.hidden_states[:num_tokens,
                                                               -hidden_size:].cpu(
                                                               ).clone()

            out_dict = {
                "id":
                self._iter,
                "input_ids":
                input_ids,
                "hidden_state_features":
                resource_manager.hidden_states[:num_tokens, :].cpu().clone(),
                "hidden_state":
                hidden_states,
            }

            self._saved_state.append(out_dict)

    def _write_to_file(self) -> None:
        if local_mpi_rank() == 0:
            output_path = os.path.join(self._output_directory,
                                       f"{self._file_prefix}_{self._iter}.pt")
            torch.save(self._saved_state, output_path)
        self._saved_state = []

    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        for request in sorted(
                scheduled_requests.context_requests,
                key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            request.py_max_new_tokens = 1
            # Pad length to `self.max_draft_len`
            draft_tokens = [0] * self.max_draft_len
            request.py_draft_tokens = draft_tokens

    def prepare_draft_tokens_post(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
        is_warmup: bool = False,
    ) -> None:
        for request in sorted(
                scheduled_requests.context_requests,
                key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            if is_warmup:
                continue
            self._process_request(request, self.spec_resource_manager)
            if self._iter % self._write_interval == 0:
                self._write_to_file()
            self._iter += 1

    def needs_draft_forward_post(self) -> bool:
        """
        If draft forward needs to be run directly after the target model forward,
        this method can be overridden to do that.
        Used in SaveHiddenStatesDrafter (to ensure correct input_ids)
        """
        return False
