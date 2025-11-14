import os
from typing import Optional

import torch

from tensorrt_llm._utils import local_mpi_rank

from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .drafter import Drafter


class SaveHiddenStatesDrafter(Drafter):

    def __init__(
        self,
        spec_config: "SaveHiddenStatesDecodingConfig",
        spec_resource_manager,
    ):
        super().__init__(spec_config.max_concurrency)
        self.spec_config = spec_config
        self.max_draft_len = spec_config.max_draft_len
        self.max_total_draft_tokens = spec_config.max_total_draft_tokens
        self._iter = 1
        self._output_directory = spec_config.output_directory
        self._file_prefix = spec_config.file_prefix
        self._write_interval = spec_config.write_interval
        self._saved_state = []
        self.spec_resource_manager = spec_resource_manager
        os.makedirs(self._output_directory, exist_ok=True)

    def _process_request(self, request: LlmRequest, resource_manager) -> None:
        out_dict = {}
        if local_mpi_rank() == 0:
            input_ids = torch.tensor(list(request.get_tokens(0)),
                                     dtype=torch.long,
                                     device='cpu')
            hidden_size = resource_manager.hidden_size
            num_tokens = input_ids.shape[0]
            hidden_states = resource_manager.hidden_states[:num_tokens,
                                                           -hidden_size:].cpu(
                                                           ).clone()

            out_dict = {
                "id": self._iter,
                "input_ids": input_ids,
                "hidden_state": hidden_states,
            }
            if len(self.spec_config.eagle3_layers_to_capture) > 1:
                if self.spec_config._last_hidden_in_save:
                    out_dict[
                        "aux_hidden_states"] = resource_manager.hidden_states[:num_tokens, :].cpu(
                        ).clone()
                else:
                    out_dict[
                        "aux_hidden_states"] = resource_manager.hidden_states[:
                                                                              num_tokens, :
                                                                              -hidden_size].cpu(
                                                                              ).clone(
                                                                              )

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

    def run_drafter_post(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
        is_warmup: bool = False,
    ) -> None:
        if is_warmup:
            return
        for request in sorted(
                scheduled_requests.context_requests,
                key=lambda r:
            (r.py_batch_idx is None, r.py_batch_idx or r.request_id),
        ):
            self._process_request(request, self.spec_resource_manager)
            if self._iter % self._write_interval == 0:
                self._write_to_file()
            self._iter += 1
