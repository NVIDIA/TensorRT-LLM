# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm.logger import logger


class AsyncTransferManager:
    """
    Handle asynchronous transfer of KV cache after a request has completed.
    When running with both the KV cache transceiver and the KV cache connector, we must ensure
    that BOTH transfers (if any) are completed before we can release the KV cache blocks.
    The AsyncTransferManager has a few key responsibilities:
    1. Track requests in transfer.
    2. Pin blocks for reuse while blocks are in transfer.
    3. Unpin blocks after all transfers are complete.

    TODO(jthomson04): This only handles async send/saving, not loading. Loading kv cache is
    handled through a separate codepath. Eventually, we'll want to merge these two paths.
    """

    class RequestTransferMetadata:
        def __init__(self, block_id: Optional[int]):
            self.block_id = block_id
            self.counter = 0

        def start_transfer(self):
            self.counter += 1

        def end_transfer(self) -> bool:
            """
            Returns:
                bool: True if there are no more transfers for this request
            """
            self.counter -= 1
            return self.counter == 0

    def __init__(self, resource_manager: "ResourceManager", should_store_blocks: bool = True):
        self.resource_manager = resource_manager
        self.kv_cache_manager = resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER
        )

        self.should_store_blocks = should_store_blocks

        # Mapping of request id to the LlmRequest
        self._requests_in_transfer: Dict[int, LlmRequest] = dict()

        # Mapping of request id to the request metadata
        self._request_transfer_metadata: Dict[int, self.RequestTransferMetadata] = dict()

    def requests_in_transfer(self) -> Dict[int, LlmRequest]:
        return self._requests_in_transfer

    def start_transfer(self, request: LlmRequest):
        """
        Called when a Cache transceiver or connector transfer is started.
        1. Increment the counter for the request.
        2. Releases all resources except for the KV cache, if not already released.
        3. Store KV cache blocks for reuse.
        """

        req_id = request.py_request_id

        if req_id not in self._requests_in_transfer:
            for resource_mgr_type in (
                ResourceManagerType.SEQ_SLOT_MANAGER,
                ResourceManagerType.SPEC_RESOURCE_MANAGER,
            ):
                if (
                    resource_mgr_type in self.resource_manager.resource_managers
                    and self.resource_manager.resource_managers[resource_mgr_type] is not None
                ):
                    self.resource_manager.resource_managers[resource_mgr_type].free_resources(
                        request
                    )

            request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

            if self.should_store_blocks:
                block_id = self.kv_cache_manager.store_blocks_for_reuse(request, True)
            else:
                block_id = None

            self._requests_in_transfer[req_id] = request
            self._request_transfer_metadata[req_id] = self.RequestTransferMetadata(block_id)

        self._request_transfer_metadata[req_id].start_transfer()

    def end_transfer(self, request: LlmRequest) -> bool:
        """
        Called after a send of KV cache is complete.
        1. Decrements counter for request.
        2. If there are no more inflight transfers for this request, unpin the blocks and mark the request as complete.

        Returns:
            bool: True if the request should be terminated after call to end_transfer
        """
        try:
            transfer_metadata = self._request_transfer_metadata[request.py_request_id]
        except KeyError:
            logger.warning(f"Request {request.py_request_id} not found in transfer manager")
            return False

        if transfer_metadata.end_transfer():
            self._requests_in_transfer.pop(request.py_request_id)
            self._request_transfer_metadata.pop(request.py_request_id)

            if self.should_store_blocks:
                self.kv_cache_manager.unpin_blocks_by_id(transfer_metadata.block_id)

            # We don't want to overwrite any error state.
            if request.state != LlmRequestState.DISAGG_TRANS_ERROR:
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

            return True

        return False

    def has_any_inflight_requests(self) -> bool:
        return len(self._requests_in_transfer) > 0
