# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional

from ...._utils import nvtx_range
from ..llm_request import LlmRequest
from ..scheduler import ScheduledRequests

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata

from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole


class ResourceManagerType(enum.Enum):
    KV_CACHE_MANAGER = "KV_CACHE_MANAGER"
    DRAFT_KV_CACHE_MANAGER = "DRAFT_KV_CACHE_MANAGER"
    PEFT_CACHE_MANAGER = "PEFT_CACHE_MANAGER"
    SEQ_SLOT_MANAGER = "SEQ_SLOT_MANAGER"
    SPEC_RESOURCE_MANAGER = "SPEC_RESOURCE_MANAGER"


class Role:
    KEY = DataRole("key")
    VALUE = DataRole("value")
    KEY_BLOCK_SCALE = DataRole("key_block_scale")
    VALUE_BLOCK_SCALE = DataRole("value_block_scale")
    ALL = DataRole("all")


def compute_page_count(token_count: int, tokens_per_page: int) -> int:
    return (token_count + tokens_per_page) // tokens_per_page


class BaseResourceManager(ABC):
    @abstractmethod
    def get_max_resource_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        raise NotImplementedError

    def add_dummy_requests(self, request_ids: List[int]):
        pass

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def shutdown(self):
        pass


def request_context(is_draft: bool, scheduled_requests: ScheduledRequests):
    class RequestContext:
        def __init__(self, is_draft: bool, scheduled_requests: ScheduledRequests):
            self.is_draft = is_draft
            self.scheduled_requests = scheduled_requests

        def __enter__(self):
            if not self.is_draft:
                return

            for req in self.scheduled_requests.all_requests():
                req.use_draft_model = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.is_draft:
                return

            # Clean up the state
            for req in self.scheduled_requests.all_requests():
                req.use_draft_model = False

    return RequestContext(is_draft, scheduled_requests)


class ResourceManager:
    def __init__(self, resource_managers: dict[ResourceManagerType, BaseResourceManager]):
        self.resource_managers = OrderedDict(resource_managers)

    def __call__(self, type: ResourceManagerType):
        return self.resource_managers[type]

    def register_resource_manager(
        self, type: ResourceManagerType, resource_manager: BaseResourceManager
    ):
        self.resource_managers[type] = resource_manager

    def get_resource_manager(self, type: ResourceManagerType) -> Optional[BaseResourceManager]:
        return self.resource_managers.get(type)

    @nvtx_range("prepare_resources")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "prepare_resources"):
                resource_manager.prepare_resources(scheduled_batch)

    @nvtx_range("update_resources")
    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ):
        # Avoid circular import at module level
        from .kv_cache_manager import KVCacheManager

        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "update_resources"):
                if isinstance(resource_manager, KVCacheManager):
                    resource_manager.update_resources(
                        scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
                    )
                else:
                    resource_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        for resource_type, resource_manager in reversed(self.resource_managers.items()):
            if hasattr(resource_manager, "free_resources"):
                resource_manager.free_resources(request)

    def reorder_pipeline(self, resource_manager_list: list[ResourceManagerType]):
        assert set(resource_manager_list) == set(self.resource_managers.keys())
        for resource_manager in resource_manager_list:
            self.resource_managers.move_to_end(resource_manager)
