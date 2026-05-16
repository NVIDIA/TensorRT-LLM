# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from abc import ABC, abstractmethod
from os import getenv
from typing import Any, Dict, List, Optional

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .mamba_cache_manager import MambaCacheManager
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType


def _request_summary(req: LlmRequest) -> str:
    disagg_params = getattr(req, "py_disaggregated_params", None)
    if disagg_params is None:
        disagg_summary = "none"
    else:
        encoded_opaque_state = getattr(disagg_params, "encoded_opaque_state",
                                       None)
        first_gen_tokens = getattr(disagg_params, "first_gen_tokens", None)
        draft_tokens = getattr(disagg_params, "draft_tokens", None)
        disagg_summary = (
            f"request_type={getattr(disagg_params, 'request_type', None)!r} "
            f"ctx_request_id={getattr(disagg_params, 'ctx_request_id', None)!r} "
            f"disagg_request_id={getattr(disagg_params, 'disagg_request_id', None)!r} "
            f"schedule_style={getattr(disagg_params, 'schedule_style', None)!r} "
            f"ctx_dp_rank={getattr(disagg_params, 'ctx_dp_rank', None)!r} "
            f"ctx_info_endpoint={getattr(disagg_params, 'ctx_info_endpoint', None)!r} "
            f"opaque_state_bytes={len(encoded_opaque_state) if encoded_opaque_state else 0} "
            f"first_gen_tokens={len(first_gen_tokens) if first_gen_tokens else 0} "
            f"draft_tokens={len(draft_tokens) if draft_tokens else 0}")

    return (
        f"request_id={getattr(req, 'py_request_id', getattr(req, 'request_id', None))!r} "
        f"state={getattr(req, 'state', None)!r} "
        f"prompt_len={getattr(req, 'py_prompt_len', getattr(req, 'prompt_len', None))!r} "
        f"max_new_tokens={getattr(req, 'py_max_new_tokens', getattr(req, 'max_new_tokens', None))!r} "
        f"seq_slot={getattr(req, 'py_seq_slot', None)!r} "
        f"client_id={getattr(req, 'py_client_id', None)!r} "
        f"is_child={getattr(req, 'is_child', None)!r} "
        f"parent_request_id={getattr(req, 'parent_request_id', None)!r} "
        f"kv_transfer_start={getattr(req, 'py_kv_transfer_start_time', None)!r} "
        f"kv_transfer_timed_out={getattr(req, 'py_kv_transfer_timed_out', None)!r} "
        f"disagg=({disagg_summary})")


def _transfer_result_summary(result: Any) -> str:
    if isinstance(result, tuple):
        parts = []
        for item in result:
            try:
                parts.append(str(len(item)))
            except TypeError:
                parts.append(type(item).__name__)
        return f"tuple_lengths={parts}"
    return repr(result)


def mapping_to_world_config(mapping: Mapping) -> WorldConfig:

    return WorldConfig(tensor_parallelism=mapping.tp_size,
                       pipeline_parallelism=mapping.pp_size,
                       context_parallelism=mapping.cp_size,
                       rank=mapping.rank,
                       gpus_per_node=mapping.gpus_per_node,
                       device_ids=None,
                       enable_attention_dp=mapping.enable_attention_dp)


def create_kv_cache_transceiver(
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig,
        mamba_cache_manager: Optional[MambaCacheManager] = None):
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        logger.info("cache_transceiver is disabled")
        return None

    if cache_transceiver_config.backend == "DEFAULT":
        # When cache_transceiver_config.backend is not set, fallback to env_vars settings
        # NIXL is the default backend for non hybrid models
        cache_transceiver_config.backend = "NIXL"
        # Ordered by priority
        env_vars = [
            ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
            ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
            ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
            ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
        ]
        for env_var, be_type in env_vars:
            if getenv(env_var) == "1":
                logger.warning(
                    f"{env_var}=1 is set, but it's recommended to set cache_transceiver_config.backend in yaml config"
                )
                cache_transceiver_config.backend = be_type
                break

    if cache_transceiver_config.backend == "MPI":
        logger.warning(
            "MPI CacheTransceiver is deprecated, UCX or NIXL is recommended")
    elif cache_transceiver_config.backend == "UCX":
        logger.info(
            "Using UCX kv-cache transceiver. If your devices are not in the same domain, please consider setting "
            "UCX_CUDA_IPC_ENABLE_MNNVL=n, UCX_RNDV_SCHEME=put_zcopy and/or unset UCX_NET_DEVICES upon server "
            "hangs or lower-than-expected performance.")

    # Select transceiver implementation based on transceiver_runtime
    # transceiver_runtime == None or "CPP" -> use C++ transceiver (default)
    # transceiver_runtime == "PYTHON" -> use Python transceiver
    if cache_transceiver_config.transceiver_runtime == "PYTHON":
        # Python transceiver currently only supports NIXL and DEFAULT backend
        if cache_transceiver_config.backend not in ("DEFAULT", "NIXL"):
            raise ValueError(
                f"Python transceiver currently only supports NIXL or DEFAULT backend, "
                f"got {cache_transceiver_config.backend}. "
                f"Please use transceiver_runtime='CPP' for MPI, UCX, or MOONCAKE backends."
            )
        from tensorrt_llm._torch.disaggregation.transceiver import \
            KvCacheTransceiverV2
        logger.info("Using KvCacheTransceiverV2")
        return KvCacheTransceiverV2(mapping, dist, kv_cache_manager,
                                    cache_transceiver_config)

    # Default: use C++ transceiver (transceiver_runtime is None or "CPP")
    return BindKvCacheTransceiver(mapping, dist, kv_cache_manager,
                                  attention_type, cache_transceiver_config,
                                  mamba_cache_manager)


class KvCacheTransceiver(ABC):

    @abstractmethod
    def respond_and_send_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def check_context_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_complete(self):
        raise NotImplementedError

    @abstractmethod
    def cancel_request(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def prepare_context_requests(self, requests: List[LlmRequest]):
        """
        Prepare the context request for the cache transceiver in generation-first mode.
        This method should set the context request state to DISAGG_CONTEXT_WAIT_SCHEDULER
        so that it won't be scheduled if the responding generation kvcache request is not
        yet received otherwise set it to CONTEXT_INIT.
        """
        ...

    @abstractmethod
    def get_disaggregated_params(self) -> Dict[str, Any]:
        """
        Return a dictionary form of DisaggregatedParams to be set in the generation request.
        The generation server will use it to get kvcache in generation-first mode.
        """
        ...

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit received KV blocks to the radix tree for prefix reuse. No-op by default."""

    def shutdown(self):
        """Shut down the transceiver and release registered resources."""


class BindKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self,
                 mapping: Mapping,
                 dist: Distributed,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig,
                 mamba_cache_manager: Optional[MambaCacheManager] = None):
        world_config = mapping_to_world_config(mapping)
        total_num_kv_heads_per_layer = kv_cache_manager.total_num_kv_heads_per_layer
        head_dim = kv_cache_manager.head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        dtype = kv_cache_manager.dtype
        # get the layer num per pp rank, which is required by cache transceiver.
        pp_layer_num = len(kv_cache_manager.pp_layers)
        pp_layer_num_per_pp_rank = dist.pp_allgather(pp_layer_num)

        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_sender_future_timeout_ms = cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        logger.info(
            f"[disagg-debug] creating C++ KV cache transceiver: "
            f"rank={mapping.rank} tp={mapping.tp_size} pp={mapping.pp_size} "
            f"cp={mapping.cp_size} gpus_per_node={mapping.gpus_per_node} "
            f"attention_dp={mapping.enable_attention_dp} "
            f"backend={cache_transceiver_config.backend} "
            f"runtime={cache_transceiver_config.transceiver_runtime} "
            f"max_tokens_in_buffer={cache_transceiver_config.max_tokens_in_buffer} "
            f"kv_transfer_timeout_ms={self.kv_transfer_timeout_ms} "
            f"kv_transfer_sender_future_timeout_ms={self.kv_transfer_sender_future_timeout_ms} "
            f"tokens_per_block={tokens_per_block} dtype={dtype} "
            f"pp_layer_num_per_pp_rank={pp_layer_num_per_pp_rank}")

        # Get RNN state manager and layer distribution if mamba_cache_manager is provided
        rnn_state_manager = None
        rnn_layer_num_per_pp_rank = []
        if mamba_cache_manager is not None:
            rnn_state_manager = mamba_cache_manager._impl.mamba_impl
            # Get the number of local RNN layers and allgather across PP ranks
            rnn_local_layer_num = rnn_state_manager.get_num_local_layers()
            rnn_layer_num_per_pp_rank = dist.pp_allgather(rnn_local_layer_num)
            logger.info(
                f"RNN state transfer enabled: rnn_layer_num_per_pp={rnn_layer_num_per_pp_rank}"
            )

        self.impl = CacheTransceiverCpp(
            kv_cache_manager.impl, total_num_kv_heads_per_layer, head_dim,
            tokens_per_block, world_config,
            pp_layer_num_per_pp_rank, dtype, attention_type,
            cache_transceiver_config._to_pybind(), rnn_state_manager,
            rnn_layer_num_per_pp_rank)
        logger.info("[disagg-debug] C++ KV cache transceiver created")

    def respond_and_send_async(self, req: LlmRequest):
        start_time = time.monotonic()
        logger.info(
            f"[disagg-debug] respond_and_send_async begin: req=({_request_summary(req)})"
        )
        result = self.impl.respond_and_send_async(req)
        logger.info(
            f"[disagg-debug] respond_and_send_async end: elapsed_s={time.monotonic() - start_time:.3f} "
            f"req=({_request_summary(req)}) result={result!r}")
        return result

    def request_and_receive_sync(self, req: LlmRequest):
        start_time = time.monotonic()
        logger.info(
            f"[disagg-debug] request_and_receive_sync begin: req=({_request_summary(req)})"
        )
        result = self.impl.request_and_receive_sync(req)
        logger.info(
            f"[disagg-debug] request_and_receive_sync end: elapsed_s={time.monotonic() - start_time:.3f} "
            f"req=({_request_summary(req)}) result={result!r}")
        return result

    def request_and_receive_async(self, req: LlmRequest):
        start_time = time.monotonic()
        logger.info(
            f"[disagg-debug] request_and_receive_async begin: req=({_request_summary(req)})"
        )
        result = self.impl.request_and_receive_async(req)
        logger.info(
            f"[disagg-debug] request_and_receive_async end: elapsed_s={time.monotonic() - start_time:.3f} "
            f"req=({_request_summary(req)}) result={result!r}")
        return result

    def check_context_transfer_status(self, at_least_request_num: int):
        start_time = time.monotonic()
        log_fn = logger.info if at_least_request_num > 0 else logger.debug
        log_fn(
            f"[disagg-debug] check_context_transfer_status begin: at_least_request_num={at_least_request_num}"
        )
        result = self.impl.check_context_transfer_status(at_least_request_num)
        log_fn(
            f"[disagg-debug] check_context_transfer_status end: at_least_request_num={at_least_request_num} "
            f"elapsed_s={time.monotonic() - start_time:.3f} result={_transfer_result_summary(result)}"
        )
        return result

    def check_gen_transfer_status(self, at_least_request_num: int):
        start_time = time.monotonic()
        log_fn = logger.info if at_least_request_num > 0 else logger.debug
        log_fn(
            f"[disagg-debug] check_gen_transfer_status begin: at_least_request_num={at_least_request_num}"
        )
        result = self.impl.check_gen_transfer_status(at_least_request_num)
        log_fn(
            f"[disagg-debug] check_gen_transfer_status end: at_least_request_num={at_least_request_num} "
            f"elapsed_s={time.monotonic() - start_time:.3f} result={_transfer_result_summary(result)}"
        )
        return result

    def check_gen_transfer_complete(self):
        result = self.impl.check_gen_transfer_complete()
        logger.debug(
            f"[disagg-debug] check_gen_transfer_complete result={result!r}")
        return result

    def cancel_request(self, req: LlmRequest):
        start_time = time.monotonic()
        logger.info(
            f"[disagg-debug] cancel_request begin: req=({_request_summary(req)})"
        )
        result = self.impl.cancel_request(req)
        logger.info(
            f"[disagg-debug] cancel_request end: elapsed_s={time.monotonic() - start_time:.3f} "
            f"req=({_request_summary(req)}) result={result!r}")
        return result

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # not implemented, an empty placeholder to allow being invoked unconditionally
        ...

    def get_disaggregated_params(self):
        # Cpp kv cache transceiver will set the disaggregated params to context response
        # Only new py cache transceiver will support gen-first disagg
        return {}


class CacheTransBufferManager:

    def __init__(self, kv_cache_manager: KVCacheManager, max_num_tokens: int):
        self.impl = CacheTransBufferManagerCpp(kv_cache_manager.impl,
                                               max_num_tokens)

    @staticmethod
    def pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window: dict[int, int],
            tokens_per_block: int,
            cache_transceiver_config: CacheTransceiverConfig):
        return CacheTransBufferManagerCpp.pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window, tokens_per_block,
            cache_transceiver_config)
