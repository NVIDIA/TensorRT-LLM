# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import multiprocessing
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import psutil
import safetensors
import torch
import tqdm
from mpi4py import MPI as _MPI

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict)
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm._utils import (ENABLE_MULTI_DEVICE, local_mpi_barrier,
                                 local_mpi_comm, local_mpi_rank, local_mpi_size,
                                 mpi_comm, mpi_disabled)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_WEIGHT_CACHE_ENV = "TRTLLM_HF_WEIGHT_CACHE"
_WEIGHT_CACHE_MAX_ENTRIES_ENV = "TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES"
_SAFETENSORS_LOAD_MODE_ENV = "TRTLLM_HF_SAFETENSORS_LOAD_MODE"
_EAGER_SAFETENSORS_LOAD_MODE = "eager"
_HYBRID_SAFETENSORS_LOAD_MODE = "hybrid"
_SUPPORTED_SAFETENSORS_LOAD_MODES = frozenset(
    (_EAGER_SAFETENSORS_LOAD_MODE, _HYBRID_SAFETENSORS_LOAD_MODE))
_HYBRID_MODEL_CLASSES = frozenset((
    ("tensorrt_llm._torch.models.modeling_llama", "LlamaForCausalLM"),
    ("tensorrt_llm._torch.models.modeling_qwen", "Qwen2ForCausalLM"),
    ("tensorrt_llm._torch.models.modeling_qwen3", "Qwen3ForCausalLM"),
))
_DEFAULT_HYBRID_PREFETCH_CHUNK_SIZE = 256 * 1024 * 1024
_HYBRID_PREFETCH_READ_SIZE = 8 * 1024 * 1024
_DEFAULT_HYBRID_PREFETCH_WORKERS_PER_RANK = 16
# Default to a single cached checkpoint: each entry pins a full copy of the
# raw weights in CPU RAM, so callers wanting cross-model caching must opt in
# via TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES.
_DEFAULT_WEIGHT_CACHE_MAX_ENTRIES = 1
_WEIGHT_CACHE_LOCK = threading.Lock()
_WEIGHT_CACHE: OrderedDict[tuple, dict[str, Any]] = OrderedDict()


@register_checkpoint_weight_loader("MX")
@register_checkpoint_weight_loader("mistral")
@register_checkpoint_weight_loader("mistral_large_3")
@register_checkpoint_weight_loader("HF")
class HfWeightLoader(BaseWeightLoader):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

    def __init__(
        self,
        *,
        safetensors_load_mode: str | None = None,
        hybrid_prefetch_chunk_size: int = _DEFAULT_HYBRID_PREFETCH_CHUNK_SIZE,
        hybrid_prefetch_workers_per_rank: int | None = None,
    ) -> None:
        self._safetensors_load_mode = safetensors_load_mode
        self._hybrid_prefetch_chunk_size = hybrid_prefetch_chunk_size
        self._hybrid_prefetch_workers_per_rank = hybrid_prefetch_workers_per_rank
        self._validate_hybrid_options()

    def _validate_hybrid_options(self) -> None:
        self._get_safetensors_load_mode()
        if self._hybrid_prefetch_chunk_size <= 0:
            raise ValueError("hybrid_prefetch_chunk_size must be positive")
        if (self._hybrid_prefetch_workers_per_rank is not None
                and self._hybrid_prefetch_workers_per_rank <= 0):
            raise ValueError(
                "hybrid_prefetch_workers_per_rank must be positive")

    def _get_safetensors_load_mode(self) -> str:
        value = self._safetensors_load_mode
        if value is None:
            value = os.environ.get(_SAFETENSORS_LOAD_MODE_ENV,
                                   _EAGER_SAFETENSORS_LOAD_MODE)
        value = value.strip().lower()
        if value not in _SUPPORTED_SAFETENSORS_LOAD_MODES:
            supported_modes = ", ".join(
                sorted(_SUPPORTED_SAFETENSORS_LOAD_MODES))
            raise ValueError(
                f"Unsupported SafeTensors load mode {value!r}; expected one of: "
                f"{supported_modes}")
        return value

    def _get_coordinated_safetensors_load_mode(
            self,
            mapping: Mapping,
            checkpoint_format: str,
            load_format: Any = None) -> tuple[str, str | None]:
        """Select one path after validating the active MPI communicator."""
        load_mode = self._get_safetensors_load_mode()
        load_format_name = getattr(load_format, "name", load_format)
        # ModelLoader selects one load format for the whole model-loading
        # group. Coordinate only the HF disk modes: GMS and format-specific
        # loaders are not guaranteed to call this disk path on every rank.
        # Including the disk-format value still catches AUTO/None divergence
        # in direct HfWeightLoader calls, where both values enter this branch.
        requires_consensus = (checkpoint_format == "HF"
                              and load_format_name in (None, "AUTO"))
        if (requires_consensus and ENABLE_MULTI_DEVICE and not mpi_disabled()
                and mpi_comm().Get_size() == mapping.world_size):
            selections = mpi_comm().allgather((load_mode, load_format_name))
            if any(selection != selections[0] for selection in selections[1:]):
                raise RuntimeError(
                    "SafeTensors load mode and load format must match across "
                    f"all MPI ranks; received {selections}")
        coordination_error = self._hybrid_coordination_error(
            mapping, checkpoint_format, load_format_name)
        return load_mode, coordination_error

    @staticmethod
    def _hybrid_coordination_error(mapping: Mapping, checkpoint_format: str,
                                   load_format: Any) -> str | None:
        if checkpoint_format != "HF":
            return None
        if (load_format in (None, "AUTO") and ENABLE_MULTI_DEVICE
                and not mpi_disabled()):
            communicator_size = mpi_comm().Get_size()
            if communicator_size != mapping.world_size:
                return (
                    "the active MPI communicator size "
                    f"({communicator_size}) does not match mapping.world_size "
                    f"({mapping.world_size})")
        if mapping.world_size > 1 and load_format is None:
            return "distributed hybrid loading requires an explicit AUTO load format"
        return None

    @staticmethod
    def _hybrid_ineligibility_reason(model: Any,
                                     mapping: Mapping,
                                     *,
                                     checkpoint_format: str | None,
                                     uses_custom_weight_mapper: bool,
                                     load_format: Any = None) -> str | None:
        if checkpoint_format != "HF":
            return f"checkpoint format {checkpoint_format or 'unknown'} is not supported"
        if uses_custom_weight_mapper:
            return "custom weight mappers are not supported"
        if load_format is not None and getattr(load_format, "name",
                                               load_format) != "AUTO":
            return f"load format {getattr(load_format, 'name', load_format)} is not supported"
        if model is None:
            return "the initialized model was not provided"
        model_class = (type(model).__module__, type(model).__name__)
        if model_class not in _HYBRID_MODEL_CLASSES:
            return f"model type {type(model).__name__} is not supported"
        if mapping.world_size > 1 and mpi_disabled():
            return "distributed hybrid loading requires MPI-launched ranks"
        if mapping.cp_size != 1:
            return "context parallelism is not supported"
        if mapping.moe_ep_size != 1 or mapping.moe_tp_size != mapping.tp_size:
            return "expert parallelism is not supported"
        if mapping.enable_attention_dp:
            return "attention data parallelism is not supported"
        if mapping.dwdp_enabled:
            return "distributed weight data parallelism is not supported"

        model_config = getattr(model, "model_config", None)
        if model_config is None:
            return "the model has no ModelConfig"
        quant_config = getattr(model_config, "quant_config", None)
        if (quant_config is not None
                and getattr(quant_config, "quant_algo", None) is not None):
            return "quantized checkpoints are not supported"
        if getattr(model_config, "quant_config_dict", None) is not None:
            return "per-layer quantization is not supported"
        if getattr(model_config, "spec_config", None) is not None:
            return "speculative decoding models are not supported"
        if getattr(model_config, "lora_config", None) is not None:
            return "LoRA-enabled models are not supported"
        if getattr(model_config, "force_dynamic_quantization", False):
            return "dynamic quantization is not supported"
        return None

    @staticmethod
    def _coordinate_hybrid_ineligibility_reason(
            reason: str | None) -> str | None:
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            reasons = mpi_comm().allgather(reason)
        else:
            reasons = [reason]
        for rank, rank_reason in enumerate(reasons):
            if rank_reason is not None:
                return f"rank {rank}: {rank_reason}"
        return None

    @staticmethod
    def _is_weight_cache_enabled() -> bool:
        return os.environ.get(_WEIGHT_CACHE_ENV,
                              "0").lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _weight_cache_max_entries() -> int:
        try:
            return max(
                0,
                int(
                    os.environ.get(_WEIGHT_CACHE_MAX_ENTRIES_ENV,
                                   _DEFAULT_WEIGHT_CACHE_MAX_ENTRIES)))
        except ValueError:
            logger.warning(
                f"Invalid {_WEIGHT_CACHE_MAX_ENTRIES_ENV} value; disabling HF weight cache."
            )
            return 0

    @staticmethod
    def _weight_files_cache_key(weight_files: List[str],
                                use_consolidated: bool) -> tuple:
        file_fingerprint = []
        for file_name in sorted(weight_files):
            stat = os.stat(file_name)
            file_fingerprint.append(
                (os.path.abspath(file_name), stat.st_size, stat.st_mtime_ns))
        return (tuple(file_fingerprint), use_consolidated)

    @staticmethod
    def _clear_weight_cache() -> None:
        with _WEIGHT_CACHE_LOCK:
            _WEIGHT_CACHE.clear()

    @staticmethod
    def _evict_to_make_room() -> None:
        """Evict LRU entries on a miss BEFORE the new load, so CPU never holds
        the old (cached) and new (loading) weights at once (a ~2x peak)."""
        max_entries = HfWeightLoader._weight_cache_max_entries()
        if max_entries <= 0:
            return
        with _WEIGHT_CACHE_LOCK:
            while len(_WEIGHT_CACHE) >= max_entries:
                _WEIGHT_CACHE.popitem(last=False)

    @staticmethod
    def _tensor_sig(t: torch.Tensor) -> tuple:
        """A cheap integrity fingerprint: shape, dtype and a sampled sum.

        Recomputing the same sum over the same (unmutated) memory is exactly
        deterministic, so plain equality detects in-place mutation. Sampling
        up to 1024 strided elements keeps this at microseconds per tensor.
        """
        flat = t.detach().reshape(-1)
        stride = max(1, flat.numel() // 1024)
        sample = flat[::stride][:1024]
        return (tuple(t.shape), str(t.dtype),
                float(torch.nan_to_num(sample.float()).sum()))

    @staticmethod
    def _fingerprint(weights: dict[str, Any]) -> dict[str, tuple]:
        return {
            key: HfWeightLoader._tensor_sig(value)
            for key, value in weights.items() if torch.is_tensor(value)
        }

    @staticmethod
    def _cache_loaded_weights(cache_key: tuple,
                              loaded_weights: dict[str, Any]) -> None:
        max_entries = HfWeightLoader._weight_cache_max_entries()
        if max_entries <= 0:
            return

        weights = dict(loaded_weights)
        # Fingerprint outside the lock; the cache shares tensors across loads
        # (read-only by contract), and the fingerprint turns a violation of
        # that contract into a detected, self-healing miss instead of
        # silently corrupted weights (see _get_cached_weights).
        sigs = HfWeightLoader._fingerprint(weights)
        # Room was already made by the caller-side evict-before-load in
        # _with_weight_cache (the load-bearing one for the memory peak).
        with _WEIGHT_CACHE_LOCK:
            _WEIGHT_CACHE[cache_key] = (weights, sigs)

    @staticmethod
    def _get_cached_weights(cache_key: tuple) -> ConsumableWeightsDict | None:
        with _WEIGHT_CACHE_LOCK:
            entry = _WEIGHT_CACHE.get(cache_key)
            if entry is None:
                return None
            weights, sigs = entry
            _WEIGHT_CACHE.move_to_end(cache_key)
        # Integrity check: cached tensors are shared, so an earlier consumer
        # mutating them in place (e.g. an in-place transform in a weight
        # mapper) would poison every later load. Detect it, name the culprit
        # keys, drop the entry and let the caller reload from disk.
        mutated = [
            key for key, sig in sigs.items()
            if HfWeightLoader._tensor_sig(weights[key]) != sig
        ]
        if mutated:
            logger.warning(
                "HF weight cache entry was mutated in place since it was "
                f"stored (keys: {mutated[:5]}{'...' if len(mutated) > 5 else ''}); "
                "dropping it and reloading from disk. Weight preprocessing "
                "must not mutate raw checkpoint tensors.")
            with _WEIGHT_CACHE_LOCK:
                if _WEIGHT_CACHE.get(cache_key) is entry:
                    del _WEIGHT_CACHE[cache_key]
            return None
        # Return a fresh dict wrapper because model loaders call
        # mark_consumed(). Tensor values are intentionally shared: this
        # cache targets read-only raw checkpoint tensors, not per-config
        # materialized module weights.
        return ConsumableWeightsDict(dict(weights))

    @staticmethod
    def _get_local_available_host_memory() -> int:
        """Determine the minimum available memory observed on the local node
        and distribute it to all local ranks

        Because psutil.virtual_memory().available is just a snapshot in time,
        it is possible for the local ranks to get different numbers due to
        timing differences. This can lead to disagreement among the local ranks
        as to whether prefetch should be enabled, which causes a deadlock,
        because the ranks that think prefetch is enabled will wait at a local
        mpi barrier indefinitely for the ranks that do not.
        """
        available_host_memory = psutil.virtual_memory().available
        if ENABLE_MULTI_DEVICE:
            return local_mpi_comm().allreduce(available_host_memory,
                                              op=_MPI.MIN)
        return available_host_memory

    def _with_weight_cache(self, weight_files: List[str],
                           use_consolidated: bool,
                           mirror_load_collectives: bool,
                           load_fn) -> ConsumableWeightsDict:
        """Wrap ``load_fn`` with the optional raw-weight cache.

        Key -> hit (optionally joining the local barrier the miss path is
        about to enter) -> evict-before-load (so CPU never holds the old
        cached and the new loading weights at once) -> load -> store.
        """
        cache_key = self._weight_files_cache_key(
            weight_files,
            use_consolidated) if self._is_weight_cache_enabled() else None
        if cache_key is not None:
            cached_weights = self._get_cached_weights(cache_key)
            if cached_weights is not None:
                logger.info("Reusing cached HF checkpoint weights.")
                if mirror_load_collectives:
                    # Rank-local caches can diverge, so a hit on one rank must
                    # enqueue EXACTLY the collectives a miss on another rank
                    # enqueues, in the same order, or the job deadlocks. The
                    # safetensors miss path performs an Allreduce (inside
                    # _get_local_available_host_memory) and then a Barrier;
                    # mirror both here (the allreduce result is unused).
                    self._get_local_available_host_memory()
                    local_mpi_barrier()
                return cached_weights
            self._evict_to_make_room()
        weights = load_fn()
        if cache_key is not None:
            self._cache_loaded_weights(cache_key, weights)
        return weights

    def load_weights(self,
                     checkpoint_dir: str,
                     mapping: Mapping,
                     use_consolidated: bool = False,
                     **kwargs) -> dict[str, Any]:
        weight_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
        # Some model checkpoint directories contain not only the sharded safetensors, but one
        # consolidated tensor. In the presence of both, we favor the former unless specified explicitly, as there really is no need
        # to prefetch the (usually) ridiculously large consolidated tensor into memory in such a case.
        filtered_weight_files = [
            x for x in weight_files
            if ("consolidated" in os.path.split(x)[1]) == use_consolidated
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        if weight_files:
            load_format = kwargs.get("_load_format")
            checkpoint_format = kwargs.get("_checkpoint_format", "HF")
            load_mode, coordination_error = (
                self._get_coordinated_safetensors_load_mode(
                    mapping, checkpoint_format, load_format))
            if load_mode == _HYBRID_SAFETENSORS_LOAD_MODE:
                if coordination_error is not None:
                    raise RuntimeError(
                        "Hybrid SafeTensors loading cannot coordinate ranks: "
                        f"{coordination_error}")
                ineligibility_reason = self._hybrid_ineligibility_reason(
                    kwargs.get("model"),
                    mapping,
                    # A directly constructed HfWeightLoader is an HF loader;
                    # checkpoint wrappers for MX/Mistral pass their own value.
                    checkpoint_format=checkpoint_format,
                    uses_custom_weight_mapper=kwargs.get(
                        "_uses_custom_weight_mapper", False),
                    load_format=load_format)
                if getattr(load_format, "name", load_format) == "AUTO":
                    ineligibility_reason = self._coordinate_hybrid_ineligibility_reason(
                        ineligibility_reason)
                if ineligibility_reason is None:
                    if self._is_weight_cache_enabled():
                        logger.warning(
                            "The HF raw-weight cache is ignored by the hybrid "
                            "SafeTensors loader because it does not yet mirror "
                            "the hybrid collective sequence.")
                    model_type = type(kwargs["model"]).__name__
                    logger.info(
                        f"Using experimental hybrid SafeTensors loading for "
                        f"{model_type} (TP={mapping.tp_size}, "
                        f"PP={mapping.pp_size}).")
                    return self._prefetch_and_load_hybrid(weight_files)
                logger.warning(
                    f"Hybrid SafeTensors loading is ineligible because "
                    f"{ineligibility_reason}; falling back to eager loading.")
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=True,
                load_fn=lambda: self._prefetch_and_load(weight_files))

        weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
        if not weight_files:
            weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

        if weight_files:
            return self._with_weight_cache(
                weight_files,
                use_consolidated,
                mirror_load_collectives=False,
                load_fn=lambda: self._load_weights_in_parallel(
                    weight_files, self._load_bin_or_path_file,
                    "Loading bin weights in parallel"))

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    def _prefetch_and_load(self,
                           weight_files: List[str]) -> ConsumableWeightsDict:
        prefetch_size, enable_prefetch = self._get_prefetch_policy(weight_files)
        if enable_prefetch:
            logger.info(
                f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
            )
            self.prefetch_files(weight_files)
        # Sync all local ranks unconditionally. `enable_prefetch` depends on
        # `psutil.virtual_memory().available`, a per-rank volatile value, so
        # different ranks may take different branches; gating the barrier on
        # it would deadlock between ranks that prefetched and ranks that
        # skipped. Ranks that didn't prefetch reach the barrier immediately.
        local_mpi_barrier()

        return self._load_weights_in_parallel(
            weight_files, self._load_safetensors_file,
            "Loading safetensors weights in parallel")

    def _prefetch_and_load_hybrid(
            self, weight_files: List[str]) -> ConsumableWeightsDict:
        """Cooperatively prefetch and mmap SafeTensors.

        When the existing memory heuristic permits full prefetch, every local
        MPI rank owns disjoint file chunks and uses a bounded thread pool to
        read them. This parallelizes a single large SafeTensors file as well as
        a sharded checkpoint. Otherwise, the existing mmap-backed SafeTensors
        loader demand-pages the ranges requested during model loading.
        """
        node_communicator = self._get_hybrid_node_communicator()
        try:
            return self._prefetch_and_load_hybrid_with_communicator(
                weight_files, node_communicator)
        finally:
            if node_communicator is not None:
                node_communicator.Free()

    def _prefetch_and_load_hybrid_with_communicator(
            self, weight_files: List[str],
            node_communicator) -> ConsumableWeightsDict:
        """Run the hybrid stages using one active-communicator node group."""
        prefetch_size, enable_prefetch = self._get_hybrid_prefetch_policy(
            weight_files, node_communicator)
        local_rank, _ = self._get_local_rank_and_size(node_communicator)
        prefetch_started = time.perf_counter()
        prefetch_error = None
        try:
            if enable_prefetch:
                logger.info(
                    f"Hybrid-prefetching {prefetch_size / (1024**3):.2f}GB "
                    "of checkpoint data in bounded chunks.")
                self.prefetch_file_chunks(weight_files, node_communicator)
            else:
                logger.info(
                    "Skipping hybrid full-checkpoint prefetch; weights will be "
                    "loaded through the existing mmap-backed path.")
        except Exception as error:
            # Coordinate ordinary failures before entering a node-local
            # barrier. Otherwise peers can wait forever after one rank exits.
            prefetch_error = error
        self._raise_on_rank_error("hybrid checkpoint prefetch", prefetch_error)

        # Available memory is reduced within each node. Different nodes may
        # make different prefetch decisions, but every rank follows the same
        # collective sequence and the barrier remains node-local.
        if node_communicator is not None:
            node_communicator.Barrier()
        if enable_prefetch and local_rank == 0:
            prefetch_elapsed = time.perf_counter() - prefetch_started
            prefetch_throughput = prefetch_size / prefetch_elapsed / (1024**3)
            logger.info(
                f"Hybrid checkpoint prefetch completed in "
                f"{prefetch_elapsed:.2f}s ({prefetch_throughput:.2f}GB/s "
                "logical read rate).")

        mmap_started = time.perf_counter()
        weights = None
        mmap_error = None
        try:
            weights = self._load_weights_in_parallel(
                weight_files,
                self._load_safetensors_file,
                "Mapping safetensors weights in parallel",
                reject_duplicate_keys=True)
        except Exception as error:
            # Complete world-rank consensus before any healthy rank returns
            # weights and begins mutating model parameters.
            mmap_error = error
        self._raise_on_rank_error("hybrid SafeTensors mmap setup", mmap_error)
        assert weights is not None
        if local_rank == 0:
            mmap_elapsed = time.perf_counter() - mmap_started
            logger.info(f"Hybrid SafeTensors mmap setup completed in "
                        f"{mmap_elapsed:.2f}s.")
        return weights

    @staticmethod
    def _raise_on_rank_error(phase: str, error: Exception | None) -> None:
        error_message = (None if error is None else
                         f"{type(error).__name__}: {error}")
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            error_messages = mpi_comm().allgather(error_message)
        else:
            error_messages = [error_message]
        for rank, rank_error in enumerate(error_messages):
            if rank_error is not None:
                raise RuntimeError(f"Rank {rank} failed during {phase}: "
                                   f"{rank_error}") from error

    @staticmethod
    def _get_hybrid_node_communicator():
        """Derive a node-local group from the active model-load communicator."""
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            return mpi_comm().Split_type(_MPI.COMM_TYPE_SHARED)
        return None

    def _get_hybrid_prefetch_policy(self,
                                    weight_files: List[str],
                                    node_communicator=None) -> tuple[int, bool]:
        """Compute a policy without entering a collective after local errors."""
        prefetch_size = None
        num_layers = None
        available_host_memory = None
        checkpoint_signature = None
        policy_error = None
        try:
            file_sizes = [(os.path.basename(file), os.path.getsize(file))
                          for file in weight_files]
            checkpoint_signature = tuple(file_sizes)
            prefetch_size = sum(size for _, size in file_sizes)
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            available_host_memory = psutil.virtual_memory().available
        except Exception as error:
            policy_error = error

        # This world collective occurs before the node-local memory allreduce,
        # so a stat/env/psutil error cannot strand healthy local peers there.
        self._raise_on_rank_error("hybrid prefetch policy", policy_error)
        assert checkpoint_signature is not None
        assert prefetch_size is not None
        assert num_layers is not None
        assert available_host_memory is not None

        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            policy_inputs = mpi_comm().allgather(
                (checkpoint_signature, num_layers))
            if any(inputs != policy_inputs[0] for inputs in policy_inputs[1:]):
                raise RuntimeError(
                    "Hybrid SafeTensors checkpoint selection and "
                    "TLLM_OVERRIDE_LAYER_NUM must match across MPI ranks")
        if node_communicator is not None:
            available_host_memory = node_communicator.allreduce(
                available_host_memory, op=_MPI.MIN)

        enable_prefetch = (prefetch_size < available_host_memory * 0.9
                           and num_layers == 0)
        return prefetch_size, enable_prefetch

    def _get_prefetch_policy(self, weight_files: List[str]) -> tuple[int, bool]:
        """Return checkpoint size and the node-consistent prefetch decision."""
        # Prefetch only when the files use less than 90% of available host
        # memory. This avoids page-cache thrashing for oversized checkpoints.
        prefetch_size = sum(os.path.getsize(file) for file in weight_files)
        # A layer override means only a model subset is loaded, so staging the
        # complete checkpoint would be wasted I/O.
        num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
        enable_prefetch = (prefetch_size
                           < self._get_local_available_host_memory() * 0.9
                           and num_layers == 0)
        return prefetch_size, enable_prefetch

    def _load_weights_in_parallel(
            self,
            weight_files: List[str],
            load_func,
            description: str,
            reject_duplicate_keys: bool = False) -> ConsumableWeightsDict:
        """
        Load weight files in parallel using the specified loading function.

        Args:
            weight_files: List of weight file paths
            load_func: Function to load individual weight files
            description: Description for the progress bar

        Returns:
            ConsumableWeightsDict containing all loaded weights
        """
        weights = {}
        pbar = tqdm.tqdm(total=len(weight_files), desc=description)

        # Note that the function is called with a tuple of arguments, hence we need to wrap the arguments in a tuple via [(w,) for w in weight_files]
        # specifically the comma right after the w is important to make it a tuple.
        def update_weights(part_weights: dict[str, Any]) -> None:
            if reject_duplicate_keys:
                duplicate_keys = weights.keys() & part_weights.keys()
                if duplicate_keys:
                    duplicate_key = min(duplicate_keys)
                    raise RuntimeError(
                        "Duplicate SafeTensors key found across checkpoint "
                        f"shards: {duplicate_key}")
            weights.update(part_weights)

        run_concurrently(load_func, [(w, ) for w in weight_files],
                         reduce_func=update_weights,
                         pbar=pbar)

        return ConsumableWeightsDict(weights)

    @staticmethod
    def _load_safetensors_file(file):
        logger.info(f"Start to load safetensor file {file}")
        return safetensors.torch.load_file(file)

    @staticmethod
    def _load_bin_or_path_file(file):
        try:
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=True)
        except Exception:
            logger.warning(
                f"Failed to load {file} with mmap=True, fallback to mmap=False")
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=False)
        finally:
            return part_weights

    def _prefetch_one_file(self, file_name):
        if os.path.exists(file_name):
            logger.info(f"Prefetching {file_name} to memory...")
            with open(file_name, 'rb') as f:
                f.read()
            logger.info(f"Finished prefetching {file_name}.")

    @staticmethod
    def _prefetch_one_chunk(file_name: str, offset: int, length: int) -> None:
        """Read a bounded file extent into the OS page cache."""
        with open(file_name, "rb", buffering=0) as f:
            file_descriptor = f.fileno()
            read_offset = offset
            remaining = length
            while remaining > 0:
                read_size = min(remaining, _HYBRID_PREFETCH_READ_SIZE)
                data = os.pread(file_descriptor, read_size, read_offset)
                if not data:
                    raise OSError(
                        f"Unexpected EOF while prefetching {file_name} at "
                        f"offset {read_offset}")
                bytes_read = len(data)
                read_offset += bytes_read
                remaining -= bytes_read

    def _local_prefetch_chunks(
            self,
            file_names: List[str],
            node_communicator=None) -> list[tuple[str, int, int]]:
        """Build this rank's deterministic share of checkpoint extents."""
        chunks = []
        for file_name in sorted(file_names):
            file_size = os.path.getsize(file_name)
            for offset in range(0, file_size, self._hybrid_prefetch_chunk_size):
                length = min(self._hybrid_prefetch_chunk_size,
                             file_size - offset)
                chunks.append((file_name, offset, length))
        local_rank, local_size = self._get_local_rank_and_size(
            node_communicator)
        return chunks[local_rank::local_size]

    @staticmethod
    def _get_local_rank_and_size(node_communicator=None) -> tuple[int, int]:
        if node_communicator is not None:
            return (node_communicator.Get_rank(), node_communicator.Get_size())
        return 0, 1

    def prefetch_file_chunks(self,
                             file_names: List[str],
                             node_communicator=None) -> None:
        """Prefetch files in chunks distributed across local ranks and CPUs."""
        local_chunks = self._local_prefetch_chunks(file_names,
                                                   node_communicator)
        if not local_chunks:
            return

        local_rank, local_size = self._get_local_rank_and_size(
            node_communicator)
        if self._hybrid_prefetch_workers_per_rank is None:
            cpu_share = max(1, multiprocessing.cpu_count() // local_size)
            max_workers = min(_DEFAULT_HYBRID_PREFETCH_WORKERS_PER_RANK,
                              cpu_share, len(local_chunks))
        else:
            max_workers = min(self._hybrid_prefetch_workers_per_rank,
                              len(local_chunks))

        logger.debug(
            f"Hybrid local rank {local_rank} prefetching {len(local_chunks)} "
            f"chunks with {max_workers} workers.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._prefetch_one_chunk, *chunk)
                for chunk in local_chunks
            ]
            for future in futures:
                future.result()

    def prefetch_files(self, file_names: List[str]):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        """
        # Find out the files to prefetch for the current rank.
        # Each rank loads files with indices local_rank, local_rank + local_mpi_size, local_rank + 2*local_mpi_size, etc.
        local_file_names = file_names[local_mpi_rank()::local_mpi_size()]
        if len(local_file_names) == 0:
            return

        max_workers = min(multiprocessing.cpu_count() * 2, 16,
                          len(local_file_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._prefetch_one_file, local_file_names))
