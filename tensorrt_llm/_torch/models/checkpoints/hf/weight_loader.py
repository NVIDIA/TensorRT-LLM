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
                                 local_mpi_comm, local_mpi_rank, local_mpi_size)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@register_checkpoint_weight_loader("mistral")
@register_checkpoint_weight_loader("HF")
class HfWeightLoader(BaseWeightLoader):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

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

    def load_weights(self, checkpoint_dir: str,
                     mapping: Mapping) -> dict[str, Any]:
        weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        # Some model checkpoint directories contain not only the sharded safetensors, but one
        # consolidated tensor. In the presence of both, we favor the former, as there really is no need
        # to prefetch the (usually) ridiculously large consolidated tensor into memory in such a case.
        filtered_weight_files = [
            x for x in weight_files if "consolidated" not in os.path.split(x)[1]
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        if weight_files:
            # Prefetch the weight files to CPU memory if the size is less than 90% of the available memory.
            # This is a heuristic to avoid prefetching files that are too large and causing file cache thrashing.
            prefetch_size = sum(os.path.getsize(file) for file in weight_files)
            # If the layer number is overridden, it indicates that only a subset of layers are loaded.
            # Prefetching all layers is unnecessary.
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            enable_prefetch = (prefetch_size
                               < self._get_local_available_host_memory() * 0.9
                               and num_layers == 0)
            if enable_prefetch:
                logger.info(
                    f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
                )
                self.prefetch_files(weight_files)
                # Ensure that all local ranks have finished prefetching before loading weights
                local_mpi_barrier()

            return self._load_weights_in_parallel(
                weight_files, self._load_safetensors_file,
                "Loading safetensors weights in parallel")

        weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
        if not weight_files:
            weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

        if weight_files:
            return self._load_weights_in_parallel(
                weight_files, self._load_bin_or_path_file,
                "Loading bin weights in parallel")

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    def _load_weights_in_parallel(self, weight_files: List[str], load_func,
                                  description: str) -> ConsumableWeightsDict:
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
        run_concurrently(load_func, [(w, ) for w in weight_files],
                         reduce_func=weights.update,
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
