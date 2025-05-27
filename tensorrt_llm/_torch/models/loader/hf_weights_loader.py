import glob
import multiprocessing
import os
from typing import Any, List

import safetensors
import torch

from .file_system_weights_loader_interface import \
    FileSystemWeightsLoaderInterface

# from tensorrt_llm.logger import logger


class HfWeightsLoader(FileSystemWeightsLoaderInterface):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

    def load_weights(self, checkpoint_dir: str) -> dict[str, Any]:
        weights = {}
        weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        if weight_files:
            self._prefetch_files(weight_files)
            for file in weight_files:
                # logger.info(f"Loading {file}")
                part_weights = safetensors.torch.load_file(file)
                weights.update(part_weights)
            return weights

        weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
        if not weight_files:
            weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

        if weight_files:
            for file in weight_files:
                # try mmap first, if failed, turn off mmap
                try:
                    part_weights = torch.load(file,
                                              weights_only=True,
                                              map_location='cpu',
                                              mmap=True)
                except Exception:
                    # logger.warning(
                    #     f"Failed to load {file} with mmap=True, fallback to mmap=False"
                    # )
                    part_weights = torch.load(file,
                                              weights_only=True,
                                              map_location='cpu',
                                              mmap=False)
                weights.update(part_weights)
            return weights

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    def _prefetch_files(self, file_names: List[str]):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        TODO: On systems with small memory, prefetching may cause file cache thrashing, so we may want to add some
        heuristics about when to prefetch and when not to.
        """

        def _prefetch_one_file(file_name, rank):
            if os.path.exists(file_name):
                # logger.info(f"Rank {rank} prefetching {file_name} to memory...")
                with open(file_name, 'rb') as f:
                    f.read()
                # logger.info(f"Rank {rank} finished prefetching {file_name}.")

        # Find out the files to prefetch for the current rank.
        # Each rank loads files with indices rank, rank + world_size, rank + 2*world_size, etc.
        local_file_names = file_names[self._mapping.rank::self._mapping.
                                      world_size]

        processes = []
        for file_name in local_file_names:
            process = multiprocessing.Process(target=_prefetch_one_file,
                                              args=(file_name,
                                                    self._mapping.rank))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
