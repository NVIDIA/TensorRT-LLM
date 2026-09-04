from dataclasses import dataclass
from typing import Any

import torch


@dataclass(repr=False)
class Scenario:
    backend: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-use_cuda_graph:{self.use_cuda_graph}"


def reduce_llama_config(mem_for_full_model: int,
                        config_dict: dict[str, Any],
                        default_num_layers: int = 32):
    _, total_mem = torch.cuda.mem_get_info()
    # scale model down if gpu memory is low
    if total_mem < mem_for_full_model:
        model_fraction = total_mem / mem_for_full_model
        num_layers = int(config_dict["num_hidden_layers"] * model_fraction)
        num_layers = min(num_layers, default_num_layers)
        config_dict["num_hidden_layers"] = num_layers
