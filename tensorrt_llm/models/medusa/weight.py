from pathlib import Path

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import MedusaLM
from tensorrt_llm.models.llama.weight import split


def load_medusa_hf(medusa_path: str,
                   trt_llm_medusa: MedusaLM,
                   mapping=Mapping(),
                   dtype='float32'):
    logger.info("Loading Medusa heads' weights ...")
    ckpt_file = Path(medusa_path) / "medusa_lm_head.pt"
    state_dict = torch.load(ckpt_file, map_location="cpu")
    torch_dtype = str_dtype_to_torch(dtype)
    for h in range(trt_llm_medusa.num_medusa_heads):
        for l in range(trt_llm_medusa.num_medusa_layers):
            w = state_dict[f"{h}.{l}.linear.weight"].clone()
            w = torch_to_numpy(w.to(torch_dtype).detach().cpu())
            trt_llm_medusa.medusa_heads[h].medusa_layers[
                l].linear.weight.value = np.ascontiguousarray(
                    split(w, mapping.tp_size, mapping.tp_rank))
            if trt_llm_medusa.medusa_heads[h].medusa_layers[
                    l].linear.bias is not None:
                # print(f"Setting bias for {h} {l}")
                b = state_dict[f"{h}.{l}.linear.bias"].clone()
                b = torch_to_numpy(b.to(torch_dtype).detach().cpu())
                trt_llm_medusa.medusa_heads[h].medusa_layers[
                    l].linear.bias.value = np.ascontiguousarray(
                        np.split(b, mapping.tp_size,
                                 axis=0)[mapping.tp_rank].copy())
        lm = state_dict[f"{h}.{trt_llm_medusa.num_medusa_layers}.weight"].clone(
        )  # LM Head
        lm = torch_to_numpy(lm.to(torch_dtype).detach().cpu())
        trt_llm_medusa.medusa_heads[
            h].lm_head.weight.value = np.ascontiguousarray(
                split(lm, mapping.tp_size, mapping.tp_rank))
    return
