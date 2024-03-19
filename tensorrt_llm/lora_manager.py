import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ._utils import (DictConversion, pad_vocab_size, str_dtype_to_torch,
                     torch_to_numpy, unpack_nemo_weights)
from .layers.linear import ColumnLinear
from .models.convert_utils import split_matrix_tp


def get_all_nemo_lora_weights(num_layers, lora_weights):
    layer_weights = [{} for _ in range(num_layers)]
    adapter_key = "self_attention.adapter_layer.lora_kqv_adapter"
    layer_pattern = re.compile(r'.*\.layers\.([0-9]+)\..*')
    for key, weights in lora_weights.items():
        if adapter_key in key:
            if key.endswith('linear_in.weight'):
                inout = 'in'
            elif key.endswith('linear_out.weight'):
                inout = 'out'
            else:
                continue
            m = layer_pattern.match(key)
            layer_idx = int(m.group(1))
            layer_weights[layer_idx][inout] = weights
    return layer_weights


@dataclass
class LoraBuildConfig(DictConversion):
    lora_dir: List[str] = field(default_factory=list)
    lora_ckpt_source: str = 'hf'
    max_lora_rank: int = 64
    lora_target_modules: List[str] = field(default_factory=list)
    trtllm_modules_to_hf_modules: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert self.lora_ckpt_source in [
            'hf', 'nemo'
        ], f"lora_ckpt_source must be one of 'hf' or 'nemo', got {self.lora_ckpt_source}"


class HfLoraLoader:

    def __init__(self, lora_dirs: List[str]):
        self.lora_target_modules = []
        self.is_valid = False
        self.lm_head = None
        self.embed_tokens = None
        self.vocab_size = 0

        if len(lora_dirs) == 0:
            return

        for lora_dir in lora_dirs:
            for filename in ["adapter_config.json", "adapter_model.bin"]:
                path = Path(f"{lora_dir}/{filename}")
                if not path.exists():
                    raise ValueError(f"{path} does not exist")
                if not path.is_file():
                    raise ValueError(f"{path} is not a file")
        self.is_valid = True

        lora_dir = lora_dirs[0]
        with open(f"{lora_dir}/adapter_config.json") as f:
            adapter_config = json.load(f)
        self.lora_target_modules = adapter_config["target_modules"]

        lora_weight = torch.load(f"{lora_dir}/adapter_model.bin")
        if adapter_config["modules_to_save"] is not None:
            if "lm_head" in adapter_config["modules_to_save"]:
                self.lm_head = lora_weight["base_model.model.lm_head.weight"]
                self.vocab_size = self.lm_head.shape[0]

            if "embed_tokens" in adapter_config["modules_to_save"]:
                self.embed_tokens = lora_weight[
                    "base_model.model.model.embed_tokens.weight"]

    def get_target_modules(self, trtllm_modules_to_hf_modules):
        hf_modules_to_trtllm_modules = {
            v: k
            for k, v in trtllm_modules_to_hf_modules.items()
        }
        lora_target_modules = []
        if self.is_valid:
            # lora_target_modules[m] can ba either a string or a list of strings
            for m in self.lora_target_modules:
                trtllm_module = hf_modules_to_trtllm_modules[m]
                if isinstance(trtllm_module, list):
                    lora_target_modules.extend(trtllm_module)
                else:
                    lora_target_modules.append(trtllm_module)
        return lora_target_modules


class NemoLoraLoader:

    def __init__(self, lora_dirs: List[str]):
        self.lora_target_modules = []
        self.is_valid = False

        if len(lora_dirs) == 0:
            return

        for lora_file in lora_dirs:
            path = Path(lora_file)
            if not path.exists():
                raise ValueError(f"{path} does not exist")
            if not path.is_file():
                raise ValueError(f"{path} is not a file")
        self.is_valid = True
        # Hardcoded since LoraManager only supports this case now
        self.lora_target_modules = ["attn_qkv"]


def load_nemo_lora(model, lora_config: LoraBuildConfig):
    lora_loader = NemoLoraLoader(lora_config.lora_dir)
    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.lora_target_modules


def load_hf_lora(
    model,
    lora_config: LoraBuildConfig,
    trtllm_modules_to_hf_modules: Dict[str, str] = None,
):
    trtllm_modules_to_hf_modules = trtllm_modules_to_hf_modules or {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_dense": "o_proj",
        "mlp_h_to_4h": "gate_proj",
        "mlp_4h_to_h": "down_proj",
        "mlp_gate": "up_proj",
    }
    lora_config.trtllm_modules_to_hf_modules = trtllm_modules_to_hf_modules

    lora_loader = HfLoraLoader(lora_config.lora_dir)

    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.get_target_modules(
            trtllm_modules_to_hf_modules)

    config = model.config
    if lora_loader.is_valid:
        # the lora checkpoint might finetune the embedding
        if lora_loader.vocab_size != 0:
            config.vocab_size = lora_loader.vocab_size
        mapping = config.mapping
        if mapping.is_first_pp_rank() and lora_loader.embed_tokens is not None:
            weight = lora_loader.embed_tokens
            if config.use_parallel_embedding:
                weight = split_matrix_tp(
                    weight,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=config.embedding_sharding_dim,
                )
            if model.transformer.vocab_embedding.weight.raw_value.shape != weight.shape:
                model.transformer.vocab_embedding = model.transformer.vocab_embedding.__class__(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.hidden_size,
                    dtype=config.dtype,
                    tp_size=mapping.tp_size
                    if config.use_parallel_embedding else 1,
                    tp_group=mapping.tp_group
                    if config.use_parallel_embedding else None,
                    sharding_dim=config.embedding_sharding_dim,
                    tp_rank=mapping.tp_rank,
                )
            model.transformer.vocab_embedding.weight.value = weight
        if mapping.is_last_pp_rank() and lora_loader.lm_head is not None:
            weight = lora_loader.lm_head
            vocab_size = lora_loader.vocab_size
            if vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size

                weight = torch.from_numpy(
                    np.pad(weight.detach().cpu().numpy(),
                           ((0, pad_width), (0, 0)),
                           'constant',
                           constant_values=0))
            else:
                vocab_size_padded = vocab_size
            if model.lm_head.weight.raw_value.shape != weight.shape:
                model.lm_head = ColumnLinear(
                    config.hidden_size,
                    vocab_size_padded,
                    bias=False,
                    dtype=config.dtype,
                    tp_group=mapping.tp_group,
                    tp_size=mapping.tp_size,
                    gather_output=True,
                )
            model.lm_head.weight.value = split_matrix_tp(
                weight,
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            )


def use_lora(
    model,
    lora_config: LoraBuildConfig,
    trtllm_modules_to_hf_modules: Dict[str, str] = None,
):
    model.lora_config = lora_config
    if lora_config.lora_ckpt_source == "nemo":
        load_nemo_lora(model, lora_config)
    elif lora_config.lora_ckpt_source == "hf":
        load_hf_lora(model, lora_config, trtllm_modules_to_hf_modules)
    else:
        raise ValueError(
            f"Unsupported lora_ckpt_source: {lora_config.lora_ckpt_source}")


class LoraConfig(object):

    def __init__(self,
                 hf_lora_dir: str = None,
                 adapter_config: dict = {},
                 tokenizer_config: dict = {},
                 lora_target_modules: list = [],
                 is_valid: bool = False,
                 has_tokenizer: bool = False,
                 lm_head_weight=None,
                 embedding_weight=None,
                 hf_modules_to_trtllm_modules: dict = {},
                 trtllm_modules_to_hf_modules: dict = {}):
        self.hf_lora_dir = hf_lora_dir
        self.adapter_config = adapter_config
        self.tokenizer_config = tokenizer_config
        self.hf_lora_target_modules = lora_target_modules
        self.lora_target_modules = []
        # lora_target_modules[m] can ba either a string or a list of strings
        for m in lora_target_modules:
            trtllm_module = hf_modules_to_trtllm_modules[m]
            if isinstance(trtllm_module, list):
                self.lora_target_modules.extend(trtllm_module)
            else:
                self.lora_target_modules.append(trtllm_module)
        self.is_valid = is_valid
        self.has_tokenizer = has_tokenizer
        self.lm_head_weight = lm_head_weight
        self.embedding_weight = embedding_weight
        self.vocab_size, self.hidden_size = self.lm_head_weight.shape if self.lm_head_weight is not None else (
            0, 0)
        self.hf_modules_to_trtllm_modules = hf_modules_to_trtllm_modules
        self.trtllm_modules_to_hf_modules = trtllm_modules_to_hf_modules

    @classmethod
    def from_hf(cls, hf_lora_dir, hf_modules_to_trtllm_modules,
                trtllm_modules_to_hf_modules):
        lora_target_modules = {}
        adapter_config = None
        tokenizer_config = None
        hf_lora_dir = hf_lora_dir
        is_valid = True
        has_tokenizer = True

        if os.path.exists(f"{hf_lora_dir}/adapter_config.json"):
            with open(f"{hf_lora_dir}/adapter_config.json") as f:
                adapter_config = json.load(f)
            lora_target_modules = adapter_config["target_modules"]
        else:
            is_valid = False

        if os.path.exists(f"{hf_lora_dir}/tokenizer_config.json"):
            with open(f"{hf_lora_dir}/tokenizer_config.json") as f:
                tokenizer_config = json.load(f)
        else:
            has_tokenizer = False

        lm_head_weight = None
        embedding_weight = None

        if os.path.exists(f"{hf_lora_dir}/adapter_model.bin"):
            lora_weight = torch.load(f"{hf_lora_dir}/adapter_model.bin")

            if adapter_config["modules_to_save"] is not None:
                if "lm_head" in adapter_config["modules_to_save"]:
                    lm_head_weight = lora_weight[
                        "base_model.model.lm_head.weight"]

                if "embed_tokens" in adapter_config["modules_to_save"]:
                    embedding_weight = lora_weight[
                        "base_model.model.model.embed_tokens.weight"]

        return cls(hf_lora_dir, adapter_config, tokenizer_config,
                   lora_target_modules, is_valid, has_tokenizer, lm_head_weight,
                   embedding_weight, hf_modules_to_trtllm_modules,
                   trtllm_modules_to_hf_modules)


class LoraManager(object):
    LORA_MODULE_IDS = {
        "attn_qkv": 0,
        "attn_q": 1,
        "attn_k": 2,
        "attn_v": 3,
        "attn_dense": 4,
        "mlp_h_to_4h": 5,
        "mlp_4h_to_h": 6,
        "mlp_gate": 7,
        "cross_attn_qkv": 8,
        "cross_attn_q": 9,
        "cross_attn_k": 10,
        "cross_attn_v": 11,
        "cross_attn_dense": 12,
    }

    def __init__(self):
        self._lora_uid_to_key = {}
        '''
        _lora_uid_to_low_ranks: dict[str -> List[dict[str -> int]]]
        {
            uid:
            [
                {
                    lora_module: int
                }, # layer_0_rank,
                {
                    lora_module: int
                }, # layer_1_rank,
                ...
            ]
        }

        _lora_weights_pointers_list:
        [
            {
               uid:
               {
                   lora_module_1: [t_in, t_out]
                   lora_module_2: [t_in, t_out]
               }
            }, # layer_0
            {

            }, # layer_1
            ...
        ]

        '''
        self._lora_uid_to_low_ranks = {}
        self._lora_weights = []
        self._lora_weights_pointers_list = []
        self._lora_cpp_weights = {}
        self._lora_weight_config = {}

    def load_from_ckpt(self, model_dir, model_config, runtime_mapping,
                       ckpt_source):
        if ckpt_source == "hf":
            self.load_from_hf(model_dir, model_config, runtime_mapping)
        elif ckpt_source == "nemo":
            self.load_from_nemo(model_dir, model_config, runtime_mapping)
        else:
            assert False, f"LoraManager does not support source {ckpt_source}"

    def load_from_nemo(self, model_files, model_config, runtime_mapping):
        tp_size = runtime_mapping.tp_size
        tp_rank = runtime_mapping.tp_rank
        lora_target_modules = model_config.lora_target_modules
        dtype = model_config.dtype

        uids = ["-1"]
        for i in range(len(model_files)):
            uids.append(str(i))
        model_files = [""] + model_files

        for uid, model_file in zip(uids, model_files):
            if uid not in self._lora_cpp_weights:
                self._lora_cpp_weights[uid] = []
            if uid not in self._lora_weight_config:
                self._lora_weight_config[uid] = []

            if model_file != "":
                _, nemo_weights = unpack_nemo_weights(model_file)
                all_lora_weights = get_all_nemo_lora_weights(
                    model_config.num_layers, nemo_weights)
            else:
                all_lora_weights = None
                nemo_weights = None

            self._lora_uid_to_low_ranks[uid] = []
            for layer_idx in range(model_config.num_layers):
                self._lora_weights_pointers_list.append({})
                self._lora_weights_pointers_list[layer_idx].update({uid: {}})

                self._lora_uid_to_low_ranks[uid].append({})

                for lora_module in lora_target_modules:
                    if uid == "-1" or lora_module != "attn_qkv" or all_lora_weights is None:
                        self._lora_uid_to_low_ranks[uid][layer_idx][
                            lora_module] = 0
                        continue

                    if lora_module == "attn_qkv":
                        t_in = all_lora_weights[layer_idx]["in"]
                        t_out = all_lora_weights[layer_idx]["out"]
                        assert t_out.shape[0] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[0] // tp_size,
                                            dim=0)[tp_rank].contiguous()
                    else:
                        t_in = None
                        t_out = None

                    if t_in is not None and t_out is not None:
                        t_in = t_in.cuda().to(
                            str_dtype_to_torch(dtype)).contiguous()
                        t_out = t_out.cuda().to(
                            str_dtype_to_torch(dtype)).contiguous()
                        rank = t_in.shape[0]
                        self._lora_weights_pointers_list[layer_idx][uid].update(
                            {lora_module: [t_in.data_ptr(),
                                           t_out.data_ptr()]})
                        self._lora_uid_to_low_ranks[uid][layer_idx][
                            lora_module] = int(rank)

                        # prevent torch free this buffer
                        self._lora_weights.append(t_in)
                        self._lora_weights.append(t_out)
                        self._lora_cpp_weights[uid].append(
                            torch.concatenate([t_in.flatten(),
                                               t_out.flatten()]))
                        self._lora_weight_config[uid].append(
                            np.array([
                                self.LORA_MODULE_IDS[lora_module], layer_idx,
                                int(rank)
                            ],
                                     dtype=np.int32))

            del nemo_weights

    def load_from_hf(self, model_dirs, model_config, runtime_mapping):
        '''
        lora config of https://huggingface.co/hfl/chinese-alpaca-2-lora-7b
        {
            "base_model_name_or_path": "/Llama-2-7b-hf",
            "bias": "none",
            "enable_lora": null,
            "fan_in_fan_out": false,
            "inference_mode": true,
            "lora_alpha": 128.0,
            "lora_dropout": 0.05,
            "merge_weights": false,
            "modules_to_save": [
                "embed_tokens",
                "lm_head"
            ],
            "peft_type": "LORA",
            "r": 64,
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            "task_type": "CAUSAL_LM"

        }

        keys in adapter_model.bin:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight torch.Size([11008, 64])
            base_model.model.model.layers.0.mlp.up_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.mlp.up_proj.lora_B.weight torch.Size([11008, 64])
            base_model.model.model.layers.0.mlp.down_proj.lora_A.weight torch.Size([64, 11008])
            base_model.model.model.layers.0.mlp.down_proj.lora_B.weight torch.Size([4096, 64])
            ...

        '''
        tp_size = runtime_mapping.tp_size
        tp_rank = runtime_mapping.tp_rank

        lora_hf_configs = [{}]
        ranks = [0]
        uids = ["-1"]
        for i, model_dir in enumerate(model_dirs):
            with open(f"{model_dir}/adapter_config.json", 'r') as f:
                config = json.load(f)
                lora_hf_configs.append(config)
                ranks.append(config["r"])
                uids.append(str(i))
        new_model_dirs = [""] + model_dirs

        lora_target_modules = model_config.lora_target_modules
        dtype = model_config.dtype

        for uid, rank, model_dir, hf_config in zip(uids, ranks, new_model_dirs,
                                                   lora_hf_configs):
            if uid not in self._lora_cpp_weights:
                self._lora_cpp_weights[uid] = []
            if uid not in self._lora_weight_config:
                self._lora_weight_config[uid] = []

            if model_dir != "":
                lora_model = torch.load(f"{model_dir}/adapter_model.bin")
            else:
                lora_model = None

            self._lora_uid_to_low_ranks[uid] = []
            for layer_idx in range(model_config.num_layers):
                self._lora_weights_pointers_list.append({})
                self._lora_weights_pointers_list[layer_idx].update({uid: {}})

                self._lora_uid_to_low_ranks[uid].append({})

                prefix = "base_model.model.model.layers"
                for lora_module in lora_target_modules:
                    if uid == "-1" or model_config.trtllm_modules_to_hf_modules[
                            lora_module] not in hf_config["target_modules"]:
                        self._lora_uid_to_low_ranks[uid][layer_idx][
                            lora_module] = 0
                        continue

                    if lora_module == "attn_q" or lora_module == "attn_k" or lora_module == "attn_v":
                        name = f"{prefix}.{layer_idx}.{lora_module.replace('attn_', 'self_attn.')}_proj"
                        # not split
                        t_in = lora_model[f"{name}.lora_A.weight"]
                        # split by column
                        t_out = lora_model[f"{name}.lora_B.weight"]
                        assert t_out.shape[0] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[0] // tp_size,
                                            dim=0)[tp_rank].contiguous()

                    elif lora_module == "attn_dense":
                        # split by row
                        t_in = lora_model[
                            f"{prefix}.{layer_idx}.self_attn.o_proj.lora_A.weight"]
                        assert t_in.shape[1] % tp_size == 0
                        t_in = torch.split(t_in,
                                           t_in.shape[1] // tp_size,
                                           dim=1)[tp_rank].contiguous()
                        # not split
                        t_out = lora_model[
                            f"{prefix}.{layer_idx}.self_attn.o_proj.lora_B.weight"]

                    elif lora_module == "mlp_h_to_4h":
                        # not split
                        t_in = lora_model[
                            f"{prefix}.{layer_idx}.mlp.gate_proj.lora_A.weight"]
                        # split by column
                        t_out = lora_model[
                            f"{prefix}.{layer_idx}.mlp.gate_proj.lora_B.weight"]
                        assert t_out.shape[0] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[0] // tp_size,
                                            dim=0)[tp_rank].contiguous()

                    elif lora_module == "mlp_gate":
                        # not split
                        t_in = lora_model[
                            f"{prefix}.{layer_idx}.mlp.up_proj.lora_A.weight"]
                        # split by column
                        t_out = lora_model[
                            f"{prefix}.{layer_idx}.mlp.up_proj.lora_B.weight"]
                        assert t_out.shape[0] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[0] // tp_size,
                                            dim=0)[tp_rank].contiguous()

                    elif lora_module == "mlp_4h_to_h":
                        # split by row
                        t_in = lora_model[
                            f"{prefix}.{layer_idx}.mlp.down_proj.lora_A.weight"]
                        assert t_in.shape[0] % tp_size == 0
                        t_in = torch.split(t_in,
                                           t_in.shape[1] // tp_size,
                                           dim=1)[tp_rank].contiguous()
                        # not split
                        t_out = lora_model[
                            f"{prefix}.{layer_idx}.mlp.down_proj.lora_B.weight"]

                    t_in = t_in.cuda().contiguous()
                    t_out = t_out.cuda().contiguous()
                    scale = float(hf_config["lora_alpha"] / hf_config["r"])
                    t_out = t_out * scale
                    t_in = t_in.float().to(str_dtype_to_torch(dtype))
                    t_out = t_out.float().to(str_dtype_to_torch(dtype))
                    self._lora_weights_pointers_list[layer_idx][uid].update(
                        {lora_module: [t_in.data_ptr(),
                                       t_out.data_ptr()]})

                    assert t_in.shape[0] == int(hf_config["r"])
                    self._lora_uid_to_low_ranks[uid][layer_idx][
                        lora_module] = int(hf_config["r"])

                    # prevent torch free this buffer
                    self._lora_weights.append(t_in)
                    self._lora_weights.append(t_out)
                    self._lora_cpp_weights[uid].append(
                        torch.concatenate([t_in.flatten(),
                                           t_out.flatten()]))
                    self._lora_weight_config[uid].append(
                        np.array([
                            self.LORA_MODULE_IDS[lora_module], layer_idx,
                            int(hf_config['r'])
                        ],
                                 dtype=np.int32))

        del lora_model

    def load_from_hf_bart(self, component, model_dirs, model_config,
                          runtime_mapping):
        '''
        lora config of https://huggingface.co/sooolee/bart-large-cnn-samsum-lora
        {
            "base_model_name_or_path": "facebook/bart-large-cnn",
            "bias": "none",
            "fan_in_fan_out": false,
            "inference_mode": true,
            "init_lora_weights": true,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "modules_to_save": null,
            "peft_type": "LORA",
            "r": 8,
            "target_modules": [
                "q_proj",
                "v_proj"
            ],
            "task_type": "SEQ_2_SEQ_LM"
        }

        For encoder, the trtllm target_modules are
            ['attn_q', 'attn_v']

        For decoder, the trtllm target_modules are
            ['attn_q', 'cross_attn_q',
             'attn_v', 'cross_attn_v']

        keys in adapter_model.bin:
            base_model.model.model.encoder.layers.0.self_attn.v_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.encoder.layers.0.self_attn.v_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.encoder.layers.0.self_attn.q_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.encoder.layers.0.self_attn.q_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.encoder.layers.1.self_attn.v_proj.lora_A.weight torch.Size([8, 1024])
            ...
            base_model.model.model.encoder.layers.11.self_attn.q_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.decoder.layers.0.encoder_attn.v_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.decoder.layers.0.encoder_attn.v_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.decoder.layers.0.encoder_attn.q_proj.lora_B.weight torch.Size([1024, 8])
            base_model.model.model.decoder.layers.1.self_attn.v_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.decoder.layers.1.self_attn.v_proj.lora_B.weight torch.Size([1024, 8])
            ...
            base_model.model.model.decoder.layers.11.encoder_attn.q_proj.lora_A.weight torch.Size([8, 1024])
            base_model.model.model.decoder.layers.11.encoder_attn.q_proj.lora_B.weight torch.Size([1024, 8])
        '''
        tp_size = runtime_mapping.tp_size
        tp_rank = runtime_mapping.tp_rank

        lora_hf_configs = [{}]
        ranks = [0]
        uids = ["-1"]
        for i, model_dir in enumerate(model_dirs):
            with open(f"{model_dir}/adapter_config.json", 'r') as f:
                config = json.load(f)
                lora_hf_configs.append(config)
                ranks.append(config["r"])
                uids.append(str(i))
        new_model_dirs = [""] + model_dirs

        # Note: lora_target_modules are trtllm_modules
        # encoder: ['attn_q', 'attn_v']
        # decoder: ['attn_q', 'cross_attn_q', 'attn_v', 'cross_attn_v']
        lora_target_modules = model_config.lora_target_modules
        dtype = model_config.dtype

        # In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time.
        # However, BART lora modules only contain two of them, so we use zero tensor to fill the missing ones.
        missing_qkv_modules = []
        if any(x in lora_target_modules
               for x in ["attn_q", "attn_k", "attn_v"]):
            for lora_module in ["attn_q", "attn_k", "attn_v"]:
                if lora_module not in lora_target_modules:
                    missing_qkv_modules.append(lora_module)
        if any(x in lora_target_modules
               for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
            for lora_module in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]:
                if lora_module not in lora_target_modules:
                    missing_qkv_modules.append(lora_module)

        self._lora_weights_pointers_list = [
            {} for _ in range(model_config.num_layers)
        ]

        for uid, rank, model_dir, hf_config in zip(uids, ranks, new_model_dirs,
                                                   lora_hf_configs):
            if uid not in self._lora_cpp_weights:
                self._lora_cpp_weights[uid] = []
            if uid not in self._lora_weight_config:
                self._lora_weight_config[uid] = []

            if model_dir != "":
                lora_model = torch.load(f"{model_dir}/adapter_model.bin")
            else:
                lora_model = None

            self._lora_uid_to_low_ranks[uid] = []
            for layer_idx in range(model_config.num_layers):
                self._lora_weights_pointers_list[layer_idx].update({uid: {}})

                self._lora_uid_to_low_ranks[uid].append({})

                prefix = f"base_model.model.model.{component}.layers"

                for lora_module in (lora_target_modules + missing_qkv_modules):
                    # fill missing q / k / v weights with zero tensors
                    if lora_module in missing_qkv_modules:
                        if uid == "-1":
                            self._lora_uid_to_low_ranks[uid][layer_idx][
                                lora_module] = 0
                            continue
                        # not split
                        t_in = torch.zeros(rank, model_config.hidden_size)
                        # split by column
                        t_out = torch.zeros(model_config.hidden_size, rank)
                        assert t_out.shape[0] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[0] // tp_size,
                                            dim=0)[tp_rank].contiguous()
                    else:
                        if uid == "-1" or model_config.trtllm_modules_to_hf_modules[
                                lora_module] not in hf_config[
                                    "target_modules"]:  # BART: q_proj, v_proj
                            self._lora_uid_to_low_ranks[uid][layer_idx][
                                lora_module] = 0
                            continue

                        if lora_module == "attn_q" or lora_module == "attn_k" or lora_module == "attn_v":
                            name = f"{prefix}.{layer_idx}.{lora_module.replace('attn_', 'self_attn.')}_proj"
                            # not split
                            t_in = lora_model[f"{name}.lora_A.weight"]
                            # split by column
                            t_out = lora_model[f"{name}.lora_B.weight"]
                            assert t_out.shape[0] % tp_size == 0
                            t_out = torch.split(t_out,
                                                t_out.shape[0] // tp_size,
                                                dim=0)[tp_rank].contiguous()
                        elif lora_module == "cross_attn_q" or lora_module == "cross_attn_k" or lora_module == "cross_attn_v":
                            name = f"{prefix}.{layer_idx}.{lora_module.replace('cross_attn_', 'encoder_attn.')}_proj"
                            # not split
                            t_in = lora_model[f"{name}.lora_A.weight"]
                            # split by column
                            t_out = lora_model[f"{name}.lora_B.weight"]
                            assert t_out.shape[0] % tp_size == 0
                            t_out = torch.split(t_out,
                                                t_out.shape[0] // tp_size,
                                                dim=0)[tp_rank].contiguous()

                    t_in = t_in.cuda().contiguous()
                    t_out = t_out.cuda().contiguous()
                    scale = float(hf_config["lora_alpha"] / hf_config["r"])
                    t_out = t_out * scale
                    t_in = t_in.float().to(str_dtype_to_torch(dtype))
                    t_out = t_out.float().to(str_dtype_to_torch(dtype))
                    self._lora_weights_pointers_list[layer_idx][uid].update(
                        {lora_module: [t_in.data_ptr(),
                                       t_out.data_ptr()]})

                    assert t_in.shape[0] == int(hf_config["r"])
                    self._lora_uid_to_low_ranks[uid][layer_idx][
                        lora_module] = int(hf_config["r"])

                    # prevent torch free this buffer
                    self._lora_weights.append(t_in)
                    self._lora_weights.append(t_out)
                    self._lora_cpp_weights[uid].append(
                        torch.concatenate([t_in.flatten(),
                                           t_out.flatten()]))
                    self._lora_weight_config[uid].append(
                        np.array([
                            self.LORA_MODULE_IDS[lora_module], layer_idx,
                            int(hf_config['r'])
                        ],
                                 dtype=np.int32))

        del lora_model

    def save_lora_weights_to_bin(self, out_dir):

        def save_val(val, dir, key, tp_num=None, write_npy=False):
            ext = "npy" if write_npy else "bin"
            suffix = ext if tp_num is None else f"{tp_num}.{ext}"
            if write_npy:
                np.save(dir / f"model.{key}.{suffix}", val)
            else:
                val.tofile(dir / f"model.{key}.{suffix}")

        if isinstance(out_dir, str):
            out_dir_path = Path(out_dir)
        elif isinstance(out_dir, Path):
            out_dir_path = out_dir
        else:
            assert False
        for uid in self._lora_cpp_weights:
            if uid == '-1':
                continue

            all_weights = np.expand_dims(
                np.stack([
                    torch_to_numpy(w.flatten().contiguous())
                    for w in self._lora_cpp_weights[uid]
                ]), 0)
            all_configs = np.expand_dims(
                np.stack(self._lora_weight_config[uid]), 0)

            uid_path = out_dir_path / f"{uid}"
            uid_path.mkdir(parents=True, exist_ok=True)
            save_val(all_weights,
                     uid_path,
                     "lora_weights",
                     tp_num=None,
                     write_npy=True)
            save_val(all_configs,
                     uid_path,
                     "lora_config",
                     tp_num=None,
                     write_npy=True)

    def uid_to_key(self, uid: str):
        assert isinstance(uid, str)
        return self._lora_uid_to_key[uid]

    def uid_to_low_ranks(self, uid: str):
        assert isinstance(uid, str)
        return self._lora_uid_to_low_ranks[uid]

    @property
    def lora_weights(self):
        return self._lora_weights

    @property
    def lora_weights_pointers_list(self):
        return self._lora_weights_pointers_list
