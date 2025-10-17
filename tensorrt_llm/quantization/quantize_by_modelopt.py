# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Adapted from examples/quantization/hf_ptq.py
"""

import copy
import json
import os
import random
import sys
import time
from importlib.metadata import version

import numpy as np
import torch
from accelerate.hooks import remove_hook_from_module
from datasets import load_dataset
from modelopt.torch.utils import print_rank_0
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer)

from .._utils import release_gc, str_dtype_to_torch
from ..logger import logger
from ..mapping import Mapping
from .image_processing import MllamaImageProcessor
from .mode import QuantAlgo

EMPTY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "enable": False,
        },
        "*input_quantizer": {
            "enable": False
        },
        "*lm_head*": {
            "enable": False
        },
        "*output_layer*": {
            "enable": False
        },
        "default": {
            "enable": False
        },
    },
    "algorithm": "max",
}

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}

KV_QUANT_CFG_CHOICES = {
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
}


def quant_cfg_choices():
    import modelopt.torch.quantization as mtq
    QUANT_CFG_CHOICES = {
        "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        "int8_wo": EMPTY_CFG,
        "int4_wo": EMPTY_CFG,
        "full_prec": EMPTY_CFG,
    }
    if hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        QUANT_CFG_CHOICES["nvfp4"] = mtq.NVFP4_DEFAULT_CFG
    return QUANT_CFG_CHOICES


def model_type_is_enc_dec(model_type):
    return model_type in ["t5", "bart"]


MODEL_NAME_PATTERN_MAP = {
    "GPT2": "gpt2",
    "Xverse": "llama",
    "MllamaForConditionalGeneration": "mllama",
    "Llama": "llama",
    "MllamaForCausalLM": "mllama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "QWen": "qwen",
    "Qwen2VLForConditionalGeneration": "qwen2_vl",
    "RecurrentGemma": "recurrentgemma",
    "Gemma3": "gemma3",
    "Gemma2": "gemma2",
    "Gemma": "gemma",
    "MixtralForCausalLM": "llama",
    "NemotronForCausalLM": "nemotron",
    "GPTBigCodeForCausalLM": "gpt_bigcode",
    "ArcticForCausalLM": "llama",
    "PhiMoEForCausalLM": "phi3",
    "Phi3SmallForCausalLM": "phi3small",
    "Phi3ForCausalLM": "phi3",
    "Phi3VForCausalLM": "phi3",
    "Starcoder2ForCausalLM": "gptnext",
    "GPTBigCodeForCausalLM": "gptnext",
    "GLM": "glm",
    "Exaone": "exaone",
    "DeciLMForCausalLM": "deci",
    "DeepseekForCausalLM": "deepseek",
    "GraniteForCausalLM": "granite",
    "GraniteMoeForCausalLM": "granitemoe",
    "T5": "t5",
    "Bart": "bart"
}

MULTIMODAL_DATASETS = ['scienceqa', 'science_qa']


class _CustomDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach().requires_grad_(False)
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class EncDecModelWrapper(torch.nn.Module):

    def __init__(self, hf_model=None):
        super().__init__()
        self.hf_model = hf_model
        self.model_type = get_model_type(hf_model)

    def forward(self, **kwargs):
        self.hf_model.generate(**kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.hf_model, name)


def get_tokenizer(ckpt_path, max_seq_length=2048, model_type=None):
    logger.info(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_length,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if model_type and model_type == "qwen":
            # qwen use token id 151643 as pad and eos tokens
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        elif model_type and model_type == "qwen2_vl":
            # qwen use token id 151643 as pad and 151643 and 151645 as eos tokens
            tokenizer.eos_token = [
                tokenizer.convert_ids_to_tokens(151643),
                tokenizer.convert_ids_to_tokens(151645)
            ]
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        else:
            tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_processor(ckpt_path, max_seq_length=2048, model_type=None, device=None):
    logger.info(f"Initializing tokenizer from {ckpt_path}")
    processor = AutoProcessor.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_length,
        padding_side="left",
        trust_remote_code=True,
    )

    if processor.tokenizer.pad_token is None:
        if model_type and model_type == "qwen":
            # qwen use token id 151643 as pad and eos tokens
            processor.tokenizer.eos_token = processor.tokenizer.convert_ids_to_tokens(
                151643)
            processor.tokenizer.pad_token = processor.tokenizer.convert_ids_to_tokens(
                151643)
        else:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    assert processor.tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    if model_type == 'mllama':
        processor = MllamaImageProcessor(processor, device)
    return processor


def _get_vila_model(model_dir):
    sys.path.append(model_dir + "/../VILA")
    from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_dir,
        device_map='auto',
        trust_remote_code=True,
    )
    return model.llm


def get_hf_config(ckpt_path):
    if "mpt" in ckpt_path:
        # MPT-7B cannot get initialized from AutoConfig
        from transformers import MptConfig
        return MptConfig.from_pretrained(ckpt_path)
    else:
        return AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)


def _get_llava_qwen_model(model_dir, dtype, device):
    if "hf" in model_dir:
        from transformers import LlavaOnevisionForConditionalGeneration
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, device_map=device)
        model = model.language_model
    else:
        from llava.model.builder import load_pretrained_model
        _, model, _, _ = load_pretrained_model(model_dir,
                                               None,
                                               'llava_qwen',
                                               torch_dtype=dtype,
                                               device_map=device)
    return model


def get_model(ckpt_path: str,
              dtype: str = 'bfloat16',
              device: str = 'cuda',
              device_map: str = "auto"):
    logger.info(f"Initializing model from {ckpt_path}")
    # Note: VILA model is not in public HF model zoo yet. We need to explicitly import from the git repo
    hf_config = get_hf_config(ckpt_path)
    torch_dtype = str_dtype_to_torch(dtype)

    model_cls = AutoModelForCausalLM
    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration
        model_cls = LlavaForConditionalGeneration
    elif hf_config.model_type == "mpt":
        from transformers import MptForCausalLM
        model_cls = MptForCausalLM
    elif hf_config.model_type == 'mllama':
        from transformers import MllamaForConditionalGeneration
        model_cls = MllamaForConditionalGeneration
    elif hf_config.model_type == 'qwen2_vl':
        from transformers import Qwen2VLForConditionalGeneration
        model_cls = Qwen2VLForConditionalGeneration

    if "vila" in ckpt_path:
        model = _get_vila_model(ckpt_path)
    elif "llava-onevision-qwen2" in ckpt_path:
        model = _get_llava_qwen_model(ckpt_path, dtype, device)
    elif hf_config.model_type == "glm":
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path,
                                                      device_map="cuda",
                                                      dtype=torch_dtype,
                                                      trust_remote_code=True)
    elif model_type_is_enc_dec(hf_config.model_type):
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path,
                                                      device_map=device,
                                                      dtype=torch_dtype,
                                                      trust_remote_code=True)
        model = EncDecModelWrapper(hf_model=model)
    else:
        model = model_cls.from_pretrained(
            ckpt_path,
            device_map=device_map if device != "cpu" else "cpu",
            dtype="auto",
            trust_remote_code=True)
        if hf_config.model_type in ["llava", "internvl_chat"]:
            model = model.language_model
        elif hf_config.model_type == "qwen2_vl":
            #WAR for Qwen2-VL because its lm_head is outside of LLM
            lm_head = model.lm_head
            model = model.model
            model.lm_head = lm_head

    model.eval()

    model_dtype = next(model.parameters()).dtype
    if torch_dtype != model_dtype:
        logger.info(
            f"[TensorRT-LLM][WARNING] The manually set model data type is {dtype}, "
            f"but the data type of the HuggingFace model is {model_dtype}.")

    return model


def get_model_type(model):
    if type(model).__name__ == "EncDecModelWrapper":
        return model.model_type
    if type(model).__name__ in MODEL_NAME_PATTERN_MAP:
        return MODEL_NAME_PATTERN_MAP[type(model).__name__]
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512,
                         device=None,
                         include_labels=False):
    logger.info("Loading calibration dataset")
    if dataset_name_or_dir == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train",
            trust_remote_code=True)
        dataset = dataset["text"][:calib_size]
    elif "scienceqa" in dataset_name_or_dir.lower(
    ) or "science_qa" in dataset_name_or_dir.lower():
        if os.path.isdir(dataset_name_or_dir):
            dataset = load_dataset(dataset_name_or_dir,
                                   split="train",
                                   trust_remote_code=True)
        else:
            dataset = load_dataset("derek-thomas/ScienceQA",
                                   split="train",
                                   trust_remote_code=True)
        dataset = dataset.select(range(calib_size))
    elif "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(
            dataset_name_or_dir,
            name="3.0.0",
            split="train",
            trust_remote_code=True,
        )
        dataset = dataset["article"][:calib_size]
    elif os.path.isdir(dataset_name_or_dir):
        logger.info(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir,
                               split="train",
                               trust_remote_code=True)
        dataset = dataset["text"][:calib_size]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    is_multimodal = False
    for dataset_name in MULTIMODAL_DATASETS:
        if dataset_name in dataset_name_or_dir:
            is_multimodal = True
    if is_multimodal:
        # Apply the preprocessing function to the dataset
        processed_dataset = dataset.map(tokenizer.preprocess_function,
                                        batched=False,
                                        remove_columns=dataset.column_names)

        # Create DataLoader with the custom collate function
        calib_dataloader = DataLoader(processed_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=tokenizer.collate_function)
    else:
        batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                    return_tensors="pt",
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=block_size)
        if device:
            batch_encoded = batch_encoded.to(device)

        if include_labels:
            # Labels are needed when backward is called in the model.
            # The labels should be a shifted version of the input_ids.
            # However, we should not shift the input_ids here since the labels are shifted by
            # Huggingface models during loss calculation as shown here -
            # https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/llama/modeling_llama.py#L1093-L1095
            batch_encoded["labels"] = torch.where(
                batch_encoded["attention_mask"] > 0.5,
                batch_encoded["input_ids"], -100)
            batch_encoded = _CustomDataset(batch_encoded)
        else:
            # For backward compatibility, if labels are not needed, we only return input_ids.
            batch_encoded = _CustomDataset(
                {"input_ids": batch_encoded["input_ids"]})

        calib_dataloader = DataLoader(batch_encoded,
                                      batch_size=batch_size,
                                      shuffle=False)

    return calib_dataloader


def quantize_model(model, quant_cfg, calib_dataloader, batch_size, qformat,
                   auto_quantize_bits):
    import modelopt.torch.quantization as mtq

    # NOTE: for ModelOpt v0.19 release
    # calibrate_loop = dataset_utils.create_forward_loop(
    #     calib_dataloader, dataloader=calib_dataloader)

    def calibrate_loop():
        if calib_dataloader is None:
            return
        with torch.no_grad():
            low_mem_mode = False
            for idx, data in enumerate(calib_dataloader):
                logger.debug(f"Calibrating batch {idx}")
                batch_size = data[list(data.keys())[0]].shape[0]
                if batch_size == 1:
                    model(**data)
                elif not low_mem_mode:
                    # Try running the forward once.
                    # If output memory, we try running inference with split input tensors
                    try:
                        model(**data)
                    except torch.OutOfMemoryError:
                        print(
                            "Warning: torch.OutOfMemoryError detected, try reducing the batch size..."
                        )
                        low_mem_mode = True

                if low_mem_mode:
                    split_data_1 = {
                        key: data[key][:batch_size // 2, ...]
                        for key in data
                    }
                    model(**split_data_1)

                    split_data_2 = {
                        key: data[key][batch_size // 2:, ...]
                        for key in data
                    }
                    model(**split_data_2)

    QUANT_CFG_CHOICES = {
        "int8": "INT8_DEFAULT_CFG",
        "int8_sq": "INT8_SMOOTHQUANT_CFG",
        "fp8": "FP8_DEFAULT_CFG",
        "fp8_pc_pt": "FP8_PER_CHANNEL_PER_TOKEN_CFG",
        "int4_awq": "INT4_AWQ_CFG",
        "w4a8_awq": "W4A8_AWQ_BETA_CFG",
    }

    logger.info("Starting quantization...")
    start_time = time.time()
    if auto_quantize_bits:
        logger.info("Starting mixed precision quantization...")

        from packaging import version as v
        opt_kwargs = {}
        modelopt_version = version('nvidia-modelopt')
        if v.parse(modelopt_version) > v.parse("0.21"):
            opt_kwargs['disabled_layers'] = ["*lm_head*"]

        model, search_history = mtq.auto_quantize(
            model,
            data_loader=calib_dataloader,
            loss_func=lambda output, batch: output.loss,
            constraints={"effective_bits": auto_quantize_bits},
            forward_step=lambda model, batch: model(**batch),
            quantization_formats=[
                QUANT_CFG_CHOICES[item] for item in qformat.split(",")
            ] + [None],
            num_calib_steps=len(calib_dataloader),
            num_score_steps=min(
                len(calib_dataloader), 128 // batch_size
            ),  # Limit the number of score steps to avoid long calibration time
            verbose=True,
            **opt_kwargs)
        mtq.print_quant_summary(model)

        # We need to explicitly calibrate for kv cache quantization
        enable_kv_cache_quantization = "int8" not in qformat
        if enable_kv_cache_quantization:
            mtq.set_quantizer_by_cfg(
                model,
                quant_cfg={
                    "*output_quantizer": {
                        "num_bits": (4, 3),
                        "axis": None,
                        "enable": True
                    }
                },
            )
            # Lets calibrate only the output quantizer this time. Let's disable all other quantizers.
            with mtq.set_quantizer_by_cfg_context(model, {
                    "*": {
                        "enable": False
                    },
                    "*output_quantizer": {
                        "enable": True
                    }
            }):
                mtq.calibrate(model,
                              algorithm="max",
                              forward_loop=calibrate_loop)
    else:
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    logger.info(
        "Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                               start_time))
    return model


def combine_medusa_weight(tp_size, pp_size, base_model_output_dir,
                          num_medusa_heads, num_medusa_layers, max_draft_len,
                          medusa_hidden_act, medusa_model_dir,
                          quant_medusa_head):

    with open(f"{medusa_model_dir}/config.json", "r") as fp:
        medusa_config = json.load(fp)

    num_medusa_heads_from_config = medusa_config.get('medusa_num_heads',
                                                     num_medusa_heads)
    num_medusa_layers = medusa_config.get('medusa_num_layers',
                                          num_medusa_layers)
    if num_medusa_heads is None:
        num_medusa_heads = num_medusa_heads_from_config

    assert max_draft_len > 0, "should have max_draft_len > 0"

    world_size = tp_size * pp_size
    # Process for each rank
    for rank in range(world_size):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=tp_size,
                          pp_size=pp_size)
        # 1. Load medusa weight for each rank
        from tensorrt_llm.models.medusa.weight import load_medusa_hf
        medusa_weights = load_medusa_hf(medusa_path=medusa_model_dir,
                                        num_medusa_heads=num_medusa_heads,
                                        num_medusa_layers=num_medusa_layers,
                                        mapping=mapping,
                                        dtype="float16")
        # 2. Load base model safetensors (after quant)
        base_model_weights = load_file(
            f"{base_model_output_dir}/rank{rank}.safetensors")

        # 3. Combine and save weight
        base_model_weights.update(medusa_weights)
        save_file(base_model_weights,
                  f"{base_model_output_dir}/rank{rank}.safetensors")

    # 4. Add medusa config into config.json
    with open(f"{base_model_output_dir}/config.json", 'r') as f:
        base_model_config = json.load(f)
        f.close()

    with open(f"{base_model_output_dir}/config.json", 'w') as f:
        base_model_config['architecture'] = "MedusaForCausalLM"
        base_model_config['quantization']['exclude_modules'] = [
            'lm_head',
            '*router',
            '*vocab_embedding',
            '*position_embedding',
            '*block_embedding',
        ]
        if not quant_medusa_head:
            base_model_config['quantization']['exclude_modules'].append(
                '*medusa_heads*')

        base_model_config['max_draft_len'] = max_draft_len
        base_model_config['num_medusa_heads'] = num_medusa_heads
        base_model_config['num_medusa_layers'] = num_medusa_layers
        json.dump(base_model_config, f, indent=4)

    torch.cuda.empty_cache()
    logger.info("Combine medusa heads' weight, done.")


def quantize_and_export(*,
                        model_dir,
                        device,
                        calib_dataset,
                        dtype,
                        qformat,
                        kv_cache_dtype,
                        calib_size,
                        batch_size,
                        calib_max_seq_length,
                        awq_block_size,
                        output_dir,
                        tp_size,
                        pp_size,
                        cp_size,
                        seed,
                        tokenizer_max_seq_length,
                        num_medusa_heads=None,
                        num_medusa_layers=None,
                        max_draft_len=None,
                        medusa_hidden_act=None,
                        medusa_model_dir=None,
                        quant_medusa_head=None,
                        auto_quantize_bits=None,
                        device_map="auto",
                        quantize_lm_head=False):
    '''
        Load model from the model_dir, call Modelopt to quantize the model, and then export
        the quantized model as TRT-LLM checkpoint
    '''
    try:
        import modelopt  # noqa
    except ImportError as e:
        logger.error(
            "Failed to import modelopt, pls check the Modelopt installation. Currently it is known to be unsupported on Windows OS"
        )
        raise e

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint

    from tensorrt_llm.models.convert_utils import infer_dtype

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(seed)
    np.random.seed(seed)

    # Check that only one quantization format is provided for non auto_quant case
    if not auto_quantize_bits:
        assert (len(qformat.split(",")) == 1
                ), "Quantization supports only one quantization format."

    hf_config = get_hf_config(model_dir)
    dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

    model = get_model(model_dir, dtype, device=device, device_map=device_map)
    model_type = get_model_type(model)
    is_enc_dec = model_type_is_enc_dec(model_type)
    if "vila" in model_dir:
        tokenizer = get_tokenizer(model_dir + "/llm",
                                  max_seq_length=tokenizer_max_seq_length,
                                  model_type=model_type)
    elif model_type == "mllama":
        tokenizer = get_processor(model_dir,
                                  max_seq_length=tokenizer_max_seq_length,
                                  model_type=model_type,
                                  device=device)
    else:
        tokenizer = get_tokenizer(model_dir,
                                  max_seq_length=tokenizer_max_seq_length,
                                  model_type=model_type)

    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        logger.info(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                logger.info(
                    f"AWQ calibration could take longer with calib_size = {calib_size}, Using"
                    " calib_size=32 instead")
                calib_size = 32
            logger.info(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument --batch_size <batch_size> to the command line.\n"
            )

        quant_cfg = None
        if not auto_quantize_bits:
            if qformat in quant_cfg_choices():
                quant_cfg = quant_cfg_choices()[qformat]
            else:
                raise ValueError(f"Unsupported quantization format: {qformat}")

            if "awq" in qformat:
                quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
                weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                if awq_block_size:
                    weight_quantizer["block_sizes"][-1] = awq_block_size

                # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
                if "w4a8_awq" == qformat and model_type in ["gemma", "mpt"]:
                    quant_cfg["algorithm"] = {
                        "method": "awq_lite",
                        "alpha_step": 1
                    }

            if kv_cache_dtype is not None:
                if kv_cache_dtype == "fp8":
                    kv_cache_quant_cfg = getattr(
                        mtq, KV_QUANT_CFG_CHOICES[kv_cache_dtype])["quant_cfg"]
                    quant_cfg["quant_cfg"].update(kv_cache_quant_cfg)
                else:
                    quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

            # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
            if model_type == "gemma" and "int8_sq" in qformat:
                quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

            if qformat == 'fp8' and quantize_lm_head:
                print_rank_0("Quantizing lm_head layer")
                del quant_cfg["quant_cfg"]["*lm_head*"]

        calib_dataloader = get_calib_dataloader(
            dataset_name_or_dir=calib_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            calib_size=calib_size,
            block_size=calib_max_seq_length,
            device=model.device,
            include_labels=auto_quantize_bits is not None,
        )

        model = quantize_model(model, quant_cfg, calib_dataloader, batch_size,
                               qformat, auto_quantize_bits)

    with torch.inference_mode():
        if model_type is None:
            logger.info(
                f"Unknown model type {type(model).__name__}. Continue exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        architecture = type(model).__name__

        export_path = output_dir
        start_time = time.time()

        # Move meta tensor back to device before exporting.
        remove_hook_from_module(model, recurse=True)

        QUANT_ALGO = {
            "int8": "INT8",
            "int8_sq": "W8A8_SQ_PER_CHANNEL",
            "fp8": "FP8",
            "int4_awq": "W4A16_AWQ",
            "w4a8_awq": "W4A8_AWQ",
        }

        if model_type == 'mllama':
            model = model.language_model

        export_tensorrt_llm_checkpoint(
            model.hf_model if is_enc_dec else model,
            model_type,
            getattr(torch, dtype),
            export_dir=export_path,
            inference_tensor_parallel=tp_size,
            inference_pipeline_parallel=pp_size,
        )

        export_paths = []
        tensorrt_llm_configs = []
        if not is_enc_dec:
            with open(f"{export_path}/config.json", "r") as f:
                tensorrt_llm_config = json.load(f)
            tensorrt_llm_configs.append(tensorrt_llm_config)
            export_paths.append(export_path)
        else:
            for component in ["encoder", "decoder"]:
                with open(f"{export_path}/{component}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_configs.append(tensorrt_llm_config)
                export_paths.append(f"{export_path}/{component}")

        for export_path, tensorrt_llm_config in zip(export_paths,
                                                    tensorrt_llm_configs):

            tensorrt_llm_config["model_type"] = model_type
            if not is_enc_dec:
                tensorrt_llm_config["architecture"] = architecture

            # Workaround for wo quantization
            if qformat in ["int8_wo", "int4_wo", "full_prec"]:
                if qformat == "int8_wo":
                    tensorrt_llm_config["quantization"][
                        "quant_algo"] = QuantAlgo.W8A16
                elif qformat == "int4_wo":
                    tensorrt_llm_config["quantization"][
                        "quant_algo"] = QuantAlgo.W4A16
                else:
                    tensorrt_llm_config["quantization"]["quant_algo"] = None

            # HF uses rope_scaling while tensorrt_llm uses rotary_scaling
            if hasattr(model.config, "rope_scaling"
                       ) and "rotary_scaling" not in tensorrt_llm_config:
                tensorrt_llm_config["rotary_scaling"] = getattr(
                    model.config, "rope_scaling")
            with open(f"{export_path}/config.json", "w") as f:
                json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for Modelopt 0.9.x fp8_kv_cache knob issue
            if qformat in ['fp8', 'nvfp4'] and kv_cache_dtype is None:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config["quantization"][
                    "kv_cache_quant_algo"] = None
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for qwen version
            if model_type == 'qwen' or model_type == 'qwen2_vl':
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                qwen_config = AutoConfig.from_pretrained(model_dir,
                                                         trust_remote_code=True)
                try:
                    from transformers import LlavaOnevisionConfig
                    if isinstance(qwen_config, LlavaOnevisionConfig):
                        qwen_config = qwen_config.text_config
                except:
                    pass
                tensorrt_llm_config["qwen_type"] = qwen_config.model_type
                if qwen_config.model_type == "qwen2":
                    tensorrt_llm_config[
                        "norm_epsilon"] = qwen_config.rms_norm_eps
                    tensorrt_llm_config["rotary_base"] = qwen_config.rope_theta
                tensorrt_llm_config[
                    "intermediate_size"] = qwen_config.intermediate_size
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Set rotary parameters correctly for chatglm.
            if model_type == 'chatglm':
                rotary_base = 10000.0
                rotary_embedding_scaling = None
                chatglm_config = AutoConfig.from_pretrained(
                    model_dir, trust_remote_code=True)
                chatglm_version = tensorrt_llm_config['chatglm_version']
                rope_ratio = tensorrt_llm_config.get('rope_ratio', 1.0)
                if chatglm_version == 'chatglm2':
                    if rope_ratio > 1:
                        rotary_embedding_scaling = {
                            'type': 'linear',
                            'factor': rope_ratio
                        }
                elif chatglm_version == 'chatglm3':
                    rotary_base *= rope_ratio

                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config['rotary_base'] = rotary_base
                tensorrt_llm_config['rotary_scaling'] = rotary_embedding_scaling
                tensorrt_llm_config['rotary_pct'] = 0.5
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # context parallel
            if cp_size > 1:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config["mapping"]["cp_size"] = cp_size
                tensorrt_llm_config["mapping"]["attn_tp_size"] = -1
                tensorrt_llm_config["mapping"]["attn_cp_size"] = -1
                tensorrt_llm_config["mapping"]["world_size"] *= cp_size
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            if model_type == 'gptnext':
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                if tensorrt_llm_config['max_position_embeddings'] is None:
                    tensorrt_llm_config['max_position_embeddings'] = getattr(
                        model.config, "n_positions", None)
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for combining medusa head
            # TODO: move these integration into modelopt to avoid redundant reading and writing
            if medusa_model_dir is not None:
                combine_medusa_weight(tp_size, pp_size, export_path,
                                      num_medusa_heads, num_medusa_layers,
                                      max_draft_len, medusa_hidden_act,
                                      medusa_model_dir, quant_medusa_head)

            # Workaround for mllama
            if model_type == 'mllama':
                from tensorrt_llm.models.mllama.config import MLLaMAConfig
                config = MLLaMAConfig.from_hugging_face(
                    model_dir,
                    dtype=dtype,
                )
                for key, value in config.to_dict().items():
                    if key not in tensorrt_llm_config:
                        tensorrt_llm_config[key] = value

                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

        end_time = time.time()
        logger.info(
            "Quantized model exported to {} \nTotal time used {:.2f} s.".format(
                export_path, end_time - start_time))

        # Need to delete the model and release memory explicitly;
        # otherwise torch may retain its GPU memory until a delayed GC running,
        # which reduces the available GPU memory for subsequent stages.
        del model
        release_gc()


def unwrap_model(model, module_instances=None):
    # Reference: https://github.com/NVIDIA/Megatron-LM/blob/core_r0.8.0/megatron/training/utils.py
    from megatron.core import DistributedDataParallel as DDP
    from megatron.core.transformer.module import Float16Module

    if module_instances is None:
        module_instances = (DDP, Float16Module)

    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def get_nemo_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                              batch_size=64,
                              calib_size=512,
                              max_sequence_length=512):
    if dataset_name_or_dir == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train",
            trust_remote_code=True)
        text_column = "text"
    elif "wikitext" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir,
                               "wikitext-103-v1",
                               split="train",
                               trust_remote_code=True)
        text_column = "text"
    elif "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir,
                               name="3.0.0",
                               split="train",
                               trust_remote_code=True)
        text_column = "article"
    elif os.path.isdir(dataset_name_or_dir):
        logger.info(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir,
                               split="train",
                               trust_remote_code=True)
        text_column = "text"
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size:(i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


def quantize_nemo_and_export(*, nemo_ckpt_path, decoder_type, calib_dataset,
                             calib_tp_size, calib_pp_size, dtype, qformat,
                             kv_cache_dtype, calib_size, batch_size,
                             calib_max_seq_length, awq_block_size, output_dir,
                             tp_size, pp_size, cp_size, seed):
    try:
        import modelopt  # noqa
    except ImportError as e:
        logger.error(
            "Failed to import modelopt, pls check the modelopt installation. Currently it is known to be unsupported on Windows OS"
        )
        raise e

    import modelopt.torch.quantization as mtq
    from megatron.core import parallel_state
    from megatron.core.transformer.module import Float16Module
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import \
        MegatronGPTModel
    from nemo.collections.nlp.modules.common.text_generation_strategy import \
        GPTModelTextGenerationStrategy
    from nemo.collections.nlp.parts.nlp_overrides import (
        NLPDDPStrategy, NLPSaveRestoreConnector)
    from nemo.utils.model_utils import load_config, save_artifacts
    from omegaconf.omegaconf import open_dict
    from pytorch_lightning.trainer.trainer import Trainer

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference.")

    random.seed(seed)
    np.random.seed(seed)

    model_cfg = load_config(nemo_ckpt_path)

    # dtype is used for non-quantized layers
    supported_dtype = ["auto", "float16", "bfloat16"]
    assert dtype in supported_dtype, f"{dtype} not supported. Supported dtypes are {supported_dtype}"

    if dtype == 'auto':
        dtype = model_cfg.get('precision', None)
        if dtype is None:
            dtype = 'float16'
        elif 'bf16' in dtype or 'bfloat16' in dtype:
            dtype = 'bfloat16'
        else:
            dtype = 'float16'
        logger.info(f"Specified dtype 'auto'; inferred dtype {dtype!r}.")
    torch_dtype = getattr(torch, dtype)

    with open_dict(model_cfg):
        model_cfg.activations_checkpoint_method = None
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.tensor_model_parallel_size = calib_tp_size
        model_cfg.pipeline_model_parallel_size = calib_pp_size
        model_cfg.sequence_parallel = False
        # Only custom modelopt spec is supported for PTQ: this custom spec is largely based on local Megatron-LM
        # layer definitions to avoid Transformer Engine implementations that are currently not supported.
        model_cfg.name = "modelopt"

    # trainer required for restoring model parallel models
    trainer_config = {
        'devices': calib_tp_size * calib_pp_size,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'logger': False,
        'precision': model_cfg.precision,
        'enable_checkpointing': False,
    }
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)
    connector = NLPSaveRestoreConnector()

    model = MegatronGPTModel.restore_from(
        restore_path=nemo_ckpt_path,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=connector,
    )
    model.freeze()

    print_rank_0(model)
    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    inference_config = {
        'greedy': False,
        'top_k': 0,
        'top_p': 0.9,
        'temperature': 1.0,
        'add_BOS': True,
        'tokens_to_generate': 30,
        'all_probs': False,
        'repetition_penalty': 1.2,
        'min_tokens_to_generate': 0,
        'compute_logprob': False,
        'batch_size': batch_size,
        'max_context_length': calib_max_seq_length,
        'strategy': GPTModelTextGenerationStrategy(model),
    }
    model.set_inference_config(inference_config)

    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        print_rank_0(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                print_rank_0(
                    "AWQ calibration could take longer with calib_size ="
                    f" {calib_size}, Using calib_size=32 instead")
                calib_size = 32
            print_rank_0(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument inference.batch_size=<batch_size> to the command"
                " line.\n")

        dataloader = get_nemo_calib_dataloader(
            dataset_name_or_dir=calib_dataset,
            batch_size=batch_size,
            calib_size=calib_size,
            max_sequence_length=calib_max_seq_length,
        )

        # =================== Start Quantization ====================
        if qformat in quant_cfg_choices():
            quant_cfg = quant_cfg_choices()[qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {qformat}")

        if "awq" in qformat:
            quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
            weight_quantizer = quant_cfg["quant_cfg"][
                "*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = awq_block_size

        if kv_cache_dtype is not None:
            if kv_cache_dtype == "fp8":
                for value in KV_CACHE_CFG.values():
                    value.update({"num_bits": (4, 3)})  # type: ignore
            quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

        print_rank_0(quant_cfg)

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we use int8 kv cache.
        # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for nemotron.
        # quant_cfg["quant_cfg"]["*output_quantizer"] = {  # type: ignore[index]
        #     "num_bits": 8 if args.qformat == "int8_sq" else (4, 3),
        #     "axis": None,
        #     "enable": args.decoder_type != "gptnext",
        # }

        dataloader = [data for data in dataloader]

        def forward_loop(model):
            for i, batch in enumerate(dataloader):
                print_rank_0(f"Calibrating batch {i}")
                model.predict_step(batch, i)

        start_time = time.time()
        model = mtq.quantize(model, quant_cfg,
                             forward_loop)  # type: ignore[arg-type]
        end_time = time.time()
        tot_time = end_time - start_time
        tput = calib_size / tot_time
        print_rank_0(
            f"Quantization done. Total time used {tot_time}s. Throughput {tput} samples/s"
        )
        # =================== End Quantization ======================

        if decoder_type == "gptnext":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            maxbound = 0
            if qformat == "fp8":
                maxbound = 448
            elif qformat == "int8_sq":
                maxbound = 127
            model = mtq.postprocess_amax(
                model, "*input_quantizer",
                lambda amax: torch.clamp(amax, min=0.01 * maxbound))

        if torch.distributed.get_rank() == 0:
            mtq.print_quant_summary(model)

    if model_cfg.megatron_amp_O2:
        model.model = unwrap_model(model.model, Float16Module)

    start_time = time.time()
    export_tensorrt_llm_checkpoint(
        model,
        decoder_type,
        torch_dtype,
        export_dir=output_dir,
        inference_tensor_parallel=tp_size,
        inference_pipeline_parallel=pp_size,
    )

    # context parallel
    if cp_size > 1:
        with open(f"{export_path}/config.json", "r") as f:
            tensorrt_llm_config = json.load(f)
        tensorrt_llm_config["mapping"]["cp_size"] = cp_size
        tensorrt_llm_config["mapping"]["world_size"] *= cp_size
        with open(f"{export_path}/config.json", "w") as f:
            json.dump(tensorrt_llm_config, f, indent=4)

    end_time = time.time()
    print_rank_0(
        f"Model config exported to: {output_dir}. Total time used {end_time - start_time}s"
    )
    if torch.distributed.get_rank() == 0:
        save_artifacts(model, output_dir, use_abspath=True)

    # Need to delete the model and release memory explicitly;
    # otherwise torch may retain its GPU memory until a delayed GC running,
    # which reduces the available GPU memory for subsequent stages.
    del model
    release_gc()
