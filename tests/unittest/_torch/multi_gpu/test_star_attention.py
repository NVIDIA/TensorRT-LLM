import json
import os

import modelopt.torch.quantization as mtq
import pytest
import torch
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root

from tensorrt_llm import SamplingParams, logger
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from tensorrt_llm.quantization.quantize_by_modelopt import get_calib_dataloader

MAX_SEQ_LEN = 4096 + 1024


def is_model_on_gpu(model) -> bool:
    """Check if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def get_model(ckpt_path, device="cuda"):
    """Load and return a model from the checkpoint path."""
    logger.info(f"Initializing model from {ckpt_path}")

    # Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = {"torch_dtype": "auto"}

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map=device,
        **model_kwargs,
        trust_remote_code=True,
    )
    model.eval()

    return model


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, model_type=None):
    """Load and return a tokenizer from the checkpoint path."""
    logger.info(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def calibrate_loop(model, calib_dataloader):
    """Calibrate the model with the calibration dataloader."""
    if calib_dataloader is None:
        return

    for idx, data in enumerate(calib_dataloader):
        logger.debug(f"Calibrating batch {idx}")
        data = data["input_ids"].to(model.device)
        model(data)


# Come from internal/examples/llm_ptq/hf_model_export.py in modelopt repo
def quantize(model_path,
             output_dir,
             quant_format,
             enable_quant_kv_cache,
             calib_samples=512):
    model = get_model(model_path, device="cuda")
    tokenizer = get_tokenizer(model_path)

    device = model.device if not hasattr(model, "model") else model.model.device

    # Calibrate the model
    calib_dataloader = get_calib_dataloader(
        dataset_name_or_dir=str(llm_models_root() / "datasets/cnn_dailymail"),
        tokenizer=tokenizer,
        batch_size=1,
        calib_size=calib_samples,
        device=device,
    )

    if quant_format == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    elif quant_format == "fp8_pc":
        quant_cfg = mtq.FP8_PER_CHANNEL_CFG
    elif quant_format == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG
    elif quant_format == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    quant_cfg["quant_cfg"]["*output_quantizer"] = {  # type: ignore[index]
        "num_bits": (4, 3),
        "axis": None,
        "enable": enable_quant_kv_cache,
    }

    model = mtq.quantize(
        model,
        quant_cfg,
        forward_loop=lambda: calibrate_loop(model, calib_dataloader),
    )
    model.to("cpu")

    # Post-process: export the quantized checkpoint with fused and packed weights, and save the quantization config
    post_state_dict, hf_quant_config, _ = export_hf_checkpoint(
        model,
        dtype=model.config.torch_dtype,
        export_dir=output_dir,
    )

    with open(f"{output_dir}/hf_quant_config.json", "w") as file:
        json.dump(hf_quant_config, file, indent=4)

    # Save the post-processed model and configs with the HF API
    model.save_pretrained(output_dir, state_dict=post_state_dict)

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)


@pytest.mark.parametrize("backend", ["pytorch"])
@pytest.mark.parametrize("model_name",
                         ["llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k"],
                         ids=["llama-3-8b-1048k"])
@pytest.mark.parametrize("quant", ["bf16", "fp8"])
@pytest.mark.parametrize("sp_size", [1, 2, 4], ids=["sp1", "sp2", "sp4"])
@pytest.mark.parametrize("sa_block_size", [256, 1024],
                         ids=["block1024", "block4096"])
@pytest.mark.parametrize("sa_anchor_size", [256, 1024],
                         ids=["anchor1024", "anchor4096"])
def test_model(backend, model_name, quant, sp_size, sa_block_size,
               sa_anchor_size):
    quant_configs = {
        "bf16":
        QuantConfig(),
        "fp8":
        QuantConfig(quant_algo=QuantAlgo.FP8),
        "fp8_kv_cache":
        QuantConfig(
            quant_algo=QuantAlgo.FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
        ),
    }
    quant_config = quant_configs[quant]
    if sp_size != 1:
        pytest.skip(f"skip multi gpu tests due to flashinfer's jitting mode")
    if torch.cuda.device_count() < sp_size:
        pytest.skip(f"Not enough GPUs available, need {sp_size} "
                    f"but only have {torch.cuda.device_count()}")
    if sa_anchor_size > sa_block_size:
        pytest.skip(
            f"Unsupported sa_anchor_size {sa_anchor_size} > sa_block_size {sa_block_size}"
        )

    if get_total_gpu_memory(0) < 32 * 1024**3:
        pytest.skip("Not enough GPU memory to run BF16 model")

    model_dir = str(llm_models_root() / model_name)
    cp_config = {
        "cp_type": "star_attention",
        "cp_anchor_size": sa_anchor_size,
        "block_size": sa_block_size
    }
    max_batch_size = 20
    max_output_tokens = 128
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.7)
    pytorch_backend_config = PyTorchConfig(
        attn_backend='FLASHINFER_STAR_ATTENTION')

    llm = LLM(model=model_dir,
              backend=backend,
              kv_cache_config=kv_cache_config,
              tensor_parallel_size=1,
              quant_config=quant_config,
              context_parallel_size=sp_size,
              cp_config=cp_config,
              pytorch_backend_config=pytorch_backend_config,
              max_batch_size=max_batch_size,
              max_input_len=MAX_SEQ_LEN - max_output_tokens,
              max_seq_len=MAX_SEQ_LEN,
              max_num_tokens=(sa_block_size + sa_anchor_size) * max_batch_size)

    contexts, queries, references = [], [], []
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    with open(f'{current_dir}/test_star_attention_input.jsonl', 'r') as f:
        for line in f:
            prompt = json.loads(line)
            contexts.append(prompt['input_context'])
            queries.append(prompt['input_query'])
            references.append(prompt['outputs'][0])
    with llm:
        outputs = llm.generate(
            contexts,
            queries=queries,
            use_tqdm=True,
            sampling_params=SamplingParams(
                max_tokens=max_output_tokens,
                add_special_tokens=False,
            ),
        )

    count = 0
    for ref, ret in zip(references, outputs):
        #print(f'reference = {ref}')
        #print(f'prediction = {ret.outputs[0].text}')
        if ref not in ret.outputs[0].text:
            print(f'reference {ref} is not in the output {ret.outputs[0].text}')
        else:
            count = count + 1
    acc = count / len(outputs)
    if acc < 1.0:
        assert False, 'accuracy test of star attention failed'


if __name__ == '__main__':
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 1, 256, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 1, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 1, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "fp8", 1, 256, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "fp8", 1, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "fp8", 1, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 2, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 2, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 2, 256, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 4, 1024, 256)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 4, 1024, 1024)
    test_model("pytorch", "llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k",
               "bf16", 4, 256, 256)
