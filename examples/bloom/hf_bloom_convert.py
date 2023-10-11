'''
Convert huggingface Bloom model. Use https://huggingface.co/bigscience/bloom as demo.
'''
import argparse
import configparser
import dataclasses
import os
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
from convert import split_and_save_weight
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers.models.bloom.modeling_bloom import BloomBlock

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "bloom"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None
    load_model_on_cpu: bool = False
    convert_model_on_cpu: bool = False

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="bloom",
            type=str,
            help="Specify Bloom variants to convert checkpoints correctly",
            choices=["bloom"])
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        parser.add_argument("--load-model-on-cpu", action="store_true")
        parser.add_argument("--convert-model-on-cpu", action="store_true")
        return ProgArgs(**vars(parser.parse_args(args)))


def reorder_torch_qkv_weight_or_bias(v, model, is_bias=False):
    """ Reorder the qkv weight.

    Note that the shape of the fused QKV weights in HF is different from the
    shape that TRT-LLM requires.
       HF: (num_heads x 3 x head_dim, hidden_size)
       TRT-LLM: (3 x num_heads x head_dim, hidden_size)
    This is unlike to the other models in HF e.g. GPT where they have the
    same shape with TRT-LLM, i.e., (3 x num_heads x head_dim, hidden_size). We reshape the qkv
        weight: (3 x num_heads x head_dim, hidden).
        bias  : (3 x num_heads x head_dim).
    """

    n_head = model.transformer.num_heads
    hidden_size = model.transformer.embed_dim
    head_dim = hidden_size // n_head

    # (3 x hidden, ...) view as (num_heads, 3, head_dim, ...)
    v = v.reshape(n_head, 3, head_dim, -1)
    # permute to (3, num_heads, head_dim, ...)
    v = v.permute((1, 0, 2, 3))
    # final shape: weight=(3 x hidden, hidden) or bias=(3 x hidden)
    if is_bias:
        return v.reshape(3 * hidden_size)
    return v.reshape(3 * hidden_size, hidden_size)


@torch.no_grad()
def smooth_bloom_model(model, scales, alpha, bloom_qkv_param, bloom_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, BloomBlock):
            continue

        # reorder qkv weight/bias and scales
        param = module.self_attention.query_key_value.weight
        param = reorder_torch_qkv_weight_or_bias(param, model, is_bias=False)

        layer_name = name + ".self_attention.query_key_value"
        act_range_qkv = scales.get(layer_name)
        # (n_head x 3 x head_dim) -> (3 x n_head x head_dim)
        act_range_qkv['w'] = reorder_torch_qkv_weight_or_bias(
            act_range_qkv['w'], model, is_bias=True)
        act_range_qkv['y'] = reorder_torch_qkv_weight_or_bias(
            act_range_qkv['y'], model, is_bias=True)
        scales[layer_name] = act_range_qkv

        # qkv_proj
        smoother = smooth_gemm(param, scales[layer_name]["x"],
                               module.input_layernorm.weight,
                               module.input_layernorm.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = param.abs().max(dim=1)[0]
        bloom_qkv_param[layer_name] = param

        # dense
        # enabled for better accuracy with perf overhead of quantiztion
        layer_name = name + ".self_attention.dense"
        smoother = smooth_gemm(module.self_attention.dense.weight,
                               scales[layer_name]["x"], None, None, alpha)
        bloom_smoother[layer_name] = smoother

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attention.dense.weight.abs().max(
            dim=1)[0]

        # fc1
        layer_name = name + ".mlp.dense_h_to_4h"
        smoother = smooth_gemm(module.mlp.dense_h_to_4h.weight,
                               scales[layer_name]["x"],
                               module.post_attention_layernorm.weight,
                               module.post_attention_layernorm.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.dense_h_to_4h.weight.abs().max(
            dim=1)[0]

        # fc2
        # enabled for better accuracy with perf overhead of quantiztion
        layer_name = name + ".mlp.dense_4h_to_h"
        smoother = smooth_gemm(module.mlp.dense_4h_to_h.weight,
                               scales[layer_name]["x"], None, None, alpha)
        bloom_smoother[layer_name] = smoother
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.dense_4h_to_h.weight.abs().max(
            dim=1)[0]


# Bloom uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = [
        "self_attention.query_key_value", "self_attention.dense",
        "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"
    ]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def bloom_to_trt_llm_name(orig_name):
    global_weights = {
        "transformer.word_embeddings.weight": "model.wpe",
        "transformer.word_embeddings_layernorm.bias":
        "model.word_embeddings_layernorm.bias",
        "transformer.word_embeddings_layernorm.weight":
        "model.word_embeddings_layernorm.weight",
        "transformer.ln_f.bias": "model.final_layernorm.bias",
        "transformer.ln_f.weight": "model.final_layernorm.weight",
        "lm_head.weight": "model.lm_head.weight"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        "transformer.input_layernorm.bias": "input_layernorm.bias",
        "transformer.input_layernorm.weight": "input_layernorm.weight",
        "transformer.self_attention.query_key_value.bias":
        "attention.query_key_value.bias",
        "transformer.self_attention.query_key_value.weight":
        "attention.query_key_value.weight",
        "transformer.self_attention.dense.bias": "attention.dense.bias",
        "transformer.self_attention.dense.weight": "attention.dense.weight",
        "transformer.post_attention_layernorm.bias":
        "post_attention_layernorm.bias",
        "transformer.post_attention_layernorm.weight":
        "post_attention_layernorm.weight",
        "transformer.mlp.dense_h_to_4h.bias": "mlp.dense_h_to_4h.bias",
        "transformer.mlp.dense_h_to_4h.weight": "mlp.dense_h_to_4h.weight",
        "transformer.mlp.dense_4h_to_h.bias": "mlp.dense_4h_to_h.bias",
        "transformer.mlp.dense_4h_to_h.weight": "mlp.dense_4h_to_h.weight",
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_bloom_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = True if args.model in ["santacoder", "starcoder"
                                              ] else False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = BloomForCausalLM.from_pretrained(args.in_file,
                                             torch_dtype="auto",
                                             device_map="auto",
                                             trust_remote_code=True)
    if args.load_model_on_cpu:
        model = model.cpu()
        torch.cuda.empty_cache()
    act_range = {}
    bloom_qkv_param = {}
    # smoother for inputs of self_attention.dense and mlp.dense_4h_to_h
    bloom_smoother = {}

    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation",
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            model, BloomTokenizerFast.from_pretrained(args.in_file), dataset)
        if args.smoothquant is not None:
            smooth_bloom_model(model, act_range, args.smoothquant,
                               bloom_qkv_param, bloom_smoother)

    config = configparser.ConfigParser()
    config["bloom"] = {}
    for key in vars(args):
        config["bloom"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["bloom"][k] = f"{v}"
    config["bloom"]["storage_dtype"] = args.storage_type
    config["bloom"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    global_trt_llm_weights = [
        "model.wpe", "model.word_embeddings_layernorm.bias",
        "model.word_embeddings_layernorm.weight", "model.final_layernorm.bias",
        "model.final_layernorm.weight", "model.lm_head.weight"
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        trt_llm_name = bloom_to_trt_llm_name(name)

        if args.convert_model_on_cpu:
            param = param.cpu()
        if name.replace(".weight", "") in bloom_smoother.keys():
            smoother = bloom_smoother[name.replace(".weight", "")]
            starmap_args.append(
                (0, saved_dir, infer_tp,
                 f"{trt_llm_name}.smoother".replace(".weight", ""),
                 smoother.to(torch.float32), torch.float32, None, {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": multi_query_mode,
                     "local_dim": None,
                 }))

        # reorder qkv weight and bias
        if "attention.query_key_value.weight" in trt_llm_name:
            if args.smoothquant is not None:
                param = bloom_qkv_param.get(name.replace(".weight", ""))
            else:
                param = reorder_torch_qkv_weight_or_bias(param,
                                                         model,
                                                         is_bias=False)
        if "attention.query_key_value.bias" in trt_llm_name:
            param = reorder_torch_qkv_weight_or_bias(param, model, is_bias=True)

        param = transpose_weights(name, param)

        if trt_llm_name in global_trt_llm_weights:
            torch_to_numpy(param.to(storage_type).cpu()).tofile(
                saved_dir / f"{trt_llm_name}.bin")
        else:
            # Needed by QKV projection weight split. With multi_query_mode one does not simply take
            # out_dim and divide it by 3 to get local_dim becuase out_dim = local_dim + 2 * head_size
            local_dim = model.transformer.h[
                0].attn.embed_dim if multi_query_mode else None
            starmap_args.append(
                (0, saved_dir, infer_tp, trt_llm_name, param.to(storage_type),
                 storage_type, act_range.get(name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": multi_query_mode,
                     "local_dim": local_dim
                 }))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    hf_bloom_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
