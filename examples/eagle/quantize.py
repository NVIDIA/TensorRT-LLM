import argparse
import json
import os
import random

import modelopt.torch.speculative as mtsp
import numpy as np
import torch
import torch.multiprocessing as mp

from tensorrt_llm._utils import release_gc, str_dtype_to_torch
from tensorrt_llm.quantization.quantize_by_modelopt import (
    KV_QUANT_CFG_CHOICES, get_calib_dataloader, get_model, get_model_type,
    get_tokenizer, quant_cfg_choices, quantize_model)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        help="Specify where the HuggingFace model is",
                        default=None)
    parser.add_argument("--eagle_model_dir",
                        help="Specify where the EAGLE checkpoint is",
                        default=None)
    parser.add_argument(
        "--eagle_num_layers",
        help="Specify the number of layers in the EAGLE checkpoint",
        type=int,
        default=1)
    parser.add_argument(
        '--device',
        help=
        "The device to run calibration; effective for HuggingFace model only.",
        default='cuda',
        choices=['cuda', 'cpu'])
    parser.add_argument(
        "--device_map",
        help="How to map the model on the devices",
        default="auto",
        choices=["auto", "sequential", "cpu", "gpu"],
    )
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="full_prec",
        choices=[
            "fp8",
        ],
    )
    parser.add_argument(
        "--seed",
        help="Seed the generate random numbers, the value will be used to call"
        "random.seed(value) and numpy.random.seed(value)",
        type=int,
        default=1234)
    parser.add_argument("--tokenizer_max_seq_length",
                        help="Max sequence length to init the tokenizers",
                        type=int,
                        default=2048)

    parser.add_argument("--batch_size",
                        help="Batch size for calibration.",
                        type=int,
                        default=1)
    parser.add_argument("--calib_size",
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--calib_max_seq_length",
                        help="Max sequence length for calibration",
                        type=int,
                        default=512)
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument("--kv_cache_dtype",
                        help="KV Cache dtype.",
                        default=None,
                        choices=["int8", "fp8", None])

    args = parser.parse_args()

    if args.model_dir is None or args.eagle_model_dir is None:
        raise ValueError(
            "One of source checkpoint (model_dir) must be specified")

    import modelopt  # noqa
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # read config.json from model_dir
    hf_config = json.load(open(os.path.join(args.model_dir, "config.json")))
    dtype = hf_config['torch_dtype']
    torch_dtype = str_dtype_to_torch(dtype)

    model = get_model(args.model_dir,
                      dtype,
                      device=args.device,
                      device_map=args.device_map)
    model_type = get_model_type(model)

    tokenizer = get_tokenizer(args.model_dir,
                              max_seq_length=args.tokenizer_max_seq_length,
                              model_type=model_type)
    quant_cfg = None
    if args.qformat in quant_cfg_choices():
        quant_cfg = quant_cfg_choices()[args.qformat]
    else:
        raise ValueError(f"Unsupported quantization format: {args.qformat}")

    if args.kv_cache_dtype is not None:
        if args.kv_cache_dtype == "fp8":
            kv_cache_quant_cfg = getattr(
                mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_dtype])["quant_cfg"]
            quant_cfg["quant_cfg"].update(kv_cache_quant_cfg)
        else:
            quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

    calib_dataloader = get_calib_dataloader(
        dataset_name_or_dir=args.calib_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        calib_size=args.calib_size,
        block_size=args.calib_max_seq_length,
        device=model.device,
    )

    last_device = list({param.device for param in model.parameters()})[-1]

    eagle_weights = torch.load(os.path.join(args.eagle_model_dir,
                                            "pytorch_model.bin"),
                               map_location=last_device)
    eagle_weights = {k: v.to(torch_dtype) for k, v in eagle_weights.items()}
    # Adjust the parameters depending on your eagle module weight
    eagle_config = {
        "eagle_num_layers": args.eagle_num_layers,
    }

    # load in your base model
    mtsp.convert(model, [("eagle", eagle_config)])

    for i in range(args.eagle_num_layers):
        # Replace the eagle weight in modelopt eagle model
        model.eagle_module.layers[
            i].self_attn.q_proj.weight = torch.nn.Parameter(
                eagle_weights[f"layers.{i}.self_attn.q_proj.weight"])
        model.eagle_module.layers[
            i].self_attn.k_proj.weight = torch.nn.Parameter(
                eagle_weights[f"layers.{i}.self_attn.k_proj.weight"])
        model.eagle_module.layers[
            i].self_attn.v_proj.weight = torch.nn.Parameter(
                eagle_weights[f"layers.{i}.self_attn.v_proj.weight"])
        model.eagle_module.layers[
            i].self_attn.o_proj.weight = torch.nn.Parameter(
                eagle_weights[f"layers.{i}.self_attn.o_proj.weight"])
        model.eagle_module.layers[i].mlp.gate_proj.weight = torch.nn.Parameter(
            eagle_weights[f"layers.{i}.mlp.gate_proj.weight"])
        model.eagle_module.layers[i].mlp.up_proj.weight = torch.nn.Parameter(
            eagle_weights[f"layers.{i}.mlp.up_proj.weight"])
        model.eagle_module.layers[i].mlp.down_proj.weight = torch.nn.Parameter(
            eagle_weights[f"layers.{i}.mlp.down_proj.weight"])
        model.eagle_module.layers[
            i].post_attention_layernorm.weight = torch.nn.Parameter(
                eagle_weights[f"layers.{i}.post_attention_layernorm.weight"])
    model.eagle_module.fc.weight = torch.nn.Parameter(
        eagle_weights[f"fc.weight"])
    if "fc.bias" in eagle_weights:
        model.eagle_module.fc.bias = torch.nn.Parameter(
            eagle_weights[f"fc.bias"])
    else:
        model.eagle_module.fc.bias = torch.nn.Parameter(
            torch.zeros(eagle_weights[f"fc.weight"].shape[0],
                        dtype=torch_dtype,
                        device=last_device))

    model = quantize_model(model, quant_cfg, calib_dataloader, args.batch_size,
                           args.qformat, None)

    with torch.inference_mode():
        export_hf_checkpoint(
            model,
            export_dir=args.output_dir,
        )

    del model
    release_gc()
