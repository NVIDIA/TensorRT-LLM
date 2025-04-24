# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import copy
import random
import time
import warnings

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.sparsity as mts
import numpy as np
import torch
from modelopt.torch.export import export_hf_checkpoint, get_model_type
from modelopt.torch.utils.dataset_utils import (create_forward_loop,
                                                get_dataset_dataloader,
                                                get_max_batch_size)
from modelopt.torch.utils.memory_monitor import launch_memory_monitor
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer, PreTrainedTokenizerFast)

RAND_SEED = 1234

QUANT_CFG_CHOICES = {
    "int8": "INT8_DEFAULT_CFG",
    "int8_sq": "INT8_SMOOTHQUANT_CFG",
    "fp8": "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",
    "nvfp4": "NVFP4_DEFAULT_CFG",
    "nvfp4_awq": "NVFP4_AWQ_LITE_CFG",
    "fp8_pb_wo": "FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG",
}

KV_QUANT_CFG_CHOICES = {
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
}

mto.enable_huggingface_checkpointing()


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def get_model(ckpt_path,
              device="cuda",
              gpu_mem_percentage=0.8,
              trust_remote_code=False):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = {"torch_dtype": "auto"}

    hf_config = AutoConfig.from_pretrained(ckpt_path,
                                           trust_remote_code=trust_remote_code)

    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_max_memory

    with init_empty_weights():
        # When computing the device_map, assuming half precision by default,
        # unless specified by the hf_config.
        torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
        model = AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code)

    max_memory = get_max_memory()
    inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

    on_cpu = "cpu" in inferred_device_map.values()

    if on_cpu:
        for device in max_memory.keys():
            if isinstance(device, int):
                max_memory[device] *= gpu_mem_percentage

        print(
            "Model does not fit to the GPU mem. "
            f"We apply the following memory limit for calibration: \n{max_memory}\n"
            "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
            "reduce the calibration `batch_size` manually.")
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map=device_map,
        **model_kwargs,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    if device == "cuda":
        if not is_model_on_gpu(model):
            print(
                "Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM"
            )

    return model


def auto_quantize(model,
                  qformat,
                  auto_quantize_bits,
                  calib_dataloader,
                  calibrate_loop,
                  batch_size=1):
    qformat_list = qformat.split(",")
    assert qformat_list, "No quantization formats provided"
    # Check if all provided quantization formats are supported
    assert all(
        qformat in
        ["fp8", "int4_awq", "nvfp4", "nvfp4_awq", "w4a8_awq", "fp8_pb_wo"]
        for qformat in qformat_list
    ), "One or more quantization formats provided are not supported for unified checkpoint export"

    model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": auto_quantize_bits},
        data_loader=calib_dataloader,
        forward_step=lambda model, batch: model(**batch),
        loss_func=lambda output, data: output.loss,
        quantization_formats=[
            QUANT_CFG_CHOICES[format] for format in qformat_list
        ] + [None],  # TRTLLM only support one quantization format or None
        num_calib_steps=len(calib_dataloader),
        num_score_steps=min(
            len(calib_dataloader), 128 // batch_size
        ),  # Limit the number of score steps to avoid long calibration time
        verbose=True,
        disabled_layers=["*lm_head*"],
    )

    # We need to explicitly calibrate for kv cache quantization
    enable_quant_kv_cache = args.kv_cache_qformat != "none"
    print(
        f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization"
    )
    if enable_quant_kv_cache:
        kv_cache_quant_cfg = getattr(
            mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"]

        mtq.set_quantizer_by_cfg(
            model,
            quant_cfg=kv_cache_quant_cfg,
        )
        # Lets calibrate only the quantizers for kv cache quantization this time. Let's disable all others.
        with mtq.set_quantizer_by_cfg_context(model, {
                "*": {
                    "enable": False
                },
                **kv_cache_quant_cfg
        }):
            mtq.calibrate(model, algorithm="max", forward_loop=calibrate_loop)
    return model


def quantize_model(model, quant_cfg, args, calib_dataloader=None):
    # The calibration loop for the model can be setup using the modelopt API.
    #
    # Example usage:
    # from modelopt.torch.utils.dataset_utils import create_forward_loop
    # model = ...  # Initialize the model
    # tokenizer = ...  # Initialize the tokenizer
    # quant_cfg = ...  # Setup quantization configuration
    # forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    # mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    # The calibrate_loop is a custom defined method to run the model with the input data.
    # The basic version looks like:
    #
    # def calibrate_loop(model, dataloader):
    #     for data in dataloader:
    #         model(**data)
    #
    # We also provided a util method to generate the forward_loop with additional error handlings.

    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

    assert not (
        args.auto_quantize_bits and args.inference_pipeline_parallel > 1
    ), "Auto Quantization is not supported for pipeline parallel size > 1"

    print("Starting quantization...")
    start_time = time.time()
    if args.auto_quantize_bits:
        model = auto_quantize(
            model,
            args.qformat,
            args.auto_quantize_bits,
            calib_dataloader,
            calibrate_loop,
            args.batch_size,
        )
    else:
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model


def get_tokenizer(ckpt_path, trust_remote_code=False, **kwargs):
    print(f"Initializing tokenizer from {ckpt_path}")

    if ckpt_path.endswith(".yaml"):
        # Model Optimizer modification
        # For Nemo models, tokenizer is instantiated based on its config
        from modelopt.deploy.llm.nemo_utils import get_nemo_tokenizer

        tokenizer = get_nemo_tokenizer(ckpt_path)

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=trust_remote_code, **kwargs)

        # can't set attribute 'pad_token' for "<unk>"
        # We skip this step for Nemo models
        if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    # launch a memory monitor to read the currently used GPU memory.
    launch_memory_monitor()

    # Check that only one quantization format is provided for non auto_quant case
    if not args.auto_quantize_bits:
        assert (len(args.qformat.split(",")) == 1
                ), "Quantization supports only one quantization format."

    # Check arguments for unified_hf export format and set to default if unsupported arguments are provided
    assert (
        args.sparsity_fmt == "dense"
    ), f"Sparsity format {args.sparsity_fmt} not supported by unified export api."

    if not args.auto_quantize_bits:
        assert args.qformat in [
            "int4_awq",
            "fp8",
            "nvfp4",
            "nvfp4_awq",
            "w4a8_awq",
            "fp8_pb_wo",
        ] or args.kv_cache_qformat in [
            "fp8",
            "nvfp4",
            "none",
            "",
        ], f"Quantization format {args.qformat} not supported for HF export path"

    model = get_model(args.pyt_ckpt_path,
                      args.device,
                      trust_remote_code=args.trust_remote_code)
    model_type = get_model_type(model)

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device
    tokenizer = None
    args.dataset = "cnn_dailymail"
    tokenizer = get_tokenizer(args.pyt_ckpt_path,
                              trust_remote_code=args.trust_remote_code)
    default_padding_side = tokenizer.padding_side
    # Left padding usually provides better calibration result.
    tokenizer.padding_side = "left"

    if args.sparsity_fmt != "dense":
        if args.batch_size == 0:
            # Sparse algorithm takes more GPU memory so we reduce the batch_size by 4.
            args.batch_size = max(get_max_batch_size(model) // 4, 1)
            if args.batch_size > args.calib_size:
                args.batch_size = args.calib_size

        print(f"Use calib batch_size {args.batch_size}")

        # Different calibration datasets are also available, e.g., "pile" and "wikipedia"
        # Please also check the docstring for the datasets available
        assert tokenizer is not None and isinstance(
            tokenizer,
            (PreTrainedTokenizer,
             PreTrainedTokenizerFast)), "The PreTrainedTokenizer must be set"
        calib_dataloader = get_dataset_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            device=device,
        )
        model = mts.sparsify(
            model,
            args.sparsity_fmt,
            config={
                "data_loader": calib_dataloader,
                "collect_func": lambda x: x
            },
        )
        mts.export(model)

    if args.auto_quantize_bits or args.qformat in QUANT_CFG_CHOICES:

        if "awq" in args.qformat:
            print(
                "\n####\nAWQ calibration could take longer than other calibration methods. "
                "Consider reducing calib_size to reduce calibration time.\n####\n"
            )

        if args.batch_size == 0:
            # TODO: Enable auto-batch size calculation for AutoQuantize
            assert (
                args.auto_quantize_bits is None
            ), "AutoQuantize requires batch_size to be specified, please specify batch_size."
            # Calibration/sparsification will actually take much more memory than regular inference
            # due to intermediate tensors for fake quantization. Setting sample_memory_usage_ratio
            # to 2 to avoid OOM for AWQ/SmoothQuant fake quantization as it will take more memory than inference.
            sample_memory_usage_ratio = (2 if "awq" in args.qformat
                                         or "sq" in args.qformat else 1.1)
            # Whisper model expects mel-spectrogram input features of length 3000
            # Whisper model needs input of shape (batch_size, num_mel_bins, 3000)
            # As the encoder of Whisper doesn't have embedding layer, input dtype has to be float
            # For non-Whisper models (language models), sample_input will be set up inside get_max_batch_size()
            if model_type == "whisper":
                max_sample_length = 3000
                num_mel_bins = model.config.num_mel_bins
                sample_input_single_batch = (
                    torch.ones([1, num_mel_bins, max_sample_length],
                               dtype=torch.float32).to(model.device) * 100)
            else:
                sample_input_single_batch = None
            args.batch_size = get_max_batch_size(
                model,
                sample_memory_usage_ratio=sample_memory_usage_ratio,
                sample_input_single_batch=sample_input_single_batch,
            )
            if args.batch_size > args.calib_size:
                args.batch_size = args.calib_size

        print(f"Use calib batch_size {args.batch_size}")

        calib_dataloader = None

        assert tokenizer is not None and isinstance(
            tokenizer,
            (PreTrainedTokenizer,
             PreTrainedTokenizerFast)), "The PreTrainedTokenizer must be set"
        calib_dataloader = get_dataset_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            device=device,
            include_labels=args.auto_quantize_bits is not None,
        )

        quant_cfg = {}
        if not args.auto_quantize_bits:
            assert (
                args.qformat in QUANT_CFG_CHOICES
            ), f"Unsupported quantization format: {args.qformat} with {args.kv_cache_qformat} KV cache"

            quant_cfg = getattr(mtq, QUANT_CFG_CHOICES[args.qformat])

            if "awq" in args.qformat:
                quant_cfg = copy.deepcopy(
                    getattr(mtq, QUANT_CFG_CHOICES[args.qformat]))
                weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                # If awq_block_size argument is provided, update weight_quantizer
                if args.awq_block_size:
                    weight_quantizer["block_sizes"][-1] = args.awq_block_size

                # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
                if "w4a8_awq" == args.qformat and model_type in [
                        "gemma", "mpt"
                ]:
                    quant_cfg["algorithm"] = {
                        "method": "awq_lite",
                        "alpha_step": 1
                    }

            enable_quant_kv_cache = args.kv_cache_qformat != "none"
            print(
                f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization"
            )

            # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
            if enable_quant_kv_cache:
                # Update KV cache related bmm quantizers
                # If quant_cfg["quant_cfg"] is None, it corresponds to only kv cache quantization case
                quant_cfg["quant_cfg"] = quant_cfg.get(
                    "quant_cfg", {"default": {
                        "enable": False
                    }})
                quant_cfg["quant_cfg"].update(
                    getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])
                    ["quant_cfg"])

                # Set default algorithm for kv cache quantization if not provided.
                if not quant_cfg.get("algorithm", None):
                    quant_cfg["algorithm"] = "max"

            # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
            if model_type == "gemma" and "int8_sq" in args.qformat:
                quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

        # Only run single sample for preview
        input_ids = next(
            iter(calib_dataloader))["input_features" if model_type ==
                                    "whisper" else "input_ids"][0:1]
        generated_ids_before_ptq = model.generate(input_ids, max_new_tokens=100)

        model = quantize_model(model, quant_cfg, args, calib_dataloader)
        if args.compress:
            mtq.compress(model)
            # Lets print the quantization summary
        if args.verbose:
            mtq.print_quant_summary(model)

        # Run some samples
        torch.cuda.empty_cache()
        generated_ids_after_ptq = None
        if model_type != "llama4":
            generated_ids_after_ptq = model.generate(input_ids,
                                                     max_new_tokens=100)
        else:
            warnings.warn(
                "Llama4 Maverick generation after quantization has a bug. Skipping generation sample."
            )

        def input_decode(input_ids):
            return tokenizer.batch_decode(input_ids)

        def output_decode(generated_ids, input_shape):
            return tokenizer.batch_decode(generated_ids[:, input_shape:])

        if generated_ids_after_ptq is not None:
            print("--------")
            print(f"example test input: {input_decode(input_ids)}")
            print("--------")
            print(
                f"example outputs before ptq: {output_decode(generated_ids_before_ptq, input_ids.shape[1])}"
            )
            print("--------")
            print(
                f"example outputs after ptq: {output_decode(generated_ids_after_ptq, input_ids.shape[1])}"
            )

    else:
        assert (model_type != "dbrx"
                ), f"Does not support export {model_type} without quantizaton"
        print(
            f"qformat: {args.qformat}. No quantization applied, export {device} model"
        )

    with torch.inference_mode():
        if model_type is None:
            print(
                f"Unknown model type {type(model).__name__}. Continue exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.export_path

        if model_type == "mllama":
            full_model_config = model.config
            model = model.language_model
            # TRT-LLM expects both the vision_config and text_config to be set for export.
            setattr(model.config, "vision_config",
                    full_model_config.vision_config)
            setattr(model.config, "text_config", full_model_config.text_config)
            setattr(model.config, "architectures",
                    full_model_config.architectures)

        start_time = time.time()
        export_hf_checkpoint(
            model,
            export_dir=export_path,
        )

        # Restore default padding and export the tokenizer as well.
        if tokenizer is not None:
            tokenizer.padding_side = default_padding_side
            tokenizer.save_pretrained(export_path)

        end_time = time.time()
        print(
            f"Quantized model exported to :{export_path}. Total time used {end_time - start_time}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyt_ckpt_path",
        help="Specify where the PyTorch checkpoint path is",
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qformat",
        help=
        ("Quantization format. If --auto_quantize_bits is set, this argument specifies the quantization "
         "format for optimal per-layer AutoQuantize search."),
        default="fp8",
    )
    parser.add_argument(
        "--batch_size",
        help=
        "Batch size for calibration. Default to 0 as we calculate max batch size on-the-fly",
        type=int,
        default=0,
    )
    parser.add_argument("--calib_size",
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument("--dataset",
                        help="name of dataset.",
                        type=str,
                        default=None)
    parser.add_argument("--inference_tensor_parallel", type=int, default=1)
    parser.add_argument("--inference_pipeline_parallel", type=int, default=1)
    parser.add_argument("--awq_block_size", default=0, type=int)
    parser.add_argument(
        "--sparsity_fmt",
        help="Sparsity format.",
        default="dense",
        choices=["dense", "sparsegpt"],
    )
    parser.add_argument(
        "--auto_quantize_bits",
        default=None,
        type=float,
        help=(
            "Effective bits constraint for AutoQuantize. If not set, "
            "regular quantization without AutoQuantize search will be applied."
        ),
    )
    parser.add_argument(
        "--kv_cache_qformat",
        required=False,
        default="fp8",
        choices=["fp8", "nvfp4", "none"],
        help=
        "Specify KV cache quantization format, default to fp8 if not provided",
    )
    parser.add_argument(
        "--vlm",
        help="Specify whether this is a visual-language model",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--compress",
        help="Compress the model weights after quantization",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help=
        "Print verbose output (e.g. quantization summary). Disable by --no_verbose.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    main(args)
