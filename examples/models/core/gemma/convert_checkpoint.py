#!/usr/bin/env python3
import argparse
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Type

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import infer_dtype
from tensorrt_llm.models.gemma.config import GEMMA_ARCHITECTURE, GemmaConfig
from tensorrt_llm.models.gemma.convert import (HfParser, JAXParser, KerasParser,
                                               Parsers, QuantizeModifiers,
                                               TorchParser, load_gemma_weights,
                                               non_modelopt_quantize_if_needed)
from tensorrt_llm.models.gemma.model import GemmaForCausalLM
from tensorrt_llm.models.modeling_utils import (QuantConfig, save_checkpoint,
                                                save_config)
from tensorrt_llm.quantization import QuantAlgo


class CheckpointType(str, Enum):
    jax = "jax"
    keras = "keras"
    torch = "torch"
    hf = "hf"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-type",
                        type=CheckpointType,
                        choices=list(CheckpointType))
    parser.add_argument("--model-dir", "--model_dir", type=Path, required=True)
    parser.add_argument("--output-model-dir",
                        "--output_dir",
                        type=Path,
                        required=True)
    parser.add_argument("--world-size",
                        type=int,
                        default=1,
                        help="world size, only support tensor parallelism now")
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        "--use-weight-only-with-precision",
        choices=["int8", "int4", "w4a8_awq", "w4a16_awq"],
        help=
        "help='Quantize weights for the various GEMMs to INT4/INT8. Define the precision for the weights.",
    )
    parser.add_argument(
        "--use-int8-weight-only-embedding",
        action="store_true",
        help=
        "Use weight only on embedding table and lm_head. (Only supported on Hopper GPU)",
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help=
        "The data type for the model weights and activations if not quantized. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
    parser.add_argument(
        "--enable_fp8",
        action="store_true",
        help="Use FP8 Linear layer for Attention QKV/Dense and MLP.")
    parser.add_argument(
        "--fp8_kv_cache",
        action="store_true",
        help=
        "By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV",
    )
    parser.add_argument(
        "--quant_ckpt_path",
        default=None,
        help=
        "Path of a directory to quantized model checkpoints in .safetensors format or \
              path of a quantized model checkpoint in .npz format")
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument('--use_smooth_quant',
                        action="store_true",
                        help="Use smooth quant.")
    parser.add_argument(
        "--int8_kv_cache",
        "--calibrate_kv_cache",
        "-kv",
        action="store_true",
        help=
        "Generate scaling factors for KV cache. Used for storing KV cache in int8."
    )
    parser.add_argument(
        '--per_channel',
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        "--smoothquant",
        "--use_smooth_quant_plugin",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument(
        '--tokenizer_dir',
        default=None,
        help='tokenizer path; defaults to jax_model_dir if left unspecified')
    parser.add_argument("--load_model_on_cpu", action="store_true")
    args = parser.parse_args()

    if args.use_weight_only:
        assert args.weight_only_precision is not None
    return args


CKPT_PARSER: Dict[CheckpointType, Type[Parsers]] = {
    CheckpointType.jax: JAXParser,
    CheckpointType.keras: KerasParser,
    CheckpointType.torch: TorchParser,
    CheckpointType.hf: HfParser
}


def compute_quant_algo(args: argparse.Namespace) -> Optional[QuantAlgo]:
    if args.weight_only_precision:
        return {
            "int8": QuantAlgo.W8A16,
            "int4": QuantAlgo.W4A16,
            "w4a8_awq": QuantAlgo.W4A8_AWQ,
            "w4a16_awq": QuantAlgo.W4A16_AWQ,
        }[args.weight_only_precision]
    elif args.enable_fp8:
        return QuantAlgo.FP8
    if args.use_smooth_quant:
        return QuantAlgo.W8A8_SQ_PER_CHANNEL
    elif args.smoothquant is not None:
        if args.per_token and args.per_channel:
            return QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
        elif not args.per_token and not args.per_channel:
            return QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        elif not args.per_token and args.per_channel:
            return QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        elif args.per_token and not args.per_channel:
            return QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
    return None


def create_quant_config(args: argparse.Namespace) -> QuantConfig:
    quant_algo = compute_quant_algo(args)
    GemmaForCausalLM.assert_valid_quant_algo(quant_algo)
    quant_config = QuantConfig(quant_algo=quant_algo,
                               smoothquant_val=args.smoothquant)

    if args.fp8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8
    if args.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    if args.weight_only_precision:
        use_awq = args.weight_only_precision.endswith("awq")
        use_int4 = args.weight_only_precision.endswith("int4")
        if use_awq:
            quant_config.group_size = 128

        if use_awq or use_int4 or not args.use_int8_weight_only_embedding:
            quant_config.has_zero_point = False
            quant_config.pre_quant_scale = True
        else:
            quant_config.exclude_modules = ['router']

    return quant_config


def main() -> None:
    args = parse_arguments()
    tik = time.time()
    quant_config = create_quant_config(args)

    ckpt_parser = CKPT_PARSER[args.ckpt_type]()

    mapping = Mapping(
        world_size=args.world_size,
        tp_size=args.world_size,
        pp_size=1,
    )
    """We don't support pipeline parallelism yet for Gemma."""

    if isinstance(ckpt_parser, HfParser):
        trt_llm_config = GemmaConfig.from_hugging_face(
            args.model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
        )
    else:
        print(f"Loading source parameters from {args.model_dir.absolute()}")
        ckpt_params = ckpt_parser.load_parameters(args.model_dir)
        input_embedding_weights = ckpt_parser.embedding_weights(ckpt_params)
        num_embed, _ = input_embedding_weights.shape
        ckpt_params_dtype = str(input_embedding_weights.dtype).split(".")[
            -1]  # np.bfloat16 -> bfloat16
        ckpt_config = ckpt_parser.get_config(args.model_dir, ckpt_params,
                                             num_embed)
        # 2B TransformerConfig(num_layers=18, num_embed=256128, embed_dim=2048, hidden_dim=16384, num_heads=8, head_dim=256, num_kv_heads=1)
        # 7B TransformerConfig(...)

        del ckpt_params

        print(f"Source configuration determined from parameters: {ckpt_config}")

        trt_llm_config = tensorrt_llm.models.GemmaConfig(
            architecture=GEMMA_ARCHITECTURE,
            dtype=infer_dtype(args.dtype, ckpt_params_dtype),
            logits_dtype="float32",
            vocab_size=ckpt_config.num_embed,
            max_position_embeddings=8192,
            hidden_size=ckpt_config.embed_dim,
            num_hidden_layers=ckpt_config.num_layers,
            num_attention_heads=ckpt_config.num_heads,
            num_key_value_heads=ckpt_config.num_kv_heads,
            head_size=ckpt_config.head_dim,
            hidden_act="gelu",
            intermediate_size=ckpt_config.hidden_dim,
            norm_epsilon=1e-6,  # hard-coded in RMSNorm from gemma/layers.py
            position_embedding_type="rope_gpt_neox",
            mapping=mapping,
            gpus_per_node=8,
            quantization=quant_config,
            use_parallel_embedding=mapping.tp_size > 1,
        )
        if hasattr(ckpt_config,
                   "model_type") and ckpt_config.model_type == "gemma2":
            trt_llm_config.inter_layernorms = True
            trt_llm_config.final_logit_softcapping = ckpt_config.final_logit_softcapping
            trt_llm_config.attn_logit_softcapping = ckpt_config.attn_logit_softcapping
            trt_llm_config.query_pre_attn_scalar = ckpt_config.query_pre_attn_scalar

    trt_llm_config_dict = trt_llm_config.to_dict()
    print(f"Determined TensorRT LLM configuration {trt_llm_config_dict}")

    save_config(trt_llm_config, output_dir=args.output_model_dir, log=True)

    for config in trt_llm_config.for_each_rank():
        hf_weights = load_gemma_weights(
            parameters_or_model_dir=args.model_dir,
            trt_llm_config=config,
            ckpt_parser=ckpt_parser,
            load_model_on_cpu=args.load_model_on_cpu)
        ranked_weights = non_modelopt_quantize_if_needed(
            hf_weights,
            model_dir=args.model_dir,
            quantize_modifiers=QuantizeModifiers.from_args(args),
            trt_llm_config=config)
        save_checkpoint(output_dir=args.output_model_dir,
                        weights=ranked_weights,
                        rank=config.mapping.rank)

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - tik))
    print(f"Total time of converting checkpoints: {elapsed}")


if __name__ == "__main__":
    main()
