import argparse
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import GPTForCausalLM
from tensorrt_llm.models.gpt.convert import (UnpackedNemoCheckpointDir,
                                             copy_tokenizer_files, load_hf_gpt,
                                             unpack_nemo_ckpt,
                                             update_tokenizer_paths)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--nemo_ckpt_path', type=str, default=None)
    parser.add_argument(
        '--gpt_variant',
        default=None,
        choices=[
            None, 'gpt2', 'santacoder', 'starcoder', 'starcoder2', 'persimmon',
            'kosmos-2', 'nemotron'
        ],
        help=
        "By default the script will try to infer the gpt_variant from model_dir. "
        "Or users may overwrite gpt_variant by explicitly passing the variant.")
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help=
        "The data type for the model weights and activations if not quantized. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
    parser.add_argument("--load_model_on_cpu", action="store_true")
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )

    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='lambada',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument("--dataset_cache_dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--nemo_rename_key',
        type=str,
        nargs='+',
        default=[],
        help=
        "Change a layer name when loading a NeMo checkpoint. Should follow <old_name_pattern>:<new_name_pattern>"
    )

    args = parser.parse_args()

    tensorrt_llm.logger.set_level(args.log_level)
    return args


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16
    elif args.smoothquant:
        quant_config.smoothquant_val = args.smoothquant
        if args.per_channel:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if args.per_token:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                quant_config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if args.int8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.INT8

    # Check if model ckpt is pre-quantized to fp8.
    hf_quant_config_path = f"{args.model_dir}/hf_quant_config.json"
    if os.path.exists(hf_quant_config_path):
        with open(hf_quant_config_path, 'r') as f:
            hf_quant_config = json.load(f)
        if hf_quant_config.get("producer", {}).get("name") == "modelopt":
            modelopt_quant_config = hf_quant_config.get("quantization", {})
            if modelopt_quant_config.get("quant_algo", None) == QuantAlgo.FP8:
                quant_config.quant_algo = QuantAlgo.FP8
            if modelopt_quant_config.get("kv_cache_quant_algo",
                                         None) == QuantAlgo.FP8:
                quant_config.kv_cache_quant_algo = QuantAlgo.FP8

    return quant_config


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_model_on_cpu = args.load_model_on_cpu
    world_size = args.tp_size * args.pp_size

    override_fields = {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'gpt_variant': args.gpt_variant,
    }

    quant_config = args_to_quant_config(args)
    is_prequantized_to_fp8 = quant_config.quant_algo == QuantAlgo.FP8
    if is_prequantized_to_fp8:
        override_fields.update({'is_prequantized_to_fp8': True})

    if args.smoothquant is not None or args.int8_kv_cache:
        mapping = Mapping(world_size=world_size,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        GPTForCausalLM.quantize(
            args.model_dir,
            args.output_dir,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            device='cpu' if args.load_model_on_cpu else 'cuda',
            calib_dataset=args.calib_dataset,
            **override_fields)
    else:
        # Defer weight loading if checkpoint is prequantized to fp8.
        if is_prequantized_to_fp8:
            hf_model_or_dir = model_dir
        else:
            hf_model_or_dir = load_hf_gpt(model_dir, load_model_on_cpu)

        def convert_and_save_rank(args, rank):
            mapping = Mapping(world_size=world_size,
                              rank=rank,
                              tp_size=args.tp_size,
                              pp_size=args.pp_size)
            model = GPTForCausalLM.from_hugging_face(hf_model_or_dir,
                                                     args.dtype,
                                                     mapping=mapping,
                                                     quant_config=quant_config,
                                                     **override_fields)
            model.save_checkpoint(args.output_dir, save_config=(rank == 0))
            del model

        execute(args.workers, [convert_and_save_rank] * world_size, args)
        release_gc()


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def convert_and_save_nemo(args):
    world_size = args.tp_size * args.pp_size
    quant_config = args_to_quant_config(args)

    override_fields = {
        'use_parallel_embedding': True,
        'embedding_sharding_dim': 0,
    }

    nemo_ckpt_dir = os.path.join(args.output_dir, "unpacked")
    nemo_ckpt_dir = unpack_nemo_ckpt(args.nemo_ckpt_path, nemo_ckpt_dir)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        model = GPTForCausalLM.from_nemo(
            nemo_ckpt_dir,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            load_model_on_cpu=args.load_model_on_cpu,
            nemo_rename_key=args.nemo_rename_key,
            **override_fields)
        model.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del model

    execute(args.workers, [convert_and_save_rank] * world_size, args)
    release_gc()

    # Copy tokenizer files
    unpacked_checkpoints_dir = UnpackedNemoCheckpointDir(
        nemo_ckpt_dir, load_checkpoints_to_cpu=args.load_model_on_cpu)
    nemo_model_config = unpacked_checkpoints_dir.model_config
    tokenizer_config = update_tokenizer_paths(
        nemo_model_config["tokenizer"],
        unpacked_checkpoints_dir.get_all_tokenizer_file_paths())
    copy_tokenizer_files(tokenizer_config, Path(args.output_dir))

    # Clean up unpacked nemo checkpoint
    shutil.rmtree(nemo_ckpt_dir)


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_dir is not None:
        convert_and_save_hf(args)
    elif args.nemo_ckpt_path is not None:
        convert_and_save_nemo(args)
    else:
        raise NotImplementedError("No source model path specified!")

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
