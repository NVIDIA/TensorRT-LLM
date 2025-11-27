import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.falcon.model import FalconForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
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

    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')
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
    args = parser.parse_args()

    tensorrt_llm.logger.set_level(args.log_level)

    # WAR for modelopt multithreading issue.
    import importlib.metadata
    if (importlib.metadata.version('nvidia-modelopt') < '0.27'
            and args.workers > 1):
        args.workers = 1
        tensorrt_llm.logger.warning(
            'Reducing workers=1 when converting checkpoint because '
            'modelopt has an issue in multi-threading, which will be '
            'fixed in 0.27.')

    return args


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    config = QuantConfig()
    if args.use_weight_only and args.weight_only_precision == 'int8':
        config.quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        config.quant_algo = QuantAlgo.W4A16
    return config


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
    }


def convert_and_save_hf(args: argparse.Namespace):
    model_dir = args.model_dir
    load_by_shard = args.load_by_shard
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    override_fields.update(args_to_build_options(args))

    quant_config = args_to_quant_config(args)
    hf_model = None
    import transformers
    if not args.load_by_shard and quant_config.quant_mode.has_any_quant():
        hf_model = transformers.FalconForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, dtype='auto', device_map='auto')
    else:
        # Initialize huggingface local cache.
        # Huggingface copies the external configuration source (`configuration_falcon.py` here) into its local cache at
        # `~/.cache/huggingface/modules/transformers_modules/<model-name>`,
        # and if multiple threads attempt to do this concurrently, weird issue can happen:
        # Some threads may see an empty configuration_falcon.py file and fail.
        # Preload the config once to initialize local cache, so subsequent multithread loading won't fail.
        _ = transformers.FalconConfig.from_pretrained(model_dir,
                                                      trust_remote_code=True,
                                                      dtype='auto',
                                                      device_map='auto')

    def convert_and_save_rank(args, rank: int):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        falcon = FalconForCausalLM.from_hugging_face(
            model_dir if hf_model is None else hf_model,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            load_by_shard=load_by_shard,
            **override_fields,
        )
        falcon.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del falcon

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


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
