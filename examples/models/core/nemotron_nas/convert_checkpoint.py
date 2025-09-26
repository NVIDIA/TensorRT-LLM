import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tensorrt_llm._utils import release_gc
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import DeciLMForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)

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

    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')

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
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')

    parser.add_argument(
        '--save_config_only',
        action="store_true",
        default=False,
        help=
        'Only save the model config w/o read and converting weights, be careful, this is for debug only'
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loading functions as needed")

    args = parser.parse_args()
    return args


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
    }


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_by_shard = args.load_by_shard
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    override_fields.update(args_to_build_options(args))

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        model = DeciLMForCausalLM.from_hugging_face(
            model_dir,
            args.dtype,
            mapping=mapping,
            load_by_shard=load_by_shard,
            load_model_on_cpu=args.load_model_on_cpu,
            trust_remote_code=args.trust_remote_code,
            **override_fields,
        )
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


def main():
    args = parse_arguments()
    tik = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO(oargov): all deci checkpoints require trust_remote_code=True at the moment, remove this when this changes
    # NOTE: we opt not to make this the default since users should be made aware of this in-case they don't want to trust remote code
    assert args.trust_remote_code, "Nemotron NAS checkpoint require --trust_remote_code"

    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
