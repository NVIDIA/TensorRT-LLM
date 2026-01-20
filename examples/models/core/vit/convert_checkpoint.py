import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import LlavaNextVisionWrapper
from tensorrt_llm.models.modeling_utils import QuantConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, default=None)

    parser.add_argument(
        '--vision_tp_size',
        type=int,
        default=1,
        help='N-way tensor parallelism size for the vision tower')

    parser.add_argument(
        '--vision_cp_size',
        type=int,
        default=1,
        help='N-way context parallelism size for the vision tower')

    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])

    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')

    parser.add_argument("--load_model_on_cpu", action="store_true")

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
    parser.add_argument('--log_level', type=str, default='info')

    args = parser.parse_args()
    return args


def args_to_build_options(args):
    return {
        'load_model_on_cpu': args.load_model_on_cpu,
    }


def convert_and_save_hf(args):
    model_dir = args.model_dir
    load_by_shard = args.load_by_shard
    world_size = args.vision_tp_size * args.vision_cp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    override_fields.update(args_to_build_options(args))

    quant_config = QuantConfig()

    # When not loading by shard, preload one complete model and then slice per rank weights from this
    # this saves the disk reloading time
    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.vision_tp_size,
                          cp_size=args.vision_cp_size)
        tik = time.time()
        llava_next_vision_wrapper = LlavaNextVisionWrapper.from_hugging_face(
            model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            load_by_shard=load_by_shard,
            **override_fields,
        )
        print(f'Total time of reading and converting {time.time()-tik} s')
        tik = time.time()
        llava_next_vision_wrapper.save_checkpoint(args.output_dir,
                                                  save_config=(rank == 0))
        del llava_next_vision_wrapper
        print(f'Total time of saving checkpoint {time.time()-tik} s')

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
    logger.set_level(args.log_level)

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
