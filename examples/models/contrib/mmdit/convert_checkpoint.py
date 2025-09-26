import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import SD3Transformer2DModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--cp_size',
                        type=int,
                        default=1,
                        help='N-way context parallelism size')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help=
        "The data type for the model weights and activations if not quantized. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
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
    return args


def convert_and_save_hf(args):
    world_size = args.tp_size * args.cp_size

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          cp_size=args.cp_size)
        tik = time.time()
        mmdit = SD3Transformer2DModel.from_pretrained(args.model_path,
                                                      args.dtype,
                                                      mapping=mapping)
        print(f'Total time of reading and converting: {time.time()-tik:.3f} s')
        tik = time.time()
        mmdit.save_checkpoint(args.output_dir, save_config=(rank == 0))
        # post-process checkpoint config
        with open(f"{args.output_dir}/config.json", 'r') as f:
            config = json.load(f)
        config["model_path"] = args.model_path
        config["use_pretrained_pos_emb"] = True if "pos_embed.pos_embed" in [
            name for name, _ in mmdit.named_parameters()
        ] else False
        with open(f"{args.output_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=4)
        del mmdit
        print(f'Total time of saving checkpoint: {time.time()-tik:.3f} s')

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

    assert args.model_path is not None
    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
