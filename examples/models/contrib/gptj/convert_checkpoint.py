import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm.llmapi import QuantConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import GPTJConfig, GPTJForCausalLM
from tensorrt_llm.models.convert_utils import infer_dtype
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
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
    parser.add_argument('--vocab_size', type=int, default=50400)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=28)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--norm_eps', type=float, default=1e-05)
    parser.add_argument('--rotary_dim', type=int, default=64)
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
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


def args_to_quant_config(args):
    quant_algo = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        quant_algo = QuantAlgo.W4A16
    return QuantConfig(quant_algo=quant_algo)


def convert_and_save_hf(args):
    model_dir = args.model_dir
    world_size = args.tp_size * args.pp_size
    quant_config = args_to_quant_config(args)

    hf_model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                    dtype='auto',
                                                    trust_remote_code=True)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)
        model = GPTJForCausalLM.from_hugging_face(hf_model,
                                                  args.dtype,
                                                  mapping=mapping,
                                                  quant_config=quant_config)
        model.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del model

    if args.workers == 1:
        for rank in range(world_size):
            convert_and_save_rank(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(convert_and_save_rank, args, rank)
                for rank in range(world_size)
            ]
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

    del hf_model


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_dir is None:
        config = GPTJConfig(architecture='GPTJForCausalLM',
                            dtype=infer_dtype(args.dtype),
                            num_hidden_layers=args.n_layer,
                            num_attention_heads=args.n_head,
                            hidden_size=args.n_embd,
                            norm_epsilon=args.norm_eps,
                            vocab_size=args.vocab_size,
                            position_embedding_type='rope_gptj',
                            max_position_embeddings=args.n_positions,
                            hidden_act='gelu',
                            rotary_dim=args.rotary_dim,
                            mapping=Mapping(world_size=args.tp_size *
                                            args.pp_size,
                                            tp_size=args.tp_size,
                                            pp_size=args.pp_size),
                            quantization=args_to_quant_config(args))
        config.to_json_file(os.path.join(args.output_dir, 'config.json'))
    else:
        convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
