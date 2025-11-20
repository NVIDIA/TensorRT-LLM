import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

from transformers import AutoConfig

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (BertForQuestionAnswering,
                                 BertForSequenceClassification, BertModel,
                                 RobertaForQuestionAnswering,
                                 RobertaForSequenceClassification, RobertaModel)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        required=True,
                        choices=[
                            'BertModel',
                            'BertForQuestionAnswering',
                            'BertForSequenceClassification',
                            'RobertaModel',
                            'RobertaForQuestionAnswering',
                            'RobertaForSequenceClassification',
                        ])
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'float16'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    # Quantization args
    parser.add_argument("--use_fp8",
                        action="store_true",
                        default=False,
                        help="Enable FP8 per-tensor quantization")
    parser.add_argument(
        '--quant_ckpt_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .safetensors format')
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )

    parser.add_argument('--log_level', type=str, default='info')

    args = parser.parse_args()

    return args


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()

    if args.use_fp8:
        quant_config.quant_algo = QuantAlgo.FP8
    return quant_config


def convert_and_save_hf(args):
    model_dir = args.model_dir

    world_size = args.tp_size * args.pp_size
    #TODO: add override_fields if needed
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.

    #TODO: add fp8 support later
    quant_config = args_to_quant_config(args)

    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    assert hf_config is not None, "Failed to load huggingface config, please check!"

    def convert_and_save_rank(args, rank, tllm_class: Union[
        BertModel,
        RobertaModel,
        BertForQuestionAnswering,
        RobertaForQuestionAnswering,
        BertForSequenceClassification,
        RobertaForSequenceClassification,
    ]):
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
        )
        tik = time.time()
        tllm_bert = tllm_class.from_hugging_face(
            model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
        )
        print(f'Total time of reading and converting {time.time()-tik} s')
        tik = time.time()
        tllm_bert.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del tllm_bert
        print(f'Total time of saving checkpoint {time.time()-tik} s')

    tllm_class = globals()[f'{args.model}']
    if not args.model == hf_config.architectures[0]:
        logger.warning(
            "The model doesn't match the architecture in huggingface config.")

    execute(args.workers, [convert_and_save_rank] * world_size, args,
            tllm_class)
    release_gc()


def execute(workers, func, args,
            tllm_class: Union[BertModel, RobertaModel, BertForQuestionAnswering,
                              RobertaForQuestionAnswering,
                              BertForSequenceClassification,
                              RobertaForSequenceClassification]):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank, tllm_class)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [
                p.submit(f, args, rank, tllm_class)
                for rank, f in enumerate(func)
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


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    logger.set_level(args.log_level)

    assert ((args.tp_size <= 2)
            and (args.pp_size == 1)), "For now we only support TP = 2!"
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.model_dir is not None
    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
