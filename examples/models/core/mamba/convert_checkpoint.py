import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import MambaForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=Path, default=None)
    parser.add_argument("--world_size",
                        type=int,
                        default=1,
                        help="world size, only support tensor parallelism now")
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
        '--output_dir',
        type=Path,
        default='mamba_tllm_checkpoint',
        help='The path to save the mamba TensorRT LLM checkpoint')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()
    return args


def load_config_hf(model_name, ckpt_type, dtype, mapping, quant_config,
                   output_dir):
    if ckpt_type == CheckpointType.hf:  # transformer compatible models
        override_fields = {}
        mamba = MambaForCausalLM.from_hugging_face(
            model_name,
            dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields,
        )
        mamba.save_checkpoint(output_dir, save_config=True)

    elif ckpt_type == CheckpointType.state_spaces:  # state-spaces/mamba models
        config = json.load(open(os.path.join(model_name, 'config.json')))
        override_fields = {}
        mamba = MambaForCausalLM.from_hugging_face(
            model_name,
            dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields,
        )
        mamba.save_checkpoint(output_dir, save_config=True)

        ssm_cfg = config.pop('ssm_cfg')
        cfg_to_mamba_cfg = {
            'd_model': 'hidden_size',
            'n_layer': 'num_hidden_layers',
            'fused_add_norm': None,
            'tie_embeddings': None,
        }
        ssm_cfg_to_mamba_cfg = {
            'd_state': 'state_size',
            'd_conv': 'conv_kernel',
            'bias': 'use_bias',
            'headdim': 'head_dim',
            'ngroups': 'n_groups',
            'chunk_size': 'chunk_size',
            'rmsnorm': 'ssm_rmsnorm',
        }
        for k in cfg_to_mamba_cfg:
            if k in config:
                v = config.pop(k)
                if cfg_to_mamba_cfg[k] is not None:
                    config[cfg_to_mamba_cfg[k]] = v
        for k in ssm_cfg_to_mamba_cfg:
            if k in ssm_cfg and ssm_cfg_to_mamba_cfg[k] is not None:
                config[ssm_cfg_to_mamba_cfg[k]] = ssm_cfg[k]

        hf_config = MambaConfig(**config)
        if 'expand' in ssm_cfg:
            expand = ssm_cfg['expand']
            hf_config.intermediate_size = expand * hf_config.hidden_size
        else:
            hf_config.intermediate_size = 2 * hf_config.hidden_size
        mamba_version = ssm_cfg.pop("layer", "Mamba1")

    elif ckpt_type == CheckpointType.mistral_inference:  # mistral inference format
        config = json.load(open(os.path.join(model_name, 'params.json')))
        cfg_to_mamba_cfg = {
            'dim': 'hidden_size',
            'n_layers': 'num_hidden_layers',
            'n_groups': 'n_groups',
            'fused_add_norm': None,
            'tie_embeddings': None,
            'model_type': None,
        }
        for k in cfg_to_mamba_cfg:
            if k in config:
                v = config.pop(k)
                if cfg_to_mamba_cfg[k] is not None:
                    config[cfg_to_mamba_cfg[k]] = v
        hf_config = MambaConfig(**config)
        if 'expand' in config:
            expand = config['expand']
            hf_config.intermediate_size = expand * hf_config.hidden_size
        else:
            hf_config.intermediate_size = 2 * hf_config.hidden_size
        mamba_version = 'Mamba2'

    return hf_config, mamba_version


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


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16

    return quant_config


def main():
    print(tensorrt_llm.__version__)

    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()
    assert args.pp_size == 1, "Pipeline parallelism is not supported."
    world_size = args.tp_size * args.pp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    quant_config = args_to_quant_config(args)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        mamba = MambaForCausalLM.from_hugging_face(
            args.model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
        )
        mamba.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del mamba

    execute(args.workers, [convert_and_save_rank] * world_size, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
