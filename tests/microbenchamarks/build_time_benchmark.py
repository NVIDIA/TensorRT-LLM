import argparse
import os
import pathlib
import time

import tensorrt_llm
from tensorrt_llm import (AutoConfig, AutoModelForCausalLM, BuildConfig,
                          Mapping, build)

# model name to the sub dir under the llm-models path
models_name_to_path = {
    'gpt2': ("gpt2", 1, 1),
    'phi2': ('phi-2', 1, 1),
    'llama-7b': ("llama-models/llama-7b-hf", 1, 1),
    'falcon-7b': ("falcon-7b-instruct", 1, 1),
    'gptj-6b': ("gpt-j-6b", 1, 1),
    'llama2-7b': ("llama-models-v2/llama-v2-7b-hf/", 1, 1),
    'llama2-70b.TP4': ("llama-models-v2/llama-v2-70b-hf", 4, 1),
    'mixtral-8x22b.TP4': ("Mixtral-8x22B-v0.1", 4, 1),
    'mixtral-8x7b.TP4': ("Mixtral-8x7B-v0.1", 4, 1),
    'mistral-7b': ("mistral-7b-v0.1", 1, 1)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "One microbenchmark to measure the engine build time for common models")

    parser.add_argument("--models_root",
                        type=str,
                        default=os.environ.get("LLM_MODELS_ROOT"),
                        help="The llm-models root path")
    parser.add_argument("--model",
                        type=str,
                        default='gpt2',
                        choices=list(models_name_to_path.keys()) + ["ALL"],
                        help="The model subdir under the models_root")
    parser.add_argument("--dtype",
                        type=str,
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        default='auto',
                        help="The data type of the fake weights for the model")
    parser.add_argument("--verbose",
                        '-v',
                        default=False,
                        action='store_true',
                        help="Turn on verbose log")
    parser.add_argument("--load",
                        default=False,
                        action='store_true',
                        help="Load Hugging Face weights")
    parser.add_argument("--opt",
                        default=3,
                        type=int,
                        choices=[0, 1, 2, 3, 4, 5],
                        help="Builder optimization level")
    parser.add_argument("--gemm",
                        type=str,
                        default='ootb',
                        choices=['plugin', 'ootb'],
                        help="Use plugin or TRT for GEMM")
    parser.add_argument("--strong_type",
                        default=False,
                        action="store_true",
                        help="Use strong type")
    parser.add_argument("--managed_weights",
                        default=False,
                        action="store_true",
                        help="Turn on TRT-LLM managed weights")
    return parser.parse_args()


def build_from_hf(args, model_tag, hf_model_dir, dtype, load_weights, tp, pp):
    '''Build model and init executor using huggingface model config and fake weights, useful for benchmarking
    '''
    world_size = tp * pp
    # TODO: Only build 1 rank for now, all the ranks shall have similar build time
    # shall we build all ranks in parallel?
    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp, pp_size=pp)

    phase_and_time = []
    if load_weights:
        start = time.time()
        trtllm_model = AutoModelForCausalLM.from_hugging_face(
            hf_model_dir, dtype, mapping)
        phase_and_time.append(('load_and_convert', time.time() - start))

    else:  # fake weights
        trtllm_config = AutoConfig.from_hugging_face(hf_model_dir, dtype,
                                                     mapping)
        trtllm_model = AutoModelForCausalLM.get_trtllm_model_class(
            hf_model_dir)(trtllm_config)

    start = time.time()
    build_config = BuildConfig(max_input_len=1024, max_batch_size=16)

    build_config.builder_opt = args.opt
    build_config.plugin_config.manage_weights = args.managed_weights
    if args.gemm == 'plugin':
        build_config.plugin_config.gemm_plugin = 'auto'
    else:
        assert args.gemm == 'ootb'
        build_config.plugin_config.gemm_plugin = None
    build.strongly_typed = args.strong_type

    engine = build(trtllm_model, build_config)
    assert engine is not None

    phase_and_time.append(('build_engine', time.time() - start))
    for (p, t) in phase_and_time:
        tensorrt_llm.logger.info(
            f"===BuildTime==== {p} {model_tag} {t} seconds")


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        tensorrt_llm.logger.set_level('verbose')
    else:
        tensorrt_llm.logger.set_level('info')

    target_models = args.model
    if target_models == "ALL":
        target_models = models_name_to_path.keys()
    else:
        target_models = [target_models]

    for model in target_models:
        model_dir, tp, pp = models_name_to_path[model]
        model_dir = pathlib.Path(args.models_root) / model_dir
        assert model_dir.exists()
        build_from_hf(args, model, str(model_dir), args.dtype, args.load, tp,
                      pp)
