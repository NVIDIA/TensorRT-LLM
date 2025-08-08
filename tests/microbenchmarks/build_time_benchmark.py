import argparse
import os
import pathlib
import sys
import tempfile
import time
import traceback

import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

import tensorrt_llm
from tensorrt_llm import (AutoConfig, AutoModelForCausalLM, BuildConfig,
                          Mapping, SamplingParams, build)
from tensorrt_llm._utils import mpi_barrier, mpi_rank
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

print(f"TensorRT version:{trt.__version__}")

# model name to the sub dir under the llm-models path

# yapf: disable
models_name_to_path = {
    'gpt2': ("{LLM_MODELS_ROOT}/gpt2", 1, 1),
    'gptj-6b': ("{LLM_MODELS_ROOT}/gpt-j-6b", 1, 1),
    'phi2': ("{LLM_MODELS_ROOT}/phi-2", 1, 1),
    "phi3-medium": ("{LLM_MODELS_ROOT}/Phi-3/Phi-3-medium-128k-instruct", 1, 1),
    'falcon-7b': ("{LLM_MODELS_ROOT}/falcon-7b-instruct", 1, 1),
    'falcon-180b.TP8': ("{LLM_MODELS_ROOT}/falcon-180b/", 8, 1),
    'llama-7b': ("{LLM_MODELS_ROOT}/llama-models/llama-7b-hf", 1, 1),
    'llama2-7b': ("{LLM_MODELS_ROOT}/llama-models-v2/llama-v2-7b-hf/", 1, 1),
    'llama2-70b.TP4': ("{LLM_MODELS_ROOT}/llama-models-v2/llama-v2-70b-hf", 4, 1),
    'llama3-70b.TP4': ("{LLM_MODELS_ROOT}/llama-models-v3/Llama-3-70B-Instruct-Gradient-1048k", 4, 1),
    'llama3.1-8b': ("{LLM_MODELS_ROOT}/llama-3.1-model/Meta-Llama-3.1-8B", 1, 2),
    'llama3.1-70b.TP4': ("{LLM_MODELS_ROOT}/llama-3.1-model/Meta-Llama-3.1-8B", 4, 1),
    'llama3.1-405b.TP8PP2': ("{LLM_MODELS_ROOT}/llama-3.1-model/Meta-Llama-3.1-405B", 8, 2),
    'llama3.1-405b.TP8': ("{LLM_MODELS_ROOT}/llama-3.1-model/Meta-Llama-3.1-405B", 8, 1),
    'mixtral-8x22b.TP4': ("{LLM_MODELS_ROOT}/Mixtral-8x22B-v0.1", 4, 1),
    'mixtral-8x22b.TP8': ("{LLM_MODELS_ROOT}/Mixtral-8x22B-v0.1", 8, 1),
    'mixtral-8x7b.TP4': ("{LLM_MODELS_ROOT}/Mixtral-8x7B-v0.1", 4, 1),
    'mistral-7b': ("{LLM_MODELS_ROOT}/mistral-7b-v0.1", 1, 1),
    "gemma-7b": ("{LLM_MODELS_ROOT}/gemma/gemma-7b/", 1, 1),
    "gemma-2-9b": ("{LLM_MODELS_ROOT}/gemma/gemma-2-9b-it/", 1, 1),
    "gemma-2-27b": ("{LLM_MODELS_ROOT}/gemma/gemma-2-27b-it/", 1, 1),
    "qwen2-72b.TP8": ("{LLM_MODELS_ROOT}/Qwen2-72B-Instruct/", 8, 1),
    "glm-4-9b": ("{LLM_MODELS_ROOT}/glm-4-9b-chat/", 1, 1),
}
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "One microbenchmark to measure the engine build time for common models")

    parser.add_argument("--models_root",
                        type=str,
                        default=os.environ.get("LLM_MODELS_ROOT"),
                        help="The llm-models root path")
    parser.add_argument(
        "--model",
        type=str,
        default='gpt2',
        help=
        f"The model to benchmark, there are multiple modes to be supported" \
        f"- Builtin offline models, a list of models stored inside the LLM_MODELS_ROOT dir: {list(models_name_to_path.keys()) + ['ALL']} " \
        f"- User specified model directory, like './llama-7b-hf', which is your local path of the hf model" \
        f"- A huggingface model id, like 'openai-community/gpt2', refer to model in https://huggingface.co/openai-community/gpt2"

    )
    parser.add_argument("--dtype",
                        type=str,
                        choices=['auto', 'float32', 'float16', 'bfloat16'],
                        default='auto',
                        help="The data type of the fake weights for the model")
    parser.add_argument(
        '--quant',
        type=str,
        default=None,
        choices=['fp8', 'sq'],
        help="The quantization algorithm to be used",
    )
    parser.add_argument("--verbose",
                        '-v',
                        default=False,
                        action='store_true',
                        help="Turn on verbose log")
    parser.add_argument("--load",
                        default=False,
                        action='store_true',
                        help="Load Hugging Face weights")
    parser.add_argument("--load_to_cpu",
                        default=False,
                        action='store_true',
                        help="Load HF model to CPU, auto otherwise")
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
    parser.add_argument("--multi_profiles",
                        default=False,
                        action="store_true",
                        help="Turn on TRT-LLM multi profiles")
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help=
        "TP size, when not specified, use the default one defined in this script if defined or 1 if not"
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help=
        "PP size, when not specified, use the default one defined in this script if defined or 1 if not"
    )
    parser.add_argument("--parallel",
                        "-P",
                        default=False,
                        action="store_true",
                        help="Convert and build multiple ranks in parallel")
    return parser.parse_args()


def log_level(args):
    tensorrt_llm.logger.set_level('info')
    if args.verbose:
        tensorrt_llm.logger.set_level('verbose')


def sanity_check(hf_model_dir, engine):
    from transformers import AutoTokenizer
    sampling = SamplingParams(max_tokens=5)
    input_str = "What should you say when someone gives you a gift? You should say:"
    executor = GenerationExecutor.create(engine)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                              trust_remote_code=True)
    if mpi_rank() == 0:
        output = executor.generate(tokenizer.encode(input_str),
                                   sampling_params=sampling)
        output_str = tokenizer.decode(output.outputs[0].token_ids)
        print(f"{input_str} {output_str}")


def update_build_config(build_config, args):
    build_config.plugin_config.manage_weights = args.managed_weights
    if args.gemm == 'plugin':
        build_config.plugin_config.gemm_plugin = 'auto'
    else:
        assert args.gemm == 'ootb'
        build_config.plugin_config.gemm_plugin = None
    build_config.strongly_typed = args.strong_type

    if args.multi_profiles:
        build_config.plugin_config.multiple_profiles = True


def build_from_hf(args,
                  model_tag,
                  hf_model_dir,
                  dtype,
                  load_weights,
                  tp,
                  pp,
                  rank=0):
    '''Build model and init executor using huggingface model config and fake weights, useful for benchmarking
    '''
    status, = cudart.cudaSetDevice(rank)
    assert status == cudart.cudaError_t.cudaSuccess, f"cuda set device to {rank} errored: {status}"
    log_level(args)
    mpi_barrier()
    world_size = tp * pp
    # TODO: Only build 1 rank for now, all the ranks shall have similar build time
    # shall we build all ranks in parallel?
    mapping = Mapping(world_size=world_size, rank=rank, tp_size=tp, pp_size=pp)

    quant_config = None
    if args.quant == 'fp8':
        quant_config = QuantConfig(QuantAlgo.FP8)

    phase_and_time = []
    if load_weights:
        quant_output_dir = tempfile.TemporaryDirectory(model_tag)
        start = time.time()
        if args.quant is None:
            trtllm_model = AutoModelForCausalLM.from_hugging_face(
                hf_model_dir,
                dtype,
                mapping,
                load_model_on_cpu=args.load_to_cpu)
        else:
            model_cls = AutoModelForCausalLM.get_trtllm_model_class(
                hf_model_dir)
            if rank == 0:
                model_cls.quantize(hf_model_dir,
                                   output_dir=quant_output_dir.name,
                                   dtype=args.dtype,
                                   mapping=mapping,
                                   quant_config=quant_config)
            mpi_barrier(
            )  # every rank must wait rank 0 to get the correct quantized checkpoint
            trtllm_model = model_cls.from_checkpoint(quant_output_dir.name)
        phase_and_time.append(('load_and_convert', time.time() - start))
        quant_output_dir.cleanup()

    else:  # fake weights
        trtllm_config = AutoConfig.from_hugging_face(hf_model_dir,
                                                     dtype,
                                                     mapping,
                                                     quant_config,
                                                     trust_remote_code=True)
        trtllm_model = AutoModelForCausalLM.get_trtllm_model_class(
            hf_model_dir)(trtllm_config)

    start = time.time()
    build_config = BuildConfig(max_input_len=1024, max_batch_size=16)
    update_build_config(build_config, args)

    engine = build(trtllm_model, build_config)
    assert engine is not None

    phase_and_time.append(('build_engine', time.time() - start))
    for (p, t) in phase_and_time:
        tensorrt_llm.logger.info(
            f"===BuildTime==== {p} {model_tag} {t:.2f} seconds")
    mpi_barrier()

    start = time.time()
    ## Since only build one engine for build time measurement, the sanity run only support TP/PP 1
    if mapping.world_size == 1 or args.parallel:
        sanity_check(hf_model_dir, engine)
        tensorrt_llm.logger.info(
            f"===EngineInit==== engine_init {model_tag} {time.time()-start:.2f} seconds"
        )

    mpi_barrier()
    return True


def run_models(models, args):
    from mpi4py.futures import MPIPoolExecutor
    for model in models:
        if model in models_name_to_path:
            model_dir, tp, pp = models_name_to_path[model]
            model_dir = pathlib.Path(
                model_dir.format(LLM_MODELS_ROOT=args.models_root))
            assert model_dir.exists(
            ), f"{model_dir} does not exist, pls check the model path"
        else:  # online model
            tensorrt_llm.logger.warning(
                "You are trying to download a model from HF online to benmark the build time, build time depends on the network time"
            )
            model_dir, tp, pp = args.model, args.tp, args.pp

        world_size = tp * pp

        tensorrt_llm.logger.info(f"build_from_hf {str(model_dir)} start")

        try:
            if not args.parallel:
                r = build_from_hf(args, model, str(model_dir), args.dtype,
                                  args.load, tp, pp, 0)
                assert r, "must return True"
            else:
                with MPIPoolExecutor(max_workers=world_size) as pool:
                    results = []
                    for rank in range(world_size):
                        r = pool.submit(build_from_hf, args, model,
                                        str(model_dir), args.dtype, args.load,
                                        tp, pp, rank)
                        results.append(r)
                    for r in results:
                        assert r.result() is True

        except Exception as e:
            traceback.print_exc()
            tensorrt_llm.logger.error(str(e))
            tensorrt_llm.logger.info(f"build_from_hf {str(model_dir)} failed")
            continue

        tensorrt_llm.logger.info(f"build_from_hf {str(model_dir)} end")


def main():
    args = parse_args()
    log_level(args)
    tensorrt_llm.logger.info(str(args))
    tensorrt_llm.logger.info(f"Running cmd: " + " ".join(sys.argv))

    # ALL is special
    target_models = models_name_to_path.keys() if args.model == "ALL" else [
        args.model
    ]

    run_models(target_models, args)


if __name__ == "__main__":
    main()
