import argparse
import os

from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models import LLaMAForCausalLM


def read_input():
    while (True):
        input_text = input("<")
        if input_text in ("q", "quit"):
            break
        yield input_text


def parse_args():
    parser = argparse.ArgumentParser(description="Llama single model example")
    parser.add_argument(
        "--engine_dir",
        type=str,
        required=True,
        help=
        "Directory to save and load the engine. When -c is specified, always rebuild and save to this dir. When -c is not specified, load engine when the engine_dir exists, rebuild otherwise"
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Read the model data and tokenizer from this directory")
    parser.add_argument(
        "-c",
        "--clean_build",
        default=False,
        action="store_true",
        help=
        "Clean build the engine even if the engine_dir exists, be careful, this overwrites the engine_dir!!"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_dir = args.hf_model_dir
    max_batch_size, max_isl, max_osl = 1, 256, 20

    if args.clean_build or not os.path.exists(args.engine_dir):
        os.makedirs(args.engine_dir, exist_ok=True)
        llama = LLaMAForCausalLM.from_hugging_face(args.hf_model_dir)
        llama.to_trt(max_batch_size, max_isl, max_osl)
        llama.save(args.engine_dir)

    executor = GenerationExecutor(args.engine_dir, tokenizer_dir)

    for inp in read_input():
        output = executor.generate(inp, max_new_tokens=20)
        print(f">{output.text}")


main()
