import argparse
import time

from utils import AutoFormatter

import tensorrt_llm.bindings
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM

default_prompts = [
    "Describe the natural environment in the image.",
    "Describe the object and the weather condition in the image.",
    "Describe the traffic condition on the road in the image.",
]

default_images = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]

default_answers = [
    "The image features a stormy ocean with large waves crashing, a gray sky with white clouds, and a dark gray horizon.",
    "The object is a large rock formation, and the weather condition is sunny with a blue sky and white clouds.",
    "The road is busy with multiple cars, including a silver SUV, a blue car, and a green double-decker bus, all driving in the same direction.",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser.add_argument("--model_dir",
                        type=str,
                        help="Model checkpoint directory.")
    parser.add_argument("--prompt",
                        type=str,
                        nargs="+",
                        default=default_prompts,
                        help="A single or a list of text prompts.")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=default_images,
        help=
        "A single or a list of filepaths / urls / tensors of image or video data."
    )
    parser.add_argument("--check_accuracy",
                        action="store_true",
                        help="Run accuracy check with the default inputs.")
    parser.add_argument("--kv_cache_fraction", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    llm = LLM(
        model=args.model_dir,
        kv_cache_config=tensorrt_llm.bindings.executor.KvCacheConfig(
            free_gpu_memory_fraction=args.kv_cache_fraction),
    )

    inputs = []
    for prompt, data in zip(args.prompt, args.data):
        inputs.append([prompt, data])
    inputs = AutoFormatter.format(llm.hf_model_dir, inputs)

    tik = time.time()
    outputs = llm.generate(inputs=inputs,
                           sampling_params=SamplingParams(
                               max_tokens=args.max_tokens,
                               temperature=args.temperature,
                               top_p=args.top_p,
                           ))
    tok = time.time()

    print(f"Time (ms): {(tok - tik) * 1000}ms")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(
            f"[{i}] Prompt: {args.prompt[i]!r}, Generated text: {generated_text!r}"
        )

    if args.check_accuracy:
        for idx in range(len(outputs)):
            assert outputs[idx].outputs[0].text == default_answers[
                idx], "Wrong answer!"
        print("All answers are correct!")


if __name__ == "__main__":
    main()
