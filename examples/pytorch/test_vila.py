import argparse

import requests
from PIL import Image

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.models.modeling_vila import VilaLlamaModel  # noqa


def get_instruction(prompt, num_images=1) -> str:
    prompt = ("<image> \\n ") * num_images + prompt
    sep = "###"
    messages = [prompt, None]
    roles = ("Human", "Assistant")
    instructions = "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions." + sep
    for role, message in zip(roles, messages):
        if message:
            if type(message) is tuple:
                message, _, _ = message
            instructions += role + ": " + message + sep
        else:
            instructions += role + ":"
    return instructions


prompts = [
    "Describe the natural environment in the image.",
    "Describe the object and the weather condition in the image.",
    "Describe the traffic condition on the road in the image.",
]

image_filenames = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]


def run_vila_pixel_values(args):
    llm = LLM(model=args.model_path, )
    images = [
        Image.open(requests.get(image_filename, stream=True).raw).convert("RGB")
        for image_filename in image_filenames
    ]
    inputs = []
    for prompt, image in zip(prompts, images):
        inputs.append({
            "prompt": get_instruction(prompt),
            "multi_modal_data": {
                "image": image
            },
            "mm_processor_kwargs": {},
        })
    outputs = llm.generate(inputs=inputs,
                           sampling_params=SamplingParams(
                               temperature=args.temperature,
                               top_p=args.top_p,
                           ))
    for idx in range(len(outputs)):
        print(f"outputs[{idx}]: {outputs[idx].outputs[0].text}")


def main(args):
    run_vila_pixel_values(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on VILA with the PyTorch backend")
    parser.add_argument(
        "--query",
        type=str,
        default="Describe the traffic condition on the road in great details.")
    parser.add_argument("--model_path",
                        type=str,
                        default="../../../models/VILA1.5-3b/")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()
    main(args)
