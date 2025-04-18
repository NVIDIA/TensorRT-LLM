import argparse
import json
import os
from typing import Any, Dict, List

from quickstart_advanced import add_llm_args, setup_llm

from tensorrt_llm.inputs import (INPUT_FORMATTER_MAP, default_image_loader,
                                 default_video_loader)

example_images = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]
example_image_prompts = [
    "Describe the natural environment in the image.",
    "Describe the object and the weather condition in the image.",
    "Describe the traffic condition on the road in the image.",
]
example_videos = [
    "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4",
    "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4",
]
example_video_prompts = [
    "Tell me what you see in the video briefly.",
    "Describe the scene in the video briefly.",
]


def prepare_multimodal_inputs(model_dir: str,
                              model_type: str,
                              modality: str,
                              prompts: List[str],
                              media: List[str],
                              image_data_format: str = "pt",
                              num_frames: int = 8) -> List[Dict[str, Any]]:

    inputs = []
    if modality == "image":
        inputs = default_image_loader(prompts, media, image_data_format)
    elif modality == "video":
        inputs = default_video_loader(prompts, media, image_data_format,
                                      num_frames)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    inputs = INPUT_FORMATTER_MAP[model_type](model_dir, inputs)

    return inputs


def add_multimodal_args(parser):
    parser.add_argument("--model_type",
                        type=str,
                        choices=INPUT_FORMATTER_MAP.keys(),
                        help="Model type.")
    parser.add_argument("--modality",
                        type=str,
                        choices=["image", "video"],
                        default="image",
                        help="Media type.")
    parser.add_argument("--media",
                        type=str,
                        nargs="+",
                        help="A single or a list of media filepaths / urls.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=8,
                        help="The number of video frames to be sampled.")
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    parser = add_multimodal_args(parser)
    args = parser.parse_args()

    args.kv_cache_enable_block_reuse = False  # kv cache reuse does not work for multimodal, force overwrite
    if args.kv_cache_fraction is None:
        args.kv_cache_fraction = 0.6  # lower the default kv cache fraction for multimodal

    return args


def main():
    args = parse_arguments()

    llm, sampling_params = setup_llm(args)

    image_format = "pt"  # ["pt", "pil"]
    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = json.load(
            open(os.path.join(llm._hf_model_dir, 'config.json')))['model_type']
    assert model_type in INPUT_FORMATTER_MAP, f"Unsupported model_type: {model_type}"

    inputs = prepare_multimodal_inputs(args.model_dir, model_type,
                                       args.modality, args.prompt, args.media,
                                       image_format, args.num_frames)

    outputs = llm.generate(inputs, sampling_params)

    for i, output in enumerate(outputs):
        prompt = inputs[i]['prompt']
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
