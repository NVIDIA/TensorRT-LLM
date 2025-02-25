import argparse

import tensorrt_llm.bindings
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm.inputs import load_image, load_video

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


def prepare_vila(inputs):

    def add_media_token(prompt, multi_modal_data):
        mm_tokens = ""
        if "image" in multi_modal_data:
            for _ in multi_modal_data["image"]:
                mm_tokens += "<image>"
        elif "video" in multi_modal_data:
            for _ in multi_modal_data["video"]:
                mm_tokens += "<vila/video>"
        return mm_tokens + prompt

    for input in inputs:
        input["prompt"] = add_media_token(input["prompt"],
                                          input["multi_modal_data"])
    return inputs


MODEL_TYPE_MAP = {
    "vila": prepare_vila,
}


def main(args):

    if args.modality == "image":
        prompts = args.prompt if args.prompt else example_image_prompts
        images = args.media if args.media else example_images
        if len(images) > len(prompts) and len(prompts) == 1:
            # 1 prompt + N media
            images = [images]
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": [load_image(i) for i in image] if isinstance(
                    image, list) else [load_image(image)]
            }
        } for prompt, image in zip(prompts, images)]
    elif args.modality == "video":
        prompts = args.prompt if args.prompt else example_video_prompts
        videos = args.media if args.media else example_videos
        if len(videos) > len(prompts) and len(prompts) == 1:
            # 1 prompt + N media
            videos = [videos]
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "video":
                [load_video(i, args.num_frames) for i in video] if isinstance(
                    video, list) else [load_video(video, args.num_frames)]
            }
        } for prompt, video in zip(prompts, videos)]
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")

    inputs = MODEL_TYPE_MAP[args.model_type](inputs)

    llm = LLM(
        model=args.model_dir,
        kv_cache_config=tensorrt_llm.bindings.executor.KvCacheConfig(
            free_gpu_memory_fraction=args.kv_cache_fraction),
    )

    outputs = llm.generate(inputs=inputs,
                           sampling_params=SamplingParams(
                               max_tokens=args.max_tokens,
                               temperature=args.temperature,
                               top_p=args.top_p,
                           ))

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(
            f"[{i}] Prompt: {inputs[i]['prompt']!r}, Generated text: {generated_text!r}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Model checkpoint directory.")
    parser.add_argument("--model_type",
                        type=str,
                        choices=MODEL_TYPE_MAP.keys(),
                        help="Model type.")
    parser.add_argument("--modality",
                        type=str,
                        choices=["image", "video"],
                        help="Media type.")
    parser.add_argument("--prompt",
                        type=str,
                        nargs="+",
                        help="A single or a list of text prompts.")
    parser.add_argument("--media",
                        type=str,
                        nargs="+",
                        help="A single or a list of media filepaths / urls.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=8,
                        help="The number of video frames to be sampled.")
    parser.add_argument("--kv_cache_fraction", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()
    main(args)
