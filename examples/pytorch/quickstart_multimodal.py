import argparse
import json
import os

from quickstart_advanced import add_llm_args, setup_llm
from transformers import AutoProcessor

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


def prepare_vila(args, inputs):

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


def prepare_llava_next(args, inputs):
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # Single-image inference chat template. For multi-image template,
    # see https://huggingface.co/docs/transformers/en/model_doc/llava_next#multi-image-inference.
    def apply_template(prompt, multimodal_data):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image"
                    },
                ],
            },
        ]
        return processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

    for input in inputs:
        input["prompt"] = apply_template(input["prompt"],
                                         input["multi_modal_data"])
    return inputs


def prepare_qwen2_vl(args, inputs):
    processor = AutoProcessor.from_pretrained(args.model_dir)

    def apply_template(prompt, multimodal_data):
        content = [{
            "type": media_type
        } for media_type, items in multimodal_data.items()
                   for _ in items] + [{
                       "type": "text",
                       "text": prompt
                   }]

        conversation = [{"role": "user", "content": content}]
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    for input in inputs:
        input["prompt"] = apply_template(input["prompt"],
                                         input["multi_modal_data"])
    return inputs


MODEL_TYPE_MAP = {
    "llava_llama": prepare_vila,
    "llava_next": prepare_llava_next,
    "qwen2_vl": prepare_qwen2_vl,
    "qwen2_5_vl": prepare_qwen2_vl,
}


def add_multimodal_args(parser):
    parser.add_argument("--model_type",
                        type=str,
                        choices=MODEL_TYPE_MAP.keys(),
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
    if args.modality == "image":
        prompts = args.prompt if args.prompt else example_image_prompts
        images = args.media if args.media else example_images
        if len(images) > len(prompts) and len(prompts) == 1:
            # 1 prompt + N media
            images = [images]
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": [
                    load_image(i, format=image_format, device="cuda")
                    for i in image
                ] if isinstance(image, list) else
                [load_image(image, format=image_format, device="cuda")]
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
                "video": [
                    load_video(
                        i, args.num_frames, format=image_format, device="cuda")
                    for i in video
                ] if isinstance(video, list) else [
                    load_video(video,
                               args.num_frames,
                               format=image_format,
                               device="cuda")
                ]
            }
        } for prompt, video in zip(prompts, videos)]
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")

    model_type = json.load(open(os.path.join(llm._hf_model_dir,
                                             'config.json')))['model_type']
    assert model_type in MODEL_TYPE_MAP, f"Unsupported model_type: {model_type}"
    inputs = MODEL_TYPE_MAP[model_type](args, inputs)

    outputs = llm.generate(inputs, sampling_params)

    for i, output in enumerate(outputs):
        prompt = inputs[i]['prompt']
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
