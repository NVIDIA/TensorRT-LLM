import argparse
import json
import os

from quickstart_advanced import add_llm_args, setup_llm

from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from tensorrt_llm.tools.importlib_utils import import_custom_module_from_dir

example_medias_and_prompts = {
    "image": {
        "media": [
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
            "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
        ],
        "prompt": [
            "Describe the natural environment in the image.",
            "Describe the object and the weather condition in the image.",
            "Describe the traffic condition on the road in the image.",
        ]
    },
    "video": {
        "media": [
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4",
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4",
        ],
        "prompt": [
            "Tell me what you see in the video briefly.",
            "Describe the scene in the video briefly.",
        ]
    },
    "audio": {
        "media": [
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_the_traffic_sign_in_the_image.wav",
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav",
        ],
        "prompt": [
            "Transcribe the audio clip into text, please don't add other text.",
            "Transcribe the audio clip into text, please don't add other text.",
        ]
    },
    "image_audio": {
        "media": [
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
            ],
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
                "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
            ],
        ],
        "prompt": [
            "Describe the scene in the image briefly.",
            "",
        ]
    },
    "multiple_image": {
        "media": [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
            "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
        ],
        "prompt": ["Describe the difference between the two images."],
    },
    "mixture_text_image": {
        "media": [
            [],
            [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
            ],
        ],
        "prompt": [
            "Who invented the internet?",
            "Describe the scene in the image briefly.",
        ],
    },
}


def add_multimodal_args(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        choices=MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types(),
        help="Model type as specified in the HuggingFace model config.")
    parser.add_argument("--modality",
                        type=str,
                        choices=[
                            "image", "video", "audio", "image_audio",
                            "multiple_image", "mixture_text_image"
                        ],
                        default="image",
                        help="Media type being used for inference.")
    parser.add_argument("--media",
                        type=str,
                        nargs="+",
                        help="A single or a list of media filepaths / urls.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=8,
                        help="The number of video frames to be sampled.")
    parser.add_argument("--image_format",
                        type=str,
                        choices=["pt", "pil"],
                        default="pt",
                        help="The format of the image.")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="The device to have the input on.")
    parser.add_argument(
        "--custom_module_dirs",
        type=str,
        nargs="+",
        default=None,
        help=
        ("Paths to an out-of-tree model directory which should be imported."
         " This is useful to load a custom model. The directory should have a structure like:"
         " <model_name>"
         " ├── __init__.py"
         " ├── <model_name>.py"
         " └── <sub_dirs>"))
    # Add multiturn conversation related parameters
    parser.add_argument("--multiturn",
                        action="store_true",
                        help="Enable multi-turn conversation mode.")
    parser.add_argument(
        "--conversation_turns",
        type=int,
        default=2,
        help="Number of conversation turns for automated testing.")
    return parser


def add_lora_args(parser):
    parser.add_argument("--load_lora",
                        default=False,
                        action='store_true',
                        help="Whether to load the LoRA model.")
    parser.add_argument("--auto_model_name",
                        type=str,
                        default=None,
                        help="The auto model name in TRTLLM repo.")
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    parser.add_argument("--kv_cache_fraction", type=float, default=0.6)
    parser = add_multimodal_args(parser)
    parser = add_lora_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.custom_module_dirs is not None:
        for custom_module_dir in args.custom_module_dirs:
            try:
                import_custom_module_from_dir(custom_module_dir)
            except Exception as e:
                print(
                    f"Failed to import custom module from {custom_module_dir}: {e}"
                )
                raise e

    lora_config = None
    if args.load_lora:
        assert args.auto_model_name is not None, "Please provide the auto model name to load LoRA config."
        import importlib
        models_module = importlib.import_module('tensorrt_llm._torch.models')
        model_class = getattr(models_module, args.auto_model_name)
        lora_config = model_class.lora_config(args.model_dir)
        # For stability - explicitly set the LoRA GPU cache & CPU cache to have space for 2 adapters
        lora_config.max_loras = 2
        lora_config.max_cpu_loras = 2

    llm, sampling_params = setup_llm(args, lora_config=lora_config)

    image_format = args.image_format
    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = json.load(
            open(os.path.join(str(llm._hf_model_dir),
                              'config.json')))['model_type']
    assert model_type in MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types(), \
        f"Unsupported model_type: {model_type} found!\n" \
        f"Supported types: {MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types()}"

    # If multiturn mode is enabled
    if args.multiturn:
        # Run predefined multiturn conversation examples
        assert args.prompt is not None, "Please provide a prompt for multiturn conversation."
        assert args.media is not None, "Please provide media for multiturn conversation."
        # Determine how many turns to run
        max_turns = min(args.conversation_turns, len(args.prompt))
        generated_outputs = []  # Store generated outputs for return

        # Initialize conversation history with the first prompt
        conversation_history = args.prompt[0] if args.prompt else ""

        for i in range(max_turns):
            print(f"\n--- Turn {i+1} ---")

            try:
                # Use multimodal input loader to process input with conversation context
                # Use accumulated conversation history instead of just the current prompt
                cur_prompt = conversation_history
                inputs = default_multimodal_input_loader(
                    tokenizer=llm.tokenizer,
                    model_dir=llm._hf_model_dir,
                    model_type=model_type,
                    modality=args.modality,
                    prompts=[cur_prompt],
                    media=args.media,
                    image_data_format="pt",
                    num_frames=8,
                    device="cpu")

                lora_request = None
                if args.load_lora:
                    if model_class is None:
                        raise ValueError(
                            "model_class must be provided when load_lora is True"
                        )
                    lora_request = model_class.lora_request(
                        len(inputs), args.modality, llm._hf_model_dir)

                # Generate response
                outputs = llm.generate(inputs,
                                       sampling_params,
                                       lora_request=lora_request)
                assert outputs and len(
                    outputs) > 0 and outputs[0].outputs and len(
                        outputs[0].outputs) > 0
                response = outputs[0].outputs[0].text.strip()

                # Store generated output
                generated_outputs.append({
                    "turn": i + 1,
                    "user_input": cur_prompt,
                    "assistant_response": response,
                    "media": args.media
                })

                conversation_history = conversation_history + "\n" + response
                if i + 1 < len(args.prompt):
                    conversation_history = conversation_history + "\n" + args.prompt[
                        i + 1]

            except Exception as e:
                print(f"Error in turn {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        for i, output in enumerate(generated_outputs):
            print(
                f"[{i}] Prompt: {output['user_input']!r}, Generated text: {output['assistant_response']!r}"
            )
        return

    # Original single-turn processing logic
    # set prompts and media to example prompts and images if they are not provided
    if args.prompt is None:
        args.prompt = example_medias_and_prompts[args.modality]["prompt"]
    if args.media is None:
        args.media = example_medias_and_prompts[args.modality]["media"]
    inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                             model_dir=str(llm._hf_model_dir),
                                             model_type=model_type,
                                             modality=args.modality,
                                             prompts=args.prompt,
                                             media=args.media,
                                             image_data_format=image_format,
                                             num_frames=args.num_frames,
                                             device=args.device)

    lora_request = None
    if args.load_lora:
        lora_request = model_class.lora_request(len(inputs), args.modality,
                                                llm._hf_model_dir)

    outputs = llm.generate(
        inputs,
        sampling_params,
        lora_request=lora_request,
    )

    for i, output in enumerate(outputs):
        prompt = args.prompt[i]
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")
        if args.return_context_logits:
            print(f"[{i}] Context logits: {output.context_logits}")
        if args.return_generation_logits:
            print(
                f"[{i}] Generation logits: {output.outputs[0].generation_logits}"
            )
        if args.logprobs:
            print(f"[{i}] Logprobs: {output.outputs[0].logprobs}")


if __name__ == "__main__":
    main()
