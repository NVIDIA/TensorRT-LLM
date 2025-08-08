import argparse
import json
import os
from typing import List, Optional, Union

from quickstart_advanced import add_llm_args, setup_llm

from tensorrt_llm.inputs import (ALL_SUPPORTED_MULTIMODAL_MODELS,
                                 default_multimodal_input_loader)

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

# Multiturn conversation examples configuration
multiturn_conversation_examples = {
    "image": {
        "conversations": [{
            "user":
            "Describe what you see in this image.",
            "assistant":
            "I can see a beautiful coastal scene with a sandy beach, clear blue water, and a rocky shoreline. The image shows a peaceful natural environment with waves gently lapping against the shore."
        }, {
            "user":
            "What colors are most prominent in this scene?",
            "assistant":
            "The most prominent colors in this coastal scene are various shades of blue from the ocean water, golden/beige tones from the sandy beach, and natural brown/gray colors from the rocky shoreline."
        }, {
            "user":
            "Based on what you see, what time of day might this be?",
            "assistant":
            "Based on the lighting and colors in the image, this appears to be taken during daylight hours, likely in the afternoon when the sun provides good illumination of the coastal landscape."
        }],
        "media": [
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png"
        ]
    },
    "video": {
        "conversations": [{
            "user":
            "What's happening in this video?",
            "assistant":
            "This video shows a dynamic urban scene with people walking and moving through what appears to be a busy city environment, possibly Tokyo based on the setting."
        }, {
            "user":
            "How would you describe the atmosphere of this scene?",
            "assistant":
            "The atmosphere appears to be lively and energetic, with people actively moving through the urban space, creating a sense of busy city life and human activity."
        }],
        "media": [
            "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4"
        ]
    },
    "audio": {
        "conversations": [{
            "user":
            "Transcribe the audio clip into text, please don't add other text.",
            "assistant": "What is the traffic sign in the image?"
        }, {
            "user":
            "What type of question is being asked in the audio?",
            "assistant":
            "The audio contains a question asking about identifying a traffic sign in an image, which is a visual recognition task."
        }],
        "media": [
            "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_the_traffic_sign_in_the_image.wav"
        ]
    },
    "mixture_text_image": {
        "conversations": [{
            "user":
            "Who invented the internet?",
            "assistant":
            "The internet was not invented by a single person, but rather developed through collaborative efforts over several decades. Key contributors include Vint Cerf and Bob Kahn, who developed TCP/IP protocols in the 1970s, and Tim Berners-Lee, who created the World Wide Web in 1989."
        }, {
            "user":
            "Now look at this image and describe what you see.",
            "assistant":
            "I can see an image that appears to show a painting or artistic scene, likely created using digital tools or artistic techniques."
        }, {
            "user":
            "How does this image relate to the previous topic about the internet?",
            "assistant":
            "This image appears to be unrelated to the internet topic we discussed earlier. It seems to be an artistic or creative work, while our previous conversation was about the technical development of the internet infrastructure."
        }],
        "media":
        [[],
         [
             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
         ],
         [
             "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
         ]]
    }
}


class MultimodalConversationManager:
    """Class for managing multimodal multiturn conversations"""

    def __init__(self,
                 llm,
                 tokenizer,
                 model_type,
                 sampling_params,
                 debug_prompt=False):
        self.llm = llm
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.sampling_params = sampling_params
        self.debug_prompt = debug_prompt
        self.conversation_history = []
        self.current_media = None
        self.current_modality = None

    def add_message(self,
                    role: str,
                    content: str,
                    media: Optional[Union[str, List[str]]] = None):
        """Add message to conversation history"""
        message = {"role": role, "content": content, "media": media}
        self.conversation_history.append(message)

    def get_conversation_text(self) -> str:
        """Get formatted conversation text for context"""
        conversation_text = ""
        for msg in self.conversation_history:
            if msg["role"] == "user":
                conversation_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_text += f"Assistant: {msg['content']}\n"
        return conversation_text

    def build_conversation_prompt(self, user_input: str) -> str:
        """Build conversation prompt with proper formatting"""
        conversation_context = self.get_conversation_text()

        if conversation_context:
            # Add the new user input to the conversation
            full_prompt = conversation_context + f"User: {user_input}\nAssistant:"
        else:
            # First message in conversation
            full_prompt = f"User: {user_input}\nAssistant:"

        return full_prompt

    def generate_response(self,
                          user_input: str,
                          media: Optional[Union[str, List[str]]] = None) -> str:
        """Generate response to user input with conversation context"""
        # Build conversation prompt with proper formatting
        full_prompt = self.build_conversation_prompt(user_input)

        # Debug: Show the full prompt if debug mode is enabled
        if self.debug_prompt:
            print(f"\n=== DEBUG: Full Prompt ===")
            print(f"Prompt: {repr(full_prompt)}")
            print(f"Media: {media}")
            print("=" * 40)

        # Use multimodal input loader to process input with conversation context
        inputs = default_multimodal_input_loader(
            tokenizer=self.tokenizer,
            model_dir=self.llm._hf_model_dir,
            model_type=self.model_type,
            modality=self._determine_modality(media),
            prompts=[full_prompt],
            media=[media] if media else None,
            image_data_format="pt",
            num_frames=8,
            device="cpu")

        # Generate response
        outputs = self.llm.generate(inputs, self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Add user message and assistant response to history
        self.add_message("user", user_input, media)
        self.add_message("assistant", response)

        return response

    def _determine_modality(self, media) -> str:
        """Determine modality type based on media content"""
        if not media:
            return "text"
        if isinstance(media, str):
            if media.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                return "image"
            elif media.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return "video"
            elif media.endswith(('.wav', '.mp3', '.flac', '.aac')):
                return "audio"
        elif isinstance(media, list):
            if len(media) == 0:
                return "text"
            elif len(media) == 1:
                return self._determine_modality(media[0])
            else:
                # Multiple images case
                return "multiple_image"
        return "text"

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.current_media = None
        self.current_modality = None

    def print_conversation_history(self):
        """Print current conversation history for debugging"""
        print("\n=== Current Conversation History ===")
        if not self.conversation_history:
            print("No conversation history yet.")
        else:
            for i, msg in enumerate(self.conversation_history):
                print(f"{i+1}. {msg['role'].title()}: {msg['content']}")
                if msg.get('media'):
                    print(f"   Media: {msg['media']}")
        print("=" * 40)


def add_multimodal_args(parser):
    parser.add_argument("--model_type",
                        type=str,
                        choices=ALL_SUPPORTED_MULTIMODAL_MODELS,
                        help="Model type.")
    parser.add_argument("--modality",
                        type=str,
                        choices=[
                            "image", "video", "audio", "image_audio",
                            "multiple_image", "mixture_text_image"
                        ],
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
    parser.add_argument("--image_format",
                        type=str,
                        choices=["pt", "pil"],
                        default="pt",
                        help="The format of the image.")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="The device to have the input on.")
    # Add multiturn conversation related parameters
    parser.add_argument("--multiturn",
                        action="store_true",
                        help="Enable multi-turn conversation mode.")
    parser.add_argument(
        "--conversation_turns",
        type=int,
        default=3,
        help="Number of conversation turns for automated testing.")
    parser.add_argument(
        "--debug_prompt",
        action="store_true",
        help="Show the full prompt being sent to the model for debugging.")
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
    parser = add_multimodal_args(parser)
    parser = add_lora_args(parser)
    args = parser.parse_args()

    args.disable_kv_cache_reuse = True  # kv cache reuse does not work for multimodal, force overwrite
    if args.kv_cache_fraction is None:
        args.kv_cache_fraction = 0.6  # lower the default kv cache fraction for multimodal

    return args


def run_multiturn_conversation_example(
        conversation_manager: MultimodalConversationManager, modality: str):
    """Run predefined multiturn conversation examples"""
    if modality not in multiturn_conversation_examples:
        print(f"No predefined conversation example for modality: {modality}")
        return

    example = multiturn_conversation_examples[modality]
    conversations = example["conversations"]
    media_list = example["media"]

    print(f"\n=== Starting {modality} multiturn conversation example ===")

    for i, (conv, media) in enumerate(zip(conversations, media_list)):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {conv['user']}")

        # Generate response
        response = conversation_manager.generate_response(conv['user'], media)
        print(f"Assistant: {response}")

        # Show expected response (for comparison)
        print(f"Expected response: {conv['assistant']}")

        # Show conversation context after each turn
        if i < len(conversations) - 1:  # Don't show after the last turn
            print("\nCurrent conversation context:")
            conversation_manager.print_conversation_history()

        print("-" * 50)


def main():
    args = parse_arguments()

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
            open(os.path.join(llm._hf_model_dir, 'config.json')))['model_type']
    assert model_type in ALL_SUPPORTED_MULTIMODAL_MODELS, f"Unsupported model_type: {model_type}"

    # If multiturn mode is enabled
    if args.multiturn:
        # Create conversation manager
        conversation_manager = MultimodalConversationManager(
            llm=llm,
            tokenizer=llm.tokenizer,
            model_type=model_type,
            sampling_params=sampling_params,
            debug_prompt=args.debug_prompt)

        # Run predefined multiturn conversation examples
        run_multiturn_conversation_example(conversation_manager, args.modality)
        return

    # Original single-turn processing logic
    # set prompts and media to example prompts and images if they are not provided
    if args.prompt is None:
        args.prompt = example_medias_and_prompts[args.modality]["prompt"]
    if args.media is None:
        args.media = example_medias_and_prompts[args.modality]["media"]
    inputs = default_multimodal_input_loader(tokenizer=llm.tokenizer,
                                             model_dir=llm._hf_model_dir,
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


if __name__ == "__main__":
    main()
