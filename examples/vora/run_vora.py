"""Example script for running VoRA models with TensorRT-LLM PyTorch backend."""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from PIL import Image

# Import VoRA support
from tensorrt_llm.models.vora.llm_integration import VoRALLM
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.logger import logger


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run VoRA model inference')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to VoRA model or HuggingFace model ID')
    parser.add_argument('--prompt',
                        type=str,
                        default="<image> Describe this image.",
                        help='Text prompt for generation')
    parser.add_argument('--image_path',
                        type=str,
                        help='Path to input image')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p',
                        type=float,
                        default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--batch_prompts',
                        type=str,
                        help='JSON file with batch prompts and images')
    parser.add_argument('--output_file',
                        type=str,
                        help='Output file for results')
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Logging level')
    
    return parser.parse_args()


def load_batch_inputs(batch_file: str) -> tuple:
    """Load batch inputs from JSON file.
    
    Expected format:
    {
        "prompts": ["prompt1", "prompt2", ...],
        "images": ["path1.jpg", "path2.jpg", ...]  # optional
    }
    """
    with open(batch_file, 'r') as f:
        data = json.load(f)
    
    prompts = data.get('prompts', [])
    images = data.get('images', None)
    
    return prompts, images


def main():
    args = parse_arguments()
    
    # Set logging level
    logger.set_level(args.log_level)
    
    # Initialize VoRA model
    logger.info(f"Loading VoRA model from {args.model_path}")
    llm = VoRALLM(
        model=args.model_path,
        dtype="float32",
        trust_remote_code=True
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Prepare inputs
    if args.batch_prompts:
        # Batch processing
        prompts, image_paths = load_batch_inputs(args.batch_prompts)
        images = None
        if image_paths:
            images = [Image.open(path) for path in image_paths]
    else:
        # Single prompt processing
        prompts = args.prompt
        images = None
        if args.image_path:
            images = Image.open(args.image_path)
    
    # Generate
    logger.info("Starting generation...")
    results = llm.generate(
        prompts=prompts,
        images=images,
        sampling_params=sampling_params
    )
    
    # Process results
    outputs = []
    for i, result in enumerate(results):
        output = {
            'prompt': result['prompt'],
            'generated_text': result['generated_text']
        }
        outputs.append(output)
        
        # Print result
        print(f"\n{'='*50}")
        print(f"Prompt {i+1}: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"{'='*50}\n")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")
    
    logger.info("Generation completed!")


def example_chat_interface():
    """Example of using VoRA with chat interface."""
    from tensorrt_llm.models.vora.llm_integration import VoRALLM
    
    # Initialize model
    llm = VoRALLM(
        model="Hon-Wong/VoRA-7B-Instruct",
        dtype="float16"
    )
    
    # Example 1: Text-only chat
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]
    response = llm.chat(messages)
    print(f"Response: {response}")
    
    # Example 2: Multimodal chat
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "path/to/image.jpg"},
                {"type": "text", "text": "What's in this image?"}
            ]
        }
    ]
    response = llm.chat(messages)
    print(f"Response: {response}")


def example_batch_processing():
    """Example of batch processing with VoRA."""
    from tensorrt_llm.models.vora.model_runner import create_vora_runner
    
    # Create runner
    runner = create_vora_runner(
        model_path="Hon-Wong/VoRA-7B-Instruct",
        device="cuda",
        dtype="float16",
        max_batch_size=4
    )
    
    # Prepare batch
    batch = {
        'prompts': [
            "Describe this image.",
            "What objects are visible?",
            "What is the main subject?",
            "Describe the colors in the image."
        ],
        'images': [
            "image1.jpg",
            "image2.jpg", 
            "image3.jpg",
            "image4.jpg"
        ]
    }
    
    # Process batch
    results = runner.process_batch(
        batch,
        max_new_tokens=256,
        temperature=0.7
    )
    
    for prompt, result in zip(batch['prompts'], results):
        print(f"Prompt: {prompt}")
        print(f"Result: {result}\n")


if __name__ == '__main__':
    main()