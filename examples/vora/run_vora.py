"""Example script for running VoRA models with TensorRT-LLM PyTorch backend."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image

# Disable flashinfer for float32 compatibility

# Import TensorRT-LLM components
from tensorrt_llm.logger import logger
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.sampling_params import SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run VoRA model inference')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to VoRA model or HuggingFace model ID')
    parser.add_argument('--prompt',
                        type=str,
                        default="Describe this image.",
                        help='Text prompt for generation (no need to include <image> token)')
    parser.add_argument('--image_path',
                        type=str,
                        default="/workspace/TensorRT-LLM/test.jpg",
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
                        default=0.8,
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
    
    # Initialize VoRA model using TensorRT-LLM LLM API
    logger.info(f"Loading VoRA model from {args.model_path}")
    
    # Initialize LLM with VoRA model using PyTorch backend
    logger.info("Initializing VoRA model through LLM API...")
    
    # Initialize LLM API for VoRA model
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",  # Use bfloat16
    )
    logger.info("✓ Model loaded successfully")
    
    # Get EOS token from tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    logger.info(f"Using EOS token ID: {eos_token_id}")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
        end_id=eos_token_id,  # Use actual EOS token from tokenizer
        stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
    )
    
    # Prepare inputs
    if args.batch_prompts:
        # Batch processing
        prompts, image_paths = load_batch_inputs(args.batch_prompts)
        images = []
        if image_paths:
            images = [Image.open(path) for path in image_paths]
    else:
        # Single prompt processing
        prompts = [args.prompt]
        images = []
        if args.image_path and Path(args.image_path).exists():
            images = [Image.open(args.image_path)]
            logger.info(f"Loaded image: {args.image_path}")
        elif args.image_path:
            logger.warning(f"Image not found: {args.image_path}")
    
    # Generate
    logger.info("Starting generation...")
    
    # Prepare prompts and images for LLM API
    if images:
        # Multimodal inputs
        prompt_inputs = []
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            prompt_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
    else:
        # Text-only inputs
        prompt_inputs = prompts
    
    # Generate with the LLM API
    outputs = llm.generate(
        prompt_inputs,
        sampling_params=sampling_params
    )
    
    # Format results
    results = []
    for i, output in enumerate(outputs):
        results.append({
            'prompt': prompts[i],
            'generated_text': output.outputs[0].text
        })
    
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


# Example functions removed - see documentation for usage examples


if __name__ == '__main__':
    main()