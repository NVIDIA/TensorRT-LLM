#!/usr/bin/env python3
"""
LongBench v2 evaluation script with TensorRT-LLM and sparse attention.

Usage:
    python eval_longbench_v2.py --model_path /path/to/model --longbench_path ./LongBench --output_dir results/

    # Run all LongBench v2 samples
    python eval_longbench_v2.py --model_path /path/to/model --longbench_path ./LongBench --output_dir results/

    # Enable CoT reasoning
    python eval_longbench_v2.py --model_path /path/to/model --longbench_path ./LongBench --output_dir results/ --cot

    # Run with different difficulty levels
    python eval_longbench_v2.py --model_path /path/to/model --longbench_path ./LongBench --output_dir results/ --difficulty easy
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

# Add tensorrt_llm imports
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, RocketSparseAttentionConfig
from tensorrt_llm.logger import logger

# Chat templates mapping
CHAT_TEMPLATES = {
    "llama3.1-8b-instruct": "llama3",
    "llama3-8b-instruct": "llama3",
    "mistral-7b-instruct-v0.2": "mistral",
    "longchat-7b-v1.5-32k": "vicuna"
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LongBench v2 evaluation with TensorRT-LLM and RocketKV")

    # Model and data arguments
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to model (HF model name or local path)')
    parser.add_argument('--longbench_path',
                        type=str,
                        default='./LongBench',
                        help='Path to LongBench directory')

    # Output arguments
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Directory to save results')
    parser.add_argument('--exp_name',
                        type=str,
                        default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Model configuration
    parser.add_argument('--attention_backend',
                        type=str,
                        default='VANILLA',
                        choices=['VANILLA', 'TRTLLM', 'FLASHINFER'],
                        help='Attention backend to use')
    parser.add_argument('--backend',
                        type=str,
                        default='pytorch',
                        choices=['pytorch', 'tensorrt'],
                        help='LLM backend to use')
    parser.add_argument('--chat_template',
                        type=str,
                        default='auto',
                        help='Chat template to use (auto-detect if "auto")')

    # Sequence and batch configuration
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=133120,
                        help='Maximum sequence length')
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=1,
                        help='Maximum batch size')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=256,
                        help='Maximum new tokens to generate')
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=133120,
        help='Maximum total tokens across all sequences in a batch')
    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=1,
                        help='Tensor parallel size')

    # RocketKV configuration
    parser.add_argument('--rocket_sparse',
                        action='store_true',
                        help='Use rocket sparse attention')
    parser.add_argument('--token_budget',
                        type=int,
                        default=2048,
                        help='Token budget for RocketKV (prompt_budget)')
    parser.add_argument('--window_size',
                        type=int,
                        default=32,
                        help='Window size for RocketKV')
    parser.add_argument('--kernel_size',
                        type=int,
                        default=63,
                        help='Kernel size for RocketKV')
    parser.add_argument('--topr',
                        type=int,
                        default=90,
                        help='Top-r for RocketKV')

    # KV cache configuration
    parser.add_argument('--kv_cache_dtype',
                        type=str,
                        default='auto',
                        help='KV cache data type')
    parser.add_argument('--kv_cache_fraction',
                        type=float,
                        default=0.7,
                        help='Fraction of GPU memory for KV cache')

    # LongBench v2 specific arguments
    parser.add_argument('--cot',
                        action='store_true',
                        help='Enable Chain-of-Thought reasoning')
    parser.add_argument('--no_context',
                        action='store_true',
                        help='Test without long context (pure memorization)')
    parser.add_argument('--rag',
                        type=int,
                        default=0,
                        help='Use top-N retrieved contexts (0 to disable)')

    # Evaluation parameters
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--start_idx',
                        type=int,
                        default=0,
                        help='Start index for evaluation')
    parser.add_argument('--difficulty',
                        type=str,
                        choices=['easy', 'hard'],
                        default=None,
                        help='Filter by difficulty level')
    parser.add_argument('--length',
                        type=str,
                        choices=['short', 'medium', 'long'],
                        default=None,
                        help='Filter by length category')
    parser.add_argument('--domain',
                        type=str,
                        default=None,
                        help='Filter by domain')
    parser.add_argument('--max_len',
                        type=int,
                        default=120000,
                        help='Maximum prompt length in tokens for truncation')

    # System arguments
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Logging level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def load_longbench_v2_config(longbench_path: str) -> Dict[str, Any]:
    """Load LongBench v2 configuration files."""
    config_dir = os.path.join(longbench_path, "config")

    # Load model2maxlen.json for v2
    maxlen_file = os.path.join(config_dir, "model2maxlen.json")
    with open(maxlen_file, 'r', encoding='utf-8') as f:
        model2maxlen = json.load(f)

    # Load prompt templates
    prompts_dir = os.path.join(longbench_path, "prompts")

    templates = {}
    template_files = {
        '0shot': '0shot.txt',
        '0shot_cot': '0shot_cot.txt',
        '0shot_cot_ans': '0shot_cot_ans.txt',
        '0shot_no_context': '0shot_no_context.txt',
        '0shot_rag': '0shot_rag.txt'
    }

    for template_name, filename in template_files.items():
        template_path = os.path.join(prompts_dir, filename)
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                templates[template_name] = f.read()

    return {'model2maxlen': model2maxlen, 'templates': templates}


def build_chat(tokenizer, prompt, chat_template):
    """Build chat prompt following LongBench's approach."""
    if chat_template == "vicuna" or chat_template == "longchat":
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        except ImportError:
            # Fallback if fastchat is not available
            prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {prompt}\nASSISTANT:"
    elif chat_template == "llama2":
        prompt = f"[INST]{prompt}[/INST]"
    elif chat_template == "llama3":
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif chat_template == "mistral":
        prompt = f"[INST] {prompt} [/INST]"
    # For other templates or "none", return prompt as-is
    return prompt


def determine_chat_template(model_path: str, chat_template: str) -> str:
    """Determine chat template based on model path."""
    if chat_template != 'auto':
        return chat_template

    model_path_lower = model_path.lower()

    for model_key, template in CHAT_TEMPLATES.items():
        if model_key.replace('-', '').replace('.',
                                              '') in model_path_lower.replace(
                                                  '-', '').replace('.', ''):
            return template

    # Default fallback
    if 'llama' in model_path_lower:
        return 'llama3'
    elif 'mistral' in model_path_lower:
        return 'mistral'
    else:
        return 'none'  # No special formatting


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from response following LongBench v2's approach."""
    response = response.replace('*', '')

    # Try to extract answer in format "The correct answer is (X)"
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)

    # Try to extract answer in format "The correct answer is X"
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)

    # Try to extract any single letter A-D
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)

    return None


def post_process(pred: str, chat_template: str) -> str:
    """Post-process prediction following LongBench v2's approach."""
    pred = pred.split("</s")[0].strip()
    if chat_template == "qwen":
        pred = pred.split("<|im_end|>")[0]
    elif "llama2" in chat_template.lower():
        pred = (pred.split("(Document")[0].split("\n\nQuestion")[0].split(
            "\n\nAnswer")[0].split("[INST]")[0].split("[/INST]")[0].split(
                "(Passage")[0].strip())

    return pred


def truncate_prompt(prompt: str, tokenizer: AutoTokenizer, max_len: int) -> str:
    """Truncate prompt following LongBench v2's approach."""
    # Encode the prompt using the tokenizer
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # If prompt exceeds max_len, truncate by taking first half and last half
    if len(input_ids) > max_len:
        half = max_len // 2
        truncated_ids = input_ids[:half] + input_ids[-half:]
        # Decode back to text
        prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    return prompt


def format_prompt(sample: Dict[str, Any], template: str,
                  args: argparse.Namespace) -> str:
    """Format prompt for LongBench v2."""
    context = sample['context']

    # Handle RAG mode
    if args.rag > 0 and 'retrieved_context' in sample:
        retrieved = sample["retrieved_context"][:args.rag]
        retrieved = sorted(retrieved, key=lambda x: x.get('c_idx', 0))
        context = '\n\n'.join([
            f"Retrieved chunk {idx+1}: {x['content']}"
            for idx, x in enumerate(retrieved)
        ])

    # Handle no context mode
    if args.no_context:
        context = ""

    # Format the prompt using the template
    prompt = template.replace('$DOC$', context.strip())
    prompt = prompt.replace('$Q$', sample['question'].strip())
    prompt = prompt.replace('$C_A$', sample['choice_A'].strip())
    prompt = prompt.replace('$C_B$', sample['choice_B'].strip())
    prompt = prompt.replace('$C_C$', sample['choice_C'].strip())
    prompt = prompt.replace('$C_D$', sample['choice_D'].strip())

    return prompt


def initialize_llm(args: argparse.Namespace) -> Tuple[LLM, AutoTokenizer]:
    """Initialize LLM and tokenizer."""
    logger.info(f"Initializing LLM with model: {args.model_path}")

    try:
        # Configure KV cache
        kv_cache_config = KvCacheConfig(
            enable_block_reuse=False,  # RocketKV doesn't support KV cache reuse
        )

        if args.rocket_sparse:
            # Configure RocketKV sparse attention
            sparse_attention_config = RocketSparseAttentionConfig(
                window_size=args.window_size,
                kernel_size=args.kernel_size,
                prompt_budget=args.token_budget,
                topr=args.topr,
            )
            logger.info(f"Using RocketKV sparse attention")
        else:
            sparse_attention_config = None
            logger.info("Using standard attention")

        # Initialize LLM
        llm = LLM(
            model=args.model_path,
            backend=args.backend,
            kv_cache_config=kv_cache_config,
            attn_backend=args.attention_backend,
            sparse_attention_config=sparse_attention_config,
            tensor_parallel_size=args.tensor_parallel_size,
            max_seq_len=args.max_seq_len,
            max_num_tokens=args.max_num_tokens,
            cuda_graph_config=None,
            torch_compile_config=None,
        )

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        logger.info("LLM and tokenizer initialized successfully")

        return llm, tokenizer

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise e


def evaluate_longbench_v2(llm: LLM, tokenizer: AutoTokenizer,
                          args: argparse.Namespace) -> Tuple[List[Dict], float]:
    """Evaluate on LongBench v2 dataset."""

    # Load LongBench v2 configuration
    config = load_longbench_v2_config(args.longbench_path)

    # Determine max_len for the model if not explicitly set via arguments
    model_name = os.path.basename(args.model_path)
    if model_name in config[
            'model2maxlen']:  # Use default from config if available
        max_len = config['model2maxlen'][model_name]
        logger.info(f"Using model-specific max_len: {max_len} for {model_name}")
    else:
        max_len = args.max_len
        logger.info(f"Using max_len: {max_len}")

    # Update args with the determined max_len
    args.max_len = max_len

    # Load dataset
    logger.info(f"Loading LongBench v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2',
                           split='train',
                           trust_remote_code=True)
    data = [item for item in dataset]

    # Apply filters
    filtered_data = data

    if args.difficulty:
        filtered_data = [
            item for item in filtered_data
            if item['difficulty'] == args.difficulty
        ]
        logger.info(
            f"Filtered by difficulty '{args.difficulty}': {len(filtered_data)} samples"
        )

    if args.length:
        filtered_data = [
            item for item in filtered_data if item['length'] == args.length
        ]
        logger.info(
            f"Filtered by length '{args.length}': {len(filtered_data)} samples")

    if args.domain:
        filtered_data = [
            item for item in filtered_data if item['domain'] == args.domain
        ]
        logger.info(
            f"Filtered by domain '{args.domain}': {len(filtered_data)} samples")

    # Apply start_idx and num_samples
    if args.num_samples:
        end_idx = min(args.start_idx + args.num_samples, len(filtered_data))
        filtered_data = filtered_data[args.start_idx:end_idx]
    else:
        filtered_data = filtered_data[args.start_idx:]

    logger.info(f"Final dataset size: {len(filtered_data)} samples")

    # Determine chat template
    chat_template = determine_chat_template(args.model_path, args.chat_template)
    logger.info(f"Using chat template: {chat_template}")

    logger.info(f"Prepare and truncate prompts...")
    # Select appropriate template
    if args.no_context:
        template_key = '0shot_no_context'
    elif args.rag > 0:
        template_key = '0shot_rag'
    elif args.cot:
        template_key = '0shot_cot'
    else:
        template_key = '0shot'

    template = config['templates'][template_key]
    logger.info(f"Using template: {template_key}")

    # Set up extra end token ids
    extra_end_token_ids = []
    if chat_template == "llama3":
        eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
        extra_end_token_ids.append(eot_id)
        logger.info(f"Added llama3 end token: {eot_id}")

    if chat_template == "qwen":
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        extra_end_token_ids.append(im_end_id)
        logger.info(f"Added qwen end token: {im_end_id}")

    # Prepare prompts
    prompts = []
    for sample in filtered_data:
        formatted_prompt = format_prompt(sample, template, args)

        # Apply chat formatting if needed
        if chat_template != 'none':
            formatted_prompt = build_chat(tokenizer, formatted_prompt,
                                          chat_template)

        # Apply truncation if prompt is too long
        formatted_prompt = truncate_prompt(formatted_prompt, tokenizer,
                                           args.max_len)

        prompts.append(formatted_prompt)

    if len(prompts) == 0:
        logger.warning(f"No prompts to evaluate")
        return [], 0.0

    # Run inference
    logger.info(f"Starting inference for {len(prompts)} samples...")
    start_time = time.time()

    # Set sampling parameters
    max_new_tokens = 1024 if args.cot else 256
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.95,
        stop_token_ids=extra_end_token_ids if extra_end_token_ids else None,
    )

    outputs = llm.generate(prompts, sampling_params)

    inference_time = time.time() - start_time
    logger.info(
        f"Inference completed in {inference_time:.2f} seconds, average time per sample: {inference_time/len(prompts):.3f} seconds"
    )

    # Process outputs
    results = []
    for i, (sample, output) in enumerate(zip(filtered_data, outputs)):
        prediction = output.outputs[0].text.strip()
        processed_prediction = post_process(prediction, chat_template)

        # Handle CoT mode
        if args.cot:
            # For CoT, we need to do a second inference to extract the final answer
            cot_response = processed_prediction

            # Format the CoT answer extraction prompt
            cot_ans_template = config['templates']['0shot_cot_ans']
            cot_ans_prompt = format_prompt(sample, cot_ans_template, args)
            cot_ans_prompt = cot_ans_prompt.replace('$COT$', cot_response)

            if chat_template != 'none':
                cot_ans_prompt = build_chat(tokenizer, cot_ans_prompt,
                                            chat_template)

            # Apply truncation to CoT answer extraction prompt
            cot_ans_prompt = truncate_prompt(cot_ans_prompt, tokenizer,
                                             args.max_len)

            # Generate final answer
            ans_sampling_params = SamplingParams(
                max_tokens=128,
                temperature=0.1,
                top_p=0.95,
                stop_token_ids=extra_end_token_ids
                if extra_end_token_ids else None,
            )

            ans_output = llm.generate([cot_ans_prompt], ans_sampling_params)[0]
            final_prediction = post_process(ans_output.outputs[0].text.strip(),
                                            chat_template)

            extracted_answer = extract_answer(final_prediction)
        else:
            extracted_answer = extract_answer(processed_prediction)

        # Calculate accuracy
        is_correct = extracted_answer == sample[
            'answer'] if extracted_answer else False

        result = {
            '_id': sample['_id'],
            'domain': sample['domain'],
            'sub_domain': sample['sub_domain'],
            'difficulty': sample['difficulty'],
            'length': sample['length'],
            'question': sample['question'],
            'choice_A': sample['choice_A'],
            'choice_B': sample['choice_B'],
            'choice_C': sample['choice_C'],
            'choice_D': sample['choice_D'],
            'answer': sample['answer'],
            'prediction': processed_prediction,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            'context_length': len(sample['context']),
            'prompt_length': len(output.prompt_token_ids),
            'output_length': len(output.outputs[0].token_ids),
        }

        if args.cot:
            result['cot_response'] = cot_response
            result['final_prediction'] = final_prediction

        results.append(result)

    return results, inference_time


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate evaluation metrics for LongBench v2."""
    if not results:
        return {}

    total_samples = len(results)
    correct_samples = sum(1 for r in results if r['is_correct'])
    overall_accuracy = correct_samples / total_samples

    metrics = {
        'overall_accuracy': round(overall_accuracy * 100, 2),
        'total_samples': total_samples,
        'correct_samples': correct_samples
    }

    # Calculate metrics by difficulty
    difficulties = ['easy', 'hard']
    for difficulty in difficulties:
        diff_results = [r for r in results if r['difficulty'] == difficulty]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r['is_correct'])
            metrics[f'{difficulty}_accuracy'] = round(
                (diff_correct / len(diff_results)) * 100, 2)

    # Calculate metrics by length
    lengths = ['short', 'medium', 'long']
    for length in lengths:
        len_results = [r for r in results if r['length'] == length]
        if len_results:
            len_correct = sum(1 for r in len_results if r['is_correct'])
            metrics[f'{length}_accuracy'] = round(
                (len_correct / len(len_results)) * 100, 2)

    # Calculate metrics by domain
    domains = list(set(r['domain'] for r in results))
    for domain in domains:
        domain_results = [r for r in results if r['domain'] == domain]
        if domain_results:
            domain_correct = sum(1 for r in domain_results if r['is_correct'])
            metrics[f'{domain}_accuracy'] = round(
                (domain_correct / len(domain_results)) * 100, 2)

    return metrics


def save_results(results: List[Dict], args: argparse.Namespace,
                 inference_time: float, output_dir: str):
    """Save evaluation results in format compatible with LongBench v2."""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate metrics
    metrics = calculate_metrics(results)
    logger.info(f"Evaluation metrics: {metrics}")

    # Save detailed results
    results_file = os.path.join(output_dir, "longbench_v2_results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    # Save prediction results in LongBench v2 format
    pred_file = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_file, 'w', encoding='utf-8') as f:
        for result in results:
            pred_data = {
                "_id": result['_id'],
                "prediction": result['extracted_answer'],
                "response": result['prediction'],
                "judge": result['is_correct']
            }
            if args.cot:
                pred_data['cot_response'] = result.get('cot_response', '')
                pred_data['final_prediction'] = result.get(
                    'final_prediction', '')

            json.dump(pred_data, f, ensure_ascii=False)
            f.write('\n')

    # Create summary
    summary = {
        'experiment_config': {
            'model_path': args.model_path,
            'attention_backend': args.attention_backend,
            'rocket_sparse': args.rocket_sparse,
            'token_budget': args.token_budget,
            'cot': args.cot,
            'no_context': args.no_context,
            'rag': args.rag,
            'difficulty_filter': args.difficulty,
            'length_filter': args.length,
            'domain_filter': args.domain,
            'max_seq_len': args.max_seq_len,
            'max_new_tokens': args.max_new_tokens
        },
        'evaluation_results': metrics,
        'timing': {
            'total_inference_time': inference_time,
            'avg_inference_time':
            inference_time / len(results) if results else 0,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    }

    # Save summary
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_dir}")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_arguments()
    logger.set_level(args.log_level)

    # Setup experiment name
    if not args.exp_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(args.model_path).replace('/', '_')
        args.exp_name = f"longbench_v2_{model_name}_{timestamp}"

    output_dir = os.path.join(args.output_dir, args.exp_name)

    logger.info(
        "=========== LongBench v2 Evaluation with TensorRT-LLM ===========")

    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to {config_file}")

    # Initialize LLM and tokenizer
    llm, tokenizer = initialize_llm(args)

    # Run evaluation
    logger.info(f"Starting LongBench v2 evaluation...")
    results, inference_time = evaluate_longbench_v2(llm, tokenizer, args)

    # Save results and get metrics
    metrics = save_results(results, args, inference_time, output_dir)

    logger.info(f"{'-'*80}")
    logger.info(f"Evaluation completed successfully!")
    logger.info(f"Total time: {inference_time:.2f} seconds")
    logger.info(f"Total samples: {len(results)}")

    if metrics:
        logger.info(
            f"Overall accuracy: {metrics.get('overall_accuracy', 'N/A')}%")

        if 'easy_accuracy' in metrics:
            logger.info(
                f"Easy difficulty accuracy: {metrics['easy_accuracy']}% ({metrics.get('easy_samples', 0)} samples)"
            )
        if 'hard_accuracy' in metrics:
            logger.info(
                f"Hard difficulty accuracy: {metrics['hard_accuracy']}% ({metrics.get('hard_samples', 0)} samples)"
            )

        for length in ['short', 'medium', 'long']:
            if f'{length}_accuracy' in metrics:
                logger.info(
                    f"{length.capitalize()} length accuracy: {metrics[f'{length}_accuracy']}% ({metrics.get(f'{length}_samples', 0)} samples)"
                )


if __name__ == '__main__':
    main()
