#!/usr/bin/env python3
"""
LongBench v1 evaluation script with TensorRT-LLM and sparse attention.

Usage:
    python longbench_rocket_eval.py --dataset narrativeqa --model_path /path/to/model --longbench_path ./LongBench --output_dir results/

    # Run all LongBench tasks
    python longbench_rocket_eval.py --model_path /path/to/model --longbench_path ./LongBench --output_dir results/ --token_budget 2048 --rocket_sparse
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Add tensorrt_llm imports
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, KvCacheConfig, MoeConfig,
                                 RocketSparseAttentionConfig)
from tensorrt_llm.logger import logger

LONGBENCH_DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

# Task categorization
TASK_DATASETS = {
    'single_doc_qa':
    ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    'multi_doc_qa': ["hotpotqa", "2wikimqa", "musique", "dureader"],
    'summarization': ["gov_report", "qmsum", "multi_news", "vcsum"],
    'few_shots': ["trec", "triviaqa", "samsum", "lsht"],
    'synthetic':
    ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    'code': ["lcc", "repobench-p"]
}

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
        description="LongBench evaluation with TensorRT-LLM and RocketKV")

    # Model and data arguments
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to model (HF model name or local path)')
    parser.add_argument('--dataset',
                        type=str,
                        nargs='+',
                        choices=LONGBENCH_DATASETS,
                        help='LongBench datasets to evaluate on')
    parser.add_argument('--run_all_tasks',
                        action='store_true',
                        help='Run evaluation on all LongBench tasks')
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

    # Parallelism
    parser.add_argument('--moe_backend',
                        type=str,
                        default='CUTLASS',
                        choices=[
                            'CUTLASS', 'TRTLLM', 'VANILLA', 'WIDEEP',
                            'DEEPGEMM', 'CUTEDSL', 'TRITON'
                        ])
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')

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

    # Runtime
    parser.add_argument('--print_iter_log',
                        default=False,
                        action='store_true',
                        help='Print iteration logs during execution')
    parser.add_argument('--use_cuda_graph', default=False, action='store_true')
    parser.add_argument('--cuda_graph_padding_enabled',
                        default=False,
                        action='store_true')
    parser.add_argument('--cuda_graph_batch_sizes',
                        nargs='+',
                        type=int,
                        default=None)

    # Evaluation parameters
    parser.add_argument('--num_samples',
                        type=int,
                        default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--start_idx',
                        type=int,
                        default=0,
                        help='Start index for evaluation')

    # System arguments
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Logging level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Validation
    if not args.run_all_tasks and not args.dataset:
        parser.error("Must specify either --dataset or --run_all_tasks")

    return args


def setup_longbench_imports(longbench_path: str):
    """Add LongBench to Python path and import required modules."""
    longbench_dir = os.path.join(longbench_path, "LongBench")  # for v1
    if not os.path.exists(longbench_dir):
        raise FileNotFoundError(
            f"LongBench directory not found: {longbench_dir}")

    # Add to path
    if longbench_dir not in sys.path:
        sys.path.insert(0, longbench_dir)


def load_longbench_config(
        longbench_path: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Load LongBench configuration files."""
    config_dir = os.path.join(longbench_path, "LongBench", "config")

    # Load dataset2prompt.json
    prompt_file = os.path.join(config_dir, "dataset2prompt.json")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        dataset2prompt = json.load(f)

    # Load dataset2maxlen.json
    maxlen_file = os.path.join(config_dir, "dataset2maxlen.json")
    with open(maxlen_file, 'r', encoding='utf-8') as f:
        dataset2maxlen = json.load(f)

    return dataset2prompt, dataset2maxlen


# LongBench's build_chat function (simplified version)
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


def post_process(pred: str, chat_template: str, dataset: str) -> str:
    """Post-process prediction following LongBench's approach."""
    pred = pred.split("</s")[0].strip()
    if chat_template == "qwen":
        pred = pred.split("<|im_end|>")[0]
    elif "llama2" in chat_template.lower():
        pred = (pred.split("(Document")[0].split("\n\nQuestion")[0].split(
            "\n\nAnswer")[0].split("[INST]")[0].split("[/INST]")[0].split(
                "(Passage")[0].strip())
    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred


def format_prompt_style(sample: Dict[str, Any], instruction: str,
                        chat_template: str, dataset: str, tokenizer) -> str:
    """Format prompt following LongBench's approach."""
    # First format the instruction using the sample data
    prompt = instruction.format(**sample)

    if dataset not in [
            "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"
    ]:
        prompt = build_chat(tokenizer, prompt, chat_template)

    return prompt


def initialize_llm(args: argparse.Namespace) -> Tuple[LLM, AutoTokenizer]:
    """Initialize LLM and tokenizer."""
    logger.info(f"Initializing LLM with model: {args.model_path}")

    try:
        # Configure KV cache
        kv_cache_config = KvCacheConfig(
            # sparse attention doesn't support KV cache reuse
            enable_block_reuse=False,
            free_gpu_memory_fraction=args.kv_cache_fraction,
        )

        # Configure CUDA graph
        cuda_graph_config = CudaGraphConfig(
            batch_sizes=args.cuda_graph_batch_sizes,
            enable_padding=args.cuda_graph_padding_enabled,
        ) if args.use_cuda_graph else None

        # Configure sparse attention
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
            max_batch_size=args.max_batch_size,
            attn_backend=args.attention_backend,
            sparse_attention_config=sparse_attention_config,
            tensor_parallel_size=args.tp_size,
            moe_expert_parallel_size=args.moe_ep_size,
            enable_attention_dp=args.enable_attention_dp,
            max_seq_len=args.max_seq_len,
            max_num_tokens=args.max_num_tokens,
            cuda_graph_config=cuda_graph_config,
            torch_compile_config=None,
            print_iter_log=args.print_iter_log,
            moe_config=MoeConfig(backend=args.moe_backend),
        )

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        logger.info("LLM and tokenizer initialized successfully")

        return llm, tokenizer

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise e


def evaluate_single_dataset(
        dataset: str, llm: LLM, tokenizer: AutoTokenizer,
        args: argparse.Namespace) -> Tuple[List[Dict], float]:
    """Evaluate a single dataset."""
    setup_longbench_imports(args.longbench_path)

    dataset2prompt, dataset2maxlen = load_longbench_config(args.longbench_path)

    # Load dataset
    logger.info(f"Loading dataset: {dataset}")
    data = [
        data_sample for data_sample in load_dataset(
            'THUDM/LongBench', dataset, split='test', trust_remote_code=True)
    ]

    # Apply data filtering
    if args.num_samples:
        end_idx = min(args.start_idx + args.num_samples, len(data))
        filtered_data = data[args.start_idx:end_idx]
    else:
        filtered_data = data[args.start_idx:]

    logger.info(f"Dataset {dataset}: {len(filtered_data)} samples to evaluate")

    # Determine chat template
    chat_template = determine_chat_template(args.model_path, args.chat_template)
    logger.info(f"Using chat template: {chat_template}")

    # Create sampling parameters
    max_new_tokens = dataset2maxlen[dataset]
    prompt_format = dataset2prompt[dataset]

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

    if dataset == "samsum":
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        extra_end_token_ids.append(newline_id)
        logger.info(f"Added samsum newline token: {newline_id}")

    # Prepare prompts
    prompts = []
    for sample in filtered_data:
        formatted_prompt = format_prompt_style(sample, prompt_format,
                                               chat_template, dataset,
                                               tokenizer)
        prompts.append(formatted_prompt)

    if len(prompts) == 0:
        logger.warning(f"No prompts to evaluate for dataset {dataset}")
        return [], 0.0

    # Run inference
    logger.info(f"Starting inference for {len(prompts)} samples...")
    start_time = time.time()

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        stop_token_ids=extra_end_token_ids if extra_end_token_ids else None,
    )

    outputs = llm.generate(prompts, sampling_params)

    inference_time = time.time() - start_time
    logger.info(
        f"Inference completed in {inference_time:.2f} seconds, average time per sample: {inference_time/len(prompts):.3f} seconds"
    )

    # Prepare results
    results = []
    for i, (sample, output) in enumerate(zip(filtered_data, outputs)):
        prediction = output.outputs[0].text.strip()
        processed_prediction = post_process(prediction, chat_template, dataset)

        result = {
            'sample_id': args.start_idx + i,
            'input': sample.get('input', ''),
            'context': sample.get('context', ''),
            'answers': sample.get('answers', []),
            'all_classes': sample.get('all_classes', []),
            'prediction': processed_prediction,
            'raw_prediction': prediction,
            'prompt_length': len(output.prompt_token_ids),
            'output_length': len(output.outputs[0].token_ids),
            'inference_time': getattr(output, 'inference_time', None),
            'length': sample.get('length', 0)
        }
        results.append(result)

    return results, inference_time


def calculate_metrics(
        dataset: str, predictions: List[str], answers_list: List[List[str]],
        all_classes_list: List[List[str]],
        longbench_path: str) -> Tuple[Dict[str, float], List[float]]:
    """Calculate evaluation metrics for a dataset following LongBench's implementation."""

    # Setup LongBench imports
    setup_longbench_imports(longbench_path)

    # Import LongBench metrics
    from metrics import (classification_score, code_sim_score, count_score,
                         qa_f1_score, qa_f1_zh_score, retrieval_score,
                         retrieval_zh_score, rouge_score, rouge_zh_score)

    # Mapping of datasets to their metric functions (from LongBench)
    dataset2metric = {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }

    if dataset not in dataset2metric:
        # Fallback to simple exact match with cleaning
        total_score = 0
        scores = []
        for pred, answers in zip(predictions, answers_list):
            cleaned_pred = pred.lstrip('\n').split('\n')[0].strip()
            score = max([
                1.0 if cleaned_pred.lower() == ans.strip().lower() else 0.0
                for ans in answers
            ])
            scores.append(score)
            total_score += score
        return {
            "exact_match": round(100 * total_score / len(predictions), 2)
        }, scores

    metric_func = dataset2metric[dataset]
    total_score = 0.0
    scores = []

    # Follow LongBench's scorer function exactly
    for pred, ground_truths, all_classes in zip(predictions, answers_list,
                                                all_classes_list):
        score = 0.0

        # Apply the same prediction cleaning as LongBench
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip('\n').split('\n')[0]

        # For code datasets, apply additional cleaning
        if dataset in ["lcc", "repobench-p"]:
            # This cleaning is done inside code_sim_score, but let's also apply it here for consistency
            all_lines = pred.lstrip('\n').split('\n')
            for line in all_lines:
                if ('`' not in line) and ('#' not in line) and ('//'
                                                                not in line):
                    pred = line
                    break

        # Calculate max score across all reference answers (exactly as in LongBench)
        for ground_truth in ground_truths:
            score = max(
                score, metric_func(pred, ground_truth, all_classes=all_classes))

        scores.append(score)
        total_score += score

    final_score = round(100 * total_score / len(predictions), 2)
    return {metric_func.__name__: final_score}, scores


def calculate_task_summary(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Calculate task-level summary statistics following long_bench_tasks_summary.py approach."""
    logger.info("Calculating task-level summary statistics...")

    summary = {}
    ind_dataset_result = {}
    task_ave_result = {}

    NA_flag = False

    # Get individual dataset results
    for dataset in LONGBENCH_DATASETS:
        if dataset in all_results and 'metrics' in all_results[dataset]:
            metrics = all_results[dataset]['metrics']
            # Get the first (and usually only) metric value
            if metrics:
                metric_key = list(metrics.keys())[0]
                val = metrics[metric_key]
                ind_dataset_result[dataset] = val
            else:
                ind_dataset_result[dataset] = 'N/A'
                NA_flag = True
        else:
            ind_dataset_result[dataset] = 'N/A'
            NA_flag = True

    summary['individual_dataset_result'] = ind_dataset_result

    # Calculate task-average results
    for task, datasets in TASK_DATASETS.items():
        task_NA_flag = False
        task_ave_result[task] = 0
        valid_count = 0

        for dataset in datasets:
            if dataset in ind_dataset_result and ind_dataset_result[
                    dataset] != 'N/A':
                task_ave_result[task] += ind_dataset_result[dataset]
                valid_count += 1
            else:
                task_NA_flag = True

        if task_NA_flag or valid_count == 0:
            task_ave_result[task] = 'N/A'
        else:
            task_ave_result[task] = np.round(task_ave_result[task] /
                                             valid_count,
                                             decimals=2)

    summary['task_average_result'] = task_ave_result

    # Calculate overall LongBench average result (excluding passage_count as in original)
    if NA_flag:
        summary['LB_average_result'] = 'N/A'
    else:
        average_result = 0
        valid_count = 0
        for dataset in LONGBENCH_DATASETS:
            if dataset != 'passage_count' and dataset in ind_dataset_result:
                if ind_dataset_result[dataset] != 'N/A':
                    average_result += ind_dataset_result[dataset]
                    valid_count += 1

        if valid_count > 0:
            summary['LB_average_result'] = np.round(average_result /
                                                    valid_count,
                                                    decimals=2)
        else:
            summary['LB_average_result'] = 'N/A'

    # Log summary statistics
    logger.info("Task Summary Results:")
    logger.info(f"Overall LongBench Average: {summary['LB_average_result']}")
    for task, score in task_ave_result.items():
        logger.info(f"{task}: {score}")

    return summary


def save_results(results: List[Dict], dataset: str, args: argparse.Namespace,
                 inference_time: float, output_dir: str):
    """Save evaluation results in format compatible with LongBench."""
    task_output_dir = os.path.join(output_dir, dataset)
    os.makedirs(task_output_dir, exist_ok=True)

    # Extract predictions, answers, and all_classes for evaluation
    predictions = [r['prediction'] for r in results]
    answers_list = [r['answers'] for r in results]
    all_classes_list = [r.get('all_classes', []) for r in results]

    # Calculate metrics
    processed_results, scores = calculate_metrics(dataset, predictions,
                                                  answers_list,
                                                  all_classes_list,
                                                  args.longbench_path)
    logger.info(f"Evaluation metrics: {processed_results}")

    # Save detailed results for manual inspection
    results_file = os.path.join(task_output_dir, f"{dataset}_results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    # Save prediction results in LongBench format for evaluation
    pred_dir = os.path.join(task_output_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, f"{dataset}.jsonl")

    with open(pred_file, 'w', encoding='utf-8') as f:
        for idx, result in enumerate(results):
            pred_data = {
                "pred": result['prediction'],
                "answers": result['answers'],
                "all_classes": result.get('all_classes', []),
                "length": result.get('length', 0),
                "score": scores[idx]
            }
            json.dump(pred_data, f, ensure_ascii=False)
            f.write('\n')

    # Create summary following LongBench format
    config = {
        'pipeline_params': {
            'model_name': args.model_path,
            'method': args.attention_backend,
            'token_budget': args.token_budget,
            'max_seq_len': args.max_seq_len,
            'max_new_tokens': args.max_new_tokens,
            'window_size': args.window_size,
            'kernel_size': args.kernel_size,
            'num_processes': 1,  # Single process
            'devices': "0"  # Single device
        },
        'eval_params': {
            'dataset': dataset,
            'num_samples': len(results)
        },
        'eval_results': {
            'processed_results': processed_results
        },
        'management': {
            'output_folder_dir': task_output_dir,
            'exp_desc':
            f'{dataset}_{os.path.basename(args.model_path)}_{args.attention_backend}_{args.token_budget}',
            'total_inference_time': inference_time,
            'avg_inference_time': inference_time / len(results),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    }

    # Save summary
    summary_file = os.path.join(task_output_dir, f"{dataset}_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"The results of {dataset} are saved to {task_output_dir}")

    return processed_results


def main():
    """Main evaluation function."""
    args = parse_arguments()
    logger.set_level(args.log_level)

    # Setup experiment name
    if not args.exp_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(args.model_path).replace('/', '_')
        args.exp_name = f"longbench_{model_name}_{timestamp}"

    output_dir = os.path.join(args.output_dir, args.exp_name)

    logger.info(
        "=========== LongBench Evaluation with TensorRT-LLM ===========")

    os.makedirs(output_dir, exist_ok=True)
    # Save configuration
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to {config_file}")

    # Determine datasets to evaluate
    if args.run_all_tasks:
        datasets = LONGBENCH_DATASETS
        logger.info(f"Running evaluation on full LongBench datasets")
    else:
        datasets = args.dataset
        logger.info(f"Running evaluation on datasets: {args.dataset}")

    # Initialize LLM and tokenizer
    llm, tokenizer = initialize_llm(args)

    # Process datasets sequentially
    all_results = {}
    for dataset_idx, dataset in enumerate(datasets):
        logger.info(f"{'='*30}")
        logger.info(
            f"Processing dataset {dataset_idx+1}/{len(datasets)}: {dataset}...")

        # Evaluate the dataset
        results, inference_time = evaluate_single_dataset(
            dataset, llm, tokenizer, args)

        # Save results and get metrics
        processed_results = save_results(results, dataset, args, inference_time,
                                         output_dir)

        all_results[dataset] = {
            'num_samples': len(results),
            'inference_time': inference_time,
            'output_dir': output_dir,
            'metrics': processed_results
        }
        logger.info(f"Dataset {dataset} completed successfully")

    total_time = sum(all_results[d]['inference_time'] for d in all_results)

    # Calculate task-level summary
    task_summary = calculate_task_summary(all_results)

    # Save overall summary with task statistics
    overall_summary = {
        'experiment_name':
        args.exp_name,
        'total_evaluation_time':
        total_time,
        'evaluated_datasets':
        list(all_results.keys()),
        'successful_datasets':
        [d for d, r in all_results.items() if 'error' not in r],
        'failed_datasets': [d for d, r in all_results.items() if 'error' in r],
        'results_by_dataset':
        all_results,
        'task_summary':
        task_summary,  # Add task-level summary
        'configuration':
        vars(args)
    }

    overall_summary_file = os.path.join(output_dir, "overall_summary.json")
    with open(overall_summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    logger.info(f"\n{'-'*80}")
    logger.info(
        f"Evaluation completed. Overall summary saved to: {overall_summary_file}"
    )
    logger.info(f"Total time: {total_time:.2f} seconds")

    # Print final summary
    if task_summary['LB_average_result'] != 'N/A':
        logger.info(f"FINAL RESULTS:")
        logger.info(
            f"LongBench Overall Average: {task_summary['LB_average_result']}")
        logger.info(f"Task-level results:")
        for task, score in task_summary['task_average_result'].items():
            logger.info(f"  {task}: {score}")

    if overall_summary['failed_datasets']:
        logger.warning(f"Failed datasets: {overall_summary['failed_datasets']}")


if __name__ == '__main__':
    main()
