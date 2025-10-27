# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LongBench v2 evaluation for TensorRT-LLM.

This module provides evaluation capabilities for the LongBench v2 benchmark,
which tests long-context understanding across multiple domains, difficulties,
and context lengths. It supports various evaluation modes including standard,
Chain-of-Thought (CoT), RAG-based, and no-context evaluations.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union

import click
from datasets import load_dataset

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator


class LongBenchV2(Evaluator):
    """Evaluator for LongBench v2 benchmark.

    This evaluator implements the LongBench v2 benchmark for evaluating long-context
    language models. It supports multiple evaluation modes and filtering options.

    Attributes:
        DIFFICULTIES: List of supported difficulty levels
        LENGTHS: List of supported context length categories
    """

    DIFFICULTIES = ['easy', 'hard']
    LENGTHS = ['short', 'medium', 'long']

    def __init__(self,
                 dataset_path: str = 'THUDM/LongBench-v2',
                 prompts_dir: Optional[str] = None,
                 num_samples: Optional[int] = None,
                 start_idx: int = 0,
                 difficulty: Optional[str] = None,
                 length: str = 'medium',
                 domain: Optional[str] = None,
                 cot: bool = False,
                 no_context: bool = False,
                 rag: int = 0,
                 max_len: int = 128000,
                 output_dir: Optional[str] = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        """Initialize LongBench v2 evaluator.

        Args:
            dataset_path: Path or HuggingFace dataset name
            prompts_dir: Directory containing custom prompt templates
            num_samples: Number of samples to evaluate (None for all)
            start_idx: Starting index for evaluation
            difficulty: Filter by difficulty ('easy' or 'hard')
            length: Filter by length ('short', 'medium', or 'long')
            domain: Filter by domain name
            cot: Enable Chain-of-Thought reasoning
            no_context: Test without context (memorization test)
            rag: Number of top retrieved contexts to use (0 to disable)
            max_len: Maximum prompt length for truncation
            output_dir: Directory to save evaluation results
            random_seed: Random seed for reproducibility
            apply_chat_template: Whether to apply model's chat template
            system_prompt: System prompt to prepend
        """
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         system_prompt=system_prompt)

        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.start_idx = start_idx
        self.difficulty = difficulty
        self.length = length
        self.domain = domain
        self.cot = cot
        self.no_context = no_context
        self.rag = rag
        self.max_len = max_len
        self.output_dir = output_dir

        # Will be set during evaluation
        self.tokenizer = None
        self.chat_template_name = 'none'

        # Load templates
        self.templates = self._load_templates(prompts_dir)

        # Load and filter dataset
        self.dataset = self._load_and_filter_dataset()

    def _load_templates(self, prompts_dir: Optional[str]) -> Dict[str, str]:
        """Load prompt templates from directory or use defaults.

        Args:
            prompts_dir: Directory containing template files (optional)

        Returns:
            Dictionary mapping template names to template strings
        """
        templates = {}

        # Default templates if prompts_dir not provided
        default_templates = {
            '0shot':
            '''Please read the following text and answer the question below.\n\n<text>\n$DOC$\n</text>\n\nWhat is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nFormat your response as follows: "The correct answer is (insert answer here)".''',
            '0shot_cot':
            '''Please read the following text and answer the questions below.\n\n<text>\n$DOC$\n</text>\n\nWhat is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nLet’s think step by step:''',
            '0shot_cot_ans':
            '''Please read the following text and answer the questions below.\n\nThe text is too long and omitted here.\n\nWhat is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nLet’s think step by step: $COT$\n\nBased on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".''',
            '0shot_no_context':
            '''What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nWhat is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".''',
            '0shot_rag':
            '''Please read the following retrieved text chunks and answer the question below.\n\n<text>\n$DOC$\n</text>\n\nWhat is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nFormat your response as follows: "The correct answer is (insert answer here)".''',
        }

        if prompts_dir and os.path.exists(prompts_dir):
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
                else:
                    templates[template_name] = default_templates.get(
                        template_name, default_templates['0shot'])
        else:
            templates = default_templates

        return templates

    def _load_and_filter_dataset(self) -> List[Dict[str, Any]]:
        """Load LongBench v2 dataset and apply filters.

        Loads the dataset from HuggingFace or local path and applies filters
        based on difficulty, length, domain, and sample range.

        Returns:
            List of filtered sample dictionaries
        """
        logger.info(f"Loading LongBench v2 dataset from {self.dataset_path}...")
        dataset = load_dataset(self.dataset_path,
                               split='train',
                               trust_remote_code=True)
        data = [item for item in dataset]

        # Apply filters
        filtered_data = data

        if self.difficulty:
            filtered_data = [
                item for item in filtered_data
                if item['difficulty'] == self.difficulty
            ]
            logger.info(
                f"Filtered by difficulty '{self.difficulty}': {len(filtered_data)} samples"
            )

        if self.length:
            filtered_data = [
                item for item in filtered_data if item['length'] == self.length
            ]
            logger.info(
                f"Filtered by length '{self.length}': {len(filtered_data)} samples"
            )

        if self.domain:
            filtered_data = [
                item for item in filtered_data if item['domain'] == self.domain
            ]
            logger.info(
                f"Filtered by domain '{self.domain}': {len(filtered_data)} samples"
            )

        # Apply start_idx and num_samples
        if self.num_samples:
            end_idx = min(self.start_idx + self.num_samples, len(filtered_data))
            filtered_data = filtered_data[self.start_idx:end_idx]
        else:
            filtered_data = filtered_data[self.start_idx:]

        logger.info(f"Final dataset size: {len(filtered_data)} samples")
        return filtered_data

    def _format_prompt(self, sample: Dict[str, Any], template: str) -> str:
        """Format a prompt using the template and sample data.

        Replaces template placeholders with actual sample content,
        handling different modes (standard, RAG, no-context).

        Args:
            sample: Sample dictionary containing question and choices
            template: Template string with placeholders

        Returns:
            Formatted prompt string
        """
        context = sample['context']

        # Handle RAG mode
        if self.rag > 0 and 'retrieved_context' in sample:
            retrieved = sample["retrieved_context"][:self.rag]
            retrieved = sorted(retrieved, key=lambda x: x.get('c_idx', 0))
            context = '\n\n'.join([
                f"Retrieved chunk {idx+1}: {x['content']}"
                for idx, x in enumerate(retrieved)
            ])

        # Handle no context mode
        if self.no_context:
            context = ""

        # Format the prompt using the template
        prompt = template.replace('$DOC$', context.strip())
        prompt = prompt.replace('$Q$', sample['question'].strip())
        prompt = prompt.replace('$C_A$', sample['choice_A'].strip())
        prompt = prompt.replace('$C_B$', sample['choice_B'].strip())
        prompt = prompt.replace('$C_C$', sample['choice_C'].strip())
        prompt = prompt.replace('$C_D$', sample['choice_D'].strip())

        return prompt

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract answer choice (A/B/C/D) from model response.

        Tries multiple patterns to extract the answer, following LongBench v2
        conventions for answer extraction.

        Args:
            response: Model's text response

        Returns:
            Extracted answer letter (A/B/C/D) or None if not found
        """
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

    def _post_process(self, pred: str) -> str:
        """Post-process prediction following LongBench v2 conventions.

        Args:
            pred: Raw prediction string

        Returns:
            Cleaned prediction string
        """
        pred = pred.split("</s")[0].strip()
        if self.chat_template_name == "qwen":
            pred = pred.split("<|im_end|>")[0]
        elif "llama2" in self.chat_template_name.lower():
            pred = (pred.split("(Document")[0].split("\n\nQuestion")[0].split(
                "\n\nAnswer")[0].split("[INST]")[0].split("[/INST]")[0].split(
                    "(Passage")[0].strip())
        elif "deepseek" in self.chat_template_name.lower():
            if "</think>" in pred:
                pred = pred.split("</think>", 1)[1].lstrip()

        return pred

    def _truncate_prompt(self, prompt: str, tokenizer: Any) -> str:
        """Truncate prompt to max_len tokens using needle-in-haystack strategy.

        If the prompt exceeds max_len, it takes the first half and last half
        to preserve both context beginning and end.

        Args:
            prompt: The prompt string to truncate
            tokenizer: Tokenizer for encoding/decoding

        Returns:
            Truncated prompt string
        """
        if tokenizer is None:
            return prompt

        try:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)

            if len(input_ids) > self.max_len:
                half = self.max_len // 2
                truncated_ids = input_ids[:half] + input_ids[-half:]
                prompt = tokenizer.decode(truncated_ids,
                                          skip_special_tokens=True)
                logger.info(
                    f"Truncated prompt from {len(input_ids)} to {len(truncated_ids)} tokens"
                )

        except Exception as e:
            logger.warning(
                f"Failed to truncate prompt: {e}. Using original prompt.")

        return prompt

    def _determine_chat_template_name(self, tokenizer: Any) -> str:
        """Determine chat template type from tokenizer.

        Args:
            tokenizer: Model tokenizer

        Returns:
            Chat template name
        """
        if tokenizer is None:
            return 'none'

        if hasattr(tokenizer, 'name_or_path'):
            model_name = str(tokenizer.name_or_path).lower()
        else:
            model_name = ''

        if 'qwen' in model_name:
            return 'qwen'
        elif 'llama2' in model_name:
            return 'llama2'
        elif 'llama3' in model_name:
            return 'llama3'
        elif 'deepseek' in model_name:
            return 'deepseek'

        return 'none'

    def _get_extra_end_token_ids(self) -> list:
        """
        Get extra end token IDs for specific chat templates.

        Returns:
            A list of token IDs that should be treated as end tokens for the current chat template.
        """
        extra_end_token_ids = []
        if self.tokenizer is not None:
            if self.chat_template_name == "llama3":
                try:
                    eot_id = self.tokenizer.encode("<|eot_id|>",
                                                   add_special_tokens=False)[0]
                    extra_end_token_ids.append(eot_id)
                    logger.info(f"Added llama3 end token: {eot_id}")
                except Exception as e:
                    logger.warning(f"Failed to add llama3 end token: {e}")

            if self.chat_template_name == "qwen":
                try:
                    im_end_id = self.tokenizer.encode(
                        "<|im_end|>", add_special_tokens=False)[0]
                    extra_end_token_ids.append(im_end_id)
                    logger.info(f"Added qwen end token: {im_end_id}")
                except Exception as e:
                    logger.warning(f"Failed to add qwen end token: {e}")
        return extra_end_token_ids

    def evaluate(self,
                 llm: Any,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False) -> float:
        """Evaluate LongBench v2 benchmark.

        This overrides the base evaluate method to initialize tokenizer and
        chat template before evaluation, and handle CoT mode with two-pass inference.

        Args:
            llm: Language model instance
            sampling_params: Sampling parameters for generation
            streaming: Whether to use streaming mode

        Returns:
            Overall accuracy score
        """
        # Initialize tokenizer and chat template
        if hasattr(llm, 'tokenizer'):
            self.tokenizer = llm.tokenizer
            self.chat_template_name = self._determine_chat_template_name(
                self.tokenizer)
            logger.info(f"Detected chat template: {self.chat_template_name}")
        else:
            logger.warning(
                "LLM does not have tokenizer attribute. Truncation disabled.")

        # Setup extra end token IDs for specific chat templates using a separate function
        self.extra_end_token_ids = self._get_extra_end_token_ids()

        # Update sampling params with extra end tokens if available
        if self.extra_end_token_ids and sampling_params is not None:
            if sampling_params.stop_token_ids is None:
                sampling_params.stop_token_ids = self.extra_end_token_ids
            else:
                sampling_params.stop_token_ids = list(
                    sampling_params.stop_token_ids) + self.extra_end_token_ids

        # Store llm reference for CoT second pass
        self.llm = llm

        # Call parent evaluate method
        return super().evaluate(llm, sampling_params, streaming)

    def generate_samples(self) -> Iterable[tuple]:
        """
        Generate samples for evaluation.

        Yields:
            tuple: A tuple containing the following items for each sample:
                - prompt (str): The generated prompt text for the evaluation.
                - sampling_args (Any or None): Sampling arguments to use for generation, or None if not specified.
                - reference (str): The ground truth or reference answer to the prompt.
                - sample_id (str): Unique identifier for the sample.
                - sample_dict (dict): The full sample dictionary from the dataset with all metadata.
        """
        # Select appropriate template
        if self.no_context:
            template_key = '0shot_no_context'
        elif self.rag > 0:
            template_key = '0shot_rag'
        elif self.cot:
            template_key = '0shot_cot'
        else:
            template_key = '0shot'

        template = self.templates[template_key]
        logger.info(f"Using template: {template_key}")

        # Generate prompts for each sample
        for sample in self.dataset:
            prompt = self._format_prompt(sample, template)

            # Apply truncation if needed and tokenizer is available
            if self.tokenizer is not None and self.max_len > 0:
                prompt = self._truncate_prompt(prompt, self.tokenizer)

            # Yield: prompt, sampling_args, reference, sample_id, sample_dict
            yield prompt, None, sample['answer'], sample['_id'], sample

    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      sample_ids: List[str], samples: List[Dict]) -> float:
        """Compute evaluation metrics and save results.

        Args:
            outputs: Model outputs
            references: Ground truth answers
            sample_ids: Sample identifiers
            samples: Full sample dictionaries

        Returns:
            Overall accuracy percentage
        """
        results = []

        for output, ref, sample_id, sample in zip(outputs, references,
                                                  sample_ids, samples):
            prediction = output.outputs[0].text.strip()
            processed_prediction = self._post_process(prediction)

            # Handle CoT mode with second inference pass
            if self.cot:
                # For CoT, we need to do a second inference to extract the final answer
                cot_response = processed_prediction

                # Check if we have the CoT answer extraction template
                if '0shot_cot_ans' in self.templates and hasattr(self, 'llm'):
                    # Format the CoT answer extraction prompt
                    cot_ans_template = self.templates['0shot_cot_ans']
                    cot_ans_prompt = self._format_prompt(
                        sample, cot_ans_template)
                    cot_ans_prompt = cot_ans_prompt.replace(
                        '$COT$', cot_response)

                    # Apply truncation if needed
                    if self.tokenizer is not None and self.max_len > 0:
                        cot_ans_prompt = self._truncate_prompt(
                            cot_ans_prompt, self.tokenizer)

                    # Generate final answer with shorter max tokens
                    ans_sampling_params = SamplingParams(
                        max_tokens=128,
                        temperature=0.6,
                        top_p=0.95,
                        stop_token_ids=self.extra_end_token_ids
                        if self.extra_end_token_ids else None,
                    )

                    try:
                        ans_output = self.llm.generate([cot_ans_prompt],
                                                       ans_sampling_params)[0]
                        final_prediction = self._post_process(
                            ans_output.outputs[0].text.strip())
                        extracted_answer = self._extract_answer(
                            final_prediction)
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract CoT answer for sample {sample_id}: {e}"
                        )
                        extracted_answer = self._extract_answer(
                            processed_prediction)
                        final_prediction = processed_prediction
                else:
                    # Fallback to basic extraction if template not available
                    extracted_answer = self._extract_answer(
                        processed_prediction)
                    final_prediction = processed_prediction
            else:
                extracted_answer = self._extract_answer(processed_prediction)
                cot_response = None
                final_prediction = None

            # Calculate accuracy
            is_correct = extracted_answer == ref if extracted_answer else False

            result = {
                '_id': sample_id,
                'domain': sample['domain'],
                'sub_domain': sample['sub_domain'],
                'difficulty': sample['difficulty'],
                'length': sample['length'],
                'question': sample['question'],
                'answer': ref,
                'prediction': processed_prediction,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'prompt_length': len(output.prompt_token_ids),
                'generation_length': len(output.outputs[0].token_ids),
            }

            # Add CoT-specific fields
            if self.cot:
                result['cot_response'] = cot_response
                result['final_prediction'] = final_prediction

            results.append(result)

        # Calculate metrics
        metrics = self._calculate_metrics(results)

        # Save results
        self._save_results(results, metrics)

        # Log detailed results similar to MMLU
        logger.info("=" * 80)
        logger.info("LongBench v2 Evaluation Results")
        logger.info("=" * 80)

        # Overall results
        logger.info(
            f"Overall Accuracy: {metrics['overall_accuracy']:.2f}% "
            f"({metrics['correct_samples']}/{metrics['total_samples']})")

        # Results by difficulty
        logger.info("-" * 80)
        logger.info("Results by Difficulty:")
        for difficulty in self.DIFFICULTIES:
            if f'{difficulty}_accuracy' in metrics:
                diff_results = [
                    r for r in results if r['difficulty'] == difficulty
                ]
                diff_correct = sum(1 for r in diff_results if r['is_correct'])
                logger.info(
                    f"  {difficulty.capitalize()}: {metrics[f'{difficulty}_accuracy']:.2f}% "
                    f"({diff_correct}/{len(diff_results)})")

        # Results by length
        logger.info("-" * 80)
        logger.info("Results by Context Length:")
        for length in self.LENGTHS:
            if f'{length}_accuracy' in metrics:
                len_results = [r for r in results if r['length'] == length]
                len_correct = sum(1 for r in len_results if r['is_correct'])
                logger.info(
                    f"  {length.capitalize()}: {metrics[f'{length}_accuracy']:.2f}% "
                    f"({len_correct}/{len(len_results)})")

        # Results by domain
        domains = sorted(set(r['domain'] for r in results))
        if domains:
            logger.info("-" * 80)
            logger.info("Results by Domain:")
            for domain in domains:
                if f'{domain}_accuracy' in metrics:
                    domain_results = [
                        r for r in results if r['domain'] == domain
                    ]
                    domain_correct = sum(1 for r in domain_results
                                         if r['is_correct'])
                    logger.info(
                        f"  {domain}: {metrics[f'{domain}_accuracy']:.2f}% "
                        f"({domain_correct}/{len(domain_results)})")

        logger.info("=" * 80)

        return metrics['overall_accuracy']

    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.

        Computes overall accuracy and breakdowns by difficulty, length, and domain.

        Args:
            results: List of result dictionaries from evaluation

        Returns:
            Dictionary containing all calculated metrics
        """
        if not results:
            return {}

        total_samples = len(results)
        correct_samples = sum(1 for r in results if r['is_correct'])
        overall_accuracy = (correct_samples / total_samples) * 100

        metrics = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'correct_samples': correct_samples
        }

        # Calculate metrics by difficulty
        for difficulty in self.DIFFICULTIES:
            diff_results = [r for r in results if r['difficulty'] == difficulty]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r['is_correct'])
                metrics[f'{difficulty}_accuracy'] = (diff_correct /
                                                     len(diff_results)) * 100
                metrics[f'{difficulty}_samples'] = len(diff_results)

        # Calculate metrics by length
        for length in self.LENGTHS:
            len_results = [r for r in results if r['length'] == length]
            if len_results:
                len_correct = sum(1 for r in len_results if r['is_correct'])
                metrics[f'{length}_accuracy'] = (len_correct /
                                                 len(len_results)) * 100
                metrics[f'{length}_samples'] = len(len_results)

        # Calculate metrics by domain
        domains = list(set(r['domain'] for r in results))
        for domain in domains:
            domain_results = [r for r in results if r['domain'] == domain]
            if domain_results:
                domain_correct = sum(1 for r in domain_results
                                     if r['is_correct'])
                metrics[f'{domain}_accuracy'] = (domain_correct /
                                                 len(domain_results)) * 100
                metrics[f'{domain}_samples'] = len(domain_results)

        return metrics

    def _save_results(self, results: List[Dict], metrics: Dict[str, float]):
        """Save evaluation results to disk.

        Saves three files:
        - longbench_v2_results.jsonl: Detailed per-sample results
        - predictions.jsonl: LongBench v2 compatible predictions format
        - summary.json: Aggregated metrics and metadata

        Args:
            results: List of result dictionaries
            metrics: Calculated metrics dictionary
        """
        if self.output_dir is None:
            logger.warning(
                "Output directory is None. Skipping saving of evaluation results."
            )
            return
        os.makedirs(self.output_dir, exist_ok=True)

        # Save detailed results
        results_file = os.path.join(self.output_dir,
                                    "longbench_v2_results.jsonl")
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

        # Save predictions
        pred_file = os.path.join(self.output_dir, "predictions.jsonl")
        with open(pred_file, 'w', encoding='utf-8') as f:
            for result in results:
                pred_data = {
                    "_id": result['_id'],
                    "extracted_answer": result['extracted_answer'],
                    "response": result['prediction'],
                    "judge": result['is_correct']
                }
                # Add CoT-specific fields if available
                if 'cot_response' in result:
                    pred_data['cot_response'] = result.get('cot_response', '')
                if 'final_prediction' in result:
                    pred_data['final_response'] = result.get(
                        'final_prediction', '')

                json.dump(pred_data, f, ensure_ascii=False)
                f.write('\n')

        # Save summary
        summary = {
            'evaluation_results': metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }

        summary_file = os.path.join(self.output_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")

    @click.command("longbench_v2")
    @click.option(
        "--dataset_path",
        type=str,
        default='THUDM/LongBench-v2',
        help="Path to LongBench v2 dataset (HF dataset name or local path).")
    @click.option("--prompts_dir",
                  type=str,
                  default=None,
                  help="Path to directory containing prompt templates.")
    @click.option("--num_samples",
                  type=int,
                  default=None,
                  help="Number of samples to evaluate (None for all).")
    @click.option("--start_idx",
                  type=int,
                  default=0,
                  help="Start index for evaluation.")
    @click.option("--difficulty",
                  type=click.Choice(['easy', 'hard']),
                  default=None,
                  help="Filter by difficulty level.")
    @click.option("--length",
                  type=click.Choice(['short', 'medium', 'long']),
                  default='medium',
                  help="Filter by length category.")
    @click.option("--domain", type=str, default=None, help="Filter by domain.")
    @click.option("--cot",
                  is_flag=True,
                  default=False,
                  help="Enable Chain-of-Thought reasoning.")
    @click.option("--no_context",
                  is_flag=True,
                  default=False,
                  help="Test without long context.")
    @click.option("--rag",
                  type=int,
                  default=0,
                  help="Use top-N retrieved contexts (0 to disable).")
    @click.option(
        "--max_len",
        type=int,
        default=128000,
        help=
        "Maximum prompt length in tokens for truncation when building prompts.")
    @click.option("--output_dir",
                  type=str,
                  default=None,
                  help="Directory to save results.")
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=True,
                  help="Whether to apply chat template.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=128000,
                  help="Maximum prompt length in sampling parameters.")
    @click.option("--max_output_length",
                  type=int,
                  default=32000,
                  help="Maximum generation length in sampling parameters.")
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, prompts_dir: Optional[str],
                num_samples: Optional[int], start_idx: int,
                difficulty: Optional[str], length: str, domain: Optional[str],
                cot: bool, no_context: bool, rag: int, max_len: int,
                output_dir: Optional[str], random_seed: int,
                apply_chat_template: bool, system_prompt: Optional[str],
                max_input_length: int, max_output_length: int) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj

        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length,
            temperature=0.6,
            top_p=0.95)

        evaluator = LongBenchV2(dataset_path=dataset_path,
                                prompts_dir=prompts_dir,
                                num_samples=num_samples,
                                start_idx=start_idx,
                                difficulty=difficulty,
                                length=length,
                                domain=domain,
                                cot=cot,
                                no_context=no_context,
                                rag=rag,
                                max_len=max_len,
                                output_dir=output_dir,
                                random_seed=random_seed,
                                apply_chat_template=apply_chat_template,
                                system_prompt=system_prompt)

        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()
