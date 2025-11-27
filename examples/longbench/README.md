# LongBench Evaluation with TensorRT-LLM and Sparse Attention

This directory contains evaluation scripts for both LongBench v1 and LongBench v2 datasets using TensorRT-LLM backend.

## Environment Setup

### 1. Clone LongBench Repository

First, clone the LongBench repository which contains the datasets and evaluation utilities:

```bash
git clone https://github.com/THUDM/LongBench.git
```

### 2. Install Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Directory Structure

After cloning, your directory structure should look like:

```text
sparse_attention/
├── eval_longbench_v1.py          # LongBench v1 evaluation script
├── eval_longbench_v2.py          # LongBench v2 evaluation script
├── README.md                     # This file
└── LongBench/                    # Cloned LongBench repository
    ├── LongBench/                # LongBench v1 data and configs
    │   ├── config/
    │   └── ...
    ├── config/                   # LongBench v2 configs
    ├── ...
    └── requirements.txt
```

## Scripts Overview

### 1. eval_longbench_v1.py

This script evaluates models on the **LongBench v1** dataset, which includes multiple specific tasks like narrativeqa, qasper, multifieldqa, etc. Key features:

- **Dataset**: LongBench v1 with task-specific evaluation
- **Tasks**: Support for 20+ different long-context tasks
- **Prompts**: Task-specific prompts from LongBench v1 configuration
- **Metrics**: Task-specific metrics (F1, ROUGE, classification scores, etc.)
- **Output**: Task-level results with comprehensive summary statistics

### 2. eval_longbench_v2.py

This script evaluates models on the **LongBench v2** dataset, which is a standardized multiple-choice format. Key features:

- **Dataset**: LongBench v2 with unified multiple-choice format
- **Format**: All questions are A/B/C/D multiple choice
- **Context Length**: 8K to 2M words (majority under 128K)
- **Difficulty**: Easy/Hard categorization
- **Length**: Short/Medium/Long categorization
- **Domains**: Various domains (single-doc QA, multi-doc QA, code, etc.)
- **CoT Support**: Chain-of-Thought reasoning support
- **Metrics**: Accuracy with breakdowns by difficulty, length, and domain

## Usage Examples

### LongBench v1 Evaluation

#### Basic Usage (Standard Attention)
```bash
python eval_longbench_v1.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v1_vanilla \
    --attention_backend VANILLA \
    --backend pytorch
```

#### Specific tasks With Sparse Attention (RocketKV)
```bash
python eval_longbench_v1.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --dataset narrativeqa qasper \
    --output_dir results/v1_rocket \
    --attention_backend VANILLA \
    --backend pytorch \
    --rocket_sparse
```

### LongBench v2 Evaluation

#### Basic Usage (Standard Attention)
```bash
python eval_longbench_v2.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v2_vanilla
```

#### With Chain-of-Thought Reasoning
```bash
python eval_longbench_v2.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v2_cot \
    --cot
```

#### Filter by Difficulty/Length/Domain
```bash
# Easy questions only
python eval_longbench_v2.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v2_easy \
    --difficulty easy

# Long context only
python eval_longbench_v2.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v2_long \
    --length long

# Specific domain
python eval_longbench_v2.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v2_code \
    --domain "Code"
```

#### Limited Sample Evaluation (for testing)
```bash
python eval_longbench_v2.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v2_test \
    --num_samples 10
```

## Output Structure

### LongBench v1 Output

```text
results/v1_experiment/
├── config.json                          # Experiment configuration
├── overall_summary.json                 # Overall experiment summary
├── narrativeqa/
│   ├── narrativeqa_results.jsonl       # Detailed results
│   ├── narrativeqa_summary.json        # Task summary
│   └── pred/
│       └── narrativeqa.jsonl           # Predictions in LongBench format
├── qasper/
│   └── ...
└── ...
```

### LongBench v2 Output

```text
results/v2_experiment/
├── config.json                          # Experiment configuration
├── summary.json                         # Evaluation summary with metrics
├── longbench_v2_results.jsonl          # Detailed results
└── predictions.jsonl                    # Predictions in LongBench v2 format
```
