# LongBench Evaluation with TensorRT-LLM and Sparse Attention

This directory contains evaluation scripts for LongBench v1 datasets using TensorRT-LLM backend.

> **Note**:  
LongBench v2 evaluation has been integrated into `trtllm-eval`. Please refer to `tensorrt_llm/evaluate/longbench_v2.py` for details.


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
├── README.md                     # This file
└── LongBench/                    # Cloned LongBench repository
    ├── LongBench/                # LongBench v1 data and configs
    │   ├── config/
    │   └── ...
    ├── ...
    └── requirements.txt
```

## Scripts Overview

The script `eval_longbench_v1.py` evaluates models on the **LongBench v1** dataset, which includes multiple specific tasks like narrativeqa, qasper, multifieldqa, etc. Key features:

- **Dataset**: LongBench v1 with task-specific evaluation
- **Tasks**: Support for 20+ different long-context tasks
- **Prompts**: Task-specific prompts from LongBench v1 configuration
- **Metrics**: Task-specific metrics (F1, ROUGE, classification scores, etc.)
- **Output**: Task-level results with comprehensive summary statistics

## Usage Examples

### Basic Usage (Standard Attention)
```bash
python eval_longbench_v1.py \
    --model_path "/path/to/your/model" \
    --longbench_path ./LongBench \
    --output_dir results/v1_vanilla \
    --attention_backend VANILLA \
    --backend pytorch
```

### Specific tasks With Sparse Attention (RocketKV)
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

## Output Structure

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
