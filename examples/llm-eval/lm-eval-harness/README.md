# Evaluation scripts for LLM tasks

This folder includes code to use the [LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness),  a unified framework to test generative language models on a large number of different evaluation tasks. The supported eval tasks are [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

The following instructions show how to evaluate TRT-LLM engines with the benchmark.

## Instructions

### TRT-LLM API

Build the TRT-LLM engine using `trtllm-build`.

Install the `lm_eval` package in the `requirements.txt` file in this folder.

Run the evaluation script with the following command:

```sh
python lm_eval_tensorrt_llm.py --model trt-llm \
    --model_args tokenizer=<HF model folder>,model=<TRT LLM engine dir>,chunk_size=<int> \
    --tasks <comma separated tasks, e.g., gsm8k-cot, mmlu>
```

In the LM-Eval-Harness, model args are submitted as a comma-separated list of the form `arg=value`. The `trt-llm` model supports the following `model_args`:

| Name                     | Description                                                       | Default Value  |
|--------------------------|-------------------------------------------------------------------|----------------|
| tokenizer                | directory containing the HF tokenizer.                            |                |
| model                    | directory containing the TRTLLM engine or torch model.            |                |
| max_gen_toks             | max number of tokens to generate (if not specified in gen_kwargs) | 256            |
| chunk_size               | number of async requests to send at once to the engine            | 200            |
| max_tokens_kv_cache      | max tokens in paged KV cache                                      | None           |
| free_gpu_memory_fraction | KV cache free GPU memory fraction                                 | 0.9            |
| trust_remote_code        | trust remote code; use if necessary to set up the tokenizer       | False          |
| tp                       | tensor parallel size (for torch backend)                          | no. of workers |
| use_cuda_graph           | enable CUDA graph                                                 | True           |
| max_context_length       | maximum context length for evaluation                             | None           |
| moe_expert_parallel_size | expert parallel size for MoE models                               | None           |
| moe_backend              | backend for MoE models (e.g., "TRTLLM")                           | "TRTLLM"       |

### Torch backend

Install the `lm_eval` package in the `requirements.txt` file in this folder.

Run the evaluation script with the same command as above, but include `backend=torch` in the `model_args`. For example:

```sh
python lm_eval_tensorrt_llm.py --model trt-llm \
    --model_args model=<HF model folder>,backend=torch,chunk_size=<int> \
    --tasks <comma separated tasks, e.g., gsm8k-cot, mmlu>
```

### trtllm-serve

Build the TRT-LLM engine using `trtllm-build` and deploy with `trtllm-serve`.

Install the `lm_eval` package in the `requirements.txt` file in this folder.

Run the evaluation script with the following command:

```sh
python lm_eval_tensorrt_llm.py --model local-completions \
    --model_args base_url=http://${HOST_NAME}:8001/v1/completions,model=<model_name>,tokenizer=<tokenizer_dir> \
    --tasks <comma separated tasks, e.g., gsm8k-cot, mmlu> \
    --batch_size <#>
```

Because `trtllm-serve` is OpenAI API compatible, we can use the `local-completions` model built in to `lm_eval`, which supports [these model_args](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/openai_completions.py#L12).
