# TensorRT-LLM Benchmarking

For benchmarking TensorRT-LLM, use
[`trtllm-bench`](../docs/source/developer-guide/perf-benchmarking.md) and
`trtllm-serve`. The legacy C++ runtime benchmarks (`gptManagerBenchmark`,
`bertBenchmark`, `disaggServerBenchmark`) were removed together with the
TensorRT backend.

This directory keeps the dataset preparation tools consumed by `trtllm-bench`:

- `prepare_dataset.py` — generate benchmark datasets from real data or with
  synthetic normal/uniform token-length distributions:

  ```bash
  python3 prepare_dataset.py \
      --tokenizer <path/to/tokenizer> \
      --output preprocessed_dataset.json \
      dataset \
      --dataset-name <name of the dataset> \
      --dataset-split <split of the dataset to use> \
      --dataset-input-key <dataset dictionary key for input> \
      --dataset-prompt-key <dataset dictionary key for prompt> \
      --dataset-output-key <dataset dictionary key for output> \
      [--num-requests 100] \
      [--max-input-len 1000] \
      [--output-len-dist 100,10]
  ```

  Synthetic variants: `python3 prepare_dataset.py ... token-norm-dist ...` and
  `... token-unif-dist ...`. Run with `--help` for the full option list.

- `utils/prepare_real_data.py`, `utils/prepare_synthetic_data.py` — the
  subcommand implementations.
- `utils/generate_rand_loras.py` — generate random LoRA adapters for
  LoRA benchmarking.
- `utils/convert_nemo_dataset.py` — convert NeMo chat datasets.
