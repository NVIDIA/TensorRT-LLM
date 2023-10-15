# Performance of TensorRT-LLM

That document summarizes performance measurements of TensorRT-LLM on A100 and
H100 GPUs for a few key models.

## Methodology

The different performance numbers below were collected using the methodology
described in the benchmarks [folder](../benchmarks/README.md).

## A100 GPUs

| Model                        | Batch Size | TP (1) | Input Length | Output Length | Precision | Time (ms) |
| :--------------------------- | ---------: | -----: | -----------: | ------------: | --------: | --------: |
| GPT 175B                     | 1          | 8      | 32           | 8             | FP16      |           |
| GPT 175B                     | 1          | 8      | 128          | 16            | FP16      |           |
| GPT 175B                     | 1          | 8      | 1024         | 128           | FP16      |           |
|                              |            |        |              |               |           |           |
| LLaMA 7B                     | 1          | 1      | 32           | 8             | FP16      |           |
| LLaMA 7B                     | 1          | 1      | 128          | 16            | FP16      |           |
| LLaMA 7B                     | 1          | 1      | 1024         | 128           | FP16      |           |
|                              |            |        |              |               |           |           |
| LLaMA 70B                    | 1          | 2      | 32           | 8             | FP16      |           |
| LLaMA 70B                    | 1          | 2      | 128          | 16            | FP16      |           |
| LLaMA 70B                    | 1          | 2      | 1024         | 128           | FP16      |           |
|                              |            |        |              |               |           |           |
| LLaMA 70B                    | 1          | 4      | 32           | 8             | FP16      |           |
| LLaMA 70B                    | 1          | 4      | 128          | 16            | FP16      |           |
| LLaMA 70B                    | 1          | 4      | 1024         | 128           | FP16      |           |

## H100 GPUs

| Model                        | Batch Size | TP (1) | Input Length | Output Length | Precision | Time (ms) |
| :--------------------------- | ---------: | -----: | -----------: | ------------: | --------: | --------: |
| GPT 175B                     | 1          | 8      | 32           | 8             | FP16      |           |
| GPT 175B                     | 1          | 8      | 128          | 16            | FP16      |           |
| GPT 175B                     | 1          | 8      | 1024         | 128           | FP16      |           |
|                              |            |        |              |               |           |           |
| LLaMA 7B                     | 1          | 1      | 32           | 8             | FP16      |           |
| LLaMA 7B                     | 1          | 1      | 128          | 16            | FP16      |           |
| LLaMA 7B                     | 1          | 1      | 1024         | 128           | FP16      |           |
|                              |            |        |              |               |           |           |
| LLaMA 70B                    | 1          | 2      | 32           | 8             | FP16      |           |
| LLaMA 70B                    | 1          | 2      | 128          | 16            | FP16      |           |
| LLaMA 70B                    | 1          | 2      | 1024         | 128           | FP16      |           |
|                              |            |        |              |               |           |           |
| LLaMA 70B                    | 1          | 4      | 32           | 8             | FP16      |           |
| LLaMA 70B                    | 1          | 4      | 128          | 16            | FP16      |           |
| LLaMA 70B                    | 1          | 4      | 1024         | 128           | FP16      |           |

## Known Issues

The following issues need are being addressed to improve the efficiency of TensorRT-LLM.

### Fused LayerNorm (All Models)

There is a custom plugin for LayerNorm in the current release of TensorRT-LLM
that works well with smaller batch sizes or input lengths but hurts performance
when the input sequence length increases (due to interactions with TensorRT).
The TensorRT-LLM team is working on fixing those issues.

### Fused Matmul + Gated-SiLU (LLaMA)

There are different possible implementations for Matmul followed by Gated-SiLU.
The simplest implementation uses two Matmul operations and combines the results
in a separate CUDA kernel. That's the current implementation in TensorRT-LLM.
The next release will include a more efficient implementation that runs a
single Matmul.

### Performance of Matmuls

The Matmul/GEMM plugin in TensorRT-LLM does not run auto-tunning to select the
best GEMM algorithm and relies entirely on the cuBLAS heuristic to select the
best algorithm. It may lead to suboptimal choices.

Also, the plugin may not be optimal when enqueueing work in the TensorRT
engine. It may affect the performance for smaller models.
