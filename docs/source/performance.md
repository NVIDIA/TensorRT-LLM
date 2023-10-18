# Performance of TensorRT-LLM

This document summarizes performance measurements of TensorRT-LLM on H100
(Hopper), L40S (Ada) and A100 (Ampere) GPUs for a few key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Methodology

The different performance numbers below were collected using the methodology
described in the benchmarks [folder](../../benchmarks/).

## High Throughput

The below tables provide reference data at large batch sizes, representating
high throughput tasks.

### H100 GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | Output Length | Throughput (out tok/s) |
| :--------------------------- | :--------- | :-------- | :----------- | :------------ | ---------------------: |
| GPT-J 6B                     | 64         | 1         | 128          | 128           |                 10,907 |
| GPT-J 6B                     | 64         | 1         | 128          | 2048          |                  6,179 |
| GPT-J 6B                     | 64         | 1         | 2048         | 128           |                  2,229 |
| GPT-J 6B                     | 64         | 1         | 2048         | 2048          |                  2,980 |
|                              |            |           |              |               |                        |
| LLaMA 7B                     | 64         | 1         | 128          | 128           |                  9,193 |
| LLaMA 7B                     | 64         | 1         | 128          | 2048          |                  5,367 |
| LLaMA 7B                     | 64         | 1         | 2048         | 128           |                  2,058 |
| LLaMA 7B                     | 32         | 1         | 2048         | 2048          |                  2,230 |
|                              |            |           |              |               |                        |
| LLaMA 70B                    | 64         | 4         | 128          | 128           |                  3,317 |
| LLaMA 70B                    | 64         | 4         | 128          | 2048          |                  2,616 |
| LLaMA 70B                    | 64         | 4         | 2048         | 128           |                    843 |
| LLaMA 70B                    | 64         | 4         | 2048         | 2048          |                  1,583 |
|                              |            |           |              |               |                        |
| Falcon 180B                  | 96         | 8         | 128          | 128           |                  2,686 |
| Falcon 180B                  | 96         | 8         | 128          | 2048          |                  2,073 |
| Falcon 180B                  | 64         | 8         | 2048         | 128           |                    465 |

### L40S GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | Output Length | Throughput (out tok/s) |
| :--------------------------- | :--------- | :-------- | :----------- | :------------ | ---------------------: |
| GPT-J 6B                     | 64         | 1         | 128          | 128           |                  3,630 |
| GPT-J 6B                     | 64         | 1         | 128          | 2048          |                  1,859 |
| GPT-J 6B                     | 32         | 1         | 2048         | 128           |                    616 |
| GPT-J 6B                     | 32         | 1         | 2048         | 2048          |                    757 |
|                              |            |           |              |               |                        |
| LLaMA 7B                     | 64         | 1         | 128          | 128           |                  3,240 |
| LLaMA 7B                     | 64         | 1         | 128          | 2048          |                  1,622 |
| LLaMA 7B                     | 32         | 1         | 2048         | 128           |                    581 |
| LLaMA 7B                     | 16         | 1         | 2048         | 2048          |                    531 |


### A100 GPUs (FP16)

| Model                        | Batch Size | TP (1)    | Input Length | Output Length | Throughput (out tok/s) |
| :--------------------------- | :--------- | :-------- | :----------- | :------------ | ---------------------: |
| GPT-J 6B                     | 64         | 1         | 128          | 128           |                  3,679 |
| GPT-J 6B                     | 32         | 1         | 128          | 2048          |                  1,558 |
| GPT-J 6B                     | 32         | 1         | 2048         | 128           |                    526 |
| GPT-J 6B                     | 16         | 1         | 2048         | 2048          |                    650 |
|                              |            |           |              |               |                        |
| LLaMA 7B                     | 64         | 1         | 128          | 128           |                  3,486 |
| LLaMA 7B                     | 32         | 1         | 128          | 2048          |                  1,459 |
| LLaMA 7B                     | 32         | 1         | 2048         | 128           |                    529 |
| LLaMA 7B                     | 16         | 1         | 2048         | 2048          |                    592 |
|                              |            |           |              |               |                        |
| LLaMA 70B                    | 64         | 4         | 128          | 128           |                  1,237 |
| LLaMA 70B                    | 64         | 4         | 128          | 2048          |                  1,181 |
| LLaMA 70B                    | 64         | 4         | 2048         | 128           |                    272 |
| LLaMA 70B                    | 64         | 4         | 2048         | 2048          |                    738 |
|                              |            |           |              |               |                        |
| Falcon 180B                  | 64         | 8         | 128          | 128           |                    929 |
| Falcon 180B                  | 64         | 8         | 128          | 2048          |                    923 |
| Falcon 180B                  | 64         | 8         | 2048         | 128           |                    202 |

(1) TP stands for Tensor Parallelism.

## Low Latency

The below tables provide reference data at batch size 1 for first token
latency, representating end-user's percieved latency for online streaming
tasks.

### H100 GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | 1st Token Latency (ms) |
| :--------------------------- | :--------- | :-------- | :----------- | ---------------------: |
| GPT-J 6B                     | 1          | 1         | 128          |                      7 |
| GPT-J 6B                     | 1          | 1         | 2048         |                     29 |
|                              |            |           |              |                        |
| LLaMA 7B                     | 1          | 1         | 128          |                      7 |
| LLaMA 7B                     | 1          | 1         | 2048         |                     36 |
|                              |            |           |              |                        |
| LLaMA 70B                    | 1          | 4         | 128          |                     26 |
| LLaMA 70B                    | 1          | 4         | 2048         |                    109 |
|                              |            |           |              |                        |
| Falcon 180B                  | 1          | 8         | 128          |                     27 |
| Falcon 180B                  | 1          | 8         | 2048         |                    205 |

### L40S GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | 1st Token Latency (ms) |
| :--------------------------- | :--------- | :-------- | :----------- | ---------------------: |
| GPT-J 6B                     | 1          | 1         | 128          |                     12 |
| GPT-J 6B                     | 1          | 1         | 2048         |                     71 |
|                              |            |           |              |                        |
| LLaMA 7B                     | 1          | 1         | 128          |                     14 |
| LLaMA 7B                     | 1          | 1         | 2048         |                     73 |

### A100 GPUs (FP16)

| Model                        | Batch Size | TP (1)    | Input Length | 1st Token Latency (ms) |
| :--------------------------- | :--------- | :-------- | :----------- | ---------------------: |
| GPT-J 6B                     | 1          | 1         | 128          |                     12 |
| GPT-J 6B                     | 1          | 1         | 2048         |                    129 |
|                              |            |           |              |                        |
| LLaMA 7B                     | 1          | 1         | 128          |                     16 |
| LLaMA 7B                     | 1          | 1         | 2048         |                    133 |
|                              |            |           |              |                        |
| LLaMA 70B                    | 1          | 4         | 128          |                     47 |
| LLaMA 70B                    | 1          | 4         | 2048         |                    377 |
|                              |            |           |              |                        |
| Falcon 180B                  | 1          | 8         | 128          |                     61 |
| Falcon 180B                  | 1          | 8         | 2048         |                    509 |

(1) TP stands for Tensor Parallelism.


## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Fused Matmul + Gated-SiLU (LLaMA)

There are different possible implementations for Matmul followed by Gated-SiLU.
The simplest implementation uses two Matmul operations and combines the results
in a separate CUDA kernel. That's the current implementation in TensorRT-LLM.
The next release will include a more efficient implementation that runs a
single Matmul.
