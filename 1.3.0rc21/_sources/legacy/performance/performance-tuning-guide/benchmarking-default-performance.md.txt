(benchmarking-default-performance)=

# Benchmarking Default Performance

This section discusses how to build an engine for the model using the LLM-API and benchmark it using TRTLLM-Bench.

> Disclaimer: While performance numbers shown here are real, they are only for demonstration purposes. Differences in environment, SKU, interconnect, and workload can all significantly affect performance and lead to your results differing from what is shown here.

## Before You Begin: TensorRT-LLM LLM-API

TensorRT-LLM's LLM-API aims to make getting started with TensorRT-LLM quick and easy. For example, the following script instantiates `Llama-3.3-70B-Instruct` and runs inference on a small set of prompts. For those familiar with TensorRT-LLM's [CLI workflow](./benchmarking-default-performance.md#building-and-saving-engines-via-cli), the call to `LLM()` handles converting the model checkpoint and building the engine in one line.

```python
#quickstart.py
from tensorrt_llm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, I am",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm  =  LLM(
    model="meta-llama/Llama-3.3-70B-Instruct", #HuggingFace model name, no need to download the checkpoint beforehand
    tensor_parallel_size=4
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
```
### Troubleshooting Tips and Pitfalls To Avoid

Since we are running on multiple GPUs, MPI is used to spawn processes for each GPU. This raises the following requirements

1. The entrypoint to the script should be guarded via `if __name__ == '__main__'`. This requirement comes from mpi4py.
2. Depending on your environment, it might be required to wrap the `python` command with `mpirun`. For example the command to run the script above could be `mpirun -n 1 --oversubscribe --allow-run-as-root python quickstart.py`. For running on multiple GPUs on one node like the example is attempting to do it is usually not required to prefix with `mpirun` but if you are getting MPI errors then you should add it. Additionally, the `-n 1` which says just to launch one process is intentional as TensorRT-LLM handles spawning the processes for the remaining GPUs
3. If you get a HuggingFace access error when loading the Llama weights, this is likely because the model is gated. Request access on the HuggingFace page for the model. Then follow the instructions on [Huggingface's quickstart guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) to authenticate in your environment.


## Building and Saving the Engine

Save the engine using `.save()`. Just like the previous example, this script and all subsequent scripts might need to be run via `mpirun`.

```python
from tensorrt_llm import LLM

def main():
    llm = LLM(
        model="/scratch/Llama-3.3-70B-Instruct",
        tensor_parallel_size=4
    )

    llm.save("baseline")

if __name__ == '__main__':
    main()
```

### Building and Saving Engines via CLI

TensorRT-LLM also has a command line interface for building and saving engines. This workflow consists of two steps

1. Convert model checkpoint (HuggingFace, Nemo) to TensorRT-LLM checkpoint via `convert_checkpoint.py`. Each supported model has a `convert_checkpoint.py` associated it with it and can be found in the examples folder. For example, the `convert_checkpoint.py` script for Llama models can be found [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama/convert_checkpoint.py)
2. Build engine by passing TensorRT-LLM checkpoint to `trtllm-build` command. The `trtllm-build` command is installed automatically when the `tensorrt_llm` package is installed.

The README in the examples folder for supported models walks through building engines using this flow for a wide variety of situations. The examples folder for Llama models can be found at [https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama).

## Benchmarking with `trtllm-bench`

`trtllm-bench` provides a command line interface for benchmarking the throughput and latency of saved engines.

### Prepare Dataset

`trtllm-bench` expects to be passed in a dataset of requests to run through the model. This guide creates a dummy dataset of 1000 requests with every request having input and output sequence length of 2048.  TensorRT-LLM provides the `prepare_dataset.py` script to produce the dataset. To use it clone the TensorRT-LLM Repo and run the following command:

`python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer /path/to/hf/Llama-3.3-70B-Instruct/ token-norm-dist --input-mean 2048 --output-mean 2048 --input-stdev 0 --output-stdev 0 --num-requests 1000 > synthetic_2048_2048.txt`

`trtllm-bench` can also take in real data, see [`trtllm-bench` documentation](../perf-benchmarking.md) for more details on the required format.

### Running Throughput and Latency Benchmarks

 To benchmark the baseline engine built in the previous script, run the following commands. Again, due to the multi-gpu nature of the workload you may need prefix the `trtllm-bench` command with `mpirun -n 1 --oversubscribe --allow-run-as-root`.

**Throughput**

```bash
trtllm-bench \
--model /path/to/hf/Llama-3.3-70B-Instruct/ \
throughput \
--dataset /path/to/dataset/synthetic_2048_2048_1000.txt \
--engine_dir /path/to/engines/baseline #replace baseline with name used in llm.save()
```

This command will send all 1000 requests to the model immediately. Run `trtllm-bench throughput -h` to see a list of options that help you control the request rate and cap the total number of requests if the benchmark is taking too long. For reference, internal testing of the above command took around 20 minutes on a 4 NVLink connected H100-sxm-80GB.

Running this command will provide a throughput overview like this:

```bash
===========================================================
= ENGINE DETAILS
===========================================================
Model:			/scratch/Llama-3.3-70B-Instruct/
Engine Directory:	/scratch/grid_search_engines/baseline
TensorRT-LLM Version:	0.16.0
Dtype:			bfloat16
KV Cache Dtype:		None
Quantization:		None
Max Sequence Length:	131072

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:		4
PP Size:		1
Max Runtime Batch Size:	2048
Max Runtime Tokens:	8192
Scheduling Policy:	Guaranteed No Evict
KV Memory Percentage:	90.00%
Issue Rate (req/sec):	7.9353E+13

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Number of requests:		1000
Average Input Length (tokens):	2048.0000
Average Output Length (tokens):	2048.0000
Token Throughput (tokens/sec):	1585.7480
Request Throughput (req/sec):	0.7743
Total Latency (ms):		1291504.1051

===========================================================
```

**Latency**

```bash
trtllm-bench \
--model /path/to/hf/Llama-3.3-70B-Instruct/ \
latency \
--dataset /path/to/dataset/synthetic_2048_2048_1000.txt \
--num-requests 100 \
--warmup 10 \
--engine_dir /path/to/engines/baseline #replace baseline with name used in llm.save()
```
The latency benchmark enforces a batch size of 1 to accurately measure latency, which can significantly increase testing duration. In the example above the total number of requests is limited to 100 via `--num-requests` to make the test duration more manageable. This example benchmark was designed to produce very stable numbers, but in real scenarios even 100 requests is likely more than you need and can take a long time to complete (in the case-study it took about an hour and a half). Reducing the number of requests to 10 would still provide accurate data and enable faster development iterations. In general you should adjust the number of requests per your needs. Run `trtllm-bench latency -h` to see other configurable options.

Running this command will provide a latency overview like this:

```bash
===========================================================
= ENGINE DETAILS
===========================================================
Model:			/scratch/Llama-3.3-70B-Instruct/
Engine Directory:	/scratch/grid_search_engines/baseline
TensorRT-LLM Version:	0.16.0
Dtype:			bfloat16
KV Cache Dtype:		None
Quantization:		None
Max Input Length:	1024
Max Sequence Length:	131072

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:		4
PP Size:		1
Max Runtime Batch Size:	1
Max Runtime Tokens:	8192
Scheduling Policy:	Guaranteed No Evict
KV Memory Percentage:	90.00%

===========================================================
= GENERAL OVERVIEW
===========================================================
Number of requests:		100
Average Input Length (tokens):	2048.0000
Average Output Length (tokens):	2048.0000
Average request latency (ms):	63456.0704

===========================================================
= THROUGHPUT OVERVIEW
===========================================================
Request Throughput (req/sec):		  0.0158
Total Token Throughput (tokens/sec):	  32.2742
Generation Token Throughput (tokens/sec): 32.3338

===========================================================
= LATENCY OVERVIEW
===========================================================
Total Latency (ms):		  6345624.0554
Average time-to-first-token (ms): 147.7502
Average inter-token latency (ms): 30.9274
Acceptance Rate (Speculative):	  1.00

===========================================================
= GENERATION LATENCY BREAKDOWN
===========================================================
MIN (ms): 63266.8804
MAX (ms): 63374.7770
AVG (ms): 63308.3201
P90 (ms): 63307.1885
P95 (ms): 63331.7136
P99 (ms): 63374.7770

===========================================================
= ACCEPTANCE BREAKDOWN
===========================================================
MIN: 1.00
MAX: 1.00
AVG: 1.00
P90: 1.00
P95: 1.00
P99: 1.00

===========================================================
```

## Results

The baseline engine achieves the following performance for token throughput, request throughput, average time to first token, and average inter-token latency. These metrics will be analyzed throughout the guide.


| Metric                        | Value         |
|-------------------------------|---------------|
| Token Throughput (tokens/sec) | 1564.3040    |
| Request Throughput (req/sec)  | 0.7638        |
| Average Time To First Token (ms) | 147.6976  |
| Average Inter-Token Latency (ms) | 31.3276    |

The following sections show ways you can improve these metrics using different configuration options.
