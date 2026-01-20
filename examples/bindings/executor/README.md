# Python Bindings Example

This example shows how to use the python bindings interface to generate tokens
using a TensorRT engine.

## Setup

Build a TensorRT engine for one of the supported TensorRT LLM model following
instructions in the corresponding `examples` folder.

## Usage

### Basic example

Run `example_basic.py`, passing in the directory where the TensorRT engine was generated. For example:

```
cd examples/bindings
python3 example_basic.py --model_path=../llama/tmp/7B/trt_engines/fp16/1-gpu/
```

### Debug example

This example shows how you can define which engine IO tensors should be kept or dumped to numpy files.
Run `example_debug.py`, passing in the directory where the TensorRT engine was generated. For example:

```
cd examples/bindings
python3 example_debug.py --model_path=../llama/tmp/7B/trt_engines/fp16/1-gpu/
```

### Advanced example

This example shows how you can use the python bindings to generate tokens for a larger number of requests concurrently and demonstrate how tokens can be returned in a streaming fashion.

The full list of supported input parameters can be obtained with:
```
pytho3 example_advanced.py -h
```

For example, assuming a CSV file named `input_tokens.csv` exist which contains the following input tokens:
```
1, 2, 3, 4, 5, 6
1, 2, 3, 4
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```
one can generate output tokens for those 3 prompts with:
```
python3 example_advanced.py --model_path <model_path> --input_tokens_csv_file input_tokens.csv
```
Upon successful completion, the output tokens will be written to file `output_tokens.csv`.

### Multi-GPU Example

To run the two examples for models requiring more than one gpu, you can run the example with MPI.

For example, the basic example can be run as follows:
```
mpirun -n 4 --allow-run-as-root python3 example_basic.py --model_path=../llama/tmp/7B/trt_engines/fp16/4gpu_tp4_pp1/
```

The advanced example can also be run using the ORCHESTRATOR mode, where the additional processes needed for multi-GPU runs will automatically be spawned.
This can be done by running:
```
python3 example_advanced.py --model_path=../llama/tmp/7B/trt_engines/fp16/4gpu_tp4_pp1/ --use_orchestrator_mode
```

### Logits post processor example

This example shows how to generate JSON structured output using LogitsPostProcessor API.

```
python3 example_logits_processor.py -t <tokenizer_path> -e <engine_path> --batch_size 8
```

LogitsPostProcessorBatched, which fuses logits processing for all samples in a batch into a single callback, is enabled by `--lpp_batched`
