# Python Bindings Example

This example shows how to use the python bindings interface to generate tokens using a TensorRT engine.

## Setup

Build a TensorRT engine for one of the supported TensorRT-LLM model following instructions in the corresponding `examples` folder.

## Usage

Run `example.py`, passing in the directory where the TensorRT engine was generated. For example:

```
cd examples/bindings
python3 example.py --model_path=../llama/tmp/7B/trt_engines/fp16/1-gpu/
```
