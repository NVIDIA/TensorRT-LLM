# Asynchronous generation in python

## Install the requirements

` pip install -r examples/server/requirements.txt`

## Directly from python, with the HL API

Due to limitation from the HLAPI implementation, currently only LLaMA models are supported:
`python3 examples/server/async.py <path_to_hf_llama_dir>`


## Using the server interface for TensorRT-LLM

### Start the server

`python3 -m examples.server.server <path_to_tllm_engine_dir> <tokenizer_type> &`

### Send requests

You can pass request arguments like "max_new_tokens", "top_p", "top_k" in your JSON dict:
`curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_new_tokens": 8}'`

You can also use the streaming interface with:
`curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_new_tokens": 8, "streaming": true}' --output -`
