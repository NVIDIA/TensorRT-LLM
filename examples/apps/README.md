# Apps examples with GenerationExecutor / High-level API

## Python chat

[chat.py](./chat.py) provides a small examples to play around with your model. You can run it with

`python3 ./chat.py --model <model_dir> --tokenizer <tokenizer_path> --tp_size <tp_size>`

Please run `python3 ./chat.py --help` for more information on the arguments.

Note that, the `model_dir` could accept the following formats:

1. A path to a built TRT-LLM engine
2. A path to a local HuggingFace model
3. The name of a HuggingFace model such as "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

## FastAPI server

### Install the additional requirements

` pip install -r ./requirements.txt`

### Start the server

Suppose you have build an engine with `trtllm-build`, you can now serve it with:

`python3 ./fastapi_server <model_dir>&`

To get more information on all the arguments, please run `python3 ./fastapi_server --help`.

### Send requests

You can pass request arguments like "max_new_tokens", "top_p", "top_k" in your JSON dict:
`curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_new_tokens": 8}'`

You can also use the streaming interface with:
`curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_new_tokens": 8, "streaming": true}' --output -`
