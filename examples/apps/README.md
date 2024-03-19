# Apps examples with GenerationExecutor / High-level API

## Python chat

[chat.py](./chat.py) provides a small examples to play around with your model. You can run it with

`python3 examples/apps/chat.py <path_to_tllm_engine_dir> <path_to_tokenizer_dir>`
or
`mpirun -n <world_size> python3 examples/apps/chat.py <path_to_tllm_engine_dir> <path_to_tokenizer_dir>`

You can modify prompt setting by entering options starting with '!!'. Type '!!help' to see available commands.

## FastAPI server

### Install the additional requirements

` pip install -r examples/apps/requirements.txt`

### Start the server

Suppose you have build an engine with `trtllm-build`, you can now serve it with:

`python3 -m examples.apps.fastapi_server <path_to_tllm_engine_dir> <tokenizer_type> &`
or
`mpirun -n <world_size> python3 -m examples.server.server <path_to_tllm_engine_dir> <tokenizer_type> &`

### Send requests

You can pass request arguments like "max_new_tokens", "top_p", "top_k" in your JSON dict:
`curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_new_tokens": 8}'`

You can also use the streaming interface with:
`curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_new_tokens": 8, "streaming": true}' --output -`
