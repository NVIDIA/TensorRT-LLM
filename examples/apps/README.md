# Apps examples with GenerationExecutor / High-level API
## OpenAI API
[openai_server.py](./openai_server.py) is an OpenAI compatible server which supports `v1/version`, `v1/completions` and `v1/chat/completions`. [openai_client.py](./openai_client.py) is a simple example using OpenAI client to query your model. To start the server, you can run
```
python3 ./openai_server.py <model_path>
```
Then you can use the following APIs by running our example client or by `curl`.
### v1/completions
Query by `curl`:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": <model_name>,
        "prompt": "Where is New York?",
        "max_tokens": 16,
        "temperature": 0
    }'
```
Query by our example:
```
python3 ./openai_client.py --prompt "Where is New York?" --api completions
```
### v1/chat/completions
Query by `curl`:
```
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": <model_name>,
        "prompt": "Where is New York?",
        "max_tokens": 16,
        "temperature": 0
    }'
```
Query by our example:
```
python3 ./openai_client.py --prompt "Where is New York?" --api chat
```
## Python chat

[chat.py](./chat.py) provides a small examples to play around with your model. Before running, install additional requirements with ` pip install -r ./requirements.txt`. Then you can run it with

```
python3 ./chat.py --model <model_dir> --tokenizer <tokenizer_path> --tp_size <tp_size>
```

Please run `python3 ./chat.py --help` for more information on the arguments.

Note that, the `model_dir` could accept the following formats:

1. A path to a built TRT-LLM engine
2. A path to a local HuggingFace model
3. The name of a HuggingFace model such as "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

## FastAPI server

### Install the additional requirements

```
pip install -r ./requirements.txt
```

### Start the server

Start the server with:

```
python3 ./fastapi_server.py <model_dir>&
```

Note that, the `model_dir` could accept same formats as in the chat example. If you are using an engine build with `trtllm-build`, remember to pass the tokenizer path:

```
python3 ./fastapi_server.py <model_dir> --tokenizer <tokenizer_dir>&
```

To get more information on all the arguments, please run `python3 ./fastapi_server.py --help`.

### Send requests

You can pass request arguments like "max_tokens", "top_p", "top_k" in your JSON dict:
```
curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_tokens": 8}'
```

You can also use the streaming interface with:
```
curl http://localhost:8000/generate -d '{"prompt": "In this example,", "max_tokens": 8, "streaming": true}'
```
